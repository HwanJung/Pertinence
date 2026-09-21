#!/usr/bin/env python3
"""Create an allowlisted modelcar context with a cryptographic inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import ast
import math
import struct
import zipfile


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def regular(root: Path, relative: str) -> Path:
    path = root / relative
    if path.is_symlink() or not path.is_file() or root.resolve() not in path.resolve().parents:
        raise ValueError(f"unsafe or missing modelcar file: {relative}")
    return path


def sealed_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON manifest must be an object: {path.name}")
    fingerprint = value.get("fingerprint")
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    actual = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if fingerprint != actual:
        raise ValueError(f"manifest fingerprint mismatch: {path.name}")
    return value


def npy_array(payload: bytes) -> tuple[str, tuple[int, ...], bytes]:
    if payload[:6] != b"\x93NUMPY" or len(payload) < 10:
        raise ValueError("dispatcher checkpoint contains an invalid NPY member")
    major = payload[6]
    if major == 1:
        header_size = struct.unpack("<H", payload[8:10])[0]
        offset = 10
    elif major in {2, 3}:
        header_size = struct.unpack("<I", payload[8:12])[0]
        offset = 12
    else:
        raise ValueError("dispatcher checkpoint uses an unsupported NPY version")
    header = ast.literal_eval(payload[offset : offset + header_size].decode("latin1").strip())
    if not isinstance(header, dict):
        raise ValueError("dispatcher NPY header must be a mapping")
    if header.get("fortran_order") is not False:
        raise ValueError("dispatcher arrays must use C order")
    shape = header.get("shape")
    if not isinstance(shape, tuple) or not all(type(item) is int for item in shape):
        raise ValueError("dispatcher array shape is invalid")
    return header.get("descr"), shape, payload[offset + header_size :]


def validate_dispatcher_npz(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist()) != 2 or set(archive.namelist()) != {
            "linear.weight.npy",
            "linear.bias.npy",
        }:
            raise ValueError("dispatcher NPZ keys differ")
        for name, expected_shape in (
            ("linear.weight.npy", (3, 1024)),
            ("linear.bias.npy", (3,)),
        ):
            dtype, shape, data = npy_array(archive.read(name))
            if dtype not in {"<f4", "|f4"} or shape != expected_shape:
                raise ValueError(f"{name} must be FP32 {expected_shape}")
            expected_bytes = math.prod(shape) * 4
            if len(data) != expected_bytes or any(
                not math.isfinite(value) for (value,) in struct.iter_unpack("<f", data)
            ):
                raise ValueError(f"{name} has invalid or non-finite data")


def select(kind: str, source: Path) -> list[str]:
    if kind == "expert":
        model = regular(source, "model.onnx")
        checksum = regular(source, "model.sha256").read_text(encoding="utf-8").split()[0]
        if checksum != sha256(model):
            raise ValueError("model.onnx does not match model.sha256")
        return ["model.onnx", "model.sha256"]
    if kind != "dynamic-router":
        raise ValueError("modelcar kind must be expert or dynamic-router")
    names = ["contract.json", "search-result.json", "final-report.json", "manifest.json"]
    documents = {name: sealed_json(regular(source, name)) for name in names}
    manifest, final, search = (
        documents["manifest.json"],
        documents["final-report.json"],
        documents["search-result.json"],
    )
    if (
        manifest.get("contract_fingerprint") != documents["contract.json"].get("fingerprint")
        or manifest.get("search_fingerprint") != search.get("fingerprint")
        or manifest.get("final_report_fingerprint") != final.get("fingerprint")
        or final.get("search_fingerprint") != search.get("fingerprint")
        or final.get("contract_fingerprint") != documents["contract.json"].get("fingerprint")
        or manifest.get("front_id") != search.get("front_id")
        or final.get("front_id") != search.get("front_id")
    ):
        raise ValueError("dispatcher evaluated-bundle lineage mismatch")
    models = [item for item in manifest.get("models", []) if item.get("model_id") == "dispatcher-000733"]
    if len(models) != 1:
        raise ValueError("evaluated bundle must contain dispatcher-000733 exactly once")
    checkpoint_name = models[0].get("checkpoint")
    expert_order = documents["contract.json"].get("config", {}).get("routing", {}).get("expert_ids")
    solutions = [
        item for item in final.get("solutions", []) if item.get("model_id") == "dispatcher-000733"
    ]
    if (
        models[0].get("dimensions") != {"D": 1024, "K": 3}
        or models[0].get("expert_order") != expert_order
        or len(solutions) != 1
        or solutions[0].get("checkpoint_sha256") != models[0].get("checkpoint_sha256")
    ):
        raise ValueError("dispatcher dimensions, expert order, or final lineage mismatch")
    if not isinstance(checkpoint_name, str) or not checkpoint_name.endswith(".npz"):
        raise ValueError("dispatcher checkpoint must be NPZ")
    checkpoint = regular(source, checkpoint_name)
    if sha256(checkpoint) != models[0].get("checkpoint_sha256"):
        raise ValueError("dispatcher checkpoint digest mismatch")
    validate_dispatcher_npz(checkpoint)
    return [*names, checkpoint_name]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("expert", "dynamic-router"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error("destination already exists")
    try:
        selected = select(args.kind, args.source)
        actual = {
            path.relative_to(args.source).as_posix()
            for path in args.source.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if actual != set(selected):
            raise ValueError(f"bundle violates file allowlist: {sorted(actual - set(selected))}")
        bundle = args.destination / "bundle"
        bundle.mkdir(parents=True)
        inventory = []
        for relative in selected:
            source = regular(args.source, relative)
            target = bundle / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            inventory.append(
                {"path": relative, "sha256": sha256(source), "size_bytes": source.stat().st_size}
            )
        (bundle / "inventory.json").write_text(
            json.dumps({"schema_version": 1, "files": inventory}, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, SyntaxError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
