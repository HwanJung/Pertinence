#!/usr/bin/env python3
"""Create an allowlisted modelcar context with a cryptographic inventory."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import zipfile


EXPERT_IDS = ("shufflenetv2_x0_5", "mobilenetv2_x0_5", "resnet56")


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


def expert_files(source: Path, prefix: str = "") -> list[str]:
    model_name = f"{prefix}model.onnx"
    checksum_name = f"{prefix}model.sha256"
    model = regular(source, model_name)
    checksum_file = regular(source, checksum_name)
    checksum_parts = checksum_file.read_text(encoding="utf-8").split()
    if (
        len(checksum_parts) not in {1, 2}
        or (len(checksum_parts) == 2 and checksum_parts[1].lstrip("*") != "model.onnx")
        or checksum_parts[0] != sha256(model)
    ):
        raise ValueError(f"{model_name} does not match {checksum_name}")
    return [model_name, checksum_name]


def evaluated_files(source: Path, prefix: str = "") -> tuple[list[str], dict]:
    names = ["contract.json", "search-result.json", "final-report.json", "manifest.json"]
    documents = {name: sealed_json(regular(source, f"{prefix}{name}")) for name in names}
    manifest, final, search = (
        documents["manifest.json"],
        documents["final-report.json"],
        documents["search-result.json"],
    )
    contract = documents["contract.json"]
    if (
        manifest.get("contract_fingerprint") != contract.get("fingerprint")
        or manifest.get("search_fingerprint") != search.get("fingerprint")
        or manifest.get("final_report_fingerprint") != final.get("fingerprint")
        or final.get("search_fingerprint") != search.get("fingerprint")
        or final.get("contract_fingerprint") != contract.get("fingerprint")
        or manifest.get("front_id") != search.get("front_id")
        or final.get("front_id") != search.get("front_id")
    ):
        raise ValueError("dispatcher evaluated-bundle lineage mismatch")
    models = [item for item in manifest.get("models", []) if item.get("model_id") == "dispatcher-000733"]
    if len(models) != 1:
        raise ValueError("evaluated bundle must contain dispatcher-000733 exactly once")
    checkpoint_name = models[0].get("checkpoint")
    expert_order = contract.get("config", {}).get("routing", {}).get("expert_ids")
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
    checkpoint_relative = f"{prefix}{checkpoint_name}"
    checkpoint = regular(source, checkpoint_relative)
    if sha256(checkpoint) != models[0].get("checkpoint_sha256"):
        raise ValueError("dispatcher checkpoint digest mismatch")
    validate_dispatcher_npz(checkpoint)
    return ([*(f"{prefix}{name}" for name in names), checkpoint_relative], contract)


def select(kind: str, source: Path) -> list[str]:
    if kind == "expert":
        return expert_files(source)
    if kind == "dynamic-router":
        selected, _ = evaluated_files(source)
        return selected
    if kind != "fused-dynamic":
        raise ValueError("modelcar kind must be expert, dynamic-router, or fused-dynamic")

    selected, contract = evaluated_files(source, "evaluated/")
    expert_order = contract.get("config", {}).get("routing", {}).get("expert_ids")
    if expert_order != list(EXPERT_IDS):
        raise ValueError("fused expert order differs from the approved evaluated contract")
    contract_experts = contract.get("experts", {}).get("experts")
    if not isinstance(contract_experts, list):
        raise ValueError("evaluated contract expert inventory is missing")
    by_id = {
        item.get("id"): item
        for item in contract_experts
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if len(by_id) != len(contract_experts) or set(by_id) != set(EXPERT_IDS):
        raise ValueError("evaluated contract experts differ from the fused expert allowlist")
    for expert_id in EXPERT_IDS:
        prefix = f"experts/{expert_id}/"
        expert_selected = expert_files(source, prefix)
        model_digest = sha256(regular(source, f"{prefix}model.onnx"))
        if by_id[expert_id].get("model_sha256") != model_digest:
            raise ValueError(f"{expert_id} model SHA-256 differs from evaluated contract")
        selected.extend(expert_selected)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("expert", "dynamic-router", "fused-dynamic"))
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
