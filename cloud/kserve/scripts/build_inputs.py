#!/usr/bin/env python3
"""Seal and fetch the exact generation-pinned KServe Cloud Build inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit


EXPERT_IDS = ("shufflenetv2_x0_5", "mobilenetv2_x0_5", "resnet56")
DISPATCHER_CHECKPOINT = "evaluated/dispatcher-000733.npz"
REQUIRED_PATHS = frozenset(
    {
        *(f"experts/{expert_id}/{name}" for expert_id in EXPERT_IDS for name in ("model.onnx", "model.sha256")),
        "evaluated/contract.json",
        "evaluated/search-result.json",
        "evaluated/final-report.json",
        "evaluated/manifest.json",
        DISPATCHER_CHECKPOINT,
    }
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if not isinstance(key, str) or key in result:
                raise ValueError("manifest keys must be unique strings")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {item}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("build input manifest must be a JSON object")
    return value


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("build input path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe build input path: {value!r}")
    return value


def _object_uri(value: Any, relative: str) -> str:
    if not isinstance(value, str) or any(character.isspace() for character in value):
        raise ValueError("build input URI must be a single-token gs:// object URI")
    parsed = urlsplit(value)
    object_name = parsed.path.lstrip("/")
    if (
        parsed.scheme != "gs"
        or not parsed.netloc
        or not object_name
        or parsed.query
        or parsed.fragment
        or "\\" in value
        or any(part in {"", ".", ".."} for part in PurePosixPath(object_name).parts)
    ):
        raise ValueError(f"unsafe GCS object URI: {value!r}")
    if not object_name.endswith("/" + relative) and object_name != relative:
        raise ValueError(f"GCS object URI does not end with its manifest path: {relative}")
    return value


def _versioned_uri(value: str) -> tuple[str, int]:
    uri, separator, raw_generation = value.rpartition("#")
    if not separator or not raw_generation.isdecimal():
        raise ValueError("manifest URI must end in #GENERATION")
    generation = int(raw_generation)
    if generation < 1:
        raise ValueError("manifest generation must be positive")
    # Reuse the strict object parser without imposing a build-bundle suffix.
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "gs"
        or not parsed.netloc
        or not parsed.path.lstrip("/")
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in uri)
        or "\\" in uri
        or any(part in {"", ".", ".."} for part in PurePosixPath(parsed.path.lstrip("/")).parts)
    ):
        raise ValueError("manifest URI must be a safe gs:// object URI with #GENERATION")
    return uri, generation


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the sealed exact build-input allowlist."""

    if set(value) != {"schema_version", "kind", "objects", "fingerprint"}:
        raise ValueError("build input manifest fields differ from the schema")
    if value.get("schema_version") != 1 or value.get("kind") != "kserve-build-inputs":
        raise ValueError("unsupported build input manifest identity")
    fingerprint = value.get("fingerprint")
    if not isinstance(fingerprint, str) or not SHA256_RE.fullmatch(fingerprint):
        raise ValueError("build input manifest fingerprint must be a SHA-256")
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    if fingerprint != _canonical_hash(body):
        raise ValueError("build input manifest fingerprint mismatch")
    raw_objects = value.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("build input manifest objects must be a list")
    objects: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_uris: set[str] = set()
    prefixes: set[str] = set()
    for raw in raw_objects:
        if not isinstance(raw, dict) or set(raw) != {
            "path", "uri", "generation", "size_bytes", "sha256"
        }:
            raise ValueError("build input object fields differ from the schema")
        relative = _safe_relative(raw["path"])
        uri = _object_uri(raw["uri"], relative)
        generation = raw["generation"]
        size = raw["size_bytes"]
        digest = raw["sha256"]
        if type(generation) is not int or generation < 1:
            raise ValueError(f"{relative}: generation must be a positive integer")
        if type(size) is not int or size < 1:
            raise ValueError(f"{relative}: size_bytes must be a positive integer")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError(f"{relative}: sha256 must be 64 lowercase hex characters")
        if relative in seen_paths or uri in seen_uris:
            raise ValueError("build input paths and URIs must be unique")
        seen_paths.add(relative)
        seen_uris.add(uri)
        prefixes.add(uri[: -(len(relative) + 1)])
        objects.append(dict(raw))
    if seen_paths != REQUIRED_PATHS:
        raise ValueError(
            "build input paths differ from the exact allowlist; "
            f"missing={sorted(REQUIRED_PATHS - seen_paths)}, "
            f"extra={sorted(seen_paths - REQUIRED_PATHS)}"
        )
    if len(prefixes) != 1:
        raise ValueError("all build inputs must be beneath one immutable GCS prefix")
    if [item["path"] for item in objects] != sorted(seen_paths):
        raise ValueError("build input objects must be sorted by path")
    return {
        "schema_version": 1,
        "kind": "kserve-build-inputs",
        "objects": objects,
        "fingerprint": fingerprint,
    }


def seal_from_bundle(
    bundle: Path, object_versions: Mapping[str, tuple[str, int]]
) -> dict[str, Any]:
    """Seal local bytes against their already-published object generations."""

    bundle = Path(bundle)
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("build bundle must be a regular directory")
    entries = list(bundle.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("build bundle must not contain symlinks")
    if any(not path.is_dir() and not path.is_file() for path in entries):
        raise ValueError("build bundle must contain only regular files and directories")
    local_paths = {
        path.relative_to(bundle).as_posix()
        for path in entries
        if path.is_file()
    }
    if local_paths != REQUIRED_PATHS or set(object_versions) != REQUIRED_PATHS:
        raise ValueError("local bundle and object mappings must exactly match the build allowlist")
    objects = []
    for relative in sorted(REQUIRED_PATHS):
        path = bundle.joinpath(*PurePosixPath(relative).parts)
        uri, generation = object_versions[relative]
        objects.append(
            {
                "path": relative,
                "uri": uri,
                "generation": generation,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    body: dict[str, Any] = {
        "schema_version": 1,
        "kind": "kserve-build-inputs",
        "objects": objects,
    }
    return validate_manifest({**body, "fingerprint": _canonical_hash(body)})


def _write_create_only(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _gcloud_copy(source: str, destination: Path) -> None:
    subprocess.run(
        ["gcloud", "storage", "cp", "--quiet", source, str(destination)],
        check=True,
    )


def fetch_inputs(
    manifest_uri: str,
    manifest_output: Path,
    output: Path,
    receipt: Path,
    *,
    fetch: Callable[[str, Path], None] = _gcloud_copy,
) -> dict[str, Any]:
    """Fetch a sealed manifest and every exact object generation into new paths."""

    manifest_base_uri, manifest_generation = _versioned_uri(manifest_uri)
    manifest_output, output, receipt = map(Path, (manifest_output, output, receipt))
    if len({path.absolute() for path in (manifest_output, output, receipt)}) != 3:
        raise ValueError("manifest, output, and receipt paths must differ")
    if any(path.exists() or path.is_symlink() for path in (manifest_output, output, receipt)):
        raise ValueError("build input outputs must not already exist")
    if not manifest_output.parent.is_dir() or not receipt.parent.is_dir():
        raise ValueError("manifest and receipt parent directories must already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    manifest_temporary = temporary / ".build-input-manifest.json"
    try:
        fetch(manifest_uri, manifest_temporary)
        if manifest_temporary.is_symlink() or not manifest_temporary.is_file():
            raise ValueError("manifest fetch did not produce a regular file")
        manifest = validate_manifest(_load_json(manifest_temporary))
        fetched = []
        for item in manifest["objects"]:
            relative = item["path"]
            destination = temporary.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            versioned = f"{item['uri']}#{item['generation']}"
            fetch(versioned, destination)
            if destination.is_symlink() or not destination.is_file():
                raise ValueError(f"fetch did not produce a regular file: {relative}")
            actual_size = destination.stat().st_size
            actual_sha = _sha256(destination)
            if actual_size != item["size_bytes"] or actual_sha != item["sha256"]:
                raise ValueError(f"downloaded bytes differ from manifest: {relative}")
            fetched.append(
                {
                    "path": relative,
                    "uri": item["uri"],
                    "generation": item["generation"],
                    "size_bytes": actual_size,
                    "sha256": actual_sha,
                }
            )
        manifest_bytes = manifest_temporary.read_bytes()
        manifest_temporary.unlink()
        if output.exists() or output.is_symlink():
            raise ValueError("build input output appeared during fetch")
        os.rename(temporary, output)
        _write_create_only(manifest_output, manifest_bytes)
        body: dict[str, Any] = {
            "schema_version": 1,
            "kind": "kserve-build-input-fetch",
            "valid": True,
            "manifest": {
                "uri": manifest_base_uri,
                "generation": manifest_generation,
                "size_bytes": len(manifest_bytes),
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "fingerprint": manifest["fingerprint"],
            },
            "objects": fetched,
        }
        result = {**body, "fingerprint": _canonical_hash(body)}
        _write_create_only(
            receipt,
            (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
        )
        return result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _parse_object_versions(values: Sequence[str]) -> dict[str, tuple[str, int]]:
    result: dict[str, tuple[str, int]] = {}
    for value in values:
        relative, separator, versioned = value.partition("=")
        if not separator or relative in result:
            raise ValueError("--object entries must be unique PATH=gs://OBJECT#GENERATION pairs")
        uri, generation = _versioned_uri(versioned)
        result[_safe_relative(relative)] = (uri, generation)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--bundle", type=Path, required=True)
    seal.add_argument("--object", action="append", default=[], required=True)
    seal.add_argument("--output", type=Path, required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--manifest-uri", required=True)
    fetch.add_argument("--manifest-output", type=Path, required=True)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "seal":
            manifest = seal_from_bundle(args.bundle, _parse_object_versions(args.object))
            encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
            _write_create_only(args.output, encoded)
            print(manifest["fingerprint"])
        else:
            result = fetch_inputs(
                args.manifest_uri,
                args.manifest_output,
                args.output,
                args.receipt,
            )
            print(result["fingerprint"])
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
