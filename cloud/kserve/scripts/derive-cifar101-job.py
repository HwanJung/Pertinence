#!/usr/bin/env python3
"""Create the sealed CIFAR-10.1 load-generator Job from a rendered workload file."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO

import yaml

from validate_rendered import validate_document


SOURCE_JOB_NAME = "pertinence-loadgen"
DESTINATION_JOB_NAME = "pertinence-loadgen-cifar101-v6"
ALLOWED_CHANGE = "/metadata/name"


class StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader which also rejects duplicate mapping keys."""


def _construct_mapping(
    loader: StrictSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(document: object) -> bytes:
    try:
        return json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"document cannot be represented as canonical JSON: {error}") from error


def _load_documents(data: bytes, source: Path) -> list[object]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{source}: input is not UTF-8") from error
    try:
        return list(yaml.load_all(text, Loader=StrictSafeLoader))
    except yaml.YAMLError as error:
        raise ValueError(f"{source}: invalid YAML: {error}") from error


def _select_source_job(documents: list[object], source: Path) -> dict[str, Any]:
    jobs = [document for document in documents if isinstance(document, dict) and document.get("kind") == "Job"]
    if len(jobs) != 1:
        raise ValueError(f"{source}: expected exactly one Job document, found {len(jobs)}")
    job = jobs[0]
    name = job.get("metadata", {}).get("name")
    if name != SOURCE_JOB_NAME:
        raise ValueError(
            f"{source}: expected Job metadata.name {SOURCE_JOB_NAME!r}, found {name!r}"
        )
    return job


def _validate_job(job: dict[str, Any], source: str) -> None:
    errors = validate_document(job, source)
    if errors:
        raise ValueError("rendered Job validation failed:\n" + "\n".join(errors))


def _reserve(path: Path) -> BinaryIO:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    return os.fdopen(descriptor, "wb")


def _write_pair_create_only(
    output: Path, output_bytes: bytes, receipt: Path, receipt_bytes: bytes
) -> None:
    if output.absolute() == receipt.absolute():
        raise ValueError("output and receipt paths must differ")
    for path in (output, receipt):
        if not path.parent.is_dir():
            raise ValueError(f"parent directory does not exist: {path.parent}")

    output_handle: BinaryIO | None = None
    receipt_handle: BinaryIO | None = None
    owned: list[Path] = []
    try:
        output_handle = _reserve(output)
        owned.append(output)
        receipt_handle = _reserve(receipt)
        owned.append(receipt)
        output_handle.write(output_bytes)
        output_handle.flush()
        os.fsync(output_handle.fileno())
        receipt_handle.write(receipt_bytes)
        receipt_handle.flush()
        os.fsync(receipt_handle.fileno())
    except Exception:
        for handle in (output_handle, receipt_handle):
            if handle is not None:
                handle.close()
        for path in reversed(owned):
            path.unlink(missing_ok=True)
        raise
    finally:
        for handle in (output_handle, receipt_handle):
            if handle is not None and not handle.closed:
                handle.close()


def derive(source: Path, output: Path, receipt: Path) -> dict[str, object]:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"source must be a regular non-symlink file: {source}")
    source_bytes = source.read_bytes()
    source_job = _select_source_job(_load_documents(source_bytes, source), source)
    _validate_job(source_job, f"{source}:Job/{SOURCE_JOB_NAME}")

    destination_job = copy.deepcopy(source_job)
    destination_job["metadata"]["name"] = DESTINATION_JOB_NAME
    _validate_job(destination_job, f"{output}:Job/{DESTINATION_JOB_NAME}")

    source_canonical = _canonical_bytes(source_job)
    destination_canonical = _canonical_bytes(destination_job)
    restored_job = copy.deepcopy(destination_job)
    restored_job["metadata"]["name"] = SOURCE_JOB_NAME
    restored_canonical = _canonical_bytes(restored_job)
    if restored_canonical != source_canonical:
        raise ValueError(f"derived Job changed fields other than {ALLOWED_CHANGE}")

    output_text = yaml.safe_dump_all(
        [destination_job], explicit_start=True, sort_keys=False, allow_unicode=True
    )
    output_bytes = output_text.encode("utf-8")
    round_trip = _load_documents(output_bytes, output)
    if len(round_trip) != 1 or not isinstance(round_trip[0], dict):
        raise ValueError("derived output did not round-trip as exactly one mapping document")
    round_trip_job = round_trip[0]
    _validate_job(round_trip_job, f"{output}:round-trip")
    if _canonical_bytes(round_trip_job) != destination_canonical:
        raise ValueError("derived output changed during YAML serialization")

    result: dict[str, object] = {
        "schema_version": 1,
        "operation": "derive-cifar10.1-v6-benchmark-job",
        "source": {
            "path": str(source.resolve()),
            "file_sha256": _sha256(source_bytes),
            "job_name": SOURCE_JOB_NAME,
            "job_canonical_sha256": _sha256(source_canonical),
        },
        "destination": {
            "path": str(output.absolute()),
            "file_sha256": _sha256(output_bytes),
            "job_name": DESTINATION_JOB_NAME,
            "job_canonical_sha256": _sha256(destination_canonical),
        },
        "equivalence": {
            "allowed_change": ALLOWED_CHANGE,
            "restored_destination_canonical_sha256": _sha256(restored_canonical),
            "canonical_bytes_equal_after_name_restore": True,
        },
        "validation": {
            "validator": "cloud/kserve/scripts/validate_rendered.py:validate_document",
            "source_valid": True,
            "destination_valid": True,
            "round_trip_valid": True,
        },
        "valid": True,
    }
    receipt_bytes = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_pair_create_only(output, output_bytes, receipt, receipt_bytes)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract pertinence-loadgen, change only metadata.name for CIFAR-10.1, "
            "and create a validation receipt"
        )
    )
    parser.add_argument("--source", required=True, type=Path, help="rendered 10-workloads.yaml")
    parser.add_argument("--output", required=True, type=Path, help="new single-Job YAML path")
    parser.add_argument("--receipt", required=True, type=Path, help="new receipt JSON path")
    args = parser.parse_args()
    try:
        result = derive(args.source, args.output, args.receipt)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
