"""Immutable benchmark trace creation and verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts.common import digest, name
from ..pipeline_artifacts import safe_path
from ..reproducibility import canonical_json_hash, sha256_file
from .contracts import BenchmarkTraceV1


FINAL_SAMPLE_ORDER_SHA256 = "323c5f237cf1a7dd81d5e726ae684c067dd28213c6a1460b023801f3c05f07f5"
CIFAR101_REVISION = "d9982abb0bfc4846b8d13a11e66b887d946205d0"
CIFAR101_DATA_SHA256 = "2997188e5816f5bd545dc77771b6227828c28146049fcecf3fa10775474cacc6"
CIFAR101_LABEL_SHA256 = "ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448"
CIFAR101_DATA_SIZE = 6_144_128
CIFAR101_LABEL_SIZE = 8_128
CIFAR101_LABEL_VALUES_SHA256 = "2b1636e92dc42ddc27bf84f4aa6bfba816ceb71844af81bbd0eff6d6474e5b72"
NORMALIZE_MEAN = np.asarray([0.4914, 0.4822, 0.4465], dtype=np.float32)[:, None, None]
NORMALIZE_STD = np.asarray([0.2023, 0.1994, 0.2010], dtype=np.float32)[:, None, None]


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _validate_source_objects(objects: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not objects:
        raise ValueError("trace provenance requires at least one source object")
    result = []
    for item in objects:
        required = {"uri", "sha256", "size_bytes"}
        if not required.issubset(item) or set(item) - required - {"generation", "revision"}:
            raise ValueError("source objects require URI, SHA-256, size, and generation or revision")
        if ("generation" in item) == ("revision" in item):
            raise ValueError("source objects require exactly one of generation or revision")
        uri = item["uri"]
        if not isinstance(uri, str) or not uri or any(character in uri for character in "\r\n"):
            raise ValueError("source object URI must be a non-empty single-line string")
        digest(item["sha256"])
        size = item["size_bytes"]
        if type(size) is not int or size < 0:
            raise ValueError("source object size_bytes must be a non-negative integer")
        result.append(dict(item))
    return result


def build_trace_manifest(
    *,
    trace_id: str,
    dataset_id: str,
    dataset_version: str,
    images: np.ndarray,
    labels: np.ndarray,
    sample_ids: Sequence[str],
    source_objects: Sequence[Mapping[str, Any]],
    provenance: str,
    usage_conditions: str,
) -> dict[str, object]:
    """Build metadata; labels are represented only by a hash in this public contract."""

    name(trace_id, "trace_id")
    name(dataset_id, "dataset_id")
    name(dataset_version, "dataset_version")
    values = np.asarray(images)
    targets = np.asarray(labels)
    identifiers = [str(item) for item in sample_ids]
    if values.dtype != np.float32 or values.ndim != 4 or values.shape[1:] != (3, 32, 32):
        raise ValueError("trace images must be normalized float32 [N,3,32,32]")
    if not np.isfinite(values).all():
        raise ValueError("trace images contain non-finite values")
    if targets.dtype != np.int64 or targets.shape != (values.shape[0],):
        raise ValueError("trace labels must be int64 [N]")
    if values.shape[0] < 1 or np.any(targets < 0) or np.any(targets >= 10):
        raise ValueError("trace must contain labeled CIFAR-10 samples")
    if len(identifiers) != values.shape[0] or len(set(identifiers)) != len(identifiers):
        raise ValueError("sample IDs must be unique and align with images")
    if any(not item or "\n" in item or "\r" in item for item in identifiers):
        raise ValueError("sample IDs must be non-empty single-line strings")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("provenance must be non-empty")
    if not isinstance(usage_conditions, str) or not usage_conditions.strip():
        raise ValueError("usage_conditions must be non-empty")
    body: dict[str, object] = {
        "schema_version": 1,
        "kind": "benchmark-trace",
        "trace_id": trace_id,
        "dataset": {"id": dataset_id, "version": dataset_version},
        "samples": int(values.shape[0]),
        "input": {"name": "image", "datatype": "FP32", "shape": [1, 3, 32, 32]},
        "source_objects": _validate_source_objects(source_objects),
        "provenance": provenance,
        "usage_conditions": usage_conditions,
        "sample_order_sha256": canonical_json_hash(identifiers),
        "label_sha256": _array_hash(targets),
        "image_sha256": _array_hash(values),
        "class_counts": [int(np.sum(targets == index)) for index in range(10)],
        "payload": "trace.npz",
    }
    manifest = {**body, "fingerprint": canonical_json_hash(body)}
    BenchmarkTraceV1.from_dict(manifest)
    return manifest


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_trace(
    output: Path,
    *,
    trace_id: str,
    dataset_id: str,
    dataset_version: str,
    images: np.ndarray,
    labels: np.ndarray,
    sample_ids: Sequence[str],
    source_objects: Sequence[Mapping[str, Any]],
    provenance: str,
    usage_conditions: str,
) -> dict[str, object]:
    """Create a trace once; an existing ID is accepted only for identical content."""

    manifest = build_trace_manifest(
        trace_id=trace_id,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        images=images,
        labels=labels,
        sample_ids=sample_ids,
        source_objects=source_objects,
        provenance=provenance,
        usage_conditions=usage_conditions,
    )
    output = Path(output)
    manifest_path = output / "trace.json"
    if output.exists():
        if not output.is_dir() or output.is_symlink() or not manifest_path.is_file():
            raise ValueError("trace output already exists but is not a valid trace")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("trace ID already exists with different content")
        verify_trace(output)
        return manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write_npz_atomic(
            temporary / "trace.npz",
            images=np.asarray(images, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.int64),
            sample_ids=np.asarray(list(sample_ids), dtype=np.str_),
        )
        with (temporary / "trace.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    return manifest


def verify_trace(root: Path) -> tuple[dict[str, object], np.ndarray, np.ndarray, list[str]]:
    root = Path(root)
    manifest_path = safe_path(root, "trace.json")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = value.get("fingerprint")
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    if fingerprint != canonical_json_hash(body):
        raise ValueError("trace manifest fingerprint mismatch")
    BenchmarkTraceV1.from_dict(value)
    payload = safe_path(root, str(value.get("payload")))
    if payload.is_symlink() or not payload.is_file():
        raise ValueError("trace payload is missing or unsafe")
    with np.load(payload, allow_pickle=False) as archive:
        if set(archive.files) != {"images", "labels", "sample_ids"}:
            raise ValueError("trace payload contains unexpected arrays")
        images = archive["images"]
        labels = archive["labels"]
        sample_ids = [str(item) for item in archive["sample_ids"].tolist()]
    rebuilt = build_trace_manifest(
        trace_id=str(value["trace_id"]),
        dataset_id=str(value["dataset"]["id"]),
        dataset_version=str(value["dataset"]["version"]),
        images=images,
        labels=labels,
        sample_ids=sample_ids,
        source_objects=value["source_objects"],
        provenance=str(value["provenance"]),
        usage_conditions=str(value["usage_conditions"]),
    )
    if rebuilt != value:
        raise ValueError("trace payload differs from sealed manifest")
    return value, images, labels, sample_ids


def source_object(
    path: Path,
    *,
    uri: str | None = None,
    generation: int | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("source object must be a regular non-symlink file")
    if (generation is None) == (revision is None):
        raise ValueError("source provenance requires exactly one of generation or revision")
    result: dict[str, Any] = {
        "uri": uri or path.resolve().as_uri(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    result["generation" if generation is not None else "revision"] = (
        generation if generation is not None else revision
    )
    return result


def normalize_cifar_images(images: np.ndarray) -> np.ndarray:
    """Convert canonical uint8 CIFAR NHWC images to normalized NCHW FP32."""

    values = np.asarray(images)
    if values.dtype != np.uint8 or values.ndim != 4 or values.shape[1:] != (32, 32, 3):
        raise ValueError("raw CIFAR images must be uint8 [N,32,32,3]")
    nchw = np.transpose(values, (0, 3, 1, 2)).astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray((nchw - NORMALIZE_MEAN) / NORMALIZE_STD, dtype=np.float32)


def build_final_trace_from_npz(
    evaluation_npz: Path,
    split_manifest: Path,
    output: Path,
    *,
    expected_label_sha256: str,
    evaluation_uri: str,
    evaluation_generation: int,
    split_uri: str,
    split_generation: int,
    trace_id: str = "cifar10-final-20260824",
) -> dict[str, object]:
    """Create the exact approved 2,000-row final trace and verify offline receipts."""

    evaluation_npz = Path(evaluation_npz)
    split_manifest = Path(split_manifest)
    if not evaluation_uri.startswith("gs://") or not split_uri.startswith("gs://"):
        raise ValueError("final trace sources must use immutable GCS object URIs")
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    final_ids = split.get("sample_ids", {}).get("final")
    if not isinstance(final_ids, list) or len(final_ids) != 2_000:
        raise ValueError("split manifest must contain exactly 2,000 final sample IDs")
    if canonical_json_hash(final_ids) != FINAL_SAMPLE_ORDER_SHA256:
        raise ValueError("final sample order does not match the approved hash")
    with np.load(evaluation_npz, allow_pickle=False) as archive:
        required = {
            "images",
            "labels",
            "sample_ids",
            "shufflenetv2_x0_5_predictions",
            "mobilenetv2_x0_5_predictions",
            "resnet56_predictions",
            "dispatcher-000733_predictions",
            "dispatcher-000733_routes",
        }
        if not required.issubset(archive.files):
            raise ValueError(f"evaluation NPZ is missing arrays: {sorted(required - set(archive.files))}")
        all_ids = [str(item) for item in archive["sample_ids"].tolist()]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("evaluation sample IDs are not unique")
        positions = {sample_id: index for index, sample_id in enumerate(all_ids)}
        if any(sample_id not in positions for sample_id in final_ids):
            raise ValueError("final sample ID is absent from evaluation bundle")
        indices = np.asarray([positions[sample_id] for sample_id in final_ids], dtype=np.int64)
        raw_images = archive["images"][indices]
        labels = np.asarray(archive["labels"][indices], dtype=np.int64)
        digest(expected_label_sha256)
        if _array_hash(labels) != expected_label_sha256:
            raise ValueError("final label hash does not match the approved receipt")
        expected_correct = {
            "shufflenetv2_x0_5_predictions": 1809,
            "mobilenetv2_x0_5_predictions": 1858,
            "resnet56_predictions": 1892,
            "dispatcher-000733_predictions": 1893,
        }
        for key, expected in expected_correct.items():
            predictions = np.asarray(archive[key][indices])
            if predictions.shape != (2_000,) or int(np.sum(predictions == labels)) != expected:
                raise ValueError(f"offline accuracy receipt mismatch for {key}")
        routes = np.asarray(archive["dispatcher-000733_routes"][indices])
        if routes.shape != (2_000,) or np.bincount(routes, minlength=3).tolist() != [0, 640, 1360]:
            raise ValueError("offline dispatcher route receipt mismatch")
    images = normalize_cifar_images(raw_images)
    return create_trace(
        output,
        trace_id=trace_id,
        dataset_id="cifar10",
        dataset_version="final-20260824",
        images=images,
        labels=labels,
        sample_ids=final_ids,
        source_objects=[
            source_object(
                evaluation_npz, uri=evaluation_uri, generation=evaluation_generation
            ),
            source_object(split_manifest, uri=split_uri, generation=split_generation),
        ],
        provenance="promoted CIFAR-10 evaluation bundle and sealed final split",
        usage_conditions="CIFAR-10 upstream terms apply",
    )


def build_cifar101_v6_trace(
    data_npy: Path,
    labels_npy: Path,
    output: Path,
    *,
    trace_id: str = "cifar10.1-v6",
) -> dict[str, object]:
    """Validate commit-pinned CIFAR-10.1 v6 NPY files and create its trace."""

    data_npy, labels_npy = Path(data_npy), Path(labels_npy)
    expected = (
        (data_npy, CIFAR101_DATA_SHA256, CIFAR101_DATA_SIZE),
        (labels_npy, CIFAR101_LABEL_SHA256, CIFAR101_LABEL_SIZE),
    )
    for path, expected_sha, expected_size in expected:
        if path.is_symlink() or not path.is_file():
            raise ValueError("CIFAR-10.1 source must be a regular file")
        if path.stat().st_size != expected_size or sha256_file(path) != expected_sha:
            raise ValueError(f"CIFAR-10.1 source checksum or size mismatch: {path}")
    images = np.load(data_npy, allow_pickle=False)
    labels = np.load(labels_npy, allow_pickle=False)
    if images.shape != (2_000, 32, 32, 3) or images.dtype != np.uint8:
        raise ValueError("CIFAR-10.1 data must be uint8 [2000,32,32,3]")
    if labels.shape != (2_000,) or labels.dtype.kind not in "iu":
        raise ValueError("CIFAR-10.1 labels must be an integer [2000] vector")
    labels = labels.astype(np.int64, copy=False)
    if _array_hash(labels) != CIFAR101_LABEL_VALUES_SHA256:
        raise ValueError("CIFAR-10.1 label value hash mismatch")
    if np.bincount(labels, minlength=10).tolist() != [200] * 10:
        raise ValueError("CIFAR-10.1 v6 must contain 200 samples per class")
    sample_ids = [f"cifar10.1-v6-{index:04d}" for index in range(2_000)]
    return create_trace(
        output,
        trace_id=trace_id,
        dataset_id="cifar10.1",
        dataset_version="v6",
        images=normalize_cifar_images(images),
        labels=labels,
        sample_ids=sample_ids,
        source_objects=[
            source_object(
                data_npy,
                uri=(
                    "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/"
                    f"{CIFAR101_REVISION}/datasets/cifar10.1_v6_data.npy"
                ),
                revision=CIFAR101_REVISION,
            ),
            source_object(
                labels_npy,
                uri=(
                    "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/"
                    f"{CIFAR101_REVISION}/datasets/cifar10.1_v6_labels.npy"
                ),
                revision=CIFAR101_REVISION,
            ),
        ],
        provenance=f"modestyachts/CIFAR-10.1 commit {CIFAR101_REVISION}",
        usage_conditions="CIFAR-10.1 and Tiny Images upstream terms apply",
    )
