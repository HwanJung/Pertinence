"""Exact offline/online prediction parity for immutable benchmark traces."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts.common import digest
from .trace import verify_trace


ARMS = (
    "shufflenetv2_x0_5",
    "mobilenetv2_x0_5",
    "resnet56",
    "static-resnet-router",
    "dispatcher-000733",
)
ONLINE_ARMS = ("resnet56", "dispatcher-000733")
FINAL_TRACE_ID = "cifar10-final-20260824"
CIFAR101_TRACE_ID = "cifar10.1-v6"
EXPECTED_CORRECT = {
    "shufflenetv2_x0_5": 1809,
    "mobilenetv2_x0_5": 1858,
    "resnet56": 1892,
    "static-resnet-router": 1892,
    "dispatcher-000733": 1893,
}
EXPECTED_FINAL_ROUTE_COUNTS = [0, 640, 1360]


def _array_hash(value: np.ndarray) -> str:
    """Hash an array without depending on a particular NPZ byte encoding."""

    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _validate_provenance(
    promotion_fingerprint: str | None, image_digests: Mapping[str, str] | None
) -> tuple[str | None, dict[str, str]]:
    if promotion_fingerprint is not None:
        digest(promotion_fingerprint)
    images: dict[str, str] = {}
    for image, value in sorted((image_digests or {}).items()):
        if not image or any(character in image for character in "\r\n"):
            raise ValueError("image digest names must be non-empty single-line strings")
        if not isinstance(value, str) or not value.startswith("sha256:"):
            raise ValueError("image digests must use sha256:<64-hex> form")
        digest(value.removeprefix("sha256:"))
        images[image] = value
    return promotion_fingerprint, images


def _online_by_arm(
    online_records: Sequence[Mapping[str, Any]], samples: int
) -> dict[str, dict[int, Mapping[str, Any]]]:
    by_arm: dict[str, dict[int, Mapping[str, Any]]] = {arm: {} for arm in ONLINE_ARMS}
    for record in online_records:
        arm = record.get("arm")
        repeat = record.get("repeat")
        # Full benchmark JSONL is supported; parity intentionally uses repeat zero.
        if arm not in by_arm or repeat != 0:
            continue
        index = record.get("sample_index")
        if type(index) is not int or not 0 <= index < samples:
            raise ValueError("online parity sample index is invalid")
        if index in by_arm[arm]:
            raise ValueError("duplicate online parity record")
        by_arm[arm][index] = record
    return by_arm


def _golden_logits_receipt(
    archive: Mapping[str, np.ndarray],
    by_arm: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    samples: int,
    sample_index: int,
    atol: float,
) -> dict[str, object]:
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("golden logits tolerance must be finite and non-negative")
    if not 0 <= sample_index < samples:
        raise ValueError("golden logits sample index is out of range")
    max_errors: dict[str, float] = {}
    for arm in ONLINE_ARMS:
        key = f"{arm}_logits"
        if key not in archive:
            raise ValueError(f"local parity NPZ is missing golden logits for {arm}")
        local = np.asarray(archive[key])
        if local.shape != (samples, 10) or not np.isfinite(local).all():
            raise ValueError(f"local logits are invalid for {arm}")
        online = np.asarray(by_arm[arm][sample_index].get("logits"), dtype=np.float32)
        if online.shape != (10,) or not np.isfinite(online).all():
            raise ValueError(f"online golden logits are invalid for {arm}")
        error = float(np.max(np.abs(local[sample_index].astype(np.float64) - online)))
        if error > atol:
            raise ValueError(f"golden logits mismatch for {arm}")
        max_errors[arm] = error
    return {
        "sample_index": sample_index,
        "absolute_tolerance": atol,
        "max_absolute_error": max_errors,
    }


def verify_offline_online_parity(
    trace_root: Path,
    local_npz: Path,
    online_records: Sequence[Mapping[str, Any]],
    *,
    promotion_fingerprint: str | None = None,
    image_digests: Mapping[str, str] | None = None,
    golden_logits_atol: float | None = None,
    golden_sample_index: int = 0,
) -> dict[str, object]:
    """Verify exact predictions, routes, and ordering for a sealed trace.

    Approved accuracy and route counts apply only to the named CIFAR-10 final
    trace. Other 2,000-row traces get the same exact comparisons without a
    predeclared accuracy claim.
    """

    manifest, _images, labels, sample_ids = verify_trace(trace_root)
    samples = len(labels)
    if samples != 2_000:
        raise ValueError("offline parity requires an approved 2,000-row benchmark trace")
    trace_id = manifest["trace_id"]
    expected_dataset = {
        FINAL_TRACE_ID: {"id": "cifar10", "version": "final-20260824"},
        CIFAR101_TRACE_ID: {"id": "cifar10.1", "version": "v6"},
    }
    if trace_id not in expected_dataset:
        raise ValueError("parity supports only the two approved benchmark traces")
    if manifest.get("dataset") != expected_dataset[trace_id]:
        raise ValueError("benchmark trace identity metadata mismatch")
    is_approved_final = trace_id == FINAL_TRACE_ID
    promotion_fingerprint, image_digests = _validate_provenance(
        promotion_fingerprint, image_digests
    )
    by_arm = _online_by_arm(online_records, samples)
    prediction_hashes: dict[str, str] = {}
    accuracy: dict[str, int] = {}
    with np.load(local_npz, allow_pickle=False) as archive:
        required = {f"{arm}_predictions" for arm in ARMS} | {"dispatcher-000733_routes"}
        allowed = required | (
            {f"{arm}_logits" for arm in ARMS} if golden_logits_atol is not None else set()
        )
        if set(archive.files) != allowed:
            raise ValueError("local parity NPZ keys differ from the expected set")
        for arm in ONLINE_ARMS:
            local = np.asarray(archive[f"{arm}_predictions"])
            if (
                local.shape != (samples,)
                or not np.issubdtype(local.dtype, np.integer)
                or np.any(local < 0)
                or np.any(local >= 10)
                or len(by_arm[arm]) != samples
            ):
                raise ValueError(f"incomplete or invalid local/online predictions for {arm}")
            try:
                online = np.asarray(
                    [by_arm[arm][index]["prediction"] for index in range(samples)]
                )
            except KeyError as error:
                raise ValueError(f"online prediction is missing for {arm}") from error
            if not np.issubdtype(online.dtype, np.integer) or not np.array_equal(local, online):
                raise ValueError(f"local/online prediction mismatch for {arm}")
            correct = int(np.sum(local == labels))
            if is_approved_final and correct != EXPECTED_CORRECT[arm]:
                raise ValueError(f"approved offline accuracy mismatch for {arm}")
            if any(
                str(by_arm[arm][index].get("sample_id")) != sample_ids[index]
                for index in range(samples)
            ):
                raise ValueError(f"sample identity mismatch for {arm}")
            accuracy[arm] = correct
            prediction_hashes[arm] = _array_hash(local)
        routes = np.asarray(archive["dispatcher-000733_routes"])
        if (
            routes.shape != (samples,)
            or not np.issubdtype(routes.dtype, np.integer)
            or np.any(routes < 0)
            or np.any(routes >= 3)
        ):
            raise ValueError("local dispatcher routes are invalid")
        online_routes = np.asarray(
            [by_arm["dispatcher-000733"][index].get("route") for index in range(samples)]
        )
        if not np.issubdtype(online_routes.dtype, np.integer) or not np.array_equal(
            routes, online_routes
        ):
            raise ValueError("dispatcher route parity mismatch")
        route_counts = np.bincount(routes, minlength=3).tolist()
        if is_approved_final and route_counts != EXPECTED_FINAL_ROUTE_COUNTS:
            raise ValueError("approved dispatcher route count mismatch")
        golden = None
        if golden_logits_atol is not None:
            golden = _golden_logits_receipt(
                archive,
                by_arm,
                samples=samples,
                sample_index=golden_sample_index,
                atol=golden_logits_atol,
            )
    receipt: dict[str, object] = {
        "schema_version": 2,
        "matched": True,
        "trace_fingerprint": manifest["fingerprint"],
        "samples": samples,
        "prediction_sha256": prediction_hashes,
        "dispatcher_route_sha256": _array_hash(routes),
        "accuracy_correct": accuracy,
        "approved_accuracy_checked": is_approved_final,
        "route_counts": route_counts,
        "approved_route_counts_checked": is_approved_final,
        "promotion_fingerprint": promotion_fingerprint,
        "image_digests": image_digests,
    }
    if golden is not None:
        receipt["golden_logits"] = golden
    return receipt


def write_parity_receipt(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(encoded)
