"""Create-only online records for sealed offline/online parity checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from ..pipeline_artifacts import atomic_directory, safe_path, write_json
from ..reproducibility import canonical_json_hash, sha256_file
from .benchmark import KServeV2Predictor
from .parity import ARMS, CIFAR101_TRACE_ID, FINAL_TRACE_ID
from .trace import verify_trace


RECORDS_FILE = "records.jsonl"
RECEIPT_FILE = "receipt.json"
SAMPLES = 2_000
DIRECT_ARMS = ARMS[:3]
STATIC_ARM = "static-resnet-router"
DISPATCHER_ARM = "dispatcher-000733"
EXPERT_BY_ROUTE = dict(enumerate(DIRECT_ARMS))
RECORD_KEYS = {
    "arm",
    "repeat",
    "sample_index",
    "sample_id",
    "prediction",
    "route",
    "logits",
    "identity",
}
ARM_RECEIPT_KEYS = {
    "schema_version",
    "kind",
    "arm",
    "repeat",
    "samples",
    "trace_id",
    "trace_fingerprint",
    "endpoint",
    "expected_identity",
    "records",
    "records_sha256",
    "fingerprint",
}


class ClosablePredictor(Protocol):
    def __call__(self, image: np.ndarray) -> Mapping[str, object]: ...

    def close(self) -> None: ...


PredictorFactory = Callable[[str, str, float], ClosablePredictor]


def _default_predictor_factory(
    endpoint: str, model_name: str, timeout_seconds: float
) -> ClosablePredictor:
    return KServeV2Predictor(endpoint, model_name, timeout_seconds=timeout_seconds)


def _sealed(value: dict[str, object]) -> dict[str, object]:
    return {**value, "fingerprint": canonical_json_hash(value)}


def _read_sealed(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"sealed receipt is missing or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("sealed receipt must be a JSON object")
    fingerprint = value.get("fingerprint")
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    if fingerprint != canonical_json_hash(body):
        raise ValueError("parity collection receipt fingerprint mismatch")
    return value


def _validate_trace(trace_root: Path) -> tuple[dict[str, object], np.ndarray, list[str]]:
    manifest, images, _labels, sample_ids = verify_trace(trace_root)
    expected_dataset = {
        FINAL_TRACE_ID: {"id": "cifar10", "version": "final-20260824"},
        CIFAR101_TRACE_ID: {"id": "cifar10.1", "version": "v6"},
    }
    trace_id = manifest.get("trace_id")
    if (
        trace_id not in expected_dataset
        or manifest.get("dataset") != expected_dataset[trace_id]
        or len(images) != SAMPLES
        or len(sample_ids) != SAMPLES
    ):
        raise ValueError("online parity requires an approved 2,000-row benchmark trace")
    return manifest, images, sample_ids


def _expected_identity_keys(arm: str) -> set[str]:
    if arm in DIRECT_ARMS:
        return {"model_name", "model_version"}
    if arm == STATIC_ARM:
        return {
            "model_name",
            "model_version",
            "promotion_id",
            "dispatcher_id",
            "expert_id",
        }
    if arm == DISPATCHER_ARM:
        return {"model_name", "model_version", "promotion_id", "dispatcher_id"}
    raise ValueError(f"unsupported parity arm: {arm}")


def _validate_expected_identity(arm: str, value: Mapping[str, str]) -> dict[str, str]:
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()):
        raise ValueError("expected identity keys and values must be strings")
    identity = dict(value)
    if set(identity) != _expected_identity_keys(arm):
        raise ValueError(f"expected identity keys are invalid for {arm}")
    if any(not key or not item or "\n" in item or "\r" in item for key, item in identity.items()):
        raise ValueError("expected identity values must be non-empty single-line strings")
    if identity["model_name"] != arm or identity["model_version"] != "1":
        raise ValueError("expected model identity does not match the selected arm")
    if arm == STATIC_ARM and (
        identity["dispatcher_id"] != "static" or identity["expert_id"] != "resnet56"
    ):
        raise ValueError("static router identity must select resnet56 through dispatcher static")
    if arm == DISPATCHER_ARM and identity["dispatcher_id"] != DISPATCHER_ARM:
        raise ValueError("dispatcher identity does not match dispatcher-000733")
    return identity


def _record_from_result(
    *,
    arm: str,
    index: int,
    sample_id: str,
    result: Mapping[str, object],
    expected_identity: Mapping[str, str],
) -> dict[str, object]:
    if not isinstance(result, Mapping):
        raise ValueError(f"predictor result is not a mapping for {arm} sample {index}")
    logits = np.asarray(result.get("logits"))
    if logits.shape != (1, 10) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ValueError(f"invalid logits for {arm} sample {index}")
    route = result.get("route")
    if arm in DIRECT_ARMS:
        if route is not None:
            raise ValueError(f"direct expert {arm} returned a route")
    elif arm == STATIC_ARM:
        if type(route) is not int or route != 2:
            raise ValueError("static-resnet-router must return route 2")
    elif type(route) is not int or route not in EXPERT_BY_ROUTE:
        raise ValueError("dispatcher-000733 returned an invalid route")

    raw_identity = result.get("identity")
    if not isinstance(raw_identity, Mapping):
        raise ValueError(f"invalid identity for {arm} sample {index}")
    if any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in raw_identity.items()
    ):
        raise ValueError(f"identity values are not strings for {arm} sample {index}")
    identity = dict(raw_identity)
    response_keys = _expected_identity_keys(arm) | ({"expert_id"} if arm == DISPATCHER_ARM else set())
    if set(identity) != response_keys:
        raise ValueError(f"response identity keys are invalid for {arm}")
    if any(identity.get(key) != item for key, item in expected_identity.items()):
        raise ValueError(f"identity mismatch for {arm} sample {index}")
    if arm == DISPATCHER_ARM and identity["expert_id"] != EXPERT_BY_ROUTE[route]:
        raise ValueError(f"dispatcher expert identity disagrees with route at sample {index}")

    flat_logits = logits.reshape(10)
    return {
        "arm": arm,
        "repeat": 0,
        "sample_index": index,
        "sample_id": sample_id,
        "prediction": int(np.argmax(flat_logits)),
        "route": route,
        "logits": flat_logits.tolist(),
        "identity": identity,
    }


def _write_records(path: Path, records: list[Mapping[str, object]]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validate_record(
    record: Mapping[str, Any],
    *,
    arm: str,
    index: int,
    sample_id: str,
    expected_identity: Mapping[str, str],
) -> None:
    if set(record) != RECORD_KEYS:
        raise ValueError(f"online parity record keys are invalid for {arm}")
    if (
        record.get("arm") != arm
        or record.get("repeat") != 0
        or record.get("sample_index") != index
        or record.get("sample_id") != sample_id
    ):
        raise ValueError(f"online parity sample order is invalid for {arm}")
    prediction = record.get("prediction")
    raw_logits = record.get("logits")
    if (
        not isinstance(raw_logits, list)
        or len(raw_logits) != 10
        or any(type(item) not in (int, float) for item in raw_logits)
    ):
        raise ValueError(f"online parity logits are invalid for {arm}")
    logits = np.asarray(raw_logits, dtype=np.float64)
    if (
        type(prediction) is not int
        or not 0 <= prediction < 10
        or logits.shape != (10,)
        or not np.issubdtype(logits.dtype, np.number)
        or not np.isfinite(logits).all()
        or prediction != int(np.argmax(logits))
    ):
        raise ValueError(f"online parity prediction/logits are invalid for {arm}")
    route = record.get("route")
    identity = record.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"online parity identity is invalid for {arm}")
    if any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in identity.items()
    ):
        raise ValueError(f"online parity identity values are invalid for {arm}")
    response_keys = _expected_identity_keys(arm) | ({"expert_id"} if arm == DISPATCHER_ARM else set())
    if set(identity) != response_keys or any(
        identity.get(key) != item for key, item in expected_identity.items()
    ):
        raise ValueError(f"online parity identity does not match its receipt for {arm}")
    if arm in DIRECT_ARMS and route is not None:
        raise ValueError(f"direct expert record contains a route for {arm}")
    if arm == STATIC_ARM and (type(route) is not int or route != 2):
        raise ValueError("static router record must contain route 2")
    if arm == DISPATCHER_ARM:
        if type(route) is not int or route not in EXPERT_BY_ROUTE:
            raise ValueError("dispatcher record contains an invalid route")
        if identity.get("expert_id") != EXPERT_BY_ROUTE[route]:
            raise ValueError("dispatcher record identity disagrees with its route")


def collect_parity_arm(
    trace_root: Path,
    *,
    arm: str,
    endpoint: str,
    expected_identity: Mapping[str, str],
    output: Path,
    timeout_seconds: float = 10.0,
    predictor_factory: PredictorFactory = _default_predictor_factory,
) -> dict[str, object]:
    """Send every sealed trace sample once and atomically seal one arm's records."""

    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    if not isinstance(endpoint, str) or not endpoint or any(item in endpoint for item in "\r\n"):
        raise ValueError("endpoint must be a non-empty single-line URL")
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive and finite")
    if arm not in ARMS:
        raise ValueError(f"unsupported parity arm: {arm}")
    identity = _validate_expected_identity(arm, expected_identity)
    manifest, images, sample_ids = _validate_trace(trace_root)

    predictor = predictor_factory(endpoint, arm, float(timeout_seconds))
    records: list[Mapping[str, object]] = []
    try:
        for index, (image, sample_id) in enumerate(zip(images, sample_ids, strict=True)):
            result = predictor(image[np.newaxis, ...])
            records.append(
                _record_from_result(
                    arm=arm,
                    index=index,
                    sample_id=sample_id,
                    result=result,
                    expected_identity=identity,
                )
            )
    finally:
        predictor.close()

    with atomic_directory(output) as stage:
        records_path = stage / RECORDS_FILE
        _write_records(records_path, records)
        receipt = _sealed(
            {
                "schema_version": 1,
                "kind": "online-parity-arm",
                "arm": arm,
                "repeat": 0,
                "samples": SAMPLES,
                "trace_id": manifest["trace_id"],
                "trace_fingerprint": manifest["fingerprint"],
                "endpoint": endpoint,
                "expected_identity": identity,
                "records": RECORDS_FILE,
                "records_sha256": sha256_file(records_path),
            }
        )
        write_json(stage / RECEIPT_FILE, receipt)
    return receipt


def _read_arm_output(
    root: Path,
    *,
    arm: str,
    trace_id: str,
    trace_fingerprint: str,
    sample_ids: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"arm output is missing or unsafe: {root}")
    if {item.name for item in root.iterdir()} != {RECORDS_FILE, RECEIPT_FILE}:
        raise ValueError(f"arm output contains unexpected entries: {root}")
    receipt = _read_sealed(safe_path(root, RECEIPT_FILE))
    if (
        set(receipt) != ARM_RECEIPT_KEYS
        or receipt.get("schema_version") != 1
        or receipt.get("kind") != "online-parity-arm"
        or receipt.get("arm") != arm
        or receipt.get("repeat") != 0
        or receipt.get("samples") != SAMPLES
        or receipt.get("trace_id") != trace_id
        or receipt.get("trace_fingerprint") != trace_fingerprint
        or receipt.get("records") != RECORDS_FILE
        or not isinstance(receipt.get("endpoint"), str)
        or not receipt["endpoint"]
    ):
        raise ValueError(f"arm receipt contract mismatch for {arm}")
    expected_identity = receipt.get("expected_identity")
    if not isinstance(expected_identity, Mapping):
        raise ValueError(f"arm receipt identity is invalid for {arm}")
    expected_identity = _validate_expected_identity(arm, expected_identity)
    records_path = safe_path(root, RECORDS_FILE)
    if not records_path.is_file() or records_path.is_symlink():
        raise ValueError(f"records are missing or unsafe for {arm}")
    if sha256_file(records_path) != receipt.get("records_sha256"):
        raise ValueError(f"records SHA-256 mismatch for {arm}")
    records: list[dict[str, Any]] = []
    with records_path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= SAMPLES or not line.strip():
                raise ValueError(f"record count or encoding is invalid for {arm}")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"online parity record is not an object for {arm}")
            _validate_record(
                record,
                arm=arm,
                index=index,
                sample_id=sample_ids[index],
                expected_identity=expected_identity,
            )
            records.append(record)
    if len(records) != SAMPLES:
        raise ValueError(f"arm output does not contain exactly {SAMPLES} records for {arm}")
    return receipt, records


def combine_parity_arms(
    trace_root: Path, *, inputs: Mapping[str, Path], output: Path
) -> dict[str, object]:
    """Combine five verified arm outputs without contacting serving endpoints."""

    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    if list(inputs) != list(ARMS):
        raise ValueError("arm inputs must contain the five approved arms in canonical order")
    manifest, _images, sample_ids = _validate_trace(trace_root)
    all_records: list[Mapping[str, object]] = []
    input_receipts: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        receipt, records = _read_arm_output(
            Path(inputs[arm]),
            arm=arm,
            trace_id=str(manifest["trace_id"]),
            trace_fingerprint=str(manifest["fingerprint"]),
            sample_ids=sample_ids,
        )
        all_records.extend(records)
        input_receipts[arm] = {
            "receipt_fingerprint": receipt["fingerprint"],
            "records_sha256": receipt["records_sha256"],
        }

    with atomic_directory(output) as stage:
        records_path = stage / RECORDS_FILE
        _write_records(records_path, all_records)
        receipt = _sealed(
            {
                "schema_version": 1,
                "kind": "online-parity-combined",
                "arms": list(ARMS),
                "repeat": 0,
                "samples_per_arm": SAMPLES,
                "records_total": len(all_records),
                "trace_id": manifest["trace_id"],
                "trace_fingerprint": manifest["fingerprint"],
                "inputs": input_receipts,
                "records": RECORDS_FILE,
                "records_sha256": sha256_file(records_path),
            }
        )
        write_json(stage / RECEIPT_FILE, receipt)
    return receipt
