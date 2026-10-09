from __future__ import annotations

import numpy as np
import pytest

from pertinence.serving.parity import (
    ARMS,
    EXPECTED_CORRECT,
    ONLINE_ARMS,
    verify_offline_online_parity,
)
from pertinence.serving.trace import create_trace


SAMPLES = 2_000


def _trace(tmp_path, trace_id: str):
    root = tmp_path / trace_id
    create_trace(
        root,
        trace_id=trace_id,
        dataset_id="cifar10" if trace_id.startswith("cifar10-final") else "cifar10.1",
        dataset_version="final-20260824" if trace_id.startswith("cifar10-final") else "v6",
        images=np.zeros((SAMPLES, 3, 32, 32), dtype=np.float32),
        labels=np.zeros(SAMPLES, dtype=np.int64),
        sample_ids=[f"sample-{index}" for index in range(SAMPLES)],
        source_objects=[
            {
                "uri": "gs://bucket/trace",
                "sha256": "a" * 64,
                "size_bytes": 1,
                "generation": 1,
            }
        ],
        provenance="sealed test trace",
        usage_conditions="test only",
    )
    return root


def _parity_inputs(tmp_path, correct_by_arm, *, logits: bool = False):
    arrays = {}
    records = []
    for arm_index, arm in enumerate(ARMS):
        predictions = np.ones(SAMPLES, dtype=np.int64)
        predictions[: correct_by_arm[arm]] = 0
        arrays[f"{arm}_predictions"] = predictions
        if logits:
            arm_logits = np.full((SAMPLES, 10), arm_index, dtype=np.float32)
            arrays[f"{arm}_logits"] = arm_logits
        for index, prediction in enumerate(predictions):
            if arm not in ONLINE_ARMS:
                continue
            record = {
                "arm": arm,
                "repeat": 0,
                "sample_index": index,
                "sample_id": f"sample-{index}",
                "prediction": int(prediction),
                "route": 1 if index < 640 else 2 if arm == "dispatcher-000733" else None,
            }
            if logits:
                record["logits"] = arrays[f"{arm}_logits"][index].tolist()
            records.append(record)
    arrays["dispatcher-000733_routes"] = np.asarray(
        [1] * 640 + [2] * 1_360, dtype=np.int64
    )
    path = tmp_path / "local.npz"
    np.savez(path, **arrays)
    return path, records


def test_final_trace_retains_approved_accuracy_routes_and_hashes(tmp_path) -> None:
    trace = _trace(tmp_path, "cifar10-final-20260824")
    local, records = _parity_inputs(tmp_path, EXPECTED_CORRECT)

    receipt = verify_offline_online_parity(
        trace,
        local,
        records,
        promotion_fingerprint="b" * 64,
        image_digests={"runtime": "sha256:" + "c" * 64},
    )

    assert receipt["matched"] is True
    assert receipt["samples"] == SAMPLES
    assert receipt["accuracy_correct"] == {
        arm: EXPECTED_CORRECT[arm] for arm in ONLINE_ARMS
    }
    assert receipt["approved_accuracy_checked"] is True
    assert receipt["route_counts"] == [0, 640, 1360]
    assert set(receipt["prediction_sha256"]) == set(ONLINE_ARMS)
    assert len(receipt["dispatcher_route_sha256"]) == 64
    assert receipt["promotion_fingerprint"] == "b" * 64
    assert receipt["image_digests"] == {"runtime": "sha256:" + "c" * 64}


def test_cifar101_checks_exact_parity_without_predeclared_accuracy(tmp_path) -> None:
    trace = _trace(tmp_path, "cifar10.1-v6")
    observed = {arm: 100 + index for index, arm in enumerate(ARMS)}
    local, records = _parity_inputs(tmp_path, observed, logits=True)

    receipt = verify_offline_online_parity(
        trace, local, records, golden_logits_atol=1e-6, golden_sample_index=17
    )

    assert receipt["accuracy_correct"] == {arm: observed[arm] for arm in ONLINE_ARMS}
    assert receipt["approved_accuracy_checked"] is False
    assert receipt["route_counts"] == [0, 640, 1360]
    assert receipt["golden_logits"]["sample_index"] == 17
    assert receipt["golden_logits"]["max_absolute_error"] == {
        arm: 0.0 for arm in ONLINE_ARMS
    }


def test_cifar101_rejects_prediction_or_sample_order_mismatch(tmp_path) -> None:
    trace = _trace(tmp_path, "cifar10.1-v6")
    observed = {arm: 100 for arm in ARMS}
    local, records = _parity_inputs(tmp_path, observed)
    next(item for item in records if item["arm"] == "dispatcher-000733")["sample_id"] = "wrong"

    with pytest.raises(ValueError, match="sample identity mismatch"):
        verify_offline_online_parity(trace, local, records)


def test_final_trace_rejects_unapproved_accuracy(tmp_path) -> None:
    trace = _trace(tmp_path, "cifar10-final-20260824")
    observed = dict(EXPECTED_CORRECT)
    observed["dispatcher-000733"] -= 1
    local, records = _parity_inputs(tmp_path, observed)

    with pytest.raises(ValueError, match="approved offline accuracy mismatch"):
        verify_offline_online_parity(trace, local, records)


def test_rejects_unregistered_trace_identity(tmp_path) -> None:
    trace = _trace(tmp_path, "cifar10.1-v7")
    observed = {arm: 100 for arm in ARMS}
    local, records = _parity_inputs(tmp_path, observed)

    with pytest.raises(ValueError, match="only the two approved benchmark traces"):
        verify_offline_online_parity(trace, local, records)
