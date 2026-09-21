from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pertinence.serving.parity import ARMS, verify_offline_online_parity
from pertinence.serving.parity_collector import collect_parity_arm, combine_parity_arms
from pertinence.serving.trace import create_trace


def _trace(tmp_path: Path, samples: int) -> Path:
    root = tmp_path / "trace"
    create_trace(
        root,
        trace_id="cifar10.1-v6",
        dataset_id="cifar10.1",
        dataset_version="v6",
        images=np.zeros((samples, 3, 32, 32), dtype=np.float32),
        labels=np.zeros(samples, dtype=np.int64),
        sample_ids=[f"sample-{index}" for index in range(samples)],
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


def _expected_identity(arm: str) -> dict[str, str]:
    value = {"model_name": arm, "model_version": "1"}
    if arm == "static-resnet-router":
        value.update(
            {"promotion_id": "promotion-1", "dispatcher_id": "static", "expert_id": "resnet56"}
        )
    elif arm == "dispatcher-000733":
        value.update(
            {"promotion_id": "promotion-1", "dispatcher_id": "dispatcher-000733"}
        )
    return value


class _FakePredictor:
    def __init__(self, arm: str, *, bad_route: bool = False) -> None:
        self.arm = arm
        self.bad_route = bad_route
        self.calls = 0
        self.closed = False

    def __call__(self, image: np.ndarray) -> dict[str, object]:
        assert image.shape == (1, 3, 32, 32)
        assert image.dtype == np.float32
        index = self.calls
        self.calls += 1
        prediction = index % 10
        logits = np.zeros((1, 10), dtype=np.float32)
        logits[0, prediction] = 1.0
        identity = _expected_identity(self.arm)
        if self.arm == "dispatcher-000733":
            route = index % 3
            identity = {**identity, "expert_id": ARMS[route]}
        elif self.arm == "static-resnet-router":
            route = None if self.bad_route else 2
        else:
            route = 0 if self.bad_route else None
        return {"logits": logits, "route": route, "identity": identity}

    def close(self) -> None:
        self.closed = True


class _Factory:
    def __init__(self, *, bad_route: bool = False) -> None:
        self.bad_route = bad_route
        self.created: list[tuple[str, str, float, _FakePredictor]] = []

    def __call__(self, endpoint: str, arm: str, timeout: float) -> _FakePredictor:
        predictor = _FakePredictor(arm, bad_route=self.bad_route)
        self.created.append((endpoint, arm, timeout, predictor))
        return predictor


def _read_records(root: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_collects_one_arm_once_and_seals_create_only_output(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("pertinence.serving.parity_collector.SAMPLES", 4)
    trace = _trace(tmp_path, 4)
    factory = _Factory()
    output = tmp_path / "arm"

    receipt = collect_parity_arm(
        trace,
        arm="shufflenetv2_x0_5",
        endpoint="http://expert.internal",
        expected_identity=_expected_identity("shufflenetv2_x0_5"),
        output=output,
        timeout_seconds=3.0,
        predictor_factory=factory,
    )

    assert receipt["samples"] == 4
    assert len(receipt["fingerprint"]) == 64
    assert [(item[0], item[1], item[2]) for item in factory.created] == [
        ("http://expert.internal", "shufflenetv2_x0_5", 3.0)
    ]
    predictor = factory.created[0][3]
    assert predictor.calls == 4
    assert predictor.closed is True
    records = _read_records(output)
    assert [item["sample_index"] for item in records] == [0, 1, 2, 3]
    assert [item["sample_id"] for item in records] == [f"sample-{index}" for index in range(4)]
    assert all(item["repeat"] == 0 and item["route"] is None for item in records)
    assert set(records[0]) == {
        "arm",
        "repeat",
        "sample_index",
        "sample_id",
        "prediction",
        "route",
        "logits",
        "identity",
    }

    with pytest.raises(ValueError, match="output already exists"):
        collect_parity_arm(
            trace,
            arm="shufflenetv2_x0_5",
            endpoint="http://expert.internal",
            expected_identity=_expected_identity("shufflenetv2_x0_5"),
            output=output,
            predictor_factory=_Factory(),
        )


def test_route_failure_closes_predictor_and_publishes_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("pertinence.serving.parity_collector.SAMPLES", 4)
    trace = _trace(tmp_path, 4)
    factory = _Factory(bad_route=True)
    output = tmp_path / "bad-arm"

    with pytest.raises(ValueError, match="static-resnet-router must return route 2"):
        collect_parity_arm(
            trace,
            arm="static-resnet-router",
            endpoint="http://router.internal",
            expected_identity=_expected_identity("static-resnet-router"),
            output=output,
            predictor_factory=factory,
        )

    assert factory.created[0][3].calls == 1
    assert factory.created[0][3].closed is True
    assert not output.exists()


def test_rejects_identity_that_does_not_match_arm_before_requests(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("pertinence.serving.parity_collector.SAMPLES", 4)
    trace = _trace(tmp_path, 4)
    factory = _Factory()

    with pytest.raises(ValueError, match="expected identity keys are invalid"):
        collect_parity_arm(
            trace,
            arm="dispatcher-000733",
            endpoint="http://router.internal",
            expected_identity={"model_name": "dispatcher-000733"},
            output=tmp_path / "bad-identity",
            predictor_factory=factory,
        )

    assert factory.created == []


def test_combines_all_arms_and_is_compatible_with_parity_verifier(tmp_path) -> None:
    trace = _trace(tmp_path, 2_000)
    inputs: dict[str, Path] = {}
    for arm in ARMS:
        output = tmp_path / f"arm-{arm}"
        collect_parity_arm(
            trace,
            arm=arm,
            endpoint=f"http://{arm}.internal",
            expected_identity=_expected_identity(arm),
            output=output,
            predictor_factory=_Factory(),
        )
        inputs[arm] = output

    combined = tmp_path / "combined"
    receipt = combine_parity_arms(trace, inputs=inputs, output=combined)
    records = _read_records(combined)

    assert receipt["records_total"] == 10_000
    assert [records[index * 2_000]["arm"] for index in range(5)] == list(ARMS)
    predictions = np.arange(2_000, dtype=np.int64) % 10
    local = tmp_path / "local.npz"
    np.savez(
        local,
        **{f"{arm}_predictions": predictions for arm in ARMS},
        **{"dispatcher-000733_routes": np.arange(2_000, dtype=np.int64) % 3},
    )
    parity = verify_offline_online_parity(trace, local, records)
    assert parity["matched"] is True
    assert parity["route_counts"] == [667, 667, 666]

    with pytest.raises(ValueError, match="output already exists"):
        combine_parity_arms(trace, inputs=inputs, output=combined)


def test_combine_detects_tampered_arm_records(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("pertinence.serving.parity_collector.SAMPLES", 4)
    trace = _trace(tmp_path, 4)
    inputs: dict[str, Path] = {}
    for arm in ARMS:
        output = tmp_path / f"arm-{arm}"
        collect_parity_arm(
            trace,
            arm=arm,
            endpoint=f"http://{arm}.internal",
            expected_identity=_expected_identity(arm),
            output=output,
            predictor_factory=_Factory(),
        )
        inputs[arm] = output
    with (inputs[ARMS[0]] / "records.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")

    combined = tmp_path / "combined"
    with pytest.raises(ValueError, match="records SHA-256 mismatch"):
        combine_parity_arms(trace, inputs=inputs, output=combined)
    assert not combined.exists()
