from __future__ import annotations

import numpy as np
import pytest

from pertinence.serving.benchmark import (
    run_scheduled_arm,
    validate_benchmark_config,
    warm_up_arm,
)
from pertinence.serving.statistics import (
    exact_mcnemar,
    hierarchical_paired_bootstrap,
    paired_accuracy_summary,
)


def test_exact_mcnemar_and_seeded_accuracy_interval() -> None:
    left = [True, True, False, False, False]
    right = [True, False, True, True, False]

    result = exact_mcnemar(left, right)
    first = paired_accuracy_summary(left, right, seed=73, bootstrap_samples=500)
    second = paired_accuracy_summary(left, right, seed=73, bootstrap_samples=500)

    assert result == {
        "left_only_correct": 1,
        "right_only_correct": 2,
        "discordant": 3,
        "exact_two_sided_p_value": 1.0,
    }
    assert first == second
    assert first["delta_right_minus_left"] == pytest.approx(0.2)


def test_hierarchical_bootstrap_preserves_pairing_and_is_reproducible() -> None:
    left = np.arange(15, dtype=float).reshape(3, 5)
    right = left + 2.5

    summary = hierarchical_paired_bootstrap(
        left, right, seed=19, bootstrap_samples=100, statistic="p95"
    )

    assert summary["delta_right_minus_left"] == pytest.approx(2.5)
    assert summary["paired_bootstrap_95_ci"] == pytest.approx([2.5, 2.5])


def test_config_accepts_only_fixed_two_arm_single_pass() -> None:
    config = {
        "schema_version": 1,
        "throughput_saturation_test": False,
        "arms": ["resnet56", "dispatcher-000733"],
        "request": {
            "send_label": False,
            "batch_size": 1,
            "concurrency": 1,
            "retry_count": 0,
            "max_in_flight": 1,
        },
        "warmup": {"requests_per_arm": 200, "distinct_samples": True},
        "steady_state": {
            "order": "single_pass",
            "arm_order": ["resnet56", "dispatcher-000733"],
            "requests_per_arm": 2_000,
            "fixed_rps": 2,
            "queue_on_missed_schedule": False,
        },
    }
    validate_benchmark_config(config)

    with pytest.raises(ValueError, match="arms/order"):
        validate_benchmark_config({**config, "arms": list(reversed(config["arms"]))})
    with pytest.raises(ValueError, match="single-pass"):
        validate_benchmark_config(
            {**config, "steady_state": {**config["steady_state"], "repeats": 2}}
        )


def test_warmup_uses_distinct_samples_and_produces_no_records() -> None:
    seen = []

    def predictor(image: np.ndarray) -> dict[str, object]:
        seen.append(float(image[0, 0, 0, 0]))
        return {
            "logits": np.zeros((1, 10), dtype=np.float32),
            "identity": {"model_name": "resnet56"},
        }

    images = np.zeros((200, 3, 32, 32), dtype=np.float32)
    images[:, 0, 0, 0] = np.arange(200)

    assert warm_up_arm(
        images=images,
        predict=predictor,
        expected_identity={"model_name": "resnet56"},
    ) is None
    assert seen == list(map(float, range(200)))


def test_scheduled_run_has_no_retry_and_never_sends_labels() -> None:
    calls = []

    def predictor(image: np.ndarray) -> dict[str, object]:
        calls.append(image.copy())
        return {
            "logits": np.asarray([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            "identity": {"model": "expert"},
        }

    ticks = iter([0, 0, 0, 10, 1_000_000_000, 1_000_000_000, 1_000_000_010])
    records, invalid = run_scheduled_arm(
        dataset_id="trace",
        arm="expert",
        repeat=0,
        images=np.zeros((2, 3, 32, 32), dtype=np.float32),
        labels=np.ones(2, dtype=np.int64),
        sample_ids=("x", "y"),
        predict=predictor,
        request_rate=1.0,
        expected_identity={"model": "expert"},
        clock_ns=lambda: next(ticks),
        sleep=lambda _: None,
    )

    assert invalid is None
    assert len(records) == len(calls) == 2
    assert records[0].correct is True
    assert records[0].logits == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert calls[0].shape == (1, 3, 32, 32)


def test_scheduled_run_invalidates_missed_schedule_without_queueing() -> None:
    ticks = iter([0, 0, 0, 10, 2_000_000_000])
    records, invalid = run_scheduled_arm(
        dataset_id="trace",
        arm="expert",
        repeat=0,
        images=np.zeros((2, 3, 32, 32), dtype=np.float32),
        labels=np.zeros(2, dtype=np.int64),
        sample_ids=("x", "y"),
        predict=lambda _: {"logits": np.zeros((1, 10), dtype=np.float32)},
        request_rate=1.0,
        clock_ns=lambda: next(ticks),
        sleep=lambda _: None,
    )

    assert len(records) == 1
    assert invalid and invalid.startswith("missed_schedule")


def test_two_arm_pass_has_exactly_4000_measured_records_after_warmup() -> None:
    images = np.zeros((2_000, 3, 32, 32), dtype=np.float32)
    images[:, 0, 0, 0] = np.arange(2_000)
    labels = np.zeros(2_000, dtype=np.int64)
    sample_ids = [f"sample-{index}" for index in range(2_000)]
    measured = []
    warmup_values: dict[str, list[int]] = {"resnet56": [], "dispatcher-000733": []}

    class Clock:
        value = 0

        def now(self) -> int:
            return self.value

        def sleep(self, seconds: float) -> None:
            self.value += round(seconds * 1_000_000_000)

    for arm in ("resnet56", "dispatcher-000733"):
        identity = {"model_name": arm}
        if arm == "dispatcher-000733":
            identity.update(
                {"promotion_id": "promotion-1", "dispatcher_id": "dispatcher-000733"}
            )

        def predictor(image: np.ndarray, *, arm: str = arm) -> dict[str, object]:
            warmup_values[arm].append(int(image[0, 0, 0, 0]))
            return {
                "logits": np.asarray([[1.0] + [0.0] * 9], dtype=np.float32),
                "route": 2 if arm == "dispatcher-000733" else None,
                "identity": identity,
            }

        assert warm_up_arm(
            images=images,
            predict=predictor,
            expected_identity=identity,
            requests=200,
        ) is None
        warmup_values[arm].clear()
        clock = Clock()
        records, invalid = run_scheduled_arm(
            dataset_id="cifar10-final-20260824",
            arm=arm,
            repeat=0,
            images=images,
            labels=labels,
            sample_ids=sample_ids,
            predict=predictor,
            request_rate=2.0,
            expected_identity=identity,
            clock_ns=clock.now,
            sleep=clock.sleep,
        )
        assert invalid is None
        measured.extend(records)

    assert len(measured) == 4_000
    assert {record.arm for record in measured} == {"resnet56", "dispatcher-000733"}
    assert all(len(values) == 2_000 for values in warmup_values.values())


def test_scheduled_run_invalidates_identity_mismatch() -> None:
    records, invalid = run_scheduled_arm(
        dataset_id="trace",
        arm="resnet56",
        repeat=0,
        images=np.zeros((1, 3, 32, 32), dtype=np.float32),
        labels=np.zeros(1, dtype=np.int64),
        sample_ids=("x",),
        predict=lambda _: {
            "logits": np.zeros((1, 10), dtype=np.float32),
            "identity": {"model_name": "mobilenetv2_x0_5"},
        },
        request_rate=2.0,
        expected_identity={"model_name": "resnet56"},
        clock_ns=lambda: 0,
        sleep=lambda _: None,
    )

    assert records == []
    assert invalid == "identity_mismatch:sample=0"
