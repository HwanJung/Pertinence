from __future__ import annotations

import numpy as np
import pytest

from pertinence.serving.benchmark import common_request_rate, latin_square, run_scheduled_arm
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


def test_rate_and_latin_square_follow_protocol() -> None:
    assert common_request_rate({"fast": [0.01] * 20, "slow": [0.2] * 20}) == 0.5
    square = latin_square(("a", "b", "c", "d", "e"))
    assert len(square) == 5
    assert all(set(row) == set(square[0]) for row in square)
    assert [row[0] for row in square] == ["a", "b", "c", "d", "e"]


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
