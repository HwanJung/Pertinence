"""Paired descriptive statistics for online serving benchmarks."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _binary(values: Sequence[bool] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if result.dtype != np.bool_:
        if not np.isin(result, (0, 1)).all():
            raise ValueError(f"{name} must contain only booleans")
        result = result.astype(np.bool_)
    return result


def exact_mcnemar(left: Sequence[bool], right: Sequence[bool]) -> dict[str, float | int]:
    """Return the two-sided exact McNemar statistic without a SciPy dependency."""

    a = _binary(left, "left")
    b = _binary(right, "right")
    if a.shape != b.shape:
        raise ValueError("paired correctness arrays must have equal length")
    left_only = int(np.sum(a & ~b))
    right_only = int(np.sum(~a & b))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant": discordant,
        "exact_two_sided_p_value": p_value,
    }


def percentile_interval(values: np.ndarray, confidence: float = 0.95) -> list[float]:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a non-empty finite vector")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    alpha = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(values, alpha, method="linear")),
        float(np.quantile(values, 1.0 - alpha, method="linear")),
    ]


def paired_accuracy_summary(
    left: Sequence[bool],
    right: Sequence[bool],
    *,
    seed: int,
    bootstrap_samples: int = 10_000,
) -> dict[str, object]:
    """Summarize accuracy difference as right minus left on unique samples."""

    a = _binary(left, "left")
    b = _binary(right, "right")
    if a.shape != b.shape:
        raise ValueError("paired correctness arrays must have equal length")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    differences = b.astype(np.float64) - a.astype(np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, differences.size, size=(bootstrap_samples, differences.size))
    draws = differences[indices].mean(axis=1)
    return {
        "samples": int(a.size),
        "left_accuracy": float(a.mean()),
        "right_accuracy": float(b.mean()),
        "delta_right_minus_left": float(differences.mean()),
        "paired_bootstrap_95_ci": percentile_interval(draws),
        "bootstrap_seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "mcnemar": exact_mcnemar(a, b),
    }


def hierarchical_paired_bootstrap(
    left: Sequence[Sequence[float]] | np.ndarray,
    right: Sequence[Sequence[float]] | np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int = 10_000,
    statistic: str = "mean",
) -> dict[str, object]:
    """Bootstrap paired repeat-by-sample matrices along both preserved dimensions.

    Rows are benchmark repeats and columns are trace samples. The same resampled
    indices are used for both arms, preserving pairing at both levels.
    """

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or not a.size or a.shape != b.shape:
        raise ValueError("paired measurements must be equal non-empty 2-D matrices")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired measurements must be finite")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if statistic not in {"mean", "p50", "p95", "p99"}:
        raise ValueError("unsupported bootstrap statistic")

    quantiles = {"p50": 0.50, "p95": 0.95, "p99": 0.99}

    def reduce(values: np.ndarray) -> float:
        if statistic == "mean":
            return float(values.mean())
        return float(np.quantile(values, quantiles[statistic], method="linear"))

    observed = reduce(b) - reduce(a)
    generator = np.random.default_rng(seed)
    draws = np.empty(bootstrap_samples, dtype=np.float64)
    repeats, samples = a.shape
    for draw in range(bootstrap_samples):
        repeat_index = generator.integers(0, repeats, size=repeats)
        sample_index = generator.integers(0, samples, size=samples)
        left_draw = a[repeat_index][:, sample_index]
        right_draw = b[repeat_index][:, sample_index]
        draws[draw] = reduce(right_draw) - reduce(left_draw)
    return {
        "statistic": statistic,
        "repeats": repeats,
        "samples_per_repeat": samples,
        "left": reduce(a),
        "right": reduce(b),
        "delta_right_minus_left": observed,
        "paired_bootstrap_95_ci": percentile_interval(draws),
        "bootstrap_seed": seed,
        "bootstrap_samples": bootstrap_samples,
    }
