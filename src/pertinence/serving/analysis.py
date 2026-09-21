"""Validation and descriptive analysis for completed benchmark request records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any
import re

import numpy as np

from .statistics import hierarchical_paired_bootstrap, paired_accuracy_summary


_METRIC_LINE = re.compile(r"^(?P<name>[a-z_]+)\{(?P<labels>[^}]*)\}\s+(?P<value>[-+0-9.eE]+)$")


def summarize_pod_counters(
    evidence_records: Sequence[Mapping[str, Any]], *, request_count: int, request_cpu_ns: int
) -> dict[str, object]:
    """Summarize filtered cAdvisor counters and cross-check request CPU accounting."""

    if request_count < 1 or len(evidence_records) < 2:
        raise ValueError("counter summary requires requests and at least two evidence samples")
    series: dict[tuple[str, str], list[float]] = defaultdict(list)
    timestamps = []
    for record in evidence_records:
        timestamps.append(int(record["timestamp_ns"]))
        for line in record.get("metrics", []):
            match = _METRIC_LINE.fullmatch(str(line))
            if match:
                labels = match.group("labels")
                pod = next(
                    (
                        item.split("=", 1)[1].strip('"')
                        for item in labels.split(",")
                        if item.startswith("pod=")
                    ),
                    "unknown",
                )
                series[(match.group("name"), pod)].append(float(match.group("value")))
    cpu_delta = 0.0
    throttled_delta = 0.0
    throttled_periods = 0.0
    periods = 0.0
    peak_memory = 0.0
    for (metric, _pod), values in series.items():
        if metric == "container_memory_working_set_bytes":
            peak_memory = max(peak_memory, max(values))
        elif len(values) >= 2:
            delta = values[-1] - values[0]
            if delta < 0:
                raise ValueError("cAdvisor counter reset detected")
            if metric == "container_cpu_usage_seconds_total":
                cpu_delta += delta
            elif metric == "container_cpu_cfs_throttled_seconds_total":
                throttled_delta += delta
            elif metric == "container_cpu_cfs_throttled_periods_total":
                throttled_periods += delta
            elif metric == "container_cpu_cfs_periods_total":
                periods += delta
    if cpu_delta <= 0 or peak_memory <= 0:
        raise ValueError("required cAdvisor CPU or memory series is missing")
    pod_cpu_per_request_ms = cpu_delta * 1_000 / request_count
    request_cpu_per_request_ms = request_cpu_ns / 1_000_000 / request_count
    return {
        "pod_cpu_per_request_ms": pod_cpu_per_request_ms,
        "request_cpu_per_request_ms": request_cpu_per_request_ms,
        "cpu_crosscheck_ratio": request_cpu_per_request_ms / pod_cpu_per_request_ms,
        "memory_peak_bytes": int(peak_memory),
        "throttled_seconds": throttled_delta,
        "throttled_period_ratio": throttled_periods / periods if periods else None,
        "metric_gap_count": sum(
            1 for left, right in zip(sorted(timestamps), sorted(timestamps)[1:]) if right - left > 7_500_000_000
        ),
    }


def summarize_arm_counters(
    evidence_records: Sequence[Mapping[str, Any]],
    markers: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Cross-check per-arm request CPU against cAdvisor deltas within marker windows."""

    snapshots: dict[int, dict[tuple[str, str], float]] = defaultdict(dict)
    for record in evidence_records:
        timestamp = int(record["timestamp_ns"])
        for line in record.get("metrics", []):
            match = _METRIC_LINE.fullmatch(str(line))
            if not match:
                continue
            pod = next(
                (
                    item.split("=", 1)[1].strip('"')
                    for item in match.group("labels").split(",")
                    if item.startswith("pod=")
                ),
                "unknown",
            )
            snapshots[timestamp][(match.group("name"), pod)] = float(match.group("value"))
    starts = {
        (str(item["arm"]), int(item["repeat"])): int(item["timestamp_ns"])
        for item in markers
        if item.get("event") == "arm_start"
    }
    ends = {
        (str(item["arm"]), int(item["repeat"])): int(item["timestamp_ns"])
        for item in markers
        if item.get("event") == "arm_end"
    }
    if starts.keys() != ends.keys() or not starts:
        raise ValueError("arm marker pairs are incomplete")
    pod_fragments = {
        "shufflenetv2_x0_5": ("expert-shufflenet",),
        "mobilenetv2_x0_5": ("expert-mobilenet",),
        "resnet56": ("expert-resnet56",),
        "static-resnet-router": ("router-static-resnet", "expert-resnet56"),
        "dispatcher-000733": (
            "router-dispatcher-000733",
            "expert-shufflenet",
            "expert-mobilenet",
            "expert-resnet56",
        ),
    }
    result: dict[str, object] = {}
    for arm in sorted({key[0] for key in starts}):
        cpu_delta = 0.0
        throttled_delta = 0.0
        throttled_periods = 0.0
        periods = 0.0
        memory_peak = 0.0
        for key in sorted(key for key in starts if key[0] == arm):
            selected = [
                (timestamp, snapshot)
                for timestamp, snapshot in sorted(snapshots.items())
                if starts[key] <= timestamp <= ends[key]
            ]
            if len(selected) < 2:
                raise ValueError(f"metric gap leaves fewer than two samples for {arm}")
            first, last = selected[0][1], selected[-1][1]
            relevant_keys = {
                metric_key
                for _, snapshot in selected
                for metric_key in snapshot
                if any(fragment in metric_key[1] for fragment in pod_fragments[arm])
            }
            for metric_key in relevant_keys:
                metric = metric_key[0]
                values = [snapshot[metric_key] for _, snapshot in selected if metric_key in snapshot]
                if metric == "container_memory_working_set_bytes":
                    memory_peak = max(memory_peak, max(values))
                elif metric_key in first and metric_key in last:
                    delta = last[metric_key] - first[metric_key]
                    if delta < 0:
                        raise ValueError("cAdvisor counter reset detected")
                    if metric == "container_cpu_usage_seconds_total":
                        cpu_delta += delta
                    elif metric == "container_cpu_cfs_throttled_seconds_total":
                        throttled_delta += delta
                    elif metric == "container_cpu_cfs_throttled_periods_total":
                        throttled_periods += delta
                    elif metric == "container_cpu_cfs_periods_total":
                        periods += delta
        arm_requests = [item for item in requests if item.get("arm") == arm]
        if not arm_requests or cpu_delta <= 0:
            raise ValueError(f"missing request or cAdvisor CPU evidence for {arm}")
        request_cpu_ms = sum(int(item.get("process_cpu_ns", 0)) for item in arm_requests) / 1e6
        pod_cpu_ms = cpu_delta * 1_000
        result[arm] = {
            "requests": len(arm_requests),
            "request_cpu_per_request_ms": request_cpu_ms / len(arm_requests),
            "pod_cpu_per_request_ms": pod_cpu_ms / len(arm_requests),
            "cpu_crosscheck_ratio": request_cpu_ms / pod_cpu_ms,
            "memory_peak_bytes": int(memory_peak),
            "throttled_seconds": throttled_delta,
            "throttled_period_ratio": throttled_periods / periods if periods else None,
        }
    timestamps = sorted(snapshots)
    return {
        "arms": result,
        "metric_gap_count": sum(
            right - left > 7_500_000_000 for left, right in zip(timestamps, timestamps[1:])
        ),
    }


def _records_by_arm(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        arm = record.get("arm")
        if not isinstance(arm, str) or not arm:
            raise ValueError("request record has an invalid arm")
        result[arm].append(record)
    if not result:
        raise ValueError("request records are empty")
    return dict(result)


def _matrix(
    records: Sequence[Mapping[str, Any]], field: str, *, repeats: int, samples: int
) -> np.ndarray:
    result = np.full((repeats, samples), np.nan, dtype=np.float64)
    for record in records:
        repeat = int(record["repeat"])
        sample = int(record["sample_index"])
        if not 0 <= repeat < repeats or not 0 <= sample < samples:
            raise ValueError("record repeat/sample index is outside the run manifest")
        if math.isfinite(result[repeat, sample]):
            raise ValueError("duplicate request record")
        result[repeat, sample] = float(record[field])
    if not np.isfinite(result).all():
        raise ValueError("request matrix is incomplete or non-finite")
    return result


def _matrix_default(
    records: Sequence[Mapping[str, Any]],
    field: str,
    *,
    repeats: int,
    samples: int,
    default: float,
) -> np.ndarray:
    expanded = [{**record, field: record.get(field, default)} for record in records]
    return _matrix(expanded, field, repeats=repeats, samples=samples)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        key: float(np.quantile(values, quantile, method="linear"))
        for key, quantile in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))
    }


def derive_validity(evidence: Mapping[str, Any]) -> dict[str, object]:
    """Derive validity from evidence; caller-provided valid flags are deliberately ignored."""

    reasons: list[str] = []
    request_reasons = evidence.get("request_invalid_reasons", [])
    if not isinstance(request_reasons, list):
        reasons.append("request_invalid_reasons_malformed")
    else:
        reasons.extend(f"request:{item}" for item in request_reasons if str(item))
    parity = evidence.get("parity_receipt")
    if not isinstance(parity, Mapping) or parity.get("matched") is not True:
        reasons.append("offline_parity_mismatch_or_missing")
    workflows = evidence.get("active_workflows")
    if type(workflows) is not int or workflows != 0:
        reasons.append("active_kfp_workflow")
    restarts = evidence.get("pod_restart_delta")
    if type(restarts) is not int or restarts != 0:
        reasons.append("pod_restart_or_missing")
    if evidence.get("node_topology_stable") is not True:
        reasons.append("node_topology_change_or_missing")
    if evidence.get("pod_identity_stable") is not True:
        reasons.append("pod_identity_change_or_missing")
    gaps = evidence.get("metric_gap_count")
    if type(gaps) is not int or gaps != 0:
        reasons.append("metric_gap_or_missing")
    return {"valid": not reasons, "invalid_reasons": sorted(set(reasons))}


def analyze_records(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_arm: str = "static-resnet-router",
    comparison_arm: str = "dispatcher-000733",
    expected_repeats: int = 5,
    expected_samples: int = 2000,
    seed: int = 73,
    bootstrap_samples: int = 10_000,
) -> dict[str, object]:
    """Analyze a valid run without imposing an effect-size pass/fail threshold."""

    if expected_repeats < 1 or expected_samples < 1:
        raise ValueError("expected run dimensions must be positive")
    dataset_ids = {record.get("dataset_id") for record in records if "dataset_id" in record}
    if len(dataset_ids) > 1:
        raise ValueError("each benchmark summary must contain exactly one dataset")
    by_arm = _records_by_arm(records)
    if baseline_arm not in by_arm or comparison_arm not in by_arm:
        raise ValueError("baseline or comparison arm is absent")

    arm_summaries: dict[str, object] = {}
    correctness: dict[str, np.ndarray] = {}
    latency: dict[str, np.ndarray] = {}
    cpu: dict[str, np.ndarray] = {}
    for arm, arm_records in sorted(by_arm.items()):
        latency_matrix = _matrix(
            arm_records, "latency_ns", repeats=expected_repeats, samples=expected_samples
        )
        correct_matrix = _matrix(
            arm_records, "correct", repeats=expected_repeats, samples=expected_samples
        ).astype(bool)
        predictions = _matrix(
            arm_records, "prediction", repeats=expected_repeats, samples=expected_samples
        ).astype(np.int64)
        # Accuracy is defined on the unique trace, not 5x duplicated traffic.
        stable_prediction = np.all(predictions == predictions[0], axis=0)
        stable_correctness = np.all(correct_matrix == correct_matrix[0], axis=0)
        route_counts: dict[str, int] = defaultdict(int)
        for item in arm_records:
            if item.get("route") is not None:
                route_counts[str(int(item["route"]))] += 1
        flattened_ms = latency_matrix.reshape(-1) / 1_000_000.0
        cpu_matrix = _matrix_default(
            arm_records,
            "process_cpu_ns",
            repeats=expected_repeats,
            samples=expected_samples,
            default=0.0,
        )
        starts = [int(item.get("started_ns", 0)) for item in arm_records]
        completed = [int(item.get("completed_ns", 0)) for item in arm_records]
        # run_scheduled_arm stops before recording a request when a schedule is missed;
        # completeness validation above therefore proves zero misses for analyzed records.
        schedule_misses = 0
        stage_values: dict[str, list[float]] = defaultdict(list)
        for item in arm_records:
            for stage, duration in dict(item.get("stage_latency_ns", {})).items():
                stage_values[str(stage)].append(float(duration) / 1_000_000.0)
        elapsed = (max(completed) - min(starts)) / 1e9 if starts and max(completed) > min(starts) else 0
        arm_summaries[arm] = {
            "requests": len(arm_records),
            "unique_samples": expected_samples,
            "accuracy": float(correct_matrix[0].mean()),
            "unstable_prediction_samples": int(np.sum(~stable_prediction)),
            "unstable_correctness_samples": int(np.sum(~stable_correctness)),
            "latency_ms": _quantiles(flattened_ms),
            "stage_latency_ms": {
                stage: _quantiles(np.asarray(values)) for stage, values in sorted(stage_values.items())
            },
            "request_cpu_ms": _quantiles(cpu_matrix.reshape(-1) / 1_000_000.0),
            "achieved_rps": len(arm_records) / elapsed if elapsed else None,
            "missed_schedule_observations": schedule_misses,
            "route_distribution_all_requests": dict(sorted(route_counts.items())),
        }
        correctness[arm] = correct_matrix[0]
        latency[arm] = latency_matrix
        cpu[arm] = cpu_matrix

    accuracy = paired_accuracy_summary(
        correctness[baseline_arm],
        correctness[comparison_arm],
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )
    performance = {
        statistic: hierarchical_paired_bootstrap(
            latency[baseline_arm] / 1_000_000.0,
            latency[comparison_arm] / 1_000_000.0,
            seed=seed + offset,
            bootstrap_samples=bootstrap_samples,
            statistic=statistic,
        )
        for offset, statistic in enumerate(("mean", "p50", "p95", "p99"))
    }
    cpu_performance = {
        statistic: hierarchical_paired_bootstrap(
            cpu[baseline_arm] / 1_000_000.0,
            cpu[comparison_arm] / 1_000_000.0,
            seed=seed + 20 + offset,
            bootstrap_samples=bootstrap_samples,
            statistic=statistic,
        )
        for offset, statistic in enumerate(("mean", "p50", "p95", "p99"))
    }
    comparison_pairs = [
        (arm, "dispatcher-000733")
        for arm in (
            "shufflenetv2_x0_5",
            "mobilenetv2_x0_5",
            "resnet56",
            "static-resnet-router",
        )
        if arm in by_arm and "dispatcher-000733" in by_arm
    ]
    if "resnet56" in by_arm and "static-resnet-router" in by_arm:
        comparison_pairs.append(("resnet56", "static-resnet-router"))
    paired_comparisons = {}
    for pair_index, (left, right) in enumerate(comparison_pairs):
        paired_comparisons[f"{left}_vs_{right}"] = {
            "left_arm": left,
            "right_arm": right,
            "accuracy": paired_accuracy_summary(
                correctness[left],
                correctness[right],
                seed=seed + 100 + pair_index,
                bootstrap_samples=bootstrap_samples,
            ),
            "latency_ms": {
                statistic: hierarchical_paired_bootstrap(
                    latency[left] / 1_000_000.0,
                    latency[right] / 1_000_000.0,
                    seed=seed + 200 + pair_index * 10 + offset,
                    bootstrap_samples=bootstrap_samples,
                    statistic=statistic,
                )
                for offset, statistic in enumerate(("mean", "p50", "p95", "p99"))
            },
            "request_cpu_ms": {
                statistic: hierarchical_paired_bootstrap(
                    cpu[left] / 1_000_000.0,
                    cpu[right] / 1_000_000.0,
                    seed=seed + 300 + pair_index * 10 + offset,
                    bootstrap_samples=bootstrap_samples,
                    statistic=statistic,
                )
                for offset, statistic in enumerate(("mean", "p50", "p95", "p99"))
            },
        }
    latency_interval = performance["p95"]["paired_bootstrap_95_ci"]
    accuracy_interval = accuracy["paired_bootstrap_95_ci"]
    uncertain = bool(accuracy_interval[0] <= 0.0 <= accuracy_interval[1])
    return {
        "schema_version": 1,
        "kind": "benchmark-summary",
        "effect_policy": "descriptive-only-no-predeclared-pass-fail-threshold",
        "interpretation": "uncertain" if uncertain else "estimated",
        "arms": arm_summaries,
        "comparison": {
            "baseline_arm": baseline_arm,
            "comparison_arm": comparison_arm,
            "accuracy": accuracy,
            "latency_ms": performance,
            "request_cpu_ms": cpu_performance,
            "p95_interval_crosses_zero": bool(latency_interval[0] <= 0.0 <= latency_interval[1]),
        },
        "paired_comparisons": paired_comparisons,
        "offline_mflops_note": "Offline MFLOPs are intentionally not treated as online CPU/latency.",
    }


def write_parquet(records: Sequence[Mapping[str, Any]], output: Path) -> None:
    """Write raw records with nested mappings encoded as canonical JSON strings."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - exercised in pipeline environment
        raise RuntimeError("pyarrow is required to write benchmark Parquet") from error
    flattened = []
    for record in records:
        row = dict(record)
        for field in ("identity", "stage_latency_ns"):
            value = row.get(field, {})
            import json

            row[field] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        flattened.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(flattened)
    temporary = output.with_name(f".{output.name}.part")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(output)
