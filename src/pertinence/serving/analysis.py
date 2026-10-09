"""Validation and descriptive analysis for completed benchmark request records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
import os
from pathlib import Path
from typing import Any
import re
import tempfile

import numpy as np


_METRIC_LINE = re.compile(
    r"^(?P<name>[a-z_]+)\{(?P<labels>[^}]*)\}\s+"
    r"(?P<value>[-+0-9.eE]+)(?:\s+[-+0-9.eE]+)?$"
)
BENCHMARK_ARMS = ("resnet56", "dispatcher-000733")
EXPECTED_CORRECT = {"resnet56": 1892, "dispatcher-000733": 1893}
EXPECTED_DISPATCHER_ROUTES = [0, 640, 1360]
POD_FRAGMENTS = {
    "resnet56": ("expert-resnet56",),
    "dispatcher-000733": (
        "router-dispatcher-000733",
        "expert-shufflenet",
        "expert-mobilenet",
        "expert-resnet56",
    ),
}


def request_matrix_inventory(
    records: Sequence[Mapping[str, Any]],
    *,
    arms: Sequence[str],
    repeats: int,
    samples: int,
) -> dict[str, object]:
    """Inventory request coordinates without computing or publishing numeric effects."""

    if not arms or len(set(arms)) != len(arms) or repeats < 1 or samples < 1:
        raise ValueError("request matrix contract is invalid")
    expected_arms = set(arms)
    cells: set[tuple[str, int, int]] = set()
    duplicate_cells = 0
    invalid_records = 0
    unexpected_arms: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            invalid_records += 1
            continue
        arm = record.get("arm")
        repeat = record.get("repeat")
        sample = record.get("sample_index")
        if not isinstance(arm, str) or arm not in expected_arms:
            invalid_records += 1
            if isinstance(arm, str):
                unexpected_arms.add(arm)
            continue
        if (
            type(repeat) is not int
            or type(sample) is not int
            or not 0 <= repeat < repeats
            or not 0 <= sample < samples
        ):
            invalid_records += 1
            continue
        cell = (arm, repeat, sample)
        if cell in cells:
            duplicate_cells += 1
        else:
            cells.add(cell)
    expected_cells = len(arms) * repeats * samples
    missing_cells = expected_cells - len(cells)
    complete = (
        missing_cells == 0
        and duplicate_cells == 0
        and invalid_records == 0
        and len(records) == expected_cells
    )
    return {
        "complete": complete,
        "arms": list(arms),
        "repeats": repeats,
        "samples_per_arm_repeat": samples,
        "expected_records": expected_cells,
        "observed_records": len(records),
        "unique_cells": len(cells),
        "missing_cells": missing_cells,
        "duplicate_cells": duplicate_cells,
        "invalid_records": invalid_records,
        "unexpected_arms": sorted(unexpected_arms),
    }


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
    """Summarize the complete serving stack used within each measured arm window."""

    snapshots: dict[int, dict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
    pod_snapshots: dict[int, list[Mapping[str, Any]]] = {}
    for record in evidence_records:
        timestamp = int(record["timestamp_ns"])
        raw_pods = record.get("pods")
        if isinstance(raw_pods, list):
            pod_snapshots[timestamp] = raw_pods
        for line in record.get("metrics", []):
            match = _METRIC_LINE.fullmatch(str(line))
            if not match:
                continue
            labels = {
                key: raw.strip('"')
                for item in match.group("labels").split(",")
                if "=" in item
                for key, raw in (item.split("=", 1),)
            }
            pod = labels.get("pod", "unknown")
            container = labels.get("container")
            if container in {"", "POD"}:
                continue
            snapshots[timestamp][(match.group("name"), pod)] += float(match.group("value"))
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
    if starts.keys() != ends.keys() or set(starts) != {(arm, 0) for arm in BENCHMARK_ARMS}:
        raise ValueError("arm marker pairs are incomplete")
    result: dict[str, object] = {}
    for arm in BENCHMARK_ARMS:
        key = (arm, 0)
        selected = [
            (timestamp, snapshot)
            for timestamp, snapshot in sorted(snapshots.items())
            if starts[key] <= timestamp <= ends[key]
        ]
        if len(selected) < 2:
            raise ValueError(f"metric gap leaves fewer than two samples for {arm}")
        relevant_pods = sorted(
            {
                pod
                for _, snapshot in selected
                for _, pod in snapshot
                if any(fragment in pod for fragment in POD_FRAGMENTS[arm])
            }
        )
        if any(not any(fragment in pod for pod in relevant_pods) for fragment in POD_FRAGMENTS[arm]):
            raise ValueError(f"serving pod metrics are incomplete for {arm}")
        per_pod: dict[str, dict[str, object]] = {}
        for pod in relevant_pods:
            memory_key = ("container_memory_working_set_bytes", pod)
            if any(memory_key not in snapshot for _, snapshot in selected):
                raise ValueError(f"cAdvisor memory series is incomplete for {pod}")
            pod_memory = [snapshot[memory_key] for _, snapshot in selected]
            deltas: dict[str, float] = {}
            for metric in (
                "container_cpu_usage_seconds_total",
                "container_cpu_cfs_throttled_seconds_total",
                "container_cpu_cfs_throttled_periods_total",
                "container_cpu_cfs_periods_total",
            ):
                metric_key = (metric, pod)
                if metric_key not in selected[0][1] or metric_key not in selected[-1][1]:
                    raise ValueError(f"cAdvisor counter series is incomplete for {pod}")
                delta = selected[-1][1][metric_key] - selected[0][1][metric_key]
                if delta < 0:
                    raise ValueError("cAdvisor counter reset detected")
                deltas[metric] = delta
            per_pod[pod] = {
                "cpu_seconds": deltas["container_cpu_usage_seconds_total"],
                "memory_working_set_average_bytes": float(np.mean(pod_memory)),
                "memory_working_set_peak_bytes": int(max(pod_memory)),
                "throttled_seconds": deltas["container_cpu_cfs_throttled_seconds_total"],
                "throttled_periods": deltas["container_cpu_cfs_throttled_periods_total"],
                "periods": deltas["container_cpu_cfs_periods_total"],
            }
        arm_requests = [item for item in requests if item.get("arm") == arm]
        cpu_delta = sum(float(item["cpu_seconds"]) for item in per_pod.values())
        if not arm_requests or cpu_delta <= 0:
            raise ValueError(f"missing request or cAdvisor CPU evidence for {arm}")
        memory_totals = [
            sum(
                snapshot.get(("container_memory_working_set_bytes", pod), 0.0)
                for pod in relevant_pods
            )
            for _, snapshot in selected
        ]
        throttled_delta = sum(float(item["throttled_seconds"]) for item in per_pod.values())
        throttled_periods = sum(float(item["throttled_periods"]) for item in per_pod.values())
        periods = sum(float(item["periods"]) for item in per_pod.values())
        pod_count = len(POD_FRAGMENTS[arm])
        result[arm] = {
            "requests": len(arm_requests),
            "cpu_seconds": cpu_delta,
            "cpu_ms_per_request": cpu_delta * 1_000 / len(arm_requests),
            "memory_working_set_average_bytes": float(np.mean(memory_totals)),
            "memory_working_set_peak_bytes": int(max(memory_totals)),
            "throttled_seconds": throttled_delta,
            "throttled_period_ratio": throttled_periods / periods if periods else None,
            "pods": per_pod,
            "kubernetes_resources": {
                "cpu_requests_cores": pod_count,
                "cpu_limits_cores": pod_count,
                "memory_requests_bytes": pod_count * 1024**3,
                "memory_limits_bytes": pod_count * 1024**3,
            },
        }
    timestamps = sorted(snapshots)
    relevant_fragments = POD_FRAGMENTS["dispatcher-000733"]
    first_pods: dict[str, tuple[str, int]] = {}
    last_pods: dict[str, tuple[str, int]] = {}
    if pod_snapshots:
        first_timestamp, last_timestamp = min(pod_snapshots), max(pod_snapshots)
        for source, destination in (
            (pod_snapshots[first_timestamp], first_pods),
            (pod_snapshots[last_timestamp], last_pods),
        ):
            for item in source:
                name = item.get("name")
                uid = item.get("uid")
                restart_count = item.get("restart_count")
                if (
                    isinstance(name, str)
                    and any(fragment in name for fragment in relevant_fragments)
                    and isinstance(uid, str)
                    and type(restart_count) is int
                ):
                    destination[name] = (uid, restart_count)
    expected_pod_fragments_present = all(
        any(fragment in name for name in first_pods) for fragment in relevant_fragments
    )
    pod_identity_stable = (
        expected_pod_fragments_present
        and first_pods.keys() == last_pods.keys()
        and all(first_pods[name][0] == last_pods[name][0] for name in first_pods)
    )
    pod_restart_delta = (
        sum(last_pods[name][1] - first_pods[name][1] for name in first_pods)
        if pod_identity_stable
        else None
    )
    return {
        "arms": result,
        "metric_gap_count": sum(
            right - left > 7_500_000_000 for left, right in zip(timestamps, timestamps[1:])
        ),
        "pod_identity_stable": pod_identity_stable,
        "pod_restart_delta": pod_restart_delta,
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
        "mean": float(np.mean(values)),
        **{
            key: float(np.quantile(values, quantile, method="linear"))
            for key, quantile in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))
        },
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
    expected_repeats: int = 1,
    expected_samples: int = 2000,
) -> dict[str, object]:
    """Analyze one complete direct-ResNet/dispatcher pass without inference statistics."""

    if expected_repeats != 1 or expected_samples < 1:
        raise ValueError("analysis requires exactly one pass and a positive sample count")
    dataset_ids = {record.get("dataset_id") for record in records if "dataset_id" in record}
    if len(dataset_ids) > 1:
        raise ValueError("each benchmark summary must contain exactly one dataset")
    by_arm = _records_by_arm(records)
    if set(by_arm) != set(BENCHMARK_ARMS):
        raise ValueError("analysis requires exactly resnet56 and dispatcher-000733")

    arm_summaries: dict[str, object] = {}
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
        # Accuracy is defined on the unique trace.
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
        correct_count = int(np.sum(correct_matrix[0]))
        arm_summaries[arm] = {
            "requests": len(arm_records),
            "unique_samples": expected_samples,
            "correct": correct_count,
            "accuracy": correct_count / expected_samples,
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
    baseline = arm_summaries["resnet56"]
    dispatcher = arm_summaries["dispatcher-000733"]
    latency_comparison = {
        metric: {
            "absolute_difference_ms": (
                dispatcher["latency_ms"][metric] - baseline["latency_ms"][metric]
            ),
            "ratio": dispatcher["latency_ms"][metric] / baseline["latency_ms"][metric],
        }
        for metric in ("mean", "p50", "p95", "p99")
    }
    latency_comparison["achieved_rps"] = {
        "absolute_difference": dispatcher["achieved_rps"] - baseline["achieved_rps"],
        "ratio": dispatcher["achieved_rps"] / baseline["achieved_rps"],
    }
    route_counts = [
        int(dispatcher["route_distribution_all_requests"].get(str(route), 0))
        for route in range(3)
    ]
    accuracy_matches = all(
        arm_summaries[arm]["correct"] == expected for arm, expected in EXPECTED_CORRECT.items()
    )
    routes_match = route_counts == EXPECTED_DISPATCHER_ROUTES
    return {
        "schema_version": 1,
        "kind": "benchmark-summary",
        "effect_policy": "two-arm-single-pass-descriptive-only",
        "interpretation": "descriptive",
        "accuracy": {
            "arms": {
                arm: {
                    "samples": expected_samples,
                    "correct": arm_summaries[arm]["correct"],
                    "value": arm_summaries[arm]["accuracy"],
                }
                for arm in BENCHMARK_ARMS
            },
            "dispatcher_minus_resnet56_percentage_points": (
                dispatcher["accuracy"] - baseline["accuracy"]
            ) * 100.0,
            "offline_expected_correct": EXPECTED_CORRECT,
            "matches_offline_expected": accuracy_matches,
            "dispatcher_route_counts": route_counts,
            "offline_expected_dispatcher_route_counts": EXPECTED_DISPATCHER_ROUTES,
            "routes_match_offline_expected": routes_match,
        },
        "performance": {
            "arms": arm_summaries,
            "latency_comparison": latency_comparison,
        },
        "offline_expected_matches": accuracy_matches and routes_match,
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
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as error:
            raise ValueError(f"output already exists: {output}") from error
    finally:
        temporary.unlink(missing_ok=True)
