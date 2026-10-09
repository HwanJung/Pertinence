from __future__ import annotations

import argparse
import json

import pytest

from pertinence.serving.analysis import (
    analyze_records,
    derive_validity,
    request_matrix_inventory,
    summarize_arm_counters,
)
from pertinence.serving.benchmark import APPROVED_ARMS
from pertinence.serving.contracts import seal_manifest
from pertinence.serving.cli.benchmark import _run_benchmark, _write_benchmark_failure


def _records(samples: int = 2_000) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    expected_correct = {"resnet56": 1892, "dispatcher-000733": 1893}
    for arm_index, arm in enumerate(APPROVED_ARMS):
        for sample in range(samples):
            latency_ns = (arm_index + 1) * 1_000_000 + sample
            started_ns = sample * 500_000_000
            route = None
            if arm == "dispatcher-000733":
                route = 1 if sample < 640 else 2
            result.append(
                {
                    "dataset_id": "cifar10-final-20260824",
                    "arm": arm,
                    "repeat": 0,
                    "sample_index": sample,
                    "sample_id": f"sample-{sample}",
                    "started_ns": started_ns,
                    "completed_ns": started_ns + latency_ns,
                    "latency_ns": latency_ns,
                    "process_cpu_ns": 500_000,
                    "prediction": 0,
                    "correct": sample < expected_correct[arm],
                    "route": route,
                    "stage_latency_ns": {},
                }
            )
    return result


def test_analysis_reports_accuracy_latency_delta_ratio_and_routes() -> None:
    summary = analyze_records(_records())

    assert summary["effect_policy"] == "two-arm-single-pass-descriptive-only"
    assert summary["accuracy"]["arms"]["resnet56"]["correct"] == 1892
    assert summary["accuracy"]["arms"]["dispatcher-000733"]["correct"] == 1893
    assert summary["accuracy"]["dispatcher_minus_resnet56_percentage_points"] == pytest.approx(
        0.05
    )
    assert summary["accuracy"]["dispatcher_route_counts"] == [0, 640, 1360]
    assert summary["offline_expected_matches"] is True
    latency = summary["performance"]["latency_comparison"]
    assert latency["mean"]["absolute_difference_ms"] == pytest.approx(1.0)
    assert latency["p95"]["ratio"] > 1.9


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "wrong-arm"))
def test_request_matrix_rejects_missing_duplicate_or_wrong_arm(mutation: str) -> None:
    records = _records(samples=2)
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(dict(records[-1]))
    else:
        records[-1]["arm"] = "mobilenetv2_x0_5"

    inventory = request_matrix_inventory(records, arms=APPROVED_ARMS, repeats=1, samples=2)

    assert inventory["complete"] is False


def _metric(name: str, pod: str, value: float) -> str:
    return f'{name}{{pod="{pod}"}} {value}'


def _pod_metrics(pod: str, *, cpu: float, memory: float) -> list[str]:
    return [
        _metric("container_cpu_usage_seconds_total", pod, cpu),
        _metric("container_memory_working_set_bytes", pod, memory),
        _metric("container_cpu_cfs_throttled_seconds_total", pod, cpu / 100),
        _metric("container_cpu_cfs_throttled_periods_total", pod, cpu),
        _metric("container_cpu_cfs_periods_total", pod, cpu * 10),
    ]


def test_resource_summary_sums_dispatcher_pods_and_memory_at_each_timestamp() -> None:
    resnet = "expert-resnet56-predictor-a"
    dispatcher_pods = [
        "router-dispatcher-000733-predictor-a",
        "expert-shufflenet-predictor-a",
        "expert-mobilenet-predictor-a",
        resnet,
    ]
    evidence = [
        {"timestamp_ns": 0, "metrics": _pod_metrics(resnet, cpu=1, memory=50)},
        {"timestamp_ns": 10, "metrics": _pod_metrics(resnet, cpu=3, memory=60)},
        {
            "timestamp_ns": 20,
            "metrics": sum(
                (
                    _pod_metrics(pod, cpu=10 + index, memory=10 * (index + 1))
                    for index, pod in enumerate(dispatcher_pods)
                ),
                [],
            ),
        },
        {
            "timestamp_ns": 30,
            "metrics": sum(
                (
                    _pod_metrics(pod, cpu=11 + index, memory=20 + 10 * index)
                    for index, pod in enumerate(dispatcher_pods)
                ),
                [],
            ),
        },
    ]
    pod_inventory = [
        {"name": pod, "uid": f"uid-{index}", "restart_count": 0}
        for index, pod in enumerate(dispatcher_pods)
    ]
    for item in evidence:
        item["pods"] = pod_inventory
    markers = [
        {"event": "arm_start", "arm": "resnet56", "repeat": 0, "timestamp_ns": 0},
        {"event": "arm_end", "arm": "resnet56", "repeat": 0, "timestamp_ns": 10},
        {
            "event": "arm_start",
            "arm": "dispatcher-000733",
            "repeat": 0,
            "timestamp_ns": 20,
        },
        {
            "event": "arm_end",
            "arm": "dispatcher-000733",
            "repeat": 0,
            "timestamp_ns": 30,
        },
    ]
    requests = [
        {"arm": "resnet56", "process_cpu_ns": 1},
        {"arm": "dispatcher-000733", "process_cpu_ns": 1},
    ]

    summary = summarize_arm_counters(evidence, markers, requests)

    assert summary["arms"]["resnet56"]["cpu_seconds"] == pytest.approx(2)
    dispatcher = summary["arms"]["dispatcher-000733"]
    assert dispatcher["cpu_seconds"] == pytest.approx(4)
    assert dispatcher["memory_working_set_average_bytes"] == pytest.approx(120)
    assert dispatcher["memory_working_set_peak_bytes"] == 140
    assert len(dispatcher["pods"]) == 4
    assert dispatcher["kubernetes_resources"]["cpu_requests_cores"] == 4
    assert summary["pod_identity_stable"] is True
    assert summary["pod_restart_delta"] == 0


def test_resource_summary_accepts_prometheus_sample_timestamps() -> None:
    resnet = "expert-resnet56-predictor-a"
    dispatcher_pods = [
        "router-dispatcher-000733-predictor-a",
        "expert-shufflenet-predictor-a",
        "expert-mobilenet-predictor-a",
        resnet,
    ]
    evidence = []
    for timestamp, pods in (
        (0, [resnet]),
        (10, [resnet]),
        (20, dispatcher_pods),
        (30, dispatcher_pods),
    ):
        metrics = sum(
            (
                _pod_metrics(pod, cpu=timestamp + index + 1, memory=100 + index)
                for index, pod in enumerate(pods)
            ),
            [],
        )
        evidence.append(
            {
                "timestamp_ns": timestamp,
                "metrics": [f"{metric} {1790093923000 + timestamp}" for metric in metrics],
                "pods": [
                    {"name": pod, "uid": f"uid-{index}", "restart_count": 0}
                    for index, pod in enumerate(dispatcher_pods)
                ],
            }
        )
    markers = [
        {"event": "arm_start", "arm": "resnet56", "repeat": 0, "timestamp_ns": 0},
        {"event": "arm_end", "arm": "resnet56", "repeat": 0, "timestamp_ns": 10},
        {
            "event": "arm_start",
            "arm": "dispatcher-000733",
            "repeat": 0,
            "timestamp_ns": 20,
        },
        {
            "event": "arm_end",
            "arm": "dispatcher-000733",
            "repeat": 0,
            "timestamp_ns": 30,
        },
    ]

    summary = summarize_arm_counters(
        evidence,
        markers,
        [{"arm": "resnet56"}, {"arm": "dispatcher-000733"}],
    )

    assert summary["arms"]["resnet56"]["cpu_seconds"] == pytest.approx(10)
    assert summary["arms"]["dispatcher-000733"]["cpu_seconds"] == pytest.approx(40)


def test_resource_summary_detects_serving_pod_restart() -> None:
    pod_names = [
        "router-dispatcher-000733-predictor-a",
        "expert-shufflenet-predictor-a",
        "expert-mobilenet-predictor-a",
        "expert-resnet56-predictor-a",
    ]
    evidence = []
    for timestamp, restart_count in ((0, 0), (10, 0), (20, 0), (30, 1)):
        evidence.append(
            {
                "timestamp_ns": timestamp,
                "metrics": sum(
                    (
                        _pod_metrics(pod, cpu=timestamp + index + 1, memory=100)
                        for index, pod in enumerate(pod_names)
                    ),
                    [],
                ),
                "pods": [
                    {
                        "name": pod,
                        "uid": f"uid-{index}",
                        "restart_count": restart_count if index == 0 else 0,
                    }
                    for index, pod in enumerate(pod_names)
                ],
            }
        )
    markers = [
        {"event": "arm_start", "arm": "resnet56", "repeat": 0, "timestamp_ns": 0},
        {"event": "arm_end", "arm": "resnet56", "repeat": 0, "timestamp_ns": 10},
        {
            "event": "arm_start",
            "arm": "dispatcher-000733",
            "repeat": 0,
            "timestamp_ns": 20,
        },
        {
            "event": "arm_end",
            "arm": "dispatcher-000733",
            "repeat": 0,
            "timestamp_ns": 30,
        },
    ]

    summary = summarize_arm_counters(
        evidence,
        markers,
        [{"arm": "resnet56"}, {"arm": "dispatcher-000733"}],
    )

    assert summary["pod_identity_stable"] is True
    assert summary["pod_restart_delta"] == 1


def test_validity_is_derived_and_ignores_external_valid_true() -> None:
    validity = derive_validity(
        {
            "valid": True,
            "request_invalid_reasons": [],
            "parity_receipt": {"matched": True},
            "active_workflows": 0,
            "pod_restart_delta": 1,
            "node_topology_stable": True,
            "pod_identity_stable": True,
            "metric_gap_count": 0,
        }
    )
    assert validity["valid"] is False
    assert "pod_restart_or_missing" in validity["invalid_reasons"]


def test_analyze_refuses_to_overwrite_output(tmp_path) -> None:
    output = tmp_path / "summary.json"
    output.write_text("owned\n", encoding="utf-8")
    args = argparse.Namespace(serving_action="benchmark-analyze", output=output)

    with pytest.raises(ValueError, match="output already exists"):
        _run_benchmark(args)

    assert output.read_text(encoding="utf-8") == "owned\n"


def test_invalid_sealed_analysis_omits_numeric_effects(tmp_path) -> None:
    records = tmp_path / "requests.jsonl"
    records.write_text(
        "".join(json.dumps(item) + "\n" for item in _records(samples=4)), encoding="utf-8"
    )
    run = seal_manifest(
        {
            "schema_version": 1,
            "benchmark_id": "invalid-run-1",
            "promotion_fingerprint": "a" * 64,
            "trace_fingerprints": {"cifar10-final-20260824": "b" * 64},
            "arms": list(APPROVED_ARMS),
            "warmup_requests": 200,
            "requests_per_arm": 4,
            "target_rps": 2.0,
            "concurrency": 1,
            "seed": 73,
            "created_at": "2026-09-21T00:00:00Z",
        }
    )
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(run), encoding="utf-8")
    environment = {
        "request_invalid_reasons": [],
        "parity_receipt": {"matched": True},
        "active_workflows": 0,
        "pod_restart_delta": 1,
        "node_topology_stable": True,
        "pod_identity_stable": True,
        "metric_gap_count": 0,
        "sources": {"metrics": {}},
    }
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(json.dumps(environment), encoding="utf-8")
    output = tmp_path / "invalid-summary.json"

    result = _run_benchmark(
        argparse.Namespace(
            serving_action="benchmark-analyze",
            records=records,
            output=output,
            samples=4,
            run_manifest=run_path,
            environment_manifest=environment_path,
        )
    )

    assert result == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["valid"] is False
    assert summary["accuracy"] == {}
    assert summary["performance"] == {}


def test_benchmark_failure_receipt_inventories_existing_evidence(tmp_path) -> None:
    output = tmp_path / "failed-run"
    output.mkdir()
    (output / "markers.jsonl").write_text("{}\n", encoding="utf-8")

    _write_benchmark_failure(output, stage="steady_state", error=RuntimeError("failed"))

    marker = json.loads((output / "INVALID.json").read_text(encoding="utf-8"))
    assert marker["valid"] is False
    assert marker["failure_stage"] == "steady_state"
    assert {item["path"] for item in marker["files"]} == {
        "loadgen-complete.json",
        "markers.jsonl",
        "validity.json",
    }
