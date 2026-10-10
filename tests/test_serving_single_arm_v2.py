from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from pertinence.serving.analysis import (
    aggregate_pod_resources,
    analyze_single_arm_records,
    analyze_three_arm_summaries,
    summarize_pod_counters,
)
from pertinence.serving.benchmark import SINGLE_ARM_IDS, validate_single_arm_config
from pertinence.serving.cli.benchmark import add_commands


def _records(arm: str) -> list[dict[str, object]]:
    model_name = "resnet56" if arm == "resnet56-direct" else "dispatcher-000733"
    serving_mode = arm.rsplit("-", 1)[-1]
    correct = 1892 if serving_mode == "direct" else 1893
    records = []
    for sample in range(2_000):
        route = None if serving_mode == "direct" else (1 if sample < 640 else 2)
        records.append(
            {
                "dataset_id": "trace",
                "arm": arm,
                "repeat": 0,
                "sample_index": sample,
                "sample_id": f"sample-{sample}",
                "started_ns": sample * 200_000_000,
                "completed_ns": sample * 200_000_000 + 1_000_000,
                "latency_ns": 1_000_000,
                "process_cpu_ns": 500_000,
                "prediction": sample % 10,
                "correct": sample < correct,
                "route": route,
                "logits": [float(index == sample % 10) for index in range(10)],
                "identity": {
                    "model_name": model_name,
                    "serving_mode": serving_mode,
                    **(
                        {"dispatcher_id": "dispatcher-000733"}
                        if serving_mode != "direct"
                        else {}
                    ),
                },
            }
        )
    return records


def _summary(arm: str, value: float = 1.0) -> dict[str, object]:
    analyzed = analyze_single_arm_records(_records(arm), arm_id=arm)
    compatibility = {
        "trace_fingerprint": "a" * 64,
        "promotion_fingerprint": "b" * 64,
        "lineage": {
            "contract_fingerprint": "e" * 64,
            "search_fingerprint": "f" * 64,
            "final_fingerprint": "0" * 64,
        },
        "runtime_image": "runtime@sha256:" + "c" * 64,
        "expert_hashes": {"resnet56": "d" * 64},
        "node_uid": "node-1",
        "machine_type": "n2-standard-8",
        "ort_config": {"intra_op_threads": 1, "inter_op_threads": 1},
    }
    analyzed["performance"].update(
        {
            "cadvisor_cpu_ms_per_request": value,
            "memory_working_set_average_bytes": value * 100,
            "memory_working_set_peak_bytes": value * 120,
        }
    )
    return {
        **analyzed,
        "valid": True,
        "compatibility": compatibility,
    }


def test_v2_config_and_cli_keep_arm_id_separate_from_endpoint_model() -> None:
    config = yaml.safe_load(
        Path("experiments/cifar10/configs/benchmark-single-arm-v2.yaml").read_text()
    )
    validate_single_arm_config(config)
    parser = argparse.ArgumentParser()
    add_commands(parser.add_subparsers(dest="command", required=True))

    args = parser.parse_args(
        [
            "benchmark",
            "run-single",
            "--arm",
            "dispatcher-000733-fused",
            "--endpoint",
            "http://service",
            "--trace",
            "trace",
            "--promotion",
            "promotion.json",
            "--config",
            "config.yaml",
            "--output",
            "output",
        ]
    )

    assert args.endpoint == "http://service"
    assert args.arm == "dispatcher-000733-fused"


def test_single_arm_analysis_requires_complete_2000_and_expected_routes() -> None:
    summary = analyze_single_arm_records(
        _records("dispatcher-000733-fused"), arm_id="dispatcher-000733-fused"
    )

    assert summary["accuracy"]["correct"] == 1893
    assert summary["accuracy"]["route_counts"] == [0, 640, 1360]
    with pytest.raises(ValueError, match="request matrix"):
        analyze_single_arm_records(
            _records("resnet56-direct")[:-1], arm_id="resnet56-direct"
        )


def test_three_arm_comparison_checks_parity_and_all_three_directions() -> None:
    summaries = [_summary(arm, index + 1.0) for index, arm in enumerate(SINGLE_ARM_IDS)]
    result = analyze_three_arm_summaries(summaries)

    assert result["valid"] is True
    assert set(result["comparisons"]) == {
        "direct_to_fused",
        "fused_to_distributed",
        "direct_to_distributed",
    }
    assert result["parity"]["matched"] is True

    mismatched = deepcopy(summaries)
    mismatched[2]["observations"][0]["logits"][0] += 2e-6
    invalid = analyze_three_arm_summaries(mismatched)
    assert invalid["valid"] is False
    assert "parity:logits_mismatch" in invalid["invalid_reasons"]


def test_comparison_rejects_environment_mismatch_without_publishing_effects() -> None:
    summaries = [_summary(arm) for arm in SINGLE_ARM_IDS]
    summaries[1]["compatibility"]["node_uid"] = "different-node"

    result = analyze_three_arm_summaries(summaries)

    assert result["valid"] is False
    assert result["arms"] == {}
    assert "compatibility_mismatch:node_uid" in result["invalid_reasons"]


def test_pod_resource_contract_includes_modelcar_sidecars() -> None:
    resources = aggregate_pod_resources(
        [
            {
                "name": "predictor-pod",
                "containers": [
                    {
                        "name": "predictor",
                        "resources": {
                            "cpu_requests_cores": 4.0,
                            "cpu_limits_cores": 4.0,
                            "memory_requests_bytes": 4 * 1024**3,
                            "memory_limits_bytes": 4 * 1024**3,
                        },
                    },
                    {
                        "name": "modelcar",
                        "role": "modelcar",
                        "resources": {
                            "cpu_requests_cores": 0.1,
                            "cpu_limits_cores": 0.2,
                            "memory_requests_bytes": 64 * 1024**2,
                            "memory_limits_bytes": 128 * 1024**2,
                        },
                    },
                ],
            }
        ]
    )

    assert resources["cpu_requests_cores"] == pytest.approx(4.1)
    assert resources["container_count"] == 2
    assert resources["modelcar_container_count"] == 1


def test_single_arm_cadvisor_sums_serving_containers_and_excludes_loadgen() -> None:
    def metric(name: str, pod: str, container: str, value: float) -> str:
        return f'{name}{{pod="{pod}",container="{container}"}} {value}'

    evidence = []
    for timestamp, predictor_cpu, modelcar_cpu in ((0, 1.0, 2.0), (10, 3.0, 5.0)):
        evidence.append(
            {
                "timestamp_ns": timestamp,
                "metrics": [
                    metric(
                        "container_cpu_usage_seconds_total",
                        "dispatcher-000733-fused-predictor",
                        "predictor",
                        predictor_cpu,
                    ),
                    metric(
                        "container_cpu_usage_seconds_total",
                        "dispatcher-000733-fused-predictor",
                        "modelcar",
                        modelcar_cpu,
                    ),
                    metric(
                        "container_memory_working_set_bytes",
                        "dispatcher-000733-fused-predictor",
                        "predictor",
                        100,
                    ),
                    metric(
                        "container_memory_working_set_bytes",
                        "dispatcher-000733-fused-predictor",
                        "modelcar",
                        20,
                    ),
                    metric("container_cpu_usage_seconds_total", "loadgen", "runner", 1000),
                    metric("container_memory_working_set_bytes", "loadgen", "runner", 1000),
                ],
            }
        )

    summary = summarize_pod_counters(
        evidence,
        request_count=2,
        request_cpu_ns=1_000_000,
        pod_fragments=["dispatcher-000733-fused"],
    )

    assert summary["pod_cpu_per_request_ms"] == pytest.approx(2_500)
    assert summary["memory_average_bytes"] == pytest.approx(120)
    assert summary["memory_peak_bytes"] == 120
