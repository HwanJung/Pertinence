from __future__ import annotations

import argparse
import json

import pytest

from pertinence.serving.analysis import analyze_records, derive_validity
from pertinence.serving.benchmark import APPROVED_ARMS
from pertinence.serving.contracts import seal_manifest
from pertinence.serving_cli import _run_benchmark, _write_benchmark_failure


def _records() -> list[dict[str, object]]:
    result = []
    predictions = {
        "static-resnet-router": [0, 0, 1, 1],
        "dispatcher-000733": [0, 1, 1, 1],
    }
    labels = [0, 1, 1, 0]
    for arm, values in predictions.items():
        for repeat in range(2):
            for sample, prediction in enumerate(values):
                result.append(
                    {
                        "arm": arm,
                        "repeat": repeat,
                        "sample_index": sample,
                        "latency_ns": 1_000_000 + sample * 10_000,
                        "prediction": prediction,
                        "correct": prediction == labels[sample],
                        "route": 2,
                    }
                )
    return result


def test_analysis_uses_unique_accuracy_and_paired_hierarchical_performance() -> None:
    summary = analyze_records(
        _records(), expected_repeats=2, expected_samples=4, bootstrap_samples=100
    )

    assert summary["effect_policy"].startswith("descriptive-only")
    comparison = summary["comparison"]
    assert comparison["accuracy"]["delta_right_minus_left"] == pytest.approx(0.25)
    assert comparison["latency_ms"]["p95"]["delta_right_minus_left"] == pytest.approx(0.0)


def test_analysis_rejects_incomplete_run() -> None:
    records = _records()
    records.pop()
    with pytest.raises(ValueError, match="incomplete"):
        analyze_records(records, expected_repeats=2, expected_samples=4, bootstrap_samples=10)


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
        "".join(json.dumps(item) + "\n" for item in _records()), encoding="utf-8"
    )
    run = seal_manifest(
        {
            "schema_version": 1,
            "benchmark_id": "invalid-run-1",
            "promotion_fingerprint": "a" * 64,
            "trace_fingerprints": {"cifar10-final-20260824": "b" * 64},
            "arms": list(APPROVED_ARMS),
            "warmup_requests": 1,
            "calibration_requests": 1,
            "requests_per_repeat": 4,
            "repeats": 2,
            "target_rps": 1.0,
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
            baseline_arm="static-resnet-router",
            comparison_arm="dispatcher-000733",
            repeats=2,
            samples=4,
            seed=73,
            bootstrap_samples=20,
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
    assert json.loads((output / "loadgen-complete.json").read_text())["complete"] is False
