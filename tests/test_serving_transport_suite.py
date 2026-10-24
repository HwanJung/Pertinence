import copy
from dataclasses import asdict

import pytest

from pertinence.serving.contracts import seal_manifest
from pertinence.serving.transport_suite import (
    VARIANTS, compare_runs, join_diagnostics, paired_blocks, run_manifest, schedule,
    validate_manifest,
)


def manifest(arm="D0", block=0, diagnostics=False):
    return run_manifest(run_id=f"run-{arm}-{block}", arm=arm, block=block,
                        diagnostics=diagnostics, promotion_fingerprint="a" * 64,
                        trace_fingerprint="b" * 64, runtime_image="runtime@sha256:" + "c" * 64,
                        serving_node_uid="serving", loadgen_node_uid="loadgen")


def test_schedule_balances_every_position():
    blocks = schedule()
    assert blocks == schedule()
    for index in range(6):
        assert {block[index] for block in blocks} == set(VARIANTS)
    assert all(sum(variant.predictor_cpu) == 4 for variant in VARIANTS.values())


def test_manifest_separates_variants_and_instrumentation():
    fingerprints = set()
    for arm in VARIANTS:
        for enabled in (False, True):
            value = manifest(arm, diagnostics=enabled)
            validate_manifest(value)
            fingerprints.add(value["fingerprint"])
    assert len(fingerprints) == 12
    value["variant"]["transport"] = "bad"
    with pytest.raises(ValueError):
        validate_manifest(value)


def test_block_bootstrap_preserves_pairing():
    result = paired_blocks([10, 20, 30, 40, 50, 60], [9, 19, 29, 39, 49, 59])
    assert result["ci95"] == [1, 1]
    with pytest.raises(ValueError):
        paired_blocks([10] * 12000, [9] * 12000)


def runs():
    return [seal_manifest({"manifest": manifest(arm, block), "valid": True, "invalid_reasons": [],
             "mean_ms": 20 if arm == "D0" else 10, "p95_ms": 25,
             "pod_cpu_ms_per_request": 10}) for block in range(6) for arm in VARIANTS]


def test_comparison_and_simpler_uncertain_candidate():
    result = compare_runs(runs())
    assert result["selected_distributed"] == "D1"
    assert result["automatic_promotion"] is False


@pytest.mark.parametrize("bad", ["nodes", "duplicate", "missing", "invalid", "diagnostics"])
def test_comparison_fails_closed(bad):
    values = runs()
    if bad == "nodes":
        old = values[0]["manifest"]
        fields = {key: old[key] for key in (
            "run_id", "arm", "block", "diagnostics", "promotion_fingerprint", "trace_fingerprint",
            "runtime_image", "serving_node_uid", "loadgen_node_uid", "seed")}
        fields["serving_node_uid"] = "changed"
        values[0]["manifest"] = run_manifest(**fields)
    elif bad == "duplicate":
        values[0] = copy.deepcopy(values[1])
    elif bad == "missing":
        values.pop()
    elif bad == "invalid":
        values[0]["valid"] = False
    else:
        values[0]["manifest"] = manifest(diagnostics=True)
    with pytest.raises(ValueError):
        compare_runs(values)


def test_diagnostic_join_disjoint_spans_and_missing():
    def row(stage):
        return {"request_id": "request", "expert": "s", "call_index": "1", "stage": stage,
                "wall_seconds": 1, "thread_cpu_seconds": 0.1, "request_bytes": 10,
                "response_bytes": 20, "created": 1, "reused": 0}

    client = [row(stage) for stage in (
        "request_encode", "connection", "rpc_exchange", "response_decode")]
    server = [row(stage) for stage in (
        "request_decode", "expert_model", "expert_output", "expert_encode", "response_write")]
    client[2]["wall_seconds"] = 7
    result = join_diagnostics(client, server)
    assert result[0]["transport_scheduling_wait_residual_seconds"] == 2
    assert result[0]["server_thread_cpu_seconds"] == 0.5
    with pytest.raises(ValueError):
        join_diagnostics(client, server[:-1])
    with pytest.raises(ValueError):
        join_diagnostics(client + client[:1], server)


def test_variants_are_immutable():
    assert asdict(VARIANTS["H1"])["predictor_cpu"] == (2, 1, 1)


def test_resource_window_excludes_warmup_and_rejects_missing_series():
    from pertinence.serving.transport_suite import summarize_resources

    rows = []
    for stamp, cpu in ((0, 0), (5, 50), (10, 55), (15, 60)):
        metrics = []
        for container in ("kserve-container", "modelcar"):
            for metric, value in (("container_cpu_usage_seconds_total", cpu),
                                  ("container_memory_working_set_bytes", 100),
                                  ("container_cpu_cfs_throttled_seconds_total", 0)):
                metrics.append(f'{metric}{{pod="router",container="{container}"}} {value}')
        rows.append({"timestamp_ns": stamp * 10**9, "metrics": metrics,
                     "pods": [{"name": "router", "uid": "uid", "restart_count": 0}]})
    result = summarize_resources(rows, start_ns=6 * 10**9, end_ns=14 * 10**9,
                                 pod_names=["router"], request_count=2000)
    assert result["pod_cpu_ms_per_request"] == 8
    assert result["modelcar_cpu_ms_per_request"] == 4
    assert result["memory_peak_bytes"] == 200
    rows[2]["metrics"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        summarize_resources(rows, start_ns=6 * 10**9, end_ns=14 * 10**9,
                            pod_names=["router"])


def test_warmup_waits_for_complete_collector_cycle(tmp_path):
    import json
    from pertinence.serving.transport_cli import wait_for_initial_evidence

    path = tmp_path / "evidence.jsonl"
    now = [0.0]
    path.write_text(json.dumps({"node": "loadgen", "metrics": []}) + "\n")

    def sleep(seconds):
        now[0] += seconds
        with path.open("a") as stream:
            stream.write(json.dumps({"node": "serving", "metrics": [
                'metric{container="kserve-container"} 1']}) + "\n")

    wait_for_initial_evidence(path, clock=lambda: now[0], sleep=sleep)
    assert now[0] == 0.1


def test_missing_initial_metrics_times_out(tmp_path):
    from pertinence.serving.transport_cli import wait_for_initial_evidence

    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    with pytest.raises(TimeoutError, match="initial"):
        wait_for_initial_evidence(tmp_path / "missing", timeout=0.2,
                                  clock=lambda: now[0], sleep=sleep)
