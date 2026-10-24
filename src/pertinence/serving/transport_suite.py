"""Independent six-arm transport experiment contracts and paired block analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..reproducibility import canonical_json_hash
from .contracts import _sealed_body, seal_manifest


@dataclass(frozen=True)
class Variant:
    structure: str
    transport: str
    connection: str
    predictor_cpu: tuple[int, ...]


VARIANTS = {
    "D0": Variant("distributed", "json", "legacy", (1, 1, 1, 1)),
    "D1": Variant("distributed", "binary", "legacy", (1, 1, 1, 1)),
    "D2": Variant("distributed", "json", "keep-alive", (1, 1, 1, 1)),
    "D3": Variant("distributed", "binary", "keep-alive", (1, 1, 1, 1)),
    "H1": Variant("hybrid", "binary", "keep-alive", (2, 1, 1)),
    "F0": Variant("fused", "local", "none", (4,)),
}
COMPARISONS = (("D0", "D1"), ("D0", "D2"), ("D0", "D3"), ("D1", "D3"),
               ("D2", "D3"), ("D3", "H1"), ("F0", "H1"))


def schedule(seed: int = 20261011) -> list[list[str]]:
    first = np.random.default_rng(seed).permutation(list(VARIANTS)).tolist()
    return [first[index:] + first[:index] for index in range(6)]


def run_manifest(*, run_id: str, arm: str, block: int, diagnostics: bool,
                 promotion_fingerprint: str, trace_fingerprint: str,
                 runtime_image: str, serving_node_uid: str, loadgen_node_uid: str,
                 seed: int = 20261011) -> dict[str, Any]:
    if arm not in VARIANTS or type(block) is not int or not 0 <= block < 6:
        raise ValueError("unknown arm or block")
    if type(diagnostics) is not bool:
        raise ValueError("diagnostics must be boolean")
    if not all(isinstance(value, str) and value for value in (
        run_id, serving_node_uid, loadgen_node_uid,
    )):
        raise ValueError("run and node identities are required")
    from ..contracts.common import digest
    from .contracts import _pinned_image

    digest(promotion_fingerprint)
    digest(trace_fingerprint)
    _pinned_image(runtime_image, "runtime image")
    return seal_manifest({
        "schema_version": 1, "kind": "transport-suite-run", "run_id": run_id,
        "arm": arm, "block": block, "variant": asdict(VARIANTS[arm]),
        "diagnostics": diagnostics, "promotion_fingerprint": promotion_fingerprint,
        "trace_fingerprint": trace_fingerprint, "runtime_image": runtime_image,
        "serving_node_uid": serving_node_uid, "loadgen_node_uid": loadgen_node_uid,
        "seed": seed, "warmup_requests": 200, "requests": 2000, "rps": 5,
        "concurrency": 1, "retry_count": 0,
        "python": "3.11", "onnxruntime": "1.20.1", "ort_threads": 1,
    })


def validate_manifest(value: dict) -> None:
    expected = run_manifest(**{key: value[key] for key in (
        "run_id", "arm", "block", "diagnostics", "promotion_fingerprint", "trace_fingerprint",
        "runtime_image", "serving_node_uid", "loadgen_node_uid", "seed",
    )})
    if value != expected:
        raise ValueError("transport run contract or fingerprint differs")


def latency_summary(records: list[dict]) -> dict:
    def summarize(rows: list[dict]) -> dict:
        values = np.asarray([row["latency_ns"] for row in rows], dtype=float) / 1e6
        return {"count": len(rows), "mean_ms": float(values.mean()),
                **{f"p{q}_ms": float(np.percentile(values, q)) for q in (50, 95, 99)}}

    if not records:
        raise ValueError("request records are empty")
    routes = sorted({row["route"] for row in records})
    return {"overall": summarize(records),
            "routes": {str(route): summarize([row for row in records if row["route"] == route])
                       for route in routes}}


def paired_blocks(baseline: list[float], candidate: list[float], *, seed: int = 20261011,
                  resamples: int = 10000) -> dict:
    left, right = np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
    if left.shape != (6,) or right.shape != (6,) or not (
        np.isfinite(left).all() and np.isfinite(right).all() and (left > 0).all()
    ):
        raise ValueError("six finite paired blocks with positive baselines are required")
    differences = left - right
    indices = np.random.default_rng(seed).integers(0, 6, size=(resamples, 6))
    return {"block_savings": differences.tolist(), "mean_saving": float(differences.mean()),
            "fraction_saved": float(differences.mean() / left.mean()),
            "ci95": np.percentile(differences[indices].mean(axis=1), [2.5, 97.5]).tolist()}


def compare_runs(runs: list[dict]) -> dict:
    """Fail closed on missing/invalid blocks; bootstrap blocks, never requests."""
    if len(runs) != 36:
        raise ValueError("comparison requires exactly six complete six-arm blocks")
    by_key = {}
    shared = set()
    for run in runs:
        _sealed_body(run)
        manifest = run["manifest"]
        validate_manifest(manifest)
        if manifest["diagnostics"]:
            raise ValueError("diagnostic runs cannot enter the performance comparison")
        shared.add(tuple(manifest[key] for key in (
            "promotion_fingerprint", "trace_fingerprint", "runtime_image", "seed",
        )))
        key = manifest["block"], manifest["arm"]
        if key in by_key:
            raise ValueError("duplicate arm/block")
        if run.get("valid") is not True or run.get("invalid_reasons") != []:
            raise ValueError("invalid run cannot enter primary comparison")
        by_key[key] = run
    if len(shared) != 1:
        raise ValueError("incompatible model, trace, runtime, or seed")
    for block in range(6):
        rows = [by_key[(block, arm)] for arm in VARIANTS]
        if len({(row["manifest"]["serving_node_uid"], row["manifest"]["loadgen_node_uid"])
                for row in rows}) != 1:
            raise ValueError("node UID changed within block; repeat the entire block")
    comparisons = {}
    for baseline, candidate in COMPARISONS:
        stats = {}
        for metric in ("mean_ms", "p95_ms", "pod_cpu_ms_per_request"):
            stats[metric] = paired_blocks(
                [by_key[(block, baseline)][metric] for block in range(6)],
                [by_key[(block, candidate)][metric] for block in range(6)],
            )
        stats["passes"] = (stats["mean_ms"]["ci95"][0] > 0
                           and stats["p95_ms"]["fraction_saved"] >= -0.05
                           and stats["pod_cpu_ms_per_request"]["fraction_saved"] >= -0.05)
        comparisons[f"{baseline}->{candidate}"] = stats
    candidates = [arm for arm in ("D1", "D2", "D3") if comparisons[f"D0->{arm}"]["passes"]]
    selected = None
    if candidates:
        selected = min(candidates, key=lambda arm: np.mean(
            [by_key[(block, arm)]["mean_ms"] for block in range(6)]))
        # Prefer a one-option change when its difference from D3 is uncertain.
        if selected == "D3":
            for simpler in ("D1", "D2"):
                if simpler in candidates and comparisons[f"{simpler}->D3"]["mean_ms"]["ci95"][0] <= 0:
                    selected = simpler
                    break
    return seal_manifest({"kind": "transport-suite-comparison", "comparisons": comparisons,
                          "selected_distributed": selected, "automatic_promotion": False,
                          "run_fingerprints": sorted(row["manifest"]["fingerprint"] for row in runs)})


def join_diagnostics(router: list[dict], experts: list[dict]) -> list[dict]:
    """Join durations without subtracting clocks from different processes."""
    server = defaultdict(dict)
    client = defaultdict(dict)
    for target, records in ((server, experts), (client, router)):
        for row in records:
            if row.get("scope") == "total":
                continue  # supplemental whole-request CPU; never sum with child spans
            if not row.get("request_id") or row.get("call_index") is None:
                continue  # health probes and non-RPC local stages
            key = row["request_id"], row["expert"], str(row["call_index"])
            if row["stage"] in target[key]:
                raise ValueError(f"duplicate diagnostic span: {key}/{row['stage']}")
            if any(not np.isfinite(row[field]) or row[field] < 0
                   for field in ("wall_seconds", "thread_cpu_seconds")):
                raise ValueError("invalid diagnostic duration")
            target[key][row["stage"]] = row
    if server.keys() != client.keys():
        raise ValueError("missing client or expert diagnostic calls")
    result = []
    for key, spans in client.items():
        required = {"request_encode", "connection", "rpc_exchange", "response_decode"}
        if spans.keys() != required:
            raise ValueError("missing or unexpected client spans")
        remote = server[key]
        encode = "expert_encode" if "expert_encode" in remote else "response_encode"
        required = {"request_decode", "expert_model", "expert_output", encode, "response_write"}
        if remote.keys() != required:
            raise ValueError("missing or unexpected expert spans")
        server_wall = sum(remote[name]["wall_seconds"] for name in required)
        result.append({"request_id": key[0], "expert": key[1], "call_index": key[2],
                       "transport_scheduling_wait_residual_seconds": (
                           spans["rpc_exchange"]["wall_seconds"] - server_wall),
                       "client_thread_cpu_seconds": sum(
                           row["thread_cpu_seconds"] for row in spans.values()),
                       "server_thread_cpu_seconds": sum(
                           row["thread_cpu_seconds"] for row in remote.values()),
                       "client_codec_cpu_seconds": sum(spans[name]["thread_cpu_seconds"]
                           for name in ("request_encode", "response_decode")),
                       "server_codec_cpu_seconds": sum(remote[name]["thread_cpu_seconds"]
                           for name in ("request_decode", "expert_output", encode)),
                       "server_model_cpu_seconds": remote["expert_model"]["thread_cpu_seconds"],
                       "request_bytes": spans["rpc_exchange"]["request_bytes"],
                       "response_bytes": spans["rpc_exchange"]["response_bytes"],
                       "connections_created": spans["connection"]["created"],
                       "connections_reused": spans["connection"]["reused"]})
    return result


def verify_run_records(records: list[dict], reference: list[dict], evidence: dict) -> dict:
    """Require complete parity and explicit resource/operational evidence."""
    reasons = []
    for label, rows in (("candidate", records), ("reference", reference)):
        if len(rows) != 2000 or [row["sample_index"] for row in rows] != list(range(2000)):
            reasons.append(f"{label}_request_matrix")
        if len({row["sample_id"] for row in rows}) != len(rows):
            reasons.append(f"{label}_duplicate_samples")
    exact = True
    error = 0.0
    for row, expected in zip(records, reference):
        if any(row[key] != expected[key] for key in ("sample_id", "prediction", "route")):
            reasons.append("prediction_route_or_sample_mismatch")
        left, right = np.asarray(row["logits"], dtype=np.float32), np.asarray(
            expected["logits"], dtype=np.float32)
        if left.shape != (10,) or right.shape != (10,) or not (
            np.isfinite(left).all() and np.isfinite(right).all()
        ):
            reasons.append("invalid_logits")
            continue
        exact &= left.tobytes() == right.tobytes()
        error = max(error, float(np.max(np.abs(left - right))))
    if error > 1e-6:
        reasons.append("logit_error")
    for field in ("errors", "missing", "duplicates", "restarts", "metric_gaps", "schedule_misses"):
        if type(evidence.get(field)) is not int or evidence[field] != 0:
            reasons.append(field)
    for field in ("pod_cpu_ms_per_request", "memory_average_bytes", "memory_peak_bytes",
                  "throttled_seconds", "modelcar_cpu_ms_per_request"):
        if not isinstance(evidence.get(field), (int, float)) or not np.isfinite(
            evidence[field]
        ) or evidence[field] < 0:
            reasons.append(field)
    return {"valid": not reasons, "invalid_reasons": sorted(set(reasons)),
            "max_logit_error": error, "byte_exact": exact,
            "records_fingerprint": canonical_json_hash(records)}


def summarize_resources(evidence: list[dict], *, start_ns: int, end_ns: int,
                        pod_names: list[str], request_count: int = 2000) -> dict:
    """Interpolate cAdvisor counters at the measured window, excluding warm-up."""
    from .analysis import _METRIC_LINE

    if not start_ns < end_ns or not pod_names or len(set(pod_names)) != len(pod_names):
        raise ValueError("invalid measurement window or pod inventory")
    snapshots = defaultdict(dict)
    inventories = {}
    for row in evidence:
        stamp = row["timestamp_ns"]
        inventories[stamp] = {pod["name"]: (pod["uid"], pod["restart_count"])
                              for pod in row.get("pods", []) if pod["name"] in pod_names}
        for line in row.get("metrics", []):
            match = _METRIC_LINE.fullmatch(line)
            if not match:
                continue
            labels = dict(item.split("=", 1) for item in match.group("labels").split(","))
            labels = {key: value.strip('"') for key, value in labels.items()}
            pod, container = labels.get("pod"), labels.get("container")
            if pod in pod_names and container in {"kserve-container", "modelcar"}:
                key = match.group("name"), pod, container
                if key in snapshots[stamp]:
                    raise ValueError("duplicate cAdvisor series")
                snapshots[stamp][key] = float(match.group("value"))
    before = [stamp for stamp in snapshots if stamp <= start_ns]
    after = [stamp for stamp in snapshots if stamp >= end_ns]
    if not before or not after:
        raise ValueError("metrics must bracket the entire steady-state window")
    stamps = sorted(stamp for stamp in snapshots if max(before) <= stamp <= min(after))
    if any(right - left > 7_500_000_000 for left, right in zip(stamps, stamps[1:])):
        raise ValueError("metric gap")
    inventory = inventories[stamps[0]]
    if set(inventory) != set(pod_names) or any(inventories[stamp] != inventory for stamp in stamps):
        raise ValueError("pod identity or restart count changed")
    if any(restarts != 0 for _, restarts in inventory.values()):
        raise ValueError("pod restart observed")
    cpu, modelcar_cpu, throttle = 0.0, 0.0, 0.0
    memory = np.zeros(len(stamps))
    required = ("container_cpu_usage_seconds_total", "container_memory_working_set_bytes",
                "container_cpu_cfs_throttled_seconds_total")
    # Subtract the epoch before interpolation to avoid loss of integer nanosecond precision.
    x = np.array([(stamp - start_ns) / 1e9 for stamp in stamps])
    for pod in pod_names:
        for container in ("kserve-container", "modelcar"):
            for metric in required:
                key = metric, pod, container
                if any(key not in snapshots[stamp] for stamp in stamps):
                    raise ValueError(f"incomplete cAdvisor series: {key}")
                values = np.array([snapshots[stamp][key] for stamp in stamps])
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError("invalid metric values")
                if metric == "container_memory_working_set_bytes":
                    memory += values
                    continue
                if (np.diff(values) < 0).any():
                    raise ValueError("counter reset")
                delta = np.interp((end_ns - start_ns) / 1e9, x, values) - np.interp(0, x, values)
                if metric == "container_cpu_usage_seconds_total":
                    cpu += delta
                    if container == "modelcar":
                        modelcar_cpu += delta
                else:
                    throttle += delta
    return {"pod_cpu_ms_per_request": cpu * 1000 / request_count,
            "modelcar_cpu_ms_per_request": modelcar_cpu * 1000 / request_count,
            "memory_average_bytes": float(memory.mean()), "memory_peak_bytes": float(memory.max()),
            "throttled_seconds": throttle, "restarts": 0, "metric_gaps": 0,
            "counter_boundary_method": "linear interpolation of bracketing five-second samples"}
