#!/usr/bin/env python3
"""Sequential GKE suite runner with local evidence handoff and fail-closed blocks."""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import subprocess
import time
import zipfile

import yaml

from pertinence.serving.cli.benchmark import _write_json_exclusive as _write_new_json
from pertinence.serving.contracts import seal_manifest
from pertinence.serving.transport_cli import read_rows
from pertinence.serving.transport_deployment import render_services
from pertinence.serving.transport_suite import (
    VARIANTS, compare_runs, join_diagnostics, latency_summary, run_manifest, schedule,
    summarize_resources, verify_run_records,
)

CONTEXT = "gke_pertinence-509116_asia-northeast3-a_pertinence-dispatcher-260919-1708"
NAMESPACE = "pertinence-serving"
SERVICES = ["expert-shufflenet", "expert-mobilenet", "expert-resnet56",
            "router-dispatcher-000733", "router-dispatcher-000733-fused"]


def _write_json_exclusive(path: Path, value: dict) -> None:
    if path.is_symlink():
        raise ValueError("artifact must not be a symlink")
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"existing artifact differs: {path}")
        return
    _write_new_json(path, value)



def kubectl(*args: str, check: bool = True,
            timeout_seconds: int = 20) -> subprocess.CompletedProcess:
    result = subprocess.run(["kubectl", "--context", CONTEXT,
                             f"--request-timeout={timeout_seconds}s", "-n", NAMESPACE, *args],
                            check=False, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(f"kubectl {args[0]} failed: {result.stderr.decode(errors='replace')}")
    return result


def get(resource: str, *args: str) -> dict:
    for attempt in range(5):
        result = kubectl("get", resource, *args, "-o", "json", check=False, timeout_seconds=60)
        if result.returncode == 0:
            return json.loads(result.stdout)
        print(json.dumps({"event": "control_read_retry", "resource": resource,
                          "attempt": attempt, "error": result.stderr.decode(errors="replace")}),
              flush=True)
        time.sleep(attempt + 1)
    raise RuntimeError(f"Kubernetes read failed after five attempts: {resource}")


def node_identity() -> tuple[str, str]:
    nodes = get("nodes")["items"]
    result = []
    for pool in ("serving-pool", "loadgen-pool"):
        selected = [node for node in nodes if node["metadata"]["labels"].get(
            "cloud.google.com/gke-nodepool") == pool]
        if len(selected) != 1 or not any(item["type"] == "Ready" and item["status"] == "True"
                                        for item in selected[0]["status"]["conditions"]):
            raise RuntimeError(f"exactly one Ready {pool} node is required")
        result.append(selected[0]["metadata"]["uid"])
    return tuple(result)


def wait_until(predicate, label: str, seconds: int = 600):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(3)
    raise TimeoutError(label)


def active_serving_pods() -> list[dict]:
    return [pod for pod in get("pods")["items"] if pod["metadata"]["labels"].get(
        "serving.kserve.io/inferenceservice") in SERVICES]


def stop_services() -> None:
    existing = {item["metadata"]["name"] for item in get("inferenceservices")["items"]}
    for name in SERVICES:
        if name in existing:
            kubectl("annotate", "inferenceservice", name, "serving.kserve.io/stop=true", "--overwrite")
    wait_until(lambda: not active_serving_pods(), "previous serving workload did not stop")


def archive(pod: str, container: str, root: str, output: Path) -> None:
    code = ("import io,pathlib,sys,zipfile; "
            f"root=pathlib.Path({root!r}); buf=io.BytesIO(); "
            "z=zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED); "
            "[z.write(p,str(p.relative_to(root))) for p in root.rglob('*') "
            "if p.is_file() and not p.is_symlink()]; z.close(); sys.stdout.buffer.write(buf.getvalue())")
    output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        result = kubectl("exec", pod, "-c", container, "--", "python", "-c", code,
                         check=False, timeout_seconds=120)
        try:
            if result.returncode:
                raise RuntimeError(result.stderr.decode(errors="replace"))
            with zipfile.ZipFile(io.BytesIO(result.stdout)) as zipped:
                if zipped.testzip() is not None:
                    raise ValueError("artifact archive CRC mismatch")
            break
        except (RuntimeError, ValueError, zipfile.BadZipFile) as error:
            (output.parent / f"{output.name}.attempt-{attempt}.error.txt").write_text(str(error))
            if attempt == 2:
                raise
            time.sleep(1)  # artifact retrieval only; inference requests are never retried
    output.mkdir(parents=True, exist_ok=False)
    (output / "archive.zip").write_bytes(result.stdout)
    with zipfile.ZipFile(io.BytesIO(result.stdout)) as zipped:
        for name in zipped.namelist():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("unsafe artifact archive")
        zipped.extractall(output)


def execute(args, arm: str, block: int, diagnostics: bool, nodes: tuple[str, str], suffix: str) -> Path:
    if node_identity() != nodes:
        raise RuntimeError("node UID changed; discard and repeat entire block")
    stop_services()
    base_id = f"{args.session}-{suffix}-{arm.lower()}"
    run_id = base_id
    attempt = 0
    while (args.output / run_id).exists():
        existing = args.output / run_id
        if (existing / "resources.json").exists():
            previous = json.loads((existing / "manifest.json").read_text())
            expected = run_manifest(
                run_id=run_id, arm=arm, block=block, diagnostics=diagnostics,
                promotion_fingerprint=json.loads(args.promotion.read_text())["fingerprint"],
                trace_fingerprint=args.trace_fingerprint, runtime_image=args.runtime,
                serving_node_uid=nodes[0], loadgen_node_uid=nodes[1],
            )
            if previous != expected:
                raise RuntimeError("completed arm is incompatible; repeat its entire block")
            print(json.dumps({"event": "reused_complete", "run": run_id}), flush=True)
            return existing
        attempt += 1
        run_id = f"{base_id}-a{attempt}"
    destination = args.output / run_id
    destination.mkdir()
    promotion = json.loads(args.promotion.read_text())
    manifest = run_manifest(run_id=run_id, arm=arm, block=block, diagnostics=diagnostics,
                            promotion_fingerprint=promotion["fingerprint"],
                            trace_fingerprint=args.trace_fingerprint, runtime_image=args.runtime,
                            serving_node_uid=nodes[0], loadgen_node_uid=nodes[1])
    _write_json_exclusive(destination / "manifest.json", manifest)
    services = render_services(list(yaml.safe_load_all(args.base_services.read_text())),
                               manifest, promotion, fused_modelcar=args.fused_modelcar,
                               hybrid_modelcar=args.hybrid_modelcar)
    cm_name = "transport-run-" + manifest["fingerprint"][:16]
    promotion_name = "transport-promotion-" + manifest["fingerprint"][:16]
    resources = [{"apiVersion": "v1", "kind": "ConfigMap", "immutable": True,
                  "metadata": {"name": name, "namespace": NAMESPACE}, "data": data}
                 for name, data in ((cm_name, {"run.json": json.dumps(manifest)}),
                                    (promotion_name, {"promotion.json": json.dumps(promotion)}))]
    for service in services:
        service["metadata"]["annotations"]["serving.kserve.io/stop"] = "false"
    path = destination / "workloads.yaml"
    path.write_text(yaml.safe_dump_all([*resources, *services]))
    kubectl("apply", "-f", str(path))
    names = {service["metadata"]["name"] for service in services}

    def ready():
        pods = active_serving_pods()
        if len(pods) != len(names):
            return False
        for pod in pods:
            if pod["metadata"].get("deletionTimestamp") or not any(
                item["type"] == "Ready" and item["status"] == "True"
                for item in pod["status"].get("conditions", [])
            ):
                return False
        return pods

    pods = wait_until(ready, "serving readiness", seconds=900)
    _write_json_exclusive(destination / "pods-start.json", {"items": pods})
    _write_json_exclusive(destination / "nodes-start.json", get("nodes"))
    job = copy.deepcopy(yaml.safe_load(args.base_job.read_text()))
    job["metadata"]["name"] = run_id
    job["metadata"]["labels"]["pertinence.dev/arm-id"] = arm
    job["spec"]["suspend"] = False
    job["spec"]["activeDeadlineSeconds"] = 1800
    template = job["spec"]["template"]
    template["metadata"]["labels"]["pertinence.dev/arm-id"] = arm
    spec = template["spec"]
    for volume in spec["volumes"]:
        if volume["name"] == "promotion":
            volume["configMap"]["name"] = promotion_name
        elif volume["name"] == "benchmark-config":
            volume["configMap"]["name"] = cm_name
    collector = spec["initContainers"][0]
    collector["image"] = args.runtime
    collector["args"] = ["benchmark", "collect-evidence", "--output", f"/work/{run_id}/evidence.jsonl",
                         "--done", "/work/loadgen.done", "--completed", "/work/collector.done"]
    loadgen = spec["containers"][0]
    loadgen["image"] = args.runtime
    loadgen["command"] = ["python", "-m", "pertinence.serving.transport_cli"]
    loadgen["args"] = ["run", "--manifest", "/etc/benchmark/run.json", "--promotion",
                       "/etc/pertinence/promotion.json", "--trace", "/work/trace", "--trace-uri",
                       args.trace_uri, "--endpoint",
                       "http://router-dispatcher-000733-predictor.pertinence-serving.svc.cluster.local",
                       "--output", f"/work/{run_id}", "--collector-done", "/work/loadgen.done",
                       "--collector-completed", "/work/collector.done",
                       "--handoff-file", "/work/handoff.done"]
    job_path = destination / "job.yaml"
    job_path.write_text(yaml.safe_dump(job))
    kubectl("apply", "-f", str(job_path))
    job_pod = wait_until(lambda: next((pod["metadata"]["name"] for pod in get(
        "pods", "-l", f"job-name={run_id}")["items"]
        if pod["status"].get("phase") == "Running"), None), "loadgen scheduling", seconds=600)
    print(json.dumps({"event": "measuring", "run": run_id}), flush=True)

    def finished():
        pod = get("pod", job_pod)
        for status in pod["status"].get("containerStatuses", []):
            if status["name"] == "loadgen" and "terminated" in status["state"]:
                log = kubectl("logs", job_pod, "-c", "loadgen", check=False).stdout
                (destination / "loadgen.log").write_bytes(log)
                raise RuntimeError(f"loadgen terminated before artifact handoff: {run_id}")
        return kubectl("exec", job_pod, "-c", "loadgen", "--", "python", "-c",
                       f"from pathlib import Path; raise SystemExit(not Path('/work/{run_id}/ready.json').exists())",
                       check=False).returncode == 0

    wait_until(finished, "measurement or artifact handoff", seconds=1000)
    archive(job_pod, "loadgen", f"/work/{run_id}", destination / "raw")
    kubectl("exec", job_pod, "-c", "loadgen", "--", "python", "-c",
            "from pathlib import Path; Path('/work/handoff.done').touch()")
    if diagnostics:
        for pod in pods:
            archive(pod["metadata"]["name"], "kserve-container", "/diagnostics",
                    destination / "diagnostics" / pod["metadata"]["name"])
    _write_json_exclusive(destination / "pods-end.json", get("pods"))
    _write_json_exclusive(destination / "nodes-end.json", get("nodes"))
    if node_identity() != nodes:
        raise RuntimeError("node UID changed; discard and repeat entire block")
    rows = read_rows(destination / "raw/requests.jsonl")
    execution = json.loads((destination / "raw/execution.json").read_text())
    if execution["request_error"] or len(rows) != 2000 or not execution["collector_complete"]:
        raise RuntimeError(f"invalid requests: {execution}")
    markers = json.loads((destination / "raw/markers.json").read_text())["markers"]
    times = {row["event"]: row["timestamp_ns"] for row in markers}
    evidence = summarize_resources(read_rows(destination / "raw/evidence.jsonl"),
                                   start_ns=times["arm_start"], end_ns=times["arm_end"],
                                   pod_names=[pod["metadata"]["name"] for pod in pods])
    evidence.update(errors=0, missing=0, duplicates=0, schedule_misses=0)
    _write_json_exclusive(destination / "resources.json", evidence)
    wait_until(lambda: get("job", run_id).get("status", {}).get("succeeded") == 1,
               "loadgen completion", seconds=120)
    stop_services()
    print(json.dumps({"event": "complete", "run": run_id}), flush=True)
    return destination


def summarize(path: Path, reference: Path) -> dict:
    rows = read_rows(path / "raw/requests.jsonl")
    evidence = json.loads((path / "resources.json").read_text())
    validity = verify_run_records(rows, read_rows(reference / "raw/requests.jsonl"), evidence)
    latency = latency_summary(rows)
    summary = {"manifest": json.loads((path / "manifest.json").read_text()), **validity,
               **latency["overall"], "latency": latency, "evidence": evidence,
               "pod_cpu_ms_per_request": evidence["pod_cpu_ms_per_request"],
               "achieved_rps": len(rows) * 1e9 / (rows[-1]["completed_ns"] - rows[0]["started_ns"])}
    summary = seal_manifest(summary)
    if (path / "summary.json").exists():
        if json.loads((path / "summary.json").read_text()) != summary:
            raise ValueError("existing summary differs; refusing replacement")
    else:
        _write_json_exclusive(path / "summary.json", summary)
    if not validity["valid"]:
        raise RuntimeError(f"parity failed: {path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("base-services", "base-job", "promotion", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("runtime", "fused-modelcar", "hybrid-modelcar", "trace-uri", "trace-fingerprint",
                 "session"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--phase", choices=("all", "diagnostics", "performance"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "schedule.json").exists():
        if json.loads((args.output / "schedule.json").read_text()) != {"blocks": schedule()}:
            raise ValueError("existing schedule differs")
    else:
        _write_json_exclusive(args.output / "schedule.json", {"blocks": schedule()})
    try:
        nodes = node_identity()
        diagnostic = ({arm: execute(args, arm, 0, True, nodes, "diag") for arm in VARIANTS} if args.phase != "performance" else {})
        for arm, path in diagnostic.items():
            summarize(path, diagnostic["F0"])
            if arm != "F0":
                router, experts = [], []
                for source in (path / "diagnostics").glob("*/spans.jsonl"):
                    (router if source.parent.name.startswith("router") else experts).extend(
                        read_rows(source))
                if not (path / "joined.json").exists():
                    _write_json_exclusive(path / "joined.json", {
                        "calls": join_diagnostics(router, experts)})
        if args.phase == "diagnostics":
            return
        summaries = []
        for block, order in enumerate(schedule()):
            nodes = node_identity()
            paths = {arm: execute(args, arm, block, False, nodes, f"b{block}") for arm in order}
            summaries.extend(summarize(path, paths["F0"]) for path in paths.values())
        comparison = compare_runs(summaries)
        _write_json_exclusive(args.output / "comparison.json", comparison)
        selected = comparison["selected_distributed"]
        overhead = {}
        for arm in dict.fromkeys(["D0", selected] if selected else ["D0"]):
            nodes = node_identity()
            paths = {enabled: execute(args, arm, 0, enabled, nodes,
                                      "overhead-on" if enabled else "overhead-off")
                     for enabled in (False, True)}
            overhead[arm] = {str(enabled): summarize(path, paths[False])
                             for enabled, path in paths.items()}
        _write_json_exclusive(args.output / "instrumentation-overhead.json", overhead)
    finally:
        stop_services()


if __name__ == "__main__":
    main()
