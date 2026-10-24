#!/usr/bin/env python3
"""Deploy and measure the three current experts exactly once, then restore idle pools."""

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

from pertinence.serving.expert_once import MEASURED, MODELS, WARMUP, finalize

CONTEXT = "gke_pertinence-509116_asia-northeast3-a_pertinence-dispatcher-260919-1708"
PROJECT = "pertinence-509116"
CLUSTER = "pertinence-dispatcher-260919-1708"
ZONE = "asia-northeast3-a"
NAMESPACE = "pertinence-serving"
SERVICES = ("expert-shufflenet", "expert-mobilenet", "expert-resnet56")
MEASUREMENT_TIMEOUT_SECONDS = 1800
JOB_DEADLINE_SECONDS = 2400
RUNTIME_DIGEST = "sha256:ce9729d2d6170636ebd1bcad51aecb23765c7d775302aff0d2f9bb629c4008ca"
TRACE = "gs://pertinence-509116-pertinence-runs/pertinence/benchmarks/traces/cifar10-final-20260824"


def command(argv: list[str]) -> bytes:
    result = subprocess.run(argv, capture_output=True, timeout=1200)
    if result.returncode:
        raise RuntimeError(f"{argv[:3]}: {result.stderr.decode(errors='replace')}")
    return result.stdout


def kube(*args: str) -> bytes:
    return command(["kubectl", "--context", CONTEXT, "--request-timeout=20s",
                    "-n", NAMESPACE, *args])


def get(*args: str) -> dict:
    return json.loads(kube("get", *args, "-o", "json"))


def cloud(*args: str) -> bytes:
    return command(["gcloud", "container", *args, "--project", PROJECT,
                    "--zone", ZONE, "--quiet"])


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def wait(predicate, label: str, seconds: int = 900):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(3)
    raise TimeoutError(label)


def serving_pods() -> list[dict]:
    return [p for p in get("pods")["items"] if p["metadata"]["labels"].get(
        "serving.kserve.io/inferenceservice")]


def ready(pod: dict) -> bool:
    return not pod["metadata"].get("deletionTimestamp") and any(
        c["type"] == "Ready" and c["status"] == "True"
        for c in pod.get("status", {}).get("conditions", []))


def environment(pods: list[dict], nodes: dict) -> dict:
    return {
        "pods": sorted((p["metadata"]["uid"], p["spec"]["nodeName"], ready(p),
                        sorted((c["name"], c["restartCount"], c.get("imageID"))
                               for c in p["status"].get("containerStatuses", []))) for p in pods),
        "nodes": sorted((n["metadata"]["name"], n["metadata"]["uid"], ready(n))
                        for n in nodes["items"] if n["metadata"]["labels"].get(
                            "cloud.google.com/gke-nodepool") in ("serving-pool", "loadgen-pool")),
    }


def retrieve(pod_name: str, output: Path) -> None:
    code = ("import io,pathlib,sys,zipfile; b=io.BytesIO(); "
            "z=zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED); "
            "[z.write(p,str(p.relative_to('/work'))) for p in pathlib.Path('/work').rglob('*') "
            "if p.is_file()]; z.close(); sys.stdout.buffer.write(b.getvalue())")
    data = kube("exec", pod_name, "--", "python", "-c", code)
    (output / "archive.zip").write_bytes(data)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if any(Path(n).is_absolute() or ".." in Path(n).parts for n in archive.namelist()):
            raise ValueError("unsafe archive")
        archive.extractall(output / "raw")


def render_service(original: dict) -> dict:
    service = {k: copy.deepcopy(original[k]) for k in ("apiVersion", "kind", "spec")}
    service["metadata"] = {k: copy.deepcopy(original["metadata"][k]) for k in
                           ("name", "namespace", "labels", "annotations")
                           if k in original["metadata"]}
    service["metadata"]["annotations"].pop("kubectl.kubernetes.io/last-applied-configuration", None)
    service["metadata"]["annotations"].update({
        "serving.kserve.io/stop": "false", "pertinence.dev/diagnostics": "true",
        "pertinence.dev/expert-transport": "json", "pertinence.dev/serving-mode": "direct",
        "pertinence.dev/connection-mode": "keep-alive",
    })
    p = service["spec"]["predictor"]
    p["minReplicas"] = p["maxReplicas"] = 1
    c = p["containers"][0]
    old = c["args"]
    model, sha = (old[old.index(flag) + 1] for flag in ("--model-name", "--model-sha256"))
    if "@sha256:" not in c["image"] or "@sha256:" not in p["storageUris"][0]["uri"]:
        raise ValueError("immutable runtime and modelcar digests required")
    c["args"] = ["serve", "expert", "--model-name", model, "--model-path",
                 "/mnt/models/model.onnx", "--model-sha256", sha, "--host", "0.0.0.0",
                 "--port", "8080", "--run-fingerprint", sha, "--serving-mode", "direct",
                 "--keep-alive", "--idle-timeout", "10",
                 "--diagnostics-path", "/diagnostics/spans.jsonl"]
    p["volumes"] = [v for v in p.get("volumes", []) if v["name"] != "diagnostics"]
    p["volumes"].append({"name": "diagnostics", "emptyDir": {}})
    c["volumeMounts"] = [v for v in c.get("volumeMounts", []) if v["name"] != "diagnostics"]
    c["volumeMounts"].append({"name": "diagnostics", "mountPath": "/diagnostics"})
    p["nodeSelector"] = {"cloud.google.com/gke-nodepool": "serving-pool"}
    c["env"] = [{"name": k, "value": "1"} for k in
                ("ORT_INTRA_OP_NUM_THREADS", "ORT_INTER_OP_NUM_THREADS", "PERTINENCE_WORKERS")]
    c["resources"] = {k: {"cpu": "1", "memory": "1Gi"} for k in ("requests", "limits")}
    return service



def render_job(base_job: dict, run_id: str, runtime_image: str) -> dict:
    """Preserve scheduling/security while replacing collectors with a single no-retry worker."""
    job = copy.deepcopy(base_job)
    job["metadata"] = {"name": run_id, "namespace": NAMESPACE}
    job["spec"].update(backoffLimit=0, completions=1, parallelism=1, suspend=False,
                        activeDeadlineSeconds=JOB_DEADLINE_SECONDS)
    spec = job["spec"]["template"]["spec"]
    spec.pop("initContainers", None)
    spec["restartPolicy"] = "Never"
    spec["volumes"] = [{"name": "code", "configMap": {"name": run_id}},
                       {"name": "work", "emptyDir": {}}]
    spec["containers"] = spec["containers"][:1]
    spec["nodeSelector"] = {"cloud.google.com/gke-nodepool": "loadgen-pool"}
    c = spec["containers"][0]
    c.update(image=runtime_image, command=["python", "/code/worker.py"],
             args=["--manifest", "/code/run.json", "--trace", "/work/trace",
                   "--output", "/work/results", "--control", "/work/control.json"],
             volumeMounts=[{"name": "code", "mountPath": "/code", "readOnly": True},
                           {"name": "work", "mountPath": "/work"}])
    c["env"] = [{"name": "PYTHONUNBUFFERED", "value": "1"}]
    return job


def validate_expert_container(actual: dict, expected: dict) -> None:
    """Check experiment settings while allowing KServe's own injected metadata env."""
    for key in ("resources", "image", "args"):
        if actual.get(key) != expected.get(key):
            raise RuntimeError(f"actual expert {key} differs from manifest: "
                               f"expected={expected.get(key)!r}; actual={actual.get(key)!r}")
    actual_env = actual.get("env", [])
    by_name = {entry["name"]: entry for entry in actual_env}
    if len(by_name) != len(actual_env):
        raise RuntimeError("duplicate actual expert environment variable")
    if any(by_name.get(entry["name"]) != entry for entry in expected.get("env", [])):
        raise RuntimeError("actual expert thread/worker environment differs from manifest")
    mounts = actual.get("volumeMounts", [])
    if not any(m.get("name") == "diagnostics" and m.get("mountPath") == "/diagnostics"
               and not m.get("readOnly", False) for m in mounts):
        raise RuntimeError("actual expert diagnostics mount missing or read-only")


def recover_spans(arm: dict, output: Path) -> None:
    data = kube("exec", arm["pod"]["name"], "-c", arm["container"], "--", "python", "-c",
                "from pathlib import Path; import sys; "
                "sys.stdout.buffer.write(Path('/diagnostics/spans.jsonl').read_bytes())")
    (output / f"{arm['model_name']}-spans.jsonl").write_bytes(data)


def write_report(output: Path, run_id: str, result: dict, cleanup_errors: list) -> None:
    columns = ["model", "requests", "errors", "model_spans", "achieved_rps",
               "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "model_mean_ms",
               "model_p50_ms", "model_p95_ms", "model_p99_ms", "model_max_ms",
               "model_thread_cpu_mean_ms"]
    lines = [f"# Expert 직접 호출 결과: {run_id}", "",
             f"완료: {result['valid']}; 정리 오류: {cleanup_errors}", "",
             f"실패 사유: {result.get('failure')}", "",
             "5 RPS, 준비 200건 제외, 본 측정 모델당 2,000건, diagnostics ON.", "",
             "응답시간은 호출부터 검증·stage 추출까지, model 시간은 expert_model span(ms).",
             "model wall은 runner.predict()와 내부 검증을 포함하며 순수 ONNX 연산자 시간이 아니다.",
             "단회 저부하 관측이며 신뢰구간이나 통계적 우월성을 주장하지 않는다.", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in result["summaries"]:
        values = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in columns]
        lines.append("| " + " | ".join(values) + " |")
    (output / "results.ko.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-job", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run_id) > 50 or not all(c in "abcdefghijklmnopqrstuvwxyz0123456789-"
                                      for c in args.run_id):
        raise ValueError("run ID must be a short lowercase Kubernetes name")
    base_job = yaml.safe_load(args.base_job.read_text())
    if base_job.get("kind") != "Job":
        raise ValueError("base-job must contain one Kubernetes Job")
    worker_source = Path("src/pertinence/serving/expert_once.py").read_text()
    output = Path("artifacts/cifar10/runs/kserve") / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "controller.py").write_text(Path(__file__).read_text())
    (output / "worker.py").write_text(worker_source)
    # Validate the complete Job before allocating nodes.
    save(output / "job-preview.json", render_job(base_job, args.run_id, "runtime-preview"))
    originals = get("isvc")["items"]
    save(output / "services-before.json", originals)
    nodes = get("nodes")
    save(output / "nodes-before.json", nodes)
    workflows = get("workflows", "-A")
    save(output / "workflows-before.json", workflows)
    if any(w.get("status", {}).get("phase") not in ("Succeeded", "Failed", "Error")
           for w in workflows["items"]):
        raise RuntimeError("active workflow")
    if (any(s["metadata"].get("annotations", {}).get("serving.kserve.io/stop") != "true"
            for s in originals) or serving_pods() or any(n["metadata"]["labels"].get("cloud.google.com/gke-nodepool")
                             != "system-pool" for n in nodes["items"])):
        raise RuntimeError("expected idle serving/loadgen/pipeline pools and stopped services")
    pools = json.loads(cloud("node-pools", "list", "--cluster", CLUSTER, "--format=json"))
    save(output / "pools-before.json", pools)
    for name, machine in (("serving-pool", "n2-standard-8"),
                          ("loadgen-pool", "n2-standard-4")):
        pool = next(p for p in pools if p["name"] == name)
        if pool["config"]["machineType"] != machine:
            raise RuntimeError(f"unexpected machine type: {name}")
        auto = next(p for p in pools if p["name"] == name)["autoscaling"]
        if not auto.get("enabled") or auto.get("minNodeCount", 0) != 0 or auto["maxNodeCount"] != 1:
            raise RuntimeError("expected autoscaling 0..1 before experiment")
    selected = [next(s for s in originals if s["metadata"]["name"] == name) for name in SERVICES]
    services = [render_service(s) for s in selected]
    if any(not s["spec"]["predictor"]["containers"][0]["image"].endswith(RUNTIME_DIGEST)
           for s in services):
        raise ValueError("expected October 11 runtime digest")
    touched, pod_name = [], None
    job_submitted = False
    failure = None
    arms = []
    result = None
    try:
        for name in ("serving-pool", "loadgen-pool"):
            touched.append(name)
            print(f"Preparing {name}", flush=True)
            cloud("clusters", "update", CLUSTER, "--node-pool", name,
                  "--enable-autoscaling", "--min-nodes=1", "--max-nodes=1")
        path = output / "services.yaml"
        path.write_text(yaml.safe_dump_all(services))
        kube("apply", "--dry-run=server", "-f", str(path))
        kube("apply", "-f", str(path))

        def ready_services():
            pods = serving_pods()
            return pods if len(pods) == 3 and all(ready(p) for p in pods) else None

        pods = wait(ready_services, "experts Ready")
        save(output / "pods-start.json", {"items": pods})
        save(output / "nodes-start.json", get("nodes"))
        if len({p["spec"]["nodeName"] for p in pods}) != 1:
            raise RuntimeError("experts must share a node")
        for model, service in zip(MODELS, services):
            p = next(p for p in pods if p["metadata"]["labels"].get(
                "serving.kserve.io/inferenceservice") == service["metadata"]["name"])
            c = service["spec"]["predictor"]["containers"][0]
            actual = next(item for item in p["spec"]["containers"] if item["name"] == c["name"])
            validate_expert_container(actual, c)
            check_code = (
                "import hashlib,inspect,json,os,sys,onnxruntime; from pathlib import Path; "
                "from pertinence.serving.benchmark import KServeV2Predictor,_validate_prediction; "
                "from pertinence.serving.trace import verify_trace; "
                "from pertinence.serving.gcs import download_trace; "
                "print(json.dumps({'python':list(sys.version_info[:2]), "
                "'onnxruntime':onnxruntime.__version__, "
                "'request_ids':'request_id_prefix' in inspect.signature(KServeV2Predictor).parameters, "
                "'model_sha256':hashlib.sha256(Path('/mnt/models/model.onnx').read_bytes()).hexdigest(), "
                "'diagnostics_writable':os.access('/diagnostics',os.W_OK)}))"
            )
            checks = json.loads(kube("exec", p["metadata"]["name"], "-c", c["name"], "--",
                                     "python", "-c", check_code))
            save(output / f"{model}-runtime-checks.json", checks)
            versions = [checks["python"], checks["onnxruntime"]]
            expected_hash = c["args"][c["args"].index("--model-sha256") + 1]
            if (versions != [[3, 11], "1.20.1"] or not checks["request_ids"]
                    or not checks["diagnostics_writable"] or checks["model_sha256"] != expected_hash):
                raise RuntimeError(f"runtime preflight failed: {model}: {checks}")
            arms.append({"model_name": model, "endpoint": "http://" + service["metadata"]["name"]
                         + "-predictor." + NAMESPACE + ".svc.cluster.local",
                         "model_sha256": c["args"][c["args"].index("--model-sha256") + 1],
                         "runtime_image": c["image"],
                         "modelcar": service["spec"]["predictor"]["storageUris"][0]["uri"],
                         "container": c["name"], "versions": versions,
                         "pod": {"name": p["metadata"]["name"], "uid": p["metadata"]["uid"],
                                 "node": p["spec"]["nodeName"]}})
        manifest = {"run_id": args.run_id, "arms": arms, "trace_uri": TRACE,
                    "warmup": WARMUP, "measured": MEASURED, "diagnostics": True,
                    "repeats": 1, "rps": 5, "concurrency": 1, "batch_size": 1,
                    "retry_count": 0, "timeout_seconds": 10, "transport": "FP32 JSON",
                    "latency": "predictor call through prediction/identity validation and stage extraction"}
        save(output / "manifest.json", manifest)
        config = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {
            "name": args.run_id, "namespace": NAMESPACE}, "immutable": True,
            "data": {"run.json": json.dumps(manifest), "worker.py": worker_source}}
        job = render_job(base_job, args.run_id, arms[0]["runtime_image"])
        path = output / "job.yaml"
        path.write_text(yaml.safe_dump_all([config, job]))
        job_submitted = True  # Also clean up an API create whose response is lost.
        kube("create", "-f", str(path))
        pod_name = wait(lambda: next((p["metadata"]["name"] for p in get(
            "pods", "-l", f"job-name={args.run_id}")["items"] if ready(p)), None), "loadgen Ready")
        all_pods = [*serving_pods(), get("pod", pod_name)]
        baseline = environment(all_pods, get("nodes"))
        save(output / "environment-start.json", baseline)
        print("Measuring ShuffleNet -> MobileNet -> ResNet56, one pass only", flush=True)
        deadline = time.monotonic() + MEASUREMENT_TIMEOUT_SECONDS
        observed = 0
        while time.monotonic() < deadline:
            current = environment([*serving_pods(), get("pod", pod_name)], get("nodes"))
            save(output / "environment-latest.json", current)
            with (output / "environment-history.jsonl").open("a") as history:
                history.write(json.dumps({"time": time.time(), "environment": current}) + "\n")
            if current != baseline:
                failure = "Pod/node identity, readiness, image, or restart count changed"
            state = json.dumps({"time": time.time(), "abort": failure})
            code = ("from pathlib import Path; import os; "
                    "import json; "
                    f"Path('/work/control.tmp').write_text({state!r}); "
                    "os.replace('/work/control.tmp','/work/control.json'); "
                    "p=Path('/work/results/requests.jsonl'); "
                    "rows=p.read_text().splitlines(keepends=True) if p.exists() else []; "
                    f"print(json.dumps({{'done': Path('/work/done').exists(), 'rows': rows[{observed}:]}}))")
            progress = json.loads(kube("exec", pod_name, "--", "python", "-c", code))
            # Keep a workstation copy even if a subsequent Pod/node failure loses emptyDir.
            with (output / "requests-live.jsonl").open("a") as journal:
                journal.writelines(progress["rows"])
            observed += len(progress["rows"])
            if progress["done"]:
                break
            time.sleep(2)
        else:
            raise TimeoutError("measurement deadline; do not retry")
        retrieve(pod_name, output)
        for arm in arms:
            recover_spans(arm, output)
        trace = json.loads((output / "raw/trace.json").read_text())
        manifest["trace"] = trace
        manifest["environment"] = baseline
        save(output / "manifest.json", manifest)
        result = finalize(output)
        if failure or not result["valid"]:
            raise RuntimeError(failure or result["failure"])
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        failure = str(error)
        save(output / "INVALID.json", {"error": failure, "rerun_allowed": False})
        raise
    finally:
        # Stop inference before removing loadgen; preserve raw evidence on failure.
        cleanup_errors = []
        if pod_name and not (output / "archive.zip").exists():
            try:
                state = json.dumps({"time": time.time(), "abort": failure or "controller stopped"})
                kube("exec", pod_name, "--", "python", "-c",
                     "from pathlib import Path; "
                     f"Path('/work/control.json').write_text({state!r})")
                time.sleep(12)  # Allow the one in-flight 10-second request to finish.
                retrieve(pod_name, output)
            except Exception as error:
                cleanup_errors.append(f"partial evidence recovery: {error}")
        for arm in arms:
            if not (output / f"{arm['model_name']}-spans.jsonl").exists():
                try:
                    recover_spans(arm, output)
                except Exception as error:
                    cleanup_errors.append(f"server span recovery: {error}")
        for service in selected:
            try:
                name = service["metadata"]["name"]
                kube("annotate", "isvc", name, "serving.kserve.io/stop=true", "--overwrite")
            except Exception as error:
                cleanup_errors.append(str(error))
        if job_submitted:
            try:
                kube("delete", "job", args.run_id, "--wait=true", "--ignore-not-found=true")
            except Exception as error:
                cleanup_errors.append(str(error))
        for name in reversed(touched):
            try:
                print(f"Restoring {name} to autoscaling 0..1, size 0", flush=True)
                cloud("clusters", "update", CLUSTER, "--node-pool", name,
                      "--enable-autoscaling", "--min-nodes=0", "--max-nodes=1")
                cloud("clusters", "resize", CLUSTER, "--node-pool", name, "--num-nodes=0")
            except Exception as error:
                cleanup_errors.append(str(error))
        try:
            save(output / "services-after.json", get("isvc"))
            final_nodes = get("nodes")
            save(output / "nodes-after.json", final_nodes)
            final_pools = json.loads(cloud("node-pools", "list", "--cluster", CLUSTER,
                                           "--format=json"))
            save(output / "pools-after.json", final_pools)
            if serving_pods() or any(n["metadata"]["labels"].get("cloud.google.com/gke-nodepool")
                                     != "system-pool" for n in final_nodes["items"]):
                cleanup_errors.append("serving Pods or experiment nodes remain")
            for name in touched:
                before = next(p for p in pools if p["name"] == name)["autoscaling"]
                after = next(p for p in final_pools if p["name"] == name)["autoscaling"]
                if before != after:
                    cleanup_errors.append(f"autoscaling not restored: {name}")
        except Exception as error:
            cleanup_errors.append(str(error))
        save(output / "cleanup.json", {"errors": cleanup_errors, "stopped": not cleanup_errors})
        save(output / "verification.json", {"complete": bool(result and result["valid"]
                                                             and not failure and not cleanup_errors),
                                            "failure": failure, "cleanup_errors": cleanup_errors})
        if result is None:
            result = {"valid": False, "failure": failure, "summaries": [],
                      "validation": "no validated statistics; inspect partial evidence"}
        if result:
            result["valid"] = bool(result["valid"] and not failure and not cleanup_errors)
            save(output / "result.json", result)
            write_report(output, args.run_id, result, cleanup_errors)
        if cleanup_errors:
            raise RuntimeError(f"cleanup incomplete: {cleanup_errors}")


if __name__ == "__main__":
    main()
