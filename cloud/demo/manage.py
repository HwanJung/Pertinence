#!/usr/bin/env python3
"""Create an isolated demo router and restore the pre-demo serving state."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import socket
import sys
import time
from string import Template
from uuid import uuid4

import yaml

from pertinence.serving.contracts import PromotionManifestV1

PROJECT = "pertinence-509116"
ZONE = "asia-northeast3-a"
CLUSTER = "pertinence-dispatcher-260919-1708"
CONTEXT = f"gke_{PROJECT}_{ZONE}_{CLUSTER}"
NAMESPACE = "pertinence-serving"
EXPERT_NAMES = ("expert-shufflenet", "expert-mobilenet", "expert-resnet56")
EXPERT_IDS = ("shufflenetv2_x0_5", "mobilenetv2_x0_5", "resnet56")
ACTIVE_EXPERTS = tuple(f"demo-{name}" for name in EXPERT_NAMES[1:])
HYBRID_MODELCAR = (
    "asia-northeast3-docker.pkg.dev/pertinence-509116/pertinence/"
    "dispatcher-000733-hybrid-modelcar@sha256:"
    "2c133fb51aef75d5e01f49e403e6ae4fe7bd7fa94962b93503d67e89e499bf17"
)
STOP = "serving.kserve.io/stop"
OWNER = "pertinence.dev/demo-session"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(*command: str, input_text: str | None = None) -> str:
    result = subprocess.run(command, input=input_text, text=True, check=True, capture_output=True)
    return result.stdout


def kube(*args: str, input_text: str | None = None) -> str:
    return run("kubectl", "--context", CONTEXT, "--request-timeout=30s", "-n", NAMESPACE,
               *args, input_text=input_text)


def get(kind: str, name: str) -> dict | None:
    value = kube("get", kind, name, "--ignore-not-found", "-o", "json")
    return json.loads(value) if value.strip() else None


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def require_pinned(image: str) -> None:
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image) or image.endswith("0" * 64):
        raise ValueError("images must use a non-sentinel SHA-256 digest")


def demo_resources(image: str, router: dict, experts: list[dict], promotion: dict,
                   session: str) -> list[dict]:
    """Render a new hybrid router using the existing sealed promotion."""
    require_pinned(image)
    PromotionManifestV1.from_dict(promotion)
    if [item["id"] for item in promotion["experts"]] != list(EXPERT_IDS):
        raise ValueError("unexpected promoted expert order")
    for expert, promoted in zip(experts, promotion["experts"], strict=True):
        predictor = expert["spec"]["predictor"]
        container = predictor["containers"][0]
        args = container["args"]
        require_pinned(container["image"])
        if (args[args.index("--model-name") + 1] != promoted["id"]
                or args[args.index("--model-sha256") + 1] != promoted["model_sha256"]
                or predictor["storageUris"] != [{"mountPath": "/mnt/models",
                                                 "uri": "oci://" + promoted["modelcar_image"]}]):
            raise ValueError("existing expert differs from the sealed promotion")
    predictor = deepcopy(router["spec"]["predictor"])
    predictor["minReplicas"] = predictor["maxReplicas"] = 1
    predictor["storageUris"] = [{"mountPath": "/mnt/models",
                                "uri": "oci://" + HYBRID_MODELCAR}]
    container = predictor["containers"][0]
    # Reuse the currently deployed runtime; the model loader verifies the sealed bundle.
    require_pinned(container["image"])
    container["args"] = ["serve", "router", "--promotion", "/etc/pertinence/promotion.json",
                         "--bundle-path", "/mnt/models", "--router-mode", "hybrid-dynamic",
                         "--expert-transport", "binary",
                         "--timeout-seconds", "10"]
    for name, expert_id in zip(ACTIVE_EXPERTS, EXPERT_IDS[1:], strict=True):
        container["args"] += ["--expert-url", (
            f"{expert_id}=http://{name}-predictor.{NAMESPACE}.svc.cluster.local"
        )]
    container["resources"] = {"requests": {"cpu": "2", "memory": "2Gi"},
                              "limits": {"cpu": "2", "memory": "2Gi"}}
    container["volumeMounts"] = [{"name": "promotion", "mountPath": "/etc/pertinence",
                                 "readOnly": True}]
    predictor["volumes"] = [{"name": "promotion", "configMap": {"name": "demo-promotion"}}]
    resources = list(yaml.safe_load_all(Template(
        (HERE / "workloads.yaml.tmpl").read_text()
    ).substitute(DEMO_IMAGE=image)))
    resources.extend([
        {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "demo-promotion"},
         "immutable": True, "data": {"promotion.json": json.dumps(promotion)}},
        {"apiVersion": "serving.kserve.io/v1beta1", "kind": "InferenceService",
         "metadata": {"name": "demo-dispatcher-000733", "annotations": {
             "serving.kserve.io/deploymentMode": "Standard", STOP: "false",
             "pertinence.dev/promotion-fingerprint": promotion["fingerprint"],
         }}, "spec": {"predictor": predictor}},
    ])
    for expert, promoted, name in zip(
        experts[1:], promotion["experts"][1:], ACTIVE_EXPERTS, strict=True,
    ):
        predictor = deepcopy(expert["spec"]["predictor"])
        predictor["minReplicas"] = predictor["maxReplicas"] = 1
        container = predictor["containers"][0]
        # HTTP/1.0 closes each connection, so a demo can remain idle without stale sockets.
        # This does not retry requests or change the benchmark runtime.
        container["args"] = ["serve", "expert", "--model-name", promoted["id"],
                             "--model-path", "/mnt/models/model.onnx",
                             "--model-sha256", promoted["model_sha256"]]
        container["resources"] = {"requests": {"cpu": "1", "memory": "1Gi"},
                                  "limits": {"cpu": "1", "memory": "1Gi"}}
        container.pop("volumeMounts", None)
        predictor.pop("volumes", None)
        resources.insert(-1, {
            "apiVersion": "serving.kserve.io/v1beta1", "kind": "InferenceService",
            "metadata": {"name": name, "annotations": {
                "serving.kserve.io/deploymentMode": "Standard", STOP: "false",
            }}, "spec": {"predictor": predictor},
        })
    for resource in resources:
        resource["metadata"]["namespace"] = NAMESPACE
        resource["metadata"].setdefault("labels", {})[OWNER] = session
    return resources


def pool_state() -> dict:
    pool = json.loads(run("gcloud", "container", "node-pools", "describe", "serving-pool",
                          "--project", PROJECT, "--zone", ZONE, "--cluster", CLUSTER,
                          "--format=json"))
    groups = pool["instanceGroupUrls"]
    if len(groups) != 1:
        raise ValueError("demo requires the existing single-zone serving pool")
    group = groups[0].rsplit("/", 1)[-1]
    size = int(run("gcloud", "compute", "instance-groups", "managed", "describe", group,
                   "--project", PROJECT, "--zone", ZONE, "--format=value(targetSize)").strip())
    if size not in (0, 1):
        raise ValueError("unexpected serving pool size")
    return {"definition": pool, "size": size}


def resize(size: int) -> None:
    print(f"serving-pool → {size} node(s)", flush=True)
    run("gcloud", "container", "clusters", "resize", CLUSTER, "--node-pool", "serving-pool",
        "--num-nodes", str(size), "--project", PROJECT, "--zone", ZONE, "--quiet")


def stop(directory: Path) -> None:
    path = directory / "state.json"
    state = json.loads(path.read_text())
    if state["context"] != CONTEXT:
        raise ValueError("state context mismatch")
    if state.get("stopped"):
        print("Demo already stopped.")
        return
    # Delete only objects created by this session, including a partially completed start.
    for item in reversed(state["resources"]):
        kind, name = item["kind"], item["metadata"]["name"]
        current = get(kind, name)
        if current:
            if current["metadata"].get("labels", {}).get(OWNER) != state["session"]:
                raise ValueError(f"refusing to delete unowned {kind}/{name}")
            kube("delete", kind, name, "--wait=true", "--timeout=180s")
    for previous in state["experts"]:
        name = previous["metadata"]["name"]
        if name not in state.get("activated_experts", EXPERT_NAMES):
            continue
        current = get("inferenceservice", name)
        if (not current or current["metadata"]["uid"] != previous["metadata"]["uid"]
                or current["spec"] != previous["spec"]):
            raise ValueError(f"{name} changed outside the demo; refusing automatic restoration")
        original = previous["metadata"].get("annotations", {}).get(STOP)
        if current["metadata"].get("annotations", {}).get(STOP) != original:
            kube("annotate", "inferenceservice", name,
                 f"{STOP}={original}" if original is not None else f"{STOP}-", "--overwrite")
        if original == "true":
            kube("wait", '--for=jsonpath={.status.conditions[?(@.type=="Stopped")].status}=True',
                 "--timeout=180s", f"inferenceservice/{name}")
            kube("wait", "--for=delete", "pod", "-l",
                 f"serving.kserve.io/inferenceservice={name}", "--timeout=180s")
    if state["pool"]["size"] == 0 and state.get("resize_requested"):
        # Do not resize under a workload that arrived independently during the demo.
        nodes = json.loads(kube("get", "nodes", "-l",
                               "cloud.google.com/gke-nodepool=serving-pool", "-o", "json"))
        node_names = {item["metadata"]["name"] for item in nodes["items"]}
        pods = json.loads(run("kubectl", "--context", CONTEXT, "get", "pods", "-A", "-o", "json"))
        for pod in pods["items"]:
            if pod["spec"].get("nodeName") not in node_names:
                continue
            owners = pod["metadata"].get("ownerReferences", [])
            static_node_agent = (
                pod["metadata"].get("namespace") == "kube-system"
                and "kubernetes.io/config.mirror" in pod["metadata"].get("annotations", {})
                and any(owner["kind"] == "Node" for owner in owners)
            )
            if (pod["status"].get("phase") not in {"Succeeded", "Failed"}
                    and not pod["metadata"].get("deletionTimestamp")
                    and not static_node_agent
                    and not any(owner["kind"] == "DaemonSet" for owner in owners)):
                raise ValueError("serving pool still has workload Pods; retry stop after they exit")
        resize(0)
    state["stopped"] = True
    save(path, state)
    save(directory / "pool-after.json", pool_state())
    print("Demo stopped; original expert stop annotations and pool size restored.", flush=True)


def start(directory: Path, image: str) -> None:
    require_pinned(image)
    directory = directory.resolve()
    if not directory.is_relative_to(ROOT / "artifacts" / "cifar10"):
        raise ValueError("state directory must be under artifacts/cifar10/")
    experts = [get("inferenceservice", name) for name in EXPERT_NAMES]
    router = get("inferenceservice", "router-dispatcher-000733")
    if router is None or any(expert is None for expert in experts):
        raise ValueError("existing router and three experts are required")
    volumes = router["spec"]["predictor"]["volumes"]
    config_name = next(v["configMap"]["name"] for v in volumes if v["name"] == "promotion")
    config = get("configmap", config_name)
    promotion = json.loads(config["data"]["promotion.json"])
    session = uuid4().hex[:16]
    resources = demo_resources(image, router, experts, promotion, session)
    for resource in resources:
        if get(resource["kind"], resource["metadata"]["name"]):
            raise ValueError("existing demo resources found; stop the previous session first")
    jobs = json.loads(kube("get", "jobs", "-o", "json"))
    if any(job.get("status", {}).get("active", 0) for job in jobs["items"]):
        raise ValueError("active serving/loadgen Jobs must finish before starting the demo")
    pool = pool_state()
    directory.mkdir(parents=True, exist_ok=False)
    state = {"context": CONTEXT, "session": session, "pool": pool, "experts": experts,
             "activated_experts": [], "serving_mode": "hybrid",
             "router_before": router, "resources": resources, "resize_requested": False}
    save(directory / "state.json", state)
    (directory / "workloads.yaml").write_text(yaml.safe_dump_all(resources, sort_keys=False))
    try:
        # Persist intent before a mutation, so stop can recover even after interruption.
        if pool["size"] == 0:
            state["resize_requested"] = True
            save(directory / "state.json", state)
            resize(1)
        for resource in resources:
            kube("create", "-f", "-", input_text=json.dumps(resource))
        print("Waiting for experts and demo router…", flush=True)
        for name in (*ACTIVE_EXPERTS, "demo-dispatcher-000733"):
            kube("wait", '--for=jsonpath={.status.conditions[?(@.type=="Ready")].status}=True',
                 "--timeout=10m", f"inferenceservice/{name}")
        kube("rollout", "status", "deployment/pertinence-demo", "--timeout=10m")
    except Exception:
        print(f"Start failed. Restore with: .venv/bin/python cloud/demo/manage.py stop {directory}",
              flush=True)
        raise
    state["ready"] = True
    save(directory / "state.json", state)
    print("Ready. Open http://localhost:8088 after running:", flush=True)
    print(f"kubectl --context {CONTEXT} -n {NAMESPACE} port-forward "
          "--address 127.0.0.1 service/pertinence-demo 8088:80", flush=True)


def owned_current(resource: dict, session: str) -> dict:
    current = get(resource["kind"], resource["metadata"]["name"])
    if not current or current["metadata"].get("labels", {}).get(OWNER) != session:
        raise ValueError("missing or unowned demo resource")
    return current


def replace_workload(resource: dict, session: str) -> None:
    """Replace only a session-owned spec, waiting out old router Pods."""
    current = owned_current(resource, session)
    kind, name = resource["kind"], resource["metadata"]["name"]
    if kind == "InferenceService":
        # Avoid holding both router generations on the single serving node.
        kube("delete", kind, name, "--wait=true", "--timeout=180s")
        kube("wait", "--for=delete", "pod", "-l",
             f"serving.kserve.io/inferenceservice={name}", "--timeout=180s")
        kube("create", "-f", "-", input_text=json.dumps(resource))
    else:
        kube("patch", kind, name, "--type=json", "-p", json.dumps([
            {"op": "test", "path": "/metadata/resourceVersion",
             "value": current["metadata"]["resourceVersion"]},
            {"op": "replace", "path": "/spec", "value": resource["spec"]},
        ]))


def wait_ready() -> None:
    for name in (*ACTIVE_EXPERTS, "demo-dispatcher-000733"):
        kube("wait", '--for=jsonpath={.status.conditions[?(@.type=="Ready")].status}=True',
             "--timeout=10m", f"inferenceservice/{name}")
    kube("rollout", "status", "deployment/pertinence-demo", "--timeout=10m")


def verify_session(directory: Path, trace: Path, mode: str, baseline: Path | None = None) -> None:
    """Use fresh, loopback-only forwards bound to the current Pods for verification."""
    import httpx

    processes = []
    ports = []
    try:
        for service in ("pertinence-demo", "demo-dispatcher-000733-predictor"):
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            ports.append(port)
            log = directory.parent / f"{directory.name}-{service}-forward.log"
            with log.open("w") as output:
                processes.append(subprocess.Popen([
                    "kubectl", "--context", CONTEXT, "-n", NAMESPACE, "port-forward",
                    "--address", "127.0.0.1", f"service/{service}", f"{port}:80",
                ], stdout=output, stderr=subprocess.STDOUT))
        with httpx.Client(trust_env=False, timeout=2) as client:
            deadline = time.monotonic() + 60
            for port in ports:
                while True:
                    if any(process.poll() is not None for process in processes):
                        raise RuntimeError("verification port-forward exited")
                    try:
                        path = "/health/ready" if port == ports[0] else "/v2/health/ready"
                        if client.get(f"http://127.0.0.1:{port}{path}").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError("verification readiness deadline")
                    time.sleep(.5)
        command = [sys.executable, str(HERE / "verify.py"), "--trace", str(trace),
                   "--output", str(directory), "--demo-url", f"http://127.0.0.1:{ports[0]}",
                   "--router-url", f"http://127.0.0.1:{ports[1]}", "--expected-mode", mode]
        if baseline is not None:
            command += ["--baseline", str(baseline)]
        print(run(*command), flush=True)
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def upgrade_hybrid(directory: Path, image: str, trace: Path) -> None:
    """Back up a distributed session, verify Hybrid parity, then retire its extractor."""
    require_pinned(image)
    path = directory / "state.json"
    state = json.loads(path.read_text())
    if state["context"] != CONTEXT or state.get("stopped") or not state.get("ready"):
        raise ValueError("upgrade requires a ready session in the expected context")
    if state.get("serving_mode") == "hybrid":
        raise ValueError("session is already hybrid")
    recovery = directory / "upgrade-hybrid"
    recovery.mkdir(exist_ok=False)
    save(recovery / "state-before.json", state)
    snapshots = [owned_current(item, state["session"]) for item in state["resources"]]
    save(recovery / "inventory-before.json", {"resources": snapshots, "pool": pool_state()})
    old = deepcopy(state["resources"])
    for item, current in zip(old, snapshots, strict=True):
        if "spec" in item:
            item["spec"] = current["spec"]
    (recovery / "workloads-before.yaml").write_text(yaml.safe_dump_all(old, sort_keys=False))
    promotion = json.loads(next(item for item in snapshots
                                if item["kind"] == "ConfigMap")["data"]["promotion.json"])
    resources = demo_resources(image, state["router_before"], state["experts"],
                               promotion, state["session"])
    (recovery / "workloads-hybrid.yaml").write_text(yaml.safe_dump_all(resources, sort_keys=False))
    verify_session(recovery / "distributed", trace, "distributed")
    targets = [item for item in resources if item["kind"] == "Deployment" or (
        item["kind"] == "InferenceService" and item["metadata"]["name"] == "demo-dispatcher-000733"
    )]
    state["upgrade"] = {"status": "applying", "recovery": str(recovery)}
    save(path, state)
    try:
        for item in reversed(targets):
            replace_workload(item, state["session"])
        wait_ready()
        verify_session(recovery / "hybrid", trace, "hybrid",
                       recovery / "distributed/verification.json")
    except BaseException:
        state["upgrade"]["status"] = "rollback-required"
        save(path, state)
        # stop can still clean every object using the original resource list if recovery fails.
        for target in targets:
            previous = next(item for item in old if item["kind"] == target["kind"]
                            and item["metadata"]["name"] == target["metadata"]["name"])
            current = get(previous["kind"], previous["metadata"]["name"])
            if current is None:
                kube("create", "-f", "-", input_text=json.dumps(previous))
            else:
                replace_workload(previous, state["session"])
        wait_ready()
        state["upgrade"]["status"] = "rolled-back"
        save(path, state)
        raise
    # Commit verified Hybrid state before cleanup; stop retains the union during interruptions.
    shuffle = next(item for item in old if item["metadata"]["name"] == "demo-expert-shufflenet")
    state["serving_mode"] = "hybrid"
    state["resources"] = [*resources, shuffle]
    state["upgrade"]["status"] = "verified-cleanup-pending"
    save(path, state)
    owned_current(shuffle, state["session"])
    kube("delete", "InferenceService", "demo-expert-shufflenet", "--wait=true", "--timeout=180s")
    state["resources"] = resources
    state["upgrade"]["status"] = "complete"
    save(path, state)
    (directory / "workloads.yaml").write_text(yaml.safe_dump_all(resources, sort_keys=False))
    print("Hybrid verified. Reconnect port-forward to http://localhost:8088.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    up = commands.add_parser("start")
    up.add_argument("directory", type=Path)
    image = up.add_mutually_exclusive_group(required=True)
    image.add_argument("--image")
    image.add_argument("--image-file", type=Path)
    upgrade = commands.add_parser("upgrade-hybrid")
    upgrade.add_argument("directory", type=Path)
    upgrade.add_argument("--trace", type=Path, required=True)
    upgrade_image = upgrade.add_mutually_exclusive_group(required=True)
    upgrade_image.add_argument("--image")
    upgrade_image.add_argument("--image-file", type=Path)
    down = commands.add_parser("stop")
    down.add_argument("directory", type=Path)
    args = parser.parse_args()
    if run("kubectl", "config", "current-context").strip() != CONTEXT:
        parser.error(f"select the expected kube context: {CONTEXT}")
    try:
        if args.command in {"start", "upgrade-hybrid"}:
            image = args.image_file.read_text().strip() if args.image_file else args.image
            if args.command == "start":
                start(args.directory, image)
            else:
                upgrade_hybrid(args.directory, image, args.trace)
        else:
            stop(args.directory)
    except subprocess.CalledProcessError as error:
        parser.exit(1, error.stderr or str(error))


if __name__ == "__main__":
    main()
