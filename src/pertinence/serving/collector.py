"""Filtered cAdvisor collection used by the benchmark sidecar."""

from __future__ import annotations

import json
from pathlib import Path
import re
import ssl
import time
from urllib.request import Request, urlopen


CADVISOR_METRICS = {
    "container_cpu_usage_seconds_total",
    "container_memory_working_set_bytes",
    "container_cpu_cfs_throttled_seconds_total",
    "container_cpu_cfs_throttled_periods_total",
    "container_cpu_cfs_periods_total",
}


def filter_cadvisor(text: str) -> list[str]:
    result = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        metric = line.split("{", 1)[0].split(" ", 1)[0]
        if metric in CADVISOR_METRICS and 'namespace="pertinence-serving"' in line:
            result.append(line)
    return result


def pod_inventory(payload: bytes) -> list[dict[str, object]]:
    """Extract stable serving Pod identity and restart counters from a Kubernetes response."""

    value = json.loads(payload)
    result = []
    for item in value.get("items", []):
        metadata = item.get("metadata", {})
        name = metadata.get("name")
        uid = metadata.get("uid")
        if not isinstance(name, str) or not isinstance(uid, str):
            raise RuntimeError("collector received malformed Pod identity")
        statuses = item.get("status", {}).get("containerStatuses", [])
        if not isinstance(statuses, list):
            raise RuntimeError("collector received malformed Pod status")
        restart_count = sum(int(status.get("restartCount", 0)) for status in statuses)
        result.append({"name": name, "uid": uid, "restart_count": restart_count})
    return sorted(result, key=lambda item: str(item["name"]))


def collect_until_done(
    output: Path,
    done: Path,
    completed: Path,
    *,
    interval_seconds: float = 5.0,
) -> None:
    if interval_seconds != 5.0:
        raise ValueError("evidence collection interval must be exactly five seconds")
    host = __import__("os").environ.get("KUBERNETES_SERVICE_HOST")
    port = __import__("os").environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    if not host:
        raise RuntimeError("Kubernetes service environment is absent")
    token = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip()
    context = ssl.create_default_context(
        cafile="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    )

    def get(path: str) -> bytes:
        request = Request(f"https://{host}:{port}{path}", headers={"Authorization": f"Bearer {token}"})
        with urlopen(request, timeout=4.0, context=context) as response:
            return response.read()

    nodes = json.loads(get("/api/v1/nodes?labelSelector=cloud.google.com%2Fgke-nodepool%20in%20(serving-pool%2Cloadgen-pool)"))
    names = [item["metadata"]["name"] for item in nodes.get("items", [])]
    if len(names) != 2 or any(not re.fullmatch(r"[a-z0-9.-]+", name) for name in names):
        raise RuntimeError("collector requires exactly one serving and one loadgen node")
    for _ in range(600):
        if Path(output).parent.is_dir():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("collector output directory was not created")
    with Path(output).open("x", encoding="utf-8") as stream:
        while not Path(done).exists():
            timestamp_ns = time.time_ns()
            pods = pod_inventory(get("/api/v1/namespaces/pertinence-serving/pods"))
            for node in names:
                payload = get(f"/api/v1/nodes/{node}/proxy/metrics/cadvisor").decode()
                stream.write(
                    json.dumps(
                        {
                            "timestamp_ns": timestamp_ns,
                            "node": node,
                            "metrics": filter_cadvisor(payload),
                            "pods": pods,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            stream.flush()
            time.sleep(interval_seconds)
    Path(completed).touch(exist_ok=False)
