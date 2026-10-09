#!/usr/bin/env python3
"""Collect fail-closed, create-only evidence before any benchmark cloud mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def run(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return completed.stdout


def write_once(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)


def quota_violations(values: dict[str, str]) -> list[str]:
    violations: list[str] = []
    quotas = json.loads(values["quota.json"]).get("quotas", [])
    cpu_quota = next((item for item in quotas if item.get("metric") == "CPUS"), None)
    if not cpu_quota or cpu_quota.get("limit", 0) - cpu_quota.get("usage", 0) < 12:
        violations.append("regional CPU quota has less than 12 vCPUs available")
    n2_quota = next((item for item in quotas if item.get("metric") == "N2_CPUS"), None)
    if not n2_quota or n2_quota.get("limit", 0) - n2_quota.get("usage", 0) < 12:
        violations.append("regional N2 CPU quota has less than 12 vCPUs available")
    global_quotas = json.loads(values["global-quota.json"]).get("quotas", [])
    all_regions_cpu_quota = next(
        (item for item in global_quotas if item.get("metric") == "CPUS_ALL_REGIONS"),
        None,
    )
    if (
        not all_regions_cpu_quota
        or all_regions_cpu_quota.get("limit", 0) - all_regions_cpu_quota.get("usage", 0) < 12
    ):
        violations.append("all-regions CPU quota has less than 12 vCPUs available")
    return violations


def write_receipt(output: Path, values: dict[str, str], violations: list[str]) -> None:
    receipt = {
        "schema_version": 1,
        "valid": not violations,
        "invalid_reasons": violations,
        "files": {
            name: hashlib.sha256(value.encode()).hexdigest() for name, value in sorted(values.items())
        },
    }
    write_once(output / "receipt.json", json.dumps(receipt, sort_keys=True, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--principal", required=True)
    parser.add_argument("--node-service-account", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quota-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    kubectl = ["kubectl", "--context", args.context]
    quota_commands = {
        "quota.json": [
            "gcloud", "compute", "regions", "describe", args.region,
            "--project", args.project, "--format=json(quotas)",
        ],
        "global-quota.json": [
            "gcloud", "compute", "project-info", "describe",
            "--project", args.project, "--format=json(quotas)",
        ],
    }
    commands = quota_commands if args.quota_only else {
        "context.txt": ["kubectl", "config", "current-context"],
        "principal.txt": [
            "gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"
        ],
        "project.txt": ["gcloud", "config", "get-value", "project"],
        "cluster.json": [
            "gcloud", "container", "clusters", "describe", args.cluster,
            "--project", args.project, "--zone", args.zone, "--format=json",
        ],
        "node-pools.json": [
            "gcloud", "container", "node-pools", "list", "--cluster", args.cluster,
            "--project", args.project, "--zone", args.zone, "--format=json",
        ],
        **quota_commands,
        "kubernetes-version.json": [*kubectl, "version", "-o", "json"],
        "nodes.json": [*kubectl, "get", "nodes", "-o", "json"],
        "kfp-namespace.json": [*kubectl, "get", "namespace", "kubeflow", "-o", "json"],
        "pvcs.json": [*kubectl, "get", "pvc", "--all-namespaces", "-o", "json"],
        "pvs.json": [*kubectl, "get", "pv", "-o", "json"],
        "workflows.json": [
            *kubectl, "get", "workflows.argoproj.io", "--all-namespaces", "-o", "json"
        ],
        "pipeline-nodes.json": [
            *kubectl, "get", "nodes", "-l",
            "cloud.google.com/gke-nodepool=pipeline-pool", "-o", "json",
        ],
        "benchmark-ksa.json": [
            *kubectl, "get", "serviceaccount", "benchmark-runner",
            "-n", "pertinence-serving", "-o", "json",
        ],
        "artifact-reader.json": [
            "gcloud", "artifacts", "repositories", "get-iam-policy", args.repository,
            "--location", args.region, "--project", args.project, "--format=json",
        ],
        "project-policy.json": [
            "gcloud", "projects", "get-iam-policy", args.project, "--format=json",
        ],
        "metrics-api.json": [*kubectl, "get", "--raw", "/apis/metrics.k8s.io/v1beta1"],
        "nodes-proxy.txt": [
            *kubectl, "auth", "can-i", "get", "nodes/proxy", "--as",
            "system:serviceaccount:pertinence-serving:benchmark-runner",
        ],
        "metrics-rbac.txt": [
            *kubectl, "auth", "can-i", "list", "pods.metrics.k8s.io", "--as",
            "system:serviceaccount:pertinence-serving:benchmark-runner",
        ],
        "services.json": [*kubectl, "get", "service", "--all-namespaces", "-o", "json"],
        "ingresses.json": [*kubectl, "get", "ingress", "--all-namespaces", "-o", "json"],
        "forwarding-rules.json": [
            "gcloud", "compute", "forwarding-rules", "list", "--project", args.project,
            "--filter=loadBalancingScheme~^EXTERNAL", "--format=json",
        ],
    }
    values: dict[str, str] = {}
    for filename, command in commands.items():
        values[filename] = run(command)
        write_once(args.output / filename, values[filename])
    if args.quota_only:
        violations = quota_violations(values)
        write_receipt(args.output, values, violations)
        if violations:
            parser.error("; ".join(violations))
        return 0
    gateway_resources = run(
        [*kubectl, "api-resources", "--api-group=gateway.networking.k8s.io", "-o", "name"]
    )
    write_once(args.output / "gateway-resources.txt", gateway_resources)
    values["gateway-resources.txt"] = gateway_resources
    if "gateways.gateway.networking.k8s.io" in gateway_resources.splitlines():
        values["gateways.json"] = run(
            [
                *kubectl,
                "get",
                "gateways.gateway.networking.k8s.io",
                "--all-namespaces",
                "-o",
                "json",
            ]
        )
    else:
        values["gateways.json"] = '{"items":[]}\n'
    write_once(args.output / "gateways.json", values["gateways.json"])
    violations = quota_violations(values)
    if values["context.txt"].strip() != args.context:
        violations.append("unexpected Kubernetes context")
    canonical_context = f"gke_{args.project}_{args.zone}_{args.cluster}"
    if args.context != canonical_context:
        violations.append("Kubernetes context does not identify the expected GKE cluster")
    if values["project.txt"].strip() != args.project:
        violations.append("active gcloud project differs")
    principals = [line for line in values["principal.txt"].splitlines() if line]
    if principals != [args.principal]:
        violations.append("active gcloud principal differs")
    cluster = json.loads(values["cluster.json"])
    if cluster.get("name") != args.cluster:
        violations.append("GKE cluster name differs")
    if cluster.get("location") != args.zone:
        violations.append("GKE cluster zone differs")
    if cluster.get("status") != "RUNNING":
        violations.append("GKE cluster is not RUNNING")
    version = json.loads(values["kubernetes-version.json"])
    server = version.get("serverVersion", {})
    try:
        server_major = int(server["major"])
        server_minor = int(str(server["minor"]).rstrip("+"))
    except (KeyError, TypeError, ValueError):
        violations.append("Kubernetes control-plane version is malformed")
        server_major = server_minor = -1
    nodes = json.loads(values["nodes.json"]).get("items", [])
    if not nodes:
        violations.append("cluster has no live nodes")
    for node in nodes:
        kubelet = node.get("status", {}).get("nodeInfo", {}).get("kubeletVersion", "")
        match = re.match(r"^v(\d+)\.(\d+)", kubelet)
        if (
            match is None
            or int(match.group(1)) != server_major
            or int(match.group(2)) not in {server_minor, server_minor - 1}
        ):
            violations.append("Kubernetes node/control-plane version skew is incompatible")
            break
    if json.loads(values["kfp-namespace.json"]).get("status", {}).get("phase") != "Active":
        violations.append("kubeflow namespace is not Active")
    pvcs = json.loads(values["pvcs.json"]).get("items", [])
    kubeflow_pvcs = [item for item in pvcs if item.get("metadata", {}).get("namespace") == "kubeflow"]
    if not kubeflow_pvcs or any(item.get("status", {}).get("phase") != "Bound" for item in kubeflow_pvcs):
        violations.append("KFP PVC inventory is absent or not fully Bound")
    pvs = {
        item.get("metadata", {}).get("name"): item
        for item in json.loads(values["pvs.json"]).get("items", [])
    }
    if any(
        not isinstance(claim.get("spec", {}).get("volumeName"), str)
        or claim["spec"]["volumeName"] not in pvs
        or pvs[claim["spec"]["volumeName"]].get("status", {}).get("phase") != "Bound"
        for claim in kubeflow_pvcs
    ):
        violations.append("KFP PVCs do not resolve to Bound PVs")
    workflows = json.loads(values["workflows.json"])
    active = [
        item for item in workflows.get("items", [])
        if item.get("status", {}).get("phase") not in {"Succeeded", "Failed", "Error"}
    ]
    if active:
        violations.append("active KFP/Argo workflows exist")
    if json.loads(values["pipeline-nodes.json"]).get("items"):
        violations.append("pipeline-pool has live nodes")
    benchmark_ksa = json.loads(values["benchmark-ksa.json"])
    if (
        benchmark_ksa.get("kind") != "ServiceAccount"
        or benchmark_ksa.get("metadata", {}).get("name") != "benchmark-runner"
        or benchmark_ksa.get("metadata", {}).get("namespace") != "pertinence-serving"
    ):
        violations.append("benchmark KSA identity differs")
    repository_policy = json.loads(values["artifact-reader.json"])
    project_policy = json.loads(values["project-policy.json"])
    member = f"serviceAccount:{args.node_service_account}"
    if not any(
        binding.get("role") == "roles/artifactregistry.reader"
        and member in binding.get("members", [])
        and "condition" not in binding
        for policy in (repository_policy, project_policy)
        for binding in policy.get("bindings", [])
    ):
        violations.append("node SA lacks Artifact Registry reader")
    if values["nodes-proxy.txt"].strip() != "yes":
        violations.append("benchmark KSA lacks nodes/proxy RBAC")
    if values["metrics-rbac.txt"].strip() != "yes":
        violations.append("benchmark KSA lacks metrics API RBAC")
    services = json.loads(values["services.json"]).get("items", [])
    if any(item.get("spec", {}).get("type") == "LoadBalancer" for item in services):
        violations.append("LoadBalancer Service exists")
    if json.loads(values["ingresses.json"]).get("items"):
        violations.append("Ingress exists")
    if json.loads(values["gateways.json"]).get("items"):
        violations.append("Gateway exists")
    if json.loads(values["forwarding-rules.json"]):
        violations.append("external forwarding rule exists")
    write_receipt(args.output, values, violations)
    if violations:
        parser.error("; ".join(violations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
