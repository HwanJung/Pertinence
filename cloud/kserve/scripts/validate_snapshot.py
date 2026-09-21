#!/usr/bin/env python3
"""Reject benchmark evidence with KFP activity, restarts, or topology churn."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise ValueError(f"invalid Kubernetes list: {path}")
    return value


def node_topology(value: dict[str, Any]) -> dict[str, tuple[str, str, str]]:
    result = {}
    for item in value["items"]:
        metadata = item["metadata"]
        labels = metadata.get("labels", {})
        result[metadata["name"]] = (
            item.get("spec", {}).get("providerID", ""),
            labels.get("cloud.google.com/gke-nodepool", ""),
            labels.get("node.kubernetes.io/instance-type", ""),
        )
    return result


def pod_identity(value: dict[str, Any]) -> dict[str, str]:
    return {item["metadata"]["name"]: item["metadata"]["uid"] for item in value["items"]}


def restarted(value: dict[str, Any]) -> list[str]:
    failures = []
    for item in value["items"]:
        statuses = item.get("status", {}).get("containerStatuses", [])
        if any(int(status.get("restartCount", 0)) > 0 for status in statuses):
            failures.append(item["metadata"]["name"])
    return failures


def restart_total(value: dict[str, Any]) -> int:
    return sum(
        int(status.get("restartCount", 0))
        for item in value["items"]
        for status in item.get("status", {}).get("containerStatuses", [])
    )


def active_workflows(value: dict[str, Any]) -> list[str]:
    active = []
    for item in value["items"]:
        phase = item.get("status", {}).get("phase", "")
        if phase not in {"Succeeded", "Failed", "Error"}:
            active.append(f"{item['metadata']['name']}:{phase or 'Unknown'}")
    return active


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes_start", type=Path)
    parser.add_argument("nodes_end", type=Path)
    parser.add_argument("pods_start", type=Path)
    parser.add_argument("pods_end", type=Path)
    parser.add_argument("kfp_start", type=Path)
    parser.add_argument("kfp_end", type=Path)
    args = parser.parse_args()
    try:
        nodes_start, nodes_end = load(args.nodes_start), load(args.nodes_end)
        pods_start, pods_end = load(args.pods_start), load(args.pods_end)
        kfp_start, kfp_end = load(args.kfp_start), load(args.kfp_end)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    violations = []
    if node_topology(nodes_start) != node_topology(nodes_end):
        violations.append("node topology changed")
    if pod_identity(pods_start) != pod_identity(pods_end):
        violations.append("serving Pod identity changed")
    restart_names = sorted(set(restarted(pods_start) + restarted(pods_end)))
    if restart_names:
        violations.append(f"Pod restart detected: {', '.join(restart_names)}")
    workflows = sorted(set(active_workflows(kfp_start) + active_workflows(kfp_end)))
    if workflows:
        violations.append(f"active KFP workflow detected: {', '.join(workflows)}")
    receipt = {
        "schema_version": 1,
        "active_workflows": len(workflows),
        "pod_restart_delta": restart_total(pods_end),
        "node_topology_stable": node_topology(nodes_start) == node_topology(nodes_end),
        "pod_identity_stable": pod_identity(pods_start) == pod_identity(pods_end),
        "snapshot_invalid_reasons": violations,
    }
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
