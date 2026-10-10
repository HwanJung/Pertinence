#!/usr/bin/env python3
"""Validate one rendered v2 arm and optionally inventory actual Pod resources."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from render_config import DIGEST_RE, read_env

_RENDERER_PATH = Path(__file__).with_name("render-single-arm-v2.py")
_RENDERER_SPEC = importlib.util.spec_from_file_location("render_single_arm_v2", _RENDERER_PATH)
if _RENDERER_SPEC is None or _RENDERER_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load {_RENDERER_PATH}")
_RENDERER = importlib.util.module_from_spec(_RENDERER_SPEC)
_RENDERER_SPEC.loader.exec_module(_RENDERER)
ARMS = _RENDERER.ARMS
validate_values = _RENDERER.validate_values
benchmark_config_map_names = _RENDERER.benchmark_config_map_names


SERVING_NODE_SELECTOR = {"cloud.google.com/gke-nodepool": "serving-pool"}
LOADGEN_NODE_SELECTOR = {"cloud.google.com/gke-nodepool": "loadgen-pool"}
SERVING_TOLERATION = [{
    "key": "workload", "operator": "Equal", "value": "pertinence-serving",
    "effect": "NoSchedule",
}]
CONTAINER_SECURITY = {
    "allowPrivilegeEscalation": False,
    "capabilities": {"drop": ["ALL"]},
    "readOnlyRootFilesystem": True,
    "runAsNonRoot": True,
    "runAsUser": 65532,
}
PREDICTOR_TOTALS = {
    "resnet56-direct": (Decimal("1"), 1024**3),
    "dispatcher-000733-fused": (Decimal("4"), 4 * 1024**3),
    "dispatcher-000733-distributed": (Decimal("4"), 4 * 1024**3),
}
CPU_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)([num]?)")
MEMORY_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)([EPTGMK]i?|[eE][+-]?[0-9]+)?")
MEMORY_SCALE = {
    "": Decimal(1), "K": Decimal(1000), "M": Decimal(1000**2),
    "G": Decimal(1000**3), "T": Decimal(1000**4), "P": Decimal(1000**5),
    "E": Decimal(1000**6), "Ki": Decimal(1024), "Mi": Decimal(1024**2),
    "Gi": Decimal(1024**3), "Ti": Decimal(1024**4), "Pi": Decimal(1024**5),
    "Ei": Decimal(1024**6),
}


def parse_cpu(value: object) -> Decimal:
    match = CPU_RE.fullmatch(str(value))
    if not match:
        raise ValueError(f"invalid CPU quantity: {value!r}")
    scale = {"": Decimal(1), "m": Decimal("0.001"), "u": Decimal("0.000001"),
             "n": Decimal("0.000000001")}[match.group(2)]
    return Decimal(match.group(1)) * scale


def parse_memory(value: object) -> int:
    text = str(value)
    match = MEMORY_RE.fullmatch(text)
    if not match:
        raise ValueError(f"invalid memory quantity: {value!r}")
    suffix = match.group(2) or ""
    if suffix.lower().startswith("e"):
        return int(Decimal(match.group(1)) * (Decimal(10) ** int(suffix[1:])))
    return int(Decimal(match.group(1)) * MEMORY_SCALE[suffix])


def _sum_resources(containers: list[object], resource_class: str) -> tuple[Decimal, int]:
    cpu = Decimal(0)
    memory = 0
    for container in containers:
        if not isinstance(container, dict):
            continue
        resources = container.get("resources", {})
        values = resources.get(resource_class, {}) if isinstance(resources, dict) else {}
        if not isinstance(values, dict):
            continue
        cpu += parse_cpu(values.get("cpu", "0"))
        memory += parse_memory(values.get("memory", "0"))
    return cpu, memory


def aggregate_pod_resources(pod: dict[str, Any]) -> dict[str, Any]:
    """Return app sums and Kubernetes-effective totals for an actual Pod."""
    spec = pod.get("spec", {})
    containers = spec.get("containers", []) if isinstance(spec, dict) else []
    init_containers = spec.get("initContainers", []) if isinstance(spec, dict) else []
    overhead = spec.get("overhead", {}) if isinstance(spec, dict) else {}
    result: dict[str, Any] = {
        "pod_uid": pod.get("metadata", {}).get("uid"),
        "pod_name": pod.get("metadata", {}).get("name"),
        "node_name": spec.get("nodeName") if isinstance(spec, dict) else None,
        "containers": [],
    }
    for container in containers:
        if not isinstance(container, dict):
            continue
        result["containers"].append({
            "name": container.get("name"),
            "image": container.get("image"),
            "requests": container.get("resources", {}).get("requests", {}),
            "limits": container.get("resources", {}).get("limits", {}),
        })
    for resource_class in ("requests", "limits"):
        app_cpu, app_memory = _sum_resources(containers, resource_class)
        init_totals = [_sum_resources([container], resource_class) for container in init_containers]
        init_cpu = max((item[0] for item in init_totals), default=Decimal(0))
        init_memory = max((item[1] for item in init_totals), default=0)
        overhead_cpu = parse_cpu(overhead.get("cpu", "0")) if isinstance(overhead, dict) else Decimal(0)
        overhead_memory = parse_memory(overhead.get("memory", "0")) if isinstance(overhead, dict) else 0
        result[resource_class] = {
            "app_cpu_cores": str(app_cpu),
            "app_memory_bytes": app_memory,
            "effective_cpu_cores": str(max(app_cpu, init_cpu) + overhead_cpu),
            "effective_memory_bytes": max(app_memory, init_memory) + overhead_memory,
        }
    return result


def _validate_image(image: object, source: str, errors: list[str]) -> None:
    match = DIGEST_RE.fullmatch(image) if isinstance(image, str) else None
    if not match or match.group(1) == "0" * 64:
        errors.append(f"{source}: image must be digest pinned and non-sentinel")


def _parse_pairs(args: object, source: str, errors: list[str]) -> dict[str, str]:
    if not isinstance(args, list) or not all(isinstance(value, str) for value in args):
        errors.append(f"{source}: args must be a string list")
        return {}
    if args[:2] != ["benchmark", "run-single"]:
        errors.append(f"{source}: must execute benchmark run-single")
        return {}
    pairs: dict[str, str] = {}
    for index in range(2, len(args), 2):
        if index + 1 >= len(args) or not args[index].startswith("--"):
            errors.append(f"{source}: malformed CLI arguments")
            break
        if args[index] in pairs:
            errors.append(f"{source}: duplicate argument {args[index]}")
        pairs[args[index]] = args[index + 1]
    return pairs


def validate_documents(documents: list[dict[str, Any]], values: dict[str, str]) -> list[str]:
    errors: list[str] = []
    arm = values["ARM_ID"]
    expected_config_maps = benchmark_config_map_names(values["BENCHMARK_ID"])
    expected_config_map_names = set(expected_config_maps.values())
    forbidden = {"Ingress", "Gateway", "HTTPRoute", "NetworkPolicy"}
    kinds = {item.get("kind") for item in documents}
    if kinds & forbidden:
        errors.append("forbidden externally exposing object is present")
    for document in documents:
        if document.get("kind") == "Service" and document.get("spec", {}).get("type", "ClusterIP") != "ClusterIP":
            errors.append("only internal ClusterIP Services are permitted")

    config_maps = [item for item in documents if item.get("kind") == "ConfigMap"]
    rendered_config_map_names = {
        item.get("metadata", {}).get("name") for item in config_maps
    }
    if rendered_config_map_names != expected_config_map_names:
        errors.append(
            "render contains unexpected ConfigMaps: "
            f"{sorted(str(name) for name in rendered_config_map_names)}"
        )
    for config_map in config_maps:
        if config_map.get("immutable") is not True:
            errors.append(f"{config_map.get('metadata', {}).get('name')}: ConfigMap must be immutable")

    def validate_config_map_refs(volumes: object, source: str) -> None:
        if not isinstance(volumes, list):
            return
        expected_by_volume = {
            "promotion": expected_config_maps["promotion-dispatcher-000733"],
            "benchmark-config": expected_config_maps["benchmark-config"],
        }
        for volume in volumes:
            if not isinstance(volume, dict) or "configMap" not in volume:
                continue
            config_map = volume.get("configMap")
            actual = config_map.get("name") if isinstance(config_map, dict) else None
            expected = expected_by_volume.get(volume.get("name"))
            if expected is None or actual != expected:
                errors.append(f"{source}: ConfigMap reference {actual!r} is not benchmark-scoped")

    services = [item for item in documents if item.get("kind") == "InferenceService"]
    names = {item.get("metadata", {}).get("name") for item in services}
    if names != ARMS[arm]["services"]:
        errors.append(f"arm contains unexpected InferenceServices: {sorted(names)}")
    total_cpu = Decimal(0)
    total_memory = 0
    for service in services:
        name = service.get("metadata", {}).get("name", "unknown")
        metadata = service.get("metadata", {})
        annotations = metadata.get("annotations", {})
        if annotations.get("pertinence.dev/arm-id") != arm:
            errors.append(f"{name}: arm annotation mismatch")
        if annotations.get("pertinence.dev/serving-mode") != ARMS[arm]["mode"]:
            errors.append(f"{name}: serving mode annotation mismatch")
        predictor = service.get("spec", {}).get("predictor", {})
        validate_config_map_refs(predictor.get("volumes", []), name)
        if predictor.get("minReplicas") != 1 or predictor.get("maxReplicas") != 1:
            errors.append(f"{name}: replicas must be fixed at one")
        if predictor.get("nodeSelector") != SERVING_NODE_SELECTOR:
            errors.append(f"{name}: must select serving-pool")
        if predictor.get("tolerations") != SERVING_TOLERATION:
            errors.append(f"{name}: serving toleration mismatch")
        if predictor.get("serviceAccountName") != "serving-runner":
            errors.append(f"{name}: service account mismatch")
        for index, storage in enumerate(predictor.get("storageUris", [])):
            uri = storage.get("uri", "") if isinstance(storage, dict) else ""
            if not uri.startswith("oci://"):
                errors.append(f"{name}: storageUris[{index}] must be OCI")
            else:
                _validate_image(uri.removeprefix("oci://"), f"{name}: modelcar", errors)
        containers = predictor.get("containers", [])
        for index, container in enumerate(containers):
            _validate_image(container.get("image"), f"{name}: container[{index}]", errors)
            if container.get("securityContext") != CONTAINER_SECURITY:
                errors.append(f"{name}: container[{index}] must run as locked-down non-root")
        cpu, memory = _sum_resources(containers, "requests")
        limit_cpu, limit_memory = _sum_resources(containers, "limits")
        if (cpu, memory) != (limit_cpu, limit_memory):
            errors.append(f"{name}: predictor requests and limits must be equal")
        total_cpu += cpu
        total_memory += memory
    if (total_cpu, total_memory) != PREDICTOR_TOTALS[arm]:
        errors.append(f"predictor aggregate differs from arm contract: {total_cpu} CPU/{total_memory} bytes")

    jobs = [item for item in documents if item.get("kind") == "Job"]
    if len(jobs) != 1:
        errors.append("render must contain exactly one unique single-arm Job")
        return errors
    job = jobs[0]
    if job.get("metadata", {}).get("name") != values["BENCHMARK_JOB_NAME"]:
        errors.append("Job name differs from sealed input")
    spec = job.get("spec", {})
    for key, wanted in (("suspend", True), ("parallelism", 1), ("completions", 1), ("backoffLimit", 0)):
        if spec.get(key) != wanted:
            errors.append(f"Job {key} must be {wanted!r}")
    pod = spec.get("template", {}).get("spec", {})
    validate_config_map_refs(pod.get("volumes", []), "loadgen")
    if pod.get("nodeSelector") != LOADGEN_NODE_SELECTOR:
        errors.append("loadgen must select loadgen-pool")
    if pod.get("restartPolicy") != "Never":
        errors.append("loadgen restartPolicy must be Never")
    containers = pod.get("containers", [])
    if len(containers) != 1:
        errors.append("Job must have exactly one loadgen container")
    else:
        loadgen = containers[0]
        _validate_image(loadgen.get("image"), "loadgen", errors)
        if loadgen.get("securityContext") != CONTAINER_SECURITY:
            errors.append("loadgen must run as locked-down non-root")
        pairs = _parse_pairs(loadgen.get("args"), "loadgen", errors)
        if pairs.get("--arm") != arm or pairs.get("--endpoint") != ARMS[arm]["endpoint"]:
            errors.append("loadgen arm/endpoint pair differs from approved mapping")
        if pairs.get("--output-uri") != values["BENCHMARK_OUTPUT_URI"]:
            errors.append("loadgen output URI differs from sealed input")
        env = {
            item.get("name"): item.get("value")
            for item in loadgen.get("env", [])
            if isinstance(item, dict)
        }
        if env.get("PERTINENCE_NODE_UID") != values["SERVING_NODE_UID"]:
            errors.append("loadgen serving node UID differs from sealed input")
        if env.get("PERTINENCE_MACHINE_TYPE") != values["SERVING_MACHINE_TYPE"]:
            errors.append("loadgen serving machine type differs from sealed input")
        required = {"--arm", "--endpoint", "--trace", "--trace-uri", "--promotion", "--config",
                    "--output", "--output-uri", "--collector-done", "--collector-completed"}
        if set(pairs) != required:
            errors.append("loadgen arguments differ from the single-arm contract")
    return errors


def pod_resource_evidence(
    snapshot: dict[str, Any], arm: str, node_snapshot: dict[str, Any] | None = None
) -> dict[str, Any]:
    items = snapshot.get("items") if isinstance(snapshot, dict) else None
    if not isinstance(items, list):
        raise ValueError("Pod snapshot must be a Kubernetes List")
    nodes: dict[str, dict[str, Any]] = {}
    if node_snapshot is not None:
        node_items = node_snapshot.get("items") if isinstance(node_snapshot, dict) else None
        if not isinstance(node_items, list):
            raise ValueError("Node snapshot must be a Kubernetes List")
        nodes = {
            item.get("metadata", {}).get("name"): item
            for item in node_items
            if isinstance(item, dict) and isinstance(item.get("metadata", {}).get("name"), str)
        }
    serving = []
    for item in items:
        if not isinstance(item, dict):
            continue
        labels = item.get("metadata", {}).get("labels", {})
        if labels.get("serving.kserve.io/inferenceservice") in ARMS[arm]["services"]:
            entry = aggregate_pod_resources(item)
            if len(entry["containers"]) < 2:
                raise ValueError(f"{entry['pod_name']}: actual serving Pod omits the modelcar sidecar")
            if node_snapshot is not None:
                node = nodes.get(entry["node_name"])
                if node is None:
                    raise ValueError(f"{entry['pod_name']}: assigned node absent from Node snapshot")
                entry["node_uid"] = node.get("metadata", {}).get("uid")
                entry["machine_type"] = node.get("metadata", {}).get("labels", {}).get(
                    "node.kubernetes.io/instance-type"
                )
                if not entry["node_uid"] or not entry["machine_type"]:
                    raise ValueError(f"{entry['pod_name']}: Node UID or machine type is missing")
            serving.append(entry)
    if len(serving) != len(ARMS[arm]["services"]):
        raise ValueError("Pod snapshot does not contain exactly one Pod per arm InferenceService")
    return {"schema_version": 1, "arm_id": arm, "pods": serving}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-env", type=Path, required=True)
    parser.add_argument("--pod-snapshot", type=Path)
    parser.add_argument("--node-snapshot", type=Path)
    parser.add_argument("--resource-evidence", type=Path)
    parser.add_argument("yaml_file", nargs="+", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    try:
        values = read_env(args.arm_env.resolve(strict=True))
        validate_values(values)
        documents: list[dict[str, Any]] = []
        for path in args.yaml_file:
            documents.extend(
                item for item in yaml.safe_load_all(path.read_text(encoding="utf-8"))
                if isinstance(item, dict)
            )
        errors.extend(validate_documents(documents, values))
        if args.pod_snapshot or args.resource_evidence:
            if not args.pod_snapshot or not args.resource_evidence:
                raise ValueError("--pod-snapshot and --resource-evidence must be used together")
            evidence = pod_resource_evidence(
                json.loads(args.pod_snapshot.read_text(encoding="utf-8")),
                values["ARM_ID"],
                json.loads(args.node_snapshot.read_text(encoding="utf-8"))
                if args.node_snapshot
                else None,
            )
            if args.resource_evidence.exists():
                raise ValueError(f"resource evidence already exists: {args.resource_evidence}")
            args.resource_evidence.write_text(
                json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as error:
        errors.append(str(error))
    if errors:
        parser.error("\n".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
