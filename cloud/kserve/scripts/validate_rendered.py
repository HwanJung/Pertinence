#!/usr/bin/env python3
"""Reject mutable images and externally exposed Kubernetes objects."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml


IMAGE_RE = re.compile(r"^[^\s@]+@sha256:([0-9a-f]{64})$")
BENCHMARK_ID_RE = re.compile(r"[a-z0-9](?:[a-z0-9.-]{0,61}[a-z0-9])?")
TRACE_URI_RE = re.compile(
    r"gs://pertinence-509116-pertinence-runs/pertinence/benchmarks/traces/"
    r"[a-z0-9.-]+"
)
OUTPUT_URI_RE = re.compile(
    r"gs://pertinence-509116-pertinence-runs/pertinence/benchmarks/runs/"
    r"([a-z0-9](?:[a-z0-9.-]{0,61}[a-z0-9])?)"
)
ALLOWED_BENCHMARK_JOBS = {
    "pertinence-loadgen",
}
LOADGEN_TOLERATION = {
    "key": "workload",
    "operator": "Equal",
    "value": "pertinence-loadgen",
    "effect": "NoSchedule",
}
LOADGEN_RESOURCES = {
    "requests": {"cpu": "1", "memory": "1Gi"},
    "limits": {"cpu": "1", "memory": "1Gi"},
}
COLLECTOR_RESOURCES = {
    "requests": {"cpu": "100m", "memory": "128Mi"},
    "limits": {"cpu": "100m", "memory": "128Mi"},
}
CONTAINER_SECURITY_CONTEXT = {
    "allowPrivilegeEscalation": False,
    "capabilities": {"drop": ["ALL"]},
    "readOnlyRootFilesystem": True,
    "runAsNonRoot": True,
    "runAsUser": 65532,
}
EXPECTED_ENDPOINTS = [
    "resnet56=http://expert-resnet56-predictor.pertinence-serving.svc.cluster.local",
    "dispatcher-000733=http://router-dispatcher-000733-predictor.pertinence-serving.svc.cluster.local",
]


def validate_image(image: object, source: str) -> list[str]:
    if not isinstance(image, str):
        return [f"{source}: image is not a string"]
    match = IMAGE_RE.fullmatch(image)
    if not match:
        return [f"{source}: image is not digest pinned: {image}"]
    if match.group(1) == "0" * 64:
        return [f"{source}: image uses the all-zero sentinel digest"]
    return []


def inspect_workload_images(value: Any, source: str) -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_source = f"{source}.{key}"
            if key in {"containers", "initContainers", "ephemeralContainers"} and isinstance(
                child, list
            ):
                for index, container in enumerate(child):
                    if isinstance(container, dict) and "image" in container:
                        errors.extend(
                            validate_image(container["image"], f"{child_source}[{index}].image")
                        )
            else:
                errors.extend(inspect_workload_images(child, child_source))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(inspect_workload_images(child, f"{source}[{index}]"))
    return errors


def inspect_config_images(value: Any, source: str) -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_source = f"{source}.{key}"
            if key == "image":
                version = value.get("defaultImageVersion")
                candidate = f"{child}:{version}" if isinstance(version, str) else child
                errors.extend(validate_image(candidate, child_source))
            elif key == "defaultJobImage":
                version = value.get("defaultJobTag")
                candidate = f"{child}:{version}" if isinstance(version, str) else child
                errors.extend(validate_image(candidate, child_source))
            else:
                errors.extend(inspect_config_images(child, child_source))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(inspect_config_images(child, f"{source}[{index}]"))
    return errors


def _parse_cli_args(args: object, source: str) -> tuple[dict[str, str], list[str], list[str]]:
    """Parse the fixed benchmark CLI form while rejecting ambiguity."""
    errors: list[str] = []
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        return {}, [], [f"{source}: args must be a list of strings"]
    if args[:2] != ["benchmark", "run"]:
        errors.append(f"{source}: loadgen must execute 'benchmark run'")
        return {}, [], errors

    values: dict[str, str] = {}
    endpoints: list[str] = []
    index = 2
    while index < len(args):
        flag = args[index]
        if not flag.startswith("--") or index + 1 >= len(args):
            errors.append(f"{source}: malformed benchmark argument at index {index}: {flag!r}")
            break
        value = args[index + 1]
        if value.startswith("--"):
            errors.append(f"{source}: benchmark argument {flag} has no value")
            break
        if flag == "--endpoint":
            endpoints.append(value)
        elif flag in values:
            errors.append(f"{source}: duplicate benchmark argument {flag}")
        else:
            values[flag] = value
        index += 2
    return values, endpoints, errors


def validate_benchmark_job(document: dict[str, Any], source: str) -> list[str]:
    """Validate the create-only benchmark Job against the execution contract."""
    errors: list[str] = []
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        return [f"{source}: benchmark Job metadata must be a mapping"]
    name = metadata.get("name")
    if name not in ALLOWED_BENCHMARK_JOBS:
        errors.append(
            f"{source}: benchmark Job name must be one of "
            + ", ".join(sorted(ALLOWED_BENCHMARK_JOBS))
        )
    if metadata.get("namespace") != "pertinence-serving":
        errors.append(f"{source}: benchmark Job namespace must be pertinence-serving")

    spec = document.get("spec")
    if not isinstance(spec, dict):
        return errors + [f"{source}: benchmark Job spec must be a mapping"]
    for key, expected in (
        ("suspend", True),
        ("parallelism", 1),
        ("completions", 1),
        ("backoffLimit", 0),
    ):
        actual = spec.get(key)
        if type(actual) is not type(expected) or actual != expected:
            errors.append(f"{source}: Job {key} must be {expected!r}")

    template = spec.get("template")
    pod = template.get("spec") if isinstance(template, dict) else None
    if not isinstance(pod, dict):
        return errors + [f"{source}: benchmark Job pod spec must be a mapping"]
    if pod.get("restartPolicy") != "Never":
        errors.append(f"{source}: benchmark Job restartPolicy must be Never")
    if pod.get("serviceAccountName") != "benchmark-runner":
        errors.append(f"{source}: benchmark Job must use benchmark-runner")
    if pod.get("nodeSelector") != {"cloud.google.com/gke-nodepool": "loadgen-pool"}:
        errors.append(f"{source}: benchmark Job must select loadgen-pool")
    if pod.get("tolerations") != [LOADGEN_TOLERATION]:
        errors.append(f"{source}: benchmark Job toleration differs from loadgen contract")
    if pod.get("securityContext") != {
        "runAsNonRoot": True,
        "runAsUser": 65532,
        "runAsGroup": 65532,
        "fsGroup": 65532,
        "seccompProfile": {"type": "RuntimeDefault"},
    }:
        errors.append(f"{source}: benchmark Job pod security context differs from contract")

    containers = pod.get("containers")
    if not isinstance(containers, list) or len(containers) != 1:
        errors.append(f"{source}: benchmark Job must have exactly one loadgen container")
        loadgen: object = None
    else:
        loadgen = containers[0]
        if not isinstance(loadgen, dict) or loadgen.get("name") != "loadgen":
            errors.append(f"{source}: benchmark Job's only container must be named loadgen")

    init_containers = pod.get("initContainers")
    if not isinstance(init_containers, list) or len(init_containers) != 1:
        errors.append(
            f"{source}: benchmark Job must have exactly one native sidecar evidence collector"
        )
        collector: object = None
    else:
        collector = init_containers[0]
        if not isinstance(collector, dict) or collector.get("name") != "evidence-collector":
            errors.append(f"{source}: native sidecar must be named evidence-collector")
        if not isinstance(collector, dict) or collector.get("restartPolicy") != "Always":
            errors.append(f"{source}: evidence collector restartPolicy must be Always")

    for container, role, resources in (
        (loadgen, "loadgen", LOADGEN_RESOURCES),
        (collector, "evidence collector", COLLECTOR_RESOURCES),
    ):
        if not isinstance(container, dict):
            continue
        errors.extend(validate_image(container.get("image"), f"{source}: {role} image"))
        if container.get("securityContext") != CONTAINER_SECURITY_CONTEXT:
            errors.append(f"{source}: {role} security context differs from contract")
        if container.get("resources") != resources:
            errors.append(f"{source}: {role} resources differ from benchmark contract")
    if isinstance(loadgen, dict) and isinstance(collector, dict):
        if loadgen.get("image") != collector.get("image"):
            errors.append(f"{source}: loadgen and evidence collector must use the same image")

    volumes = pod.get("volumes")
    if not isinstance(volumes, list):
        errors.append(f"{source}: benchmark Job volumes must be a list")
    else:
        if any(isinstance(volume, dict) and "persistentVolumeClaim" in volume for volume in volumes):
            errors.append(f"{source}: benchmark Job must use direct GCS and emptyDir, not PVC")
        work_volumes = [
            volume for volume in volumes if isinstance(volume, dict) and volume.get("name") == "work"
        ]
        if work_volumes != [{"name": "work", "emptyDir": {}}]:
            errors.append(f"{source}: benchmark Job must declare exactly one work emptyDir")

    if isinstance(loadgen, dict):
        expected_mounts = [
            {"name": "promotion", "mountPath": "/etc/pertinence", "readOnly": True},
            {"name": "benchmark-config", "mountPath": "/etc/benchmark", "readOnly": True},
            {"name": "work", "mountPath": "/work"},
        ]
        if loadgen.get("volumeMounts") != expected_mounts:
            errors.append(f"{source}: loadgen volume mounts differ from benchmark contract")
        values, endpoints, arg_errors = _parse_cli_args(
            loadgen.get("args"), f"{source}: loadgen"
        )
        errors.extend(arg_errors)
    else:
        values, endpoints = {}, []

    required_values = {
        "--trace",
        "--trace-uri",
        "--promotion",
        "--config",
        "--output",
        "--output-uri",
        "--collector-done",
        "--collector-completed",
    }
    if set(values) != required_values:
        errors.append(
            f"{source}: benchmark arguments must be exactly "
            + ", ".join(sorted(required_values))
        )
    if values.get("--trace") != "/work/trace":
        errors.append(f"{source}: benchmark trace path must be /work/trace")
    trace_uri = values.get("--trace-uri", "")
    if not TRACE_URI_RE.fullmatch(trace_uri):
        errors.append(f"{source}: benchmark trace URI uses an unexpected prefix or identifier")
    if values.get("--promotion") != "/etc/pertinence/promotion.json":
        errors.append(f"{source}: benchmark promotion path differs from contract")
    if values.get("--config") != "/etc/benchmark/benchmark.yaml":
        errors.append(f"{source}: benchmark config path differs from contract")
    if endpoints != EXPECTED_ENDPOINTS:
        errors.append(f"{source}: benchmark endpoints/order differ from contract")

    output = values.get("--output", "")
    output_match = re.fullmatch(r"/work/(.+)", output)
    benchmark_id = output_match.group(1) if output_match else ""
    if not BENCHMARK_ID_RE.fullmatch(benchmark_id):
        errors.append(f"{source}: benchmark output must be /work/<DNS-safe-benchmark-id>")
    output_uri = values.get("--output-uri", "")
    output_uri_match = OUTPUT_URI_RE.fullmatch(output_uri)
    if not output_uri_match or output_uri_match.group(1) != benchmark_id:
        errors.append(f"{source}: local and GCS output benchmark IDs must match")
    if values.get("--collector-done") != "/work/loadgen.done":
        errors.append(f"{source}: loadgen collector-done path differs from contract")
    if values.get("--collector-completed") != "/work/collector.done":
        errors.append(f"{source}: loadgen collector-completed path differs from contract")

    if isinstance(collector, dict):
        if collector.get("volumeMounts") != [{"name": "work", "mountPath": "/work"}]:
            errors.append(f"{source}: evidence collector must mount the shared work emptyDir")
        expected_collector_args = [
            "benchmark",
            "collect-evidence",
            "--output",
            f"{output}/evidence.jsonl",
            "--done",
            values.get("--collector-done", ""),
            "--completed",
            values.get("--collector-completed", ""),
        ]
        if collector.get("args") != expected_collector_args:
            errors.append(
                f"{source}: evidence collector paths must match the loadgen output and markers"
            )
    return errors


def validate_document(document: object, source: str) -> list[str]:
    if not isinstance(document, dict):
        return []
    errors: list[str] = []
    kind = document.get("kind")
    if kind in {"Ingress", "Gateway", "HTTPRoute", "NetworkPolicy"}:
        errors.append(f"{source}: forbidden kind {kind}")
    if kind == "Service" and document.get("spec", {}).get("type", "ClusterIP") != "ClusterIP":
        errors.append(f"{source}: only ClusterIP Services are permitted")
    if kind == "ServiceAccount":
        name = document.get("metadata", {}).get("name")
        annotations = document.get("metadata", {}).get("annotations", {})
        if name == "serving-runner" and any(str(key).startswith("iam.gke.io/") for key in annotations):
            errors.append(f"{source}: serving-runner must not have a GCP IAM annotation")
        if name == "benchmark-runner" and "iam.gke.io/gcp-service-account" not in annotations:
            errors.append(f"{source}: benchmark-runner must declare its Workload Identity GSA")
    if kind == "InferenceService":
        metadata = document.get("metadata", {})
        predictor = document.get("spec", {}).get("predictor", {})
        name = metadata.get("name", "")
        if predictor.get("minReplicas") != 1 or predictor.get("maxReplicas") != 1:
            errors.append(f"{source}: InferenceService replicas must be fixed at one")
        if predictor.get("nodeSelector") != {"cloud.google.com/gke-nodepool": "serving-pool"}:
            errors.append(f"{source}: InferenceService must select serving-pool")
        if predictor.get("serviceAccountName") != "serving-runner":
            errors.append(f"{source}: InferenceService must use serving-runner")
        required_toleration = {
            "key": "workload",
            "operator": "Equal",
            "value": "pertinence-serving",
            "effect": "NoSchedule",
        }
        if predictor.get("tolerations") != [required_toleration]:
            errors.append(f"{source}: InferenceService toleration differs from serving contract")
        storage = predictor.get("storageUris", [])
        for index, item in enumerate(storage):
            uri = item.get("uri") if isinstance(item, dict) else None
            if not isinstance(uri, str) or not uri.startswith("oci://"):
                errors.append(f"{source}: storageUris[{index}] must be an OCI URI")
            else:
                errors.extend(validate_image(uri.removeprefix("oci://"), f"{source}.storageUris[{index}]"))
        for index, container in enumerate(predictor.get("containers", [])):
            security = container.get("securityContext", {})
            if security.get("runAsNonRoot") is not True or security.get("runAsUser") != 65532:
                errors.append(f"{source}: container[{index}] must run as non-root UID 65532")
            resources = container.get("resources", {})
            if resources.get("requests") != {"cpu": "1", "memory": "1Gi"} or resources.get(
                "limits"
            ) != {"cpu": "1", "memory": "1Gi"}:
                errors.append(f"{source}: container[{index}] resources differ from benchmark contract")
    if kind == "ConfigMap" and document.get("metadata", {}).get("name") in {
        "promotion-dispatcher-000733",
        "benchmark-config",
    } and document.get("immutable") is not True:
        errors.append(f"{source}: serving input ConfigMaps must be immutable")
    if kind == "Job":
        metadata = document.get("metadata", {})
        name = metadata.get("name") if isinstance(metadata, dict) else None
        namespace = metadata.get("namespace") if isinstance(metadata, dict) else None
        if name in ALLOWED_BENCHMARK_JOBS or namespace == "pertinence-serving":
            errors.extend(validate_benchmark_job(document, source))
        else:
            spec = document.get("spec", {})
            template = spec.get("template", {}) if isinstance(spec, dict) else {}
            pod = template.get("spec", {}) if isinstance(template, dict) else {}
            volumes = pod.get("volumes", []) if isinstance(pod, dict) else []
            if any(
                isinstance(volume, dict) and "persistentVolumeClaim" in volume
                for volume in volumes
            ):
                errors.append(f"{source}: Job must not use a PVC")
    errors.extend(inspect_workload_images(document, source))

    if kind == "ConfigMap":
        for key, raw_value in document.get("data", {}).items():
            if not isinstance(raw_value, str):
                continue
            try:
                parsed = json.loads(raw_value)
            except json.JSONDecodeError:
                continue
            errors.extend(inspect_config_images(parsed, f"{source}.data.{key}"))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_file", nargs="+", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    namespaces: set[str] = set()
    inference_services: set[str] = set()
    benchmark_jobs: set[str] = set()
    benchmark_contract_seen = False
    for path in args.yaml_file:
        try:
            documents = yaml.safe_load_all(path.read_text(encoding="utf-8"))
            for index, document in enumerate(documents):
                if isinstance(document, dict) and document.get("kind") == "Namespace":
                    name = document.get("metadata", {}).get("name")
                    if isinstance(name, str):
                        namespaces.add(name)
                if isinstance(document, dict) and document.get("kind") in {
                    "InferenceService",
                    "Job",
                }:
                    name = document.get("metadata", {}).get("name")
                    if isinstance(name, str):
                        if document["kind"] == "InferenceService":
                            inference_services.add(name)
                        elif document.get("metadata", {}).get("namespace") == "pertinence-serving":
                            benchmark_jobs.add(name)
                if (
                    isinstance(document, dict)
                    and document.get("kind") == "ConfigMap"
                    and document.get("metadata", {}).get("name") == "benchmark-config"
                ):
                    benchmark_contract_seen = True
                errors.extend(validate_document(document, f"{path}:document[{index}]"))
        except (OSError, yaml.YAMLError) as error:
            errors.append(f"{path}: {error}")
    required_namespaces = {"cert-manager", "kserve", "pertinence-serving"}
    if len(args.yaml_file) >= 4 and not required_namespaces.issubset(namespaces):
        errors.append(
            "rendered install is missing explicit namespaces: "
            + ", ".join(sorted(required_namespaces - namespaces))
        )
    if benchmark_contract_seen:
        expected_services = {
            "expert-shufflenet",
            "expert-mobilenet",
            "expert-resnet56",
            "router-dispatcher-000733",
        }
        if inference_services != expected_services:
            errors.append("rendered benchmark must contain exactly the four approved InferenceServices")
        if benchmark_jobs != {"pertinence-loadgen"}:
            errors.append("rendered benchmark must contain exactly the final loadgen Job")
    if errors:
        parser.error("\n".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
