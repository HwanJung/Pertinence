#!/usr/bin/env python3
"""Create one immutable target manifest set for a fused-comparison arm."""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from render_config import DIGEST_RE, GSA_RE, VARIABLE_RE, read_env


ARMS = {
    "resnet56-direct": {
        "services": {"expert-resnet56"},
        "endpoint": "http://expert-resnet56-predictor.pertinence-serving.svc.cluster.local",
        "mode": "direct",
    },
    "dispatcher-000733-fused": {
        "services": {"router-dispatcher-000733-fused"},
        "endpoint": (
            "http://router-dispatcher-000733-fused-predictor."
            "pertinence-serving.svc.cluster.local"
        ),
        "mode": "fused",
    },
    "dispatcher-000733-distributed": {
        "services": {
            "expert-shufflenet",
            "expert-mobilenet",
            "expert-resnet56",
            "router-dispatcher-000733",
        },
        "endpoint": (
            "http://router-dispatcher-000733-predictor.pertinence-serving.svc.cluster.local"
        ),
        "mode": "distributed",
    },
}

DNS_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
SHA_RE = re.compile(r"[0-9a-f]{64}")
OUTPUT_RE = re.compile(
    r"gs://pertinence-509116-pertinence-runs/pertinence/benchmarks/fused-v2/runs/"
    r"[a-z0-9](?:[a-z0-9.-]{0,61}[a-z0-9])?"
)
TRACE_RE = re.compile(
    r"gs://pertinence-509116-pertinence-runs/pertinence/benchmarks/traces/"
    r"[a-z0-9.-]+"
)
REQUIRED = {
    "ARM_ID",
    "BENCHMARK_JOB_NAME",
    "RUNTIME_IMAGE",
    "SHUFFLENET_MODELCAR",
    "MOBILENET_MODELCAR",
    "RESNET56_MODELCAR",
    "DISPATCHER_MODELCAR",
    "FUSED_MODELCAR",
    "SHUFFLENET_MODEL_SHA256",
    "MOBILENET_MODEL_SHA256",
    "RESNET56_MODEL_SHA256",
    "PROMOTION_FINGERPRINT",
    "PROMOTION_JSON_BASE64",
    "BENCHMARK_CONFIG_BASE64",
    "BENCHMARK_GSA",
    "BENCHMARK_ID",
    "BENCHMARK_TRACE_URI",
    "BENCHMARK_OUTPUT_URI",
    "SERVING_NODE_UID",
    "SERVING_MACHINE_TYPE",
}

CONFIG_MAP_PREFIXES = {
    "promotion-dispatcher-000733": "promotion",
    "benchmark-config": "benchmark-config",
}


def benchmark_config_map_names(benchmark_id: str) -> dict[str, str]:
    """Return deterministic per-run ConfigMap names that fit a DNS label."""
    slug = benchmark_id.replace(".", "-")
    suffix = hashlib.sha256(benchmark_id.encode("utf-8")).hexdigest()[:10]
    names: dict[str, str] = {}
    for legacy_name, prefix in CONFIG_MAP_PREFIXES.items():
        slug_limit = 63 - len(prefix) - len(suffix) - 2
        scoped_slug = slug[:slug_limit].rstrip("-")
        name = f"{prefix}-{scoped_slug}-{suffix}"
        if not DNS_RE.fullmatch(name):  # pragma: no cover - guarded by BENCHMARK_ID validation
            raise ValueError(f"derived ConfigMap name is not a DNS label: {name}")
        names[legacy_name] = name
    return names


def _decode_json(value: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(base64.b64decode(value, validate=True))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must encode valid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must encode a mapping")
    return parsed


def _decode_yaml(value: str, label: str) -> dict[str, Any]:
    try:
        parsed = yaml.safe_load(base64.b64decode(value, validate=True))
    except (binascii.Error, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"{label} must encode valid YAML") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must encode a mapping")
    return parsed


def _validate_benchmark_config(config: dict[str, Any]) -> None:
    request = config.get("request", {})
    warmup = config.get("warmup", {})
    steady = config.get("steady_state", {})
    expected = {
        "request.concurrency": (request.get("concurrency"), 1),
        "request.retry_count": (request.get("retry_count"), 0),
        "request.max_in_flight": (request.get("max_in_flight"), 1),
        "warmup.requests": (warmup.get("requests"), 200),
        "steady_state.requests": (steady.get("requests"), 2000),
        "steady_state.fixed_rps": (steady.get("fixed_rps"), 5),
        "steady_state.queue_on_missed_schedule": (
            steady.get("queue_on_missed_schedule"),
            False,
        ),
    }
    bad = [name for name, (actual, wanted) in expected.items() if actual != wanted]
    expected_arms = [
        {"id": "resnet56-direct", "model_name": "resnet56", "serving_mode": "direct"},
        {
            "id": "dispatcher-000733-fused",
            "model_name": "dispatcher-000733",
            "serving_mode": "fused",
        },
        {
            "id": "dispatcher-000733-distributed",
            "model_name": "dispatcher-000733",
            "serving_mode": "distributed",
        },
    ]
    if config.get("schema_version") != 2:
        bad.append("schema_version")
    if config.get("throughput_saturation_test") is not False:
        bad.append("throughput_saturation_test")
    if config.get("arms") != expected_arms:
        bad.append("arms")
    runtime = config.get("runtime", {})
    for key, wanted in (
        ("workers", 1),
        ("onnx_intra_op_threads", 1),
        ("onnx_inter_op_threads", 1),
        ("retry_count", 0),
    ):
        if runtime.get(key) != wanted:
            bad.append(f"runtime.{key}")
    if bad:
        raise ValueError("single-arm benchmark config differs from fixed contract: " + ", ".join(bad))


def validate_values(values: dict[str, str]) -> dict[str, Any]:
    missing = sorted(REQUIRED - values.keys())
    extra = sorted(values.keys() - REQUIRED)
    if missing or extra:
        details = []
        if missing:
            details.append("missing variables: " + ", ".join(missing))
        if extra:
            details.append("unexpected variables: " + ", ".join(extra))
        raise ValueError("; ".join(details))

    arm = values["ARM_ID"]
    if arm not in ARMS:
        raise ValueError("ARM_ID is not one of the three approved arms")
    for key in (
        "RUNTIME_IMAGE",
        "SHUFFLENET_MODELCAR",
        "MOBILENET_MODELCAR",
        "RESNET56_MODELCAR",
        "DISPATCHER_MODELCAR",
        "FUSED_MODELCAR",
    ):
        match = DIGEST_RE.fullmatch(values[key])
        if not match or match.group(1) == "0" * 64:
            raise ValueError(f"{key} must be a non-sentinel digest-pinned image")
    for key in (
        "SHUFFLENET_MODEL_SHA256",
        "MOBILENET_MODEL_SHA256",
        "RESNET56_MODEL_SHA256",
        "PROMOTION_FINGERPRINT",
    ):
        if not SHA_RE.fullmatch(values[key]) or values[key] == "0" * 64:
            raise ValueError(f"{key} must be a non-sentinel SHA-256")
    if not DNS_RE.fullmatch(values["BENCHMARK_JOB_NAME"]):
        raise ValueError("BENCHMARK_JOB_NAME must be a DNS label")
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,61}[a-z0-9])?", values["BENCHMARK_ID"]):
        raise ValueError("BENCHMARK_ID must be a DNS-safe identifier")
    if values["BENCHMARK_ID"] not in values["BENCHMARK_JOB_NAME"]:
        raise ValueError("BENCHMARK_JOB_NAME must include the unique BENCHMARK_ID")
    if not GSA_RE.fullmatch(values["BENCHMARK_GSA"]):
        raise ValueError("BENCHMARK_GSA is not a valid GSA address")
    if not TRACE_RE.fullmatch(values["BENCHMARK_TRACE_URI"]):
        raise ValueError("BENCHMARK_TRACE_URI uses an unexpected prefix")
    if not OUTPUT_RE.fullmatch(values["BENCHMARK_OUTPUT_URI"]):
        raise ValueError("BENCHMARK_OUTPUT_URI uses an unexpected prefix")
    if not values["BENCHMARK_OUTPUT_URI"].endswith("/" + values["BENCHMARK_ID"]):
        raise ValueError("BENCHMARK_OUTPUT_URI must end with BENCHMARK_ID")
    if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f-]{27,}", values["SERVING_NODE_UID"]):
        raise ValueError("SERVING_NODE_UID must be a Kubernetes UID")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,62}", values["SERVING_MACHINE_TYPE"]):
        raise ValueError("SERVING_MACHINE_TYPE is invalid")

    promotion = _decode_json(values["PROMOTION_JSON_BASE64"], "PROMOTION_JSON_BASE64")
    fingerprint = promotion.get("fingerprint")
    body = {key: child for key, child in promotion.items() if key != "fingerprint"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if fingerprint != hashlib.sha256(encoded).hexdigest() or fingerprint != values[
        "PROMOTION_FINGERPRINT"
    ]:
        raise ValueError("promotion ConfigMap content or fingerprint is invalid")
    promoted_experts = {
        item.get("id"): item.get("modelcar_image")
        for item in promotion.get("experts", [])
        if isinstance(item, dict)
    }
    expected_experts = {
        "shufflenetv2_x0_5": values["SHUFFLENET_MODELCAR"],
        "mobilenetv2_x0_5": values["MOBILENET_MODELCAR"],
        "resnet56": values["RESNET56_MODELCAR"],
    }
    if (
        promotion.get("runtime_image") != values["RUNTIME_IMAGE"]
        or promotion.get("dispatcher", {}).get("modelcar_image")
        != values["DISPATCHER_MODELCAR"]
        or promoted_experts != expected_experts
    ):
        raise ValueError("promotion image digests differ from deployment inputs")

    config = _decode_yaml(values["BENCHMARK_CONFIG_BASE64"], "BENCHMARK_CONFIG_BASE64")
    _validate_benchmark_config(config)
    return config


def _substitute(text: str, values: dict[str, str], source: Path) -> str:
    required = set(VARIABLE_RE.findall(text))
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"{source}: missing variables: {', '.join(missing)}")
    result = VARIABLE_RE.sub(lambda match: values[match.group(1)], text)
    if VARIABLE_RE.search(result):
        raise ValueError(f"unresolved variable in {source}")
    return result


def _fused_service(router: dict[str, Any], values: dict[str, str]) -> dict[str, Any]:
    service = copy.deepcopy(router)
    metadata = service["metadata"]
    metadata["name"] = "router-dispatcher-000733-fused"
    metadata["labels"]["pertinence.dev/benchmark-role"] = "fused-router"
    metadata["annotations"].update(
        {
            "pertinence.dev/arm-id": "dispatcher-000733-fused",
            "pertinence.dev/serving-mode": "fused",
            "pertinence.dev/runtime-image": values["RUNTIME_IMAGE"],
            "pertinence.dev/shufflenet-sha256": values["SHUFFLENET_MODEL_SHA256"],
            "pertinence.dev/mobilenet-sha256": values["MOBILENET_MODEL_SHA256"],
            "pertinence.dev/resnet56-sha256": values["RESNET56_MODEL_SHA256"],
        }
    )
    predictor = service["spec"]["predictor"]
    predictor["storageUris"] = [{"uri": "oci://" + values["FUSED_MODELCAR"], "mountPath": "/mnt/models"}]
    container = predictor["containers"][0]
    container["resources"] = {
        "requests": {"cpu": "4", "memory": "4Gi"},
        "limits": {"cpu": "4", "memory": "4Gi"},
    }
    container["args"] = [
        "serve", "router", "--promotion", "/etc/pertinence/promotion.json",
        "--bundle-path", "/mnt/models", "--router-mode", "fused-dynamic",
        "--host", "0.0.0.0", "--port", "8080",
    ]
    return service


def _scope_platform_config_maps(
    platform: str, config_map_names: dict[str, str]
) -> str:
    documents = [item for item in yaml.safe_load_all(platform) if isinstance(item, dict)]
    found: set[str] = set()
    for document in documents:
        if document.get("kind") != "ConfigMap":
            continue
        metadata = document.get("metadata", {})
        original_name = metadata.get("name")
        if original_name in config_map_names:
            metadata["name"] = config_map_names[original_name]
            found.add(original_name)
    if found != set(config_map_names):
        missing = sorted(set(config_map_names) - found)
        raise ValueError("platform template omits ConfigMaps: " + ", ".join(missing))
    return "---\n".join(
        yaml.safe_dump(document, sort_keys=False).rstrip() + "\n" for document in documents
    )


def _scope_service_config_map_refs(
    services: dict[str, dict[str, Any]], config_map_names: dict[str, str]
) -> None:
    for service in services.values():
        volumes = service.get("spec", {}).get("predictor", {}).get("volumes", [])
        for volume in volumes:
            if not isinstance(volume, dict):
                continue
            config_map = volume.get("configMap")
            if not isinstance(config_map, dict):
                continue
            name = config_map.get("name")
            if name in config_map_names:
                config_map["name"] = config_map_names[name]


def render(env_path: Path, output_dir: Path, root: Path) -> None:
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    values = read_env(env_path)
    validate_values(values)
    arm = values["ARM_ID"]
    config_map_names = benchmark_config_map_names(values["BENCHMARK_ID"])
    generated = {
        **values,
        "ARM_ENDPOINT": ARMS[arm]["endpoint"],
        "PROMOTION_CONFIG_MAP_NAME": config_map_names["promotion-dispatcher-000733"],
        "BENCHMARK_CONFIG_MAP_NAME": config_map_names["benchmark-config"],
    }

    platform_path = root / "manifests" / "00-platform.yaml.tmpl"
    identity_path = root / "manifests" / "05-benchmark-workload-identity.yaml.tmpl"
    legacy_path = root / "manifests" / "10-workloads.yaml.tmpl"
    job_path = root / "manifests" / "20-single-arm-job-v2.yaml.tmpl"
    platform = _substitute(platform_path.read_text(encoding="utf-8"), generated, platform_path)
    platform = _scope_platform_config_maps(platform, config_map_names)
    identity = _substitute(identity_path.read_text(encoding="utf-8"), generated, identity_path)
    legacy = _substitute(legacy_path.read_text(encoding="utf-8"), generated, legacy_path)
    job = _substitute(job_path.read_text(encoding="utf-8"), generated, job_path)

    services = {
        item["metadata"]["name"]: item
        for item in yaml.safe_load_all(legacy)
        if isinstance(item, dict) and item.get("kind") == "InferenceService"
    }
    if arm == "dispatcher-000733-fused":
        services = {
            "router-dispatcher-000733-fused": _fused_service(
                services["router-dispatcher-000733"], values
            )
        }
    else:
        services = {name: services[name] for name in sorted(ARMS[arm]["services"])}
        if arm == "resnet56-direct":
            services["expert-resnet56"]["spec"]["predictor"]["containers"][0]["args"].extend(
                ["--serving-mode", "direct"]
            )
        for service in services.values():
            annotations = service["metadata"].setdefault("annotations", {})
            annotations.update(
                {
                    "pertinence.dev/arm-id": arm,
                    "pertinence.dev/serving-mode": ARMS[arm]["mode"],
                    "pertinence.dev/runtime-image": values["RUNTIME_IMAGE"],
                    "pertinence.dev/shufflenet-sha256": values["SHUFFLENET_MODEL_SHA256"],
                    "pertinence.dev/mobilenet-sha256": values["MOBILENET_MODEL_SHA256"],
                    "pertinence.dev/resnet56-sha256": values["RESNET56_MODEL_SHA256"],
                }
            )

    _scope_service_config_map_refs(services, config_map_names)

    target = "---\n".join(
        yaml.safe_dump(service, sort_keys=False).rstrip() + "\n" for service in services.values()
    )
    outputs = {
        "00-platform.yaml": platform,
        "05-benchmark-workload-identity.yaml": identity,
        "10-target.yaml": target,
        "20-single-arm-job.yaml": job,
    }
    output_dir.mkdir(mode=0o700, parents=False)
    for name, content in outputs.items():
        destination = output_dir / name
        destination.write_text(content, encoding="utf-8")
        destination.chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        render(args.env.resolve(strict=True), args.output_dir.resolve(), root)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
