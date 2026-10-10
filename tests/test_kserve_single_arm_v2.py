"""Regression tests for the sequential three-arm KServe deployment contract."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
KSERVE = ROOT / "cloud" / "kserve"
SCRIPTS = KSERVE / "scripts"
ARMS = {
    "resnet56-direct": {"expert-resnet56"},
    "dispatcher-000733-fused": {"router-dispatcher-000733-fused"},
    "dispatcher-000733-distributed": {
        "expert-shufflenet", "expert-mobilenet", "expert-resnet56",
        "router-dispatcher-000733",
    },
}


def _image(name: str, digest: str = "a" * 64) -> str:
    return f"asia-northeast3-docker.pkg.dev/test-project/pertinence/{name}@sha256:{digest}"


def _env(tmp_path: Path, arm: str, *, fixed_rps: int = 5) -> Path:
    runtime = _image("serving-runtime-fused")
    dispatcher = _image("dispatcher-000733-modelcar")
    experts = {
        "shufflenetv2_x0_5": _image("shufflenet-modelcar"),
        "mobilenetv2_x0_5": _image("mobilenet-modelcar"),
        "resnet56": _image("resnet56-modelcar"),
    }
    promotion_body = {
        "schema_version": 1,
        "runtime_image": runtime,
        "dispatcher": {"modelcar_image": dispatcher},
        "experts": [
            {"id": expert_id, "modelcar_image": image}
            for expert_id, image in experts.items()
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(promotion_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    promotion = {**promotion_body, "fingerprint": fingerprint}
    config = {
        "schema_version": 2,
        "throughput_saturation_test": False,
        "arms": [
            {"id": "resnet56-direct", "model_name": "resnet56", "serving_mode": "direct"},
            {"id": "dispatcher-000733-fused", "model_name": "dispatcher-000733", "serving_mode": "fused"},
            {"id": "dispatcher-000733-distributed", "model_name": "dispatcher-000733", "serving_mode": "distributed"},
        ],
        "request": {"concurrency": 1, "retry_count": 0, "max_in_flight": 1},
        "warmup": {"requests": 200, "distinct_samples": True},
        "steady_state": {
            "requests": 2000, "fixed_rps": fixed_rps,
            "queue_on_missed_schedule": False,
        },
        "runtime": {
            "workers": 1, "onnx_intra_op_threads": 1,
            "onnx_inter_op_threads": 1, "retry_count": 0,
        },
    }
    run_id = "session1-" + arm
    values = {
        "ARM_ID": arm,
        "BENCHMARK_JOB_NAME": "loadgen-" + run_id,
        "RUNTIME_IMAGE": runtime,
        "SHUFFLENET_MODELCAR": experts["shufflenetv2_x0_5"],
        "MOBILENET_MODELCAR": experts["mobilenetv2_x0_5"],
        "RESNET56_MODELCAR": experts["resnet56"],
        "DISPATCHER_MODELCAR": dispatcher,
        "FUSED_MODELCAR": _image("dispatcher-000733-fused-modelcar", "b" * 64),
        "SHUFFLENET_MODEL_SHA256": "c" * 64,
        "MOBILENET_MODEL_SHA256": "d" * 64,
        "RESNET56_MODEL_SHA256": "e" * 64,
        "PROMOTION_FINGERPRINT": fingerprint,
        "PROMOTION_JSON_BASE64": base64.b64encode(json.dumps(promotion).encode()).decode(),
        "BENCHMARK_CONFIG_BASE64": base64.b64encode(
            yaml.safe_dump(config).encode()
        ).decode(),
        "BENCHMARK_GSA": "benchmark-runner@test-project.iam.gserviceaccount.com",
        "BENCHMARK_ID": run_id,
        "BENCHMARK_TRACE_URI": (
            "gs://pertinence-509116-pertinence-runs/pertinence/benchmarks/traces/trace-1"
        ),
        "BENCHMARK_OUTPUT_URI": (
            "gs://pertinence-509116-pertinence-runs/pertinence/benchmarks/"
            f"fused-v2/runs/{run_id}"
        ),
        "SERVING_NODE_UID": "12345678-1234-1234-1234-123456789abc",
        "SERVING_MACHINE_TYPE": "n2-standard-8",
    }
    path = tmp_path / f"{arm}.env"
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")
    return path


def _render(tmp_path: Path, arm: str) -> tuple[Path, Path, list[dict[str, object]]]:
    env_path = _env(tmp_path, arm)
    output = tmp_path / f"rendered-{arm}"
    subprocess.run(
        [str(SCRIPTS / "render-single-arm-v2.sh"), str(env_path), str(output)],
        check=True,
        env={**os.environ, "PYTHON_BIN": sys.executable},
        capture_output=True,
        text=True,
    )
    documents = []
    for path in sorted(output.glob("*.yaml")):
        documents.extend(item for item in yaml.safe_load_all(path.read_text()) if item)
    return env_path, output, documents


@pytest.mark.parametrize("arm", list(ARMS))
def test_each_render_contains_only_its_arm_and_fixed_job(tmp_path: Path, arm: str) -> None:
    env_path, _, documents = _render(tmp_path, arm)
    values = dict(
        line.split("=", 1)
        for line in env_path.read_text(encoding="utf-8").splitlines()
        if line
    )
    suffix = hashlib.sha256(values["BENCHMARK_ID"].encode()).hexdigest()[:10]
    config_maps = [item for item in documents if item["kind"] == "ConfigMap"]
    config_map_names = {item["metadata"]["name"] for item in config_maps}
    assert len(config_map_names) == 2
    assert all(len(name) <= 63 and name.endswith("-" + suffix) for name in config_map_names)
    assert all(item["immutable"] is True for item in config_maps)
    services = [item for item in documents if item["kind"] == "InferenceService"]
    assert {item["metadata"]["name"] for item in services} == ARMS[arm]
    assert not {"Ingress", "Gateway", "HTTPRoute"} & {item["kind"] for item in documents}
    for service in services:
        predictor = service["spec"]["predictor"]
        assert predictor["nodeSelector"] == {"cloud.google.com/gke-nodepool": "serving-pool"}
        assert predictor["securityContext"]["runAsNonRoot"] is True
        assert "@sha256:" in predictor["storageUris"][0]["uri"]
        assert "@sha256:" in predictor["containers"][0]["image"]
        assert predictor["containers"][0]["securityContext"]["runAsNonRoot"] is True

    cpu = sum(int(item["spec"]["predictor"]["containers"][0]["resources"]["requests"]["cpu"])
              for item in services)
    memory = sum(int(item["spec"]["predictor"]["containers"][0]["resources"]["requests"]["memory"].removesuffix("Gi"))
                 for item in services)
    assert (cpu, memory) == ((1, 1) if arm == "resnet56-direct" else (4, 4))

    job = next(item for item in documents if item["kind"] == "Job")
    assert job["spec"]["suspend"] is True
    assert job["spec"]["backoffLimit"] == 0
    loadgen = job["spec"]["template"]["spec"]["containers"][0]
    job_volumes = job["spec"]["template"]["spec"]["volumes"]
    assert {
        volume["configMap"]["name"]
        for volume in job_volumes
        if "configMap" in volume
    } == config_map_names
    assert loadgen["args"][:2] == ["benchmark", "run-single"]
    assert loadgen["args"][loadgen["args"].index("--arm") + 1] == arm
    if arm == "resnet56-direct":
        direct_args = services[0]["spec"]["predictor"]["containers"][0]["args"]
        assert direct_args[direct_args.index("--serving-mode") + 1] == "direct"
    for service in services:
        for volume in service["spec"]["predictor"].get("volumes", []):
            if "configMap" in volume:
                assert volume["configMap"]["name"] in config_map_names


def test_each_arm_gets_distinct_benchmark_scoped_config_maps(tmp_path: Path) -> None:
    arm_config_maps = {}
    for arm in ARMS:
        _, _, documents = _render(tmp_path, arm)
        arm_config_maps[arm] = {
            item["metadata"]["name"]
            for item in documents
            if item["kind"] == "ConfigMap"
        }
    assert all(len(names) == 2 for names in arm_config_maps.values())
    assert len(set().union(*arm_config_maps.values())) == 2 * len(ARMS)


def test_config_map_names_are_dns_labels_for_long_dotted_benchmark_id() -> None:
    module = _validator_module()
    names = module.benchmark_config_map_names("run.with.dots-" + "a" * 49)
    assert len(set(names.values())) == 2
    assert all(len(name) <= 63 for name in names.values())
    assert all(module._RENDERER.DNS_RE.fullmatch(name) for name in names.values())


def test_fused_uses_one_modelcar_and_fused_dynamic_runtime(tmp_path: Path) -> None:
    _, _, documents = _render(tmp_path, "dispatcher-000733-fused")
    service = next(item for item in documents if item["kind"] == "InferenceService")
    predictor = service["spec"]["predictor"]
    assert len(predictor["storageUris"]) == 1
    assert "dispatcher-000733-fused-modelcar" in predictor["storageUris"][0]["uri"]
    args = predictor["containers"][0]["args"]
    assert args[args.index("--router-mode") + 1] == "fused-dynamic"
    assert "--expert-url" not in args


def test_renderer_is_create_only_and_rejects_non_five_rps(tmp_path: Path) -> None:
    env_path, output, _ = _render(tmp_path, "resnet56-direct")
    second = subprocess.run(
        [str(SCRIPTS / "render-single-arm-v2.sh"), str(env_path), str(output)],
        env={**os.environ, "PYTHON_BIN": sys.executable},
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode != 0
    assert "already exists" in second.stderr

    invalid_env = _env(tmp_path, "dispatcher-000733-fused", fixed_rps=4)
    invalid = subprocess.run(
        [str(SCRIPTS / "render-single-arm-v2.sh"), str(invalid_env), str(tmp_path / "invalid")],
        env={**os.environ, "PYTHON_BIN": sys.executable},
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode != 0
    assert "fixed contract" in invalid.stderr


def _validator_module():
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "validate-single-arm-v2.py"
    spec = importlib.util.spec_from_file_location("validate_single_arm_v2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_pod_resource_evidence_counts_modelcar_and_init_semantics() -> None:
    module = _validator_module()
    pod = {
        "metadata": {"name": "fused-pod", "uid": "uid-1"},
        "spec": {
            "nodeName": "serving-node",
            "containers": [
                {"name": "kserve-container", "image": _image("runtime"),
                 "resources": {"requests": {"cpu": "4", "memory": "4Gi"},
                               "limits": {"cpu": "4", "memory": "4Gi"}}},
                {"name": "storage-initializer", "image": _image("fused-modelcar"),
                 "resources": {"requests": {"cpu": "10m", "memory": "20Mi"},
                               "limits": {"cpu": "20m", "memory": "40Mi"}}},
            ],
            "initContainers": [
                {"name": "setup", "resources": {"requests": {"cpu": "5", "memory": "1Gi"},
                                                   "limits": {"cpu": "5", "memory": "1Gi"}}}
            ],
        },
    }
    evidence = module.aggregate_pod_resources(pod)
    assert evidence["requests"]["app_cpu_cores"] == "4.010"
    assert evidence["requests"]["effective_cpu_cores"] == "5"
    assert evidence["requests"]["app_memory_bytes"] == 4 * 1024**3 + 20 * 1024**2
    assert {item["name"] for item in evidence["containers"]} == {
        "kserve-container", "storage-initializer"
    }
