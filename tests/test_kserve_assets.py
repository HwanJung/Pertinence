"""Static regression checks for the internal-only KServe deployment assets."""

from __future__ import annotations

import os
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).parents[1]
KSERVE = ROOT / "cloud" / "kserve"


def _render_env(tmp_path: Path, *, sentinel: bool = False) -> Path:
    text = (KSERVE / "manifests" / "images.env.example").read_text(encoding="utf-8")
    image_digest = "0" * 64 if sentinel else "a" * 64
    promotion_body = {
        "schema_version": 1,
        "runtime_image": f"asia-northeast3-docker.pkg.dev/test-project/pertinence/serving-runtime@sha256:{image_digest}",
        "dispatcher": {
            "modelcar_image": f"asia-northeast3-docker.pkg.dev/test-project/pertinence/dispatcher-000733-modelcar@sha256:{image_digest}"
        },
        "experts": [
            {
                "id": name,
                "modelcar_image": f"asia-northeast3-docker.pkg.dev/test-project/pertinence/{image}-modelcar@sha256:{image_digest}",
            }
            for name, image in (
                ("shufflenetv2_x0_5", "shufflenet"),
                ("mobilenetv2_x0_5", "mobilenet"),
                ("resnet56", "resnet56"),
            )
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(promotion_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    promotion = {**promotion_body, "fingerprint": fingerprint}
    promotion_b64 = base64.b64encode(json.dumps(promotion).encode()).decode()
    config_b64 = base64.b64encode(
        (ROOT / "experiments/cifar10/configs/benchmark.yaml").read_bytes()
    ).decode()
    if not sentinel:
        text = text.replace("0" * 64, "a" * 64)
    promotion_digest = "a" * 64 if not sentinel else "0" * 64
    text = text.replace(
        "PROMOTION_FINGERPRINT=" + promotion_digest,
        "PROMOTION_FINGERPRINT=" + fingerprint,
    )
    text = text.replace("PROJECT", "test-project")
    text = text.replace("REPLACE_BENCHMARK_GSA", "benchmark-runner")
    text = text.replace("REPLACE_BENCHMARK_ID", "benchmark-1")
    text = text.replace("REPLACE_BASE64_SEALED_PROMOTION", promotion_b64)
    text = text.replace("REPLACE_BASE64_BENCHMARK_YAML", config_b64)
    text = text.replace("REPLACE_TRACE_ID", "trace-1")
    path = tmp_path / "images.env"
    path.write_text(text, encoding="utf-8")
    return path


def _render(tmp_path: Path) -> list[dict[str, object]]:
    output = tmp_path / "rendered"
    environment = dict(os.environ, PYTHON_BIN=sys.executable)
    subprocess.run(
        [str(KSERVE / "scripts" / "render-manifests.sh"), str(_render_env(tmp_path)), str(output)],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    documents: list[dict[str, object]] = []
    for path in sorted(output.glob("*.yaml")):
        documents.extend(document for document in yaml.safe_load_all(path.read_text()) if document)
    return documents


def test_rendered_workloads_are_internal_pinned_and_pool_isolated(tmp_path: Path) -> None:
    documents = _render(tmp_path)
    services = [item for item in documents if item["kind"] == "InferenceService"]
    assert {service["metadata"]["name"] for service in services} == {
        "expert-shufflenet",
        "expert-mobilenet",
        "expert-resnet56",
        "router-dispatcher-000733",
    }
    assert not {"Ingress", "Gateway", "HTTPRoute", "NetworkPolicy"} & {
        item["kind"] for item in documents
    }

    for service in services:
        predictor = service["spec"]["predictor"]
        assert predictor["minReplicas"] == predictor["maxReplicas"] == 1
        assert predictor["nodeSelector"] == {
            "cloud.google.com/gke-nodepool": "serving-pool"
        }
        assert predictor["tolerations"] == [
            {
                "key": "workload",
                "operator": "Equal",
                "value": "pertinence-serving",
                "effect": "NoSchedule",
            }
        ]
        assert predictor["storageUris"][0]["uri"].startswith("oci://")
        assert "@sha256:" in predictor["storageUris"][0]["uri"]
        container = predictor["containers"][0]
        assert "@sha256:" in container["image"]
        assert container["resources"]["requests"] == {"cpu": "1", "memory": "1Gi"}
        assert container["resources"]["limits"] == {"cpu": "1", "memory": "1Gi"}
        assert container["securityContext"]["runAsNonRoot"] is True

    assert not [item for item in documents if item["kind"] == "Service"]
    jobs = [item for item in documents if item["kind"] == "Job"]
    assert len(jobs) == 1
    job = jobs[0]
    assert job["spec"]["suspend"] is True
    pod = job["spec"]["template"]["spec"]
    assert pod["nodeSelector"] == {"cloud.google.com/gke-nodepool": "loadgen-pool"}
    assert any(volume.get("emptyDir") == {} for volume in pod["volumes"])
    assert "persistentVolumeClaim" not in json.dumps(pod)
    args = pod["containers"][0]["args"]
    endpoints = [args[index + 1] for index, item in enumerate(args) if item == "--endpoint"]
    assert endpoints == [
        "resnet56=http://expert-resnet56-predictor.pertinence-serving.svc.cluster.local",
        "dispatcher-000733=http://router-dispatcher-000733-predictor.pertinence-serving.svc.cluster.local",
    ]


def test_promoted_expert_names_and_lineage_runtime_arguments(tmp_path: Path) -> None:
    documents = _render(tmp_path)
    services = {
        item["metadata"]["name"]: item
        for item in documents
        if item["kind"] == "InferenceService"
    }
    shuffle_args = services["expert-shufflenet"]["spec"]["predictor"]["containers"][0][
        "args"
    ]
    assert "shufflenetv2_x0_5" in shuffle_args
    assert shuffle_args[shuffle_args.index("--feature-dim") + 1] == "1024"
    assert "--model-sha256" in shuffle_args
    mobile_args = services["expert-mobilenet"]["spec"]["predictor"]["containers"][0]["args"]
    assert "mobilenetv2_x0_5" in mobile_args
    router_args = services["router-dispatcher-000733"]["spec"]["predictor"]["containers"][0][
        "args"
    ]
    assert router_args[router_args.index("--bundle-path") + 1] == "/mnt/models"
    assert router_args[router_args.index("--promotion") + 1] == "/etc/pertinence/promotion.json"
    assert router_args.count("--expert-url") == 3
    promotion = next(
        item
        for item in documents
        if item["kind"] == "ConfigMap"
        and item["metadata"]["name"] == "promotion-dispatcher-000733"
    )
    assert promotion["immutable"] is True
    predictor = services["router-dispatcher-000733"]["spec"]["predictor"]
    assert predictor["volumes"] == [
        {"name": "promotion", "configMap": {"name": "promotion-dispatcher-000733"}}
    ]


def test_renderer_rejects_sentinel_digests(tmp_path: Path) -> None:
    output = tmp_path / "rendered"
    result = subprocess.run(
        [
            str(KSERVE / "scripts" / "render-manifests.sh"),
            str(_render_env(tmp_path, sentinel=True)),
            str(output),
        ],
        check=False,
        env=dict(os.environ, PYTHON_BIN=sys.executable),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "sentinel" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "script",
    sorted((KSERVE / "scripts").glob("*.sh"))
    + sorted((KSERVE / "runtime").glob("*.sh"))
    + sorted((KSERVE / "modelcar").glob("*.sh")),
)
def test_shell_scripts_parse(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_install_values_pin_versions_and_disable_ingress() -> None:
    values = (KSERVE / "install" / "kserve-values.yaml.tmpl").read_text(encoding="utf-8")
    cert_values = (KSERVE / "install" / "cert-manager-values.yaml.tmpl").read_text(
        encoding="utf-8"
    )
    assert "version: v0.20.0" in values
    assert "deploymentMode: Standard" in values
    assert "disableIngressCreation: true" in values
    assert "enableGatewayApi: false" in values
    assert "enableModelcar: true" in values
    assert "v1.20.4" in cert_values
    assert "system-pool" in values and "system-pool" in cert_values


def test_runtime_and_modelcar_images_are_non_root_and_version_locked() -> None:
    runtime = (KSERVE / "runtime" / "Dockerfile").read_text(encoding="utf-8")
    requirements = (KSERVE / "runtime" / "requirements.lock").read_text(encoding="utf-8")
    modelcar = (KSERVE / "modelcar" / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG PYTHON_BASE_IMAGE=python:3.11" in runtime
    assert "USER 65532:65532" in runtime
    assert "numpy==1.26.4" in requirements
    assert "onnxruntime==1.20.1" in requirements
    assert "requests==2.33.0" in requirements
    assert "USER 65532:65532" in modelcar
    assert "COPY --chown=65532:65532 bundle/ /models/" in modelcar
    assert (KSERVE / "runtime" / "Dockerfile.dockerignore").is_file()


def test_build_and_audit_assets_are_fail_closed() -> None:
    cloudbuild = (KSERVE / "cloudbuild.yaml").read_text(encoding="utf-8")
    audit = (KSERVE / "scripts" / "audit-exposure.sh").read_text(encoding="utf-8")
    assert "for expert in shufflenetv2_x0_5 mobilenetv2_x0_5 resnet56" in cloudbuild
    assert "dispatcher-000733-modelcar:${_BUILD_ID}" in cloudbuild
    assert "--platform linux/amd64" in cloudbuild
    assert "|| true" not in audit
    assert "serving.kserve.io/stop=true" in (
        KSERVE / "scripts" / "stop.sh"
    ).read_text(encoding="utf-8")


def test_modelcar_preparer_embeds_inventory_and_rejects_extra_files(tmp_path: Path) -> None:
    source = tmp_path / "expert"
    source.mkdir()
    model = source / "model.onnx"
    model.write_bytes(b"onnx")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    (source / "model.sha256").write_text(digest + "  model.onnx\n")
    output = tmp_path / "context"
    subprocess.run(
        [
            sys.executable,
            str(KSERVE / "modelcar" / "prepare_bundle.py"),
            "expert",
            str(source),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory = json.loads((output / "bundle" / "inventory.json").read_text())
    assert [item["path"] for item in inventory["files"]] == ["model.onnx", "model.sha256"]

    (source / "unexpected").write_text("no")
    rejected = subprocess.run(
        [
            sys.executable,
            str(KSERVE / "modelcar" / "prepare_bundle.py"),
            "expert",
            str(source),
            str(tmp_path / "rejected"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "allowlist" in rejected.stderr
