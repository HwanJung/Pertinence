"""Static regression checks for the internal-only KServe deployment assets."""

from __future__ import annotations

import os
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).parents[1]
KSERVE = ROOT / "cloud" / "kserve"
FUSED_EXPERTS = ("shufflenetv2_x0_5", "mobilenetv2_x0_5", "resnet56")


def _sealed(value: dict) -> dict:
    fingerprint = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**value, "fingerprint": fingerprint}


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fused_modelcar_source(root: Path, expert_order: tuple[str, ...] = FUSED_EXPERTS) -> Path:
    evaluated = root / "evaluated"
    evaluated.mkdir(parents=True)
    expert_rows = []
    for index, expert_id in enumerate(FUSED_EXPERTS):
        directory = root / "experts" / expert_id
        directory.mkdir(parents=True)
        model = directory / "model.onnx"
        model.write_bytes(f"onnx-{index}".encode())
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        (directory / "model.sha256").write_text(f"{digest}  model.onnx\n", encoding="utf-8")
        expert_rows.append({"id": expert_id, "model_sha256": digest})

    checkpoint = evaluated / "dispatcher-000733.npz"
    np.savez(
        checkpoint,
        **{
            "linear.weight": np.zeros((3, 1024), dtype=np.float32),
            "linear.bias": np.zeros(3, dtype=np.float32),
        },
    )
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    contract = _sealed(
        {
            "config": {"routing": {"expert_ids": list(expert_order)}},
            "experts": {"experts": expert_rows},
        }
    )
    search = _sealed({"front_id": "front-1"})
    final = _sealed(
        {
            "front_id": "front-1",
            "contract_fingerprint": contract["fingerprint"],
            "search_fingerprint": search["fingerprint"],
            "solutions": [
                {"model_id": "dispatcher-000733", "checkpoint_sha256": checkpoint_sha}
            ],
        }
    )
    manifest = _sealed(
        {
            "front_id": "front-1",
            "contract_fingerprint": contract["fingerprint"],
            "search_fingerprint": search["fingerprint"],
            "final_report_fingerprint": final["fingerprint"],
            "models": [
                {
                    "model_id": "dispatcher-000733",
                    "checkpoint": checkpoint.name,
                    "checkpoint_sha256": checkpoint_sha,
                    "dimensions": {"D": 1024, "K": 3},
                    "expert_order": list(expert_order),
                }
            ],
        }
    )
    for name, value in (
        ("contract.json", contract),
        ("search-result.json", search),
        ("final-report.json", final),
        ("manifest.json", manifest),
    ):
        _write_json(evaluated / name, value)
    return root


def _prepare_modelcar(kind: str, source: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(KSERVE / "modelcar" / "prepare_bundle.py"),
            kind,
            str(source),
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


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


def test_fused_modelcar_seals_evaluated_bundle_and_three_promoted_experts(
    tmp_path: Path,
) -> None:
    source = _fused_modelcar_source(tmp_path / "source")
    result = _prepare_modelcar("fused-dynamic", source, tmp_path / "output")

    assert result.returncode == 0, result.stderr
    inventory = json.loads((tmp_path / "output/bundle/inventory.json").read_text())
    paths = {item["path"] for item in inventory["files"]}
    assert paths == {
        "evaluated/contract.json",
        "evaluated/search-result.json",
        "evaluated/final-report.json",
        "evaluated/manifest.json",
        "evaluated/dispatcher-000733.npz",
        *(f"experts/{expert_id}/model.onnx" for expert_id in FUSED_EXPERTS),
        *(f"experts/{expert_id}/model.sha256" for expert_id in FUSED_EXPERTS),
    }


@pytest.mark.parametrize("mutation", ["model-hash", "checkpoint", "expert-order", "extra", "symlink"])
def test_fused_modelcar_rejects_unsealed_or_unexpected_inputs(
    tmp_path: Path, mutation: str
) -> None:
    order = tuple(reversed(FUSED_EXPERTS)) if mutation == "expert-order" else FUSED_EXPERTS
    source = _fused_modelcar_source(tmp_path / "source", order)
    if mutation == "model-hash":
        model = source / "experts" / FUSED_EXPERTS[0] / "model.onnx"
        model.write_bytes(b"different-promoted-bytes")
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        model.with_name("model.sha256").write_text(
            f"{digest}  model.onnx\n", encoding="utf-8"
        )
    elif mutation == "checkpoint":
        with (source / "evaluated/dispatcher-000733.npz").open("ab") as stream:
            stream.write(b"changed")
    elif mutation == "extra":
        (source / "unexpected").write_text("no", encoding="utf-8")
    elif mutation == "symlink":
        (source / "unexpected-link").symlink_to(source / "evaluated/contract.json")

    result = _prepare_modelcar("fused-dynamic", source, tmp_path / "rejected")

    assert result.returncode != 0


def test_hybrid_modelcar_contains_only_local_extractor(tmp_path):
    source = _fused_modelcar_source(tmp_path / "source")
    output = tmp_path / "hybrid"
    completed = subprocess.run(
        [sys.executable, str(KSERVE / "modelcar" / "prepare_bundle.py"),
         "hybrid-dynamic", str(source), str(output)], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output / "bundle" / "experts" / FUSED_EXPERTS[0] / "model.onnx").is_file()
    assert not (output / "bundle" / "experts" / FUSED_EXPERTS[1]).exists()
    assert not (output / "bundle" / "experts" / FUSED_EXPERTS[2]).exists()
