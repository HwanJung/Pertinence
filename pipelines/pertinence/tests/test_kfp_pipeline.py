"""Local IR compilation verifies isolation and component execution policies."""

# ruff: noqa: E402 -- skip cleanly when the optional SDK is absent.

import json
import sys

import pytest
import yaml

pytest.importorskip("kfp")

from kfp import compiler

from pipelines.pertinence.components import make_components
from pipelines.pertinence.pipeline import make_pipeline
from pipelines.pertinence.submit import main as submit_main


def test_compile_fixed_dag_and_no_final_input_to_search(tmp_path):
    image = "example/pertinence@sha256:" + "a" * 64
    path = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(
        make_pipeline(image, pipeline_root="gs://pertinence-runs/pipeline-root"), str(path)
    )
    ir = yaml.safe_load(path.read_text())
    assert set(ir["root"]["inputDefinitions"]["parameters"]) == {
        "cifar10_uri",
        "cifar10_sha256",
        "shufflenet_checkpoint_uri",
        "shufflenet_checkpoint_sha256",
        "mobilenet_checkpoint_uri",
        "mobilenet_checkpoint_sha256",
        "resnet56_checkpoint_uri",
        "resnet56_checkpoint_sha256",
        "run_config_uri",
        "run_config_sha256",
    }
    tasks = ir["root"]["dag"]["tasks"]
    train = tasks["train-dispatcher"]
    assert set(train["inputs"]["artifacts"]) == {
        "train_cache",
        "search_cache",
        "normalized_contract",
    }
    assert "final" not in json.dumps(train)
    assert "volume" not in json.dumps(ir).lower()
    assert train.get("retryPolicy", {}).get("maxRetryCount", 0) == 0
    assert (
        not tasks["register-dispatchers"].get("cachingOptions", {}).get("enableCache", False)
    )
    assert tasks["evaluate-final"]["retryPolicy"]["maxRetryCount"] == 1
    for name in (
        "prepare-dataset-bundle",
        "export-expert-bundle",
        "validate-run-contract",
        "build-routing-dataset",
        "train-dispatcher",
        "evaluate-final",
    ):
        assert not tasks[name].get("cachingOptions", {}).get("enableCache", False)
    assert "nvidia.com/gpu" not in json.dumps(ir)
    assert ir["pipelineInfo"]["name"] == "pertinence-cifar10-cpu"
    assert "gs://pertinence-runs/pipeline-root" in json.dumps(ir)
    containers = [
        executor["container"]
        for executor in ir["deploymentSpec"]["executors"].values()
        if "container" in executor
    ]
    assert len(containers) == 7
    assert all(container["image"] == image for container in containers)
    assert all(
        container["command"] == ["python", "-m", "pertinence.pipeline_cli"]
        for container in containers
    )


def test_mutable_image_tag_rejected():
    with pytest.raises(ValueError, match="digest"):
        make_components("example/pertinence:latest")


def test_non_gcs_pipeline_root_rejected():
    image = "example/pertinence@sha256:" + "a" * 64
    with pytest.raises(ValueError, match="gs://"):
        make_pipeline(image, pipeline_root="s3://bucket/runs")


def test_cache_can_be_enabled_for_recovery_runs(tmp_path):
    image = "example/pertinence@sha256:" + "a" * 64
    path = tmp_path / "pipeline-cached.yaml"
    compiler.Compiler().compile(
        make_pipeline(
            image,
            pipeline_root="gs://pertinence-runs/pipeline-root",
            enable_cache=True,
        ),
        str(path),
    )
    tasks = yaml.safe_load(path.read_text())["root"]["dag"]["tasks"]
    assert tasks["validate-run-contract"]["cachingOptions"]["enableCache"]
    assert tasks["build-routing-dataset"]["cachingOptions"]["enableCache"]
    assert tasks["train-dispatcher"]["cachingOptions"]["enableCache"]
    assert not tasks["register-dispatchers"].get("cachingOptions", {}).get("enableCache", False)


def test_gke_submit_passes_service_account_and_gcs_parameters(monkeypatch, capsys):
    calls = {}

    class Client:
        def __init__(self, **kwargs):
            calls["client"] = kwargs

        def create_run_from_pipeline_package(self, **kwargs):
            calls["run"] = kwargs
            return type("Result", (), {"run_id": "run-123"})()

    digest = "a" * 64
    arguments = [
        "submit",
        "--pipeline",
        "pipeline.yaml",
        "--pipeline-root",
        "gs://runs/root",
        "--namespace",
        "kubeflow",
        "--service-account",
        "pertinence-pipeline",
    ]
    for name in (
        "cifar10",
        "shufflenet-checkpoint",
        "mobilenet-checkpoint",
        "resnet56-checkpoint",
        "run-config",
    ):
        arguments.extend([f"--{name}-uri", f"gs://raw/{name}", f"--{name}-sha256", digest])
    monkeypatch.setattr("kfp.Client", Client)
    monkeypatch.setattr(sys, "argv", arguments)
    submit_main()
    assert capsys.readouterr().out.strip() == "run-123"
    assert calls["client"] == {"host": "http://127.0.0.1:8888", "namespace": "kubeflow"}
    assert calls["run"]["service_account"] == "pertinence-pipeline"
    assert calls["run"]["pipeline_root"] == "gs://runs/root"
    assert not calls["run"]["enable_caching"]
