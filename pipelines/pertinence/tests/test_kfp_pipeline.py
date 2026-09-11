"""Local IR compilation verifies isolation and component execution policies."""

# ruff: noqa: E402 -- skip cleanly when the optional SDK is absent.

import json

import pytest
import yaml

pytest.importorskip("kfp")

from kfp import compiler

from pipelines.pertinence.components import make_components
from pipelines.pertinence.pipeline import make_pipeline


def test_compile_fixed_dag_and_no_final_input_to_search(tmp_path):
    image = "example/pertinence@sha256:" + "a" * 64
    path = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(make_pipeline(image), str(path))
    ir = yaml.safe_load(path.read_text())
    assert set(ir["root"]["inputDefinitions"]["parameters"]) == {
        "dataset_bundle_uri",
        "dataset_bundle_sha256",
        "expert_bundle_uri",
        "expert_bundle_sha256",
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
        not tasks["register-pareto-dispatchers"].get("cachingOptions", {}).get("enableCache", False)
    )
    assert tasks["evaluate-final"]["retryPolicy"]["maxRetryCount"] == 1
    containers = [
        executor["container"]
        for executor in ir["deploymentSpec"]["executors"].values()
        if "container" in executor
    ]
    assert len(containers) == 5
    assert all(container["image"] == image for container in containers)
    assert all(
        container["command"] == ["python", "-m", "pertinence.pipeline_cli"]
        for container in containers
    )


def test_mutable_image_tag_rejected():
    with pytest.raises(ValueError, match="digest"):
        make_components("example/pertinence:latest")
