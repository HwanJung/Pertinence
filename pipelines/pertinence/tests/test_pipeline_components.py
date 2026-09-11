"""Real tiny ONNX/Parquet end-to-end tests, with no network or dataset downloads."""

# ruff: noqa: E402 -- optional dependency checks must precede runtime imports.

from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import inspect
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("pyarrow")

import onnx
from onnx import TensorProto, helper, numpy_helper
import pyarrow as pa
import pyarrow.parquet as pq

from pertinence.bundle_data import package_dataset, package_experts
from pertinence.contracts import PlatformProfile, RunConfig
from pertinence.pipeline_cli import main
from pertinence.pipeline_artifacts import (
    artifact_sha256,
    file_inventory,
    read_sealed,
    seal,
    write_json,
)
from pertinence.pipeline_cache import build_routing_dataset, load_cache, route_labels
from pertinence.pipeline_registry import SQLiteRegistry, register_pareto_dispatchers
from pertinence.pipeline_search import evaluate_final, train_dispatcher
from pertinence.pipeline_validation import validate_run_contract


def make_inputs(root: Path, classes: int = 3, experts: int = 2, feature_dim: int = 16) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    source = root / "dataset-source"
    for pool, count in (("train", 12), ("evaluation", 8)):
        path = source / pool / "part-000.parquet"
        path.parent.mkdir(parents=True)
        rows = []
        for index in range(count):
            image = Image.new("RGB", (4, 4), (index * 20, 120, 255 - index * 20))
            encoded = io.BytesIO()
            image.save(encoded, format="PNG")
            rows.append(
                {
                    "sample_id": f"{pool}-{index:03d}",
                    "image": encoded.getvalue(),
                    "target": index % classes,
                }
            )
        pq.write_table(
            pa.Table.from_pylist(
                rows,
                schema=pa.schema(
                    [
                        ("sample_id", pa.string()),
                        ("image", pa.binary()),
                        ("target", pa.int64()),
                    ]
                ),
            ),
            path,
        )
    dataset = {
        "schema_version": 1,
        "task": "image_classification",
        "dataset_id": f"test-{classes}",
        "dataset_version": "v1",
        "num_classes": classes,
        "class_names": [f"class-{index}" for index in range(classes)],
        "image_codec": "png",
        "train_glob": "train/part-*.parquet",
        "evaluation_glob": "evaluation/part-*.parquet",
    }
    write_json(source / "dataset.yaml", dataset)
    dataset_path = root / "dataset"
    dataset_sha = package_dataset(source, dataset_path)
    expert_source = root / "experts-source"
    models = []
    rng = np.random.default_rng(10)
    for index in range(experts):
        model_path = f"models/expert-{index}/model.onnx"
        path = expert_source / model_path
        path.parent.mkdir(parents=True)
        graph = helper.make_graph(
            [
                helper.make_node("Flatten", ["image"], ["flat"], axis=1),
                helper.make_node("MatMul", ["flat", "feature_weight"], ["features"]),
                helper.make_node("MatMul", ["features", "logit_weight"], ["logits"]),
            ],
            "portable-test",
            [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["N", 3, 4, 4])],
            [
                helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", classes]),
                helper.make_tensor_value_info("features", TensorProto.FLOAT, ["N", feature_dim]),
            ],
            [
                numpy_helper.from_array(
                    rng.normal(size=(48, feature_dim)).astype(np.float32), "feature_weight"
                ),
                numpy_helper.from_array(
                    rng.normal(size=(feature_dim, classes)).astype(np.float32), "logit_weight"
                ),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
        onnx.save(model, path)
        models.append(
            {
                "id": f"expert-{index}",
                "version": "v1",
                "format": "onnx",
                "model_path": model_path,
                "num_classes": classes,
                "input": {
                    "layout": "NCHW",
                    "dtype": "float32",
                    "shape": [3, 4, 4],
                    "preprocessing": {
                        "resize": [4, 4],
                        "mean": [0.0, 0.0, 0.0],
                        "std": [1.0, 1.0, 1.0],
                    },
                },
                "outputs": {"logits": "logits", "features": "features"},
                "costs": {"mflops": float(index + 1)},
            }
        )
    write_json(
        expert_source / "experts.yaml",
        {"schema_version": 1, "bundle_id": "test-experts", "experts": models},
    )
    expert_path = root / "experts"
    expert_sha = package_experts(expert_source, expert_path)
    config = {
        "schema_version": 1,
        "run_name": "tiny-run",
        "seed": 42,
        "routing": {
            "expert_ids": [model["id"] for model in models],
            "feature_extractor_id": "expert-0",
            "fallback_expert_id": "expert-1",
            "cost_key": "mflops",
            "require_cost_sorted": True,
        },
        "split": {"search_size": 4, "final_size": 4, "seed": 7},
        "dispatcher": {
            "epochs": 1,
            "batch_size": 4,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "ens_beta": 0.9999,
            "normalize_class_weights": "mean_one",
        },
        "nsga2": {
            "population_size": 2,
            "generations": 2,
            "crossover": "SBX",
            "crossover_eta": 20,
            "crossover_probability": 0.9,
            "mutation": "polynomial",
            "mutation_eta": 25,
            "mutation_probability": None,
            "penalty_min": 0.0,
            "penalty_max": 100.0,
            "weighting_schemes": ["INS", "ISNS", "ENS"],
        },
    }
    config_path = root / "run.yaml"
    write_json(config_path, config)
    return {
        "dataset_bundle_uri": str(dataset_path),
        "dataset_bundle_sha256": dataset_sha,
        "expert_bundle_uri": str(expert_path),
        "expert_bundle_sha256": expert_sha,
        "run_config_uri": str(config_path),
        "run_config_sha256": artifact_sha256(config_path),
        "normalized_contract": root / "contract.json",
        "split_manifest": root / "split.json",
        "validation_report": root / "validation.json",
    }


def build(inputs: dict, root: Path) -> dict:
    contract = validate_run_contract(**inputs)
    build_routing_dataset(
        Path(inputs["dataset_bundle_uri"]),
        Path(inputs["expert_bundle_uri"]),
        inputs["normalized_contract"],
        inputs["split_manifest"],
        train_cache=root / "train",
        search_cache=root / "search",
        final_cache=root / "final",
    )
    return contract


@pytest.mark.parametrize("classes,experts,feature_dim", [(3, 2, 16), (10, 3, 7), (100, 5, 9)])
def test_all_components_dynamic_and_registry_idempotent(
    tmp_path, monkeypatch, classes, experts, feature_dim
):
    inputs = make_inputs(tmp_path, classes, experts, feature_dim)
    contract = build(inputs, tmp_path)
    assert contract["dimensions"] == {
        "C": classes,
        "D": feature_dim,
        "K": experts,
        "chromosome_genes": experts**2 + 1,
        "evaluation_count": 4,
    }
    splits = [
        load_cache(tmp_path / split, contract, split)[0] for split in ("train", "search", "final")
    ]
    assert not set(splits[1]["sample_ids"]) & set(splits[2]["sample_ids"])
    train_dispatcher(
        tmp_path / "train",
        tmp_path / "search",
        inputs["normalized_contract"],
        search_result=tmp_path / "search.json",
        pareto_bundle=tmp_path / "pareto",
    )
    search = read_sealed(tmp_path / "search.json")
    assert len(search["evaluations"]) == 4
    assert all(len(record["chromosome"]) == experts**2 + 1 for record in search["evaluations"])

    def no_training(*args, **kwargs):
        raise AssertionError("final evaluation must not train")

    monkeypatch.setattr("pertinence.optimization.train_linear_dispatcher", no_training)
    monkeypatch.setattr("pertinence.training.train_linear_dispatcher", no_training)
    evaluate_final(
        tmp_path / "final",
        tmp_path / "search.json",
        tmp_path / "pareto",
        inputs["normalized_contract"],
        final_report=tmp_path / "final.json",
        evaluated_bundle=tmp_path / "evaluated",
    )
    report = read_sealed(tmp_path / "final.json")
    assert [item["evaluation_id"] for item in report["solutions"]] == search["pareto_ids"]
    for solution in report["solutions"]:
        metrics = solution["final_metrics"]
        assert np.shape(metrics["confusion_matrix"]) == (experts, experts)
        assert sum(metrics["selected_counts"]) == 4
        assert "checkpoint" not in solution  # report exposes logical model IDs only
    registry = SQLiteRegistry(tmp_path / "registry.db")
    first = register_pareto_dispatchers(
        tmp_path / "evaluated", registry=registry, registration_report=tmp_path / "registered.json"
    )
    second = register_pareto_dispatchers(
        tmp_path / "evaluated",
        registry=registry,
        registration_report=tmp_path / "registered-again.json",
    )
    assert first == second
    with sqlite3.connect(registry.path) as db:
        records = db.execute("SELECT metadata_json, checkpoint FROM dispatcher_versions").fetchall()
    assert len(records) == len(search["pareto_ids"])
    assert json.loads(records[0][0])["contract"] == contract
    assert records[0][1].startswith(b"PK")
    assert (
        main(
            [
                "register-pareto-dispatchers",
                "--evaluated-bundle",
                str(tmp_path / "evaluated"),
                "--registration-report",
                str(tmp_path / "cli-registered.json"),
                "--registry-db",
                str(registry.path),
            ]
        )
        == 0
    )


def modify_manifest(inputs: dict, kind: str, change) -> None:
    if kind == "config":
        path = Path(inputs["run_config_uri"])
        digest_key = "run_config_sha256"
        root = path
    else:
        key = "dataset" if kind == "dataset" else "expert"
        root = Path(inputs[f"{key}_bundle_uri"])
        path = root / ("dataset.yaml" if kind == "dataset" else "experts.yaml")
        digest_key = f"{key}_bundle_sha256"
    value = json.loads(path.read_text())
    change(value)
    write_json(path, value)
    inputs[digest_key] = artifact_sha256(root)


@pytest.mark.parametrize(
    "kind,change,error",
    [
        ("dataset", lambda d: d.update(schema_version=2), "schema_version"),
        ("dataset", lambda d: d.update(train_samples=13), "actual row count"),
        ("dataset", lambda d: d.update(class_names=["wrong"]), "class_names"),
        ("dataset", lambda d: d.update(train_glob="../escape.parquet"), "unsafe"),
        ("dataset", lambda d: d.update(content_sha256="0" * 64), "content_sha256"),
        ("config", lambda d: d["split"].update(final_size=5), "search_size"),
        (
            "config",
            lambda d: d["routing"].update(expert_ids=["expert-1", "expert-0"]),
            "cost order",
        ),
        ("config", lambda d: d["routing"].update(expert_ids=["expert-0", "expert-0"]), "duplicate"),
        ("config", lambda d: d["routing"].update(fallback_expert_id="missing"), "belong"),
        ("config", lambda d: d["routing"].update(cost_key="latency_ms"), "router_cost"),
        ("config", lambda d: d["nsga2"].update(population_size=100000), "platform limit"),
        ("experts", lambda d: d["experts"][0].update(format="python"), "runtime"),
        ("experts", lambda d: d["experts"][0].update(num_classes=99), "class count"),
        ("experts", lambda d: d["experts"][0].update(model_path="../model.onnx"), "unsafe"),
        ("experts", lambda d: d["experts"][0]["costs"].update(mflops=-1), "finite"),
        ("experts", lambda d: d["experts"][0].update(costs={"other_unit": 1}), "missing cost"),
        (
            "experts",
            lambda d: d["experts"][0]["outputs"].update(features="missing"),
            "output names",
        ),
        (
            "experts",
            lambda d: d["experts"][0]["input"]["preprocessing"].update(code="evil"),
            "unknown",
        ),
    ],
)
def test_validation_failure_matrix(tmp_path, kind, change, error):
    inputs = make_inputs(tmp_path)
    modify_manifest(inputs, kind, change)
    with pytest.raises(ValueError, match=error):
        validate_run_contract(**inputs)
    assert not inputs["normalized_contract"].exists()


def test_wrong_hash_and_resource_limits(tmp_path):
    inputs = make_inputs(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_run_contract(**{**inputs, "dataset_bundle_sha256": "0" * 64})
    with pytest.raises(ValueError, match="platform limit"):
        validate_run_contract(**inputs, profile=replace(PlatformProfile(), max_feature_dim=1))
    with pytest.raises(ValueError, match="byte limit"):
        validate_run_contract(**inputs, profile=replace(PlatformProfile(), max_bundle_bytes=1))


@pytest.mark.parametrize(
    "change,error",
    [
        (lambda rows: rows[0].update(sample_id=rows[1]["sample_id"]), "duplicate"),
        (lambda rows: rows[0].update(sample_id="evaluation-000"), "overlapping"),
        (lambda rows: rows[0].update(target=100), "target outside"),
        (lambda rows: rows[0].update(image=b"invalid"), ""),
    ],
)
def test_bad_dataset_rows(tmp_path, change, error):
    make_inputs(tmp_path)
    source = tmp_path / "dataset-source"
    path = source / "train/part-000.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    change(rows)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises((ValueError, OSError), match=error):
        package_dataset(source, tmp_path / "invalid-dataset")
    assert not (tmp_path / "invalid-dataset").exists()


def test_cache_corruption_stale_split_and_determinism(tmp_path):
    inputs = make_inputs(tmp_path)
    contract = build(inputs, tmp_path)
    again = validate_run_contract(**{**inputs, "normalized_contract": tmp_path / "again.json"})
    assert again == contract
    with pytest.raises(ValueError, match="split mismatch"):
        load_cache(tmp_path / "final", contract, "search")
    with (tmp_path / "train/features.npy").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        load_cache(tmp_path / "train", contract, "train")
    split = read_sealed(inputs["split_manifest"])
    split["sample_ids"]["final"][0] = split["sample_ids"]["search"][0]
    write_json(
        inputs["split_manifest"], seal({k: v for k, v in split.items() if k != "fingerprint"})
    )
    with pytest.raises(ValueError, match="split manifest differs"):
        build_routing_dataset(
            Path(inputs["dataset_bundle_uri"]),
            Path(inputs["expert_bundle_uri"]),
            inputs["normalized_contract"],
            inputs["split_manifest"],
            train_cache=tmp_path / "t2",
            search_cache=tmp_path / "s2",
            final_cache=tmp_path / "f2",
        )


@pytest.mark.parametrize(
    "field", ["dimensions", "expert_order", "cache_fingerprints", "checkpoint"]
)
def test_stale_or_unsafe_checkpoint_rejected(tmp_path, field):
    inputs = make_inputs(tmp_path)
    build(inputs, tmp_path)
    train_dispatcher(
        tmp_path / "train",
        tmp_path / "search",
        inputs["normalized_contract"],
        search_result=tmp_path / "search.json",
        pareto_bundle=tmp_path / "pareto",
    )
    manifest_path = tmp_path / "pareto/manifest.json"
    manifest = read_sealed(manifest_path)
    manifest["models"][0][field] = {"D": 999, "K": 2} if field == "dimensions" else "../escape.npz"
    write_json(manifest_path, seal({k: v for k, v in manifest.items() if k != "fingerprint"}))
    with pytest.raises(ValueError, match="checkpoint|unsafe"):
        evaluate_final(
            tmp_path / "final",
            tmp_path / "search.json",
            tmp_path / "pareto",
            inputs["normalized_contract"],
            final_report=tmp_path / "final.json",
            evaluated_bundle=tmp_path / "evaluated",
        )
    assert not (tmp_path / "evaluated").exists()


def test_registry_transaction_rolls_back_and_detects_conflicts(tmp_path):
    registry = SQLiteRegistry(tmp_path / "registry.db")
    original = {
        "idempotency_key": "original",
        "model_id": "m",
        "metadata": {"a": 1},
        "checkpoint": b"one",
    }
    registry.register_many([original])
    with pytest.raises(ValueError, match="different content"):
        registry.register_many(
            [{**original, "idempotency_key": "new"}, {**original, "checkpoint": b"changed"}]
        )
    with sqlite3.connect(registry.path) as db:
        assert db.execute("SELECT COUNT(*) FROM dispatcher_versions").fetchone()[0] == 1


def test_route_fallback_and_search_interface():
    assert route_labels(np.array([[0, 1, 1], [0, 0, 0]]), np.array([1, 2]), 2).tolist() == [1, 2]
    assert set(inspect.signature(train_dispatcher).parameters) == {
        "train_cache",
        "search_cache",
        "normalized_contract",
        "search_result",
        "pareto_bundle",
    }


def test_duplicate_yaml_keys_and_nonfinite_configuration():
    with pytest.raises(ValueError, match="unique"):
        RunConfig.from_text("schema_version: 1\nschema_version: 2\n")


def test_symlinks_rejected(tmp_path):
    inputs = make_inputs(tmp_path)
    source = Path(inputs["expert_bundle_uri"])
    (source / "link").symlink_to(tmp_path / "run.yaml")
    with pytest.raises(ValueError, match="symlink"):
        artifact_sha256(source)


@pytest.mark.parametrize(
    "failure,error",
    [
        ("logits_shape", "logits"),
        ("features_rank", "features"),
        ("features_nonfinite", "features"),
        ("logits_nonfinite", "logits"),
        ("static_batch", "dynamic batch"),
        ("external_data", "external tensor"),
        ("custom_operator", "custom operator"),
    ],
)
def test_invalid_onnx_io_and_runtime_rejected(tmp_path, failure, error):
    inputs = make_inputs(tmp_path)
    expert_root = Path(inputs["expert_bundle_uri"])
    path = expert_root / "models/expert-0/model.onnx"
    model = onnx.load(path)
    output_name = None
    if failure == "logits_shape":
        modify_manifest(
            inputs, "experts", lambda d: d["experts"][0]["outputs"].update(logits="features")
        )
    elif failure == "features_rank":
        model.graph.initializer.append(
            numpy_helper.from_array(np.array([2], dtype=np.int64), "axes")
        )
        model.graph.node.append(
            helper.make_node("Unsqueeze", ["features", "axes"], ["bad_features"])
        )
        model.graph.output.append(
            helper.make_tensor_value_info("bad_features", TensorProto.FLOAT, ["N", 16, 1])
        )
        output_name = "bad_features"
    elif failure == "features_nonfinite":
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(0, dtype=np.float32), "zero")
        )
        model.graph.node.append(helper.make_node("Div", ["features", "zero"], ["bad_features"]))
        model.graph.output.append(
            helper.make_tensor_value_info("bad_features", TensorProto.FLOAT, ["N", 16])
        )
        output_name = "bad_features"
    elif failure == "logits_nonfinite":
        model.graph.initializer[1].CopyFrom(
            numpy_helper.from_array(np.full((16, 3), np.nan, dtype=np.float32), "logit_weight")
        )
    elif failure == "static_batch":
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 2
    elif failure == "external_data":
        tensor = model.graph.initializer[0]
        tensor.data_location = TensorProto.EXTERNAL
        tensor.external_data.add(key="location", value="../../untrusted.bin")
    elif failure == "custom_operator":
        model.graph.node[0].domain = "untrusted.custom"
    # Serialize directly: external-data tests must not cause ONNX's writer to follow that path.
    path.write_bytes(model.SerializeToString())

    def update_metadata(value):
        value["experts"][0]["model_sha256"] = artifact_sha256(path)
        if output_name:
            value["experts"][0]["outputs"]["features"] = output_name

    modify_manifest(inputs, "experts", update_metadata)
    with pytest.raises(ValueError, match=error):
        validate_run_contract(**inputs)


@pytest.mark.parametrize("field", ["features", "sample_ids", "route_labels", "expert_predictions"])
def test_semantically_invalid_cache_even_with_recomputed_file_hash(tmp_path, field):
    inputs = make_inputs(tmp_path)
    contract = build(inputs, tmp_path)
    path = tmp_path / "train" / f"{field}.npy"
    array = np.load(path)
    if field == "features":
        array = array[:, :-1]
    elif field == "sample_ids":
        array[0] = array[1]
    elif field == "route_labels":
        array[0] = 1 - array[0]
    else:
        array[0, 0] = 100
    np.save(path, array, allow_pickle=False)
    manifest_path = tmp_path / "train/manifest.json"
    manifest = read_sealed(manifest_path)
    manifest["files"] = file_inventory(path.parent, exclude=("manifest.json",))
    write_json(manifest_path, seal({k: v for k, v in manifest.items() if k != "fingerprint"}))
    with pytest.raises(ValueError, match="cache"):
        load_cache(path.parent, contract, "train")


def test_registry_concurrent_retries_are_idempotent(tmp_path):
    registry = SQLiteRegistry(tmp_path / "registry.db")
    record = {
        "idempotency_key": "same",
        "model_id": "model",
        "metadata": {"x": 1},
        "checkpoint": b"same checkpoint",
    }
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: registry.register_many([record]), range(4)))
    assert all(result == results[0] for result in results)
    with sqlite3.connect(registry.path) as db:
        assert db.execute("SELECT COUNT(*) FROM dispatcher_versions").fetchone()[0] == 1


def test_scientific_json_numbers_preserve_fingerprint(tmp_path):
    path = tmp_path / "scientific.json"
    value = seal({"tiny": 9e-5})
    write_json(path, value)
    assert read_sealed(path) == value
