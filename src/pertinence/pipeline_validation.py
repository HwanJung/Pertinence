"""Fail-fast validation component and normalized execution contract."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .bundle_data import DatasetBundleReader, decode_image
from .contracts import (
    DatasetBundle,
    ExpertBundle,
    NormalizedRunContract,
    PlatformProfile,
    RunConfig,
)
from .pipeline_artifacts import (
    code_fingerprint,
    file_inventory,
    local_path,
    read_sealed,
    safe_path,
    seal,
    verify_artifact,
    write_json,
)
from .reproducibility import canonical_json_hash
from .runtimes.onnx import ExpertRuntime


def load_contract(path: Path) -> dict:
    value = NormalizedRunContract.from_dict(read_sealed(path)).to_dict()
    if value["lineage"]["code_sha256"] != code_fingerprint():
        raise ValueError("component code differs from validated run; revalidate inputs")
    for package, expected in value["lineage"]["packages"].items():
        if version(package) != expected:
            raise ValueError(f"runtime dependency differs from validated run: {package}")
    return value


def selected_experts(contract: dict) -> list[dict]:
    inventory = {expert["id"]: expert for expert in contract["experts"]["experts"]}
    return [inventory[expert_id] for expert_id in contract["config"]["routing"]["expert_ids"]]


def validate_run_contract(
    dataset_bundle_uri: str,
    dataset_bundle_sha256: str,
    expert_bundle_uri: str,
    expert_bundle_sha256: str,
    run_config_uri: str,
    run_config_sha256: str,
    *,
    normalized_contract: Path,
    split_manifest: Path,
    validation_report: Path,
    profile: PlatformProfile | None = None,
) -> dict:
    platform = profile or PlatformProfile()
    dataset_root, expert_root, config_path = map(
        local_path, (dataset_bundle_uri, expert_bundle_uri, run_config_uri)
    )
    for path, expected in (
        (dataset_root, dataset_bundle_sha256),
        (expert_root, expert_bundle_sha256),
        (config_path, run_config_sha256),
    ):
        verify_artifact(path, expected, max_bytes=platform.max_bundle_bytes)
    dataset = DatasetBundle.from_text((dataset_root / "dataset.yaml").read_text()).to_dict()
    experts = ExpertBundle.from_text((expert_root / "experts.yaml").read_text()).to_dict()
    config = RunConfig.from_text(config_path.read_text()).to_dict()
    if dataset["content_sha256"] != canonical_json_hash(
        file_inventory(dataset_root, exclude=("dataset.yaml",))
    ):
        raise ValueError("dataset content_sha256 mismatch")
    samples = dataset["train_samples"] + dataset["evaluation_samples"]
    if samples > platform.max_samples or dataset["num_classes"] > platform.max_classes:
        raise ValueError("dataset exceeds platform sample/class limits")
    split = config["split"]
    if split["search_size"] + split["final_size"] != dataset["evaluation_samples"]:
        raise ValueError("search_size + final_size must equal evaluation_samples")
    routing = config["routing"]
    if len(experts["experts"]) > platform.max_experts:
        raise ValueError("expert bundle exceeds platform expert limit")
    inventory = {expert["id"]: expert for expert in experts["experts"]}
    if set(routing["expert_ids"]) - inventory.keys():
        raise ValueError("routing expert_ids contains unknown experts")
    chosen = [inventory[item] for item in routing["expert_ids"]]
    for expert in experts["experts"]:
        if expert["num_classes"] != dataset["num_classes"]:
            raise ValueError("expert class count differs from dataset")
        if expert.get("class_names", dataset["class_names"]) != dataset["class_names"]:
            raise ValueError("expert class index ordering differs from dataset")
        shape = expert["input"]["shape"]
        if shape[1] * shape[2] > platform.max_image_pixels:
            raise ValueError("expert input exceeds platform pixel limit")
        verify_artifact(
            safe_path(expert_root, expert["model_path"]),
            expert["model_sha256"],
            max_bytes=platform.max_model_bytes,
        )
    cost_key = routing["cost_key"]
    if any(cost_key not in expert["costs"] for expert in chosen):
        raise ValueError("missing cost key/unit in selected experts")
    costs = [expert["costs"][cost_key] for expert in chosen]
    if any(left > right for left, right in zip(costs, costs[1:])):
        raise ValueError(
            "expert_ids must be in ascending cost order; automatic reordering is disabled"
        )
    evaluations = config["nsga2"]["population_size"] * config["nsga2"]["generations"]
    if evaluations > platform.max_evaluations:
        raise ValueError("NSGA-II evaluation count exceeds platform limit")
    if evaluations * config["dispatcher"]["epochs"] * samples > platform.max_training_sample_visits:
        raise ValueError("dispatcher training workload exceeds platform limit")
    reader = DatasetBundleReader(dataset_root, DatasetBundle.from_dict(dataset))
    pools = reader.validate_rows(platform)
    smoke_rows = next(reader.batches("train", 2))
    smoke_images = [
        decode_image(row["image"], dataset["image_codec"], platform.max_image_pixels)
        for row in smoke_rows
    ]
    if len(smoke_images) == 1:
        smoke_images *= 2
    feature_dim = None
    for expert in chosen:
        runtime = ExpertRuntime(expert_root, expert, device=platform.device)
        extractor = expert["id"] == routing["feature_extractor_id"]
        _, features = runtime.predict(smoke_images, features=extractor)
        _, single = runtime.predict(smoke_images[:1], features=extractor)
        if extractor:
            feature_dim = features.shape[1]
            if single.shape[1] != feature_dim:
                raise ValueError("feature dimension changes with batch size")
        del runtime
    k, d = len(chosen), int(feature_dim)
    estimated_bytes = samples * (4 * d + 8 * k + 16 + 4 * platform.max_sample_id_bytes)
    if d > platform.max_feature_dim or estimated_bytes > platform.max_cache_bytes:
        raise ValueError("derived feature dimension/cache size exceeds platform limit")
    if evaluations * 4 * k * (d + 1) > platform.max_dispatcher_state_bytes:
        raise ValueError("search checkpoint state memory exceeds platform limit")
    # Split by stable IDs, independent of shard order.
    evaluation_ids = np.asarray(sorted(pools["evaluation"]))
    permutation = np.random.default_rng(split["seed"]).permutation(len(evaluation_ids))
    ids = {
        "train": sorted(pools["train"]),
        "search": evaluation_ids[permutation[: split["search_size"]]].tolist(),
        "final": evaluation_ids[permutation[split["search_size"] :]].tolist(),
    }
    split_payload = seal({"schema_version": 1, "seed": split["seed"], "sample_ids": ids})
    derived_splits = {
        key: {"samples": len(items), "sample_ids_sha256": canonical_json_hash(items)}
        for key, items in ids.items()
    }
    router_cost = routing.get("router_cost", 2.0 * d * k / 1_000_000.0)
    contract = seal(
        {
            "schema_version": 1,
            "dataset": dataset,
            "experts": experts,
            "config": config,
            "inputs": {
                "dataset_sha256": dataset_bundle_sha256,
                "experts_sha256": expert_bundle_sha256,
                "config_sha256": run_config_sha256,
            },
            "dimensions": {
                "C": dataset["num_classes"],
                "D": d,
                "K": k,
                "chromosome_genes": k * k + 1,
                "evaluation_count": evaluations,
            },
            "cost": {
                "key": cost_key,
                "unit": cost_key,
                "vector": costs,
                "router_cost": router_cost,
                "router_rule": "explicit" if "router_cost" in routing else "2*D*K/1e6",
                "system_rule": "extractor + router + selected (zero if extractor reused)",
            },
            "splits": {"fingerprint": split_payload["fingerprint"], **derived_splits},
            "lineage": {
                "code_sha256": code_fingerprint(),
                "image_digest": platform.image_digest,
                "preprocessing_version": 1,
                "packages": {
                    key: version(key)
                    for key in (
                        "numpy",
                        "torch",
                        "pymoo",
                        "onnx",
                        "onnxruntime",
                        "pyarrow",
                        "Pillow",
                        "protobuf",
                        "PyYAML",
                    )
                },
            },
            "platform": asdict(platform),
        }
    )
    NormalizedRunContract.from_dict(contract)
    write_json(normalized_contract, contract)
    write_json(split_manifest, split_payload)
    write_json(
        validation_report,
        seal(
            {
                "schema_version": 1,
                "valid": True,
                "contract_fingerprint": contract["fingerprint"],
                "dimensions": contract["dimensions"],
                "estimated_cache_bytes": estimated_bytes,
                "split_fingerprint": split_payload["fingerprint"],
            }
        ),
    )
    return contract
