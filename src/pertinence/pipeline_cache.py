"""Split-isolated routing cache construction and integrity checks."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .bundle_data import DatasetBundleReader, decode_image
from .contracts import DatasetBundle
from .pipeline_artifacts import (
    atomic_directory,
    file_inventory,
    local_path,
    read_sealed,
    seal,
    verify_artifact,
    write_json,
)
from .pipeline_validation import load_contract, selected_experts
from .reproducibility import canonical_json_hash
from .runtimes.onnx import ExpertRuntime


def route_labels(predictions: np.ndarray, targets: np.ndarray, fallback: int) -> np.ndarray:
    correct = predictions == targets[:, None]
    return np.where(correct.any(axis=1), correct.argmax(axis=1), fallback).astype(np.int64)


def build_routing_dataset(
    dataset_bundle: Path,
    expert_bundle: Path,
    normalized_contract: Path,
    split_manifest: Path,
    *,
    train_cache: Path,
    search_cache: Path,
    final_cache: Path,
) -> None:
    contract = load_contract(normalized_contract)
    dataset_root, expert_root = local_path(dataset_bundle), local_path(expert_bundle)
    verify_artifact(dataset_root, contract["inputs"]["dataset_sha256"])
    verify_artifact(expert_root, contract["inputs"]["experts_sha256"])
    split = read_sealed(split_manifest)
    if split["fingerprint"] != contract["splits"]["fingerprint"]:
        raise ValueError("split manifest differs from normalized contract")
    ids = split["sample_ids"]
    all_ids = [sample_id for items in ids.values() for sample_id in items]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("split sample IDs overlap")
    destinations = {"train": train_cache, "search": search_cache, "final": final_cache}
    if len({Path(path).resolve() for path in destinations.values()}) != 3:
        raise ValueError("cache output paths must be distinct")
    k, d = contract["dimensions"]["K"], contract["dimensions"]["D"]
    settings = contract["platform"]
    routing = contract["config"]["routing"]
    reader = DatasetBundleReader(dataset_root, DatasetBundle.from_dict(contract["dataset"]))
    with ExitStack() as stack:
        stages = {
            key: stack.enter_context(atomic_directory(path)) for key, path in destinations.items()
        }
        arrays = {}
        positions = {}
        for split_name, stage in stages.items():
            samples = len(ids[split_name])
            if (
                samples != contract["splits"][split_name]["samples"]
                or canonical_json_hash(ids[split_name])
                != contract["splits"][split_name]["sample_ids_sha256"]
            ):
                raise ValueError("split sample IDs differ from normalized contract")
            specs = {
                "features": (np.float32, (samples, d)),
                "expert_predictions": (np.int64, (samples, k)),
                "targets": (np.int64, (samples,)),
                "route_labels": (np.int64, (samples,)),
            }
            arrays[split_name] = {
                key: np.lib.format.open_memmap(
                    stage / f"{key}.npy", mode="w+", dtype=dtype, shape=shape
                )
                for key, (dtype, shape) in specs.items()
            }
            np.save(stage / "sample_ids.npy", np.asarray(ids[split_name]), allow_pickle=False)
            positions.update(
                {sample_id: (split_name, index) for index, sample_id in enumerate(ids[split_name])}
            )
        for expert_index, expert in enumerate(selected_experts(contract)):
            runtime = ExpertRuntime(expert_root, expert, device=settings["device"])
            extractor = expert["id"] == routing["feature_extractor_id"]
            visited = set()
            for pool in ("train", "evaluation"):
                for batch in reader.batches(pool, settings["inference_batch_size"]):
                    images = [
                        decode_image(
                            row["image"],
                            contract["dataset"]["image_codec"],
                            settings["max_image_pixels"],
                        )
                        for row in batch
                    ]
                    predictions, features = runtime.predict(images, features=extractor)
                    if extractor and features.shape[1] != d:
                        raise ValueError("feature dimension differs from smoke inference")
                    for index, row in enumerate(batch):
                        sample_id = row["sample_id"]
                        if sample_id not in positions or sample_id in visited:
                            raise ValueError("dataset IDs changed after validation")
                        visited.add(sample_id)
                        split_name, position = positions[sample_id]
                        cache = arrays[split_name]
                        cache["targets"][position] = row["target"]
                        cache["expert_predictions"][position, expert_index] = predictions[index]
                        if extractor:
                            cache["features"][position] = features[index]
            if visited != set(positions):
                raise ValueError("dataset sample coverage differs from split manifest")
            del runtime
        fallback = routing["expert_ids"].index(routing["fallback_expert_id"])
        for split_name, stage in stages.items():
            cache = arrays[split_name]
            cache["route_labels"][:] = route_labels(
                cache["expert_predictions"], cache["targets"], fallback
            )
            for array in cache.values():
                array.flush()
            manifest = seal(
                {
                    "schema_version": 1,
                    "kind": "routing-cache",
                    "split": split_name,
                    "contract_fingerprint": contract["fingerprint"],
                    "split_fingerprint": contract["splits"]["fingerprint"],
                    "sample_ids_sha256": contract["splits"][split_name]["sample_ids_sha256"],
                    "expert_order": routing["expert_ids"],
                    "dimensions": {"D": d, "K": k},
                    "samples": len(ids[split_name]),
                    "inputs": contract["inputs"],
                    "cost": contract["cost"],
                    "lineage": contract["lineage"],
                    "preprocessing": [expert["input"] for expert in selected_experts(contract)],
                    "files": file_inventory(stage),
                }
            )
            write_json(stage / "manifest.json", manifest)


def load_cache(path: Path, contract: dict, split: str) -> tuple[dict, dict]:
    manifest = read_sealed(path / "manifest.json")
    if (
        manifest.get("kind") != "routing-cache"
        or manifest.get("schema_version") != 1
        or manifest["contract_fingerprint"] != contract["fingerprint"]
        or manifest["split"] != split
        or manifest["expert_order"] != contract["config"]["routing"]["expert_ids"]
        or manifest["split_fingerprint"] != contract["splits"]["fingerprint"]
    ):
        raise ValueError("cache lineage, expert order, or split mismatch")
    expected_metadata = {
        "dimensions": {key: contract["dimensions"][key] for key in ("D", "K")},
        "samples": contract["splits"][split]["samples"],
        "sample_ids_sha256": contract["splits"][split]["sample_ids_sha256"],
        "inputs": contract["inputs"],
        "cost": contract["cost"],
        "lineage": contract["lineage"],
        "preprocessing": [expert["input"] for expert in selected_experts(contract)],
    }
    if any(manifest.get(key) != expected for key, expected in expected_metadata.items()):
        raise ValueError("cache metadata differs from normalized contract")
    if file_inventory(path, exclude=("manifest.json",)) != manifest["files"]:
        raise ValueError("cache file SHA-256 mismatch")
    arrays = {
        key: np.load(path / f"{key}.npy", mmap_mode="r", allow_pickle=False)
        for key in ("features", "expert_predictions", "targets", "route_labels", "sample_ids")
    }
    n = contract["splits"][split]["samples"]
    d, k = contract["dimensions"]["D"], contract["dimensions"]["K"]
    expected = {
        "features": (n, d),
        "expert_predictions": (n, k),
        "targets": (n,),
        "route_labels": (n,),
        "sample_ids": (n,),
    }
    for key, array in arrays.items():
        if array.shape != expected[key]:
            raise ValueError(f"cache {key} shape mismatch")
        if key == "sample_ids":
            if array.dtype.kind != "U":
                raise ValueError("cache sample_ids must be strings")
        elif key == "features":
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError("cache features must be finite float32")
        else:
            maximum = k if key == "route_labels" else contract["dimensions"]["C"]
            if array.dtype != np.int64 or np.any(array < 0) or np.any(array >= maximum):
                raise ValueError(f"cache {key} dtype/range mismatch")
    actual_ids = arrays["sample_ids"].tolist()
    if (
        len(set(actual_ids)) != n
        or canonical_json_hash(actual_ids) != contract["splits"][split]["sample_ids_sha256"]
    ):
        raise ValueError("cache sample IDs differ from contract")
    fallback = contract["config"]["routing"]["expert_ids"].index(
        contract["config"]["routing"]["fallback_expert_id"]
    )
    if not np.array_equal(
        arrays["route_labels"],
        route_labels(arrays["expert_predictions"], arrays["targets"], fallback),
    ):
        raise ValueError("cache route labels are inconsistent with predictions")
    return arrays, manifest
