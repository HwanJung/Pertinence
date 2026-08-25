"""Offline, fingerprinted caches for expert routing experiments."""

from __future__ import annotations

import gc
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .assets import AssetManifest
from .config import ExperimentConfig
from .data import SplitIndices
from .experts import FrozenFeatureExtractor, load_expert
from .reproducibility import canonical_json_hash


CACHE_SCHEMA_VERSION = 1
SPLIT_NAMES = ("dispatcher_train", "ga_search", "final_evaluation")


class CacheError(RuntimeError):
    """Base class for prediction-cache errors."""


class CacheFingerprintMismatch(CacheError):
    """Raised before inference when an existing cache belongs to another run."""


@dataclass(frozen=True, slots=True)
class CacheBuildRequest:
    """All paths and identities needed by the cache builder."""

    model_order: tuple[str, ...]
    feature_extractor: str
    source_root: Path
    checkpoint_paths: Mapping[str, Path]
    output_directory: Path
    asset_fingerprint: str
    config_fingerprint: str
    batch_size: int
    num_workers: int
    device: str


@dataclass(frozen=True, slots=True)
class SplitCacheResult:
    split: str
    path: str
    fingerprint: str
    status: Literal["planned", "written", "reused"]
    samples: int


@dataclass(frozen=True, slots=True)
class CacheBuildReport:
    schema_version: int
    results: tuple[SplitCacheResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "results": [asdict(result) for result in self.results],
        }


class _IndexedSubset(Dataset[tuple[Tensor, int, int]]):
    def __init__(self, dataset: Dataset[Any], indices: np.ndarray) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> tuple[Tensor, int, int]:
        sample_index = int(self.indices[position])
        image, target = self.dataset[sample_index]
        if not isinstance(image, Tensor):
            raise TypeError("cache datasets must return transformed torch.Tensor images")
        return image, int(target), sample_index


def asset_manifest_fingerprint(manifest: AssetManifest) -> str:
    """Fingerprint asset identities independent of manifest entry order."""

    assets = sorted(
        (asdict(asset) for asset in manifest.assets),
        key=lambda item: (str(item["kind"]), str(item["name"])),
    )
    return canonical_json_hash(
        {"schema_version": manifest.schema_version, "assets": assets}
    )


def experiment_config_fingerprint(config: ExperimentConfig) -> str:
    return canonical_json_hash(config.raw)


def checkpoint_paths_from_manifest(
    manifest: AssetManifest, artifact_root: str | Path
) -> dict[str, Path]:
    """Resolve checkpoint asset paths without downloading or opening them."""

    root = Path(artifact_root).resolve()
    return {
        asset.name: root / asset.path
        for asset in manifest.assets
        if asset.kind == "checkpoint"
    }


def make_cache_request(
    config: ExperimentConfig,
    manifest: AssetManifest,
    *,
    artifact_root: str | Path,
    source_root: str | Path,
    output_directory: str | Path,
    device: str | None = None,
) -> CacheBuildRequest:
    """Translate config and manifest data into a loosely coupled build request."""

    return CacheBuildRequest(
        model_order=config.candidates,
        feature_extractor=config.feature_extractor,
        source_root=Path(source_root),
        checkpoint_paths=checkpoint_paths_from_manifest(manifest, artifact_root),
        output_directory=Path(output_directory),
        asset_fingerprint=asset_manifest_fingerprint(manifest),
        config_fingerprint=experiment_config_fingerprint(config),
        batch_size=int(config.raw["data"]["batch_size"]),
        num_workers=int(config.raw["data"]["num_workers"]),
        device=str(device or config.raw["experiment"]["device"]),
    )


def route_labels_from_predictions(predictions: Tensor, targets: Tensor) -> Tensor:
    """Choose the first (cheapest) correct expert, falling back to expert zero."""

    if predictions.ndim != 2:
        raise ValueError("predictions must have shape [samples, models]")
    if targets.ndim != 1 or predictions.shape[0] != targets.shape[0]:
        raise ValueError("targets must have shape [samples]")
    if predictions.shape[1] == 0:
        raise ValueError("at least one expert prediction is required")

    correct = predictions.eq(targets.unsqueeze(1))
    any_correct = correct.any(dim=1)
    first_correct = correct.to(torch.int64).argmax(dim=1)
    return torch.where(any_correct, first_correct, torch.zeros_like(first_correct))


def _split_arrays(splits: SplitIndices) -> dict[str, np.ndarray]:
    return {
        "dispatcher_train": splits.dispatcher_train,
        "ga_search": splits.ga_search,
        "final_evaluation": splits.final_evaluation,
    }


def _split_fingerprint(name: str, indices: np.ndarray, splits: SplitIndices) -> str:
    return canonical_json_hash(
        {
            "name": name,
            "indices": np.asarray(indices, dtype=np.int64).tolist(),
            "split_manifest_fingerprint": splits.fingerprint,
        }
    )


def expected_cache_fingerprint(
    *,
    split: str,
    split_fingerprint: str,
    model_order: Sequence[str],
    feature_extractor: str,
    asset_fingerprint: str,
    config_fingerprint: str,
) -> str:
    return canonical_json_hash(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "split": split,
            "split_fingerprint": split_fingerprint,
            "model_order": list(model_order),
            "feature_extractor": feature_extractor,
            "asset_fingerprint": asset_fingerprint,
            "config_fingerprint": config_fingerprint,
        }
    )


def cache_path(output_directory: str | Path, split: str) -> Path:
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown cache split: {split!r}")
    return Path(output_directory) / f"{split}.pt"


def save_cache_atomically(payload: Mapping[str, object], destination: str | Path) -> None:
    """Write a torch cache through a same-directory fsync and atomic rename."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".part", dir=target.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _existing_cache_matches(
    path: Path,
    expected_fingerprint: str,
    *,
    split: str,
    expected_indices: np.ndarray,
    model_order: Sequence[str],
    feature_extractor: str,
) -> bool:
    if not path.exists():
        return False
    if not path.is_file():
        raise CacheFingerprintMismatch(f"cache path is not a regular file: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CacheFingerprintMismatch(f"cannot safely inspect existing cache: {path}") from error
    actual = payload.get("fingerprint") if isinstance(payload, Mapping) else None
    if actual != expected_fingerprint:
        raise CacheFingerprintMismatch(
            f"existing cache fingerprint mismatch at {path}: "
            f"expected {expected_fingerprint}, got {actual!r}"
        )
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise CacheFingerprintMismatch(f"cache schema mismatch at {path}")
    if payload.get("split") != split:
        raise CacheFingerprintMismatch(f"cache split mismatch at {path}")
    if tuple(payload.get("model_order", ())) != tuple(model_order):
        raise CacheFingerprintMismatch(f"cache model order mismatch at {path}")
    if payload.get("feature_extractor") != feature_extractor:
        raise CacheFingerprintMismatch(f"cache extractor mismatch at {path}")
    tensors = {
        key: payload.get(key)
        for key in (
            "features",
            "expert_predictions",
            "targets",
            "sample_indices",
            "route_labels",
        )
    }
    if not all(isinstance(value, Tensor) for value in tensors.values()):
        raise CacheFingerprintMismatch(f"cache has missing or non-tensor fields at {path}")
    sample_count = len(expected_indices)
    if (
        tensors["features"].ndim != 2
        or tensors["features"].shape[0] != sample_count
        or tuple(tensors["expert_predictions"].shape) != (sample_count, len(model_order))
        or tuple(tensors["targets"].shape) != (sample_count,)
        or tuple(tensors["sample_indices"].shape) != (sample_count,)
        or tuple(tensors["route_labels"].shape) != (sample_count,)
    ):
        raise CacheFingerprintMismatch(f"cache tensor shape mismatch at {path}")
    expected_index_tensor = torch.as_tensor(expected_indices, dtype=torch.int64)
    if not torch.equal(tensors["sample_indices"].to(torch.int64), expected_index_tensor):
        raise CacheFingerprintMismatch(f"cache sample indices mismatch at {path}")
    return True


def _make_loaders(
    *,
    train_dataset: Dataset[Any],
    test_dataset: Dataset[Any],
    split_arrays: Mapping[str, np.ndarray],
    pending_splits: Sequence[str],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> dict[str, DataLoader[tuple[Tensor, Tensor, Tensor]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    datasets = {
        "dispatcher_train": train_dataset,
        "ga_search": test_dataset,
        "final_evaluation": test_dataset,
    }
    return {
        name: DataLoader(
            _IndexedSubset(datasets[name], split_arrays[name]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        for name in pending_splits
    }


def _collect_one_model(
    model: nn.Module,
    loaders: Mapping[str, DataLoader[tuple[Tensor, Tensor, Tensor]]],
    *,
    device: torch.device,
    capture_features: bool,
) -> dict[str, tuple[Tensor, Tensor, Tensor, Tensor | None]]:
    model.requires_grad_(False)
    model.eval()
    adapter = FrozenFeatureExtractor(model) if capture_features else None
    collected: dict[str, tuple[Tensor, Tensor, Tensor, Tensor | None]] = {}

    for split, loader in loaders.items():
        prediction_batches: list[Tensor] = []
        target_batches: list[Tensor] = []
        index_batches: list[Tensor] = []
        feature_batches: list[Tensor] = []
        with torch.no_grad():
            for images, targets, sample_indices in loader:
                images = images.to(device, non_blocking=device.type == "cuda")
                if adapter is None:
                    logits = model(images)
                else:
                    output = adapter(images)
                    logits = output.logits
                    feature_batches.append(output.features.detach().cpu())
                if not isinstance(logits, Tensor) or logits.ndim != 2:
                    raise CacheError("expert must return logits with shape [batch, classes]")
                prediction_batches.append(logits.argmax(dim=1).to(torch.int64).cpu())
                target_batches.append(targets.to(torch.int64).cpu())
                index_batches.append(sample_indices.to(torch.int64).cpu())

        predictions = torch.cat(prediction_batches)
        targets = torch.cat(target_batches)
        indices = torch.cat(index_batches)
        features = torch.cat(feature_batches) if capture_features else None
        collected[split] = predictions, targets, indices, features
    return collected


ModelLoader = Callable[..., nn.Module]


def build_prediction_caches(
    request: CacheBuildRequest,
    splits: SplitIndices,
    *,
    train_dataset: Dataset[Any] | None = None,
    test_dataset: Dataset[Any] | None = None,
    dry_run: bool = False,
    model_loader: ModelLoader = load_expert,
) -> CacheBuildReport:
    """Build all pending split caches while holding one device model at a time.

    ``dry_run`` computes only paths and fingerprints.  It neither requires
    datasets nor invokes ``model_loader`` nor creates the output directory.
    """

    if not request.model_order:
        raise ValueError("model_order must not be empty")
    if request.feature_extractor not in request.model_order:
        raise ValueError("feature extractor must be present in model_order")
    missing_checkpoints = set(request.model_order) - set(request.checkpoint_paths)
    if missing_checkpoints:
        raise ValueError(f"checkpoint paths missing for: {sorted(missing_checkpoints)}")

    arrays = _split_arrays(splits)
    split_fingerprints = {
        name: _split_fingerprint(name, arrays[name], splits) for name in SPLIT_NAMES
    }
    fingerprints = {
        name: expected_cache_fingerprint(
            split=name,
            split_fingerprint=split_fingerprints[name],
            model_order=request.model_order,
            feature_extractor=request.feature_extractor,
            asset_fingerprint=request.asset_fingerprint,
            config_fingerprint=request.config_fingerprint,
        )
        for name in SPLIT_NAMES
    }
    paths = {name: cache_path(request.output_directory, name) for name in SPLIT_NAMES}

    if dry_run:
        return CacheBuildReport(
            schema_version=CACHE_SCHEMA_VERSION,
            results=tuple(
                SplitCacheResult(
                    split=name,
                    path=str(paths[name]),
                    fingerprint=fingerprints[name],
                    status="planned",
                    samples=int(arrays[name].size),
                )
                for name in SPLIT_NAMES
            ),
        )

    reused = {
        name: _existing_cache_matches(
            paths[name],
            fingerprints[name],
            split=name,
            expected_indices=arrays[name],
            model_order=request.model_order,
            feature_extractor=request.feature_extractor,
        )
        for name in SPLIT_NAMES
    }
    pending = [name for name in SPLIT_NAMES if not reused[name]]
    if not pending:
        return CacheBuildReport(
            schema_version=CACHE_SCHEMA_VERSION,
            results=tuple(
                SplitCacheResult(
                    split=name,
                    path=str(paths[name]),
                    fingerprint=fingerprints[name],
                    status="reused",
                    samples=int(arrays[name].size),
                )
                for name in SPLIT_NAMES
            ),
        )
    if train_dataset is None or test_dataset is None:
        raise ValueError("train_dataset and test_dataset are required outside dry-run mode")

    device = torch.device(request.device)
    loaders = _make_loaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        split_arrays=arrays,
        pending_splits=pending,
        batch_size=request.batch_size,
        num_workers=request.num_workers,
        pin_memory=device.type == "cuda",
    )
    predictions_by_split: dict[str, list[Tensor]] = {name: [] for name in pending}
    targets_by_split: dict[str, Tensor] = {}
    indices_by_split: dict[str, Tensor] = {}
    features_by_split: dict[str, Tensor] = {}

    for model_name in request.model_order:
        model = model_loader(
            model_name,
            source_root=request.source_root,
            checkpoint=request.checkpoint_paths[model_name],
            device=device,
        )
        capture_features = model_name == request.feature_extractor
        collected = _collect_one_model(
            model, loaders, device=device, capture_features=capture_features
        )
        for split, (predictions, targets, indices, features) in collected.items():
            predictions_by_split[split].append(predictions)
            if split not in targets_by_split:
                targets_by_split[split] = targets
                indices_by_split[split] = indices
            elif not torch.equal(targets_by_split[split], targets) or not torch.equal(
                indices_by_split[split], indices
            ):
                raise CacheError(f"dataset ordering changed between experts for {split}")
            if features is not None:
                features_by_split[split] = features

        del collected
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results: list[SplitCacheResult] = []
    for split in SPLIT_NAMES:
        if reused[split]:
            status: Literal["written", "reused"] = "reused"
        else:
            if split not in features_by_split:
                raise CacheError(f"feature extractor produced no features for {split}")
            # Sample-major layout is consumed directly by CachedEvaluationData
            # and system_metrics: [samples, experts].
            predictions = torch.stack(predictions_by_split[split], dim=1)
            targets = targets_by_split[split]
            indices = indices_by_split[split]
            features = features_by_split[split]
            expected_size = int(arrays[split].size)
            if (
                predictions.shape != (expected_size, len(request.model_order))
                or targets.shape != (expected_size,)
                or indices.shape != (expected_size,)
                or features.shape[0] != expected_size
            ):
                raise CacheError(f"incomplete tensors collected for split {split}")
            route_labels = route_labels_from_predictions(predictions, targets)
            payload: dict[str, object] = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "split": split,
                "fingerprint": fingerprints[split],
                "asset_fingerprint": request.asset_fingerprint,
                "config_fingerprint": request.config_fingerprint,
                "split_fingerprint": split_fingerprints[split],
                "model_order": list(request.model_order),
                "feature_extractor": request.feature_extractor,
                "targets": targets,
                "sample_indices": indices,
                "expert_predictions": predictions,
                "features": features,
                "route_labels": route_labels,
            }
            save_cache_atomically(payload, paths[split])
            status = "written"
        results.append(
            SplitCacheResult(
                split=split,
                path=str(paths[split]),
                fingerprint=fingerprints[split],
                status=status,
                samples=int(arrays[split].size),
            )
        )

    return CacheBuildReport(schema_version=CACHE_SCHEMA_VERSION, results=tuple(results))
