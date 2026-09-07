"""Read-only preflight checks for the staged CIFAR-10 experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
import gc
import platform
import shutil
import sys
import tomllib
from typing import Any

import torch
from packaging.version import InvalidVersion, Version

from .assets import load_asset_manifest, prepare_assets, verify_extracted_source
from .cache import build_prediction_caches, make_cache_request
from .config import load_config
from .data import evaluation_transform, load_cifar10, make_splits
from .experts import FrozenFeatureExtractor, load_expert


MINIMUM_FREE_BYTES = 2 * 1024**3


class PreflightError(RuntimeError):
    """Raised when an experiment must not start in the current environment."""


@dataclass(frozen=True, slots=True)
class PreflightReport:
    status: str
    python: str
    device: str
    device_name: str
    free_bytes: int
    source_fingerprint: str
    dataset_sizes: dict[str, int]
    model_output_shapes: dict[str, list[int]]
    cache_plan: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_python_and_dependencies(workspace_root: Path) -> None:
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        raise PreflightError(
            f"Python 3.11 or 3.12 is required, got {platform.python_version()}"
        )
    with (workspace_root / "pyproject.toml").open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["dependencies"]
    mismatches: list[str] = []
    for requirement in dependencies:
        name, expected = requirement.split("==", maxsplit=1)
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            mismatches.append(f"{name}: missing (expected {expected})")
            continue
        try:
            versions_match = Version(actual) == Version(expected)
        except InvalidVersion:
            versions_match = actual == expected
        if not versions_match:
            mismatches.append(f"{name}: {actual} (expected {expected})")
    if mismatches:
        raise PreflightError("pinned dependency mismatch: " + "; ".join(mismatches))


def _check_device(device_value: str) -> tuple[torch.device, str]:
    device = torch.device(device_value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise PreflightError(
                "CUDA was requested but is unavailable; check GPU allocation, driver visibility, "
                "and the container runtime before starting cache inference"
            )
        index = device.index if device.index is not None else torch.cuda.current_device()
        try:
            probe = torch.empty(1, device=device)
            del probe
            return device, torch.cuda.get_device_name(index)
        except RuntimeError as error:
            raise PreflightError(f"CUDA allocation probe failed on {device}: {error}") from error
    if device.type != "cpu":
        raise PreflightError(f"only cpu and cuda devices are supported, got {device_value!r}")
    return device, platform.processor() or "CPU"


def run_preflight(
    *,
    config_path: Path,
    manifest_path: Path,
    artifact_root: Path,
    dataset_root: Path,
    source_root: Path,
    cache_directory: Path,
    device_override: str | None = None,
    smoke_samples: int = 2,
) -> PreflightReport:
    """Validate all local inputs and execute a tiny forward pass without writes."""

    if smoke_samples <= 0:
        raise PreflightError("smoke_samples must be positive")
    config = load_config(config_path)
    workspace_root = config.workspace_root
    _check_python_and_dependencies(workspace_root)
    device_value = str(device_override or config.raw["experiment"]["device"])
    device, device_name = _check_device(device_value)

    manifest = load_asset_manifest(manifest_path)
    prepare_assets(manifest, artifact_root, verify_only=True)
    source_fingerprint = verify_extracted_source(manifest, artifact_root, source_root)

    disk_probe = cache_directory.resolve()
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    free_bytes = shutil.disk_usage(disk_probe).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise PreflightError(
            f"at least {MINIMUM_FREE_BYTES} free bytes are required near {cache_directory}, "
            f"found {free_bytes}"
        )

    transform = evaluation_transform(
        config.raw["data"]["normalize_mean"], config.raw["data"]["normalize_std"]
    )
    train_dataset = load_cifar10(dataset_root, train=True, transform=transform)
    test_dataset = load_cifar10(dataset_root, train=False, transform=transform)
    expected_sizes = {"train": 50_000, "test": 10_000}
    actual_sizes = {"train": len(train_dataset), "test": len(test_dataset)}
    if actual_sizes != expected_sizes:
        raise PreflightError(
            f"unexpected CIFAR-10 dataset sizes: {actual_sizes}, expected {expected_sizes}"
        )
    images = torch.stack([test_dataset[index][0] for index in range(smoke_samples)]).to(device)

    request = make_cache_request(
        config,
        manifest,
        artifact_root=artifact_root,
        source_root=source_root,
        output_directory=cache_directory,
        device=device_value,
    )
    splits = make_splits(
        int(config.raw["data"]["ga_search_size"]),
        int(config.raw["data"]["final_evaluation_size"]),
        int(config.raw["data"]["split_seed"]),
    )
    cache_plan = build_prediction_caches(request, splits, dry_run=True)

    output_shapes: dict[str, list[int]] = {}
    for model_name in request.model_order:
        model = load_expert(
            model_name,
            source_root=request.source_root,
            checkpoint=request.checkpoint_paths[model_name],
            device=device,
        )
        with torch.no_grad():
            if model_name == request.feature_extractor:
                extracted = FrozenFeatureExtractor(model)(images)
                logits = extracted.logits
                if extracted.features.ndim != 2 or extracted.features.shape[0] != smoke_samples:
                    raise PreflightError("feature extractor returned an invalid feature tensor")
            else:
                logits = model(images)
        if logits.shape != (smoke_samples, 10) or not torch.isfinite(logits).all():
            raise PreflightError(
                f"expert {model_name!r} returned invalid smoke output {tuple(logits.shape)}"
            )
        output_shapes[model_name] = list(logits.shape)
        del logits
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return PreflightReport(
        status="passed",
        python=platform.python_version(),
        device=device_value,
        device_name=device_name,
        free_bytes=free_bytes,
        source_fingerprint=source_fingerprint,
        dataset_sizes=actual_sizes,
        model_output_shapes=output_shapes,
        cache_plan=[asdict(result) for result in cache_plan.results],
    )
