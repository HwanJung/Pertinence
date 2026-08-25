"""Offline loading and feature extraction for the CIFAR-10 expert pool.

The upstream project normally exposes these models through PyTorch Hub.  Hub is
deliberately not used here: both the checked-out upstream source tree and each
checkpoint must be supplied as local paths by the caller.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import NamedTuple
from urllib.parse import urlparse

import torch
from torch import Tensor, nn


UPSTREAM_REPOSITORY = "chenyaofo/pytorch-cifar-models"
UPSTREAM_REVISION = "786c16252c0fc58ee9adac063f8337cc4a7a497a"
CATALOG_URL = "https://github.com/chenyaofo/pytorch-cifar-models#model-zoo"


@dataclass(frozen=True, slots=True)
class ExpertSpec:
    """Immutable catalog record for one cost-sorted Pareto expert."""

    name: str
    family: str
    source_module: str
    hub_entry: str
    checkpoint_url: str
    top1_accuracy_pct: float
    top5_accuracy_pct: float
    params_m: float
    madds_m: float

    @property
    def checkpoint_filename(self) -> str:
        return Path(urlparse(self.checkpoint_url).path).name

    @property
    def mflops(self) -> float:
        """FLOPs under this project's explicit convention, 1 MAC = 2 FLOPs."""

        return 2.0 * self.madds_m


_RELEASES = "https://github.com/chenyaofo/pytorch-cifar-models/releases/download"

# This is intentionally a tuple, in increasing catalog MAdds order.  It is the
# Pareto front computed over the 19 CNN checkpoints in the upstream CIFAR-10
# Model Zoo table, rather than an upstream collection carrying a Pareto label.
EXPERT_REGISTRY: tuple[ExpertSpec, ...] = (
    ExpertSpec(
        "shufflenetv2_x0_5",
        "ShuffleNetV2",
        "shufflenetv2.py",
        "cifar10_shufflenetv2_x0_5",
        f"{_RELEASES}/shufflenetv2/cifar10_shufflenetv2_x0_5-1308b4e9.pt",
        90.13,
        99.70,
        0.35,
        10.90,
    ),
    ExpertSpec(
        "mobilenetv2_x0_5",
        "MobileNetV2",
        "mobilenetv2.py",
        "cifar10_mobilenetv2_x0_5",
        f"{_RELEASES}/mobilenetv2/cifar10_mobilenetv2_x0_5-ca14ced9.pt",
        92.88,
        99.86,
        0.70,
        27.97,
    ),
    ExpertSpec(
        "shufflenetv2_x1_0",
        "ShuffleNetV2",
        "shufflenetv2.py",
        "cifar10_shufflenetv2_x1_0",
        f"{_RELEASES}/shufflenetv2/cifar10_shufflenetv2_x1_0-98807be3.pt",
        92.98,
        99.73,
        1.26,
        45.00,
    ),
    ExpertSpec(
        "mobilenetv2_x0_75",
        "MobileNetV2",
        "mobilenetv2.py",
        "cifar10_mobilenetv2_x0_75",
        f"{_RELEASES}/mobilenetv2/cifar10_mobilenetv2_x0_75-a53c314e.pt",
        93.72,
        99.79,
        1.37,
        59.31,
    ),
    ExpertSpec(
        "mobilenetv2_x1_0",
        "MobileNetV2",
        "mobilenetv2.py",
        "cifar10_mobilenetv2_x1_0",
        f"{_RELEASES}/mobilenetv2/cifar10_mobilenetv2_x1_0-fe6a5b48.pt",
        93.79,
        99.73,
        2.24,
        87.98,
    ),
    ExpertSpec(
        "resnet44",
        "ResNet",
        "resnet.py",
        "cifar10_resnet44",
        f"{_RELEASES}/resnet/cifar10_resnet44-2a3cabcb.pt",
        94.01,
        99.77,
        0.66,
        97.44,
    ),
    ExpertSpec(
        "resnet56",
        "ResNet",
        "resnet.py",
        "cifar10_resnet56",
        f"{_RELEASES}/resnet/cifar10_resnet56-187c023a.pt",
        94.37,
        99.83,
        0.86,
        125.75,
    ),
    ExpertSpec(
        "repvgg_a0",
        "RepVGG",
        "repvgg.py",
        "cifar10_repvgg_a0",
        f"{_RELEASES}/repvgg/cifar10_repvgg_a0-ef08a50e.pt",
        94.39,
        99.82,
        7.84,
        489.08,
    ),
    ExpertSpec(
        "repvgg_a1",
        "RepVGG",
        "repvgg.py",
        "cifar10_repvgg_a1",
        f"{_RELEASES}/repvgg/cifar10_repvgg_a1-38d2431b.pt",
        94.89,
        99.83,
        12.82,
        851.33,
    ),
    ExpertSpec(
        "repvgg_a2",
        "RepVGG",
        "repvgg.py",
        "cifar10_repvgg_a2",
        f"{_RELEASES}/repvgg/cifar10_repvgg_a2-09488915.pt",
        94.98,
        99.82,
        26.82,
        1850.10,
    ),
)

EXPERTS_BY_NAME: Mapping[str, ExpertSpec] = {spec.name: spec for spec in EXPERT_REGISTRY}


def get_expert_spec(name: str) -> ExpertSpec:
    """Look up an expert while producing a useful error for invalid config."""

    try:
        return EXPERTS_BY_NAME[name]
    except KeyError as error:
        choices = ", ".join(EXPERTS_BY_NAME)
        raise KeyError(f"unknown CIFAR-10 expert {name!r}; expected one of: {choices}") from error


def checkpoint_path(checkpoint_directory: str | Path, expert: str | ExpertSpec) -> Path:
    """Return the expected local checkpoint path without touching the network."""

    spec = get_expert_spec(expert) if isinstance(expert, str) else expert
    return Path(checkpoint_directory) / spec.checkpoint_filename


def _source_package_directory(source_root: str | Path) -> Path:
    root = Path(source_root)
    nested = root / "pytorch_cifar_models"
    return nested if nested.is_dir() else root


def _load_local_source_module(source_root: str | Path, filename: str) -> ModuleType:
    module_path = _source_package_directory(source_root) / filename
    if not module_path.is_file():
        raise FileNotFoundError(f"upstream model source not found: {module_path}")

    # Upstream modules use sys.modules[__name__] while creating their Hub entry
    # functions.  Give each local path a private, deterministic module name and
    # remove it after execution so two asset roots cannot silently share code.
    path_hash = hashlib.sha256(str(module_path.resolve()).encode()).hexdigest()[:16]
    module_name = f"_pertinence_chenyaofo_{module_path.stem}_{path_hash}"
    import_spec = importlib.util.spec_from_file_location(module_name, module_path)
    if import_spec is None or import_spec.loader is None:
        raise ImportError(f"cannot create an import specification for {module_path}")
    module = importlib.util.module_from_spec(import_spec)
    sys.modules[module_name] = module
    try:
        import_spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def build_expert(name: str, *, source_root: str | Path) -> nn.Module:
    """Instantiate one model solely from a local upstream source checkout."""

    spec = get_expert_spec(name)
    module = _load_local_source_module(source_root, spec.source_module)
    try:
        constructor = getattr(module, spec.hub_entry)
    except AttributeError as error:
        raise ImportError(
            f"{spec.hub_entry!r} is missing from local source module "
            f"{spec.source_module!r}"
        ) from error

    # `pretrained=False` is important: the upstream constructor otherwise uses
    # load_state_dict_from_url and would introduce an implicit network access.
    model = constructor(pretrained=False)
    if not isinstance(model, nn.Module):
        raise TypeError(f"{spec.hub_entry} returned {type(model).__name__}, not nn.Module")
    return model


def load_plain_state_dict(path: str | Path) -> dict[str, Tensor]:
    """Safely load a checkpoint that must be a plain tensor state dictionary."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"expert checkpoint not found: {checkpoint}")

    value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"checkpoint must contain a non-empty plain state_dict: {checkpoint}")
    plain_tensor_items = all(
        isinstance(key, str) and isinstance(tensor, Tensor) for key, tensor in value.items()
    )
    if not plain_tensor_items:
        raise TypeError(
            f"checkpoint must map parameter names directly to tensors (wrappers are rejected): "
            f"{checkpoint}"
        )
    return dict(value)


def load_expert(
    name: str,
    *,
    source_root: str | Path,
    checkpoint: str | Path,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Build and load a frozen expert from explicit local asset paths.

    RepVGG is deliberately kept in the checkpoint's original multi-branch form;
    this function never calls ``convert_to_inference_model`` or ``switch_to_deploy``.
    """

    model = build_expert(name, source_root=source_root)
    state_dict = load_plain_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.requires_grad_(False)
    model.eval()
    if device is not None:
        model.to(device)
    return model


def load_expert_from_directory(
    name: str,
    *,
    source_root: str | Path,
    checkpoint_directory: str | Path,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Convenience wrapper for a directory using registry checkpoint names."""

    path = checkpoint_path(checkpoint_directory, name)
    return load_expert(name, source_root=source_root, checkpoint=path, device=device)


class ExpertForward(NamedTuple):
    """Classifier logits and the exact tensor consumed by its final Linear."""

    logits: Tensor
    features: Tensor


class FrozenFeatureExtractor(nn.Module):
    """Expose a frozen expert's logits and penultimate features in one pass."""

    def __init__(self, expert: nn.Module) -> None:
        super().__init__()
        linear_layers = [module for module in expert.modules() if isinstance(module, nn.Linear)]
        if not linear_layers:
            raise ValueError("expert must contain at least one nn.Linear classifier")
        self.expert = expert
        self._classifier = linear_layers[-1]
        self.expert.requires_grad_(False)
        self.expert.eval()

    @property
    def feature_dim(self) -> int:
        return self._classifier.in_features

    def train(self, mode: bool = True) -> FrozenFeatureExtractor:
        # The eventual dispatcher head may enter training mode, but its frozen
        # feature extractor must retain BatchNorm statistics and disable dropout.
        super().train(mode)
        self.expert.eval()
        return self

    def forward(self, inputs: Tensor) -> ExpertForward:
        captured: list[Tensor] = []

        def capture_linear_input(_module: nn.Module, args: tuple[object, ...]) -> None:
            if not args or not isinstance(args[0], Tensor):
                raise RuntimeError("final nn.Linear did not receive a tensor input")
            captured.append(args[0])

        handle = self._classifier.register_forward_pre_hook(capture_linear_input)
        try:
            # no_grad (rather than inference_mode) produces detached ordinary
            # tensors that can subsequently be consumed by a trainable FC head.
            with torch.no_grad():
                logits = self.expert(inputs)
        finally:
            handle.remove()

        if not isinstance(logits, Tensor):
            raise TypeError("expert classifier must return one logits tensor")
        if len(captured) != 1:
            raise RuntimeError(
                f"expected final nn.Linear to run exactly once, observed {len(captured)} calls"
            )
        features = captured[0]
        if features.shape[-1] != self.feature_dim:
            raise RuntimeError(
                f"captured feature width {features.shape[-1]} does not match "
                f"classifier input width {self.feature_dim}"
            )
        return ExpertForward(logits=logits, features=features)
