"""Train and evaluate the linear dispatcher from cached tensors only.

This module intentionally has no image loader or expert-model dependency.  The
expensive feature extraction and expert inference stages are expected to have
been cached before a dispatcher experiment starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import SystemMetrics, system_metrics
from .routing import (
    LinearDispatcher,
    WeightingScheme,
    class_weights,
    paper_penalty_loss,
)


@dataclass(frozen=True)
class DispatcherTrainingConfig:
    """FC training choices left unspecified by the paper.

    The epoch default is the paper value.  Adam and its learning rate are the
    explicit, documented assumptions used by this reproduction configuration.
    """

    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"
    ens_beta: float = 0.9999
    normalize_class_weights: bool = True
    seed: int = 0
    device: str = "cpu"

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.optimizer.lower() not in {"adam", "sgd"}:
            raise ValueError("optimizer must be 'adam' or 'sgd'")


@dataclass(frozen=True)
class CachedEvaluationData:
    """Cached inputs required for search or final system evaluation."""

    features: Tensor | np.ndarray
    expert_predictions: Tensor | np.ndarray
    targets: Tensor | np.ndarray
    ideal_routes: Tensor | np.ndarray

    @property
    def num_samples(self) -> int:
        return int(np.shape(self.features)[0])

    @property
    def num_experts(self) -> int:
        shape = np.shape(self.expert_predictions)
        if len(shape) != 2:
            raise ValueError("expert_predictions must be [samples, experts]")
        return int(shape[1])

    def validate(self) -> None:
        feature_shape = np.shape(self.features)
        prediction_shape = np.shape(self.expert_predictions)
        target_shape = np.shape(self.targets)
        route_shape = np.shape(self.ideal_routes)
        if len(feature_shape) != 2:
            raise ValueError("features must be [samples, feature_dim]")
        if len(prediction_shape) != 2:
            raise ValueError("expert_predictions must be [samples, experts]")
        samples = feature_shape[0]
        if prediction_shape[0] != samples:
            raise ValueError("features and expert_predictions have different sample counts")
        if target_shape != (samples,) or route_shape != (samples,):
            raise ValueError("targets and ideal_routes must contain one value per sample")
        if prediction_shape[1] == 0:
            raise ValueError("at least one expert is required")


@dataclass
class DispatcherTrainingResult:
    dispatcher: LinearDispatcher
    epoch_losses: tuple[float, ...]
    weighting_scheme: WeightingScheme
    class_weight_values: Tensor

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        """Return a portable snapshot without retaining accelerator tensors."""

        return {
            key: value.detach().cpu().numpy().copy()
            for key, value in self.dispatcher.state_dict().items()
        }


def _as_tensor(value: Tensor | np.ndarray, *, dtype: torch.dtype) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _as_numpy(value: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _optimizer_for(
    model: nn.Module, config: DispatcherTrainingConfig
) -> torch.optim.Optimizer:
    parameters = model.parameters()
    if config.optimizer.lower() == "adam":
        return torch.optim.Adam(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    return torch.optim.SGD(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def train_linear_dispatcher(
    features: Tensor | np.ndarray,
    route_labels: Tensor | np.ndarray,
    penalty_matrix: Tensor | np.ndarray,
    weighting_scheme: WeightingScheme,
    *,
    config: DispatcherTrainingConfig | None = None,
) -> DispatcherTrainingResult:
    """Train a newly initialized FC dispatcher using cached features.

    The function resets the FC initialization and mini-batch order to
    ``config.seed``.  Using the same seed for every individual makes the GA
    compare loss configurations rather than random initializations.
    """

    settings = config or DispatcherTrainingConfig()
    settings.validate()
    cached_features = _as_tensor(features, dtype=torch.float32)
    cached_routes = _as_tensor(route_labels, dtype=torch.long)
    penalties = _as_tensor(penalty_matrix, dtype=torch.float32)
    if cached_features.ndim != 2 or cached_routes.ndim != 1:
        raise ValueError("features must be [samples, dim] and route_labels [samples]")
    if cached_features.shape[0] != cached_routes.shape[0] or cached_features.shape[0] == 0:
        raise ValueError("features and route_labels must have the same non-zero sample count")
    if penalties.ndim != 2 or penalties.shape[0] != penalties.shape[1]:
        raise ValueError("penalty_matrix must be square")
    num_experts = int(penalties.shape[0])
    if torch.any(cached_routes < 0) or torch.any(cached_routes >= num_experts):
        raise ValueError("route label is outside the penalty-matrix class range")

    device = torch.device(settings.device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(settings.seed)
    loader = DataLoader(
        TensorDataset(cached_features, cached_routes),
        batch_size=settings.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(settings.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(settings.seed)
        dispatcher = LinearDispatcher(
            feature_dim=int(cached_features.shape[1]),
            num_experts=num_experts,
        ).to(device)
        optimizer = _optimizer_for(dispatcher, settings)
        weights = class_weights(
            cached_routes,
            num_experts,
            weighting_scheme,
            beta=settings.ens_beta,
            normalize=settings.normalize_class_weights,
        ).to(device)
        penalties = penalties.to(device)

        epoch_losses: list[float] = []
        for _ in range(settings.epochs):
            dispatcher.train()
            total_loss = 0.0
            total_samples = 0
            for batch_features, batch_routes in loader:
                batch_features = batch_features.to(device)
                batch_routes = batch_routes.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = paper_penalty_loss(
                    dispatcher(batch_features),
                    batch_routes,
                    penalties,
                    weights,
                )
                loss.backward()
                optimizer.step()
                batch_samples = int(batch_routes.shape[0])
                total_loss += float(loss.detach()) * batch_samples
                total_samples += batch_samples
            epoch_losses.append(total_loss / total_samples)

    dispatcher.eval()
    return DispatcherTrainingResult(
        dispatcher=dispatcher,
        epoch_losses=tuple(epoch_losses),
        weighting_scheme=weighting_scheme,
        class_weight_values=weights.detach().cpu(),
    )


def predict_routes(
    dispatcher: LinearDispatcher,
    features: Tensor | np.ndarray,
    *,
    batch_size: int = 1024,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Predict route IDs from cached features without expert execution."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    cached_features = _as_tensor(features, dtype=torch.float32)
    if cached_features.ndim != 2:
        raise ValueError("features must be [samples, feature_dim]")
    try:
        model_device = next(dispatcher.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    inference_device = torch.device(device) if device is not None else model_device
    dispatcher = dispatcher.to(inference_device)
    dispatcher.eval()
    routes: list[Tensor] = []
    with torch.no_grad():
        for (batch_features,) in DataLoader(
            TensorDataset(cached_features), batch_size=batch_size, shuffle=False, num_workers=0
        ):
            routes.append(dispatcher(batch_features.to(inference_device)).argmax(dim=1).cpu())
    if not routes:
        return np.empty(0, dtype=np.int64)
    return torch.cat(routes).numpy()


def evaluate_cached_dispatcher(
    dispatcher: LinearDispatcher,
    data: CachedEvaluationData,
    expert_mflops: Tensor | np.ndarray,
    *,
    feature_extractor_index: int,
    feature_extractor_mflops: float,
    router_mflops: float,
    batch_size: int = 1024,
    device: str | torch.device | None = None,
) -> SystemMetrics:
    """Evaluate system accuracy and cost using cached expert predictions."""

    data.validate()
    routed = predict_routes(
        dispatcher,
        data.features,
        batch_size=batch_size,
        device=device,
    )
    return system_metrics(
        routed,
        _as_numpy(data.ideal_routes),
        _as_numpy(data.expert_predictions),
        _as_numpy(data.targets),
        _as_numpy(expert_mflops),
        feature_extractor_index=feature_extractor_index,
        feature_extractor_mflops=feature_extractor_mflops,
        router_mflops=router_mflops,
    )


def evaluate_final_test(
    dispatcher: LinearDispatcher,
    final_test: CachedEvaluationData,
    expert_mflops: Tensor | np.ndarray,
    *,
    feature_extractor_index: int,
    feature_extractor_mflops: float,
    router_mflops: float,
    batch_size: int = 1024,
    device: str | torch.device | None = None,
) -> SystemMetrics:
    """Evaluate a chosen dispatcher on the isolated final test cache.

    This deliberately separate entry point makes it harder for a GA fitness
    problem to accidentally consume the final evaluation data.
    """

    return evaluate_cached_dispatcher(
        dispatcher,
        final_test,
        expert_mflops,
        feature_extractor_index=feature_extractor_index,
        feature_extractor_mflops=feature_extractor_mflops,
        router_mflops=router_mflops,
        batch_size=batch_size,
        device=device,
    )


def load_dispatcher_state(
    feature_dim: int,
    num_experts: int,
    state: Mapping[str, Tensor | np.ndarray],
    *,
    device: str = "cpu",
) -> LinearDispatcher:
    """Rebuild a dispatcher from an evaluation-record snapshot."""

    model = LinearDispatcher(feature_dim, num_experts)
    model.load_state_dict(
        {key: _as_tensor(value, dtype=torch.float32) for key, value in state.items()}
    )
    return model.to(device).eval()
