"""Ideal labels, imbalance weighting, paper loss, and linear dispatcher."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


WeightingScheme = Literal["INS", "ISNS", "ENS"]


def cheapest_correct_labels(predictions: Tensor, targets: Tensor) -> Tensor:
    """Return cheapest correct expert indices; fall back to expert zero.

    ``predictions`` is shaped ``[samples, experts]`` and experts must already be
    sorted by ascending compute cost.
    """
    if predictions.ndim != 2 or targets.ndim != 1:
        raise ValueError("expected predictions [samples, experts] and targets [samples]")
    if predictions.shape[0] != targets.shape[0] or predictions.shape[1] == 0:
        raise ValueError("prediction and target shapes are incompatible")
    correct = predictions.eq(targets[:, None])
    # argmax returns zero both for first-correct and all-false, which implements
    # the paper's cheapest-model fallback without a special branch.
    return correct.to(torch.int64).argmax(dim=1)


def class_weights(
    route_labels: Tensor,
    num_classes: int,
    scheme: WeightingScheme,
    *,
    beta: float = 0.9999,
    normalize: bool = True,
) -> Tensor:
    counts = torch.bincount(route_labels, minlength=num_classes).to(torch.float64)
    if torch.any(counts == 0):
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"routing classes have no training samples: {missing}")
    if scheme == "INS":
        weights = counts.reciprocal()
    elif scheme == "ISNS":
        weights = counts.rsqrt()
    elif scheme == "ENS":
        if not 0.0 <= beta < 1.0:
            raise ValueError("ENS beta must satisfy 0 <= beta < 1")
        weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    else:
        raise ValueError(f"unknown weighting scheme: {scheme}")
    if normalize:
        weights = weights / weights.mean()
    return weights.to(dtype=torch.float32, device=route_labels.device)


def paper_penalty_loss(
    logits: Tensor,
    route_labels: Tensor,
    penalty_matrix: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    """Implement PERTINENCE Eq. (6)-(7), including its argmax semantics."""
    classes = logits.shape[1]
    if penalty_matrix.shape != (classes, classes):
        raise ValueError("penalty matrix must be [classes, classes]")
    if weights is not None and weights.shape != (classes,):
        raise ValueError("class weights must have one value per route class")
    predicted = logits.argmax(dim=1)
    base = F.cross_entropy(logits, route_labels, reduction="none")
    penalties = penalty_matrix.to(logits.device)[route_labels, predicted]
    sample_weights: Tensor | float = 1.0
    if weights is not None:
        sample_weights = weights.to(logits.device)[route_labels]
    return (base * penalties * sample_weights).mean()


def directional_penalty_matrix(
    num_classes: int,
    underestimation: float,
    overestimation: float,
) -> Tensor:
    """Build penalties for classes ordered from cheapest to most expensive."""
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    for true_class in range(num_classes):
        for predicted_class in range(num_classes):
            if predicted_class < true_class:
                matrix[true_class, predicted_class] = underestimation
            elif predicted_class > true_class:
                matrix[true_class, predicted_class] = overestimation
    return matrix


class LinearDispatcher(nn.Module):
    def __init__(self, feature_dim: int, num_experts: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_experts)

    def forward(self, features: Tensor) -> Tensor:
        return self.linear(features)

