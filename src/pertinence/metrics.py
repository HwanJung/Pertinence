"""System-level metrics and Pareto filtering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SystemMetrics:
    system_accuracy: float
    route_accuracy: float
    average_mflops: float
    confusion_matrix: np.ndarray


def system_metrics(
    routed_experts: np.ndarray,
    ideal_routes: np.ndarray,
    expert_predictions: np.ndarray,
    targets: np.ndarray,
    expert_mflops: np.ndarray,
    *,
    feature_extractor_index: int,
    feature_extractor_mflops: float,
    router_mflops: float,
) -> SystemMetrics:
    routed = np.asarray(routed_experts, dtype=np.int64)
    ideal = np.asarray(ideal_routes, dtype=np.int64)
    predictions = np.asarray(expert_predictions)
    labels = np.asarray(targets)
    costs = np.asarray(expert_mflops, dtype=np.float64)
    n = labels.shape[0]
    if routed.shape != (n,) or ideal.shape != (n,) or predictions.shape[0] != n:
        raise ValueError("sample dimensions do not match")
    if predictions.shape[1] != costs.shape[0]:
        raise ValueError("expert dimensions do not match")
    selected_predictions = predictions[np.arange(n), routed]
    system_accuracy = float(np.mean(selected_predictions == labels))
    route_accuracy = float(np.mean(routed == ideal))
    # Dispatcher always pays for its backbone. If that same model is selected,
    # its already-computed classification output is reused.
    selected_cost = costs[routed].copy()
    selected_cost[routed == feature_extractor_index] = 0.0
    average_mflops = float(
        feature_extractor_mflops + router_mflops + np.mean(selected_cost)
    )
    num_experts = predictions.shape[1]
    confusion = np.zeros((num_experts, num_experts), dtype=np.int64)
    np.add.at(confusion, (ideal, routed), 1)
    return SystemMetrics(system_accuracy, route_accuracy, average_mflops, confusion)


def pareto_mask(accuracy: np.ndarray, cost: np.ndarray) -> np.ndarray:
    accuracy = np.asarray(accuracy, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    if accuracy.shape != cost.shape or accuracy.ndim != 1:
        raise ValueError("accuracy and cost must be same-length vectors")
    keep = np.ones(len(accuracy), dtype=bool)
    for index in range(len(accuracy)):
        dominates = (
            (accuracy >= accuracy[index])
            & (cost <= cost[index])
            & ((accuracy > accuracy[index]) | (cost < cost[index]))
        )
        dominates[index] = False
        keep[index] = not np.any(dominates)
    return keep

