"""Post-hoc analysis of completed search and untouched-final artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from .metrics import pareto_mask


def _interpolated_catalog_accuracy(
    catalog_rows: Sequence[Mapping[str, Any]], cost: float
) -> float | None:
    """Return the plotted catalog boundary at ``cost`` on the log-MFLOPs axis."""

    if not catalog_rows:
        return None
    if cost < float(catalog_rows[0]["standalone_mflops"]) or cost > float(
        catalog_rows[-1]["standalone_mflops"]
    ):
        return None
    if len(catalog_rows) == 1:
        only_cost = float(catalog_rows[0]["standalone_mflops"])
        return (
            float(catalog_rows[0]["catalog_accuracy_pct"])
            if math.isclose(cost, only_cost)
            else None
        )
    for lower, upper in zip(catalog_rows, catalog_rows[1:]):
        lower_cost = float(lower["standalone_mflops"])
        upper_cost = float(upper["standalone_mflops"])
        if lower_cost <= cost <= upper_cost:
            log_fraction = (math.log10(cost) - math.log10(lower_cost)) / (
                math.log10(upper_cost) - math.log10(lower_cost)
            )
            lower_accuracy = float(lower["catalog_accuracy_pct"])
            upper_accuracy = float(upper["catalog_accuracy_pct"])
            return lower_accuracy + log_fraction * (upper_accuracy - lower_accuracy)
    return None


def _validated_metrics(
    metrics: Mapping[str, Any], *, samples: int, num_experts: int
) -> tuple[float, float, float, np.ndarray]:
    accuracy = float(metrics["system_accuracy"])
    route_accuracy = float(metrics["route_accuracy"])
    cost = float(metrics["average_mflops"])
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    if not all(math.isfinite(value) for value in (accuracy, route_accuracy, cost)):
        raise ValueError("solution metrics contain a non-finite value")
    if not 0.0 <= accuracy <= 1.0 or not 0.0 <= route_accuracy <= 1.0 or cost <= 0.0:
        raise ValueError("solution metrics are outside their valid ranges")
    if confusion.shape != (num_experts, num_experts) or int(confusion.sum()) != samples:
        raise ValueError("solution confusion matrix has an invalid shape or sample count")
    selected = np.asarray(metrics["selected_counts"], dtype=np.int64)
    if selected.shape != (num_experts,) or not np.array_equal(
        selected, confusion.sum(axis=0)
    ):
        raise ValueError("selected_counts differ from the confusion matrix")
    return accuracy, route_accuracy, cost, selected


def analyze_payloads(
    search: Mapping[str, Any],
    final: Mapping[str, Any],
    *,
    candidate_names: Sequence[str],
    candidate_mflops: Sequence[float],
    static_pareto_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate, join, and summarize search/final records without reselection."""

    names = tuple(candidate_names)
    costs = np.asarray(candidate_mflops, dtype=np.float64)
    if not names or len(set(names)) != len(names) or costs.shape != (len(names),):
        raise ValueError("candidate names and costs are invalid")
    if search.get("kind") != "nsga2_search" or final.get("kind") != "final_evaluation":
        raise ValueError("unexpected search or final artifact kind")
    expected_final_fingerprints = dict(search["cache_fingerprints"])
    actual_final_fingerprints = dict(final["cache_fingerprints"])
    if any(
        actual_final_fingerprints.get(name) != fingerprint
        for name, fingerprint in expected_final_fingerprints.items()
    ):
        raise ValueError("search and final cache fingerprints differ")

    evaluations = list(search["evaluations"])
    by_id = {int(item["evaluation_id"]): item for item in evaluations}
    if len(by_id) != len(evaluations):
        raise ValueError("search evaluation IDs are not unique")
    retained = {identifier: item for identifier, item in by_id.items() if item["pareto"]}
    solutions = list(final["solutions"])
    final_ids = [int(item["search_evaluation_id"]) for item in solutions]
    if len(set(final_ids)) != len(final_ids) or set(final_ids) != set(retained):
        raise ValueError("final solutions are not exactly the retained search front")

    baselines = final["final_baselines_and_oracle"]
    samples = int(baselines["samples"])
    if samples <= 0 or set(baselines["expert_top1"]) != set(names):
        raise ValueError("final baseline metadata differs from the candidate pool")

    rows: list[dict[str, Any]] = []
    for solution in solutions:
        identifier = int(solution["search_evaluation_id"])
        search_record = retained[identifier]
        if (
            solution["chromosome"] != search_record["chromosome"]
            or solution["weighting"] != search_record["weighting"]
            or not search_record.get("dispatcher_state")
        ):
            raise ValueError(f"final solution {identifier} differs from its search record")
        final_accuracy, route_accuracy, final_cost, selected = _validated_metrics(
            solution["metrics"], samples=samples, num_experts=len(names)
        )
        search_metrics = search_record["metrics"]
        search_accuracy = float(search_metrics["system_accuracy"])
        search_cost = float(search_metrics["average_mflops"])
        if not all(math.isfinite(value) for value in (search_accuracy, search_cost)):
            raise ValueError("search metrics contain a non-finite value")
        row: dict[str, Any] = {
            "search_evaluation_id": identifier,
            "weighting": str(solution["weighting"]),
            "search_accuracy_pct": 100.0 * search_accuracy,
            "final_accuracy_pct": 100.0 * final_accuracy,
            "accuracy_delta_pp": 100.0 * (final_accuracy - search_accuracy),
            "search_mflops": search_cost,
            "final_mflops": final_cost,
            "final_route_accuracy_pct": 100.0 * route_accuracy,
            "underestimation_count": int(solution["metrics"]["underestimation_count"]),
            "overestimation_count": int(solution["metrics"]["overestimation_count"]),
            "dispatcher_checkpoint": str(solution["dispatcher_checkpoint"]),
            "chromosome": list(solution["chromosome"]),
        }
        for index, name in enumerate(names):
            row[f"selected_{name}_count"] = int(selected[index])
            row[f"selected_{name}_fraction"] = float(selected[index] / samples)
        rows.append(row)

    accuracy = np.asarray([row["final_accuracy_pct"] for row in rows])
    final_cost = np.asarray([row["final_mflops"] for row in rows])
    keep = pareto_mask(accuracy, final_cost)
    for row, is_pareto in zip(rows, keep):
        row["final_pareto"] = bool(is_pareto)

    front_rows = [row for row in rows if row["final_pareto"]]
    unique_front = {
        (round(float(row["final_mflops"]), 9), round(float(row["final_accuracy_pct"]), 9))
        for row in front_rows
    }
    highest_accuracy = min(
        (row for row in rows if row["final_accuracy_pct"] == float(accuracy.max())),
        key=lambda row: (row["final_mflops"], row["search_evaluation_id"]),
    )
    lowest_cost = max(
        (row for row in rows if row["final_mflops"] == float(final_cost.min())),
        key=lambda row: (row["final_accuracy_pct"], -row["search_evaluation_id"]),
    )

    baseline_comparisons: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        baseline_accuracy = 100.0 * float(baselines["expert_top1"][name])
        baseline_cost = float(costs[index])
        dominators = [
            row
            for row in rows
            if row["final_accuracy_pct"] >= baseline_accuracy
            and row["final_mflops"] <= baseline_cost
            and (
                row["final_accuracy_pct"] > baseline_accuracy
                or row["final_mflops"] < baseline_cost
            )
        ]
        best = (
            min(
                dominators,
                key=lambda row: (
                    row["final_mflops"],
                    -row["final_accuracy_pct"],
                    row["search_evaluation_id"],
                ),
            )
            if dominators
            else None
        )
        baseline_comparisons.append(
            {
                "model": name,
                "final_accuracy_pct": baseline_accuracy,
                "standalone_mflops": baseline_cost,
                "dynamic_dominator_count": len(dominators),
                "lowest_cost_dynamic_dominator": (
                    None
                    if best is None
                    else {
                        "search_evaluation_id": best["search_evaluation_id"],
                        "final_accuracy_pct": best["final_accuracy_pct"],
                        "final_mflops": best["final_mflops"],
                    }
                ),
            }
        )

    catalog_rows = []
    for item in static_pareto_rows:
        catalog_rows.append(
            {
                "model": str(item["model"]),
                "catalog_accuracy_pct": float(item["top1_accuracy_pct"]),
                "standalone_mflops": float(item["mflops"]),
            }
        )
    catalog_rows.sort(key=lambda item: item["standalone_mflops"])
    if any(
        not math.isfinite(item["catalog_accuracy_pct"])
        or not math.isfinite(item["standalone_mflops"])
        or item["standalone_mflops"] <= 0.0
        for item in catalog_rows
    ):
        raise ValueError("static catalog Pareto rows contain invalid values")
    if any(
        lower["standalone_mflops"] >= upper["standalone_mflops"]
        for lower, upper in zip(catalog_rows, catalog_rows[1:])
    ):
        raise ValueError("static catalog Pareto costs must be strictly increasing")

    for row in rows:
        interpolated_accuracy = _interpolated_catalog_accuracy(
            catalog_rows, float(row["final_mflops"])
        )
        margin = (
            None
            if interpolated_accuracy is None
            else float(row["final_accuracy_pct"]) - interpolated_accuracy
        )
        row["catalog_interpolated_accuracy_pct"] = interpolated_accuracy
        row["catalog_interpolated_margin_pp"] = margin
        row["above_catalog_interpolated_front"] = bool(
            margin is not None and margin > 0.0
        )

    above_catalog_rows = [
        row for row in rows if row["above_catalog_interpolated_front"]
    ]
    above_catalog_final_front_rows = [
        row
        for row in above_catalog_rows
        if row["final_pareto"]
    ]
    unique_above_catalog: dict[tuple[float, float], dict[str, Any]] = {}
    for row in above_catalog_final_front_rows:
        key = (
            round(float(row["final_mflops"]), 9),
            round(float(row["final_accuracy_pct"]), 9),
        )
        previous = unique_above_catalog.get(key)
        if previous is None or int(row["search_evaluation_id"]) < int(
            previous["search_evaluation_id"]
        ):
            unique_above_catalog[key] = row
    above_catalog_representatives = sorted(
        unique_above_catalog.values(), key=lambda row: float(row["final_mflops"])
    )

    oracle = baselines["oracle"]
    summary = {
        "schema_version": 1,
        "kind": "pertinence_final_analysis",
        "cache_fingerprints": actual_final_fingerprints,
        "search_evaluations": len(evaluations),
        "search_pareto_solutions": len(retained),
        "final_solutions": len(rows),
        "final_nondominated_solutions": len(front_rows),
        "final_unique_nondominated_objectives": len(unique_front),
        "final_accuracy_pct_range": [float(accuracy.min()), float(accuracy.max())],
        "final_mflops_range": [float(final_cost.min()), float(final_cost.max())],
        "accuracy_delta_pp_range": [
            min(float(row["accuracy_delta_pp"]) for row in rows),
            max(float(row["accuracy_delta_pp"]) for row in rows),
        ],
        "mean_accuracy_delta_pp": float(
            np.mean([row["accuracy_delta_pp"] for row in rows])
        ),
        "highest_accuracy_observation": highest_accuracy,
        "lowest_cost_observation": lowest_cost,
        "final_oracle": {
            "accuracy_pct": 100.0 * float(oracle["system_accuracy"]),
            "average_mflops": float(oracle["average_mflops"]),
            "no_correct_samples": int(baselines["no_correct_samples"]),
        },
        "same_split_expert_baselines": baseline_comparisons,
        "catalog_static_pareto": catalog_rows,
        "catalog_interpolated_front_comparison": {
            "interpolation": "linear_accuracy_vs_log10_mflops",
            "accuracy_sources_comparable": False,
            "above_all_solution_count": len(above_catalog_rows),
            "above_final_nondominated_solution_count": len(
                above_catalog_final_front_rows
            ),
            "above_unique_final_nondominated_objectives": len(
                above_catalog_representatives
            ),
            "representative_solutions": [
                {
                    "search_evaluation_id": row["search_evaluation_id"],
                    "weighting": row["weighting"],
                    "final_accuracy_pct": row["final_accuracy_pct"],
                    "final_mflops": row["final_mflops"],
                    "catalog_interpolated_accuracy_pct": row[
                        "catalog_interpolated_accuracy_pct"
                    ],
                    "catalog_interpolated_margin_pp": row[
                        "catalog_interpolated_margin_pp"
                    ],
                    "dispatcher_checkpoint": row["dispatcher_checkpoint"],
                }
                for row in above_catalog_representatives
            ],
        },
    }
    return {"summary": summary, "rows": rows}
