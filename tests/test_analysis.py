from __future__ import annotations

from copy import deepcopy
import math

import pytest

from pertinence.analysis import analyze_payloads


def _metrics(accuracy: float, cost: float, confusion: list[list[int]]) -> dict[str, object]:
    selected = [sum(row[column] for row in confusion) for column in range(2)]
    return {
        "system_accuracy": accuracy,
        "route_accuracy": 0.5,
        "average_mflops": cost,
        "confusion_matrix": confusion,
        "selected_counts": selected,
        "underestimation_count": 1,
        "overestimation_count": 1,
    }


def _payloads() -> tuple[dict[str, object], dict[str, object]]:
    chromosomes = [[1.0, 2.0], [3.0, 4.0]]
    search = {
        "kind": "nsga2_search",
        "cache_fingerprints": {"dispatcher_train": "a", "ga_search": "b"},
        "evaluations": [
            {
                "evaluation_id": identifier,
                "pareto": True,
                "chromosome": chromosomes[identifier],
                "weighting": "INS",
                "dispatcher_state": {"linear.weight": [[1.0]]},
                "metrics": {
                    "system_accuracy": accuracy,
                    "average_mflops": cost,
                },
            }
            for identifier, accuracy, cost in ((0, 0.79, 10.0), (1, 0.88, 15.0))
        ],
    }
    final = {
        "kind": "final_evaluation",
        "cache_fingerprints": {
            "dispatcher_train": "a",
            "ga_search": "b",
            "final_evaluation": "c",
        },
        "final_baselines_and_oracle": {
            "samples": 4,
            "expert_top1": {"a": 0.75, "b": 0.85},
            "no_correct_samples": 1,
            "oracle": {"system_accuracy": 0.95, "average_mflops": 8.0},
        },
        "solutions": [
            {
                "search_evaluation_id": identifier,
                "chromosome": chromosomes[identifier],
                "weighting": "INS",
                "dispatcher_checkpoint": f"dispatcher-{identifier}.pt",
                "metrics": _metrics(accuracy, cost, [[1, 1], [1, 1]]),
            }
            for identifier, accuracy, cost in ((0, 0.80, 10.0), (1, 0.90, 15.0))
        ],
    }
    return search, final


def test_analysis_joins_exact_search_front_and_computes_final_front() -> None:
    search, final = _payloads()
    analysis = analyze_payloads(
        search,
        final,
        candidate_names=("a", "b"),
        candidate_mflops=(5.0, 20.0),
        static_pareto_rows=(
            {"model": "static", "top1_accuracy_pct": 91.0, "mflops": 30.0},
        ),
    )

    summary = analysis["summary"]
    assert summary["search_evaluations"] == 2
    assert summary["final_solutions"] == 2
    assert summary["final_unique_nondominated_objectives"] == 2
    assert summary["highest_accuracy_observation"]["search_evaluation_id"] == 1
    assert summary["same_split_expert_baselines"][1]["dynamic_dominator_count"] == 1
    assert all(row["final_pareto"] for row in analysis["rows"])


def test_analysis_marks_solutions_above_log_interpolated_catalog_front() -> None:
    search, final = _payloads()
    analysis = analyze_payloads(
        search,
        final,
        candidate_names=("a", "b"),
        candidate_mflops=(5.0, 20.0),
        static_pareto_rows=(
            {"model": "small", "top1_accuracy_pct": 70.0, "mflops": 10.0},
            {"model": "large", "top1_accuracy_pct": 90.0, "mflops": 100.0},
        ),
    )

    first, second = analysis["rows"]
    assert first["catalog_interpolated_accuracy_pct"] == pytest.approx(70.0)
    assert first["catalog_interpolated_margin_pp"] == pytest.approx(10.0)
    assert first["above_catalog_interpolated_front"] is True
    assert second["catalog_interpolated_accuracy_pct"] == pytest.approx(
        70.0 + 20.0 * math.log10(1.5)
    )
    assert second["above_catalog_interpolated_front"] is True

    comparison = analysis["summary"]["catalog_interpolated_front_comparison"]
    assert comparison["interpolation"] == "linear_accuracy_vs_log10_mflops"
    assert comparison["accuracy_sources_comparable"] is False
    assert comparison["above_all_solution_count"] == 2
    assert comparison["above_final_nondominated_solution_count"] == 2
    assert comparison["above_unique_final_nondominated_objectives"] == 2
    assert [
        row["search_evaluation_id"] for row in comparison["representative_solutions"]
    ] == [0, 1]


def test_analysis_rejects_mismatched_final_fingerprint() -> None:
    search, final = _payloads()
    changed = deepcopy(final)
    changed["cache_fingerprints"]["ga_search"] = "wrong"  # type: ignore[index]

    with pytest.raises(ValueError, match="fingerprints differ"):
        analyze_payloads(
            search,
            changed,
            candidate_names=("a", "b"),
            candidate_mflops=(5.0, 20.0),
            static_pareto_rows=(),
        )
