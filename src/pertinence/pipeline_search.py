"""Artifact-only dispatcher search and isolated, inference-only final evaluation."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np

from .metrics import pareto_mask
from .optimization import DispatcherNSGA2Problem, NSGA2Config, run_nsga2
from .pipeline_artifacts import (
    atomic_directory,
    read_sealed,
    safe_path,
    seal,
    verify_artifact,
    write_json,
)
from .pipeline_cache import load_cache
from .pipeline_validation import load_contract
from .reproducibility import canonical_json_hash, seed_everything, sha256_file
from .training import (
    CachedEvaluationData,
    DispatcherTrainingConfig,
    evaluate_cached_dispatcher,
    load_dispatcher_state,
)


def evaluation_data(cache: dict) -> CachedEvaluationData:
    # Copy read-only memmaps before conversion to writable torch tensors.
    return CachedEvaluationData(
        **{
            key: np.array(cache[source])
            for key, source in (
                ("features", "features"),
                ("expert_predictions", "expert_predictions"),
                ("targets", "targets"),
                ("ideal_routes", "route_labels"),
            )
        }
    )


def cost_context(contract: dict) -> dict:
    routing = contract["config"]["routing"]
    costs = contract["cost"]["vector"]
    index = routing["expert_ids"].index(routing["feature_extractor_id"])
    # The existing core's mflops argument names implement unit-independent arithmetic.
    return {
        "expert_mflops": np.asarray(costs),
        "feature_extractor_index": index,
        "feature_extractor_mflops": costs[index],
        "router_mflops": contract["cost"]["router_cost"],
    }


def metrics_dict(metrics, contract: dict) -> dict:
    confusion = metrics.confusion_matrix
    return {
        "system_accuracy": metrics.system_accuracy,
        "route_accuracy": metrics.route_accuracy,
        "average_cost": metrics.average_mflops,
        "cost_unit": contract["cost"]["unit"],
        "confusion_matrix": confusion.tolist(),
        "selected_counts": confusion.sum(axis=0).tolist(),
        "underestimation_count": int(np.tril(confusion, k=-1).sum()),
        "overestimation_count": int(np.triu(confusion, k=1).sum()),
    }


def train_dispatcher(
    train_cache: Path,
    search_cache: Path,
    normalized_contract: Path,
    *,
    search_result: Path,
    pareto_bundle: Path,
) -> None:
    """No final artifact, URI, or shared cache directory is accepted by this API."""
    contract = load_contract(normalized_contract)
    train, train_meta = load_cache(train_cache, contract, "train")
    search, search_meta = load_cache(search_cache, contract, "search")
    if set(train["sample_ids"]) & set(search["sample_ids"]):
        raise ValueError("training/search cache sample IDs overlap")
    raw = contract["config"]
    seed_everything(raw["seed"])
    settings = DispatcherTrainingConfig(
        **{
            key: value
            for key, value in raw["dispatcher"].items()
            if key != "normalize_class_weights"
        },
        normalize_class_weights=True,
        seed=raw["seed"],
        device=contract["platform"]["device"],
    )
    ga = NSGA2Config(
        **{
            key: tuple(value) if key == "weighting_schemes" else value
            for key, value in raw["nsga2"].items()
            if key not in {"crossover", "mutation"}
        }
    )
    problem = DispatcherNSGA2Problem(
        np.array(train["features"]),
        np.array(train["route_labels"]),
        evaluation_data(search),
        **cost_context(contract),
        training_config=settings,
        weighting_schemes=ga.weighting_schemes,
        penalty_min=ga.penalty_min,
        penalty_max=ga.penalty_max,
        keep_dispatcher_states=True,
    )
    run = run_nsga2(problem, config=ga, seed=raw["seed"])
    if len(run.evaluations) != contract["dimensions"]["evaluation_count"]:
        raise RuntimeError("NSGA-II evaluation count differs from normalized contract")
    pareto_ids = [record.evaluation_id for record in run.pareto_evaluations]
    front_id = canonical_json_hash({"contract": contract["fingerprint"], "evaluations": pareto_ids})
    cache_fingerprints = {"train": train_meta["fingerprint"], "search": search_meta["fingerprint"]}
    result = seal(
        {
            "schema_version": 1,
            "kind": "search-result",
            "contract_fingerprint": contract["fingerprint"],
            "cache_fingerprints": cache_fingerprints,
            "front_id": front_id,
            "pareto_ids": pareto_ids,
            "evaluations": [
                {
                    "evaluation_id": record.evaluation_id,
                    "chromosome": record.chromosome.tolist(),
                    "penalty_matrix": record.penalty_matrix.tolist(),
                    "weighting": record.weighting_scheme,
                    "metrics": metrics_dict(record.metrics, contract),
                    "objectives": list(record.objectives),
                    "epoch_losses": list(record.epoch_losses),
                }
                for record in run.evaluations
            ],
        }
    )
    with atomic_directory(pareto_bundle) as stage:
        models = []
        for record in run.pareto_evaluations:
            model_id = f"dispatcher-{record.evaluation_id:06d}"
            path = stage / f"{model_id}.npz"
            np.savez(path, **record.dispatcher_state)
            models.append(
                {
                    "model_id": model_id,
                    "evaluation_id": record.evaluation_id,
                    "checkpoint": path.name,
                    "checkpoint_sha256": sha256_file(path),
                    "dimensions": {key: contract["dimensions"][key] for key in ("D", "K")},
                    "expert_order": raw["routing"]["expert_ids"],
                    "cache_fingerprints": cache_fingerprints,
                }
            )
        write_json(
            stage / "manifest.json",
            seal(
                {
                    "schema_version": 1,
                    "kind": "pareto-dispatchers",
                    "contract_fingerprint": contract["fingerprint"],
                    "front_id": front_id,
                    "search_fingerprint": result["fingerprint"],
                    "models": models,
                }
            ),
        )
    write_json(search_result, result)


def validate_search(result: dict, contract: dict) -> dict[int, dict]:
    if (
        result.get("kind") != "search-result"
        or result.get("schema_version") != 1
        or result["contract_fingerprint"] != contract["fingerprint"]
    ):
        raise ValueError("search result contract mismatch")
    records = result["evaluations"]
    ids = [record["evaluation_id"] for record in records]
    if ids != list(range(contract["dimensions"]["evaluation_count"])):
        raise ValueError("search evaluation IDs/count mismatch")
    accuracy = np.asarray([record["metrics"]["system_accuracy"] for record in records])
    costs = np.asarray([record["metrics"]["average_cost"] for record in records])
    if (
        not np.isfinite(accuracy).all()
        or not np.isfinite(costs).all()
        or np.any(accuracy < 0)
        or np.any(accuracy > 1)
        or np.any(costs < 0)
    ):
        raise ValueError("invalid search metrics")
    expected = [item for item, keep in zip(ids, pareto_mask(accuracy, costs)) if keep]
    if result["pareto_ids"] != expected:
        raise ValueError("search Pareto IDs differ from metrics")
    front_id = canonical_json_hash({"contract": contract["fingerprint"], "evaluations": expected})
    if result["front_id"] != front_id:
        raise ValueError("search Pareto front fingerprint mismatch")
    return {record["evaluation_id"]: record for record in records}


def validate_models(root: Path, manifest: dict, result: dict, contract: dict) -> None:
    if (
        manifest.get("schema_version") != 1
        or manifest["contract_fingerprint"] != contract["fingerprint"]
        or manifest["search_fingerprint"] != result["fingerprint"]
        or manifest["front_id"] != result["front_id"]
    ):
        raise ValueError("checkpoint bundle lineage mismatch")
    models = manifest["models"]
    if [model["evaluation_id"] for model in models] != result["pareto_ids"]:
        raise ValueError("checkpoint IDs differ from Search Pareto IDs")
    if len({model["model_id"] for model in models}) != len(models):
        raise ValueError("duplicate checkpoint model IDs")
    for model in models:
        if (
            model["dimensions"] != {key: contract["dimensions"][key] for key in ("D", "K")}
            or model["expert_order"] != contract["config"]["routing"]["expert_ids"]
            or model["cache_fingerprints"] != result["cache_fingerprints"]
        ):
            raise ValueError("checkpoint dimensions, expert order, or cache fingerprint mismatch")
        path = safe_path(root, model["checkpoint"])
        verify_artifact(path, model["checkpoint_sha256"])
        with np.load(path, allow_pickle=False) as state:
            d, k = contract["dimensions"]["D"], contract["dimensions"]["K"]
            if set(state.files) != {"linear.weight", "linear.bias"}:
                raise ValueError("checkpoint state keys mismatch")
            for key, shape in (("linear.weight", (k, d)), ("linear.bias", (k,))):
                if (
                    state[key].shape != shape
                    or state[key].dtype != np.float32
                    or not np.isfinite(state[key]).all()
                ):
                    raise ValueError("checkpoint tensor shape/dtype/finiteness mismatch")


def evaluate_final(
    final_cache: Path,
    search_result: Path,
    pareto_bundle: Path,
    normalized_contract: Path,
    *,
    final_report: Path,
    evaluated_bundle: Path,
) -> None:
    contract = load_contract(normalized_contract)
    cache, cache_meta = load_cache(final_cache, contract, "final")
    result = read_sealed(search_result)
    records = validate_search(result, contract)
    bundle = read_sealed(pareto_bundle / "manifest.json")
    if bundle.get("kind") != "pareto-dispatchers":
        raise ValueError("expected Search Pareto dispatcher bundle")
    validate_models(pareto_bundle, bundle, result, contract)
    data = evaluation_data(cache)
    solutions = []
    device = contract["platform"]["device"]
    for model in bundle["models"]:
        with np.load(safe_path(pareto_bundle, model["checkpoint"]), allow_pickle=False) as state:
            dispatcher = load_dispatcher_state(
                contract["dimensions"]["D"], contract["dimensions"]["K"], dict(state), device=device
            )
        metrics = evaluate_cached_dispatcher(
            dispatcher,
            data,
            **cost_context(contract),
            device=device,
            batch_size=contract["config"]["dispatcher"]["batch_size"],
        )
        solutions.append(
            {
                "model_id": model["model_id"],
                "evaluation_id": model["evaluation_id"],
                "checkpoint_sha256": model["checkpoint_sha256"],
                "search_metrics": records[model["evaluation_id"]]["metrics"],
                "final_metrics": metrics_dict(metrics, contract),
            }
        )
    report = seal(
        {
            "schema_version": 1,
            "kind": "final-report",
            "contract_fingerprint": contract["fingerprint"],
            "search_fingerprint": result["fingerprint"],
            "front_id": result["front_id"],
            "final_cache_fingerprint": cache_meta["fingerprint"],
            "solutions": solutions,
        }
    )
    with atomic_directory(evaluated_bundle) as stage:
        for model in bundle["models"]:
            target = safe_path(stage, model["checkpoint"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(pareto_bundle, model["checkpoint"]), target)
        write_json(stage / "contract.json", contract)
        write_json(stage / "search-result.json", result)
        write_json(stage / "final-report.json", report)
        write_json(
            stage / "manifest.json",
            seal(
                {
                    **{key: value for key, value in bundle.items() if key != "fingerprint"},
                    "kind": "evaluated-dispatchers",
                    "status": "evaluated",
                    "final_report_fingerprint": report["fingerprint"],
                }
            ),
        )
    write_json(final_report, report)
