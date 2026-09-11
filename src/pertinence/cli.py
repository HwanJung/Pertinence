"""Run the cached fixed, NSGA-II search, and isolated final stages."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from .config import ExperimentConfig, load_config
from .cache import expected_cache_fingerprint, experiment_config_fingerprint
from .data import make_splits
from .experts import EXPERTS_BY_NAME
from .metrics import SystemMetrics, system_metrics
from .optimization import (
    DispatcherNSGA2Problem,
    NSGA2Config,
    run_nsga2,
)
from .routing import directional_penalty_matrix
from .reproducibility import canonical_json_hash, seed_everything
from .training import (
    CachedEvaluationData,
    DispatcherTrainingConfig,
    evaluate_cached_dispatcher,
    evaluate_final_test,
    load_dispatcher_state,
    train_linear_dispatcher,
)


def _load_cache(path: Path, config: ExperimentConfig, expected_split: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required cache does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid cache payload: {path}")
    if payload.get("split") != expected_split:
        raise ValueError(f"expected {expected_split!r} cache at {path}")
    if tuple(payload.get("model_order", ())) != config.candidates:
        raise ValueError(f"cache model order differs from config: {path}")
    if payload.get("feature_extractor") != config.feature_extractor:
        raise ValueError(f"cache feature extractor differs from config: {path}")
    required = {
        "asset_fingerprint",
        "config_fingerprint",
        "expert_predictions",
        "features",
        "fingerprint",
        "route_labels",
        "sample_indices",
        "split_fingerprint",
        "targets",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"cache is missing fields {sorted(missing)}: {path}")
    splits = make_splits(
        int(config.raw["data"]["ga_search_size"]),
        int(config.raw["data"]["final_evaluation_size"]),
        int(config.raw["data"]["split_seed"]),
    )
    expected_indices = getattr(splits, expected_split)
    actual_indices = payload.get("sample_indices")
    if actual_indices is None or not np.array_equal(actual_indices.numpy(), expected_indices):
        raise ValueError(f"cache sample indices differ from configured split: {path}")
    expected_config_fingerprint = experiment_config_fingerprint(config)
    if payload.get("config_fingerprint") != expected_config_fingerprint:
        raise ValueError(f"cache config fingerprint differs from config: {path}")
    split_fingerprint = canonical_json_hash(
        {
            "name": expected_split,
            "indices": expected_indices.tolist(),
            "split_manifest_fingerprint": splits.fingerprint,
        }
    )
    recomputed = expected_cache_fingerprint(
        split=expected_split,
        split_fingerprint=split_fingerprint,
        model_order=config.candidates,
        feature_extractor=config.feature_extractor,
        asset_fingerprint=str(payload.get("asset_fingerprint")),
        config_fingerprint=expected_config_fingerprint,
    )
    if payload.get("split_fingerprint") != split_fingerprint or payload.get("fingerprint") != recomputed:
        raise ValueError(f"cache fingerprint cannot be reproduced: {path}")
    return payload


def _evaluation_data(payload: Mapping[str, Any]) -> CachedEvaluationData:
    value = CachedEvaluationData(
        features=payload["features"],
        expert_predictions=payload["expert_predictions"],
        targets=payload["targets"],
        ideal_routes=payload["route_labels"],
    )
    value.validate()
    return value


def _training_config(config: ExperimentConfig) -> DispatcherTrainingConfig:
    raw = config.raw["dispatcher"]
    normalization = raw["normalize_class_weights"]
    if normalization != "mean_one":
        raise ValueError("only mean_one class-weight normalization is implemented")
    return DispatcherTrainingConfig(
        epochs=int(raw["epochs"]),
        batch_size=int(raw["batch_size"]),
        learning_rate=float(raw["learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        optimizer=str(raw["optimizer"]),
        ens_beta=float(raw["ens_beta"]),
        normalize_class_weights=True,
        seed=config.seed,
        device=str(config.raw["experiment"]["device"]),
    )


def _cost_context(config: ExperimentConfig, feature_dim: int) -> dict[str, Any]:
    costs = np.asarray([EXPERTS_BY_NAME[name].mflops for name in config.candidates])
    extractor_index = config.candidates.index(config.feature_extractor)
    # A Linear(in, out) performs in*out MACs. Bias bookkeeping is omitted to
    # match THOP's conventional Linear MAC accounting.
    router_mflops = 2.0 * feature_dim * len(config.candidates) / 1_000_000.0
    return {
        "expert_mflops": costs,
        "feature_extractor_index": extractor_index,
        "feature_extractor_mflops": float(costs[extractor_index]),
        "router_mflops": router_mflops,
    }


def _metrics_dict(metrics: SystemMetrics) -> dict[str, Any]:
    confusion = metrics.confusion_matrix
    return {
        "system_accuracy": metrics.system_accuracy,
        "route_accuracy": metrics.route_accuracy,
        "average_mflops": metrics.average_mflops,
        "confusion_matrix": confusion.tolist(),
        "selected_counts": confusion.sum(axis=0).tolist(),
        "underestimation_count": int(np.tril(confusion, k=-1).sum()),
        "overestimation_count": int(np.triu(confusion, k=1).sum()),
    }


def _write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cache_pair(cache_directory: Path, config: ExperimentConfig):
    train = _load_cache(cache_directory / "dispatcher_train.pt", config, "dispatcher_train")
    search = _load_cache(cache_directory / "ga_search.pt", config, "ga_search")
    if train["asset_fingerprint"] != search["asset_fingerprint"]:
        raise ValueError("training and search caches use different assets")
    fingerprints = {
        "dispatcher_train": train["fingerprint"],
        "ga_search": search["fingerprint"],
    }
    return train, search, fingerprints


def _cache_set(cache_directory: Path, config: ExperimentConfig):
    train, search, fingerprints = _cache_pair(cache_directory, config)
    final = _load_cache(
        cache_directory / "final_evaluation.pt", config, "final_evaluation"
    )
    if train["asset_fingerprint"] != final["asset_fingerprint"]:
        raise ValueError("training and final caches use different assets")
    fingerprints["final_evaluation"] = final["fingerprint"]
    return train, search, final, fingerprints


def _summarize_baseline(
    config: ExperimentConfig, payload: Mapping[str, Any], costs: Mapping[str, Any]
) -> dict[str, Any]:
    predictions = payload["expert_predictions"].numpy()
    targets = payload["targets"].numpy()
    ideal = payload["route_labels"].numpy()
    correct = predictions == targets[:, None]
    oracle = system_metrics(
        ideal,
        ideal,
        predictions,
        targets,
        costs["expert_mflops"],
        feature_extractor_index=costs["feature_extractor_index"],
        feature_extractor_mflops=costs["feature_extractor_mflops"],
        router_mflops=costs["router_mflops"],
    )
    return {
        "samples": int(len(targets)),
        "expert_top1": {
            name: float(correct[:, index].mean())
            for index, name in enumerate(config.candidates)
        },
        "route_distribution": {
            name: int(np.sum(ideal == index))
            for index, name in enumerate(config.candidates)
        },
        "no_correct_samples": int(np.sum(~correct.any(axis=1))),
        "oracle": _metrics_dict(oracle),
    }


def run_baseline(config: ExperimentConfig, cache_directory: Path, output: Path) -> None:
    train, search, fingerprints = _cache_pair(cache_directory, config)
    costs = _cost_context(config, int(train["features"].shape[1]))

    _write_json_atomic(
        {
            "schema_version": 1,
            "kind": "expert_baselines_and_oracle",
            "cache_fingerprints": fingerprints,
            "catalog_top1": {
                name: EXPERTS_BY_NAME[name].top1_accuracy_pct / 100.0
                for name in config.candidates
            },
            "dispatcher_train": _summarize_baseline(config, train, costs),
            "ga_search": _summarize_baseline(config, search, costs),
        },
        output,
    )


def run_fixed(config: ExperimentConfig, cache_directory: Path, output: Path) -> None:
    train, search, fingerprints = _cache_pair(cache_directory, config)
    number_of_experts = len(config.candidates)
    fixed = config.raw["fixed_solution"]
    penalties = directional_penalty_matrix(
        number_of_experts,
        underestimation=float(fixed["underestimation_penalty"]),
        overestimation=float(fixed["overestimation_penalty"]),
    )
    settings = _training_config(config)
    training = train_linear_dispatcher(
        train["features"],
        train["route_labels"],
        penalties,
        str(fixed["weighting"]),
        config=settings,
    )
    costs = _cost_context(config, int(train["features"].shape[1]))
    search_metrics = evaluate_cached_dispatcher(
        training.dispatcher, _evaluation_data(search), **costs, device=settings.device
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(training.dispatcher.state_dict(), output.with_suffix(".pt"))
    _write_json_atomic(
        {
            "schema_version": 1,
            "kind": "fixed",
            "cache_fingerprints": fingerprints,
            "weighting": fixed["weighting"],
            "penalty_matrix": penalties.tolist(),
            "epoch_losses": list(training.epoch_losses),
            "search_metrics": _metrics_dict(search_metrics),
        },
        output,
    )


def run_search(config: ExperimentConfig, cache_directory: Path, output: Path) -> None:
    train, search, fingerprints = _cache_pair(cache_directory, config)
    settings = _training_config(config)
    costs = _cost_context(config, int(train["features"].shape[1]))
    raw = config.raw["nsga2"]
    ga_config = NSGA2Config(
        population_size=int(raw["population_size"]),
        generations=int(raw["generations"]),
        crossover_eta=float(raw["crossover_eta"]),
        crossover_probability=float(raw["crossover_probability"]),
        mutation_eta=float(raw["mutation_eta"]),
        mutation_probability=(
            None if raw["mutation_probability"] is None else float(raw["mutation_probability"])
        ),
        penalty_min=float(raw["penalty_min"]),
        penalty_max=float(raw["penalty_max"]),
        weighting_schemes=tuple(raw["weighting_schemes"]),
    )
    problem = DispatcherNSGA2Problem(
        train["features"],
        train["route_labels"],
        _evaluation_data(search),
        costs["expert_mflops"],
        feature_extractor_index=costs["feature_extractor_index"],
        feature_extractor_mflops=costs["feature_extractor_mflops"],
        router_mflops=costs["router_mflops"],
        training_config=settings,
        weighting_schemes=ga_config.weighting_schemes,
        penalty_min=ga_config.penalty_min,
        penalty_max=ga_config.penalty_max,
        keep_dispatcher_states=True,
    )
    run = run_nsga2(problem, config=ga_config, seed=config.seed, verbose=True)
    pareto_ids = {record.evaluation_id for record in run.pareto_evaluations}
    records = [
        {
            "evaluation_id": record.evaluation_id,
            "chromosome": record.chromosome.tolist(),
            "weighting": record.weighting_scheme,
            "objectives": list(record.objectives),
            "metrics": _metrics_dict(record.metrics),
            "epoch_losses": list(record.epoch_losses),
            "pareto": record.evaluation_id in pareto_ids,
            "dispatcher_state": (
                {key: value.tolist() for key, value in record.dispatcher_state.items()}
                if record.evaluation_id in pareto_ids and record.dispatcher_state is not None
                else None
            ),
        }
        for record in run.evaluations
    ]
    _write_json_atomic(
        {
            "schema_version": 1,
            "kind": "nsga2_search",
            "cache_fingerprints": fingerprints,
            "experiment_seed": config.seed,
            "training": asdict(settings),
            "nsga2": asdict(ga_config),
            "evaluations": records,
        },
        output,
    )


def run_final(
    config: ExperimentConfig, cache_directory: Path, search_result: Path, output: Path
) -> None:
    train, _search, final, fingerprints = _cache_set(cache_directory, config)
    search_payload = json.loads(search_result.read_text(encoding="utf-8"))
    expected_search_fingerprints = {
        "dispatcher_train": fingerprints["dispatcher_train"],
        "ga_search": fingerprints["ga_search"],
    }
    if search_payload.get("cache_fingerprints") != expected_search_fingerprints:
        raise ValueError("search result and cache fingerprints differ")
    candidates = [item for item in search_payload["evaluations"] if item["pareto"]]
    settings = _training_config(config)
    costs = _cost_context(config, int(train["features"].shape[1]))
    head_directory = output.with_suffix("")
    head_directory.mkdir(parents=True, exist_ok=True)
    results = []
    for candidate in candidates:
        state = candidate.get("dispatcher_state")
        if not isinstance(state, Mapping):
            raise ValueError("Pareto search record is missing its trained dispatcher state")
        dispatcher = load_dispatcher_state(
            int(train["features"].shape[1]),
            len(config.candidates),
            state,
            device=settings.device,
        )
        metrics = evaluate_final_test(
            dispatcher,
            _evaluation_data(final),
            costs["expert_mflops"],
            feature_extractor_index=costs["feature_extractor_index"],
            feature_extractor_mflops=costs["feature_extractor_mflops"],
            router_mflops=costs["router_mflops"],
            device=settings.device,
        )
        head_path = head_directory / f"dispatcher-{candidate['evaluation_id']:05d}.pt"
        torch.save(dispatcher.state_dict(), head_path)
        results.append(
            {
                "search_evaluation_id": candidate["evaluation_id"],
                "chromosome": candidate["chromosome"],
                "weighting": candidate["weighting"],
                "dispatcher_checkpoint": str(head_path),
                "metrics": _metrics_dict(metrics),
            }
        )
    _write_json_atomic(
        {
            "schema_version": 1,
            "kind": "final_evaluation",
            "cache_fingerprints": fingerprints,
            "final_baselines_and_oracle": _summarize_baseline(config, final, costs),
            "solutions": results,
        },
        output,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    from .pipeline_cli import add_commands
    add_commands(subparsers)
    for name in ("baseline", "fixed", "search", "final"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--cache-dir", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "final":
            command.add_argument("--search-result", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if hasattr(args, "component_action"):
            from .pipeline_cli import run_command
            return run_command(args)
        config = load_config(args.config)
        seed_everything(config.seed)
        if args.command == "baseline":
            run_baseline(config, args.cache_dir, args.output)
        elif args.command == "fixed":
            run_fixed(config, args.cache_dir, args.output)
        elif args.command == "search":
            run_search(config, args.cache_dir, args.output)
        else:
            run_final(config, args.cache_dir, args.search_result, args.output)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"{args.command} failed: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
