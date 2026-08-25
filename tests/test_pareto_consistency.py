from __future__ import annotations

import csv
from pathlib import Path

from pertinence.config import PARETO_MODELS, ROUTING_CANDIDATES, load_config
from pertinence.experts import EXPERT_REGISTRY


ROOT = Path(__file__).resolve().parents[1]


def test_computed_pareto_front_is_the_single_model_ordering_source() -> None:
    with (ROOT / "data" / "chenyaofo_cifar10_models.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = tuple(csv.DictReader(handle))

    csv_pareto_models = tuple(
        row["model"]
        for row in sorted(
            (row for row in rows if row["pareto"].strip().lower() == "true"),
            key=lambda row: float(row["madds_m"]),
        )
    )
    config = load_config(ROOT / "configs" / "cifar10_pareto.yaml")

    assert PARETO_MODELS == csv_pareto_models
    assert config.pareto_boundary_models == csv_pareto_models
    assert config.candidates == ROUTING_CANDIDATES
    assert set(config.candidates).issubset(csv_pareto_models)
    assert tuple(spec.name for spec in EXPERT_REGISTRY) == csv_pareto_models
