"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PARETO_MODELS = (
    "shufflenetv2_x0_5",
    "mobilenetv2_x0_5",
    "shufflenetv2_x1_0",
    "mobilenetv2_x0_75",
    "mobilenetv2_x1_0",
    "resnet44",
    "resnet56",
    "repvgg_a0",
    "repvgg_a1",
    "repvgg_a2",
)

ROUTING_CANDIDATES = (
    "shufflenetv2_x0_5",
    "mobilenetv2_x0_5",
    "resnet56",
    "repvgg_a1",
)


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def candidates(self) -> tuple[str, ...]:
        return tuple(self.raw["experts"]["candidates"])

    @property
    def pareto_boundary_models(self) -> tuple[str, ...]:
        return tuple(self.raw["experts"]["pareto_boundary_models"])

    @property
    def feature_extractor(self) -> str:
        return str(self.raw["experts"]["feature_extractor"])

    @property
    def seed(self) -> int:
        return int(self.raw["experiment"]["seed"])

    @property
    def artifacts_root(self) -> Path:
        return (self.path.parent.parent / self.raw["assets"]["root"]).resolve()


def load_config(path: str | Path) -> ExperimentConfig:
    resolved = Path(path).resolve()
    with resolved.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    config = ExperimentConfig(raw=raw, path=resolved)
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.raw["data"]["dataset"] != "CIFAR10":
        raise ValueError("this reproduction intentionally supports CIFAR-10 only")
    if config.pareto_boundary_models != PARETO_MODELS:
        raise ValueError(
            "experts.pareto_boundary_models must be the cost-sorted "
            "chenyaofo Pareto front"
        )
    if config.candidates != ROUTING_CANDIDATES:
        raise ValueError(
            "experts.candidates must be the fixed four-model routing configuration"
        )
    if not set(config.candidates).issubset(config.pareto_boundary_models):
        raise ValueError("routing candidates must belong to the Pareto boundary")
    if config.feature_extractor not in config.candidates:
        raise ValueError("feature extractor must also be a routing candidate")
    if config.feature_extractor != config.candidates[0]:
        raise ValueError("the routing pool requires its cheapest model as feature extractor")
    train_size = int(config.raw["data"]["dispatcher_train_size"])
    search_size = int(config.raw["data"]["ga_search_size"])
    final_size = int(config.raw["data"]["final_evaluation_size"])
    if train_size != 50_000:
        raise ValueError("dispatcher training must use all 50,000 official training samples")
    if search_size != 8_000 or final_size != 2_000:
        raise ValueError("the fixed reproduction split is GA search 8,000 + final 2,000")
    if config.raw["nsga2"]["crossover"] != "SBX":
        raise ValueError("the paper reproduction requires SBX crossover")
    if config.raw["nsga2"]["mutation"] != "polynomial":
        raise ValueError("the paper reproduction requires polynomial mutation")
    if float(config.raw["experts"]["mac_to_flop"]) != 2.0:
        raise ValueError("the project-wide convention is fixed to 1 MAC = 2 FLOPs")
