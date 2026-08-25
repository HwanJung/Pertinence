from pathlib import Path

from pertinence.config import PARETO_MODELS, ROUTING_CANDIDATES, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_default_config_separates_pareto_boundary_and_routing_pool() -> None:
    config = load_config(ROOT / "configs" / "cifar10_pareto.yaml")
    assert config.pareto_boundary_models == PARETO_MODELS
    assert config.candidates == ROUTING_CANDIDATES
    assert config.feature_extractor == "shufflenetv2_x0_5"
