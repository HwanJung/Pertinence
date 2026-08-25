import numpy as np

from pertinence.metrics import pareto_mask, system_metrics


def test_system_accuracy_is_not_route_accuracy_and_extractor_is_shared() -> None:
    # Both routed experts happen to classify correctly, although one route differs
    # from cheapest-correct. The extractor (expert 0) is not charged twice.
    predictions = np.array([[1, 1], [2, 2]])
    targets = np.array([1, 2])
    routed = np.array([1, 0])
    ideal = np.array([0, 0])
    result = system_metrics(
        routed,
        ideal,
        predictions,
        targets,
        np.array([10.0, 100.0]),
        feature_extractor_index=0,
        feature_extractor_mflops=10.0,
        router_mflops=0.1,
    )
    assert result.system_accuracy == 1.0
    assert result.route_accuracy == 0.5
    assert result.average_mflops == 60.1


def test_pareto_dominance_maximizes_accuracy_and_minimizes_cost() -> None:
    mask = pareto_mask(
        np.array([0.90, 0.91, 0.90, 0.95]),
        np.array([10.0, 12.0, 15.0, 30.0]),
    )
    assert mask.tolist() == [True, True, False, True]

