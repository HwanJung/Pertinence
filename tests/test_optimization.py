import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("pymoo")

from pertinence.optimization import (  # noqa: E402
    DispatcherNSGA2Problem,
    NSGA2Config,
    chromosome_size,
    decode_chromosome,
    evaluate_final_solution,
    run_nsga2,
)
from pertinence.training import (  # noqa: E402
    CachedEvaluationData,
    DispatcherTrainingConfig,
    evaluate_cached_dispatcher,
    train_linear_dispatcher,
)


def _tiny_caches():
    features = torch.tensor(
        [
            [-2.0, -1.0],
            [-1.5, -2.0],
            [-1.0, -1.5],
            [1.0, 1.5],
            [1.5, 2.0],
            [2.0, 1.0],
        ]
    )
    ideal = torch.tensor([0, 0, 0, 1, 1, 1])
    targets = torch.tensor([0, 1, 2, 3, 4, 5])
    predictions = torch.tensor(
        [
            [0, 9],
            [1, 9],
            [2, 9],
            [9, 3],
            [9, 4],
            [9, 5],
        ]
    )
    return features, ideal, CachedEvaluationData(features, predictions, targets, ideal)


def test_chromosome_is_n_squared_plus_one_and_decodes_diagonal() -> None:
    defaults = NSGA2Config()
    assert (defaults.population_size, defaults.generations) == (50, 50)
    assert (defaults.crossover_eta, defaults.crossover_probability) == (20.0, 0.9)
    assert defaults.mutation_eta == 25.0
    assert (defaults.penalty_min, defaults.penalty_max) == (0.0, 100.0)
    assert chromosome_size(2) == 5
    chromosome = np.array([99.0, 10.0, 0.001, 77.0, 1.2])
    decoded = decode_chromosome(chromosome, 2)
    assert decoded.weighting_scheme == "ISNS"
    np.testing.assert_allclose(
        decoded.penalty_matrix,
        np.array([[0.0, 10.0], [0.001, 0.0]], dtype=np.float32),
    )


def test_cached_training_and_system_evaluation_are_deterministic() -> None:
    features, ideal, evaluation = _tiny_caches()
    config = DispatcherTrainingConfig(
        epochs=20,
        batch_size=6,
        learning_rate=0.1,
        seed=7,
    )
    penalty = np.array([[0.0, 10.0], [10.0, 0.0]], dtype=np.float32)
    first = train_linear_dispatcher(features, ideal, penalty, "INS", config=config)
    second = train_linear_dispatcher(features, ideal, penalty, "INS", config=config)
    assert first.epoch_losses == pytest.approx(second.epoch_losses)
    for key, value in first.dispatcher.state_dict().items():
        assert torch.equal(value, second.dispatcher.state_dict()[key])
    # Make routing exact so this assertion tests cached system evaluation rather
    # than depending on optimizer convergence details.
    with torch.no_grad():
        first.dispatcher.linear.weight.copy_(torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))
        first.dispatcher.linear.bias.zero_()
    metrics = evaluate_cached_dispatcher(
        first.dispatcher,
        evaluation,
        np.array([10.0, 100.0]),
        feature_extractor_index=0,
        feature_extractor_mflops=10.0,
        router_mflops=0.1,
    )
    assert metrics.system_accuracy == 1.0
    assert metrics.route_accuracy == 1.0
    assert metrics.average_mflops == pytest.approx(60.1)


def test_problem_records_objective_and_final_test_is_separate() -> None:
    features, ideal, search = _tiny_caches()
    training_config = DispatcherTrainingConfig(
        epochs=2,
        batch_size=6,
        learning_rate=0.1,
        seed=11,
    )
    problem = DispatcherNSGA2Problem(
        features,
        ideal,
        search,
        np.array([10.0, 100.0]),
        feature_extractor_index=0,
        feature_extractor_mflops=10.0,
        router_mflops=0.1,
        training_config=training_config,
    )
    chromosome = np.array([0.0, 10.0, 10.0, 0.0, 0.2])
    output = {}
    problem._evaluate(chromosome, output)
    assert len(problem.evaluations) == 1
    record = problem.evaluations[0]
    assert output["F"].tolist() == pytest.approx(
        [-record.metrics.system_accuracy, record.metrics.average_mflops]
    )

    final = evaluate_final_solution(
        chromosome,
        features,
        ideal,
        search,
        np.array([10.0, 100.0]),
        feature_extractor_index=0,
        feature_extractor_mflops=10.0,
        router_mflops=0.1,
        training_config=training_config,
    )
    assert final.decoded.weighting_scheme == "INS"
    assert final.metrics.system_accuracy >= 0.0


def test_small_nsga2_run_keeps_every_evaluated_solution() -> None:
    features, ideal, search = _tiny_caches()
    problem = DispatcherNSGA2Problem(
        features,
        ideal,
        search,
        np.array([10.0, 100.0]),
        feature_extractor_index=0,
        feature_extractor_mflops=10.0,
        router_mflops=0.1,
        training_config=DispatcherTrainingConfig(
            epochs=1,
            batch_size=6,
            learning_rate=0.1,
            seed=5,
        ),
    )
    run = run_nsga2(
        problem,
        config=NSGA2Config(population_size=4, generations=2),
        seed=3,
    )
    assert len(run.evaluations) >= 4
    assert all(record.objectives[0] <= 0 for record in run.evaluations)
    assert run.pareto_evaluations
