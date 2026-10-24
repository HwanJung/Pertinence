"""NSGA-II exploration of PERTINENCE dispatcher training configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .metrics import SystemMetrics, pareto_mask
from .routing import WeightingScheme
from .training import (
    CachedEvaluationData,
    DispatcherTrainingConfig,
    DispatcherTrainingResult,
    evaluate_cached_dispatcher,
    evaluate_final_test,
    train_linear_dispatcher,
)


DEFAULT_WEIGHTING_SCHEMES: tuple[WeightingScheme, ...] = ("INS", "ISNS", "ENS")


@dataclass(frozen=True)
class DecodedChromosome:
    penalty_matrix: np.ndarray
    weighting_scheme: WeightingScheme


@dataclass(frozen=True)
class EvaluationRecord:
    evaluation_id: int
    chromosome: np.ndarray
    penalty_matrix: np.ndarray
    weighting_scheme: WeightingScheme
    metrics: SystemMetrics
    objectives: tuple[float, float]
    epoch_losses: tuple[float, ...]
    dispatcher_state: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class NSGA2Config:
    population_size: int = 50
    generations: int = 50
    crossover_eta: float = 20.0
    crossover_probability: float = 0.9
    mutation_eta: float = 25.0
    mutation_probability: float | None = None
    penalty_min: float = 0.0
    penalty_max: float = 100.0
    weighting_schemes: tuple[WeightingScheme, ...] = DEFAULT_WEIGHTING_SCHEMES

    def validate(self) -> None:
        if self.population_size <= 1 or self.generations <= 0:
            raise ValueError("NSGA-II requires population_size > 1 and generations > 0")
        if self.crossover_eta <= 0 or self.mutation_eta <= 0:
            raise ValueError("SBX and polynomial-mutation eta must be positive")
        if not 0 <= self.crossover_probability <= 1:
            raise ValueError("crossover_probability must be in [0, 1]")
        if self.mutation_probability is not None and not 0 <= self.mutation_probability <= 1:
            raise ValueError("mutation_probability must be in [0, 1]")
        if self.penalty_min < 0 or self.penalty_max <= self.penalty_min:
            raise ValueError("invalid penalty bounds")
        if not self.weighting_schemes:
            raise ValueError("at least one weighting scheme is required")
        if len(set(self.weighting_schemes)) != len(self.weighting_schemes):
            raise ValueError("weighting_schemes must not contain duplicates")


@dataclass(frozen=True)
class OptimizationRun:
    pymoo_result: Any
    evaluations: tuple[EvaluationRecord, ...]

    @property
    def pareto_evaluations(self) -> tuple[EvaluationRecord, ...]:
        """Non-dominated records across every solution evaluated in the run."""

        if not self.evaluations:
            return ()
        accuracy = np.asarray([record.metrics.system_accuracy for record in self.evaluations])
        cost = np.asarray([record.metrics.average_mflops for record in self.evaluations])
        mask = pareto_mask(accuracy, cost)
        return tuple(record for record, keep in zip(self.evaluations, mask) if keep)


@dataclass(frozen=True)
class FinalSolutionEvaluation:
    chromosome: np.ndarray
    decoded: DecodedChromosome
    training: DispatcherTrainingResult
    metrics: SystemMetrics


def chromosome_size(num_experts: int) -> int:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    return num_experts * num_experts + 1


def decode_chromosome(
    chromosome: Sequence[float] | np.ndarray,
    num_experts: int,
    *,
    weighting_schemes: Sequence[WeightingScheme] = DEFAULT_WEIGHTING_SCHEMES,
    penalty_min: float = 0.0,
    penalty_max: float = 100.0,
) -> DecodedChromosome:
    """Decode the paper's N^2 penalty genes plus one weighting gene.

    The final continuous gene is divided into equal-width categorical bins.
    Diagonal penalty genes remain in the chromosome, as described in the paper,
    but are forced to zero when decoded.
    """

    values = np.asarray(chromosome, dtype=np.float64)
    if values.shape != (chromosome_size(num_experts),):
        raise ValueError(f"expected chromosome with {chromosome_size(num_experts)} genes")
    if not np.all(np.isfinite(values)):
        raise ValueError("chromosome contains non-finite values")
    if not weighting_schemes:
        raise ValueError("at least one weighting scheme is required")
    penalties = np.clip(values[:-1], penalty_min, penalty_max).reshape(
        num_experts, num_experts
    )
    penalties = penalties.copy()
    np.fill_diagonal(penalties, 0.0)
    weight_index = int(np.floor(values[-1]))
    weight_index = int(np.clip(weight_index, 0, len(weighting_schemes) - 1))
    return DecodedChromosome(
        penalty_matrix=penalties.astype(np.float32),
        weighting_scheme=weighting_schemes[weight_index],
    )


class DispatcherNSGA2Problem(ElementwiseProblem):
    """Pymoo problem whose fitness uses only training and GA-search caches."""

    def __init__(
        self,
        train_features,
        train_route_labels,
        search_data: CachedEvaluationData,
        expert_mflops,
        *,
        feature_extractor_index: int,
        feature_extractor_mflops: float,
        router_mflops: float,
        training_config: DispatcherTrainingConfig | None = None,
        weighting_schemes: Sequence[WeightingScheme] = DEFAULT_WEIGHTING_SCHEMES,
        penalty_min: float = 0.0,
        penalty_max: float = 100.0,
        keep_dispatcher_states: bool = False,
    ) -> None:
        search_data.validate()
        self.num_experts = search_data.num_experts
        costs = np.asarray(expert_mflops)
        if costs.shape != (self.num_experts,):
            raise ValueError("expert_mflops must have one value per expert")
        if not 0 <= feature_extractor_index < self.num_experts:
            raise ValueError("feature_extractor_index is outside the expert range")
        if penalty_min < 0 or penalty_max <= penalty_min:
            raise ValueError("invalid penalty bounds")
        if not weighting_schemes:
            raise ValueError("at least one weighting scheme is required")

        self.train_features = train_features
        self.train_route_labels = train_route_labels
        self.search_data = search_data
        self.expert_mflops = expert_mflops
        self.feature_extractor_index = feature_extractor_index
        self.feature_extractor_mflops = float(feature_extractor_mflops)
        self.router_mflops = float(router_mflops)
        self.training_config = training_config or DispatcherTrainingConfig()
        self.weighting_schemes = tuple(weighting_schemes)
        self.penalty_min = float(penalty_min)
        self.penalty_max = float(penalty_max)
        self.keep_dispatcher_states = keep_dispatcher_states
        self.evaluations: list[EvaluationRecord] = []

        variable_count = chromosome_size(self.num_experts)
        lower = np.full(variable_count, self.penalty_min, dtype=np.float64)
        upper = np.full(variable_count, self.penalty_max, dtype=np.float64)
        lower[-1] = 0.0
        # Equal-width [0,1), [1,2), ... bins; the exact upper endpoint is
        # clipped by decode_chromosome into the last category.
        upper[-1] = float(len(self.weighting_schemes))
        super().__init__(n_var=variable_count, n_obj=2, xl=lower, xu=upper)

    def _evaluate(self, chromosome, out, *args, **kwargs) -> None:
        decoded = decode_chromosome(
            chromosome,
            self.num_experts,
            weighting_schemes=self.weighting_schemes,
            penalty_min=self.penalty_min,
            penalty_max=self.penalty_max,
        )
        training = train_linear_dispatcher(
            self.train_features,
            self.train_route_labels,
            decoded.penalty_matrix,
            decoded.weighting_scheme,
            config=self.training_config,
        )
        metrics = evaluate_cached_dispatcher(
            training.dispatcher,
            self.search_data,
            self.expert_mflops,
            feature_extractor_index=self.feature_extractor_index,
            feature_extractor_mflops=self.feature_extractor_mflops,
            router_mflops=self.router_mflops,
            batch_size=self.training_config.batch_size,
            device=self.training_config.device,
        )
        objectives = (-metrics.system_accuracy, metrics.average_mflops)
        out["F"] = np.asarray(objectives, dtype=np.float64)
        self.evaluations.append(
            EvaluationRecord(
                evaluation_id=len(self.evaluations),
                chromosome=np.asarray(chromosome, dtype=np.float64).copy(),
                penalty_matrix=decoded.penalty_matrix.copy(),
                weighting_scheme=decoded.weighting_scheme,
                metrics=metrics,
                objectives=objectives,
                epoch_losses=training.epoch_losses,
                dispatcher_state=(
                    training.state_dict_numpy() if self.keep_dispatcher_states else None
                ),
            )
        )


def run_nsga2(
    problem: DispatcherNSGA2Problem,
    *,
    config: NSGA2Config | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> OptimizationRun:
    """Run paper-default NSGA-II and retain every evaluated solution."""

    settings = config or NSGA2Config(
        penalty_min=problem.penalty_min,
        penalty_max=problem.penalty_max,
        weighting_schemes=problem.weighting_schemes,
    )
    settings.validate()
    if tuple(settings.weighting_schemes) != problem.weighting_schemes:
        raise ValueError("problem and NSGA-II weighting schemes differ")
    if (
        settings.penalty_min != problem.penalty_min
        or settings.penalty_max != problem.penalty_max
    ):
        raise ValueError("problem and NSGA-II penalty bounds differ")
    mutation_kwargs: dict[str, float] = {"eta": settings.mutation_eta}
    if settings.mutation_probability is not None:
        mutation_kwargs["prob"] = settings.mutation_probability
    algorithm = NSGA2(
        pop_size=settings.population_size,
        crossover=SBX(
            prob=settings.crossover_probability,
            eta=settings.crossover_eta,
        ),
        mutation=PM(**mutation_kwargs),
        eliminate_duplicates=False,
    )
    result = minimize(
        problem,
        algorithm,
        get_termination("n_gen", settings.generations),
        seed=seed,
        verbose=verbose,
        save_history=False,
    )
    return OptimizationRun(pymoo_result=result, evaluations=tuple(problem.evaluations))


def evaluate_final_solution(
    chromosome: Sequence[float] | np.ndarray,
    train_features,
    train_route_labels,
    final_test: CachedEvaluationData,
    expert_mflops,
    *,
    feature_extractor_index: int,
    feature_extractor_mflops: float,
    router_mflops: float,
    training_config: DispatcherTrainingConfig | None = None,
    weighting_schemes: Sequence[WeightingScheme] = DEFAULT_WEIGHTING_SCHEMES,
    penalty_min: float = 0.0,
    penalty_max: float = 100.0,
) -> FinalSolutionEvaluation:
    """Train one selected chromosome and evaluate only on final-test caches."""

    settings = training_config or DispatcherTrainingConfig()
    decoded = decode_chromosome(
        chromosome,
        final_test.num_experts,
        weighting_schemes=weighting_schemes,
        penalty_min=penalty_min,
        penalty_max=penalty_max,
    )
    training = train_linear_dispatcher(
        train_features,
        train_route_labels,
        decoded.penalty_matrix,
        decoded.weighting_scheme,
        config=settings,
    )
    metrics = evaluate_final_test(
        training.dispatcher,
        final_test,
        expert_mflops,
        feature_extractor_index=feature_extractor_index,
        feature_extractor_mflops=feature_extractor_mflops,
        router_mflops=router_mflops,
        batch_size=settings.batch_size,
        device=settings.device,
    )
    return FinalSolutionEvaluation(
        chromosome=np.asarray(chromosome, dtype=np.float64).copy(),
        decoded=decoded,
        training=training,
        metrics=metrics,
    )
