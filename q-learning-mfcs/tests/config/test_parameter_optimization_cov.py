import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

from config.parameter_optimization import (
    OptimizationMethod,
    ObjectiveType,
    OptimizationObjective,
    OptimizationConstraint,
    OptimizationResult,
    ParameterOptimizer,
    GeneticOptimizer,
    GradientOptimizer,
    calculate_pareto_frontier,
    hypervolume_indicator,
)


class TestEnums:
    def test_optimization_method(self):
        assert OptimizationMethod.BAYESIAN.value == "bayesian"
        assert OptimizationMethod.GENETIC.value == "genetic"
        assert OptimizationMethod.GRADIENT_BASED.value == "gradient_based"
        assert OptimizationMethod.PARTICLE_SWARM.value == "particle_swarm"

    def test_objective_type(self):
        assert ObjectiveType.MINIMIZE.value == "minimize"
        assert ObjectiveType.MAXIMIZE.value == "maximize"


class TestOptimizationObjective:
    def test_creation(self):
        obj = OptimizationObjective(name="loss", type=ObjectiveType.MINIMIZE)
        assert obj.name == "loss"
        assert obj.weight == 1.0
        assert obj.tolerance == 1e-6


class TestOptimizationConstraint:
    def test_creation(self):
        c = OptimizationConstraint(
            name="bound", constraint_function=lambda x: x[0] - 1.0
        )
        assert c.name == "bound"
        assert c.constraint_type == "ineq"


class TestOptimizationResult:
    def test_creation(self):
        r = OptimizationResult(method=OptimizationMethod.GENETIC)
        assert r.total_evaluations == 0
        assert r.converged is False

    def test_set_end_time(self):
        r = OptimizationResult(method=OptimizationMethod.GENETIC)
        r.set_end_time()
        assert r.end_time is not None

    def test_get_optimization_time_no_end(self):
        r = OptimizationResult(method=OptimizationMethod.GENETIC)
        assert r.get_optimization_time() == 0.0

    def test_get_optimization_time_with_end(self):
        r = OptimizationResult(method=OptimizationMethod.GENETIC)
        r.set_end_time()
        assert r.get_optimization_time() >= 0.0


@dataclass
class MockBounds:
    min_value: float
    max_value: float


@dataclass
class MockParam:
    name: str
    bounds: MockBounds


class MockParameterSpace:
    def __init__(self, n_params=2):
        self.parameters = [
            MockParam(f"p{i}", MockBounds(0.0, 10.0)) for i in range(n_params)
        ]

    def sample(self, n, method="random", seed=None):
        np.random.seed(seed)
        return np.random.uniform(
            0, 10, size=(n, len(self.parameters))
        )


def _obj_fn(params):
    return {"loss": np.sum(params ** 2)}


class TestParameterOptimizerBase:
    def test_evaluate_objectives_success(self):
        space = MockParameterSpace()
        objs = [OptimizationObjective("loss", ObjectiveType.MINIMIZE)]
        go = GeneticOptimizer(space, objs, random_seed=42, population_size=4)
        result = go._evaluate_objectives(np.array([1.0, 2.0]), _obj_fn)
        assert result["loss"] == pytest.approx(5.0)

    def test_evaluate_objectives_failure(self):
        space = MockParameterSpace()
        objs = [OptimizationObjective("loss", ObjectiveType.MINIMIZE)]
        go = GeneticOptimizer(space, objs, random_seed=42, population_size=4)

        def bad_fn(x):
            raise RuntimeError("fail")

        result = go._evaluate_objectives(np.array([1.0, 2.0]), bad_fn)
        assert result["loss"] == float("inf")

    def test_check_constraints_empty(self):
        space = MockParameterSpace()
        objs = [OptimizationObjective("loss", ObjectiveType.MINIMIZE)]
        go = GeneticOptimizer(space, objs, random_seed=42, population_size=4)
        assert go._check_constraints(np.array([1.0, 2.0])) is True

    def test_check_constraints_ineq_satisfied(self):
        space = MockParameterSpace()
        objs = [OptimizationObjective("loss", ObjectiveType.MINIMIZE)]
        c = OptimizationConstraint("c1", lambda x: x[0] - 0.5, "ineq")
        go = GeneticOptimizer(space, objs, constraints=[c], random_seed=42, population_size=4)
        assert go._check_constraints(np.array([1.0, 2.0])) is True

    def test_check_constraints_ineq_violated(self):
        space = MockParameterSpace()
        objs = [OptimizationObjective("loss", ObjectiveType.MINIMIZE)]
        c = OptimizationConstraint("c1", lambda x: x[0] - 5.0, "ineq")
        go = GeneticOptimizer(space, objs, constraints=[c], random_seed=42, population_size=4)
        assert go._check_constraints(np.array([1.0, 2.0])) is False

    def test_check_constraints_eq_satisfied(self):
        space = MockParameterSpace()
        objs = [OptimizationObjective("loss", ObjectiveType.MINIMIZE)]
        c = OptimizationConstraint("c1", lambda x: 0.0, "eq")
        go = GeneticOptimizer(space, objs, constraints=[c], random_seed=42, population_size=4)
        assert go._check_constraints(np.array([1.0, 2.0])) is True

    def test_check_constraints_eq_violated(self):
        space = MockParameterSpace()
        objs = [OptimizationObjective("loss", ObjectiveType.MINIMIZE)]
        c = OptimizationConstraint("c1", lambda x: 5.0, "eq")
        go = GeneticOptimizer(space, objs, constraints=[c], random_seed=42, population_size=4)
        assert go._check_constraints(np.array([1.0, 2.0])) is False

    def test_check_constraints_exception(self):
        space = MockParameterSpace()
        objs = [OptimizationObjective("loss", ObjectiveType.MINIMIZE)]
        c = OptimizationConstraint("c1", lambda x: 1 / 0, "ineq")
        go = GeneticOptimizer(space, objs, constraints=[c], random_seed=42, population_size=4)
        assert go._check_constraints(np.array([1.0, 2.0])) is False

    def test_calculate_overall_score_minimize(self):
        space = MockParameterSpace()
        objs = [OptimizationObjective("loss", ObjectiveType.MINIMIZE, weight=2.0)]
        go = GeneticOptimizer(space, objs, random_seed=42, population_size=4)
        score = go._calculate_overall_score({"loss": 5.0})
        assert score == pytest.approx(-10.0)

    def test_calculate_overall_score_maximize(self):
        space = MockParameterSpace()
        objs = [OptimizationObjective("reward", ObjectiveType.MAXIMIZE, weight=2.0)]
        go = GeneticOptimizer(space, objs, random_seed=42, population_size=4)
        score = go._calculate_overall_score({"reward": 5.0})
        assert score == pytest.approx(10.0)


class TestGeneticOptimizer:
    def _make_go(self, pop_size=6, sel="tournament"):
        space = MockParameterSpace()
        objs = [OptimizationObjective("loss", ObjectiveType.MINIMIZE)]
        return GeneticOptimizer(space, objs, population_size=pop_size,
                                selection_method=sel, random_seed=42)

    def test_optimize(self):
        go = self._make_go(pop_size=6)
        result = go.optimize(_obj_fn, max_evaluations=30)
        assert result.method == OptimizationMethod.GENETIC
        assert result.best_parameters is not None

    def test_tournament_selection(self):
        go = self._make_go(sel="tournament")
        fitness = np.array([1.0, 5.0, 3.0, 2.0, 4.0, 6.0])
        selected = go._tournament_selection(fitness, 4)
        assert len(selected) == 4

    def test_roulette_selection(self):
        go = self._make_go(sel="roulette")
        fitness = np.array([1.0, 5.0, 3.0, 2.0, 4.0, 6.0])
        selected = go._roulette_selection(fitness, 4)
        assert len(selected) == 4

    def test_roulette_selection_negative(self):
        go = self._make_go(sel="roulette")
        fitness = np.array([-1.0, 5.0, -3.0])
        selected = go._roulette_selection(fitness, 2)
        assert len(selected) == 2

    def test_rank_selection(self):
        go = self._make_go(sel="rank")
        fitness = np.array([1.0, 5.0, 3.0, 2.0, 4.0, 6.0])
        selected = go._rank_selection(fitness, 4)
        assert len(selected) == 4

    def test_crossover(self):
        go = self._make_go()
        p1 = np.array([1.0, 2.0])
        p2 = np.array([3.0, 4.0])
        c1, c2 = go._crossover(p1, p2)
        assert len(c1) == 2 and len(c2) == 2

    def test_mutation(self):
        go = self._make_go()
        go.mutation_rate = 1.0
        ind = np.array([5.0, 5.0])
        mutated = go._mutation(ind)
        assert len(mutated) == 2

    def test_clip_to_bounds(self):
        go = self._make_go()
        ind = np.array([-1.0, 15.0])
        clipped = go._clip_to_bounds(ind)
        assert clipped[0] == 0.0
        assert clipped[1] == 10.0

    def test_check_convergence_short(self):
        go = self._make_go()
        assert go._check_convergence([1.0, 2.0]) is False

    def test_selection_dispatch_roulette(self):
        go = self._make_go(sel="roulette")
        fitness = np.array([1.0, 5.0, 3.0, 2.0])
        sel = go._selection(fitness, 2)
        assert len(sel) == 2

    def test_selection_dispatch_rank(self):
        go = self._make_go(sel="rank")
        fitness = np.array([1.0, 5.0, 3.0, 2.0])
        sel = go._selection(fitness, 2)
        assert len(sel) == 2

    def test_selection_dispatch_default(self):
        go = self._make_go(sel="unknown")
        fitness = np.array([1.0, 5.0, 3.0, 2.0])
        sel = go._selection(fitness, 2)
        assert len(sel) == 2


class TestParetoAndHypervolume:
    def test_pareto_frontier(self):
        objs = np.array([[1, 3], [2, 2], [3, 1], [2, 3]])
        is_pareto = calculate_pareto_frontier(objs)
        assert is_pareto.sum() >= 1

    def test_hypervolume_2d(self):
        front = np.array([[1, 3], [2, 2], [3, 1]])
        ref = np.array([0, 0])
        hv = hypervolume_indicator(front, ref)
        assert hv > 0

    def test_hypervolume_empty(self):
        front = np.array([]).reshape(0, 2)
        ref = np.array([0, 0])
        hv = hypervolume_indicator(front, ref)
        assert hv == 0.0

    def test_hypervolume_3d(self):
        np.random.seed(42)
        front = np.array([[1, 2, 3], [2, 1, 3], [3, 2, 1]])
        ref = np.array([0, 0, 0])
        hv = hypervolume_indicator(front, ref)
        assert hv >= 0
