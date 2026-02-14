"""Tests for parameter_optimization.py - coverage part 3.

Missing: GeneticOptimizer (selection methods, crossover, mutation,
convergence), GradientOptimizer optimize, ParameterOptimizer
(_evaluate_objectives failure, _check_constraints, _calculate_overall_score),
OptimizationResult methods, calculate_pareto_frontier, hypervolume_indicator.
"""
import sys
import os
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.parameter_optimization import (
    BayesianOptimizer,
    GeneticOptimizer,
    GradientOptimizer,
    ObjectiveType,
    OptimizationConstraint,
    OptimizationMethod,
    OptimizationObjective,
    OptimizationResult,
    calculate_pareto_frontier,
    hypervolume_indicator,
)


@dataclass
class _FakeParam:
    name: str
    bounds: "_FakeBounds"


@dataclass
class _FakeBounds:
    min_value: float
    max_value: float


class _FakeParamSpace:
    """Mock parameter space."""
    def __init__(self, n_params=2):
        self.parameters = [
            _FakeParam(f"p{i}", _FakeBounds(0.0, 1.0))
            for i in range(n_params)
        ]

    def sample(self, n, method="random", seed=None):
        return np.random.uniform(0.0, 1.0, (n, len(self.parameters)))


def _make_objective(name="obj", otype=ObjectiveType.MAXIMIZE):
    return OptimizationObjective(name=name, type=otype, weight=1.0)


def _simple_obj_func(params):
    return {"obj": -np.sum((params - 0.5) ** 2)}


@pytest.mark.coverage_extra
class TestOptimizationResult:
    """Cover OptimizationResult methods."""

    def test_set_end_time(self):
        r = OptimizationResult(method=OptimizationMethod.BAYESIAN)
        r.set_end_time()
        assert r.end_time is not None

    def test_get_optimization_time(self):
        r = OptimizationResult(method=OptimizationMethod.BAYESIAN)
        assert r.get_optimization_time() == 0.0
        r.set_end_time()
        assert r.get_optimization_time() >= 0.0


@pytest.mark.coverage_extra
class TestParameterOptimizerBase:
    """Cover _evaluate_objectives, _check_constraints, _calculate_overall_score."""

    def test_evaluate_objectives_success(self):
        space = _FakeParamSpace()
        obj = _make_objective()
        opt = GeneticOptimizer(space, [obj], random_seed=42)
        result = opt._evaluate_objectives(np.array([0.5, 0.5]), _simple_obj_func)
        assert "obj" in result

    def test_evaluate_objectives_failure(self):
        space = _FakeParamSpace()
        obj = _make_objective()
        opt = GeneticOptimizer(space, [obj], random_seed=42)
        def bad_func(p):
            raise ValueError("fail")
        result = opt._evaluate_objectives(np.array([0.5, 0.5]), bad_func)
        assert result["obj"] == -float("inf")

    def test_evaluate_objectives_failure_minimize(self):
        space = _FakeParamSpace()
        obj = _make_objective(otype=ObjectiveType.MINIMIZE)
        opt = GeneticOptimizer(space, [obj], random_seed=42)
        def bad_func(p):
            raise ValueError("fail")
        result = opt._evaluate_objectives(np.array([0.5, 0.5]), bad_func)
        assert result["obj"] == float("inf")

    def test_check_constraints_satisfied(self):
        space = _FakeParamSpace()
        obj = _make_objective()
        constraint = OptimizationConstraint(
            name="c1",
            constraint_function=lambda x: 1.0,
            constraint_type="ineq",
        )
        opt = GeneticOptimizer(space, [obj], constraints=[constraint])
        assert opt._check_constraints(np.array([0.5, 0.5])) is True

    def test_check_constraints_eq_violated(self):
        space = _FakeParamSpace()
        obj = _make_objective()
        constraint = OptimizationConstraint(
            name="c1",
            constraint_function=lambda x: 10.0,
            constraint_type="eq",
            tolerance=1e-6,
        )
        opt = GeneticOptimizer(space, [obj], constraints=[constraint])
        assert opt._check_constraints(np.array([0.5, 0.5])) is False

    def test_check_constraints_ineq_violated(self):
        space = _FakeParamSpace()
        obj = _make_objective()
        constraint = OptimizationConstraint(
            name="c1",
            constraint_function=lambda x: -10.0,
            constraint_type="ineq",
        )
        opt = GeneticOptimizer(space, [obj], constraints=[constraint])
        assert opt._check_constraints(np.array([0.5, 0.5])) is False

    def test_check_constraints_exception(self):
        space = _FakeParamSpace()
        obj = _make_objective()
        constraint = OptimizationConstraint(
            name="c1",
            constraint_function=lambda x: 1 / 0,
        )
        opt = GeneticOptimizer(space, [obj], constraints=[constraint])
        assert opt._check_constraints(np.array([0.5, 0.5])) is False

    def test_calculate_overall_score_maximize(self):
        space = _FakeParamSpace()
        obj = _make_objective(otype=ObjectiveType.MAXIMIZE)
        opt = GeneticOptimizer(space, [obj])
        score = opt._calculate_overall_score({"obj": 5.0})
        assert score == 5.0

    def test_calculate_overall_score_minimize(self):
        space = _FakeParamSpace()
        obj = _make_objective(otype=ObjectiveType.MINIMIZE)
        opt = GeneticOptimizer(space, [obj])
        score = opt._calculate_overall_score({"obj": 5.0})
        assert score == -5.0


@pytest.mark.coverage_extra
class TestGeneticOptimizer:
    """Cover GeneticOptimizer methods."""

    def test_optimize_basic(self):
        space = _FakeParamSpace()
        obj = _make_objective()
        opt = GeneticOptimizer(
            space, [obj], population_size=10, random_seed=42,
        )
        result = opt.optimize(_simple_obj_func, max_evaluations=30)
        assert result.best_parameters is not None
        assert result.best_overall_score is not None

    def test_tournament_selection(self):
        space = _FakeParamSpace()
        obj = _make_objective()
        opt = GeneticOptimizer(
            space, [obj], selection_method="tournament",
        )
        fitness = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        selected = opt._selection(fitness, 3)
        assert len(selected) == 3

    def test_roulette_selection(self):
        space = _FakeParamSpace()
        obj = _make_objective()
        opt = GeneticOptimizer(
            space, [obj], selection_method="roulette",
        )
        fitness = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        selected = opt._selection(fitness, 3)
        assert len(selected) == 3

    def test_roulette_negative_fitness(self):
        space = _FakeParamSpace()
        obj = _make_objective()
        opt = GeneticOptimizer(
            space, [obj], selection_method="roulette",
        )
        fitness = np.array([-1.0, -2.0, 3.0, 4.0, 5.0])
        selected = opt._roulette_selection(fitness, 3)
        assert len(selected) == 3

    def test_rank_selection(self):
        space = _FakeParamSpace()
        obj = _make_objective()
        opt = GeneticOptimizer(
            space, [obj], selection_method="rank",
        )
        fitness = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        selected = opt._selection(fitness, 3)
        assert len(selected) == 3

    def test_crossover(self):
        space = _FakeParamSpace()
        obj = _make_objective()
        opt = GeneticOptimizer(space, [obj])
        p1 = np.array([0.2, 0.8])
        p2 = np.array([0.7, 0.3])
        c1, c2 = opt._crossover(p1, p2)
        assert len(c1) == 2
        assert len(c2) == 2
        assert all(0.0 <= v <= 1.0 for v in c1)

    def test_mutation(self):
        space = _FakeParamSpace()
        obj = _make_objective()
        opt = GeneticOptimizer(space, [obj], mutation_rate=1.0)
        ind = np.array([0.5, 0.5])
        mutated = opt._mutation(ind)
        assert len(mutated) == 2

    def test_clip_to_bounds(self):
        space = _FakeParamSpace()
        obj = _make_objective()
        opt = GeneticOptimizer(space, [obj])
        ind = np.array([-5.0, 10.0])
        clipped = opt._clip_to_bounds(ind)
        assert clipped[0] == 0.0
        assert clipped[1] == 1.0

    def test_check_convergence_short_history(self):
        space = _FakeParamSpace()
        obj = _make_objective()
        opt = GeneticOptimizer(space, [obj])
        assert opt._check_convergence([1.0, 2.0], window=20) is False

    def test_check_convergence_converged(self):
        space = _FakeParamSpace()
        obj = _make_objective()
        opt = GeneticOptimizer(space, [obj])
        history = [5.0] * 50  # No improvement
        assert opt._check_convergence(history, window=20) is True


@pytest.mark.coverage_extra
class TestGradientOptimizer:
    """Cover GradientOptimizer methods."""

    def test_optimize_basic(self):
        space = _FakeParamSpace()
        obj = _make_objective(otype=ObjectiveType.MINIMIZE)
        opt = GradientOptimizer(space, [obj], random_seed=42)
        result = opt.optimize(_simple_obj_func, max_evaluations=20, n_restarts=2)
        assert isinstance(result, OptimizationResult)

    def test_optimize_with_constraints(self):
        space = _FakeParamSpace()
        obj = _make_objective(otype=ObjectiveType.MINIMIZE)
        constraint = OptimizationConstraint(
            name="c1",
            constraint_function=lambda x: x[0] - 0.1,
            constraint_type="ineq",
        )
        opt = GradientOptimizer(
            space, [obj], constraints=[constraint], method="SLSQP",
        )
        result = opt.optimize(_simple_obj_func, max_evaluations=20, n_restarts=1)
        assert isinstance(result, OptimizationResult)

    def test_optimize_maximize(self):
        space = _FakeParamSpace()
        obj = _make_objective(otype=ObjectiveType.MAXIMIZE)
        opt = GradientOptimizer(space, [obj], random_seed=42)
        result = opt.optimize(_simple_obj_func, max_evaluations=20, n_restarts=1)
        assert isinstance(result, OptimizationResult)


@pytest.mark.coverage_extra
class TestParetoFrontier:
    """Cover calculate_pareto_frontier."""

    def test_pareto_basic(self):
        objectives = np.array([
            [1.0, 5.0],
            [2.0, 4.0],
            [3.0, 3.0],
            [1.5, 3.5],
        ])
        is_pareto = calculate_pareto_frontier(objectives)
        assert isinstance(is_pareto, np.ndarray)
        assert len(is_pareto) == 4

    def test_pareto_single_point(self):
        objectives = np.array([[1.0, 2.0]])
        is_pareto = calculate_pareto_frontier(objectives)
        assert bool(is_pareto[0]) is True


@pytest.mark.coverage_extra
class TestHypervolumeIndicator:
    """Cover hypervolume_indicator."""

    def test_hypervolume_2d(self):
        pareto_front = np.array([
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0],
        ])
        ref_point = np.array([0.0, 0.0])
        vol = hypervolume_indicator(pareto_front, ref_point)
        assert vol > 0

    def test_hypervolume_empty(self):
        pareto_front = np.empty((0, 2))
        ref_point = np.array([0.0, 0.0])
        vol = hypervolume_indicator(pareto_front, ref_point)
        assert vol == 0.0

    def test_hypervolume_3d(self):
        pareto_front = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 2.0],
        ])
        ref_point = np.array([0.0, 0.0, 0.0])
        vol = hypervolume_indicator(pareto_front, ref_point)
        assert isinstance(vol, float)
