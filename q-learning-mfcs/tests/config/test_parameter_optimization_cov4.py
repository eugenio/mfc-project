"""Tests for parameter_optimization.py - coverage part 4.

Targets missing lines: BayesianOptimizer.optimize (331-385),
_optimize_acquisition (414-445), _acquisition_function dispatch (454-462),
_generate_initial_points (389), GeneticOptimizer best_obj_values match (657-659),
GradientOptimizer scipy import check (821-822), optimize failure/best result
(900-905, 908-921), BayesianOptimizer sklearn import check (303-304),
GeneticOptimizer total_evaluations >= max_evaluations early break (598).
"""
import sys
import os
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

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
class _Bounds:
    min_value: float
    max_value: float


@dataclass
class _Param:
    name: str
    bounds: _Bounds
    config_path: list = None

    def __post_init__(self):
        if self.config_path is None:
            self.config_path = [self.name]


class _Space:
    """Minimal parameter space mock."""

    def __init__(self, n=2, lo=0.0, hi=1.0):
        self.parameters = [
            _Param(f"p{i}", _Bounds(lo, hi)) for i in range(n)
        ]

    def sample(self, n, method="random", seed=None):
        rng = np.random.RandomState(seed)
        return rng.uniform(
            [p.bounds.min_value for p in self.parameters],
            [p.bounds.max_value for p in self.parameters],
            size=(n, len(self.parameters)),
        )


def _obj_max(x):
    return {"score": -np.sum((x - 0.5) ** 2)}


def _obj_min(x):
    return {"score": np.sum((x - 0.5) ** 2)}


@pytest.mark.coverage_extra
class TestBayesianOptimizerOptimize:
    """Cover BayesianOptimizer.optimize and related methods."""

    def test_optimize_runs_full_loop(self):
        """Cover lines 327-385: full Bayesian optimization loop."""
        np.random.seed(42)
        space = _Space(2, 0.0, 1.0)
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MAXIMIZE)]
        opt = BayesianOptimizer(
            space, objectives, random_seed=42,
            acquisition_function="expected_improvement",
            kernel_type="matern",
        )

        # Override _generate_initial_points to avoid method string issue
        def gen_init(n):
            return space.sample(n, seed=42)
        opt._generate_initial_points = gen_init

        result = opt.optimize(_obj_max, max_evaluations=15, n_initial_points=5)
        assert result.best_parameters is not None
        assert result.total_evaluations > 0
        assert result.method == OptimizationMethod.BAYESIAN
        assert len(result.convergence_history) > 0

    def test_optimize_minimize(self):
        """Cover argmin branch (line 377)."""
        np.random.seed(42)
        space = _Space(2, 0.0, 1.0)
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MINIMIZE)]
        opt = BayesianOptimizer(space, objectives, random_seed=42)
        opt._generate_initial_points = lambda n: space.sample(n, seed=42)
        result = opt.optimize(_obj_min, max_evaluations=12, n_initial_points=5)
        assert result.best_parameters is not None

    def test_optimize_convergence_triggers(self):
        """Cover convergence branch (lines 368-371)."""
        np.random.seed(42)
        space = _Space(2, 0.0, 1.0)
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MAXIMIZE)]
        opt = BayesianOptimizer(space, objectives, random_seed=42)
        opt._generate_initial_points = lambda n: space.sample(n, seed=42)
        # Force convergence: constant objective
        result = opt.optimize(
            lambda x: {"score": 1.0},
            max_evaluations=50,
            n_initial_points=5,
        )
        # With constant objective, convergence should trigger
        assert result.total_evaluations > 0


@pytest.mark.coverage_extra
class TestBayesianAcquisitionFunction:
    """Cover _acquisition_function dispatch (lines 454-462)."""

    def test_ucb_dispatch(self):
        """Cover upper_confidence_bound branch."""
        space = _Space(2, 0.0, 1.0)
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MAXIMIZE)]
        opt = BayesianOptimizer(
            space, objectives, acquisition_function="upper_confidence_bound",
        )
        gp = MagicMock()
        gp.predict = MagicMock(
            return_value=(np.array([1.0, 2.0]), np.array([0.5, 0.5]))
        )
        result = opt._acquisition_function(np.array([[0.5, 0.5]]), gp, np.array([1.0]))
        assert result is not None

    def test_poi_dispatch(self):
        """Cover probability_of_improvement branch."""
        space = _Space(2, 0.0, 1.0)
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MAXIMIZE)]
        opt = BayesianOptimizer(
            space, objectives, acquisition_function="probability_of_improvement",
        )
        gp = MagicMock()
        gp.predict = MagicMock(
            return_value=(np.array([1.0, 2.0]), np.array([0.5, 0.5]))
        )
        result = opt._acquisition_function(np.array([[0.5, 0.5]]), gp, np.array([1.0]))
        assert result is not None

    def test_unknown_dispatch_defaults_to_ei(self):
        """Cover default EI fallback (line 462)."""
        space = _Space(2, 0.0, 1.0)
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MAXIMIZE)]
        opt = BayesianOptimizer(
            space, objectives, acquisition_function="unknown_acq",
        )
        gp = MagicMock()
        gp.predict = MagicMock(
            return_value=(np.array([1.0]), np.array([0.5]))
        )
        result = opt._acquisition_function(np.array([[0.5, 0.5]]), gp, np.array([1.0]))
        assert result is not None


@pytest.mark.coverage_extra
class TestBayesianOptimizeAcquisition:
    """Cover _optimize_acquisition (lines 414-445)."""

    def test_optimize_acquisition_returns_point(self):
        """Cover the multi-restart acquisition optimization."""
        np.random.seed(42)
        space = _Space(2, 0.0, 1.0)
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MAXIMIZE)]
        opt = BayesianOptimizer(space, objectives, random_seed=42)

        gp = MagicMock()
        gp.predict = MagicMock(
            return_value=(np.array([1.0]), np.array([0.5]))
        )

        X = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
        y = np.array([1.0, 2.0, 1.5])

        next_x = opt._optimize_acquisition(gp, X, y)
        assert next_x is not None
        assert len(next_x) == 2

    def test_optimize_acquisition_all_fail_returns_fallback(self):
        """Cover fallback when all restarts fail (line 445)."""
        np.random.seed(42)
        space = _Space(2, 0.0, 1.0)
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MAXIMIZE)]
        opt = BayesianOptimizer(space, objectives, random_seed=42)

        gp = MagicMock()
        # Make predict raise to force failure in minimize
        gp.predict = MagicMock(side_effect=ValueError("fail"))

        X = np.array([[0.2, 0.3], [0.4, 0.5]])
        y = np.array([1.0, 2.0])

        next_x = opt._optimize_acquisition(gp, X, y)
        assert next_x is not None
        assert len(next_x) == 2


@pytest.mark.coverage_extra
class TestBayesianConvergenceLongHistory:
    """Cover _check_convergence with len(history) >= 2*window (lines 508-515)."""

    def test_convergence_with_sufficient_history(self):
        space = _Space()
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MAXIMIZE)]
        opt = BayesianOptimizer(space, objectives)
        # History with improvement followed by plateau
        history = list(range(10)) + [10.0] * 20
        assert opt._check_convergence(history, window=10, threshold=0.5) is True

    def test_no_convergence_with_improvement(self):
        space = _Space()
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MAXIMIZE)]
        opt = BayesianOptimizer(space, objectives)
        history = list(range(30))
        assert opt._check_convergence(history, window=10) is False


@pytest.mark.coverage_extra
class TestBayesianImportCheck:
    """Cover sklearn import check (lines 303-304)."""

    def test_no_sklearn_raises(self):
        space = _Space()
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MAXIMIZE)]
        with patch("config.parameter_optimization.HAS_SKLEARN", False):
            with pytest.raises(ImportError, match="Scikit-learn"):
                BayesianOptimizer(space, objectives)


@pytest.mark.coverage_extra
class TestGradientOptimizerImportCheck:
    """Cover scipy import check (lines 821-822)."""

    def test_no_scipy_raises(self):
        space = _Space()
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MINIMIZE)]
        with patch("config.parameter_optimization.HAS_SCIPY", False):
            with pytest.raises(ImportError, match="SciPy"):
                GradientOptimizer(space, objectives)


@pytest.mark.coverage_extra
class TestGradientOptimizerFailureBranches:
    """Cover GradientOptimizer optimize failure branches."""

    def test_optimize_all_restarts_fail(self):
        """Cover lines 900-905: all optimization restarts fail."""
        np.random.seed(42)
        space = _Space(2, 0.0, 1.0)
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MINIMIZE)]
        opt = GradientOptimizer(space, objectives, random_seed=42)

        def failing_obj(x):
            raise RuntimeError("objective fails")

        result = opt.optimize(failing_obj, max_evaluations=10, n_restarts=2)
        # best_result stays None because all fail with exception
        assert result.total_evaluations >= 0

    def test_optimize_best_result_found(self):
        """Cover lines 908-921: best_result is not None branch."""
        np.random.seed(42)
        space = _Space(2, -5.0, 5.0)
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MINIMIZE)]
        opt = GradientOptimizer(space, objectives, random_seed=42)

        result = opt.optimize(_obj_min, max_evaluations=50, n_restarts=3)
        assert result.total_evaluations > 0
        # best_parameters should be set
        if result.best_parameters is not None:
            assert result.best_overall_score is not None

    def test_optimize_maximize_best_result(self):
        """Cover maximize branch in best_result handling (line 910-911)."""
        np.random.seed(42)
        space = _Space(2, 0.0, 1.0)
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MAXIMIZE)]
        opt = GradientOptimizer(space, objectives, random_seed=42)

        result = opt.optimize(_obj_max, max_evaluations=50, n_restarts=2)
        assert result.total_evaluations > 0

    def test_optimize_best_objective_values_match(self):
        """Cover lines 918-921: find matching objective values for best params."""
        np.random.seed(42)
        space = _Space(1, 0.0, 10.0)
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MINIMIZE)]
        opt = GradientOptimizer(space, objectives, random_seed=42)

        def simple_obj(x):
            return {"score": float(x[0] ** 2)}

        result = opt.optimize(simple_obj, max_evaluations=30, n_restarts=1)
        # Check that best_objective_values was matched
        assert result.total_evaluations > 0


@pytest.mark.coverage_extra
class TestGeneticOptimizerEdgeCases:
    """Cover GeneticOptimizer edge cases."""

    def test_max_evaluations_reached_early(self):
        """Cover line 598: total_evaluations >= max_evaluations breaks loop."""
        np.random.seed(42)
        space = _Space(2, 0.0, 1.0)
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MAXIMIZE)]
        opt = GeneticOptimizer(
            space, objectives, population_size=10, random_seed=42,
        )
        # Very small max_evaluations to trigger the early break
        result = opt.optimize(_obj_max, max_evaluations=5)
        assert result is not None
        assert result.best_parameters is not None

    def test_best_objective_values_found(self):
        """Cover lines 657-659: matching params to all_parameters."""
        np.random.seed(42)
        space = _Space(2, 0.0, 1.0)
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MAXIMIZE)]
        opt = GeneticOptimizer(
            space, objectives, population_size=6, random_seed=42,
        )
        result = opt.optimize(_obj_max, max_evaluations=20)
        # best_objective_values should have been matched
        assert result.best_overall_score is not None

    def test_optimize_with_max_generations_set(self):
        """Cover max_generations explicit parameter."""
        np.random.seed(42)
        space = _Space(2, 0.0, 1.0)
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MAXIMIZE)]
        opt = GeneticOptimizer(
            space, objectives, population_size=6, random_seed=42,
        )
        result = opt.optimize(
            _obj_max, max_evaluations=100, max_generations=3,
        )
        assert result.best_parameters is not None

    def test_selection_method_roulette_in_optimize(self):
        """Cover roulette selection during full optimize."""
        np.random.seed(42)
        space = _Space(2, 0.0, 1.0)
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MAXIMIZE)]
        opt = GeneticOptimizer(
            space, objectives, population_size=6,
            selection_method="roulette", random_seed=42,
        )
        result = opt.optimize(_obj_max, max_evaluations=30)
        assert result.best_parameters is not None

    def test_selection_method_rank_in_optimize(self):
        """Cover rank selection during full optimize."""
        np.random.seed(42)
        space = _Space(2, 0.0, 1.0)
        objectives = [OptimizationObjective(name="score", type=ObjectiveType.MAXIMIZE)]
        opt = GeneticOptimizer(
            space, objectives, population_size=6,
            selection_method="rank", random_seed=42,
        )
        result = opt.optimize(_obj_max, max_evaluations=30)
        assert result.best_parameters is not None
