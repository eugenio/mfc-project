"""Deep coverage tests for ml_optimization page module."""
import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_DIR = str((Path(__file__).resolve().parent / ".." / ".." / "src").resolve())
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_ML_OPT_PATH = str(
    (Path(__file__).resolve().parent / ".." / ".." / "src" / "gui" / "pages" / "ml_optimization.py").resolve()
)


class _SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None


def _make_mock_st():
    mock_st = MagicMock()
    mock_st.session_state = _SessionState()

    def _smart_columns(n_or_spec):
        n = len(n_or_spec) if isinstance(n_or_spec, list | tuple) else int(n_or_spec)
        cols = []
        for _ in range(n):
            col = MagicMock()
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=False)
            cols.append(col)
        return cols

    mock_st.columns.side_effect = _smart_columns

    def _smart_tabs(labels):
        tabs = []
        for _ in labels:
            tab = MagicMock()
            tab.__enter__ = MagicMock(return_value=tab)
            tab.__exit__ = MagicMock(return_value=False)
            tabs.append(tab)
        return tabs

    mock_st.tabs.side_effect = _smart_tabs

    exp = MagicMock()
    exp.__enter__ = MagicMock(return_value=exp)
    exp.__exit__ = MagicMock(return_value=False)
    mock_st.expander.return_value = exp

    mock_st.sidebar = MagicMock()

    form = MagicMock()
    form.__enter__ = MagicMock(return_value=form)
    form.__exit__ = MagicMock(return_value=False)
    mock_st.form.return_value = form

    spinner = MagicMock()
    spinner.__enter__ = MagicMock(return_value=spinner)
    spinner.__exit__ = MagicMock(return_value=False)
    mock_st.spinner.return_value = spinner

    container = MagicMock()
    container.__enter__ = MagicMock(return_value=container)
    container.__exit__ = MagicMock(return_value=False)
    mock_st.container.return_value = container

    status = MagicMock()
    status.__enter__ = MagicMock(return_value=status)
    status.__exit__ = MagicMock(return_value=False)
    mock_st.status.return_value = status

    progress = MagicMock()
    progress.progress = MagicMock()
    progress.empty = MagicMock()
    mock_st.progress.return_value = progress

    return mock_st


_MOD_NAME = "gui.pages.ml_optimization"

# Load the module once at import time to avoid the C extension re-import issue,
# then just re-assign st for each test.
_loaded_mod = None


def _ensure_module_loaded():
    """Load the module once on first use."""
    global _loaded_mod  # noqa: PLW0603
    if _loaded_mod is not None:
        return _loaded_mod

    mock_st_init = MagicMock()
    spec = importlib.util.spec_from_file_location(_MOD_NAME, _ML_OPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MOD_NAME] = mod
    old_st = sys.modules.get("streamlit")
    sys.modules["streamlit"] = mock_st_init
    try:
        spec.loader.exec_module(mod)
    finally:
        if old_st is not None:
            sys.modules["streamlit"] = old_st
        else:
            sys.modules.pop("streamlit", None)
    _loaded_mod = mod
    return mod


def _load_module(mock_st):
    """Get the ml_optimization module with st pointing to mock_st."""
    mod = _ensure_module_loaded()
    mod.st = mock_st
    return mod


# ---------------------------------------------------------------------------
# Helper: common parameters and objectives for optimization tests
# ---------------------------------------------------------------------------
_TEST_PARAMS = {"conductivity": (100.0, 10000.0), "surface_area": (1.0, 50.0)}
_TEST_OBJECTIVES = ["power_density", "cost"]


# ===================================================================
# MLOptimizer class tests
# ===================================================================
@pytest.mark.apptest
class TestMLOptimizerInit:
    """Test MLOptimizer initialization."""

    def test_init_sets_defaults(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        assert opt.optimization_active is False
        assert opt.current_iteration == 0
        assert opt.history == []


@pytest.mark.apptest
class TestMLOptimizerRunOptimization:
    """Test the run_optimization dispatch method."""

    def test_run_bayesian(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        result = opt.run_optimization(
            mod.OptimizationMethod.BAYESIAN,
            _TEST_OBJECTIVES,
            _TEST_PARAMS,
            max_iterations=2,
        )
        assert result.success is True
        assert result.method == mod.OptimizationMethod.BAYESIAN
        assert result.execution_time is not None
        assert opt.optimization_active is False

    def test_run_nsga_ii(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        result = opt.run_optimization(
            mod.OptimizationMethod.NSGA_II,
            _TEST_OBJECTIVES,
            _TEST_PARAMS,
            max_iterations=2,
        )
        assert result.success is True
        assert result.method == mod.OptimizationMethod.NSGA_II

    def test_run_neural_surrogate(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        result = opt.run_optimization(
            mod.OptimizationMethod.NEURAL_SURROGATE,
            _TEST_OBJECTIVES,
            _TEST_PARAMS,
            max_iterations=2,
        )
        assert result.success is True
        assert result.method == mod.OptimizationMethod.NEURAL_SURROGATE

    def test_run_q_learning(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        result = opt.run_optimization(
            mod.OptimizationMethod.Q_LEARNING,
            _TEST_OBJECTIVES,
            _TEST_PARAMS,
            max_iterations=2,
        )
        assert result.success is True
        assert result.method == mod.OptimizationMethod.Q_LEARNING

    def test_run_optimization_exception_handling(self):
        """Test that exceptions are caught and returned as failed results."""
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        # Force an exception by making progress raise
        mock_st.progress.side_effect = RuntimeError("test error")
        result = opt.run_optimization(
            mod.OptimizationMethod.BAYESIAN,
            _TEST_OBJECTIVES,
            _TEST_PARAMS,
            max_iterations=2,
        )
        assert result.success is False
        assert "test error" in result.error_message
        assert opt.optimization_active is False

    def test_run_unknown_method_raises(self):
        """Unknown method value triggers the else branch and ValueError."""
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        # Create a fake method that doesn't match any known OptimizationMethod
        fake_method = MagicMock()
        fake_method.value = "Fake Method"
        # The ValueError is caught by the except block and returned as failed
        result = opt.run_optimization(
            fake_method,
            _TEST_OBJECTIVES,
            _TEST_PARAMS,
            max_iterations=2,
        )
        assert result.success is False
        assert "Unknown optimization method" in result.error_message

    def test_progress_bar_empty_called(self):
        """Test that progress_bar.empty() is called on success."""
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        result = opt.run_optimization(
            mod.OptimizationMethod.BAYESIAN,
            _TEST_OBJECTIVES,
            _TEST_PARAMS,
            max_iterations=2,
        )
        assert result.success is True
        progress_bar = mock_st.progress.return_value
        progress_bar.empty.assert_called()


@pytest.mark.apptest
class TestBayesianOptimization:
    """Test _run_bayesian_optimization in detail."""

    def test_bayesian_single_iteration(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        progress_bar = MagicMock()
        result = opt._run_bayesian_optimization(
            _TEST_OBJECTIVES, _TEST_PARAMS, 1, progress_bar
        )
        assert result.success is True
        assert result.iterations == 1
        assert result.best_parameters is not None
        assert result.optimization_history is not None

    def test_bayesian_multi_iteration_gp_sampling(self):
        """Multiple iterations exercise the GP-guided sampling branch."""
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        progress_bar = MagicMock()
        result = opt._run_bayesian_optimization(
            _TEST_OBJECTIVES, _TEST_PARAMS, 3, progress_bar
        )
        assert result.success is True
        assert result.iterations == 3
        assert len(result.optimization_history) == 3

    def test_bayesian_history_has_acquisition(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        progress_bar = MagicMock()
        result = opt._run_bayesian_optimization(
            _TEST_OBJECTIVES, _TEST_PARAMS, 2, progress_bar
        )
        assert "acquisition" in result.optimization_history.columns


@pytest.mark.apptest
class TestNSGAII:
    """Test _run_nsga_ii in detail."""

    def test_nsga_ii_basic(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        progress_bar = MagicMock()
        result = opt._run_nsga_ii(
            _TEST_OBJECTIVES, _TEST_PARAMS, 2, progress_bar
        )
        assert result.success is True
        assert result.method == mod.OptimizationMethod.NSGA_II
        assert result.pareto_front is not None

    def test_nsga_ii_with_no_pareto_solutions(self):
        """Test the else branch when pareto_solutions is empty."""
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        progress_bar = MagicMock()
        with patch.object(opt, "_non_dominated_sort", return_value=[[]]):
            result = opt._run_nsga_ii(
                _TEST_OBJECTIVES, _TEST_PARAMS, 1, progress_bar
            )
        assert result.success is True
        assert result.best_parameters is not None


@pytest.mark.apptest
class TestNeuralSurrogate:
    """Test _run_neural_surrogate in detail."""

    def test_neural_surrogate_data_collection_phase(self):
        """With few iterations, stays in data collection phase."""
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        progress_bar = MagicMock()
        result = opt._run_neural_surrogate(
            _TEST_OBJECTIVES, _TEST_PARAMS, 3, progress_bar
        )
        assert result.success is True
        assert all(
            row == "collection"
            for row in result.optimization_history["phase"]
        )

    def test_neural_surrogate_optimization_phase(self):
        """With enough iterations, enters optimization phase."""
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        progress_bar = MagicMock()
        result = opt._run_neural_surrogate(
            _TEST_OBJECTIVES, _TEST_PARAMS, 12, progress_bar
        )
        assert result.success is True
        phases = result.optimization_history["phase"].tolist()
        assert "optimization" in phases
        assert "collection" in phases

    def test_neural_surrogate_no_best_params_branch(self):
        """Test the else branch in neural network guided sampling."""
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        progress_bar = MagicMock()
        with patch.object(
            opt,
            "_evaluate_objectives",
            return_value={"power_density": float("inf"), "cost": float("inf")},
        ):
            result = opt._run_neural_surrogate(
                _TEST_OBJECTIVES, _TEST_PARAMS, 12, progress_bar
            )
        assert result.success is True


@pytest.mark.apptest
class TestQLearning:
    """Test _run_q_learning in detail."""

    def test_q_learning_basic(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        progress_bar = MagicMock()
        result = opt._run_q_learning(
            _TEST_OBJECTIVES, _TEST_PARAMS, 3, progress_bar
        )
        assert result.success is True
        assert result.method == mod.OptimizationMethod.Q_LEARNING
        assert "episode" in result.optimization_history.columns
        assert "reward" in result.optimization_history.columns
        assert "epsilon" in result.optimization_history.columns

    def test_q_learning_exploit_with_best_params(self):
        """Ensure the exploit branch (epsilon < random) is exercised."""
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        progress_bar = MagicMock()
        import numpy as np

        random_vals = iter([0.5, 0.99, 0.99])
        with patch.object(np.random, "random", side_effect=lambda: next(random_vals)):
            result = opt._run_q_learning(
                _TEST_OBJECTIVES, _TEST_PARAMS, 3, progress_bar
            )
        assert result.success is True

    def test_q_learning_exploit_no_best_params(self):
        """Exercise the else branch: exploit but no best_params yet."""
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        progress_bar = MagicMock()
        import numpy as np

        random_vals = iter([0.99, 0.5, 0.5])
        with patch.object(np.random, "random", side_effect=lambda: next(random_vals)):
            result = opt._run_q_learning(
                _TEST_OBJECTIVES, _TEST_PARAMS, 3, progress_bar
            )
        assert result.success is True

    def test_q_learning_epsilon_decay(self):
        """Verify epsilon decays over episodes."""
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        progress_bar = MagicMock()
        result = opt._run_q_learning(
            _TEST_OBJECTIVES, _TEST_PARAMS, 5, progress_bar
        )
        epsilons = result.optimization_history["epsilon"].tolist()
        for i in range(1, len(epsilons)):
            assert epsilons[i] <= epsilons[i - 1]


@pytest.mark.apptest
class TestEvaluateObjectives:
    """Test _evaluate_objectives with all objective types."""

    def test_power_density_objective(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        params = {"conductivity": 1000.0, "surface_area": 10.0}
        result = opt._evaluate_objectives(params, ["power_density"])
        assert "power_density" in result
        assert isinstance(result["power_density"], float)

    def test_treatment_efficiency_objective(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        params = {"flow_rate": 1e-4, "biofilm_thickness": 100.0}
        result = opt._evaluate_objectives(params, ["treatment_efficiency"])
        assert "treatment_efficiency" in result

    def test_cost_objective(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        params = {"electrode_area": 10.0, "flow_rate": 1e-4}
        result = opt._evaluate_objectives(params, ["cost"])
        assert "cost" in result

    def test_stability_objective(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        params = {"ph": 7.0, "temperature": 25.0}
        result = opt._evaluate_objectives(params, ["stability"])
        assert "stability" in result

    def test_all_objectives(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        params = {
            "conductivity": 1000.0,
            "surface_area": 10.0,
            "flow_rate": 1e-4,
            "biofilm_thickness": 100.0,
            "electrode_area": 10.0,
            "ph": 7.0,
            "temperature": 25.0,
        }
        objectives = ["power_density", "treatment_efficiency", "cost", "stability"]
        result = opt._evaluate_objectives(params, objectives)
        assert len(result) == 4
        for obj in objectives:
            assert obj in result

    def test_unknown_objective_returns_empty(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        result = opt._evaluate_objectives({}, ["unknown_objective"])
        assert "unknown_objective" not in result

    def test_objectives_with_default_params(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        result = opt._evaluate_objectives(
            {}, ["power_density", "treatment_efficiency", "cost", "stability"]
        )
        assert len(result) == 4


@pytest.mark.apptest
class TestNonDominatedSort:
    """Test _non_dominated_sort."""

    def test_single_element(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        pop = [{"objectives": {"a": 1.0, "b": 2.0}}]
        fronts = opt._non_dominated_sort(pop, ["a", "b"])
        assert len(fronts) >= 1
        assert len(fronts[0]) == 1

    def test_two_non_dominating(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        pop = [
            {"objectives": {"a": 1.0, "b": 3.0}},
            {"objectives": {"a": 3.0, "b": 1.0}},
        ]
        fronts = opt._non_dominated_sort(pop, ["a", "b"])
        assert len(fronts[0]) == 2

    def test_one_dominates_other(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        pop = [
            {"objectives": {"a": 1.0, "b": 1.0}},
            {"objectives": {"a": 2.0, "b": 2.0}},
        ]
        fronts = opt._non_dominated_sort(pop, ["a", "b"])
        assert len(fronts) >= 2
        assert len(fronts[0]) == 1
        assert fronts[0][0]["objectives"]["a"] == 1.0

    def test_multiple_fronts(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        pop = [
            {"objectives": {"a": 1.0, "b": 1.0}},
            {"objectives": {"a": 2.0, "b": 2.0}},
            {"objectives": {"a": 3.0, "b": 3.0}},
        ]
        fronts = opt._non_dominated_sort(pop, ["a", "b"])
        assert len(fronts) >= 2


@pytest.mark.apptest
class TestDominates:
    """Test _dominates method."""

    def test_dominates_true(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        s1 = {"objectives": {"a": 1.0, "b": 1.0}}
        s2 = {"objectives": {"a": 2.0, "b": 2.0}}
        assert opt._dominates(s1, s2, ["a", "b"]) is True

    def test_dominates_false_worse_in_one(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        s1 = {"objectives": {"a": 1.0, "b": 3.0}}
        s2 = {"objectives": {"a": 2.0, "b": 2.0}}
        assert opt._dominates(s1, s2, ["a", "b"]) is False

    def test_dominates_false_equal(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        s1 = {"objectives": {"a": 1.0, "b": 1.0}}
        s2 = {"objectives": {"a": 1.0, "b": 1.0}}
        assert opt._dominates(s1, s2, ["a", "b"]) is False

    def test_dominates_better_in_one_equal_other(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        s1 = {"objectives": {"a": 1.0, "b": 2.0}}
        s2 = {"objectives": {"a": 1.0, "b": 3.0}}
        assert opt._dominates(s1, s2, ["a", "b"]) is True


@pytest.mark.apptest
class TestGenerateNextPopulation:
    """Test _generate_next_population."""

    def test_fills_from_fronts(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        fronts = [
            [{"parameters": {"x": 1.0}}, {"parameters": {"x": 2.0}}],
            [{"parameters": {"x": 3.0}}],
        ]
        params = {"x": (0.0, 10.0)}
        pop = opt._generate_next_population(fronts, 3, params)
        assert len(pop) == 3

    def test_fills_remaining_with_random(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        fronts = [[{"parameters": {"x": 1.0}}]]
        params = {"x": (0.0, 10.0)}
        pop = opt._generate_next_population(fronts, 5, params)
        assert len(pop) == 5

    def test_partial_front_selection(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        fronts = [
            [{"parameters": {"x": float(i)}} for i in range(10)],
        ]
        params = {"x": (0.0, 10.0)}
        pop = opt._generate_next_population(fronts, 3, params)
        assert len(pop) == 3


@pytest.mark.apptest
class TestDiscretizeParameters:
    """Test _discretize_parameters."""

    def test_no_values_returns_initial_state(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        result = opt._discretize_parameters({"x": (0.0, 10.0)})
        assert result == "initial_state"

    def test_with_values(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        result = opt._discretize_parameters(
            {"x": (0.0, 10.0), "y": (0.0, 100.0)},
            {"x": 5.0, "y": 50.0},
        )
        assert "x:" in result
        assert "y:" in result
        assert "_" in result

    def test_edge_value_at_max(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        result = opt._discretize_parameters(
            {"x": (0.0, 10.0)},
            {"x": 10.0},
        )
        assert "x:9" in result

    def test_edge_value_at_min(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        result = opt._discretize_parameters(
            {"x": (0.0, 10.0)},
            {"x": 0.0},
        )
        assert "x:0" in result

    def test_missing_value_uses_min(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        result = opt._discretize_parameters(
            {"x": (0.0, 10.0)},
            {},
        )
        assert "x:0" in result


@pytest.mark.apptest
class TestUpdateQTable:
    """Test _update_q_table."""

    def test_new_states_initialized(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        q_table = {}
        opt._update_q_table(q_table, "s1", "s2", 1.0, 0.1, 0.95)
        assert "s1" in q_table
        assert "s2" in q_table

    def test_q_value_updated(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        q_table = {"s1": 0.0, "s2": 0.0}
        opt._update_q_table(q_table, "s1", "s2", 1.0, 0.1, 0.95)
        assert abs(q_table["s1"] - 0.1) < 1e-9

    def test_existing_q_values(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        q_table = {"s1": 5.0, "s2": 3.0}
        opt._update_q_table(q_table, "s1", "s2", 2.0, 0.1, 0.95)
        assert abs(q_table["s1"] - 4.985) < 1e-9

    def test_only_state_not_in_table(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        q_table = {"s2": 1.0}
        opt._update_q_table(q_table, "s1", "s2", 0.5, 0.2, 0.9)
        assert "s1" in q_table

    def test_only_next_state_not_in_table(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        q_table = {"s1": 1.0}
        opt._update_q_table(q_table, "s1", "s2", 0.5, 0.2, 0.9)
        assert "s2" in q_table


# ===================================================================
# create_optimization_visualizations tests
# ===================================================================
@pytest.mark.apptest
class TestCreateOptimizationVisualizations:
    """Test create_optimization_visualizations function."""

    def test_failed_result_shows_error(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        result = mod.OptimizationResult(
            success=False,
            method=mod.OptimizationMethod.BAYESIAN,
            error_message="test failure",
        )
        mod.create_optimization_visualizations(result)
        mock_st.error.assert_called_once()

    def test_successful_result_no_history(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        result = mod.OptimizationResult(
            success=True,
            method=mod.OptimizationMethod.BAYESIAN,
            best_parameters={"x": 1.0},
            optimization_history=None,
        )
        mod.create_optimization_visualizations(result)
        mock_st.plotly_chart.assert_not_called()

    def test_successful_result_with_history(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        import pandas as pd

        history_df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "power_density": [-0.5, -0.6, -0.7],
            "cost": [0.3, 0.2, 0.1],
            "iteration": [1, 2, 3],
            "acquisition": [0.5, 0.6, 0.7],
        })
        result = mod.OptimizationResult(
            success=True,
            method=mod.OptimizationMethod.BAYESIAN,
            best_parameters={"x": 3.0},
            best_objectives={"power_density": -0.7, "cost": 0.1},
            optimization_history=history_df,
        )
        mod.create_optimization_visualizations(result)
        mock_st.subheader.assert_called()
        mock_st.plotly_chart.assert_called()

    def test_successful_result_empty_history(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        import pandas as pd

        result = mod.OptimizationResult(
            success=True,
            method=mod.OptimizationMethod.BAYESIAN,
            best_parameters={"x": 1.0},
            optimization_history=pd.DataFrame(),
        )
        mod.create_optimization_visualizations(result)
        mock_st.plotly_chart.assert_not_called()

    def test_with_pareto_front(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        import pandas as pd

        history_df = pd.DataFrame({
            "x": [1.0, 2.0],
            "power_density": [-0.5, -0.6],
            "cost": [0.3, 0.2],
            "generation": [1, 2],
        })
        pareto_df = pd.DataFrame({
            "power_density": [-0.5, -0.6, -0.7],
            "cost": [0.3, 0.2, 0.1],
            "generation": [1, 2, 3],
        })
        result = mod.OptimizationResult(
            success=True,
            method=mod.OptimizationMethod.NSGA_II,
            best_parameters={"x": 1.0},
            best_objectives={"power_density": -0.7, "cost": 0.1},
            optimization_history=history_df,
            pareto_front=pareto_df,
        )
        mod.create_optimization_visualizations(result)
        assert mock_st.plotly_chart.call_count >= 1

    def test_with_empty_pareto_front(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        import pandas as pd

        result = mod.OptimizationResult(
            success=True,
            method=mod.OptimizationMethod.NSGA_II,
            best_parameters={"x": 1.0},
            optimization_history=pd.DataFrame(),
            pareto_front=pd.DataFrame(),
        )
        mod.create_optimization_visualizations(result)

    def test_pareto_front_single_objective_col(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        import pandas as pd

        pareto_df = pd.DataFrame({
            "x": [1.0, 2.0],
            "power_density": [-0.5, -0.6],
            "generation": [1, 2],
        })
        history_df = pd.DataFrame({
            "x": [1.0, 2.0],
            "power_density": [-0.5, -0.6],
            "iteration": [1, 2],
        })
        result = mod.OptimizationResult(
            success=True,
            method=mod.OptimizationMethod.NSGA_II,
            best_parameters={"x": 1.0},
            optimization_history=history_df,
            pareto_front=pareto_df,
        )
        mod.create_optimization_visualizations(result)

    def test_history_with_no_param_cols(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        import pandas as pd

        history_df = pd.DataFrame({
            "power_density": [-0.5, -0.6],
            "cost": [0.3, 0.2],
            "iteration": [1, 2],
        })
        result = mod.OptimizationResult(
            success=True,
            method=mod.OptimizationMethod.BAYESIAN,
            best_parameters={"z_not_in_cols": 1.0},
            best_objectives={"power_density": -0.6, "cost": 0.2},
            optimization_history=history_df,
        )
        mod.create_optimization_visualizations(result)

    def test_history_no_obj_cols(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        import pandas as pd

        history_df = pd.DataFrame({
            "x": [1.0, 2.0],
            "iteration": [1, 2],
        })
        result = mod.OptimizationResult(
            success=True,
            method=mod.OptimizationMethod.BAYESIAN,
            best_parameters={"x": 2.0},
            optimization_history=history_df,
        )
        mod.create_optimization_visualizations(result)


# ===================================================================
# render_ml_optimization_page tests
# ===================================================================
@pytest.mark.apptest
class TestRenderMLOptimizationPage:
    """Test render_ml_optimization_page function."""

    def _setup_st_for_render(self, mock_st, method_value="Bayesian Optimization",
                              objectives=None, selected_params=None,
                              max_iterations=50, run_button=False,
                              real_time_plots=False):
        if objectives is None:
            objectives = ["power_density", "cost"]
        if selected_params is None:
            selected_params = ["conductivity", "surface_area", "flow_rate"]

        mock_st.radio.return_value = method_value

        multiselect_calls = [objectives, selected_params]
        multiselect_iter = iter(multiselect_calls)
        mock_st.multiselect.side_effect = lambda *args, **kwargs: next(multiselect_iter)

        mock_st.number_input.return_value = max_iterations
        mock_st.selectbox.return_value = "Expected Improvement"
        mock_st.slider.return_value = 0.9

        checkbox_returns = iter([True, True, True, real_time_plots, True])
        mock_st.checkbox.side_effect = lambda *args, **kwargs: next(checkbox_returns, True)

        button_returns = iter([run_button, False, False, False])
        mock_st.button.side_effect = lambda *args, **kwargs: next(button_returns, False)

        return mock_st

    def test_render_basic_no_run(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(mock_st)
        mod.render_ml_optimization_page()
        mock_st.title.assert_called_once()
        mock_st.info.assert_called()

    def test_render_no_objectives(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(mock_st, objectives=[])
        mod.render_ml_optimization_page()
        mock_st.warning.assert_called()

    def test_render_no_params(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(mock_st, selected_params=[])
        mod.render_ml_optimization_page()
        mock_st.warning.assert_called()

    def test_render_bayesian_settings(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(mock_st, method_value="Bayesian Optimization")
        mod.render_ml_optimization_page()
        mock_st.title.assert_called_once()

    def test_render_nsga_ii_settings(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(mock_st, method_value="Multi-Objective (NSGA-II)")
        mod.render_ml_optimization_page()
        mock_st.title.assert_called_once()

    def test_render_neural_surrogate_settings(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(mock_st, method_value="Neural Network Surrogate")
        mod.render_ml_optimization_page()
        mock_st.title.assert_called_once()

    def test_render_q_learning_settings(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(mock_st, method_value="Q-Learning Reinforcement")
        mod.render_ml_optimization_page()
        mock_st.title.assert_called_once()

    def test_render_run_optimization_success(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(mock_st, run_button=True, max_iterations=2)
        mod.render_ml_optimization_page()
        mock_st.success.assert_called()

    def test_render_run_optimization_with_real_time_plots(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(
            mock_st, run_button=True, max_iterations=2, real_time_plots=True
        )
        checkbox_vals = iter([True, True, True, True, True])
        mock_st.checkbox.side_effect = lambda *a, **kw: next(checkbox_vals, True)
        mod.render_ml_optimization_page()
        mock_st.empty.assert_called()

    def test_render_run_optimization_failure(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(mock_st, run_button=True, max_iterations=2)
        mock_st.progress.side_effect = RuntimeError("forced failure")
        mod.render_ml_optimization_page()
        mock_st.error.assert_called()

    def test_render_run_with_download_params(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(mock_st, run_button=True, max_iterations=2)
        button_vals = iter([True, True, False, False])
        mock_st.button.side_effect = lambda *a, **kw: next(button_vals, False)
        mod.render_ml_optimization_page()

    def test_render_run_with_download_history(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(mock_st, run_button=True, max_iterations=2)
        button_vals = iter([True, False, True, False])
        mock_st.button.side_effect = lambda *a, **kw: next(button_vals, False)
        mod.render_ml_optimization_page()

    def test_render_run_with_generate_report(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(mock_st, run_button=True, max_iterations=2)
        button_vals = iter([True, False, False, True])
        mock_st.button.side_effect = lambda *a, **kw: next(button_vals, False)
        mod.render_ml_optimization_page()

    def test_render_param_bounds_with_small_values(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(mock_st, selected_params=["flow_rate"])
        mod.render_ml_optimization_page()

    def test_render_param_bounds_with_large_values(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(mock_st, selected_params=["conductivity", "surface_area"])
        mod.render_ml_optimization_page()

    def test_render_run_best_params_display_small_values(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(
            mock_st, run_button=True, max_iterations=2, selected_params=["flow_rate"]
        )
        mod.render_ml_optimization_page()

    def test_render_run_no_best_objectives(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(mock_st, run_button=True, max_iterations=2)
        with patch.object(
            mod.MLOptimizer,
            "run_optimization",
            return_value=mod.OptimizationResult(
                success=True,
                method=mod.OptimizationMethod.BAYESIAN,
                best_parameters={"conductivity": 500.0},
                best_objectives=None,
                iterations=2,
                execution_time=0.5,
            ),
        ):
            mod.render_ml_optimization_page()

    def test_render_run_no_best_parameters(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(mock_st, run_button=True, max_iterations=2)
        with patch.object(
            mod.MLOptimizer,
            "run_optimization",
            return_value=mod.OptimizationResult(
                success=True,
                method=mod.OptimizationMethod.BAYESIAN,
                best_parameters=None,
                best_objectives={"power_density": -0.5},
                iterations=2,
                execution_time=0.5,
            ),
        ):
            mod.render_ml_optimization_page()

    def test_render_expander_info_panel(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        self._setup_st_for_render(mock_st)
        mod.render_ml_optimization_page()
        mock_st.expander.assert_called()
        mock_st.markdown.assert_called()


@pytest.mark.apptest
class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_default_values(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        result = mod.OptimizationResult(
            success=True,
            method=mod.OptimizationMethod.BAYESIAN,
        )
        assert result.best_parameters is None
        assert result.best_objectives is None
        assert result.optimization_history is None
        assert result.convergence_data is None
        assert result.pareto_front is None
        assert result.execution_time is None
        assert result.iterations == 0
        assert result.error_message is None

    def test_all_values(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        import pandas as pd
        result = mod.OptimizationResult(
            success=False,
            method=mod.OptimizationMethod.NSGA_II,
            best_parameters={"x": 1.0},
            best_objectives={"y": 2.0},
            optimization_history=pd.DataFrame(),
            convergence_data=pd.DataFrame(),
            pareto_front=pd.DataFrame(),
            execution_time=1.5,
            iterations=10,
            error_message="err",
        )
        assert result.success is False
        assert result.iterations == 10
        assert result.error_message == "err"


@pytest.mark.apptest
class TestOptimizationMethodEnum:
    """Test OptimizationMethod enum."""

    def test_all_methods(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        assert mod.OptimizationMethod.BAYESIAN.value == "Bayesian Optimization"
        assert mod.OptimizationMethod.NSGA_II.value == "Multi-Objective (NSGA-II)"
        assert mod.OptimizationMethod.NEURAL_SURROGATE.value == "Neural Network Surrogate"
        assert mod.OptimizationMethod.Q_LEARNING.value == "Q-Learning Reinforcement"

    def test_enum_from_value(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        method = mod.OptimizationMethod("Bayesian Optimization")
        assert method == mod.OptimizationMethod.BAYESIAN


# ===================================================================
# Edge case and integration tests
# ===================================================================
@pytest.mark.apptest
class TestEdgeCases:
    """Edge cases and additional branch coverage."""

    def test_render_all_methods_advanced_settings(self):
        methods = [
            "Bayesian Optimization",
            "Multi-Objective (NSGA-II)",
            "Neural Network Surrogate",
            "Q-Learning Reinforcement",
        ]
        for method_val in methods:
            mock_st = _make_mock_st()
            mod = _load_module(mock_st)
            mock_st.radio.return_value = method_val
            _multiselect_calls = iter([["power_density"], ["conductivity"]])
            mock_st.multiselect.side_effect = lambda *a, _mc=_multiselect_calls, **kw: next(_mc)
            mock_st.number_input.return_value = 10
            mock_st.selectbox.return_value = "Expected Improvement"
            mock_st.slider.return_value = 0.5
            mock_st.checkbox.return_value = True
            mock_st.button.return_value = False
            mod.render_ml_optimization_page()
            mock_st.title.assert_called_once()

    def test_render_run_nsga_ii_method(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        mock_st.radio.return_value = "Multi-Objective (NSGA-II)"
        multiselect_calls = iter([["power_density", "cost"], ["conductivity", "surface_area"]])
        mock_st.multiselect.side_effect = lambda *a, **kw: next(multiselect_calls)
        mock_st.number_input.return_value = 2
        mock_st.selectbox.return_value = "Expected Improvement"
        mock_st.slider.return_value = 0.9
        mock_st.checkbox.return_value = True
        button_vals = iter([True, False, False, False])
        mock_st.button.side_effect = lambda *a, **kw: next(button_vals, False)
        mod.render_ml_optimization_page()

    def test_render_run_neural_surrogate_method(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        mock_st.radio.return_value = "Neural Network Surrogate"
        multiselect_calls = iter([["power_density"], ["conductivity"]])
        mock_st.multiselect.side_effect = lambda *a, **kw: next(multiselect_calls)
        mock_st.number_input.return_value = 2
        mock_st.selectbox.return_value = "MLP"
        mock_st.slider.return_value = 0.5
        mock_st.checkbox.return_value = True
        button_vals = iter([True, False, False, False])
        mock_st.button.side_effect = lambda *a, **kw: next(button_vals, False)
        mod.render_ml_optimization_page()

    def test_render_run_q_learning_method(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        mock_st.radio.return_value = "Q-Learning Reinforcement"
        multiselect_calls = iter([["power_density"], ["conductivity"]])
        mock_st.multiselect.side_effect = lambda *a, **kw: next(multiselect_calls)
        mock_st.number_input.return_value = 2
        mock_st.selectbox.return_value = "Expected Improvement"
        mock_st.slider.return_value = 0.1
        mock_st.checkbox.return_value = True
        button_vals = iter([True, False, False, False])
        mock_st.button.side_effect = lambda *a, **kw: next(button_vals, False)
        mod.render_ml_optimization_page()

    def test_render_many_params_columns(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        mock_st.radio.return_value = "Bayesian Optimization"
        many_params = ["conductivity", "surface_area", "flow_rate", "ph", "temperature"]
        multiselect_calls = iter([["power_density"], many_params])
        mock_st.multiselect.side_effect = lambda *a, **kw: next(multiselect_calls)
        mock_st.number_input.return_value = 10
        mock_st.selectbox.return_value = "RBF"
        mock_st.slider.return_value = 0.5
        mock_st.checkbox.return_value = True
        mock_st.button.return_value = False
        mod.render_ml_optimization_page()

    def test_render_best_params_with_param_in_available_and_not(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        mock_st.radio.return_value = "Bayesian Optimization"
        multiselect_calls = iter([["power_density"], ["conductivity"]])
        mock_st.multiselect.side_effect = lambda *a, **kw: next(multiselect_calls)
        mock_st.number_input.return_value = 2
        mock_st.selectbox.return_value = "RBF"
        mock_st.slider.return_value = 0.5
        mock_st.checkbox.return_value = True
        button_vals = iter([True, False, False, False])
        mock_st.button.side_effect = lambda *a, **kw: next(button_vals, False)

        with patch.object(
            mod.MLOptimizer,
            "run_optimization",
            return_value=mod.OptimizationResult(
                success=True,
                method=mod.OptimizationMethod.BAYESIAN,
                best_parameters={"conductivity": 5000.0, "unknown_param": 0.001},
                best_objectives={"power_density": -0.5},
                iterations=2,
                execution_time=0.3,
            ),
        ):
            mod.render_ml_optimization_page()

    def test_render_status_update_on_failure(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        mock_st.radio.return_value = "Bayesian Optimization"
        multiselect_calls = iter([["power_density"], ["conductivity"]])
        mock_st.multiselect.side_effect = lambda *a, **kw: next(multiselect_calls)
        mock_st.number_input.return_value = 2
        mock_st.selectbox.return_value = "RBF"
        mock_st.slider.return_value = 0.5
        mock_st.checkbox.return_value = True
        button_vals = iter([True, False, False, False])
        mock_st.button.side_effect = lambda *a, **kw: next(button_vals, False)

        with patch.object(
            mod.MLOptimizer,
            "run_optimization",
            return_value=mod.OptimizationResult(
                success=False,
                method=mod.OptimizationMethod.BAYESIAN,
                error_message="simulated failure",
            ),
        ):
            mod.render_ml_optimization_page()

        status_cm = mock_st.status.return_value.__enter__.return_value
        status_cm.update.assert_called()

    def test_bayesian_best_params_update(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        progress_bar = MagicMock()
        import numpy as np
        np.random.seed(42)
        result = opt._run_bayesian_optimization(
            ["power_density"], {"x": (0.0, 10.0)}, 5, progress_bar
        )
        assert result.best_parameters is not None
        assert "x" in result.best_parameters

    def test_nsga_ii_population_init_multiple_params(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        progress_bar = MagicMock()
        params = {"x": (0.0, 10.0), "y": (0.0, 5.0), "z": (1.0, 100.0)}
        result = opt._run_nsga_ii(
            ["power_density", "cost"], params, 2, progress_bar
        )
        assert result.success is True

    def test_neural_surrogate_with_many_params(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        progress_bar = MagicMock()
        params = {f"p{i}": (0.0, 10.0) for i in range(8)}
        result = opt._run_neural_surrogate(
            ["power_density"], params, 3, progress_bar
        )
        assert result.success is True

    def test_generate_next_population_empty_fronts(self):
        mock_st = _make_mock_st()
        mod = _load_module(mock_st)
        opt = mod.MLOptimizer()
        pop = opt._generate_next_population([], 5, {"x": (0.0, 10.0)})
        assert len(pop) == 5
