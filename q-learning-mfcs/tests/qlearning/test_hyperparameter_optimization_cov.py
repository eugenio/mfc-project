"""Coverage tests for hyperparameter_optimization.py (98%+ target)."""
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Mock ray, optuna, and ray.tune before importing the module
_mock_ray = MagicMock()
_mock_ray.is_initialized.return_value = False
_mock_ray.init = MagicMock()
_mock_ray.shutdown = MagicMock()

_mock_tune = MagicMock()
_mock_tune.uniform = lambda low, high: {"low": low, "high": high}
_mock_tune.Tuner = MagicMock()
_mock_tune.TuneConfig = MagicMock()
_mock_tune.RunConfig = MagicMock()

_mock_optuna = MagicMock()
_mock_optuna.samplers.TPESampler.return_value = MagicMock()

_mock_asha = MagicMock()
_mock_optuna_search = MagicMock()

sys.modules.setdefault("ray", _mock_ray)
sys.modules.setdefault("ray.tune", _mock_tune)
sys.modules.setdefault("ray.tune.schedulers", MagicMock(ASHAScheduler=_mock_asha))
sys.modules.setdefault("ray.tune.search", MagicMock())
sys.modules.setdefault("ray.tune.search.optuna", MagicMock(OptunaSearch=_mock_optuna_search))
sys.modules.setdefault("optuna", _mock_optuna)

# Now patch ray and tune at module level
import ray as _ray_mod
_ray_mod.tune = _mock_tune
_ray_mod.tune.RunConfig = MagicMock()

from hyperparameter_optimization import (
    SubstrateControlObjective,
    apply_optimized_config,
    run_bayesian_optimization,
    setup_optimization_search_space,
)


class TestSubstrateControlObjective:
    def test_init_defaults(self):
        obj = SubstrateControlObjective()
        assert obj.duration_hours == 200
        assert obj.target_concentration == 25.0
        assert obj.tolerance == 2.0

    def test_init_custom(self):
        obj = SubstrateControlObjective(
            duration_hours=100, target_concentration=30.0, tolerance=3.0
        )
        assert obj.duration_hours == 100
        assert obj.target_concentration == 30.0
        assert obj.tolerance == 3.0

    def test_call_success(self):
        obj = SubstrateControlObjective(duration_hours=0.05)
        config = {
            "learning_rate": 0.1,
            "discount_factor": 0.95,
            "epsilon_initial": 0.3,
            "epsilon_decay": 0.9995,
            "epsilon_min": 0.01,
            "power_weight": 10.0,
            "substrate_reward_multiplier": 30.0,
            "substrate_penalty_multiplier": 60.0,
            "substrate_excess_penalty": -100.0,
            "biofilm_weight": 50.0,
            "efficiency_weight": 20.0,
            "outlet_penalty_multiplier": 1.15,
            "substrate_penalty_base_multiplier": 1.0,
        }
        result = obj(config)
        assert "loss" in result
        assert "substrate_deviation" in result
        assert "stability_score" in result
        assert "power_efficiency" in result
        assert "final_concentration" in result

    def test_call_exception(self):
        obj = SubstrateControlObjective(duration_hours=0.05)
        # Pass invalid config that will cause an error
        with patch(
            "hyperparameter_optimization.run_mfc_simulation",
            side_effect=RuntimeError("test error"),
        ):
            result = obj({"learning_rate": 0.1})
        assert result["loss"] == 1000.0
        assert result["substrate_deviation"] == 100.0

    def test_call_empty_results(self):
        obj = SubstrateControlObjective(duration_hours=0.05)
        with patch(
            "hyperparameter_optimization.run_mfc_simulation",
            return_value={},
        ):
            result = obj({
                "learning_rate": 0.1,
                "discount_factor": 0.95,
                "epsilon_initial": 0.3,
                "epsilon_decay": 0.9995,
                "epsilon_min": 0.01,
                "power_weight": 10.0,
                "substrate_reward_multiplier": 30.0,
                "substrate_penalty_multiplier": 60.0,
                "substrate_excess_penalty": -100.0,
                "biofilm_weight": 50.0,
                "efficiency_weight": 20.0,
                "outlet_penalty_multiplier": 1.15,
                "substrate_penalty_base_multiplier": 1.0,
            })
        assert result["loss"] == 1000.0

    def test_build_qlearning_config(self):
        obj = SubstrateControlObjective()
        config = {
            "learning_rate": 0.15,
            "discount_factor": 0.92,
            "epsilon_initial": 0.4,
            "epsilon_decay": 0.998,
            "epsilon_min": 0.02,
            "power_weight": 12.0,
            "substrate_reward_multiplier": 35.0,
            "substrate_penalty_multiplier": 70.0,
            "substrate_excess_penalty": -120.0,
            "biofilm_weight": 55.0,
            "efficiency_weight": 25.0,
            "outlet_penalty_multiplier": 1.2,
            "substrate_penalty_base_multiplier": 1.5,
        }
        qconfig = obj._build_qlearning_config(config)
        assert qconfig.learning_rate == 0.15
        assert qconfig.discount_factor == 0.92
        assert qconfig.epsilon == 0.4
        assert qconfig.epsilon_decay == 0.998
        assert qconfig.epsilon_min == 0.02
        assert qconfig.outlet_penalty_multiplier == 1.2
        assert qconfig.substrate_penalty_base_multiplier == 1.5

    def test_calculate_performance_metrics_normal(self):
        obj = SubstrateControlObjective(target_concentration=25.0, tolerance=2.0)
        results = {
            "reservoir_concentration": list(np.linspace(20, 26, 100)),
            "outlet_concentration": list(np.linspace(18, 24, 100)),
            "total_power": list(np.linspace(0.01, 0.1, 100)),
        }
        metrics = obj._calculate_performance_metrics(results)
        assert metrics["substrate_control_loss"] > 0
        assert metrics["substrate_deviation"] > 0
        assert 0 <= metrics["stability_score"] <= 1.0
        assert metrics["power_efficiency"] > 0
        assert metrics["final_concentration"] == 26.0

    def test_calculate_performance_metrics_empty(self):
        obj = SubstrateControlObjective()
        metrics = obj._calculate_performance_metrics({
            "reservoir_concentration": [],
            "outlet_concentration": [],
            "total_power": [],
        })
        assert metrics["substrate_control_loss"] == 1000.0

    def test_calculate_performance_metrics_too_short(self):
        obj = SubstrateControlObjective()
        metrics = obj._calculate_performance_metrics({
            "reservoir_concentration": [25.0, 25.0],
            "outlet_concentration": [20.0, 20.0],
            "total_power": [0.01, 0.01],
        })
        assert metrics["substrate_control_loss"] == 1000.0

    def test_calculate_performance_metrics_perfect(self):
        obj = SubstrateControlObjective(target_concentration=25.0)
        results = {
            "reservoir_concentration": [25.0] * 100,
            "outlet_concentration": [25.0] * 100,
            "total_power": [0.05] * 100,
        }
        metrics = obj._calculate_performance_metrics(results)
        assert metrics["substrate_deviation"] == 0.0
        assert metrics["stability_score"] == 1.0

    def test_calculate_performance_metrics_large_excursions(self):
        obj = SubstrateControlObjective(target_concentration=25.0, tolerance=2.0)
        # Concentrations far from target
        results = {
            "reservoir_concentration": [50.0] * 100,
            "outlet_concentration": [10.0] * 100,
            "total_power": [10.0] * 100,
        }
        metrics = obj._calculate_performance_metrics(results)
        assert metrics["substrate_control_loss"] > 100


class TestSetupOptimizationSearchSpace:
    def test_search_space_keys(self):
        space = setup_optimization_search_space()
        expected_keys = [
            "learning_rate", "discount_factor", "epsilon_initial",
            "epsilon_decay", "epsilon_min", "power_weight",
            "substrate_reward_multiplier", "substrate_penalty_multiplier",
            "substrate_excess_penalty", "biofilm_weight", "efficiency_weight",
            "outlet_penalty_multiplier", "substrate_penalty_base_multiplier",
        ]
        for key in expected_keys:
            assert key in space, f"Missing key: {key}"


class TestRunBayesianOptimization:
    @patch("hyperparameter_optimization.ray")
    @patch("hyperparameter_optimization.tune")
    @patch("hyperparameter_optimization.OptunaSearch")
    @patch("hyperparameter_optimization.ASHAScheduler")
    def test_run_bayesian_optimization(
        self, mock_scheduler, mock_optuna_search, mock_tune, mock_ray
    ):
        mock_ray.is_initialized.return_value = False
        mock_ray.tune = MagicMock()
        mock_ray.tune.RunConfig = MagicMock()

        mock_result = MagicMock()
        mock_result.config = {"learning_rate": 0.1, "discount_factor": 0.95}
        mock_result.metrics = {"loss": 5.0}

        mock_results = MagicMock()
        mock_results.get_best_result.return_value = mock_result

        mock_tuner_instance = MagicMock()
        mock_tuner_instance.fit.return_value = mock_results
        mock_tune.Tuner.return_value = mock_tuner_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("hyperparameter_optimization.os.makedirs"):
                with patch("builtins.open", MagicMock()):
                    with patch("json.dump"):
                        best_config, results_file = run_bayesian_optimization(
                            num_samples=2,
                            max_concurrent_trials=1,
                            duration_hours=10,
                        )

        assert best_config["learning_rate"] == 0.1
        mock_ray.init.assert_called_once()
        mock_ray.shutdown.assert_called_once()


class TestApplyOptimizedConfig:
    def test_apply_config(self):
        best_config = {
            "learning_rate": 0.15,
            "discount_factor": 0.92,
            "epsilon_initial": 0.4,
            "epsilon_decay": 0.998,
            "epsilon_min": 0.02,
            "power_weight": 12.0,
            "substrate_reward_multiplier": 35.0,
            "substrate_penalty_multiplier": 70.0,
            "substrate_excess_penalty": -120.0,
            "biofilm_weight": 55.0,
            "efficiency_weight": 25.0,
            "outlet_penalty_multiplier": 1.2,
            "substrate_penalty_base_multiplier": 1.5,
        }
        config = apply_optimized_config(best_config)
        assert config.learning_rate == 0.15
        assert config.discount_factor == 0.92
        assert config.epsilon == 0.4
        assert config.epsilon_decay == 0.998
        assert config.epsilon_min == 0.02
        assert config.outlet_penalty_multiplier == 1.2
        assert config.substrate_penalty_base_multiplier == 1.5
        # Note: source uses config.rewards (not reward_weights), a dynamic attribute
        assert config.rewards.power_weight == 12.0
        assert config.rewards.substrate_reward_multiplier == 35.0
