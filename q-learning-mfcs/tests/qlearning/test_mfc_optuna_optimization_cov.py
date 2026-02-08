"""Coverage tests for mfc_optuna_optimization.py (98%+ target)."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Mock optuna before importing modules that depend on it
mock_optuna = MagicMock()
mock_optuna.trial.TrialState.COMPLETE = "COMPLETE"
mock_optuna.pruners.MedianPruner = MagicMock()
sys.modules.setdefault("optuna", mock_optuna)
sys.modules.setdefault("optuna.trial", mock_optuna.trial)
sys.modules.setdefault("optuna.pruners", mock_optuna.pruners)
sys.modules.setdefault("optuna.samplers", mock_optuna.samplers)

# Need to patch config validation for unified controller
import mfc_unified_qlearning_control as _umod

_orig_validate_u = _umod.validate_qlearning_config


def _patched_validate_u(config):
    if not hasattr(config, "flow_rate_adjustments_ml_per_h"):
        config.flow_rate_adjustments_ml_per_h = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
    _orig_validate_u(config)


_umod.validate_qlearning_config = _patched_validate_u

from mfc_optuna_optimization import MFCOptunaOptimizer, main


# Standard test params used across multiple tests
_STANDARD_PARAMS = {
    "learning_rate": 0.1, "discount_factor": 0.95,
    "initial_epsilon": 0.3, "epsilon_decay": 0.998,
    "epsilon_min": 0.05, "max_flow_decrease": -8,
    "max_flow_increase": 5, "max_substrate_decrease": -2.0,
    "max_substrate_increase": 1.5,
    "substrate_increment_fineness": "coarse",
    "biofilm_base_reward": 40.0, "biofilm_steady_bonus": 20.0,
    "biofilm_penalty_multiplier": 70.0,
    "power_increase_multiplier": 50.0,
    "power_decrease_multiplier": 100.0,
    "substrate_increase_multiplier": 30.0,
    "substrate_decrease_multiplier": 60.0,
    "conc_precise_reward": 20.0, "conc_acceptable_reward": 5.0,
    "conc_poor_penalty": -10.0, "flow_penalty_threshold": 20.0,
    "flow_penalty_multiplier": 25.0,
    "biofilm_threshold_ratio": 0.9,
}


def _make_mock_trial():
    """Create a mock Optuna trial with proper suggest methods."""
    trial = MagicMock()
    float_values = {}
    int_values = {}
    cat_values = {}

    def suggest_float(name, low, high):
        val = (low + high) / 2.0
        float_values[name] = val
        return val

    def suggest_int(name, low, high):
        val = (low + high) // 2
        int_values[name] = val
        return val

    def suggest_categorical(name, choices):
        val = choices[0]
        cat_values[name] = val
        return val

    trial.suggest_float = suggest_float
    trial.suggest_int = suggest_int
    trial.suggest_categorical = suggest_categorical
    trial.number = 0
    trial.set_user_attr = MagicMock()
    return trial


def _make_optimizer(**kwargs):
    """Create an MFCOptunaOptimizer with mocked logging."""
    with patch.object(MFCOptunaOptimizer, "setup_logging"):
        opt = MFCOptunaOptimizer(**kwargs)
        opt.logger = MagicMock()
    return opt


class TestMFCOptunaOptimizer:
    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(MFCOptunaOptimizer, "setup_logging"):
                opt = MFCOptunaOptimizer(n_trials=2, n_jobs=1)
                opt.results_dir = Path(tmpdir)
                opt.logger = MagicMock()
        assert opt.n_trials == 2
        assert opt.n_jobs == 1
        assert opt.target_biofilm == 1.3
        assert opt.target_outlet_conc == 12.0

    def test_setup_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = MFCOptunaOptimizer.__new__(MFCOptunaOptimizer)
            opt.results_dir = Path(tmpdir)
            opt.setup_logging()
            assert opt.logger is not None

    def test_define_search_space(self):
        opt = _make_optimizer(n_trials=1)
        trial = _make_mock_trial()
        params = opt.define_search_space(trial)
        assert "biofilm_base_reward" in params
        assert "learning_rate" in params
        assert "max_flow_decrease" in params
        assert "substrate_increment_fineness" in params

    def test_create_modified_simulation(self):
        opt = _make_optimizer(n_trials=1)
        trial = _make_mock_trial()
        params = opt.define_search_space(trial)
        sim = opt.create_modified_simulation(params)
        assert sim is not None
        assert sim.total_time == 120 * 3600

    def test_update_action_space_coarse(self):
        opt = _make_optimizer(n_trials=1)
        trial = _make_mock_trial()
        params = opt.define_search_space(trial)
        params["substrate_increment_fineness"] = "coarse"
        sim = opt.create_modified_simulation(params)
        assert len(sim.unified_controller.actions) > 0

    def test_update_action_space_medium(self):
        opt = _make_optimizer(n_trials=1)
        trial = _make_mock_trial()
        params = opt.define_search_space(trial)
        params["substrate_increment_fineness"] = "medium"
        sim = opt.create_modified_simulation(params)
        assert len(sim.unified_controller.actions) > 0

    def test_update_action_space_fine(self):
        opt = _make_optimizer(n_trials=1)
        trial = _make_mock_trial()
        params = opt.define_search_space(trial)
        params["substrate_increment_fineness"] = "fine"
        sim = opt.create_modified_simulation(params)
        assert len(sim.unified_controller.actions) > 0

    # ---- Tests for the INNER optimized_reward_function in create_modified_simulation ----

    def test_reward_function_power_increase(self):
        """Cover inner reward function: power_change > 0 branch."""
        opt = _make_optimizer(n_trials=1)
        sim = opt.create_modified_simulation(_STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=0.5, prev_biofilm_dev=0.02,
            prev_substrate_util=15.0, prev_outlet_conc=14.0,
        )
        assert isinstance(result, float)

    def test_reward_function_power_decrease(self):
        """Cover inner reward function: power_change <= 0 branch."""
        opt = _make_optimizer(n_trials=1)
        sim = opt.create_modified_simulation(_STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=0.5, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=1.0, prev_biofilm_dev=0.02,
            prev_substrate_util=25.0, prev_outlet_conc=14.0,
        )
        assert isinstance(result, float)

    def test_reward_function_biofilm_steady_state(self):
        """Cover steady state bonus with biofilm history."""
        opt = _make_optimizer(n_trials=1)
        sim = opt.create_modified_simulation(_STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
            biofilm_thickness_history=[1.3, 1.3, 1.3],
        )
        assert isinstance(result, float)
        assert result > 0  # should get steady state bonus

    def test_reward_function_biofilm_growing(self):
        """Cover biofilm history with growth rate >= 0.01."""
        opt = _make_optimizer(n_trials=1)
        sim = opt.create_modified_simulation(_STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
            biofilm_thickness_history=[1.0, 1.1, 1.3],
        )
        assert isinstance(result, float)

    def test_reward_function_biofilm_outside_threshold(self):
        """Cover biofilm deviation > threshold branch."""
        opt = _make_optimizer(n_trials=1)
        sim = opt.create_modified_simulation(_STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.5, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=1.0, prev_biofilm_dev=0.5,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
        )
        assert isinstance(result, float)
        assert result < 0  # penalty

    def test_reward_function_concentration_precise(self):
        """Cover outlet_error <= 0.5 branch."""
        opt = _make_optimizer(n_trials=1)
        sim = opt.create_modified_simulation(_STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.2, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=13.0,
        )
        assert isinstance(result, float)

    def test_reward_function_concentration_acceptable(self):
        """Cover outlet_error <= 2.0 branch."""
        opt = _make_optimizer(n_trials=1)
        sim = opt.create_modified_simulation(_STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=13.5, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=13.0,
        )
        assert isinstance(result, float)

    def test_reward_function_concentration_poor(self):
        """Cover outlet_error > 2.0 branch."""
        opt = _make_optimizer(n_trials=1)
        sim = opt.create_modified_simulation(_STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=20.0, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=13.0,
        )
        assert isinstance(result, float)

    def test_reward_function_stability_bonus(self):
        """Cover stability bonus: all conditions met."""
        opt = _make_optimizer(n_trials=1)
        sim = opt.create_modified_simulation(_STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        # All values nearly identical -> stability bonus
        result = reward_fn(
            power=0.01, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=0.01, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
        )
        assert isinstance(result, float)

    def test_reward_function_flow_penalty(self):
        """Cover flow penalty path."""
        opt = _make_optimizer(n_trials=1)
        params = _STANDARD_PARAMS.copy()
        params["biofilm_threshold_ratio"] = 0.99  # Makes it easier to trigger
        params["flow_penalty_threshold"] = 5.0  # Low threshold
        sim = opt.create_modified_simulation(params)
        # Set a high current_flow_rate on the controller
        sim.unified_controller.current_flow_rate = 30.0
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
            biofilm_thickness_history=[0.5, 0.5, 0.5],  # Below threshold
        )
        assert isinstance(result, float)

    def test_reward_function_combined_penalty(self):
        """Cover combined penalty: all metrics worsening."""
        opt = _make_optimizer(n_trials=1)
        sim = opt.create_modified_simulation(_STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        # All metrics worsening + biofilm deviation > threshold
        result = reward_fn(
            power=0.5, biofilm_deviation=0.5, substrate_utilization=10.0,
            outlet_conc=20.0, prev_power=1.0, prev_biofilm_dev=0.5,
            prev_substrate_util=20.0, prev_outlet_conc=13.0,
        )
        assert isinstance(result, float)
        assert result < -100  # Should include -200 combined penalty

    def test_reward_function_low_power_base(self):
        """Cover power_base -5.0 (power <= 0.005)."""
        opt = _make_optimizer(n_trials=1)
        sim = opt.create_modified_simulation(_STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=0.001, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=0.001, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
        )
        assert isinstance(result, float)

    def test_reward_function_low_substrate_base(self):
        """Cover substrate_base -2.0 (substrate_utilization <= 10)."""
        opt = _make_optimizer(n_trials=1)
        sim = opt.create_modified_simulation(_STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=5.0,
            outlet_conc=12.0, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=5.0, prev_outlet_conc=12.0,
        )
        assert isinstance(result, float)

    # ---- Tests for _apply_parameters_to_simulation inner reward function ----

    def test_apply_params_reward_function_power_increase(self):
        """Cover inner reward in _apply_parameters_to_simulation: power up."""
        opt = _make_optimizer(n_trials=1)
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        opt._apply_parameters_to_simulation(sim, _STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=0.5, prev_biofilm_dev=0.02,
            prev_substrate_util=15.0, prev_outlet_conc=14.0,
        )
        assert isinstance(result, float)

    def test_apply_params_reward_function_power_decrease(self):
        """Cover inner reward in _apply_parameters_to_simulation: power down."""
        opt = _make_optimizer(n_trials=1)
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        opt._apply_parameters_to_simulation(sim, _STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=0.5, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=1.0, prev_biofilm_dev=0.02,
            prev_substrate_util=25.0, prev_outlet_conc=14.0,
        )
        assert isinstance(result, float)

    def test_apply_params_reward_biofilm_steady(self):
        """Cover biofilm steady state bonus in _apply_parameters."""
        opt = _make_optimizer(n_trials=1)
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        opt._apply_parameters_to_simulation(sim, _STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
            biofilm_thickness_history=[1.3, 1.3, 1.3],
        )
        assert isinstance(result, float)

    def test_apply_params_reward_biofilm_growing(self):
        """Cover biofilm growing (growth_rate >= 0.01) in _apply_parameters."""
        opt = _make_optimizer(n_trials=1)
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        opt._apply_parameters_to_simulation(sim, _STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
            biofilm_thickness_history=[1.0, 1.1, 1.3],
        )
        assert isinstance(result, float)

    def test_apply_params_reward_biofilm_outside_threshold(self):
        """Cover biofilm penalty in _apply_parameters."""
        opt = _make_optimizer(n_trials=1)
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        opt._apply_parameters_to_simulation(sim, _STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.5, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=1.0, prev_biofilm_dev=0.5,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
        )
        assert isinstance(result, float)

    def test_apply_params_reward_conc_precise(self):
        """Cover outlet_error <= 0.5 in _apply_parameters reward fn."""
        opt = _make_optimizer(n_trials=1)
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        opt._apply_parameters_to_simulation(sim, _STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.3, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=13.0,
        )
        assert isinstance(result, float)

    def test_apply_params_reward_conc_acceptable(self):
        """Cover outlet_error <= 2.0 in _apply_parameters reward fn."""
        opt = _make_optimizer(n_trials=1)
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        opt._apply_parameters_to_simulation(sim, _STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=13.5, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=13.0,
        )
        assert isinstance(result, float)

    def test_apply_params_reward_conc_poor(self):
        """Cover outlet_error > 2.0 in _apply_parameters reward fn."""
        opt = _make_optimizer(n_trials=1)
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        opt._apply_parameters_to_simulation(sim, _STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=20.0, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=13.0,
        )
        assert isinstance(result, float)

    def test_apply_params_reward_stability_bonus(self):
        """Cover stability bonus in _apply_parameters reward fn."""
        opt = _make_optimizer(n_trials=1)
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        opt._apply_parameters_to_simulation(sim, _STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=0.01, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=0.01, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
        )
        assert isinstance(result, float)

    def test_apply_params_reward_flow_penalty(self):
        """Cover flow penalty in _apply_parameters reward fn."""
        opt = _make_optimizer(n_trials=1)
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        params = _STANDARD_PARAMS.copy()
        params["biofilm_threshold_ratio"] = 0.99
        params["flow_penalty_threshold"] = 5.0
        opt._apply_parameters_to_simulation(sim, params)
        sim.unified_controller.current_flow_rate = 30.0
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
            biofilm_thickness_history=[0.5, 0.5, 0.5],
        )
        assert isinstance(result, float)

    def test_apply_params_reward_combined_penalty(self):
        """Cover combined penalty in _apply_parameters reward fn."""
        opt = _make_optimizer(n_trials=1)
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        opt._apply_parameters_to_simulation(sim, _STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=0.5, biofilm_deviation=0.5, substrate_utilization=10.0,
            outlet_conc=20.0, prev_power=1.0, prev_biofilm_dev=0.5,
            prev_substrate_util=20.0, prev_outlet_conc=13.0,
        )
        assert isinstance(result, float)

    def test_apply_params_reward_low_power_base(self):
        """Cover power_base -5.0 in _apply_parameters."""
        opt = _make_optimizer(n_trials=1)
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        opt._apply_parameters_to_simulation(sim, _STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=0.001, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=0.001, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
        )
        assert isinstance(result, float)

    def test_apply_params_reward_low_substrate_base(self):
        """Cover substrate_base -2.0 in _apply_parameters."""
        opt = _make_optimizer(n_trials=1)
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        opt._apply_parameters_to_simulation(sim, _STANDARD_PARAMS.copy())
        reward_fn = sim.unified_controller.calculate_unified_reward
        result = reward_fn(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=5.0,
            outlet_conc=12.0, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=5.0, prev_outlet_conc=12.0,
        )
        assert isinstance(result, float)

    # ---- Objective tests ----

    def test_objective_success(self):
        opt = _make_optimizer(n_trials=1)
        trial = _make_mock_trial()
        mock_sim = MagicMock()
        mock_sim.stack_powers = np.ones(100) * 0.01
        mock_sim.concentration_errors = np.ones(100) * 0.5
        mock_sim.substrate_utilizations = np.ones(100) * 20.0
        mock_sim.q_rewards = np.ones(100) * 10.0
        mock_sim.biofilm_thickness = np.ones((100, 5)) * 1.3
        mock_sim.unified_controller = MagicMock()
        mock_sim.unified_controller.actions = [(0, 0)]
        mock_sim.dt = 10.0
        with patch.object(opt, "create_modified_simulation", return_value=mock_sim):
            result = opt.objective(trial)
        assert isinstance(result, float)

    def test_objective_failure(self):
        opt = _make_optimizer(n_trials=1)
        trial = _make_mock_trial()
        with patch.object(opt, "create_modified_simulation", side_effect=Exception("fail")):
            result = opt.objective(trial)
        assert result == 1000.0

    def test_objective_no_valid_errors(self):
        opt = _make_optimizer(n_trials=1)
        trial = _make_mock_trial()
        mock_sim = MagicMock()
        mock_sim.stack_powers = np.ones(100) * 0.01
        mock_sim.concentration_errors = np.zeros(100)
        mock_sim.substrate_utilizations = np.ones(100) * 20.0
        mock_sim.q_rewards = np.ones(100) * 10.0
        mock_sim.biofilm_thickness = np.array([[0.5]])
        mock_sim.unified_controller = MagicMock()
        mock_sim.unified_controller.actions = [(0, 0)]
        mock_sim.dt = 10.0
        with patch.object(opt, "create_modified_simulation", return_value=mock_sim):
            result = opt.objective(trial)
        assert isinstance(result, float)

    def test_objective_1d_biofilm(self):
        """Cover line 498: biofilm without 2D shape."""
        opt = _make_optimizer(n_trials=1)
        trial = _make_mock_trial()
        mock_sim = MagicMock()
        mock_sim.stack_powers = np.ones(100) * 0.01
        mock_sim.concentration_errors = np.ones(100) * 0.5
        mock_sim.substrate_utilizations = np.ones(100) * 20.0
        mock_sim.q_rewards = np.ones(100) * 10.0
        # 1D biofilm (not 2D) - triggers else branch
        mock_sim.biofilm_thickness = np.array([0.5])
        mock_sim.unified_controller = MagicMock()
        mock_sim.unified_controller.actions = [(0, 0)]
        mock_sim.dt = 10.0
        with patch.object(opt, "create_modified_simulation", return_value=mock_sim):
            result = opt.objective(trial)
        assert isinstance(result, float)

    # ---- run_optimization tests ----

    def test_run_optimization(self):
        import optuna
        opt = _make_optimizer(n_trials=1, n_jobs=1)
        mock_study = MagicMock()
        mock_study.best_params = {"param1": 1.0}
        mock_study.best_value = 0.5
        mock_study.best_trial = MagicMock()
        mock_study.best_trial.user_attrs = {"metric1": 1.0}
        mock_study.trials = []
        mock_study.study_name = "test"
        with patch("optuna.create_study", return_value=mock_study):
            with patch.object(opt, "save_results"):
                study = opt.run_optimization()
        assert study is not None

    def test_run_optimization_parallel(self):
        opt = _make_optimizer(n_trials=2, n_jobs=2)
        mock_study = MagicMock()
        mock_study.best_params = {"param1": 1.0}
        mock_study.best_value = 0.5
        mock_study.best_trial = MagicMock()
        mock_study.best_trial.user_attrs = {}
        mock_study.trials = []
        mock_study.study_name = "test"
        with patch("optuna.create_study", return_value=mock_study):
            with patch.object(opt, "save_results"):
                study = opt.run_optimization()
        assert study is not None

    def test_run_optimization_with_storage(self):
        opt = _make_optimizer(n_trials=1, n_jobs=1, storage="sqlite:///test.db")
        mock_study = MagicMock()
        mock_study.best_params = {"param1": 1.0}
        mock_study.best_value = 0.5
        mock_study.best_trial = MagicMock()
        mock_study.best_trial.user_attrs = {}
        mock_study.trials = []
        mock_study.study_name = "test"
        with patch("optuna.create_study", return_value=mock_study):
            with patch.object(opt, "save_results"):
                study = opt.run_optimization()
        assert study is not None

    def test_run_optimization_keyboard_interrupt(self):
        opt = _make_optimizer(n_trials=1, n_jobs=1)
        mock_study = MagicMock()
        mock_study.optimize.side_effect = KeyboardInterrupt()
        mock_study.best_params = {"param1": 1.0}
        mock_study.best_value = 0.5
        mock_study.best_trial = MagicMock()
        mock_study.best_trial.user_attrs = {}
        mock_study.trials = []
        mock_study.study_name = "test"
        with patch("optuna.create_study", return_value=mock_study):
            with patch.object(opt, "save_results"):
                study = opt.run_optimization()
        assert study is not None

    # ---- save/print/get tests ----

    def test_save_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = _make_optimizer(n_trials=1)
            opt.results_dir = Path(tmpdir)
            mock_study = MagicMock()
            mock_study.best_params = {"param1": 1.0}
            mock_study.best_value = 0.5
            mock_study.best_trial = MagicMock()
            mock_study.best_trial.user_attrs = {"metric1": 2.0}
            mock_study.study_name = "test"
            mock_trial = MagicMock()
            mock_trial.number = 0
            mock_trial.value = 0.5
            mock_trial.params = {"param1": 1.0}
            mock_trial.state = MagicMock()
            mock_trial.state.name = "COMPLETE"
            mock_trial.user_attrs = {"metric1": 2.0}
            mock_study.trials = [mock_trial]
            opt.save_results(mock_study)

    def test_print_optimization_summary(self):
        opt = _make_optimizer(n_trials=1)
        mock_study = MagicMock()
        mock_study.best_params = {"param1": 1.0}
        mock_study.best_trial = MagicMock()
        mock_study.best_trial.user_attrs = {"metric1": 2.0}
        opt.print_optimization_summary(mock_study)

    def test_print_optimization_summary_no_attrs(self):
        opt = _make_optimizer(n_trials=1)
        mock_study = MagicMock()
        mock_study.best_params = {"param1": 1.0}
        mock_study.best_trial = MagicMock()
        mock_study.best_trial.user_attrs = {}
        opt.print_optimization_summary(mock_study)

    def test_get_top_configurations(self):
        import optuna
        opt = _make_optimizer(n_trials=1)
        mock_trial = MagicMock()
        mock_trial.number = 0
        mock_trial.value = 0.5
        mock_trial.params = {"param1": 1.0}
        mock_trial.state = optuna.trial.TrialState.COMPLETE
        mock_trial.user_attrs = {"metric1": 2.0}
        mock_study = MagicMock()
        mock_study.trials = [mock_trial]
        configs = opt.get_top_configurations(mock_study, n_configs=1)
        assert len(configs) == 1
        assert configs[0]["trial_number"] == 0

    # ---- Extended validation tests ----

    def test_run_extended_validation_parallel(self):
        opt = _make_optimizer(n_trials=1)
        config = {
            "trial_number": 0,
            "objective_value": 0.5,
            "params": _STANDARD_PARAMS.copy(),
            "original_metrics": {"energy_total": 5.0, "control_rmse": 4.0,
                                 "biofilm_error": 0.1},
        }
        mock_result = {
            "config": config, "status": "completed",
            "extended_objective": 0.5, "extended_simulation_time": 10.0,
            "extended_metrics": {"energy_total": 5.0},
        }
        with patch.object(opt, "_run_single_extended_validation", return_value=mock_result):
            with patch.object(opt, "_save_validation_results"):
                with patch.object(opt, "_print_validation_summary"):
                    results = opt.run_extended_validation([config])
        assert len(results) == 1

    def test_run_extended_validation_failure(self):
        opt = _make_optimizer(n_trials=1)
        config = {"trial_number": 0, "params": {}, "original_metrics": {}}
        with patch.object(opt, "_run_single_extended_validation", side_effect=Exception("fail")):
            with patch.object(opt, "_save_validation_results"):
                with patch.object(opt, "_print_validation_summary"):
                    results = opt.run_extended_validation([config])
        assert len(results) == 1
        assert results[0]["status"] == "failed"

    def test_run_extended_validation_sequential(self):
        """Test sequential execution path (>14 configs)."""
        opt = _make_optimizer(n_trials=1)
        configs = [{"trial_number": i, "params": {}, "original_metrics": {}} for i in range(16)]
        mock_result = {"config": configs[0], "status": "completed", "extended_objective": 0.5}
        with patch.object(opt, "_run_single_extended_validation", return_value=mock_result):
            with patch.object(opt, "_save_validation_results"):
                with patch.object(opt, "_print_validation_summary"):
                    results = opt.run_extended_validation(configs)
        assert len(results) == 16

    def test_run_extended_validation_sequential_failure(self):
        opt = _make_optimizer(n_trials=1)
        configs = [{"trial_number": i, "params": {}, "original_metrics": {}} for i in range(16)]
        with patch.object(opt, "_run_single_extended_validation", side_effect=Exception("fail")):
            with patch.object(opt, "_save_validation_results"):
                with patch.object(opt, "_print_validation_summary"):
                    results = opt.run_extended_validation(configs)
        assert all(r["status"] == "failed" for r in results)

    def test_run_single_extended_validation(self):
        """Actually run _run_single_extended_validation to cover lines 809-949."""
        opt = _make_optimizer(n_trials=1)
        config = {
            "trial_number": 0,
            "params": _STANDARD_PARAMS.copy(),
            "original_metrics": {"energy_total": 5.0, "control_rmse": 4.0,
                                 "biofilm_error": 0.1},
        }
        # Mock the simulation run to be fast
        mock_sim = MagicMock()
        mock_sim.use_gpu = False
        mock_sim.num_cells = 5
        mock_sim.dt = 10.0
        mock_sim.stack_powers = np.ones(100) * 0.01
        mock_sim.concentration_errors = np.ones(100) * 0.5
        mock_sim.substrate_utilizations = np.ones(100) * 20.0
        mock_sim.q_rewards = np.ones(100) * 10.0
        mock_sim.biofilm_thickness = np.ones((100, 5)) * 1.3
        mock_sim.unified_controller = MagicMock()

        with patch("mfc_optuna_optimization.MFCUnifiedQLearningSimulation", return_value=mock_sim):
            result = opt._run_single_extended_validation(config, 1)
        assert result["status"] == "completed"
        assert "extended_objective" in result
        assert "extended_metrics" in result
        assert "improvement_vs_short" in result

    def test_run_single_extended_validation_no_valid_errors(self):
        """Cover lines 874-876: no valid errors branch."""
        opt = _make_optimizer(n_trials=1)
        config = {
            "trial_number": 0,
            "params": _STANDARD_PARAMS.copy(),
            "original_metrics": {"energy_total": 5.0, "control_rmse": 4.0,
                                 "biofilm_error": 0.1},
        }
        mock_sim = MagicMock()
        mock_sim.use_gpu = False
        mock_sim.num_cells = 5
        mock_sim.dt = 10.0
        mock_sim.stack_powers = np.ones(100) * 0.01
        mock_sim.concentration_errors = np.zeros(100)  # All zeros
        mock_sim.substrate_utilizations = np.ones(100) * 20.0
        mock_sim.q_rewards = np.ones(100) * 10.0
        mock_sim.biofilm_thickness = np.ones((100, 5)) * 1.3
        mock_sim.unified_controller = MagicMock()

        with patch("mfc_optuna_optimization.MFCUnifiedQLearningSimulation", return_value=mock_sim):
            result = opt._run_single_extended_validation(config, 1)
        assert result["extended_metrics"]["control_rmse"] == 100.0

    def test_run_single_extended_validation_1d_biofilm(self):
        """Cover lines 890-892: 1D biofilm else branch."""
        opt = _make_optimizer(n_trials=1)
        config = {
            "trial_number": 0,
            "params": _STANDARD_PARAMS.copy(),
            "original_metrics": {"energy_total": 5.0, "control_rmse": 4.0,
                                 "biofilm_error": 0.1},
        }
        mock_sim = MagicMock()
        mock_sim.use_gpu = False
        mock_sim.num_cells = 5
        mock_sim.dt = 10.0
        mock_sim.stack_powers = np.ones(100) * 0.01
        mock_sim.concentration_errors = np.ones(100) * 0.5
        mock_sim.substrate_utilizations = np.ones(100) * 20.0
        mock_sim.q_rewards = np.ones(100) * 10.0
        mock_sim.unified_controller = MagicMock()

        # After run_simulation, replace biofilm_thickness with 1D array
        def set_1d_biofilm():
            mock_sim.biofilm_thickness = np.array([0.5])

        mock_sim.run_simulation.side_effect = set_1d_biofilm

        with patch("mfc_optuna_optimization.MFCUnifiedQLearningSimulation", return_value=mock_sim):
            result = opt._run_single_extended_validation(config, 1)
        assert result["extended_metrics"]["avg_final_biofilm"] == 0.5
        assert result["extended_metrics"]["biofilm_stability"] == 0.0

    def test_run_single_extended_validation_exception(self):
        """Cover lines 947-949: exception re-raise."""
        opt = _make_optimizer(n_trials=1)
        config = {
            "trial_number": 0,
            "params": _STANDARD_PARAMS.copy(),
            "original_metrics": {},
        }
        with patch("mfc_optuna_optimization.MFCUnifiedQLearningSimulation", side_effect=Exception("sim fail")):
            with pytest.raises(Exception, match="sim fail"):
                opt._run_single_extended_validation(config, 1)

    def test_apply_parameters_to_simulation(self):
        opt = _make_optimizer(n_trials=1)
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        opt._apply_parameters_to_simulation(sim, _STANDARD_PARAMS.copy())
        assert sim.unified_controller.learning_rate == 0.1

    # ---- save/print validation tests ----

    def test_save_validation_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = _make_optimizer(n_trials=1)
            opt.results_dir = Path(tmpdir)
            results = [
                {
                    "config": {"trial_number": 0},
                    "status": "completed",
                    "extended_metrics": {"energy_total": np.float64(5.0)},
                }
            ]
            opt._save_validation_results(results)

    def test_print_validation_summary_success(self):
        opt = _make_optimizer(n_trials=1)
        results = [
            {"config": {"trial_number": 0}, "status": "completed",
             "extended_objective": 0.5, "extended_metrics": {"energy_total": 5.0}},
            {"config": {"trial_number": 1}, "status": "completed",
             "extended_objective": 0.6, "extended_metrics": {"energy_total": 4.0}},
            {"config": {"trial_number": 2}, "status": "completed",
             "extended_objective": 0.7, "extended_metrics": {"energy_total": 3.0}},
        ]
        opt._print_validation_summary(results)

    def test_print_validation_summary_with_failures(self):
        opt = _make_optimizer(n_trials=1)
        results = [
            {"config": {"trial_number": 0}, "status": "failed", "error": "test"},
        ]
        opt._print_validation_summary(results)

    def test_print_validation_summary_empty(self):
        opt = _make_optimizer(n_trials=1)
        opt._print_validation_summary([])


class TestMain:
    def test_main_success(self):
        with patch.object(MFCOptunaOptimizer, "setup_logging"):
            with patch.object(MFCOptunaOptimizer, "run_optimization") as mock_run:
                mock_study = MagicMock()
                mock_study.best_params = {"param1": 1.0}
                mock_study.best_value = 0.5
                mock_study.best_trial = MagicMock()
                mock_study.best_trial.user_attrs = {}
                mock_study.trials = []
                mock_run.return_value = mock_study
                with patch.object(MFCOptunaOptimizer, "print_optimization_summary"):
                    with patch.object(MFCOptunaOptimizer, "get_top_configurations", return_value=[]):
                        with patch.object(MFCOptunaOptimizer, "run_extended_validation"):
                            result = main()
        assert result == 0

    def test_main_failure(self):
        with patch.object(MFCOptunaOptimizer, "setup_logging"):
            with patch.object(MFCOptunaOptimizer, "run_optimization", side_effect=Exception("fail")):
                result = main()
        assert result == 1
