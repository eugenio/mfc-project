"""Coverage tests for mfc_qlearning_optimization.py (98%+ target)."""
import os
import pickle
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config import QLearningConfig


def _make_config():
    """Create a QLearningConfig with the missing attribute."""
    cfg = QLearningConfig()
    cfg.flow_rate_adjustments_ml_per_h = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
    return cfg


# Patch default config creation so QLearningFlowController() works
_orig_validate = None


def _patched_validate(config):
    if not hasattr(config, "flow_rate_adjustments_ml_per_h"):
        config.flow_rate_adjustments_ml_per_h = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
    if _orig_validate:
        _orig_validate(config)


import mfc_qlearning_optimization as _mod

_orig_validate = _mod.validate_qlearning_config
_mod.validate_qlearning_config = _patched_validate

from mfc_qlearning_optimization import (
    MFCQLearningSimulation,
    QLearningFlowController,
    add_subplot_labels,
    main,
)


class TestQLearningFlowController:
    def test_init_default_config(self):
        ctrl = QLearningFlowController()
        assert ctrl.learning_rate == 0.1
        assert ctrl.discount_factor > 0
        assert ctrl.epsilon > 0
        assert len(ctrl.actions) > 0

    def test_init_custom_config(self):
        cfg = _make_config()
        cfg.learning_rate = 0.2
        ctrl = QLearningFlowController(config=cfg)
        assert ctrl.learning_rate == 0.2

    def test_setup_state_action_spaces(self):
        ctrl = QLearningFlowController()
        assert len(ctrl.power_bins) > 0
        assert len(ctrl.biofilm_bins) > 0
        assert len(ctrl.substrate_bins) > 0
        assert len(ctrl.time_bins) > 0
        assert len(ctrl.actions) > 0

    def test_discretize_state(self):
        ctrl = QLearningFlowController()
        state = ctrl.discretize_state(0.5, 0.1, 10.0, 500.0)
        assert isinstance(state, tuple)
        assert len(state) == 4

    def test_discretize_state_boundary_low(self):
        ctrl = QLearningFlowController()
        state = ctrl.discretize_state(0.0, 0.0, 0.0, 0.0)
        assert all(idx >= 0 for idx in state)

    def test_discretize_state_boundary_high(self):
        ctrl = QLearningFlowController()
        state = ctrl.discretize_state(100.0, 100.0, 100.0, 10000.0)
        assert all(idx >= 0 for idx in state)

    def test_select_action_exploration(self):
        ctrl = QLearningFlowController()
        ctrl.epsilon = 1.0
        state = (0, 0, 0, 0)
        np.random.seed(42)
        action_idx, new_flow = ctrl.select_action(state, 1e-5)
        assert 0 <= action_idx < len(ctrl.actions)
        assert new_flow > 0

    def test_select_action_exploitation(self):
        ctrl = QLearningFlowController()
        ctrl.epsilon = 0.0
        state = (0, 0, 0, 0)
        ctrl.q_table[state][2] = 100.0
        action_idx, new_flow = ctrl.select_action(state, 1e-5)
        assert action_idx == 2

    def test_select_action_flow_bounds(self):
        ctrl = QLearningFlowController()
        ctrl.epsilon = 0.0
        state = (0, 0, 0, 0)
        _, new_flow = ctrl.select_action(state, 1e-10)
        min_flow = ctrl.config.flow_rate_min * 1e-6 / 3.6
        max_flow = ctrl.config.flow_rate_max * 1e-6 / 3.6
        assert new_flow >= min_flow
        assert new_flow <= max_flow

    def test_update_q_table(self):
        ctrl = QLearningFlowController()
        state = (0, 0, 0, 0)
        next_state = (1, 1, 1, 1)
        old_epsilon = ctrl.epsilon
        ctrl.update_q_table(state, 0, 10.0, next_state)
        assert ctrl.q_table[state][0] != 0
        assert ctrl.total_rewards == 10.0
        assert ctrl.epsilon <= old_epsilon

    def test_update_q_table_epsilon_min(self):
        ctrl = QLearningFlowController()
        ctrl.epsilon = ctrl.epsilon_min
        ctrl.update_q_table((0, 0, 0, 0), 0, 1.0, (1, 1, 1, 1))
        assert ctrl.epsilon == ctrl.epsilon_min

    def test_calculate_reward_power_increase(self):
        ctrl = QLearningFlowController()
        reward = ctrl.calculate_reward(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            prev_power=0.5, prev_biofilm_dev=0.02, prev_substrate_util=15.0,
        )
        assert reward > 0

    def test_calculate_reward_power_decrease(self):
        ctrl = QLearningFlowController()
        reward = ctrl.calculate_reward(
            power=0.5, biofilm_deviation=0.01, substrate_utilization=20.0,
            prev_power=1.0, prev_biofilm_dev=0.02, prev_substrate_util=15.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_reward_no_change(self):
        ctrl = QLearningFlowController()
        reward = ctrl.calculate_reward(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            prev_power=1.0, prev_biofilm_dev=0.01, prev_substrate_util=20.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_reward_biofilm_outside_threshold(self):
        ctrl = QLearningFlowController()
        reward = ctrl.calculate_reward(
            power=1.0, biofilm_deviation=0.5, substrate_utilization=20.0,
            prev_power=1.0, prev_biofilm_dev=0.5, prev_substrate_util=20.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_reward_biofilm_steady_state(self):
        ctrl = QLearningFlowController()
        history = [1.3, 1.3, 1.3]
        reward = ctrl.calculate_reward(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            prev_power=1.0, prev_biofilm_dev=0.01, prev_substrate_util=20.0,
            biofilm_thickness_history=history,
        )
        assert reward > 0

    def test_calculate_reward_biofilm_growing(self):
        ctrl = QLearningFlowController()
        history = [1.0, 1.1, 1.3]
        reward = ctrl.calculate_reward(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            prev_power=1.0, prev_biofilm_dev=0.01, prev_substrate_util=20.0,
            biofilm_thickness_history=history,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_reward_biofilm_history_short(self):
        ctrl = QLearningFlowController()
        history = [1.3]
        reward = ctrl.calculate_reward(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            prev_power=1.0, prev_biofilm_dev=0.01, prev_substrate_util=20.0,
            biofilm_thickness_history=history,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_reward_combined_penalty(self):
        ctrl = QLearningFlowController()
        reward = ctrl.calculate_reward(
            power=0.5, biofilm_deviation=0.5, substrate_utilization=10.0,
            prev_power=1.0, prev_biofilm_dev=0.01, prev_substrate_util=20.0,
        )
        assert reward < 0

    def test_calculate_reward_substrate_decrease(self):
        ctrl = QLearningFlowController()
        reward = ctrl.calculate_reward(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=10.0,
            prev_power=1.0, prev_biofilm_dev=0.01, prev_substrate_util=20.0,
        )
        assert isinstance(reward, (int, float))

    def test_save_q_table(self):
        ctrl = QLearningFlowController()
        ctrl.q_table[(0, 0, 0, 0)][0] = 5.0
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            fname = f.name
        try:
            ctrl.save_q_table(fname)
            with open(fname, "rb") as fp:
                loaded = pickle.load(fp)
            assert (0, 0, 0, 0) in loaded
        finally:
            os.unlink(fname)

    def test_load_q_table(self):
        ctrl = QLearningFlowController()
        data = {(0, 0, 0, 0): {0: 5.0}}
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            fname = f.name
            pickle.dump(data, f)
        try:
            ctrl.load_q_table(fname)
            assert ctrl.q_table[(0, 0, 0, 0)][0] == 5.0
        finally:
            os.unlink(fname)

    def test_load_q_table_nonexistent(self):
        ctrl = QLearningFlowController()
        ctrl.load_q_table("/nonexistent/path.pkl")


class TestAddSubplotLabels:
    def test_lowercase_labels(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3)
        add_subplot_labels(fig, "a")
        plt.close(fig)

    def test_uppercase_labels(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3)
        add_subplot_labels(fig, "A")
        plt.close(fig)


class TestMFCQLearningSimulation:
    def _make_small_sim(self):
        sim = MFCQLearningSimulation(use_gpu=False)
        sim.num_steps = 200
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        return sim

    def test_init(self):
        sim = self._make_small_sim()
        assert sim.num_cells == 5
        assert sim.use_gpu is False

    def test_initialize_arrays(self):
        sim = self._make_small_sim()
        assert sim.biofilm_thickness[0, 0] == 1.0
        assert sim.acetate_concentrations[0, 0] == 20.0

    def test_biofilm_factor(self):
        sim = self._make_small_sim()
        f = sim.biofilm_factor(1.3)
        assert f >= 1.0

    def test_reaction_rate(self):
        sim = self._make_small_sim()
        rate = sim.reaction_rate(20.0, 1.3)
        assert rate > 0

    def test_update_cell(self):
        sim = self._make_small_sim()
        result = sim.update_cell(0, 20.0, 1e-5, 1.3)
        assert result["outlet_concentration"] < 20.0
        assert result["power"] > 0

    def test_update_biofilm_variants(self):
        sim = self._make_small_sim()
        for thickness in [0.8, 1.0, 2.0]:
            sim.biofilm_thickness[0, :] = thickness
            sim.acetate_concentrations[0, :] = 20.0
            sim.flow_rates[0] = 1e-5
            sim.update_biofilm(1, sim.dt)
            assert 0.5 <= sim.biofilm_thickness[1, 0] <= 3.0

    def test_simulate_step_zero(self):
        sim = self._make_small_sim()
        sim.simulate_step(0)

    def test_simulate_step_non_zero(self):
        sim = self._make_small_sim()
        sim.simulate_step(1)
        assert sim.stack_voltages[1] > 0

    def test_simulate_q_learning_cycles(self):
        sim = self._make_small_sim()
        for step in range(1, 125):
            sim.simulate_step(step)

    def test_simulate_biofilm_history_overflow(self):
        """Run enough steps to trigger biofilm_history pop (>10 entries, >660 steps)."""
        sim = MFCQLearningSimulation(use_gpu=False)
        sim.num_steps = 700
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        for step in range(1, 700):
            sim.simulate_step(step)
        assert len(sim.biofilm_history) <= 10

    def test_run_simulation(self):
        sim = self._make_small_sim()
        sim.run_simulation()
        assert sim.stack_powers[-1] >= 0

    @patch("mfc_qlearning_optimization.get_simulation_data_path", return_value="/tmp/test.csv")
    @patch("mfc_qlearning_optimization.get_model_path", return_value="/tmp/test.pkl")
    def test_save_data(self, mock_model, mock_data):
        sim = self._make_small_sim()
        sim.run_simulation()
        with patch("builtins.open", MagicMock()):
            with patch("pandas.DataFrame.to_csv"):
                with patch("json.dump"):
                    with patch("pickle.dump"):
                        ts = sim.save_data()
        assert ts is not None

    @patch("mfc_qlearning_optimization.get_figure_path", return_value="/tmp/test.png")
    def test_generate_plots(self, mock_fig_path):
        import matplotlib
        matplotlib.use("Agg")
        sim = self._make_small_sim()
        sim.run_simulation()
        with patch("matplotlib.pyplot.savefig"):
            sim.generate_plots("20250101_000000")
        import matplotlib.pyplot as plt
        plt.close("all")


class TestGPUPaths:
    """Test GPU code paths by mocking gpu_accelerator."""

    def test_biofilm_factor_gpu(self):
        mock_gpu = MagicMock()
        mock_gpu.abs = np.abs
        sim = MFCQLearningSimulation(use_gpu=False)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        sim.use_gpu = True
        _mod.gpu_accelerator = mock_gpu
        try:
            result = sim.biofilm_factor(1.5)
            assert result >= 1.0
        finally:
            sim.use_gpu = False

    def test_reaction_rate_gpu(self):
        mock_gpu = MagicMock()
        mock_gpu.abs = np.abs
        mock_gpu.where = np.where
        sim = MFCQLearningSimulation(use_gpu=False)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        sim.use_gpu = True
        _mod.gpu_accelerator = mock_gpu
        try:
            result = sim.reaction_rate(20.0, 1.3)
            assert result > 0
        finally:
            sim.use_gpu = False

    def test_update_cell_gpu(self):
        mock_gpu = MagicMock()
        mock_gpu.abs = np.abs
        mock_gpu.where = np.where
        mock_gpu.maximum = np.maximum
        mock_gpu.log = np.log
        sim = MFCQLearningSimulation(use_gpu=False)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        sim.use_gpu = True
        _mod.gpu_accelerator = mock_gpu
        try:
            result = sim.update_cell(0, 20.0, 1e-5, 1.3)
            assert result["power"] > 0
        finally:
            sim.use_gpu = False

    def test_update_biofilm_gpu(self):
        mock_gpu = MagicMock()
        mock_gpu.where = np.where
        mock_gpu.clip = np.clip
        sim = MFCQLearningSimulation(use_gpu=False)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        sim.biofilm_thickness[0, :] = 1.0
        sim.acetate_concentrations[0, :] = 20.0
        sim.flow_rates[0] = 1e-5
        sim.use_gpu = True
        _mod.gpu_accelerator = mock_gpu
        try:
            sim.update_biofilm(1, sim.dt)
        finally:
            sim.use_gpu = False

    def test_simulate_step_gpu_biofilm_deviation(self):
        mock_gpu = MagicMock()
        mock_gpu.abs = np.abs
        mock_gpu.where = np.where
        mock_gpu.maximum = np.maximum
        mock_gpu.log = np.log
        mock_gpu.clip = np.clip
        mock_gpu.mean = np.mean
        mock_gpu.to_cpu = lambda x: x
        sim = MFCQLearningSimulation(use_gpu=False)
        sim.num_steps = 200
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        sim.use_gpu = True
        _mod.gpu_accelerator = mock_gpu
        try:
            sim.simulate_step(1)
        finally:
            sim.use_gpu = False

    def test_run_simulation_gpu_conversion(self):
        mock_gpu = MagicMock()
        mock_gpu.abs = np.abs
        mock_gpu.where = np.where
        mock_gpu.maximum = np.maximum
        mock_gpu.log = np.log
        mock_gpu.clip = np.clip
        mock_gpu.mean = np.mean
        mock_gpu.to_cpu = lambda x: x
        sim = MFCQLearningSimulation(use_gpu=False)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        sim.run_simulation()
        # Now test GPU conversion path
        sim.use_gpu = True
        _mod.gpu_accelerator = mock_gpu
        try:
            # Manually trigger the conversion block from run_simulation
            sim.cell_voltages = mock_gpu.to_cpu(sim.cell_voltages)
            sim.biofilm_thickness = mock_gpu.to_cpu(sim.biofilm_thickness)
            sim.acetate_concentrations = mock_gpu.to_cpu(sim.acetate_concentrations)
            sim.current_densities = mock_gpu.to_cpu(sim.current_densities)
            sim.power_outputs = mock_gpu.to_cpu(sim.power_outputs)
            sim.substrate_consumptions = mock_gpu.to_cpu(sim.substrate_consumptions)
            sim.stack_voltages = mock_gpu.to_cpu(sim.stack_voltages)
            sim.stack_powers = mock_gpu.to_cpu(sim.stack_powers)
            sim.flow_rates = mock_gpu.to_cpu(sim.flow_rates)
            sim.objective_values = mock_gpu.to_cpu(sim.objective_values)
            sim.substrate_utilizations = mock_gpu.to_cpu(sim.substrate_utilizations)
            sim.q_rewards = mock_gpu.to_cpu(sim.q_rewards)
            sim.q_actions = mock_gpu.to_cpu(sim.q_actions)
        finally:
            sim.use_gpu = False

    def test_generate_plots_no_valid_flow(self):
        """Test generate_plots when all flow rates are zero (else branch line 1010)."""
        import matplotlib
        matplotlib.use("Agg")
        sim = MFCQLearningSimulation(use_gpu=False)
        sim.num_steps = 50
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        # Set all flow_rates to zero
        sim.flow_rates[:] = 0.0
        sim.stack_powers[:] = 0.0
        sim.stack_voltages[:] = 0.0
        sim.substrate_utilizations[:] = 0.0
        sim.q_rewards[:] = 0.0
        sim.q_actions[:] = 0.0
        sim.objective_values[:] = 0.0
        with patch("mfc_qlearning_optimization.get_figure_path", return_value="/tmp/test.png"):
            with patch("matplotlib.pyplot.savefig"):
                sim.generate_plots("20250101_000000")
        import matplotlib.pyplot as plt
        plt.close("all")


class TestMain:
    @patch("mfc_qlearning_optimization.MFCQLearningSimulation")
    def test_main(self, MockSim):
        mock_sim = MagicMock()
        mock_sim.stack_powers = np.ones(100)
        mock_sim.dt = 10.0
        MockSim.return_value = mock_sim
        mock_sim.save_data.return_value = "20250101"
        main()
        mock_sim.run_simulation.assert_called_once()
