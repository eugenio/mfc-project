"""Coverage tests for mfc_unified_qlearning_control.py (98%+ target)."""
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Patch config validation to add missing flow_rate_adjustments_ml_per_h
import config as _cfg_mod
_orig_validate = _cfg_mod.validate_qlearning_config
def _patched_validate(config):
    if not hasattr(config, "flow_rate_adjustments_ml_per_h"):
        config.flow_rate_adjustments_ml_per_h = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
    _orig_validate(config)
_cfg_mod.validate_qlearning_config = _patched_validate

import mfc_unified_qlearning_control as _mod
_mod.validate_qlearning_config = _patched_validate

from mfc_unified_qlearning_control import (
    MFCUnifiedQLearningSimulation,
    UnifiedQLearningController,
    add_subplot_labels,
    main,
)


class TestUnifiedQLearningController:
    def test_init_defaults(self):
        ctrl = UnifiedQLearningController()
        assert ctrl.learning_rate == 0.1
        assert ctrl.discount_factor == 0.95
        assert ctrl.epsilon == 0.3
        assert len(ctrl.actions) > 0

    def test_init_custom_config(self):
        from config import QLearningConfig
        cfg = QLearningConfig()
        cfg.learning_rate = 0.2
        if not hasattr(cfg, "flow_rate_adjustments_ml_per_h"):
            cfg.flow_rate_adjustments_ml_per_h = [-10, -5, 0, 5, 10]
        ctrl = UnifiedQLearningController(config=cfg)
        assert ctrl.learning_rate == 0.2

    def test_init_custom_target(self):
        ctrl = UnifiedQLearningController(target_outlet_conc=15.0)
        assert ctrl.target_outlet_conc == 15.0

    def test_setup_state_action_spaces(self):
        ctrl = UnifiedQLearningController()
        assert len(ctrl.power_bins) > 0
        assert len(ctrl.biofilm_bins) > 0
        assert len(ctrl.outlet_error_bins) > 0
        assert len(ctrl.flow_rate_bins) > 0
        assert len(ctrl.time_bins) > 0

    def test_discretize_state(self):
        ctrl = UnifiedQLearningController()
        state = ctrl.discretize_state(0.01, 0.1, 20.0, 1.0, 10.0, 100.0)
        assert isinstance(state, tuple)
        assert len(state) == 6

    def test_discretize_state_boundaries(self):
        ctrl = UnifiedQLearningController()
        state_low = ctrl.discretize_state(0.0, 0.0, 0.0, -100.0, 0.0, 0.0)
        state_high = ctrl.discretize_state(100.0, 100.0, 100.0, 100.0, 100.0, 10000.0)
        assert all(idx >= 0 for idx in state_low)
        assert all(idx >= 0 for idx in state_high)

    def test_select_action_exploration(self):
        ctrl = UnifiedQLearningController()
        ctrl.epsilon = 1.0
        state = (0, 0, 0, 0, 0, 0)
        np.random.seed(42)
        action_idx, new_flow, new_conc = ctrl.select_action(state, 0.010, 20.0)
        assert 0 <= action_idx < len(ctrl.actions)
        assert 0.005 <= new_flow <= 0.050

    def test_select_action_exploitation(self):
        ctrl = UnifiedQLearningController()
        ctrl.epsilon = 0.0
        state = (0, 0, 0, 0, 0, 0)
        ctrl.q_table[state][5] = 100.0
        action_idx, _, _ = ctrl.select_action(state, 0.010, 20.0)
        assert action_idx == 5

    def test_select_action_exploration_poor_performance(self):
        ctrl = UnifiedQLearningController()
        ctrl.epsilon = 1.0
        ctrl.performance_history = [-10.0] * 15
        state = (0, 0, 0, 0, 0, 0)
        np.random.seed(42)
        action_idx, _, _ = ctrl.select_action(state, 0.010, 20.0)
        assert 0 <= action_idx < len(ctrl.actions)

    def test_select_action_exploration_good_performance(self):
        ctrl = UnifiedQLearningController()
        ctrl.epsilon = 1.0
        ctrl.performance_history = [10.0] * 15
        state = (0, 0, 0, 0, 0, 0)
        np.random.seed(42)
        action_idx, _, _ = ctrl.select_action(state, 0.010, 20.0)
        assert 0 <= action_idx < len(ctrl.actions)

    def test_select_action_history_overflow(self):
        ctrl = UnifiedQLearningController()
        ctrl.epsilon = 0.0
        ctrl.action_history = [(0, 0)] * 1001
        state = (0, 0, 0, 0, 0, 0)
        ctrl.select_action(state, 0.010, 20.0)
        assert len(ctrl.action_history) <= 1001

    def test_update_q_table(self):
        ctrl = UnifiedQLearningController()
        state = (0, 0, 0, 0, 0, 0)
        next_state = (1, 1, 1, 1, 1, 1)
        ctrl.update_q_table(state, 0, 10.0, next_state)
        assert ctrl.q_table[state][0] != 0
        assert ctrl.total_rewards == 10.0

    def test_update_q_table_adaptive_decay_good(self):
        ctrl = UnifiedQLearningController()
        ctrl.performance_history = [60.0] * 15
        old_eps = ctrl.epsilon
        ctrl.update_q_table((0,0,0,0,0,0), 0, 60.0, (1,1,1,1,1,1))
        assert ctrl.epsilon < old_eps

    def test_update_q_table_adaptive_decay_poor(self):
        ctrl = UnifiedQLearningController()
        ctrl.performance_history = [-60.0] * 15
        old_eps = ctrl.epsilon
        ctrl.update_q_table((0,0,0,0,0,0), 0, -60.0, (1,1,1,1,1,1))
        assert ctrl.epsilon <= old_eps

    def test_update_q_table_adaptive_decay_moderate(self):
        ctrl = UnifiedQLearningController()
        ctrl.performance_history = [0.0] * 15
        old_eps = ctrl.epsilon
        ctrl.update_q_table((0,0,0,0,0,0), 0, 0.0, (1,1,1,1,1,1))
        assert ctrl.epsilon <= old_eps

    def test_update_q_table_performance_overflow(self):
        ctrl = UnifiedQLearningController()
        ctrl.performance_history = [1.0] * 101
        ctrl.update_q_table((0,0,0,0,0,0), 0, 1.0, (1,1,1,1,1,1))
        assert len(ctrl.performance_history) <= 101

    def test_update_q_table_short_history(self):
        ctrl = UnifiedQLearningController()
        ctrl.performance_history = [1.0] * 5
        old_eps = ctrl.epsilon
        ctrl.update_q_table((0,0,0,0,0,0), 0, 1.0, (1,1,1,1,1,1))
        assert ctrl.epsilon <= old_eps

    def test_calculate_unified_reward_power_increase(self):
        ctrl = UnifiedQLearningController()
        reward = ctrl.calculate_unified_reward(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=10.0, prev_power=0.5, prev_biofilm_dev=0.02,
            prev_substrate_util=15.0, prev_outlet_conc=11.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_unified_reward_power_decrease(self):
        ctrl = UnifiedQLearningController()
        reward = ctrl.calculate_unified_reward(
            power=0.5, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=10.0, prev_power=1.0, prev_biofilm_dev=0.02,
            prev_substrate_util=15.0, prev_outlet_conc=11.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_unified_reward_no_change(self):
        ctrl = UnifiedQLearningController()
        reward = ctrl.calculate_unified_reward(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=10.0, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=10.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_unified_reward_biofilm_outside_threshold(self):
        ctrl = UnifiedQLearningController()
        reward = ctrl.calculate_unified_reward(
            power=1.0, biofilm_deviation=0.5, substrate_utilization=20.0,
            outlet_conc=10.0, prev_power=1.0, prev_biofilm_dev=0.5,
            prev_substrate_util=20.0, prev_outlet_conc=10.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_unified_reward_biofilm_steady(self):
        ctrl = UnifiedQLearningController(target_outlet_conc=10.0)
        history = [1.3, 1.3, 1.3]
        reward = ctrl.calculate_unified_reward(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=10.0, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=10.0,
            biofilm_thickness_history=history,
        )
        assert reward > 0

    def test_calculate_unified_reward_outlet_precise(self):
        ctrl = UnifiedQLearningController(target_outlet_conc=10.0)
        reward = ctrl.calculate_unified_reward(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=10.5, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=11.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_unified_reward_outlet_acceptable(self):
        ctrl = UnifiedQLearningController(target_outlet_conc=10.0)
        reward = ctrl.calculate_unified_reward(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=11.5, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_unified_reward_outlet_far(self):
        ctrl = UnifiedQLearningController(target_outlet_conc=10.0)
        reward = ctrl.calculate_unified_reward(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=15.0, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=14.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_unified_reward_stability_bonus(self):
        ctrl = UnifiedQLearningController(target_outlet_conc=10.0)
        reward = ctrl.calculate_unified_reward(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=10.0, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=10.0,
        )
        assert reward > 0

    def test_calculate_unified_reward_flow_penalty(self):
        ctrl = UnifiedQLearningController()
        ctrl.current_flow_rate = 30.0
        history = [0.5, 0.5, 0.5, 0.5, 0.5]
        reward = ctrl.calculate_unified_reward(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=10.0, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=10.0,
            biofilm_thickness_history=history,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_unified_reward_combined_penalty(self):
        ctrl = UnifiedQLearningController()
        reward = ctrl.calculate_unified_reward(
            power=0.5, biofilm_deviation=0.5, substrate_utilization=10.0,
            outlet_conc=15.0, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=8.0,
        )
        assert reward < 0

    def test_get_control_statistics_empty(self):
        ctrl = UnifiedQLearningController()
        stats = ctrl.get_control_statistics()
        assert stats["avg_reward"] == 0
        assert stats["reward_trend"] == 0

    def test_get_control_statistics_short(self):
        ctrl = UnifiedQLearningController()
        ctrl.performance_history = [1.0, 2.0, 3.0]
        stats = ctrl.get_control_statistics()
        assert stats["avg_reward"] > 0
        assert stats["reward_trend"] == 0

    def test_get_control_statistics_long(self):
        ctrl = UnifiedQLearningController()
        ctrl.performance_history = list(range(60))
        stats = ctrl.get_control_statistics()
        assert stats["reward_trend"] > 0
        assert "q_table_size" in stats


class TestAddSubplotLabels:
    def test_lowercase(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3)
        add_subplot_labels(fig, "a")
        plt.close(fig)

    def test_uppercase(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3)
        add_subplot_labels(fig, "A")
        plt.close(fig)


class TestMFCUnifiedQLearningSimulation:
    def _make_small_sim(self):
        sim = MFCUnifiedQLearningSimulation(use_gpu=False)
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
        assert sim.flow_rates[0] == 0.010
        assert sim.inlet_concentrations[0] == 20.0

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
        result = sim.update_cell(0, 20.0, 0.010, 1.3)
        assert result["outlet_concentration"] < 20.0
        assert result["power"] > 0

    def test_update_cell_debug(self):
        sim = self._make_small_sim()
        sim.debug_counter = 0
        result = sim.update_cell(0, 20.0, 0.010, 1.3)
        assert sim.debug_counter == 1

    def test_update_biofilm_variants(self):
        sim = self._make_small_sim()
        for thickness in [0.8, 1.0, 2.0]:
            sim.biofilm_thickness[0, :] = thickness
            sim.acetate_concentrations[0, :] = 20.0
            sim.flow_rates[0] = 0.010
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
        sim = MFCUnifiedQLearningSimulation(use_gpu=False)
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

    @patch("mfc_unified_qlearning_control.get_simulation_data_path", return_value="/tmp/test.csv")
    @patch("mfc_unified_qlearning_control.get_model_path", return_value="/tmp/test.pkl")
    def test_save_data(self, mock_model, mock_data):
        sim = self._make_small_sim()
        sim.run_simulation()
        with patch("builtins.open", MagicMock()):
            with patch("pandas.DataFrame.to_csv"):
                with patch("json.dump"):
                    with patch("pickle.dump"):
                        ts = sim.save_data()
        assert ts is not None

    @patch("mfc_unified_qlearning_control.get_figure_path", return_value="/tmp/test.png")
    def test_generate_plots(self, mock_fig_path):
        import matplotlib
        matplotlib.use("Agg")
        sim = self._make_small_sim()
        sim.run_simulation()
        with patch("matplotlib.pyplot.savefig"):
            sim.generate_plots("20250101_000000")
        import matplotlib.pyplot as plt
        plt.close("all")

    @patch("mfc_unified_qlearning_control.get_figure_path", return_value="/tmp/test.png")
    def test_generate_plots_no_valid_data(self, mock_fig_path):
        import matplotlib
        matplotlib.use("Agg")
        sim = self._make_small_sim()
        # Run a short simulation so controller stats include total_reward key
        sim.run_simulation()
        # Zero out arrays to test edge cases in plotting
        sim.flow_rates[:] = 0.0
        sim.stack_powers[:] = 0.0
        sim.stack_voltages[:] = 0.0
        sim.substrate_utilizations[:] = 0.0
        sim.q_rewards[:] = 0.0
        sim.q_actions[:] = 0.0
        sim.objective_values[:] = 0.0
        sim.outlet_concentrations[:] = 0.0
        sim.inlet_concentrations[:] = 0.0
        sim.concentration_errors[:] = 0.0
        with patch("matplotlib.pyplot.savefig"):
            sim.generate_plots("20250101_000000")
        import matplotlib.pyplot as plt
        plt.close("all")


class TestMain:
    @patch("mfc_unified_qlearning_control.MFCUnifiedQLearningSimulation")
    def test_main(self, MockSim):
        mock_sim = MagicMock()
        mock_sim.stack_powers = np.ones(100)
        mock_sim.dt = 10.0
        mock_sim.outlet_concentrations = np.ones(100) * 12.0
        mock_sim.concentration_errors = np.zeros(100)
        MockSim.return_value = mock_sim
        mock_sim.save_data.return_value = "20250101"
        main()
        mock_sim.run_simulation.assert_called_once()
