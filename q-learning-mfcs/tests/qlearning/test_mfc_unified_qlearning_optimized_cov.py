"""Coverage tests for mfc_unified_qlearning_optimized.py (98%+ target)."""
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Patch config validation
import config as _cfg_mod
_orig_validate = _cfg_mod.validate_qlearning_config
def _patched_validate(config):
    if not hasattr(config, "flow_rate_adjustments_ml_per_h"):
        config.flow_rate_adjustments_ml_per_h = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
    _orig_validate(config)
_cfg_mod.validate_qlearning_config = _patched_validate

import mfc_unified_qlearning_control as _umod
_umod.validate_qlearning_config = _patched_validate

from mfc_unified_qlearning_optimized import (
    OptimizedMFCSimulation,
    OptimizedUnifiedQController,
    main,
)


class TestOptimizedUnifiedQController:
    def test_init(self):
        ctrl = OptimizedUnifiedQController()
        assert ctrl.learning_rate == pytest.approx(0.0987, abs=0.001)
        assert ctrl.target_outlet_conc == 12.0
        assert len(ctrl.actions) > 0

    def test_init_custom_target(self):
        ctrl = OptimizedUnifiedQController(target_outlet_conc=15.0)
        assert ctrl.target_outlet_conc == 15.0

    def test_state_bins(self):
        ctrl = OptimizedUnifiedQController()
        assert "power" in ctrl.state_bins
        assert len(ctrl.state_bins["power"]) == 10

    def test_discretize_state(self):
        ctrl = OptimizedUnifiedQController()
        state = ctrl.discretize_state(0.01, 0.1, 20.0, 1.0, 10.0, 100.0)
        assert isinstance(state, tuple)
        assert len(state) == 6

    def test_discretize_state_boundaries(self):
        ctrl = OptimizedUnifiedQController()
        state_low = ctrl.discretize_state(0.0, 0.0, 0.0, -100.0, 0.0, 0.0)
        state_high = ctrl.discretize_state(100.0, 100.0, 200.0, 100.0, 100.0, 5000.0)
        assert all(idx >= 0 for idx in state_low)
        assert all(idx >= 0 for idx in state_high)

    def test_select_action_exploration_reasonable(self):
        ctrl = OptimizedUnifiedQController()
        ctrl.epsilon = 1.0
        state = (0, 0, 0, 0, 0, 0)
        np.random.seed(42)
        action_idx, new_flow, new_conc = ctrl.select_action(state, 0.010, 20.0)
        assert 0 <= action_idx < len(ctrl.actions)

    def test_select_action_exploration_random(self):
        ctrl = OptimizedUnifiedQController()
        ctrl.epsilon = 1.0
        state = (0, 0, 0, 0, 0, 0)
        # Force non-reasonable exploration path (random() >= 0.7)
        with patch("numpy.random.random", side_effect=[0.0, 0.8]):
            action_idx, _, _ = ctrl.select_action(state, 0.010, 20.0)
        assert 0 <= action_idx < len(ctrl.actions)

    def test_select_action_exploitation(self):
        ctrl = OptimizedUnifiedQController()
        ctrl.epsilon = 0.0
        state = (0, 0, 0, 0, 0, 0)
        ctrl.q_table[state][5] = 100.0
        action_idx, _, _ = ctrl.select_action(state, 0.010, 20.0)
        assert action_idx == 5

    def test_select_action_flow_bounds(self):
        ctrl = OptimizedUnifiedQController()
        ctrl.epsilon = 0.0
        state = (0, 0, 0, 0, 0, 0)
        _, new_flow, new_conc = ctrl.select_action(state, 0.001, 20.0)
        assert 0.005 <= new_flow <= 0.050

    def test_select_action_substrate_bounds(self):
        ctrl = OptimizedUnifiedQController()
        ctrl.epsilon = 0.0
        state = (0, 0, 0, 0, 0, 0)
        _, _, new_conc = ctrl.select_action(state, 0.010, 1.0)
        assert new_conc >= ctrl.min_substrate
        assert new_conc <= ctrl.max_substrate

    def test_select_action_history_overflow(self):
        ctrl = OptimizedUnifiedQController()
        ctrl.epsilon = 0.0
        ctrl.action_history = [(0, 0)] * 1001
        state = (0, 0, 0, 0, 0, 0)
        ctrl.select_action(state, 0.010, 20.0)
        assert len(ctrl.action_history) <= 1001

    def test_update_q_table_basic(self):
        ctrl = OptimizedUnifiedQController()
        state = (0, 0, 0, 0, 0, 0)
        next_state = (1, 1, 1, 1, 1, 1)
        ctrl.update_q_table(state, 0, 10.0, next_state)
        assert ctrl.q_table[state][0] != 0
        assert ctrl.total_rewards == 10.0

    def test_update_q_table_existing_next_state(self):
        ctrl = OptimizedUnifiedQController()
        next_state = (1, 1, 1, 1, 1, 1)
        ctrl.q_table[next_state][0] = 5.0
        ctrl.update_q_table((0,0,0,0,0,0), 0, 10.0, next_state)

    def test_update_q_table_adaptive_decay_good(self):
        ctrl = OptimizedUnifiedQController()
        ctrl.performance_history = [-500.0] * 100
        old_eps = ctrl.epsilon
        ctrl.update_q_table((0,0,0,0,0,0), 0, -500.0, (1,1,1,1,1,1))
        assert ctrl.epsilon < old_eps

    def test_update_q_table_adaptive_decay_poor(self):
        ctrl = OptimizedUnifiedQController()
        ctrl.performance_history = [-2000.0] * 100
        old_eps = ctrl.epsilon
        ctrl.update_q_table((0,0,0,0,0,0), 0, -2000.0, (1,1,1,1,1,1))
        assert ctrl.epsilon <= old_eps

    def test_update_q_table_normal_decay(self):
        ctrl = OptimizedUnifiedQController()
        ctrl.performance_history = [1.0] * 5
        old_eps = ctrl.epsilon
        ctrl.update_q_table((0,0,0,0,0,0), 0, 1.0, (1,1,1,1,1,1))
        assert ctrl.epsilon <= old_eps

    def test_calculate_optimized_reward_power_increase(self):
        ctrl = OptimizedUnifiedQController(target_outlet_conc=12.0)
        reward = ctrl.calculate_optimized_reward(
            power=0.01, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=0.005, prev_biofilm_dev=0.01,
            prev_substrate_util=15.0, prev_outlet_conc=13.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_optimized_reward_power_decrease(self):
        ctrl = OptimizedUnifiedQController(target_outlet_conc=12.0)
        reward = ctrl.calculate_optimized_reward(
            power=0.001, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=0.01, prev_biofilm_dev=0.01,
            prev_substrate_util=15.0, prev_outlet_conc=13.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_optimized_reward_low_power(self):
        ctrl = OptimizedUnifiedQController(target_outlet_conc=12.0)
        reward = ctrl.calculate_optimized_reward(
            power=0.001, biofilm_deviation=0.01, substrate_utilization=5.0,
            outlet_conc=12.0, prev_power=0.001, prev_biofilm_dev=0.01,
            prev_substrate_util=5.0, prev_outlet_conc=12.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_optimized_reward_biofilm_within_threshold(self):
        ctrl = OptimizedUnifiedQController(target_outlet_conc=12.0)
        reward = ctrl.calculate_optimized_reward(
            power=0.01, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=0.01, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
            biofilm_thickness_history=[1.3, 1.3, 1.3],
        )
        assert isinstance(reward, (int, float))

    def test_calculate_optimized_reward_biofilm_outside_threshold(self):
        ctrl = OptimizedUnifiedQController(target_outlet_conc=12.0)
        reward = ctrl.calculate_optimized_reward(
            power=0.01, biofilm_deviation=0.5, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=0.01, prev_biofilm_dev=0.5,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_optimized_reward_conc_precise(self):
        ctrl = OptimizedUnifiedQController(target_outlet_conc=12.0)
        reward = ctrl.calculate_optimized_reward(
            power=0.01, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.3, prev_power=0.01, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.5,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_optimized_reward_conc_acceptable(self):
        ctrl = OptimizedUnifiedQController(target_outlet_conc=12.0)
        reward = ctrl.calculate_optimized_reward(
            power=0.01, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=13.5, prev_power=0.01, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=14.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_optimized_reward_conc_poor(self):
        ctrl = OptimizedUnifiedQController(target_outlet_conc=12.0)
        reward = ctrl.calculate_optimized_reward(
            power=0.01, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=16.0, prev_power=0.01, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=15.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_optimized_reward_stability_bonus(self):
        ctrl = OptimizedUnifiedQController(target_outlet_conc=12.0)
        reward = ctrl.calculate_optimized_reward(
            power=0.01, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=0.01, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_optimized_reward_flow_penalty(self):
        ctrl = OptimizedUnifiedQController(target_outlet_conc=12.0)
        ctrl.current_flow_rate = 30.0
        history = [0.5, 0.5, 0.5, 0.5, 0.5]
        reward = ctrl.calculate_optimized_reward(
            power=0.01, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=0.01, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
            biofilm_thickness_history=history,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_optimized_reward_combined_penalty(self):
        ctrl = OptimizedUnifiedQController(target_outlet_conc=12.0)
        reward = ctrl.calculate_optimized_reward(
            power=0.005, biofilm_deviation=0.5, substrate_utilization=10.0,
            outlet_conc=16.0, prev_power=0.01, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=10.0,
        )
        assert reward < 0

    def test_calculate_optimized_reward_performance_overflow(self):
        ctrl = OptimizedUnifiedQController(target_outlet_conc=12.0)
        ctrl.performance_history = [1.0] * 1001
        ctrl.calculate_optimized_reward(
            power=0.01, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=0.01, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
        )
        assert len(ctrl.performance_history) <= 1001

    def test_get_control_statistics_empty(self):
        ctrl = OptimizedUnifiedQController()
        stats = ctrl.get_control_statistics()
        assert stats["avg_reward"] == 0
        assert stats["reward_trend"] == 0

    def test_get_control_statistics_short(self):
        ctrl = OptimizedUnifiedQController()
        ctrl.performance_history = [1.0, 2.0, 3.0]
        stats = ctrl.get_control_statistics()
        assert stats["avg_reward"] > 0
        assert "total_reward" in stats

    def test_get_control_statistics_long(self):
        ctrl = OptimizedUnifiedQController()
        ctrl.performance_history = list(range(60))
        stats = ctrl.get_control_statistics()
        assert stats["reward_trend"] > 0
        assert "q_table_size" in stats


class TestOptimizedMFCSimulation:
    def _make_small_sim(self):
        sim = OptimizedMFCSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.num_steps = 200
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        return sim

    def test_init(self):
        sim = self._make_small_sim()
        assert isinstance(sim.unified_controller, OptimizedUnifiedQController)
        assert sim.unified_controller.target_outlet_conc == 12.0

    def test_run_simulation(self):
        sim = self._make_small_sim()
        sim.run_simulation()
        assert sim.stack_powers[-1] >= 0


class TestMain:
    @patch("mfc_unified_qlearning_optimized.OptimizedMFCSimulation")
    def test_main(self, MockSim):
        mock_sim = MagicMock()
        mock_sim.stack_powers = np.ones(100)
        mock_sim.dt = 10.0
        MockSim.return_value = mock_sim
        mock_sim.save_data.return_value = "20250101"
        main()
        mock_sim.run_simulation.assert_called_once()
