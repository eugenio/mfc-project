"""Tests for MFC Unified Q-Learning Control module.

Coverage target: 50%+ (from 19.55%)
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.fixture
def mock_config():
    """Create a mock QLearningConfig for testing."""
    config = MagicMock()
    config.learning_rate = 0.1
    config.discount_factor = 0.95
    config.epsilon = 0.3
    config.epsilon_decay = 0.9995
    config.epsilon_min = 0.01
    config.power_max = 2.0
    config.power_bins = 10
    config.biofilm_max_deviation = 1.0
    config.biofilm_deviation_bins = 6
    config.substrate_utilization_max = 50.0
    config.substrate_utilization_bins = 8
    config.outlet_error_range = (-10.0, 10.0)
    config.outlet_error_bins = 8
    config.flow_rate_range = (5.0, 50.0)
    config.flow_rate_bins = 6
    config.time_phase_hours = [200, 500, 800, 1000]
    config.unified_flow_actions = [-8, -4, -2, -1, 0, 1, 2, 3, 4]
    config.substrate_actions = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    config.substrate_concentration_min = 5.0
    config.substrate_concentration_max = 45.0
    config.stability_target_outlet_concentration = 12.0
    return config


@pytest.fixture
def controller(mock_config):
    """Create a UnifiedQLearningController instance for testing."""
    with patch(
        "mfc_unified_qlearning_control.validate_qlearning_config", return_value=True
    ):
        from mfc_unified_qlearning_control import UnifiedQLearningController
        return UnifiedQLearningController(config=mock_config)


class TestUnifiedQLearningControllerInit:
    """Tests for controller initialization."""

    def test_init_with_config(self, mock_config):
        with patch(
            "mfc_unified_qlearning_control.validate_qlearning_config", return_value=True
        ):
            from mfc_unified_qlearning_control import UnifiedQLearningController
            ctrl = UnifiedQLearningController(config=mock_config)
            assert ctrl.learning_rate == 0.1
            assert ctrl.discount_factor == 0.95

    def test_target_outlet_override(self, mock_config):
        with patch(
            "mfc_unified_qlearning_control.validate_qlearning_config", return_value=True
        ):
            from mfc_unified_qlearning_control import UnifiedQLearningController
            ctrl = UnifiedQLearningController(
                config=mock_config, target_outlet_conc=15.0
            )
            assert ctrl.target_outlet_conc == 15.0

    def test_q_table_is_defaultdict(self, controller):
        assert isinstance(controller.q_table, defaultdict)
        state = (0, 0, 0, 0, 0, 0)
        assert controller.q_table[state][0] == 0.0

    def test_statistics_initialized(self, controller):
        assert controller.total_rewards == 0
        assert controller.action_history == []


class TestStateActionSpaces:
    """Tests for state and action space setup."""

    def test_power_bins(self, controller):
        assert len(controller.power_bins) == 10

    def test_action_space_size(self, controller):
        assert len(controller.actions) == 9 * 7

    def test_actions_are_tuples(self, controller):
        for action in controller.actions:
            assert isinstance(action, tuple)
            assert len(action) == 2


class TestDiscretizeState:
    """Tests for state discretization."""

    def test_returns_6_tuple(self, controller):
        state = controller.discretize_state(0.5, 0.2, 10.0, 0.0, 15.0, 300)
        assert isinstance(state, tuple)
        assert len(state) == 6

    def test_min_values(self, controller):
        state = controller.discretize_state(0.0, 0.0, 0.0, -10.0, 5.0, 0)
        assert state == (0, 0, 0, 0, 0, 0)

    def test_clips_high_values(self, controller):
        state = controller.discretize_state(100.0, 100.0, 1000.0, 100.0, 1000.0, 5000)
        for idx in state:
            assert idx >= 0


class TestSelectAction:
    """Tests for action selection."""

    def test_returns_correct_types(self, controller):
        state = (0, 0, 0, 0, 0, 0)
        action_idx, new_flow, new_conc = controller.select_action(
            state, current_flow_rate=0.010, current_inlet_conc=20.0
        )
        assert isinstance(action_idx, (int, np.integer))

    def test_exploration_mode(self, controller):
        controller.epsilon = 1.0
        state = (0, 0, 0, 0, 0, 0)
        np.random.seed(42)
        actions = set()
        for _ in range(50):
            action_idx, _, _ = controller.select_action(
                state, current_flow_rate=0.010, current_inlet_conc=20.0
            )
            actions.add(action_idx)
        assert len(actions) > 1

    def test_exploitation_mode(self, controller):
        controller.epsilon = 0.0
        state = (0, 0, 0, 0, 0, 0)
        controller.q_table[state][5] = 100.0
        action_idx, _, _ = controller.select_action(
            state, current_flow_rate=0.010, current_inlet_conc=20.0
        )
        assert action_idx == 5

    def test_flow_rate_bounds(self, controller):
        controller.epsilon = 1.0
        state = (0, 0, 0, 0, 0, 0)
        for _ in range(20):
            _, new_flow, _ = controller.select_action(
                state, current_flow_rate=0.010, current_inlet_conc=20.0
            )
            assert 0.005 <= new_flow <= 0.050


class TestUpdateQTable:
    """Tests for Q-table updates."""

    def test_positive_reward_increases_q(self, controller):
        state = (0, 0, 0, 0, 0, 0)
        next_state = (1, 1, 1, 1, 1, 1)
        initial_q = controller.q_table[state][0]
        controller.update_q_table(state, 0, 100.0, next_state)
        assert controller.q_table[state][0] > initial_q

    def test_updates_total_rewards(self, controller):
        state = (0, 0, 0, 0, 0, 0)
        next_state = (1, 1, 1, 1, 1, 1)
        controller.update_q_table(state, 0, 50.0, next_state)
        assert controller.total_rewards == 50.0

    def test_epsilon_decays(self, controller):
        state = (0, 0, 0, 0, 0, 0)
        next_state = (1, 1, 1, 1, 1, 1)
        initial_epsilon = controller.epsilon
        controller.update_q_table(state, 0, 0.0, next_state)
        assert controller.epsilon <= initial_epsilon


class TestCalculateUnifiedReward:
    """Tests for reward calculation."""

    def test_positive_power_change(self, controller):
        reward = controller.calculate_unified_reward(
            power=1.0, biofilm_deviation=0.05, substrate_utilization=10.0,
            outlet_conc=12.0, prev_power=0.5, prev_biofilm_dev=0.05,
            prev_substrate_util=10.0, prev_outlet_conc=12.0,
        )
        assert reward > 0

    def test_biofilm_optimal(self, controller):
        reward = controller.calculate_unified_reward(
            power=1.0, biofilm_deviation=0.0, substrate_utilization=10.0,
            outlet_conc=12.0, prev_power=1.0, prev_biofilm_dev=0.0,
            prev_substrate_util=10.0, prev_outlet_conc=12.0,
        )
        assert reward > 0

    def test_concentration_improvement(self, controller):
        reward = controller.calculate_unified_reward(
            power=1.0, biofilm_deviation=0.05, substrate_utilization=10.0,
            outlet_conc=12.0, prev_power=1.0, prev_biofilm_dev=0.05,
            prev_substrate_util=10.0, prev_outlet_conc=14.0,
        )
        assert reward > 0

    def test_with_biofilm_history(self, controller):
        reward = controller.calculate_unified_reward(
            power=1.0, biofilm_deviation=0.01, substrate_utilization=10.0,
            outlet_conc=12.0, prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=10.0, prev_outlet_conc=12.0,
            biofilm_thickness_history=[1.3, 1.3, 1.3],
        )
        assert isinstance(reward, (int, float))


class TestGetControlStatistics:
    """Tests for control statistics."""

    def test_empty_history(self, controller):
        stats = controller.get_control_statistics()
        assert stats["avg_reward"] == 0
        assert stats["reward_trend"] == 0

    def test_with_history(self, controller):
        controller.performance_history = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = controller.get_control_statistics()
        assert stats["avg_reward"] == pytest.approx(30.0)


class TestMFCSimulation:
    """Tests for MFCUnifiedQLearningSimulation."""

    @pytest.fixture
    def mock_deps(self):
        path_se = lambda x: f"/tmp/{x}"  # noqa: E731
        with (
            patch("mfc_unified_qlearning_control.GPU_AVAILABLE", False),
            patch("mfc_unified_qlearning_control.gpu_accelerator"),
            patch(
                "mfc_unified_qlearning_control.validate_qlearning_config",
                return_value=True,
            ),
            patch("mfc_unified_qlearning_control.QLearningConfig") as MC,
            patch(
                "mfc_unified_qlearning_control.get_figure_path",
                side_effect=path_se,
            ),
            patch(
                "mfc_unified_qlearning_control.get_model_path",
                side_effect=path_se,
            ),
            patch(
                "mfc_unified_qlearning_control.get_simulation_data_path",
                side_effect=path_se,
            ),
            patch("os.makedirs"),
        ):
            MC.return_value = MagicMock(
                learning_rate=0.1, discount_factor=0.95, epsilon=0.3,
                epsilon_decay=0.9995, epsilon_min=0.01,
                stability_target_outlet_concentration=12.0,
                substrate_concentration_min=5.0, substrate_concentration_max=45.0,
                power_max=2.0, power_bins=10, biofilm_max_deviation=1.0,
                biofilm_deviation_bins=6, substrate_utilization_max=50.0,
                substrate_utilization_bins=8, outlet_error_range=(-10.0, 10.0),
                outlet_error_bins=8, flow_rate_range=(5.0, 50.0), flow_rate_bins=6,
                time_phase_hours=[200, 500, 800, 1000],
                unified_flow_actions=[-8, -4, -2, -1, 0, 1, 2, 3, 4],
                substrate_actions=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            )
            yield

    def test_init_no_gpu(self, mock_deps):
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        assert sim.use_gpu is False

    def test_parameters(self, mock_deps):
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        assert sim.dt == 10.0
        assert sim.num_cells == 5

    def test_arrays_initialized(self, mock_deps):
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        assert sim.cell_voltages.shape == (sim.num_steps, 5)

    def test_biofilm_factor(self, mock_deps):
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        factor = sim.biofilm_factor(1.3)
        assert factor == pytest.approx(1.0, abs=0.01)

    def test_reaction_rate(self, mock_deps):
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        rate = sim.reaction_rate(c_ac=20.0, biofilm=1.3)
        assert rate > 0

    def test_update_cell(self, mock_deps):
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        result = sim.update_cell(0, 20.0, 0.010, 1.3)
        assert result["outlet_concentration"] < 20.0
        assert result["power"] > 0

    def test_update_biofilm(self, mock_deps):
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.update_biofilm(step=1, dt=10.0)
        for i in range(sim.num_cells):
            assert sim.biofilm_thickness[1, i] > 0

    def test_simulate_step_zero(self, mock_deps):
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.simulate_step(0)
        assert sim.stack_powers[0] == 0.0

    def test_simulate_step_updates(self, mock_deps):
        from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.simulate_step(1)
        assert sim.stack_powers[1] > 0


class TestAddSubplotLabels:
    """Tests for subplot labels helper."""

    def test_lowercase(self):
        with patch("mfc_unified_qlearning_control.validate_qlearning_config"):
            import matplotlib.pyplot as plt
            from mfc_unified_qlearning_control import add_subplot_labels
            fig, _ = plt.subplots(2, 2)
            add_subplot_labels(fig, start_letter="a")
            plt.close(fig)

    def test_uppercase(self):
        with patch("mfc_unified_qlearning_control.validate_qlearning_config"):
            import matplotlib.pyplot as plt
            from mfc_unified_qlearning_control import add_subplot_labels
            fig, _ = plt.subplots(2, 2)
            add_subplot_labels(fig, start_letter="A")
            plt.close(fig)


class TestIntegration:
    """Integration tests."""

    def test_control_loop(self, controller):
        state = controller.discretize_state(0.5, 0.1, 10.0, 2.0, 15.0, 100)
        action_idx, new_flow, new_conc = controller.select_action(
            state, current_flow_rate=0.015, current_inlet_conc=20.0
        )
        next_state = controller.discretize_state(
            0.6, 0.08, 12.0, 1.5, new_flow * 1000, 110
        )
        reward = controller.calculate_unified_reward(
            0.6, 0.08, 12.0, 13.5, 0.5, 0.1, 10.0, 14.0,
        )
        controller.update_q_table(state, action_idx, reward, next_state)
        assert controller.q_table[state][action_idx] != 0.0

    def test_learning(self, controller):
        state = (0, 0, 0, 0, 0, 0)
        next_state = (1, 1, 1, 1, 1, 1)
        for _ in range(100):
            controller.update_q_table(state, 10, 100.0, next_state)
        num_actions = len(controller.actions)
        q_values = [controller.q_table[state][i] for i in range(num_actions)]
        assert np.argmax(q_values) == 10


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_power(self, controller):
        reward = controller.calculate_unified_reward(
            0.0, 0.1, 5.0, 12.0, 0.0, 0.1, 5.0, 12.0,
        )
        assert isinstance(reward, (int, float))

    def test_empty_biofilm_history(self, controller):
        reward = controller.calculate_unified_reward(
            1.0, 0.1, 10.0, 12.0, 0.5, 0.1, 10.0, 12.0,
            biofilm_thickness_history=[],
        )
        assert isinstance(reward, (int, float))
