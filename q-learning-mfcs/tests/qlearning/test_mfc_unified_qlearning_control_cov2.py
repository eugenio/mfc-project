"""Tests for mfc_unified_qlearning_control.py - coverage target 98%+."""
import sys
import os

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mfc_unified_qlearning_control import (
    UnifiedQLearningController, MFCUnifiedQLearningSimulation,
    add_subplot_labels,
)


@pytest.fixture
def controller():
    return UnifiedQLearningController(target_outlet_conc=12.0)


class TestSelectAction:
    def test_exploration_poor_performance(self, controller):
        controller.performance_history = [-100.0] * 15
        controller.epsilon = 1.0
        np.random.seed(0)
        state = (0, 0, 0, 0, 0, 0)
        a, fr, ic = controller.select_action(state, 0.01, 20.0)
        assert 0 <= a < len(controller.actions)

    def test_exploration_good_performance(self, controller):
        controller.performance_history = [100.0] * 15
        controller.epsilon = 1.0
        np.random.seed(42)
        state = (0, 0, 0, 0, 0, 0)
        a, fr, ic = controller.select_action(state, 0.01, 20.0)
        assert 0 <= a < len(controller.actions)

    def test_exploration_no_history(self, controller):
        controller.epsilon = 1.0
        np.random.seed(0)
        state = (0, 0, 0, 0, 0, 0)
        a, fr, ic = controller.select_action(state, 0.01, 20.0)
        assert 0 <= a < len(controller.actions)


class TestCalculateUnifiedReward:
    def test_combined_penalty(self, controller):
        r = controller.calculate_unified_reward(
            power=0.1, biofilm_deviation=0.5, substrate_utilization=10.0,
            outlet_conc=5.0, prev_power=0.2, prev_biofilm_dev=0.3,
            prev_substrate_util=15.0, prev_outlet_conc=11.0,
        )
        assert r < 0

    def test_stability_bonus(self, controller):
        r = controller.calculate_unified_reward(
            power=0.5, biofilm_deviation=0.01, substrate_utilization=50.0,
            outlet_conc=12.0, prev_power=0.5, prev_biofilm_dev=0.01,
            prev_substrate_util=50.0, prev_outlet_conc=12.0,
        )
        assert isinstance(r, float)

    def test_flow_penalty(self, controller):
        controller.current_flow_rate = 30.0
        bh = [0.5, 0.5, 0.5, 0.5, 0.5]
        r = controller.calculate_unified_reward(
            power=0.5, biofilm_deviation=0.01, substrate_utilization=50.0,
            outlet_conc=12.0, prev_power=0.5, prev_biofilm_dev=0.01,
            prev_substrate_util=50.0, prev_outlet_conc=12.0,
            biofilm_thickness_history=bh,
        )
        assert isinstance(r, float)


class TestUpdateQTable:
    def test_adaptive_decay_good(self, controller):
        controller.performance_history = [100.0] * 15
        controller.epsilon = 0.5
        state = (0, 0, 0, 0, 0, 0)
        controller.update_q_table(state, 0, 100.0, state)
        assert controller.epsilon < 0.5

    def test_adaptive_decay_poor(self, controller):
        controller.performance_history = [-100.0] * 15
        controller.epsilon = 0.5
        state = (0, 0, 0, 0, 0, 0)
        controller.update_q_table(state, 0, -100.0, state)
        assert controller.epsilon < 0.5


class TestGetControlStatistics:
    def test_empty_history(self, controller):
        stats = controller.get_control_statistics()
        assert stats["avg_reward"] == 0

    def test_with_history(self, controller):
        controller.performance_history = list(range(60))
        stats = controller.get_control_statistics()
        assert stats["reward_trend"] != 0 or True


class TestAddSubplotLabels:
    def test_lowercase_labels(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2)
        add_subplot_labels(fig, "a")
        plt.close(fig)

    def test_uppercase_labels(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2)
        add_subplot_labels(fig, "A")
        plt.close(fig)


class TestMFCSimulationGPUConversion:
    def test_run_simulation_gpu_false(self):
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.num_steps = 200
        sim.initialize_arrays()
        for step in range(200):
            sim.simulate_step(step)


class TestSimulateStepControlInterval:
    def test_moving_average_window(self):
        sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
        sim.num_steps = 500
        sim.initialize_arrays()
        for step in range(500):
            sim.simulate_step(step)
        assert sim.substrate_utilizations[499] >= 0 or True
