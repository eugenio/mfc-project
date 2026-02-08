"""Coverage boost tests for mfc_qlearning_optimization_parallel.py - targeting remaining uncovered lines."""
import os
import sys
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mfc_qlearning_optimization_parallel import (
    MFCQLearningSimulationParallel,
    QLearningFlowController,
    add_subplot_labels,
)


class TestQLearningFlowControllerEdgeCases:
    def test_discretize_state_negative_values(self):
        ctrl = QLearningFlowController()
        state = ctrl.discretize_state(-1.0, -0.5, -10.0, -100.0)
        assert all(idx >= 0 for idx in state)

    def test_select_action_all_zeros(self):
        ctrl = QLearningFlowController()
        ctrl.epsilon = 0.0
        state = (0, 0, 0, 0)
        action_idx, flow = ctrl.select_action(state, 0.025)
        assert isinstance(action_idx, (int, np.integer))

    def test_update_q_table_multiple_times(self):
        ctrl = QLearningFlowController()
        for i in range(5):
            ctrl.update_q_table((i, 0, 0, 0), i % 9, float(i), (i + 1, 0, 0, 0))
        assert ctrl.total_rewards == sum(range(5))

    def test_calculate_reward_biofilm_at_threshold(self):
        ctrl = QLearningFlowController()
        optimal = 1.3
        threshold = 0.05 * optimal
        reward = ctrl.calculate_reward(
            power=1.0, biofilm_deviation=threshold,
            substrate_utilization=20.0,
            prev_power=1.0, prev_biofilm_dev=threshold,
            prev_substrate_util=20.0,
        )
        assert isinstance(reward, (int, float))

    def test_calculate_reward_history_two_entries(self):
        ctrl = QLearningFlowController()
        reward = ctrl.calculate_reward(
            power=1.0, biofilm_deviation=0.01,
            substrate_utilization=20.0,
            prev_power=1.0, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0,
            biofilm_thickness_history=[1.3, 1.3],
        )
        assert isinstance(reward, (int, float))

    def test_epsilon_decays_but_not_below_min(self):
        ctrl = QLearningFlowController()
        ctrl.epsilon = 0.06
        for _ in range(100):
            ctrl.update_q_table((0, 0, 0, 0), 0, 1.0, (1, 1, 1, 1))
        assert ctrl.epsilon >= ctrl.epsilon_min


class TestSimulationEdgeCases:
    def _make_sim(self, steps=200):
        sim = MFCQLearningSimulationParallel(use_gpu=False)
        sim.num_steps = steps
        sim.total_time = sim.num_steps * sim.dt
        sim.initialize_arrays()
        return sim

    def test_biofilm_factor_at_optimal(self):
        sim = self._make_sim()
        f = sim.biofilm_factor(sim.optimal_biofilm_thickness)
        assert f == pytest.approx(1.0, abs=0.01)

    def test_biofilm_factor_far_from_optimal(self):
        sim = self._make_sim()
        f = sim.biofilm_factor(3.0)
        assert f > 1.0

    def test_reaction_rate_low_substrate(self):
        sim = self._make_sim()
        rate = sim.reaction_rate(0.001, 1.3)
        assert rate > 0

    def test_reaction_rate_far_from_optimal_biofilm(self):
        sim = self._make_sim()
        rate_optimal = sim.reaction_rate(20.0, 1.3)
        rate_far = sim.reaction_rate(20.0, 3.0)
        assert rate_optimal > rate_far

    def test_update_cell_parallel_low_flow(self):
        sim = self._make_sim()
        result = sim.update_cell_parallel(0, 20.0, 0.005, 1.3)
        assert result["outlet_concentration"] >= 0.001

    def test_update_biofilm_high_flow(self):
        sim = self._make_sim()
        sim.biofilm_thickness[0, :] = 1.3
        sim.acetate_concentrations[0, :] = 20.0
        sim.flow_rates[0] = 0.050
        sim.update_biofilm(1, sim.dt)
        for i in range(sim.num_cells):
            assert 0.5 <= sim.biofilm_thickness[1, i] <= 3.0

    def test_update_biofilm_below_threshold(self):
        sim = self._make_sim()
        sim.biofilm_thickness[0, :] = 0.5
        sim.acetate_concentrations[0, :] = 20.0
        sim.flow_rates[0] = 0.010
        sim.update_biofilm(1, sim.dt)

    def test_update_biofilm_above_optimal(self):
        sim = self._make_sim()
        sim.biofilm_thickness[0, :] = 2.5
        sim.acetate_concentrations[0, :] = 20.0
        sim.flow_rates[0] = 0.010
        sim.update_biofilm(1, sim.dt)

    def test_simulate_step_q_learning_update(self):
        """Test that Q-learning updates occur at step % 60 == 0."""
        sim = self._make_sim(steps=200)
        # Simulate enough steps to trigger Q-learning decisions
        for step in range(1, 130):
            sim.simulate_step(step)
        # After step 120, should have prev_state set
        assert hasattr(sim, "prev_state")
        assert hasattr(sim, "prev_action")

    def test_simulate_objective_value(self):
        sim = self._make_sim()
        for step in range(1, 10):
            sim.simulate_step(step)
        assert sim.objective_values[5] >= 0


class TestAddSubplotLabelsEdge:
    def test_single_subplot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        add_subplot_labels(fig, "a")
        plt.close(fig)

    def test_many_subplots(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 3)
        add_subplot_labels(fig, "a")
        plt.close(fig)
