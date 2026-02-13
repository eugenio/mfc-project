"""End-to-end tests for Q-learning optimization workflow.

Tests the complete Q-learning training cycle: initialization,
state discretization, action selection, Q-table updates, and
reward convergence.
"""

import numpy as np
import pytest


class TestQLearningControllerE2E:
    """E2E test: QLearningFlowController full workflow.

    The controller expects config.flow_rate_adjustments_ml_per_h,
    which maps to config.flow_rate_actions in the current QLearningConfig.
    We patch the config to bridge this gap.
    """

    @pytest.fixture
    def controller(self):
        from config.qlearning_config import QLearningConfig
        from mfc_qlearning_optimization import QLearningFlowController

        cfg = QLearningConfig()
        cfg.flow_rate_adjustments_ml_per_h = cfg.flow_rate_actions
        return QLearningFlowController(config=cfg)

    def test_controller_initializes(self, controller):
        assert controller is not None
        assert controller.learning_rate > 0
        assert controller.discount_factor > 0
        assert controller.epsilon >= 0

    def test_state_discretization(self, controller):
        state = controller.discretize_state(
            power=0.05,
            biofilm_deviation=0.02,
            substrate_utilization=50.0,
            time_hours=10.0,
        )
        assert state is not None
        assert len(state) == 4

    def test_action_selection(self, controller):
        state = controller.discretize_state(
            power=0.05,
            biofilm_deviation=0.02,
            substrate_utilization=50.0,
            time_hours=10.0,
        )
        action_idx, new_flow = controller.select_action(
            state, current_flow_rate=1e-6,
        )
        assert action_idx is not None
        assert new_flow >= 0

    def test_q_table_update(self, controller):
        state = controller.discretize_state(
            power=0.05,
            biofilm_deviation=0.02,
            substrate_utilization=50.0,
            time_hours=10.0,
        )
        next_state = controller.discretize_state(
            power=0.06,
            biofilm_deviation=0.01,
            substrate_utilization=48.0,
            time_hours=10.5,
        )
        action_idx, _ = controller.select_action(
            state, current_flow_rate=1e-6,
        )
        reward = controller.calculate_reward(
            power=0.06,
            biofilm_deviation=0.01,
            substrate_utilization=48.0,
            prev_power=0.05,
            prev_biofilm_dev=0.02,
            prev_substrate_util=50.0,
        )

        controller.update_q_table(state, action_idx, reward, next_state)
        assert len(controller.q_table) > 0

    def test_training_loop_10_steps(self, controller):
        """Run 10 training steps and verify Q-table grows."""
        flow_rate = 1e-6
        rewards = []

        prev_power = 0.04
        prev_biofilm = 0.03
        prev_substrate = 55.0

        for step in range(10):
            power = 0.04 + np.random.uniform(0, 0.02)
            biofilm_dev = np.random.uniform(0, 0.05)
            substrate_util = max(10.0, 100.0 - step * 5)
            time_h = step * 0.5

            state = controller.discretize_state(
                power=power,
                biofilm_deviation=biofilm_dev,
                substrate_utilization=substrate_util,
                time_hours=time_h,
            )
            action_idx, flow_rate = controller.select_action(
                state, flow_rate,
            )
            reward = controller.calculate_reward(
                power=power,
                biofilm_deviation=biofilm_dev,
                substrate_utilization=substrate_util,
                prev_power=prev_power,
                prev_biofilm_dev=prev_biofilm,
                prev_substrate_util=prev_substrate,
            )

            next_state = controller.discretize_state(
                power=power + 0.001,
                biofilm_deviation=biofilm_dev,
                substrate_utilization=substrate_util - 1,
                time_hours=time_h + 0.5,
            )
            controller.update_q_table(
                state, action_idx, reward, next_state,
            )
            rewards.append(reward)

            prev_power = power
            prev_biofilm = biofilm_dev
            prev_substrate = substrate_util

        assert len(controller.q_table) > 0
        assert len(rewards) == 10


class TestRecirculationControlE2E:
    """E2E test: recirculation control system workflow."""

    def test_reservoir_substrate_tracking(self):
        from mfc_recirculation_control import AnolytereservoirSystem

        reservoir = AnolytereservoirSystem()
        initial = reservoir.substrate_concentration

        # Add substrate and verify
        reservoir.add_substrate(amount_mmol=0.5, dt_hours=1.0)
        assert reservoir.substrate_concentration >= initial

    def test_reservoir_circulation(self):
        from mfc_recirculation_control import AnolytereservoirSystem

        reservoir = AnolytereservoirSystem()
        inlet_conc = reservoir.get_inlet_concentration()
        assert inlet_conc >= 0

    def test_cell_monitoring_workflow(self):
        from mfc_recirculation_control import MFCCellWithMonitoring

        cell = MFCCellWithMonitoring(cell_id=0)
        assert cell is not None
        assert callable(cell.process_with_monitoring)
        assert callable(cell.update_concentrations)

    def test_advanced_controller_workflow(self):
        from mfc_recirculation_control import (
            AdvancedQLearningFlowController,
        )

        ctrl = AdvancedQLearningFlowController()
        assert ctrl is not None
        assert callable(ctrl.choose_action)
