"""Coverage tests for mfc_100h_simulation module."""
import os
import sys
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pytest

# Mock the entire dependency chain before importing
_mock_odes = MagicMock()
_mock_mfc_stack = MagicMock()

# Create mock MFCStack class
class MockMFCStack:
    def __init__(self):
        self.time = 0
        self.stack_power = 0.5
        self.stack_voltage = 3.0
        self.cells = [MagicMock() for _ in range(5)]
        for c in self.cells:
            c.state = np.array([1.0, 0.05, 1e-4, 0.1, 0.25, 1e-7, 0.05, 0.01, -0.01, 1.0, 1.0])
            c.actuators = {"ph_buffer": MagicMock(get_value=MagicMock(return_value=0.5))}
            c.get_sensor_readings = MagicMock(return_value={"voltage": 0.6, "pH": 7.0, "acetate": 1.0})
            c.get_power = MagicMock(return_value=0.1)
            c.is_reversed = False

    def update_stack(self):
        self.time += 1

    def apply_control_actions(self, actions):
        pass

    def get_stack_state(self):
        return np.zeros(55)

    def check_system_health(self):
        pass


class MockQLearningController:
    def __init__(self, stack):
        self.stack = stack
        self.epsilon = 0.5
        self.q_table = {}

    def calculate_reward(self, state, actions):
        return 1.0

    def get_action(self, state):
        return np.random.uniform(0, 1, 15)

    def update_q_table(self, state, actions, reward, next_state):
        pass


_mock_mfc_stack.MFCStack = MockMFCStack
_mock_mfc_stack.MFCStackQLearningController = MockQLearningController

sys.modules["odes"] = _mock_odes
sys.modules["mfc_stack_simulation"] = _mock_mfc_stack

from mfc_100h_simulation import (
    LongTermController,
    LongTermMFCStack,
)


@pytest.mark.coverage_extra
class TestLongTermMFCStack:
    def test_init(self):
        stack = LongTermMFCStack()
        assert stack.substrate_tank_level == 100.0
        assert stack.ph_buffer_tank_level == 100.0
        assert stack.maintenance_cycles == 0
        assert stack.total_energy_produced == 0.0
        assert len(stack.cell_aging_factors) == 5
        assert len(stack.biofilm_thickness) == 5

    def test_apply_long_term_effects(self):
        stack = LongTermMFCStack()
        stack.stack_power = 0.5
        stack.time = 3600
        initial_aging = stack.cell_aging_factors[0]
        stack.apply_long_term_effects(1.0)
        assert stack.cell_aging_factors[0] < initial_aging
        assert stack.total_energy_produced > 0

    def test_apply_aging_clamp(self):
        stack = LongTermMFCStack()
        stack.stack_power = 0.5
        for _ in range(600):
            stack.apply_long_term_effects(1.0)
        assert all(f >= 0.5 for f in stack.cell_aging_factors)
        assert all(t <= 2.0 for t in stack.biofilm_thickness)

    def test_biofilm_mass_transfer(self):
        stack = LongTermMFCStack()
        stack.stack_power = 0.1
        # Force biofilm > 1.2
        for i in range(5):
            stack.biofilm_thickness[i] = 1.5
        stack.apply_long_term_effects(0.01)

    def test_check_maintenance_substrate_low(self):
        stack = LongTermMFCStack()
        stack.substrate_tank_level = 10.0
        result = stack.check_maintenance_needs()
        assert result is True
        assert stack.substrate_tank_level == 100.0
        assert stack.maintenance_cycles == 1

    def test_check_maintenance_ph_low(self):
        stack = LongTermMFCStack()
        stack.ph_buffer_tank_level = 10.0
        result = stack.check_maintenance_needs()
        assert result is True
        assert stack.ph_buffer_tank_level == 100.0

    def test_check_maintenance_no_need(self):
        stack = LongTermMFCStack()
        stack.time = 1000  # Not on 24h boundary
        result = stack.check_maintenance_needs()
        assert result is False

    def test_check_maintenance_biofilm_cleaning(self):
        stack = LongTermMFCStack()
        stack.time = 5  # Within 10s of 24h boundary
        for i in range(5):
            stack.biofilm_thickness[i] = 1.8
        result = stack.check_maintenance_needs()
        assert result is True
        assert any(t == 1.0 for t in stack.biofilm_thickness)

    def test_log_hourly_data(self):
        stack = LongTermMFCStack()
        stack.time = 5  # Within 10s of hour boundary
        stack.log_hourly_data()
        assert len(stack.hourly_data["hour"]) == 1

    def test_log_hourly_data_not_on_boundary(self):
        stack = LongTermMFCStack()
        stack.time = 500  # Not within 10s of hour boundary
        stack.log_hourly_data()
        assert len(stack.hourly_data["hour"]) == 0

    def test_hourly_data_structure(self):
        stack = LongTermMFCStack()
        assert "hour" in stack.hourly_data
        assert "stack_power" in stack.hourly_data
        assert "system_efficiency" in stack.hourly_data


@pytest.mark.coverage_extra
class TestLongTermController:
    def test_init(self):
        stack = LongTermMFCStack()
        ctrl = LongTermController(stack)
        assert ctrl.current_phase == "exploration"
        assert ctrl.substrate_management_learned is False

    def test_update_learning_phase_to_optimization(self):
        stack = LongTermMFCStack()
        ctrl = LongTermController(stack)
        stack.time = 25 * 3600  # Past 24h exploration
        ctrl.update_learning_phase()
        assert ctrl.current_phase == "optimization"

    def test_update_learning_phase_to_maintenance(self):
        stack = LongTermMFCStack()
        ctrl = LongTermController(stack)
        ctrl.current_phase = "optimization"
        ctrl.phase_start_time = 0
        stack.time = 49 * 3600  # Past 48h optimization
        ctrl.update_learning_phase()
        assert ctrl.current_phase == "maintenance"

    def test_calculate_long_term_reward(self):
        stack = LongTermMFCStack()
        ctrl = LongTermController(stack)
        stack.substrate_tank_level = 80
        stack.ph_buffer_tank_level = 80
        stack.cell_aging_factors = [0.9] * 5
        stack.total_energy_produced = 20.0
        state = np.zeros(55)
        actions = np.zeros(15)
        # Patch base class calculate_reward to isolate long-term bonus logic
        with patch.object(
            type(ctrl).__mro__[1], "calculate_reward", return_value=1.0
        ):
            reward = ctrl.calculate_long_term_reward(state, actions)
        assert reward > 0  # base + sustainability bonuses

    def test_calculate_reward_low_resources(self):
        stack = LongTermMFCStack()
        ctrl = LongTermController(stack)
        stack.substrate_tank_level = 30
        stack.ph_buffer_tank_level = 30
        stack.cell_aging_factors = [0.5] * 5
        stack.total_energy_produced = 5.0
        state = np.zeros(55)
        actions = np.zeros(15)
        reward = ctrl.calculate_long_term_reward(state, actions)
        assert isinstance(reward, float)

    def test_train_step(self):
        stack = LongTermMFCStack()
        ctrl = LongTermController(stack)
        reward, power, maintenance = ctrl.train_step()
        assert isinstance(reward, float)
        assert isinstance(power, float)

    def test_train_step_maintenance_phase(self):
        stack = LongTermMFCStack()
        ctrl = LongTermController(stack)
        ctrl.current_phase = "maintenance"
        reward, power, maintenance = ctrl.train_step()
        assert isinstance(reward, float)

    def test_performance_windows_pruning(self):
        stack = LongTermMFCStack()
        ctrl = LongTermController(stack)
        ctrl.performance_windows["hourly"] = [0.1] * 3601
        ctrl.train_step()
        assert len(ctrl.performance_windows["hourly"]) <= 3601
