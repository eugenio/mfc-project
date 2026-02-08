import sys
import os
import importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from collections import deque

# Force-clean any stale mocks from other test files (e.g. test_mfc_stack_demo_cov.py)
for _stale in ["mfc_stack_simulation", "mfc_100h_simulation"]:
    if _stale in sys.modules:
        del sys.modules[_stale]

# Mock odes and path_config before importing
mock_odes = MagicMock()
mock_model_instance = MagicMock()
mock_model_instance.mfc_odes = MagicMock(return_value=[0.0]*9)
mock_odes.MFCModel = MagicMock(return_value=mock_model_instance)
sys.modules["odes"] = mock_odes

mock_pc = MagicMock()
mock_pc.get_figure_path = MagicMock(return_value="/tmp/test_fig.png")
mock_pc.get_simulation_data_path = MagicMock(return_value="/tmp/test_data.json")
sys.modules["path_config"] = mock_pc

import matplotlib
matplotlib.use("Agg")

from mfc_stack_simulation import MFCSensor, MFCActuator, MFCCell, MFCStack, MFCStackQLearningController
from mfc_100h_simulation import LongTermMFCStack, LongTermController


class TestLongTermMFCStack:
    def setup_method(self):
        self.stack = LongTermMFCStack()

    def test_init(self):
        assert self.stack.substrate_tank_level == 100.0
        assert self.stack.ph_buffer_tank_level == 100.0
        assert self.stack.maintenance_cycles == 0
        assert self.stack.total_energy_produced == 0.0
        assert len(self.stack.cell_aging_factors) == 5
        assert len(self.stack.biofilm_thickness) == 5

    def test_apply_long_term_effects(self):
        initial_aging = self.stack.cell_aging_factors[0]
        self.stack.stack_power = 1.0
        self.stack.apply_long_term_effects(1.0)
        assert self.stack.cell_aging_factors[0] < initial_aging
        assert self.stack.total_energy_produced > 0

    def test_apply_long_term_effects_biofilm(self):
        for _ in range(100):
            self.stack.apply_long_term_effects(1.0)
        for t in self.stack.biofilm_thickness:
            assert t <= 2.0

    def test_apply_long_term_effects_aging_floor(self):
        for _ in range(1000):
            self.stack.apply_long_term_effects(1.0)
        for a in self.stack.cell_aging_factors:
            assert a >= 0.5

    def test_apply_long_term_effects_substrate_depletion(self):
        self.stack.stack_power = 100.0
        self.stack.apply_long_term_effects(10.0)
        assert self.stack.substrate_tank_level < 100.0

    def test_apply_long_term_effects_ph_buffer(self):
        for cell in self.stack.cells:
            cell.actuators["ph_buffer"].current_value = 0.5
        self.stack.apply_long_term_effects(10.0)
        assert self.stack.ph_buffer_tank_level < 100.0

    def test_apply_long_term_effects_mass_transfer(self):
        for i in range(5):
            self.stack.biofilm_thickness[i] = 1.5
        self.stack.apply_long_term_effects(0.1)

    def test_check_maintenance_substrate(self):
        self.stack.substrate_tank_level = 15.0
        result = self.stack.check_maintenance_needs()
        assert result is True
        assert self.stack.substrate_tank_level == 100.0
        assert self.stack.maintenance_cycles == 1

    def test_check_maintenance_ph_buffer(self):
        self.stack.ph_buffer_tank_level = 10.0
        result = self.stack.check_maintenance_needs()
        assert result is True
        assert self.stack.ph_buffer_tank_level == 100.0

    def test_check_maintenance_biofilm_cleaning(self):
        self.stack.time = 24 * 3600  # Set to exact day boundary
        for i in range(5):
            self.stack.biofilm_thickness[i] = 1.8
        result = self.stack.check_maintenance_needs()
        assert result is True
        for t in self.stack.biofilm_thickness:
            assert t <= 1.5

    def test_check_maintenance_no_need(self):
        self.stack.substrate_tank_level = 80.0
        self.stack.ph_buffer_tank_level = 80.0
        self.stack.time = 100  # Not a day boundary
        result = self.stack.check_maintenance_needs()
        assert result is False

    def test_log_hourly_data(self):
        self.stack.time = 3600  # Exactly 1 hour
        self.stack.stack_power = 1.5
        self.stack.stack_voltage = 2.0
        self.stack.log_hourly_data()
        assert len(self.stack.hourly_data["hour"]) == 1
        assert self.stack.hourly_data["stack_power"][0] == 1.5

    def test_log_hourly_data_not_on_hour(self):
        self.stack.time = 1800  # 30 min
        self.stack.log_hourly_data()
        assert len(self.stack.hourly_data["hour"]) == 0


class TestLongTermController:
    def setup_method(self):
        self.stack = LongTermMFCStack()
        self.ctrl = LongTermController(self.stack)

    def test_init(self):
        assert self.ctrl.current_phase == "exploration"
        assert self.ctrl.epsilon == 0.3

    def test_update_learning_phase_exploration_to_optimization(self):
        self.ctrl.stack.time = 25 * 3600
        self.ctrl.phase_start_time = 0
        self.ctrl.update_learning_phase()
        assert self.ctrl.current_phase == "optimization"

    def test_update_learning_phase_optimization_to_maintenance(self):
        self.ctrl.current_phase = "optimization"
        self.ctrl.phase_start_time = 0
        self.ctrl.stack.time = 49 * 3600
        self.ctrl.update_learning_phase()
        assert self.ctrl.current_phase == "maintenance"

    def test_update_learning_phase_no_change(self):
        self.ctrl.stack.time = 10 * 3600
        self.ctrl.phase_start_time = 0
        self.ctrl.update_learning_phase()
        assert self.ctrl.current_phase == "exploration"

    def test_calculate_long_term_reward_bonuses(self):
        self.stack.substrate_tank_level = 60.0
        self.stack.ph_buffer_tank_level = 60.0
        self.stack.cell_aging_factors = [0.9] * 5
        self.stack.total_energy_produced = 15.0
        state = self.stack.get_stack_state()
        actions = np.random.uniform(0.3, 0.7, 15)
        reward = self.ctrl.calculate_long_term_reward(state, actions)
        assert isinstance(reward, float)

    def test_calculate_long_term_reward_no_bonuses(self):
        self.stack.substrate_tank_level = 30.0
        self.stack.ph_buffer_tank_level = 30.0
        self.stack.cell_aging_factors = [0.6] * 5
        self.stack.total_energy_produced = 5.0
        state = self.stack.get_stack_state()
        actions = np.random.uniform(0.3, 0.7, 15)
        reward = self.ctrl.calculate_long_term_reward(state, actions)
        assert isinstance(reward, float)

    def test_train_step(self):
        reward, power, maint = self.ctrl.train_step()
        assert isinstance(reward, float)
        assert isinstance(power, float)
        assert isinstance(maint, bool)

    def test_train_step_maintenance_phase(self):
        self.ctrl.current_phase = "maintenance"
        reward, power, maint = self.ctrl.train_step()
        assert isinstance(reward, float)

    def test_train_step_performance_window(self):
        for _ in range(10):
            self.ctrl.train_step()
        assert len(self.ctrl.performance_windows["hourly"]) > 0

    def test_train_step_window_trim(self):
        self.ctrl.performance_windows["hourly"] = list(range(3601))
        self.ctrl.train_step()
        assert len(self.ctrl.performance_windows["hourly"]) <= 3601
