"""Coverage tests for mfc_stack_simulation.py - targeting all paths."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from collections import deque

for _stale in ["mfc_stack_simulation", "mfc_100h_simulation"]:
    if _stale in sys.modules:
        del sys.modules[_stale]

mock_odes = MagicMock()
mock_model = MagicMock()
mock_model.mfc_odes = MagicMock(return_value=[0.001] * 9)
mock_odes.MFCModel = MagicMock(return_value=mock_model)
sys.modules["odes"] = mock_odes

mock_pc = MagicMock()
mock_pc.get_figure_path = MagicMock(return_value="/tmp/fig.png")
mock_pc.get_simulation_data_path = MagicMock(return_value="/tmp/data.json")
sys.modules["path_config"] = mock_pc

import matplotlib
matplotlib.use("Agg")

from mfc_stack_simulation import (
    MFCSensor, MFCActuator, MFCCell, MFCStack,
    MFCStackQLearningController, save_simulation_data,
    run_stack_simulation, plot_simulation_results,
)


@pytest.mark.coverage_extra
class TestMFCSensorCov2:
    def test_read_unknown_type(self):
        s = MFCSensor("unknown")
        val = s.read(5.0)
        assert isinstance(val, float)

    def test_filtered_reading_exactly_5(self):
        s = MFCSensor("voltage", noise_level=0.0)
        s.calibration_offset = 0.0
        for _ in range(5):
            s.read(1.0)
        val = s.get_filtered_reading()
        assert abs(val - 1.0) < 0.01

    def test_maxlen_deque(self):
        s = MFCSensor("voltage", noise_level=0.0)
        s.calibration_offset = 0.0
        for i in range(200):
            s.read(float(i))
        assert len(s.readings) == 100


@pytest.mark.coverage_extra
class TestMFCActuatorCov2:
    def test_response_dynamics(self):
        a = MFCActuator("duty_cycle", 0, 1)
        a.set_value(1.0)
        v1 = a.current_value
        a.set_value(1.0)
        v2 = a.current_value
        assert v2 >= v1

    def test_history_maxlen(self):
        a = MFCActuator("duty_cycle", 0, 1)
        for i in range(1100):
            a.set_value(0.5)
        assert len(a.history) == 1000


@pytest.mark.coverage_extra
class TestMFCCellCov2:
    def test_update_state_all_actuators_active(self):
        cell = MFCCell(0)
        cell.actuators["duty_cycle"].set_value(0.5)
        cell.actuators["ph_buffer"].set_value(0.5)
        cell.actuators["acetate_pump"].set_value(0.5)
        cell.update_state(1.0)

    def test_check_reversal_true(self):
        cell = MFCCell(0)
        cell.state[7] = 0.0
        cell.state[8] = 0.0
        result = cell.check_reversal()
        assert isinstance(result, (bool, np.bool_))

    def test_get_state_vector_reversed(self):
        cell = MFCCell(0)
        cell.is_reversed = True
        sv = cell.get_state_vector()
        assert sv[-1] == 1.0

    def test_get_state_vector_not_reversed(self):
        cell = MFCCell(0)
        cell.is_reversed = False
        sv = cell.get_state_vector()
        assert sv[-1] == 0.0

    def test_ph_calculation(self):
        cell = MFCCell(0)
        cell.state[2] = 1e-7
        readings = cell.get_sensor_readings()
        assert "pH" in readings


@pytest.mark.coverage_extra
class TestMFCStackCov2:
    def test_update_stack_multiple(self):
        stack = MFCStack()
        for _ in range(10):
            stack.update_stack()
        assert stack.time == 10.0
        assert len(stack.data_log["time"]) == 10

    def test_stack_power_calculation(self):
        stack = MFCStack()
        stack.update_stack()
        assert isinstance(stack.stack_power, float)

    def test_check_system_health_metrics(self):
        stack = MFCStack()
        stack.update_stack()
        h = stack.check_system_health()
        assert h["reversed_cells"] >= 0
        assert h["low_power_cells"] >= 0
        assert isinstance(h["stack_efficiency"], float)
        assert isinstance(h["power_stability"], float)


@pytest.mark.coverage_extra
class TestQLearningControllerCov2:
    def test_calculate_reward_low_duty(self):
        stack = MFCStack()
        ctrl = MFCStackQLearningController(stack)
        actions = np.full(15, 0.1)
        reward = ctrl.calculate_reward(np.zeros(40), actions)
        assert isinstance(reward, float)

    def test_calculate_reward_high_ph_buffer(self):
        stack = MFCStack()
        ctrl = MFCStackQLearningController(stack)
        actions = np.full(15, 0.5)
        for i in range(5):
            actions[i * 3 + 1] = 0.9
        reward = ctrl.calculate_reward(np.zeros(40), actions)
        assert isinstance(reward, float)

    def test_calculate_reward_high_acetate(self):
        stack = MFCStack()
        ctrl = MFCStackQLearningController(stack)
        actions = np.full(15, 0.5)
        for i in range(5):
            actions[i * 3 + 2] = 0.9
        reward = ctrl.calculate_reward(np.zeros(40), actions)
        assert isinstance(reward, float)

    def test_update_q_table_existing_entries(self):
        stack = MFCStack()
        ctrl = MFCStackQLearningController(stack)
        s = np.zeros(40)
        ns = np.ones(40) * 0.5
        key_s = ctrl.discretize_state(s)
        key_ns = ctrl.discretize_state(ns)
        ctrl.q_table[key_s] = np.full(15, 0.5)
        ctrl.q_table[key_ns] = np.full(15, 0.6)
        ctrl.update_q_table(s, np.zeros(15), 1.0, ns)

    def test_train_step_multiple(self):
        stack = MFCStack()
        ctrl = MFCStackQLearningController(stack)
        for _ in range(5):
            reward, power = ctrl.train_step()
        assert len(ctrl.reward_history) == 5
        assert len(ctrl.action_history) == 5


@pytest.mark.coverage_extra
class TestRunStackSimulation:
    def test_run_stack_simulation(self):
        mock_plt_mod = MagicMock()
        mock_axes = np.empty((3, 2), dtype=object)
        for i in range(3):
            for j in range(2):
                mock_axes[i, j] = MagicMock()
        mock_plt_mod.subplots.return_value = (MagicMock(), mock_axes)
        with patch("mfc_stack_simulation.time") as mock_time:
            mock_time.time.return_value = 0.0
            with patch("mfc_stack_simulation.plt", mock_plt_mod):
                with patch("mfc_stack_simulation.save_simulation_data"):
                    with patch("mfc_stack_simulation.plot_simulation_results"):
                        with patch.object(
                            MFCStackQLearningController, "train_step",
                            return_value=(0.5, 0.2)
                        ):
                            with patch.object(MFCStack, "check_system_health",
                                              return_value={"healthy": True}):
                                import mfc_stack_simulation as mod
                                orig_sim_time = 1000 * 3600
                                with patch.object(mod, "__name__", "__not_main__"):
                                    stack, ctrl = run_stack_simulation()
                                assert isinstance(stack, MFCStack)
