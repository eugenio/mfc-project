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

# Mock odes and path_config
mock_odes = MagicMock()
mock_model = MagicMock()
mock_model.mfc_odes = MagicMock(return_value=[0.001]*9)
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


class TestMFCSensor:
    def test_init(self):
        s = MFCSensor("voltage", noise_level=0.01)
        assert s.sensor_type == "voltage"
        assert s.noise_level == 0.01

    def test_read_voltage(self):
        s = MFCSensor("voltage")
        val = s.read(0.5)
        assert 0 <= val <= 2.0

    def test_read_current(self):
        s = MFCSensor("current")
        val = s.read(0.1)
        assert val >= 0

    def test_read_ph(self):
        s = MFCSensor("pH")
        val = s.read(7.0)
        assert 0 <= val <= 14

    def test_read_acetate(self):
        s = MFCSensor("acetate")
        val = s.read(1.0)
        assert val >= 0

    def test_get_filtered_empty(self):
        s = MFCSensor("voltage")
        assert s.get_filtered_reading() == 0

    def test_get_filtered_few(self):
        s = MFCSensor("voltage", noise_level=0.0)
        s.calibration_offset = 0.0
        for _ in range(3):
            s.read(1.0)
        val = s.get_filtered_reading()
        assert abs(val - 1.0) < 0.1

    def test_get_filtered_many(self):
        s = MFCSensor("voltage", noise_level=0.0)
        s.calibration_offset = 0.0
        for _ in range(10):
            s.read(1.0)
        val = s.get_filtered_reading()
        assert abs(val - 1.0) < 0.1


class TestMFCActuator:
    def test_init(self):
        a = MFCActuator("duty_cycle", 0, 1)
        assert a.actuator_type == "duty_cycle"
        assert a.current_value == 0

    def test_set_value(self):
        a = MFCActuator("duty_cycle", 0, 1)
        val = a.set_value(0.5)
        assert 0 <= val <= 1

    def test_set_value_clamp_high(self):
        a = MFCActuator("duty_cycle", 0, 1)
        a.set_value(5.0)
        assert a.current_value <= 1.0

    def test_set_value_clamp_low(self):
        a = MFCActuator("duty_cycle", 0, 1)
        a.set_value(-1.0)
        assert a.current_value >= 0.0

    def test_get_value(self):
        a = MFCActuator("duty_cycle", 0, 1)
        a.set_value(0.5)
        assert a.get_value() >= 0


class TestMFCCell:
    def setup_method(self):
        self.cell = MFCCell(0)

    def test_init(self):
        assert self.cell.cell_id == 0
        assert len(self.cell.state) == 9
        assert not self.cell.is_reversed

    def test_update_state(self):
        self.cell.actuators["duty_cycle"].set_value(0.5)
        self.cell.update_state(1.0)

    def test_update_state_ph_buffer(self):
        self.cell.actuators["ph_buffer"].set_value(0.5)
        self.cell.update_state(1.0)

    def test_update_state_acetate(self):
        self.cell.actuators["acetate_pump"].set_value(0.5)
        self.cell.update_state(1.0)

    def test_update_state_exception(self):
        mock_model.mfc_odes = MagicMock(side_effect=Exception("fail"))
        self.cell.update_state(1.0)
        mock_model.mfc_odes = MagicMock(return_value=[0.001]*9)

    def test_get_sensor_readings(self):
        r = self.cell.get_sensor_readings()
        assert "voltage" in r
        assert "current" in r
        assert "pH" in r
        assert "acetate" in r

    def test_get_power(self):
        p = self.cell.get_power()
        assert isinstance(p, float)

    def test_check_reversal(self):
        result = self.cell.check_reversal()
        assert bool(result) == result

    def test_get_state_vector(self):
        sv = self.cell.get_state_vector()
        assert len(sv) == 7


class TestMFCStack:
    def setup_method(self):
        self.stack = MFCStack()

    def test_init(self):
        assert len(self.stack.cells) == 5
        assert self.stack.time == 0

    def test_update_stack(self):
        self.stack.update_stack()
        assert self.stack.time == 1.0
        assert len(self.stack.data_log["time"]) == 1

    def test_log_data(self):
        self.stack.log_data()
        assert len(self.stack.data_log["time"]) == 1

    def test_get_stack_state(self):
        state = self.stack.get_stack_state()
        assert len(state) == 40  # 7*5 + 5

    def test_apply_control_actions(self):
        actions = np.random.uniform(0, 1, 15)
        self.stack.apply_control_actions(actions)

    def test_check_system_health(self):
        h = self.stack.check_system_health()
        assert "reversed_cells" in h
        assert "low_power_cells" in h
        assert "stack_efficiency" in h
        assert "power_stability" in h


class TestMFCStackQLearningController:
    def setup_method(self):
        self.stack = MFCStack()
        self.ctrl = MFCStackQLearningController(self.stack)

    def test_init(self):
        assert self.ctrl.epsilon == 0.3
        assert len(self.ctrl.q_table) == 0

    def test_discretize_state(self):
        state = np.random.uniform(0, 1, 40)
        key = self.ctrl.discretize_state(state)
        assert len(key) == 10

    def test_get_action_explore(self):
        self.ctrl.epsilon = 1.0
        state = np.random.uniform(0, 1, 40)
        actions = self.ctrl.get_action(state)
        assert len(actions) == 15

    def test_get_action_exploit_known(self):
        self.ctrl.epsilon = 0.0
        state = np.zeros(40)
        key = self.ctrl.discretize_state(state)
        self.ctrl.q_table[key] = np.full(15, 0.5)
        actions = self.ctrl.get_action(state)
        assert len(actions) == 15

    def test_get_action_exploit_unknown(self):
        self.ctrl.epsilon = 0.0
        state = np.random.uniform(0, 1, 40)
        actions = self.ctrl.get_action(state)
        assert len(actions) == 15

    def test_get_action_constraints(self):
        actions = self.ctrl.get_action(np.random.uniform(0, 1, 40))
        for i in range(5):
            assert actions[i*3] >= 0.1
            assert actions[i*3] <= 0.9

    def test_calculate_reward(self):
        state = np.random.uniform(0, 1, 40)
        actions = np.random.uniform(0.3, 0.7, 15)
        reward = self.ctrl.calculate_reward(state, actions)
        assert isinstance(reward, float)

    def test_calculate_reward_extreme_actions(self):
        state = np.random.uniform(0, 1, 40)
        actions = np.full(15, 0.9)
        reward = self.ctrl.calculate_reward(state, actions)
        assert isinstance(reward, float)

    def test_update_q_table(self):
        s = np.random.uniform(0, 1, 40)
        ns = np.random.uniform(0, 1, 40)
        a = np.random.uniform(0, 1, 15)
        self.ctrl.update_q_table(s, a, 1.0, ns)
        assert len(self.ctrl.q_table) > 0

    def test_train_step(self):
        reward, power = self.ctrl.train_step()
        assert isinstance(reward, float)
        assert isinstance(power, float)
        assert len(self.ctrl.reward_history) == 1

    def test_train_step_epsilon_decay(self):
        initial = self.ctrl.epsilon
        self.ctrl.train_step()
        assert self.ctrl.epsilon <= initial

    def test_train_step_epsilon_min(self):
        self.ctrl.epsilon = self.ctrl.epsilon_min
        self.ctrl.train_step()
        assert self.ctrl.epsilon >= self.ctrl.epsilon_min


class TestSaveSimulationData:
    def test_save(self, tmp_path):
        stack = MFCStack()
        stack.update_stack()
        ctrl = MFCStackQLearningController(stack)
        mock_pc.get_simulation_data_path.return_value = str(tmp_path / "data.json")
        data = save_simulation_data(stack, ctrl)
        assert "metadata" in data
        assert "time_series" in data
        assert "final_states" in data

class TestPlotSimulationResults:
    def _make_axes(self):
        mock_axes = np.empty((3, 2), dtype=object)
        for i in range(3):
            for j in range(2):
                mock_axes[i, j] = MagicMock()
        return MagicMock(), mock_axes

    def test_plot_long_rewards(self):
        stack = MFCStack()
        for _ in range(5):
            stack.update_stack()
        stack.data_log["cell_powers"] = {i: [0.1] * 5 for i in range(5)}
        ctrl = MFCStackQLearningController(stack)
        ctrl.reward_history = list(range(100))
        with patch("mfc_stack_simulation.plt.subplots", return_value=self._make_axes()):
            with patch("mfc_stack_simulation.plt.tight_layout"):
                with patch("mfc_stack_simulation.plt.savefig"):
                    plot_simulation_results(stack, ctrl)

    def test_plot_short_rewards(self):
        stack = MFCStack()
        stack.update_stack()
        stack.data_log["cell_powers"] = {i: [0.1] for i in range(5)}
        ctrl = MFCStackQLearningController(stack)
        ctrl.reward_history = [1.0, 2.0]
        with patch("mfc_stack_simulation.plt.subplots", return_value=self._make_axes()):
            with patch("mfc_stack_simulation.plt.tight_layout"):
                with patch("mfc_stack_simulation.plt.savefig"):
                    plot_simulation_results(stack, ctrl)


