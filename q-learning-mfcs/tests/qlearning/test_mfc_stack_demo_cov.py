"""Coverage tests for mfc_stack_demo.py (98%+ target)."""
import os
import sys
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import matplotlib
matplotlib.use("Agg")

# Mock the mfc_stack_simulation module
_mock_cell = MagicMock()
_mock_cell.state = [1.0, 0.05, 1e-4, 0.1, 0.25, 1e-7, 0.05, 0.01, -0.01]
_mock_cell.is_reversed = False
_mock_cell.get_sensor_readings.return_value = {"voltage": 0.6, "current": 1.0}
_mock_cell.get_power.return_value = 0.6
_mock_cell.actuators = {
    "duty_cycle": MagicMock(get_value=MagicMock(return_value=0.8)),
    "ph_buffer": MagicMock(get_value=MagicMock(return_value=0.1)),
    "acetate_pump": MagicMock(get_value=MagicMock(return_value=0.05)),
}


def _make_mock_stack():
    """Create a fresh mock stack with proper data_log structure."""
    stack = MagicMock()
    n_steps = 1200  # 5 phases: 200+300+200+200+300

    stack.cells = [MagicMock() for _ in range(5)]
    for cell in stack.cells:
        cell.state = [1.0, 0.05, 1e-4, 0.1, 0.25, 1e-7, 0.05, 0.01, -0.01]
        cell.is_reversed = False
        cell.get_sensor_readings.return_value = {"voltage": 0.6}
        cell.get_power.return_value = 0.6
        cell.actuators = {
            "duty_cycle": MagicMock(get_value=MagicMock(return_value=0.8)),
            "ph_buffer": MagicMock(get_value=MagicMock(return_value=0.1)),
            "acetate_pump": MagicMock(get_value=MagicMock(return_value=0.05)),
        }

    stack.stack_voltage = 3.0
    stack.check_system_health.return_value = {"status": "ok"}

    # Build data_log with proper lists
    time_list = list(range(n_steps))
    stack.data_log = {
        "time": time_list,
        "stack_power": [0.5 + 0.001 * i for i in range(n_steps)],
        "stack_voltage": [3.0 + 0.001 * i for i in range(n_steps)],
        "cell_voltages": {i: [0.6 + 0.0001 * j for j in range(n_steps)] for i in range(5)},
        "cell_reversals": {i: [0] * n_steps for i in range(5)},
        "duty_cycles": {i: [0.8] * n_steps for i in range(5)},
        "ph_buffers": {i: [0.1] * n_steps for i in range(5)},
        "acetate_additions": {i: [0.0] * n_steps for i in range(5)},
    }

    return stack


def _make_mock_controller():
    """Create a fresh mock controller."""
    ctrl = MagicMock()
    ctrl.train_step.return_value = (1.0, 0.5)
    ctrl.reward_history = list(range(100))
    ctrl.enable_acetate_addition = False
    return ctrl


_mock_stack_simulation = MagicMock()
_mock_stack_simulation.MFCStack = MagicMock(side_effect=lambda: _make_mock_stack())
_mock_stack_simulation.MFCStackQLearningController = MagicMock(
    side_effect=lambda stack: _make_mock_controller()
)
sys.modules["mfc_stack_simulation"] = _mock_stack_simulation

from mfc_stack_demo import (
    plot_comprehensive_results,
    run_comprehensive_demo,
)


class TestRunComprehensiveDemo:
    @patch("matplotlib.pyplot.savefig")
    def test_demo_runs(self, mock_savefig):
        stack, controller, phase_results = run_comprehensive_demo()
        assert len(phase_results) == 5
        assert phase_results[0]["name"] == "Initialization"
        assert phase_results[1]["name"] == "Normal Operation"
        assert phase_results[2]["name"] == "Disturbance Recovery"
        assert phase_results[3]["name"] == "pH Management"
        assert phase_results[4]["name"] == "Long-term Operation"
        import matplotlib.pyplot as plt
        plt.close("all")

    @patch("matplotlib.pyplot.savefig")
    def test_demo_phase_data_structure(self, mock_savefig):
        _, _, phase_results = run_comprehensive_demo()
        for phase in phase_results:
            assert "name" in phase
            assert "start_step" in phase
            assert "end_step" in phase
            assert "powers" in phase
            assert "voltages" in phase
            assert "reversals" in phase
            assert "rewards" in phase
            assert len(phase["powers"]) > 0
        import matplotlib.pyplot as plt
        plt.close("all")


class TestPlotComprehensiveResults:
    @patch("matplotlib.pyplot.savefig")
    def test_plot_results(self, mock_savefig):
        stack = _make_mock_stack()
        ctrl = _make_mock_controller()
        phase_results = [
            {"name": "Phase1", "start_step": 0, "end_step": 200, "powers": [0.5]*200,
             "voltages": [3.0]*200, "reversals": [0]*200, "rewards": [1.0]*200},
            {"name": "Phase2", "start_step": 200, "end_step": 500, "powers": [0.6]*300,
             "voltages": [3.1]*300, "reversals": [0]*300, "rewards": [1.1]*300},
        ]
        plot_comprehensive_results(stack, ctrl, phase_results)
        mock_savefig.assert_called_once()
        import matplotlib.pyplot as plt
        plt.close("all")

    @patch("matplotlib.pyplot.savefig")
    def test_plot_results_short_rewards(self, mock_savefig):
        stack = _make_mock_stack()
        ctrl = _make_mock_controller()
        ctrl.reward_history = [1.0, 2.0, 3.0]  # Less than window_size=20
        phase_results = [
            {"name": "Phase1", "start_step": 0, "end_step": 200, "powers": [0.5]*200,
             "voltages": [3.0]*200, "reversals": [0]*200, "rewards": [1.0]*200},
        ]
        plot_comprehensive_results(stack, ctrl, phase_results)
        import matplotlib.pyplot as plt
        plt.close("all")

    @patch("matplotlib.pyplot.savefig")
    def test_plot_results_with_reversals(self, mock_savefig):
        stack = _make_mock_stack()
        # Add some reversals
        for i in range(5):
            for j in range(100, 200):
                stack.data_log["cell_reversals"][i][j] = 1
        ctrl = _make_mock_controller()
        phase_results = [
            {"name": "Phase1", "start_step": 0, "end_step": 500, "powers": [0.5]*500,
             "voltages": [3.0]*500, "reversals": [0]*500, "rewards": [1.0]*500},
        ]
        plot_comprehensive_results(stack, ctrl, phase_results)
        import matplotlib.pyplot as plt
        plt.close("all")
