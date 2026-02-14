"""Extended coverage tests for run_gpu_simulation module.

Targets the run_accelerated_python_simulation function (lines 117-383).
"""

import os
import sys
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

_mock_pc = MagicMock()
_mock_pc.get_figure_path = MagicMock(return_value="/tmp/fig.png")
_mock_pc.get_simulation_data_path = MagicMock(return_value="/tmp/d.json")
sys.modules.setdefault("path_config", _mock_pc)

import run_gpu_simulation as rgs


@pytest.fixture(autouse=True)
def mock_paths():
    with patch("run_gpu_simulation.get_figure_path", return_value="/tmp/f.png"), \
         patch("run_gpu_simulation.get_simulation_data_path", return_value="/tmp/d.json"):
        yield


@pytest.mark.coverage_extra
class TestRunAcceleratedPythonSimulation:
    """Test run_accelerated_python_simulation covering lines 117-383."""

    def test_full_run(self):
        """Run the actual function with mocked I/O."""
        with patch("run_gpu_simulation.save_performance_data") as ms, \
             patch("run_gpu_simulation.generate_plots") as mg:
            result = rgs.run_accelerated_python_simulation()
            assert result is not None
            assert len(result) > 0
            ms.assert_called_once()
            mg.assert_called_once()


@pytest.mark.coverage_extra
class TestMainFunction:
    """Test main() function (lines 670-697)."""

    def test_above_target(self):
        best = {"name": "B", "energy_output": 100.0,
                "avg_power": 1.0, "max_power": 2.0,
                "runtime": 10.0, "success": True}
        with patch("run_gpu_simulation.run_all_mojo_simulations",
                    return_value=[best]), \
             patch("run_gpu_simulation.analyze_and_compare_results",
                    return_value=[best]):
            rgs.main()

    def test_below_target(self):
        best = {"name": "B", "energy_output": 50.0,
                "avg_power": 0.5, "max_power": 1.0,
                "runtime": 10.0, "success": True}
        with patch("run_gpu_simulation.run_all_mojo_simulations",
                    return_value=[best]), \
             patch("run_gpu_simulation.analyze_and_compare_results",
                    return_value=[best]):
            rgs.main()

    def test_no_results_fallback(self):
        with patch("run_gpu_simulation.run_all_mojo_simulations",
                    return_value=[]), \
             patch("run_gpu_simulation.analyze_and_compare_results",
                    return_value=None), \
             patch("run_gpu_simulation.run_accelerated_python_simulation") as mp:
            rgs.main()
            mp.assert_called_once()


@pytest.mark.coverage_extra
class TestAnalyzeEdgeCases:
    """Edge cases for analyze_and_compare_results."""

    def test_all_technologies(self):
        results = [
            {"success": True, "name": "Simple MFC",
             "runtime": 10.0, "energy_output": 50.0,
             "avg_power": 0.5, "max_power": 1.0},
            {"success": True, "name": "Q-Learning MFC",
             "runtime": 12.0, "energy_output": 60.0,
             "avg_power": 0.6, "max_power": 1.1},
            {"success": True, "name": "Enhanced Q-Learning MFC",
             "runtime": 15.0, "energy_output": 70.0,
             "avg_power": 0.7, "max_power": 1.3},
            {"success": True, "name": "Advanced Q-Learning MFC",
             "runtime": 18.0, "energy_output": 80.0,
             "avg_power": 0.8, "max_power": 1.5},
        ]
        with patch("run_gpu_simulation.generate_comparison_plots"):
            ret = rgs.analyze_and_compare_results(results)
            assert len(ret) == 4

    def test_zero_baseline(self):
        results = [
            {"success": True, "name": "Simple MFC",
             "runtime": 10.0, "energy_output": 0.0,
             "avg_power": 0.0, "max_power": 0.0},
            {"success": True, "name": "Q-Learning MFC",
             "runtime": 15.0, "energy_output": 50.0,
             "avg_power": 0.5, "max_power": 1.0},
        ]
        with patch("run_gpu_simulation.generate_comparison_plots"):
            ret = rgs.analyze_and_compare_results(results)
            assert len(ret) == 2

    def test_with_failures(self):
        results = [
            {"success": True, "name": "Simple MFC",
             "runtime": 10.0, "energy_output": 50.0,
             "avg_power": 0.5, "max_power": 1.0},
            {"success": False, "name": "Failed MFC",
             "runtime": 0.0, "energy_output": 0,
             "avg_power": 0, "max_power": 0,
             "error": "compilation failed"},
        ]
        with patch("run_gpu_simulation.generate_comparison_plots"):
            ret = rgs.analyze_and_compare_results(results)
            assert len(ret) == 1


@pytest.mark.coverage_extra
class TestSavePerformanceDataEdge:
    def test_zero_points(self):
        log = np.zeros((10, 8))
        with patch("builtins.open", mock_open()), patch("json.dump"):
            data = rgs.save_performance_data(log, 0)
            assert "metadata" in data

    def test_single_point(self):
        log = np.zeros((1, 8))
        log[0] = [1.0, 0.5, 0.3, 0.1, 10.0, 90.0, 85.0, 0]
        with patch("builtins.open", mock_open()), patch("json.dump"):
            data = rgs.save_performance_data(log, 1)
            assert data["metadata"]["data_points"] == 1


@pytest.mark.coverage_extra
class TestRunMojoEdge:
    def test_empty_output(self):
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("run_gpu_simulation.subprocess.run",
                    return_value=mock_result):
            r = rgs.run_mojo_simulation("Test", "t.mojo")
            assert r["success"] is True
            assert r["energy_output"] == 0

    def test_partial_parse(self):
        mock_result = MagicMock(
            returncode=0,
            stdout="Total energy produced: 42.0 Wh\nAverage power: 0.5 W\nMaximum power: 1.0 W\n",
            stderr="",
        )
        with patch("run_gpu_simulation.subprocess.run",
                    return_value=mock_result):
            r = rgs.run_mojo_simulation("Test", "t.mojo")
            assert r["energy_output"] == 42.0
            assert r["avg_power"] == 0.5
            assert r["max_power"] == 1.0
