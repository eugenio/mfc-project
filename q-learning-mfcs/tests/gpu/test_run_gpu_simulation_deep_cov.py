"""Comprehensive coverage tests for run_gpu_simulation module."""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pytest


class TestRunMojoSimulation:
    def test_success(self):
        from run_gpu_simulation import run_mojo_simulation
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "Total energy produced: 42.5 Wh\n"
            "Average power: 0.425 W\n"
            "Maximum power: 0.850 W\n"
        )
        with patch("run_gpu_simulation.subprocess.run", return_value=mock_result):
            result = run_mojo_simulation("test", "test.mojo", timeout=10)
        assert result["success"] is True
        assert result["energy_output"] == pytest.approx(42.5)
        assert result["avg_power"] == pytest.approx(0.425)
        assert result["max_power"] == pytest.approx(0.850)

    def test_failure(self):
        from run_gpu_simulation import run_mojo_simulation
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "compilation error"
        with patch("run_gpu_simulation.subprocess.run", return_value=mock_result):
            result = run_mojo_simulation("test", "test.mojo")
        assert result["success"] is False
        assert result["error"] == "compilation error"

    def test_timeout(self):
        import subprocess as sp
        from run_gpu_simulation import run_mojo_simulation
        with patch("run_gpu_simulation.subprocess.run", side_effect=sp.TimeoutExpired("cmd", 5)):
            result = run_mojo_simulation("test", "test.mojo", timeout=5)
        assert result["success"] is False
        assert result["error"] == "Timeout"

    def test_file_not_found(self):
        from run_gpu_simulation import run_mojo_simulation
        with patch("run_gpu_simulation.subprocess.run", side_effect=FileNotFoundError()):
            result = run_mojo_simulation("test", "test.mojo")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_general_exception(self):
        from run_gpu_simulation import run_mojo_simulation
        with patch("run_gpu_simulation.subprocess.run", side_effect=RuntimeError("err")):
            result = run_mojo_simulation("test", "test.mojo")
        assert result["success"] is False

    def test_parse_no_energy(self):
        from run_gpu_simulation import run_mojo_simulation
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Done.\n"
        with patch("run_gpu_simulation.subprocess.run", return_value=mock_result):
            result = run_mojo_simulation("test", "test.mojo")
        assert result["success"] is True
        assert result["energy_output"] == 0


class TestRunAllMojoSimulations:
    def test_runs_all(self):
        from run_gpu_simulation import run_all_mojo_simulations
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        with patch("run_gpu_simulation.subprocess.run", return_value=mock_result):
            results = run_all_mojo_simulations()
        assert len(results) == 4


class TestAnalyzeAndCompareResults:
    def test_no_successful(self):
        from run_gpu_simulation import analyze_and_compare_results
        results = [{"success": False, "name": "test", "error": "fail"}]
        assert analyze_and_compare_results(results) is None

    def test_two_successful(self):
        from run_gpu_simulation import analyze_and_compare_results
        results = [
            {"success": True, "name": "Simple MFC", "runtime": 10.0, "energy_output": 50.0, "avg_power": 0.5, "max_power": 1.0},
            {"success": True, "name": "Q-Learning MFC", "runtime": 15.0, "energy_output": 75.0, "avg_power": 0.75, "max_power": 1.2},
        ]
        with patch("run_gpu_simulation.generate_comparison_plots"):
            result = analyze_and_compare_results(results)
        assert len(result) == 2

    def test_mixed_with_failures(self):
        from run_gpu_simulation import analyze_and_compare_results
        results = [
            {"success": True, "name": "Simple MFC", "runtime": 10.0, "energy_output": 50.0, "avg_power": 0.5, "max_power": 1.0},
            {"success": False, "name": "Failed", "error": "err"},
        ]
        with patch("run_gpu_simulation.generate_comparison_plots"):
            result = analyze_and_compare_results(results)
        assert len(result) == 1


class TestSavePerformanceData:
    def test_saves_data(self, tmp_path):
        from run_gpu_simulation import save_performance_data
        log = np.zeros((100, 8))
        log[:, 0] = np.arange(100)
        with patch("run_gpu_simulation.get_simulation_data_path", return_value=str(tmp_path / "data.json")):
            data = save_performance_data(log, 50)
        assert data["metadata"]["data_points"] == 50


class TestGeneratePlots:
    def test_few_points(self):
        from run_gpu_simulation import generate_plots
        log = np.zeros((1, 8))
        generate_plots(log, 1)

    def test_normal(self):
        from run_gpu_simulation import generate_plots
        import matplotlib
        matplotlib.use("Agg")
        log = np.random.rand(100, 8)
        log[:, 0] = np.arange(100)
        with patch("run_gpu_simulation.get_figure_path", return_value="/tmp/test.png"):
            generate_plots(log, 100)


class TestGenerateComparisonPlots:
    def test_too_few_results(self):
        from run_gpu_simulation import generate_comparison_plots
        generate_comparison_plots([{"name": "x", "energy_output": 1, "avg_power": 0.1, "max_power": 0.2, "runtime": 1}])

    def test_normal(self):
        from run_gpu_simulation import generate_comparison_plots
        import matplotlib
        matplotlib.use("Agg")
        results = [
            {"name": "A MFC", "energy_output": 50, "avg_power": 0.5, "max_power": 1.0, "runtime": 10},
            {"name": "B MFC", "energy_output": 75, "avg_power": 0.75, "max_power": 1.2, "runtime": 15},
        ]
        with patch("run_gpu_simulation.get_figure_path", return_value="/tmp/test.png"):
            generate_comparison_plots(results)
