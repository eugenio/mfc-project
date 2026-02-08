"""Tests for run_gpu_simulation.py - coverage target 98%+."""
import sys
import os
import json
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.fixture(autouse=True)
def mock_path_config():
    with patch("run_gpu_simulation.get_figure_path", return_value="/tmp/fig.png"):
        with patch(
            "run_gpu_simulation.get_simulation_data_path",
            return_value="/tmp/data.json",
        ):
            yield


class TestRunMojoSimulation:
    def test_success(self):
        from run_gpu_simulation import run_mojo_simulation

        mock_result = MagicMock(
            returncode=0,
            stdout="Total energy produced: 100.5 Wh\nAverage power: 1.2 W\nMaximum power: 2.5 W\n",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_result):
            r = run_mojo_simulation("Test", "test.mojo")
            assert r["success"] is True
            assert r["energy_output"] == 100.5
            assert r["avg_power"] == 1.2
            assert r["max_power"] == 2.5

    def test_failure(self):
        from run_gpu_simulation import run_mojo_simulation

        mock_result = MagicMock(returncode=1, stdout="", stderr="error")
        with patch("subprocess.run", return_value=mock_result):
            r = run_mojo_simulation("Test", "test.mojo")
            assert r["success"] is False
            assert r["error"] == "error"

    def test_timeout(self):
        import subprocess

        from run_gpu_simulation import run_mojo_simulation

        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 600)
        ):
            r = run_mojo_simulation("Test", "test.mojo", timeout=600)
            assert r["success"] is False
            assert r["error"] == "Timeout"
            assert r["runtime"] == 600

    def test_not_found(self):
        from run_gpu_simulation import run_mojo_simulation

        with patch("subprocess.run", side_effect=FileNotFoundError):
            r = run_mojo_simulation("Test", "test.mojo")
            assert r["success"] is False
            assert "not found" in r["error"]

    def test_generic_exception(self):
        from run_gpu_simulation import run_mojo_simulation

        with patch("subprocess.run", side_effect=RuntimeError("boom")):
            r = run_mojo_simulation("Test", "test.mojo")
            assert r["success"] is False
            assert "boom" in r["error"]

    def test_parse_error(self):
        from run_gpu_simulation import run_mojo_simulation

        mock_result = MagicMock(
            returncode=0,
            stdout="Total energy produced: bad_value Wh\nAverage power: bad W\nMaximum power: bad W\n",
        )
        with patch("subprocess.run", return_value=mock_result):
            r = run_mojo_simulation("Test", "test.mojo")
            assert r["success"] is True
            assert r["energy_output"] == 0
            assert r["avg_power"] == 0


class TestRunAllMojoSimulations:
    def test_parallel(self):
        from run_gpu_simulation import run_all_mojo_simulations

        def fake_run(name, file, timeout=600):
            return {"name": name, "file": file, "success": True, "runtime": 1.0}

        with patch("run_gpu_simulation.run_mojo_simulation", side_effect=fake_run):
            results = run_all_mojo_simulations()
            assert len(results) == 4


class TestAnalyzeAndCompareResults:
    def test_no_successful(self):
        from run_gpu_simulation import analyze_and_compare_results

        results = [{"success": False, "name": "A", "error": "err"}]
        with patch("run_gpu_simulation.generate_comparison_plots"):
            ret = analyze_and_compare_results(results)
            assert ret is None

    def test_with_successful(self):
        from run_gpu_simulation import analyze_and_compare_results

        results = [
            {
                "success": True,
                "name": "Simple MFC",
                "runtime": 10.0,
                "energy_output": 100.0,
                "avg_power": 1.0,
                "max_power": 2.0,
            },
            {
                "success": True,
                "name": "Q-Learning MFC",
                "runtime": 15.0,
                "energy_output": 120.0,
                "avg_power": 1.2,
                "max_power": 2.5,
            },
        ]
        with patch("run_gpu_simulation.generate_comparison_plots"):
            ret = analyze_and_compare_results(results)
            assert ret is not None
            assert len(ret) == 2

    def test_with_failed(self):
        from run_gpu_simulation import analyze_and_compare_results

        results = [
            {
                "success": True,
                "name": "Simple MFC",
                "runtime": 10.0,
                "energy_output": 50.0,
                "avg_power": 0.5,
                "max_power": 1.0,
            },
            {"success": False, "name": "Failed MFC", "error": "oops"},
        ]
        with patch("run_gpu_simulation.generate_comparison_plots"):
            ret = analyze_and_compare_results(results)
            assert len(ret) == 1


class TestSavePerformanceData:
    def test_save(self):
        from run_gpu_simulation import save_performance_data

        log = np.zeros((10, 8))
        log[:, 0] = np.arange(10)
        log[:, 3] = 1.0
        log[:, 4] = np.arange(10)

        m = mock_open()
        with patch("builtins.open", m):
            data = save_performance_data(log, 10)
            assert "metadata" in data
            assert "time_series" in data
            assert data["metadata"]["data_points"] == 10


class TestGeneratePlots:
    def test_too_few_points(self):
        from run_gpu_simulation import generate_plots

        log = np.zeros((1, 8))
        generate_plots(log, 1)

    def test_normal(self):
        from run_gpu_simulation import generate_plots

        with patch("run_gpu_simulation.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_axes = np.empty((2, 2), dtype=object)
            for r in range(2):
                for c in range(2):
                    mock_axes[r, c] = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            log = np.random.rand(20, 8)
            log[:, 0] = np.linspace(0, 100, 20)
            generate_plots(log, 20)
            mock_plt.savefig.assert_called_once()


class TestGenerateComparisonPlots:
    def test_too_few(self):
        from run_gpu_simulation import generate_comparison_plots

        generate_comparison_plots([{"name": "A", "success": True}])

    def test_normal(self):
        from run_gpu_simulation import generate_comparison_plots

        results = [
            {
                "name": "Simple MFC",
                "energy_output": 100.0,
                "avg_power": 1.0,
                "max_power": 2.0,
                "runtime": 10.0,
            },
            {
                "name": "Q-Learning MFC",
                "energy_output": 120.0,
                "avg_power": 1.2,
                "max_power": 2.5,
                "runtime": 15.0,
            },
        ]
        with patch("run_gpu_simulation.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_axes = np.empty((2, 2), dtype=object)
            for r in range(2):
                for c in range(2):
                    mock_axes[r, c] = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            generate_comparison_plots(results)
            mock_plt.savefig.assert_called_once()
