"""Coverage tests for run_gpu_simulation.py - lines 117-383, 673-693."""
import os
import sys
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


@pytest.fixture(autouse=True)
def mock_paths():
    with patch("run_gpu_simulation.get_figure_path", return_value="/tmp/f.png"), \
         patch("run_gpu_simulation.get_simulation_data_path", return_value="/tmp/d.json"):
        yield


class TestSavePerformanceData:
    def test_basic(self):
        from run_gpu_simulation import save_performance_data
        perf_log = np.zeros((10, 8))
        perf_log[:, 0] = np.arange(10) * 100
        perf_log[:, 3] = 0.5
        perf_log[:, 4] = 50.0
        with patch('builtins.open', mock_open()):
            with patch('json.dump'):
                result = save_performance_data(perf_log, 10)
        assert 'metadata' in result
        assert 'time_series' in result


class TestGeneratePlots:
    def test_min_points(self):
        from run_gpu_simulation import generate_plots
        with patch('run_gpu_simulation.plt') as mp:
            generate_plots(np.zeros((1, 8)), 1)
            assert not mp.subplots.called

    def test_valid(self):
        from run_gpu_simulation import generate_plots
        with patch('run_gpu_simulation.plt') as mp:
            mock_axes = MagicMock()
            mp.subplots.return_value = (MagicMock(), mock_axes)
            perf = np.zeros((10, 8))
            perf[:, 0] = np.arange(10) * 100
            generate_plots(perf, 10)
            assert mp.tight_layout.called


class TestAnalyzeAndCompareResults:
    def test_no_success(self):
        from run_gpu_simulation import analyze_and_compare_results
        assert analyze_and_compare_results(
            [{"success": False, "name": "T", "runtime": 0}]
        ) is None

    def test_single(self):
        from run_gpu_simulation import analyze_and_compare_results
        r = [{
            "success": True, "name": "Simple MFC",
            "runtime": 10.0, "energy_output": 50.0,
            "avg_power": 0.5, "max_power": 1.0,
        }]
        with patch('run_gpu_simulation.generate_comparison_plots'):
            assert len(analyze_and_compare_results(r)) == 1

    def test_multi_with_technologies(self):
        from run_gpu_simulation import analyze_and_compare_results
        r = [
            {"success": True, "name": "Simple MFC",
             "runtime": 10.0, "energy_output": 50.0,
             "avg_power": 0.5, "max_power": 1.0},
            {"success": True, "name": "Enhanced Q-Learning MFC",
             "runtime": 20.0, "energy_output": 80.0,
             "avg_power": 0.8, "max_power": 1.5},
            {"success": True, "name": "Advanced Q-Learning MFC",
             "runtime": 15.0, "energy_output": 70.0,
             "avg_power": 0.7, "max_power": 1.2},
            {"success": False, "name": "Failed",
             "runtime": 0, "energy_output": 0,
             "avg_power": 0, "max_power": 0},
        ]
        with patch('run_gpu_simulation.generate_comparison_plots'):
            res = analyze_and_compare_results(r)
        assert len(res) == 3


class TestGenerateComparisonPlots:
    def test_single_result(self):
        from run_gpu_simulation import generate_comparison_plots
        with patch('run_gpu_simulation.plt') as mp:
            generate_comparison_plots([{"name": "A"}])
            assert not mp.subplots.called

    def test_valid(self):
        from run_gpu_simulation import generate_comparison_plots
        with patch('run_gpu_simulation.plt') as mp:
            mock_axes = MagicMock()
            mp.subplots.return_value = (MagicMock(), mock_axes)
            r = [
                {"name": "Simple MFC", "energy_output": 50.0,
                 "avg_power": 0.5, "max_power": 1.0, "runtime": 10.0},
                {"name": "Q-Learning MFC", "energy_output": 80.0,
                 "avg_power": 0.8, "max_power": 1.5, "runtime": 20.0},
            ]
            generate_comparison_plots(r)
            assert mp.tight_layout.called


class TestMain:
    def test_no_results(self):
        from run_gpu_simulation import main
        with patch('run_gpu_simulation.run_all_mojo_simulations', return_value=[]), \
             patch('run_gpu_simulation.analyze_and_compare_results', return_value=None), \
             patch('run_gpu_simulation.run_accelerated_python_simulation') as mp:
            main()
            mp.assert_called_once()

    def test_above_target(self):
        from run_gpu_simulation import main
        best = {"name": "B", "energy_output": 100.0,
                "avg_power": 1.0, "max_power": 2.0, "runtime": 10.0}
        with patch('run_gpu_simulation.run_all_mojo_simulations', return_value=[{"success": True}]), \
             patch('run_gpu_simulation.analyze_and_compare_results', return_value=[best]), \
             patch('run_gpu_simulation.run_accelerated_python_simulation') as mp:
            main()
            mp.assert_not_called()

    def test_below_target(self):
        from run_gpu_simulation import main
        best = {"name": "B", "energy_output": 50.0,
                "avg_power": 0.5, "max_power": 1.0, "runtime": 10.0}
        with patch('run_gpu_simulation.run_all_mojo_simulations', return_value=[{"success": True}]), \
             patch('run_gpu_simulation.analyze_and_compare_results', return_value=[best]), \
             patch('run_gpu_simulation.run_accelerated_python_simulation') as mp:
            main()
            mp.assert_not_called()
