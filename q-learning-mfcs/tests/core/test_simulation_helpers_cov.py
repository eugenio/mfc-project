"""Tests for simulation_helpers.py - Simulation runner with chronology."""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


@pytest.fixture
def mock_chronology():
    mgr = MagicMock()
    entry = MagicMock()
    entry.id = "test-entry-001"
    entry.timestamp = "2025-01-01T00:00:00"
    mgr.create_entry.return_value = entry
    return mgr


@pytest.fixture
def runner(tmp_path, mock_chronology):
    with patch(
        'simulation_helpers.get_chronology_manager',
        return_value=mock_chronology,
    ):
        from simulation_helpers import SimulationRunner
        return SimulationRunner(output_dir=str(tmp_path / "sim_out"))


class TestSimulationRunnerInit:
    def test_creates_output_dir(self, runner):
        assert Path(runner.output_dir).exists()


class TestRunWithTracking:
    def test_success_path(self, runner, mock_chronology):
        def sim_func():
            return {"total_energy": 42.0}
        result = runner.run_simulation_with_tracking(
            simulation_name="test_sim",
            simulation_func=sim_func,
            description="desc",
            tags=["test"],
        )
        assert result["simulation_metadata"]["success"] is True
        assert result["total_energy"] == 42.0

    def test_failure_path(self, runner):
        def failing():
            raise ValueError("boom")
        result = runner.run_simulation_with_tracking(
            simulation_name="fail", simulation_func=failing,
        )
        assert result["simulation_metadata"]["success"] is False
        assert "error" in result

    def test_no_browser_download(self, runner):
        result = runner.run_simulation_with_tracking(
            simulation_name="x",
            simulation_func=lambda: {"a": 1},
            enable_browser_download=False,
        )
        assert result["download_files"] == {}


class TestExtractSummary:
    def test_key_fields(self, runner):
        r = {"total_energy": 1, "average_power": 2, "peak_power": 3}
        s = runner._extract_results_summary(r)
        assert s["total_energy"] == 1

    def test_time_series_stats(self, runner):
        r = {"time_series": {
            "time": [0, 1], "power": [1.0, 3.0],
            "voltage": [0.5, 0.7], "current": [2.0, 4.0],
        }}
        s = runner._extract_results_summary(r)
        assert s["power_mean"] == 2.0
        assert s["power_max"] == 3.0

    def test_time_series_non_dict(self, runner):
        s = runner._extract_results_summary({"time_series": "str"})
        assert "time_series_length" not in s

    def test_empty_series(self, runner):
        s = runner._extract_results_summary(
            {"time_series": {"time": [], "power": []}}
        )
        assert s["time_series_length"] == 0

    def test_no_fields(self, runner):
        assert runner._extract_results_summary({"x": 1}) == {}


class TestBrowserDownload:
    def test_basic_files(self, runner):
        files = runner._prepare_browser_download_files(
            "e1", {"total_energy": 1.0}
        )
        assert "results_json" in files
        assert "summary_report" in files

    def test_with_time_series(self, runner):
        results = {
            "time_series": {"time": [0, 1], "power": [1.0, 2.0]},
        }
        files = runner._prepare_browser_download_files("e2", results)
        assert "time_series_csv" in files

    def test_with_configs(self, runner):
        mock_ql = MagicMock()
        mock_sensor = MagicMock()
        with patch('config.config_io.save_config'):
            files = runner._prepare_browser_download_files(
                "e3", {"x": 1},
                qlearning_config=mock_ql,
                sensor_config=mock_sensor,
            )

    def test_exception_in_prep(self, runner):
        with patch('builtins.open', side_effect=OSError("fail")):
            files = runner._prepare_browser_download_files("e4", {"x": 1})
            assert isinstance(files, dict)


class TestExportCSV:
    def test_with_pandas(self, runner, tmp_path):
        ts = {"time": [0, 1], "power": [1.0, 2.0]}
        out = tmp_path / "t.csv"
        assert runner._export_time_series_csv(ts, out) == out

    def test_empty(self, runner, tmp_path):
        assert runner._export_time_series_csv(
            {"t": [], "p": []}, tmp_path / "e.csv"
        ) is None

    def test_non_list(self, runner, tmp_path):
        assert runner._export_time_series_csv(
            {"scalar": 42}, tmp_path / "n.csv"
        ) is None

    def test_pandas_error(self, runner, tmp_path):
        with patch('pandas.DataFrame', side_effect=RuntimeError("x")):
            assert runner._export_time_series_csv(
                {"t": [1]}, tmp_path / "err.csv"
            ) is None


class TestSummaryReport:
    def test_full_report(self, runner, tmp_path):
        results = {
            "total_energy": 100.0, "average_power": 2.5,
            "peak_power": 5.0, "coulombic_efficiency": 85.0,
            "final_biofilm_thickness": 0.1,
            "final_current_density": 10.0,
            "time_series": {"time": [0, 1], "power": [1, 2]},
            "simulation_metadata": {
                "name": "n", "description": "d",
                "execution_time_seconds": 1.0,
                "success": True, "tags": ["t1"],
            },
        }
        out = tmp_path / "r.txt"
        assert runner._create_summary_report("e1", results, out) == out
        c = out.read_text()
        assert "MFC SIMULATION" in c
        assert "t1" in c

    def test_na_values(self, runner, tmp_path):
        out = tmp_path / "na.txt"
        runner._create_summary_report("e2", {}, out)
        assert "N/A" in out.read_text()

    def test_error_path(self, runner, tmp_path):
        bad = tmp_path / "no" / "deep" / "r.txt"
        assert runner._create_summary_report("e3", {}, bad) is None


class TestQuickSimulation:
    def test_runs(self):
        with patch('simulation_helpers.get_chronology_manager') as m:
            entry = MagicMock()
            entry.id = "q1"
            entry.timestamp = "2025"
            m.return_value.create_entry.return_value = entry
            from simulation_helpers import quick_simulation_with_chronology
            r = quick_simulation_with_chronology(
                "q", lambda: {"p": 1}, tags=["t"],
            )
            assert r["p"] == 1
