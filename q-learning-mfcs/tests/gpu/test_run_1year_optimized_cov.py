"""Tests for run_1year_optimized.py - coverage target 98%+."""
import sys
import os
import json
import signal
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


class TestSetupGpuAcceleration:
    def test_sets_env_vars(self):
        from run_1year_optimized import setup_gpu_acceleration

        setup_gpu_acceleration()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
        assert os.environ.get("OMP_NUM_THREADS") == "8"
        assert os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"


class TestCalculateMaintenanceRequirements:
    def test_basic(self):
        from run_1year_optimized import calculate_maintenance_requirements

        result = calculate_maintenance_requirements(1000.0, 8784)
        assert "substrate_requirements" in result
        assert "buffer_requirements" in result
        assert "maintenance_schedule" in result

        sr = result["substrate_requirements"]
        assert sr["total_consumed_mmol"] == 1000.0
        assert sr["consumption_rate_mmol_per_hour"] == pytest.approx(
            1000.0 / 8784
        )
        assert "refill_interval_days" in sr
        assert "stock_bottles_per_year" in sr

    def test_high_consumption(self):
        from run_1year_optimized import calculate_maintenance_requirements

        result = calculate_maintenance_requirements(50000.0, 8784)
        sr = result["substrate_requirements"]
        assert sr["refills_per_year"] > 1

    def test_low_consumption(self):
        from run_1year_optimized import calculate_maintenance_requirements

        result = calculate_maintenance_requirements(10.0, 8784)
        sr = result["substrate_requirements"]
        assert sr["refills_per_year"] < 1


class TestRun1YearSimulation:
    def test_success(self, tmp_path):
        from run_1year_optimized import run_1year_simulation

        mock_results = {
            "time_hours": [0, 1, 2, 3],
            "substrate_addition_rate": [0.1, 0.2, 0.15, 0.1],
            "reservoir_concentration": [25.0, 24.0, 24.5, 25.0],
            "outlet_concentration": [12.0, 11.0, 11.5, 12.0],
            "total_power": [0.5, 0.45, 0.48, 0.5],
        }
        mock_cells = [MagicMock()]
        mock_reservoir = MagicMock()
        mock_controller = MagicMock()
        mock_q = MagicMock()

        with patch(
            "run_1year_optimized.simulate_mfc_with_recirculation",
            return_value=(
                mock_results, mock_cells, mock_reservoir,
                mock_controller, mock_q,
            ),
        ):
            with patch(
                "run_1year_optimized.setup_gpu_acceleration"
            ):
                with patch("pathlib.Path.mkdir"):
                    with patch("builtins.open", mock_open()):
                        with patch("pandas.DataFrame.to_csv"):
                            with patch(
                                "run_1year_optimized.send_completion_email",
                                create=True,
                            ):
                                result, output_dir = run_1year_simulation()
                                assert result is not None
                                assert "simulation_info" in result
                                assert "performance_summary" in result
                                assert "maintenance_requirements" in result

    def test_keyboard_interrupt(self):
        from run_1year_optimized import run_1year_simulation

        with patch(
            "run_1year_optimized.simulate_mfc_with_recirculation",
            side_effect=KeyboardInterrupt,
        ):
            with patch("run_1year_optimized.setup_gpu_acceleration"):
                with patch("pathlib.Path.mkdir"):
                    result, output_dir = run_1year_simulation()
                    assert result is None
                    assert output_dir is None

    def test_exception(self):
        from run_1year_optimized import run_1year_simulation

        with patch(
            "run_1year_optimized.simulate_mfc_with_recirculation",
            side_effect=RuntimeError("sim fail"),
        ):
            with patch("run_1year_optimized.setup_gpu_acceleration"):
                with patch("pathlib.Path.mkdir"):
                    result, output_dir = run_1year_simulation()
                    assert result is None


class TestSignalHandler:
    def test_exits(self):
        from run_1year_optimized import signal_handler

        with pytest.raises(SystemExit):
            signal_handler(signal.SIGINT, None)
