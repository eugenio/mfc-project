"""Coverage tests for run_1year_optimized.py."""
import sys
import os
import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

mock_config = MagicMock()
mock_config.DEFAULT_QLEARNING_CONFIG = MagicMock()
mock_config.DEFAULT_QLEARNING_CONFIG.enhanced_learning_rate = 0.1
mock_config.DEFAULT_QLEARNING_CONFIG.enhanced_discount_factor = 0.95
mock_config.DEFAULT_QLEARNING_CONFIG.enhanced_epsilon = 0.3
mock_config.DEFAULT_QLEARNING_CONFIG.advanced_epsilon_decay = 0.995
mock_config.DEFAULT_QLEARNING_CONFIG.reward_weights = MagicMock()
mock_config.DEFAULT_QLEARNING_CONFIG.reward_weights.power_weight = 0.4
mock_config.DEFAULT_QLEARNING_CONFIG.reward_weights.substrate_reward_multiplier = 1.0
mock_config.DEFAULT_QLEARNING_CONFIG.reward_weights.biofilm_weight = 0.3

sys.modules["config"] = MagicMock()
sys.modules["config.qlearning_config"] = mock_config

mock_recirc = MagicMock()
mock_recirc.simulate_mfc_with_recirculation = MagicMock(return_value=(
    {
        "substrate_addition_rate": [0.1, 0.2, 0.3],
        "time_hours": [0, 1, 2],
        "reservoir_concentration": [25.0, 24.0, 23.0],
        "outlet_concentration": [20.0, 19.0, 18.0],
        "total_power": [0.5, 0.6, 0.7],
    },
    MagicMock(),
    MagicMock(),
    MagicMock(),
    MagicMock(),
))
sys.modules["mfc_recirculation_control"] = mock_recirc

for _stale in ["run_1year_optimized"]:
    if _stale in sys.modules:
        del sys.modules[_stale]

from run_1year_optimized import (
    setup_gpu_acceleration,
    calculate_maintenance_requirements,
    run_1year_simulation,
    signal_handler,
)


@pytest.mark.coverage_extra
class TestSetupGPU:
    def test_sets_env_vars(self):
        setup_gpu_acceleration()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
        assert os.environ.get("OMP_NUM_THREADS") == "8"


@pytest.mark.coverage_extra
class TestCalculateMaintenanceRequirements:
    def test_basic_calculation(self):
        result = calculate_maintenance_requirements(1000.0, 100.0)
        assert "substrate_requirements" in result
        assert "buffer_requirements" in result
        assert "maintenance_schedule" in result

    def test_substrate_fields(self):
        result = calculate_maintenance_requirements(5000.0, 8784.0)
        sr = result["substrate_requirements"]
        assert sr["total_consumed_mmol"] == 5000.0
        assert sr["consumption_rate_mmol_per_hour"] > 0
        assert sr["refills_per_year"] > 0
        assert sr["stock_bottles_per_year"] >= 1

    def test_buffer_fields(self):
        result = calculate_maintenance_requirements(5000.0, 8784.0)
        br = result["buffer_requirements"]
        assert br["total_consumed_mmol"] > 0
        assert br["refill_interval_days"] > 0

    def test_maintenance_schedule(self):
        result = calculate_maintenance_requirements(5000.0, 8784.0)
        ms = result["maintenance_schedule"]
        assert "Every" in ms["substrate_refill_frequency"]
        assert "Every" in ms["buffer_refill_frequency"]
        assert "Every" in ms["recommended_check_frequency"]


@pytest.mark.coverage_extra
class TestRun1YearSimulation:
    def test_successful_run(self, tmp_path):
        mock_output_dir = MagicMock()
        mock_output_dir.mkdir = MagicMock()
        mock_output_dir.__truediv__ = MagicMock(return_value=tmp_path / "result.json")
        mock_pd = MagicMock()
        mock_pd.DataFrame.return_value = MagicMock()
        with patch("run_1year_optimized.Path", return_value=mock_output_dir):
            with patch.dict("sys.modules", {"pandas": mock_pd}):
                with patch("builtins.open", MagicMock()):
                    with patch("run_1year_optimized.json.dump"):
                        results, output = run_1year_simulation()

    def test_keyboard_interrupt(self):
        mock_recirc.simulate_mfc_with_recirculation.side_effect = KeyboardInterrupt
        results, output = run_1year_simulation()
        assert results is None
        assert output is None
        mock_recirc.simulate_mfc_with_recirculation.side_effect = None
        mock_recirc.simulate_mfc_with_recirculation.return_value = (
            {
                "substrate_addition_rate": [0.1, 0.2, 0.3],
                "time_hours": [0, 1, 2],
                "reservoir_concentration": [25.0, 24.0, 23.0],
                "outlet_concentration": [20.0, 19.0, 18.0],
                "total_power": [0.5, 0.6, 0.7],
            },
            MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )

    def test_general_exception(self):
        mock_recirc.simulate_mfc_with_recirculation.side_effect = Exception("fail")
        results, output = run_1year_simulation()
        assert results is None
        mock_recirc.simulate_mfc_with_recirculation.side_effect = None
        mock_recirc.simulate_mfc_with_recirculation.return_value = (
            {
                "substrate_addition_rate": [0.1, 0.2, 0.3],
                "time_hours": [0, 1, 2],
                "reservoir_concentration": [25.0, 24.0, 23.0],
                "outlet_concentration": [20.0, 19.0, 18.0],
                "total_power": [0.5, 0.6, 0.7],
            },
            MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )

    def test_few_step_results(self):
        mock_recirc.simulate_mfc_with_recirculation.return_value = (
            {"substrate_addition_rate": [0.1, 0.2], "time_hours": [0, 1],
             "reservoir_concentration": [25.0, 24.0], "outlet_concentration": [20.0, 19.0],
             "total_power": [0.5, 0.6]},
            MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )
        mock_output_dir = MagicMock()
        mock_output_dir.mkdir = MagicMock()
        mock_output_dir.__truediv__ = MagicMock(return_value=MagicMock())
        mock_pd = MagicMock()
        mock_pd.DataFrame.return_value = MagicMock()
        with patch("run_1year_optimized.Path", return_value=mock_output_dir):
            with patch.dict("sys.modules", {"pandas": mock_pd}):
                with patch("builtins.open", MagicMock()):
                    with patch("run_1year_optimized.json.dump"):
                        results, output = run_1year_simulation()
        mock_recirc.simulate_mfc_with_recirculation.return_value = (
            {
                "substrate_addition_rate": [0.1, 0.2, 0.3],
                "time_hours": [0, 1, 2],
                "reservoir_concentration": [25.0, 24.0, 23.0],
                "outlet_concentration": [20.0, 19.0, 18.0],
                "total_power": [0.5, 0.6, 0.7],
            },
            MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )


@pytest.mark.coverage_extra
class TestSignalHandler:
    def test_signal_handler_exits(self):
        with pytest.raises(SystemExit):
            signal_handler(signal.SIGINT, None)
