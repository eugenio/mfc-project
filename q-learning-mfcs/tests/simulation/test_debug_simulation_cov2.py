"""Coverage tests for debug_simulation.py."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

mock_gpu_mod = MagicMock()
mock_mfc_sim = MagicMock()
mock_mfc_sim.reservoir_concentration = 25.0
mock_mfc_sim.outlet_concentration = 20.0
mock_mfc_sim.biofilm_thicknesses = [1.0, 1.1, 1.2, 1.3, 1.4]
mock_mfc_sim.simulate_timestep.return_value = {
    "total_power": 0.5,
    "action": 1,
    "reward": 5.0,
    "epsilon": 0.1,
    "substrate_addition": 0.01,
}
mock_gpu_mod.GPUAcceleratedMFC = MagicMock(return_value=mock_mfc_sim)
sys.modules["mfc_gpu_accelerated"] = mock_gpu_mod

mock_config_mod = MagicMock()
mock_config_mod.DEFAULT_QLEARNING_CONFIG = MagicMock()
mock_config_mod.DEFAULT_QLEARNING_CONFIG.n_cells = 5
mock_config_mod.DEFAULT_QLEARNING_CONFIG.electrode_area_per_cell = 10e-4
mock_config_mod.DEFAULT_QLEARNING_CONFIG.substrate_target_concentration = 25.0
sys.modules["config"] = MagicMock()
sys.modules["config.qlearning_config"] = mock_config_mod

from debug_simulation import test_simulation_steps, test_gui_simulation_setup


@pytest.mark.coverage_extra
class TestSimulationSteps:
    def test_basic_run(self):
        result = test_simulation_steps()
        assert result is True
        mock_mfc_sim.cleanup_gpu_resources.assert_called()

    def test_failure(self):
        mock_gpu_mod.GPUAcceleratedMFC.side_effect = Exception("fail")
        result = test_simulation_steps()
        assert result is False
        mock_gpu_mod.GPUAcceleratedMFC.side_effect = None
        mock_gpu_mod.GPUAcceleratedMFC.return_value = mock_mfc_sim

    def test_slow_step(self):
        import time as time_mod
        call_count = [0]
        orig_time = time_mod.time
        def slow_time():
            call_count[0] += 1
            return call_count[0] * 2.0
        with patch("debug_simulation.time.time", side_effect=slow_time):
            result = test_simulation_steps()


@pytest.mark.coverage_extra
class TestGuiSimulationSetup:
    def test_basic_run(self):
        mock_gpu_mod.GPUAcceleratedMFC.return_value = mock_mfc_sim
        result = test_gui_simulation_setup()
        assert result is True

    def test_failure(self):
        mock_gpu_mod.GPUAcceleratedMFC.side_effect = Exception("fail")
        result = test_gui_simulation_setup()
        assert result is False
        mock_gpu_mod.GPUAcceleratedMFC.side_effect = None
        mock_gpu_mod.GPUAcceleratedMFC.return_value = mock_mfc_sim
