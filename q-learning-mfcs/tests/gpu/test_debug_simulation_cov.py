"""Tests for debug_simulation.py - coverage target 98%+."""
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


class TestSimulationSteps:
    def test_success(self):
        from debug_simulation import test_simulation_steps

        mock_mfc = MagicMock()
        mock_mfc.reservoir_concentration = 25.0
        mock_mfc.simulate_timestep.return_value = {
            "total_power": 0.5,
            "action": 1,
            "reward": 10.0,
            "epsilon": 0.3,
        }
        mock_config = MagicMock()
        with patch("debug_simulation.DEFAULT_QLEARNING_CONFIG", mock_config):
            with patch(
                "debug_simulation.GPUAcceleratedMFC",
                return_value=mock_mfc,
            ):
                result = test_simulation_steps()
                assert result is True
                mock_mfc.cleanup_gpu_resources.assert_called_once()

    def test_failure(self):
        from debug_simulation import test_simulation_steps

        mock_config = MagicMock()
        with patch("debug_simulation.DEFAULT_QLEARNING_CONFIG", mock_config):
            with patch(
                "debug_simulation.GPUAcceleratedMFC",
                side_effect=RuntimeError("GPU fail"),
            ):
                result = test_simulation_steps()
                assert result is False

    def test_slow_step_warning(self):
        import time as _time

        from debug_simulation import test_simulation_steps

        mock_mfc = MagicMock()
        mock_mfc.reservoir_concentration = 25.0

        call_count = 0

        def slow_timestep(dt):
            nonlocal call_count
            call_count += 1
            _time.sleep(0.01)
            return {
                "total_power": 0.5,
                "action": 1,
                "reward": 10.0,
                "epsilon": 0.3,
            }

        mock_mfc.simulate_timestep.side_effect = slow_timestep
        mock_config = MagicMock()

        with patch("debug_simulation.DEFAULT_QLEARNING_CONFIG", mock_config):
            with patch(
                "debug_simulation.GPUAcceleratedMFC",
                return_value=mock_mfc,
            ):
                result = test_simulation_steps()
                assert result is True


class TestGUISimulationSetup:
    def test_success(self):
        from debug_simulation import test_gui_simulation_setup

        mock_mfc = MagicMock()
        mock_mfc.reservoir_concentration = 25.0
        mock_mfc.outlet_concentration = 12.0
        mock_mfc.biofilm_thicknesses = [1.0, 1.0, 1.0, 1.0, 1.0]
        mock_mfc.simulate_timestep.return_value = {
            "total_power": 0.5,
            "substrate_addition": 0.1,
            "action": 1,
            "epsilon": 0.3,
            "reward": 10.0,
        }
        mock_config = MagicMock()

        with patch("debug_simulation.DEFAULT_QLEARNING_CONFIG", mock_config):
            with patch(
                "debug_simulation.GPUAcceleratedMFC",
                return_value=mock_mfc,
            ):
                result = test_gui_simulation_setup()
                assert result is True
                mock_mfc.cleanup_gpu_resources.assert_called_once()

    def test_failure(self):
        from debug_simulation import test_gui_simulation_setup

        mock_config = MagicMock()
        with patch("debug_simulation.DEFAULT_QLEARNING_CONFIG", mock_config):
            with patch(
                "debug_simulation.GPUAcceleratedMFC",
                side_effect=RuntimeError("init fail"),
            ):
                result = test_gui_simulation_setup()
                assert result is False
