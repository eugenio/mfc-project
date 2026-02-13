"""Coverage tests for run_simulation.py - targeting uncovered lines.

Covers: _run_100h with both import paths, _run_gpu with successful GPU init,
_run_stack with successful import, _run_comprehensive with successful import,
_run_demo interrupted mid-loop, _save_results pickle failure,
_generate_plots with all branch variants, main with all CLI flags,
simplified Q-learning edge cases, and error branches.
"""
import sys
import os
import json
import pickle
import signal
import time
import argparse

import matplotlib
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Clean stale path_config mock and force fresh import of run_simulation
from run_simulation import (
class TestRunSimulationCov2ConfigEdgeCases:
    """Cover config edge cases not tested before."""

    def test_from_mode_with_all_kwargs(self):
        """Cover all keyword argument overrides at once."""
        c = SimulationConfig.from_mode(
            "demo",
            n_cells=7,
            duration_hours=2.5,
            time_step=5.0,
            use_gpu=True,
            enable_sensors=True,
            output_dir="/tmp/override",
            verbose=False,
        )
        assert c.n_cells == 7
        assert c.duration_hours == 2.5
        assert c.time_step == 5.0
        assert c.use_gpu is True
        assert c.enable_sensors is True
        assert c.output_dir == "/tmp/override"
        assert c.verbose is False

    def test_default_output_dir_varies_by_mode(self):
        """Cover _default_output_dir with different modes."""
        c1 = SimulationConfig(mode="100h")
        c2 = SimulationConfig(mode="stack")
        assert "100h_simulation" in c1.output_dir
        assert "stack_simulation" in c2.output_dir

class TestRun100hWithMFCStackImport:
    """Cover _run_100h lines where mfc_stack_simulation imports succeed
    but mfc_100h_simulation import fails (fallback path using MFCStack)."""

    def test_100h_with_stack_fallback(self, tmp_path):
        """Cover lines 299-328: import MFCStack succeeds, import 100h fails."""
        c = SimulationConfig.from_mode(
            "100h",
            output_dir=str(tmp_path),
            verbose=False,
            duration_hours=0.001,
            time_step=60.0,
        )
        r = UnifiedSimulationRunner(c)

        # Create mock stack and controller
        mock_stack = MagicMock()
        mock_stack.data_log = {"stack_power": [0.1, 0.2]}
        mock_stack.stack_power = 0.2

        mock_ctrl = MagicMock()
        mock_ctrl.q_table = {"s1": [0.1]}

        mock_stack_module = MagicMock()
        mock_stack_module.MFCStack = MagicMock(return_value=mock_stack)
        mock_stack_module.MFCStackQLearningController = MagicMock(
            return_value=mock_ctrl
        )

        with patch.dict("sys.modules", {
            "mfc_stack_simulation": mock_stack_module,
            "mfc_100h_simulation": None,  # Force ImportError
        }):
            # Force reimport-level bypass: patch the import inside method
            def fake_import(name, *args, **kwargs):
                if name == "mfc_100h_simulation":
                    raise ImportError("no 100h module")
                if name == "mfc_stack_simulation":
                    return mock_stack_module
                return __builtins__.__import__(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                result = r._run_100h()

        # It should either succeed with stack data or fall through to simplified
        assert isinstance(result, dict)

    def test_100h_both_imports_fail(self, tmp_path):
        """Cover lines 295-297: both imports fail, falls back to simplified."""
        c = SimulationConfig.from_mode(
            "100h",
            output_dir=str(tmp_path),
            verbose=False,
            duration_hours=0.001,
            time_step=60.0,
        )
        r = UnifiedSimulationRunner(c)

        def fake_import(name, *args, **kwargs):
            if name in ("mfc_stack_simulation", "mfc_100h_simulation"):
                raise ImportError("mocked")
            return __builtins__.__import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_100h()

        assert "time_series" in result
        assert "q_table_size" in result

    def test_100h_with_long_term_stack(self, tmp_path):
        """Cover lines 330-353: both imports succeed, uses LongTermMFCStack."""
        c = SimulationConfig.from_mode(
            "100h",
            output_dir=str(tmp_path),
            verbose=False,
            duration_hours=0.001,
            time_step=60.0,
        )
        r = UnifiedSimulationRunner(c)

        mock_stack = MagicMock()
        mock_stack.hourly_data = {"test": [1, 2]}
        mock_stack.stack_power = 0.3
        mock_stack.total_energy_produced = 10.0
        mock_stack.maintenance_cycles = 2

        mock_ctrl = MagicMock()
        mock_ctrl.q_table = {"s1": [0.1], "s2": [0.2]}

        mock_stack_mod = MagicMock()
        mock_100h_mod = MagicMock()
        mock_100h_mod.LongTermMFCStack = MagicMock(return_value=mock_stack)
        mock_100h_mod.LongTermController = MagicMock(return_value=mock_ctrl)

        with patch.dict("sys.modules", {
            "mfc_stack_simulation": mock_stack_mod,
            "mfc_100h_simulation": mock_100h_mod,
        }):
            def fake_import(name, *args, **kwargs):
                if name == "mfc_stack_simulation":
                    return mock_stack_mod
                if name == "mfc_100h_simulation":
                    return mock_100h_mod
                return __builtins__.__import__(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                result = r._run_100h()

        assert isinstance(result, dict)

class TestRunGPUWithSuccessfulInit:
    """Cover _run_gpu lines where GPU module loads and runs."""

    def test_gpu_successful_run(self, tmp_path):
        """Cover lines 364-413: GPU module loads, runs, cleanup."""
        c = SimulationConfig.from_mode(
            "gpu",
            output_dir=str(tmp_path),
            verbose=False,
            duration_hours=0.3,  # Very short for speed
            time_step=60.0,
        )
        r = UnifiedSimulationRunner(c)

        mock_mfc_sim = MagicMock()
        mock_mfc_sim.gpu_available = True
        mock_mfc_sim.reservoir_concentration = 25.0
        mock_mfc_sim.simulate_timestep.return_value = {"total_power": 0.5}

        mock_gpu_mod = MagicMock()
        mock_gpu_mod.GPUAcceleratedMFC = MagicMock(return_value=mock_mfc_sim)

        mock_config_mod = MagicMock()
        mock_config_mod.DEFAULT_QLEARNING_CONFIG = MagicMock()
        mock_config_mod.DEFAULT_QLEARNING_CONFIG.n_cells = 5

        def fake_import(name, *args, **kwargs):
            if name == "mfc_gpu_accelerated":
                return mock_gpu_mod
            if name == "config.qlearning_config":
                return mock_config_mod
            return __builtins__.__import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_gpu()

        assert result["gpu_accelerated"] is True
        assert "time_series" in result
        mock_mfc_sim.cleanup_gpu_resources.assert_called_once()

    def test_gpu_init_failure(self, tmp_path):
        """Cover lines 377-379: GPU init raises exception, falls back."""
        c = SimulationConfig.from_mode(
            "gpu",
            output_dir=str(tmp_path),
            verbose=False,
            duration_hours=0.001,
        )
        r = UnifiedSimulationRunner(c)

        mock_gpu_mod = MagicMock()
        mock_gpu_mod.GPUAcceleratedMFC = MagicMock(
            side_effect=Exception("GPU fail")
        )

        mock_config_mod = MagicMock()
        mock_config_mod.DEFAULT_QLEARNING_CONFIG = MagicMock()

        def fake_import(name, *args, **kwargs):
            if name == "mfc_gpu_accelerated":
                return mock_gpu_mod
            if name == "config.qlearning_config":
                return mock_config_mod
            return __builtins__.__import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_gpu()

        # Should fall back to simplified Q-learning
        assert "time_series" in result

    def test_gpu_interrupted(self, tmp_path):
        """Cover GPU run with interruption mid-loop."""
        c = SimulationConfig.from_mode(
            "gpu",
            output_dir=str(tmp_path),
            verbose=False,
            duration_hours=1.0,
        )
        r = UnifiedSimulationRunner(c)
        r._interrupted = True

        mock_mfc_sim = MagicMock()
        mock_mfc_sim.gpu_available = True
        mock_mfc_sim.reservoir_concentration = 25.0
        mock_mfc_sim.simulate_timestep.return_value = {"total_power": 0.5}

        mock_gpu_mod = MagicMock()
        mock_gpu_mod.GPUAcceleratedMFC = MagicMock(return_value=mock_mfc_sim)

        mock_config_mod = MagicMock()
        mock_config_mod.DEFAULT_QLEARNING_CONFIG = MagicMock()

        def fake_import(name, *args, **kwargs):
            if name == "mfc_gpu_accelerated":
                return mock_gpu_mod
            if name == "config.qlearning_config":
                return mock_config_mod
            return __builtins__.__import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_gpu()

        mock_mfc_sim.cleanup_gpu_resources.assert_called_once()

    def test_gpu_no_gpu_available_attr(self, tmp_path):
        """Cover line 376: GPU sim without gpu_available attribute."""
        c = SimulationConfig.from_mode(
            "gpu",
            output_dir=str(tmp_path),
            verbose=True,
            duration_hours=0.2,
        )
        r = UnifiedSimulationRunner(c)

        mock_mfc_sim = MagicMock(spec=[])  # no gpu_available attr
        mock_mfc_sim.reservoir_concentration = 25.0
        mock_mfc_sim.simulate_timestep = MagicMock(
            return_value={"total_power": 0.5}
        )
        mock_mfc_sim.cleanup_gpu_resources = MagicMock()

        mock_gpu_mod = MagicMock()
        mock_gpu_mod.GPUAcceleratedMFC = MagicMock(return_value=mock_mfc_sim)

        mock_config_mod = MagicMock()
        mock_config_mod.DEFAULT_QLEARNING_CONFIG = MagicMock()

        def fake_import(name, *args, **kwargs):
            if name == "mfc_gpu_accelerated":
                return mock_gpu_mod
            if name == "config.qlearning_config":
                return mock_config_mod
            return __builtins__.__import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_gpu()

        assert isinstance(result, dict)

class TestRunStackWithSuccessfulImport:
    """Cover _run_stack lines where mfc_stack_simulation import succeeds."""

    def test_stack_successful_run(self, tmp_path):
        """Cover lines 419-460: stack simulation runs normally."""
        c = SimulationConfig.from_mode(
            "stack",
            output_dir=str(tmp_path),
            verbose=False,
            duration_hours=0.001,
            time_step=60.0,
        )
        r = UnifiedSimulationRunner(c)

        mock_cell = MagicMock()
        mock_cell.get_power.return_value = 0.2
        mock_cell.is_reversed = False

        mock_stack = MagicMock()
        mock_stack.data_log = {"stack_power": [0.5, 0.6]}
        mock_stack.stack_power = 0.6
        mock_stack.stack_voltage = 2.5
        mock_stack.cells = [mock_cell] * 5
        mock_stack.check_system_health.return_value = {"reversed_cells": 0}

        mock_ctrl = MagicMock()
        mock_ctrl.q_table = {"s1": [0.1]}

        mock_mod = MagicMock()
        mock_mod.MFCStack = MagicMock(return_value=mock_stack)
        mock_mod.MFCStackQLearningController = MagicMock(return_value=mock_ctrl)

        def fake_import(name, *args, **kwargs):
            if name == "mfc_stack_simulation":
                return mock_mod
            return __builtins__.__import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_stack()

        assert "final_cell_states" in result
        assert len(result["final_cell_states"]) == 5

    def test_stack_interrupted(self, tmp_path):
        """Cover stack simulation with interruption."""
        c = SimulationConfig.from_mode(
            "stack",
            output_dir=str(tmp_path),
            verbose=False,
            duration_hours=1.0,
            time_step=1.0,
        )
        r = UnifiedSimulationRunner(c)
        r._interrupted = True

        mock_cell = MagicMock()
        mock_cell.get_power.return_value = 0.2
        mock_cell.is_reversed = False

        mock_stack = MagicMock()
        mock_stack.data_log = {"stack_power": [0.5]}
        mock_stack.stack_power = 0.5
        mock_stack.stack_voltage = 2.0
        mock_stack.cells = [mock_cell] * 5

        mock_ctrl = MagicMock()
        mock_ctrl.q_table = {}

        mock_mod = MagicMock()
        mock_mod.MFCStack = MagicMock(return_value=mock_stack)
        mock_mod.MFCStackQLearningController = MagicMock(return_value=mock_ctrl)

        def fake_import(name, *args, **kwargs):
            if name == "mfc_stack_simulation":
                return mock_mod
            return __builtins__.__import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_stack()

        assert isinstance(result, dict)

class TestRunComprehensiveWithSuccessfulImport:
    """Cover _run_comprehensive lines where imports succeed."""

    def test_comprehensive_successful(self, tmp_path):
        """Cover lines 462-513: comprehensive simulation runs."""
        c = SimulationConfig.from_mode(
            "comprehensive",
            output_dir=str(tmp_path),
            verbose=False,
            duration_hours=2.0,
            time_step=1.0,
        )
        r = UnifiedSimulationRunner(c)

        mock_state = MagicMock()
        mock_state.average_power = 0.5
        mock_state.coulombic_efficiency = 0.85

        mock_model = MagicMock()
        mock_model.gpu_available = True
        mock_model.step_integrated_dynamics.return_value = mock_state
        mock_model._compile_results.return_value = {
            "total_energy": 5.0,
            "average_power": 0.5,
        }

        mock_fusion = MagicMock()
        mock_fusion.KALMAN_FILTER = "kalman"

        mock_sensor_mod = MagicMock()
        mock_sensor_mod.FusionMethod = mock_fusion

        mock_sim_mod = MagicMock()
        mock_sim_mod.SensorIntegratedMFCModel = MagicMock(
            return_value=mock_model
        )

        def fake_import(name, *args, **kwargs):
            if name == "sensing_models.sensor_fusion":
                return mock_sensor_mod
            if name == "sensor_integrated_mfc_model":
                return mock_sim_mod
            return __builtins__.__import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_comprehensive()

        assert "sensor_integration" in result
        assert result["sensor_integration"]["eis_enabled"] is True

    def test_comprehensive_model_init_failure(self, tmp_path):
        """Cover lines 486-488: model init raises, falls back to _run_100h."""
        c = SimulationConfig.from_mode(
            "comprehensive",
            output_dir=str(tmp_path),
            verbose=False,
            duration_hours=0.001,
        )
        r = UnifiedSimulationRunner(c)

        mock_fusion = MagicMock()
        mock_sensor_mod = MagicMock()
        mock_sensor_mod.FusionMethod = mock_fusion

        mock_sim_mod = MagicMock()
        mock_sim_mod.SensorIntegratedMFCModel = MagicMock(
            side_effect=Exception("model fail")
        )

        def fake_import(name, *args, **kwargs):
            if name == "sensing_models.sensor_fusion":
                return mock_sensor_mod
            if name == "sensor_integrated_mfc_model":
                return mock_sim_mod
            if name in ("mfc_stack_simulation", "mfc_100h_simulation"):
                raise ImportError("no module")
            return __builtins__.__import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_comprehensive()

        # Should fall back to _run_100h which falls back to simplified
        assert isinstance(result, dict)

    def test_comprehensive_interrupted(self, tmp_path):
        """Cover comprehensive run with interruption."""
        c = SimulationConfig.from_mode(
            "comprehensive",
            output_dir=str(tmp_path),
            verbose=False,
            duration_hours=10.0,
            time_step=1.0,
        )
        r = UnifiedSimulationRunner(c)
        r._interrupted = True

        mock_state = MagicMock()
        mock_state.average_power = 0.5
        mock_state.coulombic_efficiency = 0.85

        mock_model = MagicMock()
        mock_model.gpu_available = False
        mock_model.step_integrated_dynamics.return_value = mock_state
        mock_model._compile_results.return_value = {"total_energy": 0}

        mock_fusion = MagicMock()
        mock_sensor_mod = MagicMock()
        mock_sensor_mod.FusionMethod = mock_fusion
        mock_sim_mod = MagicMock()
        mock_sim_mod.SensorIntegratedMFCModel = MagicMock(
            return_value=mock_model
        )

        def fake_import(name, *args, **kwargs):
            if name == "sensing_models.sensor_fusion":
                return mock_sensor_mod
            if name == "sensor_integrated_mfc_model":
                return mock_sim_mod
            return __builtins__.__import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_comprehensive()

        assert "sensor_integration" in result

class TestSimplifiedQLearningEdgeCases:
    """Cover edge cases in _run_simplified_qlearning."""

    def test_q_table_exploitation(self, tmp_path):
        """Cover lines 558-559: when state_key is in q_states."""
        c = SimulationConfig(
            mode="demo",
            output_dir=str(tmp_path),
            verbose=False,
            duration_hours=0.02,
            time_step=60,
            n_cells=2,
        )
        r = UnifiedSimulationRunner(c)
        # Force epsilon to 0 to always exploit (line 556)
        with patch("numpy.random.random", return_value=1.0):
            result = r._run_simplified_qlearning()
        assert "q_table_size" in result

    def test_logging_intervals(self, tmp_path):
        """Cover progress logging at exact step intervals."""
        c = SimulationConfig(
            mode="demo",
            output_dir=str(tmp_path),
            verbose=True,
            duration_hours=0.05,
            time_step=60,
            n_cells=2,
        )
        r = UnifiedSimulationRunner(c)
        result = r._run_simplified_qlearning()
        assert len(result["time_series"]["time_hours"]) > 0

class TestSaveResultsEdgeCases:
    """Cover _save_results edge paths."""

    def test_save_results_successful(self, tmp_path):
        """Cover lines 599-624: successful JSON + pickle + plots save."""
        c = SimulationConfig(output_dir=str(tmp_path), verbose=False)
        r = UnifiedSimulationRunner(c)
        r.results = {
            "time_series": {
                "time": [0, 1],
                "power": [0.1, 0.2],
            },
            "final_power": 0.2,
            "total_energy": 0.3,
            "average_power": 0.15,
        }
        os.makedirs(str(tmp_path), exist_ok=True)
        r._save_results()

        # Check JSON was written
        json_file = os.path.join(str(tmp_path), "results.json")
        assert os.path.exists(json_file)

        # Check pickle was written
        pkl_file = os.path.join(str(tmp_path), "results.pkl")
        assert os.path.exists(pkl_file)

    def test_save_results_plots_exception(self, tmp_path):
        """Cover line 624: _generate_plots raises exception."""
        c = SimulationConfig(output_dir=str(tmp_path), verbose=False)
        r = UnifiedSimulationRunner(c)
        r.results = {"test": True}
        os.makedirs(str(tmp_path), exist_ok=True)
        with patch.object(r, "_generate_plots", side_effect=Exception("plot fail")):
            r._save_results()  # Should not raise

class TestGeneratePlotsEdgeCases:
    """Cover _generate_plots edge branches."""

    def _make_mock_plt(self):
        """Create a mock plt with subplots returning (fig, 2x2 axes) tuple."""
        mock_plt = MagicMock()
        mock_fig = MagicMock()
        mock_axes = [[MagicMock() for _ in range(2)] for _ in range(2)]

        class _FakeAxes:
            def __init__(self, data):
                self._data = data
            def __getitem__(self, key):
                if isinstance(key, tuple):
                    return self._data[key[0]][key[1]]
                return self._data[key]

        mock_plt.subplots.return_value = (mock_fig, _FakeAxes(mock_axes))
        return mock_plt

    def test_plots_with_stack_power_keys(self, tmp_path):
        """Cover lines 653-656: stack_power and stack_voltage keys."""
        c = SimulationConfig(output_dir=str(tmp_path))
        r = UnifiedSimulationRunner(c)
        r.results = {
            "time_series": {
                "time_hours": [0, 1, 2],
                "stack_power": [0.1, 0.2, 0.3],
                "stack_voltage": [0.5, 0.6, 0.7],
                "total_energy": [0.1, 0.3, 0.6],
            },
            "final_power": 0.3,
            "total_energy": 0.6,
            "average_power": 0.2,
        }
        mock_plt = self._make_mock_plt()
        mock_mpl = MagicMock()
        mock_mpl.pyplot = mock_plt
        with patch.dict("sys.modules", {
            "matplotlib": mock_mpl,
            "matplotlib.pyplot": mock_plt,
        }):
            r._generate_plots()

    def test_plots_without_total_energy(self, tmp_path):
        """Cover branch where total_energy is not in time_series."""
        c = SimulationConfig(output_dir=str(tmp_path))
        r = UnifiedSimulationRunner(c)
        r.results = {
            "time_series": {
                "time": [0, 1],
                "power": [0.1, 0.2],
            },
            "final_power": 0.2,
            "total_energy": 0.3,
            "average_power": 0.15,
        }
        mock_plt = self._make_mock_plt()
        mock_mpl = MagicMock()
        mock_mpl.pyplot = mock_plt
        with patch.dict("sys.modules", {
            "matplotlib": mock_mpl,
            "matplotlib.pyplot": mock_plt,
        }):
            r._generate_plots()

    def test_plots_without_voltage(self, tmp_path):
        """Cover branch where voltage keys absent."""
        c = SimulationConfig(output_dir=str(tmp_path))
        r = UnifiedSimulationRunner(c)
        r.results = {
            "time_series": {
                "time": [0, 1],
                "power": [0.1, 0.2],
                "total_energy": [0.1, 0.3],
                "time_hours": [0, 1],
            },
            "final_power": 0.2,
            "total_energy": 0.3,
            "average_power": 0.15,
        }
        mock_plt = self._make_mock_plt()
        mock_mpl = MagicMock()
        mock_mpl.pyplot = mock_plt
        with patch.dict("sys.modules", {
            "matplotlib": mock_mpl,
            "matplotlib.pyplot": mock_plt,
        }):
            r._generate_plots()

class TestMainVerboseOutputBranches:
    """Cover main function verbose output branches."""

    def test_main_failure_returns_1(self, tmp_path):
        """Cover main returning 1 on failure."""
        with patch("sys.argv", ["prog", "demo", "-o", str(tmp_path)]):
            with patch.object(
                UnifiedSimulationRunner,
                "run",
                return_value={
                    "success": False,
                    "metadata": {"execution_time": 1.0},
                },
            ):
                result = main()
                assert result == 1

    def test_main_verbose_with_energy_and_power(self, tmp_path, capsys):
        """Cover lines 869-873: verbose output with total_energy and final_power."""
        with patch("sys.argv", ["prog", "demo", "-o", str(tmp_path)]):
            result = main()
        captured = capsys.readouterr()
        assert "Total energy" in captured.out
        assert "Final power" in captured.out

    def test_main_verbose_without_energy(self, tmp_path, capsys):
        """Cover branch where total_energy is not in results."""
        with patch("sys.argv", ["prog", "demo", "-o", str(tmp_path)]):
            with patch.object(
                UnifiedSimulationRunner,
                "run",
                return_value={
                    "success": True,
                    "metadata": {"execution_time": 0.5},
                },
            ):
                result = main()
        captured = capsys.readouterr()
        assert "Total energy" not in captured.out

    def test_main_with_all_options(self, tmp_path):
        """Cover all kwargs paths in main."""
        with patch("sys.argv", [
            "prog", "100h",
            "-c", "3",
            "-d", "0.01",
            "-t", "60",
            "-o", str(tmp_path),
            "--gpu",
            "-q",
        ]):
            result = main()
            assert result == 0

class TestRunDemoInterruptedMidLoop:
    """Cover _run_demo interrupt inside the step loop."""

    def test_demo_interrupt_midway(self, tmp_path):
        """Cover line 260-261: interrupt inside the for loop."""
        c = SimulationConfig.from_mode(
            "demo",
            output_dir=str(tmp_path),
            verbose=False,
        )
        r = UnifiedSimulationRunner(c)

        call_count = [0]
        original_clip = np.clip

        def interrupt_after_5(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 10:
                r._interrupted = True
            return original_clip(*args, **kwargs)

        with patch("numpy.clip", side_effect=interrupt_after_5):
            result = r._run_demo()

        assert len(result["time_series"]["time"]) < 360

class TestRunExtended:
    """Cover _run_extended method."""

    def test_extended_mode(self, tmp_path):
        """Cover lines 355-358."""
        c = SimulationConfig.from_mode(
            "1year",
            output_dir=str(tmp_path),
            verbose=False,
            duration_hours=0.01,
        )
        r = UnifiedSimulationRunner(c)
        result = r._run_extended()
        assert "time_series" in result

class TestPrepareForJsonNested:
    """Cover nested _prepare_for_json paths."""

    def test_nested_dict_with_arrays(self):
        c = SimulationConfig()
        r = UnifiedSimulationRunner(c)
        data = {
            "outer": {
                "inner": np.array([1, 2, 3]),
                "list": [np.int64(1), np.float64(2.0)],
                "bool": np.bool_(False),
            }
        }
        result = r._prepare_for_json(data)
        assert result["outer"]["inner"] == [1, 2, 3]
        assert result["outer"]["list"] == [1.0, 2.0]
        assert result["outer"]["bool"] is False