"""Coverage tests for run_simulation.py - targeting uncovered lines."""
import sys
import os
import json
import signal

import matplotlib
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from run_simulation import (
class TestConfigEdgeCases:
    def test_from_mode_all_kwargs(self):
        c = SimulationConfig.from_mode(
            "demo", n_cells=7, duration_hours=2.5, time_step=5.0,
            use_gpu=True, enable_sensors=True, output_dir="/tmp/x", verbose=False,
        )
        assert c.n_cells == 7
        assert c.use_gpu is True

    def test_default_output_dir_varies(self):
        c1 = SimulationConfig(mode="100h")
        c2 = SimulationConfig(mode="stack")
        assert "100h" in c1.output_dir
        assert "stack" in c2.output_dir

class TestRun100hPaths:
    def test_both_imports_fail(self, tmp_path):
        c = SimulationConfig.from_mode("100h", output_dir=str(tmp_path),
                                        verbose=False, duration_hours=0.001, time_step=60.0)
        r = UnifiedSimulationRunner(c)
        def fake_import(name, *a, **kw):
            if name in ("mfc_stack_simulation", "mfc_100h_simulation"):
                raise ImportError("mocked")
            return __builtins__.__import__(name, *a, **kw)
        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_100h()
        assert "time_series" in result

    def test_stack_only_fallback(self, tmp_path):
        c = SimulationConfig.from_mode("100h", output_dir=str(tmp_path),
                                        verbose=False, duration_hours=0.001, time_step=60.0)
        r = UnifiedSimulationRunner(c)
        mock_stack = MagicMock()
        mock_stack.data_log = {"stack_power": [0.1]}
        mock_stack.stack_power = 0.2
        mock_ctrl = MagicMock()
        mock_ctrl.q_table = {"s1": [0.1]}
        mock_mod = MagicMock()
        mock_mod.MFCStack = MagicMock(return_value=mock_stack)
        mock_mod.MFCStackQLearningController = MagicMock(return_value=mock_ctrl)
        def fake_import(name, *a, **kw):
            if name == "mfc_100h_simulation":
                raise ImportError("no")
            if name == "mfc_stack_simulation":
                return mock_mod
            return __builtins__.__import__(name, *a, **kw)
        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_100h()
        assert isinstance(result, dict)

    def test_both_imports_succeed(self, tmp_path):
        c = SimulationConfig.from_mode("100h", output_dir=str(tmp_path),
                                        verbose=False, duration_hours=0.001, time_step=60.0)
        r = UnifiedSimulationRunner(c)
        mock_stack = MagicMock()
        mock_stack.hourly_data = {"t": [1]}
        mock_stack.stack_power = 0.3
        mock_stack.total_energy_produced = 10.0
        mock_stack.maintenance_cycles = 2
        mock_ctrl = MagicMock()
        mock_ctrl.q_table = {"s": [0.1]}
        mock_stack_mod = MagicMock()
        mock_100h = MagicMock()
        mock_100h.LongTermMFCStack = MagicMock(return_value=mock_stack)
        mock_100h.LongTermController = MagicMock(return_value=mock_ctrl)
        def fake_import(name, *a, **kw):
            if name == "mfc_stack_simulation":
                return mock_stack_mod
            if name == "mfc_100h_simulation":
                return mock_100h
            return __builtins__.__import__(name, *a, **kw)
        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_100h()
        assert isinstance(result, dict)

class TestRunGPUPaths:
    def test_gpu_successful(self, tmp_path):
        c = SimulationConfig.from_mode("gpu", output_dir=str(tmp_path),
                                        verbose=False, duration_hours=0.3)
        r = UnifiedSimulationRunner(c)
        mock_sim = MagicMock()
        mock_sim.gpu_available = True
        mock_sim.reservoir_concentration = 25.0
        mock_sim.simulate_timestep.return_value = {"total_power": 0.5}
        mock_gpu = MagicMock()
        mock_gpu.GPUAcceleratedMFC = MagicMock(return_value=mock_sim)
        mock_cfg = MagicMock()
        mock_cfg.DEFAULT_QLEARNING_CONFIG = MagicMock()
        def fake_import(name, *a, **kw):
            if name == "mfc_gpu_accelerated":
                return mock_gpu
            if name == "config.qlearning_config":
                return mock_cfg
            return __builtins__.__import__(name, *a, **kw)
        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_gpu()
        assert result["gpu_accelerated"] is True
        mock_sim.cleanup_gpu_resources.assert_called_once()

    def test_gpu_init_failure(self, tmp_path):
        c = SimulationConfig.from_mode("gpu", output_dir=str(tmp_path),
                                        verbose=False, duration_hours=0.001)
        r = UnifiedSimulationRunner(c)
        mock_gpu = MagicMock()
        mock_gpu.GPUAcceleratedMFC = MagicMock(side_effect=Exception("fail"))
        mock_cfg = MagicMock()
        mock_cfg.DEFAULT_QLEARNING_CONFIG = MagicMock()
        def fake_import(name, *a, **kw):
            if name == "mfc_gpu_accelerated":
                return mock_gpu
            if name == "config.qlearning_config":
                return mock_cfg
            return __builtins__.__import__(name, *a, **kw)
        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_gpu()
        assert "time_series" in result

    def test_gpu_interrupted(self, tmp_path):
        c = SimulationConfig.from_mode("gpu", output_dir=str(tmp_path),
                                        verbose=False, duration_hours=1.0)
        r = UnifiedSimulationRunner(c)
        r._interrupted = True
        mock_sim = MagicMock()
        mock_sim.gpu_available = True
        mock_sim.reservoir_concentration = 25.0
        mock_sim.simulate_timestep.return_value = {"total_power": 0.5}
        mock_gpu = MagicMock()
        mock_gpu.GPUAcceleratedMFC = MagicMock(return_value=mock_sim)
        mock_cfg = MagicMock()
        mock_cfg.DEFAULT_QLEARNING_CONFIG = MagicMock()
        def fake_import(name, *a, **kw):
            if name == "mfc_gpu_accelerated":
                return mock_gpu
            if name == "config.qlearning_config":
                return mock_cfg
            return __builtins__.__import__(name, *a, **kw)
        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_gpu()
        mock_sim.cleanup_gpu_resources.assert_called_once()

class TestRunStackPaths:
    def test_stack_successful(self, tmp_path):
        c = SimulationConfig.from_mode("stack", output_dir=str(tmp_path),
                                        verbose=False, duration_hours=0.001, time_step=60.0)
        r = UnifiedSimulationRunner(c)
        mock_cell = MagicMock()
        mock_cell.get_power.return_value = 0.2
        mock_cell.is_reversed = False
        mock_stack = MagicMock()
        mock_stack.data_log = {"stack_power": [0.5]}
        mock_stack.stack_power = 0.6
        mock_stack.stack_voltage = 2.5
        mock_stack.cells = [mock_cell] * 5
        mock_stack.check_system_health.return_value = {"reversed_cells": 0}
        mock_ctrl = MagicMock()
        mock_ctrl.q_table = {"s1": [0.1]}
        mock_mod = MagicMock()
        mock_mod.MFCStack = MagicMock(return_value=mock_stack)
        mock_mod.MFCStackQLearningController = MagicMock(return_value=mock_ctrl)
        def fake_import(name, *a, **kw):
            if name == "mfc_stack_simulation":
                return mock_mod
            return __builtins__.__import__(name, *a, **kw)
        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_stack()
        assert "final_cell_states" in result
        assert len(result["final_cell_states"]) == 5

    def test_stack_interrupted(self, tmp_path):
        c = SimulationConfig.from_mode("stack", output_dir=str(tmp_path),
                                        verbose=False, duration_hours=1.0)
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
        def fake_import(name, *a, **kw):
            if name == "mfc_stack_simulation":
                return mock_mod
            return __builtins__.__import__(name, *a, **kw)
        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_stack()
        assert isinstance(result, dict)

class TestRunComprehensivePaths:
    def test_comprehensive_successful(self, tmp_path):
        c = SimulationConfig.from_mode("comprehensive", output_dir=str(tmp_path),
                                        verbose=False, duration_hours=2.0, time_step=1.0)
        r = UnifiedSimulationRunner(c)
        mock_state = MagicMock()
        mock_state.average_power = 0.5
        mock_state.coulombic_efficiency = 0.85
        mock_model = MagicMock()
        mock_model.gpu_available = True
        mock_model.step_integrated_dynamics.return_value = mock_state
        mock_model._compile_results.return_value = {"total_energy": 5.0}
        mock_fusion = MagicMock()
        mock_sensor = MagicMock()
        mock_sensor.FusionMethod = mock_fusion
        mock_sim = MagicMock()
        mock_sim.SensorIntegratedMFCModel = MagicMock(return_value=mock_model)
        def fake_import(name, *a, **kw):
            if name == "sensing_models.sensor_fusion":
                return mock_sensor
            if name == "sensor_integrated_mfc_model":
                return mock_sim
            return __builtins__.__import__(name, *a, **kw)
        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_comprehensive()
        assert "sensor_integration" in result

    def test_comprehensive_model_init_failure(self, tmp_path):
        c = SimulationConfig.from_mode("comprehensive", output_dir=str(tmp_path),
                                        verbose=False, duration_hours=0.001)
        r = UnifiedSimulationRunner(c)
        mock_sensor = MagicMock()
        mock_sensor.FusionMethod = MagicMock()
        mock_sim = MagicMock()
        mock_sim.SensorIntegratedMFCModel = MagicMock(side_effect=Exception("fail"))
        def fake_import(name, *a, **kw):
            if name == "sensing_models.sensor_fusion":
                return mock_sensor
            if name == "sensor_integrated_mfc_model":
                return mock_sim
            if name in ("mfc_stack_simulation", "mfc_100h_simulation"):
                raise ImportError("no")
            return __builtins__.__import__(name, *a, **kw)
        with patch("builtins.__import__", side_effect=fake_import):
            result = r._run_comprehensive()
        assert isinstance(result, dict)

class TestSimplifiedQLearningEdge:
    def test_logging_intervals(self, tmp_path):
        c = SimulationConfig(mode="demo", output_dir=str(tmp_path), verbose=True,
                              duration_hours=0.05, time_step=60, n_cells=2)
        r = UnifiedSimulationRunner(c)
        result = r._run_simplified_qlearning()
        assert len(result["time_series"]["time_hours"]) > 0

class TestSaveResultsEdge:
    def test_save_results_success(self, tmp_path):
        c = SimulationConfig(output_dir=str(tmp_path), verbose=False)
        r = UnifiedSimulationRunner(c)
        r.results = {"time_series": {"time": [0]}, "final_power": 0.2,
                      "total_energy": 0.3, "average_power": 0.15}
        os.makedirs(str(tmp_path), exist_ok=True)
        r._save_results()
        assert os.path.exists(os.path.join(str(tmp_path), "results.json"))
        assert os.path.exists(os.path.join(str(tmp_path), "results.pkl"))

    def test_save_results_plots_fail(self, tmp_path):
        c = SimulationConfig(output_dir=str(tmp_path), verbose=False)
        r = UnifiedSimulationRunner(c)
        r.results = {"test": True}
        os.makedirs(str(tmp_path), exist_ok=True)
        with patch.object(r, "_generate_plots", side_effect=Exception("fail")):
            r._save_results()

class TestGeneratePlotsEdge:
    def _make_mock_plt(self):
        mock_plt = MagicMock()
        mock_fig = MagicMock()
        mock_axes = [[MagicMock() for _ in range(2)] for _ in range(2)]
        class _FA:
            def __init__(self, d):
                self._d = d
            def __getitem__(self, k):
                if isinstance(k, tuple):
                    return self._d[k[0]][k[1]]
                return self._d[k]
        mock_plt.subplots.return_value = (mock_fig, _FA(mock_axes))
        return mock_plt

    def test_plots_stack_keys(self, tmp_path):
        c = SimulationConfig(output_dir=str(tmp_path))
        r = UnifiedSimulationRunner(c)
        r.results = {"time_series": {"time_hours": [0, 1], "stack_power": [0.1, 0.2],
                      "stack_voltage": [0.5, 0.6], "total_energy": [0.1, 0.3]},
                      "final_power": 0.2, "total_energy": 0.3, "average_power": 0.15}
        mp = self._make_mock_plt()
        mm = MagicMock()
        mm.pyplot = mp
        with patch.dict("sys.modules", {"matplotlib": mm, "matplotlib.pyplot": mp}):
            r._generate_plots()

    def test_plots_no_voltage(self, tmp_path):
        c = SimulationConfig(output_dir=str(tmp_path))
        r = UnifiedSimulationRunner(c)
        r.results = {"time_series": {"time": [0, 1], "power": [0.1, 0.2],
                      "total_energy": [0.1, 0.3], "time_hours": [0, 1]},
                      "final_power": 0.2, "total_energy": 0.3, "average_power": 0.15}
        mp = self._make_mock_plt()
        mm = MagicMock()
        mm.pyplot = mp
        with patch.dict("sys.modules", {"matplotlib": mm, "matplotlib.pyplot": mp}):
            r._generate_plots()

class TestMainEdge:
    def test_main_failure(self, tmp_path):
        with patch("sys.argv", ["prog", "demo", "-o", str(tmp_path)]):
            with patch.object(UnifiedSimulationRunner, "run",
                              return_value={"success": False, "metadata": {"execution_time": 1.0}}):
                assert main() == 1

    def test_main_verbose_output(self, tmp_path, capsys):
        with patch("sys.argv", ["prog", "demo", "-o", str(tmp_path)]):
            main()
        captured = capsys.readouterr()
        assert "Total energy" in captured.out

    def test_main_all_options(self, tmp_path):
        with patch("sys.argv", ["prog", "100h", "-c", "3", "-d", "0.01",
                                  "-t", "60", "-o", str(tmp_path), "--gpu", "-q"]):
            assert main() == 0

    def test_extended_mode(self, tmp_path):
        c = SimulationConfig.from_mode("1year", output_dir=str(tmp_path),
                                        verbose=False, duration_hours=0.01)
        r = UnifiedSimulationRunner(c)
        result = r._run_extended()
        assert "time_series" in result

    def test_demo_interrupt_midway(self, tmp_path):
        c = SimulationConfig.from_mode("demo", output_dir=str(tmp_path), verbose=False)
        r = UnifiedSimulationRunner(c)
        count = [0]
        orig_clip = np.clip
        def interrupt_after(a, b, cc):
            count[0] += 1
            if count[0] == 10:
                r._interrupted = True
            return orig_clip(a, b, cc)
        with patch("numpy.clip", side_effect=interrupt_after):
            result = r._run_demo()
        assert len(result["time_series"]["time"]) < 360

    def test_prepare_for_json_nested(self):
        c = SimulationConfig()
        r = UnifiedSimulationRunner(c)
        data = {"outer": {"arr": np.array([1, 2]),
                           "list": [np.int64(1)], "b": np.bool_(False)}}
        result = r._prepare_for_json(data)
        assert result["outer"]["arr"] == [1, 2]
        assert result["outer"]["b"] is False