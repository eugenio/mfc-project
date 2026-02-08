import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import matplotlib
matplotlib.use("Agg")

import json
import pickle
import signal
import time
import argparse
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Clean stale path_config mock and force fresh import of run_simulation
# run_simulation.py imports path_config with try/except fallback
for _mod_name in ['run_simulation', 'path_config']:
    if _mod_name in sys.modules and hasattr(sys.modules[_mod_name], '_mock_name'):
        del sys.modules[_mod_name]

from run_simulation import (
    SimulationConfig,
    UnifiedSimulationRunner,
    create_argument_parser,
    list_modes,
    main,
)


class TestSimulationConfig:
    def test_default(self):
        c = SimulationConfig()
        assert c.mode == "demo"
        assert c.n_cells == 5
        assert c.duration_hours == 1.0
        assert c.verbose is True

    def test_custom(self):
        c = SimulationConfig(mode="100h", n_cells=3, duration_hours=50.0)
        assert c.mode == "100h"
        assert c.n_cells == 3

    def test_custom_output_dir(self):
        c = SimulationConfig(output_dir="/tmp/test")
        assert c.output_dir == "/tmp/test"

    def test_default_output_dir(self):
        c = SimulationConfig()
        assert "demo_simulation" in str(c.output_dir)

    def test_from_mode_demo(self):
        c = SimulationConfig.from_mode("demo")
        assert c.mode == "demo"
        assert c.duration_hours == 1.0
        assert c.time_step == 10.0

    def test_from_mode_100h(self):
        c = SimulationConfig.from_mode("100h")
        assert c.duration_hours == 100.0

    def test_from_mode_1year(self):
        c = SimulationConfig.from_mode("1year")
        assert c.duration_hours == 1000.0

    def test_from_mode_gpu(self):
        c = SimulationConfig.from_mode("gpu")
        assert c.use_gpu is True

    def test_from_mode_stack(self):
        c = SimulationConfig.from_mode("stack")
        assert c.n_cells == 5

    def test_from_mode_comprehensive(self):
        c = SimulationConfig.from_mode("comprehensive")
        assert c.enable_sensors is True

    def test_from_mode_unknown(self):
        c = SimulationConfig.from_mode("unknown")
        assert c.mode == "unknown"

    def test_from_mode_override(self):
        c = SimulationConfig.from_mode("demo", n_cells=10, duration_hours=5.0)
        assert c.n_cells == 10
        assert c.duration_hours == 5.0


class TestUnifiedSimulationRunner:
    def test_init(self):
        c = SimulationConfig(mode="demo")
        r = UnifiedSimulationRunner(c)
        assert r.start_time is None
        assert r._interrupted is False

    def test_signal_handler(self):
        c = SimulationConfig()
        r = UnifiedSimulationRunner(c)
        r._signal_handler(signal.SIGINT, None)
        assert r._interrupted is True

    def test_log_verbose(self, capsys):
        c = SimulationConfig(verbose=True)
        r = UnifiedSimulationRunner(c)
        r.log("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_log_quiet(self, capsys):
        c = SimulationConfig(verbose=False)
        r = UnifiedSimulationRunner(c)
        r.log("test message")
        captured = capsys.readouterr()
        assert "test message" not in captured.out

    def test_run_demo(self, tmp_path):
        c = SimulationConfig.from_mode("demo", output_dir=str(tmp_path), verbose=False)
        r = UnifiedSimulationRunner(c)
        results = r.run()
        assert results["success"] is True
        assert "time_series" in results
        assert "metadata" in results

    def test_run_demo_interrupted(self, tmp_path):
        c = SimulationConfig.from_mode("demo", output_dir=str(tmp_path), verbose=False)
        r = UnifiedSimulationRunner(c)
        r._interrupted = True
        results = r.run()
        assert results["metadata"]["interrupted"] is True

    def test_run_extended(self, tmp_path):
        c = SimulationConfig.from_mode("1year", output_dir=str(tmp_path), verbose=False, duration_hours=0.01)
        r = UnifiedSimulationRunner(c)
        results = r.run()
        assert results["success"] is True

    def test_run_100h_import_error(self, tmp_path):
        c = SimulationConfig.from_mode("100h", output_dir=str(tmp_path), verbose=False, duration_hours=0.001)
        r = UnifiedSimulationRunner(c)
        results = r.run()
        assert "success" in results

    def test_run_gpu_import_error(self, tmp_path):
        c = SimulationConfig.from_mode("gpu", output_dir=str(tmp_path), verbose=False, duration_hours=0.001)
        r = UnifiedSimulationRunner(c)
        results = r.run()
        assert "success" in results

    def test_run_stack_import_error(self, tmp_path):
        c = SimulationConfig.from_mode("stack", output_dir=str(tmp_path), verbose=False, duration_hours=0.001)
        r = UnifiedSimulationRunner(c)
        results = r.run()
        assert "success" in results

    def test_run_comprehensive_import_error(self, tmp_path):
        c = SimulationConfig.from_mode("comprehensive", output_dir=str(tmp_path), verbose=False, duration_hours=0.001)
        r = UnifiedSimulationRunner(c)
        results = r.run()
        assert "success" in results

    def test_run_unknown_mode(self, tmp_path):
        c = SimulationConfig(mode="unknown", output_dir=str(tmp_path), verbose=False)
        r = UnifiedSimulationRunner(c)
        results = r.run()
        assert results["success"] is True

    def test_run_failure(self, tmp_path):
        c = SimulationConfig(mode="demo", output_dir=str(tmp_path), verbose=False)
        r = UnifiedSimulationRunner(c)
        with patch.object(r, "_run_demo", side_effect=Exception("test error")):
            results = r.run()
            assert results["success"] is False
            assert "error" in results
            assert "traceback" in results

    def test_simplified_qlearning(self, tmp_path):
        c = SimulationConfig(mode="demo", output_dir=str(tmp_path), verbose=False, duration_hours=0.01, time_step=60)
        r = UnifiedSimulationRunner(c)
        result = r._run_simplified_qlearning()
        assert "time_series" in result
        assert "q_table_size" in result

    def test_simplified_qlearning_interrupted(self, tmp_path):
        c = SimulationConfig(mode="demo", output_dir=str(tmp_path), verbose=False, duration_hours=0.01, time_step=60)
        r = UnifiedSimulationRunner(c)
        r._interrupted = True
        result = r._run_simplified_qlearning()

    def test_prepare_for_json_dict(self):
        c = SimulationConfig()
        r = UnifiedSimulationRunner(c)
        assert r._prepare_for_json({"a": 1}) == {"a": 1}

    def test_prepare_for_json_list(self):
        c = SimulationConfig()
        r = UnifiedSimulationRunner(c)
        assert r._prepare_for_json([1, 2]) == [1, 2]

    def test_prepare_for_json_ndarray(self):
        c = SimulationConfig()
        r = UnifiedSimulationRunner(c)
        assert r._prepare_for_json(np.array([1, 2])) == [1, 2]

    def test_prepare_for_json_np_int(self):
        c = SimulationConfig()
        r = UnifiedSimulationRunner(c)
        assert r._prepare_for_json(np.int64(5)) == 5.0

    def test_prepare_for_json_np_float(self):
        c = SimulationConfig()
        r = UnifiedSimulationRunner(c)
        assert r._prepare_for_json(np.float64(5.5)) == 5.5

    def test_prepare_for_json_np_bool(self):
        c = SimulationConfig()
        r = UnifiedSimulationRunner(c)
        assert r._prepare_for_json(np.bool_(True)) is True

    def test_prepare_for_json_string(self):
        c = SimulationConfig()
        r = UnifiedSimulationRunner(c)
        assert r._prepare_for_json("hello") == "hello"

    def test_save_results_json_fail(self, tmp_path):
        c = SimulationConfig(output_dir=str(tmp_path), verbose=False)
        r = UnifiedSimulationRunner(c)
        r.results = {"data": object()}  # Not serializable
        r._save_results()

    def test_save_results_pkl_fail(self, tmp_path):
        c = SimulationConfig(output_dir=str(tmp_path), verbose=False)
        r = UnifiedSimulationRunner(c)
        r.results = {"test": True}
        # Patch open to fail on pkl write
        original_open = open
        def mock_open_fn(path, *a, **kw):
            if str(path).endswith(".pkl"):
                raise OSError("mock pkl failure")
            return original_open(path, *a, **kw)
        with patch("builtins.open", side_effect=mock_open_fn):
            r._save_results()

    def test_generate_plots_empty(self, tmp_path):
        c = SimulationConfig(output_dir=str(tmp_path))
        r = UnifiedSimulationRunner(c)
        r.results = {}
        r._generate_plots()

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

    def test_generate_plots_power(self, tmp_path):
        c = SimulationConfig(output_dir=str(tmp_path))
        r = UnifiedSimulationRunner(c)
        r.results = {
            "time_series": {
                "time": [0, 1, 2],
                "power": [0.1, 0.2, 0.3],
                "voltage": [0.5, 0.6, 0.7],
                "total_energy": [0.1, 0.3, 0.6],
                "time_hours": [0, 1, 2],
            },
            "final_power": 0.3,
            "total_energy": 0.6,
            "average_power": 0.2,
        }
        mock_plt = self._make_mock_plt()
        mock_mpl = MagicMock()
        mock_mpl.pyplot = mock_plt
        with patch.dict(
            "sys.modules",
            {"matplotlib": mock_mpl, "matplotlib.pyplot": mock_plt},
        ):
            r._generate_plots()

    def test_generate_plots_stack(self, tmp_path):
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
        with patch.dict(
            "sys.modules",
            {"matplotlib": mock_mpl, "matplotlib.pyplot": mock_plt},
        ):
            r._generate_plots()


class TestCreateArgumentParser:
    def test_parser(self):
        p = create_argument_parser()
        assert p is not None

    def test_parse_demo(self):
        p = create_argument_parser()
        args = p.parse_args(["demo"])
        assert args.mode == "demo"

    def test_parse_with_options(self):
        p = create_argument_parser()
        args = p.parse_args(["100h", "--cells", "3", "--duration", "50", "--timestep", "2"])
        assert args.mode == "100h"
        assert args.cells == 3
        assert args.duration == 50.0
        assert args.timestep == 2.0

    def test_parse_gpu_flag(self):
        p = create_argument_parser()
        args = p.parse_args(["demo", "--gpu"])
        assert args.gpu is True

    def test_parse_quiet(self):
        p = create_argument_parser()
        args = p.parse_args(["demo", "-q"])
        assert args.quiet is True

    def test_parse_output(self):
        p = create_argument_parser()
        args = p.parse_args(["demo", "-o", "/tmp/test"])
        assert args.output == "/tmp/test"

    def test_parse_list_modes(self):
        p = create_argument_parser()
        args = p.parse_args(["--list-modes"])
        assert args.list_modes is True


class TestListModes:
    def test_list_modes(self, capsys):
        list_modes()
        captured = capsys.readouterr()
        assert "demo" in captured.out
        assert "100h" in captured.out
        assert "gpu" in captured.out


class TestMain:
    def test_list_modes(self):
        with patch("sys.argv", ["prog", "--list-modes"]):
            result = main()
            assert result == 0

    def test_no_mode(self):
        with patch("sys.argv", ["prog"]):
            result = main()
            assert result == 1

    def test_demo(self, tmp_path):
        with patch("sys.argv", ["prog", "demo", "-q", "-o", str(tmp_path)]):
            result = main()
            assert result == 0

    def test_demo_verbose(self, tmp_path, capsys):
        with patch("sys.argv", ["prog", "demo", "-o", str(tmp_path)]):
            result = main()
            captured = capsys.readouterr()
            assert "SIMULATION COMPLETE" in captured.out

    def test_with_cells(self, tmp_path):
        with patch("sys.argv", ["prog", "demo", "-q", "-c", "3", "-o", str(tmp_path)]):
            result = main()
            assert result == 0

    def test_with_duration(self, tmp_path):
        with patch("sys.argv", ["prog", "demo", "-q", "-d", "0.5", "-o", str(tmp_path)]):
            result = main()
            assert result == 0

    def test_with_timestep(self, tmp_path):
        with patch("sys.argv", ["prog", "demo", "-q", "-t", "5", "-o", str(tmp_path)]):
            result = main()
            assert result == 0

    def test_with_gpu(self, tmp_path):
        with patch("sys.argv", ["prog", "demo", "-q", "--gpu", "-o", str(tmp_path)]):
            result = main()
            assert result == 0
