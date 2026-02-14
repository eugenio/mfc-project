"""Coverage tests for simulation_runner.py - lines 135-491."""
import os
import queue
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'gui'),
)

from simulation_runner import SimulationRunner


@pytest.mark.coverage_extra
class TestStopSimulation:
    def test_not_running(self):
        sr = SimulationRunner()
        assert sr.stop_simulation() is False

    def test_normal_stop(self):
        sr = SimulationRunner()
        sr.is_running = True
        sr.thread = MagicMock()
        sr.thread.is_alive.return_value = False
        sr.current_output_dir = "/tmp"
        result = sr.stop_simulation()
        assert result is True
        assert not sr.is_running

    def test_force_stop(self):
        sr = SimulationRunner()
        sr.is_running = True
        sr.thread = MagicMock()
        sr.thread.is_alive.return_value = True
        sr.current_output_dir = "/tmp"
        result = sr.stop_simulation()
        assert result is False


@pytest.mark.coverage_extra
class TestForceCleanup:
    def test_cleanup(self):
        sr = SimulationRunner()
        sr.is_running = True
        sr.should_stop = True
        sr._force_cleanup()
        assert not sr.is_running
        assert not sr.should_stop


@pytest.mark.coverage_extra
class TestCleanupResources:
    def test_no_deps(self):
        sr = SimulationRunner()
        with patch.dict(sys.modules, {'jax': None, 'torch': None}):
            sr._cleanup_resources()

    def test_with_mock_deps(self):
        sr = SimulationRunner()
        mock_jax = MagicMock()
        mock_jax.clear_backends = MagicMock()
        mock_jax.clear_caches = MagicMock()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {'jax': mock_jax, 'torch': mock_torch}):
            sr._cleanup_resources()


@pytest.mark.coverage_extra
class TestGetStatus:
    def test_empty_queue(self):
        sr = SimulationRunner()
        assert sr.get_status() is None

    def test_with_status(self):
        sr = SimulationRunner()
        sr.results_queue.put(("completed", {}, "/tmp"))
        result = sr.get_status()
        assert result[0] == "completed"


@pytest.mark.coverage_extra
class TestGetLiveData:
    def test_empty(self):
        sr = SimulationRunner()
        data = sr.get_live_data()
        assert data == []

    def test_with_data(self):
        sr = SimulationRunner()
        sr.data_queue.put({"time_hours": 1.0, "power": 0.5})
        sr.data_queue.put({"time_hours": 2.0, "power": 0.6})
        data = sr.get_live_data()
        assert len(data) == 2
        assert len(sr.live_data_buffer) == 2

    def test_buffer_trim(self):
        sr = SimulationRunner()
        sr.live_data_buffer = [{}] * 999
        sr.data_queue.put({"time_hours": 1.0})
        sr.data_queue.put({"time_hours": 2.0})
        sr.get_live_data()
        assert len(sr.live_data_buffer) == 1000


@pytest.mark.coverage_extra
class TestGetBufferedData:
    def test_empty(self):
        sr = SimulationRunner()
        assert sr.get_buffered_data() is None

    def test_with_data(self):
        sr = SimulationRunner()
        sr.live_data_buffer = [
            {"time_hours": 1.0, "power": 0.5},
            {"time_hours": 2.0, "power": 0.6},
        ]
        df = sr.get_buffered_data()
        assert df is not None
        assert len(df) == 2


@pytest.mark.coverage_extra
class TestHasDataChanged:
    def test_no_change(self):
        sr = SimulationRunner()
        sr.last_data_count = 0
        assert sr.has_data_changed() is False

    def test_with_change(self):
        sr = SimulationRunner()
        sr.live_data_buffer = [{}]
        sr.last_data_count = 0
        assert sr.has_data_changed() is True
        assert sr.plot_dirty_flag is True
        assert sr.metrics_dirty_flag is True


@pytest.mark.coverage_extra
class TestShouldUpdatePlots:
    def test_not_dirty(self):
        sr = SimulationRunner()
        sr.plot_dirty_flag = False
        assert sr.should_update_plots() is False

    def test_dirty(self):
        sr = SimulationRunner()
        sr.plot_dirty_flag = True
        assert sr.should_update_plots() is True
        assert sr.plot_dirty_flag is False

    def test_force(self):
        sr = SimulationRunner()
        sr.plot_dirty_flag = False
        assert sr.should_update_plots(force=True) is True


@pytest.mark.coverage_extra
class TestShouldUpdateMetrics:
    def test_not_dirty(self):
        sr = SimulationRunner()
        sr.metrics_dirty_flag = False
        assert sr.should_update_metrics() is False

    def test_dirty(self):
        sr = SimulationRunner()
        sr.metrics_dirty_flag = True
        assert sr.should_update_metrics() is True

    def test_force(self):
        sr = SimulationRunner()
        sr.metrics_dirty_flag = False
        assert sr.should_update_metrics(force=True) is True


@pytest.mark.coverage_extra
class TestGetIncrementalUpdateInfo:
    def test_info(self):
        sr = SimulationRunner()
        info = sr.get_incremental_update_info()
        assert "has_new_data" in info
        assert "data_count" in info
        assert "needs_plot_update" in info
        assert "needs_metrics_update" in info


@pytest.mark.coverage_extra
class TestCreateParquetSchema:
    def test_disabled(self):
        sr = SimulationRunner()
        sr.enable_parquet = False
        assert sr.create_parquet_schema({}) is None

    def test_empty(self):
        sr = SimulationRunner()
        assert sr.create_parquet_schema({}) is None

    def test_valid(self):
        sr = SimulationRunner()
        sample = {"time_hours": 1.0, "power": 0.5, "action": 1}
        result = sr.create_parquet_schema(sample)
        assert result is not None
        assert sr.parquet_schema is not None

    def test_exception(self):
        sr = SimulationRunner()
        with patch('pandas.DataFrame', side_effect=Exception("err")):
            result = sr.create_parquet_schema({"x": 1})
        assert result is None
        assert sr.enable_parquet is False


@pytest.mark.coverage_extra
class TestInitParquetWriter:
    def test_disabled(self):
        sr = SimulationRunner()
        sr.enable_parquet = False
        assert sr.init_parquet_writer("/tmp") is False

    def test_no_schema(self):
        sr = SimulationRunner()
        sr.parquet_schema = None
        assert sr.init_parquet_writer("/tmp") is False

    def test_success(self, tmp_path):
        sr = SimulationRunner()
        sample = {"time_hours": 1.0, "power": 0.5}
        sr.create_parquet_schema(sample)
        result = sr.init_parquet_writer(tmp_path)
        assert result is True
        sr.close_parquet_writer()

    def test_exception(self):
        sr = SimulationRunner()
        sr.parquet_schema = MagicMock()
        with patch('pyarrow.parquet.ParquetWriter', side_effect=Exception("e")):
            result = sr.init_parquet_writer("/nonexistent")
        assert result is False
        assert sr.enable_parquet is False


@pytest.mark.coverage_extra
class TestWriteParquetBatch:
    def test_disabled(self):
        sr = SimulationRunner()
        sr.enable_parquet = False
        assert sr.write_parquet_batch() is None

    def test_below_threshold(self):
        sr = SimulationRunner()
        sr.parquet_writer = MagicMock()
        sr.parquet_buffer = [{}] * 50
        assert sr.write_parquet_batch() is None

    def test_success(self, tmp_path):
        sr = SimulationRunner()
        sample = {"time_hours": 1.0, "power": 0.5}
        sr.create_parquet_schema(sample)
        sr.init_parquet_writer(tmp_path)
        sr.parquet_buffer = [sample.copy() for _ in range(100)]
        result = sr.write_parquet_batch()
        assert result is True
        assert len(sr.parquet_buffer) == 0
        sr.close_parquet_writer()


@pytest.mark.coverage_extra
class TestCloseParquetWriter:
    def test_no_writer(self):
        sr = SimulationRunner()
        assert sr.close_parquet_writer() is True

    def test_with_writer(self, tmp_path):
        sr = SimulationRunner()
        sample = {"time_hours": 1.0, "power": 0.5}
        sr.create_parquet_schema(sample)
        sr.init_parquet_writer(tmp_path)
        sr.parquet_buffer = [sample.copy()]
        result = sr.close_parquet_writer()
        assert result is True
        assert sr.parquet_writer is None

    def test_exception(self):
        sr = SimulationRunner()
        sr.parquet_writer = MagicMock()
        sr.parquet_writer.close.side_effect = Exception("err")
        sr.parquet_buffer = []
        sr.parquet_schema = MagicMock()
        result = sr.close_parquet_writer()
        assert result is False


@pytest.mark.coverage_extra
class TestStartSimulation:
    def test_already_running(self):
        sr = SimulationRunner()
        sr.is_running = True
        assert sr.start_simulation(MagicMock(), 1.0) is False

    def test_start(self):
        sr = SimulationRunner()
        with patch.object(sr, '_run_simulation'):
            result = sr.start_simulation(MagicMock(), 1.0)
        assert result is True
        assert sr.is_running is True
        sr.should_stop = True
        if sr.thread:
            sr.thread.join(timeout=2)
