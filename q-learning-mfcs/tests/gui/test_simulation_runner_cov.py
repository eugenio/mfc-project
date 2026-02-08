"""Tests for gui/simulation_runner.py - coverage target 98%+."""
import sys
import os
import queue
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gui.simulation_runner import SimulationRunner


class TestSimulationRunnerInit:
    def test_defaults(self):
        r = SimulationRunner()
        assert r.is_running is False
        assert r.should_stop is False
        assert r.enable_parquet is True
        assert r.parquet_batch_size == 100
        assert r.gui_refresh_interval == 5.0
        assert r.last_data_count == 0


class TestStartStopSimulation:
    def test_start_when_running(self):
        r = SimulationRunner()
        r.is_running = True
        assert r.start_simulation(MagicMock(), 1.0) is False

    def test_start_creates_thread(self):
        r = SimulationRunner()
        with patch("gui.simulation_runner.threading.Thread") as mt:
            mock_t = MagicMock()
            mt.return_value = mock_t
            result = r.start_simulation(MagicMock(), 1.0, n_cells=3)
            assert result is True
            assert r.is_running is True
            mock_t.start.assert_called_once()

    def test_stop_not_running(self):
        r = SimulationRunner()
        assert r.stop_simulation() is False

    def test_stop_running(self):
        r = SimulationRunner()
        r.is_running = True
        mock_t = MagicMock()
        mock_t.is_alive.return_value = False
        r.thread = mock_t
        result = r.stop_simulation()
        assert result is True
        assert r.is_running is False

    def test_stop_thread_alive_timeout(self):
        r = SimulationRunner()
        r.is_running = True
        mock_t = MagicMock()
        mock_t.is_alive.return_value = True
        r.thread = mock_t
        result = r.stop_simulation()
        assert result is False


class TestForceCleanup:
    def test_force_cleanup(self):
        r = SimulationRunner()
        r.is_running = True
        r._force_cleanup()
        assert r.is_running is False
        assert r.should_stop is False


class TestCleanupResources:
    def test_cleanup(self):
        r = SimulationRunner()
        r._cleanup_resources()


class TestGetStatus:
    def test_empty_queue(self):
        r = SimulationRunner()
        assert r.get_status() is None

    def test_has_status(self):
        r = SimulationRunner()
        r.results_queue.put(("completed", {}, None))
        status = r.get_status()
        assert status[0] == "completed"


class TestGetLiveData:
    def test_empty(self):
        r = SimulationRunner()
        data = r.get_live_data()
        assert data == []

    def test_with_data(self):
        r = SimulationRunner()
        r.data_queue.put({"time": 1.0, "value": 42})
        data = r.get_live_data()
        assert len(data) == 1
        assert len(r.live_data_buffer) == 1

    def test_buffer_truncation(self):
        r = SimulationRunner()
        r.live_data_buffer = [{"t": i} for i in range(1005)]
        r.data_queue.put({"t": 9999})
        r.get_live_data()
        assert len(r.live_data_buffer) <= 1001


class TestGetBufferedData:
    def test_empty(self):
        r = SimulationRunner()
        assert r.get_buffered_data() is None

    def test_with_buffer(self):
        r = SimulationRunner()
        r.live_data_buffer = [{"a": 1}, {"a": 2}]
        df = r.get_buffered_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2


class TestHasDataChanged:
    def test_no_change(self):
        r = SimulationRunner()
        r.last_data_count = 0
        assert r.has_data_changed() is False

    def test_change_detected(self):
        r = SimulationRunner()
        r.live_data_buffer = [{"a": 1}]
        assert r.has_data_changed() is True
        assert r.plot_dirty_flag is True
        assert r.metrics_dirty_flag is True


class TestShouldUpdatePlots:
    def test_dirty(self):
        r = SimulationRunner()
        r.plot_dirty_flag = True
        assert r.should_update_plots() is True
        assert r.plot_dirty_flag is False

    def test_clean(self):
        r = SimulationRunner()
        r.plot_dirty_flag = False
        assert r.should_update_plots() is False

    def test_force(self):
        r = SimulationRunner()
        r.plot_dirty_flag = False
        assert r.should_update_plots(force=True) is True


class TestShouldUpdateMetrics:
    def test_dirty(self):
        r = SimulationRunner()
        r.metrics_dirty_flag = True
        assert r.should_update_metrics() is True

    def test_clean(self):
        r = SimulationRunner()
        r.metrics_dirty_flag = False
        assert r.should_update_metrics() is False

    def test_force(self):
        r = SimulationRunner()
        r.metrics_dirty_flag = False
        assert r.should_update_metrics(force=True) is True


class TestGetIncrementalUpdateInfo:
    def test_info(self):
        r = SimulationRunner()
        info = r.get_incremental_update_info()
        assert "has_new_data" in info
        assert "data_count" in info
        assert "needs_plot_update" in info


class TestParquetSchema:
    def test_create_schema(self):
        r = SimulationRunner()
        sample = {"time": 1.0, "value": 42.0, "name": "test"}
        schema = r.create_parquet_schema(sample)
        assert schema is not None

    def test_create_schema_disabled(self):
        r = SimulationRunner()
        r.enable_parquet = False
        assert r.create_parquet_schema({"t": 1}) is None

    def test_create_schema_empty(self):
        r = SimulationRunner()
        assert r.create_parquet_schema(None) is None


class TestParquetWriter:
    def test_init_no_schema(self):
        r = SimulationRunner()
        assert r.init_parquet_writer("/tmp") is False

    def test_init_disabled(self):
        r = SimulationRunner()
        r.enable_parquet = False
        assert r.init_parquet_writer("/tmp") is False

    def test_init_success(self):
        r = SimulationRunner()
        sample = {"time": 1.0, "power": 0.5}
        r.create_parquet_schema(sample)
        with tempfile.TemporaryDirectory() as td:
            result = r.init_parquet_writer(td)
            assert result is True
            r.close_parquet_writer()


class TestWriteParquetBatch:
    def test_below_threshold(self):
        r = SimulationRunner()
        r.parquet_buffer = [{"t": 1}]
        assert r.write_parquet_batch() is None

    def test_disabled(self):
        r = SimulationRunner()
        r.enable_parquet = False
        assert r.write_parquet_batch() is None

    def test_no_writer(self):
        r = SimulationRunner()
        assert r.write_parquet_batch() is None


class TestCloseParquetWriter:
    def test_no_writer(self):
        r = SimulationRunner()
        assert r.close_parquet_writer() is True

    def test_with_writer_no_buffer(self):
        r = SimulationRunner()
        mock_w = MagicMock()
        r.parquet_writer = mock_w
        result = r.close_parquet_writer()
        assert result is True
        mock_w.close.assert_called_once()
        assert r.parquet_writer is None
