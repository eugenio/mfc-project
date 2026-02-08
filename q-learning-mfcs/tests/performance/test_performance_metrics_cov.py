"""Tests for performance_metrics module - targeting 98%+ coverage."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Mock torch before importing performance package (gpu_memory_manager needs it)
from unittest.mock import MagicMock
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.cuda.device_count.return_value = 0
mock_torch.device.return_value = MagicMock(type="cpu")
sys.modules["torch"] = mock_torch

import pytest
import time
import threading
from datetime import datetime
from unittest.mock import patch
import numpy as np

from performance.performance_metrics import (
    MetricType,
    MetricValue,
    MetricSummary,
    MetricsCollector,
    MFCMetricsCollector,
    get_default_collector,
    start_metrics_collection,
    stop_metrics_collection,
    record_metric,
)


class TestMetricType:
    def test_enum_values(self):
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.TIMER.value == "timer"


class TestMetricValue:
    def test_defaults(self):
        mv = MetricValue(name="test", value=1.0, timestamp=datetime.now())
        assert mv.labels == {}
        assert mv.metric_type == MetricType.GAUGE

    def test_with_labels(self):
        mv = MetricValue(
            name="test", value=1.0, timestamp=datetime.now(),
            labels={"env": "prod"}, metric_type=MetricType.COUNTER
        )
        assert mv.labels == {"env": "prod"}
        assert mv.metric_type == MetricType.COUNTER


class TestMetricSummary:
    def test_dataclass(self):
        ms = MetricSummary(
            name="test", count=10, min_value=0.0, max_value=10.0,
            mean_value=5.0, std_value=2.0, percentile_50=5.0,
            percentile_95=9.0, percentile_99=9.9, last_updated=datetime.now()
        )
        assert ms.count == 10
        assert ms.mean_value == 5.0


class TestMetricsCollector:
    def test_init(self):
        mc = MetricsCollector(collection_interval=2.0, retention_hours=12)
        assert mc.collection_interval == 2.0
        assert mc.retention_hours == 12
        assert mc.running is False

    def test_record_metric(self):
        mc = MetricsCollector()
        mc.record_metric("test.metric", 42.0)
        values = mc.get_metric_values("test.metric")
        assert len(values) == 1
        assert values[0].value == 42.0

    def test_record_metric_with_timestamp(self):
        mc = MetricsCollector()
        ts = datetime(2025, 1, 1)
        mc.record_metric("test.metric", 42.0, timestamp=ts)
        values = mc.get_metric_values("test.metric")
        assert values[0].timestamp == ts

    def test_record_metric_with_labels(self):
        mc = MetricsCollector()
        mc.record_metric("test.metric", 42.0, labels={"env": "test"})
        values = mc.get_metric_values("test.metric")
        assert values[0].labels == {"env": "test"}

    def test_get_metric_values_empty(self):
        mc = MetricsCollector()
        assert mc.get_metric_values("nonexistent") == []

    def test_get_metric_values_since(self):
        mc = MetricsCollector()
        ts1 = datetime(2025, 1, 1)
        ts2 = datetime(2025, 6, 1)
        mc.record_metric("test", 1.0, timestamp=ts1)
        mc.record_metric("test", 2.0, timestamp=ts2)
        values = mc.get_metric_values("test", since=datetime(2025, 3, 1))
        assert len(values) == 1
        assert values[0].value == 2.0

    def test_get_metric_summary(self):
        mc = MetricsCollector()
        for i in range(10):
            mc.record_metric("test", float(i))
        summary = mc.get_metric_summary("test")
        assert summary is not None
        assert summary.count == 10
        assert summary.min_value == 0.0
        assert summary.max_value == 9.0

    def test_get_metric_summary_empty(self):
        mc = MetricsCollector()
        assert mc.get_metric_summary("nonexistent") is None

    def test_get_all_metric_names(self):
        mc = MetricsCollector()
        mc.record_metric("a", 1.0)
        mc.record_metric("b", 2.0)
        names = mc.get_all_metric_names()
        assert "a" in names
        assert "b" in names

    def test_clear_metrics_specific(self):
        mc = MetricsCollector()
        mc.record_metric("a", 1.0)
        mc.record_metric("b", 2.0)
        mc.clear_metrics("a")
        assert mc.get_metric_values("a") == []
        assert len(mc.get_metric_values("b")) == 1

    def test_clear_metrics_all(self):
        mc = MetricsCollector()
        mc.record_metric("a", 1.0)
        mc.record_metric("b", 2.0)
        mc.clear_metrics()
        assert mc.get_all_metric_names() == []

    def test_clear_metrics_nonexistent(self):
        mc = MetricsCollector()
        mc.clear_metrics("nonexistent")  # Should not raise

    def test_export_prometheus(self):
        mc = MetricsCollector()
        mc.record_metric("test.metric", 42.0)
        output = mc.export_metrics("prometheus")
        assert "test_metric" in output
        assert "HELP" in output
        assert "TYPE" in output

    def test_export_json(self):
        mc = MetricsCollector()
        mc.record_metric("test.metric", 42.0)
        output = mc.export_metrics("json")
        assert "test.metric" in output
        import json
        data = json.loads(output)
        assert "test.metric" in data

    def test_export_unsupported(self):
        mc = MetricsCollector()
        with pytest.raises(ValueError, match="Unsupported"):
            mc.export_metrics("xml")

    def test_start_stop_collection(self):
        mc = MetricsCollector(collection_interval=0.1)
        mc.start_collection()
        assert mc.running is True
        # Start again should be no-op
        mc.start_collection()
        time.sleep(0.3)
        mc.stop_collection()
        assert mc.running is False

    def test_register_metric_callback(self):
        mc = MetricsCollector()
        mc.register_metric_callback("custom", lambda: 99.0)
        assert "custom" in mc.metric_callbacks

    def test_register_custom_collector(self):
        mc = MetricsCollector()
        initial_count = len(mc.custom_collectors)
        mc.register_custom_collector(lambda: {"custom": 1.0})
        assert len(mc.custom_collectors) == initial_count + 1

    def test_collect_all_metrics(self):
        mc = MetricsCollector()
        mc.register_metric_callback("callback_metric", lambda: 5.0)
        mc._collect_all_metrics()
        values = mc.get_metric_values("callback_metric")
        assert len(values) == 1

    def test_collect_all_metrics_callback_error(self):
        mc = MetricsCollector()
        mc.register_metric_callback("bad_metric", lambda: (_ for _ in ()).throw(Exception("fail")))
        mc._collect_all_metrics()  # Should not raise

    def test_collect_all_metrics_callback_returns_none(self):
        mc = MetricsCollector()
        mc.register_metric_callback("none_metric", lambda: None)
        mc._collect_all_metrics()
        assert mc.get_metric_values("none_metric") == []

    def test_collect_all_metrics_custom_collector_error(self):
        mc = MetricsCollector()
        mc.custom_collectors = []
        mc.register_custom_collector(lambda: (_ for _ in ()).throw(Exception("fail")))
        mc._collect_all_metrics()  # Should not raise

    def test_collect_all_metrics_custom_returns_nondict(self):
        mc = MetricsCollector()
        mc.custom_collectors = []
        mc.register_custom_collector(lambda: "not a dict")
        mc._collect_all_metrics()  # Should not raise

    def test_collect_all_metrics_custom_none_value(self):
        mc = MetricsCollector()
        mc.custom_collectors = []
        mc.register_custom_collector(lambda: {"good": 1.0, "bad": None})
        mc._collect_all_metrics()
        assert len(mc.get_metric_values("good")) == 1
        assert mc.get_metric_values("bad") == []

    def test_collect_system_metrics(self):
        mc = MetricsCollector()
        result = mc._collect_system_metrics()
        assert isinstance(result, dict)
        assert "system.cpu.utilization" in result

    def test_collect_process_metrics(self):
        mc = MetricsCollector()
        result = mc._collect_process_metrics()
        assert isinstance(result, dict)
        assert "process.memory.rss_mb" in result

    def test_collection_loop_runs(self):
        mc = MetricsCollector(collection_interval=0.05)
        mc.start_collection()
        time.sleep(0.15)
        mc.stop_collection()
        # Should have collected some system metrics
        names = mc.get_all_metric_names()
        assert len(names) > 0

    def test_collection_loop_handles_exception(self):
        mc = MetricsCollector(collection_interval=0.05)
        # Monkeypatch _collect_all_metrics to raise
        original = mc._collect_all_metrics
        call_count = [0]
        def failing_collect():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError("collection failure")
            # After 2 failures, stop looping
            mc.running = False
        mc._collect_all_metrics = failing_collect
        mc.start_collection()
        time.sleep(0.3)
        mc.stop_collection()
        # Should have handled exception without crashing
        assert call_count[0] >= 1

    def test_collect_system_metrics_exception(self):
        mc = MetricsCollector()
        with patch("performance.performance_metrics.psutil") as mock_psutil:
            mock_psutil.cpu_percent.side_effect = RuntimeError("system fail")
            result = mc._collect_system_metrics()
            assert result == {}

    def test_collect_process_metrics_exception(self):
        mc = MetricsCollector()
        with patch("performance.performance_metrics.psutil") as mock_psutil:
            mock_psutil.Process.side_effect = RuntimeError("process fail")
            result = mc._collect_process_metrics()
            assert result == {}


class TestMFCMetricsCollector:
    def test_init(self):
        mmc = MFCMetricsCollector()
        assert len(mmc.custom_collectors) > 2  # default + MFC + qlearning

    def test_collect_mfc_simulation_metrics(self):
        mmc = MFCMetricsCollector()
        result = mmc._collect_mfc_simulation_metrics()
        assert "mfc.simulation.power_output_mw" in result

    def test_collect_qlearning_metrics(self):
        mmc = MFCMetricsCollector()
        result = mmc._collect_qlearning_metrics()
        assert "qlearning.epsilon_value" in result

    def test_record_simulation_step(self):
        mmc = MFCMetricsCollector()
        mmc.record_simulation_step(1.5, 80.0, 0.5, 25.0)
        assert len(mmc.get_metric_values("mfc.simulation.power_output_mw")) == 1

    def test_record_qlearning_episode(self):
        mmc = MFCMetricsCollector()
        mmc.record_qlearning_episode(100, 0.1, 50.0, 0.95)
        assert len(mmc.get_metric_values("qlearning.episodes_completed")) == 1


class TestModuleFunctions:
    def test_get_default_collector(self):
        import performance.performance_metrics as pm
        pm._default_collector = None
        collector = get_default_collector()
        assert isinstance(collector, MFCMetricsCollector)
        # Second call returns same instance
        collector2 = get_default_collector()
        assert collector is collector2

    def test_start_stop_metrics_collection(self):
        import performance.performance_metrics as pm
        pm._default_collector = None
        start_metrics_collection()
        collector = get_default_collector()
        assert collector.running is True
        stop_metrics_collection()
        assert collector.running is False

    def test_record_metric_function(self):
        import performance.performance_metrics as pm
        pm._default_collector = None
        record_metric("test.func", 123.0)
        collector = get_default_collector()
        values = collector.get_metric_values("test.func")
        assert len(values) == 1
        assert values[0].value == 123.0
