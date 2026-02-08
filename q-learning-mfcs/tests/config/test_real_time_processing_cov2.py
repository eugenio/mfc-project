"""Tests for config/real_time_processing.py - coverage target 98%+."""
import sys
import os
import time

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.real_time_processing import (
    DataPoint, DataQuality, StreamBuffer, MFCDataStream, StreamProcessor,
    RealTimeAnalyzer, AlertSystem, Alert, AlertLevel, StreamingStats,
    create_sample_mfc_config, create_sample_processing_config,
    create_sample_alert_config,
)


class TestDataPointConversion:
    def test_to_dict_and_from_dict(self):
        dp = DataPoint(timestamp=datetime.now(), sensor_id="s1", value=1.5,
            quality=DataQuality.GOOD, metadata={"x": 1})
        d = dp.to_dict()
        dp2 = DataPoint.from_dict(d)
        assert dp2.sensor_id == "s1"
        assert dp2.value == 1.5


class TestStreamBuffer:
    def test_get_sensor_data(self):
        buf = StreamBuffer(100)
        for i in range(10):
            buf.add(DataPoint(datetime.now(), "s1", float(i)))
        buf.add(DataPoint(datetime.now(), "s2", 99.0))
        assert len(buf.get_sensor_data("s1")) == 10


class TestStreamingStats:
    def test_update_stats(self):
        ss = StreamingStats(sensor_id="s1", window_size=100)
        for i in range(5):
            ss.update(float(i), DataQuality.GOOD)
        assert ss.count == 5
        assert ss.get_std() > 0

    def test_update_poor_quality(self):
        ss = StreamingStats(sensor_id="s1", window_size=100)
        ss.update(1.0, DataQuality.POOR)
        assert ss.quality_score < 1.0


class TestMFCDataStream:
    def test_start_stop(self):
        cfg = {"s1": {"base_value": 10.0, "noise_level": 0.1, "drift": 0, "error_rate": 0}}
        stream = MFCDataStream("test", cfg, sampling_rate=10.0)
        stream.start()
        time.sleep(0.2)
        stream.stop()
        assert stream.buffer.size() > 0

    def test_callback_notification(self):
        cfg = {"s1": {"base_value": 10.0, "noise_level": 0.1, "drift": 0, "error_rate": 0}}
        stream = MFCDataStream("test", cfg, sampling_rate=10.0)
        results = []
        stream.add_callback(lambda dp: results.append(dp))
        stream.start()
        time.sleep(0.2)
        stream.stop()
        assert len(results) > 0

    def test_read_sensor_quality_issue(self):
        cfg = {"s1": {"base_value": 10.0, "noise_level": 0.1, "drift": 0, "error_rate": 1.0}}
        stream = MFCDataStream("test", cfg, sampling_rate=10.0)
        dp = stream._read_sensor("s1", cfg["s1"])
        assert dp.quality == DataQuality.POOR


class TestStreamProcessor:
    def test_smoothing_processor(self):
        proc_cfg = {"pipeline": [{"type": "smoothing", "method": "moving_average", "window_size": 3}]}
        proc = StreamProcessor(proc_cfg)
        dps = [DataPoint(datetime.now(), "s1", float(i)) for i in range(10)]
        result = proc.process(dps)
        assert len(result) == 10

    def test_outlier_processor_iqr(self):
        proc_cfg = {"pipeline": [{"type": "outlier_detection", "method": "iqr", "threshold": 1.5}]}
        proc = StreamProcessor(proc_cfg)
        dps = [DataPoint(datetime.now(), "s1", float(i)) for i in range(20)]
        dps.append(DataPoint(datetime.now(), "s1", 1000.0))
        result = proc.process(dps)
        assert any(dp.quality == DataQuality.POOR for dp in result)

    def test_outlier_processor_zscore(self):
        proc_cfg = {"pipeline": [{"type": "outlier_detection", "method": "zscore", "threshold": 2.0}]}
        proc = StreamProcessor(proc_cfg)
        dps = [DataPoint(datetime.now(), "s1", float(i)) for i in range(20)]
        dps.append(DataPoint(datetime.now(), "s1", 1000.0))
        result = proc.process(dps)
        assert len(result) >= 20

    def test_trend_processor(self):
        proc_cfg = {"pipeline": [{"type": "trend_analysis", "window_size": 5}]}
        proc = StreamProcessor(proc_cfg)
        dps = [DataPoint(datetime.now(), "s1", float(i)) for i in range(30)]
        result = proc.process(dps)
        assert any("trend_slope" in dp.metadata for dp in result)

    def test_anomaly_processor(self):
        proc_cfg = {"pipeline": [{"type": "anomaly_detection", "method": "isolation_forest", "contamination": 0.1}]}
        proc = StreamProcessor(proc_cfg)
        dps = [DataPoint(datetime.now(), "s1", float(i)) for i in range(50)]
        result = proc.process(dps)
        assert len(result) >= 50


class TestRealTimeAnalyzer:
    def test_analyze_stream_no_data(self):
        cfg = create_sample_alert_config()
        analyzer = RealTimeAnalyzer(cfg)
        stream = MFCDataStream("test", {}, sampling_rate=1.0)
        result = analyzer.analyze_stream(stream)
        assert result["status"] == "no_data"

    def test_analyze_stream_with_data(self):
        cfg = create_sample_alert_config()
        analyzer = RealTimeAnalyzer(cfg)
        stream = MFCDataStream("test", create_sample_mfc_config(), sampling_rate=10.0)
        stream.start(); time.sleep(0.3); stream.stop()
        result = analyzer.analyze_stream(stream, timedelta(minutes=5))
        assert "sensors" in result

    def test_check_alerts_min_value(self):
        cfg = {"alert_thresholds": {"s1": {"min_value": 100.0}}}
        analyzer = RealTimeAnalyzer(cfg)
        alerts = []
        analyzer.add_alert_callback(lambda a: alerts.append(a))
        analysis = {"min": 5.0, "max": 10.0}
        dps = [DataPoint(datetime.now(), "s1", 5.0)]
        analyzer._check_alerts("s1", analysis, dps)
        assert len(alerts) >= 1

    def test_check_alerts_max_value(self):
        cfg = {"alert_thresholds": {"s1": {"max_value": 1.0}}}
        analyzer = RealTimeAnalyzer(cfg)
        alerts = []
        analyzer.add_alert_callback(lambda a: alerts.append(a))
        analysis = {"min": 5.0, "max": 10.0}
        analyzer._check_alerts("s1", analysis, [])
        assert len(alerts) >= 1

    def test_check_alerts_quality(self):
        cfg = {"alert_thresholds": {"s1": {"min_quality": 0.99}}}
        analyzer = RealTimeAnalyzer(cfg)
        alerts = []
        analyzer.add_alert_callback(lambda a: alerts.append(a))
        analysis = {"quality_score": 0.5}
        analyzer._check_alerts("s1", analysis, [])
        assert len(alerts) >= 1

    def test_check_alerts_trend(self):
        cfg = {"alert_thresholds": {"s1": {"max_trend_slope": 0.001}}}
        analyzer = RealTimeAnalyzer(cfg)
        alerts = []
        analyzer.add_alert_callback(lambda a: alerts.append(a))
        analysis = {"trend": {"slope": 1.0}}
        analyzer._check_alerts("s1", analysis, [])
        assert len(alerts) >= 1


class TestAlertSystem:
    def test_start_stop(self):
        als = AlertSystem({"max_history": 10})
        als.start()
        alert = Alert(datetime.now(), AlertLevel.WARNING, "s1", "test alert")
        als.send_alert(alert)
        time.sleep(0.3)
        als.stop()
        assert len(als.get_recent_alerts()) >= 1

    def test_critical_alert(self):
        als = AlertSystem({"max_history": 10})
        als.start()
        alert = Alert(datetime.now(), AlertLevel.CRITICAL, "s1", "critical!")
        als.send_alert(alert)
        time.sleep(0.3)
        als.stop()

    def test_error_alert(self):
        als = AlertSystem({"max_history": 10})
        als.start()
        alert = Alert(datetime.now(), AlertLevel.ERROR, "s1", "error!")
        als.send_alert(alert)
        time.sleep(0.3)
        als.stop()


class TestSampleConfigs:
    def test_create_sample_configs(self):
        assert "power_sensor" in create_sample_mfc_config()
        assert "pipeline" in create_sample_processing_config()
        assert "alert_thresholds" in create_sample_alert_config()
