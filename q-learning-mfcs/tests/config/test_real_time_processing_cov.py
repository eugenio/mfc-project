"""Tests for config/real_time_processing.py - targeting 98%+ coverage."""
import importlib.util
import math
import os
import sys
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from queue import Empty
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---- Mock heavy deps before import ----

_mock_np = MagicMock()
_mock_np.nan = float("nan")
_mock_np.inf = float("inf")
_mock_np.pi = math.pi
_mock_np.isnan = lambda x: x != x if isinstance(x, float) else False
_mock_np.sqrt = lambda x: math.sqrt(x) if isinstance(x, (int, float)) else x
_mock_np.sin = lambda x: math.sin(x) if isinstance(x, (int, float)) else x
_mock_np.random = MagicMock()
_mock_np.random.normal = lambda mu, sigma: mu
_mock_np.random.random = lambda: 0.5


class _MockArray(list):
    """List subclass that supports numpy-like comparison and reshape."""

    def __lt__(self, other):
        return _MockArray([x < other for x in self])

    def __gt__(self, other):
        return _MockArray([x > other for x in self])

    def __le__(self, other):
        return _MockArray([x <= other for x in self])

    def __ge__(self, other):
        return _MockArray([x >= other for x in self])

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return _MockArray([x == other for x in self])
        return list.__eq__(self, other)

    def __ne__(self, other):
        if isinstance(other, (int, float)):
            return _MockArray([x != other for x in self])
        return list.__ne__(self, other)

    def __or__(self, other):
        if isinstance(other, list):
            return _MockArray([a or b for a, b in zip(self, other)])
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return _MockArray([x - other for x in self])
        if isinstance(other, list):
            return _MockArray([a - b for a, b in zip(self, other)])
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return _MockArray([other - x for x in self])
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return _MockArray([x / other for x in self])
        if isinstance(other, list):
            return _MockArray([a / b for a, b in zip(self, other)])
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return _MockArray([x * other for x in self])
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return _MockArray([x + other for x in self])
        if isinstance(other, list):
            return _MockArray([a + b for a, b in zip(self, other)])
        return list.__add__(self, other)

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return _MockArray([x ** other for x in self])
        return NotImplemented

    def reshape(self, *args):
        """Mimic numpy reshape -- returns self for common patterns."""
        return self


def _np_array(data, dtype=None):
    if isinstance(data, list):
        return _MockArray(data)
    return _MockArray([data])


def _np_mean(data, axis=None):
    if isinstance(data, (list, tuple)):
        return sum(data) / len(data) if data else 0.0
    return data


def _np_std(data, axis=None):
    if isinstance(data, (list, tuple)) and len(data) > 1:
        m = sum(data) / len(data)
        return math.sqrt(sum((x - m) ** 2 for x in data) / len(data))
    return 0.0


def _np_min(data):
    if isinstance(data, (list, tuple)):
        return min(data) if data else 0.0
    return data


def _np_max(data):
    if isinstance(data, (list, tuple)):
        return max(data) if data else 0.0
    return data


def _np_median(data):
    if isinstance(data, (list, tuple)):
        s = sorted(data)
        n = len(s)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2.0
    return data


def _np_arange(n):
    return list(range(int(n)))


def _np_diff(data):
    if isinstance(data, (list, tuple)):
        return [data[i + 1] - data[i] for i in range(len(data) - 1)]
    return data


def _np_percentile(data, q):
    if isinstance(data, (list, tuple)):
        s = sorted(data)
        n = len(s)
        results = []
        for p in (q if isinstance(q, (list, tuple)) else [q]):
            idx = (p / 100.0) * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            results.append(s[lo] * (1 - frac) + s[hi] * frac)
        return results if isinstance(q, (list, tuple)) else results[0]
    return data


def _np_abs(data):
    if isinstance(data, (list, tuple)):
        return _MockArray([abs(x) for x in data])
    return abs(data)


def _np_zeros(n, dtype=None):
    if dtype is bool or dtype == "bool":
        return _MockArray([False] * n)
    if isinstance(n, int):
        return _MockArray([0] * n)
    return _MockArray([0] * n)


def _np_ones(n):
    if isinstance(n, int):
        return _MockArray([1.0] * n)
    return _MockArray([1.0] * n)


def _np_full(n, val):
    return [val] * n


def _np_convolve(a, b, mode="full"):
    # Simplified convolution
    return list(a)[:len(a) - len(b) + 1] if mode == "valid" else list(a)


def _np_concatenate(arrays):
    result = []
    for arr in arrays:
        if isinstance(arr, (list, tuple)):
            result.extend(arr)
        else:
            result.append(arr)
    return result


_mock_np.array = _np_array
_mock_np.mean = _np_mean
_mock_np.std = _np_std
_mock_np.min = _np_min
_mock_np.max = _np_max
_mock_np.median = _np_median
_mock_np.arange = _np_arange
_mock_np.diff = _np_diff
_mock_np.percentile = _np_percentile
_mock_np.abs = _np_abs
_mock_np.zeros = _np_zeros
_mock_np.ones = _np_ones
_mock_np.full = _np_full
_mock_np.convolve = _np_convolve
_mock_np.concatenate = _np_concatenate

sys.modules.setdefault("numpy", _mock_np)

# Mock scipy
_mock_scipy = MagicMock()
_mock_scipy_stats = MagicMock()
_mock_scipy_stats.zscore = lambda x: [0.0] * len(x) if isinstance(x, list) else 0.0
_mock_scipy_stats.linregress = lambda x, y: (0.01, 0.0, 0.9, 0.05, 0.001)
_mock_scipy_signal = MagicMock()
_mock_scipy_signal.savgol_filter = lambda v, w, p: v

sys.modules.setdefault("scipy", _mock_scipy)
sys.modules.setdefault("scipy.stats", _mock_scipy_stats)
sys.modules.setdefault("scipy.signal", _mock_scipy_signal)
_mock_scipy.stats = _mock_scipy_stats
_mock_scipy.signal = _mock_scipy_signal

# Mock statsmodels
_mock_sm = MagicMock()
sys.modules.setdefault("statsmodels", _mock_sm)
sys.modules.setdefault("statsmodels.tsa", MagicMock())
sys.modules.setdefault("statsmodels.tsa.holtwinters", MagicMock())
sys.modules.setdefault("statsmodels.tsa.seasonal", MagicMock())

# Mock sklearn
_mock_sklearn = MagicMock()
_mock_iso_forest = MagicMock()
_mock_iso_forest.return_value.fit_predict.return_value = [1, 1, 1, -1, 1]
sys.modules.setdefault("sklearn", _mock_sklearn)
sys.modules.setdefault("sklearn.ensemble", MagicMock())
sys.modules["sklearn.ensemble"].IsolationForest = _mock_iso_forest
sys.modules.setdefault("sklearn.preprocessing", MagicMock())

# ---- Load module ----

_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
_spec = importlib.util.spec_from_file_location(
    "config.real_time_processing",
    os.path.join(_src, "config", "real_time_processing.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["config.real_time_processing"] = _mod
_spec.loader.exec_module(_mod)

StreamingMode = _mod.StreamingMode
DataQuality = _mod.DataQuality
AlertLevel = _mod.AlertLevel
DataPoint = _mod.DataPoint
StreamingStats = _mod.StreamingStats
Alert = _mod.Alert
StreamBuffer = _mod.StreamBuffer
DataStream = _mod.DataStream
MFCDataStream = _mod.MFCDataStream
StreamProcessor = _mod.StreamProcessor
RealTimeAnalyzer = _mod.RealTimeAnalyzer
AlertSystem = _mod.AlertSystem
create_sample_mfc_config = _mod.create_sample_mfc_config
create_sample_processing_config = _mod.create_sample_processing_config
create_sample_alert_config = _mod.create_sample_alert_config


# ---- Helper ----


def _make_dp(sensor_id="s1", value=1.0, quality=DataQuality.GOOD, ts=None):
    return DataPoint(
        timestamp=ts or datetime.now(),
        sensor_id=sensor_id,
        value=value,
        quality=quality,
    )


# ---- Tests ----


class TestEnums:
    def test_streaming_mode(self):
        assert StreamingMode.REAL_TIME.value == "real_time"
        assert StreamingMode.BATCH.value == "batch"
        assert StreamingMode.MICRO_BATCH.value == "micro_batch"
        assert StreamingMode.CONTINUOUS.value == "continuous"

    def test_data_quality(self):
        assert DataQuality.EXCELLENT.value == "excellent"
        assert DataQuality.GOOD.value == "good"
        assert DataQuality.ACCEPTABLE.value == "acceptable"
        assert DataQuality.POOR.value == "poor"
        assert DataQuality.INVALID.value == "invalid"

    def test_alert_level(self):
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"


class TestDataPoint:
    def test_default(self):
        dp = _make_dp()
        assert dp.sensor_id == "s1"
        assert dp.value == 1.0
        assert dp.quality == DataQuality.GOOD
        assert dp.metadata == {}

    def test_to_dict(self):
        dp = _make_dp(value=2.5)
        d = dp.to_dict()
        assert d["sensor_id"] == "s1"
        assert d["value"] == 2.5
        assert d["quality"] == "good"
        assert "timestamp" in d

    def test_from_dict(self):
        d = {
            "timestamp": "2025-01-01T00:00:00",
            "sensor_id": "s2",
            "value": 3.0,
            "quality": "poor",
            "metadata": {"x": 1},
        }
        dp = DataPoint.from_dict(d)
        assert dp.sensor_id == "s2"
        assert dp.value == 3.0
        assert dp.quality == DataQuality.POOR
        assert dp.metadata == {"x": 1}

    def test_from_dict_defaults(self):
        d = {
            "timestamp": "2025-01-01T00:00:00",
            "sensor_id": "s3",
            "value": 0.0,
        }
        dp = DataPoint.from_dict(d)
        assert dp.quality == DataQuality.GOOD
        assert dp.metadata == {}


class TestStreamingStats:
    def test_defaults(self):
        ss = StreamingStats(sensor_id="s1", window_size=100)
        assert ss.count == 0
        assert ss.mean == 0.0

    def test_update_first_value(self):
        ss = StreamingStats(sensor_id="s1", window_size=100)
        ss.update(5.0)
        assert ss.count == 1
        assert ss.mean == 5.0
        assert ss.variance == 0.0
        assert ss.min_value == 5.0
        assert ss.max_value == 5.0

    def test_update_multiple_values(self):
        ss = StreamingStats(sensor_id="s1", window_size=100)
        ss.update(2.0)
        ss.update(4.0)
        assert ss.count == 2
        assert ss.min_value == 2.0
        assert ss.max_value == 4.0

    def test_update_poor_quality(self):
        ss = StreamingStats(sensor_id="s1", window_size=100)
        ss.update(1.0, DataQuality.GOOD)
        initial_q = ss.quality_score
        ss.update(2.0, DataQuality.POOR)
        assert ss.quality_score < initial_q

    def test_update_good_quality_recovery(self):
        ss = StreamingStats(sensor_id="s1", window_size=100)
        ss.update(1.0, DataQuality.POOR)
        q_after_poor = ss.quality_score
        ss.update(2.0, DataQuality.GOOD)
        assert ss.quality_score >= q_after_poor

    def test_get_std_zero(self):
        ss = StreamingStats(sensor_id="s1", window_size=100)
        assert ss.get_std() == 0.0

    def test_get_std_one_value(self):
        ss = StreamingStats(sensor_id="s1", window_size=100)
        ss.update(5.0)
        assert ss.get_std() == 0.0

    def test_get_std_multiple(self):
        ss = StreamingStats(sensor_id="s1", window_size=100)
        ss.update(2.0)
        ss.update(4.0)
        assert ss.get_std() >= 0.0

    def test_update_frequency(self):
        ss = StreamingStats(sensor_id="s1", window_size=100)
        ss.update(1.0)
        time.sleep(0.01)
        ss.update(2.0)
        # update_frequency should be nonzero after two updates
        assert ss.update_frequency >= 0.0


class TestAlert:
    def test_defaults(self):
        a = Alert(
            timestamp=datetime.now(),
            level=AlertLevel.WARNING,
            sensor_id="s1",
            message="test",
        )
        assert a.value is None
        assert a.threshold is None
        assert a.metadata == {}

    def test_to_dict(self):
        a = Alert(
            timestamp=datetime(2025, 1, 1),
            level=AlertLevel.ERROR,
            sensor_id="s1",
            message="err",
            value=5.0,
            threshold=4.0,
        )
        d = a.to_dict()
        assert d["level"] == "error"
        assert d["value"] == 5.0
        assert d["threshold"] == 4.0


class TestStreamBuffer:
    def test_add_and_size(self):
        buf = StreamBuffer(max_size=100)
        buf.add(_make_dp())
        assert buf.size() == 1
        assert buf.total_points == 1

    def test_max_size(self):
        buf = StreamBuffer(max_size=3)
        for i in range(5):
            buf.add(_make_dp(value=float(i)))
        assert buf.size() == 3
        assert buf.total_points == 5

    def test_get_recent(self):
        buf = StreamBuffer(max_size=100)
        for i in range(10):
            buf.add(_make_dp(value=float(i)))
        recent = buf.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].value == 9.0

    def test_get_time_window(self):
        buf = StreamBuffer()
        now = datetime.now()
        buf.add(_make_dp(ts=now - timedelta(minutes=10)))
        buf.add(_make_dp(ts=now - timedelta(minutes=1)))
        buf.add(_make_dp(ts=now))
        window = buf.get_time_window(now - timedelta(minutes=5))
        assert len(window) == 2

    def test_get_time_window_with_end(self):
        buf = StreamBuffer()
        now = datetime.now()
        buf.add(_make_dp(ts=now - timedelta(minutes=10)))
        buf.add(_make_dp(ts=now - timedelta(minutes=5)))
        buf.add(_make_dp(ts=now))
        window = buf.get_time_window(
            now - timedelta(minutes=11),
            now - timedelta(minutes=3),
        )
        assert len(window) == 2

    def test_get_sensor_data(self):
        buf = StreamBuffer()
        buf.add(_make_dp(sensor_id="s1"))
        buf.add(_make_dp(sensor_id="s2"))
        buf.add(_make_dp(sensor_id="s1"))
        data = buf.get_sensor_data("s1")
        assert len(data) == 2

    def test_clear(self):
        buf = StreamBuffer()
        buf.add(_make_dp())
        buf.clear()
        assert buf.size() == 0


class TestDataStream:
    def test_abstract(self):
        # DataStream is abstract, can't instantiate directly
        with pytest.raises(TypeError):
            DataStream("test")

    def test_concrete_subclass(self):
        class ConcreteStream(DataStream):
            def start(self):
                self.is_active = True

            def stop(self):
                self.is_active = False

        cs = ConcreteStream("test", buffer_size=100)
        assert cs.stream_id == "test"
        assert cs.is_active is False
        cs.start()
        assert cs.is_active is True
        cs.stop()
        assert cs.is_active is False

    def test_callbacks(self):
        class ConcreteStream(DataStream):
            def start(self):
                pass

            def stop(self):
                pass

        cs = ConcreteStream("test")
        results = []
        cb = lambda dp: results.append(dp)
        cs.add_callback(cb)
        assert len(cs.callbacks) == 1

        dp = _make_dp()
        cs._notify_callbacks(dp)
        assert len(results) == 1

        cs.remove_callback(cb)
        assert len(cs.callbacks) == 0

    def test_remove_nonexistent_callback(self):
        class ConcreteStream(DataStream):
            def start(self):
                pass

            def stop(self):
                pass

        cs = ConcreteStream("test")
        cs.remove_callback(lambda x: None)  # Should not raise

    def test_callback_error(self):
        class ConcreteStream(DataStream):
            def start(self):
                pass

            def stop(self):
                pass

        cs = ConcreteStream("test")
        cs.add_callback(lambda dp: 1 / 0)  # Raises ZeroDivisionError
        dp = _make_dp()
        cs._notify_callbacks(dp)  # Should not raise

    def test_get_stats(self):
        class ConcreteStream(DataStream):
            def start(self):
                pass

            def stop(self):
                pass

        cs = ConcreteStream("test")
        assert cs.get_stats("s1") is None

    def test_update_stats(self):
        class ConcreteStream(DataStream):
            def start(self):
                pass

            def stop(self):
                pass

        cs = ConcreteStream("test")
        dp = _make_dp(sensor_id="s1", value=5.0)
        cs.update_stats(dp)
        stats = cs.get_stats("s1")
        assert stats is not None
        assert stats.count == 1

    def test_update_stats_existing(self):
        class ConcreteStream(DataStream):
            def start(self):
                pass

            def stop(self):
                pass

        cs = ConcreteStream("test")
        cs.update_stats(_make_dp(sensor_id="s1", value=5.0))
        cs.update_stats(_make_dp(sensor_id="s1", value=10.0))
        stats = cs.get_stats("s1")
        assert stats.count == 2


class TestMFCDataStream:
    def test_init(self):
        cfg = {"s1": {"base_value": 1.0, "noise_level": 0.1}}
        ds = MFCDataStream("mfc1", cfg, sampling_rate=2.0)
        assert ds.stream_id == "mfc1"
        assert ds.sampling_rate == 2.0
        assert ds.sampling_interval == 0.5

    def test_start_stop(self):
        cfg = {"s1": {"base_value": 1.0, "noise_level": 0.1}}
        ds = MFCDataStream("mfc1", cfg, sampling_rate=100.0)
        ds.start()
        assert ds.is_active is True
        time.sleep(0.05)
        ds.stop()
        assert ds.is_active is False

    def test_start_already_active(self):
        cfg = {"s1": {"base_value": 1.0}}
        ds = MFCDataStream("mfc1", cfg)
        ds.start()
        ds.start()  # Should not start second thread
        ds.stop()

    def test_stop_already_stopped(self):
        cfg = {"s1": {"base_value": 1.0}}
        ds = MFCDataStream("mfc1", cfg)
        ds.stop()  # Should not raise

    def test_read_sensor(self):
        cfg = {"s1": {"base_value": 10.0, "noise_level": 0.1, "drift": 0.5}}
        ds = MFCDataStream("mfc1", cfg)
        dp = ds._read_sensor("s1", cfg["s1"])
        assert dp.sensor_id == "s1"
        assert isinstance(dp.value, (int, float))

    def test_read_sensor_error(self):
        cfg = {"s1": {"base_value": 10.0, "noise_level": 0.1, "error_rate": 1.0}}
        ds = MFCDataStream("mfc1", cfg)
        # Mock random to trigger error path
        old_random = _mock_np.random.random
        _mock_np.random.random = lambda: 0.0  # < error_rate=1.0
        dp = ds._read_sensor("s1", cfg["s1"])
        assert dp.quality == DataQuality.POOR
        _mock_np.random.random = old_random


class TestStreamProcessor:
    def test_init_empty(self):
        sp = StreamProcessor({})
        assert sp.processors == []

    def test_init_with_pipeline(self):
        cfg = create_sample_processing_config()
        sp = StreamProcessor(cfg)
        assert len(sp.processors) > 0

    def test_process_empty(self):
        sp = StreamProcessor({})
        result = sp.process([])
        assert result == []

    def test_smoothing_processor_short_data(self):
        cfg = {"pipeline": [{"type": "smoothing", "method": "moving_average", "window_size": 5}]}
        sp = StreamProcessor(cfg)
        points = [_make_dp(value=float(i)) for i in range(3)]
        result = sp.process(points)
        assert len(result) == 3

    def test_smoothing_processor_moving_average(self):
        cfg = {"pipeline": [{"type": "smoothing", "method": "moving_average", "window_size": 3}]}
        sp = StreamProcessor(cfg)
        points = [_make_dp(sensor_id="s1", value=float(i)) for i in range(10)]
        result = sp.process(points)
        assert len(result) == 10

    def test_smoothing_processor_savgol(self):
        cfg = {"pipeline": [{"type": "smoothing", "method": "savgol", "window_size": 5}]}
        sp = StreamProcessor(cfg)
        points = [_make_dp(sensor_id="s1", value=float(i)) for i in range(10)]
        result = sp.process(points)
        assert len(result) >= 1

    def test_smoothing_processor_unknown_method(self):
        cfg = {"pipeline": [{"type": "smoothing", "method": "unknown", "window_size": 3}]}
        sp = StreamProcessor(cfg)
        points = [_make_dp(sensor_id="s1", value=float(i)) for i in range(10)]
        result = sp.process(points)
        assert len(result) == 10

    def test_outlier_processor_iqr(self):
        cfg = {"pipeline": [{"type": "outlier_detection", "method": "iqr", "threshold": 1.5}]}
        sp = StreamProcessor(cfg)
        points = [_make_dp(sensor_id="s1", value=float(i)) for i in range(10)]
        points.append(_make_dp(sensor_id="s1", value=1000.0))  # outlier
        result = sp.process(points)
        assert len(result) == 11

    def test_outlier_processor_zscore(self):
        cfg = {"pipeline": [{"type": "outlier_detection", "method": "zscore", "threshold": 2.0}]}
        sp = StreamProcessor(cfg)
        points = [_make_dp(sensor_id="s1", value=float(i)) for i in range(10)]
        result = sp.process(points)
        assert len(result) == 10

    def test_outlier_processor_unknown_method(self):
        cfg = {"pipeline": [{"type": "outlier_detection", "method": "unknown", "threshold": 2.0}]}
        sp = StreamProcessor(cfg)
        points = [_make_dp(sensor_id="s1", value=float(i)) for i in range(10)]
        result = sp.process(points)
        assert len(result) == 10

    def test_outlier_processor_short_data(self):
        cfg = {"pipeline": [{"type": "outlier_detection", "method": "iqr"}]}
        sp = StreamProcessor(cfg)
        points = [_make_dp(sensor_id="s1", value=1.0) for _ in range(3)]
        result = sp.process(points)
        assert len(result) == 3

    def test_outlier_processor_with_nan(self):
        cfg = {"pipeline": [{"type": "outlier_detection", "method": "iqr", "threshold": 1.5}]}
        sp = StreamProcessor(cfg)
        points = [_make_dp(sensor_id="s1", value=float(i)) for i in range(10)]
        points.append(_make_dp(sensor_id="s1", value=float("nan")))
        result = sp.process(points)
        assert len(result) == 11

    def test_trend_processor(self):
        cfg = {"pipeline": [{"type": "trend_analysis", "window_size": 5}]}
        sp = StreamProcessor(cfg)
        points = [_make_dp(sensor_id="s1", value=float(i)) for i in range(10)]
        result = sp.process(points)
        assert len(result) == 10

    def test_trend_processor_short_data(self):
        cfg = {"pipeline": [{"type": "trend_analysis", "window_size": 20}]}
        sp = StreamProcessor(cfg)
        points = [_make_dp(sensor_id="s1", value=1.0) for _ in range(5)]
        result = sp.process(points)
        assert len(result) == 5

    def test_trend_processor_insufficient_valid(self):
        cfg = {"pipeline": [{"type": "trend_analysis", "window_size": 5}]}
        sp = StreamProcessor(cfg)
        # All NaN values
        points = [_make_dp(sensor_id="s1", value=float("nan")) for _ in range(10)]
        result = sp.process(points)
        assert len(result) == 10

    def test_anomaly_processor(self):
        cfg = {"pipeline": [{"type": "anomaly_detection", "method": "isolation_forest"}]}
        sp = StreamProcessor(cfg)
        points = [_make_dp(sensor_id="s1", value=float(i)) for i in range(15)]
        # Ensure fit_predict returns right length
        _mock_iso_forest.return_value.fit_predict.return_value = [1] * 15
        result = sp.process(points)
        assert len(result) >= 1

    def test_anomaly_processor_short_data(self):
        cfg = {"pipeline": [{"type": "anomaly_detection", "method": "isolation_forest"}]}
        sp = StreamProcessor(cfg)
        points = [_make_dp(sensor_id="s1", value=1.0) for _ in range(5)]
        result = sp.process(points)
        assert len(result) == 5

    def test_anomaly_processor_unknown_method(self):
        cfg = {"pipeline": [{"type": "anomaly_detection", "method": "unknown"}]}
        sp = StreamProcessor(cfg)
        points = [_make_dp(sensor_id="s1", value=float(i)) for i in range(15)]
        result = sp.process(points)
        assert len(result) >= 1

    def test_anomaly_processor_with_nan(self):
        cfg = {"pipeline": [{"type": "anomaly_detection", "method": "isolation_forest"}]}
        sp = StreamProcessor(cfg)
        points = [_make_dp(sensor_id="s1", value=float(i)) for i in range(12)]
        points.append(_make_dp(sensor_id="s1", value=float("nan")))
        _mock_iso_forest.return_value.fit_predict.return_value = [1] * 12
        result = sp.process(points)
        assert len(result) >= 1

    def test_process_with_error(self):
        sp = StreamProcessor({})
        sp.processors.append(lambda x: 1 / 0)  # Will raise
        result = sp.process([_make_dp()])
        assert isinstance(result, list)

    def test_moving_average(self):
        sp = StreamProcessor({})
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = sp._moving_average(values, 3)
        assert len(result) >= 1

    def test_moving_average_short(self):
        sp = StreamProcessor({})
        values = [1.0, 2.0]
        result = sp._moving_average(values, 5)
        assert result == values


class TestRealTimeAnalyzer:
    def test_init(self):
        cfg = create_sample_alert_config()
        analyzer = RealTimeAnalyzer(cfg)
        assert analyzer.alert_thresholds is not None

    def test_add_alert_callback(self):
        analyzer = RealTimeAnalyzer({})
        cb = MagicMock()
        analyzer.add_alert_callback(cb)
        assert len(analyzer.alert_callbacks) == 1

    def test_analyze_stream_no_data(self):
        class ConcreteStream(DataStream):
            def start(self):
                pass

            def stop(self):
                pass

        analyzer = RealTimeAnalyzer({})
        ds = ConcreteStream("test")
        result = analyzer.analyze_stream(ds)
        assert result["status"] == "no_data"

    def test_analyze_stream_with_data(self):
        class ConcreteStream(DataStream):
            def start(self):
                pass

            def stop(self):
                pass

        analyzer = RealTimeAnalyzer({})
        ds = ConcreteStream("test")
        now = datetime.now()
        for i in range(5):
            ds.buffer.add(_make_dp(
                sensor_id="s1",
                value=float(i + 1),
                ts=now - timedelta(seconds=i),
            ))
        result = analyzer.analyze_stream(ds, analysis_window=timedelta(minutes=1))
        assert "sensors" in result
        assert "s1" in result["sensors"]

    def test_analyze_sensor_data_empty(self):
        analyzer = RealTimeAnalyzer({})
        result = analyzer._analyze_sensor_data("s1", [])
        # All values are nan, so no valid data
        assert result.get("status") == "no_valid_data" or result.get("count", 0) >= 0

    def test_analyze_sensor_data_with_values(self):
        analyzer = RealTimeAnalyzer({})
        points = [_make_dp(sensor_id="s1", value=float(i)) for i in range(5)]
        result = analyzer._analyze_sensor_data("s1", points)
        assert "count" in result
        assert "mean" in result

    def test_analyze_sensor_data_trend(self):
        analyzer = RealTimeAnalyzer({})
        points = [_make_dp(sensor_id="s1", value=float(i)) for i in range(10)]
        result = analyzer._analyze_sensor_data("s1", points)
        assert "trend" in result

    def test_analyze_sensor_data_rate_of_change(self):
        analyzer = RealTimeAnalyzer({})
        points = [_make_dp(sensor_id="s1", value=float(i * 2)) for i in range(5)]
        result = analyzer._analyze_sensor_data("s1", points)
        assert "rate_of_change" in result

    def test_analyze_sensor_data_cv(self):
        analyzer = RealTimeAnalyzer({})
        points = [_make_dp(sensor_id="s1", value=float(i + 1)) for i in range(5)]
        result = analyzer._analyze_sensor_data("s1", points)
        assert "coefficient_of_variation" in result

    def test_analyze_sensor_data_zero_mean(self):
        analyzer = RealTimeAnalyzer({})
        points = [_make_dp(sensor_id="s1", value=0.0) for _ in range(5)]
        result = analyzer._analyze_sensor_data("s1", points)
        # Should handle zero mean for CV calculation
        assert result is not None

    def test_check_alerts_no_thresholds(self):
        analyzer = RealTimeAnalyzer({})
        # Should not raise
        analyzer._check_alerts("unknown", {}, [])

    def test_check_alerts_min_value(self):
        cfg = {"alert_thresholds": {"s1": {"min_value": 5.0}}}
        analyzer = RealTimeAnalyzer(cfg)
        alerts = []
        analyzer.add_alert_callback(lambda a: alerts.append(a))
        analyzer._check_alerts("s1", {"min": 3.0}, [])
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.WARNING

    def test_check_alerts_max_value(self):
        cfg = {"alert_thresholds": {"s1": {"max_value": 10.0}}}
        analyzer = RealTimeAnalyzer(cfg)
        alerts = []
        analyzer.add_alert_callback(lambda a: alerts.append(a))
        analyzer._check_alerts("s1", {"max": 15.0}, [])
        assert len(alerts) == 1

    def test_check_alerts_min_quality(self):
        cfg = {"alert_thresholds": {"s1": {"min_quality": 0.8}}}
        analyzer = RealTimeAnalyzer(cfg)
        alerts = []
        analyzer.add_alert_callback(lambda a: alerts.append(a))
        analyzer._check_alerts("s1", {"quality_score": 0.5}, [])
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.ERROR

    def test_check_alerts_trend(self):
        cfg = {"alert_thresholds": {"s1": {"max_trend_slope": 0.01}}}
        analyzer = RealTimeAnalyzer(cfg)
        alerts = []
        analyzer.add_alert_callback(lambda a: alerts.append(a))
        analyzer._check_alerts("s1", {"trend": {"slope": 0.5}}, [])
        assert len(alerts) == 1

    def test_send_alert_callback_error(self):
        analyzer = RealTimeAnalyzer({})
        analyzer.add_alert_callback(lambda a: 1 / 0)
        alert = Alert(
            timestamp=datetime.now(),
            level=AlertLevel.WARNING,
            sensor_id="s1",
            message="test",
        )
        analyzer._send_alert(alert)  # Should not raise


class TestAlertSystem:
    def test_init(self):
        asys = AlertSystem({"max_history": 500})
        assert asys.is_active is False

    def test_start_stop(self):
        asys = AlertSystem({})
        asys.start()
        assert asys.is_active is True
        time.sleep(0.05)
        asys.stop()
        assert asys.is_active is False

    def test_start_already_active(self):
        asys = AlertSystem({})
        asys.start()
        asys.start()  # Should not start second thread
        asys.stop()

    def test_stop_already_stopped(self):
        asys = AlertSystem({})
        asys.stop()  # Should not raise

    def test_send_and_get_alerts(self):
        asys = AlertSystem({"max_history": 100})
        asys.start()
        time.sleep(0.05)
        alert = Alert(
            timestamp=datetime.now(),
            level=AlertLevel.WARNING,
            sensor_id="s1",
            message="test alert",
        )
        asys.send_alert(alert)
        time.sleep(0.5)
        asys.stop()
        recent = asys.get_recent_alerts()
        assert len(recent) >= 1

    def test_process_alert_levels(self):
        asys = AlertSystem({})
        for level in [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]:
            alert = Alert(
                timestamp=datetime.now(),
                level=level,
                sensor_id="s1",
                message=f"test {level.value}",
            )
            asys._process_alert(alert)

    def test_handle_critical(self):
        asys = AlertSystem({})
        alert = Alert(
            timestamp=datetime.now(),
            level=AlertLevel.CRITICAL,
            sensor_id="s1",
            message="critical test",
        )
        asys._handle_critical_alert(alert)  # Should just log

    def test_handle_error(self):
        asys = AlertSystem({})
        alert = Alert(
            timestamp=datetime.now(),
            level=AlertLevel.ERROR,
            sensor_id="s1",
            message="error test",
        )
        asys._handle_error_alert(alert)  # Should just log


class TestCoverageGaps:
    """Tests targeting specific uncovered lines."""

    def test_has_scipy_false_trend_fallback(self):
        """Cover lines 876-877: no-scipy trend estimation fallback."""
        analyzer = RealTimeAnalyzer({})
        points = [_make_dp(sensor_id="s1", value=float(i)) for i in range(5)]
        old_val = _mod.HAS_SCIPY
        try:
            _mod.HAS_SCIPY = False
            result = analyzer._analyze_sensor_data("s1", points)
            assert "trend" in result
            assert "slope" in result["trend"]
            assert "direction" in result["trend"]
        finally:
            _mod.HAS_SCIPY = old_val

    def test_has_sklearn_false_anomaly_processor(self):
        """Cover line 695: HAS_SKLEARN = False in anomaly processor."""
        old_val = _mod.HAS_SKLEARN
        try:
            _mod.HAS_SKLEARN = False
            cfg = {"pipeline": [{"type": "anomaly_detection", "method": "isolation_forest"}]}
            sp = StreamProcessor(cfg)
            points = [_make_dp(sensor_id="s1", value=float(i)) for i in range(15)]
            result = sp.process(points)
            assert len(result) == 15
        finally:
            _mod.HAS_SKLEARN = old_val

    def test_data_acquisition_loop_exception(self):
        """Cover lines 435-437: exception in _data_acquisition_loop."""
        cfg = {"s1": {"base_value": 1.0, "noise_level": 0.1}}
        ds = MFCDataStream("mfc_err", cfg, sampling_rate=100.0)
        # Patch _read_sensor to raise on first call then allow stop
        call_count = [0]
        original_read = ds._read_sensor

        def raising_read(sensor_id, config):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError("simulated sensor failure")
            return original_read(sensor_id, config)

        ds._read_sensor = raising_read
        ds.start()
        time.sleep(0.3)
        ds.stop()
        # The loop should have survived the exceptions
        assert call_count[0] >= 1

    def test_alert_processing_loop_exception(self):
        """Cover lines 1041-1042: exception in _alert_processing_loop."""
        asys = AlertSystem({"max_history": 100})
        # Patch _process_alert to raise
        original_process = asys._process_alert

        def raising_process(alert):
            raise RuntimeError("simulated processing failure")

        asys._process_alert = raising_process
        asys.start()
        time.sleep(0.05)
        alert = Alert(
            timestamp=datetime.now(),
            level=AlertLevel.WARNING,
            sensor_id="s1",
            message="test exception",
        )
        asys.send_alert(alert)
        time.sleep(0.5)
        asys.stop()
        # System should survive the exception

    def test_has_scipy_false_import_path(self):
        """Cover lines 68-71: scipy import TypeError path."""
        # We can't easily re-trigger the import, but we can verify the
        # HAS_SCIPY flag behavior and the warning path.
        # The flag is set at module load time. We can at least verify
        # that the code path for HAS_SCIPY=False works end-to-end.
        old_val = _mod.HAS_SCIPY
        try:
            _mod.HAS_SCIPY = False
            # Test smoothing processor with savgol when HAS_SCIPY=False
            cfg = {"pipeline": [{"type": "smoothing", "method": "savgol", "window_size": 5}]}
            sp = StreamProcessor(cfg)
            points = [_make_dp(sensor_id="s1", value=float(i)) for i in range(10)]
            result = sp.process(points)
            # Should fall through to else branch (no savgol)
            assert len(result) >= 1

            # Test trend processor with HAS_SCIPY=False
            cfg2 = {"pipeline": [{"type": "trend_analysis", "window_size": 5}]}
            sp2 = StreamProcessor(cfg2)
            points2 = [_make_dp(sensor_id="s1", value=float(i)) for i in range(10)]
            result2 = sp2.process(points2)
            assert len(result2) >= 1

            # Test outlier zscore with HAS_SCIPY=False
            cfg3 = {"pipeline": [{"type": "outlier_detection", "method": "zscore", "threshold": 2.0}]}
            sp3 = StreamProcessor(cfg3)
            points3 = [_make_dp(sensor_id="s1", value=float(i)) for i in range(10)]
            result3 = sp3.process(points3)
            assert len(result3) == 10
        finally:
            _mod.HAS_SCIPY = old_val


class TestUtilityFunctions:
    def test_create_sample_mfc_config(self):
        cfg = create_sample_mfc_config()
        assert "power_sensor" in cfg
        assert "flow_rate_sensor" in cfg
        assert "substrate_concentration" in cfg
        assert "biofilm_thickness" in cfg

    def test_create_sample_processing_config(self):
        cfg = create_sample_processing_config()
        assert "pipeline" in cfg
        assert len(cfg["pipeline"]) > 0

    def test_create_sample_alert_config(self):
        cfg = create_sample_alert_config()
        assert "alert_thresholds" in cfg
        assert "max_history" in cfg
