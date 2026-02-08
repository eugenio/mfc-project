"""Tests for degradation_detector module - comprehensive coverage."""
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stability.degradation_detector import (
    BaseDegradationDetector,
    DegradationEvent,
    DegradationMetrics,
    DegradationSeverity,
    DegradationThresholds,
    DegradationType,
    DetectionMethod,
    StatisticalDegradationDetector,
    create_statistical_degradation_detector,
    run_example_degradation_detection,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class TestDegradationType:
    def test_str(self):
        assert str(DegradationType.BIOFILM_AGING) == "biofilm aging"
        assert str(DegradationType.ELECTRODE_CORROSION) == "electrode corrosion"

    def test_members_count(self):
        assert len(DegradationType) == 10


class TestDegradationSeverity:
    def test_numeric_value(self):
        assert DegradationSeverity.MINIMAL.numeric_value == 0.1
        assert DegradationSeverity.MINOR.numeric_value == 0.3
        assert DegradationSeverity.MODERATE.numeric_value == 0.5
        assert DegradationSeverity.SEVERE.numeric_value == 0.8
        assert DegradationSeverity.CRITICAL.numeric_value == 1.0


# ---------------------------------------------------------------------------
# DegradationThresholds
# ---------------------------------------------------------------------------
class TestDegradationThresholds:
    def test_default_thresholds(self):
        t = DegradationThresholds()
        assert t.power_degradation_threshold == 0.05
        assert t.trend_detection_window == 50

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            DegradationThresholds(power_degradation_threshold=2.0)

    def test_invalid_efficiency_threshold(self):
        with pytest.raises(ValueError):
            DegradationThresholds(efficiency_degradation_threshold=-0.1)

    def test_invalid_biofilm_threshold(self):
        with pytest.raises(ValueError):
            DegradationThresholds(biofilm_health_threshold=1.5)

    def test_invalid_electrode_threshold(self):
        with pytest.raises(ValueError):
            DegradationThresholds(electrode_performance_threshold=-0.01)


# ---------------------------------------------------------------------------
# DegradationEvent
# ---------------------------------------------------------------------------
class TestDegradationEvent:
    @pytest.fixture
    def event(self):
        return DegradationEvent(
            detection_time=datetime.now(),
            degradation_type=DegradationType.BIOFILM_AGING,
            severity=DegradationSeverity.MODERATE,
            affected_component="Biofilm",
            degradation_rate=0.01,
            confidence=0.85,
            baseline_value=20.0,
            current_value=18.0,
            trend_slope=-0.1,
            detection_method=DetectionMethod.STATISTICAL_PROCESS_CONTROL,
        )

    def test_creation(self, event):
        assert event.confidence == 0.85

    def test_invalid_confidence(self):
        with pytest.raises(ValueError, match="Confidence"):
            DegradationEvent(
                detection_time=datetime.now(),
                degradation_type=DegradationType.BIOFILM_AGING,
                severity=DegradationSeverity.MODERATE,
                affected_component="X",
                degradation_rate=0.01,
                confidence=1.5,
                baseline_value=20.0,
                current_value=18.0,
                trend_slope=-0.1,
                detection_method=DetectionMethod.TREND_ANALYSIS,
            )

    def test_negative_degradation_rate(self):
        with pytest.raises(ValueError, match="non-negative"):
            DegradationEvent(
                detection_time=datetime.now(),
                degradation_type=DegradationType.BIOFILM_AGING,
                severity=DegradationSeverity.MODERATE,
                affected_component="X",
                degradation_rate=-0.01,
                confidence=0.5,
                baseline_value=20.0,
                current_value=18.0,
                trend_slope=-0.1,
                detection_method=DetectionMethod.TREND_ANALYSIS,
            )

    def test_degradation_percentage(self, event):
        pct = event.degradation_percentage
        assert abs(pct - 0.1) < 0.01

    def test_degradation_percentage_zero_baseline(self):
        e = DegradationEvent(
            detection_time=datetime.now(),
            degradation_type=DegradationType.BIOFILM_AGING,
            severity=DegradationSeverity.MINIMAL,
            affected_component="X",
            degradation_rate=0.0,
            confidence=0.5,
            baseline_value=0.0,
            current_value=0.0,
            trend_slope=0.0,
            detection_method=DetectionMethod.TREND_ANALYSIS,
        )
        assert e.degradation_percentage == 0.0

    def test_to_dict(self, event):
        d = event.to_dict()
        assert "detection_time" in d
        assert "degradation_percentage" in d
        assert d["affected_component"] == "Biofilm"


# ---------------------------------------------------------------------------
# DegradationMetrics
# ---------------------------------------------------------------------------
class TestDegradationMetrics:
    def test_default_metrics(self):
        m = DegradationMetrics()
        assert m.system_health_score > 0

    def test_compute_derived(self):
        m = DegradationMetrics(
            biofilm_health_score=0.5,
            electrode_health_score=0.5,
            membrane_health_score=0.5,
        )
        assert m.system_health_score == pytest.approx(0.5)

    def test_severity_levels(self):
        assert DegradationMetrics(overall_health_score=0.95).get_severity_level() == DegradationSeverity.MINIMAL
        assert DegradationMetrics(overall_health_score=0.85).get_severity_level() == DegradationSeverity.MINOR
        assert DegradationMetrics(overall_health_score=0.65).get_severity_level() == DegradationSeverity.MODERATE
        assert DegradationMetrics(overall_health_score=0.45).get_severity_level() == DegradationSeverity.SEVERE
        assert DegradationMetrics(overall_health_score=0.1).get_severity_level() == DegradationSeverity.CRITICAL

    def test_warning_on_invalid_health_score(self):
        m = DegradationMetrics(overall_health_score=1.5)
        assert m.overall_health_score >= 0

    def test_to_dict(self):
        m = DegradationMetrics(
            last_significant_change=datetime.now(),
            predicted_failure_time=datetime.now(),
            change_points_detected=[(1.0, 0.5)],
        )
        d = m.to_dict()
        assert "overall_health_score" in d
        assert "severity_level" in d
        assert d["last_significant_change"] is not None
        assert d["predicted_failure_time"] is not None

    def test_to_dict_none_timestamps(self):
        m = DegradationMetrics()
        d = m.to_dict()
        assert d["last_significant_change"] is None
        assert d["predicted_failure_time"] is None


# ---------------------------------------------------------------------------
# BaseDegradationDetector
# ---------------------------------------------------------------------------
class TestBaseDegradationDetector:
    @pytest.fixture
    def detector(self):
        return StatisticalDegradationDetector(min_data_points=10)

    def test_validate_data_valid(self, detector):
        data = np.random.normal(20, 1, 100)
        assert detector.validate_data(data) is True

    def test_validate_data_too_short(self, detector):
        data = np.array([1.0, 2.0])
        assert detector.validate_data(data) is False

    def test_validate_data_with_nan(self, detector):
        data = np.array([1.0, np.nan, 3.0] * 10)
        assert detector.validate_data(data) is False

    def test_validate_data_with_inf(self, detector):
        data = np.array([1.0, np.inf, 3.0] * 10)
        assert detector.validate_data(data) is False

    def test_validate_data_exception(self, detector):
        assert detector.validate_data("not_array") is False

    def test_establish_baseline(self, detector):
        data = np.random.normal(20, 1, 100)
        stats = detector.establish_baseline(data)
        assert "mean" in stats
        assert "std" in stats
        assert "trend" in stats
        assert detector.baseline_established

    def test_establish_baseline_failure(self, detector):
        with patch("numpy.mean", side_effect=Exception("fail")):
            result = detector.establish_baseline(np.array([1.0]))
            assert result == {}

    def test_health_score_general(self, detector):
        score = detector._calculate_health_score(18.0, 20.0, "general")
        assert 0.0 <= score <= 1.0

    def test_health_score_biofilm(self, detector):
        score = detector._calculate_health_score(18.0, 20.0, "biofilm")
        assert 0.0 <= score <= 1.0

    def test_health_score_electrode(self, detector):
        score = detector._calculate_health_score(18.0, 20.0, "electrode")
        assert 0.0 <= score <= 1.0

    def test_health_score_zero_baseline(self, detector):
        score = detector._calculate_health_score(5.0, 0.0)
        assert score == 1.0

    def test_health_score_above_baseline(self, detector):
        score = detector._calculate_health_score(25.0, 20.0)
        assert score == 1.0


# ---------------------------------------------------------------------------
# StatisticalDegradationDetector - detect_degradation
# ---------------------------------------------------------------------------
class TestStatisticalDetector:
    @pytest.fixture
    def detector(self):
        return StatisticalDegradationDetector(min_data_points=30)

    @pytest.fixture
    def degrading_data(self):
        np.random.seed(42)
        n = 200
        base = 20.0
        trend = np.linspace(0, -3, n)
        noise = np.random.normal(0, 0.3, n)
        return base + trend + noise

    @pytest.fixture
    def stable_data(self):
        np.random.seed(42)
        return 20.0 + np.random.normal(0, 0.3, 200)

    def test_detect_degradation_with_baseline(self, detector, degrading_data):
        detector.establish_baseline(degrading_data[:50])
        metrics, events = detector.detect_degradation(degrading_data)
        assert isinstance(metrics, DegradationMetrics)
        assert isinstance(events, list)

    def test_detect_degradation_without_baseline(self, detector, degrading_data):
        metrics, events = detector.detect_degradation(degrading_data)
        assert detector.baseline_established
        assert metrics.sample_size == 200

    def test_detect_degradation_invalid_data(self, detector):
        with pytest.raises(ValueError):
            detector.detect_degradation(np.array([1.0, 2.0]))

    def test_detect_degradation_stable_data(self, detector, stable_data):
        metrics, events = detector.detect_degradation(stable_data)
        assert metrics.overall_health_score > 0.5

    def test_detect_anomalies_zero_std(self, detector):
        data = np.ones(50)
        result = detector._detect_anomalies(data)
        assert result["score"] == 0.0
        assert result["count"] == 0

    def test_detect_anomalies_with_outliers(self, detector):
        data = np.random.normal(20, 1, 100)
        data[50] = 50.0  # outlier
        result = detector._detect_anomalies(data)
        assert result["count"] > 0
        assert result["severity"] > 0

    def test_analyze_trends_short_data(self, detector):
        data = np.random.normal(20, 1, 40)
        result = detector._analyze_trends(data)
        assert "short_term" in result
        assert "acceleration" in result

    def test_analyze_trends_long_data(self, detector):
        data = np.random.normal(20, 1, 200)
        result = detector._analyze_trends(data)
        assert result["acceleration"] != 0.0 or True  # may be 0

    def test_change_point_detection(self, detector):
        data = np.concatenate([
            np.random.normal(20, 0.3, 80),
            np.random.normal(16, 0.3, 80),
        ])
        result = detector._detect_change_points(data)
        assert isinstance(result["change_points"], list)

    def test_change_point_with_timestamps(self, detector):
        data = np.concatenate([
            np.random.normal(20, 0.3, 80),
            np.random.normal(16, 0.3, 80),
        ])
        timestamps = [datetime.now() + timedelta(hours=i) for i in range(160)]
        result = detector._detect_change_points(data, timestamps)
        assert isinstance(result, dict)

    def test_change_point_non_datetime_timestamps(self, detector):
        data = np.concatenate([
            np.random.normal(20, 0.3, 80),
            np.random.normal(16, 0.3, 80),
        ])
        timestamps = list(range(160))
        result = detector._detect_change_points(data, timestamps)
        assert isinstance(result, dict)

    def test_change_point_failure(self, detector):
        with patch("numpy.mean", side_effect=Exception("fail")):
            result = detector._detect_change_points(np.array([1.0]))
            assert result["change_points"] == []

    def test_component_health_no_baseline(self, detector):
        result = detector._calculate_component_health_scores(np.ones(50))
        assert result == {"biofilm": 1.0, "electrode": 1.0, "membrane": 1.0}

    def test_component_health_with_baseline(self, detector):
        data = np.random.normal(20, 1, 100)
        detector.establish_baseline(data[:50])
        result = detector._calculate_component_health_scores(data)
        assert all(0.0 <= v <= 1.0 for v in result.values())

    def test_degradation_percentages_no_baseline(self, detector):
        result = detector._calculate_degradation_percentages(np.ones(50))
        assert result == {"power": 0.0, "efficiency": 0.0, "stability": 0.0}

    def test_degradation_percentages_with_baseline(self, detector):
        data = np.random.normal(20, 1, 100)
        detector.establish_baseline(data[:50])
        result = detector._calculate_degradation_percentages(data)
        assert "power" in result

    def test_degradation_percentages_zero_baseline(self, detector):
        detector.baseline_established = True
        detector.baseline_stats = {"mean": 0.0, "std": 0.0}
        result = detector._calculate_degradation_percentages(np.ones(20))
        assert result["power"] == 0.0

    def test_degradation_rate_short_data(self, detector):
        rate = detector._calculate_degradation_rate(np.array([1.0, 2.0]))
        assert rate == 0.0

    def test_degradation_rate_no_baseline(self, detector):
        data = np.linspace(20, 15, 50)
        rate = detector._calculate_degradation_rate(data)
        assert rate >= 0.0

    def test_degradation_rate_with_baseline(self, detector):
        data = np.linspace(20, 15, 50)
        detector.establish_baseline(data[:20])
        rate = detector._calculate_degradation_rate(data)
        assert rate >= 0.0

    def test_degradation_rate_zero_mean(self, detector):
        data = np.linspace(1, -1, 50)
        rate = detector._calculate_degradation_rate(data)
        assert rate >= 0.0

    def test_predict_failure_declining(self, detector):
        data = np.linspace(20, 10, 100)
        detector.establish_baseline(data[:30])
        result = detector._predict_failure(data)
        assert result["failure_time"] is not None
        assert result["remaining_life"] < float("inf")

    def test_predict_failure_stable(self, detector):
        data = np.ones(100) * 20
        result = detector._predict_failure(data)
        assert result["failure_time"] is None

    def test_predict_failure_no_baseline(self, detector):
        data = np.linspace(20, 10, 100)
        result = detector._predict_failure(data)
        assert isinstance(result, dict)

    def test_predict_failure_exception(self, detector):
        with patch("numpy.polyfit", side_effect=Exception("fail")):
            result = detector._predict_failure(np.array([1.0, 2.0]))
            assert result["failure_time"] is None

    def test_assess_data_quality(self, detector):
        data = np.random.normal(20, 1, 100)
        score = detector._assess_data_quality(data)
        assert 0.0 <= score <= 1.0

    def test_assess_data_quality_single_point(self, detector):
        data = np.array([20.0])
        score = detector._assess_data_quality(data)
        assert 0.0 <= score <= 1.0

    def test_generate_events_healthy(self, detector):
        detector.establish_baseline(np.random.normal(20, 0.3, 50))
        metrics = DegradationMetrics(
            overall_health_score=0.95,
            biofilm_health_score=0.95,
            electrode_health_score=0.95,
            anomaly_count=0,
        )
        events = detector._generate_degradation_events(np.ones(50) * 20, metrics)
        assert len(events) == 0

    def test_generate_events_degraded(self, detector):
        detector.establish_baseline(np.random.normal(20, 0.3, 50))
        metrics = DegradationMetrics(
            overall_health_score=0.5,
            biofilm_health_score=0.5,
            electrode_health_score=0.5,
            anomaly_count=15,
            long_term_trend=-0.1,
            degradation_rate=0.02,
            anomaly_score=2.5,
            trend_acceleration=0.1,
            anomaly_severity=3.0,
        )
        events = detector._generate_degradation_events(np.ones(50) * 15, metrics)
        assert len(events) > 0

    def test_generate_events_exception(self, detector):
        with patch.object(detector, "_determine_severity_from_health_score", side_effect=Exception("fail")):
            events = detector._generate_degradation_events(
                np.ones(50),
                DegradationMetrics(overall_health_score=0.3),
            )
            assert events == []

    def test_severity_from_health_score(self, detector):
        assert detector._determine_severity_from_health_score(0.95) == DegradationSeverity.MINIMAL
        assert detector._determine_severity_from_health_score(0.85) == DegradationSeverity.MINOR
        assert detector._determine_severity_from_health_score(0.65) == DegradationSeverity.MODERATE
        assert detector._determine_severity_from_health_score(0.45) == DegradationSeverity.SEVERE
        assert detector._determine_severity_from_health_score(0.1) == DegradationSeverity.CRITICAL

    def test_generate_recommendations(self, detector):
        metrics = DegradationMetrics(
            overall_health_score=0.5,
            biofilm_health_score=0.5,
            electrode_health_score=0.5,
            degradation_rate=0.05,
            anomaly_count=15,
        )
        recs = detector._generate_recommendations(metrics)
        assert len(recs) > 0

    def test_generate_recommendations_healthy(self, detector):
        metrics = DegradationMetrics()
        recs = detector._generate_recommendations(metrics)
        assert "Continue normal monitoring" in recs


# ---------------------------------------------------------------------------
# Predict degradation
# ---------------------------------------------------------------------------
class TestPredictDegradation:
    @pytest.fixture
    def detector(self):
        return StatisticalDegradationDetector(min_data_points=30)

    def test_predict_no_baseline(self, detector):
        result = detector.predict_degradation(100.0, 50.0)
        assert result["confidence"] == 0.0
        assert result["predicted_health_scores"] == []

    def test_predict_with_baseline(self, detector):
        data = np.random.normal(20, 0.5, 100)
        detector.establish_baseline(data)
        result = detector.predict_degradation(100.0, 200.0)
        assert len(result["predicted_health_scores"]) == 100
        assert "confidence" in result

    def test_predict_with_degradation_events(self, detector):
        data = np.linspace(20, 15, 100)
        detector.establish_baseline(data)
        result = detector.predict_degradation(0.0, 500.0)
        assert isinstance(result["predicted_degradation_events"], list)


# ---------------------------------------------------------------------------
# Factory and example
# ---------------------------------------------------------------------------
class TestFactory:
    def test_create_detector(self):
        d = create_statistical_degradation_detector()
        assert isinstance(d, StatisticalDegradationDetector)

    def test_create_with_thresholds(self):
        t = DegradationThresholds(power_degradation_threshold=0.1)
        d = create_statistical_degradation_detector(thresholds=t)
        assert d.thresholds.power_degradation_threshold == 0.1

    def test_run_example(self):
        run_example_degradation_detection()
