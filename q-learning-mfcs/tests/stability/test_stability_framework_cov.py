"""Tests for stability_framework module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from stability.stability_framework import (
    AnalysisMethod,
    BaseStabilityAnalyzer,
    StabilityFramework,
    StabilityLevel,
    StabilityMetrics,
    StabilityThresholds,
    StatisticalStabilityAnalyzer,
    create_default_framework,
    create_statistical_analyzer,
)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def stable_data():
    np.random.seed(42)
    return 20.0 + np.random.normal(0, 0.5, 100)


@pytest.fixture
def analyzer():
    return StatisticalStabilityAnalyzer(min_data_points=10)


@pytest.fixture
def framework():
    return create_default_framework()


class TestStabilityLevel:
    def test_str(self):
        assert str(StabilityLevel.CRITICAL) == "critical"
        assert str(StabilityLevel.EXCELLENT) == "excellent"

    def test_numeric_value(self):
        assert StabilityLevel.CRITICAL.numeric_value == 0.0
        assert StabilityLevel.POOR.numeric_value == 0.25
        assert StabilityLevel.FAIR.numeric_value == 0.5
        assert StabilityLevel.GOOD.numeric_value == 0.75
        assert StabilityLevel.EXCELLENT.numeric_value == 1.0


class TestAnalysisMethod:
    def test_values(self):
        assert AnalysisMethod.STATISTICAL is not None
        assert AnalysisMethod.SPECTRAL is not None
        assert AnalysisMethod.TREND_ANALYSIS is not None


class TestStabilityThresholds:
    def test_defaults(self):
        t = StabilityThresholds()
        assert t.power_variation_threshold == 0.05
        assert t.efficiency_degradation_threshold == 0.10

    def test_custom(self):
        t = StabilityThresholds(power_variation_threshold=0.1)
        assert t.power_variation_threshold == 0.1

    def test_invalid_power_threshold(self):
        with pytest.raises(ValueError):
            StabilityThresholds(power_variation_threshold=2.0)

    def test_invalid_efficiency_threshold(self):
        with pytest.raises(ValueError):
            StabilityThresholds(efficiency_degradation_threshold=-0.1)


class TestStabilityMetrics:
    def test_defaults(self):
        m = StabilityMetrics()
        assert m.overall_stability_score >= 0.0
        assert m.stability_level is not None

    def test_compute_overall_stability(self):
        m = StabilityMetrics(
            power_stability_coefficient=0.9,
            efficiency_stability=0.85,
            biofilm_health_index=0.8,
            power_autocorrelation=0.7,
            biofilm_thickness_stability=0.9,
        )
        assert m.overall_stability_score > 0.5

    def test_determine_level_excellent(self):
        m = StabilityMetrics(
            power_stability_coefficient=1.0,
            efficiency_stability=1.0,
            biofilm_health_index=1.0,
            power_autocorrelation=1.0,
            biofilm_thickness_stability=1.0,
        )
        assert m.stability_level == StabilityLevel.EXCELLENT

    def test_determine_level_critical(self):
        m = StabilityMetrics()
        assert m.stability_level == StabilityLevel.CRITICAL

    def test_to_dict(self):
        m = StabilityMetrics()
        d = m.to_dict()
        assert "power_stability_coefficient" in d
        assert "stability_level" in d
        assert "analysis_method" in d

    def test_validate_metrics_warning(self):
        m = StabilityMetrics(power_stability_coefficient=1.5)
        assert m.power_stability_coefficient == 1.5


class TestStatisticalStabilityAnalyzer:
    def test_init(self):
        a = StatisticalStabilityAnalyzer()
        assert a.min_data_points == 30

    def test_init_custom_window(self):
        a = StatisticalStabilityAnalyzer(window_size=5)
        assert a.window_size == 5

    def test_validate_data_valid(self, analyzer, stable_data):
        assert analyzer.validate_data(stable_data) is True

    def test_validate_data_short(self, analyzer):
        assert analyzer.validate_data([1.0, 2.0]) is False

    def test_validate_data_nan(self, analyzer):
        data = np.array([1.0, float("nan"), 3.0] + [1.0] * 20)
        assert analyzer.validate_data(data) is False

    def test_validate_data_inf(self, analyzer):
        data = np.array([1.0, float("inf"), 3.0] + [1.0] * 20)
        assert analyzer.validate_data(data) is False

    def test_validate_data_constant(self, analyzer):
        assert analyzer.validate_data(np.ones(20)) is False

    def test_validate_data_exception(self, analyzer):
        assert analyzer.validate_data("not_array") is False

    def test_analyze(self, analyzer, stable_data):
        metrics = analyzer.analyze(stable_data)
        assert isinstance(metrics, StabilityMetrics)
        assert metrics.sample_size == len(stable_data)
        assert 0 <= metrics.power_stability_coefficient <= 1

    def test_analyze_invalid_data(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.analyze([1.0, 2.0])

    def test_power_stability_zero_mean(self, analyzer):
        data = np.array([1, -1] * 10, dtype=float)
        result = analyzer._calculate_power_stability(data)
        assert isinstance(result, float)

    def test_power_variation_zero_mean(self, analyzer):
        data = np.zeros(10)
        assert analyzer._calculate_power_variation(data) == 0.0

    def test_trend_slope(self, analyzer):
        data = np.arange(20, dtype=float)
        slope = analyzer._calculate_trend_slope(data)
        assert slope == pytest.approx(1.0, abs=0.01)

    def test_autocorrelation_short(self, analyzer):
        assert analyzer._calculate_autocorrelation(np.array([1.0])) == 0.0

    def test_autocorrelation_normal(self, analyzer, stable_data):
        ac = analyzer._calculate_autocorrelation(stable_data)
        assert -1.0 <= ac <= 1.0

    def test_efficiency_stability(self, analyzer, stable_data):
        es = analyzer._calculate_efficiency_stability(stable_data)
        assert 0.0 <= es <= 1.0

    def test_degradation_rate_short(self, analyzer):
        assert analyzer._calculate_degradation_rate(np.array([1.0])) == 0.0

    def test_degradation_rate_zero_mean(self, analyzer):
        data = np.zeros(20)
        assert analyzer._calculate_degradation_rate(data) == 0.0

    def test_degradation_rate_positive_slope(self, analyzer):
        data = np.arange(20, dtype=float)
        assert analyzer._calculate_degradation_rate(data) == 0.0

    def test_degradation_rate_negative_slope(self, analyzer):
        data = np.arange(20, 0, -1, dtype=float)
        rate = analyzer._calculate_degradation_rate(data)
        assert rate >= 0.0

    def test_recovery_factor_short(self, analyzer):
        assert analyzer._calculate_recovery_factor(np.array([1, 2, 3])) == 1.0

    def test_recovery_factor_no_minima(self, analyzer):
        data = np.arange(20, dtype=float)
        assert analyzer._calculate_recovery_factor(data) == 1.0

    def test_recovery_factor_with_dips(self, analyzer):
        data = np.array([10, 8, 3, 7, 10, 9, 2, 8, 10, 9, 8, 10, 11, 12, 13])
        rf = analyzer._calculate_recovery_factor(data)
        assert isinstance(rf, float)

    def test_biofilm_health(self, analyzer, stable_data):
        health = analyzer._estimate_biofilm_health(stable_data)
        assert 0.0 <= health <= 1.0

    def test_biofilm_health_zero_mean(self, analyzer):
        data = np.zeros(20)
        data[0] = 1  # avoid constant
        health = analyzer._estimate_biofilm_health(data)
        assert isinstance(health, float)

    def test_biofilm_thickness_stability(self, analyzer, stable_data):
        ts = analyzer._calculate_biofilm_thickness_stability(stable_data)
        assert 0.0 <= ts <= 1.0

    def test_biofilm_thickness_zero_max(self, analyzer):
        data = np.zeros(20)
        data[0] = 0.001
        ts = analyzer._calculate_biofilm_thickness_stability(data)
        assert isinstance(ts, float)

    def test_biofilm_conductivity_stability(self, analyzer, stable_data):
        cs = analyzer._calculate_biofilm_conductivity_stability(stable_data)
        assert 0.0 <= cs <= 1.0

    def test_data_quality(self, analyzer, stable_data):
        dq = analyzer._assess_data_quality(stable_data)
        assert 0.0 <= dq <= 1.0

    def test_confidence_interval(self, analyzer, stable_data):
        ci = analyzer._calculate_confidence_interval(stable_data)
        assert ci[0] < ci[1]

    def test_confidence_interval_custom_level(self, analyzer, stable_data):
        ci = analyzer._calculate_confidence_interval(stable_data, 0.90)
        assert ci[0] < ci[1]


class TestStabilityFramework:
    def test_init(self, framework):
        assert framework.analyzer is not None
        assert framework.thresholds is not None

    def test_analyze_stability(self, framework, stable_data):
        metrics = framework.analyze_stability(stable_data)
        assert isinstance(metrics, StabilityMetrics)
        assert len(framework.analysis_history) == 1

    def test_analyze_stability_failure(self, framework):
        with pytest.raises(ValueError):
            framework.analyze_stability([1.0, 2.0])

    def test_get_stability_trends_insufficient(self, framework):
        assert framework.get_stability_trends() is None

    def test_get_stability_trends(self, framework, stable_data):
        framework.analyze_stability(stable_data)
        framework.analyze_stability(stable_data + 1)
        trends = framework.get_stability_trends()
        assert trends is not None
        assert "timestamps" in trends
        assert len(trends["overall_stability"]) == 2

    def test_export_json(self, framework, stable_data, tmp_dir):
        framework.analyze_stability(stable_data)
        out = Path(tmp_dir) / "results.json"
        framework.export_analysis_results(out, "json")
        assert out.exists()

    def test_export_csv(self, framework, stable_data, tmp_dir):
        framework.analyze_stability(stable_data)
        out = Path(tmp_dir) / "results.csv"
        framework.export_analysis_results(out, "csv")
        assert out.exists()

    def test_export_unsupported(self, framework, stable_data, tmp_dir):
        framework.analyze_stability(stable_data)
        with pytest.raises(ValueError):
            framework.export_analysis_results(Path(tmp_dir) / "x", "xyz")

    def test_export_empty(self, framework, tmp_dir):
        with pytest.raises(ValueError):
            framework.export_analysis_results(Path(tmp_dir) / "x", "json")


class TestFactoryFunctions:
    def test_create_statistical_analyzer(self):
        a = create_statistical_analyzer()
        assert isinstance(a, StatisticalStabilityAnalyzer)

    def test_create_default_framework(self):
        f = create_default_framework()
        assert isinstance(f, StabilityFramework)

    def test_create_default_framework_unknown_type(self):
        with pytest.raises(ValueError):
            create_default_framework(analyzer_type="unknown")
