"""Tests for biofilm_health_monitor.py."""
import os
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sensing_models.sensor_fusion import BacterialSpecies

from biofilm_health_monitor import (
    BiofimHealthOptimizer,
    HealthAlert,
    HealthMetrics,
    HealthStatus,
    HealthTrend,
    InterventionRecommendation,
    PredictiveBiofimHealthMonitor,
    create_predictive_health_monitor,
)


def _make_measurement(thickness=15.0, conductivity=0.05, ts=1.0,
                      confidence=0.9, agreement=0.85):
    m = MagicMock()
    m.thickness_um = thickness
    m.conductivity_S_per_m = conductivity
    m.timestamp = ts
    m.fusion_confidence = confidence
    m.sensor_agreement = agreement
    return m


def _make_growth(phase="exponential", rate=0.5, confidence=0.8):
    g = MagicMock()
    g.growth_phase = phase
    g.growth_rate_um_per_hour = rate
    g.pattern_confidence = confidence
    return g


class TestHealthEnums:
    def test_health_status_values(self):
        assert HealthStatus.EXCELLENT.value == "excellent"
        assert HealthStatus.CRITICAL.value == "critical"

    def test_health_trend_values(self):
        assert HealthTrend.IMPROVING.value == "improving"
        assert HealthTrend.VOLATILE.value == "volatile"


class TestBiofimHealthOptimizer:
    def test_init_mixed(self):
        opt = BiofimHealthOptimizer()
        assert opt.species == BacterialSpecies.MIXED
        assert opt.optimal_parameters["optimal_thickness_um"] == 15.0

    def test_init_geobacter(self):
        opt = BiofimHealthOptimizer(species=BacterialSpecies.GEOBACTER)
        assert opt.optimal_parameters["optimal_thickness_um"] == 12.0

    def test_init_shewanella(self):
        opt = BiofimHealthOptimizer(species=BacterialSpecies.SHEWANELLA)
        assert opt.optimal_parameters["optimal_thickness_um"] == 18.0

    def test_init_with_config_thresholds(self):
        cfg = MagicMock()
        cfg.health_thresholds = {"thickness_optimal_um": 20.0}
        opt = BiofimHealthOptimizer(config=cfg)
        assert opt.health_thresholds["thickness_optimal_um"] == 20.0

    def test_init_config_no_thresholds(self):
        cfg = MagicMock(spec=[])
        opt = BiofimHealthOptimizer(config=cfg)
        assert opt.health_thresholds["thickness_optimal_um"] == 15.0

    def test_thickness_health_optimal(self):
        opt = BiofimHealthOptimizer()
        h = opt.calculate_thickness_health(15.0, 0.0)
        assert h > 0.9

    def test_thickness_health_excessive(self):
        opt = BiofimHealthOptimizer()
        h = opt.calculate_thickness_health(40.0, 0.0)
        assert h < 0.5

    def test_thickness_health_positive_trend(self):
        opt = BiofimHealthOptimizer()
        h = opt.calculate_thickness_health(15.0, 0.2)
        assert h > 0.9

    def test_thickness_health_negative_trend(self):
        opt = BiofimHealthOptimizer()
        h = opt.calculate_thickness_health(15.0, -0.2)
        assert 0.0 <= h <= 1.0

    def test_thickness_health_zero_optimal(self):
        opt = BiofimHealthOptimizer()
        opt.optimal_parameters["optimal_thickness_um"] = 0.0
        h = opt.calculate_thickness_health(5.0, 0.0)
        assert 0.0 <= h <= 1.0

    def test_conductivity_health_optimal(self):
        opt = BiofimHealthOptimizer()
        h = opt.calculate_conductivity_health(0.05, 0.0)
        assert h > 0.9

    def test_conductivity_health_zero(self):
        opt = BiofimHealthOptimizer()
        h = opt.calculate_conductivity_health(0.0, 0.0)
        assert h == 0.0

    def test_conductivity_health_zero_optimal(self):
        opt = BiofimHealthOptimizer()
        opt.optimal_parameters["optimal_conductivity_S_per_m"] = 0.0
        h = opt.calculate_conductivity_health(0.05, 0.0)
        assert h == 0.5

    def test_conductivity_positive_trend(self):
        opt = BiofimHealthOptimizer()
        h = opt.calculate_conductivity_health(0.05, 0.01)
        assert h > 0.9

    def test_conductivity_negative_trend(self):
        opt = BiofimHealthOptimizer()
        h = opt.calculate_conductivity_health(0.05, -0.01)
        assert 0.0 <= h <= 1.0

    def test_growth_health_exponential(self):
        opt = BiofimHealthOptimizer()
        g = _make_growth("exponential", 0.5, 0.8)
        h = opt.calculate_growth_health(g)
        assert 0.0 <= h <= 1.0

    def test_growth_health_decline(self):
        opt = BiofimHealthOptimizer()
        g = _make_growth("decline", 0.1, 0.8)
        h = opt.calculate_growth_health(g)
        assert h < 0.8

    def test_growth_health_unknown_phase(self):
        opt = BiofimHealthOptimizer()
        g = _make_growth("alien_phase", 0.5, 0.8)
        h = opt.calculate_growth_health(g)
        assert 0.0 <= h <= 1.0

    def test_growth_health_low_confidence(self):
        opt = BiofimHealthOptimizer()
        g = _make_growth("exponential", 0.5, 0.1)
        h = opt.calculate_growth_health(g)
        assert h == 0.5

    def test_growth_health_none(self):
        opt = BiofimHealthOptimizer()
        h = opt.calculate_growth_health(None)
        assert h == 0.5

    def test_growth_health_zero_optimal_rate(self):
        opt = BiofimHealthOptimizer()
        opt.optimal_parameters["optimal_growth_rate_um_per_h"] = 0.0
        g = _make_growth("stationary", 0.0, 0.8)
        h = opt.calculate_growth_health(g)
        assert 0.0 <= h <= 1.0

    def test_growth_large_rate_deviation(self):
        opt = BiofimHealthOptimizer()
        g = _make_growth("exponential", 5.0, 0.9)
        h = opt.calculate_growth_health(g)
        assert 0.0 <= h <= 1.0

    def test_assess_risks_nominal(self):
        opt = BiofimHealthOptimizer()
        f, d, s = opt.assess_risks(15.0, 0.05, 0.5, 0.8)
        assert 0.0 <= f <= 1.0
        assert 0.0 <= d <= 1.0
        assert 0.0 <= s <= 1.0

    def test_assess_risks_high_fouling(self):
        opt = BiofimHealthOptimizer()
        f, d, s = opt.assess_risks(35.0, 0.001, 0.5, 0.8)
        assert f > 0.0

    def test_assess_risks_high_detachment(self):
        opt = BiofimHealthOptimizer()
        f, d, s = opt.assess_risks(15.0, 0.05, 2.0, 0.1)
        assert d > 0.0

    def test_assess_risks_stagnation(self):
        opt = BiofimHealthOptimizer()
        f, d, s = opt.assess_risks(15.0, 0.01, 0.01, 0.99)
        assert s > 0.0


class TestPredictiveMonitor:
    @pytest.fixture
    def monitor(self):
        return PredictiveBiofimHealthMonitor()

    def test_init(self, monitor):
        assert monitor.current_health is None
        assert len(monitor.health_history) == 0

    def test_assess_health_basic(self, monitor):
        m = _make_measurement()
        h = monitor.assess_health(m)
        assert isinstance(h, HealthMetrics)
        assert 0.0 <= h.overall_health_score <= 1.0

    def test_assess_with_growth(self, monitor):
        m = _make_measurement()
        g = _make_growth()
        h = monitor.assess_health(m, growth_pattern=g)
        assert h.growth_health > 0.0

    def test_assess_with_anomalies(self, monitor):
        m = _make_measurement()
        anom = MagicMock()
        anom.severity = "high"
        h = monitor.assess_health(m, anomalies=[anom])
        assert h.overall_health_score < 1.0

    def test_assess_critical_anomaly(self, monitor):
        m = _make_measurement()
        anom = MagicMock()
        anom.severity = "critical"
        h = monitor.assess_health(m, anomalies=[anom, anom, anom, anom])
        assert h.overall_health_score < 1.0

    def test_measurement_stability_insufficient(self, monitor):
        s = monitor._calculate_measurement_stability()
        assert s == 0.5

    def test_measurement_stability_with_data(self, monitor):
        for i in range(10):
            monitor.measurement_history.append(
                _make_measurement(15.0 + i * 0.01, ts=float(i))
            )
        s = monitor._calculate_measurement_stability()
        assert 0.0 <= s <= 1.0

    def test_thickness_trend_insufficient(self, monitor):
        assert monitor._calculate_thickness_trend() == 0.0

    def test_thickness_trend_with_data(self, monitor):
        for i in range(10):
            monitor.measurement_history.append(
                _make_measurement(15.0 + i * 0.1, ts=float(i))
            )
        t = monitor._calculate_thickness_trend()
        assert isinstance(t, float)

    def test_thickness_trend_same_timestamps(self, monitor):
        for _ in range(10):
            monitor.measurement_history.append(
                _make_measurement(15.0, ts=1.0)
            )
        assert monitor._calculate_thickness_trend() == 0.0

    def test_conductivity_trend_insufficient(self, monitor):
        assert monitor._calculate_conductivity_trend() == 0.0

    def test_conductivity_trend_with_data(self, monitor):
        for i in range(10):
            monitor.measurement_history.append(
                _make_measurement(conductivity=0.05 + i * 0.001, ts=float(i))
            )
        t = monitor._calculate_conductivity_trend()
        assert isinstance(t, float)

    def test_conductivity_trend_same_timestamps(self, monitor):
        for _ in range(10):
            monitor.measurement_history.append(
                _make_measurement(conductivity=0.05, ts=1.0)
            )
        assert monitor._calculate_conductivity_trend() == 0.0

    def test_categorize_health_status(self, monitor):
        assert monitor._categorize_health_status(0.95) == HealthStatus.EXCELLENT
        assert monitor._categorize_health_status(0.8) == HealthStatus.GOOD
        assert monitor._categorize_health_status(0.6) == HealthStatus.FAIR
        assert monitor._categorize_health_status(0.4) == HealthStatus.POOR
        assert monitor._categorize_health_status(0.1) == HealthStatus.CRITICAL

    def test_health_trend_unknown(self, monitor):
        assert monitor._calculate_health_trend() == HealthTrend.UNKNOWN

    def test_health_trend_with_history(self, monitor):
        for i in range(10):
            hm = MagicMock()
            hm.overall_health_score = 0.7 + i * 0.01
            monitor.health_history.append(hm)
        t = monitor._calculate_health_trend()
        assert t in [HealthTrend.IMPROVING, HealthTrend.STABLE,
                     HealthTrend.DECLINING, HealthTrend.VOLATILE]

    def test_health_trend_volatile(self, monitor):
        for i in range(10):
            hm = MagicMock()
            hm.overall_health_score = 0.3 if i % 2 == 0 else 0.9
            monitor.health_history.append(hm)
        t = monitor._calculate_health_trend()
        assert t == HealthTrend.VOLATILE

    def test_health_trend_declining(self, monitor):
        for i in range(10):
            hm = MagicMock()
            hm.overall_health_score = 0.9 - i * 0.05
            monitor.health_history.append(hm)
        t = monitor._calculate_health_trend()
        assert t in [HealthTrend.DECLINING, HealthTrend.VOLATILE]

    def test_predict_health_24h_improving(self, monitor):
        h = monitor._predict_health_24h(0.8, HealthTrend.IMPROVING)
        assert 0.0 <= h <= 1.0

    def test_predict_health_24h_declining(self, monitor):
        h = monitor._predict_health_24h(0.8, HealthTrend.DECLINING)
        assert h < 0.8

    def test_predict_health_24h_volatile(self, monitor):
        h = monitor._predict_health_24h(0.8, HealthTrend.VOLATILE)
        assert 0.0 <= h <= 1.0

    def test_predict_health_24h_stable(self, monitor):
        h = monitor._predict_health_24h(0.8, HealthTrend.STABLE)
        assert 0.0 <= h <= 1.0

    def test_predict_intervention_immediate(self, monitor):
        t = monitor._predict_intervention_time(0.3, HealthTrend.DECLINING)
        assert t == 0.0

    def test_predict_intervention_declining(self, monitor):
        for i in range(10):
            hm = MagicMock()
            hm.overall_health_score = 0.9 - i * 0.05
            monitor.health_history.append(hm)
        t = monitor._predict_intervention_time(0.6, HealthTrend.DECLINING)
        assert t is not None and t >= 0

    def test_predict_intervention_volatile(self, monitor):
        t = monitor._predict_intervention_time(0.6, HealthTrend.VOLATILE)
        assert t == 12.0

    def test_predict_intervention_stable(self, monitor):
        t = monitor._predict_intervention_time(0.8, HealthTrend.STABLE)
        assert t is None

    def test_predict_intervention_declining_no_history(self, monitor):
        t = monitor._predict_intervention_time(0.6, HealthTrend.DECLINING)
        assert t is None


class TestGenerateAlerts:
    @pytest.fixture
    def monitor_with_health(self):
        mon = PredictiveBiofimHealthMonitor()
        mon.last_assessment_time = 100.0
        return mon

    def test_critical_health_alert(self, monitor_with_health):
        hm = MagicMock(spec=HealthMetrics)
        hm.health_status = HealthStatus.CRITICAL
        hm.overall_health_score = 0.1
        hm.health_trend = HealthTrend.DECLINING
        hm.predicted_intervention_time = None
        hm.fouling_risk = 0.0
        hm.detachment_risk = 0.0
        hm.stagnation_risk = 0.0
        hm.assessment_confidence = 0.9
        hm.prediction_confidence = 0.7
        alerts = monitor_with_health.generate_alerts(hm)
        assert any(a.severity == "critical" for a in alerts)

    def test_declining_health_alert(self, monitor_with_health):
        hm = MagicMock(spec=HealthMetrics)
        hm.health_status = HealthStatus.POOR
        hm.overall_health_score = 0.35
        hm.health_trend = HealthTrend.DECLINING
        hm.predicted_intervention_time = None
        hm.fouling_risk = 0.0
        hm.detachment_risk = 0.0
        hm.stagnation_risk = 0.0
        hm.assessment_confidence = 0.8
        alerts = monitor_with_health.generate_alerts(hm)
        assert any(a.alert_type == "health_decline" for a in alerts)

    def test_intervention_needed_soon(self, monitor_with_health):
        hm = MagicMock(spec=HealthMetrics)
        hm.health_status = HealthStatus.FAIR
        hm.overall_health_score = 0.55
        hm.health_trend = HealthTrend.STABLE
        hm.predicted_intervention_time = 0.5
        hm.fouling_risk = 0.0
        hm.detachment_risk = 0.0
        hm.stagnation_risk = 0.0
        hm.prediction_confidence = 0.8
        hm.assessment_confidence = 0.8
        alerts = monitor_with_health.generate_alerts(hm)
        assert any(a.alert_type == "intervention_needed" for a in alerts)

    def test_intervention_high_severity(self, monitor_with_health):
        hm = MagicMock(spec=HealthMetrics)
        hm.health_status = HealthStatus.FAIR
        hm.overall_health_score = 0.55
        hm.health_trend = HealthTrend.STABLE
        hm.predicted_intervention_time = 3.0
        hm.fouling_risk = 0.0
        hm.detachment_risk = 0.0
        hm.stagnation_risk = 0.0
        hm.prediction_confidence = 0.8
        hm.assessment_confidence = 0.8
        alerts = monitor_with_health.generate_alerts(hm)
        intervention_alerts = [a for a in alerts if a.alert_type == "intervention_needed"]
        if intervention_alerts:
            assert intervention_alerts[0].severity == "high"

    def test_risk_alerts(self, monitor_with_health):
        hm = MagicMock(spec=HealthMetrics)
        hm.health_status = HealthStatus.GOOD
        hm.overall_health_score = 0.75
        hm.health_trend = HealthTrend.STABLE
        hm.predicted_intervention_time = None
        hm.fouling_risk = 0.9
        hm.detachment_risk = 0.9
        hm.stagnation_risk = 0.9
        hm.assessment_confidence = 0.9
        alerts = monitor_with_health.generate_alerts(hm)
        assert any(a.alert_type == "risk_assessment" for a in alerts)

    def test_anomaly_alerts(self, monitor_with_health):
        hm = MagicMock(spec=HealthMetrics)
        hm.health_status = HealthStatus.GOOD
        hm.overall_health_score = 0.75
        hm.health_trend = HealthTrend.STABLE
        hm.predicted_intervention_time = None
        hm.fouling_risk = 0.0
        hm.detachment_risk = 0.0
        hm.stagnation_risk = 0.0
        hm.assessment_confidence = 0.9
        anom = MagicMock()
        anom.severity = "critical"
        anom.affected_sensors = ["eis"]
        anom.recommended_action = "recalibrate"
        anom.confidence = 0.9
        alerts = monitor_with_health.generate_alerts(hm, anomalies=[anom])
        assert any(a.alert_type == "anomaly_detected" for a in alerts)

    def test_old_alerts_cleaned(self, monitor_with_health):
        old = HealthAlert(
            timestamp=50.0, alert_type="old", severity="low",
            message="old", affected_components=[], recommended_actions=[],
            predicted_consequences="", confidence=0.5,
        )
        monitor_with_health.alert_history = [old]
        hm = MagicMock(spec=HealthMetrics)
        hm.health_status = HealthStatus.GOOD
        hm.overall_health_score = 0.8
        hm.health_trend = HealthTrend.STABLE
        hm.predicted_intervention_time = None
        hm.fouling_risk = 0.0
        hm.detachment_risk = 0.0
        hm.stagnation_risk = 0.0
        hm.assessment_confidence = 0.9
        monitor_with_health.generate_alerts(hm)
        assert old not in monitor_with_health.alert_history

    def test_no_alerts_healthy(self, monitor_with_health):
        hm = MagicMock(spec=HealthMetrics)
        hm.health_status = HealthStatus.EXCELLENT
        hm.overall_health_score = 0.95
        hm.health_trend = HealthTrend.STABLE
        hm.predicted_intervention_time = None
        hm.fouling_risk = 0.0
        hm.detachment_risk = 0.0
        hm.stagnation_risk = 0.0
        hm.assessment_confidence = 0.95
        alerts = monitor_with_health.generate_alerts(hm)
        assert len(alerts) == 0


class TestInterventionRecommendations:
    @pytest.fixture
    def monitor(self):
        return PredictiveBiofimHealthMonitor()

    def test_low_health_recommendations(self, monitor):
        hm = MagicMock(spec=HealthMetrics)
        hm.overall_health_score = 0.2
        hm.thickness_health = 0.3
        hm.conductivity_health = 0.3
        hm.growth_health = 0.3
        hm.fouling_risk = 0.8
        hm.detachment_risk = 0.8
        hm.stagnation_risk = 0.8
        recs = monitor.generate_intervention_recommendations(hm)
        assert len(recs) > 0
        assert recs[0].urgency == "immediate"

    def test_moderate_health(self, monitor):
        hm = MagicMock(spec=HealthMetrics)
        hm.overall_health_score = 0.35
        hm.thickness_health = 0.6
        hm.conductivity_health = 0.6
        hm.growth_health = 0.6
        hm.fouling_risk = 0.3
        hm.detachment_risk = 0.3
        hm.stagnation_risk = 0.3
        recs = monitor.generate_intervention_recommendations(hm)
        assert len(recs) > 0

    def test_thickness_low_no_fouling(self, monitor):
        hm = MagicMock(spec=HealthMetrics)
        hm.overall_health_score = 0.6
        hm.thickness_health = 0.3
        hm.conductivity_health = 0.8
        hm.growth_health = 0.8
        hm.fouling_risk = 0.2
        hm.detachment_risk = 0.0
        hm.stagnation_risk = 0.0
        recs = monitor.generate_intervention_recommendations(hm)
        assert any(r.intervention_type == "substrate_change" for r in recs)

    def test_conductivity_low(self, monitor):
        hm = MagicMock(spec=HealthMetrics)
        hm.overall_health_score = 0.6
        hm.thickness_health = 0.8
        hm.conductivity_health = 0.3
        hm.growth_health = 0.8
        hm.fouling_risk = 0.0
        hm.detachment_risk = 0.0
        hm.stagnation_risk = 0.0
        recs = monitor.generate_intervention_recommendations(hm)
        assert any(r.intervention_type == "substrate_change" for r in recs)

    def test_growth_low(self, monitor):
        hm = MagicMock(spec=HealthMetrics)
        hm.overall_health_score = 0.6
        hm.thickness_health = 0.8
        hm.conductivity_health = 0.8
        hm.growth_health = 0.3
        hm.fouling_risk = 0.0
        hm.detachment_risk = 0.0
        hm.stagnation_risk = 0.0
        recs = monitor.generate_intervention_recommendations(hm)
        assert any(r.intervention_type == "monitoring" for r in recs)

    def test_healthy_no_recommendations(self, monitor):
        hm = MagicMock(spec=HealthMetrics)
        hm.overall_health_score = 0.95
        hm.thickness_health = 0.9
        hm.conductivity_health = 0.9
        hm.growth_health = 0.9
        hm.fouling_risk = 0.0
        hm.detachment_risk = 0.0
        hm.stagnation_risk = 0.0
        recs = monitor.generate_intervention_recommendations(hm)
        assert len(recs) == 0

    def test_max_5_recommendations(self, monitor):
        hm = MagicMock(spec=HealthMetrics)
        hm.overall_health_score = 0.15
        hm.thickness_health = 0.1
        hm.conductivity_health = 0.1
        hm.growth_health = 0.1
        hm.fouling_risk = 0.9
        hm.detachment_risk = 0.9
        hm.stagnation_risk = 0.9
        recs = monitor.generate_intervention_recommendations(hm)
        assert len(recs) <= 5


class TestDashboardData:
    def test_no_data(self):
        mon = PredictiveBiofimHealthMonitor()
        d = mon.get_health_dashboard_data()
        assert "error" in d

    def test_with_data(self):
        mon = PredictiveBiofimHealthMonitor()
        m = _make_measurement()
        mon.assess_health(m)
        d = mon.get_health_dashboard_data()
        assert "current_health" in d
        assert "component_health" in d
        assert "risk_assessment" in d
        assert "predictions" in d


class TestFactoryFunction:
    def test_create_monitor(self):
        mon = create_predictive_health_monitor()
        assert isinstance(mon, PredictiveBiofimHealthMonitor)
        assert mon.species == BacterialSpecies.MIXED

    def test_create_with_species(self):
        mon = create_predictive_health_monitor(
            species=BacterialSpecies.GEOBACTER
        )
        assert mon.species == BacterialSpecies.GEOBACTER


class TestTrendFallbacks:
    """Test linregress fallback paths."""

    def test_thickness_trend_no_scipy(self):
        mon = PredictiveBiofimHealthMonitor()
        for i in range(10):
            mon.measurement_history.append(
                _make_measurement(15.0 + i * 0.1, ts=float(i))
            )
        with patch('biofilm_health_monitor.SCIPY_AVAILABLE', False):
            t = mon._calculate_thickness_trend()
            assert isinstance(t, float)

    def test_conductivity_trend_no_scipy(self):
        mon = PredictiveBiofimHealthMonitor()
        for i in range(10):
            mon.measurement_history.append(
                _make_measurement(conductivity=0.05 + i * 0.001, ts=float(i))
            )
        with patch('biofilm_health_monitor.SCIPY_AVAILABLE', False):
            t = mon._calculate_conductivity_trend()
            assert isinstance(t, float)

    def test_thickness_trend_scipy_exception(self):
        mon = PredictiveBiofimHealthMonitor()
        for i in range(10):
            mon.measurement_history.append(
                _make_measurement(15.0, ts=float(i))
            )
        with patch('biofilm_health_monitor.SCIPY_AVAILABLE', True):
            with patch('biofilm_health_monitor.stats.linregress', side_effect=Exception("fail")):
                t = mon._calculate_thickness_trend()
                assert isinstance(t, float)

    def test_conductivity_trend_scipy_exception(self):
        mon = PredictiveBiofimHealthMonitor()
        for i in range(10):
            mon.measurement_history.append(
                _make_measurement(conductivity=0.05, ts=float(i))
            )
        with patch('biofilm_health_monitor.SCIPY_AVAILABLE', True):
            with patch('biofilm_health_monitor.stats.linregress', side_effect=Exception("fail")):
                t = mon._calculate_conductivity_trend()
                assert isinstance(t, float)

    def test_thickness_fallback_insufficient_data(self):
        mon = PredictiveBiofimHealthMonitor()
        for i in range(5):
            mon.measurement_history.append(
                _make_measurement(15.0, ts=float(i))
            )
        with patch('biofilm_health_monitor.SCIPY_AVAILABLE', False):
            t = mon._calculate_thickness_trend()
            assert isinstance(t, float)

    def test_conductivity_fallback_zero_dt(self):
        mon = PredictiveBiofimHealthMonitor()
        for i in range(10):
            ts = 0.0 if i < 9 else 0.0
            mon.measurement_history.append(
                _make_measurement(conductivity=0.05, ts=ts)
            )
        with patch('biofilm_health_monitor.SCIPY_AVAILABLE', False):
            t = mon._calculate_conductivity_trend()
            assert t == 0.0
