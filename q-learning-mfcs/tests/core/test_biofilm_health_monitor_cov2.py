"""Coverage boost tests for biofilm_health_monitor.py."""
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

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
from sensing_models.sensor_fusion import BacterialSpecies


def _make_fused_measurement(**kwargs):
    m = MagicMock()
    m.timestamp = kwargs.get("timestamp", 100.0)
    m.thickness_um = kwargs.get("thickness_um", 15.0)
    m.conductivity_S_per_m = kwargs.get("conductivity_S_per_m", 0.05)
    m.fusion_confidence = kwargs.get("fusion_confidence", 0.9)
    m.sensor_agreement = kwargs.get("sensor_agreement", 0.85)
    return m


def _make_growth_pattern(**kwargs):
    gp = MagicMock()
    gp.pattern_confidence = kwargs.get("pattern_confidence", 0.8)
    gp.growth_rate_um_per_hour = kwargs.get("growth_rate_um_per_hour", 0.5)
    gp.growth_phase = kwargs.get("growth_phase", "exponential")
    return gp


@pytest.mark.coverage_extra
class TestBiofimHealthOptimizer:
    def test_init_geobacter(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.GEOBACTER)
        assert opt.optimal_parameters["optimal_thickness_um"] == 12.0

    def test_init_shewanella(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.SHEWANELLA)
        assert opt.optimal_parameters["optimal_thickness_um"] == 18.0

    def test_init_mixed(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.MIXED)
        assert opt.optimal_parameters["optimal_thickness_um"] == 15.0

    def test_thickness_health_optimal(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.MIXED)
        h = opt.calculate_thickness_health(15.0, 0.0)
        assert h > 0.9

    def test_thickness_health_excessive(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.MIXED)
        h = opt.calculate_thickness_health(40.0, 0.0)
        assert h < 0.5

    def test_thickness_health_growing(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.MIXED)
        h = opt.calculate_thickness_health(15.0, 0.2)
        assert h > 0.9

    def test_thickness_health_declining(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.MIXED)
        h_decline = opt.calculate_thickness_health(15.0, -0.2)
        h_stable = opt.calculate_thickness_health(15.0, 0.0)
        assert h_decline < h_stable

    def test_conductivity_health_optimal(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.MIXED)
        h = opt.calculate_conductivity_health(0.05, 0.0)
        assert h > 0.9

    def test_conductivity_health_zero(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.MIXED)
        h = opt.calculate_conductivity_health(0.0, 0.0)
        assert h == 0.0

    def test_conductivity_health_improving(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.MIXED)
        h = opt.calculate_conductivity_health(0.05, 0.01)
        assert h > 0.9

    def test_growth_health_exponential(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.MIXED)
        gp = _make_growth_pattern(growth_phase="exponential")
        h = opt.calculate_growth_health(gp)
        assert h > 0.5

    def test_growth_health_decline(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.MIXED)
        gp = _make_growth_pattern(growth_phase="decline")
        h = opt.calculate_growth_health(gp)
        assert h < 0.5

    def test_growth_health_low_confidence(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.MIXED)
        gp = _make_growth_pattern(pattern_confidence=0.1)
        h = opt.calculate_growth_health(gp)
        assert h == 0.5

    def test_growth_health_none(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.MIXED)
        h = opt.calculate_growth_health(None)
        assert h == 0.5

    def test_assess_risks(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.MIXED)
        f, d, s = opt.assess_risks(
            thickness=28.0,
            conductivity=0.01,
            growth_rate=0.01,
            measurement_stability=0.5,
        )
        assert 0 <= f <= 1
        assert 0 <= d <= 1
        assert 0 <= s <= 1

    def test_assess_risks_high_fouling(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.MIXED)
        f, _, _ = opt.assess_risks(35.0, 0.001, 0.5, 0.8)
        assert f > 0.3

    def test_assess_risks_detachment(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.MIXED)
        _, d, _ = opt.assess_risks(15.0, 0.05, 2.0, 0.1)
        assert d > 0.3

    def test_assess_risks_stagnation(self):
        opt = BiofimHealthOptimizer(BacterialSpecies.MIXED)
        _, _, s = opt.assess_risks(15.0, 0.01, 0.01, 0.99)
        assert s > 0.5


@pytest.mark.coverage_extra
class TestPredictiveHealthMonitor:
    @pytest.fixture
    def monitor(self):
        return PredictiveBiofimHealthMonitor(BacterialSpecies.MIXED)

    def test_init(self, monitor):
        assert monitor.current_health is None
        assert len(monitor.health_history) == 0

    def test_assess_health(self, monitor):
        fm = _make_fused_measurement()
        gp = _make_growth_pattern()
        metrics = monitor.assess_health(fm, gp)
        assert isinstance(metrics, HealthMetrics)
        assert 0 <= metrics.overall_health_score <= 1

    def test_assess_health_with_anomalies(self, monitor):
        fm = _make_fused_measurement()
        anomaly = MagicMock()
        anomaly.severity = "critical"
        metrics = monitor.assess_health(fm, anomalies=[anomaly])
        assert metrics.overall_health_score < 1.0

    def test_categorize_health_status(self, monitor):
        assert monitor._categorize_health_status(0.95) == HealthStatus.EXCELLENT
        assert monitor._categorize_health_status(0.8) == HealthStatus.GOOD
        assert monitor._categorize_health_status(0.6) == HealthStatus.FAIR
        assert monitor._categorize_health_status(0.4) == HealthStatus.POOR
        assert monitor._categorize_health_status(0.2) == HealthStatus.CRITICAL

    def test_health_trend_unknown(self, monitor):
        assert monitor._calculate_health_trend() == HealthTrend.UNKNOWN

    def test_health_trend_stable(self, monitor):
        for _ in range(10):
            m = MagicMock()
            m.overall_health_score = 0.8
            monitor.health_history.append(m)
        assert monitor._calculate_health_trend() == HealthTrend.STABLE

    def test_health_trend_improving(self, monitor):
        for i in range(10):
            m = MagicMock()
            m.overall_health_score = 0.5 + i * 0.05
            monitor.health_history.append(m)
        assert monitor._calculate_health_trend() == HealthTrend.IMPROVING

    def test_health_trend_declining(self, monitor):
        for i in range(10):
            m = MagicMock()
            m.overall_health_score = 0.9 - i * 0.05
            monitor.health_history.append(m)
        assert monitor._calculate_health_trend() == HealthTrend.DECLINING

    def test_predict_health_24h(self, monitor):
        p = monitor._predict_health_24h(0.8, HealthTrend.STABLE)
        assert 0 <= p <= 1

    def test_predict_health_improving(self, monitor):
        p = monitor._predict_health_24h(0.8, HealthTrend.IMPROVING)
        assert p > 0

    def test_predict_health_declining(self, monitor):
        p = monitor._predict_health_24h(0.8, HealthTrend.DECLINING)
        assert p >= 0

    def test_predict_intervention_immediate(self, monitor):
        t = monitor._predict_intervention_time(0.3, HealthTrend.STABLE)
        assert t == 0.0

    def test_predict_intervention_volatile(self, monitor):
        t = monitor._predict_intervention_time(0.8, HealthTrend.VOLATILE)
        assert t == 12.0

    def test_predict_intervention_stable(self, monitor):
        t = monitor._predict_intervention_time(0.8, HealthTrend.STABLE)
        assert t is None

    def test_predict_intervention_declining(self, monitor):
        for i in range(5):
            m = MagicMock()
            m.overall_health_score = 0.8 - i * 0.05
            monitor.health_history.append(m)
        t = monitor._predict_intervention_time(0.6, HealthTrend.DECLINING)
        assert t is not None and t >= 0

    def test_generate_alerts_critical(self, monitor):
        metrics = MagicMock()
        metrics.health_status = HealthStatus.CRITICAL
        metrics.overall_health_score = 0.2
        metrics.health_trend = HealthTrend.DECLINING
        metrics.predicted_intervention_time = None
        metrics.fouling_risk = 0.3
        metrics.detachment_risk = 0.3
        metrics.stagnation_risk = 0.3
        metrics.assessment_confidence = 0.9
        metrics.prediction_confidence = 0.7
        monitor.last_assessment_time = 100.0
        alerts = monitor.generate_alerts(metrics)
        assert len(alerts) > 0
        assert any(a.severity == "critical" for a in alerts)

    def test_generate_alerts_intervention_needed(self, monitor):
        metrics = MagicMock()
        metrics.health_status = HealthStatus.FAIR
        metrics.overall_health_score = 0.55
        metrics.health_trend = HealthTrend.DECLINING
        metrics.predicted_intervention_time = 0.5
        metrics.fouling_risk = 0.3
        metrics.detachment_risk = 0.3
        metrics.stagnation_risk = 0.3
        metrics.assessment_confidence = 0.8
        metrics.prediction_confidence = 0.6
        monitor.last_assessment_time = 100.0
        alerts = monitor.generate_alerts(metrics)
        assert any(a.alert_type == "intervention_needed" for a in alerts)

    def test_generate_alerts_high_risks(self, monitor):
        metrics = MagicMock()
        metrics.health_status = HealthStatus.GOOD
        metrics.overall_health_score = 0.7
        metrics.health_trend = HealthTrend.STABLE
        metrics.predicted_intervention_time = None
        metrics.fouling_risk = 0.9
        metrics.detachment_risk = 0.8
        metrics.stagnation_risk = 0.8
        metrics.assessment_confidence = 0.9
        metrics.prediction_confidence = 0.7
        monitor.last_assessment_time = 100.0
        alerts = monitor.generate_alerts(metrics)
        assert any(a.alert_type == "risk_assessment" for a in alerts)

    def test_generate_alerts_anomalies(self, monitor):
        metrics = MagicMock()
        metrics.health_status = HealthStatus.GOOD
        metrics.overall_health_score = 0.8
        metrics.health_trend = HealthTrend.STABLE
        metrics.predicted_intervention_time = None
        metrics.fouling_risk = 0.1
        metrics.detachment_risk = 0.1
        metrics.stagnation_risk = 0.1
        metrics.assessment_confidence = 0.9
        metrics.prediction_confidence = 0.7
        monitor.last_assessment_time = 100.0
        anomaly = MagicMock()
        anomaly.severity = "critical"
        anomaly.affected_sensors = ["thickness"]
        anomaly.recommended_action = "Check sensor"
        anomaly.confidence = 0.9
        alerts = monitor.generate_alerts(metrics, anomalies=[anomaly])
        assert any(a.alert_type == "anomaly_detected" for a in alerts)

    def test_intervention_recommendations_low_health(self, monitor):
        metrics = MagicMock()
        metrics.overall_health_score = 0.25
        metrics.thickness_health = 0.3
        metrics.conductivity_health = 0.3
        metrics.growth_health = 0.3
        metrics.fouling_risk = 0.8
        metrics.detachment_risk = 0.3
        metrics.stagnation_risk = 0.3
        recs = monitor.generate_intervention_recommendations(metrics)
        assert len(recs) > 0
        assert any(r.urgency == "immediate" for r in recs)

    def test_intervention_recommendations_detachment(self, monitor):
        metrics = MagicMock()
        metrics.overall_health_score = 0.6
        metrics.thickness_health = 0.7
        metrics.conductivity_health = 0.7
        metrics.growth_health = 0.7
        metrics.fouling_risk = 0.1
        metrics.detachment_risk = 0.9
        metrics.stagnation_risk = 0.1
        recs = monitor.generate_intervention_recommendations(metrics)
        assert any(r.intervention_type == "flow_adjustment" for r in recs)

    def test_intervention_recommendations_stagnation(self, monitor):
        metrics = MagicMock()
        metrics.overall_health_score = 0.6
        metrics.thickness_health = 0.7
        metrics.conductivity_health = 0.7
        metrics.growth_health = 0.7
        metrics.fouling_risk = 0.1
        metrics.detachment_risk = 0.1
        metrics.stagnation_risk = 0.9
        recs = monitor.generate_intervention_recommendations(metrics)
        assert any(r.intervention_type == "substrate_change" for r in recs)

    def test_dashboard_data_no_health(self, monitor):
        data = monitor.get_health_dashboard_data()
        assert "error" in data

    def test_dashboard_data_with_health(self, monitor):
        fm = _make_fused_measurement()
        monitor.assess_health(fm)
        data = monitor.get_health_dashboard_data()
        assert "current_health" in data
        assert "component_health" in data
        assert "risk_assessment" in data

    def test_measurement_stability_few(self, monitor):
        s = monitor._calculate_measurement_stability()
        assert s == 0.5

    def test_thickness_trend_few(self, monitor):
        t = monitor._calculate_thickness_trend()
        assert t == 0.0

    def test_conductivity_trend_few(self, monitor):
        t = monitor._calculate_conductivity_trend()
        assert t == 0.0


@pytest.mark.coverage_extra
class TestFactoryFunction:
    def test_create_monitor(self):
        m = create_predictive_health_monitor(BacterialSpecies.GEOBACTER)
        assert isinstance(m, PredictiveBiofimHealthMonitor)
        assert m.species == BacterialSpecies.GEOBACTER
