"""Tests for adaptive_mfc_controller.py - coverage target 98%+.

Covers ControlStrategy, AdaptationMode, ControlDecision, SystemState,
HealthAwareQLearning, AdaptiveMFCController, and factory function.
"""
import sys
import os
from unittest.mock import MagicMock, patch
from collections import defaultdict

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


# --- Mock the heavy dependencies before importing the module ---
mock_sensor_fusion = MagicMock()


class FakeFusedMeasurement:
    def __init__(self, **kw):
        self.eis_thickness = kw.get("eis_thickness", 10.0)
        self.qcm_thickness = kw.get("qcm_thickness", 11.0)
        self.conductivity_S_per_m = kw.get("conductivity_S_per_m", 0.1)
        self.biomass_density_g_per_L = kw.get("biomass_density_g_per_L", 0.5)
        self.fusion_confidence = kw.get("fusion_confidence", 0.85)
        self.sensor_agreement = kw.get("sensor_agreement", 0.9)
        self.cross_validation_error = kw.get("cross_validation_error", 0.05)
        self.eis_status = kw.get("eis_status", "good")
        self.qcm_status = kw.get("qcm_status", "good")
        self.thickness_um = kw.get("thickness_um", 10.5)
        self.timestamp = kw.get("timestamp", 1.0)


class FakePredictiveState:
    pass


class FakeAnomalyDetection:
    def __init__(self, severity="low"):
        self.severity = severity


mock_sensor_fusion.FusedMeasurement = FakeFusedMeasurement
mock_sensor_fusion.PredictiveState = FakePredictiveState
mock_sensor_fusion.AnomalyDetection = FakeAnomalyDetection
mock_sensor_fusion.create_advanced_sensor_fusion = MagicMock(
    return_value=MagicMock(
        fuse_measurements_with_prediction=MagicMock(
            return_value=(FakeFusedMeasurement(), None, [])
        ),
        analyze_biofilm_growth_pattern=MagicMock(return_value={}),
        get_system_health_assessment=MagicMock(return_value={}),
    )
)


class FakeHealthTrend:
    def __init__(self, v="stable"):
        self.value = v


class FakeHealthStatus:
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class FakeHealthMetrics:
    def __init__(self, **kw):
        self.overall_health_score = kw.get("overall_health_score", 0.8)
        self.health_trend = FakeHealthTrend(kw.get("health_trend", "stable"))
        self.health_status = kw.get("health_status", FakeHealthStatus.GOOD)
        self.thickness_health = kw.get("thickness_health", 0.9)
        self.conductivity_health = kw.get("conductivity_health", 0.85)
        self.growth_health = kw.get("growth_health", 0.7)
        self.predicted_health_24h = kw.get("predicted_health_24h", 0.82)
        self.fouling_risk = kw.get("fouling_risk", 0.1)
        self.detachment_risk = kw.get("detachment_risk", 0.1)
        self.stagnation_risk = kw.get("stagnation_risk", 0.1)
        self.assessment_confidence = kw.get("assessment_confidence", 0.9)
        self.prediction_confidence = kw.get("prediction_confidence", 0.8)


class FakeHealthAlert:
    def __init__(self, severity="warning", message="test"):
        self.severity = severity
        self.message = message


class FakeInterventionRecommendation:
    def __init__(self, urgency="low", success_probability=0.5):
        self.urgency = urgency
        self.success_probability = success_probability
        self.intervention_type = "flow_adjustment"
        self.description = "Adjust flow"
        self.expected_benefit = 0.1


mock_health = MagicMock()
mock_health.HealthAlert = FakeHealthAlert
mock_health.HealthMetrics = FakeHealthMetrics
mock_health.HealthStatus = FakeHealthStatus
mock_health.InterventionRecommendation = FakeInterventionRecommendation
mock_health.create_predictive_health_monitor = MagicMock(
    return_value=MagicMock(
        assess_health=MagicMock(return_value=FakeHealthMetrics()),
        generate_alerts=MagicMock(return_value=[]),
        generate_intervention_recommendations=MagicMock(return_value=[]),
        get_health_dashboard_data=MagicMock(return_value={}),
    )
)

# Patch modules before import
with patch.dict(
    sys.modules,
    {
        "sensing_models.advanced_sensor_fusion": mock_sensor_fusion,
        "biofilm_health_monitor": mock_health,
    },
):
    from adaptive_mfc_controller import (
        AdaptationMode,
        AdaptiveMFCController,
        ControlDecision,
        ControlStrategy,
        HealthAwareQLearning,
        SystemState,
        create_adaptive_mfc_controller,
    )


def _make_fused(**kw):
    return FakeFusedMeasurement(**kw)


def _make_health(**kw):
    return FakeHealthMetrics(**kw)


def _make_system_state(**kw):
    defaults = dict(
        fused_measurement=_make_fused(),
        prediction=None,
        anomalies=[],
        health_metrics=_make_health(),
        health_alerts=[],
        flow_rate=15.0,
        inlet_concentration=10.0,
        outlet_concentration=8.0,
        current_density=0.5,
        power_output=0.1,
        current_strategy=ControlStrategy.BALANCED,
        adaptation_mode=AdaptationMode.MODERATE,
        intervention_active=False,
    )
    defaults.update(kw)
    return SystemState(**defaults)


# ==== Tests ====

class TestEnums:
    def test_control_strategy_values(self):
        assert ControlStrategy.PERFORMANCE_FOCUSED.value == "performance_focused"
        assert ControlStrategy.HEALTH_FOCUSED.value == "health_focused"
        assert ControlStrategy.BALANCED.value == "balanced"
        assert ControlStrategy.CONSERVATIVE.value == "conservative"
        assert ControlStrategy.RECOVERY.value == "recovery"

    def test_adaptation_mode_values(self):
        assert AdaptationMode.AGGRESSIVE.value == "aggressive"
        assert AdaptationMode.MODERATE.value == "moderate"
        assert AdaptationMode.CONSERVATIVE.value == "conservative"
        assert AdaptationMode.DISABLED.value == "disabled"


class TestControlDecision:
    def test_creation(self):
        d = ControlDecision(
            action_index=0,
            action_description="Maintain",
            expected_outcome={"power_change": 0.0},
            confidence=0.8,
            rationale="Test",
            risk_assessment={"biofilm_damage": 0.0},
        )
        assert d.action_index == 0
        assert d.intervention_type is None

    def test_with_intervention(self):
        d = ControlDecision(
            action_index=7,
            action_description="Emergency",
            expected_outcome={},
            confidence=0.5,
            rationale="Critical",
            risk_assessment={},
            intervention_type="emergency_intervention",
        )
        assert d.intervention_type == "emergency_intervention"


class TestSystemState:
    def test_creation(self):
        s = _make_system_state()
        assert s.flow_rate == 15.0
        assert s.current_strategy == ControlStrategy.BALANCED


class TestHealthAwareQLearning:
    @pytest.fixture
    def ctrl(self):
        return HealthAwareQLearning()

    def test_init(self, ctrl):
        assert ctrl.health_weight == 0.4
        assert ctrl.power_weight == 0.6
        assert len(ctrl.actions) > 0

    def test_calculate_health_aware_reward_basic(self, ctrl):
        hm = _make_health()
        ss = _make_system_state()
        r = ctrl.calculate_health_aware_reward(1.0, hm, ss)
        assert isinstance(r, float)

    def test_reward_with_intervention_improving(self, ctrl):
        hm = _make_health(health_trend="improving")
        ss = _make_system_state()
        r = ctrl.calculate_health_aware_reward(1.0, hm, ss, intervention_active=True)
        assert isinstance(r, float)

    def test_reward_with_intervention_declining(self, ctrl):
        hm = _make_health(health_trend="declining")
        ss = _make_system_state()
        r = ctrl.calculate_health_aware_reward(1.0, hm, ss, intervention_active=True)
        assert isinstance(r, float)

    def test_health_reward_improving(self, ctrl):
        hm = _make_health(health_trend="improving")
        ss = _make_system_state()
        r = ctrl._calculate_health_reward(hm, ss)
        assert r > 0

    def test_health_reward_declining(self, ctrl):
        hm = _make_health(health_trend="declining")
        ss = _make_system_state()
        r = ctrl._calculate_health_reward(hm, ss)
        assert isinstance(r, float)

    def test_health_reward_volatile(self, ctrl):
        hm = _make_health(health_trend="volatile")
        ss = _make_system_state()
        r = ctrl._calculate_health_reward(hm, ss)
        assert isinstance(r, float)

    def test_health_reward_high_components(self, ctrl):
        hm = _make_health(
            thickness_health=0.9,
            conductivity_health=0.9,
            growth_health=0.9,
            predicted_health_24h=0.95,
        )
        ss = _make_system_state()
        r = ctrl._calculate_health_reward(hm, ss)
        assert r > 0

    def test_risk_penalty_high_fouling(self, ctrl):
        hm = _make_health(fouling_risk=0.8)
        p = ctrl._calculate_risk_penalty(hm)
        assert p > 0

    def test_risk_penalty_high_detachment(self, ctrl):
        hm = _make_health(detachment_risk=0.9)
        p = ctrl._calculate_risk_penalty(hm)
        assert p > 0

    def test_risk_penalty_high_stagnation(self, ctrl):
        hm = _make_health(stagnation_risk=0.8)
        p = ctrl._calculate_risk_penalty(hm)
        assert p > 0

    def test_risk_penalty_critical(self, ctrl):
        hm = _make_health(health_status=FakeHealthStatus.CRITICAL)
        p = ctrl._calculate_risk_penalty(hm)
        assert p >= 0.5

    def test_risk_penalty_poor(self, ctrl):
        hm = _make_health(health_status=FakeHealthStatus.POOR)
        p = ctrl._calculate_risk_penalty(hm)
        assert p >= 0.2

    def test_stability_bonus_high_confidence(self, ctrl):
        ss = _make_system_state(
            fused_measurement=_make_fused(
                fusion_confidence=0.9, sensor_agreement=0.9
            )
        )
        b = ctrl._calculate_stability_bonus(ss)
        assert b > 0

    def test_stability_bonus_no_anomalies(self, ctrl):
        ss = _make_system_state(anomalies=[])
        b = ctrl._calculate_stability_bonus(ss)
        assert b >= 0.05

    def test_stability_bonus_low_anomalies(self, ctrl):
        ss = _make_system_state(
            anomalies=[FakeAnomalyDetection(severity="low")]
        )
        b = ctrl._calculate_stability_bonus(ss)
        assert b >= 0.02

    def test_stability_bonus_high_anomalies(self, ctrl):
        ss = _make_system_state(
            anomalies=[FakeAnomalyDetection(severity="high")]
        )
        b = ctrl._calculate_stability_bonus(ss)
        assert isinstance(b, float)

    def test_adapt_parameters_critical(self, ctrl):
        hm = _make_health(health_status=FakeHealthStatus.CRITICAL)
        ss = _make_system_state(health_metrics=hm)
        ctrl.adapt_parameters(hm, ss)
        assert ctrl.learning_rate <= 0.8
        assert len(ctrl.adaptation_history) > 0

    def test_adapt_parameters_volatile(self, ctrl):
        hm = _make_health(health_trend="volatile")
        ss = _make_system_state(health_metrics=hm)
        ctrl.adapt_parameters(hm, ss)
        assert isinstance(ctrl.learning_rate, float)

    def test_adapt_parameters_normal(self, ctrl):
        hm = _make_health()
        ss = _make_system_state(health_metrics=hm)
        ctrl.adapt_parameters(hm, ss)
        assert isinstance(ctrl.learning_rate, float)

    def test_adapt_parameters_high_risk(self, ctrl):
        hm = _make_health(fouling_risk=0.9)
        ss = _make_system_state(health_metrics=hm)
        ctrl.adapt_parameters(hm, ss)
        assert ctrl.epsilon > 0

    def test_adapt_parameters_critical_anomalies(self, ctrl):
        hm = _make_health()
        ss = _make_system_state(
            health_metrics=hm,
            anomalies=[FakeAnomalyDetection(severity="critical")],
        )
        ctrl.adapt_parameters(hm, ss)

    def test_adapt_parameters_low_health(self, ctrl):
        hm = _make_health(overall_health_score=0.3)
        ss = _make_system_state(health_metrics=hm)
        ctrl.adapt_parameters(hm, ss)

    def test_get_adaptation_trigger_critical(self, ctrl):
        hm = _make_health(health_status=FakeHealthStatus.CRITICAL)
        ss = _make_system_state(health_metrics=hm)
        t = ctrl._get_adaptation_trigger(hm, ss)
        assert "critical_health" in t

    def test_get_adaptation_trigger_volatile(self, ctrl):
        hm = _make_health(health_trend="volatile")
        ss = _make_system_state(health_metrics=hm)
        t = ctrl._get_adaptation_trigger(hm, ss)
        assert "volatile_trend" in t

    def test_get_adaptation_trigger_anomalies(self, ctrl):
        hm = _make_health()
        ss = _make_system_state(
            anomalies=[FakeAnomalyDetection(severity="critical")]
        )
        t = ctrl._get_adaptation_trigger(hm, ss)
        assert "critical_anomalies" in t

    def test_get_adaptation_trigger_high_risk(self, ctrl):
        hm = _make_health(fouling_risk=0.9)
        ss = _make_system_state(health_metrics=hm)
        t = ctrl._get_adaptation_trigger(hm, ss)
        assert "high_risk" in t

    def test_get_adaptation_trigger_routine(self, ctrl):
        hm = _make_health()
        ss = _make_system_state(health_metrics=hm)
        t = ctrl._get_adaptation_trigger(hm, ss)
        assert t == "routine_adaptation"

    def test_system_state_to_qlearning_state(self, ctrl):
        ss = _make_system_state()
        result = ctrl._system_state_to_qlearning_state(ss)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_prepare_sensor_data(self, ctrl):
        ss = _make_system_state()
        data = ctrl._prepare_sensor_data(ss)
        assert "eis" in data
        assert "qcm" in data
        assert "fusion" in data

    def test_predict_action_outcomes_action1(self, ctrl):
        ss = _make_system_state()
        o = ctrl._predict_action_outcomes(1, ss)
        assert o["thickness_change"] == -0.5

    def test_predict_action_outcomes_action2(self, ctrl):
        ss = _make_system_state()
        o = ctrl._predict_action_outcomes(2, ss)
        assert o["thickness_change"] == 0.3

    def test_predict_action_outcomes_action5(self, ctrl):
        ss = _make_system_state()
        o = ctrl._predict_action_outcomes(5, ss)
        assert o["power_change"] == 0.1

    def test_predict_action_outcomes_action6(self, ctrl):
        ss = _make_system_state()
        o = ctrl._predict_action_outcomes(6, ss)
        assert o["health_change"] == 0.15

    def test_predict_action_outcomes_default(self, ctrl):
        ss = _make_system_state()
        o = ctrl._predict_action_outcomes(99, ss)
        assert o["power_change"] == 0.0

    def test_calculate_decision_confidence(self, ctrl):
        ss = _make_system_state()
        c = ctrl._calculate_decision_confidence(0, ss)
        assert 0.1 <= c <= 1.0

    def test_decision_confidence_critical(self, ctrl):
        ss = _make_system_state(
            health_metrics=_make_health(
                health_status=FakeHealthStatus.CRITICAL
            )
        )
        c = ctrl._calculate_decision_confidence(0, ss)
        assert c < 1.0

    def test_decision_confidence_poor(self, ctrl):
        ss = _make_system_state(
            health_metrics=_make_health(
                health_status=FakeHealthStatus.POOR
            )
        )
        c = ctrl._calculate_decision_confidence(0, ss)
        assert c < 1.0

    def test_decision_confidence_anomalies(self, ctrl):
        ss = _make_system_state(
            anomalies=[FakeAnomalyDetection(), FakeAnomalyDetection()]
        )
        c = ctrl._calculate_decision_confidence(0, ss)
        assert c < 1.0

    def test_assess_decision_risks_large_flow(self, ctrl):
        ss = _make_system_state()
        r = ctrl._assess_decision_risks(3, ss)
        assert r["biofilm_damage"] > 0

    def test_assess_decision_risks_emergency(self, ctrl):
        ss = _make_system_state()
        r = ctrl._assess_decision_risks(7, ss)
        assert r["performance_loss"] > 0

    def test_assess_decision_risks_substrate(self, ctrl):
        ss = _make_system_state()
        r = ctrl._assess_decision_risks(8, ss)
        assert r["system_instability"] > 0

    def test_assess_decision_risks_detachment(self, ctrl):
        ss = _make_system_state(
            health_metrics=_make_health(detachment_risk=0.9)
        )
        r = ctrl._assess_decision_risks(3, ss)
        assert r["biofilm_damage"] > 0.3

    def test_assess_decision_risks_many_anomalies(self, ctrl):
        ss = _make_system_state(
            anomalies=[FakeAnomalyDetection() for _ in range(3)]
        )
        r = ctrl._assess_decision_risks(0, ss)
        assert r["sensor_interference"] == 0.2

    def test_generate_decision_rationale_critical(self, ctrl):
        ss = _make_system_state(
            health_metrics=_make_health(
                health_status=FakeHealthStatus.CRITICAL
            )
        )
        d = ctrl._generate_decision_rationale(7, ss)
        assert d.intervention_type == "emergency_intervention"
        assert "intervention" in d.rationale.lower() or "critical" in d.rationale.lower()

    def test_generate_decision_rationale_poor_health(self, ctrl):
        ss = _make_system_state(
            health_metrics=_make_health(
                health_status=FakeHealthStatus.POOR
            )
        )
        d = ctrl._generate_decision_rationale(6, ss)
        assert d.intervention_type == "health_intervention"

    def test_generate_decision_rationale_risks(self, ctrl):
        ss = _make_system_state(
            health_metrics=_make_health(fouling_risk=0.9)
        )
        d = ctrl._generate_decision_rationale(0, ss)
        assert "fouling" in d.rationale.lower()

    def test_generate_decision_rationale_anomalies(self, ctrl):
        ss = _make_system_state(
            anomalies=[FakeAnomalyDetection()]
        )
        d = ctrl._generate_decision_rationale(0, ss)
        assert "anomal" in d.rationale.lower()

    def test_generate_decision_rationale_health_focused(self, ctrl):
        ss = _make_system_state(
            current_strategy=ControlStrategy.HEALTH_FOCUSED
        )
        d = ctrl._generate_decision_rationale(0, ss)
        assert "health" in d.rationale.lower()

    def test_generate_decision_rationale_performance_focused(self, ctrl):
        ss = _make_system_state(
            current_strategy=ControlStrategy.PERFORMANCE_FOCUSED
        )
        d = ctrl._generate_decision_rationale(0, ss)
        assert "performance" in d.rationale.lower()

    def test_generate_decision_rationale_declining(self, ctrl):
        ss = _make_system_state(
            health_metrics=_make_health(health_trend="declining")
        )
        d = ctrl._generate_decision_rationale(0, ss)
        assert "declining" in d.rationale.lower()

    def test_generate_decision_rationale_standard(self, ctrl):
        ss = _make_system_state()
        d = ctrl._generate_decision_rationale(0, ss)
        assert d.rationale == "Standard operational decision"

    def test_choose_health_aware_action(self, ctrl):
        ss = _make_system_state()
        d = ctrl.choose_health_aware_action(ss)
        assert isinstance(d, ControlDecision)
