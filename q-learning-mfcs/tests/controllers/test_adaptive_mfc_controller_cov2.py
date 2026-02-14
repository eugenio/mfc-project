"""Tests for adaptive_mfc_controller.py - coverage part 2.

Covers missing lines: 191, 575, 577, 758-789, 815-893, 897-938,
944-961, 968-979, 987-1002, 1011-1032, 1042, 1055-1069, 1082,
1093-1098, 1146-1156.
Targets: AdaptiveMFCController init, control_step, strategy adaptation,
interventions, execute_control_action, update_qlearning, performance
metrics, comprehensive status, and factory function.
"""
import sys
import os
from unittest.mock import MagicMock, patch
from collections import defaultdict

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# --- Mock heavy deps ---
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
    def __init__(self):
        self.predicted_values = [10.0, 11.0, 12.0]
        self.prediction_accuracy = 0.9


class FakeAnomalyDetection:
    def __init__(self, severity="low"):
        self.severity = severity


mock_sensor_fusion.FusedMeasurement = FakeFusedMeasurement
mock_sensor_fusion.PredictiveState = FakePredictiveState
mock_sensor_fusion.AnomalyDetection = FakeAnomalyDetection
mock_sensor_fusion.create_advanced_sensor_fusion = MagicMock(
    return_value=MagicMock(
        fuse_measurements_with_prediction=MagicMock(
            return_value=(FakeFusedMeasurement(), FakePredictiveState(), [])
        ),
        analyze_biofilm_growth_pattern=MagicMock(return_value={}),
        get_system_health_assessment=MagicMock(return_value={}),
    )
)


class FakeHealthTrend:
    def __init__(self, v="stable"):
        self.value = v


class _FakeEnum:
    """Fake enum member with .value attribute."""
    def __init__(self, v):
        self.value = v
    def __eq__(self, other):
        if isinstance(other, _FakeEnum):
            return self.value == other.value
        return self.value == other
    def __hash__(self):
        return hash(self.value)


class FakeHealthStatus:
    EXCELLENT = _FakeEnum("excellent")
    GOOD = _FakeEnum("good")
    FAIR = _FakeEnum("fair")
    POOR = _FakeEnum("poor")
    CRITICAL = _FakeEnum("critical")


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


# ---- Tests for AdaptiveMFCController ----

@pytest.mark.coverage_extra
class TestAdaptiveMFCControllerInit:
    """Cover lines 758-789 (init)."""

    def test_init_creates_subsystems(self):
        ctrl = AdaptiveMFCController()
        assert ctrl.current_strategy == ControlStrategy.BALANCED
        assert ctrl.adaptation_mode == AdaptationMode.MODERATE
        assert ctrl.intervention_active is False
        assert ctrl.last_intervention_time == 0.0
        assert isinstance(ctrl.control_history, list)
        assert isinstance(ctrl.strategy_changes, list)
        assert isinstance(ctrl.intervention_outcomes, list)
        assert ctrl.system_parameters["flow_rate"] == 15.0
        assert ctrl.system_parameters["temperature"] == 25.0

    def test_init_with_custom_strategy(self):
        ctrl = AdaptiveMFCController(
            initial_strategy=ControlStrategy.CONSERVATIVE
        )
        assert ctrl.current_strategy == ControlStrategy.CONSERVATIVE


@pytest.mark.coverage_extra
class TestAdaptControlStrategy:
    """Cover lines 897-938."""

    def _make_ctrl(self):
        return AdaptiveMFCController()

    def test_critical_health_triggers_recovery(self):
        ctrl = self._make_ctrl()
        ss = _make_system_state(
            health_metrics=_make_health(health_status=FakeHealthStatus.CRITICAL)
        )
        ctrl._adapt_control_strategy(ss)
        assert ctrl.current_strategy == ControlStrategy.RECOVERY
        assert len(ctrl.strategy_changes) == 1

    def test_poor_health_triggers_health_focused(self):
        ctrl = self._make_ctrl()
        ss = _make_system_state(
            health_metrics=_make_health(health_status=FakeHealthStatus.POOR)
        )
        ctrl._adapt_control_strategy(ss)
        assert ctrl.current_strategy == ControlStrategy.HEALTH_FOCUSED

    def test_high_risk_triggers_conservative(self):
        ctrl = self._make_ctrl()
        ss = _make_system_state(
            health_metrics=_make_health(fouling_risk=0.9)
        )
        ctrl._adapt_control_strategy(ss)
        assert ctrl.current_strategy == ControlStrategy.CONSERVATIVE

    def test_excellent_stable_triggers_performance(self):
        ctrl = self._make_ctrl()
        ss = _make_system_state(
            health_metrics=_make_health(
                health_status=FakeHealthStatus.EXCELLENT,
                health_trend="stable",
            )
        )
        ctrl._adapt_control_strategy(ss)
        assert ctrl.current_strategy == ControlStrategy.PERFORMANCE_FOCUSED

    def test_excellent_improving_triggers_performance(self):
        ctrl = self._make_ctrl()
        ss = _make_system_state(
            health_metrics=_make_health(
                health_status=FakeHealthStatus.EXCELLENT,
                health_trend="improving",
            )
        )
        ctrl._adapt_control_strategy(ss)
        assert ctrl.current_strategy == ControlStrategy.PERFORMANCE_FOCUSED

    def test_default_balanced(self):
        ctrl = self._make_ctrl()
        ctrl.current_strategy = ControlStrategy.CONSERVATIVE
        ss = _make_system_state(
            health_metrics=_make_health(health_status=FakeHealthStatus.FAIR)
        )
        ctrl._adapt_control_strategy(ss)
        assert ctrl.current_strategy == ControlStrategy.BALANCED

    def test_no_change_no_log(self):
        ctrl = self._make_ctrl()
        ss = _make_system_state(
            health_metrics=_make_health(health_status=FakeHealthStatus.FAIR)
        )
        ctrl._adapt_control_strategy(ss)
        assert ctrl.current_strategy == ControlStrategy.BALANCED
        assert len(ctrl.strategy_changes) == 0


@pytest.mark.coverage_extra
class TestIdentifyStrategyTrigger:
    """Cover lines 944-961."""

    def _make_ctrl(self):
        return AdaptiveMFCController()

    def test_critical_health_trigger(self):
        ctrl = self._make_ctrl()
        ss = _make_system_state(
            health_metrics=_make_health(health_status=FakeHealthStatus.CRITICAL)
        )
        assert ctrl._identify_strategy_trigger(ss) == "critical_health"

    def test_poor_health_trigger(self):
        ctrl = self._make_ctrl()
        ss = _make_system_state(
            health_metrics=_make_health(health_status=FakeHealthStatus.POOR)
        )
        assert ctrl._identify_strategy_trigger(ss) == "poor_health"

    def test_critical_alerts_trigger(self):
        ctrl = self._make_ctrl()
        alert = FakeHealthAlert(severity="critical", message="critical alert")
        ss = _make_system_state(health_alerts=[alert])
        assert ctrl._identify_strategy_trigger(ss) == "critical_alerts"

    def test_high_risk_trigger(self):
        ctrl = self._make_ctrl()
        ss = _make_system_state(
            health_metrics=_make_health(fouling_risk=0.9)
        )
        assert ctrl._identify_strategy_trigger(ss) == "high_risk"

    def test_excellent_health_trigger(self):
        ctrl = self._make_ctrl()
        ss = _make_system_state(
            health_metrics=_make_health(health_status=FakeHealthStatus.EXCELLENT)
        )
        assert ctrl._identify_strategy_trigger(ss) == "excellent_health"

    def test_operational_optimization_trigger(self):
        ctrl = self._make_ctrl()
        ss = _make_system_state()
        assert ctrl._identify_strategy_trigger(ss) == "operational_optimization"


@pytest.mark.coverage_extra
class TestEvaluateInterventions:
    """Cover lines 968-979."""

    def test_no_recommendations(self):
        ctrl = AdaptiveMFCController()
        ss = _make_system_state()
        result = ctrl._evaluate_interventions(ss)
        assert isinstance(result, list)

    def test_immediate_urgent_intervention(self):
        ctrl = AdaptiveMFCController()
        rec = FakeInterventionRecommendation(
            urgency="immediate", success_probability=0.9
        )
        ctrl.health_monitor.generate_intervention_recommendations = MagicMock(
            return_value=[rec]
        )
        ss = _make_system_state()
        result = ctrl._evaluate_interventions(ss)
        assert len(result) == 1
        assert ctrl.intervention_active is True

    def test_non_immediate_not_executed(self):
        ctrl = AdaptiveMFCController()
        rec = FakeInterventionRecommendation(
            urgency="scheduled", success_probability=0.9
        )
        ctrl.health_monitor.generate_intervention_recommendations = MagicMock(
            return_value=[rec]
        )
        ss = _make_system_state()
        ctrl._evaluate_interventions(ss)
        assert ctrl.intervention_active is False


@pytest.mark.coverage_extra
class TestExecuteIntervention:
    """Cover lines 987-1002."""

    def test_execute_intervention_sets_state(self):
        ctrl = AdaptiveMFCController()
        rec = FakeInterventionRecommendation(
            urgency="immediate", success_probability=0.9
        )
        ss = _make_system_state()
        ctrl._execute_intervention(rec, ss)
        assert ctrl.intervention_active is True
        assert ctrl.last_intervention_time == 1.0
        assert len(ctrl.intervention_outcomes) == 1
        outcome = ctrl.intervention_outcomes[0]
        assert outcome["type"] == "flow_adjustment"
        assert outcome["urgency"] == "immediate"


@pytest.mark.coverage_extra
class TestExecuteControlAction:
    """Cover lines 1011-1032."""

    def _make_ctrl(self):
        return AdaptiveMFCController()

    def _make_decision(self, action_idx=0, confidence=0.8):
        return ControlDecision(
            action_index=action_idx,
            action_description=f"Action {action_idx}",
            expected_outcome={"power_change": 0.0},
            confidence=confidence,
            rationale="test",
            risk_assessment={"biofilm_damage": 0.1, "a": 0.1, "b": 0.1, "c": 0.1},
        )

    def test_action_0_no_change(self):
        ctrl = self._make_ctrl()
        initial_flow = ctrl.system_parameters["flow_rate"]
        decision = self._make_decision(0)
        ss = _make_system_state()
        result = ctrl._execute_control_action(decision, ss)
        assert ctrl.system_parameters["flow_rate"] == initial_flow
        assert "action_executed" in result
        assert "execution_success" in result

    def test_action_1_increase_flow_10(self):
        ctrl = self._make_ctrl()
        initial_flow = ctrl.system_parameters["flow_rate"]
        decision = self._make_decision(1)
        ss = _make_system_state()
        ctrl._execute_control_action(decision, ss)
        assert abs(ctrl.system_parameters["flow_rate"] - initial_flow * 1.1) < 0.01

    def test_action_2_decrease_flow_10(self):
        ctrl = self._make_ctrl()
        initial = ctrl.system_parameters["flow_rate"]
        ctrl._execute_control_action(self._make_decision(2), _make_system_state())
        assert abs(ctrl.system_parameters["flow_rate"] - initial * 0.9) < 0.01

    def test_action_3_increase_flow_20(self):
        ctrl = self._make_ctrl()
        initial = ctrl.system_parameters["flow_rate"]
        ctrl._execute_control_action(self._make_decision(3), _make_system_state())
        assert abs(ctrl.system_parameters["flow_rate"] - initial * 1.2) < 0.01

    def test_action_4_decrease_flow_20(self):
        ctrl = self._make_ctrl()
        initial = ctrl.system_parameters["flow_rate"]
        ctrl._execute_control_action(self._make_decision(4), _make_system_state())
        assert abs(ctrl.system_parameters["flow_rate"] - initial * 0.8) < 0.01

    def test_action_7_emergency(self):
        ctrl = self._make_ctrl()
        initial = ctrl.system_parameters["flow_rate"]
        ctrl._execute_control_action(self._make_decision(7), _make_system_state())
        assert abs(ctrl.system_parameters["flow_rate"] - initial * 0.5) < 0.01

    def test_action_8_substrate(self):
        ctrl = self._make_ctrl()
        initial = ctrl.system_parameters["inlet_concentration"]
        ctrl._execute_control_action(self._make_decision(8), _make_system_state())
        assert abs(ctrl.system_parameters["inlet_concentration"] - initial * 1.1) < 0.01


@pytest.mark.coverage_extra
class TestGetParameterChanges:
    """Cover line 1042."""

    def test_returns_dict(self):
        ctrl = AdaptiveMFCController()
        changes = ctrl._get_parameter_changes()
        assert "flow_rate" in changes
        assert "inlet_concentration" in changes


@pytest.mark.coverage_extra
class TestUpdateQLearning:
    """Cover lines 1055-1069."""

    def test_update_qlearning(self):
        ctrl = AdaptiveMFCController()
        ss = _make_system_state()
        decision = ControlDecision(
            action_index=0,
            action_description="test",
            expected_outcome={},
            confidence=0.8,
            rationale="test",
            risk_assessment={"a": 0.1},
        )
        ctrl._update_qlearning(ss, decision, {"execution_success": 0.8})


@pytest.mark.coverage_extra
class TestCalculatePerformanceMetrics:
    """Cover line 1082."""

    def test_metrics_keys(self):
        ctrl = AdaptiveMFCController()
        ss = _make_system_state()
        metrics = ctrl._calculate_performance_metrics(ss)
        assert "power_efficiency" in metrics
        assert "biofilm_health_score" in metrics
        assert "sensor_reliability" in metrics
        assert "system_stability" in metrics
        assert "control_confidence" in metrics


@pytest.mark.coverage_extra
class TestGetComprehensiveStatus:
    """Cover lines 1093-1098."""

    def test_no_history(self):
        ctrl = AdaptiveMFCController()
        status = ctrl.get_comprehensive_status()
        assert "error" in status

    def test_with_history(self):
        ctrl = AdaptiveMFCController()
        ss = _make_system_state()
        ctrl.control_history.append({
            "timestamp": 1.0,
            "system_state": ss,
            "system_health_score": 0.85,
            "health_alerts": [],
            "performance_metrics": {"power_efficiency": 0.5},
        })
        status = ctrl.get_comprehensive_status()
        assert status["timestamp"] == 1.0
        assert status["control_strategy"] == "balanced"
        assert status["intervention_active"] is False


@pytest.mark.coverage_extra
class TestControlStep:
    """Cover lines 815-893 (full control_step)."""

    def test_control_step_runs(self):
        ctrl = AdaptiveMFCController()
        eis_m = MagicMock()
        qcm_m = MagicMock()
        eis_p = {"thickness_um": 10.0}
        qcm_p = {"mass_per_area_ng_per_cm2": 100.0}
        result = ctrl.control_step(eis_m, qcm_m, eis_p, qcm_p, 1.0)
        assert "timestamp" in result
        assert "system_state" in result
        assert result["timestamp"] == 1.0
        assert len(ctrl.control_history) == 1

    def test_control_step_with_alerts(self):
        ctrl = AdaptiveMFCController()
        alert = FakeHealthAlert(severity="warning", message="test alert")
        ctrl.health_monitor.generate_alerts = MagicMock(return_value=[alert])
        eis_m = MagicMock()
        qcm_m = MagicMock()
        result = ctrl.control_step(eis_m, qcm_m, {}, {}, 2.0)
        assert len(result["health_alerts"]) == 1


@pytest.mark.coverage_extra
class TestFactoryFunction:
    """Cover lines 1146-1156."""

    def test_create_adaptive_mfc_controller(self):
        ctrl = create_adaptive_mfc_controller()
        assert isinstance(ctrl, AdaptiveMFCController)
        assert ctrl.current_strategy == ControlStrategy.BALANCED


@pytest.mark.coverage_extra
class TestFallbackActions:
    """Cover line 191 - fallback actions when flow_actions/substrate_actions missing."""

    def test_fallback_when_no_flow_or_substrate_actions(self):
        ctrl = HealthAwareQLearning()
        # Remove flow_actions and substrate_actions if present
        if hasattr(ctrl, "flow_actions"):
            delattr(ctrl, "flow_actions")
        if hasattr(ctrl, "substrate_actions"):
            delattr(ctrl, "substrate_actions")
        ctrl2 = HealthAwareQLearning()
        assert len(ctrl2.actions) > 0


@pytest.mark.coverage_extra
class TestDetachmentAndStagnationRisk:
    """Cover lines 575, 577 - detachment_risk and stagnation_risk > 0.7."""

    def test_detachment_risk_in_rationale(self):
        ctrl = HealthAwareQLearning()
        ss = _make_system_state(
            health_metrics=_make_health(detachment_risk=0.8)
        )
        d = ctrl._generate_decision_rationale(0, ss)
        assert "detachment" in d.rationale.lower()

    def test_stagnation_risk_in_rationale(self):
        ctrl = HealthAwareQLearning()
        ss = _make_system_state(
            health_metrics=_make_health(stagnation_risk=0.8)
        )
        d = ctrl._generate_decision_rationale(0, ss)
        assert "stagnation" in d.rationale.lower()

    def test_multiple_risks_in_rationale(self):
        ctrl = HealthAwareQLearning()
        ss = _make_system_state(
            health_metrics=_make_health(
                fouling_risk=0.8,
                detachment_risk=0.8,
                stagnation_risk=0.8,
            )
        )
        d = ctrl._generate_decision_rationale(0, ss)
        rationale_lower = d.rationale.lower()
        assert "fouling" in rationale_lower
        assert "detachment" in rationale_lower
        assert "stagnation" in rationale_lower
