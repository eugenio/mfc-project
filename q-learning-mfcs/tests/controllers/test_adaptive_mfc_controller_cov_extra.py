"""Extra coverage tests for adaptive_mfc_controller.py.

Covers remaining uncovered lines:
- Line 191: fallback actions when flow_actions/substrate_actions are present
- Line 891: control_decision.intervention_type logging in control_step
"""
import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# --- Mock heavy dependencies before importing ---
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


class _FakeEnum:
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


@pytest.mark.coverage_extra
class TestHealthAwareQLearningFlowSubstrateActions:
    """Cover line 191: fallback actions when flow_actions and substrate_actions
    are NOT present (else branch)."""

    def test_fallback_actions_when_no_flow_substrate(self):
        """When flow_actions / substrate_actions do not exist on self before
        HealthAwareQLearning.__init__ runs, the else branch at line 191 fires
        and creates 15 hard-coded (flow, substrate) tuples."""
        # Patch the parent __init__ to set the minimum required attrs
        # but NOT flow_actions / substrate_actions, so the else branch fires.
        def _minimal_parent_init(self, *a, **kw):
            self.learning_rate = 0.1
            self.epsilon = 0.2
            self.discount_factor = 0.95
            self.q_table = {}
            # Deliberately do NOT set self.flow_actions or self.substrate_actions

        with patch.object(
            HealthAwareQLearning.__bases__[0], "__init__", _minimal_parent_init
        ):
            ctrl = HealthAwareQLearning()

        # Fallback path sets 15 actions
        assert len(ctrl.actions) == 15
        for action in ctrl.actions:
            assert isinstance(action, tuple)
            assert len(action) == 2


@pytest.mark.coverage_extra
class TestControlStepInterventionLogging:
    """Cover line 891: intervention_type logging in control_step."""

    def test_control_step_logs_intervention_type(self):
        """Ensure line 891 is hit when control_decision has intervention_type."""
        ctrl = AdaptiveMFCController()

        # Make the q_controller return a decision with intervention_type set
        intervention_decision = ControlDecision(
            action_index=7,
            action_description="Emergency flow reduction",
            expected_outcome={"power_change": -0.1},
            confidence=0.6,
            rationale="Critical health requires intervention",
            risk_assessment={
                "biofilm_damage": 0.1,
                "performance_loss": 0.4,
                "system_instability": 0.1,
                "sensor_interference": 0.0,
            },
            intervention_type="emergency_intervention",
        )
        ctrl.q_controller.choose_health_aware_action = MagicMock(
            return_value=intervention_decision
        )

        eis_m = MagicMock()
        qcm_m = MagicMock()
        result = ctrl.control_step(eis_m, qcm_m, {}, {}, 1.0)

        assert result["control_decision"].intervention_type == "emergency_intervention"
        assert len(ctrl.control_history) == 1

    def test_control_step_with_health_alerts(self):
        """Ensure both alert logging and intervention logging paths are hit."""
        ctrl = AdaptiveMFCController()

        # Make health monitor generate alerts
        alert = FakeHealthAlert(severity="warning", message="high fouling risk")
        ctrl.health_monitor.generate_alerts = MagicMock(return_value=[alert])

        # Make q_controller return a decision with intervention_type
        decision = ControlDecision(
            action_index=6,
            action_description="Optimize for biofilm health",
            expected_outcome={"health_change": 0.15},
            confidence=0.7,
            rationale="Health intervention",
            risk_assessment={
                "biofilm_damage": 0.0,
                "performance_loss": 0.05,
                "system_instability": 0.0,
                "sensor_interference": 0.0,
            },
            intervention_type="health_intervention",
        )
        ctrl.q_controller.choose_health_aware_action = MagicMock(
            return_value=decision
        )

        result = ctrl.control_step(MagicMock(), MagicMock(), {}, {}, 2.0)

        assert len(result["health_alerts"]) == 1
        assert result["control_decision"].intervention_type == "health_intervention"
