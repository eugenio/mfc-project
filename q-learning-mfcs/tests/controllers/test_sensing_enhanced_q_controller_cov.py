"""Tests for sensing_enhanced_q_controller.py - coverage target 98%+.

Covers SensingEnhancedQLearningController: init, state space,
discretization, action selection, rewards, fault handling, summaries.
"""
import sys
import os
from unittest.mock import MagicMock, patch
from collections import defaultdict

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sensing_enhanced_q_controller import SensingEnhancedQLearningController


def _sensor_data(eis_status="good", qcm_status="good", quality=0.9):
    return {
        "eis": {
            "thickness_um": 10.0,
            "conductivity_S_per_m": 0.1,
            "measurement_quality": quality,
            "status": eis_status,
        },
        "qcm": {
            "thickness_um": 11.0,
            "mass_per_area_ng_per_cm2": 500.0,
            "frequency_shift_Hz": -200.0,
            "dissipation": 1e-6,
            "measurement_quality": quality,
            "status": qcm_status,
        },
        "fusion": {
            "sensor_agreement": 0.9,
            "fusion_confidence": 0.85,
            "cross_validation_error": 0.05,
        },
    }


@pytest.fixture
def ctrl():
    c = SensingEnhancedQLearningController()
    if not hasattr(c, "actions"):
        c.actions = [(-10, -5), (-5, 0), (0, 0), (0, 5), (5, 0), (5, 5),
                     (10, -5), (10, 0), (10, 5)]
    return c


class TestInit:
    def test_default_init(self, ctrl):
        assert ctrl.enable_sensor_state is True
        assert ctrl.fault_tolerance is True
        assert ctrl.sensor_fault_count == 0
        assert ctrl.sensor_degradation_factor == 1.0

    def test_custom_init(self):
        c = SensingEnhancedQLearningController(
            enable_sensor_state=False,
            fault_tolerance=False,
        )
        assert c.enable_sensor_state is False
        assert c.fault_tolerance is False

    def test_multi_objective_weights(self, ctrl):
        assert ctrl.power_weight > 0
        assert ctrl.biofilm_health_weight > 0
        assert ctrl.sensor_agreement_weight > 0
        assert ctrl.stability_weight > 0


class TestStateSpace:
    def test_sensor_bins_exist(self, ctrl):
        assert hasattr(ctrl, "eis_thickness_bins")
        assert hasattr(ctrl, "qcm_mass_bins")
        assert hasattr(ctrl, "sensor_agreement_bins")

    def test_no_sensor_state(self):
        c = SensingEnhancedQLearningController(enable_sensor_state=False)
        assert c.enable_sensor_state is False


class TestDiscretizeState:
    def test_with_sensor_data(self, ctrl):
        base = (1, 2, 3)
        sd = _sensor_data()
        result = ctrl.discretize_sensor_enhanced_state(base, sd)
        assert len(result) > len(base)

    def test_without_sensor_data(self, ctrl):
        base = (1, 2, 3)
        result = ctrl.discretize_sensor_enhanced_state(base, None)
        assert result == base

    def test_sensor_disabled(self):
        c = SensingEnhancedQLearningController(enable_sensor_state=False)
        base = (1, 2, 3)
        result = c.discretize_sensor_enhanced_state(base, _sensor_data())
        assert result == base

    def test_various_statuses(self, ctrl):
        for eis_s in ["good", "degraded", "failed", "unavailable"]:
            for qcm_s in ["good", "degraded", "failed", "unavailable"]:
                sd = _sensor_data(eis_status=eis_s, qcm_status=qcm_s)
                base = (1, 2, 3)
                result = ctrl.discretize_sensor_enhanced_state(base, sd)
                assert len(result) > 3


class TestEncodeSensorStatus:
    def test_both_good(self, ctrl):
        assert ctrl._encode_sensor_status("good", "good") == 3

    def test_both_degraded(self, ctrl):
        assert ctrl._encode_sensor_status("degraded", "degraded") == 2

    def test_both_failed(self, ctrl):
        assert ctrl._encode_sensor_status("failed", "failed") == 1

    def test_both_unavailable(self, ctrl):
        assert ctrl._encode_sensor_status("unavailable", "unavailable") == 0

    def test_mixed(self, ctrl):
        r = ctrl._encode_sensor_status("good", "failed")
        assert r == 2

    def test_unknown(self, ctrl):
        r = ctrl._encode_sensor_status("unknown", "unknown")
        assert r == 0


class TestChooseActionWithSensors:
    def test_exploration(self, ctrl):
        ctrl.epsilon = 1.0
        a = ctrl.choose_action_with_sensors((1, 2, 3), _sensor_data())
        assert isinstance(a, (int, np.integer))

    def test_exploitation(self, ctrl):
        ctrl.epsilon = 0.0
        a = ctrl.choose_action_with_sensors((1, 2, 3), _sensor_data())
        assert isinstance(a, (int, np.integer))

    def test_with_available_actions_explore(self, ctrl):
        ctrl.epsilon = 1.0
        a = ctrl.choose_action_with_sensors(
            (1, 2, 3), _sensor_data(), available_actions=[0, 1, 2]
        )
        assert isinstance(a, (int, np.integer))

    def test_with_available_actions_exploit(self, ctrl):
        ctrl.epsilon = 0.0
        a = ctrl.choose_action_with_sensors(
            (1, 2, 3), _sensor_data(), available_actions=[0, 1, 2]
        )
        assert isinstance(a, (int, np.integer))

    def test_no_sensor_data(self, ctrl):
        ctrl.epsilon = 0.5
        a = ctrl.choose_action_with_sensors((1, 2, 3), None)
        assert isinstance(a, (int, np.integer))

    def test_tracks_decisions(self, ctrl):
        ctrl.epsilon = 0.0
        ctrl.choose_action_with_sensors((1, 2, 3), _sensor_data(quality=0.9))
        total = ctrl.sensor_guided_decisions + ctrl.model_guided_decisions
        assert total > 0

    def test_low_confidence_model_guided(self, ctrl):
        ctrl.epsilon = 0.0
        ctrl.choose_action_with_sensors(
            (1, 2, 3), _sensor_data(quality=0.01)
        )
        assert ctrl.model_guided_decisions > 0


class TestSensorConfidence:
    def test_no_data(self, ctrl):
        assert ctrl._calculate_sensor_confidence(None) == 0.0

    def test_with_data(self, ctrl):
        c = ctrl._calculate_sensor_confidence(_sensor_data())
        assert 0 <= c <= 1

    def test_empty_sections(self, ctrl):
        c = ctrl._calculate_sensor_confidence({})
        assert c == 0.0

    def test_degraded_factor(self, ctrl):
        ctrl.sensor_degradation_factor = 0.5
        c = ctrl._calculate_sensor_confidence(_sensor_data())
        assert c <= 0.5


class TestAdaptiveEpsilon:
    def test_low_confidence(self, ctrl):
        e = ctrl._calculate_adaptive_epsilon(0.0)
        assert e >= ctrl.epsilon

    def test_high_confidence(self, ctrl):
        e = ctrl._calculate_adaptive_epsilon(0.99)
        assert isinstance(e, float)


class TestChooseBestAvailableAction:
    def test_empty_actions(self, ctrl):
        assert ctrl._choose_best_available_action((1, 2, 3), []) == 0

    def test_valid_actions(self, ctrl):
        a = ctrl._choose_best_available_action((1, 2, 3), [0, 1])
        assert isinstance(a, (int, np.integer))


class TestSensorEnhancedReward:
    def test_no_data(self, ctrl):
        r = ctrl.calculate_sensor_enhanced_reward(1.0)
        assert r == 1.0

    def test_with_data(self, ctrl):
        sd = _sensor_data()
        ss = {"flow_rate": 15.0, "outlet_concentration": 10.0}
        r = ctrl.calculate_sensor_enhanced_reward(1.0, sd, ss)
        assert isinstance(r, float)

    def test_reward_components_tracked(self, ctrl):
        sd = _sensor_data()
        ss = {"flow_rate": 15.0, "outlet_concentration": 10.0}
        ctrl.calculate_sensor_enhanced_reward(1.0, sd, ss)
        assert any(v != 0.0 for v in ctrl.total_reward_components.values())


class TestBiofilmHealthReward:
    def test_with_eis(self, ctrl):
        sd = _sensor_data()
        ss = {}
        r = ctrl._calculate_biofilm_health_reward(sd, ss)
        assert r >= 0

    def test_mass_growth(self, ctrl):
        ctrl.sensor_confidence_history = [0.5, 0.6]
        ctrl._previous_qcm_mass = 400.0
        sd = _sensor_data()
        r = ctrl._calculate_biofilm_health_reward(sd, {})
        assert isinstance(r, float)


class TestSensorAgreementReward:
    def test_with_fusion(self, ctrl):
        sd = _sensor_data()
        r = ctrl._calculate_sensor_agreement_reward(sd)
        assert r > 0

    def test_no_fusion(self, ctrl):
        assert ctrl._calculate_sensor_agreement_reward({}) == 0.0


class TestStabilityReward:
    def test_basic(self, ctrl):
        ss = {"flow_rate": 15.0, "outlet_concentration": 10.0}
        r = ctrl._calculate_stability_reward(ss)
        assert isinstance(r, float)


class TestUpdateQValue:
    def test_without_sensors(self, ctrl):
        state = (1, 2, 3)
        ctrl.update_q_value_with_sensors(state, 0, 1.0, state)
        assert ctrl.total_rewards > 0

    def test_with_sensors(self, ctrl):
        state = (1, 2, 3)
        sd = _sensor_data()
        ctrl.update_q_value_with_sensors(state, 0, 1.0, state, sd)
        assert ctrl.total_rewards > 0

    def test_low_confidence(self, ctrl):
        state = (1, 2, 3)
        sd = _sensor_data(quality=0.01)
        ctrl.update_q_value_with_sensors(state, 0, 1.0, state, sd)


class TestSensorFault:
    def test_failed_both(self, ctrl):
        ctrl.handle_sensor_fault("failed", "both")
        assert ctrl.sensor_degradation_factor == 0.1
        assert ctrl.sensor_fault_count == 1

    def test_failed_single(self, ctrl):
        ctrl.handle_sensor_fault("failed", "eis")
        assert ctrl.sensor_degradation_factor == 0.5

    def test_degraded(self, ctrl):
        ctrl.handle_sensor_fault("degraded", "qcm")
        assert ctrl.sensor_degradation_factor < 1.0

    def test_noisy(self, ctrl):
        ctrl.handle_sensor_fault("noisy", "eis")
        assert ctrl.sensor_degradation_factor < 1.0

    def test_low_degradation_boosts_epsilon(self, ctrl):
        ctrl.handle_sensor_fault("failed", "both")
        assert ctrl.epsilon > 0


class TestPerformanceSummary:
    def test_basic(self, ctrl):
        s = ctrl.get_controller_performance_summary()
        assert "basic_performance" in s
        assert "sensor_integration" in s
        assert "decision_statistics" in s
        assert "reward_components" in s
        assert "sensor_confidence" in s

    def test_with_decisions(self, ctrl):
        ctrl.sensor_guided_decisions = 10
        ctrl.model_guided_decisions = 5
        s = ctrl.get_controller_performance_summary()
        assert s["decision_statistics"]["total_decisions"] == 15

    def test_with_confidence_history(self, ctrl):
        ctrl.sensor_confidence_history = [0.5] * 20
        s = ctrl.get_controller_performance_summary()
        assert s["sensor_confidence"]["recent_confidence"] == 0.5


class TestResetSensorTracking:
    def test_reset(self, ctrl):
        ctrl.sensor_confidence_history = [0.5]
        ctrl.sensor_fault_count = 3
        ctrl.sensor_degradation_factor = 0.2
        ctrl.sensor_guided_decisions = 10
        ctrl.model_guided_decisions = 5
        ctrl.total_reward_components["power"] = 10.0
        ctrl.reset_sensor_tracking()
        assert len(ctrl.sensor_confidence_history) == 0
        assert ctrl.sensor_fault_count == 0
        assert ctrl.sensor_degradation_factor == 1.0
        assert ctrl.sensor_guided_decisions == 0
        assert ctrl.total_reward_components["power"] == 0.0


class TestAdaptToSensorAvailability:
    def test_no_sensors(self, ctrl):
        ctrl.adapt_to_sensor_availability(False, False)
        assert ctrl.enable_sensor_state is False
        assert ctrl.sensor_weight == 0.0

    def test_both_sensors(self, ctrl):
        ctrl.adapt_to_sensor_availability(True, True)
        assert ctrl.enable_sensor_state is True
        assert ctrl.sensor_weight == 0.3

    def test_single_sensor(self, ctrl):
        ctrl.adapt_to_sensor_availability(True, False)
        assert ctrl.enable_sensor_state is True
        assert ctrl.sensor_weight == 0.2


class TestValidateOperation:
    def test_basic(self, ctrl):
        v = ctrl.validate_sensor_enhanced_operation()
        assert "sensor_state_enabled" in v
        assert v["sensor_state_enabled"] is True
        assert v["state_space_extended"] is True
        assert v["adaptive_exploration"] is True
        assert v["fault_tolerance"] is True


class TestGetStateHash:
    def test_basic(self, ctrl):
        h = ctrl.get_state_hash(10.0, 8.0, 0.5)
        assert isinstance(h, str)
        assert "_" in h


class TestChooseAction:
    def test_explore(self, ctrl):
        ctrl.epsilon = 1.0
        a = ctrl.choose_action("1_2_3")
        assert 0 <= a < 10

    def test_exploit_no_q(self, ctrl):
        ctrl.epsilon = 0.0
        a = ctrl.choose_action("nonexistent_state")
        assert 0 <= a < 10

    def test_exploit_with_q(self, ctrl):
        ctrl.epsilon = 0.0
        ctrl.q_table["test_state"] = {0: 0.5, 1: 0.9, 2: 0.1}
        a = ctrl.choose_action("test_state")
        assert a == 1
