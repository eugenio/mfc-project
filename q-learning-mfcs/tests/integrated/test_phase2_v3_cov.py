"""Tests for phase2_demonstration.py - coverage for lines 20-287.

Covers: create_sample_measurements, demonstrate_phase2_enhancements,
demonstrate_individual_components.
"""
import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Mock heavy deps
mock_ml = MagicMock()
mock_ml.OptimizationStrategy = MagicMock()
mock_ml.OptimizationStrategy.BAYESIAN = "bayesian"
mock_ml.FeatureEngineer = MagicMock

mock_eis = MagicMock()
mock_qcm = MagicMock()
mock_fusion = MagicMock()
mock_fusion.BacterialSpecies = MagicMock()
mock_fusion.BacterialSpecies.MIXED = MagicMock(value="mixed")

mock_adv_fusion = MagicMock()
mock_health_mon = MagicMock()
mock_adaptive = MagicMock()

if "ml_optimization" not in sys.modules:
    sys.modules["ml_optimization"] = mock_ml
if "sensing_models.eis_model" not in sys.modules:
    sys.modules["sensing_models.eis_model"] = mock_eis
if "sensing_models.qcm_model" not in sys.modules:
    sys.modules["sensing_models.qcm_model"] = mock_qcm
if "sensing_models.sensor_fusion" not in sys.modules:
    sys.modules["sensing_models.sensor_fusion"] = mock_fusion
if "sensing_models.advanced_sensor_fusion" not in sys.modules:
    sys.modules["sensing_models.advanced_sensor_fusion"] = mock_adv_fusion
if "biofilm_health_monitor" not in sys.modules:
    sys.modules["biofilm_health_monitor"] = mock_health_mon
if "adaptive_mfc_controller" not in sys.modules:
    sys.modules["adaptive_mfc_controller"] = mock_adaptive

from phase2_demonstration import create_sample_measurements


class TestCreateSampleMeasurements:
    """Cover lines 31-81."""

    def test_returns_tuple_of_4(self):
        result = create_sample_measurements(1.0, 15.0)
        assert len(result) == 4

    def test_default_base_thickness(self):
        result = create_sample_measurements(0.0)
        assert len(result) == 4

    def test_at_different_times(self):
        r1 = create_sample_measurements(0.0, 10.0)
        r2 = create_sample_measurements(10.0, 10.0)
        assert len(r1) == 4
        assert len(r2) == 4


class TestDemonstratePhase2:
    """Cover lines 84-193 (demonstrate_phase2_enhancements)."""

    def test_demonstrate_phase2(self):
        # Set up the mock controller
        mock_ctrl = MagicMock()
        mock_fused = MagicMock()
        mock_fused.fusion_confidence = 0.9
        mock_fused.sensor_agreement = 0.85
        mock_state = MagicMock()
        mock_state.fused_measurement = mock_fused
        mock_state.anomalies = []
        mock_pred = MagicMock()
        mock_pred.predicted_values = [10.0]
        mock_pred.prediction_accuracy = 0.9

        mock_decision = MagicMock()
        mock_decision.intervention_type = None

        step_result = {
            "system_health_score": 0.8,
            "system_state": mock_state,
            "prediction": mock_pred,
            "health_alerts": [],
            "control_decision": mock_decision,
            "ml_insights": {
                "learning_status": "active",
                "optimization_recommendations": [],
            },
        }
        mock_ctrl.control_step_with_learning.return_value = step_result
        mock_ctrl.get_ml_status_report.return_value = {}
        mock_ctrl.base_controller = MagicMock()
        mock_ctrl.base_controller.get_comprehensive_status.return_value = {}

        mock_ml.create_ml_optimized_controller = MagicMock(
            return_value=mock_ctrl
        )

        from phase2_demonstration import demonstrate_phase2_enhancements
        demonstrate_phase2_enhancements()


class TestDemonstrateIndividualComponents:
    """Cover lines 195-287."""

    def test_demonstrate_individual(self):
        mock_fusion_sys = MagicMock()
        fused = MagicMock()
        fused.fusion_confidence = 0.9
        fused.sensor_agreement = 0.85
        fused.thickness_um = 15.0
        fused.biomass_density_g_per_L = 0.5
        fused.timestamp = 1.0

        pred = MagicMock()
        pred.predicted_values = [10.0]
        pred.prediction_accuracy = 0.9

        anomalies = []
        mock_fusion_sys.fuse_measurements_with_prediction.return_value = (
            fused, pred, anomalies
        )
        mock_adv_fusion.create_advanced_sensor_fusion = MagicMock(
            return_value=mock_fusion_sys
        )

        mock_hm = MagicMock()
        mock_hm.overall_health_score = 0.8
        mock_health_monitor = MagicMock()
        mock_health_monitor.assess_health.return_value = mock_hm
        mock_health_monitor.generate_alerts.return_value = []
        mock_health_monitor.generate_intervention_recommendations.return_value = []
        mock_health_mon.create_predictive_health_monitor = MagicMock(
            return_value=mock_health_monitor
        )

        mock_adaptive.AdaptationMode = MagicMock()
        mock_adaptive.AdaptationMode.MODERATE = "moderate"
        mock_adaptive.ControlStrategy = MagicMock()
        mock_adaptive.ControlStrategy.BALANCED = "balanced"
        mock_adaptive.SystemState = MagicMock()

        mock_fe = MagicMock()
        mock_fe.extract_features.return_value = {
            "f1": 1.0, "f2": 2.0, "f3": 3.0, "f4": 4.0, "f5": 5.0
        }
        mock_ml.FeatureEngineer = MagicMock(return_value=mock_fe)

        from phase2_demonstration import demonstrate_individual_components
        demonstrate_individual_components()
