"""Deep coverage tests for phase2_demonstration.py - covers remaining branches."""
import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


class TestDemonstratePhase2BranchCoverage:
    """Target lines 146, 163, 223-224 missed by existing tests."""

    def test_optimization_result_branch(self):
        """Cover line 146: result['optimization_result'] access."""
        from phase2_demonstration import (
            create_sample_measurements,
            demonstrate_phase2_enhancements,
        )
        from ml_optimization import (
            OptimizationStrategy,
            create_ml_optimized_controller,
        )
        from sensing_models.sensor_fusion import BacterialSpecies

        np.random.seed(99)
        controller = create_ml_optimized_controller(
            species=BacterialSpecies.MIXED,
            optimization_strategy=OptimizationStrategy.BAYESIAN,
        )

        # Patch control_step_with_learning to return result with
        # optimization_result key
        fused_mock = MagicMock()
        fused_mock.fusion_confidence = 0.9
        fused_mock.sensor_agreement = 0.85

        state_mock = MagicMock()
        state_mock.fused_measurement = fused_mock
        state_mock.anomalies = []

        decision_mock = MagicMock()
        decision_mock.intervention_type = None

        prediction_mock = MagicMock()
        prediction_mock.predicted_values = [0.5]
        prediction_mock.prediction_accuracy = 0.95

        result_with_opt = {
            "system_health_score": 0.8,
            "system_state": state_mock,
            "health_alerts": ["alert1"],
            "prediction": prediction_mock,
            "ml_insights": {
                "learning_status": "active",
                "optimization_recommendations": ["rec1"],
            },
            "optimization_result": {"best_score": 0.95},
            "feature_importance": [],
            "control_decision": decision_mock,
        }

        with patch.object(
            controller,
            "control_step_with_learning",
            return_value=result_with_opt,
        ):
            with patch.object(controller, "get_ml_status_report"):
                with patch.object(
                    controller.base_controller,
                    "get_comprehensive_status",
                ):
                    with patch(
                        "phase2_demonstration.create_ml_optimized_controller",
                        return_value=controller,
                    ):
                        demonstrate_phase2_enhancements()

    def test_feature_importance_empty_branch(self):
        """Cover line 163: else pass when feature_importance is empty."""
        from phase2_demonstration import demonstrate_phase2_enhancements

        fused_mock = MagicMock()
        fused_mock.fusion_confidence = 0.9
        fused_mock.sensor_agreement = 0.85

        state_mock = MagicMock()
        state_mock.fused_measurement = fused_mock
        state_mock.anomalies = []

        decision_mock = MagicMock()
        decision_mock.intervention_type = None

        result_empty_fi = {
            "system_health_score": 0.8,
            "system_state": state_mock,
            "health_alerts": [],
            "prediction": None,
            "ml_insights": {
                "learning_status": "active",
                "optimization_recommendations": [],
            },
            "feature_importance": [],
            "control_decision": decision_mock,
        }

        controller = MagicMock()
        controller.control_step_with_learning.return_value = result_empty_fi

        with patch(
            "phase2_demonstration.create_ml_optimized_controller",
            return_value=controller,
        ):
            demonstrate_phase2_enhancements()

    def test_anomalies_branch_in_demonstrate_individual(self):
        """Cover lines 223-224: for _anomaly in anomalies."""
        from phase2_demonstration import demonstrate_individual_components

        anomaly_mock = MagicMock()
        anomaly_list = [anomaly_mock]

        fused_mock = MagicMock()
        fused_mock.fusion_confidence = 0.9

        prediction_mock = MagicMock()

        health_metrics_mock = MagicMock()
        health_metrics_mock.overall_health_score = 0.8

        fusion_system = MagicMock()
        fusion_system.fuse_measurements_with_prediction.return_value = (
            fused_mock,
            prediction_mock,
            anomaly_list,
        )

        health_monitor = MagicMock()
        health_monitor.assess_health.return_value = health_metrics_mock
        health_monitor.generate_alerts.return_value = ["alert"]
        health_monitor.generate_intervention_recommendations.return_value = [
            "rec",
        ]

        feature_engineer = MagicMock()
        feature_engineer.extract_features.return_value = {
            "f1": 0.1,
            "f2": 0.2,
        }

        with patch(
            "phase2_demonstration.create_sample_measurements",
            return_value=(MagicMock(), MagicMock(), {}, {}),
        ):
            with patch(
                "phase2_demonstration.create_advanced_sensor_fusion",
                create=True,
            ) as mock_create_fusion:
                # We need to patch the local import
                import phase2_demonstration as p2d

                orig_func = p2d.demonstrate_individual_components

                def patched_demonstrate():
                    """Run with mocked fusion system."""
                    import builtins

                    orig_import = builtins.__import__

                    def custom_import(name, *args, **kwargs):
                        if name == "sensing_models.advanced_sensor_fusion":
                            mod = MagicMock()
                            mod.create_advanced_sensor_fusion = (
                                lambda: fusion_system
                            )
                            return mod
                        if name == "biofilm_health_monitor":
                            mod = MagicMock()
                            mod.create_predictive_health_monitor = (
                                lambda *a: health_monitor
                            )
                            return mod
                        if name == "ml_optimization":
                            mod = MagicMock()
                            mod.FeatureEngineer = (
                                lambda: feature_engineer
                            )
                            return mod
                        if name == "adaptive_mfc_controller":
                            mod = MagicMock()
                            mod.SystemState = MagicMock
                            mod.ControlStrategy = MagicMock()
                            mod.ControlStrategy.BALANCED = "balanced"
                            mod.AdaptationMode = MagicMock()
                            mod.AdaptationMode.MODERATE = "moderate"
                            return mod
                        return orig_import(name, *args, **kwargs)

                    with patch.object(
                        builtins, "__import__", side_effect=custom_import
                    ):
                        orig_func()

                patched_demonstrate()


class TestCreateSampleMeasurementsEdge:
    def test_zero_time(self):
        from phase2_demonstration import create_sample_measurements

        np.random.seed(0)
        eis_m, qcm_m, eis_p, qcm_p = create_sample_measurements(0.0)
        assert eis_m.timestamp == 0.0
        assert qcm_m.timestamp == 0.0

    def test_large_thickness(self):
        from phase2_demonstration import create_sample_measurements

        np.random.seed(0)
        eis_m, qcm_m, _, _ = create_sample_measurements(10.0, 100.0)
        assert eis_m is not None
        assert qcm_m is not None
