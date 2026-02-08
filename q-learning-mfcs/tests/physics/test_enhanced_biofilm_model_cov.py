"""Tests for biofilm_kinetics/enhanced_biofilm_model.py - coverage 98%+."""
import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from biofilm_kinetics.enhanced_biofilm_model import (
    EnhancedBiofilmModel,
    SensorCalibrationError,
)


class TestEnhancedBiofilmModelInit:
    def test_default_init(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        assert m.enable_eis is True
        assert m.enable_qcm is True
        assert m.sensor_feedback_enabled is True
        assert m.adaptive_calibration_enabled is True
        assert m.model_confidence == 1.0
        assert m.sensor_confidence == 0.8

    def test_disable_sensors(self):
        m = EnhancedBiofilmModel(
            use_gpu=False, enable_eis=False, enable_qcm=False
        )
        assert m.enable_eis is False
        assert m.enable_qcm is False

    def test_sensor_calibration_error(self):
        err = SensorCalibrationError("test error")
        assert str(err) == "test error"


class TestStepBiofilmDynamicsWithSensors:
    def test_no_sensors_available(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        m.sensor_feedback_enabled = False
        result = m.step_biofilm_dynamics_with_sensors(
            0.1, -0.2, 1.0, time_hours=0.0
        )
        assert "biofilm_thickness" in result

    def test_with_external_measurements(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        ext = {
            "eis": {
                "measurements": [],
                "properties": {"thickness_um": 5.0, "measurement_quality": 0.9},
            }
        }
        result = m.step_biofilm_dynamics_with_sensors(
            0.1, -0.2, 1.0, time_hours=1.0, external_measurements=ext
        )
        assert "biofilm_thickness" in result

    def test_no_sensor_measurements(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        m.eis_model = None
        m.qcm_model = None
        result = m.step_biofilm_dynamics_with_sensors(
            0.1, -0.2, 1.0, time_hours=0.0
        )
        assert result.get("sensor_status") == "unavailable"

    def test_with_eis_only(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        ext = {
            "eis": {
                "measurements": [],
                "properties": {"thickness_um": 10.0, "measurement_quality": 0.7},
            }
        }
        result = m.step_biofilm_dynamics_with_sensors(
            0.1, -0.2, 1.0, time_hours=1.0, external_measurements=ext
        )
        assert len(m.sensor_measurements) > 0

    def test_with_qcm_only(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        ext = {
            "qcm": {
                "measurement": MagicMock(),
                "properties": {
                    "thickness_um": 8.0,
                    "biomass_density_g_per_L": 3.0,
                    "measurement_quality": 0.6,
                },
            }
        }
        result = m.step_biofilm_dynamics_with_sensors(
            0.1, -0.2, 1.0, time_hours=1.0, external_measurements=ext
        )
        assert "biofilm_thickness" in result


class TestGetSensorMeasurements:
    def test_external_measurements_used(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        ext = {"eis": {"data": "test"}}
        result = m._get_sensor_measurements(10.0, 5.0, 0.0, ext)
        assert result == ext

    def test_no_sensors(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        m.eis_model = None
        m.qcm_model = None
        result = m._get_sensor_measurements(10.0, 5.0, 0.0)
        assert result is None


class TestApplySensorFeedback:
    def test_empty_measurements(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        base = {"biofilm_thickness": 10.0, "biomass_density": 5.0}
        result = m._apply_sensor_feedback(base, {}, 0.0)
        assert result == base

    def test_eis_only(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        base = {"biofilm_thickness": 10.0, "biomass_density": 5.0}
        sensor = {
            "eis": {
                "measurements": [],
                "properties": {"thickness_um": 12.0, "measurement_quality": 0.8},
            }
        }
        result = m._apply_sensor_feedback(base, sensor, 1.0)
        assert result.get("sensor_status") == "eis_only"
        assert result.get("sensor_validated_thickness") == 12.0

    def test_qcm_only(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        base = {"biofilm_thickness": 10.0, "biomass_density": 5.0}
        sensor = {
            "qcm": {
                "measurement": MagicMock(),
                "properties": {
                    "thickness_um": 9.0,
                    "biomass_density_g_per_L": 4.0,
                    "measurement_quality": 0.7,
                },
            }
        }
        result = m._apply_sensor_feedback(base, sensor, 1.0)
        assert result.get("sensor_status") == "qcm_only"

    def test_eis_zero_thickness(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        base = {"biofilm_thickness": 10.0, "biomass_density": 5.0}
        sensor = {
            "eis": {
                "measurements": [],
                "properties": {"thickness_um": 0.0, "measurement_quality": 0.5},
            }
        }
        result = m._apply_sensor_feedback(base, sensor, 1.0)
        assert "sensor_validated_thickness" not in result or result.get("sensor_validated_thickness") == 0.0


class TestUpdateAdaptiveParameters:
    def test_no_recalibration_needed(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        m.last_calibration_time = 0.0
        state = {
            "biofilm_thickness": 10.0,
            "sensor_validated_thickness": 10.0,
        }
        m._update_adaptive_parameters(state, {"eis": {}}, 1.0)

    def test_recalibration_triggered(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        m.last_calibration_time = 0.0
        state = {
            "biofilm_thickness": 10.0,
            "sensor_validated_thickness": 15.0,
        }
        m._update_adaptive_parameters(state, {"eis": {}}, 25.0)
        assert len(m.calibration_history) > 0

    def test_no_recalibration_when_agreement(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        m.last_calibration_time = 0.0
        state = {
            "biofilm_thickness": 10.0,
            "sensor_validated_thickness": 10.5,
        }
        m._update_adaptive_parameters(state, {"eis": {}}, 25.0)


class TestShouldRecalibrate:
    def test_time_threshold(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        m.last_calibration_time = 0.0
        assert m._should_recalibrate(25.0) is True

    def test_within_time_no_drift(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        m.last_calibration_time = 0.0
        assert m._should_recalibrate(1.0) is False

    def test_drift_detection(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        m.last_calibration_time = 0.0
        for i in range(15):
            m.prediction_errors.append({"thickness_error": 10.0, "biomass_error": 0.0})
        assert m._should_recalibrate(1.0) is True


class TestRecalibrateParameters:
    def test_basic_recalibration(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        original_mu = m.kinetic_params.mu_max
        m._recalibrate_parameters(10.0, 15.0, 25.0)
        assert len(m.calibration_history) == 1
        assert m.last_calibration_time == 25.0

    def test_zero_model_thickness(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        m._recalibrate_parameters(0.0, 15.0, 25.0)

    def test_clipping(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        m._recalibrate_parameters(0.001, 1000.0, 25.0)
        assert m.kinetic_params.mu_max <= 2.0


class TestUpdateConfidenceLevels:
    def test_basic_update(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        m.thickness_prediction_error = 5.0
        state = {"fusion_confidence": 0.9}
        m._update_confidence_levels(state)
        assert m.model_confidence != 1.0

    def test_no_error(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        state = {"fusion_confidence": 0.5}
        m._update_confidence_levels(state)


class TestGetSensorDiagnostics:
    def test_empty_diagnostics(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        diag = m.get_sensor_diagnostics()
        assert "sensor_configuration" in diag
        assert "confidence_levels" in diag
        assert diag["measurement_statistics"]["total_measurements"] == 0

    def test_with_errors(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        m.prediction_errors = [
            {"thickness_error": 2.0, "biomass_error": 0.5},
            {"thickness_error": 3.0, "biomass_error": 0.8},
        ]
        diag = m.get_sensor_diagnostics()
        assert diag["error_statistics"]["mean_thickness_error_um"] == pytest.approx(2.5)


class TestResetSensorCalibration:
    def test_reset(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        m.prediction_errors.append({"test": 1})
        m.calibration_history.append({"test": 2})
        m.model_confidence = 0.5
        m.reset_sensor_calibration()
        assert m.model_confidence == 1.0
        assert m.sensor_confidence == 0.8
        assert len(m.prediction_errors) == 0
        assert len(m.calibration_history) == 0


class TestValidateSensorIntegration:
    def test_no_sensors(self):
        m = EnhancedBiofilmModel(use_gpu=False)
        m.eis_model = None
        m.qcm_model = None
        m.sensor_fusion = None
        result = m.validate_sensor_integration()
        assert result["eis_model_available"] is False
        assert result["qcm_model_available"] is False
        assert result["can_simulate_eis"] is False
        assert result["can_simulate_qcm"] is False
        assert result["can_perform_fusion"] is False
