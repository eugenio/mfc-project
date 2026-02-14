"""Coverage boost tests for enhanced_biofilm_model.py targeting remaining uncovered lines."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Snapshot sys.modules before mocking
_original_modules = dict(sys.modules)

# Pre-mock sensing_models so the source try/except succeeds
_mock_eis_mod = MagicMock()
_mock_qcm_mod = MagicMock()
_mock_fusion_mod = MagicMock()

_BacterialSpecies = SimpleNamespace(
    GEOBACTER="geobacter",
    SHEWANELLA="shewanella",
    MIXED="mixed",
)
_FusionMethod = SimpleNamespace(
    KALMAN_FILTER="kalman_filter",
    WEIGHTED_AVERAGE="weighted_average",
    MAXIMUM_LIKELIHOOD="maximum_likelihood",
    BAYESIAN_FUSION="bayesian",
)

_mock_eis_mod.BacterialSpecies = _BacterialSpecies
_mock_eis_mod.EISModel = MagicMock
_mock_qcm_mod.QCMModel = MagicMock
_mock_fusion_mod.FusionMethod = _FusionMethod
_mock_fusion_mod.SensorFusion = MagicMock

sys.modules.setdefault("sensing_models", MagicMock())
sys.modules.setdefault("sensing_models.eis_model", _mock_eis_mod)
sys.modules.setdefault("sensing_models.qcm_model", _mock_qcm_mod)
sys.modules.setdefault("sensing_models.sensor_fusion", _mock_fusion_mod)
sys.modules.setdefault("gpu_acceleration", MagicMock())

from biofilm_kinetics.enhanced_biofilm_model import (  # noqa: E402
    EnhancedBiofilmModel,
    SensorCalibrationError,
)

# --- Restore sys.modules to prevent mock leakage ---
for _mock_key in list(sys.modules):
    if _mock_key not in _original_modules:
        if isinstance(sys.modules[_mock_key], MagicMock):
            del sys.modules[_mock_key]
    elif isinstance(sys.modules[_mock_key], MagicMock):
        sys.modules[_mock_key] = _original_modules[_mock_key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(enable_eis=True, enable_qcm=True, fusion="kalman_filter"):
    """Create an EnhancedBiofilmModel with mocked internals."""
    with patch.object(EnhancedBiofilmModel, "__init__", lambda self, *a, **kw: None):
        m = EnhancedBiofilmModel()

    # Replicate minimal __init__ state
    m.species = "geobacter"
    m.substrate = "acetate"
    m.temperature = 303.0
    m.ph = 7.0
    m.enable_eis = enable_eis
    m.enable_qcm = enable_qcm
    m.sensor_fusion_method = fusion
    m.eis_model = MagicMock() if enable_eis else None
    m.qcm_model = MagicMock() if enable_qcm else None
    m.sensor_fusion = MagicMock() if (enable_eis or enable_qcm) else None
    m.sensor_feedback_enabled = True
    m.adaptive_calibration_enabled = True
    m.calibration_threshold = 0.2
    m.sensor_weight_decay = 0.95
    m.model_confidence = 1.0
    m.sensor_confidence = 0.8
    m.fusion_history_length = 50
    m.sensor_measurements = []
    m.model_predictions = []
    m.fusion_results = []
    m.calibration_history = []
    m.prediction_errors = []
    m.sensor_residuals = []
    m.last_calibration_time = 0.0
    m.sensor_validated_thickness = 0.0
    m.sensor_validated_biomass = 0.0
    m.thickness_prediction_error = 0.0
    m.biomass_prediction_error = 0.0
    return m


# ---------------------------------------------------------------------------
# Tests targeting uncovered branches and lines
# ---------------------------------------------------------------------------


@pytest.mark.coverage_extra
class TestSensorFeedbackBranches:
    """Cover _apply_sensor_feedback branches: EIS-only, QCM-only, fusion exception."""

    def test_apply_sensor_feedback_eis_only_positive_thickness(self):
        """Cover EIS-only branch with positive thickness."""
        m = _make_model(enable_eis=True, enable_qcm=False)
        m.sensor_fusion = None  # no fusion
        base_state = {
            "biofilm_thickness": 10.0,
            "biomass_density": 5.0,
        }
        sensor_data = {
            "eis": {
                "measurements": [MagicMock()],
                "properties": {
                    "thickness_um": 12.0,
                    "measurement_quality": 0.9,
                },
            },
        }
        result = m._apply_sensor_feedback(base_state, sensor_data, 1.0)
        assert result["sensor_validated_thickness"] == 12.0
        assert result["sensor_status"] == "eis_only"
        assert abs(m.thickness_prediction_error - 2.0) < 1e-9

    def test_apply_sensor_feedback_eis_only_zero_thickness(self):
        """Cover EIS-only branch with zero thickness (skips update)."""
        m = _make_model(enable_eis=True, enable_qcm=False)
        m.sensor_fusion = None
        base_state = {"biofilm_thickness": 10.0, "biomass_density": 5.0}
        sensor_data = {
            "eis": {
                "measurements": [MagicMock()],
                "properties": {"thickness_um": 0.0, "measurement_quality": 0.5},
            },
        }
        result = m._apply_sensor_feedback(base_state, sensor_data, 1.0)
        # Should not set sensor_validated_thickness from EIS because thickness is 0
        assert "sensor_status" not in result or result.get("sensor_status") != "eis_only"

    def test_apply_sensor_feedback_qcm_only_positive_thickness(self):
        """Cover QCM-only branch with positive thickness."""
        m = _make_model(enable_eis=False, enable_qcm=True)
        m.sensor_fusion = None
        base_state = {"biofilm_thickness": 10.0, "biomass_density": 5.0}
        sensor_data = {
            "qcm": {
                "measurement": MagicMock(),
                "properties": {
                    "thickness_um": 11.0,
                    "biomass_density_g_per_L": 4.5,
                    "measurement_quality": 0.85,
                },
            },
        }
        result = m._apply_sensor_feedback(base_state, sensor_data, 1.0)
        assert result["sensor_validated_thickness"] == 11.0
        assert result["sensor_validated_biomass"] == 4.5
        assert result["sensor_status"] == "qcm_only"
        assert abs(m.biomass_prediction_error - 0.5) < 1e-9

    def test_apply_sensor_feedback_qcm_only_zero_thickness(self):
        """Cover QCM-only branch with zero thickness."""
        m = _make_model(enable_eis=False, enable_qcm=True)
        m.sensor_fusion = None
        base_state = {"biofilm_thickness": 10.0, "biomass_density": 5.0}
        sensor_data = {
            "qcm": {
                "measurement": MagicMock(),
                "properties": {
                    "thickness_um": 0.0,
                    "biomass_density_g_per_L": 0.0,
                    "measurement_quality": 0.5,
                },
            },
        }
        result = m._apply_sensor_feedback(base_state, sensor_data, 1.0)
        assert result.get("sensor_status") != "qcm_only"

    def test_apply_sensor_feedback_fusion_exception(self):
        """Cover fusion exception fallback branch."""
        m = _make_model(enable_eis=True, enable_qcm=True)
        m.sensor_fusion.fuse_measurements.side_effect = Exception("fusion error")
        base_state = {"biofilm_thickness": 10.0, "biomass_density": 5.0}
        sensor_data = {
            "eis": {
                "measurements": [MagicMock()],
                "properties": {"thickness_um": 12.0, "measurement_quality": 0.9},
            },
            "qcm": {
                "measurement": MagicMock(),
                "properties": {
                    "thickness_um": 11.5,
                    "biomass_density_g_per_L": 4.5,
                    "measurement_quality": 0.85,
                },
            },
        }
        result = m._apply_sensor_feedback(base_state, sensor_data, 1.0)
        # Fusion failed, should fall back to EIS-only
        assert result["sensor_validated_thickness"] == 12.0
        assert result["sensor_status"] == "eis_only"

    def test_apply_sensor_feedback_no_sensors(self):
        """Cover branch with no eis_data and no qcm_data."""
        m = _make_model()
        base_state = {"biofilm_thickness": 10.0, "biomass_density": 5.0}
        sensor_data = {}
        result = m._apply_sensor_feedback(base_state, sensor_data, 1.0)
        assert result["biofilm_thickness"] == 10.0

    def test_apply_sensor_feedback_fusion_success(self):
        """Cover successful fusion branch."""
        m = _make_model(enable_eis=True, enable_qcm=True)
        fused = MagicMock()
        fused.thickness_um = 11.0
        fused.biomass_density_g_per_L = 4.8
        fused.thickness_uncertainty = 0.5
        fused.biomass_uncertainty = 0.2
        fused.fusion_confidence = 0.95
        fused.sensor_agreement = 0.9
        fused.eis_weight = 0.6
        fused.qcm_weight = 0.4
        m.sensor_fusion.fuse_measurements.return_value = fused
        base_state = {"biofilm_thickness": 10.0, "biomass_density": 5.0}
        sensor_data = {
            "eis": {
                "measurements": [MagicMock()],
                "properties": {"thickness_um": 11.0},
            },
            "qcm": {
                "measurement": MagicMock(),
                "properties": {"thickness_um": 11.0},
            },
        }
        result = m._apply_sensor_feedback(base_state, sensor_data, 1.0)
        assert result["sensor_status"] == "fused"
        assert result["sensor_validated_thickness"] == 11.0
        assert len(m.fusion_results) == 1


@pytest.mark.coverage_extra
class TestAdaptiveCalibration:
    """Cover _update_adaptive_parameters, _should_recalibrate, _recalibrate_parameters."""

    def test_should_recalibrate_24h(self):
        """Cover 24h recalibration trigger."""
        m = _make_model()
        m.last_calibration_time = 0.0
        assert m._should_recalibrate(25.0) is True

    def test_should_recalibrate_drift(self):
        """Cover systematic drift detection."""
        m = _make_model()
        m.last_calibration_time = 20.0
        m.prediction_errors = [{"thickness_error": 6.0} for _ in range(15)]
        assert m._should_recalibrate(21.0) is True

    def test_should_not_recalibrate(self):
        """Cover false return."""
        m = _make_model()
        m.last_calibration_time = 20.0
        m.prediction_errors = [{"thickness_error": 1.0} for _ in range(15)]
        assert m._should_recalibrate(21.0) is False

    def test_should_not_recalibrate_few_errors(self):
        """Cover branch with < 10 errors."""
        m = _make_model()
        m.last_calibration_time = 20.0
        m.prediction_errors = [{"thickness_error": 6.0} for _ in range(5)]
        assert m._should_recalibrate(21.0) is False

    def test_recalibrate_parameters_with_kinetic_params(self):
        """Cover kinetic parameter adjustment branch."""
        m = _make_model()
        m.kinetic_params = SimpleNamespace(mu_max=0.5)
        m._recalibrate_parameters(10.0, 15.0, 48.0)
        # correction_factor = 15.0 / 10.0 = 1.5
        assert m.kinetic_params.mu_max == pytest.approx(0.75)
        assert len(m.calibration_history) == 1
        assert m.last_calibration_time == 48.0

    def test_recalibrate_parameters_clamp(self):
        """Cover mu_max clamping to bounds."""
        m = _make_model()
        m.kinetic_params = SimpleNamespace(mu_max=1.5)
        m._recalibrate_parameters(1.0, 100.0, 48.0)
        # correction_factor = 100, 1.5 * 100 = 150 -> clipped to 2.0
        assert m.kinetic_params.mu_max == 2.0

    def test_recalibrate_parameters_zero_model_thickness(self):
        """Cover zero model_thickness case."""
        m = _make_model()
        m.kinetic_params = SimpleNamespace(mu_max=0.5)
        m._recalibrate_parameters(0.0, 15.0, 48.0)
        # correction_factor = 1.0 (default)
        assert m.kinetic_params.mu_max == pytest.approx(0.5)

    def test_recalibrate_parameters_no_kinetic_params(self):
        """Cover branch where model has no kinetic_params."""
        m = _make_model()
        # No kinetic_params attribute at all
        m._recalibrate_parameters(10.0, 15.0, 48.0)
        assert len(m.calibration_history) == 0

    def test_update_adaptive_parameters_no_recalibration(self):
        """Cover path where recalibration is not needed."""
        m = _make_model()
        m.last_calibration_time = 0.0
        validated_state = {
            "biofilm_thickness": 10.0,
            "sensor_validated_thickness": 10.5,
            "fusion_confidence": 0.9,
        }
        m._update_adaptive_parameters(validated_state, {"eis": {}}, 1.0)
        # Should not recalibrate since time < 24h and no drift

    def test_update_adaptive_parameters_triggers_recalibration(self):
        """Cover path where recalibration triggers and relative_error > threshold."""
        m = _make_model()
        m.last_calibration_time = 0.0
        m.kinetic_params = SimpleNamespace(mu_max=0.5)
        validated_state = {
            "biofilm_thickness": 10.0,
            "sensor_validated_thickness": 15.0,  # 50% error > 20% threshold
            "fusion_confidence": 0.9,
        }
        m._update_adaptive_parameters(validated_state, {"eis": {}}, 48.0)
        assert len(m.calibration_history) == 1

    def test_update_adaptive_parameters_below_threshold(self):
        """Cover path where relative_error < threshold (no recalibration)."""
        m = _make_model()
        m.last_calibration_time = 0.0
        validated_state = {
            "biofilm_thickness": 10.0,
            "sensor_validated_thickness": 10.5,  # 5% error < 20% threshold
            "fusion_confidence": 0.9,
        }
        m._update_adaptive_parameters(validated_state, {"eis": {}}, 48.0)
        assert len(m.calibration_history) == 0


@pytest.mark.coverage_extra
class TestUpdateConfidenceLevels:
    """Cover _update_confidence_levels."""

    def test_confidence_update_with_prediction_error(self):
        m = _make_model()
        m.thickness_prediction_error = 5.0
        validated_state = {"fusion_confidence": 0.9}
        m._update_confidence_levels(validated_state)
        # error_factor = max(0.1, 1.0 - 5.0/20.0) = 0.75
        expected_model = 1.0 * 0.9 + 0.75 * 0.1
        assert m.model_confidence == pytest.approx(expected_model)
        expected_sensor = 0.8 * 0.9 + 0.9 * 0.1
        assert m.sensor_confidence == pytest.approx(expected_sensor)

    def test_confidence_update_large_error(self):
        """Cover clamping of error_factor to 0.1."""
        m = _make_model()
        m.thickness_prediction_error = 25.0
        validated_state = {"fusion_confidence": 0.5}
        m._update_confidence_levels(validated_state)
        # error_factor = max(0.1, 1.0 - 25.0/20.0) = max(0.1, -0.25) = 0.1
        expected_model = 1.0 * 0.9 + 0.1 * 0.1
        assert m.model_confidence == pytest.approx(expected_model)


@pytest.mark.coverage_extra
class TestSensorDiagnostics:
    """Cover get_sensor_diagnostics branches."""

    def test_diagnostics_with_errors(self):
        m = _make_model()
        m.prediction_errors = [
            {"thickness_error": 2.0, "biomass_error": 0.5},
            {"thickness_error": 3.0, "biomass_error": 0.8},
        ]
        diag = m.get_sensor_diagnostics()
        assert "error_statistics" in diag
        assert diag["error_statistics"]["mean_thickness_error_um"] == pytest.approx(2.5)
        assert "eis_diagnostics" in diag
        assert "qcm_diagnostics" in diag
        assert "fusion_diagnostics" in diag
        assert "sensor_faults" in diag

    def test_diagnostics_without_errors(self):
        m = _make_model()
        m.prediction_errors = []
        diag = m.get_sensor_diagnostics()
        assert diag["error_statistics"] == {}

    def test_diagnostics_no_sensors(self):
        m = _make_model(enable_eis=False, enable_qcm=False)
        m.sensor_fusion = None
        diag = m.get_sensor_diagnostics()
        assert "eis_diagnostics" not in diag
        assert "qcm_diagnostics" not in diag
        assert "fusion_diagnostics" not in diag


@pytest.mark.coverage_extra
class TestResetSensorCalibration:
    """Cover reset_sensor_calibration."""

    def test_reset_with_sensor_models(self):
        m = _make_model()
        m.sensor_measurements = [1, 2]
        m.model_predictions = [3, 4]
        m.prediction_errors = [5]
        m.calibration_history = [6]
        m.fusion_results = [7]
        m.sensor_residuals = [8]
        m.eis_model.measurement_history = [1]
        m.eis_model.thickness_history = [2]
        m.eis_model.time_history = [3]
        m.qcm_model.measurement_history = [1]
        m.qcm_model.frequency_history = [2]
        m.qcm_model.mass_history = [3]
        m.qcm_model.time_history = [4]
        m.reset_sensor_calibration()
        assert m.model_confidence == 1.0
        assert m.sensor_confidence == 0.8
        assert m.last_calibration_time == 0.0
        assert len(m.sensor_measurements) == 0
        assert len(m.model_predictions) == 0

    def test_reset_without_sensor_models(self):
        m = _make_model(enable_eis=False, enable_qcm=False)
        m.eis_model = None
        m.qcm_model = None
        m.reset_sensor_calibration()
        assert m.model_confidence == 1.0


@pytest.mark.coverage_extra
class TestValidateSensorIntegration:
    """Cover validate_sensor_integration."""

    def test_validate_all_sensors(self):
        m = _make_model()
        m.eis_model.simulate_measurement.return_value = [MagicMock()]
        m.qcm_model.simulate_measurement.return_value = MagicMock()
        result = m.validate_sensor_integration()
        assert result["eis_model_available"] is True
        assert result["qcm_model_available"] is True
        assert result["sensor_fusion_available"] is True
        assert result["can_simulate_eis"] is True
        assert result["can_simulate_qcm"] is True
        assert result["can_perform_fusion"] is True

    def test_validate_eis_failure(self):
        m = _make_model()
        m.eis_model.simulate_measurement.side_effect = Exception("fail")
        m.qcm_model.simulate_measurement.return_value = MagicMock()
        result = m.validate_sensor_integration()
        assert result["can_simulate_eis"] is False
        # Fusion requires both EIS and QCM
        assert result["can_perform_fusion"] is False

    def test_validate_qcm_failure(self):
        m = _make_model()
        m.eis_model.simulate_measurement.return_value = [MagicMock()]
        m.qcm_model.simulate_measurement.side_effect = Exception("fail")
        result = m.validate_sensor_integration()
        assert result["can_simulate_qcm"] is False

    def test_validate_no_sensors(self):
        m = _make_model(enable_eis=False, enable_qcm=False)
        m.eis_model = None
        m.qcm_model = None
        m.sensor_fusion = None
        result = m.validate_sensor_integration()
        assert result["eis_model_available"] is False
        assert result["can_simulate_eis"] is False
        assert result["can_simulate_qcm"] is False
        assert result["can_perform_fusion"] is False


@pytest.mark.coverage_extra
class TestStepBiofilmDynamicsWithSensors:
    """Cover step_biofilm_dynamics_with_sensors."""

    def test_step_with_sensor_feedback(self):
        m = _make_model()
        base_state = {"biofilm_thickness": 10.0, "biomass_density": 5.0}
        with patch.object(m, "step_biofilm_dynamics", return_value=base_state):
            with patch.object(m, "_get_sensor_measurements", return_value={"eis": {}}):
                with patch.object(m, "_apply_sensor_feedback", return_value={
                    **base_state,
                    "sensor_validated_thickness": 10.5,
                    "sensor_status": "eis_only",
                }):
                    with patch.object(m, "_update_adaptive_parameters"):
                        result = m.step_biofilm_dynamics_with_sensors(
                            dt=0.1, anode_potential=-0.3,
                            substrate_supply=5.0, time_hours=1.0,
                        )
        assert result["sensor_validated_thickness"] == 10.5
        assert len(m.model_predictions) == 1
        assert len(m.sensor_measurements) == 1

    def test_step_without_sensor_data(self):
        m = _make_model()
        base_state = {"biofilm_thickness": 10.0, "biomass_density": 5.0}
        with patch.object(m, "step_biofilm_dynamics", return_value=base_state):
            with patch.object(m, "_get_sensor_measurements", return_value=None):
                result = m.step_biofilm_dynamics_with_sensors(
                    dt=0.1, anode_potential=-0.3,
                    substrate_supply=5.0, time_hours=1.0,
                )
        assert result["sensor_status"] == "unavailable"
        assert result["sensor_confidence"] == 0.0

    def test_step_with_feedback_disabled(self):
        m = _make_model()
        m.sensor_feedback_enabled = False
        base_state = {"biofilm_thickness": 10.0, "biomass_density": 5.0}
        with patch.object(m, "step_biofilm_dynamics", return_value=base_state):
            with patch.object(m, "_get_sensor_measurements", return_value={"eis": {}}):
                result = m.step_biofilm_dynamics_with_sensors(
                    dt=0.1, anode_potential=-0.3,
                    substrate_supply=5.0, time_hours=1.0,
                )
        assert result["sensor_status"] == "unavailable"

    def test_step_with_external_measurements(self):
        m = _make_model()
        base_state = {"biofilm_thickness": 10.0, "biomass_density": 5.0}
        external = {"eis": {"measurements": [], "properties": {"thickness_um": 12.0}}}
        with patch.object(m, "step_biofilm_dynamics", return_value=base_state):
            with patch.object(m, "_apply_sensor_feedback", return_value={
                **base_state, "sensor_status": "fused",
            }):
                with patch.object(m, "_update_adaptive_parameters"):
                    result = m.step_biofilm_dynamics_with_sensors(
                        dt=0.1, anode_potential=-0.3,
                        substrate_supply=5.0, time_hours=1.0,
                        external_measurements=external,
                    )
        assert len(m.sensor_measurements) == 1


@pytest.mark.coverage_extra
class TestGetSensorMeasurements:
    """Cover _get_sensor_measurements."""

    def test_external_measurements_returned(self):
        m = _make_model()
        ext = {"eis": {"data": 123}}
        result = m._get_sensor_measurements(10.0, 5.0, 1.0, external_measurements=ext)
        assert result == ext

    def test_eis_simulation(self):
        m = _make_model(enable_eis=True, enable_qcm=False)
        m.qcm_model = None
        m.eis_model.simulate_measurement.return_value = [MagicMock()]
        m.eis_model.get_biofilm_properties.return_value = {"thickness_um": 12.0}
        result = m._get_sensor_measurements(10.0, 5.0, 1.0)
        assert "eis" in result

    def test_qcm_simulation(self):
        m = _make_model(enable_eis=False, enable_qcm=True)
        m.eis_model = None
        m.qcm_model.electrode_area = 0.196e-4
        m.qcm_model.simulate_measurement.return_value = MagicMock()
        m.qcm_model.estimate_biofilm_properties.return_value = {"thickness_um": 11.0}
        result = m._get_sensor_measurements(10.0, 5.0, 1.0)
        assert "qcm" in result

    def test_eis_exception(self):
        m = _make_model(enable_eis=True, enable_qcm=False)
        m.qcm_model = None
        m.eis_model.simulate_measurement.side_effect = Exception("fail")
        result = m._get_sensor_measurements(10.0, 5.0, 1.0)
        assert result is None

    def test_qcm_exception(self):
        m = _make_model(enable_eis=False, enable_qcm=True)
        m.eis_model = None
        m.qcm_model.electrode_area = 0.196e-4
        m.qcm_model.simulate_measurement.side_effect = Exception("fail")
        result = m._get_sensor_measurements(10.0, 5.0, 1.0)
        assert result is None

    def test_no_sensors(self):
        m = _make_model(enable_eis=False, enable_qcm=False)
        m.eis_model = None
        m.qcm_model = None
        result = m._get_sensor_measurements(10.0, 5.0, 1.0)
        assert result is None


@pytest.mark.coverage_extra
class TestSensorCalibrationErrorException:
    def test_exception(self):
        with pytest.raises(SensorCalibrationError):
            raise SensorCalibrationError("test error")
