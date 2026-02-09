"""Coverage tests for enhanced_biofilm_model.py targeting 95%+ coverage."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Pre-mock sensing_models so the source try/except succeeds
_mock_eis_mod = MagicMock()
_mock_qcm_mod = MagicMock()
_mock_fusion_mod = MagicMock()

# Create real-ish enums via SimpleNamespace
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

    # Sensor models (mocked)
    m.eis_model = MagicMock() if enable_eis else None
    m.qcm_model = MagicMock() if enable_qcm else None
    m.sensor_fusion = MagicMock() if (enable_eis or enable_qcm) else None
    if m.qcm_model:
        m.qcm_model.electrode_area = 0.196e-4

    # Feedback params
    m.sensor_feedback_enabled = True
    m.adaptive_calibration_enabled = True
    m.calibration_threshold = 0.2
    m.sensor_weight_decay = 0.95

    # Confidence levels
    m.model_confidence = 1.0
    m.sensor_confidence = 0.8
    m.fusion_history_length = 50

    # Tracking
    m.sensor_measurements = []
    m.model_predictions = []
    m.fusion_results = []
    m.calibration_history = []
    m.prediction_errors = []
    m.sensor_residuals = []
    m.last_calibration_time = 0.0

    # State
    m.sensor_validated_thickness = 0.0
    m.sensor_validated_biomass = 0.0
    m.thickness_prediction_error = 0.0
    m.biomass_prediction_error = 0.0

    return m


def _base_state(thickness=10.0, biomass=5.0):
    """Return a minimal base state dict."""
    return {
        "biofilm_thickness": thickness,
        "biomass_density": biomass,
        "growth_rate": 0.1,
        "current_density": 2.5,
    }


# ---------------------------------------------------------------------------
# Tests: __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_real_init_default(self):
        """Test real __init__ with default params (sensors resolve to mocks)."""
        model = EnhancedBiofilmModel(
            species="geobacter",
            substrate="acetate",
            use_gpu=False,
        )
        assert model.enable_eis is True
        assert model.enable_qcm is True
        assert model.sensor_feedback_enabled is True
        assert model.model_confidence == 1.0
        assert model.sensor_confidence == 0.8
        assert model.sensor_measurements == []

    def test_real_init_no_sensors(self):
        model = EnhancedBiofilmModel(
            species="mixed",
            substrate="lactate",
            use_gpu=False,
            enable_eis=False,
            enable_qcm=False,
        )
        assert model.enable_eis is False
        assert model.enable_qcm is False

    def test_real_init_species_shewanella(self):
        model = EnhancedBiofilmModel(
            species="shewanella",
            substrate="lactate",
            use_gpu=False,
        )
        assert model.sensor_fusion_method == "kalman_filter"

    def test_real_init_fusion_methods(self):
        for method in ("kalman_filter", "weighted_average", "maximum_likelihood", "bayesian"):
            model = EnhancedBiofilmModel(
                species="mixed",
                use_gpu=False,
                sensor_fusion_method=method,
            )
            assert model.sensor_fusion_method == method


class TestSensorCalibrationError:
    def test_exception(self):
        with pytest.raises(SensorCalibrationError):
            raise SensorCalibrationError("calibration failed")


# ---------------------------------------------------------------------------
# Tests: step_biofilm_dynamics_with_sensors
# ---------------------------------------------------------------------------

class TestStepWithSensors:
    def test_basic_step(self):
        m = _make_model()
        m.step_biofilm_dynamics = MagicMock(return_value=_base_state())
        m._get_sensor_measurements = MagicMock(return_value=None)

        result = m.step_biofilm_dynamics_with_sensors(
            dt=0.1, anode_potential=-0.3, substrate_supply=1.0, time_hours=1.0,
        )
        assert result["sensor_status"] == "unavailable"
        assert result["sensor_confidence"] == 0.0
        assert len(m.model_predictions) == 1

    def test_step_with_sensor_feedback(self):
        m = _make_model()
        m.step_biofilm_dynamics = MagicMock(return_value=_base_state())
        sensor_data = {"eis": {"properties": {"thickness_um": 12.0, "measurement_quality": 0.9}, "measurements": []}}
        m._get_sensor_measurements = MagicMock(return_value=sensor_data)
        m._apply_sensor_feedback = MagicMock(return_value={**_base_state(), "sensor_validated_thickness": 12.0})
        m._update_adaptive_parameters = MagicMock()

        m.step_biofilm_dynamics_with_sensors(
            dt=0.1, anode_potential=-0.3, substrate_supply=1.0, time_hours=2.0,
        )
        m._apply_sensor_feedback.assert_called_once()
        m._update_adaptive_parameters.assert_called_once()
        assert len(m.sensor_measurements) == 1
        assert m.sensor_measurements[0]["time"] == 2.0

    def test_step_feedback_disabled(self):
        m = _make_model()
        m.sensor_feedback_enabled = False
        m.step_biofilm_dynamics = MagicMock(return_value=_base_state())
        sensor_data = {"eis": {"properties": {"thickness_um": 12.0}}}
        m._get_sensor_measurements = MagicMock(return_value=sensor_data)

        result = m.step_biofilm_dynamics_with_sensors(
            dt=0.1, anode_potential=-0.3, substrate_supply=1.0,
        )
        assert result["sensor_status"] == "unavailable"

    def test_step_adaptive_calibration_disabled(self):
        m = _make_model()
        m.adaptive_calibration_enabled = False
        m.step_biofilm_dynamics = MagicMock(return_value=_base_state())
        sensor_data = {"eis": {"properties": {"thickness_um": 12.0}}}
        m._get_sensor_measurements = MagicMock(return_value=sensor_data)
        m._apply_sensor_feedback = MagicMock(return_value=_base_state())
        m._update_adaptive_parameters = MagicMock()

        m.step_biofilm_dynamics_with_sensors(
            dt=0.1, anode_potential=-0.3, substrate_supply=1.0,
        )
        m._update_adaptive_parameters.assert_not_called()

    def test_step_with_external_measurements(self):
        m = _make_model()
        m.step_biofilm_dynamics = MagicMock(return_value=_base_state())
        ext = {"eis": {"properties": {"thickness_um": 15.0}, "measurements": []}}
        m._apply_sensor_feedback = MagicMock(return_value={**_base_state(), "sensor_validated_thickness": 15.0})
        m._update_adaptive_parameters = MagicMock()

        m.step_biofilm_dynamics_with_sensors(
            dt=0.1, anode_potential=-0.3, substrate_supply=1.0,
            external_measurements=ext,
        )
        assert len(m.sensor_measurements) == 1


# ---------------------------------------------------------------------------
# Tests: _get_sensor_measurements
# ---------------------------------------------------------------------------

class TestGetSensorMeasurements:
    def test_external_measurements_override(self):
        m = _make_model()
        ext = {"eis": {"thickness": 10.0}}
        result = m._get_sensor_measurements(10.0, 5.0, 1.0, external_measurements=ext)
        assert result is ext

    def test_simulated_eis_measurement(self):
        m = _make_model(enable_qcm=False)
        m.sensor_fusion = None
        m.eis_model.simulate_measurement.return_value = [{"impedance": 100}]
        m.eis_model.get_biofilm_properties.return_value = {"thickness_um": 11.0}

        result = m._get_sensor_measurements(10.0, 5.0, 1.0)
        assert "eis" in result
        assert result["eis"]["properties"]["thickness_um"] == 11.0

    def test_simulated_qcm_measurement(self):
        m = _make_model(enable_eis=False)
        m.eis_model = None
        m.sensor_fusion = None
        m.qcm_model.electrode_area = 0.196e-4
        m.qcm_model.simulate_measurement.return_value = {"frequency_shift": -500}
        m.qcm_model.estimate_biofilm_properties.return_value = {"thickness_um": 9.0}

        result = m._get_sensor_measurements(10.0, 5.0, 1.0)
        assert "qcm" in result
        assert result["qcm"]["properties"]["thickness_um"] == 9.0

    def test_both_sensors(self):
        m = _make_model()
        m.eis_model.simulate_measurement.return_value = [{"impedance": 100}]
        m.eis_model.get_biofilm_properties.return_value = {"thickness_um": 11.0}
        m.qcm_model.electrode_area = 0.196e-4
        m.qcm_model.simulate_measurement.return_value = {"frequency_shift": -500}
        m.qcm_model.estimate_biofilm_properties.return_value = {"thickness_um": 9.0}

        result = m._get_sensor_measurements(10.0, 5.0, 1.0)
        assert "eis" in result
        assert "qcm" in result

    def test_eis_exception_handled(self):
        m = _make_model(enable_qcm=False)
        m.sensor_fusion = None
        m.eis_model.simulate_measurement.side_effect = RuntimeError("EIS failed")

        result = m._get_sensor_measurements(10.0, 5.0, 1.0)
        assert result is None  # Empty measurements → None

    def test_qcm_exception_handled(self):
        m = _make_model(enable_eis=False)
        m.eis_model = None
        m.sensor_fusion = None
        m.qcm_model.simulate_measurement.side_effect = RuntimeError("QCM failed")

        result = m._get_sensor_measurements(10.0, 5.0, 1.0)
        assert result is None

    def test_no_sensors_returns_none(self):
        m = _make_model(enable_eis=False, enable_qcm=False)
        m.eis_model = None
        m.qcm_model = None
        m.sensor_fusion = None

        result = m._get_sensor_measurements(10.0, 5.0, 1.0)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: _apply_sensor_feedback
# ---------------------------------------------------------------------------

class TestApplySensorFeedback:
    def test_no_sensor_data(self):
        m = _make_model()
        result = m._apply_sensor_feedback(_base_state(), {}, 1.0)
        assert result["biofilm_thickness"] == 10.0

    def test_fusion_path(self):
        m = _make_model()
        fused = SimpleNamespace(
            thickness_um=11.5,
            biomass_density_g_per_L=5.2,
            thickness_uncertainty=0.5,
            biomass_uncertainty=0.3,
            fusion_confidence=0.92,
            sensor_agreement=0.95,
            eis_weight=0.6,
            qcm_weight=0.4,
        )
        m.sensor_fusion.fuse_measurements.return_value = fused

        sensor_data = {
            "eis": {"measurements": [{"impedance": 100}], "properties": {"thickness_um": 11.0}},
            "qcm": {"measurement": {"frequency_shift": -500}, "properties": {"thickness_um": 12.0}},
        }
        result = m._apply_sensor_feedback(_base_state(), sensor_data, 1.0)

        assert result["sensor_status"] == "fused"
        assert result["sensor_validated_thickness"] == 11.5
        assert result["sensor_validated_biomass"] == 5.2
        assert result["fusion_confidence"] == 0.92
        assert m.thickness_prediction_error == abs(10.0 - 11.5)
        assert len(m.fusion_results) == 1

    def test_fusion_exception_falls_back(self):
        m = _make_model()
        m.sensor_fusion.fuse_measurements.side_effect = RuntimeError("fusion error")

        sensor_data = {
            "eis": {"measurements": [{"impedance": 100}], "properties": {"thickness_um": 11.0, "measurement_quality": 0.85}},
            "qcm": {"measurement": {"freq": -500}, "properties": {"thickness_um": 12.0}},
        }
        result = m._apply_sensor_feedback(_base_state(), sensor_data, 1.0)
        # Falls back to EIS-only since fusion failed
        assert result["sensor_status"] == "eis_only"
        assert result["sensor_validated_thickness"] == 11.0

    def test_eis_only_path(self):
        m = _make_model(enable_qcm=False)
        m.sensor_fusion = None

        sensor_data = {
            "eis": {"measurements": [], "properties": {"thickness_um": 12.5, "measurement_quality": 0.88}},
        }
        result = m._apply_sensor_feedback(_base_state(), sensor_data, 1.0)

        assert result["sensor_status"] == "eis_only"
        assert result["sensor_validated_thickness"] == 12.5
        assert result["sensor_confidence"] == 0.88
        assert m.thickness_prediction_error == abs(10.0 - 12.5)

    def test_qcm_only_path(self):
        m = _make_model(enable_eis=False)
        m.sensor_fusion = None

        sensor_data = {
            "qcm": {"measurement": {"freq": -500}, "properties": {"thickness_um": 9.5, "biomass_density_g_per_L": 4.8, "measurement_quality": 0.75}},
        }
        result = m._apply_sensor_feedback(_base_state(), sensor_data, 1.0)

        assert result["sensor_status"] == "qcm_only"
        assert result["sensor_validated_thickness"] == 9.5
        assert result["sensor_validated_biomass"] == 4.8
        assert m.biomass_prediction_error == abs(5.0 - 4.8)

    def test_eis_zero_thickness_ignored(self):
        m = _make_model(enable_qcm=False)
        m.sensor_fusion = None

        sensor_data = {
            "eis": {"measurements": [], "properties": {"thickness_um": 0.0}},
        }
        result = m._apply_sensor_feedback(_base_state(), sensor_data, 1.0)
        assert "sensor_status" not in result or result.get("sensor_status") != "eis_only"

    def test_qcm_zero_thickness_ignored(self):
        m = _make_model(enable_eis=False)
        m.sensor_fusion = None

        sensor_data = {
            "qcm": {"measurement": {}, "properties": {"thickness_um": 0.0, "biomass_density_g_per_L": 0.0}},
        }
        result = m._apply_sensor_feedback(_base_state(), sensor_data, 1.0)
        assert result.get("sensor_status") != "qcm_only"

    def test_error_tracking_appended(self):
        m = _make_model(enable_qcm=False)
        m.sensor_fusion = None

        sensor_data = {
            "eis": {"measurements": [], "properties": {"thickness_um": 15.0, "measurement_quality": 0.9}},
        }
        m._apply_sensor_feedback(_base_state(), sensor_data, 5.0)
        assert len(m.prediction_errors) == 1
        assert m.prediction_errors[0]["time"] == 5.0
        assert m.prediction_errors[0]["thickness_error"] == 5.0

    def test_fusion_with_empty_eis_measurements(self):
        """Test fusion path when EIS measurements list is empty."""
        m = _make_model()
        sensor_data = {
            "eis": {"measurements": [], "properties": {"thickness_um": 11.0}},
            "qcm": {"measurement": {"freq": -500}, "properties": {"thickness_um": 12.0}},
        }
        # eis_measurement will be None because list is empty
        result = m._apply_sensor_feedback(_base_state(), sensor_data, 1.0)
        # Should fall through to individual sensor processing
        assert result.get("sensor_status") in ("eis_only", "qcm_only", None)


# ---------------------------------------------------------------------------
# Tests: _update_adaptive_parameters
# ---------------------------------------------------------------------------

class TestUpdateAdaptiveParameters:
    def test_no_recalibrate(self):
        m = _make_model()
        m._should_recalibrate = MagicMock(return_value=False)
        m._update_confidence_levels = MagicMock()

        state = {**_base_state(), "sensor_validated_thickness": 10.5}
        m._update_adaptive_parameters(state, {"eis": {}}, 1.0)
        m._update_confidence_levels.assert_not_called()

    def test_recalibrate_below_threshold(self):
        m = _make_model()
        m._should_recalibrate = MagicMock(return_value=True)
        m._recalibrate_parameters = MagicMock()
        m._update_confidence_levels = MagicMock()

        # 5% error < 20% threshold → no recalibration
        state = {**_base_state(thickness=10.0), "sensor_validated_thickness": 10.5}
        m._update_adaptive_parameters(state, {"eis": {}}, 1.0)
        m._recalibrate_parameters.assert_not_called()
        m._update_confidence_levels.assert_called_once()

    def test_recalibrate_above_threshold(self):
        m = _make_model()
        m._should_recalibrate = MagicMock(return_value=True)
        m._recalibrate_parameters = MagicMock()
        m._update_confidence_levels = MagicMock()

        # 50% error > 20% threshold → recalibrate
        state = {**_base_state(thickness=10.0), "sensor_validated_thickness": 15.0}
        m._update_adaptive_parameters(state, {"eis": {}}, 25.0)
        m._recalibrate_parameters.assert_called_once_with(10.0, 15.0, 25.0)

    def test_zero_sensor_thickness_skips(self):
        m = _make_model()
        m._should_recalibrate = MagicMock(return_value=True)
        m._recalibrate_parameters = MagicMock()
        m._update_confidence_levels = MagicMock()

        state = {**_base_state(thickness=10.0), "sensor_validated_thickness": 0.0}
        m._update_adaptive_parameters(state, {"eis": {}}, 1.0)
        m._recalibrate_parameters.assert_not_called()

    def test_missing_sensor_thickness_uses_model(self):
        m = _make_model()
        m._should_recalibrate = MagicMock(return_value=True)
        m._recalibrate_parameters = MagicMock()
        m._update_confidence_levels = MagicMock()

        state = _base_state(thickness=10.0)  # No sensor_validated_thickness
        m._update_adaptive_parameters(state, {"eis": {}}, 1.0)
        # sensor_thickness defaults to model_thickness → error = 0 → no recalibrate
        m._recalibrate_parameters.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: _should_recalibrate
# ---------------------------------------------------------------------------

class TestShouldRecalibrate:
    def test_time_based_recalibration(self):
        m = _make_model()
        m.last_calibration_time = 0.0
        assert m._should_recalibrate(25.0) is True

    def test_no_recalibration_within_24h(self):
        m = _make_model()
        m.last_calibration_time = 0.0
        assert m._should_recalibrate(10.0) is False

    def test_drift_based_recalibration(self):
        m = _make_model()
        m.last_calibration_time = 0.0
        # 10 errors with mean > 5.0 → drift detected
        m.prediction_errors = [{"thickness_error": 6.0} for _ in range(10)]
        assert m._should_recalibrate(10.0) is True

    def test_no_drift_with_small_errors(self):
        m = _make_model()
        m.last_calibration_time = 0.0
        m.prediction_errors = [{"thickness_error": 2.0} for _ in range(10)]
        assert m._should_recalibrate(10.0) is False

    def test_not_enough_errors_for_drift(self):
        m = _make_model()
        m.last_calibration_time = 0.0
        m.prediction_errors = [{"thickness_error": 10.0} for _ in range(5)]
        assert m._should_recalibrate(10.0) is False


# ---------------------------------------------------------------------------
# Tests: _recalibrate_parameters
# ---------------------------------------------------------------------------

class TestRecalibrateParameters:
    def test_recalibration_adjusts_mu_max(self):
        m = _make_model()
        m.kinetic_params = SimpleNamespace(mu_max=0.5)

        m._recalibrate_parameters(model_thickness=10.0, sensor_thickness=15.0, time_hours=25.0)

        # correction_factor = 15/10 = 1.5, new mu_max = 0.5 * 1.5 = 0.75
        assert abs(m.kinetic_params.mu_max - 0.75) < 1e-6
        assert len(m.calibration_history) == 1
        assert m.calibration_history[0]["correction_factor"] == 1.5
        assert m.last_calibration_time == 25.0

    def test_mu_max_clipped_to_upper_bound(self):
        m = _make_model()
        m.kinetic_params = SimpleNamespace(mu_max=1.5)

        m._recalibrate_parameters(model_thickness=5.0, sensor_thickness=50.0, time_hours=25.0)
        # correction = 10x, mu_max = 1.5 * 10 = 15.0, clipped to 2.0
        assert m.kinetic_params.mu_max == 2.0

    def test_mu_max_clipped_to_lower_bound(self):
        m = _make_model()
        m.kinetic_params = SimpleNamespace(mu_max=0.1)

        m._recalibrate_parameters(model_thickness=100.0, sensor_thickness=1.0, time_hours=25.0)
        # correction = 0.01, mu_max = 0.1 * 0.01 = 0.001, clipped to 0.01
        assert m.kinetic_params.mu_max == 0.01

    def test_zero_model_thickness(self):
        m = _make_model()
        m.kinetic_params = SimpleNamespace(mu_max=0.5)

        m._recalibrate_parameters(model_thickness=0.0, sensor_thickness=10.0, time_hours=25.0)
        # correction_factor = 1.0 when model_thickness is 0
        assert m.kinetic_params.mu_max == 0.5

    def test_no_kinetic_params(self):
        m = _make_model()
        # No kinetic_params attribute → exception silently caught
        m._recalibrate_parameters(model_thickness=10.0, sensor_thickness=15.0, time_hours=25.0)
        assert len(m.calibration_history) == 0


# ---------------------------------------------------------------------------
# Tests: _update_confidence_levels
# ---------------------------------------------------------------------------

class TestUpdateConfidenceLevels:
    def test_updates_model_confidence(self):
        m = _make_model()
        m.thickness_prediction_error = 5.0  # 5 μm error
        m.model_confidence = 0.8

        state = {"fusion_confidence": 0.9}
        m._update_confidence_levels(state)

        # error_factor = max(0.1, 1 - 5/20) = 0.75
        # new_confidence = 0.8 * 0.9 + 0.75 * 0.1 = 0.72 + 0.075 = 0.795
        assert abs(m.model_confidence - 0.795) < 1e-6

    def test_updates_sensor_confidence(self):
        m = _make_model()
        m.thickness_prediction_error = 0.0
        m.sensor_confidence = 0.5

        state = {"fusion_confidence": 0.95}
        m._update_confidence_levels(state)

        # sensor_confidence = 0.5 * 0.9 + 0.95 * 0.1 = 0.45 + 0.095 = 0.545
        assert abs(m.sensor_confidence - 0.545) < 1e-6

    def test_large_error_clamps_factor(self):
        m = _make_model()
        m.thickness_prediction_error = 30.0  # Very large error
        m.model_confidence = 0.5

        state = {"fusion_confidence": 0.5}
        m._update_confidence_levels(state)

        # error_factor = max(0.1, 1 - 30/20) = max(0.1, -0.5) = 0.1
        # new_confidence = 0.5 * 0.9 + 0.1 * 0.1 = 0.45 + 0.01 = 0.46
        assert abs(m.model_confidence - 0.46) < 1e-6

    def test_missing_fusion_confidence_defaults(self):
        m = _make_model()
        m.thickness_prediction_error = 0.0

        state = {}  # No fusion_confidence
        m._update_confidence_levels(state)
        # Uses default 0.5; just verify no error is raised
        assert m.sensor_confidence is not None


# ---------------------------------------------------------------------------
# Tests: get_sensor_diagnostics
# ---------------------------------------------------------------------------

class TestGetSensorDiagnostics:
    def test_basic_diagnostics(self):
        m = _make_model()
        diag = m.get_sensor_diagnostics()

        assert diag["sensor_configuration"]["eis_enabled"] is True
        assert diag["sensor_configuration"]["qcm_enabled"] is True
        assert diag["confidence_levels"]["model_confidence"] == 1.0
        assert diag["measurement_statistics"]["total_measurements"] == 0

    def test_with_prediction_errors(self):
        m = _make_model()
        m.prediction_errors = [
            {"thickness_error": 2.0, "biomass_error": 0.5},
            {"thickness_error": 4.0, "biomass_error": 1.0},
        ]
        diag = m.get_sensor_diagnostics()

        assert diag["error_statistics"]["mean_thickness_error_um"] == 3.0
        assert diag["error_statistics"]["max_thickness_error_um"] == 4.0

    def test_with_eis_qcm_diagnostics(self):
        m = _make_model()
        m.eis_model.get_measurement_summary.return_value = {"count": 10}
        m.qcm_model.get_measurement_summary.return_value = {"count": 5}
        m.sensor_fusion.get_fusion_summary.return_value = {"method": "kalman"}
        m.sensor_fusion.detect_sensor_faults.return_value = {"faults": []}

        diag = m.get_sensor_diagnostics()
        assert diag["eis_diagnostics"]["count"] == 10
        assert diag["qcm_diagnostics"]["count"] == 5
        assert "fusion_diagnostics" in diag
        assert "sensor_faults" in diag

    def test_no_sensors(self):
        m = _make_model(enable_eis=False, enable_qcm=False)
        m.eis_model = None
        m.qcm_model = None
        m.sensor_fusion = None

        diag = m.get_sensor_diagnostics()
        assert "eis_diagnostics" not in diag
        assert "qcm_diagnostics" not in diag
        assert "fusion_diagnostics" not in diag


# ---------------------------------------------------------------------------
# Tests: reset_sensor_calibration
# ---------------------------------------------------------------------------

class TestResetSensorCalibration:
    def test_resets_confidence(self):
        m = _make_model()
        m.model_confidence = 0.5
        m.sensor_confidence = 0.3
        m.last_calibration_time = 100.0
        m.sensor_measurements = [{"time": 1}]
        m.model_predictions = [{"time": 1}]
        m.fusion_results = [1, 2, 3]
        m.calibration_history = [{"time": 1}]
        m.prediction_errors = [{"error": 1}]
        m.sensor_residuals = [0.1]

        m.reset_sensor_calibration()

        assert m.model_confidence == 1.0
        assert m.sensor_confidence == 0.8
        assert m.last_calibration_time == 0.0
        assert m.sensor_measurements == []
        assert m.model_predictions == []
        assert m.fusion_results == []
        assert m.calibration_history == []
        assert m.prediction_errors == []
        assert m.sensor_residuals == []

    def test_resets_eis_history(self):
        m = _make_model(enable_qcm=False)
        # Leave attributes as MagicMock so .clear() is trackable
        m.reset_sensor_calibration()
        m.eis_model.measurement_history.clear.assert_called_once()
        m.eis_model.thickness_history.clear.assert_called_once()
        m.eis_model.time_history.clear.assert_called_once()

    def test_resets_qcm_history(self):
        m = _make_model(enable_eis=False)
        # Leave attributes as MagicMock so .clear() is trackable
        m.reset_sensor_calibration()
        m.qcm_model.measurement_history.clear.assert_called_once()
        m.qcm_model.frequency_history.clear.assert_called_once()
        m.qcm_model.mass_history.clear.assert_called_once()
        m.qcm_model.time_history.clear.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: validate_sensor_integration
# ---------------------------------------------------------------------------

class TestValidateSensorIntegration:
    def test_all_sensors_available(self):
        m = _make_model()
        m.eis_model.simulate_measurement.return_value = [{"impedance": 100}]
        m.qcm_model.simulate_measurement.return_value = {"freq": -500}

        result = m.validate_sensor_integration()
        assert result["eis_model_available"] is True
        assert result["qcm_model_available"] is True
        assert result["sensor_fusion_available"] is True
        assert result["can_simulate_eis"] is True
        assert result["can_simulate_qcm"] is True
        assert result["can_perform_fusion"] is True

    def test_no_sensors(self):
        m = _make_model(enable_eis=False, enable_qcm=False)
        m.eis_model = None
        m.qcm_model = None
        m.sensor_fusion = None

        result = m.validate_sensor_integration()
        assert result["eis_model_available"] is False
        assert result["qcm_model_available"] is False
        assert result["can_simulate_eis"] is False
        assert result["can_simulate_qcm"] is False
        assert result["can_perform_fusion"] is False

    def test_eis_simulation_fails(self):
        m = _make_model()
        m.eis_model.simulate_measurement.side_effect = RuntimeError("fail")
        m.qcm_model.simulate_measurement.return_value = {"freq": -500}

        result = m.validate_sensor_integration()
        assert result["can_simulate_eis"] is False
        assert result["can_simulate_qcm"] is True
        assert result["can_perform_fusion"] is False  # Needs both

    def test_qcm_simulation_fails(self):
        m = _make_model()
        m.eis_model.simulate_measurement.return_value = [{"impedance": 100}]
        m.qcm_model.simulate_measurement.side_effect = RuntimeError("fail")

        result = m.validate_sensor_integration()
        assert result["can_simulate_eis"] is True
        assert result["can_simulate_qcm"] is False
        assert result["can_perform_fusion"] is False

    def test_eis_empty_measurements(self):
        m = _make_model()
        m.eis_model.simulate_measurement.return_value = []
        m.qcm_model.simulate_measurement.return_value = {"freq": -500}

        result = m.validate_sensor_integration()
        assert result["can_simulate_eis"] is False
        assert result["can_perform_fusion"] is False

    def test_qcm_returns_none(self):
        m = _make_model()
        m.eis_model.simulate_measurement.return_value = [{"impedance": 100}]
        m.qcm_model.simulate_measurement.return_value = None

        result = m.validate_sensor_integration()
        assert result["can_simulate_qcm"] is False


# Clean up mocked sys.modules after import
for _k in list(sys.modules):
    if _k in ("sensing_models", "sensing_models.eis_model",
              "sensing_models.qcm_model", "sensing_models.sensor_fusion",
              "gpu_acceleration"):
        del sys.modules[_k]
