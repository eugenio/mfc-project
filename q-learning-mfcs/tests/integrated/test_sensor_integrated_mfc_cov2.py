"""Tests for sensor_integrated_mfc_model.py - coverage part 2.

Covers: _aggregate_sensor_data, _calculate_biofilm_current_enhancement,
_calculate_model_validation_score, _handle_sensor_faults,
_compile_sensor_results, get_sensor_diagnostics, _save_sensor_checkpoint,
run_sensor_integrated_simulation, main().
"""
import sys
import os
import pickle
from unittest.mock import MagicMock, patch, mock_open
from dataclasses import dataclass

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Mock heavy dependencies before import
mock_biofilm = MagicMock()
mock_integrated = MagicMock()
mock_sensing_ctrl = MagicMock()
mock_gpu = MagicMock()
mock_recirc = MagicMock()
mock_path_config = MagicMock()
mock_metabolic = MagicMock()

# Create mock classes
mock_integrated.IntegratedMFCModel = type("IntegratedMFCModel", (), {})
mock_integrated.IntegratedMFCState = type("IntegratedMFCState", (), {})

sys.modules["biofilm_kinetics"] = MagicMock()
sys.modules["biofilm_kinetics.enhanced_biofilm_model"] = mock_biofilm
sys.modules["integrated_mfc_model"] = mock_integrated
sys.modules["sensing_enhanced_q_controller"] = mock_sensing_ctrl
sys.modules["gpu_acceleration"] = mock_gpu
sys.modules["mfc_recirculation_control"] = mock_recirc
sys.modules["path_config"] = mock_path_config
sys.modules["metabolic_model"] = mock_metabolic
sys.modules["sensing_models"] = MagicMock()
sys.modules["sensing_models.eis_model"] = MagicMock()
sys.modules["sensing_models.qcm_model"] = MagicMock()
sys.modules["sensing_models.sensor_fusion"] = MagicMock()

mock_path_config.get_model_path = MagicMock(return_value="/tmp/test.pkl")
mock_gpu.get_gpu_accelerator = MagicMock(return_value=MagicMock(
    is_gpu_available=MagicMock(return_value=False)))

import sensor_integrated_mfc_model as sim_mod

# Clean up mocked sys.modules entries after import to avoid polluting
# other tests in the full suite (e.g. metabolic_model, sensing_models).
for _k in list(sys.modules):
    if _k in (
        "biofilm_kinetics", "biofilm_kinetics.enhanced_biofilm_model",
        "integrated_mfc_model", "sensing_enhanced_q_controller",
        "gpu_acceleration", "mfc_recirculation_control", "path_config",
        "metabolic_model", "sensing_models", "sensing_models.eis_model",
        "sensing_models.qcm_model", "sensing_models.sensor_fusion",
    ):
        del sys.modules[_k]


def _make_model():
    """Create a SensorIntegratedMFCModel with mocked internals."""
    with patch.object(sim_mod.SensorIntegratedMFCModel, "__init__",
                      lambda self, *a, **kw: None):
        model = sim_mod.SensorIntegratedMFCModel()

    model.n_cells = 3
    model.enable_eis = False
    model.enable_qcm = False
    model.sensor_fusion_method = "kalman_filter"
    model.sensor_feedback_weight = 0.3
    model.time = 10.0
    model.simulation_hours = 100
    model.history = []
    model.sensor_history = []
    model.sensor_fault_history = []
    model.adaptive_calibration_events = []
    model.total_energy_generated = 5.0
    model.pump_power_consumed = 0.1
    model.flow_rate_ml_h = 10.0
    model.eis_models = []
    model.qcm_models = []
    model.sensor_fusion_models = []
    model.biofilm_models = [MagicMock() for _ in range(3)]
    model.flow_controller = MagicMock()
    model.flow_controller.q_table = {"s1": [0.1]}
    model.flow_controller.epsilon = 0.1
    model.performance_metrics = {"total_energy": 0.0}
    model.sensor_states = [
        {"eis_status": "disabled", "qcm_status": "disabled"}
        for _ in range(3)
    ]
    return model


@pytest.mark.coverage_extra
class TestAggregateSensorData:
    """Cover _aggregate_sensor_data lines 644-715."""

    def test_empty_measurements(self):
        model = _make_model()
        result = model._aggregate_sensor_data([])
        assert result == {}

    def test_none_measurements(self):
        model = _make_model()
        result = model._aggregate_sensor_data([{}, {}, {}])
        assert result == {}

    def test_eis_only(self):
        model = _make_model()
        measurements = [
            {"eis": {
                "properties": {
                    "thickness_um": 25.0,
                    "conductivity_S_per_m": 0.5,
                    "measurement_quality": 0.9,
                },
            }},
        ]
        result = model._aggregate_sensor_data(measurements)
        assert "eis" in result
        assert result["eis"]["thickness_um"] == 25.0

    def test_qcm_only(self):
        model = _make_model()
        mock_meas = MagicMock()
        mock_meas.frequency_shift = -500.0
        mock_meas.dissipation = 1e-6
        measurements = [
            {"qcm": {
                "properties": {"mass_per_area_ng_per_cm2": 100.0},
                "measurement": mock_meas,
            }},
        ]
        result = model._aggregate_sensor_data(measurements)
        assert "qcm" in result

    def test_fusion_data(self):
        model = _make_model()
        measurements = [
            {"fusion": {
                "sensor_agreement": 0.9,
                "fusion_confidence": 0.85,
            }},
        ]
        result = model._aggregate_sensor_data(measurements)
        assert "fusion" in result
        assert result["fusion"]["sensor_agreement"] == 0.9


@pytest.mark.coverage_extra
class TestBiofilmCurrentEnhancement:
    """Cover _calculate_biofilm_current_enhancement lines 717-742."""

    def test_with_fusion_data(self):
        model = _make_model()
        biofilm_state = {"biofilm_thickness": 20.0}
        sensor_data = {
            "fusion": {"thickness": 25.0, "fusion_confidence": 0.9},
        }
        result = model._calculate_biofilm_current_enhancement(
            biofilm_state, sensor_data)
        assert result > 0

    def test_with_eis_data(self):
        model = _make_model()
        biofilm_state = {"biofilm_thickness": 20.0}
        sensor_data = {
            "eis": {
                "properties": {
                    "thickness_um": 22.0,
                    "measurement_quality": 0.8,
                },
            },
        }
        result = model._calculate_biofilm_current_enhancement(
            biofilm_state, sensor_data)
        assert result > 0

    def test_no_sensor_data(self):
        model = _make_model()
        biofilm_state = {"biofilm_thickness": 20.0}
        sensor_data = {}
        result = model._calculate_biofilm_current_enhancement(
            biofilm_state, sensor_data)
        assert result > 0


@pytest.mark.coverage_extra
class TestModelValidationScore:
    """Cover _calculate_model_validation_score lines 744-783."""

    def test_no_sensor_data(self):
        model = _make_model()
        score = model._calculate_model_validation_score(
            [{"biofilm_thickness": 20.0, "biomass_density": 5.0}],
            [{}],
        )
        assert score == 0.5

    def test_with_fusion_data(self):
        model = _make_model()
        biofilm_states = [
            {"biofilm_thickness": 20.0, "biomass_density": 5.0},
        ]
        sensor_data = [
            {"fusion": {"thickness": 21.0, "biomass": 5.2}},
        ]
        score = model._calculate_model_validation_score(
            biofilm_states, sensor_data)
        assert 0.0 <= score <= 1.0

    def test_empty_sensor_list(self):
        model = _make_model()
        score = model._calculate_model_validation_score([], [])
        assert score == 0.5


@pytest.mark.coverage_extra
class TestHandleSensorFaults:
    """Cover _handle_sensor_faults lines 785-812."""

    def test_eis_fault(self):
        model = _make_model()
        model.flow_controller.handle_sensor_fault = MagicMock()
        measurements = [
            {"eis": {"status": "failed"}},
        ]
        model._handle_sensor_faults(measurements)
        assert len(model.sensor_fault_history) == 1
        assert model.sensor_fault_history[0]["sensor"] == "eis"

    def test_qcm_fault(self):
        model = _make_model()
        model.flow_controller.handle_sensor_fault = MagicMock()
        measurements = [
            {"qcm": {"status": "failed"}},
        ]
        model._handle_sensor_faults(measurements)
        assert len(model.sensor_fault_history) == 1
        assert model.sensor_fault_history[0]["sensor"] == "qcm"

    def test_no_faults(self):
        model = _make_model()
        measurements = [
            {"eis": {"status": "good"}, "qcm": {"status": "good"}},
        ]
        model._handle_sensor_faults(measurements)
        assert len(model.sensor_fault_history) == 0


@pytest.mark.coverage_extra
class TestCompileSensorResults:
    """Cover _compile_sensor_results lines 1016-1060."""

    def test_empty_history(self):
        model = _make_model()
        result = model._compile_sensor_results()
        assert result == {}

    def test_with_history(self):
        model = _make_model()
        mock_state = MagicMock()
        mock_state.model_sensor_agreement = 0.85
        mock_state.fusion_confidence = [0.9, 0.85]
        mock_state.thickness_prediction_error = [0.5, 0.3]
        model.history = [mock_state]
        # Patch SENSING_AVAILABLE
        with patch.object(sim_mod, "SENSING_AVAILABLE", True):
            result = model._compile_sensor_results()
        assert "sensor_integration_score" in result

    def test_with_controller_performance(self):
        model = _make_model()
        mock_state = MagicMock()
        mock_state.model_sensor_agreement = 0.8
        mock_state.fusion_confidence = [0.9]
        mock_state.thickness_prediction_error = [0.5]
        model.history = [mock_state]
        model.flow_controller.get_controller_performance_summary = MagicMock(
            return_value={"metric": 0.5})
        with patch.object(sim_mod, "SENSING_AVAILABLE", True):
            result = model._compile_sensor_results()
        assert "controller_performance" in result


@pytest.mark.coverage_extra
class TestGetSensorDiagnostics:
    """Cover get_sensor_diagnostics lines 1062-1103."""

    def test_basic_diagnostics(self):
        model = _make_model()
        with patch.object(sim_mod, "SENSING_AVAILABLE", False):
            diag = model.get_sensor_diagnostics()
        assert "sensor_availability" in diag
        assert "sensor_performance" in diag
        assert diag["sensor_availability"]["eis_enabled"] is False

    def test_with_biofilm_diagnostics(self):
        model = _make_model()
        model.biofilm_models[0].get_sensor_diagnostics = MagicMock(
            return_value={"health": "good"})
        with patch.object(sim_mod, "SENSING_AVAILABLE", False):
            diag = model.get_sensor_diagnostics()
        assert "cell_0_biofilm_sensor_diagnostics" in diag

    def test_with_controller_diagnostics(self):
        model = _make_model()
        model.flow_controller.get_controller_performance_summary = MagicMock(
            return_value={"quality": 0.9})
        with patch.object(sim_mod, "SENSING_AVAILABLE", False):
            diag = model.get_sensor_diagnostics()
        assert "controller_diagnostics" in diag


@pytest.mark.coverage_extra
class TestSaveSensorCheckpoint:
    """Cover _save_sensor_checkpoint lines 995-1014."""

    def test_save_checkpoint(self):
        model = _make_model()
        model.history = [MagicMock()]
        model.sensor_history = [MagicMock()]
        m = mock_open()
        with patch("builtins.open", m):
            with patch("pickle.dump") as mock_dump:
                model._save_sensor_checkpoint(10)
                mock_dump.assert_called_once()


@pytest.mark.coverage_extra
class TestRunSensorIntegratedSimulation:
    """Cover run_sensor_integrated_simulation lines 954-993."""

    def test_short_simulation(self):
        model = _make_model()
        model.simulation_hours = 3
        model.step_sensor_integrated_dynamics = MagicMock()
        model._save_sensor_checkpoint = MagicMock()
        model._compile_sensor_results = MagicMock(return_value={"ok": True})
        with patch.object(sim_mod, "SENSING_AVAILABLE", False):
            result = model.run_sensor_integrated_simulation(dt=1.0, save_interval=2)
        assert "computation_time" in result
        assert model.step_sensor_integrated_dynamics.call_count == 3

    def test_simulation_with_save(self):
        model = _make_model()
        model.simulation_hours = 5
        model.step_sensor_integrated_dynamics = MagicMock()
        model._save_sensor_checkpoint = MagicMock()
        model._compile_sensor_results = MagicMock(return_value={})
        with patch.object(sim_mod, "SENSING_AVAILABLE", False):
            model.run_sensor_integrated_simulation(dt=1.0, save_interval=2)
        # Should save at hour 2 and 4
        assert model._save_sensor_checkpoint.call_count >= 1


@pytest.mark.coverage_extra
class TestMain:
    """Cover main() function lines 1106-1181."""

    def test_main_basic(self):
        mock_model = MagicMock()
        mock_model.run_sensor_integrated_simulation.return_value = {}
        mock_model.get_sensor_diagnostics.return_value = {
            "sensor_availability": {"eis_enabled": True},
            "sensor_performance": {"total": 100},
            "simple_key": "value",
        }
        with patch.object(sim_mod, "SensorIntegratedMFCModel",
                          return_value=mock_model):
            with patch("sys.argv", ["prog", "--cells", "2", "--hours", "10"]):
                sim_mod.main()
        mock_model.run_sensor_integrated_simulation.assert_called_once()

    def test_main_with_save_and_plot(self):
        mock_model = MagicMock()
        mock_model.run_sensor_integrated_simulation.return_value = {}
        mock_model.get_sensor_diagnostics.return_value = {}
        mock_model.save_results = MagicMock()
        mock_model.plot_results = MagicMock()
        with patch.object(sim_mod, "SensorIntegratedMFCModel",
                          return_value=mock_model):
            with patch("sys.argv", [
                "prog", "--cells", "2", "--hours", "5", "--plot"]):
                sim_mod.main()
