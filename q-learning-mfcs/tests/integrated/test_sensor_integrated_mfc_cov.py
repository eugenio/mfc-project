"""Tests for sensor_integrated_mfc_model.py - coverage target 98%+."""
import sys
import os
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.fixture
def model():
    """Create a SensorIntegratedMFCModel for testing."""
    from sensor_integrated_mfc_model import SensorIntegratedMFCModel

    m = SensorIntegratedMFCModel(
        n_cells=2,
        species="mixed",
        substrate="lactate",
        use_gpu=False,
        simulation_hours=10,
        enable_eis=True,
        enable_qcm=True,
    )
    # Ensure the flow controller has `actions` attribute
    if not hasattr(m.flow_controller, "actions"):
        m.flow_controller.actions = np.array(
            [-10, -5, -2, -1, 0, 1, 2, 5, 10]
        )
    return m


class TestSensorIntegratedMFCState:
    def test_dataclass_fields(self):
        from sensor_integrated_mfc_model import SensorIntegratedMFCState

        state = SensorIntegratedMFCState(
            time=1.0,
            total_energy=0.5,
            average_power=0.05,
            coulombic_efficiency=0.7,
            biofilm_thickness=[10.0, 12.0],
            biomass_density=[0.5, 0.6],
            attachment_fraction=[0.5, 0.5],
            substrate_concentration=[1.0, 0.8],
            nadh_ratio=[0.3, 0.3],
            atp_level=[2.0, 2.0],
            electron_flux=[0.01, 0.01],
            cell_voltages=[0.35, 0.35],
            current_densities=[0.1, 0.1],
            anode_potentials=[-0.3, -0.3],
            reservoir_concentration=20.0,
            flow_rate=10.0,
            pump_power=0.001,
            epsilon=0.3,
            q_table_size=100,
            learning_progress=0.5,
            eis_thickness=[10.0, 12.0],
            eis_conductivity=[0.1, 0.1],
            eis_measurement_quality=[0.9, 0.9],
            qcm_mass_per_area=[100.0, 120.0],
            qcm_frequency_shift=[-200.0, -220.0],
            qcm_dissipation=[1e-6, 1e-6],
            fused_thickness=[10.0, 12.0],
            fused_biomass=[0.5, 0.6],
            sensor_agreement=[0.9, 0.9],
            fusion_confidence=[0.85, 0.85],
            eis_sensor_status=["good", "good"],
            qcm_sensor_status=["good", "good"],
            sensor_fault_flags=[False, False],
            thickness_prediction_error=[0.1, 0.2],
            biomass_prediction_error=[0.05, 0.05],
            model_sensor_agreement=0.9,
            adaptive_calibration_active=False,
        )
        assert state.time == 1.0
        assert len(state.eis_thickness) == 2


class TestSensorIntegratedMFCModel:
    def test_init(self, model):
        assert model.n_cells == 2
        assert model.enable_eis is True
        assert model.enable_qcm is True
        assert len(model.biofilm_models) == 2
        assert len(model.metabolic_models) == 2

    def test_init_no_sensors(self):
        from sensor_integrated_mfc_model import SensorIntegratedMFCModel

        m = SensorIntegratedMFCModel(
            n_cells=2,
            use_gpu=False,
            simulation_hours=5,
            enable_eis=False,
            enable_qcm=False,
        )
        assert m.enable_eis is False
        assert m.enable_qcm is False

    def test_initialize_tracking(self, model):
        assert model.time == 0.0
        assert model.history == []
        assert "total_energy" in model.performance_metrics

    def test_initialize_recirculation(self, model):
        assert model.reservoir is not None
        assert len(model.mfc_cells) == 2
        assert model.flow_controller is not None
        assert model.substrate_controller is not None


class TestStepSensorIntegratedDynamics:
    def test_single_step(self, model):
        state = model.step_sensor_integrated_dynamics(dt=1.0)
        assert state.time > 0
        assert len(state.biofilm_thickness) == 2
        assert len(state.eis_thickness) == 2

    def test_multiple_steps(self, model):
        for _ in range(3):
            state = model.step_sensor_integrated_dynamics(dt=1.0)
        assert state.time == pytest.approx(3.0, abs=0.1)
        assert len(model.history) == 3


class TestCollectCellSensorData:
    def test_with_sensors(self, model):
        biofilm_state = {
            "biofilm_thickness": 15.0,
            "biomass_density": 0.5,
            "sensor_validated_thickness": 15.0,
            "sensor_validated_biomass": 0.5,
        }
        data = model._collect_cell_sensor_data(0, biofilm_state, 1.0)
        assert "eis" in data
        assert "qcm" in data
        assert "fusion" in data


class TestAggregateSensorData:
    def test_empty(self, model):
        result = model._aggregate_sensor_data([])
        assert result == {}

    def test_with_data(self, model):
        sensor_data = [{
            "eis": {
                "properties": {
                    "thickness_um": 15.0,
                    "conductivity_S_per_m": 0.1,
                    "measurement_quality": 0.9,
                },
            },
            "qcm": {
                "properties": {"mass_per_area_ng_per_cm2": 100.0},
                "measurement": MagicMock(
                    frequency_shift=-200.0, dissipation=1e-6
                ),
            },
            "fusion": {
                "sensor_agreement": 0.9,
                "fusion_confidence": 0.85,
            },
        }]
        result = model._aggregate_sensor_data(sensor_data)
        assert "eis" in result
        assert "qcm" in result
        assert "fusion" in result


class TestBiofilmCurrentEnhancement:
    def test_with_fusion(self, model):
        biofilm_state = {"biofilm_thickness": 15.0}
        sensor_data = {
            "fusion": {"thickness": 15.0, "fusion_confidence": 0.9},
        }
        current = model._calculate_biofilm_current_enhancement(
            biofilm_state, sensor_data
        )
        assert current > 0

    def test_with_eis_only(self, model):
        biofilm_state = {"biofilm_thickness": 15.0}
        sensor_data = {
            "eis": {
                "properties": {
                    "thickness_um": 15.0,
                    "measurement_quality": 0.8,
                },
            },
        }
        current = model._calculate_biofilm_current_enhancement(
            biofilm_state, sensor_data
        )
        assert current > 0

    def test_no_sensor(self, model):
        biofilm_state = {"biofilm_thickness": 15.0}
        sensor_data = {}
        current = model._calculate_biofilm_current_enhancement(
            biofilm_state, sensor_data
        )
        assert current > 0


class TestModelValidationScore:
    def test_no_sensors(self, model):
        score = model._calculate_model_validation_score([], [])
        assert score == 0.5

    def test_with_fusion(self, model):
        biofilm_states = [
            {"biofilm_thickness": 15.0, "biomass_density": 0.5},
        ]
        sensor_data = [
            {"fusion": {"thickness": 15.0, "biomass": 0.5}},
        ]
        score = model._calculate_model_validation_score(
            biofilm_states, sensor_data
        )
        assert score >= 0.0


class TestHandleSensorFaults:
    def test_no_faults(self, model):
        sensor_data = [{"eis": {"status": "good"}, "qcm": {"status": "good"}}]
        model._handle_sensor_faults(sensor_data)
        assert len(model.sensor_fault_history) == 0

    def test_eis_fault(self, model):
        sensor_data = [
            {"eis": {"status": "failed"}, "qcm": {"status": "good"}},
        ]
        model._handle_sensor_faults(sensor_data)
        assert len(model.sensor_fault_history) == 1
        assert model.sensor_fault_history[0]["sensor"] == "eis"

    def test_qcm_fault(self, model):
        sensor_data = [
            {"eis": {"status": "good"}, "qcm": {"status": "failed"}},
        ]
        model._handle_sensor_faults(sensor_data)
        assert len(model.sensor_fault_history) == 1
        assert model.sensor_fault_history[0]["sensor"] == "qcm"


class TestRunSensorIntegratedSimulation:
    def test_short_run(self):
        from sensor_integrated_mfc_model import SensorIntegratedMFCModel

        m = SensorIntegratedMFCModel(
            n_cells=2,
            use_gpu=False,
            simulation_hours=3,
            enable_eis=True,
            enable_qcm=True,
        )
        if not hasattr(m.flow_controller, "actions"):
            m.flow_controller.actions = np.array(
                [-10, -5, -2, -1, 0, 1, 2, 5, 10]
            )
        with patch(
            "sensor_integrated_mfc_model.get_model_path",
            return_value="/tmp/checkpoint.pkl",
        ):
            with patch("builtins.open", mock_open()):
                results = m.run_sensor_integrated_simulation(
                    dt=1.0, save_interval=2
                )
                assert isinstance(results, dict)
                assert "computation_time" in results


class TestGetSensorDiagnostics:
    def test_diagnostics(self, model):
        diag = model.get_sensor_diagnostics()
        assert "sensor_availability" in diag
        assert "sensor_performance" in diag
        assert diag["sensor_availability"]["eis_enabled"] is True
        assert diag["sensor_availability"]["qcm_enabled"] is True
