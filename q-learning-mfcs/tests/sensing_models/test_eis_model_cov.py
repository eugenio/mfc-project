"""Tests for eis_model module - targeting 98%+ coverage."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Mock gpu_acceleration before importing
sys.modules["gpu_acceleration"] = MagicMock()

from sensing_models.eis_model import (
    BacterialSpecies,
    EISMeasurement,
    EISCircuitModel,
    EISModel,
)


class TestBacterialSpecies:
    def test_enum_values(self):
        assert BacterialSpecies.GEOBACTER.value == "geobacter_sulfurreducens"
        assert BacterialSpecies.SHEWANELLA.value == "shewanella_oneidensis"
        assert BacterialSpecies.MIXED.value == "mixed_culture"


class TestEISMeasurement:
    def test_complex_impedance(self):
        m = EISMeasurement(
            frequency=1000.0,
            impedance_magnitude=100.0,
            impedance_phase=0.5,
            real_impedance=80.0,
            imaginary_impedance=-60.0,
            timestamp=0.0,
            temperature=303.0,
        )
        z = m.complex_impedance
        assert z.real == 80.0
        assert z.imag == -60.0

    def test_from_complex(self):
        z = complex(100, -50)
        m = EISMeasurement.from_complex(frequency=1000.0, impedance=z)
        assert m.frequency == 1000.0
        assert m.real_impedance == 100.0
        assert m.imaginary_impedance == -50.0
        assert m.timestamp == 0.0
        assert m.temperature == 303.0
        assert m.impedance_magnitude == abs(z)

    def test_from_complex_custom_params(self):
        z = complex(50, -30)
        m = EISMeasurement.from_complex(
            frequency=500.0, impedance=z, timestamp=1.5, temperature=310.0
        )
        assert m.timestamp == 1.5
        assert m.temperature == 310.0


class TestEISCircuitModel:
    def test_init_geobacter(self):
        model = EISCircuitModel(BacterialSpecies.GEOBACTER)
        assert model.species == BacterialSpecies.GEOBACTER
        assert model.species_params["base_resistivity"] == 100.0

    def test_init_shewanella(self):
        model = EISCircuitModel(BacterialSpecies.SHEWANELLA)
        assert model.species_params["base_resistivity"] == 500.0

    def test_init_mixed(self):
        model = EISCircuitModel(BacterialSpecies.MIXED)
        assert model.species_params["base_resistivity"] == 250.0

    def test_reset_parameters(self):
        model = EISCircuitModel()
        model.Rs = 999
        model.reset_parameters()
        assert model.Rs == 50.0

    def test_update_from_biofilm_state(self):
        model = EISCircuitModel()
        model.update_from_biofilm_state(
            thickness=20.0, biomass_density=5.0, porosity=0.7
        )
        assert model.Rbio >= 10.0
        assert model.Cbio >= 1e-9
        assert model.Rct > 0

    def test_update_from_biofilm_state_thin(self):
        model = EISCircuitModel()
        model.update_from_biofilm_state(
            thickness=0.1, biomass_density=0.1, porosity=0.99
        )
        assert model.Rbio >= 10.0
        assert model.Cbio >= 1e-9

    def test_calculate_impedance(self):
        model = EISCircuitModel()
        z = model.calculate_impedance(1000.0)
        assert isinstance(z, complex)
        assert z.real > 0

    def test_fit_parameters(self):
        model = EISCircuitModel()
        measurements = []
        for freq in [100, 1000, 10000, 100000]:
            z = model.calculate_impedance(freq)
            m = EISMeasurement.from_complex(frequency=freq, impedance=z)
            measurements.append(m)
        params = model.fit_parameters(measurements)
        assert "Rs" in params
        assert "Cdl" in params
        assert "Rbio" in params
        assert "Rct" in params


class TestEISModel:
    def test_init_mixed(self):
        with patch("sensing_models.eis_model.get_gpu_accelerator", None):
            model = EISModel(use_gpu=False)
            assert model.species == BacterialSpecies.MIXED
            assert model.gpu_available is False

    def test_init_geobacter(self):
        with patch("sensing_models.eis_model.get_gpu_accelerator", None):
            model = EISModel(species=BacterialSpecies.GEOBACTER, use_gpu=False)
            assert model.calibration["max_thickness"] == 80.0

    def test_init_shewanella(self):
        with patch("sensing_models.eis_model.get_gpu_accelerator", None):
            model = EISModel(species=BacterialSpecies.SHEWANELLA, use_gpu=False)
            assert model.calibration["max_thickness"] == 60.0

    def test_init_with_gpu(self):
        mock_gpu = MagicMock()
        mock_gpu.is_gpu_available.return_value = True
        with patch(
            "sensing_models.eis_model.get_gpu_accelerator", return_value=mock_gpu
        ):
            model = EISModel(use_gpu=True)
            assert model.gpu_available is True

    def test_simulate_measurement_cpu(self):
        with patch("sensing_models.eis_model.get_gpu_accelerator", None):
            model = EISModel(use_gpu=False)
            np.random.seed(42)
            measurements = model.simulate_measurement(
                biofilm_thickness=10.0, biomass_density=2.0, time_hours=1.0
            )
            assert len(measurements) == 50
            assert model.current_thickness == 10.0
            assert model.current_biomass == 2.0
            assert len(model.measurement_history) == 1

    def test_simulate_measurement_gpu(self):
        mock_gpu = MagicMock()
        mock_gpu.is_gpu_available.return_value = True
        mock_gpu.array.return_value = np.logspace(2, 6, 50)
        mock_gpu.multiply.return_value = None
        mock_gpu.to_cpu.return_value = np.logspace(2, 6, 50)
        with patch(
            "sensing_models.eis_model.get_gpu_accelerator", return_value=mock_gpu
        ):
            model = EISModel(use_gpu=True)
            np.random.seed(42)
            measurements = model.simulate_measurement(
                biofilm_thickness=10.0, biomass_density=2.0
            )
            assert len(measurements) == 50

    def test_estimate_thickness_empty(self):
        with patch("sensing_models.eis_model.get_gpu_accelerator", None):
            model = EISModel(use_gpu=False)
            assert model.estimate_thickness([]) == 0.0

    def test_estimate_thickness_low_frequency(self):
        with patch("sensing_models.eis_model.get_gpu_accelerator", None):
            model = EISModel(use_gpu=False)
            np.random.seed(42)
            measurements = model.simulate_measurement(
                biofilm_thickness=20.0, biomass_density=3.0
            )
            thickness = model.estimate_thickness(measurements, method="low_frequency")
            assert thickness >= 0.0

    def test_estimate_thickness_low_frequency_no_low_freq(self):
        with patch("sensing_models.eis_model.get_gpu_accelerator", None):
            model = EISModel(use_gpu=False)
            # Create measurements with only high frequencies
            measurements = []
            for freq in [10000, 100000, 1000000]:
                z = model.circuit.calculate_impedance(freq)
                m = EISMeasurement.from_complex(frequency=freq, impedance=z)
                measurements.append(m)
            thickness = model.estimate_thickness(measurements, method="low_frequency")
            assert thickness >= 0.0

    def test_estimate_thickness_characteristic(self):
        with patch("sensing_models.eis_model.get_gpu_accelerator", None):
            model = EISModel(use_gpu=False)
            np.random.seed(42)
            measurements = model.simulate_measurement(
                biofilm_thickness=15.0, biomass_density=2.0
            )
            thickness = model.estimate_thickness(
                measurements, method="characteristic"
            )
            assert thickness >= 0.0

    def test_estimate_thickness_fitting(self):
        with patch("sensing_models.eis_model.get_gpu_accelerator", None):
            model = EISModel(use_gpu=False)
            np.random.seed(42)
            measurements = model.simulate_measurement(
                biofilm_thickness=10.0, biomass_density=2.0
            )
            thickness = model.estimate_thickness(measurements, method="fitting")
            assert thickness >= 0.0

    def test_estimate_thickness_unknown_method(self):
        with patch("sensing_models.eis_model.get_gpu_accelerator", None):
            model = EISModel(use_gpu=False)
            np.random.seed(42)
            measurements = model.simulate_measurement(
                biofilm_thickness=10.0, biomass_density=2.0
            )
            with pytest.raises(ValueError, match="Unknown estimation method"):
                model.estimate_thickness(measurements, method="unknown")

    def test_get_biofilm_properties_empty(self):
        with patch("sensing_models.eis_model.get_gpu_accelerator", None):
            model = EISModel(use_gpu=False)
            assert model.get_biofilm_properties([]) == {}

    def test_get_biofilm_properties(self):
        with patch("sensing_models.eis_model.get_gpu_accelerator", None):
            model = EISModel(use_gpu=False)
            np.random.seed(42)
            measurements = model.simulate_measurement(
                biofilm_thickness=10.0, biomass_density=2.0
            )
            props = model.get_biofilm_properties(measurements)
            assert "thickness_um" in props
            assert "conductivity_S_per_m" in props
            assert "measurement_quality" in props

    def test_assess_measurement_quality_few(self):
        with patch("sensing_models.eis_model.get_gpu_accelerator", None):
            model = EISModel(use_gpu=False)
            measurements = [
                EISMeasurement.from_complex(100.0, complex(100, -50))
            ] * 5
            quality = model._assess_measurement_quality(measurements)
            assert quality == 0.5

    def test_assess_measurement_quality_many(self):
        with patch("sensing_models.eis_model.get_gpu_accelerator", None):
            model = EISModel(use_gpu=False)
            np.random.seed(42)
            measurements = model.simulate_measurement(
                biofilm_thickness=10.0, biomass_density=2.0
            )
            quality = model._assess_measurement_quality(measurements)
            assert 0.0 <= quality <= 1.0

    def test_calibrate_for_species_few_points(self):
        with patch("sensing_models.eis_model.get_gpu_accelerator", None):
            model = EISModel(use_gpu=False)
            old_slope = model.calibration["thickness_slope"]
            model.calibrate_for_species([(10.0, [])])
            assert model.calibration["thickness_slope"] == old_slope

    def test_calibrate_for_species(self):
        with patch("sensing_models.eis_model.get_gpu_accelerator", None):
            model = EISModel(use_gpu=False)
            np.random.seed(42)
            ref_data = []
            for thickness in [5.0, 15.0, 25.0, 35.0]:
                measurements = model.simulate_measurement(
                    biofilm_thickness=thickness, biomass_density=2.0
                )
                ref_data.append((thickness, measurements))
            model.calibrate_for_species(ref_data)
            # Calibration should have been updated
            assert model.calibration["thickness_slope"] != 0

    def test_get_measurement_summary(self):
        with patch("sensing_models.eis_model.get_gpu_accelerator", None):
            model = EISModel(use_gpu=False)
            summary = model.get_measurement_summary()
            assert "species" in summary
            assert "electrode_area_m2" in summary
            assert "gpu_available" in summary
            assert "circuit_parameters" in summary
