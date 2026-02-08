"""Tests for qcm_model module - targeting 98%+ coverage."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

sys.modules["gpu_acceleration"] = MagicMock()

from sensing_models.qcm_model import (
    CrystalType,
    ElectrodeType,
    QCMMeasurement,
    SauerbreyModel,
    ViscoelasticModel,
    QCMModel,
)


class TestEnums:
    def test_crystal_types(self):
        assert CrystalType.AT_CUT_5MHZ.value == "AT_cut_5MHz"
        assert CrystalType.AT_CUT_10MHZ.value == "AT_cut_10MHz"
        assert CrystalType.BT_CUT_5MHZ.value == "BT_cut_5MHz"

    def test_electrode_types(self):
        assert ElectrodeType.GOLD.value == "gold"
        assert ElectrodeType.SILVER.value == "silver"
        assert ElectrodeType.ALUMINUM.value == "aluminum"
        assert ElectrodeType.PLATINUM.value == "platinum"


class TestQCMMeasurement:
    def test_calculate_sauerbrey_mass(self):
        m = QCMMeasurement(
            frequency=5e6, frequency_shift=-100.0, dissipation=1e-5,
            quality_factor=50000, timestamp=0.0, temperature=303.0,
        )
        mass = m.calculate_sauerbrey_mass(17.7)
        assert mass == pytest.approx(1770.0)

    def test_calculate_thickness_positive_mass(self):
        m = QCMMeasurement(
            frequency=5e6, frequency_shift=-100.0, dissipation=1e-5,
            quality_factor=50000, timestamp=0.0, temperature=303.0,
            mass_per_area=1000.0,
        )
        thickness = m.calculate_thickness(density=1.1)
        assert thickness > 0

    def test_calculate_thickness_zero_mass(self):
        m = QCMMeasurement(
            frequency=5e6, frequency_shift=-100.0, dissipation=1e-5,
            quality_factor=50000, timestamp=0.0, temperature=303.0,
            mass_per_area=0.0,
        )
        assert m.calculate_thickness(density=1.1) == 0.0


class TestSauerbreyModel:
    def test_init_at_cut_5mhz_gold(self):
        model = SauerbreyModel(CrystalType.AT_CUT_5MHZ, ElectrodeType.GOLD)
        assert model.properties["fundamental_frequency"] == 5e6
        assert model.electrode_props["density"] == 19320
        assert model.practical_sensitivity == 17.7

    def test_init_at_cut_10mhz(self):
        model = SauerbreyModel(CrystalType.AT_CUT_10MHZ, ElectrodeType.GOLD)
        assert model.properties["fundamental_frequency"] == 10e6
        assert model.practical_sensitivity == 4.4

    def test_init_bt_cut_5mhz(self):
        model = SauerbreyModel(CrystalType.BT_CUT_5MHZ, ElectrodeType.GOLD)
        assert model.practical_sensitivity == 17.7

    def test_init_silver_electrode(self):
        model = SauerbreyModel(CrystalType.AT_CUT_5MHZ, ElectrodeType.SILVER)
        assert model.electrode_props["density"] == 10490

    def test_init_aluminum_electrode(self):
        model = SauerbreyModel(CrystalType.AT_CUT_5MHZ, ElectrodeType.ALUMINUM)
        # Falls through to default gold-like properties
        assert model.electrode_props["density"] == 19320

    def test_calculate_mass_from_frequency(self):
        model = SauerbreyModel()
        mass = model.calculate_mass_from_frequency(-100.0)
        assert mass > 0
        mass_zero = model.calculate_mass_from_frequency(100.0)
        assert mass_zero == 0  # Positive shift means no mass

    def test_calculate_frequency_from_mass(self):
        model = SauerbreyModel()
        freq_shift = model.calculate_frequency_from_mass(1000.0)
        assert freq_shift < 0

    def test_estimate_thickness(self):
        model = SauerbreyModel()
        thickness = model.estimate_thickness(1000.0, 1.1)
        assert thickness > 0

    def test_estimate_thickness_zero(self):
        model = SauerbreyModel()
        assert model.estimate_thickness(0.0, 1.1) == 0.0
        assert model.estimate_thickness(100.0, 0.0) == 0.0
        assert model.estimate_thickness(-1.0, 1.1) == 0.0

    def test_get_model_parameters(self):
        model = SauerbreyModel()
        params = model.get_model_parameters()
        assert "crystal_type" in params
        assert "electrode_type" in params
        assert "practical_sensitivity_ng_per_cm2_Hz" in params


class TestViscoelasticModel:
    def test_init(self):
        model = ViscoelasticModel()
        assert model.default_biofilm_props["density"] == 1.1

    def test_calculate_correction_thin_film(self):
        model = ViscoelasticModel()
        freq_corr, dissipation = model.calculate_viscoelastic_correction(
            frequency=5e6, shear_modulus=1e4, viscosity=0.01,
            density=1100, thickness=1e-7,
        )
        assert freq_corr > 0
        assert dissipation >= 0

    def test_calculate_correction_thick_film(self):
        model = ViscoelasticModel()
        freq_corr, dissipation = model.calculate_viscoelastic_correction(
            frequency=5e6, shear_modulus=1e4, viscosity=0.01,
            density=1100, thickness=1e-3,
        )
        assert freq_corr > 0
        assert dissipation >= 0

    def test_correct_sauerbrey_mass(self):
        model = ViscoelasticModel()
        corrected = model.correct_sauerbrey_mass(
            sauerbrey_mass=1000.0, frequency=5e6,
            biofilm_properties={"density": 1.1, "shear_modulus": 1e4, "viscosity": 0.01},
        )
        assert corrected > 0

    def test_correct_sauerbrey_mass_defaults(self):
        model = ViscoelasticModel()
        corrected = model.correct_sauerbrey_mass(
            sauerbrey_mass=1000.0, frequency=5e6, biofilm_properties={},
        )
        assert corrected > 0


class TestQCMModel:
    def test_init_no_gpu(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            model = QCMModel(use_gpu=False)
            assert model.gpu_available is False
            assert model.fundamental_frequency == 5e6

    def test_init_with_gpu(self):
        mock_gpu = MagicMock()
        mock_gpu.is_gpu_available.return_value = True
        with patch("sensing_models.qcm_model.get_gpu_accelerator", return_value=mock_gpu):
            model = QCMModel(use_gpu=True)
            assert model.gpu_available is True

    def test_set_biofilm_species_geobacter(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            model = QCMModel(use_gpu=False)
            model.set_biofilm_species("geobacter")
            assert model.current_biofilm_props["density"] == 1.15

    def test_set_biofilm_species_shewanella(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            model = QCMModel(use_gpu=False)
            model.set_biofilm_species("shewanella")
            assert model.current_biofilm_props["density"] == 1.08

    def test_set_biofilm_species_mixed(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            model = QCMModel(use_gpu=False)
            model.set_biofilm_species("mixed")
            assert model.current_biofilm_props["density"] == 1.12

    def test_simulate_measurement_thin(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            model = QCMModel(use_gpu=False)
            np.random.seed(42)
            m = model.simulate_measurement(
                biofilm_mass=0.01, biofilm_thickness=1.0, time_hours=1.0
            )
            assert isinstance(m, QCMMeasurement)
            assert len(model.measurement_history) == 1

    def test_simulate_measurement_thick(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            model = QCMModel(use_gpu=False)
            np.random.seed(42)
            m = model.simulate_measurement(
                biofilm_mass=1.0, biofilm_thickness=20.0, time_hours=5.0
            )
            assert m.thickness_estimate == 20.0

    def test_simulate_measurement_very_thin(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            model = QCMModel(use_gpu=False)
            np.random.seed(42)
            m = model.simulate_measurement(
                biofilm_mass=0.001, biofilm_thickness=0.5
            )
            assert m is not None

    def test_estimate_biofilm_properties_with_mass(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            model = QCMModel(use_gpu=False)
            np.random.seed(42)
            m = model.simulate_measurement(biofilm_mass=1.0, biofilm_thickness=20.0)
            props = model.estimate_biofilm_properties(m)
            assert "mass_per_area_ng_per_cm2" in props
            assert "thickness_um" in props
            assert "measurement_quality" in props

    def test_estimate_biofilm_properties_no_mass(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            model = QCMModel(use_gpu=False)
            m = QCMMeasurement(
                frequency=5e6, frequency_shift=-50.0, dissipation=2e-5,
                quality_factor=50000, timestamp=0.0, temperature=303.0,
                mass_per_area=0.0,
            )
            props = model.estimate_biofilm_properties(m)
            assert "mass_per_area_ng_per_cm2" in props

    def test_calibrate_for_biofilm_few_points(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            model = QCMModel(use_gpu=False)
            old_density = model.current_biofilm_props["density"]
            model.calibrate_for_biofilm([(1.0, 1.0, QCMMeasurement(
                5e6, -10.0, 1e-5, 50000, 0.0, 303.0))])
            assert model.current_biofilm_props["density"] == old_density

    def test_calibrate_for_biofilm(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            model = QCMModel(use_gpu=False)
            np.random.seed(42)
            ref_data = []
            for mass, thick in [(0.1, 5.0), (0.5, 15.0), (1.0, 25.0), (2.0, 40.0)]:
                m = model.simulate_measurement(biofilm_mass=mass, biofilm_thickness=thick)
                ref_data.append((mass, thick, m))
            model.calibrate_for_biofilm(ref_data)

    def test_reset_baseline_no_history(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            model = QCMModel(use_gpu=False)
            model.reset_baseline()
            assert model.baseline_frequency == model.fundamental_frequency

    def test_reset_baseline_with_history(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            model = QCMModel(use_gpu=False)
            np.random.seed(42)
            m = model.simulate_measurement(biofilm_mass=1.0, biofilm_thickness=10.0)
            model.reset_baseline()
            assert model.baseline_frequency == m.frequency

    def test_get_measurement_summary(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            model = QCMModel(use_gpu=False)
            summary = model.get_measurement_summary()
            assert "crystal_type" in summary
            assert "gpu_available" in summary

    def test_get_frequency_stability_insufficient_data(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            model = QCMModel(use_gpu=False)
            result = model.get_frequency_stability_metrics()
            assert result.get("insufficient_data") is True

    def test_get_frequency_stability(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            model = QCMModel(use_gpu=False)
            np.random.seed(42)
            for i in range(15):
                model.simulate_measurement(
                    biofilm_mass=0.5, biofilm_thickness=10.0, time_hours=i * 0.1
                )
            result = model.get_frequency_stability_metrics(window_hours=10.0)
            assert "mean_frequency_Hz" in result

    def test_get_frequency_stability_insufficient_recent(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            model = QCMModel(use_gpu=False)
            np.random.seed(42)
            for i in range(12):
                model.simulate_measurement(
                    biofilm_mass=0.5, biofilm_thickness=10.0, time_hours=i * 0.1
                )
            result = model.get_frequency_stability_metrics(window_hours=0.001)
            assert "insufficient_recent_data" in result or "mean_frequency_Hz" in result

    def test_init_various_crystal_electrode(self):
        with patch("sensing_models.qcm_model.get_gpu_accelerator", None):
            m1 = QCMModel(crystal_type=CrystalType.AT_CUT_10MHZ, use_gpu=False)
            assert m1.fundamental_frequency == 10e6
            m2 = QCMModel(
                crystal_type=CrystalType.BT_CUT_5MHZ,
                electrode_type=ElectrodeType.SILVER,
                use_gpu=False,
            )
            assert m2 is not None
