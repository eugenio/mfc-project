"""Coverage tests for membrane_fouling.py targeting 98%+ statement coverage."""
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
if "jax" not in sys.modules:
    _jax = MagicMock()
    _jax.numpy = MagicMock()
    for _a in dir(np):
        if not _a.startswith("_"):
            setattr(_jax.numpy, _a, getattr(np, _a))
    sys.modules["jax"] = _jax
    sys.modules["jax.numpy"] = _jax.numpy

from membrane_models.membrane_fouling import (
    FoulingModel,
    FoulingParameters,
    FoulingType,
    calculate_fouling_resistance,
)


class TestFoulingType(unittest.TestCase):
    def test_enum_values(self):
        self.assertEqual(FoulingType.BIOLOGICAL.value, "biological")
        self.assertEqual(FoulingType.CHEMICAL.value, "chemical")
        self.assertEqual(FoulingType.PHYSICAL.value, "physical")
        self.assertEqual(FoulingType.DEGRADATION.value, "degradation")


class TestFoulingParameters(unittest.TestCase):
    def test_defaults(self):
        p = FoulingParameters()
        self.assertAlmostEqual(p.biofilm_growth_rate, 0.1)
        self.assertAlmostEqual(p.biofilm_detachment_rate, 0.01)
        self.assertAlmostEqual(p.biofilm_thickness_max, 100e-6)
        self.assertAlmostEqual(p.precipitation_rate, 1e-8)
        self.assertAlmostEqual(p.particle_deposition_rate, 1e-10)
        self.assertAlmostEqual(p.thermal_degradation_rate, 1e-6)
        self.assertAlmostEqual(p.chemical_degradation_rate, 1e-7)
        self.assertAlmostEqual(p.temperature, 298.15)
        self.assertAlmostEqual(p.ph, 7.0)


class TestFoulingModelInit(unittest.TestCase):
    def test_init(self):
        fm = FoulingModel(FoulingParameters())
        self.assertAlmostEqual(fm.biofilm_thickness, 0.0)
        self.assertAlmostEqual(fm.chemical_layer_thickness, 0.0)
        self.assertAlmostEqual(fm.particle_layer_thickness, 0.0)
        self.assertAlmostEqual(fm.degradation_fraction, 0.0)
        self.assertEqual(fm.fouling_history, [])
        self.assertEqual(fm.resistance_history, [])
        self.assertAlmostEqual(fm.operating_time, 0.0)
        self.assertEqual(fm.biofilm_density, 1200)
        self.assertAlmostEqual(fm.cake_porosity, 0.4)


class TestBiofilmGrowth(unittest.TestCase):
    def test_initial_nucleation(self):
        fm = FoulingModel(FoulingParameters())
        change = fm.calculate_biofilm_growth(dt_hours=1.0, nutrient_conc=0.01)
        self.assertGreater(change, 0.0)
        self.assertGreater(fm.biofilm_thickness, 0.0)

    def test_established_biofilm(self):
        fm = FoulingModel(FoulingParameters())
        fm.biofilm_thickness = 10e-6
        change = fm.calculate_biofilm_growth(dt_hours=1.0, nutrient_conc=0.01)
        self.assertTrue(float(change) == float(change))  # numeric check

    def test_positive_current(self):
        fm = FoulingModel(FoulingParameters())
        fm.biofilm_thickness = 10e-6
        change = fm.calculate_biofilm_growth(
            dt_hours=1.0, nutrient_conc=0.01, current_density=1000.0
        )
        self.assertTrue(float(change) == float(change))  # numeric check

    def test_negative_current(self):
        fm = FoulingModel(FoulingParameters())
        fm.biofilm_thickness = 10e-6
        change = fm.calculate_biofilm_growth(
            dt_hours=1.0, nutrient_conc=0.01, current_density=-1000.0
        )
        self.assertTrue(float(change) == float(change))  # numeric check

    def test_max_thickness_capping(self):
        fm = FoulingModel(FoulingParameters(biofilm_thickness_max=5e-6))
        fm.biofilm_thickness = 5e-6
        fm.calculate_biofilm_growth(dt_hours=10.0, nutrient_conc=0.1)
        self.assertLessEqual(fm.biofilm_thickness, 5e-6)


class TestChemicalFouling(unittest.TestCase):
    def test_supersaturated(self):
        fm = FoulingModel(FoulingParameters())
        ions = {"Ca2+": 0.01, "CO3--": 0.01}
        change = fm.calculate_chemical_fouling(
            dt_hours=1.0, ion_concentrations=ions, temperature=298.15
        )
        self.assertGreater(change, 0.0)

    def test_undersaturated(self):
        fm = FoulingModel(FoulingParameters())
        ions = {"Ca2+": 1e-6, "CO3--": 1e-6}
        change = fm.calculate_chemical_fouling(
            dt_hours=1.0, ion_concentrations=ions, temperature=298.15
        )
        self.assertAlmostEqual(change, 0.0)

    def test_default_concentrations(self):
        fm = FoulingModel(FoulingParameters())
        change = fm.calculate_chemical_fouling(
            dt_hours=1.0, ion_concentrations={}, temperature=298.15
        )
        self.assertTrue(float(change) == float(change))  # numeric check


class TestParticleFouling(unittest.TestCase):
    def test_basic(self):
        fm = FoulingModel(FoulingParameters())
        change = fm.calculate_particle_fouling(
            dt_hours=1.0, particle_concentration=0.01, flow_velocity=0.1
        )
        self.assertGreater(change, 0.0)
        self.assertGreater(fm.particle_layer_thickness, 0.0)

    def test_zero_velocity(self):
        fm = FoulingModel(FoulingParameters())
        change = fm.calculate_particle_fouling(
            dt_hours=1.0, particle_concentration=0.01, flow_velocity=0.0
        )
        self.assertIsInstance(change, float)


class TestThermalDegradation(unittest.TestCase):
    def test_basic(self):
        fm = FoulingModel(FoulingParameters())
        deg = fm.calculate_thermal_degradation(dt_hours=100.0, temperature=350.0)
        self.assertGreater(deg, 0.0)

    def test_reference_temp(self):
        fm = FoulingModel(FoulingParameters())
        deg = fm.calculate_thermal_degradation(dt_hours=1.0, temperature=298.15)
        self.assertGreater(deg, 0.0)


class TestChemicalDegradation(unittest.TestCase):
    def test_neutral_ph(self):
        fm = FoulingModel(FoulingParameters())
        deg = fm.calculate_chemical_degradation(dt_hours=1.0, ph=7.0)
        self.assertGreater(deg, 0.0)

    def test_extreme_ph(self):
        fm = FoulingModel(FoulingParameters())
        deg_neutral = fm.calculate_chemical_degradation(dt_hours=1.0, ph=7.0)
        deg_extreme = fm.calculate_chemical_degradation(dt_hours=1.0, ph=14.0)
        self.assertGreater(deg_extreme, deg_neutral)

    def test_with_oxidizer(self):
        fm = FoulingModel(FoulingParameters())
        deg_no = fm.calculate_chemical_degradation(dt_hours=1.0, ph=7.0, oxidizing_species=0.0)
        deg_ox = fm.calculate_chemical_degradation(dt_hours=1.0, ph=7.0, oxidizing_species=0.1)
        self.assertGreater(deg_ox, deg_no)


class TestTotalResistance(unittest.TestCase):
    def test_clean_membrane(self):
        fm = FoulingModel(FoulingParameters())
        res = fm.calculate_total_resistance(0.1)
        self.assertAlmostEqual(res["clean_membrane_resistance"], 0.1)
        self.assertAlmostEqual(res["biofilm_resistance"], 0.0)
        self.assertAlmostEqual(res["chemical_fouling_resistance"], 0.0)
        self.assertAlmostEqual(res["particle_fouling_resistance"], 0.0)

    def test_fouled_membrane(self):
        fm = FoulingModel(FoulingParameters())
        fm.biofilm_thickness = 10e-6
        fm.chemical_layer_thickness = 5e-6
        fm.particle_layer_thickness = 3e-6
        fm.degradation_fraction = 0.1
        res = fm.calculate_total_resistance(0.1)
        self.assertGreater(res["biofilm_resistance"], 0.0)
        self.assertGreater(res["chemical_fouling_resistance"], 0.0)
        self.assertGreater(res["particle_fouling_resistance"], 0.0)
        self.assertGreater(res["total_resistance"], 0.1)
        self.assertGreater(res["fouling_resistance_fraction"], 0.0)


class TestUpdateFouling(unittest.TestCase):
    def test_basic_update(self):
        fm = FoulingModel(FoulingParameters())
        fm.update_fouling(
            dt_hours=1.0,
            operating_conditions={
                "temperature": 310.0,
                "ph": 7.5,
                "nutrient_concentration": 0.005,
                "current_density": 500.0,
                "ion_concentrations": {"Ca2+": 0.005, "CO3--": 0.005},
                "particle_concentration": 0.01,
                "flow_velocity": 0.05,
                "oxidizing_species": 0.001,
            },
        )
        self.assertAlmostEqual(fm.operating_time, 1.0)
        self.assertEqual(len(fm.fouling_history), 1)

    def test_defaults_used(self):
        fm = FoulingModel(FoulingParameters())
        fm.update_fouling(dt_hours=1.0, operating_conditions={})
        self.assertAlmostEqual(fm.operating_time, 1.0)

    def test_degradation_capped(self):
        fm = FoulingModel(FoulingParameters())
        fm.degradation_fraction = 0.89
        fm.update_fouling(
            dt_hours=1000.0,
            operating_conditions={"temperature": 400.0, "ph": 14.0, "oxidizing_species": 1.0},
        )
        self.assertLessEqual(fm.degradation_fraction, 0.9)


class TestPredictFoulingTrajectory(unittest.TestCase):
    def test_basic_prediction(self):
        fm = FoulingModel(FoulingParameters())
        result = fm.predict_fouling_trajectory(
            simulation_hours=10.0,
            operating_conditions={},
            time_step=1.0,
        )
        self.assertIn("time_hours", result)
        self.assertIn("biofilm_thickness_um", result)
        self.assertIn("chemical_layer_thickness_um", result)
        self.assertIn("particle_layer_thickness_um", result)
        self.assertIn("degradation_fraction", result)
        self.assertIn("total_resistance_ohm_m2", result)
        self.assertIn("final_fouling_thickness_um", result)
        self.assertIn("resistance_increase_factor", result)
        # State should be restored
        self.assertAlmostEqual(fm.biofilm_thickness, 0.0)
        self.assertAlmostEqual(fm.operating_time, 0.0)


class TestCleaningEffectiveness(unittest.TestCase):
    def test_chemical_cleaning(self):
        fm = FoulingModel(FoulingParameters())
        fm.biofilm_thickness = 50e-6
        fm.chemical_layer_thickness = 20e-6
        fm.particle_layer_thickness = 10e-6
        result = fm.get_cleaning_effectiveness("chemical_cleaning")
        self.assertIn("cleaning_method", result)
        self.assertIn("resistance_reduction_percent", result)
        self.assertGreater(result["resistance_reduction_percent"], 0.0)

    def test_backwash(self):
        fm = FoulingModel(FoulingParameters())
        fm.biofilm_thickness = 50e-6
        result = fm.get_cleaning_effectiveness("backwash")
        self.assertEqual(result["cleaning_method"], "backwash")

    def test_ultrasonic(self):
        fm = FoulingModel(FoulingParameters())
        fm.particle_layer_thickness = 30e-6
        result = fm.get_cleaning_effectiveness("ultrasonic_cleaning")
        self.assertEqual(result["cleaning_method"], "ultrasonic_cleaning")

    def test_electrochemical(self):
        fm = FoulingModel(FoulingParameters())
        fm.biofilm_thickness = 50e-6
        result = fm.get_cleaning_effectiveness("electrochemical_cleaning")
        self.assertEqual(result["cleaning_method"], "electrochemical_cleaning")

    def test_unknown_method(self):
        fm = FoulingModel(FoulingParameters())
        result = fm.get_cleaning_effectiveness("unknown_method")
        self.assertIn("error", result)


class TestFoulingStatus(unittest.TestCase):
    def test_low_severity(self):
        fm = FoulingModel(FoulingParameters())
        fm.biofilm_thickness = 1e-6
        status = fm.get_fouling_status()
        self.assertEqual(status["fouling_severity"], "Low")
        self.assertFalse(status["cleaning_recommended"])

    def test_moderate_severity(self):
        fm = FoulingModel(FoulingParameters())
        fm.biofilm_thickness = 20e-6
        status = fm.get_fouling_status()
        self.assertEqual(status["fouling_severity"], "Moderate")

    def test_high_severity(self):
        fm = FoulingModel(FoulingParameters())
        fm.biofilm_thickness = 60e-6
        status = fm.get_fouling_status()
        self.assertEqual(status["fouling_severity"], "High")
        self.assertTrue(status["cleaning_recommended"])

    def test_severe(self):
        fm = FoulingModel(FoulingParameters())
        fm.biofilm_thickness = 110e-6
        status = fm.get_fouling_status()
        self.assertEqual(status["fouling_severity"], "Severe")

    def test_dominant_type(self):
        fm = FoulingModel(FoulingParameters())
        fm.biofilm_thickness = 50e-6
        fm.chemical_layer_thickness = 10e-6
        fm.particle_layer_thickness = 5e-6
        status = fm.get_fouling_status()
        self.assertEqual(status["dominant_fouling_type"], "biofilm")


class TestCalculateFoulingResistance(unittest.TestCase):
    def test_basic(self):
        r = calculate_fouling_resistance(10e-6, 1e-6)
        self.assertAlmostEqual(r, 10.0)

    def test_zero_conductivity(self):
        r = calculate_fouling_resistance(10e-6, 0.0)
        self.assertEqual(r, float("inf"))

    def test_negative_conductivity(self):
        r = calculate_fouling_resistance(10e-6, -1.0)
        self.assertEqual(r, float("inf"))


if __name__ == "__main__":
    unittest.main()
