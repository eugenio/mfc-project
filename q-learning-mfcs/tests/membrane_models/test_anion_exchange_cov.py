"""Coverage tests for anion_exchange.py targeting 98%+ statement coverage."""
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
if "jax" not in sys.modules:
    mock_jax = MagicMock()
    mock_jax.numpy = MagicMock()
    for attr in dir(np):
        if not attr.startswith("_"):
            setattr(mock_jax.numpy, attr, getattr(np, attr))
    sys.modules["jax"] = mock_jax
    sys.modules["jax.numpy"] = mock_jax.numpy

from membrane_models.anion_exchange import (
    AEMParameters,
    AnionExchangeMembrane,
    create_aem_membrane,
)
from membrane_models.base_membrane import IonType


class TestAEMParameters(unittest.TestCase):
    def test_defaults(self):
        p = AEMParameters()
        self.assertEqual(p.membrane_type, "Quaternary Ammonium")
        self.assertEqual(p.polymer_backbone, "PPO")
        self.assertAlmostEqual(p.hydroxide_conductivity_ref, 0.04)
        self.assertAlmostEqual(p.conductivity_activation, 20000)
        self.assertAlmostEqual(p.hydroxide_selectivity, 0.95)
        self.assertAlmostEqual(p.bicarbonate_permeability, 0.8)
        self.assertAlmostEqual(p.water_uptake_max, 15.0)
        self.assertAlmostEqual(p.electro_osmotic_drag, 4.0)
        self.assertAlmostEqual(p.alkaline_stability_factor, 0.8)
        self.assertAlmostEqual(p.degradation_activation, 50000)
        self.assertAlmostEqual(p.max_operating_ph, 14.0)
        self.assertAlmostEqual(p.co2_absorption_rate, 1e-8)
        self.assertAlmostEqual(p.carbonate_conductivity_ratio, 0.3)
        self.assertAlmostEqual(p.material_cost_per_m2, 400.0)


class TestAnionExchangeMembraneInit(unittest.TestCase):
    def test_init(self):
        params = AEMParameters()
        aem = AnionExchangeMembrane(params)
        self.assertIn(IonType.HYDROXIDE, aem.ion_transport)
        self.assertIn(IonType.BICARBONATE, aem.ion_transport)
        self.assertIn(IonType.CARBONATE, aem.ion_transport)
        self.assertIn(IonType.CHLORIDE, aem.ion_transport)
        self.assertAlmostEqual(aem.carbonate_fraction, 0.0)
        self.assertAlmostEqual(aem.cumulative_degradation, 0.0)
        self.assertEqual(aem.ph_gradient_history, [])


class TestAEMWaterContent(unittest.TestCase):
    def test_vapor_equilibrated(self):
        aem = create_aem_membrane()
        wc = aem.calculate_water_content(0.5)
        self.assertGreater(wc, 0.0)
        self.assertLessEqual(wc, 15.0)

    def test_liquid_equilibrated(self):
        aem = create_aem_membrane()
        wc = aem.calculate_water_content(1.5)
        self.assertAlmostEqual(wc, 15.0)

    def test_full_activity(self):
        aem = create_aem_membrane()
        wc = aem.calculate_water_content(1.0)
        self.assertAlmostEqual(wc, 15.0)

    def test_zero_activity(self):
        aem = create_aem_membrane()
        wc = aem.calculate_water_content(0.0)
        self.assertAlmostEqual(wc, 0.0)


class TestAEMIonicConductivity(unittest.TestCase):
    def test_defaults(self):
        aem = create_aem_membrane()
        cond = aem.calculate_ionic_conductivity()
        self.assertGreater(cond, 0.0)

    def test_custom_temp_water(self):
        aem = create_aem_membrane()
        cond = aem.calculate_ionic_conductivity(temperature=333.0, water_content=10.0)
        self.assertGreater(cond, 0.0)

    def test_carbonation_reduces_conductivity(self):
        aem = create_aem_membrane()
        cond_fresh = aem.calculate_ionic_conductivity()
        aem.carbonate_fraction = 0.5
        cond_carbonated = aem.calculate_ionic_conductivity()
        self.assertLess(cond_carbonated, cond_fresh)

    def test_degradation_reduces_conductivity(self):
        aem = create_aem_membrane()
        cond_fresh = aem.calculate_ionic_conductivity()
        aem.cumulative_degradation = 0.3
        cond_degraded = aem.calculate_ionic_conductivity()
        self.assertLess(cond_degraded, cond_fresh)


class TestUpdateCarbonation(unittest.TestCase):
    def test_no_exposure(self):
        aem = create_aem_membrane()
        initial_cf = aem.carbonate_fraction
        aem.update_carbonation(exposure_time=0.0)
        self.assertAlmostEqual(aem.carbonate_fraction, initial_cf)

    def test_exposure(self):
        aem = create_aem_membrane()
        aem.update_carbonation(co2_partial_pressure=40.0, exposure_time=100.0)
        self.assertGreater(aem.carbonate_fraction, 0.0)

    def test_capping_at_one(self):
        aem = create_aem_membrane()
        aem.update_carbonation(co2_partial_pressure=1e6, exposure_time=1e6)
        self.assertLessEqual(aem.carbonate_fraction, 1.0)


class TestPHGradientEffect(unittest.TestCase):
    def test_basic(self):
        aem = create_aem_membrane()
        result = aem.calculate_ph_gradient_effect(ph_anode=7.0, ph_cathode=13.0)
        self.assertIn("ph_gradient", result)
        self.assertIn("OH_concentration_gradient", result)
        self.assertIn("diffusion_potential_V", result)
        self.assertIn("ph_stress_factor", result)
        self.assertAlmostEqual(result["ph_gradient"], 6.0)

    def test_zero_gradient(self):
        aem = create_aem_membrane()
        result = aem.calculate_ph_gradient_effect(ph_anode=13.0, ph_cathode=13.0)
        self.assertAlmostEqual(result["ph_gradient"], 0.0)
        self.assertAlmostEqual(result["diffusion_potential_V"], 0.0, places=3)

    def test_zero_oh_one_side(self):
        aem = create_aem_membrane()
        # pH = 0 means pOH = 14, C_OH = 1e-14*1000 = 1e-11
        # Still positive so diffusion_potential should be computed
        result = aem.calculate_ph_gradient_effect(ph_anode=0.0, ph_cathode=14.0)
        self.assertIn("diffusion_potential_V", result)

    def test_extremely_negative_ph_gives_zero_oh(self):
        """Cover line 291: C_OH = 0 when pH is extremely negative (underflow)."""
        aem = create_aem_membrane()
        # pH = -1000 -> pOH = 1014 -> C_OH = 10^(-1014)*1000 = 0.0
        result = aem.calculate_ph_gradient_effect(ph_anode=-1000.0, ph_cathode=14.0)
        self.assertAlmostEqual(result["diffusion_potential_V"], 0.0)


class TestAEMDegradationRate(unittest.TestCase):
    def test_basic(self):
        aem = create_aem_membrane()
        rate = aem.calculate_degradation_rate(
            temperature=353.15, ph=14.0, current_density=5000.0
        )
        self.assertGreater(rate, 0.0)

    def test_mild_vs_harsh(self):
        aem = create_aem_membrane()
        rate_mild = aem.calculate_degradation_rate(
            temperature=303.15, ph=10.0, current_density=1000.0
        )
        rate_harsh = aem.calculate_degradation_rate(
            temperature=363.15, ph=14.0, current_density=10000.0
        )
        self.assertGreater(rate_harsh, rate_mild)


class TestCO2Mitigation(unittest.TestCase):
    def test_no_scrubber_no_pulse(self):
        aem = create_aem_membrane()
        result = aem.simulate_co2_mitigation({"co2_ppm": 400, "ph_cathode": 13.0})
        self.assertAlmostEqual(result["effective_co2_ppm"], 400.0)
        self.assertAlmostEqual(result["pulse_mitigation_factor"], 0.0)

    def test_with_scrubber(self):
        aem = create_aem_membrane()
        result = aem.simulate_co2_mitigation(
            {"co2_ppm": 400, "use_co2_scrubber": True}
        )
        self.assertAlmostEqual(result["effective_co2_ppm"], 20.0)

    def test_low_ph(self):
        aem = create_aem_membrane()
        result = aem.simulate_co2_mitigation({"ph_cathode": 10.0})
        self.assertAlmostEqual(result["carbonation_rate_factor"], 1.5)

    def test_with_pulse(self):
        aem = create_aem_membrane()
        result = aem.simulate_co2_mitigation({"pulse_frequency_hz": 10.0})
        self.assertGreater(result["pulse_mitigation_factor"], 0.0)

    def test_performance_retention(self):
        aem = create_aem_membrane()
        result = aem.simulate_co2_mitigation({})
        self.assertGreater(result["performance_retention_percent"], 0.0)
        self.assertLessEqual(result["performance_retention_percent"], 100.0)


class TestWaterBalance(unittest.TestCase):
    def test_basic(self):
        aem = create_aem_membrane()
        result = aem.calculate_water_balance(
            current_density=1000.0, rh_anode=80.0, rh_cathode=100.0
        )
        self.assertIn("electro_osmotic_flux_mol_m2_s", result)
        self.assertIn("diffusion_flux_mol_m2_s", result)
        self.assertIn("net_water_flux_mol_m2_s", result)
        self.assertIn("orr_water_consumption_mol_m2_s", result)
        self.assertIn("effective_drag_coefficient", result)
        self.assertIn("water_content_anode", result)
        self.assertIn("water_content_cathode", result)

    def test_carbonate_effect(self):
        aem = create_aem_membrane()
        aem.carbonate_fraction = 0.5
        result = aem.calculate_water_balance(
            current_density=1000.0, rh_anode=80.0, rh_cathode=100.0
        )
        self.assertGreater(result["effective_drag_coefficient"], 0.0)


class TestStabilityAssessment(unittest.TestCase):
    def test_basic(self):
        aem = create_aem_membrane()
        result = aem.get_stability_assessment(
            operating_hours=1000.0,
            average_temperature=333.15,
            average_ph=13.0,
        )
        self.assertIn("cumulative_degradation_fraction", result)
        self.assertIn("conductivity_retention_percent", result)
        self.assertIn("estimated_lifetime_hours", result)
        self.assertIn("remaining_lifetime_hours", result)
        self.assertIn("temperature_margin_K", result)
        self.assertIn("carbonate_fraction", result)
        self.assertIn("degradation_rate_per_hour", result)
        self.assertIn("membrane_type", result)

    def test_quaternary_ammonium(self):
        aem = create_aem_membrane(membrane_type="Quaternary Ammonium")
        result = aem.get_stability_assessment(300.0, 333.15, 13.0)
        self.assertAlmostEqual(result["temperature_margin_K"], 0.0)

    def test_imidazolium(self):
        aem = create_aem_membrane(membrane_type="Imidazolium")
        result = aem.get_stability_assessment(300.0, 333.15, 13.0)
        self.assertAlmostEqual(result["temperature_margin_K"], 20.0)

    def test_other_type(self):
        aem = create_aem_membrane(membrane_type="Phosphonium")
        result = aem.get_stability_assessment(300.0, 333.15, 13.0)
        self.assertAlmostEqual(result["temperature_margin_K"], 10.0)

    def test_zero_degradation_rate(self):
        """Cover line 520: deg_rate=0 -> remaining_hours=inf.

        Extremely negative pH causes OH_conc underflow to 0, making deg_rate=0.
        """
        aem = create_aem_membrane()
        result = aem.get_stability_assessment(
            operating_hours=100.0,
            average_temperature=300.0,
            average_ph=-1000.0,  # Extreme pH -> OH_conc=0 -> deg_rate=0
        )
        self.assertEqual(result["remaining_lifetime_hours"], float("inf"))


class TestGetAEMProperties(unittest.TestCase):
    def test_low_carbonate(self):
        aem = create_aem_membrane()
        aem.carbonate_fraction = 0.05
        props = aem.get_aem_properties()
        self.assertIn("membrane_type", props)
        self.assertIn("polymer_backbone", props)
        self.assertIn("hydroxide_conductivity_S_cm", props)
        self.assertIn("carbonate_fraction", props)
        self.assertEqual(props["carbonation_sensitivity"], "Low")

    def test_high_carbonate(self):
        aem = create_aem_membrane()
        aem.carbonate_fraction = 0.2
        props = aem.get_aem_properties()
        self.assertEqual(props["carbonation_sensitivity"], "High")


class TestCreateAEMMembrane(unittest.TestCase):
    def test_quaternary_ammonium(self):
        aem = create_aem_membrane(membrane_type="Quaternary Ammonium")
        self.assertEqual(aem.aem_params.membrane_type, "Quaternary Ammonium")
        self.assertEqual(aem.aem_params.polymer_backbone, "PPO")

    def test_imidazolium(self):
        aem = create_aem_membrane(membrane_type="Imidazolium")
        self.assertEqual(aem.aem_params.polymer_backbone, "PEEK")

    def test_phosphonium(self):
        aem = create_aem_membrane(membrane_type="Phosphonium")
        self.assertEqual(aem.aem_params.polymer_backbone, "Fluorinated")

    def test_unknown_type(self):
        aem = create_aem_membrane(membrane_type="Unknown")
        self.assertEqual(aem.aem_params.polymer_backbone, "Polymer")

    def test_custom_params(self):
        aem = create_aem_membrane(
            thickness_um=80.0,
            area_cm2=2.0,
            temperature_C=50.0,
            ion_exchange_capacity=2.5,
        )
        self.assertAlmostEqual(aem.thickness, 80e-6)
        self.assertAlmostEqual(aem.area, 2e-4)
        self.assertAlmostEqual(aem.temperature, 323.15)


if __name__ == "__main__":
    unittest.main()
