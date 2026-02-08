"""Coverage tests for proton_exchange.py targeting 98%+ statement coverage."""
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

from membrane_models.proton_exchange import (
    PEMParameters,
    ProtonExchangeMembrane,
    create_nafion_membrane,
    create_speek_membrane,
)
from membrane_models.base_membrane import IonType


class TestPEMParameters(unittest.TestCase):
    def test_defaults(self):
        p = PEMParameters()
        self.assertEqual(p.membrane_type, "Nafion")
        self.assertEqual(p.membrane_grade, "117")
        self.assertAlmostEqual(p.conductivity_ref, 0.1)
        self.assertAlmostEqual(p.conductivity_activation, 15000)
        self.assertAlmostEqual(p.water_uptake_max, 22.0)
        self.assertAlmostEqual(p.water_diffusion, 2.5e-10)
        self.assertAlmostEqual(p.electro_osmotic_drag, 2.5)
        self.assertAlmostEqual(p.hydraulic_permeability, 1e-18)
        self.assertAlmostEqual(p.youngs_modulus, 250e6)
        self.assertAlmostEqual(p.tensile_strength, 25e6)
        self.assertAlmostEqual(p.max_swelling, 0.25)
        self.assertAlmostEqual(p.peroxide_rate_constant, 1e-8)
        self.assertAlmostEqual(p.fluoride_release_rate, 1e-10)
        self.assertAlmostEqual(p.methanol_permeability, 2e-6)
        self.assertAlmostEqual(p.material_cost_per_m2, 800.0)


class TestProtonExchangeMembraneInit(unittest.TestCase):
    def test_nafion_init(self):
        params = PEMParameters(membrane_type="Nafion")
        pem = ProtonExchangeMembrane(params)
        self.assertIn(IonType.PROTON, pem.ion_transport)
        self.assertIn(IonType.SODIUM, pem.ion_transport)
        self.assertIn(IonType.POTASSIUM, pem.ion_transport)
        self.assertEqual(pem.water_content_history, [])
        self.assertEqual(pem.degradation_history, [])
        self.assertAlmostEqual(pem.cumulative_fluoride_loss, 0.0)

    def test_speek_init(self):
        params = PEMParameters(membrane_type="SPEEK")
        pem = ProtonExchangeMembrane(params)
        self.assertIn(IonType.PROTON, pem.ion_transport)

    def test_generic_init(self):
        params = PEMParameters(membrane_type="Generic")
        pem = ProtonExchangeMembrane(params)
        self.assertIn(IonType.PROTON, pem.ion_transport)


class TestWaterContent(unittest.TestCase):
    def test_nafion_vapor(self):
        params = PEMParameters(membrane_type="Nafion")
        pem = ProtonExchangeMembrane(params)
        wc = pem.calculate_water_content(0.5)
        self.assertGreater(wc, 0.0)
        self.assertLessEqual(wc, params.water_uptake_max)

    def test_nafion_liquid(self):
        params = PEMParameters(membrane_type="Nafion")
        pem = ProtonExchangeMembrane(params)
        wc = pem.calculate_water_content(1.5)
        self.assertGreater(wc, 14.0)

    def test_speek_vapor(self):
        params = PEMParameters(membrane_type="SPEEK")
        pem = ProtonExchangeMembrane(params)
        wc = pem.calculate_water_content(0.5)
        self.assertAlmostEqual(wc, 4.0)

    def test_speek_liquid(self):
        params = PEMParameters(membrane_type="SPEEK")
        pem = ProtonExchangeMembrane(params)
        wc = pem.calculate_water_content(1.5)
        self.assertAlmostEqual(wc, 9.0)

    def test_generic_membrane(self):
        params = PEMParameters(membrane_type="Custom", water_uptake_max=15.0)
        pem = ProtonExchangeMembrane(params)
        wc = pem.calculate_water_content(0.5)
        self.assertAlmostEqual(wc, 7.5)

    def test_water_content_capped(self):
        params = PEMParameters(membrane_type="Nafion", water_uptake_max=5.0)
        pem = ProtonExchangeMembrane(params)
        wc = pem.calculate_water_content(1.0)
        self.assertLessEqual(wc, 5.0)


class TestIonicConductivity(unittest.TestCase):
    def test_nafion_high_water(self):
        params = PEMParameters(membrane_type="Nafion")
        pem = ProtonExchangeMembrane(params)
        cond = pem.calculate_ionic_conductivity(temperature=303.0, water_content=14.0)
        self.assertGreater(cond, 0.0)

    def test_nafion_dry(self):
        params = PEMParameters(membrane_type="Nafion")
        pem = ProtonExchangeMembrane(params)
        cond = pem.calculate_ionic_conductivity(temperature=303.0, water_content=0.5)
        self.assertGreater(cond, 0.0)
        # Low conductivity when dry
        cond_wet = pem.calculate_ionic_conductivity(temperature=303.0, water_content=14.0)
        self.assertGreater(cond_wet, cond)

    def test_non_nafion(self):
        params = PEMParameters(membrane_type="SPEEK")
        pem = ProtonExchangeMembrane(params)
        cond = pem.calculate_ionic_conductivity(temperature=303.0, water_content=8.0)
        self.assertGreater(cond, 0.0)

    def test_defaults_used(self):
        params = PEMParameters(membrane_type="Nafion")
        pem = ProtonExchangeMembrane(params)
        cond = pem.calculate_ionic_conductivity()
        self.assertGreater(cond, 0.0)

    def test_degradation_effect(self):
        params = PEMParameters(membrane_type="Nafion")
        pem = ProtonExchangeMembrane(params)
        cond_fresh = pem.calculate_ionic_conductivity()
        pem.update_degradation(10000.0)
        cond_aged = pem.calculate_ionic_conductivity()
        self.assertLess(cond_aged, cond_fresh)


class TestElectroOsmoticDrag(unittest.TestCase):
    def test_nafion_drag(self):
        params = PEMParameters(membrane_type="Nafion")
        pem = ProtonExchangeMembrane(params)
        drag = pem.calculate_electro_osmotic_drag(water_content=14.0)
        self.assertAlmostEqual(drag, 2.5 * 14.0 / 22.0, places=5)

    def test_non_nafion_drag(self):
        params = PEMParameters(membrane_type="SPEEK", electro_osmotic_drag=1.5)
        pem = ProtonExchangeMembrane(params)
        drag = pem.calculate_electro_osmotic_drag(water_content=8.0)
        self.assertAlmostEqual(drag, 1.5 * 8.0 / 14.0, places=5)

    def test_defaults(self):
        params = PEMParameters(membrane_type="Nafion")
        pem = ProtonExchangeMembrane(params)
        drag = pem.calculate_electro_osmotic_drag()
        self.assertGreater(drag, 0.0)


class TestWaterFlux(unittest.TestCase):
    def test_water_flux_basic(self):
        pem = create_nafion_membrane()
        result = pem.calculate_water_flux(
            current_density=1000.0,
            water_activity_anode=0.9,
            water_activity_cathode=1.0,
        )
        self.assertIn("electro_osmotic_flux", result)
        self.assertIn("diffusion_flux", result)
        self.assertIn("hydraulic_flux", result)
        self.assertIn("net_water_flux", result)
        self.assertIn("drag_coefficient", result)
        self.assertAlmostEqual(result["hydraulic_flux"], 0.0)

    def test_water_flux_with_pressure(self):
        pem = create_nafion_membrane()
        result = pem.calculate_water_flux(
            current_density=1000.0,
            water_activity_anode=0.9,
            water_activity_cathode=1.0,
            pressure_difference=10000.0,
        )
        self.assertNotAlmostEqual(result["hydraulic_flux"], 0.0)


class TestMethanolCrossover(unittest.TestCase):
    def test_crossover(self):
        pem = create_nafion_membrane()
        flux = pem.calculate_methanol_crossover(
            methanol_conc_anode=1000.0,
            methanol_conc_cathode=0.0,
            current_density=1000.0,
        )
        self.assertGreater(flux, 0.0)


class TestDegradationRate(unittest.TestCase):
    def test_default_conditions(self):
        pem = create_nafion_membrane()
        rate = pem.calculate_degradation_rate({})
        self.assertGreater(rate, 0.0)

    def test_harsh_conditions(self):
        pem = create_nafion_membrane()
        rate_mild = pem.calculate_degradation_rate(
            {"temperature": 303.0, "relative_humidity": 100.0, "cathode_potential": 0.6}
        )
        rate_harsh = pem.calculate_degradation_rate(
            {"temperature": 363.0, "relative_humidity": 20.0, "cathode_potential": 0.9}
        )
        self.assertGreater(rate_harsh, rate_mild)


class TestGasCrossover(unittest.TestCase):
    def test_known_gas(self):
        pem = create_nafion_membrane()
        flux = pem.calculate_gas_crossover("O2", 21000.0, 101325.0)
        self.assertIsInstance(flux, float)

    def test_unknown_gas(self):
        pem = create_nafion_membrane()
        flux = pem.calculate_gas_crossover("Ar", 21000.0, 101325.0)
        self.assertEqual(flux, 0.0)

    def test_all_known_gases(self):
        pem = create_nafion_membrane()
        for gas in ["O2", "H2", "CO2", "N2"]:
            flux = pem.calculate_gas_crossover(gas, 0.0, 101325.0)
            self.assertIsInstance(flux, float)


class TestHumidityCycling(unittest.TestCase):
    def test_cycling(self):
        pem = create_nafion_membrane()
        result = pem.simulate_humidity_cycling(n_cycles=100, RH_high=100.0, RH_low=30.0)
        self.assertIn("n_cycles", result)
        self.assertIn("swelling_strain", result)
        self.assertIn("cycles_to_failure", result)
        self.assertIn("damage_fraction", result)
        self.assertIn("conductivity_loss_percent", result)
        self.assertIn("initial_conductivity_S_cm", result)
        self.assertIn("final_conductivity_S_cm", result)
        self.assertEqual(result["n_cycles"], 100)


class TestCostAnalysis(unittest.TestCase):
    def test_cost_analysis(self):
        pem = create_nafion_membrane()
        cost = pem.get_cost_analysis()
        self.assertIn("material_cost_USD", cost)
        self.assertIn("cost_per_m2_USD", cost)
        self.assertIn("cost_per_kW_USD", cost)
        self.assertIn("cost_per_kWh_USD", cost)
        self.assertIn("membrane_type", cost)
        self.assertIn("membrane_grade", cost)
        self.assertIn("thickness_um", cost)
        self.assertEqual(cost["membrane_type"], "Nafion")


class TestGetPEMProperties(unittest.TestCase):
    def test_nafion_properties(self):
        pem = create_nafion_membrane()
        props = pem.get_pem_properties()
        self.assertIn("membrane_type", props)
        self.assertIn("membrane_grade", props)
        self.assertIn("water_content", props)
        self.assertIn("conductivity_S_cm", props)
        self.assertIn("drag_coefficient", props)
        self.assertIn("max_operating_temp_C", props)
        self.assertAlmostEqual(props["max_operating_temp_C"], 80.0)

    def test_speek_properties(self):
        pem = create_speek_membrane()
        props = pem.get_pem_properties()
        self.assertAlmostEqual(props["max_operating_temp_C"], 160.0)


class TestCreateNafionMembrane(unittest.TestCase):
    def test_thick_nafion(self):
        pem = create_nafion_membrane(thickness_um=250.0)
        self.assertEqual(pem.pem_params.membrane_grade, "N117")

    def test_nafion_115(self):
        pem = create_nafion_membrane(thickness_um=180.0)
        self.assertEqual(pem.pem_params.membrane_grade, "N115")

    def test_nafion_1110(self):
        pem = create_nafion_membrane(thickness_um=120.0)
        self.assertEqual(pem.pem_params.membrane_grade, "N1110")

    def test_nafion_112(self):
        pem = create_nafion_membrane(thickness_um=80.0)
        self.assertEqual(pem.pem_params.membrane_grade, "N112")

    def test_nafion_212(self):
        pem = create_nafion_membrane(thickness_um=50.0)
        self.assertEqual(pem.pem_params.membrane_grade, "N212")

    def test_custom_area_temp(self):
        pem = create_nafion_membrane(area_cm2=5.0, temperature_C=60.0)
        self.assertAlmostEqual(pem.area, 5e-4)
        self.assertAlmostEqual(pem.temperature, 333.15)


class TestCreateSPEEKMembrane(unittest.TestCase):
    def test_default(self):
        pem = create_speek_membrane()
        self.assertEqual(pem.pem_params.membrane_type, "SPEEK")
        self.assertEqual(pem.pem_params.membrane_grade, "DS70")
        self.assertAlmostEqual(pem.pem_params.material_cost_per_m2, 200.0)

    def test_custom_ds(self):
        pem = create_speek_membrane(degree_sulfonation=0.5)
        self.assertEqual(pem.pem_params.membrane_grade, "DS50")

    def test_custom_params(self):
        pem = create_speek_membrane(
            thickness_um=80.0, area_cm2=2.0, temperature_C=50.0
        )
        self.assertAlmostEqual(pem.thickness, 80e-6)
        self.assertAlmostEqual(pem.area, 2e-4)
        self.assertAlmostEqual(pem.temperature, 323.15)


if __name__ == "__main__":
    unittest.main()
