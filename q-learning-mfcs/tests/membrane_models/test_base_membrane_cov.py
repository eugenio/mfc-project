"""Coverage tests for base_membrane.py targeting 98%+ statement coverage."""
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

from membrane_models.base_membrane import (
    BaseMembraneModel,
    IonTransportDatabase,
    IonTransportMechanisms,
    IonType,
    MembraneParameters,
    TransportMechanism,
)


class ConcreteMembraneModel(BaseMembraneModel):
    """Concrete subclass for testing the abstract BaseMembraneModel."""

    def _setup_ion_transport(self):
        self.ion_transport = {}
        self.ion_transport[IonType.PROTON] = IonTransportDatabase.get_proton_transport()
        self.ion_transport[IonType.HYDROXIDE] = IonTransportDatabase.get_hydroxide_transport()
        self.ion_transport[IonType.SODIUM] = IonTransportDatabase.get_sodium_transport()

    def _calculate_membrane_properties(self):
        pass

    def calculate_ionic_conductivity(self, temperature=None, water_content=None):
        return 10.0  # S/m


class TestIonType(unittest.TestCase):
    def test_all_ion_types(self):
        expected = {
            "PROTON": "H+",
            "HYDROXIDE": "OH-",
            "SODIUM": "Na+",
            "POTASSIUM": "K+",
            "CHLORIDE": "Cl-",
            "BICARBONATE": "HCO3-",
            "CARBONATE": "CO3--",
            "SULFATE": "SO4--",
            "PHOSPHATE": "PO4---",
        }
        for name, value in expected.items():
            self.assertEqual(IonType[name].value, value)


class TestTransportMechanism(unittest.TestCase):
    def test_all_transport_mechanisms(self):
        expected = {
            "MIGRATION": "migration",
            "DIFFUSION": "diffusion",
            "CONVECTION": "convection",
            "ELECTROOSMOTIC": "electroosmotic",
        }
        for name, value in expected.items():
            self.assertEqual(TransportMechanism[name].value, value)


class TestMembraneParameters(unittest.TestCase):
    def test_default_values(self):
        params = MembraneParameters()
        self.assertAlmostEqual(params.thickness, 100e-6)
        self.assertAlmostEqual(params.area, 1e-4)
        self.assertAlmostEqual(params.porosity, 0.3)
        self.assertAlmostEqual(params.tortuosity, 2.5)
        self.assertAlmostEqual(params.pore_size, 1e-9)
        self.assertAlmostEqual(params.temperature, 298.15)
        self.assertAlmostEqual(params.pressure_anode, 101325)
        self.assertAlmostEqual(params.pressure_cathode, 101325)
        self.assertAlmostEqual(params.ion_exchange_capacity, 1.2)
        self.assertAlmostEqual(params.water_content, 10.0)
        self.assertAlmostEqual(params.fixed_charge_density, 1200)
        self.assertAlmostEqual(params.faraday_constant, 96485.0)
        self.assertAlmostEqual(params.gas_constant, 8.314)
        self.assertAlmostEqual(params.initial_conductivity, 0.1)
        self.assertAlmostEqual(params.degradation_rate, 1e-6)
        self.assertAlmostEqual(params.fouling_resistance, 0.0)

    def test_custom_values(self):
        params = MembraneParameters(thickness=200e-6, area=2e-4, temperature=310.0)
        self.assertAlmostEqual(params.thickness, 200e-6)
        self.assertAlmostEqual(params.area, 2e-4)
        self.assertAlmostEqual(params.temperature, 310.0)


class TestIonTransportMechanisms(unittest.TestCase):
    def test_creation(self):
        transport = IonTransportMechanisms(
            ion_type=IonType.PROTON,
            diffusion_coefficient=9.3e-9,
            mobility=3.6e-7,
            partition_coefficient=1.0,
            hydration_number=3.0,
            charge=1,
            stokes_radius=2.8e-10,
        )
        self.assertEqual(transport.ion_type, IonType.PROTON)
        self.assertAlmostEqual(transport.diffusion_coefficient, 9.3e-9)
        self.assertEqual(transport.charge, 1)


class TestIonTransportDatabase(unittest.TestCase):
    def test_get_proton_transport(self):
        transport = IonTransportDatabase.get_proton_transport()
        self.assertEqual(transport.ion_type, IonType.PROTON)
        self.assertAlmostEqual(transport.diffusion_coefficient, 9.3e-9)
        self.assertAlmostEqual(transport.mobility, 3.6e-7)
        self.assertAlmostEqual(transport.partition_coefficient, 1.0)
        self.assertAlmostEqual(transport.hydration_number, 3.0)
        self.assertEqual(transport.charge, 1)
        self.assertAlmostEqual(transport.stokes_radius, 2.8e-10)

    def test_get_hydroxide_transport(self):
        transport = IonTransportDatabase.get_hydroxide_transport()
        self.assertEqual(transport.ion_type, IonType.HYDROXIDE)
        self.assertAlmostEqual(transport.diffusion_coefficient, 5.3e-9)
        self.assertAlmostEqual(transport.mobility, 2.0e-7)
        self.assertAlmostEqual(transport.partition_coefficient, 0.8)
        self.assertEqual(transport.charge, -1)

    def test_get_sodium_transport(self):
        transport = IonTransportDatabase.get_sodium_transport()
        self.assertEqual(transport.ion_type, IonType.SODIUM)
        self.assertAlmostEqual(transport.diffusion_coefficient, 1.3e-9)
        self.assertAlmostEqual(transport.mobility, 5.2e-8)
        self.assertAlmostEqual(transport.partition_coefficient, 0.3)
        self.assertEqual(transport.charge, 1)


class TestBaseMembraneModelInit(unittest.TestCase):
    def test_initialization(self):
        params = MembraneParameters()
        model = ConcreteMembraneModel(params)
        self.assertEqual(model.thickness, params.thickness)
        self.assertEqual(model.area, params.area)
        self.assertEqual(model.temperature, params.temperature)
        self.assertAlmostEqual(model.operating_hours, 0.0)
        self.assertIn(IonType.PROTON, model.ion_transport)
        self.assertIn(IonType.HYDROXIDE, model.ion_transport)
        self.assertIn(IonType.SODIUM, model.ion_transport)


class TestNernstPlanckFlux(unittest.TestCase):
    def setUp(self):
        self.model = ConcreteMembraneModel(MembraneParameters())

    def test_flux_with_known_ion(self):
        flux = self.model.calculate_nernst_planck_flux(
            IonType.PROTON, 100.0, 10.0, 1000.0
        )
        self.assertIsInstance(flux, float)
        # Diffusion from anode to cathode (high to low conc)
        # Migration depends on potential gradient sign

    def test_flux_with_unknown_ion(self):
        flux = self.model.calculate_nernst_planck_flux(
            IonType.POTASSIUM, 100.0, 10.0, 1000.0
        )
        self.assertEqual(flux, 0.0)

    def test_flux_zero_gradient(self):
        flux = self.model.calculate_nernst_planck_flux(
            IonType.PROTON, 50.0, 50.0, 0.0
        )
        self.assertAlmostEqual(flux, 0.0, places=15)

    def test_flux_negative_potential(self):
        flux = self.model.calculate_nernst_planck_flux(
            IonType.PROTON, 100.0, 10.0, -500.0
        )
        self.assertIsInstance(flux, float)


class TestDonnanPotential(unittest.TestCase):
    def setUp(self):
        self.model = ConcreteMembraneModel(MembraneParameters())

    def test_donnan_with_valid_concentrations(self):
        anode = {IonType.PROTON: 100.0, IonType.SODIUM: 50.0}
        cathode = {IonType.PROTON: 10.0, IonType.SODIUM: 50.0}
        potential = self.model.calculate_donnan_potential(anode, cathode)
        self.assertIsInstance(potential, float)

    def test_donnan_equal_concentrations(self):
        conc = {IonType.PROTON: 100.0}
        potential = self.model.calculate_donnan_potential(conc, conc)
        self.assertAlmostEqual(potential, 0.0, places=5)

    def test_donnan_zero_one_side(self):
        anode = {IonType.PROTON: 0.0}
        cathode = {IonType.PROTON: 100.0}
        potential = self.model.calculate_donnan_potential(anode, cathode)
        self.assertEqual(potential, 0.0)

    def test_donnan_unknown_ions_ignored(self):
        anode = {IonType.POTASSIUM: 100.0}  # Not in model transport
        cathode = {IonType.POTASSIUM: 10.0}
        potential = self.model.calculate_donnan_potential(anode, cathode)
        self.assertEqual(potential, 0.0)


class TestWaterTransport(unittest.TestCase):
    def setUp(self):
        self.model = ConcreteMembraneModel(MembraneParameters())

    def test_water_transport_positive_current(self):
        flux = self.model.calculate_water_transport(1000.0, IonType.PROTON)
        self.assertGreater(flux, 0.0)

    def test_water_transport_unknown_ion(self):
        flux = self.model.calculate_water_transport(1000.0, IonType.POTASSIUM)
        self.assertEqual(flux, 0.0)

    def test_water_transport_zero_current(self):
        flux = self.model.calculate_water_transport(0.0, IonType.PROTON)
        self.assertAlmostEqual(flux, 0.0)

    def test_water_activity_capping(self):
        params = MembraneParameters(water_content=20.0)
        model = ConcreteMembraneModel(params)
        flux = model.calculate_water_transport(1000.0, IonType.PROTON)
        self.assertGreater(flux, 0.0)


class TestGasPermeability(unittest.TestCase):
    def setUp(self):
        self.model = ConcreteMembraneModel(MembraneParameters())

    def test_known_gases(self):
        for gas in ["O2", "H2", "CO2", "N2", "CH4"]:
            flux = self.model.calculate_gas_permeability(gas, 1000.0)
            self.assertGreater(flux, 0.0, f"{gas} flux should be positive")

    def test_unknown_gas(self):
        flux = self.model.calculate_gas_permeability("Ar", 1000.0)
        self.assertEqual(flux, 0.0)

    def test_zero_pressure(self):
        flux = self.model.calculate_gas_permeability("O2", 0.0)
        self.assertAlmostEqual(flux, 0.0, places=20)

    def test_temperature_effect(self):
        params_cold = MembraneParameters(temperature=280.0)
        params_hot = MembraneParameters(temperature=350.0)
        model_cold = ConcreteMembraneModel(params_cold)
        model_hot = ConcreteMembraneModel(params_hot)

        flux_cold = model_cold.calculate_gas_permeability("O2", 1000.0)
        flux_hot = model_hot.calculate_gas_permeability("O2", 1000.0)
        # Higher temperature should generally increase permeability
        self.assertGreater(flux_hot, flux_cold)


class TestMembraneResistance(unittest.TestCase):
    def setUp(self):
        self.model = ConcreteMembraneModel(MembraneParameters())

    def test_resistance_default_conductivity(self):
        resistance = self.model.calculate_membrane_resistance()
        self.assertGreater(resistance, 0.0)

    def test_resistance_custom_conductivity(self):
        resistance = self.model.calculate_membrane_resistance(ionic_conductivity=5.0)
        self.assertGreater(resistance, 0.0)

    def test_resistance_with_fouling(self):
        self.model.add_fouling_resistance(0.01)
        resistance = self.model.calculate_membrane_resistance()
        self.assertGreater(resistance, 0.0)

    def test_resistance_after_degradation(self):
        self.model.update_degradation(1000.0)
        resistance = self.model.calculate_membrane_resistance()
        self.assertGreater(resistance, 0.0)


class TestSelectivity(unittest.TestCase):
    def setUp(self):
        self.model = ConcreteMembraneModel(MembraneParameters())

    def test_same_ion_selectivity(self):
        selectivity = self.model.calculate_selectivity(IonType.PROTON, IonType.PROTON)
        # Same ion selectivity with same D and K ratios => charge selectivity=1
        self.assertAlmostEqual(selectivity, 1.0, places=5)

    def test_opposite_charge_selectivity(self):
        selectivity = self.model.calculate_selectivity(IonType.PROTON, IonType.HYDROXIDE)
        # Opposite charges -> high selectivity factor (10.0)
        self.assertGreater(selectivity, 1.0)

    def test_unknown_ion_selectivity(self):
        selectivity = self.model.calculate_selectivity(IonType.PROTON, IonType.POTASSIUM)
        self.assertAlmostEqual(selectivity, 1.0)

    def test_both_unknown_ions(self):
        selectivity = self.model.calculate_selectivity(IonType.POTASSIUM, IonType.CHLORIDE)
        self.assertAlmostEqual(selectivity, 1.0)

    def test_same_charge_same_magnitude(self):
        selectivity = self.model.calculate_selectivity(IonType.PROTON, IonType.SODIUM)
        # Same charge, same magnitude => charge_selectivity=1.0
        self.assertIsInstance(selectivity, float)

    def test_same_sign_different_magnitude(self):
        """Cover line 373: same sign but different charge magnitudes."""
        # Manually add a divalent cation transport entry to trigger else branch
        self.model.ion_transport[IonType.POTASSIUM] = IonTransportMechanisms(
            ion_type=IonType.POTASSIUM,
            diffusion_coefficient=1.96e-9,
            mobility=7.6e-8,
            partition_coefficient=0.2,
            hydration_number=6.0,
            charge=2,  # Divalent (different from +1)
            stokes_radius=1.3e-10,
        )
        selectivity = self.model.calculate_selectivity(IonType.PROTON, IonType.POTASSIUM)
        # z1=1, z2=2, same sign -> charge_selectivity = abs(1/2) = 0.5
        self.assertIsInstance(selectivity, float)
        self.assertGreater(selectivity, 0.0)


class TestTransportNumber(unittest.TestCase):
    def setUp(self):
        self.model = ConcreteMembraneModel(MembraneParameters())

    def test_transport_number_single_ion(self):
        concentrations = {IonType.PROTON: 100.0}
        t_num = self.model.calculate_transport_number(
            IonType.PROTON, concentrations, 1000.0
        )
        self.assertAlmostEqual(t_num, 1.0, places=5)

    def test_transport_number_multiple_ions(self):
        concentrations = {IonType.PROTON: 100.0, IonType.SODIUM: 100.0}
        t_proton = self.model.calculate_transport_number(
            IonType.PROTON, concentrations, 1000.0
        )
        t_sodium = self.model.calculate_transport_number(
            IonType.SODIUM, concentrations, 1000.0
        )
        self.assertGreater(t_proton, 0.0)
        self.assertGreater(t_sodium, 0.0)
        self.assertAlmostEqual(t_proton + t_sodium, 1.0, places=5)

    def test_transport_number_unknown_ion(self):
        concentrations = {IonType.POTASSIUM: 100.0}
        t_num = self.model.calculate_transport_number(
            IonType.POTASSIUM, concentrations, 1000.0
        )
        self.assertEqual(t_num, 0.0)

    def test_transport_number_ion_not_in_concentrations(self):
        concentrations = {IonType.SODIUM: 100.0}
        t_num = self.model.calculate_transport_number(
            IonType.PROTON, concentrations, 1000.0
        )
        self.assertEqual(t_num, 0.0)

    def test_transport_number_zero_conductivity(self):
        concentrations = {}
        t_num = self.model.calculate_transport_number(
            IonType.PROTON, concentrations, 1000.0
        )
        self.assertEqual(t_num, 0.0)

    def test_transport_number_zero_concentration(self):
        """Cover line 423: total_conductivity=0 when all concentrations are 0."""
        concentrations = {IonType.PROTON: 0.0}
        t_num = self.model.calculate_transport_number(
            IonType.PROTON, concentrations, 1000.0
        )
        self.assertEqual(t_num, 0.0)


class TestUpdateOperatingConditions(unittest.TestCase):
    def setUp(self):
        self.model = ConcreteMembraneModel(MembraneParameters())

    def test_update_temperature(self):
        self.model.update_operating_conditions(temperature=330.0)
        self.assertAlmostEqual(self.model.temperature, 330.0)
        self.assertAlmostEqual(self.model.params.temperature, 330.0)

    def test_update_pressures(self):
        self.model.update_operating_conditions(
            pressure_anode=110000.0, pressure_cathode=120000.0
        )
        self.assertAlmostEqual(self.model.params.pressure_anode, 110000.0)
        self.assertAlmostEqual(self.model.params.pressure_cathode, 120000.0)

    def test_update_none_values(self):
        orig_temp = self.model.temperature
        self.model.update_operating_conditions()
        self.assertAlmostEqual(self.model.temperature, orig_temp)


class TestDegradationAndFouling(unittest.TestCase):
    def setUp(self):
        self.model = ConcreteMembraneModel(MembraneParameters())

    def test_update_degradation(self):
        self.model.update_degradation(500.0)
        self.assertAlmostEqual(self.model.operating_hours, 500.0)

    def test_add_fouling_resistance(self):
        initial = self.model.params.fouling_resistance
        self.model.add_fouling_resistance(0.05)
        self.assertAlmostEqual(self.model.params.fouling_resistance, initial + 0.05)


class TestPerformanceMetrics(unittest.TestCase):
    def setUp(self):
        self.model = ConcreteMembraneModel(MembraneParameters())

    def test_performance_metrics(self):
        ion_conc = {IonType.PROTON: 100.0, IonType.SODIUM: 50.0}
        metrics = self.model.get_performance_metrics(1000.0, ion_conc)

        self.assertIn("ionic_conductivity_S_m", metrics)
        self.assertIn("membrane_resistance_ohm", metrics)
        self.assertIn("voltage_drop_V", metrics)
        self.assertIn("power_loss_W", metrics)
        self.assertIn("thickness_um", metrics)
        self.assertIn("area_cm2", metrics)
        self.assertIn("operating_hours", metrics)
        self.assertIn("fouling_resistance_ohm_m2", metrics)
        self.assertIn("transport_number_H+", metrics)
        self.assertIn("transport_number_Na+", metrics)

    def test_performance_metrics_with_no_matching_ions(self):
        ion_conc = {IonType.POTASSIUM: 100.0}
        metrics = self.model.get_performance_metrics(1000.0, ion_conc)
        self.assertIn("ionic_conductivity_S_m", metrics)
        # No transport numbers for unknown ions
        self.assertNotIn("transport_number_K+", metrics)


class TestGetTransportProperties(unittest.TestCase):
    def test_transport_properties(self):
        model = ConcreteMembraneModel(MembraneParameters())
        props = model.get_transport_properties()

        self.assertEqual(props["membrane_type"], "ConcreteMembraneModel")
        self.assertAlmostEqual(props["thickness_m"], 100e-6)
        self.assertAlmostEqual(props["area_m2"], 1e-4)
        self.assertAlmostEqual(props["porosity"], 0.3)
        self.assertAlmostEqual(props["tortuosity"], 2.5)
        self.assertIn("ion_transport_mechanisms", props)
        self.assertIn("H+", props["ion_transport_mechanisms"])
        self.assertIn("diffusion_coefficient_m2_s", props["ion_transport_mechanisms"]["H+"])


class TestRepr(unittest.TestCase):
    def test_repr(self):
        model = ConcreteMembraneModel(MembraneParameters())
        r = repr(model)
        self.assertIn("ConcreteMembraneModel", r)
        self.assertIn("100.0", r)  # thickness in um
        self.assertIn("1.0", r)  # area in cm2
        self.assertIn("298.1", r)  # temperature


if __name__ == "__main__":
    unittest.main()
