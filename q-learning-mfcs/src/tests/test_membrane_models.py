#!/usr/bin/env python3
"""
Test suite for membrane models

Tests all membrane types, transport mechanisms, and fouling models
with comprehensive unit tests and integration tests.

Created: 2025-07-27
"""

import unittest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

try:
    from membrane_models.base_membrane import (
        BaseMembraneModel, MembraneParameters, IonType, IonTransportDatabase
    )
    from membrane_models.proton_exchange import (
        ProtonExchangeMembrane, PEMParameters, create_nafion_membrane, create_speek_membrane
    )
    from membrane_models.anion_exchange import (
        AnionExchangeMembrane, AEMParameters, create_aem_membrane
    )
    from membrane_models.membrane_fouling import FoulingModel, FoulingParameters
    from membrane_models.bipolar_membrane import create_bipolar_membrane
    from membrane_models.ceramic_membrane import create_ceramic_membrane
    MEMBRANE_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Membrane model import error: {e}")
    MEMBRANE_IMPORTS_AVAILABLE = False


class TestBaseMembraneModel(unittest.TestCase):
    """Test base membrane model functionality."""
    
    def setUp(self):
        """Set up test environment."""
        if not MEMBRANE_IMPORTS_AVAILABLE:
            self.skipTest("Membrane model modules not available")
    
    def test_ion_transport_database(self):
        """Test ion transport database access."""
        proton = IonTransportDatabase.get_proton_transport()
        self.assertEqual(proton.ion_type, IonType.PROTON)
        self.assertEqual(proton.charge, 1)
        self.assertGreater(proton.diffusion_coefficient, 0)
        
        hydroxide = IonTransportDatabase.get_hydroxide_transport()
        self.assertEqual(hydroxide.ion_type, IonType.HYDROXIDE)
        self.assertEqual(hydroxide.charge, -1)
    
    def test_nernst_planck_calculation(self):
        """Test Nernst-Planck flux calculation."""
        
        # Create a simple test membrane
        class TestMembrane(BaseMembraneModel):
            def _setup_ion_transport(self):
                self.ion_transport = {
                    IonType.PROTON: IonTransportDatabase.get_proton_transport()
                }
            
            def _calculate_membrane_properties(self):
                pass
            
            def calculate_ionic_conductivity(self, temperature=None, water_content=None):
                return 1.0  # S/m
        
        params = MembraneParameters(thickness=100e-6, area=1e-4)
        membrane = TestMembrane(params)
        
        # Test flux calculation
        flux = membrane.calculate_nernst_planck_flux(
            ion=IonType.PROTON,
            concentration_anode=1000,    # mol/m³
            concentration_cathode=500,   # mol/m³
            potential_gradient=1000      # V/m
        )
        
        # Should give reasonable flux
        self.assertIsInstance(flux, float)
        self.assertNotEqual(flux, 0.0)
    
    def test_gas_permeability(self):
        """Test gas permeability calculations."""
        
        class TestMembrane(BaseMembraneModel):
            def _setup_ion_transport(self):
                pass
            def _calculate_membrane_properties(self):
                pass  
            def calculate_ionic_conductivity(self, temperature=None, water_content=None):
                return 1.0
        
        params = MembraneParameters(thickness=100e-6, area=1e-4)
        membrane = TestMembrane(params)
        
        # Test oxygen permeability
        o2_flux = membrane.calculate_gas_permeability("O2", 10000)  # 10 kPa pressure diff
        self.assertGreater(o2_flux, 0)
        
        # Test unknown gas
        unknown_flux = membrane.calculate_gas_permeability("Ar", 10000)
        self.assertEqual(unknown_flux, 0.0)


class TestProtonExchangeMembrane(unittest.TestCase):
    """Test proton exchange membrane functionality."""
    
    def setUp(self):
        if not MEMBRANE_IMPORTS_AVAILABLE:
            self.skipTest("Membrane model modules not available")
        
        self.nafion = create_nafion_membrane(
            thickness_um=183.0,
            area_cm2=1.0,
            temperature_C=30.0
        )
    
    def test_nafion_creation(self):
        """Test Nafion membrane creation."""
        self.assertIsInstance(self.nafion, ProtonExchangeMembrane)
        self.assertEqual(self.nafion.pem_params.membrane_type, "Nafion")
        self.assertAlmostEqual(self.nafion.thickness, 183e-6, places=8)
    
    def test_water_content_calculation(self):
        """Test water content calculation with sorption isotherm."""
        
        # Test at different water activities
        lambda_dry = self.nafion.calculate_water_content(0.0)
        lambda_humid = self.nafion.calculate_water_content(1.0)
        lambda_liquid = self.nafion.calculate_water_content(1.2)
        
        # Should increase with water activity
        self.assertLess(lambda_dry, lambda_humid)
        self.assertLess(lambda_humid, lambda_liquid)
        
        # Reasonable values for Nafion
        self.assertGreater(lambda_humid, 10.0)
        self.assertLess(lambda_humid, 25.0)
    
    def test_proton_conductivity(self):
        """Test proton conductivity calculation."""
        
        # Test temperature dependence
        conductivity_25C = self.nafion.calculate_ionic_conductivity(298.15)
        conductivity_80C = self.nafion.calculate_ionic_conductivity(353.15)
        
        # Higher temperature should give higher conductivity
        self.assertGreater(conductivity_80C, conductivity_25C)
        
        # Reasonable range (0.01 - 1 S/cm → 1 - 100 S/m)
        self.assertGreater(conductivity_80C, 1.0)
        self.assertLess(conductivity_80C, 100.0)
    
    def test_electro_osmotic_drag(self):
        """Test electro-osmotic drag coefficient."""
        
        drag = self.nafion.calculate_electro_osmotic_drag()
        
        # Should be positive and reasonable (1-4 H2O/H+)
        self.assertGreater(drag, 0.5)
        self.assertLess(drag, 5.0)
    
    def test_water_transport(self):
        """Test water transport mechanisms."""
        
        water_fluxes = self.nafion.calculate_water_flux(
            current_density=5000,        # A/m²
            water_activity_anode=0.8,
            water_activity_cathode=1.0,
            pressure_difference=0.0
        )
        
        # Check required keys
        required_keys = ['electro_osmotic_flux', 'diffusion_flux', 'net_water_flux']
        for key in required_keys:
            self.assertIn(key, water_fluxes)
            self.assertIsInstance(water_fluxes[key], float)
        
        # Net flux should be reasonable
        self.assertNotEqual(water_fluxes['net_water_flux'], 0.0)
    
    def test_methanol_crossover(self):
        """Test methanol crossover for DMFC."""
        
        methanol_flux = self.nafion.calculate_methanol_crossover(
            methanol_conc_anode=1000,    # mol/m³
            methanol_conc_cathode=10,    # mol/m³
            current_density=5000         # A/m²
        )
        
        # Should be positive (anode to cathode)
        self.assertGreater(methanol_flux, 0.0)
        
        # Reasonable magnitude
        self.assertLess(methanol_flux, 1.0)  # mol/m²/s
    
    def test_gas_crossover(self):
        """Test gas crossover with water content effects."""
        
        o2_flux = self.nafion.calculate_gas_crossover(
            gas_type="O2",
            partial_pressure_anode=5000,     # Pa
            partial_pressure_cathode=20000   # Pa
        )
        
        # Should be positive (low to high pressure)
        self.assertGreater(o2_flux, 0.0)
        
        # Test hydrogen
        h2_flux = self.nafion.calculate_gas_crossover(
            gas_type="H2",
            partial_pressure_anode=50000,
            partial_pressure_cathode=10000
        )
        
        # Should be negative (high to low pressure)
        self.assertLess(h2_flux, 0.0)
    
    def test_humidity_cycling(self):
        """Test humidity cycling simulation."""
        
        cycling_results = self.nafion.simulate_humidity_cycling(
            n_cycles=100,
            RH_high=95.0,
            RH_low=30.0,
            cycle_time=0.5
        )
        
        # Check required keys
        required_keys = [
            'swelling_strain', 'cycles_to_failure', 'conductivity_loss_percent'
        ]
        for key in required_keys:
            self.assertIn(key, cycling_results)
        
        # Should show some degradation
        self.assertGreater(cycling_results['conductivity_loss_percent'], 0)
        self.assertLess(cycling_results['conductivity_loss_percent'], 100)
    
    def test_cost_analysis(self):
        """Test PEM cost analysis."""
        
        cost_data = self.nafion.get_cost_analysis()
        
        # Check cost components
        self.assertIn('material_cost_USD', cost_data)
        self.assertIn('cost_per_kW_USD', cost_data)
        self.assertGreater(cost_data['material_cost_USD'], 0)
        
        # Nafion should be expensive
        self.assertGreater(cost_data['cost_per_m2_USD'], 100)
    
    def test_speek_membrane(self):
        """Test SPEEK membrane creation and properties."""
        
        speek = create_speek_membrane(
            degree_sulfonation=0.7,
            thickness_um=50.0,
            area_cm2=1.0
        )
        
        self.assertEqual(speek.pem_params.membrane_type, "SPEEK")
        
        # Should have lower cost than Nafion
        speek_cost = speek.get_cost_analysis()
        nafion_cost = self.nafion.get_cost_analysis()
        self.assertLess(speek_cost['cost_per_m2_USD'], nafion_cost['cost_per_m2_USD'])


class TestAnionExchangeMembrane(unittest.TestCase):
    """Test anion exchange membrane functionality."""
    
    def setUp(self):
        if not MEMBRANE_IMPORTS_AVAILABLE:
            self.skipTest("Membrane model modules not available")
        
        self.aem = create_aem_membrane(
            membrane_type="Quaternary Ammonium",
            thickness_um=100.0,
            area_cm2=1.0,
            temperature_C=30.0
        )
    
    def test_aem_creation(self):
        """Test AEM creation."""
        self.assertIsInstance(self.aem, AnionExchangeMembrane)
        self.assertEqual(self.aem.aem_params.membrane_type, "Quaternary Ammonium")
    
    def test_hydroxide_conductivity(self):
        """Test hydroxide conductivity calculation."""
        
        conductivity = self.aem.calculate_ionic_conductivity()
        
        # Should be positive but typically lower than PEM
        self.assertGreater(conductivity, 0.1)  # S/m
        self.assertLess(conductivity, 50.0)    # Lower than PEM
    
    def test_carbonation_effects(self):
        """Test CO2 carbonation effects."""
        
        initial_conductivity = self.aem.calculate_ionic_conductivity()
        
        # Simulate CO2 exposure
        self.aem.update_carbonation(
            co2_partial_pressure=1000,  # Pa
            exposure_time=10.0          # hours
        )
        
        carbonated_conductivity = self.aem.calculate_ionic_conductivity()
        
        # Conductivity should decrease
        self.assertLess(carbonated_conductivity, initial_conductivity)
        self.assertGreater(self.aem.carbonate_fraction, 0.0)
    
    def test_ph_gradient_effects(self):
        """Test pH gradient calculations."""
        
        ph_effects = self.aem.calculate_ph_gradient_effect(
            ph_anode=12.0,
            ph_cathode=14.0
        )
        
        required_keys = ['ph_gradient', 'OH_concentration_gradient', 'diffusion_potential_V']
        for key in required_keys:
            self.assertIn(key, ph_effects)
        
        # pH gradient should be positive
        self.assertAlmostEqual(ph_effects['ph_gradient'], 2.0, places=1)
    
    def test_degradation_calculation(self):
        """Test AEM degradation rate calculation."""
        
        degradation_rate = self.aem.calculate_degradation_rate(
            temperature=333.15,  # 60°C
            ph=13.0,
            current_density=5000
        )
        
        # Should be positive
        self.assertGreater(degradation_rate, 0.0)
        
        # Should be reasonable (not too fast)
        self.assertLess(degradation_rate, 1e-3)  # h⁻¹
    
    def test_co2_mitigation_strategies(self):
        """Test CO2 mitigation simulation."""
        
        operating_conditions = {
            'co2_ppm': 400,
            'ph_cathode': 13.0,
            'use_co2_scrubber': True,
            'pulse_frequency_hz': 5.0
        }
        
        mitigation = self.aem.simulate_co2_mitigation(operating_conditions)
        
        # Should reduce effective CO2
        self.assertLess(mitigation['effective_co2_ppm'], 400)
        self.assertGreater(mitigation['performance_retention_percent'], 50)
    
    def test_water_balance(self):
        """Test AEM water balance."""
        
        water_balance = self.aem.calculate_water_balance(
            current_density=5000,
            rh_anode=80.0,
            rh_cathode=90.0
        )
        
        # Check required keys
        required_keys = ['electro_osmotic_flux_mol_m2_s', 'net_water_flux_mol_m2_s']
        for key in required_keys:
            self.assertIn(key, water_balance)
        
        # AEM should have higher drag than PEM
        self.assertGreater(water_balance['effective_drag_coefficient'], 3.0)
    
    def test_stability_assessment(self):
        """Test stability assessment."""
        
        stability = self.aem.get_stability_assessment(
            operating_hours=1000,
            average_temperature=333.15,
            average_ph=13.0
        )
        
        # Check metrics
        self.assertIn('conductivity_retention_percent', stability)
        self.assertIn('estimated_lifetime_hours', stability)
        
        # Should show some degradation after 1000 hours
        self.assertLess(stability['conductivity_retention_percent'], 100)


class TestFoulingModel(unittest.TestCase):
    """Test membrane fouling model."""
    
    def setUp(self):
        if not MEMBRANE_IMPORTS_AVAILABLE:
            self.skipTest("Membrane model modules not available")
        
        self.fouling_params = FoulingParameters()
        self.fouling_model = FoulingModel(self.fouling_params)
    
    def test_fouling_model_creation(self):
        """Test fouling model initialization."""
        self.assertIsInstance(self.fouling_model, FoulingModel)
        self.assertEqual(self.fouling_model.biofilm_thickness, 0.0)
        self.assertEqual(self.fouling_model.degradation_fraction, 0.0)
    
    def test_biofilm_growth(self):
        """Test biofilm growth calculation."""
        
        initial_thickness = 1e-6  # 1 μm
        self.fouling_model.biofilm_thickness = initial_thickness
        
        thickness_change = self.fouling_model.calculate_biofilm_growth(
            dt_hours=10.0,
            nutrient_conc=0.01,  # mol/L
            current_density=1000  # A/m²
        )
        
        # Should show growth
        self.assertGreater(self.fouling_model.biofilm_thickness, initial_thickness)
    
    def test_chemical_fouling(self):
        """Test chemical fouling (scaling)."""
        
        ion_concentrations = {
            'Ca2+': 0.01,    # mol/L
            'CO3--': 0.01    # mol/L (supersaturated)
        }
        
        thickness_change = self.fouling_model.calculate_chemical_fouling(
            dt_hours=24.0,
            ion_concentrations=ion_concentrations,
            temperature=323.15  # 50°C
        )
        
        # Should show scaling
        self.assertGreater(self.fouling_model.chemical_layer_thickness, 0.0)
    
    def test_particle_fouling(self):
        """Test particle deposition fouling."""
        
        thickness_change = self.fouling_model.calculate_particle_fouling(
            dt_hours=1.0,
            particle_concentration=0.001,  # kg/m³
            flow_velocity=0.1              # m/s
        )
        
        # Should show deposition
        self.assertGreaterEqual(self.fouling_model.particle_layer_thickness, 0.0)
    
    def test_total_resistance_calculation(self):
        """Test total resistance including fouling."""
        
        # Add some fouling
        self.fouling_model.biofilm_thickness = 10e-6     # 10 μm
        self.fouling_model.chemical_layer_thickness = 5e-6  # 5 μm
        self.fouling_model.degradation_fraction = 0.1    # 10% degradation
        
        base_resistance = 0.1  # Ω·m²
        resistance_data = self.fouling_model.calculate_total_resistance(base_resistance)
        
        # Total should be higher than base
        self.assertGreater(resistance_data['total_resistance'], base_resistance)
        
        # Should have individual components
        self.assertGreater(resistance_data['biofilm_resistance'], 0.0)
        self.assertGreater(resistance_data['chemical_fouling_resistance'], 0.0)
    
    def test_fouling_trajectory_prediction(self):
        """Test fouling trajectory prediction."""
        
        operating_conditions = {
            'temperature': 308.15,
            'ph': 7.5,
            'nutrient_concentration': 0.005,
            'current_density': 2000,
            'particle_concentration': 0.0005,
            'flow_velocity': 0.05
        }
        
        trajectory = self.fouling_model.predict_fouling_trajectory(
            simulation_hours=100.0,
            operating_conditions=operating_conditions,
            time_step=1.0
        )
        
        # Check data structure
        self.assertIn('time_hours', trajectory)
        self.assertIn('biofilm_thickness_um', trajectory)
        self.assertIn('total_resistance_ohm_m2', trajectory)
        
        # Should show progression
        biofilm_data = trajectory['biofilm_thickness_um']
        self.assertGreater(biofilm_data[-1], biofilm_data[0])  # Growth over time
    
    def test_cleaning_effectiveness(self):
        """Test cleaning method effectiveness."""
        
        # Add significant fouling
        self.fouling_model.biofilm_thickness = 50e-6     # 50 μm
        self.fouling_model.chemical_layer_thickness = 20e-6  # 20 μm
        self.fouling_model.particle_layer_thickness = 10e-6  # 10 μm
        
        cleaning_results = self.fouling_model.get_cleaning_effectiveness('chemical_cleaning')
        
        # Should show resistance reduction
        self.assertGreater(cleaning_results['resistance_reduction_percent'], 0)
        self.assertIn('cleaning_cost_per_m2', cleaning_results)
        self.assertIn('downtime_hours', cleaning_results)
    
    def test_fouling_status(self):
        """Test fouling status assessment."""
        
        # Add moderate fouling
        self.fouling_model.biofilm_thickness = 25e-6  # 25 μm
        self.fouling_model.operating_time = 500.0     # 500 hours
        
        status = self.fouling_model.get_fouling_status()
        
        # Check status fields
        self.assertIn('fouling_severity', status)
        self.assertIn('dominant_fouling_type', status)
        self.assertIn('cleaning_recommended', status)
        
        # Should be moderate severity
        self.assertIn(status['fouling_severity'], ['Low', 'Moderate', 'High', 'Severe'])


class TestMembraneIntegration(unittest.TestCase):
    """Test integration between different membrane models."""
    
    def setUp(self):
        if not MEMBRANE_IMPORTS_AVAILABLE:
            self.skipTest("Membrane model modules not available")
    
    def test_membrane_comparison(self):
        """Compare different membrane types."""
        
        # Create different membranes
        nafion = create_nafion_membrane(thickness_um=100, area_cm2=1.0)
        aem = create_aem_membrane(thickness_um=100, area_cm2=1.0)
        bipolar = create_bipolar_membrane(thickness_um=200, area_cm2=1.0)
        
        # Compare conductivities
        nafion_cond = nafion.calculate_ionic_conductivity()
        aem_cond = aem.calculate_ionic_conductivity()
        bipolar_cond = bipolar.calculate_ionic_conductivity()
        
        # Nafion should typically have highest conductivity
        self.assertGreater(nafion_cond, aem_cond)
        
        # All should be positive
        self.assertGreater(nafion_cond, 0)
        self.assertGreater(aem_cond, 0)
        self.assertGreater(bipolar_cond, 0)
    
    def test_membrane_with_fouling(self):
        """Test membrane performance with fouling."""
        
        membrane = create_nafion_membrane()
        fouling = FoulingModel(FoulingParameters())
        
        # Initial resistance
        initial_resistance = membrane.calculate_membrane_resistance()
        
        # Add fouling
        fouling.biofilm_thickness = 20e-6  # 20 μm biofilm
        
        # Calculate fouled resistance
        fouling_data = fouling.calculate_total_resistance(initial_resistance)
        
        # Should be higher
        self.assertGreater(fouling_data['total_resistance'], initial_resistance)
    
    def test_ceramic_membrane(self):
        """Test ceramic membrane functionality."""
        
        ceramic = create_ceramic_membrane(
            ceramic_type="YSZ",
            thickness_um=500,
            area_cm2=1.0
        )
        
        # Should have very low conductivity at room temperature
        low_temp_cond = ceramic.calculate_ionic_conductivity(temperature=298.15)
        
        # But higher at elevated temperature
        high_temp_cond = ceramic.calculate_ionic_conductivity(temperature=1073.15)  # 800°C
        
        self.assertGreater(high_temp_cond, low_temp_cond)


def run_membrane_tests():
    """Run all membrane model tests."""
    print("🧪 Running Membrane Model Tests")
    print("=" * 50)
    
    if not MEMBRANE_IMPORTS_AVAILABLE:
        print("❌ Cannot run tests - membrane models not available")
        return False
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBaseMembraneModel,
        TestProtonExchangeMembrane,
        TestAnionExchangeMembrane,
        TestFoulingModel,
        TestMembraneIntegration
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 Membrane Model Test Summary")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed.'}")
    
    return success


if __name__ == "__main__":
    success = run_membrane_tests()
    sys.exit(0 if success else 1)