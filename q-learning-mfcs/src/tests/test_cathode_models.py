#!/usr/bin/env python3
"""
Test suite for cathode models

Tests the base cathode model, platinum cathode, and biological cathode models
with various operating conditions and parameter combinations.

Created: 2025-07-26
"""

import unittest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

try:
    from cathode_models.base_cathode import BaseCathodeModel, CathodeParameters, ButlerVolmerKinetics
    from cathode_models.platinum_cathode import PlatinumCathodeModel, create_platinum_cathode
    from cathode_models.biological_cathode import BiologicalCathodeModel, create_biological_cathode
    CATHODE_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Cathode model import error: {e}")
    CATHODE_IMPORTS_AVAILABLE = False


class TestBaseCathodeModel(unittest.TestCase):
    """Test base cathode model functionality."""
    
    def setUp(self):
        """Set up test environment."""
        if not CATHODE_IMPORTS_AVAILABLE:
            self.skipTest("Cathode model modules not available")
        
        self.params = CathodeParameters(
            area_m2=1e-4,  # 1 cm²
            temperature_K=298.15,  # 25°C
            oxygen_concentration=8e-3  # 8 mM
        )
    
    def test_equilibrium_potential_calculation(self):
        """Test Nernst equation for O₂/H₂O equilibrium potential."""
        
        class TestCathode(BaseCathodeModel):
            def _setup_kinetic_parameters(self):
                pass
            def calculate_current_density(self, overpotential, oxygen_conc=None):
                return 0.0
        
        cathode = TestCathode(self.params)
        
        # Test at standard conditions (pH=7, 8mM O₂)
        E_eq = cathode.calculate_equilibrium_potential()
        
        # Expected: ~0.816 V vs SHE at pH=7
        self.assertAlmostEqual(E_eq, 0.816, places=2)
        
        # Test pH dependency
        E_eq_ph0 = cathode.calculate_equilibrium_potential(ph=0)
        E_eq_ph14 = cathode.calculate_equilibrium_potential(ph=14)
        
        # Should decrease by ~0.059V per pH unit
        expected_diff = 14 * 0.059
        actual_diff = E_eq_ph0 - E_eq_ph14
        self.assertAlmostEqual(actual_diff, expected_diff, places=1)
    
    def test_overpotential_calculation(self):
        """Test overpotential calculation."""
        
        class TestCathode(BaseCathodeModel):
            def _setup_kinetic_parameters(self):
                pass
            def calculate_current_density(self, overpotential, oxygen_conc=None):
                return 0.0
        
        cathode = TestCathode(self.params)
        
        cathode_potential = 0.6  # V vs SHE
        overpotential = cathode.calculate_overpotential(cathode_potential)
        
        # Overpotential should be positive (E_eq - E_cathode)
        self.assertGreater(overpotential, 0)
        self.assertAlmostEqual(overpotential, 0.216, places=2)
    
    def test_temperature_dependency(self):
        """Test temperature effects on equilibrium potential."""
        
        class TestCathode(BaseCathodeModel):
            def _setup_kinetic_parameters(self):
                pass
            def calculate_current_density(self, overpotential, oxygen_conc=None):
                return 0.0
        
        cathode = TestCathode(self.params)
        
        E_25C = cathode.calculate_equilibrium_potential()
        
        # Change temperature to 50°C
        cathode.update_temperature(323.15)
        E_50C = cathode.calculate_equilibrium_potential()
        
        # Equilibrium potential should change with temperature
        self.assertNotEqual(E_25C, E_50C)
        
        # For 25K temperature increase, expect ~20-60 mV change
        temp_diff_mV = abs(E_25C - E_50C) * 1000  # Convert to mV
        self.assertGreater(temp_diff_mV, 20)  # Should show significant temperature effect
        self.assertLess(temp_diff_mV, 80)     # But not unreasonably large


class TestButlerVolmerKinetics(unittest.TestCase):
    """Test Butler-Volmer kinetics utility class."""
    
    def setUp(self):
        if not CATHODE_IMPORTS_AVAILABLE:
            self.skipTest("Cathode model modules not available")
    
    def test_butler_volmer_calculation(self):
        """Test Butler-Volmer current density calculation."""
        
        # Standard parameters
        i0 = 1e-3  # A/m²
        alpha = 0.5
        overpotential = 0.1  # V
        temperature = 298.15  # K
        
        current_density = ButlerVolmerKinetics.calculate_current_density(
            exchange_current_density=i0,
            transfer_coefficient=alpha,
            overpotential=overpotential,
            temperature_K=temperature
        )
        
        # Should give positive current for positive overpotential
        self.assertGreater(current_density, 0)
        self.assertLess(current_density, 1.0)  # Reasonable magnitude
    
    def test_tafel_calculation(self):
        """Test Tafel equation calculation."""
        
        i0 = 1e-3  # A/m²
        tafel_slope = 0.060  # V/decade
        overpotential = 0.2  # V
        
        current_density = ButlerVolmerKinetics.calculate_tafel_current(
            exchange_current_density=i0,
            tafel_slope=tafel_slope,
            overpotential=overpotential
        )
        
        # Should give reasonable current density for 200 mV overpotential
        # Expected: i = 1e-3 * 10^(0.2/0.06) = 1e-3 * 2154 = 2.15 A/m²
        self.assertGreater(current_density, i0)
        self.assertLess(current_density, 5.0)  # Updated to realistic threshold
        self.assertAlmostEqual(current_density, 2.154, places=2)  # Verify expected value
    
    def test_zero_overpotential(self):
        """Test behavior at zero overpotential."""
        
        current = ButlerVolmerKinetics.calculate_tafel_current(1e-3, 0.060, 0.0)
        self.assertEqual(current, 0.0)


class TestPlatinumCathodeModel(unittest.TestCase):
    """Test platinum cathode model."""
    
    def setUp(self):
        if not CATHODE_IMPORTS_AVAILABLE:
            self.skipTest("Cathode model modules not available")
        
        self.cathode = create_platinum_cathode(
            area_cm2=1.0,
            temperature_C=25.0,
            oxygen_mg_L=8.0
        )
    
    def test_platinum_cathode_creation(self):
        """Test platinum cathode creation and basic properties."""
        
        self.assertIsInstance(self.cathode, PlatinumCathodeModel)
        self.assertEqual(self.cathode.area_m2, 1e-4)  # 1 cm²
        self.assertEqual(self.cathode.temperature_K, 298.15)
        
        # Check that kinetic parameters are set
        self.assertGreater(self.cathode.exchange_current_density, 0)
        self.assertEqual(self.cathode.transfer_coefficient, 0.5)
    
    def test_current_density_calculation(self):
        """Test current density calculation for platinum cathode."""
        
        overpotential = 0.1  # V
        current_density = self.cathode.calculate_current_density(overpotential)
        
        # Should give reasonable current density
        self.assertGreater(current_density, 0)
        self.assertLess(current_density, 10.0)  # A/m² - reasonable range
        
        # Higher overpotential should give higher current
        higher_current = self.cathode.calculate_current_density(0.2)
        self.assertGreater(higher_current, current_density)
    
    def test_dual_tafel_regions(self):
        """Test transition between low and high overpotential regions."""
        
        # Low overpotential (should use Butler-Volmer)
        low_eta = 0.05  # V
        low_current = self.cathode.calculate_current_density(low_eta)
        
        # High overpotential (should use Tafel)
        high_eta = 0.15  # V
        high_current = self.cathode.calculate_current_density(high_eta)
        
        # High overpotential should give higher current
        self.assertGreater(high_current, low_current)
        
        # Test at transition point
        transition_eta = self.cathode.overpotential_transition
        transition_current = self.cathode.calculate_current_density(transition_eta)
        
        self.assertGreater(transition_current, low_current)
        self.assertLess(transition_current, high_current)
    
    def test_temperature_effects(self):
        """Test temperature effects on platinum cathode performance."""
        
        overpotential = 0.1  # V
        
        # Current at 25°C
        current_25C = self.cathode.calculate_current_density(overpotential)
        
        # Change to 50°C
        self.cathode.update_temperature(323.15)
        current_50C = self.cathode.calculate_current_density(overpotential)
        
        # Higher temperature should generally increase current (Arrhenius)
        self.assertGreater(current_50C, current_25C)
    
    def test_oxygen_concentration_effects(self):
        """Test oxygen concentration effects."""
        
        overpotential = 0.1  # V
        
        # Standard oxygen concentration (explicitly pass it)
        standard_current = self.cathode.calculate_current_density(overpotential, oxygen_conc=8e-3)
        
        # Higher oxygen concentration
        high_o2_current = self.cathode.calculate_current_density(
            overpotential, 
            oxygen_conc=16e-3  # 16 mM
        )
        
        # Lower oxygen concentration
        low_o2_current = self.cathode.calculate_current_density(
            overpotential,
            oxygen_conc=2e-3  # 2 mM
        )
        
        # Higher O₂ should give higher current
        self.assertGreater(high_o2_current, standard_current)
        self.assertGreater(standard_current, low_o2_current)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        
        metrics = self.cathode.calculate_performance_metrics(0.1)
        
        # Check required metrics are present
        required_metrics = [
            'current_density_A_m2', 'power_loss_W', 'voltage_efficiency_percent',
            'overpotential_V', 'equilibrium_potential_V'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertGreater(metrics[metric], 0)
    
    def test_cost_analysis(self):
        """Test cost analysis functionality."""
        
        cost = self.cathode.estimate_cost_per_area()
        
        # Should be positive and reasonable ($/m²)
        self.assertGreater(cost, 0)
        self.assertLess(cost, 1e6)  # Less than $1M/m²
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison functionality."""
        
        comparison = self.cathode.compare_to_benchmark()
        
        # Check structure
        self.assertIn('performance_vs_benchmark', comparison)
        self.assertIn('operating_conditions', comparison)
        self.assertIn('cost_analysis', comparison)
        
        # Check performance ratios
        perf = comparison['performance_vs_benchmark']
        self.assertIn('power_density_ratio', perf)
        self.assertIn('exchange_current_ratio', perf)


class TestBiologicalCathodeModel(unittest.TestCase):
    """Test biological cathode model."""
    
    def setUp(self):
        if not CATHODE_IMPORTS_AVAILABLE:
            self.skipTest("Cathode model modules not available")
        
        self.biocathode = create_biological_cathode(
            area_cm2=1.0,
            temperature_C=30.0,
            ph=7.0,
            oxygen_mg_L=8.0
        )
    
    def test_biological_cathode_creation(self):
        """Test biological cathode creation."""
        
        self.assertIsInstance(self.biocathode, BiologicalCathodeModel)
        self.assertEqual(self.biocathode.area_m2, 1e-4)
        self.assertEqual(self.biocathode.temperature_K, 303.15)  # 30°C
        
        # Check biofilm initialization
        self.assertGreater(self.biocathode.biofilm_thickness, 0)
        self.assertGreater(self.biocathode.biomass_density, 0)
    
    def test_monod_growth_rate(self):
        """Test Monod kinetics for microbial growth."""
        
        oxygen_conc = 8e-3  # mol/L
        electrode_potential = 0.4  # V
        
        growth_rate = self.biocathode.calculate_monod_growth_rate(
            oxygen_conc, electrode_potential
        )
        
        # Should give positive growth rate under favorable conditions
        self.assertGreater(growth_rate, 0)
        self.assertLess(growth_rate, self.biocathode.bio_params.max_growth_rate)
    
    def test_biofilm_dynamics_update(self):
        """Test biofilm thickness and biomass updates."""
        
        initial_thickness = self.biocathode.biofilm_thickness
        
        # Update for 24 hours under favorable conditions
        self.biocathode.update_biofilm_dynamics(
            dt_hours=24.0,
            oxygen_conc=8e-3,
            electrode_potential=0.4
        )
        
        # Biofilm should grow under favorable conditions
        self.assertGreater(self.biocathode.biofilm_thickness, initial_thickness)
        
        # Biomass density may increase or decrease depending on growth vs decay
        final_biomass = self.biocathode.biomass_density
        self.assertGreater(final_biomass, 0)  # Should remain positive
    
    def test_biofilm_current_density(self):
        """Test current density calculation with biofilm effects."""
        
        overpotential = 0.1  # V
        current_density = self.biocathode.calculate_current_density(overpotential)
        
        # Should give positive current
        self.assertGreater(current_density, 0)
        
        # Test with different biofilm thicknesses
        original_thickness = self.biocathode.biofilm_thickness
        
        # Thicker biofilm
        self.biocathode.biofilm_thickness = 100e-6  # 100 μm
        self.biocathode._setup_biofilm_parameters()
        thick_current = self.biocathode.calculate_current_density(overpotential)
        
        # Restore original thickness
        self.biocathode.biofilm_thickness = original_thickness
        self.biocathode._setup_biofilm_parameters()
        
        # Thicker biofilm might give different current due to resistance/activity trade-off
        self.assertNotEqual(thick_current, current_density)
    
    def test_environmental_factors(self):
        """Test pH and temperature effects on biological cathode."""
        
        # Test pH effects
        original_factor = self.biocathode.environmental_factor
        
        # Change pH away from optimum
        self.biocathode.params.ph = 5.0  # Acidic
        self.biocathode._setup_kinetic_parameters()
        acidic_factor = self.biocathode.environmental_factor
        
        # Should reduce environmental factor
        self.assertLess(acidic_factor, original_factor)
        
        # Test temperature effects
        self.biocathode.params.ph = 7.0  # Restore optimal pH
        self.biocathode.update_temperature(283.15)  # 10°C (cold)
        cold_factor = self.biocathode.environmental_factor
        
        # Cold temperature should reduce factor
        self.assertLess(cold_factor, original_factor)
    
    def test_biofilm_performance_metrics(self):
        """Test biofilm-specific performance metrics."""
        
        metrics = self.biocathode.calculate_biofilm_performance_metrics(0.1)
        
        required_metrics = [
            'biofilm_thickness_um', 'biomass_density_kg_m3', 'current_per_biomass_A_kg',
            'growth_rate_h_inv', 'biofilm_resistance_ohm_m2'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_long_term_prediction(self):
        """Test long-term biofilm development prediction."""
        
        prediction = self.biocathode.predict_long_term_performance(
            simulation_days=7,  # 1 week
            oxygen_conc=8e-3,
            electrode_potential=0.4
        )
        
        # Check data structure
        self.assertIn('time_hours', prediction)
        self.assertIn('biofilm_thickness_um', prediction)
        self.assertIn('current_density_A_m2', prediction)
        
        # Check data length
        time_points = len(prediction['time_hours'])
        self.assertEqual(len(prediction['biofilm_thickness_um']), time_points)
        self.assertEqual(len(prediction['current_density_A_m2']), time_points)
        
        # Check final values
        self.assertGreater(prediction['final_thickness_um'], 0)
        # Allow for very small positive current density instead of zero
        self.assertGreaterEqual(prediction['final_current_density_A_m2'], 0)
        # Check that at least some current was generated during the simulation
        self.assertGreater(prediction['average_current_density_A_m2'], 0)
    
    def test_economic_analysis(self):
        """Test economic analysis for biological cathode."""
        
        economics = self.biocathode.estimate_economic_analysis()
        
        required_fields = [
            'inoculation_cost_per_m2', 'annual_maintenance_cost_per_m2',
            'total_lifetime_cost_per_m2', 'cost_per_kW'
        ]
        
        for field in required_fields:
            self.assertIn(field, economics)
            self.assertGreater(economics[field], 0)


class TestCathodeModelComparison(unittest.TestCase):
    """Test comparison between different cathode models."""
    
    def setUp(self):
        if not CATHODE_IMPORTS_AVAILABLE:
            self.skipTest("Cathode model modules not available")
        
        # Create both cathode types with same conditions
        self.pt_cathode = create_platinum_cathode(
            area_cm2=1.0, temperature_C=25.0, oxygen_mg_L=8.0
        )
        
        self.bio_cathode = create_biological_cathode(
            area_cm2=1.0, temperature_C=25.0, ph=7.0, oxygen_mg_L=8.0
        )
    
    def test_performance_comparison(self):
        """Compare performance between platinum and biological cathodes."""
        
        overpotential = 0.1  # V
        
        # Calculate performance for both
        pt_current = self.pt_cathode.calculate_current_density(overpotential)
        bio_current = self.bio_cathode.calculate_current_density(overpotential)
        
        # Both should give positive current
        self.assertGreater(pt_current, 0)
        self.assertGreater(bio_current, 0)
        
        # Platinum typically has higher exchange current initially
        self.assertGreater(self.pt_cathode.exchange_current_density,
                          self.bio_cathode.exchange_current_density)
    
    def test_cost_comparison(self):
        """Compare costs between cathode types."""
        
        pt_cost = self.pt_cathode.estimate_cost_per_area()
        bio_economics = self.bio_cathode.estimate_economic_analysis()
        bio_cost = bio_economics['total_lifetime_cost_per_m2']
        
        # Both should have positive costs
        self.assertGreater(pt_cost, 0)
        self.assertGreater(bio_cost, 0)
        
        # Platinum is typically more expensive initially
        self.assertGreater(pt_cost, bio_cost)
    
    def test_parameter_access(self):
        """Test parameter access for both cathode types."""
        
        pt_params = self.pt_cathode.get_all_parameters()
        bio_params = self.bio_cathode.get_all_parameters()
        
        # Both should have comprehensive parameter sets
        self.assertIsInstance(pt_params, dict)
        self.assertIsInstance(bio_params, dict)
        
        # Should have different specific parameters
        self.assertIn('platinum_kinetic_parameters', pt_params)
        self.assertIn('microbial_kinetics', bio_params)


def run_cathode_tests():
    """Run all cathode model tests."""
    print("🧪 Running Cathode Model Tests")
    print("=" * 50)
    
    if not CATHODE_IMPORTS_AVAILABLE:
        print("❌ Cannot run tests - cathode models not available")
        return False
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBaseCathodeModel,
        TestButlerVolmerKinetics,
        TestPlatinumCathodeModel,
        TestBiologicalCathodeModel,
        TestCathodeModelComparison
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 Cathode Model Test Summary")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed.'}")
    
    return success


if __name__ == "__main__":
    success = run_cathode_tests()
    sys.exit(0 if success else 1)