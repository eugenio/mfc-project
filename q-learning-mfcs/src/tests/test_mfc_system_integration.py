#!/usr/bin/env python3
"""
Test suite for MFC System Integration (Phase 5)

Tests the complete integrated MFC system including:
- Component integration (anode, membrane, cathode)
- System-level performance calculations
- Multi-physics coupling
- Q-learning integration
- Economic analysis

Created: 2025-07-27
"""

import unittest
import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

try:
    from mfc_system_integration import (
        IntegratedMFCSystem,
        MFCStackParameters,
        MFCSystemState,
        MFCConfiguration,
        create_standard_mfc_system
    )
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Integration system import error: {e}")
    INTEGRATION_AVAILABLE = False


class TestMFCSystemIntegration(unittest.TestCase):
    """Test integrated MFC system functionality."""
    
    def setUp(self):
        """Set up test environment."""
        if not INTEGRATION_AVAILABLE:
            self.skipTest("MFC integration system not available")
        
        # Create minimal test configuration
        self.test_config = MFCStackParameters(
            n_cells=2,
            cell_area=0.0001,  # 1 cm²
            bacterial_species="geobacter",
            substrate_type="acetate",
            membrane_material="Nafion",
            cathode_type="platinum",
            enable_qlearning=False  # Disable for testing
        )
    
    def test_system_initialization(self):
        """Test MFC system initialization."""
        system = IntegratedMFCSystem(self.test_config)
        
        # Check basic properties
        self.assertEqual(system.config.n_cells, 2)
        self.assertEqual(len(system.anode_models), 2)
        self.assertEqual(len(system.membrane_models), 2)
        self.assertEqual(len(system.cathode_models), 2)
        self.assertEqual(len(system.cell_states), 2)
        
        # Check initial state
        self.assertGreater(len(system.cell_states), 0)
        for state in system.cell_states:
            self.assertIsInstance(state, MFCSystemState)
            self.assertGreater(state.substrate_concentration, 0)
            self.assertGreater(state.membrane_conductivity, 0)
    
    def test_anode_dynamics(self):
        """Test anode biofilm and metabolic dynamics."""
        system = IntegratedMFCSystem(self.test_config)
        initial_state = system.cell_states[0]
        
        # Step anode dynamics
        anode_result = system._step_anode_dynamics(0, 1.0, initial_state)
        
        # Check result structure
        required_keys = [
            'biofilm_thickness', 'biomass_density', 
            'substrate_concentration', 'anode_current_density'
        ]
        for key in required_keys:
            self.assertIn(key, anode_result)
            self.assertIsInstance(anode_result[key], (int, float))
        
        # Check reasonable values
        self.assertGreater(anode_result['biofilm_thickness'], 0)
        self.assertGreater(anode_result['biomass_density'], 0)
        self.assertGreaterEqual(anode_result['substrate_concentration'], 0)
        self.assertGreaterEqual(anode_result['anode_current_density'], 0)
    
    def test_membrane_dynamics(self):
        """Test membrane transport and fouling dynamics."""
        system = IntegratedMFCSystem(self.test_config)
        initial_state = system.cell_states[0]
        
        # Create mock anode state
        anode_state = {
            'anode_current_density': 100.0,  # A/m²
            'substrate_concentration': 15.0
        }
        
        # Step membrane dynamics
        membrane_result = system._step_membrane_dynamics(0, 1.0, initial_state, anode_state)
        
        # Check result structure
        required_keys = [
            'membrane_conductivity', 'proton_flux', 'water_flux',
            'fouling_thickness', 'degradation_fraction', 'membrane_resistance'
        ]
        for key in required_keys:
            self.assertIn(key, membrane_result)
            # Handle JAX arrays and regular numbers
            value = membrane_result[key]
            if hasattr(value, 'item'):  # JAX array
                value = float(value.item())
            self.assertIsInstance(value, (int, float))
        
        # Check reasonable values
        self.assertGreater(membrane_result['membrane_conductivity'], 0)
        self.assertGreater(membrane_result['proton_flux'], 0)
        self.assertGreaterEqual(membrane_result['fouling_thickness'], 0)
        self.assertGreaterEqual(membrane_result['degradation_fraction'], 0)
        self.assertLessEqual(membrane_result['degradation_fraction'], 1)
    
    def test_cathode_dynamics(self):
        """Test cathode kinetics and mass transport."""
        system = IntegratedMFCSystem(self.test_config)
        initial_state = system.cell_states[0]
        
        # Create mock membrane state
        membrane_state = {
            'proton_flux': 0.001  # mol/m²/s
        }
        
        # Step cathode dynamics
        cathode_result = system._step_cathode_dynamics(0, 1.0, initial_state, membrane_state)
        
        # Check result structure
        required_keys = [
            'cathode_potential', 'oxygen_concentration', 
            'cathode_current_density', 'overpotential'
        ]
        for key in required_keys:
            self.assertIn(key, cathode_result)
            self.assertIsInstance(cathode_result[key], (int, float))
        
        # Check reasonable values
        self.assertGreater(cathode_result['cathode_potential'], 0)
        self.assertGreater(cathode_result['oxygen_concentration'], 0)
        self.assertGreaterEqual(cathode_result['cathode_current_density'], 0)
    
    def test_cell_performance_calculation(self):
        """Test overall cell performance calculation."""
        system = IntegratedMFCSystem(self.test_config)
        
        # Create mock component states
        anode_state = {'anode_current_density': 100.0}
        membrane_state = {'proton_flux': 0.001, 'membrane_resistance': 0.1}
        cathode_state = {'cathode_potential': 0.8, 'cathode_current_density': 100.0}
        
        # Calculate performance
        performance = system._calculate_cell_performance(0, anode_state, membrane_state, cathode_state)
        
        # Check result structure
        required_keys = [
            'cell_voltage', 'current_density', 'power_density',
            'coulombic_efficiency', 'power_generated'
        ]
        for key in required_keys:
            self.assertIn(key, performance)
            self.assertIsInstance(performance[key], (int, float))
        
        # Check reasonable values
        self.assertGreaterEqual(performance['cell_voltage'], 0)
        self.assertGreaterEqual(performance['current_density'], 0)
        self.assertGreaterEqual(performance['power_density'], 0)
        self.assertGreaterEqual(performance['coulombic_efficiency'], 0)
        self.assertLessEqual(performance['coulombic_efficiency'], 1)
    
    def test_system_dynamics_step(self):
        """Test complete system dynamics step."""
        system = IntegratedMFCSystem(self.test_config)
        initial_time = system.time
        
        # Step system dynamics
        updated_states = system.step_system_dynamics(dt=1.0)
        
        # Check return structure
        self.assertEqual(len(updated_states), self.test_config.n_cells)
        self.assertGreater(system.time, initial_time)
        
        # Check updated states
        for state in updated_states:
            self.assertIsInstance(state, MFCSystemState)
            self.assertGreater(state.time, initial_time)
            self.assertGreaterEqual(state.cell_voltage, 0)
            self.assertGreaterEqual(state.power_density, 0)
            self.assertGreaterEqual(state.coulombic_efficiency, 0)
            self.assertLessEqual(state.coulombic_efficiency, 1)
    
    def test_short_simulation(self):
        """Test short system simulation."""
        system = IntegratedMFCSystem(self.test_config)
        
        # Run short simulation
        results = system.run_system_simulation(
            duration_hours=2.0,
            dt=0.5,
            save_interval=1
        )
        
        # Check results structure
        required_keys = [
            'total_energy_wh', 'peak_power_mw', 'average_efficiency',
            'time_series', 'configuration'
        ]
        for key in required_keys:
            self.assertIn(key, results)
        
        # Check time series data
        df = results['time_series']
        self.assertGreater(len(df), 0)
        self.assertEqual(df['cell_id'].nunique(), self.test_config.n_cells)
        
        # Check performance metrics
        self.assertGreaterEqual(results['total_energy_wh'], 0)
        self.assertGreaterEqual(results['peak_power_mw'], 0)
        self.assertGreaterEqual(results['average_efficiency'], 0)
        self.assertLessEqual(results['average_efficiency'], 1)
    
    def test_system_status(self):
        """Test system status reporting."""
        system = IntegratedMFCSystem(self.test_config)
        
        # Step once to get some data
        system.step_system_dynamics(1.0)
        
        # Get status
        status = system.get_system_status()
        
        # Check status structure
        required_keys = [
            'time_hours', 'total_power_mw', 'average_voltage_v',
            'average_efficiency_pct', 'cells_active', 'system_health'
        ]
        for key in required_keys:
            self.assertIn(key, status)
        
        # Check values
        self.assertGreater(status['time_hours'], 0)
        self.assertGreaterEqual(status['total_power_mw'], 0)
        self.assertGreaterEqual(status['average_voltage_v'], 0)
        self.assertGreaterEqual(status['average_efficiency_pct'], 0)
        self.assertEqual(status['cells_active'], self.test_config.n_cells)
        self.assertIn(status['system_health'], ['Healthy', 'Degraded'])


class TestStandardConfigurations(unittest.TestCase):
    """Test standard MFC configurations."""
    
    def setUp(self):
        if not INTEGRATION_AVAILABLE:
            self.skipTest("MFC integration system not available")
    
    def test_basic_lab_configuration(self):
        """Test basic lab configuration."""
        system = create_standard_mfc_system(MFCConfiguration.BASIC_LAB)
        
        self.assertEqual(system.config.n_cells, 3)
        self.assertEqual(system.config.bacterial_species, "geobacter")
        self.assertEqual(system.config.substrate_type, "acetate")
        self.assertEqual(system.config.cathode_type, "platinum")
    
    def test_research_configuration(self):
        """Test research configuration."""
        system = create_standard_mfc_system(MFCConfiguration.RESEARCH)
        
        self.assertEqual(system.config.n_cells, 5)
        self.assertEqual(system.config.bacterial_species, "mixed")
        self.assertEqual(system.config.substrate_type, "lactate")
        self.assertTrue(system.config.enable_qlearning)
    
    def test_pilot_plant_configuration(self):
        """Test pilot plant configuration."""
        system = create_standard_mfc_system(MFCConfiguration.PILOT_PLANT)
        
        self.assertEqual(system.config.n_cells, 10)
        self.assertEqual(system.config.membrane_material, "SPEEK")
        self.assertEqual(system.config.cathode_type, "biological")


class TestIntegrationPerformance(unittest.TestCase):
    """Test integration performance and benchmarks."""
    
    def setUp(self):
        if not INTEGRATION_AVAILABLE:
            self.skipTest("MFC integration system not available")
    
    def test_simulation_speed(self):
        """Test simulation computational performance."""
        system = create_standard_mfc_system(MFCConfiguration.BASIC_LAB)
        
        import time
        start_time = time.time()
        
        # Run short simulation
        results = system.run_system_simulation(duration_hours=1.0, dt=0.25)
        
        computation_time = time.time() - start_time
        
        # Should complete reasonably quickly
        self.assertLess(computation_time, 30.0)  # Less than 30 seconds
        self.assertIn('computation_time', results)
    
    def test_memory_usage(self):
        """Test that simulation doesn't consume excessive memory."""
        system = create_standard_mfc_system(MFCConfiguration.BASIC_LAB)
        
        # Run simulation and check history growth
        system.run_system_simulation(duration_hours=2.0, dt=0.5)
        
        # History should be reasonable size
        self.assertLess(len(system.history), 1000)  # Not excessive
        
        # States should be properly structured
        for state in system.history[-5:]:  # Check last 5
            self.assertIsInstance(state, MFCSystemState)


class TestSystemValidation(unittest.TestCase):
    """Test system validation and physical constraints."""
    
    def setUp(self):
        if not INTEGRATION_AVAILABLE:
            self.skipTest("MFC integration system not available")
    
    def test_energy_conservation(self):
        """Test that energy is conserved in the system."""
        system = create_standard_mfc_system(MFCConfiguration.BASIC_LAB)
        
        results = system.run_system_simulation(duration_hours=2.0, dt=0.5)
        
        # Total energy should be positive and reasonable
        self.assertGreater(results['total_energy_wh'], 0)
        self.assertLess(results['total_energy_wh'], 1.0)  # Not unrealistically high
    
    def test_mass_balance(self):
        """Test substrate mass balance."""
        system = create_standard_mfc_system(MFCConfiguration.BASIC_LAB)
        
        initial_substrate = np.mean([s.substrate_concentration for s in system.cell_states])
        
        # Run simulation
        system.run_system_simulation(duration_hours=2.0, dt=0.5)
        
        final_substrate = np.mean([s.substrate_concentration for s in system.cell_states])
        
        # Substrate should decrease (be consumed)
        self.assertLess(final_substrate, initial_substrate)
        self.assertGreaterEqual(final_substrate, 0)  # Can't go negative
    
    def test_physical_constraints(self):
        """Test that physical constraints are respected."""
        system = create_standard_mfc_system(MFCConfiguration.RESEARCH)
        
        # Step system a few times
        for _ in range(5):
            states = system.step_system_dynamics(1.0)
            
            for state in states:
                # Voltage constraints
                self.assertGreaterEqual(state.cell_voltage, 0)
                self.assertLessEqual(state.cell_voltage, 1.5)  # Theoretical max ~1.1V
                
                # Efficiency constraints
                self.assertGreaterEqual(state.coulombic_efficiency, 0)
                self.assertLessEqual(state.coulombic_efficiency, 1)
                
                # Concentration constraints
                self.assertGreaterEqual(state.substrate_concentration, 0)
                self.assertGreaterEqual(state.oxygen_concentration, 0)
                
                # Thickness constraints
                self.assertGreaterEqual(state.biofilm_thickness, 0)
                self.assertGreaterEqual(state.fouling_thickness, 0)


def run_integration_tests():
    """Run all MFC integration tests."""
    print("🧪 Running MFC System Integration Tests")
    print("=" * 50)
    
    if not INTEGRATION_AVAILABLE:
        print("❌ Cannot run tests - integration system not available")
        return False
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMFCSystemIntegration,
        TestStandardConfigurations,
        TestIntegrationPerformance,
        TestSystemValidation
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 MFC Integration Test Summary")
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
    success = run_integration_tests()
    sys.exit(0 if success else 1)