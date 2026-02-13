#!/usr/bin/env python3
"""
Test suite for Hydraulic System Models (Phase 6)

Tests the hydraulic system including:
- Pump models (peristaltic, centrifugal, diaphragm)
- Flow calculations and pressure dynamics
- Hydraulic network modeling
- Power consumption and cost analysis
- Control system functionality

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
    from hydraulic_system.hydraulic_models import (
        PumpType, FluidType, PumpParameters, CellGeometry, PipingNetwork,
        BasePump, PeristalticPump, CentrifugalPump, DiaphragmPump,
        FlowCalculator, HydraulicNetwork, HydraulicController,
        create_standard_hydraulic_system, calculate_hydraulic_costs,
        WATER_DENSITY, WATER_VISCOSITY, GRAVITY, ATMOSPHERIC_PRESSURE
    )
    HYDRAULIC_AVAILABLE = True
except ImportError as e:
    print(f"Hydraulic system import error: {e}")
    HYDRAULIC_AVAILABLE = False


class TestPumpModels(unittest.TestCase):
    """Test individual pump models."""
    
    def setUp(self):
        if not HYDRAULIC_AVAILABLE:
            self.skipTest("Hydraulic system not available")
        
        self.pump_params = PumpParameters(
            max_flow_rate=100.0,
            max_pressure=50000.0,
            efficiency=0.7,
            power_rating=5.0,
            cost=200.0
        )
    
    def test_peristaltic_pump_creation(self):
        """Test peristaltic pump creation and basic properties."""
        pump = PeristalticPump("test_pump", self.pump_params)
        
        self.assertEqual(pump.pump_id, "test_pump")
        self.assertEqual(pump.pump_type, PumpType.PERISTALTIC)
        self.assertFalse(pump.is_running)
        self.assertEqual(pump.current_flow_rate, 0.0)
        self.assertEqual(pump.current_power, 0.0)
    
    def test_centrifugal_pump_creation(self):
        """Test centrifugal pump creation."""
        pump = CentrifugalPump("centrifugal_test", self.pump_params)
        
        self.assertEqual(pump.pump_type, PumpType.CENTRIFUGAL)
        self.assertIsInstance(pump, BasePump)
    
    def test_diaphragm_pump_creation(self):
        """Test diaphragm pump creation."""
        pump = DiaphragmPump("diaphragm_test", self.pump_params)
        
        self.assertEqual(pump.pump_type, PumpType.DIAPHRAGM)
        self.assertIsInstance(pump, BasePump)
    
    def test_pump_start_stop(self):
        """Test pump start/stop functionality."""
        pump = PeristalticPump("test_pump", self.pump_params)
        
        # Initially stopped
        self.assertFalse(pump.is_running)
        
        # Start pump
        pump.start_pump()
        self.assertTrue(pump.is_running)
        
        # Stop pump
        pump.stop_pump()
        self.assertFalse(pump.is_running)
        self.assertEqual(pump.current_flow_rate, 0.0)
        self.assertEqual(pump.current_power, 0.0)
    
    def test_peristaltic_flow_calculation(self):
        """Test peristaltic pump flow rate calculation."""
        pump = PeristalticPump("test_pump", self.pump_params)
        
        # Normal pressure - should maintain flow
        flow_rate = pump.calculate_flow_rate(10000.0)  # 0.1 bar
        self.assertGreater(flow_rate, 0)
        self.assertLessEqual(flow_rate, self.pump_params.max_flow_rate)
        
        # Excessive pressure - should stall
        flow_rate = pump.calculate_flow_rate(100000.0)  # 1 bar > max
        self.assertEqual(flow_rate, 0.0)
    
    def test_centrifugal_flow_calculation(self):
        """Test centrifugal pump flow rate calculation."""
        pump = CentrifugalPump("test_pump", self.pump_params)
        
        # No pressure - should give max flow
        flow_rate = pump.calculate_flow_rate(0.0)
        self.assertAlmostEqual(flow_rate, self.pump_params.max_flow_rate, places=1)
        
        # Moderate pressure - should reduce flow
        flow_rate = pump.calculate_flow_rate(25000.0)  # Half max pressure
        self.assertGreater(flow_rate, 0)
        self.assertLess(flow_rate, self.pump_params.max_flow_rate)
        
        # Max pressure - should stop flow
        flow_rate = pump.calculate_flow_rate(self.pump_params.max_pressure)
        self.assertAlmostEqual(flow_rate, 0.0, places=1)
    
    def test_power_consumption_calculation(self):
        """Test pump power consumption calculations."""
        pump = PeristalticPump("test_pump", self.pump_params)
        
        # No flow - standby power
        power = pump.calculate_power_consumption(0.0, ATMOSPHERIC_PRESSURE)
        self.assertGreater(power, 0)
        self.assertLess(power, self.pump_params.power_rating)
        
        # Normal operation
        power = pump.calculate_power_consumption(50.0, 20000.0)
        self.assertGreater(power, 0)
        self.assertLessEqual(power, self.pump_params.power_rating * 1.2)
    
    def test_pump_operation_update(self):
        """Test pump operation update over time."""
        pump = PeristalticPump("test_pump", self.pump_params)
        pump.start_pump()
        
        initial_hours = pump.operating_hours
        
        # Update operation with reasonable system pressure
        pump.update_operation(dt=1.0, target_flow_rate=50.0, system_pressure=ATMOSPHERIC_PRESSURE + 10000.0)
        
        # Check updates
        self.assertGreater(pump.operating_hours, initial_hours)
        self.assertGreater(pump.current_flow_rate, 0)
        self.assertGreater(pump.current_power, 0)
    
    def test_pump_status_reporting(self):
        """Test pump status reporting."""
        pump = PeristalticPump("test_pump", self.pump_params)
        pump.start_pump()
        pump.update_operation(dt=1.0, target_flow_rate=50.0, system_pressure=ATMOSPHERIC_PRESSURE + 10000.0)
        
        status = pump.get_pump_status()
        
        required_keys = [
            'pump_id', 'pump_type', 'is_running', 'flow_rate_ml_min',
            'power_consumption_w', 'operating_hours', 'maintenance_due', 'efficiency_pct'
        ]
        
        for key in required_keys:
            self.assertIn(key, status)
        
        self.assertEqual(status['pump_id'], 'test_pump')
        self.assertEqual(status['pump_type'], 'peristaltic')
        self.assertTrue(status['is_running'])


class TestFlowCalculations(unittest.TestCase):
    """Test hydraulic flow calculation utilities."""
    
    def setUp(self):
        if not HYDRAULIC_AVAILABLE:
            self.skipTest("Hydraulic system not available")
    
    def test_reynolds_number_calculation(self):
        """Test Reynolds number calculation."""
        # Typical values
        velocity = 0.1  # m/s
        diameter = 0.003  # 3mm
        
        reynolds = FlowCalculator.calculate_reynolds_number(velocity, diameter)
        
        self.assertGreater(reynolds, 0)
        self.assertIsInstance(reynolds, float)
        
        # Known case: should be around 300 for these values
        self.assertGreater(reynolds, 200)
        self.assertLess(reynolds, 500)
    
    def test_friction_factor_laminar(self):
        """Test friction factor calculation for laminar flow."""
        reynolds = 1000  # Laminar flow
        roughness = 1e-6
        diameter = 0.003
        
        friction_factor = FlowCalculator.calculate_friction_factor(reynolds, roughness, diameter)
        
        # For laminar flow: f = 64/Re
        expected = 64.0 / reynolds
        self.assertAlmostEqual(friction_factor, expected, places=3)
    
    def test_friction_factor_turbulent(self):
        """Test friction factor calculation for turbulent flow."""
        reynolds = 10000  # Turbulent flow
        roughness = 1e-6
        diameter = 0.003
        
        friction_factor = FlowCalculator.calculate_friction_factor(reynolds, roughness, diameter)
        
        self.assertGreater(friction_factor, 0)
        self.assertLess(friction_factor, 0.1)  # Reasonable range
    
    def test_pressure_drop_calculation(self):
        """Test pressure drop calculation."""
        flow_rate = 50.0  # mL/min
        pipe_diameter = 0.003  # 3mm
        pipe_length = 1.0  # 1m
        roughness = 1e-6
        
        pressure_drop = FlowCalculator.calculate_pressure_drop(
            flow_rate, pipe_diameter, pipe_length, roughness
        )
        
        self.assertGreaterEqual(pressure_drop, 0)
        self.assertIsInstance(pressure_drop, float)
        
        # Zero flow should give zero pressure drop
        zero_pressure_drop = FlowCalculator.calculate_pressure_drop(
            0.0, pipe_diameter, pipe_length, roughness
        )
        self.assertEqual(zero_pressure_drop, 0.0)
    
    def test_hydrostatic_pressure_calculation(self):
        """Test hydrostatic pressure calculation."""
        elevation_change = 0.1  # 10cm elevation
        
        pressure = FlowCalculator.calculate_hydrostatic_pressure(elevation_change)
        
        # Should be ρgh
        expected = WATER_DENSITY * GRAVITY * elevation_change
        self.assertAlmostEqual(pressure, expected, places=1)


class TestCellGeometry(unittest.TestCase):
    """Test cell geometry models."""
    
    def setUp(self):
        if not HYDRAULIC_AVAILABLE:
            self.skipTest("Hydraulic system not available")
    
    def test_cell_geometry_creation(self):
        """Test cell geometry creation."""
        geometry = CellGeometry(
            length=0.1,
            width=0.05,
            height=0.02,
            volume=0.0001
        )
        
        self.assertEqual(geometry.length, 0.1)
        self.assertEqual(geometry.width, 0.05)
        self.assertEqual(geometry.height, 0.02)
        self.assertEqual(geometry.volume, 0.0001)
        
        # Check default values
        self.assertGreater(geometry.anode_area, 0)
        self.assertGreater(geometry.cathode_area, 0)
        self.assertGreater(geometry.inlet_diameter, 0)


class TestHydraulicNetwork(unittest.TestCase):
    """Test hydraulic network modeling."""
    
    def setUp(self):
        if not HYDRAULIC_AVAILABLE:
            self.skipTest("Hydraulic system not available")
        
        # Create test components
        self.cell_geometry = CellGeometry()
        self.piping = PipingNetwork()
        self.network = HydraulicNetwork([self.cell_geometry], self.piping)
        
        # Add test pump
        pump_params = PumpParameters(max_flow_rate=50.0, power_rating=3.0)
        test_pump = PeristalticPump("test_pump", pump_params)
        self.network.add_pump(test_pump, "test_flow")
    
    def test_network_creation(self):
        """Test hydraulic network creation."""
        self.assertEqual(len(self.network.cells), 1)
        self.assertIsInstance(self.network.piping, PipingNetwork)
        self.assertIn("test_flow", self.network.pumps)
        self.assertIn("test_flow", self.network.flow_rates)
        self.assertIn("test_flow", self.network.pressures)
    
    def test_pressure_calculation(self):
        """Test network pressure calculations."""
        # Set flow rate
        self.network.flow_rates["test_flow"] = 25.0  # mL/min
        
        # Calculate pressures
        self.network.calculate_network_pressures()
        
        # Check that pressure was calculated
        pressure = self.network.pressures["test_flow"]
        self.assertGreaterEqual(pressure, ATMOSPHERIC_PRESSURE)
    
    def test_network_update(self):
        """Test hydraulic network update."""
        # Start pump
        self.network.pumps["test_flow"].start_pump()
        
        # Update network
        target_flows = {"test_flow": 25.0}
        self.network.update_hydraulic_system(dt=1.0, target_flows=target_flows)
        
        # Check updates
        self.assertGreater(self.network.total_power_consumption, 0)
        pump = self.network.pumps["test_flow"]
        self.assertGreater(pump.current_power, 0)
    
    def test_network_status(self):
        """Test network status reporting."""
        status = self.network.get_network_status()
        
        required_keys = ['pumps', 'flow_rates', 'pressures', 'total_power_w', 'network_efficiency']
        for key in required_keys:
            self.assertIn(key, status)
        
        self.assertIsInstance(status['pumps'], dict)
        self.assertIsInstance(status['flow_rates'], dict)
        self.assertIsInstance(status['total_power_w'], float)


class TestHydraulicController(unittest.TestCase):
    """Test hydraulic control system."""
    
    def setUp(self):
        if not HYDRAULIC_AVAILABLE:
            self.skipTest("Hydraulic system not available")
        
        # Create test network
        self.network = create_standard_hydraulic_system(n_cells=1)
        self.controller = HydraulicController(self.network)
    
    def test_controller_creation(self):
        """Test hydraulic controller creation."""
        self.assertIsInstance(self.controller.network, HydraulicNetwork)
        self.assertIsInstance(self.controller.control_setpoints, dict)
        self.assertIsInstance(self.controller.pid_controllers, dict)
    
    def test_setpoint_setting(self):
        """Test flow setpoint setting."""
        self.controller.set_flow_setpoint("test_flow", 30.0)
        
        self.assertIn("test_flow", self.controller.control_setpoints)
        self.assertEqual(self.controller.control_setpoints["test_flow"], 30.0)
        self.assertIn("test_flow", self.controller.pid_controllers)
    
    def test_control_calculation(self):
        """Test PID control calculation."""
        self.controller.set_flow_setpoint("test_flow", 30.0)
        
        # Calculate control output
        output = self.controller.calculate_control_output("test_flow", 25.0, 1.0)
        
        self.assertIsInstance(output, float)
        self.assertGreaterEqual(output, 0)
    
    def test_control_update(self):
        """Test control system update."""
        # Set setpoints for existing pumps
        for flow_path in self.network.pumps.keys():
            self.controller.set_flow_setpoint(flow_path, 20.0)
        
        # Update control
        control_outputs = self.controller.update_control(dt=1.0)
        
        self.assertIsInstance(control_outputs, dict)
        for flow_path in self.controller.control_setpoints:
            self.assertIn(flow_path, control_outputs)


class TestStandardSystem(unittest.TestCase):
    """Test standard hydraulic system creation."""
    
    def setUp(self):
        if not HYDRAULIC_AVAILABLE:
            self.skipTest("Hydraulic system not available")
    
    def test_standard_system_creation(self):
        """Test standard hydraulic system creation."""
        system = create_standard_hydraulic_system(n_cells=3)
        
        self.assertEqual(len(system.cells), 3)
        self.assertGreater(len(system.pumps), 0)
        
        # Check that standard pumps are included
        pump_types = [pump.pump_type for pump in system.pumps.values()]
        self.assertIn(PumpType.PERISTALTIC, pump_types)  # Substrate pump
    
    def test_single_cell_system(self):
        """Test single cell hydraulic system."""
        system = create_standard_hydraulic_system(n_cells=1)
        
        self.assertEqual(len(system.cells), 1)
        self.assertIsInstance(system.piping, PipingNetwork)
    
    def test_multi_cell_system(self):
        """Test multi-cell hydraulic system."""
        system = create_standard_hydraulic_system(n_cells=5)
        
        self.assertEqual(len(system.cells), 5)
        # Piping should scale with number of cells
        self.assertGreater(system.piping.total_length, 2.0)


class TestCostCalculations(unittest.TestCase):
    """Test hydraulic cost calculations."""
    
    def setUp(self):
        if not HYDRAULIC_AVAILABLE:
            self.skipTest("Hydraulic system not available")
        
        self.system = create_standard_hydraulic_system(n_cells=2)
        
        # Set some power consumption
        self.system.total_power_consumption = 10.0  # W
    
    def test_cost_calculation(self):
        """Test hydraulic cost calculation."""
        costs = calculate_hydraulic_costs(
            network=self.system,
            operating_hours=100.0,
            electricity_cost=0.12
        )
        
        required_keys = [
            'capital_cost_usd', 'energy_cost_usd', 'maintenance_cost_usd',
            'total_cost_usd', 'cost_per_hour_usd', 'power_consumption_kwh'
        ]
        
        for key in required_keys:
            self.assertIn(key, costs)
            self.assertIsInstance(costs[key], float)
            self.assertGreaterEqual(costs[key], 0)
        
        # Capital cost should be sum of pump costs
        expected_capital = sum(pump.params.cost for pump in self.system.pumps.values())
        self.assertEqual(costs['capital_cost_usd'], expected_capital)
        
        # Energy cost should be reasonable
        self.assertGreater(costs['energy_cost_usd'], 0)
    
    def test_zero_hours_cost(self):
        """Test cost calculation with zero operating hours."""
        costs = calculate_hydraulic_costs(
            network=self.system,
            operating_hours=0.0
        )
        
        self.assertEqual(costs['cost_per_hour_usd'], 0.0)
        self.assertGreaterEqual(costs['capital_cost_usd'], 0)


class TestSystemIntegration(unittest.TestCase):
    """Test complete hydraulic system integration."""
    
    def setUp(self):
        if not HYDRAULIC_AVAILABLE:
            self.skipTest("Hydraulic system not available")
    
    def test_complete_system_operation(self):
        """Test complete hydraulic system operation."""
        # Create system
        system = create_standard_hydraulic_system(n_cells=2)
        controller = HydraulicController(system)
        
        # Set up control
        for flow_path in system.pumps.keys():
            controller.set_flow_setpoint(flow_path, 15.0)
            system.pumps[flow_path].start_pump()
        
        # Run simulation
        total_time = 0.0
        dt = 0.1  # hours
        
        for step in range(5):
            # Update control
            target_flows = controller.update_control(dt)
            
            # Update hydraulic system
            system.update_hydraulic_system(dt, target_flows)
            
            total_time += dt
        
        # Check that system operated
        self.assertGreater(total_time, 0)
        self.assertGreater(system.total_power_consumption, 0)
        
        # Get final status
        status = system.get_network_status()
        self.assertIsInstance(status, dict)
        self.assertGreater(status['total_power_w'], 0)
    
    def test_system_efficiency_calculation(self):
        """Test system efficiency calculation."""
        system = create_standard_hydraulic_system(n_cells=1)
        
        # Set up operation
        for pump in system.pumps.values():
            pump.start_pump()
        
        target_flows = {path: 10.0 for path in system.pumps.keys()}
        system.update_hydraulic_system(dt=1.0, target_flows=target_flows)
        
        status = system.get_network_status()
        efficiency = status['network_efficiency']
        
        self.assertIsInstance(efficiency, float)
        self.assertGreaterEqual(efficiency, 0.0)
        self.assertLessEqual(efficiency, 1.0)


def run_hydraulic_tests():
    """Run all hydraulic system tests."""
    print("🔧 Running Hydraulic System Tests")
    print("=" * 50)
    
    if not HYDRAULIC_AVAILABLE:
        print("❌ Cannot run tests - hydraulic system not available")
        return False
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestPumpModels,
        TestFlowCalculations,
        TestCellGeometry,
        TestHydraulicNetwork,
        TestHydraulicController,
        TestStandardSystem,
        TestCostCalculations,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 Hydraulic System Test Summary")
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
    success = run_hydraulic_tests()
    sys.exit(0 if success else 1)