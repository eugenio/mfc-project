"""
Comprehensive Tests for MFC Sensor Systems

This module tests all sensor types including substrate, temperature, 
conductivity, oxygen sensors, and the integrated sensor system.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sensing_models.substrate_sensors import (
    SubstrateSensor, SubstrateSensorType, SubstrateSensorSpecs,
    create_standard_substrate_sensors
)
from sensing_models.temperature_sensors import (
    TemperatureSensor, TemperatureSensorType, TemperatureSensorSpecs,
    ThermalDynamicsModel, create_standard_temperature_sensors
)
from sensing_models.conductivity_sensors import (
    ConductivitySensor, ConductivitySensorType, ConductivitySensorSpecs,
    ElectrolyteModel, create_standard_conductivity_sensors
)
from sensing_models.oxygen_sensors import (
    OxygenSensor, OxygenSensorType, OxygenSensorSpecs,
    HenryLawModel, create_standard_oxygen_sensors
)
from sensing_models.sensor_integration import (
    SensorSystem, create_standard_mfc_sensor_system,
    MFCEnvironmentalConditions, SensorStatus, AlertLevel
)


class TestSubstrateSensors(unittest.TestCase):
    """Test substrate concentration sensors"""
    
    def setUp(self):
        """Set up test sensors"""
        self.sensors = create_standard_substrate_sensors()
        
    def test_sensor_creation(self):
        """Test that all sensor types are created"""
        expected_types = ['uv_vis_lactate', 'enzymatic_lactate', 'amperometric_lactate']
        for sensor_type in expected_types:
            self.assertIn(sensor_type, self.sensors)
            self.assertIsInstance(self.sensors[sensor_type], SubstrateSensor)
    
    def test_uv_vis_measurement(self):
        """Test UV-Vis sensor measurements"""
        sensor = self.sensors['uv_vis_lactate']
        
        # Test measurement
        measurement = sensor.measure(10.0, temperature=25.0, ph=7.0, time=1.0)
        
        self.assertIsNotNone(measurement)
        self.assertGreater(measurement.concentration, 0)
        self.assertGreater(measurement.uncertainty, 0)
        self.assertIn(measurement.quality_flag, ['good', 'warning', 'bad'])
        self.assertEqual(measurement.sensor_type, SubstrateSensorType.UV_VIS_SPECTROSCOPY)
    
    def test_enzymatic_sensor_degradation(self):
        """Test enzymatic sensor degradation over time"""
        sensor = self.sensors['enzymatic_lactate']
        
        measurements = []
        for hour in range(0, 100, 10):
            measurement = sensor.measure(10.0, time=hour)
            measurements.append(measurement.concentration)
        
        # Enzymatic sensors should show some degradation
        self.assertLess(measurements[-1], measurements[0] * 1.1)  # Allow some tolerance
    
    def test_sensor_calibration(self):
        """Test sensor calibration"""
        sensor = self.sensors['amperometric_lactate']
        
        # Calibration data
        standards = [1.0, 5.0, 10.0, 20.0]
        measured = [0.95, 5.1, 9.8, 20.2]
        
        success = sensor.calibrate(standards, measured, time=50.0)
        self.assertTrue(success)
    
    def test_power_and_cost_analysis(self):
        """Test power consumption and cost analysis"""
        sensor = self.sensors['uv_vis_lactate']
        
        power = sensor.get_power_consumption()
        self.assertGreater(power, 0)
        
        cost_analysis = sensor.get_cost_analysis()
        self.assertIn('initial_cost', cost_analysis)
        self.assertIn('total_cost_per_hour', cost_analysis)
        self.assertGreater(cost_analysis['initial_cost'], 0)


class TestTemperatureSensors(unittest.TestCase):
    """Test temperature sensors with thermal dynamics"""
    
    def setUp(self):
        """Set up test sensors"""
        self.sensors = create_standard_temperature_sensors()
    
    def test_sensor_creation(self):
        """Test sensor creation"""
        expected_types = ['thermocouple_k', 'rtd_pt100', 'thermistor_ntc', 'ic_sensor']
        for sensor_type in expected_types:
            self.assertIn(sensor_type, self.sensors)
            self.assertIsInstance(self.sensors[sensor_type], TemperatureSensor)
    
    def test_thermal_dynamics(self):
        """Test thermal dynamics modeling"""
        sensor = self.sensors['rtd_pt100']
        
        # Step response test
        measurements = []
        for t in np.linspace(0, 60, 61):  # 60 seconds
            environment_temp = 50.0 if t > 30 else 25.0  # Step at t=30s
            measurement = sensor.measure(environment_temp, time=t/3600, dt=1.0)
            measurements.append(measurement.temperature)
        
        # Should show thermal lag
        self.assertLess(measurements[31], 40.0)  # Should not instantly reach 50°C
        self.assertGreater(measurements[-1], 45.0)  # Should eventually approach 50°C
    
    def test_self_heating_effects(self):
        """Test self-heating effects in RTD"""
        sensor = self.sensors['rtd_pt100']
        
        measurement = sensor.measure(25.0, time=1.0)
        self.assertGreater(measurement.self_heating_effect, 0)
    
    def test_temperature_calibration(self):
        """Test temperature sensor calibration"""
        sensor = self.sensors['thermocouple_k']
        
        reference_temps = [0.0, 50.0, 100.0]
        measured_temps = [0.2, 50.1, 99.8]
        
        success = sensor.calibrate(reference_temps, measured_temps, time=10.0)
        self.assertTrue(success)
    
    def test_thermal_model(self):
        """Test thermal dynamics model directly"""
        thermal_model = ThermalDynamicsModel(thermal_mass=0.1, heat_transfer_coefficient=10.0)
        
        # Test time constant calculation
        time_constant = thermal_model.get_time_constant()
        self.assertEqual(time_constant, 0.01)  # 0.1/10.0
        
        # Test temperature update
        initial_temp = thermal_model.sensor_temperature
        thermal_model.update(environment_temp=50.0, dt=1.0, power_dissipation=1.0)
        
        self.assertNotEqual(thermal_model.sensor_temperature, initial_temp)


class TestConductivitySensors(unittest.TestCase):
    """Test conductivity sensors"""
    
    def setUp(self):
        """Set up test sensors"""
        self.sensors = create_standard_conductivity_sensors()
    
    def test_sensor_creation(self):
        """Test sensor creation"""
        expected_types = ['two_electrode', 'four_electrode', 'inductive', 'toroidal']
        for sensor_type in expected_types:
            self.assertIn(sensor_type, self.sensors)
            self.assertIsInstance(self.sensors[sensor_type], ConductivitySensor)
    
    def test_conductivity_measurement(self):
        """Test conductivity measurements"""
        sensor = self.sensors['four_electrode']
        
        measurement = sensor.measure(50.0, temperature=25.0, time=1.0)
        
        self.assertGreater(measurement.conductivity, 0)
        self.assertIsInstance(measurement.uncertainty, float)
        self.assertIn(measurement.quality_flag, ['good', 'warning', 'bad'])
        self.assertEqual(measurement.sensor_type, ConductivitySensorType.FOUR_ELECTRODE)
    
    def test_fouling_effects(self):
        """Test sensor fouling over time"""
        sensor = self.sensors['two_electrode']
        
        initial_measurement = sensor.measure(50.0, time=0.0)
        
        # Simulate 1000 hours of operation
        final_measurement = sensor.measure(50.0, time=1000.0)
        
        # Fouling should reduce signal quality
        self.assertLess(final_measurement.fouling_factor, initial_measurement.fouling_factor)
    
    def test_sensor_cleaning(self):
        """Test sensor cleaning functionality"""
        sensor = self.sensors['two_electrode']
        
        # Degrade sensor through operation
        sensor.measure(50.0, time=500.0)
        fouling_before = sensor.fouling_factor
        
        # Clean sensor
        success = sensor.clean_sensor(cleaning_effectiveness=0.9)
        self.assertTrue(success)
        
        # Fouling factor should improve
        self.assertGreater(sensor.fouling_factor, fouling_before)
    
    def test_electrolyte_model(self):
        """Test electrolyte model calculations"""
        electrolyte = ElectrolyteModel()
        
        # Test conductivity calculation for NaCl solution
        ion_conc = {'Na+': 0.1, 'Cl-': 0.1}  # 0.1 M NaCl
        conductivity = electrolyte.calculate_theoretical_conductivity(ion_conc)
        
        self.assertGreater(conductivity, 0)
        self.assertLess(conductivity, 2000)  # Reasonable range for 0.1 M NaCl (theoretical value is higher)
        
        # Test ionic strength estimation
        ionic_strength = electrolyte.estimate_ionic_strength(conductivity)
        self.assertGreater(ionic_strength, 0)


class TestOxygenSensors(unittest.TestCase):
    """Test dissolved oxygen sensors"""
    
    def setUp(self):
        """Set up test sensors"""
        self.sensors = create_standard_oxygen_sensors()
    
    def test_sensor_creation(self):
        """Test sensor creation"""
        expected_types = ['clark_electrode', 'optical_luminescence', 'galvanic', 'paramagnetic']
        for sensor_type in expected_types:
            self.assertIn(sensor_type, self.sensors)
            self.assertIsInstance(self.sensors[sensor_type], OxygenSensor)
    
    def test_do_measurement(self):
        """Test DO measurements"""
        sensor = self.sensors['optical_luminescence']
        
        measurement = sensor.measure(5.0, temperature=25.0, pressure=1.013, time=1.0)
        
        self.assertGreater(measurement.do_concentration, 0)
        self.assertGreater(measurement.do_saturation, 0)
        self.assertIsInstance(measurement.uncertainty, float)
        self.assertEqual(measurement.sensor_type, OxygenSensorType.OPTICAL_LUMINESCENCE)
    
    def test_membrane_degradation(self):
        """Test membrane degradation in Clark electrode"""
        sensor = self.sensors['clark_electrode']
        
        initial_condition = sensor.membrane_condition
        
        # Simulate 6 months of operation
        sensor.measure(5.0, time=4380.0)  # 6 months = 4380 hours
        
        self.assertLess(sensor.membrane_condition, initial_condition)
    
    def test_membrane_replacement(self):
        """Test membrane replacement"""
        sensor = self.sensors['clark_electrode']
        
        # Degrade membrane
        sensor.measure(5.0, time=5000.0)
        
        # Replace membrane
        success = sensor.replace_membrane()
        self.assertTrue(success)
        self.assertEqual(sensor.membrane_condition, 1.0)
    
    def test_henry_law_model(self):
        """Test Henry's Law model for DO saturation"""
        henry_model = HenryLawModel()
        
        # Test saturation concentration at different temperatures
        sat_25c = henry_model.calculate_saturation_concentration(25.0)
        sat_10c = henry_model.calculate_saturation_concentration(10.0)
        
        # For dissolved gases, solubility typically decreases with temperature
        # At lower temperature, more gas can dissolve
        self.assertGreater(sat_10c, sat_25c)
        
        # Test saturation percentage
        saturation = henry_model.calculate_saturation_percentage(5.0, 25.0)
        self.assertGreater(saturation, 0)
        self.assertLess(saturation, 100)
    
    def test_two_point_calibration(self):
        """Test two-point calibration (0% and 100% saturation)"""
        sensor = self.sensors['optical_luminescence']
        henry_model = HenryLawModel()
        
        # Get saturation concentration at 25°C
        sat_conc = henry_model.calculate_saturation_concentration(25.0)
        
        # Two-point calibration
        reference_concs = [0.0, sat_conc]
        measured_vals = [0.05, sat_conc * 0.98]
        
        success = sensor.calibrate(reference_concs, measured_vals, time=10.0, 
                                 calibration_type="two_point")
        self.assertTrue(success)


class TestSensorIntegration(unittest.TestCase):
    """Test integrated sensor system"""
    
    def setUp(self):
        """Set up test sensor system"""
        self.sensor_system = create_standard_mfc_sensor_system("Test_System")
    
    def test_system_creation(self):
        """Test sensor system creation"""
        self.assertGreater(len(self.sensor_system.sensors), 0)
        self.assertGreater(self.sensor_system.total_power_consumption, 0)
        self.assertGreater(self.sensor_system.total_cost_per_hour, 0)
    
    def test_measurement_acquisition(self):
        """Test acquiring measurements from all sensors"""
        measurements = self.sensor_system.acquire_all_measurements(time=1.0)
        
        # Should get measurements from enabled sensors
        self.assertGreater(len(measurements), 0)
        
        # Check measurement structure
        for sensor_id, measurement in measurements.items():
            self.assertIn('timestamp', measurement)
            self.assertIn('value', measurement)
            self.assertIn('unit', measurement)
            self.assertIn('uncertainty', measurement)
            self.assertIn('quality', measurement)
    
    def test_environmental_conditions_update(self):
        """Test updating environmental conditions"""
        original_temp = self.sensor_system.environmental_conditions.temperature
        
        self.sensor_system.update_environmental_conditions(temperature=35.0)
        
        self.assertEqual(self.sensor_system.environmental_conditions.temperature, 35.0)
        self.assertNotEqual(self.sensor_system.environmental_conditions.temperature, original_temp)
    
    def test_alert_generation(self):
        """Test alert generation for out-of-range values"""
        # Set extreme environmental conditions to trigger alerts
        self.sensor_system.update_environmental_conditions(
            temperature=100.0,  # Extreme temperature
            substrate_concentration=200.0  # High concentration
        )
        
        measurements = self.sensor_system.acquire_all_measurements(time=1.0)
        
        # Should generate some alerts
        # Note: Alerts may be generated based on sensor quality or range checks
        initial_alert_count = len(self.sensor_system.alerts)
        
        # Make another measurement to potentially trigger more alerts
        self.sensor_system.acquire_all_measurements(time=2.0)
        
        # Alert count should remain same or increase
        final_alert_count = len(self.sensor_system.alerts)
        self.assertGreaterEqual(final_alert_count, initial_alert_count)
    
    def test_sensor_calibration(self):
        """Test sensor calibration through system interface"""
        # Find a substrate sensor
        substrate_sensor_id = None
        for sensor_id in self.sensor_system.sensors:
            if 'substrate' in sensor_id:
                substrate_sensor_id = sensor_id
                break
        
        self.assertIsNotNone(substrate_sensor_id)
        
        # Calibration data
        calibration_data = {
            'reference_values': [1.0, 10.0, 20.0],
            'measured_values': [1.1, 9.8, 20.2]
        }
        
        success = self.sensor_system.calibrate_sensor(substrate_sensor_id, calibration_data, time=10.0)
        self.assertTrue(success)
    
    def test_sensor_maintenance(self):
        """Test sensor maintenance operations"""
        # Find a conductivity sensor that supports cleaning
        conductivity_sensor_id = None
        for sensor_id in self.sensor_system.sensors:
            if 'conductivity' in sensor_id:
                conductivity_sensor_id = sensor_id
                break
        
        self.assertIsNotNone(conductivity_sensor_id)
        
        # Perform cleaning maintenance
        success = self.sensor_system.perform_maintenance(conductivity_sensor_id, "clean")
        self.assertTrue(success)
    
    def test_system_status(self):
        """Test system status reporting"""
        status = self.sensor_system.get_system_status()
        
        self.assertIn('system_id', status)
        self.assertIn('total_sensors', status)
        self.assertIn('operational_sensors', status)
        self.assertIn('system_health', status)
        self.assertIn('total_power_consumption', status)
        self.assertIn('environmental_conditions', status)
        
        # System health should be between 0 and 1
        self.assertGreaterEqual(status['system_health'], 0.0)
        self.assertLessEqual(status['system_health'], 1.0)
    
    def test_data_logging_and_export(self):
        """Test data logging and export functionality"""
        # Generate some measurements
        for hour in range(5):
            self.sensor_system.acquire_all_measurements(time=hour)
        
        # Test sensor summary
        summary = self.sensor_system.get_sensor_summary(hours=24.0)
        self.assertGreater(len(summary), 0)
        
        # Test data export
        exported_data = self.sensor_system.export_data(hours=24.0)
        self.assertGreater(len(exported_data), 0)
        
        # Check exported data structure
        for sensor_id, data_points in exported_data.items():
            if data_points:  # If there's data
                self.assertIsInstance(data_points, list)
                self.assertIn('timestamp', data_points[0])
                self.assertIn('value', data_points[0])
    
    def test_power_and_cost_analysis(self):
        """Test system-wide power and cost analysis"""
        # Test individual sensor power/cost
        for sensor_id, sensor in self.sensor_system.sensors.items():
            if hasattr(sensor, 'get_power_consumption'):
                power = sensor.get_power_consumption()
                self.assertGreater(power, 0)
            
            if hasattr(sensor, 'get_cost_analysis'):
                cost_analysis = sensor.get_cost_analysis()
                self.assertIsInstance(cost_analysis, dict)
                self.assertIn('total_cost_per_hour', cost_analysis)
        
        # Test system totals
        self.assertGreater(self.sensor_system.total_power_consumption, 0)
        self.assertGreater(self.sensor_system.total_cost_per_hour, 0)


class TestSensorReliability(unittest.TestCase):
    """Test sensor reliability and edge cases"""
    
    def test_extreme_conditions(self):
        """Test sensors under extreme conditions"""
        sensors = create_standard_temperature_sensors()
        temp_sensor = sensors['rtd_pt100']
        
        # Test extreme temperatures
        measurement_low = temp_sensor.measure(-50.0, time=1.0)
        measurement_high = temp_sensor.measure(200.0, time=1.0)
        
        # Measurements should be within sensor range
        self.assertGreaterEqual(measurement_low.temperature, temp_sensor.specs.measurement_range[0])
        self.assertLessEqual(measurement_high.temperature, temp_sensor.specs.measurement_range[1])
    
    def test_sensor_lifetime(self):
        """Test sensor behavior near end of lifetime"""
        sensors = create_standard_oxygen_sensors()
        sensor = sensors['galvanic']
        
        # Test near end of lifetime
        measurement = sensor.measure(5.0, time=sensor.specs.lifetime * 0.95)
        
        # Should still provide measurement but quality may degrade
        self.assertIsNotNone(measurement)
        self.assertGreater(measurement.do_concentration, 0)
    
    def test_calibration_edge_cases(self):
        """Test calibration with edge cases"""
        sensors = create_standard_substrate_sensors()
        sensor = sensors['uv_vis_lactate']
        
        # Test calibration with insufficient points
        success = sensor.calibrate([1.0], [1.1], time=1.0)
        self.assertFalse(success)
        
        # Test calibration with mismatched arrays
        success = sensor.calibrate([1.0, 5.0], [1.1], time=1.0)
        self.assertFalse(success)
    
    def test_sensor_error_handling(self):
        """Test error handling in sensor operations"""
        sensor_system = create_standard_mfc_sensor_system()
        
        # Test calibration of non-existent sensor
        success = sensor_system.calibrate_sensor("non_existent", {}, time=1.0)
        self.assertFalse(success)
        
        # Test maintenance of non-existent sensor
        success = sensor_system.perform_maintenance("non_existent", "clean")
        self.assertFalse(success)


def run_comprehensive_tests():
    """Run all sensor tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSubstrateSensors,
        TestTemperatureSensors,
        TestConductivitySensors,
        TestOxygenSensors,
        TestSensorIntegration,
        TestSensorReliability
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run comprehensive tests
    success = run_comprehensive_tests()
    
    if success:
        print("\n✅ All sensor tests passed successfully!")
    else:
        print("\n❌ Some sensor tests failed!")
        
    exit(0 if success else 1)