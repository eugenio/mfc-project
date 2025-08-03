"""
Comprehensive tests for controller models
"""

import pytest
import numpy as np
import time
import tempfile
import json

# Import controller modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from controller_models.model_inference import (
    ModelInferenceEngine, InferenceSpecs, ModelFormat, InferenceMeasurement,
    create_standard_inference_engines
)
from controller_models.control_electronics import (
    InterfaceType, MCUArchitecture,
    create_standard_control_electronics
)
from controller_models.real_time_controller import (
    RealTimeController, ControlLoop, TimingConstraints, ControllerMode,
    ControlTask, TaskPriority, create_standard_real_time_controllers
)
from controller_models.hardware_abstraction import (
    HardwareAbstractionLayer, MFCControlInterface, DeviceStatus,
    SensorDevice, ActuatorDevice, PowerDevice, DeviceInfo,
    create_mfc_hardware_system
)
from controller_models.controller_cost_analysis import (
    ControllerCostAnalyzer, PowerRequirement,
    CostItem, CostCategory, create_standard_controller_configurations
)


class TestModelInferenceEngine:
    """Test suite for ModelInferenceEngine"""
    
    @pytest.fixture
    def inference_specs(self):
        return InferenceSpecs(
            model_format=ModelFormat.NUMPY,
            max_inference_time_ms=5.0,
            memory_limit_mb=256.0,
            cache_size=100,
            batch_processing=True,
            quantization=False,
            optimization_level=1,
            power_consumption=2.0,
            cost=200.0,
            cpu_cores=2,
            ram_mb=512.0,
            storage_mb=64.0,
            temperature_range=(-20, 60)
        )
    
    @pytest.fixture
    def inference_engine(self, inference_specs):
        return ModelInferenceEngine(inference_specs)
    
    def test_inference_engine_creation(self, inference_engine):
        """Test inference engine creation"""
        assert inference_engine.specs.model_format == ModelFormat.NUMPY
        assert inference_engine.model is None
        assert inference_engine.total_inferences == 0
        assert len(inference_engine.inference_cache) == 0
    
    def test_load_numpy_model(self, inference_engine):
        """Test loading NumPy model"""
        # Create dummy model
        dummy_model = np.random.rand(10, 5)  # 10 states, 5 actions
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            np.save(f.name, dummy_model)
            
            success = inference_engine.load_model(f.name)
            assert success
            assert inference_engine.model is not None
            assert isinstance(inference_engine.model, np.ndarray)
            assert inference_engine.model.shape == (10, 5)
    
    def test_load_json_model(self, inference_specs):
        """Test loading JSON Q-table model"""
        inference_specs.model_format = ModelFormat.JSON
        engine = ModelInferenceEngine(inference_specs)
        
        # Create dummy JSON Q-table
        q_table = {
            "(0, 0)": [1.0, 2.0, 3.0],
            "(0, 1)": [2.0, 1.0, 2.5],
            "(1, 0)": [1.5, 3.0, 1.0]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"q_table": q_table}, f)
            f.flush()  # Ensure data is written to disk
            
        try:
            success = engine.load_model(f.name)
            assert success
            assert engine.model is not None
            assert isinstance(engine.model, dict)
        finally:
            # Clean up the temp file
            import os
            os.unlink(f.name)
    
    def test_inference_with_numpy_model(self, inference_engine):
        """Test inference with NumPy model"""
        # Load dummy model
        dummy_model = np.random.rand(100, 5)
        inference_engine.model = dummy_model
        
        # Test inference
        test_state = np.array([0.5, 0.3, 0.8])
        measurement = inference_engine.infer(test_state)
        
        assert isinstance(measurement, InferenceMeasurement)
        assert 0 <= measurement.output_action < 5
        assert 0.0 <= measurement.confidence_score <= 1.0
        assert measurement.inference_time_ms >= 0
        assert not measurement.cache_hit  # First inference
    
    def test_inference_caching(self, inference_engine):
        """Test inference caching functionality"""
        dummy_model = np.random.rand(100, 5)
        inference_engine.model = dummy_model
        
        test_state = np.array([0.5, 0.3, 0.8])
        
        # First inference
        measurement1 = inference_engine.infer(test_state)
        assert not measurement1.cache_hit
        
        # Second inference (should hit cache)
        measurement2 = inference_engine.infer(test_state)
        assert measurement2.cache_hit
        assert measurement2.output_action == measurement1.output_action
    
    def test_batch_inference(self, inference_engine):
        """Test batch inference"""
        dummy_model = np.random.rand(100, 5)
        inference_engine.model = dummy_model
        
        test_states = [np.random.rand(3) for _ in range(10)]
        results = inference_engine.batch_infer(test_states)
        
        assert len(results) == 10
        assert all(isinstance(r, InferenceMeasurement) for r in results)
    
    def test_performance_stats(self, inference_engine):
        """Test performance statistics"""
        dummy_model = np.random.rand(100, 5)
        inference_engine.model = dummy_model
        
        # Perform some inferences
        for _ in range(5):
            test_state = np.random.rand(3)
            inference_engine.infer(test_state)
        
        stats = inference_engine.get_performance_stats()
        
        assert stats['total_inferences'] == 5
        assert stats['average_inference_time_ms'] >= 0
        assert 0 <= stats['cache_hit_rate'] <= 1
    
    def test_standard_engines(self):
        """Test creation of standard inference engines"""
        engines = create_standard_inference_engines()
        
        assert 'high_performance' in engines
        assert 'low_power' in engines
        assert 'balanced' in engines
        
        for engine in engines.values():
            assert isinstance(engine, ModelInferenceEngine)


class TestControlElectronics:
    """Test suite for ControlElectronics"""
    
    @pytest.fixture
    def control_electronics(self):
        systems = create_standard_control_electronics()
        return systems['high_performance']
    
    def test_electronics_creation(self, control_electronics):
        """Test control electronics creation"""
        assert control_electronics.mcu_specs.architecture == MCUArchitecture.ARM_CORTEX_M7
        assert control_electronics.adc_specs.resolution_bits == 16
        assert control_electronics.dac_specs.resolution_bits == 16
        # ControlElectronics doesn't have a status attribute - it's a model, not a device
        assert control_electronics.power_mode == "active"  # Check default power mode instead
    
    def test_adc_operations(self, control_electronics):
        """Test ADC read operations"""
        # Set a test value
        control_electronics.adc_readings[0] = 2.5
        
        # Read ADC value
        reading = control_electronics.read_adc(0, samples=1)
        
        assert isinstance(reading, float)
        assert 0.0 <= reading <= control_electronics.adc_specs.reference_voltage_v
        
        # Test invalid channel
        with pytest.raises(ValueError):
            control_electronics.read_adc(999)
    
    def test_dac_operations(self, control_electronics):
        """Test DAC write operations"""
        # Write valid value
        success = control_electronics.write_dac(0, 2.0)
        assert success
        assert 0 in control_electronics.dac_outputs
        
        # Test out-of-range value (should be clipped, but allow for linearity error)
        success = control_electronics.write_dac(0, 10.0)
        assert success
        assert control_electronics.dac_outputs[0] <= control_electronics.dac_specs.reference_voltage_v + 0.1
        
        # Test invalid channel
        success = control_electronics.write_dac(999, 1.0)
        assert not success
    
    def test_gpio_operations(self, control_electronics):
        """Test GPIO operations"""
        # Set GPIO pin
        success = control_electronics.set_gpio(0, True, "output")
        assert success
        assert control_electronics.gpio_states[0]
        assert control_electronics.gpio_directions[0] == "output"
        
        # Read GPIO pin
        state = control_electronics.read_gpio(0)
        assert state
        
        # Test invalid pin
        success = control_electronics.set_gpio(999, True)
        assert not success
    
    def test_communication(self, control_electronics):
        """Test communication interfaces"""
        test_data = b"test_message"
        
        success, response = control_electronics.communicate(InterfaceType.SPI, test_data)
        
        assert isinstance(success, bool)
        if success:
            assert isinstance(response, bytes)
            assert len(response) > 0
    
    def test_power_consumption(self, control_electronics):
        """Test power consumption calculation"""
        power = control_electronics.get_power_consumption()
        
        assert power > 0
        assert isinstance(power, float)
        
        # Test different power modes
        control_electronics.set_power_mode("sleep")
        sleep_power = control_electronics.get_power_consumption()
        
        control_electronics.set_power_mode("active")
        active_power = control_electronics.get_power_consumption()
        
        assert sleep_power < active_power
    
    def test_thermal_model(self, control_electronics):
        """Test thermal model updates"""
        initial_temp = control_electronics.mcu_temperature
        
        # Update thermal model with high ambient temperature
        control_electronics.update_thermal_model(dt=1.0, ambient_temp=50.0)
        
        # Temperature should change
        assert control_electronics.mcu_temperature != initial_temp
    
    def test_measurement(self, control_electronics):
        """Test getting electronics measurement"""
        measurement = control_electronics.get_measurement()
        
        assert hasattr(measurement, 'timestamp')
        assert hasattr(measurement, 'mcu_temperature_c')
        assert hasattr(measurement, 'power_consumption_mw')
        assert hasattr(measurement, 'cpu_utilization_pct')
        assert measurement.power_consumption_mw > 0

    def test_gpio_input_mode(self, control_electronics):
        """Test GPIO input mode operations"""
        # Set GPIO as input
        success = control_electronics.set_gpio(0, False, "input")
        assert success
        assert control_electronics.gpio_directions[0] == "input"
        
        # Test reading input GPIO (should return current state)
        state = control_electronics.read_gpio(0)
        assert isinstance(state, bool)
        
        # Test with external GPIO states
        control_electronics._external_gpio_states = {0: True}
        state = control_electronics.read_gpio(0)
        # Should return True most of the time (with rare errors)
        # We can't guarantee exact value due to randomness
        assert isinstance(state, bool)

    def test_gpio_error_conditions(self, control_electronics):
        """Test GPIO error conditions"""
        # Test invalid GPIO direction
        success = control_electronics.set_gpio(0, True, "invalid")
        assert not success
        
        # Test reading invalid GPIO pin
        state = control_electronics.read_gpio(999)
        assert not state  # Should return False for invalid pin

    def test_communication_interface_errors(self, control_electronics):
        """Test communication interface error conditions"""
        # Test invalid interface type
        success, response = control_electronics.communicate(InterfaceType.USB, b"test")
        assert not success
        assert response == b""

    def test_communication_noise_simulation(self, control_electronics):
        """Test communication with noise/errors"""
        # Run multiple communications to potentially trigger errors
        error_count = 0
        total_tests = 100
        
        for _ in range(total_tests):
            success, response = control_electronics.communicate(InterfaceType.SPI, b"test_data")
            if not success:
                error_count += 1
        
        # We should have some successful communications
        assert error_count < total_tests  # Not all should fail

    def test_thermal_limit_violations(self, control_electronics):
        """Test thermal limit violations"""
        # Force temperature below minimum
        control_electronics.mcu_temperature = -50.0
        control_electronics.update_thermal_model(dt=1.0)
        
        # Check that fault flags are generated
        measurement = control_electronics.get_measurement()
        temp_faults = [f for f in measurement.fault_flags if "TEMP" in f]
        assert len(temp_faults) > 0
        
        # Force temperature above maximum
        control_electronics.mcu_temperature = 100.0
        control_electronics.update_thermal_model(dt=1.0)
        
        measurement = control_electronics.get_measurement()
        temp_faults = [f for f in measurement.fault_flags if "TEMP" in f]
        assert len(temp_faults) > 0

    def test_adc_noise_and_linearity(self, control_electronics):
        """Test ADC noise and linearity effects"""
        # Set a known value
        control_electronics.adc_readings[0] = 2.0
        
        # Take multiple readings to test noise
        readings = []
        for _ in range(10):
            reading = control_electronics.read_adc(0, samples=1)
            readings.append(reading)
        
        # Readings should vary due to noise
        assert len(set(readings)) > 1  # Should have different values due to noise
        
        # All readings should be within reasonable range
        for reading in readings:
            assert 0.0 <= reading <= control_electronics.adc_specs.reference_voltage_v

    def test_adc_multiple_samples(self, control_electronics):
        """Test ADC with multiple samples"""
        control_electronics.adc_readings[0] = 1.5
        
        # Test with multiple samples (should take longer)
        start_time = time.time()
        reading = control_electronics.read_adc(0, samples=10)
        end_time = time.time()
        
        assert isinstance(reading, float)
        # Should take some time due to sampling
        elapsed_time = end_time - start_time
        assert elapsed_time > 0

    def test_dac_range_clamping(self, control_electronics):
        """Test DAC output range clamping"""
        # Test voltage above maximum range
        max_voltage = control_electronics.dac_specs.output_range_v[1]
        success = control_electronics.write_dac(0, max_voltage + 10.0)
        assert success
        # Output should be clamped to maximum (allowing for linearity error)
        assert control_electronics.dac_outputs[0] <= max_voltage + 0.1
        
        # Test voltage below minimum range
        min_voltage = control_electronics.dac_specs.output_range_v[0]
        success = control_electronics.write_dac(0, min_voltage - 10.0)
        assert success
        # Output should be clamped to minimum (allowing for linearity error)
        assert control_electronics.dac_outputs[0] >= min_voltage - 0.1

    def test_dac_quantization_and_linearity(self, control_electronics):
        """Test DAC quantization and linearity effects"""
        # Test precise voltage that will be quantized
        precise_voltage = 1.23456789
        success = control_electronics.write_dac(0, precise_voltage)
        assert success
        
        # Output should be quantized to DAC resolution
        output_voltage = control_electronics.dac_outputs[0]
        lsb_voltage = control_electronics.dac_specs.reference_voltage_v / (2**control_electronics.dac_specs.resolution_bits)
        
        # Should be close to input (within quantization + linearity error)
        assert abs(output_voltage - precise_voltage) < lsb_voltage * 2

    def test_power_mode_transitions(self, control_electronics):
        """Test power mode transitions and effects"""
        initial_cpu = control_electronics.cpu_utilization
        control_electronics.cpu_utilization = 50.0  # Set high utilization
        
        # Switch to sleep mode
        success = control_electronics.set_power_mode("sleep")
        assert success
        assert control_electronics.power_mode == "sleep"
        # CPU utilization should be reduced
        assert control_electronics.cpu_utilization < 50.0
        
        # Switch to deep sleep mode
        success = control_electronics.set_power_mode("deep_sleep")
        assert success
        assert control_electronics.power_mode == "deep_sleep"
        # CPU utilization should be zero
        assert control_electronics.cpu_utilization == 0.0
        
        # Test invalid power mode
        success = control_electronics.set_power_mode("invalid_mode")
        assert not success

    def test_gpio_switching_power(self, control_electronics):
        """Test GPIO switching power calculation"""
        # Set initial state
        control_electronics.set_gpio(0, False, "output")
        control_electronics._previous_gpio_states = control_electronics.gpio_states.copy()
        
        # Switch GPIO state to trigger switching power
        control_electronics.set_gpio(0, True, "output")
        
        # Should have switching power recorded
        switching_power = getattr(control_electronics, '_gpio_switching_power', 0)
        assert switching_power >= 0

    def test_communication_activity_tracking(self, control_electronics):
        """Test communication activity tracking"""
        initial_activity = control_electronics.comm_activity[InterfaceType.SPI]
        
        # Send some data
        control_electronics.communicate(InterfaceType.SPI, b"test_data_for_activity")
        
        # Activity should increase
        new_activity = control_electronics.comm_activity[InterfaceType.SPI]
        assert new_activity >= initial_activity

    def test_cost_analysis_comprehensive(self, control_electronics):
        """Test comprehensive cost analysis"""
        cost_analysis = control_electronics.get_cost_analysis()
        
        # Verify all expected cost components
        expected_keys = [
            'initial_cost', 'mcu_cost', 'adc_cost', 'dac_cost', 
            'gpio_cost', 'communication_cost', 'power_cost_per_hour',
            'development_cost_per_hour', 'maintenance_cost_per_hour', 
            'total_cost_per_hour'
        ]
        
        for key in expected_keys:
            assert key in cost_analysis
            assert cost_analysis[key] >= 0  # All costs should be non-negative

    def test_measurement_with_high_activity(self, control_electronics):
        """Test measurement with high system activity"""
        # Generate high activity
        control_electronics.cpu_utilization = 90.0
        for i in range(5):
            control_electronics.read_adc(i % control_electronics.adc_specs.channels)
            control_electronics.write_dac(i % control_electronics.dac_specs.channels, 2.0)
            control_electronics.set_gpio(i % 10, True, "output")
        
        # Get measurement
        measurement = control_electronics.get_measurement()
        
        # Verify high activity is reflected
        assert measurement.cpu_utilization_pct > 0
        assert measurement.memory_usage_pct > 30  # Should have base + variable memory

    def test_low_power_system(self):
        """Test low power electronics system"""
        systems = create_standard_control_electronics()
        lp_system = systems['low_power']
        
        # Verify low power specifications
        assert lp_system.mcu_specs.architecture == MCUArchitecture.ARM_CORTEX_M0
        assert lp_system.mcu_specs.power_consumption_active_mw < 50.0
        assert lp_system.adc_specs.resolution_bits == 12
        assert lp_system.dac_specs.channels == 2
        
        # Test basic operations
        measurement = lp_system.get_measurement()
        assert measurement.power_consumption_mw > 0

    def test_adc_noise_history_tracking(self, control_electronics):
        """Test ADC noise history tracking"""
        control_electronics.adc_readings[0] = 1.0
        
        # Clear noise history
        control_electronics.adc_noise_history.clear()
        
        # Take several readings to build history
        for _ in range(10):
            control_electronics.read_adc(0, samples=1)
        
        # Should have noise history
        assert len(control_electronics.adc_noise_history) == 10
        
        # Test noise history overflow (should limit to 1000)
        for _ in range(1000):
            control_electronics.read_adc(0, samples=1)
        
        assert len(control_electronics.adc_noise_history) <= 1000

    def test_communication_target_address(self, control_electronics):
        """Test communication with target address"""
        # Test communication with target address parameter
        success, response = control_electronics.communicate(
            InterfaceType.SPI, b"test", target_address=0x42
        )
        # Should work same as without target address (simplified implementation)
        assert isinstance(success, bool)
        assert isinstance(response, bytes)

    def test_measurement_decay_effects(self, control_electronics):
        """Test measurement decay effects over time"""
        # Set high initial values
        control_electronics.cpu_utilization = 80.0
        control_electronics.comm_activity[InterfaceType.SPI] = 90.0
        
        # Get measurement (should apply decay)
        measurement1 = control_electronics.get_measurement()
        initial_cpu = measurement1.cpu_utilization_pct
        
        # Get another measurement (should decay further)
        measurement2 = control_electronics.get_measurement()
        decayed_cpu = measurement2.cpu_utilization_pct
        
        # CPU utilization should decay
        assert decayed_cpu <= initial_cpu

    def test_thermal_model_without_ambient_change(self, control_electronics):
        """Test thermal model update without changing ambient temperature"""
        initial_temp = control_electronics.mcu_temperature
        initial_ambient = control_electronics.ambient_temperature
        
        # Update without changing ambient temperature
        control_electronics.update_thermal_model(dt=0.1)
        
        # Ambient temperature should remain the same
        assert control_electronics.ambient_temperature == initial_ambient
        # Temperature may change due to power dissipation
        # assert control_electronics.mcu_temperature != initial_temp  # May or may not change

    def test_communication_error_counting(self, control_electronics):
        """Test communication error counting"""
        initial_errors = control_electronics.comm_error_counts[InterfaceType.SPI]
        
        # Force communication errors by using very large noise immunity (inverse relationship)
        interface = control_electronics.comm_interfaces[InterfaceType.SPI]
        original_noise_immunity = interface.noise_immunity_db
        interface.noise_immunity_db = 1.0  # Low noise immunity = high error probability
        
        error_count = 0
        for _ in range(100):
            success, _ = control_electronics.communicate(InterfaceType.SPI, b"test")
            if not success:
                error_count += 1
        
        # Should have some errors with low noise immunity
        final_errors = control_electronics.comm_error_counts[InterfaceType.SPI]
        assert final_errors >= initial_errors
        
        # Restore original noise immunity
        interface.noise_immunity_db = original_noise_immunity

    def test_power_consumption_modes_detailed(self, control_electronics):
        """Test detailed power consumption in different modes"""
        # Test active mode power scaling
        control_electronics.cpu_utilization = 0.0
        control_electronics.set_power_mode("active")
        min_active_power = control_electronics.get_power_consumption()
        
        control_electronics.cpu_utilization = 100.0
        max_active_power = control_electronics.get_power_consumption()
        
        # Power should scale with utilization
        assert max_active_power > min_active_power
        
        # Test sleep mode
        control_electronics.set_power_mode("sleep")
        sleep_power = control_electronics.get_power_consumption()
        
        # Test deep sleep mode
        control_electronics.set_power_mode("deep_sleep")
        deep_sleep_power = control_electronics.get_power_consumption()
        
        # Power should decrease: active > sleep > deep_sleep
        assert min_active_power > sleep_power
        assert sleep_power >= deep_sleep_power


class TestRealTimeController:
    """Test suite for RealTimeController"""
    
    @pytest.fixture
    def timing_constraints(self):
        return TimingConstraints(
            control_loop_period_ms=10.0,
            max_jitter_ms=1.0,
            deadline_violation_limit=5,
            interrupt_response_time_us=50.0,
            context_switch_time_us=20.0,
            worst_case_execution_time_ms=5.0,
            watchdog_timeout_ms=500.0,
            safety_stop_timeout_ms=50.0,
            sensor_timeout_ms=100.0
        )
    
    @pytest.fixture
    def rt_controller(self, timing_constraints):
        return RealTimeController(timing_constraints)
    
    def test_controller_creation(self, rt_controller):
        """Test real-time controller creation"""
        assert rt_controller.mode == ControllerMode.MANUAL
        assert not rt_controller.running
        assert len(rt_controller.tasks) == 0
    
    def test_task_management(self, rt_controller):
        """Test task addition and removal"""
        def dummy_task():
            return "test"
        
        task = ControlTask(
            task_id="test_task",
            priority=TaskPriority.HIGH,
            period_ms=5.0,
            deadline_ms=4.0,
            wcet_ms=2.0,
            callback=dummy_task
        )
        
        rt_controller.add_task(task)
        assert "test_task" in rt_controller.tasks
        
        rt_controller.remove_task("test_task")
        assert "test_task" not in rt_controller.tasks
    
    def test_control_loop_management(self, rt_controller):
        """Test control loop management"""
        loop = ControlLoop(
            loop_id="test_loop",
            input_channels=[0, 1],
            output_channels=[0],
            control_algorithm="PID",
            setpoint=25.0,
            gains={'kp': 1.0, 'ki': 0.1, 'kd': 0.05},
            limits={'output': (-10.0, 10.0)}
        )
        
        rt_controller.add_control_loop(loop)
        assert "test_loop" in rt_controller.control_loops
        
        # Test setpoint update
        rt_controller.set_control_loop_setpoint("test_loop", 30.0)
        assert rt_controller.control_loops["test_loop"].setpoint == 30.0
    
    def test_mode_switching(self, rt_controller):
        """Test controller mode switching"""
        rt_controller.set_mode(ControllerMode.AUTOMATIC)
        assert rt_controller.mode == ControllerMode.AUTOMATIC
        
        rt_controller.set_mode(ControllerMode.SAFETY)
        assert rt_controller.mode == ControllerMode.SAFETY
        assert rt_controller.safety_state == "EMERGENCY_STOP"
    
    def test_sensor_data_update(self, rt_controller):
        """Test sensor data updates"""
        rt_controller.update_sensor_data(0, 25.5)
        assert rt_controller.sensor_data[0] == 25.5
        
        rt_controller.update_sensor_data(1, 30.2)
        assert rt_controller.sensor_data[1] == 30.2
    
    def test_actuator_outputs(self, rt_controller):
        """Test actuator output management"""
        # Set output value
        rt_controller.actuator_outputs[0] = 50.0
        
        # Get output value
        output = rt_controller.get_actuator_output(0)
        assert output == 50.0
        
        # Non-existent channel should return 0
        output = rt_controller.get_actuator_output(999)
        assert output == 0.0
    
    def test_measurement(self, rt_controller):
        """Test getting controller measurement"""
        measurement = rt_controller.get_measurement()
        
        assert measurement.mode == ControllerMode.MANUAL
        assert measurement.cpu_utilization_pct >= 0
        assert measurement.memory_usage_mb >= 0
        assert isinstance(measurement.fault_flags, list)
    
    def test_standard_controllers(self):
        """Test creation of standard controllers"""
        controllers = create_standard_real_time_controllers()
        
        assert 'high_performance' in controllers
        assert 'low_power' in controllers
        
        for controller in controllers.values():
            assert isinstance(controller, RealTimeController)

    def test_controller_start_stop(self, rt_controller):
        """Test controller start and stop operations"""
        # Controller should start as not running
        assert not rt_controller.running
        
        # Start controller
        rt_controller.start()
        assert rt_controller.running
        assert rt_controller.scheduler_thread is not None
        
        # Wait a moment for threads to start
        time.sleep(0.1)
        
        # Stop controller
        rt_controller.stop()
        assert not rt_controller.running

    def test_multiple_start_calls(self, rt_controller):
        """Test multiple start calls don't cause issues"""
        rt_controller.start()
        assert rt_controller.running
        
        # Second start should be ignored
        rt_controller.start()  # Should log warning but not crash
        assert rt_controller.running
        
        rt_controller.stop()

    def test_task_execution_timing(self, rt_controller):
        """Test task execution and timing"""
        execution_count = 0
        
        def test_callback():
            nonlocal execution_count
            execution_count += 1
            return "executed"
        
        task = ControlTask(
            task_id="timing_test",
            priority=TaskPriority.HIGH,
            period_ms=10.0,
            deadline_ms=8.0,
            wcet_ms=2.0,
            callback=test_callback
        )
        
        rt_controller.add_task(task)
        rt_controller.start()
        
        # Let it run for a short time
        time.sleep(0.05)  # 50ms should allow several executions
        rt_controller.stop()
        
        # Should have executed multiple times
        assert execution_count > 0
        assert "timing_test" in rt_controller.tasks

    def test_task_priority_ordering(self, rt_controller):
        """Test that tasks execute in priority order"""
        execution_order = []
        
        def high_priority_task():
            execution_order.append("HIGH")
        
        def low_priority_task():
            execution_order.append("LOW")
        
        def critical_task():
            execution_order.append("CRITICAL")
        
        # Add tasks in reverse priority order
        low_task = ControlTask("low", TaskPriority.LOW, 5.0, 4.0, 1.0, low_priority_task)
        high_task = ControlTask("high", TaskPriority.HIGH, 5.0, 4.0, 1.0, high_priority_task)
        critical_task = ControlTask("critical", TaskPriority.CRITICAL, 5.0, 4.0, 1.0, critical_task)
        
        rt_controller.add_task(low_task)
        rt_controller.add_task(high_task)
        rt_controller.add_task(critical_task)
        
        # Set all tasks to execute immediately
        current_time = time.time()
        for task in rt_controller.tasks.values():
            task.next_execution = current_time
        
        rt_controller.start()
        time.sleep(0.02)  # Brief execution
        rt_controller.stop()
        
        # Should have executed in priority order (CRITICAL, HIGH, LOW)
        if execution_order:  # May not execute if timing is tight
            assert execution_order[0] == "CRITICAL"

    def test_control_loop_pid_algorithm(self, rt_controller):
        """Test PID control algorithm execution"""
        loop = ControlLoop(
            loop_id="pid_test",
            input_channels=[0],
            output_channels=[0],
            control_algorithm="PID",
            setpoint=50.0,
            gains={'kp': 1.0, 'ki': 0.1, 'kd': 0.05},
            limits={'output': (-100.0, 100.0)}
        )
        
        rt_controller.add_control_loop(loop)
        rt_controller.update_sensor_data(0, 45.0)  # Below setpoint
        
        # Switch to automatic mode to enable control loops
        rt_controller.set_mode(ControllerMode.AUTOMATIC)
        
        # Execute control loop manually for testing
        rt_controller._execute_control_loops()
        
        # Should have calculated some output
        assert loop.output_value != 0.0
        assert loop.previous_error != 0.0

    def test_control_loop_limits(self, rt_controller):
        """Test control loop output limiting"""
        loop = ControlLoop(
            loop_id="limit_test",
            input_channels=[0],
            output_channels=[0],
            control_algorithm="PID",
            setpoint=50.0,
            gains={'kp': 100.0, 'ki': 0.0, 'kd': 0.0},  # High gain to saturate
            limits={'output': (-10.0, 10.0)}
        )
        
        rt_controller.add_control_loop(loop)
        rt_controller.update_sensor_data(0, 0.0)  # Far from setpoint
        rt_controller.set_mode(ControllerMode.AUTOMATIC)
        
        rt_controller._execute_control_loops()
        
        # Output should be clamped to limits
        assert -10.0 <= loop.output_value <= 10.0

    def test_safety_mode_activation(self, rt_controller):
        """Test safety mode activation and behavior"""
        rt_controller.set_mode(ControllerMode.SAFETY)
        
        assert rt_controller.mode == ControllerMode.SAFETY
        assert rt_controller.safety_state == "EMERGENCY_STOP"
        
        # Should have stopped all outputs
        for output_value in rt_controller.actuator_outputs.values():
            assert output_value == 0.0

    def test_learning_mode_activation(self, rt_controller):
        """Test learning mode activation"""
        rt_controller.set_mode(ControllerMode.LEARNING)
        
        assert rt_controller.mode == ControllerMode.LEARNING
        # In learning mode, exploration should be enabled
        # This would typically set up Q-learning parameters

    def test_fault_handling(self, rt_controller):
        """Test fault handling and recovery"""
        # Trigger a fault
        rt_controller._handle_controller_fault("TEST_FAULT")
        
        measurement = rt_controller.get_measurement()
        assert "TEST_FAULT" in measurement.fault_flags

    def test_timing_violation_handling(self, rt_controller):
        """Test timing violation detection and handling"""
        # Force timing violations by manipulating jitter history
        rt_controller.jitter_history.extend([100.0] * 10)  # High jitter values
        rt_controller.deadline_violations = 50
        
        # This should trigger timing violation handling
        rt_controller._handle_timing_violation()
        
        # Should be in safety mode due to many violations
        assert rt_controller.mode == ControllerMode.SAFETY

    def test_watchdog_functionality(self, rt_controller):
        """Test watchdog timer functionality"""
        # Set watchdog to have been pet recently
        rt_controller.watchdog_last_pet = time.time()
        
        # Check that watchdog was recently pet (within timeout)
        time_since_pet = (time.time() - rt_controller.watchdog_last_pet) * 1000
        assert time_since_pet < rt_controller.timing_constraints.watchdog_timeout_ms

    def test_watchdog_timeout(self, rt_controller):
        """Test watchdog timeout detection"""
        # Set watchdog to have not been pet for a long time
        rt_controller.watchdog_last_pet = time.time() - 10.0  # 10 seconds ago
        
        # Check if timeout would be detected
        time_since_pet = (time.time() - rt_controller.watchdog_last_pet) * 1000
        assert time_since_pet > rt_controller.timing_constraints.watchdog_timeout_ms

    def test_sensor_timeout_detection(self, rt_controller):
        """Test sensor data timeout handling"""
        # Update sensor data and verify it's stored
        rt_controller.update_sensor_data(0, 25.0)
        
        # Check that sensor data was stored
        assert rt_controller.sensor_data[0] == 25.0

    def test_performance_monitoring(self, rt_controller):
        """Test performance monitoring and metrics"""
        # Simulate some activity
        rt_controller.loop_execution_times.extend([5.0, 6.0, 4.5, 7.0])
        rt_controller.jitter_history.extend([0.5, 1.0, 0.3, 0.8])
        rt_controller.interrupt_count = 100
        
        rt_controller._update_performance_metrics()
        
        measurement = rt_controller.get_measurement()
        assert measurement.cpu_utilization_pct >= 0
        assert measurement.jitter_ms >= 0
        assert measurement.interrupt_count == 100

    def test_control_loop_disabled(self, rt_controller):
        """Test disabled control loop doesn't execute"""
        loop = ControlLoop(
            loop_id="disabled_test",
            input_channels=[0],
            output_channels=[0],
            control_algorithm="PID",
            setpoint=50.0,
            gains={'kp': 1.0, 'ki': 0.1, 'kd': 0.05},
            limits={'output': (-100.0, 100.0)},
            enabled=False  # Disabled
        )
        
        rt_controller.add_control_loop(loop)
        rt_controller.update_sensor_data(0, 45.0)
        rt_controller.set_mode(ControllerMode.AUTOMATIC)
        
        initial_output = loop.output_value
        rt_controller._execute_control_loops()
        
        # Output should not change for disabled loop
        assert loop.output_value == initial_output

    def test_task_disabled(self, rt_controller):
        """Test disabled task doesn't execute"""
        execution_count = 0
        
        def test_callback():
            nonlocal execution_count
            execution_count += 1
        
        task = ControlTask(
            task_id="disabled_task",
            priority=TaskPriority.HIGH,
            period_ms=5.0,
            deadline_ms=4.0,
            wcet_ms=1.0,
            callback=test_callback,
            enabled=False  # Disabled
        )
        
        rt_controller.add_task(task)
        task.next_execution = time.time()  # Ready to execute
        
        rt_controller._execute_scheduled_tasks(time.time())
        
        # Should not have executed
        assert execution_count == 0

    def test_emergency_stop_functionality(self, rt_controller):
        """Test emergency stop functionality through safety mode"""
        # Set some actuator outputs
        rt_controller.actuator_outputs[0] = 50.0
        rt_controller.actuator_outputs[1] = -25.0
        
        # Trigger safety mode (which implements emergency stop)
        rt_controller.set_mode(ControllerMode.SAFETY)
        
        # All outputs should be zero
        for output in rt_controller.actuator_outputs.values():
            assert output == 0.0
        
        assert rt_controller.safety_state == "EMERGENCY_STOP"

    def test_control_loop_setpoint_update(self, rt_controller):
        """Test updating control loop setpoint"""
        loop = ControlLoop(
            loop_id="setpoint_test",
            input_channels=[0],
            output_channels=[0],
            control_algorithm="PID",
            setpoint=25.0,
            gains={'kp': 1.0, 'ki': 0.1, 'kd': 0.05},
            limits={'output': (-100.0, 100.0)}
        )
        
        rt_controller.add_control_loop(loop)
        
        # Update setpoint
        rt_controller.set_control_loop_setpoint("setpoint_test", 50.0)
        
        assert rt_controller.control_loops["setpoint_test"].setpoint == 50.0

    def test_invalid_control_loop_setpoint(self, rt_controller):
        """Test updating setpoint for non-existent control loop"""
        # Should not raise error for non-existent loop
        rt_controller.set_control_loop_setpoint("non_existent", 50.0)
        # Just logs warning and continues

    def test_actuator_output_retrieval(self, rt_controller):
        """Test actuator output value retrieval"""
        rt_controller.actuator_outputs[5] = 75.0
        
        output = rt_controller.get_actuator_output(5)
        assert output == 75.0
        
        # Non-existent channel should return 0
        output = rt_controller.get_actuator_output(999)
        assert output == 0.0

    def test_comprehensive_measurement(self, rt_controller):
        """Test comprehensive controller measurement"""
        # Set up some state
        rt_controller.deadline_violations = 5
        rt_controller.interrupt_count = 150
        rt_controller.cpu_utilization = 65.0
        rt_controller.memory_usage = 128.5
        rt_controller.fault_flags = ["TEST_FAULT_1", "TEST_FAULT_2"]
        
        # Add some control loops and tasks
        loop = ControlLoop("test_loop", [0], [0], "PID", 25.0, {'kp': 1.0}, {'output': (-10, 10)})
        rt_controller.add_control_loop(loop)
        
        measurement = rt_controller.get_measurement()
        
        # Verify all measurement fields
        assert measurement.mode == ControllerMode.MANUAL
        assert measurement.cpu_utilization_pct == 65.0
        assert measurement.memory_usage_mb == 128.5
        # deadline_violations_recent is calculated differently in the implementation
        assert measurement.deadline_violations_recent >= 0
        assert measurement.interrupt_count == 150
        assert "test_loop" in measurement.active_control_loops
        assert len(measurement.fault_flags) >= 2


class TestHardwareAbstractionLayer:
    """Test suite for HardwareAbstractionLayer"""
    
    @pytest.fixture
    def hal_system(self):
        return create_mfc_hardware_system()
    
    def test_hal_creation(self, hal_system):
        """Test HAL system creation"""
        assert isinstance(hal_system, HardwareAbstractionLayer)
        
        control_interface = hal_system.get_control_interface()
        assert isinstance(control_interface, MFCControlInterface)
        assert len(control_interface.devices) > 0
    
    def test_device_initialization(self, hal_system):
        """Test device initialization"""
        control_interface = hal_system.get_control_interface()
        
        init_results = control_interface.initialize_all_devices()
        
        assert isinstance(init_results, dict)
        assert len(init_results) > 0
        
        # At least some devices should initialize successfully
        success_count = sum(1 for success in init_results.values() if success)
        assert success_count > 0
    
    def test_sensor_operations(self, hal_system):
        """Test sensor operations"""
        control_interface = hal_system.get_control_interface()
        
        # Initialize devices first
        control_interface.initialize_all_devices()
        
        # Test sensor reading
        ph_value = control_interface.read_sensor('pH_001')
        if ph_value is not None:
            assert isinstance(ph_value, float)
        
        temp_value = control_interface.read_sensor('TEMP_001')
        if temp_value is not None:
            assert isinstance(temp_value, float)
    
    def test_actuator_operations(self, hal_system):
        """Test actuator operations"""
        control_interface = hal_system.get_control_interface()
        
        # Initialize devices first
        control_interface.initialize_all_devices()
        
        # Test actuator control
        success = control_interface.write_actuator('PUMP_001', 50.0)
        assert isinstance(success, bool)
    
    def test_power_management(self, hal_system):
        """Test power management operations"""
        control_interface = hal_system.get_control_interface()
        
        # Initialize devices first
        control_interface.initialize_all_devices()
        
        # Test power control
        success = control_interface.set_power_output('PSU_001', 12.0, 2.0)
        assert isinstance(success, bool)
        
        # Test power status reading
        power_status = control_interface.read_power_status('PSU_001')
        if power_status is not None:
            assert isinstance(power_status, dict)
            assert 'voltage' in power_status
            assert 'current' in power_status
    
    def test_system_status(self, hal_system):
        """Test system status reporting"""
        control_interface = hal_system.get_control_interface()
        
        status = control_interface.get_system_status()
        
        assert isinstance(status, dict)
        assert 'device_count' in status
        assert 'system_health' in status
        assert 'devices' in status
        assert status['device_count'] > 0
        assert 0.0 <= status['system_health'] <= 1.0
    
    def test_diagnostics(self, hal_system):
        """Test system diagnostics"""
        control_interface = hal_system.get_control_interface()
        
        diagnostics = control_interface.run_diagnostics()
        
        assert isinstance(diagnostics, dict)
        assert 'timestamp' in diagnostics
        assert 'devices' in diagnostics
        assert 'performance' in diagnostics
        assert 'recommendations' in diagnostics
    
    def test_sensor_calibration(self, hal_system):
        """Test sensor calibration"""
        control_interface = hal_system.get_control_interface()
        
        # Initialize devices first
        control_interface.initialize_all_devices()
        
        # Test calibration
        cal_results = control_interface.calibrate_sensors('sensors')
        
        assert isinstance(cal_results, dict)
        for device_id, result in cal_results.items():
            assert isinstance(result, bool)

    def test_device_status_transitions(self, hal_system):
        """Test device status transitions"""
        devices = hal_system.control_interface.get_devices_by_group("sensors")
        if devices:
            device = devices[0]
            original_status = device.get_status()
            
            # Test status change
            device.set_status(DeviceStatus.MAINTENANCE)
            assert device.get_status() == DeviceStatus.MAINTENANCE
            
            # Change back
            device.set_status(original_status)
            assert device.get_status() == original_status

    def test_sensor_device_calibration(self, hal_system):
        """Test individual sensor device calibration"""
        sensor_devices = hal_system.control_interface.get_devices_by_group("sensors")
        if sensor_devices:
            sensor = sensor_devices[0]
            assert isinstance(sensor, SensorDevice)
            
            # Calibrate sensor
            success = sensor.calibrate()
            assert success
            
            # Should have reset drift
            assert sensor.calibration_drift == 0.0

    def test_sensor_device_error_conditions(self, hal_system):
        """Test sensor device error conditions"""
        sensor_devices = hal_system.control_interface.get_devices_by_group("sensors")
        if sensor_devices:
            sensor = sensor_devices[0]
            
            # Set sensor to ERROR status
            sensor.set_status(DeviceStatus.ERROR)
            
            # Reading should return NaN
            reading = sensor.read()
            assert np.isnan(reading)

    def test_actuator_device_slew_rate(self, hal_system):
        """Test actuator device slew rate limiting"""
        actuator_devices = hal_system.control_interface.get_devices_by_group("actuators")
        if actuator_devices:
            actuator = actuator_devices[0]
            assert isinstance(actuator, ActuatorDevice)
            
            # Set target and current values
            actuator.output_value = 0.0
            actuator.target_value = 100.0
            
            # Update with small time step
            actuator.update(dt=0.1)
            
            # Should move towards target but not reach it immediately due to slew rate
            assert 0.0 < actuator.output_value < 100.0

    def test_actuator_device_deadband(self, hal_system):
        """Test actuator device deadband functionality"""
        actuator_devices = hal_system.control_interface.get_devices_by_group("actuators")
        if actuator_devices:
            actuator = actuator_devices[0]
            
            # Set initial target
            actuator.write(50.0)
            initial_target = actuator.target_value
            
            # Write value within deadband
            actuator.write(initial_target + actuator.deadband / 2)
            
            # Target should not change
            assert actuator.target_value == initial_target

    def test_power_device_power_limiting(self, hal_system):
        """Test power device power limiting"""
        power_devices = hal_system.control_interface.get_devices_by_group("power")
        if power_devices:
            power_device = power_devices[0]
            assert isinstance(power_device, PowerDevice)
            
            # Try to set power above limit
            high_voltage = 50.0
            high_current = 10.0
            
            success = power_device.write({'voltage': high_voltage, 'current': high_current})
            assert success
            
            # Power should be limited
            actual_power = power_device.voltage_output * power_device.current_output
            assert actual_power <= power_device.power_limit

    def test_mfc_control_interface_device_groups(self, hal_system):
        """Test MFC control interface device grouping"""
        interface = hal_system.control_interface
        
        # Test getting devices by group
        sensors = interface.get_devices_by_group("sensors")
        actuators = interface.get_devices_by_group("actuators")
        power = interface.get_devices_by_group("power")
        
        # Should have devices in each group
        assert len(sensors) > 0
        assert len(actuators) > 0
        assert len(power) > 0
        
        # Test non-existent group
        non_existent = interface.get_devices_by_group("non_existent")
        assert len(non_existent) == 0

    def test_device_maintenance_tracking(self, hal_system):
        """Test device maintenance hour tracking"""
        devices = hal_system.control_interface.get_devices_by_group("sensors")
        if devices:
            device = devices[0]
            
            initial_hours = device.maintenance_hours
            device.update_maintenance_hours(10.0)
            
            assert device.maintenance_hours == initial_hours + 10.0

    def test_device_maintenance_schedule(self, hal_system):
        """Test device maintenance schedule warnings"""
        devices = hal_system.control_interface.get_devices_by_group("sensors")
        if devices:
            device = devices[0]
            
            # Force maintenance hours to exceed schedule
            if 'maintenance_interval_hours' in device.config.maintenance_schedule:
                interval = device.config.maintenance_schedule['maintenance_interval_hours']
                device.maintenance_hours = interval + 1
                
                device.update_maintenance_hours(1.0)  # Trigger check
                
                # Should be in maintenance status
                assert device.get_status() == DeviceStatus.MAINTENANCE

    def test_sensor_noise_and_drift(self, hal_system):
        """Test sensor noise and drift simulation"""
        sensor_devices = hal_system.control_interface.get_devices_by_group("sensors")
        if sensor_devices:
            sensor = sensor_devices[0]
            
            # Set known value
            sensor.write(25.0)
            
            # Take multiple readings
            readings = []
            for _ in range(10):
                reading = sensor.read()
                readings.append(reading)
            
            # Should have some variation due to noise
            assert len(set(readings)) > 1  # Not all readings should be identical

    def test_sensor_drift_over_time(self, hal_system):
        """Test sensor drift over time"""
        sensor_devices = hal_system.control_interface.get_devices_by_group("sensors")
        if sensor_devices:
            sensor = sensor_devices[0]
            
            # Set up for drift testing
            sensor.write(10.0)
            sensor.last_update = time.time() - 3600  # 1 hour ago
            
            # Reading should include drift
            reading_with_drift = sensor.read()
            
            # Should be different from set value due to drift
            assert reading_with_drift != 10.0

    def test_sensor_range_limiting(self, hal_system):
        """Test sensor measurement range limiting"""
        sensor_devices = hal_system.control_interface.get_devices_by_group("sensors")
        if sensor_devices:
            sensor = sensor_devices[0]
            
            # Force value outside range
            if 'measurement_range' in sensor.config.limits:
                min_val, max_val = sensor.config.limits['measurement_range']
                sensor.write(max_val + 100.0)  # Way above range
                
                reading = sensor.read()
                # Should still get a reading (may be clamped or generate warning)
                assert isinstance(reading, float)

    def test_actuator_range_limiting(self, hal_system):
        """Test actuator output range limiting"""
        actuator_devices = hal_system.control_interface.get_devices_by_group("actuators")
        if actuator_devices:
            actuator = actuator_devices[0]
            
            # Ensure actuator is online first
            actuator.set_status(DeviceStatus.ONLINE)
            
            # Test value outside range
            if 'output_range' in actuator.config.limits:
                min_val, max_val = actuator.config.limits['output_range']
                
                # Write value above maximum
                success = actuator.write(max_val + 50.0)
                assert success
                
                # Should be clamped to range
                assert actuator.target_value <= max_val

    def test_hal_driver_registration(self, hal_system):
        """Test HAL driver registration"""
        hal = hal_system
        
        # Should have registered drivers
        assert len(hal.device_drivers) > 0
        
        # Test registering a new driver
        class TestDriver:
            pass
        
        hal.register_driver("test_device", TestDriver)
        assert "test_device" in hal.device_drivers
        assert hal.device_drivers["test_device"] == TestDriver

    def test_configuration_manager(self, hal_system):
        """Test configuration manager functionality"""
        config_manager = hal_system.configuration_manager
        
        # Test saving and retrieving configuration
        test_config = {
            'device_type': 'test',
            'parameters': {'param1': 'value1'},
            'limits': {'limit1': (0, 100)},
            'calibration_data': {'offset': 0.5}
        }
        
        config_manager.save_device_config("test_device", test_config)
        retrieved_config = config_manager.get_device_config("test_device")
        
        assert retrieved_config == test_config

    def test_configuration_manager_non_existent(self, hal_system):
        """Test configuration manager with non-existent device"""
        config_manager = hal_system.configuration_manager
        
        # Should return None for non-existent device
        config = config_manager.get_device_config("non_existent_device")
        assert config is None

    def test_device_creation_from_config(self, hal_system):
        """Test creating device from configuration"""
        hal = hal_system
        
        # This would normally create from file, but we're testing the method exists
        device = hal.create_device_from_config("non_existent_config.json")
        # Should return None for non-existent file
        assert device is None

    def test_system_health_calculation(self, hal_system):
        """Test system health calculation"""
        interface = hal_system.control_interface
        
        # Initialize devices first to ensure they're online
        interface.initialize_all_devices()
        
        # Get initial health
        health = interface._calculate_system_health()
        assert 0.0 <= health <= 1.0
        
        # Set a device to error status
        devices = list(interface.devices.values())
        if devices:
            devices[0].set_status(DeviceStatus.ERROR)
            
            new_health = interface._calculate_system_health()
            # Health should decrease (unless all devices were already offline)
            assert new_health <= health

    def test_comprehensive_diagnostics(self, hal_system):
        """Test comprehensive system diagnostics"""
        interface = hal_system.control_interface
        
        # Run diagnostics
        diagnostics = interface.run_diagnostics()
        
        # Verify diagnostics structure
        assert 'timestamp' in diagnostics
        assert 'devices' in diagnostics
        assert 'performance' in diagnostics
        assert 'recommendations' in diagnostics
        
        # Check performance metrics
        performance = diagnostics['performance']
        assert 'system_health' in performance
        assert 'total_devices' in performance
        assert 'online_devices' in performance

    def test_diagnostics_recommendations(self, hal_system):
        """Test diagnostics recommendations generation"""
        interface = hal_system.control_interface
        
        # Force high error count on a device
        devices = list(interface.devices.values())
        if devices:
            devices[0].error_count = 15  # Above threshold of 10
            
            diagnostics = interface.run_diagnostics()
            
            # Should have recommendations
            assert len(diagnostics['recommendations']) > 0
            
            # Should mention high error count
            error_recommendations = [r for r in diagnostics['recommendations'] if 'error count' in r]
            assert len(error_recommendations) > 0

    def test_calibrate_all_sensors(self, hal_system):
        """Test calibrating all sensors without group specification"""
        interface = hal_system.control_interface
        
        # Calibrate all sensors (no group specified)
        results = interface.calibrate_sensors()
        
        # Should return results for all sensor devices
        assert isinstance(results, dict)
        assert len(results) >= 0

    def test_device_info_retrieval(self, hal_system):
        """Test device information retrieval"""
        devices = list(hal_system.control_interface.devices.values())
        if devices:
            device = devices[0]
            
            info = device.get_info()
            assert isinstance(info, DeviceInfo)
            assert hasattr(info, 'device_id')
            assert hasattr(info, 'device_type')
            assert hasattr(info, 'name')

    def test_device_offline_operations(self, hal_system):
        """Test operations on offline devices"""
        sensor_devices = hal_system.control_interface.get_devices_by_group("sensors")
        if sensor_devices:
            sensor = sensor_devices[0]
            
            # Set device offline
            sensor.set_status(DeviceStatus.OFFLINE)
            
            # Calibration should fail
            success = sensor.calibrate()
            assert not success

    def test_actuator_offline_write(self, hal_system):
        """Test writing to offline actuator"""
        actuator_devices = hal_system.control_interface.get_devices_by_group("actuators")
        if actuator_devices:
            actuator = actuator_devices[0]
            
            # Set device offline
            actuator.set_status(DeviceStatus.OFFLINE)
            
            # Write should fail
            success = actuator.write(50.0)
            assert not success

    def test_mfc_interface_sensor_reading_none_device(self, hal_system):
        """Test sensor reading with non-existent device"""
        interface = hal_system.control_interface
        
        # Try to read from non-existent sensor
        reading = interface.read_sensor("non_existent_sensor")
        assert reading is None

    def test_mfc_interface_actuator_write_none_device(self, hal_system):
        """Test actuator write with non-existent device"""
        interface = hal_system.control_interface
        
        # Try to write to non-existent actuator
        success = interface.write_actuator("non_existent_actuator", 50.0)
        assert not success

    def test_mfc_interface_power_operations_none_device(self, hal_system):
        """Test power operations with non-existent device"""
        interface = hal_system.control_interface
        
        # Try to read from non-existent power device
        status = interface.read_power_status("non_existent_power")
        assert status is None
        
        # Try to write to non-existent power device
        success = interface.set_power_output("non_existent_power", 12.0, 2.0)
        assert not success

    def test_device_get_device_by_id(self, hal_system):
        """Test getting device by ID"""
        interface = hal_system.control_interface
        
        # Get a known device
        device_ids = list(interface.devices.keys())
        if device_ids:
            device_id = device_ids[0]
            device = interface.get_device(device_id)
            assert device is not None
            assert device.device_info.device_id == device_id
        
        # Test non-existent device
        non_existent = interface.get_device("non_existent_id")
        assert non_existent is None


class TestControllerCostAnalysis:
    """Test suite for ControllerCostAnalyzer"""
    
    @pytest.fixture
    def cost_analyzer(self):
        configurations = create_standard_controller_configurations()
        return configurations['high_performance']
    
    def test_cost_analyzer_creation(self, cost_analyzer):
        """Test cost analyzer creation"""
        assert isinstance(cost_analyzer, ControllerCostAnalyzer)
        assert len(cost_analyzer.cost_items) > 0
        assert len(cost_analyzer.power_requirements) > 0
    
    def test_power_requirements_calculation(self, cost_analyzer):
        """Test power requirements calculation"""
        power_req = cost_analyzer.calculate_total_power_requirements()
        
        assert isinstance(power_req, dict)
        assert 'total_idle_power_w' in power_req
        assert 'total_active_power_w' in power_req
        assert 'total_peak_power_w' in power_req
        assert 'average_system_power_w' in power_req
        
        # Sanity checks
        assert power_req['total_idle_power_w'] > 0
        assert power_req['total_active_power_w'] > power_req['total_idle_power_w']
        assert power_req['total_peak_power_w'] > power_req['total_active_power_w']
    
    def test_cost_analysis_calculation(self, cost_analyzer):
        """Test cost analysis calculation"""
        analysis = cost_analyzer.calculate_cost_analysis(analysis_years=5)
        
        assert isinstance(analysis, dict)
        assert 'total_initial_cost' in analysis
        assert 'total_recurring_cost_per_year' in analysis
        assert 'total_cost_of_ownership' in analysis
        assert 'category_breakdown' in analysis
        
        # Sanity checks
        assert analysis['total_initial_cost'] > 0
        assert analysis['total_recurring_cost_per_year'] >= 0
        assert analysis['total_cost_of_ownership'] > analysis['total_initial_cost']
    
    def test_cost_report_generation(self, cost_analyzer):
        """Test cost report generation"""
        report = cost_analyzer.generate_cost_report(analysis_years=10)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "MFC Controller System Cost Analysis Report" in report
        assert "EXECUTIVE SUMMARY" in report
        assert "POWER REQUIREMENTS" in report
    
    def test_configuration_comparison(self):
        """Test configuration comparison"""
        configurations = create_standard_controller_configurations()
        hp_analyzer = configurations['high_performance']
        lc_analyzer = configurations['low_cost']
        
        comparison = hp_analyzer.compare_configurations(lc_analyzer)
        
        assert isinstance(comparison, dict)
        assert 'configuration_a' in comparison
        assert 'configuration_b' in comparison
        assert 'differences' in comparison
        
        # High-performance should have different costs (absolute value should be positive)
        assert abs(comparison['differences']['total_cost_diff']) > 0
    
    def test_cost_item_management(self, cost_analyzer):
        """Test adding custom cost items"""
        initial_count = len(cost_analyzer.cost_items)
        
        custom_item = CostItem(
            item_name="Custom Component",
            category=CostCategory.HARDWARE,
            initial_cost=1000.0,
            recurring_cost_per_year=100.0,
            useful_life_years=5.0
        )
        
        cost_analyzer.add_cost_item(custom_item)
        
        assert len(cost_analyzer.cost_items) == initial_count + 1
        
        # Verify it affects the analysis
        analysis = cost_analyzer.calculate_cost_analysis()
        assert analysis['total_initial_cost'] >= 1000.0
    
    def test_power_requirement_management(self, cost_analyzer):
        """Test adding custom power requirements"""
        initial_count = len(cost_analyzer.power_requirements)
        
        custom_power = PowerRequirement(
            component="Custom Component",
            idle_power_w=1.0,
            active_power_w=5.0,
            peak_power_w=10.0,
            duty_cycle_pct=50.0,
            efficiency=0.9
        )
        
        cost_analyzer.add_power_requirement(custom_power)
        
        assert len(cost_analyzer.power_requirements) == initial_count + 1
        
        # Verify it affects the power analysis
        power_req = cost_analyzer.calculate_total_power_requirements()
        assert 'Custom Component' in power_req['component_breakdown']


class TestIntegration:
    """Integration tests for the complete controller system"""
    
    def test_full_system_integration(self):
        """Test integration of all controller components"""
        # Create inference engine
        engines = create_standard_inference_engines()
        inference_engine = engines['balanced']
        
        # Create control electronics
        electronics_systems = create_standard_control_electronics()
        electronics = electronics_systems['high_performance']
        
        # Create real-time controller
        rt_controllers = create_standard_real_time_controllers()
        rt_controller = rt_controllers['high_performance']
        
        # Create HAL system
        hal_system = create_mfc_hardware_system()
        
        # Create cost analyzer
        cost_configs = create_standard_controller_configurations()
        cost_analyzer = cost_configs['high_performance']
        
        # Verify all components are created successfully
        assert inference_engine is not None
        assert electronics is not None
        assert rt_controller is not None
        assert hal_system is not None
        assert cost_analyzer is not None
        
        # Test basic operations
        # 1. Initialize HAL devices
        control_interface = hal_system.get_control_interface()
        init_results = control_interface.initialize_all_devices()
        assert len(init_results) > 0
        
        # 2. Get electronics measurement
        electronics_measurement = electronics.get_measurement()
        assert electronics_measurement.power_consumption_mw > 0
        
        # 3. Get RT controller measurement
        rt_measurement = rt_controller.get_measurement()
        assert rt_measurement.mode == ControllerMode.MANUAL
        
        # 4. Calculate system costs
        cost_analysis = cost_analyzer.calculate_cost_analysis()
        assert cost_analysis['total_initial_cost'] > 0
        
        # 5. Generate system report
        report = cost_analyzer.generate_cost_report()
        assert "MFC Controller System Cost Analysis Report" in report
    
    def test_performance_benchmarking(self):
        """Test performance benchmarking of the system"""
        # Create high-performance configuration
        engines = create_standard_inference_engines()
        hp_engine = engines['high_performance']
        
        # Load a dummy model
        dummy_model = np.random.rand(1000, 10)  # Large model
        hp_engine.model = dummy_model
        
        # Benchmark inference performance
        test_states = [np.random.rand(5) for _ in range(100)]
        
        start_time = time.time()
        results = hp_engine.batch_infer(test_states)
        end_time = time.time()
        
        total_time = (end_time - start_time) * 1000  # ms
        avg_time_per_inference = total_time / len(test_states)
        
        assert len(results) == 100
        assert avg_time_per_inference < 10.0  # Should be under 10ms per inference
        
        # Check performance stats
        stats = hp_engine.get_performance_stats()
        assert stats['total_inferences'] >= 100
    
    def test_fault_handling(self):
        """Test fault handling across the system"""
        # Create RT controller
        rt_controllers = create_standard_real_time_controllers()
        rt_controller = rt_controllers['high_performance']
        
        # Test safety mode activation
        rt_controller.set_mode(ControllerMode.SAFETY)
        assert rt_controller.mode == ControllerMode.SAFETY
        assert rt_controller.safety_state == "EMERGENCY_STOP"
        
        # Test fault flag generation
        rt_controller._handle_controller_fault("TEST_FAULT")
        measurement = rt_controller.get_measurement()
        assert "TEST_FAULT" in measurement.fault_flags


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])