"""
Comprehensive tests for controller models
"""

import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch
import tempfile
import json




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
            
            success = engine.load_model(f.name)
            assert success
            assert engine.model is not None
            assert isinstance(engine.model, dict)
    
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
        assert control_electronics.status == DeviceStatus.OFFLINE
    
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
        
        # Test out-of-range value (should be clipped)
        success = control_electronics.write_dac(0, 10.0)
        assert success
        assert control_electronics.dac_outputs[0] <= control_electronics.dac_specs.reference_voltage_v
        
        # Test invalid channel
        success = control_electronics.write_dac(999, 1.0)
        assert not success
    
    def test_gpio_operations(self, control_electronics):
        """Test GPIO operations"""
        # Set GPIO pin
        success = control_electronics.set_gpio(0, True, "output")
        assert success
        assert control_electronics.gpio_states[0] == True
        assert control_electronics.gpio_directions[0] == "output"
        
        # Read GPIO pin
        state = control_electronics.read_gpio(0)
        assert state == True
        
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
        
        # High-performance should cost more
        assert comparison['differences']['total_cost_diff'] > 0
    
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