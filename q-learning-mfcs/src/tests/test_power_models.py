"""
Comprehensive tests for power models in MFC systems

This module tests all power-related components including boost controllers,
battery systems, supercapacitors, electronic loads, supercapacitor controllers,
and the complete power management system.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.power_models import (
    # Boost Controller
    BoostController, BoostControllerSpecs, create_standard_boost_controllers,
    # Battery Storage
    BatterySystem, BatterySpecs, BatteryChemistry, create_standard_battery_systems,
    # Supercapacitor Storage
    SupercapacitorSystem, SupercapacitorSpecs, SupercapacitorType, 
    create_standard_supercapacitor_systems,
    # Electronic Load
    ElectronicLoad, ElectronicLoadSpecs, LoadMode, create_standard_electronic_loads,
    # Supercapacitor Controller
    SupercapacitorController, SupercapacitorControllerSpecs, ControllerTopology,
    ControllerControlMode, create_standard_supercapacitor_controllers,
    # Power Management System
    PowerManagementSystem, PowerSystemConfiguration, PowerManagementMode,
    EnergyManagementStrategy, create_standard_power_management_systems
)


class TestBoostController:
    """Test boost controller functionality"""
    
    def test_boost_controller_creation(self):
        """Test boost controller creation with valid specs"""
        specs = BoostControllerSpecs(
            topology="boost",
            control_mode="voltage",
            input_voltage_range=(1.0, 5.0),
            output_voltage_range=(5.0, 15.0),
            max_current=10.0,
            max_power=50.0,
            switching_frequency=100000.0,
            efficiency=0.90,
            ripple_current=10.0,
            ripple_voltage=2.0,
            response_time=0.001,
            power_consumption=1.0,
            cost=100.0,
            mass=0.2,
            volume=0.1
        )
        
        controller = BoostController(specs)
        assert controller.specs == specs
        assert controller.duty_cycle == 0.5
        assert controller.efficiency > 0
    
    def test_boost_controller_operation(self):
        """Test boost controller operation"""
        controllers = create_standard_boost_controllers()
        controller = controllers['high_efficiency_boost']
        
        # Test control step
        measurement = controller.control_step(
            input_voltage=3.0,
            load_current=2.0,
            dt=0.001,
            temperature=25.0
        )
        
        assert measurement.input_voltage == 3.0
        assert measurement.output_voltage > measurement.input_voltage  # Boost operation
        assert measurement.efficiency > 0.7
        assert measurement.duty_cycle > 0.0
        assert measurement.duty_cycle < 1.0
    
    def test_boost_controller_spice_generation(self):
        """Test SPICE model generation"""
        controllers = create_standard_boost_controllers()
        controller = controllers['high_efficiency_boost']
        
        spice_model = controller.generate_spice_model()
        assert len(spice_model) > 1000  # Should be substantial
        assert 'boost_controller' in spice_model.lower()
        assert '.subckt' in spice_model
        assert '.ends' in spice_model
    
    def test_boost_controller_cost_analysis(self):
        """Test cost analysis"""
        controllers = create_standard_boost_controllers()
        controller = controllers['high_efficiency_boost']
        
        cost_analysis = controller.get_cost_analysis()
        assert 'initial_cost' in cost_analysis
        assert 'total_cost_per_hour' in cost_analysis
        assert 'lifetime_cost' in cost_analysis
        assert cost_analysis['initial_cost'] > 0
        assert cost_analysis['total_cost_per_hour'] > 0


class TestBatterySystem:
    """Test battery system functionality"""
    
    def test_battery_system_creation(self):
        """Test battery system creation"""
        specs = BatterySpecs(
            chemistry=BatteryChemistry.LITHIUM_ION,
            nominal_voltage=3.7,
            capacity=10.0,
            max_charge_current=5.0,
            max_discharge_current=10.0,
            charge_voltage=4.2,
            discharge_cutoff=3.0,
            internal_resistance=0.1,
            cycle_life=2000,
            self_discharge_rate=0.1,
            operating_temp_range=(-20, 60),
            power_consumption=0.1,
            cost=200.0,
            mass=0.5,
            volume=0.2
        )
        
        battery = BatterySystem(specs)
        assert battery.specs == specs
        assert battery.voltage == specs.nominal_voltage
        assert battery.get_state_of_charge() == 0.5  # Default 50% SOC
    
    def test_battery_charging(self):
        """Test battery charging"""
        systems = create_standard_battery_systems()
        battery = systems['lithium_ion_standard']
        
        initial_soc = battery.get_state_of_charge()
        
        # Charge for 1 hour at 1A
        measurement = battery.charge(
            charging_current=1.0,
            dt=3600.0,
            temperature=25.0
        )
        
        assert measurement.current > 0  # Charging
        assert battery.get_state_of_charge() > initial_soc
        assert measurement.state_of_charge == battery.get_state_of_charge()
    
    def test_battery_discharging(self):
        """Test battery discharging"""
        systems = create_standard_battery_systems()
        battery = systems['lithium_ion_standard']
        
        # Start with higher SOC
        battery.set_state_of_charge(0.8)
        initial_soc = battery.get_state_of_charge()
        
        # Discharge for 1 hour at 2A
        measurement = battery.discharge(
            load_current=2.0,
            dt=3600.0,
            temperature=25.0
        )
        
        assert measurement.current > 0  # Discharging
        assert battery.get_state_of_charge() < initial_soc
        assert measurement.power > 0
    
    def test_battery_spice_generation(self):
        """Test battery SPICE model generation"""
        systems = create_standard_battery_systems()
        battery = systems['lithium_ion_standard']
        
        spice_model = battery.generate_spice_model()
        assert len(spice_model) > 500
        assert 'battery' in spice_model.lower()
        assert 'lithium' in spice_model.lower()


class TestSupercapacitorSystem:
    """Test supercapacitor system functionality"""
    
    def test_supercapacitor_creation(self):
        """Test supercapacitor creation"""
        specs = SupercapacitorSpecs(
            capacitor_type=SupercapacitorType.EDLC,
            rated_capacitance=100.0,
            rated_voltage=2.7,
            max_current=50.0,
            esr=0.02,
            esl=5e-9,
            leakage_current=0.005,
            cycle_life=1000000,
            operating_temp_range=(-40, 65),
            power_consumption=0.2,
            cost=200.0,
            mass=0.5,
            volume=0.2
        )
        
        supercap = SupercapacitorSystem(specs)
        assert supercap.specs == specs
        assert supercap.voltage == 0.0  # Starts discharged
        assert supercap.get_state_of_charge() == 0.0
    
    def test_supercapacitor_charging(self):
        """Test supercapacitor charging"""
        systems = create_standard_supercapacitor_systems()
        supercap = systems['edlc_medium']
        
        initial_soc = supercap.get_state_of_charge()
        
        # Charge for 10 seconds at 10A
        measurement = supercap.charge(
            charging_current=10.0,
            dt=10.0,
            temperature=25.0
        )
        
        assert measurement.current == 10.0
        assert supercap.get_state_of_charge() > initial_soc
        assert measurement.voltage > 0
        assert measurement.stored_energy > 0
    
    def test_supercapacitor_discharging(self):
        """Test supercapacitor discharging"""
        systems = create_standard_supercapacitor_systems()
        supercap = systems['edlc_medium']
        
        # Charge first
        supercap.set_voltage(2.0)  # Set to 2V
        initial_energy = supercap.stored_energy
        
        # Discharge for 5 seconds at 5A
        measurement = supercap.discharge(
            load_current=5.0,
            dt=5.0,
            temperature=25.0
        )
        
        assert measurement.current == 5.0
        assert supercap.stored_energy < initial_energy
        assert measurement.power > 0
    
    def test_supercapacitor_spice_generation(self):
        """Test supercapacitor SPICE model generation"""
        systems = create_standard_supercapacitor_systems()
        supercap = systems['edlc_medium']
        
        spice_model = supercap.generate_spice_model()
        assert len(spice_model) > 1000
        assert 'supercap' in spice_model.lower()
        assert 'capacitance' in spice_model.lower()


class TestElectronicLoad:
    """Test electronic load functionality"""
    
    def test_electronic_load_creation(self):
        """Test electronic load creation"""
        specs = ElectronicLoadSpecs(
            load_type="active",
            max_voltage=50.0,
            max_current=20.0,
            max_power=500.0,
            min_voltage=1.0,
            current_resolution=0.001,
            voltage_resolution=0.001,
            power_resolution=0.01,
            slew_rate=1000.0,
            operating_modes=[LoadMode.CC, LoadMode.CV, LoadMode.CP],
            power_consumption=10.0,
            cost=800.0,
            mass=2.0,
            volume=1.0
        )
        
        load = ElectronicLoad(specs)
        assert load.specs == specs
        assert load.current_mode == LoadMode.CC
    
    def test_electronic_load_operation(self):
        """Test electronic load operation"""
        loads = create_standard_electronic_loads()
        load = loads['active_load_medium']
        
        # Test constant current mode
        load.set_operating_mode(LoadMode.CC, setpoint=2.0)
        measurement = load.apply_load(
            input_voltage=12.0,
            dt=0.1,
            temperature=25.0
        )
        
        assert abs(measurement.current - 2.0) < 0.1  # Should be close to setpoint
        assert measurement.voltage == 12.0
        assert measurement.power > 0
    
    def test_electronic_load_spice_generation(self):
        """Test electronic load SPICE model generation"""
        loads = create_standard_electronic_loads()
        load = loads['active_load_medium']
        
        spice_model = load.generate_spice_model()
        assert len(spice_model) > 1000
        assert 'electronic_load' in spice_model.lower()
        assert 'mosfet' in spice_model.lower()


class TestSupercapacitorController:
    """Test supercapacitor controller functionality"""
    
    def test_controller_creation(self):
        """Test controller creation"""
        specs = SupercapacitorControllerSpecs(
            topology=ControllerTopology.BUCK_BOOST,
            control_mode=ControllerControlMode.CC_CV,
            input_voltage_range=(5.0, 60.0),
            output_voltage_range=(1.0, 50.0),
            max_current=20.0,
            max_power=500.0,
            switching_frequency=100000.0,
            efficiency=0.92,
            ripple_current=5.0,
            ripple_voltage=2.0,
            response_time=0.001,
            power_consumption=2.0,
            cost=150.0,
            mass=0.3,
            volume=0.1
        )
        
        controller = SupercapacitorController(specs)
        assert controller.specs == specs
        assert controller.duty_cycle == 0.5
    
    def test_controller_operation(self):
        """Test controller operation"""
        controllers = create_standard_supercapacitor_controllers()
        controller = controllers['buck_boost_cc_cv']
        
        # Set control parameters
        controller.set_control_parameters(voltage_setpoint=12.0, current_setpoint=5.0)
        
        # Test control step
        measurement = controller.control_step(
            input_voltage=24.0,
            load_current=3.0,
            dt=0.001,
            temperature=25.0
        )
        
        assert measurement.input_voltage == 24.0
        assert measurement.efficiency > 0.8
        assert measurement.fault_status in ['ok', 'warning', 'fault']
    
    def test_controller_spice_generation(self):
        """Test controller SPICE model generation"""
        controllers = create_standard_supercapacitor_controllers()
        controller = controllers['buck_boost_cc_cv']
        
        spice_model = controller.generate_spice_model()
        assert len(spice_model) > 2000  # Should be comprehensive
        assert 'buck_boost' in spice_model.lower()
        assert 'controller' in spice_model.lower()


class TestPowerManagementSystem:
    """Test comprehensive power management system"""
    
    def test_power_management_creation(self):
        """Test power management system creation"""
        config = PowerSystemConfiguration(
            management_mode=PowerManagementMode.STANDALONE,
            energy_strategy=EnergyManagementStrategy.BALANCED,
            enable_battery=True,
            enable_supercapacitor=True,
            max_charge_rate=50.0,
            max_discharge_rate=100.0
        )
        
        pms = PowerManagementSystem(config)
        assert pms.config == config
        assert pms.target_bus_voltage == 12.0
        assert len(pms.fault_flags) > 0
    
    def test_power_management_operation(self):
        """Test power management system operation"""
        systems = create_standard_power_management_systems()
        pms = systems['standalone_balanced']
        
        # Test power management cycle
        measurement = pms.manage_power(
            mfc_voltage=2.5,
            mfc_current=2.0,
            load_power_demand=3.0,
            dt=1.0,
            temperature=25.0
        )
        
        assert measurement.mfc_voltage == 2.5
        assert measurement.mfc_current == 2.0
        assert measurement.mfc_power == 5.0  # 2.5V * 2.0A
        assert measurement.load_power == 3.0
        assert measurement.system_efficiency > 0
        assert measurement.system_status in ['normal', 'warning', 'alarm']
    
    def test_energy_management_strategies(self):
        """Test different energy management strategies"""
        strategies = [
            EnergyManagementStrategy.LOAD_PRIORITY,
            EnergyManagementStrategy.BATTERY_PRIORITY,
            EnergyManagementStrategy.SUPERCAP_PRIORITY,
            EnergyManagementStrategy.BALANCED,
            EnergyManagementStrategy.EFFICIENCY_OPTIMIZED
        ]
        
        for strategy in strategies:
            config = PowerSystemConfiguration(
                management_mode=PowerManagementMode.STANDALONE,
                energy_strategy=strategy,
                enable_battery=True,
                enable_supercapacitor=True
            )
            
            pms = PowerManagementSystem(config)
            measurement = pms.manage_power(
                mfc_voltage=3.0,
                mfc_current=1.5,
                load_power_demand=2.0,
                dt=1.0,
                temperature=25.0
            )
            
            assert measurement.energy_strategy == strategy
            assert measurement.total_input_power > 0
    
    def test_fault_detection(self):
        """Test fault detection in power management system"""
        systems = create_standard_power_management_systems()
        pms = systems['standalone_balanced']
        
        # Test with high temperature (should trigger fault)
        measurement = pms.manage_power(
            mfc_voltage=2.0,
            mfc_current=1.0,
            load_power_demand=1.0,
            dt=1.0,
            temperature=80.0  # High temperature
        )
        
        assert measurement.fault_flags['overtemperature'] == True
        assert measurement.system_status in ['warning', 'alarm']
    
    def test_power_management_spice_generation(self):
        """Test power management system SPICE model generation"""
        systems = create_standard_power_management_systems()
        pms = systems['standalone_balanced']
        
        spice_model = pms.generate_spice_model()
        assert len(spice_model) > 3000  # Should be very comprehensive
        assert 'power_management_system' in spice_model.lower()
        assert 'testbench' in spice_model.lower()
        assert 'efficiency' in spice_model.lower()
    
    def test_cost_analysis(self):
        """Test comprehensive cost analysis"""
        systems = create_standard_power_management_systems()
        pms = systems['standalone_balanced']
        
        cost_analysis = pms.get_cost_analysis()
        assert 'total_initial_cost' in cost_analysis
        assert 'component_costs' in cost_analysis
        assert 'integration_cost' in cost_analysis
        assert 'lifetime_cost_10_years' in cost_analysis
        assert cost_analysis['total_initial_cost'] > 0
        assert cost_analysis['lifetime_cost_10_years'] > cost_analysis['total_initial_cost']


class TestIntegrationScenarios:
    """Test integration scenarios across all power components"""
    
    def test_complete_system_integration(self):
        """Test complete system with all components"""
        # Create individual components
        boost_controllers = create_standard_boost_controllers()
        battery_systems = create_standard_battery_systems()
        supercap_systems = create_standard_supercapacitor_systems()
        supercap_controllers = create_standard_supercapacitor_controllers()
        electronic_loads = create_standard_electronic_loads()
        
        # Create power management system
        config = PowerSystemConfiguration(
            management_mode=PowerManagementMode.STANDALONE,
            energy_strategy=EnergyManagementStrategy.BALANCED,
            enable_battery=True,
            enable_supercapacitor=True
        )
        
        pms = PowerManagementSystem(
            config=config,
            boost_controller=boost_controllers['high_efficiency_boost'],
            battery_system=battery_systems['lithium_ion_standard'],
            supercap_system=supercap_systems['edlc_medium'],
            supercap_controller=supercap_controllers['buck_boost_cc_cv'],
            electronic_load=electronic_loads['active_load_medium']
        )
        
        # Run simulation for multiple time steps
        for i in range(10):
            measurement = pms.manage_power(
                mfc_voltage=2.0 + 0.5 * np.sin(i * 0.1),
                mfc_current=1.0 + 0.3 * np.cos(i * 0.2),
                load_power_demand=3.0 + np.sin(i * 0.3),
                dt=1.0,
                temperature=25.0 + i * 0.5
            )
            
            # Verify basic functionality
            assert measurement.total_input_power > 0
            assert measurement.system_efficiency > 0
            assert measurement.system_status in ['normal', 'warning', 'alarm']
    
    def test_energy_flow_tracking(self):
        """Test energy flow tracking across components"""
        systems = create_standard_power_management_systems()
        pms = systems['energy_harvesting']
        
        # Run for several cycles to track energy flows
        total_generated = 0.0
        total_consumed = 0.0
        
        for i in range(20):
            mfc_power = 2.0 + np.sin(i * 0.1)
            load_power = 1.5 + 0.5 * np.cos(i * 0.2)
            
            measurement = pms.manage_power(
                mfc_voltage=2.0,
                mfc_current=mfc_power / 2.0,
                load_power_demand=load_power,
                dt=1.0,
                temperature=25.0
            )
            
            total_generated += measurement.total_input_power
            total_consumed += measurement.load_power
        
        # Energy accounting should be reasonable
        assert total_generated > 0
        assert total_consumed > 0
        assert pms.energy_generated > 0
        assert pms.energy_consumed > 0
    
    def test_spice_model_compatibility(self):
        """Test SPICE model compatibility across components"""
        # Create all component types
        boost_controllers = create_standard_boost_controllers()
        battery_systems = create_standard_battery_systems()
        supercap_systems = create_standard_supercapacitor_systems()
        supercap_controllers = create_standard_supercapacitor_controllers()
        electronic_loads = create_standard_electronic_loads()
        power_management_systems = create_standard_power_management_systems()
        
        # Generate SPICE models for all components
        spice_models = {}
        
        # Test each component type
        for name, component in boost_controllers.items():
            spice_models[f'boost_{name}'] = component.generate_spice_model()
        
        for name, component in battery_systems.items():
            spice_models[f'battery_{name}'] = component.generate_spice_model()
        
        for name, component in supercap_systems.items():
            spice_models[f'supercap_{name}'] = component.generate_spice_model()
        
        for name, component in supercap_controllers.items():
            spice_models[f'controller_{name}'] = component.generate_spice_model()
        
        for name, component in electronic_loads.items():
            spice_models[f'load_{name}'] = component.generate_spice_model()
        
        for name, component in power_management_systems.items():
            spice_models[f'pms_{name}'] = component.generate_spice_model()
        
        # Verify all models are generated
        assert len(spice_models) > 15  # Should have many models
        
        # Verify all models contain essential SPICE elements
        for model_name, spice_code in spice_models.items():
            assert len(spice_code) > 100  # Should be substantial
            assert any(keyword in spice_code.lower() for keyword in 
                      ['.subckt', '.model', '.include', '.end'])


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])