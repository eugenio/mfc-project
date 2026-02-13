"""
Comprehensive Power Management System for MFC Applications

This module implements a complete power management system that integrates MFC generators,
boost controllers, battery storage, supercapacitors, electronic loads, and charge controllers
with advanced energy management algorithms and comprehensive SPICE model generation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import logging

from .boost_controller import BoostController, BoostControllerSpecs
from .battery_storage import BatterySystem, BatterySpecs, BatteryChemistry
from .supercapacitor_storage import SupercapacitorSystem, SupercapacitorSpecs, SupercapacitorType
from .supercapacitor_controller import SupercapacitorController
from .electronic_load import ElectronicLoad

logger = logging.getLogger(__name__)


class PowerManagementMode(Enum):
    """Power management operating modes"""
    GRID_TIE = "grid_tie"  # Grid-tied operation
    STANDALONE = "standalone"  # Standalone operation
    HYBRID = "hybrid"  # Hybrid grid-tie/standalone
    ENERGY_HARVESTING = "energy_harvesting"  # Pure energy harvesting
    LOAD_FOLLOWING = "load_following"  # Load following mode
    POWER_SMOOTHING = "power_smoothing"  # Power smoothing mode


class EnergyManagementStrategy(Enum):
    """Energy management strategies"""
    BATTERY_PRIORITY = "battery_priority"  # Prioritize battery charging
    SUPERCAP_PRIORITY = "supercap_priority"  # Prioritize supercapacitor charging
    BALANCED = "balanced"  # Balanced energy storage
    LOAD_PRIORITY = "load_priority"  # Prioritize load supply
    EFFICIENCY_OPTIMIZED = "efficiency_optimized"  # Optimize for efficiency
    COST_OPTIMIZED = "cost_optimized"  # Optimize for cost


@dataclass
class PowerSystemConfiguration:
    """Configuration for power management system"""
    management_mode: PowerManagementMode
    energy_strategy: EnergyManagementStrategy
    enable_battery: bool = True
    enable_supercapacitor: bool = True
    enable_grid_interface: bool = False
    max_charge_rate: float = 100.0  # W
    max_discharge_rate: float = 200.0  # W
    voltage_regulation_tolerance: float = 0.05  # ±5%
    efficiency_threshold: float = 0.80  # Minimum efficiency
    safety_margins: Dict[str, float] = field(default_factory=lambda: {
        'voltage_margin': 0.1,  # 10% voltage margin
        'current_margin': 0.2,  # 20% current margin
        'temperature_margin': 10.0  # 10°C temperature margin
    })


@dataclass
class PowerSystemMeasurement:
    """Comprehensive power system measurement"""
    timestamp: float
    mfc_voltage: float  # V
    mfc_current: float  # A
    mfc_power: float  # W
    battery_voltage: float  # V
    battery_current: float  # A
    battery_soc: float  # 0-1
    supercap_voltage: float  # V
    supercap_current: float  # A
    supercap_soc: float  # 0-1
    load_voltage: float  # V
    load_current: float  # A
    load_power: float  # W
    total_input_power: float  # W
    total_output_power: float  # W
    system_efficiency: float  # 0-1
    energy_balance: float  # W (positive = surplus, negative = deficit)
    management_mode: PowerManagementMode
    energy_strategy: EnergyManagementStrategy
    system_status: str  # 'normal', 'warning', 'alarm'
    fault_flags: Dict[str, bool]


@dataclass
class EnergyFlow:
    """Energy flow tracking"""
    source: str
    destination: str
    power: float  # W
    efficiency: float  # 0-1
    timestamp: float


class PowerManagementSystem:
    """Comprehensive power management system"""
    
    def __init__(self, 
                 config: PowerSystemConfiguration,
                 boost_controller: Optional[BoostController] = None,
                 battery_system: Optional[BatterySystem] = None,
                 supercap_system: Optional[SupercapacitorSystem] = None,
                 supercap_controller: Optional[SupercapacitorController] = None,
                 electronic_load: Optional[ElectronicLoad] = None):
        
        self.config = config
        self.boost_controller = boost_controller
        self.battery_system = battery_system
        self.supercap_system = supercap_system
        self.supercap_controller = supercap_controller
        self.electronic_load = electronic_load
        
        # System state
        self.operating_hours = 0.0
        self.temperature = 25.0
        self.grid_voltage = 0.0
        self.grid_available = False
        
        # Energy management variables
        self.target_bus_voltage = 12.0  # V
        self.load_power_demand = 0.0  # W
        self.energy_flows: List[EnergyFlow] = []
        
        # Control algorithms
        self.voltage_controller = PIController(kp=0.1, ki=10.0)
        self.power_controller = PIController(kp=0.05, ki=5.0)
        self.soc_controller = PIController(kp=0.02, ki=1.0)
        
        # Fault detection
        self.fault_flags = {
            'overvoltage': False,
            'undervoltage': False,
            'overcurrent': False,
            'overtemperature': False,
            'battery_fault': False,
            'supercap_fault': False,
            'controller_fault': False,
            'efficiency_fault': False
        }
        
        # Energy accounting
        self.energy_stored = 0.0  # Wh
        self.energy_consumed = 0.0  # Wh
        self.energy_generated = 0.0  # Wh
        self.energy_wasted = 0.0  # Wh
        
        # Initialize default components if not provided
        self._initialize_default_components()
        
    def _initialize_default_components(self):
        """Initialize default components if not provided"""
        if self.boost_controller is None and self.config.enable_battery:
            # Create default boost controller
            from .boost_controller import SwitchingTopology
            boost_specs = BoostControllerSpecs(
                topology=SwitchingTopology.BOOST,
                input_voltage_range=(0.5, 5.0),
                output_voltage_range=(5.0, 15.0),
                max_output_current=10.0,
                max_output_power=50.0,
                switching_frequency=100000.0,
                efficiency_at_rated=0.90,
                quiescent_current=10.0,
                cost=100.0,
                lifetime=50000.0,
                thermal_resistance=20.0,
                inductance=100e-6,
                capacitance_input=100e-6,
                capacitance_output=220e-6,
                control_bandwidth=10000.0
            )
            self.boost_controller = BoostController(boost_specs)
            
        if self.battery_system is None and self.config.enable_battery:
            # Create default battery system
            battery_specs = BatterySpecs(
                chemistry=BatteryChemistry.LITHIUM_ION,
                nominal_capacity=10.0,
                nominal_voltage=3.7,
                max_charge_current=5.0,
                max_discharge_current=10.0,
                charge_voltage=4.2,
                cutoff_voltage=3.0,
                internal_resistance=0.1,
                self_discharge_rate=0.1,
                cycle_life=2000,
                temperature_range=(-20, 60),
                cost=200.0,
                weight=0.5,
                energy_density=150.0,
                power_density=300.0
            )
            self.battery_system = BatterySystem(battery_specs)
            
        if self.supercap_system is None and self.config.enable_supercapacitor:
            # Create default supercapacitor system
            supercap_specs = SupercapacitorSpecs(
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
            self.supercap_system = SupercapacitorSystem(supercap_specs)
    
    def manage_power(self, mfc_voltage: float, mfc_current: float, 
                    load_power_demand: float, dt: float, 
                    temperature: float = 25.0) -> PowerSystemMeasurement:
        """
        Execute power management cycle
        
        Args:
            mfc_voltage: MFC output voltage (V)
            mfc_current: MFC output current (A)
            load_power_demand: Required load power (W)
            dt: Time step (seconds)
            temperature: Operating temperature (°C)
            
        Returns:
            PowerSystemMeasurement object
        """
        self.temperature = temperature
        self.operating_hours += dt / 3600.0
        self.load_power_demand = load_power_demand
        
        # Clear energy flows for this cycle
        self.energy_flows.clear()
        
        # Calculate MFC power
        mfc_power = mfc_voltage * mfc_current
        
        # Execute energy management strategy
        power_allocation = self._execute_energy_management_strategy(
            mfc_power, load_power_demand, dt
        )
        
        # Control power converters
        converter_outputs = self._control_power_converters(
            mfc_voltage, mfc_current, power_allocation, dt
        )
        
        # Manage energy storage
        storage_states = self._manage_energy_storage(
            power_allocation, dt, temperature
        )
        
        # Calculate system performance metrics
        metrics = self._calculate_system_metrics(
            mfc_power, load_power_demand, storage_states, converter_outputs
        )
        
        # Fault detection and handling
        self._detect_and_handle_faults(metrics, temperature)
        
        # Update energy accounting
        self._update_energy_accounting(mfc_power, load_power_demand, dt)
        
        return PowerSystemMeasurement(
            timestamp=self.operating_hours * 3600,
            mfc_voltage=mfc_voltage,
            mfc_current=mfc_current,
            mfc_power=mfc_power,
            battery_voltage=storage_states.get('battery_voltage', 0.0),
            battery_current=storage_states.get('battery_current', 0.0),
            battery_soc=storage_states.get('battery_soc', 0.0),
            supercap_voltage=storage_states.get('supercap_voltage', 0.0),
            supercap_current=storage_states.get('supercap_current', 0.0),
            supercap_soc=storage_states.get('supercap_soc', 0.0),
            load_voltage=converter_outputs.get('load_voltage', 0.0),
            load_current=converter_outputs.get('load_current', 0.0),
            load_power=load_power_demand,
            total_input_power=mfc_power,
            total_output_power=metrics['total_output_power'],
            system_efficiency=metrics['system_efficiency'],
            energy_balance=metrics['energy_balance'],
            management_mode=self.config.management_mode,
            energy_strategy=self.config.energy_strategy,
            system_status=metrics['system_status'],
            fault_flags=self.fault_flags.copy()
        )
    
    def _execute_energy_management_strategy(self, mfc_power: float, 
                                          load_power: float, dt: float) -> Dict[str, float]:
        """Execute the configured energy management strategy"""
        power_allocation = {
            'load_power': 0.0,
            'battery_power': 0.0,
            'supercap_power': 0.0,
            'grid_power': 0.0,
            'waste_power': 0.0
        }
        
        available_power = mfc_power
        remaining_load = load_power
        
        if self.config.energy_strategy == EnergyManagementStrategy.LOAD_PRIORITY:
            # Satisfy load first, then charge storage
            power_allocation['load_power'] = min(available_power, remaining_load)
            available_power -= power_allocation['load_power']
            remaining_load -= power_allocation['load_power']
            
            # Distribute remaining power to storage
            if available_power > 0:
                battery_power, supercap_power = self._distribute_charging_power(
                    available_power, EnergyManagementStrategy.BALANCED
                )
                power_allocation['battery_power'] = battery_power
                power_allocation['supercap_power'] = supercap_power
            
            # If load not fully satisfied, discharge storage
            if remaining_load > 0:
                discharge_power = self._get_storage_discharge_power(remaining_load)
                power_allocation['battery_power'] -= discharge_power['battery']
                power_allocation['supercap_power'] -= discharge_power['supercap']
                power_allocation['load_power'] += discharge_power['total']
                
        elif self.config.energy_strategy == EnergyManagementStrategy.BATTERY_PRIORITY:
            # Charge battery first, then supercap, then load
            if self.battery_system and self.battery_system.state_of_charge / 100.0 < 0.9:
                battery_power = min(available_power, self.config.max_charge_rate)
                power_allocation['battery_power'] = battery_power
                available_power -= battery_power
            
            if available_power > 0 and self.supercap_system:
                supercap_power = min(available_power, self.config.max_charge_rate)
                power_allocation['supercap_power'] = supercap_power
                available_power -= supercap_power
                
            power_allocation['load_power'] = min(available_power, remaining_load)
            
        elif self.config.energy_strategy == EnergyManagementStrategy.SUPERCAP_PRIORITY:
            # Charge supercap first for fast response
            if self.supercap_system and self.supercap_system.get_state_of_charge() < 0.95:
                supercap_power = min(available_power, self.config.max_charge_rate)
                power_allocation['supercap_power'] = supercap_power
                available_power -= supercap_power
            
            power_allocation['load_power'] = min(available_power, remaining_load)
            available_power -= power_allocation['load_power']
            
            if available_power > 0 and self.battery_system:
                power_allocation['battery_power'] = min(available_power, self.config.max_charge_rate)
                
        elif self.config.energy_strategy == EnergyManagementStrategy.BALANCED:
            # Balanced approach
            power_allocation['load_power'] = min(available_power, remaining_load)
            available_power -= power_allocation['load_power']
            
            if available_power > 0:
                battery_power, supercap_power = self._distribute_charging_power(
                    available_power, EnergyManagementStrategy.BALANCED
                )
                power_allocation['battery_power'] = battery_power
                power_allocation['supercap_power'] = supercap_power
                
        elif self.config.energy_strategy == EnergyManagementStrategy.EFFICIENCY_OPTIMIZED:
            # Optimize for maximum system efficiency
            power_allocation = self._optimize_for_efficiency(mfc_power, load_power)
            
        elif self.config.energy_strategy == EnergyManagementStrategy.COST_OPTIMIZED:
            # Optimize for minimum operating cost
            power_allocation = self._optimize_for_cost(mfc_power, load_power)
        
        # Calculate waste power
        total_allocated = sum(max(0, p) for p in power_allocation.values())
        power_allocation['waste_power'] = max(0, mfc_power - total_allocated)
        
        return power_allocation
    
    def _distribute_charging_power(self, available_power: float, 
                                 strategy: EnergyManagementStrategy) -> Tuple[float, float]:
        """Distribute charging power between battery and supercapacitor"""
        battery_power = 0.0
        supercap_power = 0.0
        
        if strategy == EnergyManagementStrategy.BALANCED:
            # Distribute based on SOC difference from target
            if self.battery_system and self.supercap_system:
                battery_soc = self.battery_system.state_of_charge / 100.0
                supercap_soc = self.supercap_system.get_state_of_charge()
                
                # Target SOCs
                battery_target = 0.8
                supercap_target = 0.9
                
                battery_deficit = max(0, battery_target - battery_soc)
                supercap_deficit = max(0, supercap_target - supercap_soc)
                
                total_deficit = battery_deficit + supercap_deficit
                
                if total_deficit > 0:
                    battery_fraction = battery_deficit / total_deficit
                    supercap_fraction = supercap_deficit / total_deficit
                    
                    battery_power = available_power * battery_fraction
                    supercap_power = available_power * supercap_fraction
                else:
                    # Both at target, split evenly
                    battery_power = available_power * 0.5
                    supercap_power = available_power * 0.5
        
        # Apply power limits
        if self.battery_system:
            battery_power = min(battery_power, self.config.max_charge_rate)
        if self.supercap_system:
            supercap_power = min(supercap_power, self.config.max_charge_rate)
            
        return battery_power, supercap_power
    
    def _get_storage_discharge_power(self, required_power: float) -> Dict[str, float]:
        """Get available discharge power from storage systems"""
        discharge_power = {'battery': 0.0, 'supercap': 0.0, 'total': 0.0}
        
        # Prioritize supercapacitor for fast response
        if self.supercap_system:
            supercap_available = self.supercap_system.get_power_capability(60)  # 1 minute capability
            supercap_power = min(required_power, supercap_available)
            discharge_power['supercap'] = supercap_power
            required_power -= supercap_power
        
        # Use battery for remaining power
        if required_power > 0 and self.battery_system:
            battery_available = self.battery_system.get_power_capability(3600)  # 1 hour capability
            battery_power = min(required_power, battery_available)
            discharge_power['battery'] = battery_power
            required_power -= battery_power
        
        discharge_power['total'] = discharge_power['battery'] + discharge_power['supercap']
        return discharge_power
    
    def _optimize_for_efficiency(self, mfc_power: float, load_power: float) -> Dict[str, float]:
        """Optimize power allocation for maximum efficiency"""
        # Simple efficiency optimization
        power_allocation = {
            'load_power': min(mfc_power, load_power),
            'battery_power': 0.0,
            'supercap_power': 0.0,
            'grid_power': 0.0,
            'waste_power': 0.0
        }
        
        remaining_power = mfc_power - power_allocation['load_power']
        
        # Choose storage with better efficiency
        if remaining_power > 0:
            if self.battery_system and self.supercap_system:
                # Supercapacitors typically have better round-trip efficiency
                if self.supercap_system.get_state_of_charge() < 0.9:
                    power_allocation['supercap_power'] = min(remaining_power, self.config.max_charge_rate)
                    remaining_power -= power_allocation['supercap_power']
                
                if remaining_power > 0:
                    power_allocation['battery_power'] = min(remaining_power, self.config.max_charge_rate)
        
        return power_allocation
    
    def _optimize_for_cost(self, mfc_power: float, load_power: float) -> Dict[str, float]:
        """Optimize power allocation for minimum cost"""
        # Simple cost optimization - prioritize cheaper storage
        power_allocation = {
            'load_power': min(mfc_power, load_power),
            'battery_power': 0.0,
            'supercap_power': 0.0,
            'grid_power': 0.0,
            'waste_power': 0.0
        }
        
        remaining_power = mfc_power - power_allocation['load_power']
        
        # Battery typically has lower cost per Wh
        if remaining_power > 0 and self.battery_system:
            if self.battery_system.state_of_charge / 100.0 < 0.9:
                power_allocation['battery_power'] = min(remaining_power, self.config.max_charge_rate)
                remaining_power -= power_allocation['battery_power']
        
        if remaining_power > 0 and self.supercap_system:
            power_allocation['supercap_power'] = min(remaining_power, self.config.max_charge_rate)
        
        return power_allocation
    
    def _control_power_converters(self, mfc_voltage: float, mfc_current: float,
                                power_allocation: Dict[str, float], dt: float) -> Dict[str, float]:
        """Control power converters based on power allocation"""
        converter_outputs = {
            'load_voltage': 0.0,
            'load_current': 0.0,
            'boost_efficiency': 0.0,
            'controller_efficiency': 0.0
        }
        
        # Control boost controller
        if self.boost_controller:
            # Set target based on load requirements
            if power_allocation['load_power'] > 0:
                target_voltage = self.target_bus_voltage
                from .boost_controller import ControlMode
                self.boost_controller.set_control_mode(
                    ControlMode.VOLTAGE_MODE, target_voltage
                )
            
            boost_measurement = self.boost_controller.process(
                input_voltage=mfc_voltage,
                load_current=power_allocation['load_power'] / self.target_bus_voltage if self.target_bus_voltage > 0 else 0.0
            )
            
            converter_outputs['load_voltage'] = boost_measurement.output_voltage
            converter_outputs['load_current'] = boost_measurement.output_current
            converter_outputs['boost_efficiency'] = boost_measurement.efficiency
        
        # Control supercapacitor controller
        if self.supercap_controller:
            if power_allocation['supercap_power'] != 0:
                target_current = power_allocation['supercap_power'] / self.target_bus_voltage
                self.supercap_controller.set_control_parameters(
                    current_setpoint=target_current
                )
                
                controller_measurement = self.supercap_controller.control_step(
                    input_voltage=self.target_bus_voltage,
                    load_current=target_current,
                    dt=dt,
                    temperature=self.temperature
                )
                
                converter_outputs['controller_efficiency'] = controller_measurement.efficiency
        
        return converter_outputs
    
    def _manage_energy_storage(self, power_allocation: Dict[str, float],
                             dt: float, temperature: float) -> Dict[str, float]:
        """Manage energy storage systems"""
        storage_states = {}
        
        # Manage battery system
        if self.battery_system:
            if power_allocation['battery_power'] > 0:
                # Charging
                battery_measurement = self.battery_system.charge(
                    charging_current=power_allocation['battery_power'] / self.battery_system.voltage,
                    dt=dt,
                    temperature=temperature
                )
            elif power_allocation['battery_power'] < 0:
                # Discharging
                battery_measurement = self.battery_system.discharge(
                    load_current=abs(power_allocation['battery_power']) / self.battery_system.voltage,
                    dt=dt,
                    temperature=temperature
                )
            else:
                # Idle
                battery_measurement = self.battery_system.idle(dt, temperature)
            
            storage_states.update({
                'battery_voltage': battery_measurement.voltage,
                'battery_current': battery_measurement.current,
                'battery_soc': battery_measurement.state_of_charge
            })
        
        # Manage supercapacitor system
        if self.supercap_system:
            if power_allocation['supercap_power'] > 0:
                # Charging
                supercap_measurement = self.supercap_system.charge(
                    charging_current=power_allocation['supercap_power'] / self.supercap_system.voltage,
                    dt=dt,
                    temperature=temperature
                )
            elif power_allocation['supercap_power'] < 0:
                # Discharging
                supercap_measurement = self.supercap_system.discharge(
                    load_current=abs(power_allocation['supercap_power']) / self.supercap_system.voltage,
                    dt=dt,
                    temperature=temperature
                )
            else:
                # Self-discharge only
                supercap_measurement = self.supercap_system.charge(0.0, dt, temperature)
            
            storage_states.update({
                'supercap_voltage': supercap_measurement.voltage,
                'supercap_current': supercap_measurement.current,
                'supercap_soc': supercap_measurement.state_of_charge
            })
        
        return storage_states
    
    def _calculate_system_metrics(self, mfc_power: float, load_power: float,
                                storage_states: Dict[str, float],
                                converter_outputs: Dict[str, float]) -> Dict[str, Any]:
        """Calculate comprehensive system performance metrics"""
        # Total output power
        total_output_power = (converter_outputs.get('load_voltage', 0.0) * 
                            converter_outputs.get('load_current', 0.0))
        
        # System efficiency
        if mfc_power > 0:
            system_efficiency = total_output_power / mfc_power
        else:
            system_efficiency = 0.0
        
        # Energy balance
        energy_balance = mfc_power - load_power
        
        # System status
        if system_efficiency < self.config.efficiency_threshold:
            system_status = "warning"
        elif any(self.fault_flags.values()):
            system_status = "alarm"
        else:
            system_status = "normal"
        
        return {
            'total_output_power': total_output_power,
            'system_efficiency': system_efficiency,
            'energy_balance': energy_balance,
            'system_status': system_status
        }
    
    def _detect_and_handle_faults(self, metrics: Dict[str, Any], temperature: float):
        """Detect and handle system faults"""
        # Temperature fault
        self.fault_flags['overtemperature'] = temperature > 70.0
        
        # Efficiency fault
        self.fault_flags['efficiency_fault'] = metrics['system_efficiency'] < 0.5
        
        # Voltage faults
        if self.boost_controller:
            boost_voltage = getattr(self.boost_controller, 'output_voltage', 0.0)
            self.fault_flags['overvoltage'] = boost_voltage > self.target_bus_voltage * 1.2
            self.fault_flags['undervoltage'] = boost_voltage < self.target_bus_voltage * 0.8
        
        # Current faults
        total_current = 0.0
        if self.battery_system:
            total_current += abs(getattr(self.battery_system, 'current', 0.0))
        if self.supercap_system:
            total_current += abs(getattr(self.supercap_system, 'current', 0.0))
        
        self.fault_flags['overcurrent'] = total_current > 50.0  # Example limit
        
        # Component-specific faults
        if self.battery_system:
            self.fault_flags['battery_fault'] = (
                self.battery_system.state_of_charge < 10.0 or  # Less than 10%
                getattr(self.battery_system, 'temperature', 25.0) > 60.0
            )
        
        if self.supercap_system:
            self.fault_flags['supercap_fault'] = (
                getattr(self.supercap_system, 'health_factor', 1.0) < 0.5
            )
        
        # Handle critical faults
        if any([self.fault_flags['overvoltage'], 
                self.fault_flags['overcurrent'], 
                self.fault_flags['overtemperature']]):
            logger.warning("Critical fault detected - implementing safety shutdown")
            self._emergency_shutdown()
    
    def _emergency_shutdown(self):
        """Implement emergency shutdown procedures"""
        # Disable all power converters
        if self.boost_controller:
            self.boost_controller.duty_cycle = 0.0
        if self.supercap_controller:
            self.supercap_controller.duty_cycle = 0.0
        
        logger.critical("Emergency shutdown activated")
    
    def _update_energy_accounting(self, mfc_power: float, load_power: float, dt: float):
        """Update energy accounting"""
        dt_hours = dt / 3600.0
        
        self.energy_generated += mfc_power * dt_hours
        self.energy_consumed += load_power * dt_hours
        
        # Calculate stored energy
        stored_energy = 0.0
        if self.battery_system:
            # Calculate available energy from battery SOC
            max_energy = self.battery_system.specs.nominal_capacity * self.battery_system.specs.nominal_voltage  # Wh
            stored_energy += max_energy * (self.battery_system.state_of_charge / 100.0)
        if self.supercap_system:
            stored_energy += self.supercap_system.get_available_energy() / 3600  # Convert J to Wh
        
        self.energy_stored = stored_energy
        
        # Wasted energy (simplified calculation)
        if mfc_power > load_power and stored_energy > 0.9 * self._get_max_storage_energy():
            self.energy_wasted += (mfc_power - load_power) * dt_hours
    
    def _get_max_storage_energy(self) -> float:
        """Get maximum storage energy capacity in Wh"""
        max_energy = 0.0
        if self.battery_system:
            max_energy += self.battery_system.specs.nominal_capacity * self.battery_system.specs.nominal_voltage  # Wh
        if self.supercap_system:
            max_energy += self.supercap_system.max_energy / 3600  # J to Wh
        return max_energy
    
    def generate_spice_model(self, output_file: Optional[str] = None,
                           operating_point: Optional[Dict[str, float]] = None) -> str:
        """
        Generate comprehensive SPICE model for the entire power management system
        
        Args:
            output_file: Optional file to write the SPICE model
            operating_point: Optional operating point for linearization
            
        Returns:
            SPICE netlist as string
        """
        if operating_point is None:
            operating_point = {
                'mfc_voltage': 2.0,
                'mfc_current': 1.0,
                'load_power': 5.0,
                'temperature': 25.0
            }
        
        spice_model = f"""* Comprehensive Power Management System for MFC
* Generated for operating point: VMFC={operating_point['mfc_voltage']:.2f}V, 
* IMFC={operating_point['mfc_current']:.3f}A, Pload={operating_point['load_power']:.1f}W

* Top-level power management system
.subckt power_management_system mfc_pos mfc_neg load_pos load_neg gnd
"""
        
        # Add MFC model
        spice_model += """
* MFC equivalent circuit
V_mfc mfc_pos n_mfc_internal DC {:.2f}
R_mfc_internal n_mfc_internal mfc_neg 1.0
""".format(operating_point['mfc_voltage'])
        
        # Add boost controller if present
        if self.boost_controller:
            boost_spice = self.boost_controller.generate_spice_model(operating_point=operating_point)
            spice_model += "\n* Boost Controller\n"
            spice_model += boost_spice.split('.subckt')[1].split('.ends')[0]
            spice_model += "X_boost mfc_pos n_bus gnd boost_controller\n"
        else:
            spice_model += "* Direct connection (no boost controller)\n"
            spice_model += "R_direct mfc_pos n_bus 0.1\n"
        
        # Add battery system if present
        if self.battery_system:
            battery_spice = self.battery_system.generate_spice_model(operating_point=operating_point)
            spice_model += "\n* Battery System\n"
            spice_model += battery_spice.split('.subckt')[1].split('.ends')[0]
            spice_model += "X_battery n_bus n_battery_neg gnd battery_system\n"
        
        # Add supercapacitor system if present
        if self.supercap_system:
            supercap_spice = self.supercap_system.generate_spice_model(operating_point=operating_point)
            spice_model += "\n* Supercapacitor System\n"
            spice_model += supercap_spice.split('.subckt')[1].split('.ends')[0]
            spice_model += "X_supercap n_bus gnd supercap_model\n"
        
        # Add supercapacitor controller if present
        if self.supercap_controller:
            controller_spice = self.supercap_controller.generate_spice_model(operating_point=operating_point)
            spice_model += "\n* Supercapacitor Controller\n"
            spice_model += controller_spice.split('.subckt')[1].split('.ends')[0]
            spice_model += "X_supercap_ctrl n_bus n_supercap_out gnd supercap_controller\n"
        
        # Add load interface
        spice_model += f"""
* Load interface
R_load_interface n_bus load_pos 0.01
V_load_return load_neg gnd DC 0

* Bus capacitance for stability
C_bus n_bus gnd 1000u

* Power management control circuit
E_bus_regulation n_bus_reg gnd VALUE={{if(V(n_bus)>{self.target_bus_voltage:.1f}, {self.target_bus_voltage:.1f}, V(n_bus))}}
R_regulation n_bus_reg n_bus 0.001

.ends power_management_system

* System-level testbench
.include "power_management_system.cir"

* MFC source
I_mfc n_mfc_source 0 DC {operating_point['mfc_current']:.3f}
V_mfc_bias n_mfc_source 0 DC {operating_point['mfc_voltage']:.2f}

* Load
R_load n_load_pos n_load_neg {self.target_bus_voltage**2 / operating_point['load_power']:.2f}

* System instantiation
X_pms n_mfc_source 0 n_load_pos n_load_neg 0 power_management_system

* Analysis
.tran 0.1m 10 0 0.1m
.probe V(n_mfc_source) I(X_pms.X_boost.L_main) V(n_load_pos) I(R_load)

* DC sweep analysis
.dc I_mfc 0 5 0.1
.probe V(n_load_pos) I(R_load)

* AC analysis for stability
.ac dec 100 1 100k
.probe V(n_load_pos) V(X_pms.n_bus)

* Efficiency calculation
.measure TRAN pin AVG {{V(n_mfc_source)*I(I_mfc)}} FROM 1 TO 10
.measure TRAN pout AVG {{V(n_load_pos)*I(R_load)}} FROM 1 TO 10
.measure TRAN efficiency PARAM {{pout/pin*100}}

* Energy management parameters
.param max_charge_rate={self.config.max_charge_rate:.1f}
.param max_discharge_rate={self.config.max_discharge_rate:.1f}
.param target_bus_voltage={self.target_bus_voltage:.1f}
.param efficiency_threshold={self.config.efficiency_threshold:.2f}

* Management mode: {self.config.management_mode.value}
* Energy strategy: {self.config.energy_strategy.value}

.end
"""
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(spice_model)
                logger.info(f"Power management system SPICE model written to {output_file}")
            except Exception as e:
                logger.error(f"Failed to write SPICE model to {output_file}: {e}")
        
        return spice_model
    
    def get_power_consumption(self) -> float:
        """Get total system power consumption"""
        total_power = 0.0
        
        if self.boost_controller:
            total_power += self.boost_controller.get_power_consumption()
        if self.battery_system:
            total_power += self.battery_system.get_power_consumption()
        if self.supercap_system:
            total_power += self.supercap_system.get_power_consumption()
        if self.supercap_controller:
            total_power += self.supercap_controller.get_power_consumption()
        if self.electronic_load:
            total_power += self.electronic_load.get_power_consumption()
        
        return total_power
    
    def get_cost_analysis(self) -> Dict[str, float]:
        """Get comprehensive system cost analysis"""
        total_initial_cost = 0.0
        total_operating_cost_per_hour = 0.0
        total_maintenance_cost_per_hour = 0.0
        
        components = []
        if self.boost_controller:
            components.append(('boost_controller', self.boost_controller))
        if self.battery_system:
            components.append(('battery_system', self.battery_system))
        if self.supercap_system:
            components.append(('supercap_system', self.supercap_system))
        if self.supercap_controller:
            components.append(('supercap_controller', self.supercap_controller))
        if self.electronic_load:
            components.append(('electronic_load', self.electronic_load))
        
        component_costs = {}
        for name, component in components:
            cost_analysis = component.get_cost_analysis()
            component_costs[name] = cost_analysis
            total_initial_cost += cost_analysis['initial_cost']
            total_operating_cost_per_hour += cost_analysis.get('power_cost_per_hour', 0.0)
            total_maintenance_cost_per_hour += cost_analysis.get('maintenance_cost_per_hour', 0.0)
        
        # System integration cost
        integration_cost = total_initial_cost * 0.2  # 20% integration overhead
        
        # Management system cost
        management_cost_per_hour = 0.001  # Software/firmware maintenance
        
        total_cost_per_hour = (total_operating_cost_per_hour + 
                             total_maintenance_cost_per_hour + 
                             management_cost_per_hour)
        
        return {
            'total_initial_cost': total_initial_cost + integration_cost,
            'component_costs': component_costs,
            'integration_cost': integration_cost,
            'total_operating_cost_per_hour': total_operating_cost_per_hour,
            'total_maintenance_cost_per_hour': total_maintenance_cost_per_hour,
            'management_cost_per_hour': management_cost_per_hour,
            'total_cost_per_hour': total_cost_per_hour,
            'lifetime_cost_10_years': (total_initial_cost + integration_cost + 
                                     total_cost_per_hour * 10 * 365 * 24)
        }


class PIController:
    """Simple PI controller for power management"""
    
    def __init__(self, kp: float, ki: float):
        self.kp = kp
        self.ki = ki
        self.integrator = 0.0
        self.last_error = 0.0
    
    def update(self, setpoint: float, measured_value: float, dt: float) -> float:
        """Update PI controller"""
        error = setpoint - measured_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integrator += error * dt
        # Anti-windup
        self.integrator = np.clip(self.integrator, -10.0, 10.0)
        i_term = self.ki * self.integrator
        
        # PI output
        output = p_term + i_term
        
        self.last_error = error
        return output


def create_standard_power_management_systems() -> Dict[str, PowerManagementSystem]:
    """Create standard power management system configurations"""
    
    # Standalone MFC system
    standalone_config = PowerSystemConfiguration(
        management_mode=PowerManagementMode.STANDALONE,
        energy_strategy=EnergyManagementStrategy.BALANCED,
        enable_battery=True,
        enable_supercapacitor=True,
        enable_grid_interface=False,
        max_charge_rate=50.0,
        max_discharge_rate=100.0
    )
    
    # Energy harvesting optimized system
    harvesting_config = PowerSystemConfiguration(
        management_mode=PowerManagementMode.ENERGY_HARVESTING,
        energy_strategy=EnergyManagementStrategy.EFFICIENCY_OPTIMIZED,
        enable_battery=True,
        enable_supercapacitor=True,
        enable_grid_interface=False,
        max_charge_rate=20.0,
        max_discharge_rate=50.0
    )
    
    # Load following system
    load_following_config = PowerSystemConfiguration(
        management_mode=PowerManagementMode.LOAD_FOLLOWING,
        energy_strategy=EnergyManagementStrategy.LOAD_PRIORITY,
        enable_battery=True,
        enable_supercapacitor=True,
        enable_grid_interface=False,
        max_charge_rate=100.0,
        max_discharge_rate=200.0
    )
    
    systems = {
        'standalone_balanced': PowerManagementSystem(standalone_config),
        'energy_harvesting': PowerManagementSystem(harvesting_config),
        'load_following': PowerManagementSystem(load_following_config)
    }
    
    return systems


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create standard power management systems
    systems = create_standard_power_management_systems()
    
    # Test standalone system
    pms = systems['standalone_balanced']
    
    print(f"Testing {pms.config.management_mode.value} power management system")
    print(f"Energy strategy: {pms.config.energy_strategy.value}")
    
    # Simulate operation
    print("\nPower management simulation:")
    for i in range(20):
        # Simulate varying MFC output and load
        mfc_voltage = 2.0 + 0.5 * np.sin(i * 0.3)
        mfc_current = 1.0 + 0.3 * np.cos(i * 0.2)
        load_power = 5.0 + 2.0 * np.sin(i * 0.1)
        
        measurement = pms.manage_power(
            mfc_voltage=mfc_voltage,
            mfc_current=mfc_current,
            load_power_demand=load_power,
            dt=1.0,
            temperature=25.0 + i * 0.5
        )
        
        print(f"Step {i+1:2d}: MFC={measurement.mfc_power:.2f}W, "
              f"Load={measurement.load_power:.2f}W, "
              f"BatSOC={measurement.battery_soc:.1%}, "
              f"SupSOC={measurement.supercap_soc:.1%}, "
              f"Eff={measurement.system_efficiency:.1%}, "
              f"Status={measurement.system_status}")
    
    # Cost analysis
    cost_analysis = pms.get_cost_analysis()
    print("\nCost analysis:")
    print(f"Total initial cost: ${cost_analysis['total_initial_cost']:.2f}")
    print(f"Total cost per hour: ${cost_analysis['total_cost_per_hour']:.4f}")
    print(f"10-year lifetime cost: ${cost_analysis['lifetime_cost_10_years']:.2f}")
    
    # Generate SPICE model
    spice_model = pms.generate_spice_model()
    print(f"\nSPICE model generated ({len(spice_model)} characters)")
    print("Comprehensive power management system model created")