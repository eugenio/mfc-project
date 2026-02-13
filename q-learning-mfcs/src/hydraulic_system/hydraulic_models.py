#!/usr/bin/env python3
"""
Hydraulic System Models for MFC - Phase 6
=========================================

Comprehensive hydraulic system modeling including:
- Pump models (peristaltic, centrifugal, diaphragm)
- Flow calculations and pressure dynamics
- Plumbing network modeling
- Cell geometry and flow distribution
- Power consumption and cost analysis
- Recirculation, aeration, and substrate delivery

Created: 2025-07-27 (Phase 6)
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Physical constants
WATER_DENSITY = 1000.0  # kg/m³
WATER_VISCOSITY = 0.001  # Pa·s at 20°C
GRAVITY = 9.81  # m/s²
ATMOSPHERIC_PRESSURE = 101325.0  # Pa


class PumpType(Enum):
    """Types of pumps used in MFC systems."""
    PERISTALTIC = "peristaltic"
    CENTRIFUGAL = "centrifugal"
    DIAPHRAGM = "diaphragm"
    GEAR = "gear"
    SYRINGE = "syringe"


class FluidType(Enum):
    """Types of fluids in MFC hydraulic system."""
    SUBSTRATE = "substrate"
    RECIRCULATION = "recirculation"
    AERATION = "aeration"
    BUFFER = "buffer"
    CLEANING = "cleaning"
    WATER = "water"


@dataclass
class PumpParameters:
    """Parameters for pump models."""
    max_flow_rate: float = 100.0  # mL/min
    max_pressure: float = 50000.0  # Pa (0.5 bar)
    efficiency: float = 0.7  # 70% efficiency
    power_rating: float = 5.0  # W
    cost: float = 200.0  # USD
    tubing_inner_diameter: float = 1.6e-3  # m (1.6mm)
    tubing_wall_thickness: float = 0.8e-3  # m (0.8mm)
    maintenance_interval: float = 8760.0  # hours (1 year)
    maintenance_cost: float = 50.0  # USD


@dataclass
class CellGeometry:
    """MFC cell geometry parameters."""
    length: float = 0.1  # m
    width: float = 0.05  # m
    height: float = 0.02  # m
    anode_area: float = 0.005  # m²
    cathode_area: float = 0.005  # m²
    membrane_area: float = 0.005  # m²
    volume: float = 0.0001  # m³ (100 mL)
    inlet_diameter: float = 3.0e-3  # m (3mm)
    outlet_diameter: float = 3.0e-3  # m (3mm)
    electrode_spacing: float = 0.01  # m


@dataclass
class PipingNetwork:
    """Piping network configuration."""
    total_length: float = 2.0  # m
    pipe_diameter: float = 3.0e-3  # m (3mm)
    pipe_roughness: float = 1.5e-6  # m (smooth plastic)
    fittings_equivalent_length: float = 0.5  # m
    elevation_change: float = 0.1  # m
    ambient_temperature: float = 293.15  # K (20°C)


class BasePump(ABC):
    """Abstract base class for pump models."""
    
    def __init__(self, pump_id: str, pump_type: PumpType, parameters: PumpParameters):
        self.pump_id = pump_id
        self.pump_type = pump_type
        self.params = parameters
        self.is_running = False
        self.current_flow_rate = 0.0  # mL/min
        self.current_power = 0.0  # W
        self.operating_hours = 0.0
        self.maintenance_due = False
        
    @abstractmethod
    def calculate_flow_rate(self, pressure_differential: float) -> float:
        """Calculate flow rate based on pressure differential."""
        pass
    
    @abstractmethod
    def calculate_power_consumption(self, flow_rate: float, pressure: float) -> float:
        """Calculate power consumption for given operating conditions."""
        pass
    
    def start_pump(self):
        """Start the pump."""
        self.is_running = True
        
    def stop_pump(self):
        """Stop the pump."""
        self.is_running = False
        self.current_flow_rate = 0.0
        self.current_power = 0.0
    
    def update_operation(self, dt: float, target_flow_rate: float, system_pressure: float):
        """Update pump operation over time step dt."""
        if not self.is_running:
            self.current_flow_rate = 0.0
            self.current_power = 0.0
            return
        
        # Update operating hours
        self.operating_hours += dt
        
        # Check maintenance schedule
        if self.operating_hours > self.params.maintenance_interval:
            self.maintenance_due = True
        
        # Calculate actual flow rate considering pressure
        pressure_differential = system_pressure - ATMOSPHERIC_PRESSURE
        max_achievable_flow = self.calculate_flow_rate(pressure_differential)
        self.current_flow_rate = min(target_flow_rate, max_achievable_flow)
        
        # Calculate power consumption
        self.current_power = self.calculate_power_consumption(
            self.current_flow_rate, system_pressure
        )
    
    def get_pump_status(self) -> Dict[str, Any]:
        """Get current pump status."""
        return {
            'pump_id': self.pump_id,
            'pump_type': self.pump_type.value,
            'is_running': self.is_running,
            'flow_rate_ml_min': self.current_flow_rate,
            'power_consumption_w': self.current_power,
            'operating_hours': self.operating_hours,
            'maintenance_due': self.maintenance_due,
            'efficiency_pct': self.params.efficiency * 100
        }


class PeristalticPump(BasePump):
    """Peristaltic pump model - ideal for precise, low-flow applications."""
    
    def __init__(self, pump_id: str, parameters: PumpParameters):
        super().__init__(pump_id, PumpType.PERISTALTIC, parameters)
        self.rpm = 0.0
        
    def calculate_flow_rate(self, pressure_differential: float) -> float:
        """Calculate flow rate. Peristaltic pumps are positive displacement."""
        # Peristaltic pumps maintain flow rate regardless of moderate pressures
        if abs(pressure_differential) > self.params.max_pressure:
            return 0.0  # Pump stalled
        
        # Flow rate proportional to RPM, independent of pressure
        max_flow = self.params.max_flow_rate
        return max_flow
    
    def calculate_power_consumption(self, flow_rate: float, pressure: float) -> float:
        """Calculate power consumption."""
        if flow_rate == 0:
            return 0.1  # Standby power
        
        # Base power + pressure-dependent power
        base_power = self.params.power_rating * 0.3
        pressure_power = (flow_rate / self.params.max_flow_rate) * self.params.power_rating * 0.7
        pressure_factor = 1.0 + pressure / self.params.max_pressure * 0.5
        
        total_power = (base_power + pressure_power) * pressure_factor / self.params.efficiency
        return min(total_power, self.params.power_rating * 1.2)  # 120% overload limit


class CentrifugalPump(BasePump):
    """Centrifugal pump model - good for high flow rates."""
    
    def __init__(self, pump_id: str, parameters: PumpParameters):
        super().__init__(pump_id, PumpType.CENTRIFUGAL, parameters)
        
    def calculate_flow_rate(self, pressure_differential: float) -> float:
        """Calculate flow rate using pump curve approximation."""
        # Simplified pump curve: Q = Q_max * (1 - (H/H_max)^0.5)
        pressure_head = pressure_differential / (WATER_DENSITY * GRAVITY)  # m
        max_head = self.params.max_pressure / (WATER_DENSITY * GRAVITY)  # m
        
        if pressure_head >= max_head:
            return 0.0
        
        head_ratio = pressure_head / max_head
        flow_ratio = 1.0 - (head_ratio ** 0.5)
        
        return self.params.max_flow_rate * max(0, flow_ratio)
    
    def calculate_power_consumption(self, flow_rate: float, pressure: float) -> float:
        """Calculate power consumption."""
        if flow_rate == 0:
            return 0.05  # Lower standby power than peristaltic
        
        # Power = (Q * H * ρ * g) / efficiency
        pressure_head = pressure / (WATER_DENSITY * GRAVITY)
        flow_m3_s = (flow_rate / 60000.0)  # Convert mL/min to m³/s
        
        hydraulic_power = flow_m3_s * pressure_head * WATER_DENSITY * GRAVITY
        total_power = hydraulic_power / self.params.efficiency
        
        # Add motor losses
        motor_power = max(total_power, self.params.power_rating * 0.1)
        return min(motor_power, self.params.power_rating)


class DiaphragmPump(BasePump):
    """Diaphragm pump model - good for corrosive fluids."""
    
    def __init__(self, pump_id: str, parameters: PumpParameters):
        super().__init__(pump_id, PumpType.DIAPHRAGM, parameters)
        
    def calculate_flow_rate(self, pressure_differential: float) -> float:
        """Calculate flow rate. Similar to peristaltic but with some pressure sensitivity."""
        if abs(pressure_differential) > self.params.max_pressure:
            return 0.0
        
        # Slight pressure dependency
        pressure_factor = 1.0 - abs(pressure_differential) / self.params.max_pressure * 0.2
        return self.params.max_flow_rate * max(0, pressure_factor)
    
    def calculate_power_consumption(self, flow_rate: float, pressure: float) -> float:
        """Calculate power consumption."""
        if flow_rate == 0:
            return 0.08  # Standby power
        
        # Similar to peristaltic but slightly higher power
        base_power = self.params.power_rating * 0.4
        flow_power = (flow_rate / self.params.max_flow_rate) * self.params.power_rating * 0.6
        pressure_factor = 1.0 + abs(pressure) / self.params.max_pressure * 0.3
        
        total_power = (base_power + flow_power) * pressure_factor / self.params.efficiency
        return min(total_power, self.params.power_rating * 1.1)


class FlowCalculator:
    """Utility class for hydraulic flow calculations."""
    
    @staticmethod
    def calculate_reynolds_number(velocity: float, diameter: float, 
                                 density: float = WATER_DENSITY,
                                 viscosity: float = WATER_VISCOSITY) -> float:
        """Calculate Reynolds number."""
        return (density * velocity * diameter) / viscosity
    
    @staticmethod
    def calculate_friction_factor(reynolds: float, roughness: float, diameter: float) -> float:
        """Calculate Darcy friction factor using Colebrook equation approximation."""
        if reynolds < 2300:  # Laminar flow
            return 64.0 / reynolds
        
        # Turbulent flow - Swamee-Jain approximation
        relative_roughness = roughness / diameter
        term1 = (relative_roughness / 3.7) ** 1.11
        term2 = 6.9 / reynolds
        
        friction_factor = 0.25 / (np.log10(term1 + term2)) ** 2
        return friction_factor
    
    @staticmethod
    def calculate_pressure_drop(flow_rate: float, pipe_diameter: float, 
                              pipe_length: float, roughness: float,
                              fittings_length: float = 0.0,
                              density: float = WATER_DENSITY,
                              viscosity: float = WATER_VISCOSITY) -> float:
        """Calculate pressure drop through pipe using Darcy-Weisbach equation."""
        # Convert flow rate from mL/min to m³/s
        flow_m3_s = flow_rate / 60000000.0
        
        # Calculate velocity
        area = np.pi * (pipe_diameter / 2) ** 2
        velocity = flow_m3_s / area
        
        if velocity == 0:
            return 0.0
        
        # Calculate Reynolds number and friction factor
        reynolds = FlowCalculator.calculate_reynolds_number(velocity, pipe_diameter, density, viscosity)
        friction_factor = FlowCalculator.calculate_friction_factor(reynolds, roughness, pipe_diameter)
        
        # Total length including fittings
        total_length = pipe_length + fittings_length
        
        # Darcy-Weisbach equation
        pressure_drop = friction_factor * (total_length / pipe_diameter) * (density * velocity ** 2) / 2
        
        return pressure_drop
    
    @staticmethod
    def calculate_hydrostatic_pressure(elevation_change: float, 
                                     density: float = WATER_DENSITY) -> float:
        """Calculate hydrostatic pressure due to elevation change."""
        return density * GRAVITY * elevation_change


class HydraulicNetwork:
    """Models the complete hydraulic network of the MFC system."""
    
    def __init__(self, cell_geometries: List[CellGeometry], piping: PipingNetwork):
        self.cells = cell_geometries
        self.piping = piping
        self.pumps: Dict[str, BasePump] = {}
        self.flow_rates: Dict[str, float] = {}  # mL/min for each flow path
        self.pressures: Dict[str, float] = {}  # Pa at various points
        self.total_power_consumption = 0.0
        
    def add_pump(self, pump: BasePump, flow_path: str):
        """Add a pump to the hydraulic network."""
        self.pumps[flow_path] = pump
        self.flow_rates[flow_path] = 0.0
        self.pressures[flow_path] = ATMOSPHERIC_PRESSURE
    
    def calculate_network_pressures(self):
        """Calculate pressures throughout the hydraulic network."""
        # Simplified network analysis - can be extended for complex networks
        for flow_path, flow_rate in self.flow_rates.items():
            # Calculate pressure drop through piping
            pressure_drop = FlowCalculator.calculate_pressure_drop(
                flow_rate=flow_rate,
                pipe_diameter=self.piping.pipe_diameter,
                pipe_length=self.piping.total_length / len(self.cells),
                roughness=self.piping.pipe_roughness,
                fittings_length=self.piping.fittings_equivalent_length / len(self.cells)
            )
            
            # Add hydrostatic pressure
            hydrostatic_pressure = FlowCalculator.calculate_hydrostatic_pressure(
                self.piping.elevation_change
            )
            
            # Update pressure at this point
            self.pressures[flow_path] = ATMOSPHERIC_PRESSURE + pressure_drop + hydrostatic_pressure
    
    def update_hydraulic_system(self, dt: float, target_flows: Dict[str, float]):
        """Update the entire hydraulic system."""
        # Update target flow rates
        for flow_path, target_flow in target_flows.items():
            if flow_path in self.flow_rates:
                self.flow_rates[flow_path] = target_flow
        
        # Calculate network pressures
        self.calculate_network_pressures()
        
        # Update all pumps
        total_power = 0.0
        for flow_path, pump in self.pumps.items():
            target_flow = target_flows.get(flow_path, 0.0)
            system_pressure = self.pressures.get(flow_path, ATMOSPHERIC_PRESSURE)
            
            pump.update_operation(dt, target_flow, system_pressure)
            total_power += pump.current_power
        
        self.total_power_consumption = total_power
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get complete hydraulic network status."""
        pump_statuses = {path: pump.get_pump_status() for path, pump in self.pumps.items()}
        
        return {
            'pumps': pump_statuses,
            'flow_rates': self.flow_rates.copy(),
            'pressures': {path: p - ATMOSPHERIC_PRESSURE for path, p in self.pressures.items()},  # Gauge pressure
            'total_power_w': self.total_power_consumption,
            'network_efficiency': self._calculate_network_efficiency()
        }
    
    def _calculate_network_efficiency(self) -> float:
        """Calculate overall network efficiency."""
        if self.total_power_consumption == 0:
            return 0.0
        
        # Calculate hydraulic power delivered
        hydraulic_power = 0.0
        for flow_path, flow_rate in self.flow_rates.items():
            pressure = self.pressures.get(flow_path, ATMOSPHERIC_PRESSURE) - ATMOSPHERIC_PRESSURE
            flow_m3_s = flow_rate / 60000000.0  # Convert mL/min to m³/s
            hydraulic_power += flow_m3_s * pressure
        
        return hydraulic_power / self.total_power_consumption if self.total_power_consumption > 0 else 0.0


class HydraulicController:
    """Controller for hydraulic system operations."""
    
    def __init__(self, network: HydraulicNetwork):
        self.network = network
        self.control_setpoints: Dict[str, float] = {}
        self.pid_controllers: Dict[str, Dict[str, float]] = {}
        
    def set_flow_setpoint(self, flow_path: str, setpoint: float):
        """Set flow rate setpoint for a specific path."""
        self.control_setpoints[flow_path] = setpoint
        
        # Initialize PID controller if not exists
        if flow_path not in self.pid_controllers:
            self.pid_controllers[flow_path] = {
                'kp': 1.0, 'ki': 0.1, 'kd': 0.01,
                'error_integral': 0.0, 'last_error': 0.0
            }
    
    def calculate_control_output(self, flow_path: str, actual_flow: float, dt: float) -> float:
        """Calculate PID control output."""
        if flow_path not in self.control_setpoints:
            return actual_flow
        
        setpoint = self.control_setpoints[flow_path]
        pid = self.pid_controllers[flow_path]
        
        # Calculate error
        error = setpoint - actual_flow
        
        # PID calculations
        pid['error_integral'] += error * dt
        error_derivative = (error - pid['last_error']) / dt if dt > 0 else 0.0
        
        # Control output
        output = (pid['kp'] * error + 
                 pid['ki'] * pid['error_integral'] + 
                 pid['kd'] * error_derivative)
        
        pid['last_error'] = error
        
        # Apply output limits
        return max(0, setpoint + output)
    
    def update_control(self, dt: float) -> Dict[str, float]:
        """Update all flow control loops."""
        control_outputs = {}
        
        for flow_path in self.control_setpoints:
            actual_flow = self.network.flow_rates.get(flow_path, 0.0)
            control_output = self.calculate_control_output(flow_path, actual_flow, dt)
            control_outputs[flow_path] = control_output
        
        return control_outputs


def create_standard_hydraulic_system(n_cells: int = 1) -> HydraulicNetwork:
    """Create a standard hydraulic system configuration."""
    
    # Create cell geometries
    cell_geometries = []
    for i in range(n_cells):
        cell = CellGeometry(
            length=0.1,  # 10 cm
            width=0.05,  # 5 cm  
            height=0.02, # 2 cm
            volume=0.0001  # 100 mL
        )
        cell_geometries.append(cell)
    
    # Create piping network
    piping = PipingNetwork(
        total_length=2.0 + n_cells * 0.3,  # Scale with number of cells
        pipe_diameter=3.0e-3,  # 3mm ID tubing
        pipe_roughness=1.5e-6,
        fittings_equivalent_length=0.5 + n_cells * 0.1
    )
    
    # Create hydraulic network
    network = HydraulicNetwork(cell_geometries, piping)
    
    # Add standard pumps
    # Substrate delivery pump
    substrate_pump_params = PumpParameters(
        max_flow_rate=50.0,  # mL/min
        max_pressure=30000.0,  # 0.3 bar
        efficiency=0.75,
        power_rating=3.0,  # W
        cost=150.0
    )
    substrate_pump = PeristalticPump("substrate_pump", substrate_pump_params)
    network.add_pump(substrate_pump, "substrate")
    
    # Recirculation pump
    recirculation_pump_params = PumpParameters(
        max_flow_rate=200.0,  # mL/min
        max_pressure=20000.0,  # 0.2 bar
        efficiency=0.8,
        power_rating=5.0,  # W
        cost=200.0
    )
    recirculation_pump = CentrifugalPump("recirculation_pump", recirculation_pump_params)
    network.add_pump(recirculation_pump, "recirculation")
    
    # Aeration pump (air)
    aeration_pump_params = PumpParameters(
        max_flow_rate=1000.0,  # mL/min (air)
        max_pressure=10000.0,  # 0.1 bar
        efficiency=0.6,
        power_rating=2.0,  # W
        cost=100.0
    )
    aeration_pump = DiaphragmPump("aeration_pump", aeration_pump_params)
    network.add_pump(aeration_pump, "aeration")
    
    return network


def calculate_hydraulic_costs(network: HydraulicNetwork, operating_hours: float,
                            electricity_cost: float = 0.12) -> Dict[str, float]:
    """Calculate hydraulic system costs."""
    
    # Capital costs
    capital_cost = sum(pump.params.cost for pump in network.pumps.values())
    
    # Operating costs
    total_energy = network.total_power_consumption * operating_hours / 1000.0  # kWh
    energy_cost = total_energy * electricity_cost
    
    # Maintenance costs
    maintenance_cost = 0.0
    for pump in network.pumps.values():
        maintenance_cycles = pump.operating_hours / pump.params.maintenance_interval
        maintenance_cost += maintenance_cycles * pump.params.maintenance_cost
    
    # Total cost
    total_cost = capital_cost + energy_cost + maintenance_cost
    
    return {
        'capital_cost_usd': capital_cost,
        'energy_cost_usd': energy_cost,
        'maintenance_cost_usd': maintenance_cost,
        'total_cost_usd': total_cost,
        'cost_per_hour_usd': total_cost / operating_hours if operating_hours > 0 else 0.0,
        'power_consumption_kwh': total_energy
    }


if __name__ == "__main__":
    # Demonstration
    print("🔧 Hydraulic System Models Demo")
    
    # Create hydraulic system
    hydraulic_system = create_standard_hydraulic_system(n_cells=3)
    controller = HydraulicController(hydraulic_system)
    
    # Set flow setpoints
    controller.set_flow_setpoint("substrate", 10.0)  # mL/min
    controller.set_flow_setpoint("recirculation", 50.0)  # mL/min
    controller.set_flow_setpoint("aeration", 100.0)  # mL/min
    
    # Start pumps
    for pump in hydraulic_system.pumps.values():
        pump.start_pump()
    
    # Run simulation
    print("\n📊 Running hydraulic simulation...")
    simulation_time = 0.0
    dt = 0.1  # 6 minutes
    
    for step in range(10):
        # Update control
        target_flows = controller.update_control(dt)
        
        # Update hydraulic system
        hydraulic_system.update_hydraulic_system(dt, target_flows)
        
        simulation_time += dt
        
        if step % 2 == 0:  # Print every 12 minutes
            status = hydraulic_system.get_network_status()
            print(f"t={simulation_time:.1f}h: Power={status['total_power_w']:.1f}W, "
                  f"Efficiency={status['network_efficiency']:.2%}")
    
    # Calculate costs
    costs = calculate_hydraulic_costs(hydraulic_system, 100.0)  # 100 hours
    print("\n💰 Cost Analysis (100h operation):")
    print(f"   Capital: ${costs['capital_cost_usd']:.2f}")
    print(f"   Energy: ${costs['energy_cost_usd']:.2f}")
    print(f"   Maintenance: ${costs['maintenance_cost_usd']:.2f}")
    print(f"   Total: ${costs['total_cost_usd']:.2f}")
    
    print("\n✅ Hydraulic system demo complete!")