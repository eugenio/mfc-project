"""
Supercapacitor Energy Storage Models for MFC Systems

This module implements supercapacitor/ultracapacitor energy storage models including
electrochemical double-layer capacitors (EDLC), pseudocapacitors, and hybrid 
supercapacitors with SPICE model generation for circuit simulation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SupercapacitorType(Enum):
    """Types of supercapacitors"""
    EDLC = "edlc"  # Electrochemical Double Layer Capacitor
    PSEUDOCAPACITOR = "pseudocapacitor"  # Faradaic pseudocapacitor
    HYBRID = "hybrid"  # Hybrid supercapacitor
    ASYMMETRIC = "asymmetric"  # Asymmetric supercapacitor


@dataclass
class SupercapacitorSpecs:
    """Specifications for supercapacitor systems"""
    capacitor_type: SupercapacitorType
    rated_capacitance: float  # F (farads)
    rated_voltage: float  # V
    max_current: float  # A
    esr: float  # Equivalent Series Resistance (ohms)
    esl: float  # Equivalent Series Inductance (H)
    leakage_current: float  # A at rated voltage
    cycle_life: int  # number of cycles
    operating_temp_range: Tuple[float, float]  # min, max °C
    power_consumption: float  # W (for management system)
    cost: float  # USD
    mass: float  # kg
    volume: float  # L


@dataclass
class SupercapacitorMeasurement:
    """Single supercapacitor measurement"""
    timestamp: float
    voltage: float  # V
    current: float  # A
    capacitance: float  # F
    esr: float  # ohms
    stored_energy: float  # J
    power: float  # W
    state_of_charge: float  # 0-1
    temperature: float  # °C
    cycle_count: int
    health_factor: float  # 0-1
    self_discharge_rate: float  # %/hour


class SupercapacitorSystem:
    """Comprehensive supercapacitor energy storage system"""
    
    def __init__(self, specs: SupercapacitorSpecs):
        self.specs = specs
        self.voltage = 0.0
        self.current = 0.0
        self.temperature = 25.0
        self.cycle_count = 0
        self.health_factor = 1.0
        self.last_voltage = 0.0
        
        # Initialize capacitor-specific parameters
        self._initialize_capacitor_parameters()
        
        # Energy storage parameters
        self.max_energy = 0.5 * self.specs.rated_capacitance * self.specs.rated_voltage**2
        self.stored_energy = 0.0
        
        # Degradation parameters
        self.capacitance_fade_rate = 0.2  # % per 100k cycles
        self.esr_increase_rate = 10.0  # % per 100k cycles
        
        # Self-discharge parameters
        self.self_discharge_time_constant = 24.0  # hours
        
    def _initialize_capacitor_parameters(self):
        """Initialize capacitor-specific parameters"""
        if self.specs.capacitor_type == SupercapacitorType.EDLC:
            # EDLC parameters
            self.electrode_material = "activated_carbon"
            self.electrolyte_type = "organic"
            self.specific_capacitance = 100  # F/g
            self.energy_density = 5.0  # Wh/kg
            self.power_density = 10000  # W/kg
            self.voltage_efficiency = 0.95
            
        elif self.specs.capacitor_type == SupercapacitorType.PSEUDOCAPACITOR:
            # Pseudocapacitor parameters
            self.electrode_material = "metal_oxide"
            self.electrolyte_type = "aqueous"
            self.specific_capacitance = 400  # F/g
            self.energy_density = 20.0  # Wh/kg
            self.power_density = 5000  # W/kg
            self.voltage_efficiency = 0.90
            
        elif self.specs.capacitor_type == SupercapacitorType.HYBRID:
            # Hybrid supercapacitor parameters
            self.anode_material = "lithium_titanate"
            self.cathode_material = "activated_carbon"
            self.electrolyte_type = "organic"
            self.specific_capacitance = 200  # F/g
            self.energy_density = 15.0  # Wh/kg
            self.power_density = 7500  # W/kg
            self.voltage_efficiency = 0.92
            
        elif self.specs.capacitor_type == SupercapacitorType.ASYMMETRIC:
            # Asymmetric supercapacitor parameters
            self.positive_electrode = "metal_oxide"
            self.negative_electrode = "activated_carbon"
            self.electrolyte_type = "aqueous"
            self.specific_capacitance = 150  # F/g
            self.energy_density = 12.0  # Wh/kg
            self.power_density = 8000  # W/kg
            self.voltage_efficiency = 0.88
    
    def charge(self, charging_current: float, dt: float, temperature: float = 25.0) -> SupercapacitorMeasurement:
        """
        Simulate charging the supercapacitor
        
        Args:
            charging_current: Charging current (A, positive for charging)
            dt: Time step (seconds)
            temperature: Operating temperature (°C)
            
        Returns:
            SupercapacitorMeasurement object
        """
        self.temperature = temperature
        
        # Update health factor based on cycling
        self._update_health_factor()
        
        # Current effective capacitance accounting for degradation
        effective_capacitance = self.specs.rated_capacitance * self.health_factor
        
        # Temperature effects on capacitance
        temp_factor = 1.0 + 0.005 * (temperature - 25.0)  # 0.5%/°C
        effective_capacitance *= temp_factor
        
        # Current ESR accounting for degradation and temperature
        current_esr = self.specs.esr * (1.0 + (1.0 - self.health_factor))
        temp_esr_factor = 1.0 - 0.01 * (temperature - 25.0)  # ESR decreases with temperature
        current_esr *= temp_esr_factor
        
        # Limit charging current
        max_charge_current = min(self.specs.max_current, 
                               (self.specs.rated_voltage - self.voltage) / current_esr)
        charging_current = np.clip(charging_current, -self.specs.max_current, max_charge_current)
        
        # Update voltage using capacitor equation: I = C * dV/dt
        if effective_capacitance > 0:
            dv_dt = charging_current / effective_capacitance
            
            # Update internal voltage
            self.voltage += dv_dt * dt
            
            # Apply voltage limits
            self.voltage = np.clip(self.voltage, 0.0, self.specs.rated_voltage)
        
        # Self-discharge
        if dt > 0:
            self._apply_self_discharge(dt)
        
        # Update current
        self.current = charging_current
        
        # Calculate stored energy
        self.stored_energy = 0.5 * effective_capacitance * self.voltage**2
        
        # Calculate power
        power = self.voltage * charging_current
        
        # State of charge
        soc = (self.voltage / self.specs.rated_voltage)**2  # Energy-based SOC
        
        # Update cycle count
        if abs(self.voltage - self.last_voltage) > 0.1 * self.specs.rated_voltage:
            self.cycle_count += 1
        self.last_voltage = self.voltage
        
        # Self-discharge rate calculation
        self_discharge_rate = self._calculate_self_discharge_rate()
        
        return SupercapacitorMeasurement(
            timestamp=0.0,  # Will be set by caller
            voltage=self.voltage,
            current=charging_current,
            capacitance=effective_capacitance,
            esr=current_esr,
            stored_energy=self.stored_energy,
            power=power,
            state_of_charge=soc,
            temperature=temperature,
            cycle_count=self.cycle_count,
            health_factor=self.health_factor,
            self_discharge_rate=self_discharge_rate
        )
    
    def discharge(self, load_current: float, dt: float, temperature: float = 25.0) -> SupercapacitorMeasurement:
        """
        Simulate discharging the supercapacitor
        
        Args:
            load_current: Discharge current (A, positive for discharging)
            dt: Time step (seconds)
            temperature: Operating temperature (°C)
            
        Returns:
            SupercapacitorMeasurement object
        """
        # Discharge is just charging with negative current
        return self.charge(-abs(load_current), dt, temperature)
    
    def _update_health_factor(self):
        """Update health factor based on cycle count and degradation"""
        # Capacitance fade
        capacitance_degradation = (self.capacitance_fade_rate / 100.0) * (self.cycle_count / 100000.0)
        self.health_factor = max(0.5, 1.0 - capacitance_degradation)
    
    def _apply_self_discharge(self, dt: float):
        """Apply self-discharge to the supercapacitor"""
        # Exponential self-discharge: V(t) = V0 * exp(-t/τ)
        # where τ = RC (time constant)
        time_constant = self.self_discharge_time_constant * 3600  # Convert to seconds
        discharge_factor = np.exp(-dt / time_constant)
        self.voltage *= discharge_factor
    
    def _calculate_self_discharge_rate(self) -> float:
        """Calculate current self-discharge rate"""
        # Self-discharge rate depends on voltage and temperature
        base_rate = (self.voltage / self.specs.rated_voltage) * 2.0  # %/hour at rated voltage
        temp_factor = 1.0 + 0.1 * (self.temperature - 25.0)  # 10%/°C
        return base_rate * temp_factor
    
    def set_voltage(self, voltage: float):
        """Set supercapacitor voltage directly"""
        self.voltage = np.clip(voltage, 0.0, self.specs.rated_voltage)
        effective_capacitance = self.specs.rated_capacitance * self.health_factor
        self.stored_energy = 0.5 * effective_capacitance * self.voltage**2
    
    def get_state_of_charge(self) -> float:
        """Get current state of charge (energy-based)"""
        return (self.voltage / self.specs.rated_voltage)**2
    
    def get_available_energy(self) -> float:
        """Get available energy that can be extracted"""
        # Available energy down to some minimum voltage (e.g., 10% of rated)
        min_voltage = 0.1 * self.specs.rated_voltage
        effective_capacitance = self.specs.rated_capacitance * self.health_factor
        current_energy = 0.5 * effective_capacitance * self.voltage**2
        min_energy = 0.5 * effective_capacitance * min_voltage**2
        return max(0.0, current_energy - min_energy)
    
    def get_power_capability(self, duration: float) -> float:
        """Get maximum sustainable power for given duration"""
        # Power limited by current capability or energy availability
        current_limit_power = self.voltage * self.specs.max_current
        
        # Energy limit
        available_energy = self.get_available_energy()
        energy_limit_power = available_energy / duration if duration > 0 else float('inf')
        
        return min(current_limit_power, energy_limit_power)
    
    def generate_spice_model(self, output_file: Optional[str] = None,
                           operating_point: Optional[Dict[str, float]] = None) -> str:
        """
        Generate SPICE model for the supercapacitor
        
        Args:
            output_file: Optional file to write the SPICE model
            operating_point: Optional operating point for linearization
            
        Returns:
            SPICE netlist as string
        """
        if operating_point is None:
            operating_point = {
                'voltage': self.voltage,
                'temperature': self.temperature
            }
        
        # Current effective values
        effective_capacitance = self.specs.rated_capacitance * self.health_factor
        current_esr = self.specs.esr * (1.0 + (1.0 - self.health_factor))
        
        # Temperature corrections
        temp_factor = 1.0 + 0.005 * (operating_point['temperature'] - 25.0)
        effective_capacitance *= temp_factor
        temp_esr_factor = 1.0 - 0.01 * (operating_point['temperature'] - 25.0)
        current_esr *= temp_esr_factor
        
        spice_model = f"""* Supercapacitor Model: {self.specs.capacitor_type.value}
* Generated for operating point: V={operating_point['voltage']:.3f}V, T={operating_point['temperature']:.1f}°C
* Health factor: {self.health_factor:.3f}

.subckt supercap_model pos neg
* Main capacitance (voltage-dependent for realistic behavior)
C_main pos n1 {effective_capacitance:.6f}
* Initial condition
.IC V(pos,n1)={operating_point['voltage']:.3f}

* Equivalent Series Resistance (ESR)
R_esr n1 n2 {current_esr:.6f}

* Equivalent Series Inductance (ESL)
L_esl n2 neg {self.specs.esl:.9f}

* Leakage resistance (parallel to capacitor)
R_leak pos n3 {self.specs.rated_voltage / self.specs.leakage_current:.0f}

* Self-discharge model (voltage-dependent resistance)
E_vdep n4 0 pos neg 1.0
R_selfdis n3 neg 1e12

* Voltage-dependent capacitance model (if needed for advanced modeling)
* B_cap_var n5 0 V={{if(V(pos,neg)>0, {effective_capacitance}*(1-0.1*V(pos,neg)/{self.specs.rated_voltage}), {effective_capacitance})}}

.ends supercap_model

* Usage example:
* X_supercap node_pos node_neg supercap_model

* Model parameters:
* Rated capacitance: {self.specs.rated_capacitance:.3f} F
* Effective capacitance: {effective_capacitance:.6f} F
* ESR: {current_esr:.6f} ohms
* ESL: {self.specs.esl:.9f} H
* Rated voltage: {self.specs.rated_voltage:.1f} V
* Max current: {self.specs.max_current:.1f} A
* Type: {self.specs.capacitor_type.value}
* Health factor: {self.health_factor:.3f}
"""
        
        # Add temperature-dependent models
        if operating_point['temperature'] != 25.0:
            spice_model += f"""
* Temperature effects included for T={operating_point['temperature']:.1f}°C
* Capacitance temperature coefficient: +0.5%/°C
* ESR temperature coefficient: -1.0%/°C
"""
        
        # Add nonlinear models for advanced supercapacitors
        if self.specs.capacitor_type in [SupercapacitorType.PSEUDOCAPACITOR, SupercapacitorType.HYBRID]:
            spice_model += f"""
* Nonlinear behavior model for {self.specs.capacitor_type.value}
.subckt supercap_nonlinear pos neg
* Voltage-dependent capacitance
B_cap_nl n1 0 V=Q(pos,neg)
C_nl pos n1 1
E_cap n1 0 pos neg 1
R_cap_model n1 neg 1

* Faradaic resistance (for pseudocapacitive behavior)
R_faradaic pos n2 {current_esr * 0.1:.6f}
C_faradaic n2 neg {effective_capacitance * 0.3:.6f}
.ends supercap_nonlinear
"""
        
        # Add thermal model
        spice_model += f"""
* Thermal model
.subckt thermal_model power temp_rise
R_thermal power temp_rise {1.0 / (self.specs.mass * 1000):.3f}  ; K/W (thermal resistance)
C_thermal temp_rise 0 {self.specs.mass * 1000:.0f}  ; J/K (thermal capacitance)
.ends thermal_model
"""
        
        # Add testbench
        spice_model += f"""
* Testbench for supercapacitor model
.include "supercap_model.cir"

* Test circuit
V_source n_supply 0 DC {self.specs.rated_voltage}
R_charge n_supply n_pos 1.0
X_supercap n_pos 0 supercap_model

* Analysis
.tran 0.1 1000 0 0.1
.probe V(n_pos) I(X_supercap)

* Charge/discharge test
.param charge_current={self.specs.max_current * 0.5:.3f}
I_charge 0 n_pos DC {{charge_current}}

* AC analysis for impedance characterization
.ac dec 100 0.01 100k
.probe V(n_pos) I(X_supercap)

.end
"""
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(spice_model)
                logger.info(f"SPICE model written to {output_file}")
            except Exception as e:
                logger.error(f"Failed to write SPICE model to {output_file}: {e}")
        
        return spice_model
    
    def get_power_consumption(self) -> float:
        """Get system power consumption"""
        # Base management system power
        base_power = self.specs.power_consumption
        
        # Additional power for monitoring and balancing (if multi-cell)
        monitoring_power = 0.05  # W
        
        # Temperature-dependent power
        if self.temperature < 0 or self.temperature > 40:
            heating_cooling_power = 2.0  # W for thermal management
        else:
            heating_cooling_power = 0.0
        
        return base_power + monitoring_power + heating_cooling_power
    
    def get_cost_analysis(self) -> Dict[str, float]:
        """Get comprehensive cost analysis"""
        initial_cost = self.specs.cost
        
        # Operating cost per hour
        power_cost_per_hour = self.get_power_consumption() * 0.15 / 1000  # $0.15/kWh
        
        # Maintenance cost (very low for supercapacitors)
        maintenance_cost_per_hour = 0.001  # $0.001/hour
        
        # Replacement cost amortized over cycle life
        cycles_per_hour = 0.1  # Estimated average
        replacement_cost_per_hour = initial_cost / (self.specs.cycle_life / cycles_per_hour)
        
        total_cost_per_hour = power_cost_per_hour + maintenance_cost_per_hour + replacement_cost_per_hour
        
        # Lifetime cost
        estimated_lifetime_hours = self.specs.cycle_life / cycles_per_hour
        lifetime_cost = initial_cost + total_cost_per_hour * estimated_lifetime_hours
        
        return {
            'initial_cost': initial_cost,
            'power_cost_per_hour': power_cost_per_hour,
            'maintenance_cost_per_hour': maintenance_cost_per_hour,
            'replacement_cost_per_hour': replacement_cost_per_hour,
            'total_cost_per_hour': total_cost_per_hour,
            'estimated_lifetime_hours': estimated_lifetime_hours,
            'lifetime_cost': lifetime_cost,
            'cost_per_wh': initial_cost / (self.max_energy / 3600),  # $/Wh
            'cost_per_w': initial_cost / self.get_power_capability(3600)  # $/W
        }


def create_standard_supercapacitor_systems() -> Dict[str, SupercapacitorSystem]:
    """Create standard supercapacitor system configurations"""
    
    # Small EDLC for buffering
    edlc_small_specs = SupercapacitorSpecs(
        capacitor_type=SupercapacitorType.EDLC,
        rated_capacitance=10.0,  # F
        rated_voltage=2.7,  # V
        max_current=10.0,  # A
        esr=0.1,  # ohms
        esl=10e-9,  # H
        leakage_current=0.001,  # A
        cycle_life=1000000,
        operating_temp_range=(-40, 65),
        power_consumption=0.1,  # W
        cost=50.0,  # USD
        mass=0.1,  # kg
        volume=0.05  # L
    )
    
    # Medium EDLC for energy storage
    edlc_medium_specs = SupercapacitorSpecs(
        capacitor_type=SupercapacitorType.EDLC,
        rated_capacitance=100.0,  # F
        rated_voltage=2.7,  # V
        max_current=50.0,  # A
        esr=0.02,  # ohms
        esl=5e-9,  # H
        leakage_current=0.005,  # A
        cycle_life=1000000,
        operating_temp_range=(-40, 65),
        power_consumption=0.2,  # W
        cost=200.0,  # USD
        mass=0.5,  # kg
        volume=0.2  # L
    )
    
    # High-energy pseudocapacitor
    pseudo_specs = SupercapacitorSpecs(
        capacitor_type=SupercapacitorType.PSEUDOCAPACITOR,
        rated_capacitance=500.0,  # F
        rated_voltage=1.2,  # V (aqueous electrolyte)
        max_current=100.0,  # A
        esr=0.005,  # ohms
        esl=2e-9,  # H
        leakage_current=0.01,  # A
        cycle_life=500000,
        operating_temp_range=(-20, 60),
        power_consumption=0.3,  # W
        cost=800.0,  # USD
        mass=1.0,  # kg
        volume=0.5  # L
    )
    
    # Hybrid supercapacitor for balanced performance
    hybrid_specs = SupercapacitorSpecs(
        capacitor_type=SupercapacitorType.HYBRID,
        rated_capacitance=200.0,  # F
        rated_voltage=3.8,  # V
        max_current=75.0,  # A
        esr=0.01,  # ohms
        esl=3e-9,  # H
        leakage_current=0.002,  # A
        cycle_life=750000,
        operating_temp_range=(-30, 60),
        power_consumption=0.25,  # W
        cost=600.0,  # USD
        mass=0.8,  # kg
        volume=0.4  # L
    )
    
    # Asymmetric supercapacitor
    asymmetric_specs = SupercapacitorSpecs(
        capacitor_type=SupercapacitorType.ASYMMETRIC,
        rated_capacitance=300.0,  # F
        rated_voltage=2.5,  # V
        max_current=80.0,  # A
        esr=0.008,  # ohms
        esl=4e-9,  # H
        leakage_current=0.003,  # A
        cycle_life=800000,
        operating_temp_range=(-25, 65),
        power_consumption=0.2,  # W
        cost=500.0,  # USD
        mass=0.7,  # kg
        volume=0.35  # L
    )
    
    systems = {
        'edlc_small': SupercapacitorSystem(edlc_small_specs),
        'edlc_medium': SupercapacitorSystem(edlc_medium_specs),
        'pseudocapacitor': SupercapacitorSystem(pseudo_specs),
        'hybrid': SupercapacitorSystem(hybrid_specs),
        'asymmetric': SupercapacitorSystem(asymmetric_specs)
    }
    
    return systems


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create standard supercapacitor systems
    systems = create_standard_supercapacitor_systems()
    
    # Test hybrid supercapacitor
    hybrid_system = systems['hybrid']
    
    print(f"Testing {hybrid_system.specs.capacitor_type.value} supercapacitor")
    print(f"Rated: {hybrid_system.specs.rated_capacitance:.1f}F, {hybrid_system.specs.rated_voltage:.1f}V")
    print(f"Max energy: {hybrid_system.max_energy:.2f}J ({hybrid_system.max_energy/3600:.4f}Wh)")
    
    # Simulate charging
    print("\nCharging simulation:")
    charging_current = 20.0  # A
    time_step = 1.0  # second
    
    for i in range(20):
        measurement = hybrid_system.charge(charging_current, time_step, temperature=25.0)
        print(f"t={i+1:2d}s: V={measurement.voltage:.3f}V, "
              f"SOC={measurement.state_of_charge:.1%}, "
              f"Energy={measurement.stored_energy:.2f}J, "
              f"Power={measurement.power:.1f}W")
        
        if measurement.voltage >= hybrid_system.specs.rated_voltage * 0.95:
            print("Near full charge, stopping...")
            break
    
    # Simulate discharge
    print("\nDischarge simulation:")
    discharge_current = 30.0  # A
    
    for i in range(15):
        measurement = hybrid_system.discharge(discharge_current, time_step, temperature=25.0)
        print(f"t={i+1:2d}s: V={measurement.voltage:.3f}V, "
              f"SOC={measurement.state_of_charge:.1%}, "
              f"Energy={measurement.stored_energy:.2f}J, "
              f"Power={measurement.power:.1f}W")
        
        if measurement.voltage <= hybrid_system.specs.rated_voltage * 0.1:
            print("Low voltage, stopping discharge...")
            break
    
    # Test power capability
    print(f"\nPower capability for 1 hour: {hybrid_system.get_power_capability(3600):.2f}W")
    print(f"Power capability for 1 minute: {hybrid_system.get_power_capability(60):.2f}W")
    
    # Cost analysis
    cost_analysis = hybrid_system.get_cost_analysis()
    print("\nCost analysis:")
    print(f"Initial cost: ${cost_analysis['initial_cost']:.2f}")
    print(f"Cost per Wh: ${cost_analysis['cost_per_wh']:.2f}/Wh")
    print(f"Cost per W: ${cost_analysis['cost_per_w']:.3f}/W")
    print(f"Lifetime cost: ${cost_analysis['lifetime_cost']:.2f}")
    
    # Generate SPICE model
    spice_model = hybrid_system.generate_spice_model()
    print(f"\nSPICE model generated ({len(spice_model)} characters)")
    print("First few lines:")
    for line in spice_model.split('\n')[:10]:
        print(f"  {line}")