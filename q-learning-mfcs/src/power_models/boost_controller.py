"""
Boost Controller Models for MFC Power Management

This module implements various boost converter topologies for stepping up
MFC voltage to usable levels, including efficiency modeling, thermal effects,
control algorithms, and SPICE model generation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ControlMode(Enum):
    """Boost controller control modes"""
    VOLTAGE_MODE = "voltage_mode"
    CURRENT_MODE = "current_mode"
    POWER_MODE = "power_mode"
    MPPT = "maximum_power_point_tracking"


class SwitchingTopology(Enum):
    """Switching converter topologies"""
    BOOST = "boost"
    SYNCHRONOUS_BOOST = "synchronous_boost"
    INTERLEAVED_BOOST = "interleaved_boost"
    FOUR_SWITCH_BOOST = "four_switch_boost"


@dataclass
class BoostControllerSpecs:
    """Specifications for boost controller"""
    topology: SwitchingTopology
    input_voltage_range: Tuple[float, float]  # V (min, max)
    output_voltage_range: Tuple[float, float]  # V (min, max)
    max_output_current: float  # A
    max_output_power: float  # W
    switching_frequency: float  # Hz
    efficiency_at_rated: float  # % at rated power
    quiescent_current: float  # mA
    cost: float  # USD
    lifetime: float  # hours
    thermal_resistance: float  # °C/W
    inductance: float  # H
    capacitance_input: float  # F
    capacitance_output: float  # F
    control_bandwidth: float  # Hz


@dataclass
class BoostControllerMeasurement:
    """Single boost controller measurement"""
    timestamp: float
    input_voltage: float  # V
    input_current: float  # A
    output_voltage: float  # V
    output_current: float  # A
    duty_cycle: float  # %
    switching_frequency: float  # Hz
    efficiency: float  # %
    temperature: float  # °C
    power_dissipation: float  # W
    control_mode: ControlMode
    error_voltage: float  # V
    inductor_current_ripple: float  # A
    output_voltage_ripple: float  # V


class BoostController:
    """Boost converter model with detailed efficiency and thermal modeling"""
    
    def __init__(self, specs: BoostControllerSpecs):
        self.specs = specs
        self.control_mode = ControlMode.VOLTAGE_MODE
        self.target_voltage = 5.0  # V
        self.target_current = 1.0  # A
        self.target_power = 5.0  # W
        
        # Controller state
        self.duty_cycle = 0.5
        self.switching_frequency = specs.switching_frequency
        self.operating_hours = 0.0
        self.temperature = 25.0  # °C
        
        # Control loop parameters
        self.voltage_controller_kp = 0.1
        self.voltage_controller_ki = 100.0
        self.current_controller_kp = 0.01
        self.current_controller_ki = 10.0
        self.voltage_error_integral = 0.0
        self.current_error_integral = 0.0
        
        # Component parameters
        self._initialize_component_parameters()
        
        # Efficiency map
        self._build_efficiency_map()
        
    def _initialize_component_parameters(self):
        """Initialize component-specific parameters"""
        if self.specs.topology == SwitchingTopology.BOOST:
            # Standard boost converter
            self.mosfet_rds_on = 0.01  # Ω
            self.diode_forward_voltage = 0.7  # V
            self.inductor_dcr = 0.005  # Ω
            self.capacitor_esr_input = 0.01  # Ω
            self.capacitor_esr_output = 0.005  # Ω
            
        elif self.specs.topology == SwitchingTopology.SYNCHRONOUS_BOOST:
            # Synchronous boost (MOSFET replaces diode)
            self.mosfet_hs_rds_on = 0.01  # Ω (high-side)
            self.mosfet_ls_rds_on = 0.008  # Ω (low-side)
            self.inductor_dcr = 0.003  # Ω
            self.capacitor_esr_input = 0.01  # Ω
            self.capacitor_esr_output = 0.003  # Ω
            self.dead_time = 50e-9  # s
            
        elif self.specs.topology == SwitchingTopology.INTERLEAVED_BOOST:
            # Interleaved boost (two phases)
            self.num_phases = 2
            self.mosfet_rds_on = 0.015  # Ω per phase
            self.diode_forward_voltage = 0.6  # V
            self.inductor_dcr = 0.008  # Ω per phase
            self.phase_shift = 180.0  # degrees
            
        elif self.specs.topology == SwitchingTopology.FOUR_SWITCH_BOOST:
            # Four-switch buck-boost
            self.mosfet_rds_on = 0.012  # Ω
            self.inductor_dcr = 0.004  # Ω
            self.capacitor_esr_input = 0.008  # Ω
            self.capacitor_esr_output = 0.004  # Ω
    
    def _build_efficiency_map(self):
        """Build efficiency map based on topology and load conditions"""
        # Create efficiency lookup table
        load_percentages = np.linspace(0.1, 1.0, 10)
        input_voltages = np.linspace(self.specs.input_voltage_range[0], 
                                   self.specs.input_voltage_range[1], 5)
        
        self.efficiency_map = {}
        
        for vin in input_voltages:
            self.efficiency_map[vin] = {}
            for load_pct in load_percentages:
                # Calculate efficiency based on topology and operating point
                efficiency = self._calculate_theoretical_efficiency(vin, load_pct)
                self.efficiency_map[vin][load_pct] = efficiency
    
    def _calculate_theoretical_efficiency(self, vin: float, load_percentage: float) -> float:
        """Calculate theoretical efficiency for given operating point"""
        if self.specs.topology == SwitchingTopology.BOOST:
            # Standard boost efficiency model
            base_efficiency = 0.85 + 0.1 * load_percentage - 0.05 * load_percentage**2
            voltage_factor = 1.0 - 0.02 * abs(vin - 3.0)  # Optimal at 3V
            
        elif self.specs.topology == SwitchingTopology.SYNCHRONOUS_BOOST:
            # Higher efficiency due to synchronous rectification
            base_efficiency = 0.90 + 0.08 * load_percentage - 0.03 * load_percentage**2
            voltage_factor = 1.0 - 0.015 * abs(vin - 3.5)
            
        elif self.specs.topology == SwitchingTopology.INTERLEAVED_BOOST:
            # Lower ripple current improves efficiency
            base_efficiency = 0.88 + 0.09 * load_percentage - 0.04 * load_percentage**2
            voltage_factor = 1.0 - 0.018 * abs(vin - 3.2)
            
        elif self.specs.topology == SwitchingTopology.FOUR_SWITCH_BOOST:
            # Buck-boost topology
            base_efficiency = 0.82 + 0.12 * load_percentage - 0.06 * load_percentage**2
            voltage_factor = 1.0 - 0.025 * abs(vin - 2.5)
            
        else:
            base_efficiency = 0.80
            voltage_factor = 1.0
        
        # Apply temperature derating
        temp_factor = 1.0 - 0.001 * max(0, self.temperature - 25.0)
        
        # Apply aging factor
        aging_factor = 1.0 - 0.0001 * self.operating_hours
        
        total_efficiency = base_efficiency * voltage_factor * temp_factor * aging_factor
        return np.clip(total_efficiency, 0.1, 0.98)
    
    def set_control_mode(self, mode: ControlMode, target_value: float):
        """Set controller operating mode and target value"""
        self.control_mode = mode
        
        if mode == ControlMode.VOLTAGE_MODE:
            self.target_voltage = target_value
        elif mode == ControlMode.CURRENT_MODE:
            self.target_current = target_value
        elif mode == ControlMode.POWER_MODE:
            self.target_power = target_value
        elif mode == ControlMode.MPPT:
            # MPPT mode tracks maximum power point
            pass
    
    def process(self, input_voltage: float, load_current: float, 
                dt: float = 1e-4, time: float = 0.0) -> BoostControllerMeasurement:
        """
        Process boost controller operation for one time step
        
        Args:
            input_voltage: Input voltage from MFC (V)
            load_current: Required load current (A)
            dt: Time step (s)
            time: Current time (hours)
            
        Returns:
            BoostControllerMeasurement object
        """
        # Update operating hours
        self.operating_hours = time
        
        # Calculate output voltage based on duty cycle
        if self.specs.topology in [SwitchingTopology.BOOST, SwitchingTopology.SYNCHRONOUS_BOOST]:
            # Basic boost relationship: Vout = Vin / (1 - D)
            ideal_output_voltage = input_voltage / (1.0 - self.duty_cycle)
        else:
            # More complex topologies have different relationships
            ideal_output_voltage = input_voltage / (1.0 - self.duty_cycle)
        
        # Apply voltage drops and non-idealities
        output_voltage = self._calculate_actual_output_voltage(
            ideal_output_voltage, input_voltage, load_current
        )
        
        # Calculate input current
        input_current = self._calculate_input_current(
            input_voltage, output_voltage, load_current
        )
        
        # Update controller based on mode
        self._update_controller(input_voltage, output_voltage, load_current, dt)
        
        # Calculate efficiency
        input_power = input_voltage * input_current
        output_power = output_voltage * load_current
        efficiency = (output_power / input_power * 100) if input_power > 0 else 0
        
        # Calculate power dissipation and temperature
        power_dissipation = input_power - output_power
        self._update_temperature(power_dissipation, dt)
        
        # Calculate ripple currents and voltages
        inductor_ripple = self._calculate_inductor_current_ripple(
            input_voltage, output_voltage
        )
        output_ripple = self._calculate_output_voltage_ripple(load_current)
        
        # Calculate control error
        if self.control_mode == ControlMode.VOLTAGE_MODE:
            error_voltage = self.target_voltage - output_voltage
        elif self.control_mode == ControlMode.CURRENT_MODE:
            error_voltage = (self.target_current - load_current) * 0.1  # Convert to voltage error
        else:
            error_voltage = 0.0
        
        return BoostControllerMeasurement(
            timestamp=time,
            input_voltage=input_voltage,
            input_current=input_current,
            output_voltage=output_voltage,
            output_current=load_current,
            duty_cycle=self.duty_cycle * 100,  # Convert to percentage
            switching_frequency=self.switching_frequency,
            efficiency=efficiency,
            temperature=self.temperature,
            power_dissipation=power_dissipation,
            control_mode=self.control_mode,
            error_voltage=error_voltage,
            inductor_current_ripple=inductor_ripple,
            output_voltage_ripple=output_ripple
        )
    
    def _calculate_actual_output_voltage(self, ideal_vout: float, vin: float, iload: float) -> float:
        """Calculate actual output voltage considering non-idealities"""
        # Voltage drops due to component resistances
        if self.specs.topology == SwitchingTopology.BOOST:
            # Diode drop and inductor DCR
            diode_drop = self.diode_forward_voltage
            inductor_drop = iload * self.inductor_dcr
            mosfet_drop = vin * self.duty_cycle * self.mosfet_rds_on / (1 - self.duty_cycle)
            
        elif self.specs.topology == SwitchingTopology.SYNCHRONOUS_BOOST:
            # Synchronous MOSFET drop instead of diode
            sync_mosfet_drop = iload * self.mosfet_ls_rds_on
            inductor_drop = iload * self.inductor_dcr
            mosfet_drop = vin * self.duty_cycle * self.mosfet_hs_rds_on / (1 - self.duty_cycle)
            diode_drop = sync_mosfet_drop
            
        else:
            # Generic voltage drops
            diode_drop = 0.5
            inductor_drop = iload * 0.01
            mosfet_drop = vin * self.duty_cycle * 0.01
        
        # ESR drops
        esr_drop = iload * self.capacitor_esr_output
        
        actual_vout = ideal_vout - diode_drop - inductor_drop - mosfet_drop - esr_drop
        
        # Clamp to valid range
        return np.clip(actual_vout, self.specs.output_voltage_range[0], 
                      self.specs.output_voltage_range[1])
    
    def _calculate_input_current(self, vin: float, vout: float, iload: float) -> float:
        """Calculate input current based on power balance and efficiency"""
        output_power = vout * iload
        
        # Look up efficiency from map
        load_percentage = output_power / self.specs.max_output_power
        efficiency = self._get_efficiency_from_map(vin, load_percentage)
        
        input_power = output_power / efficiency if efficiency > 0 else 0
        input_current = input_power / vin if vin > 0 else 0
        
        # Add quiescent current
        input_current += self.specs.quiescent_current / 1000.0  # Convert mA to A
        
        return input_current
    
    def _get_efficiency_from_map(self, vin: float, load_percentage: float) -> float:
        """Get efficiency from lookup table with interpolation"""
        if not self.efficiency_map:
            return 0.85  # Default efficiency
        
        # Find closest voltage points
        voltages = sorted(self.efficiency_map.keys())
        vin_low = max([v for v in voltages if v <= vin], default=voltages[0])
        vin_high = min([v for v in voltages if v >= vin], default=voltages[-1])
        
        # Find closest load points
        load_pcts = sorted(self.efficiency_map[vin_low].keys())
        load_percentage = np.clip(load_percentage, min(load_pcts), max(load_pcts))
        load_low = max([load_pct for load_pct in load_pcts if load_pct <= load_percentage], default=load_pcts[0])
        load_high = min([load_pct for load_pct in load_pcts if load_pct >= load_percentage], default=load_pcts[-1])
        
        # Bilinear interpolation
        if vin_low == vin_high and load_low == load_high:
            return self.efficiency_map[vin_low][load_low]
        elif vin_low == vin_high:
            # Linear interpolation in load
            eff_low = self.efficiency_map[vin_low][load_low]
            eff_high = self.efficiency_map[vin_low][load_high]
            alpha = (load_percentage - load_low) / (load_high - load_low)
            return eff_low + alpha * (eff_high - eff_low)
        elif load_low == load_high:
            # Linear interpolation in voltage
            eff_low = self.efficiency_map[vin_low][load_low]
            eff_high = self.efficiency_map[vin_high][load_low]
            alpha = (vin - vin_low) / (vin_high - vin_low)
            return eff_low + alpha * (eff_high - eff_low)
        else:
            # Bilinear interpolation
            eff_ll = self.efficiency_map[vin_low][load_low]
            eff_lh = self.efficiency_map[vin_low][load_high]
            eff_hl = self.efficiency_map[vin_high][load_low]
            eff_hh = self.efficiency_map[vin_high][load_high]
            
            alpha_v = (vin - vin_low) / (vin_high - vin_low)
            alpha_l = (load_percentage - load_low) / (load_high - load_low)
            
            eff_l = eff_ll + alpha_l * (eff_lh - eff_ll)
            eff_h = eff_hl + alpha_l * (eff_hh - eff_hl)
            
            return eff_l + alpha_v * (eff_h - eff_l)
    
    def _update_controller(self, vin: float, vout: float, iload: float, dt: float):
        """Update control loop and duty cycle"""
        if self.control_mode == ControlMode.VOLTAGE_MODE:
            # Voltage mode control
            error = self.target_voltage - vout
            self.voltage_error_integral += error * dt
            
            # PI controller
            control_output = (self.voltage_controller_kp * error + 
                            self.voltage_controller_ki * self.voltage_error_integral)
            
            # Update duty cycle
            self.duty_cycle += control_output * dt
            
        elif self.control_mode == ControlMode.CURRENT_MODE:
            # Current mode control
            error = self.target_current - iload
            self.current_error_integral += error * dt
            
            # PI controller
            control_output = (self.current_controller_kp * error + 
                            self.current_controller_ki * self.current_error_integral)
            
            self.duty_cycle += control_output * dt
            
        elif self.control_mode == ControlMode.POWER_MODE:
            # Power mode control
            current_power = vout * iload
            error = self.target_power - current_power
            
            # Simple proportional control for power
            control_output = 0.001 * error
            self.duty_cycle += control_output * dt
            
        elif self.control_mode == ControlMode.MPPT:
            # Maximum Power Point Tracking
            self._update_mppt_controller(vin, vout, iload, dt)
        
        # Clamp duty cycle to valid range
        if self.specs.topology == SwitchingTopology.BOOST:
            # Boost converter duty cycle must be < 1
            self.duty_cycle = np.clip(self.duty_cycle, 0.05, 0.95)
        else:
            self.duty_cycle = np.clip(self.duty_cycle, 0.05, 0.95)
    
    def _update_mppt_controller(self, vin: float, vout: float, iload: float, dt: float):
        """Update Maximum Power Point Tracking controller"""
        # Perturb and Observe MPPT algorithm
        input_power = vin * self._calculate_input_current(vin, vout, iload)
        
        if not hasattr(self, 'prev_power'):
            self.prev_power = input_power
            self.prev_duty_cycle = self.duty_cycle
            self.mppt_step_size = 0.01
            return
        
        # Calculate power and duty cycle changes
        dp = input_power - self.prev_power
        dd = self.duty_cycle - self.prev_duty_cycle
        
        if dp > 0:
            # Power increased
            if dd > 0:
                # Duty cycle increased and power increased -> continue in same direction
                self.duty_cycle += self.mppt_step_size
            else:
                # Duty cycle decreased and power increased -> reverse direction
                self.duty_cycle -= self.mppt_step_size
        else:
            # Power decreased
            if dd > 0:
                # Duty cycle increased and power decreased -> reverse direction
                self.duty_cycle -= self.mppt_step_size
            else:
                # Duty cycle decreased and power decreased -> continue in same direction
                self.duty_cycle += self.mppt_step_size
        
        # Update previous values
        self.prev_power = input_power
        self.prev_duty_cycle = self.duty_cycle
    
    def _calculate_inductor_current_ripple(self, vin: float, vout: float) -> float:
        """Calculate inductor current ripple"""
        # Inductor current ripple: ΔIL = Vin * D / (L * fsw)
        delta_il = vin * self.duty_cycle / (self.specs.inductance * self.switching_frequency)
        return delta_il
    
    def _calculate_output_voltage_ripple(self, iload: float) -> float:
        """Calculate output voltage ripple"""
        # Output voltage ripple depends on output capacitance and ESR
        # Simplified model: ΔVout ≈ Iload / (8 * L * C * fsw²) + Iload * ESR / 8
        capacitive_ripple = iload / (8 * self.specs.inductance * 
                                   self.specs.capacitance_output * 
                                   self.switching_frequency**2)
        
        esr_ripple = iload * self.capacitor_esr_output / 8
        
        return capacitive_ripple + esr_ripple
    
    def _update_temperature(self, power_dissipation: float, dt: float):
        """Update junction temperature based on power dissipation"""
        # Simple thermal model: dT/dt = (P*Rth - (T-Tamb)) / τ
        ambient_temp = 25.0  # °C
        thermal_time_constant = 30.0  # seconds
        
        temp_rise = power_dissipation * self.specs.thermal_resistance
        target_temp = ambient_temp + temp_rise
        
        # First-order thermal response
        self.temperature += (target_temp - self.temperature) * dt / thermal_time_constant
    
    def get_power_consumption(self) -> float:
        """Get controller power consumption"""
        # Quiescent power consumption
        quiescent_power = self.specs.quiescent_current / 1000.0 * 3.3  # Assume 3.3V supply
        
        # Switching losses (proportional to frequency)
        switching_power = self.switching_frequency / 100000.0 * 0.1  # W
        
        return quiescent_power + switching_power
    
    def get_cost_analysis(self) -> Dict[str, float]:
        """Get comprehensive cost analysis"""
        initial_cost = self.specs.cost
        
        # Operating cost per hour
        power_consumption = self.get_power_consumption()
        power_cost_per_hour = power_consumption * 0.15 / 1000  # $0.15/kWh
        
        # Maintenance cost (component replacement)
        if self.specs.topology == SwitchingTopology.SYNCHRONOUS_BOOST:
            maintenance_cost_per_hour = 0.02  # Higher complexity
        else:
            maintenance_cost_per_hour = 0.01
        
        # Replacement cost amortized over lifetime
        replacement_cost_per_hour = initial_cost / self.specs.lifetime
        
        total_cost_per_hour = power_cost_per_hour + maintenance_cost_per_hour + replacement_cost_per_hour
        
        return {
            'initial_cost': initial_cost,
            'power_cost_per_hour': power_cost_per_hour,
            'maintenance_cost_per_hour': maintenance_cost_per_hour,
            'replacement_cost_per_hour': replacement_cost_per_hour,
            'total_cost_per_hour': total_cost_per_hour,
            'lifetime_cost': initial_cost + total_cost_per_hour * self.specs.lifetime
        }
    
    def generate_spice_model(self, output_file: Optional[str] = None, 
                           operating_point: Optional[Dict[str, float]] = None) -> str:
        """
        Generate SPICE netlist model of the boost converter
        
        Args:
            output_file: Optional file path to save the SPICE model
            operating_point: Optional operating point for component values
            
        Returns:
            SPICE netlist as string
        """
        if operating_point is None:
            operating_point = {
                'input_voltage': 2.0,  # V
                'output_voltage': 5.0,  # V
                'output_current': 1.0,  # A
                'duty_cycle': 0.6  # 60%
            }
        
        # Calculate component values based on operating point
        vin = operating_point['input_voltage']
        vout = operating_point['output_voltage']
        iout = operating_point['output_current']
        duty = operating_point['duty_cycle']
        
        # Generate SPICE netlist based on topology
        spice_model = self._generate_spice_header()
        spice_model += self._generate_spice_parameters(vin, vout, iout, duty)
        spice_model += self._generate_spice_circuit()
        spice_model += self._generate_spice_analysis()
        spice_model += self._generate_spice_footer()
        
        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(spice_model)
            logger.info(f"SPICE model saved to {output_file}")
        
        return spice_model
    
    def _generate_spice_header(self) -> str:
        """Generate SPICE model header"""
        header = f"""* Boost Converter SPICE Model
* Topology: {self.specs.topology.value}
* Generated by MFC Power Models
* 
* Input Voltage Range: {self.specs.input_voltage_range[0]:.1f}V - {self.specs.input_voltage_range[1]:.1f}V
* Output Voltage Range: {self.specs.output_voltage_range[0]:.1f}V - {self.specs.output_voltage_range[1]:.1f}V
* Maximum Output Power: {self.specs.max_output_power:.1f}W
* Switching Frequency: {self.specs.switching_frequency/1000:.0f}kHz

"""
        return header
    
    def _generate_spice_parameters(self, vin: float, vout: float, iout: float, duty: float) -> str:
        """Generate SPICE parameter definitions"""
        fsw = self.specs.switching_frequency
        
        params = f"""* Circuit Parameters
.param Vin={vin:.2f}
.param Vout={vout:.2f}
.param Iout={iout:.3f}
.param Duty={duty:.3f}
.param Fsw={fsw:.0f}
.param Period={{1/Fsw}}
.param Ton={{Duty*Period}}
.param Toff={{(1-Duty)*Period}}

* Component Values
.param L={self.specs.inductance*1e6:.1f}u
.param Cin={self.specs.capacitance_input*1e6:.0f}u
.param Cout={self.specs.capacitance_output*1e6:.0f}u
.param Rload={{Vout/Iout}}

"""
        return params
    
    def _generate_spice_circuit(self) -> str:
        """Generate SPICE circuit netlist based on topology"""
        if self.specs.topology == SwitchingTopology.BOOST:
            return self._generate_basic_boost_circuit()
        elif self.specs.topology == SwitchingTopology.SYNCHRONOUS_BOOST:
            return self._generate_synchronous_boost_circuit()
        elif self.specs.topology == SwitchingTopology.INTERLEAVED_BOOST:
            return self._generate_interleaved_boost_circuit()
        elif self.specs.topology == SwitchingTopology.FOUR_SWITCH_BOOST:
            return self._generate_four_switch_boost_circuit()
        else:
            return self._generate_basic_boost_circuit()
    
    def _generate_basic_boost_circuit(self) -> str:
        """Generate basic boost converter SPICE circuit"""
        circuit = f"""* Basic Boost Converter Circuit
* Input voltage source
Vin input 0 DC {{Vin}}

* Input capacitor with ESR
Cin input n_vin {{Cin}}
Rcin_esr n_vin sw_node {self.capacitor_esr_input:.4f}

* Main inductor with DCR
L1 n_vin n_lx {{L}}
Rl_dcr n_lx sw_node {self.inductor_dcr:.4f}

* Main switching MOSFET (ideal switch for simplicity)
Sm1 sw_node 0 vcont 0 SWITCH Roff=1Meg Ron={self.mosfet_rds_on:.4f}

* Freewheeling diode
D1 sw_node output DIODE

* Output capacitor with ESR
Cout output n_out {{Cout}}
Rcout_esr n_out 0 {self.capacitor_esr_output:.4f}

* Load resistor
Rload output 0 {{Rload}}

* PWM Control voltage source
Vcont vcont 0 PULSE(0 5 0 1n 1n {{Ton}} {{Period}})

* Diode model
.model DIODE D(Is=1e-14 Rs=0.01 Vj={self.diode_forward_voltage:.2f})

* Switch model  
.model SWITCH SW(Vt=2.5 Vh=0.1)

"""
        return circuit
    
    def _generate_synchronous_boost_circuit(self) -> str:
        """Generate synchronous boost converter SPICE circuit"""
        circuit = f"""* Synchronous Boost Converter Circuit
* Input voltage source
Vin input 0 DC {{Vin}}

* Input capacitor with ESR
Cin input n_vin {{Cin}}
Rcin_esr n_vin sw_node {self.capacitor_esr_input:.4f}

* Main inductor with DCR
L1 n_vin n_lx {{L}}
Rl_dcr n_lx sw_node {self.inductor_dcr:.4f}

* High-side switching MOSFET
Sm1 sw_node 0 vcont_hs 0 SWITCH Roff=1Meg Ron={self.mosfet_hs_rds_on:.4f}

* Low-side synchronous MOSFET
Sm2 output sw_node vcont_ls 0 SWITCH Roff=1Meg Ron={self.mosfet_ls_rds_on:.4f}

* Output capacitor with ESR
Cout output n_out {{Cout}}
Rcout_esr n_out 0 {self.capacitor_esr_output:.4f}

* Load resistor
Rload output 0 {{Rload}}

* PWM Control voltage sources with dead time
Vcont_hs vcont_hs 0 PULSE(0 5 0 1n 1n {{Ton}} {{Period}})
Vcont_ls vcont_ls 0 PULSE(5 0 {{Ton+{self.dead_time*1e9:.0f}n}} 1n 1n {{Toff-{self.dead_time*1e9:.0f}n}} {{Period}})

* Switch model  
.model SWITCH SW(Vt=2.5 Vh=0.1)

"""
        return circuit
    
    def _generate_interleaved_boost_circuit(self) -> str:
        """Generate interleaved boost converter SPICE circuit"""
        circuit = f"""* Interleaved Boost Converter Circuit (2-Phase)
* Input voltage source
Vin input 0 DC {{Vin}}

* Input capacitor with ESR
Cin input n_vin {{Cin}}
Rcin_esr n_vin common {self.capacitor_esr_input:.4f}

* Phase 1 - Main inductor with DCR
L1 n_vin n_lx1 {{L}}
Rl1_dcr n_lx1 sw_node1 {self.inductor_dcr:.4f}

* Phase 1 - Main switching MOSFET
Sm1 sw_node1 0 vcont1 0 SWITCH Roff=1Meg Ron={self.mosfet_rds_on:.4f}

* Phase 1 - Freewheeling diode
D1 sw_node1 output DIODE

* Phase 2 - Main inductor with DCR
L2 common n_lx2 {{L}}
Rl2_dcr n_lx2 sw_node2 {self.inductor_dcr:.4f}

* Phase 2 - Main switching MOSFET
Sm2 sw_node2 0 vcont2 0 SWITCH Roff=1Meg Ron={self.mosfet_rds_on:.4f}

* Phase 2 - Freewheeling diode
D2 sw_node2 output DIODE

* Output capacitor with ESR
Cout output n_out {{Cout}}
Rcout_esr n_out 0 {self.capacitor_esr_output:.4f}

* Load resistor
Rload output 0 {{Rload}}

* PWM Control voltage sources (180° phase shift)
Vcont1 vcont1 0 PULSE(0 5 0 1n 1n {{Ton}} {{Period}})
Vcont2 vcont2 0 PULSE(0 5 {{Period/2}} 1n 1n {{Ton}} {{Period}})

* Component models
.model DIODE D(Is=1e-14 Rs=0.01 Vj={self.diode_forward_voltage:.2f})
.model SWITCH SW(Vt=2.5 Vh=0.1)

"""
        return circuit
    
    def _generate_four_switch_boost_circuit(self) -> str:
        """Generate four-switch buck-boost converter SPICE circuit"""
        circuit = f"""* Four-Switch Buck-Boost Converter Circuit
* Input voltage source
Vin input 0 DC {{Vin}}

* Input capacitor with ESR
Cin input n_vin {{Cin}}
Rcin_esr n_vin n_in {self.capacitor_esr_input:.4f}

* Main inductor with DCR
L1 n_in n_lx {{L}}
Rl_dcr n_lx n_mid {self.inductor_dcr:.4f}

* Input-side switches
Sm1 n_in n_mid vcont1 0 SWITCH Roff=1Meg Ron={self.mosfet_rds_on:.4f}
Sm2 n_mid 0 vcont2 0 SWITCH Roff=1Meg Ron={self.mosfet_rds_on:.4f}

* Output-side switches  
Sm3 n_mid output vcont3 0 SWITCH Roff=1Meg Ron={self.mosfet_rds_on:.4f}
Sm4 output 0 vcont4 0 SWITCH Roff=1Meg Ron={self.mosfet_rds_on:.4f}

* Output capacitor with ESR
Cout output n_out {{Cout}}
Rcout_esr n_out 0 {self.capacitor_esr_output:.4f}

* Load resistor
Rload output 0 {{Rload}}

* PWM Control voltage sources (complementary switching)
Vcont1 vcont1 0 PULSE(0 5 0 1n 1n {{Ton}} {{Period}})
Vcont2 vcont2 0 PULSE(5 0 {{Ton}} 1n 1n {{Toff}} {{Period}})
Vcont3 vcont3 0 PULSE(5 0 0 1n 1n {{Ton}} {{Period}})
Vcont4 vcont4 0 PULSE(0 5 {{Ton}} 1n 1n {{Toff}} {{Period}})

* Switch model  
.model SWITCH SW(Vt=2.5 Vh=0.1)

"""
        return circuit
    
    def _generate_spice_analysis(self) -> str:
        """Generate SPICE analysis commands"""
        period = 1.0 / self.specs.switching_frequency
        sim_time = period * 20  # Simulate 20 switching cycles
        
        analysis = f"""* Analysis Commands
* Transient analysis
.tran 1n {sim_time*1e6:.0f}u

* Operating point analysis
.op

* AC analysis for stability
.ac dec 100 1 1Meg

* Measurements
.measure tran avg_vout AVG V(output) FROM {sim_time*0.5*1e6:.0f}u TO {sim_time*1e6:.0f}u
.measure tran avg_iout AVG I(Rload) FROM {sim_time*0.5*1e6:.0f}u TO {sim_time*1e6:.0f}u
.measure tran avg_iin AVG I(Vin) FROM {sim_time*0.5*1e6:.0f}u TO {sim_time*1e6:.0f}u
.measure tran ripple_vout PP V(output) FROM {sim_time*0.5*1e6:.0f}u TO {sim_time*1e6:.0f}u
.measure tran efficiency PARAM='avg_vout*avg_iout/({{Vin}}*avg_iin)*100'

* Output commands
.print tran V(input) V(output) I(L1) I(Rload)
.plot tran V(output) V(sw_node)

"""
        return analysis
    
    def _generate_spice_footer(self) -> str:
        """Generate SPICE model footer"""
        footer = """* End of boost converter model
.end
"""
        return footer
    
    def export_spice_testbench(self, output_dir: str, test_conditions: List[Dict[str, float]]) -> List[str]:
        """
        Export SPICE testbench with multiple operating conditions
        
        Args:
            output_dir: Directory to save testbench files
            test_conditions: List of operating condition dictionaries
            
        Returns:
            List of generated file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        for i, condition in enumerate(test_conditions):
            filename = f"boost_converter_test_{i+1:02d}.cir"
            filepath = output_path / filename
            
            # Generate SPICE model for this condition
            spice_model = self.generate_spice_model(operating_point=condition)
            
            # Add testbench-specific analysis
            testbench_header = f"""* Boost Converter Testbench #{i+1}
* Test Condition: Vin={condition.get('input_voltage', 2.0):.1f}V, 
*                 Vout={condition.get('output_voltage', 5.0):.1f}V,
*                 Iout={condition.get('output_current', 1.0):.1f}A

"""
            
            # Write complete testbench
            with open(filepath, 'w') as f:
                f.write(testbench_header)
                f.write(spice_model)
            
            generated_files.append(str(filepath))
            logger.info(f"Generated SPICE testbench: {filepath}")
        
        # Generate batch simulation script
        batch_script = self._generate_batch_script(output_path, generated_files)
        script_path = output_path / "run_all_tests.sh"
        with open(script_path, 'w') as f:
            f.write(batch_script)
        os.chmod(script_path, 0o755)  # Make executable
        
        generated_files.append(str(script_path))
        
        return generated_files
    
    def _generate_batch_script(self, output_dir: Path, spice_files: List[str]) -> str:
        """Generate batch script to run all SPICE simulations"""
        script = """#!/bin/bash
# Batch script to run all boost converter SPICE simulations

echo "Running boost converter SPICE simulations..."

"""
        
        for spice_file in spice_files:
            if spice_file.endswith('.cir'):
                filename = Path(spice_file).name
                script += f"""
echo "Running {filename}..."
ngspice -b {filename} > {filename.replace('.cir', '.log')} 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ {filename} completed successfully"
else
    echo "  ✗ {filename} failed"
fi
"""
        
        script += """
echo "All simulations completed."
echo "Check individual .log files for results."
"""
        
        return script


def create_standard_boost_controllers() -> Dict[str, BoostController]:
    """Create standard boost controller configurations"""
    
    # Basic boost converter
    basic_boost_specs = BoostControllerSpecs(
        topology=SwitchingTopology.BOOST,
        input_voltage_range=(0.5, 5.0),  # V
        output_voltage_range=(3.3, 24.0),  # V
        max_output_current=2.0,  # A
        max_output_power=10.0,  # W
        switching_frequency=100000.0,  # 100 kHz
        efficiency_at_rated=85.0,  # %
        quiescent_current=2.0,  # mA
        cost=25.0,  # USD
        lifetime=87600.0,  # 10 years
        thermal_resistance=50.0,  # °C/W
        inductance=22e-6,  # 22 µH
        capacitance_input=100e-6,  # 100 µF
        capacitance_output=220e-6,  # 220 µF
        control_bandwidth=10000.0  # 10 kHz
    )
    
    # Synchronous boost converter
    sync_boost_specs = BoostControllerSpecs(
        topology=SwitchingTopology.SYNCHRONOUS_BOOST,
        input_voltage_range=(0.3, 5.5),  # V
        output_voltage_range=(3.3, 28.0),  # V
        max_output_current=3.0,  # A
        max_output_power=20.0,  # W
        switching_frequency=200000.0,  # 200 kHz
        efficiency_at_rated=92.0,  # %
        quiescent_current=1.5,  # mA
        cost=45.0,  # USD
        lifetime=87600.0,  # 10 years
        thermal_resistance=30.0,  # °C/W
        inductance=10e-6,  # 10 µH
        capacitance_input=100e-6,  # 100 µF
        capacitance_output=470e-6,  # 470 µF
        control_bandwidth=20000.0  # 20 kHz
    )
    
    # High-power interleaved boost
    interleaved_boost_specs = BoostControllerSpecs(
        topology=SwitchingTopology.INTERLEAVED_BOOST,
        input_voltage_range=(0.8, 8.0),  # V
        output_voltage_range=(5.0, 48.0),  # V
        max_output_current=10.0,  # A
        max_output_power=100.0,  # W
        switching_frequency=150000.0,  # 150 kHz
        efficiency_at_rated=88.0,  # %
        quiescent_current=5.0,  # mA
        cost=120.0,  # USD
        lifetime=70080.0,  # 8 years
        thermal_resistance=20.0,  # °C/W
        inductance=15e-6,  # 15 µH per phase
        capacitance_input=220e-6,  # 220 µF
        capacitance_output=1000e-6,  # 1000 µF
        control_bandwidth=15000.0  # 15 kHz
    )
    
    # Ultra-low power boost for energy harvesting
    micro_boost_specs = BoostControllerSpecs(
        topology=SwitchingTopology.BOOST,
        input_voltage_range=(0.1, 1.5),  # V
        output_voltage_range=(1.8, 5.0),  # V
        max_output_current=0.1,  # A
        max_output_power=0.5,  # W
        switching_frequency=500000.0,  # 500 kHz
        efficiency_at_rated=78.0,  # %
        quiescent_current=0.5,  # mA
        cost=15.0,  # USD
        lifetime=131400.0,  # 15 years
        thermal_resistance=100.0,  # °C/W
        inductance=4.7e-6,  # 4.7 µH
        capacitance_input=47e-6,  # 47 µF
        capacitance_output=100e-6,  # 100 µF
        control_bandwidth=50000.0  # 50 kHz
    )
    
    controllers = {
        'basic_boost': BoostController(basic_boost_specs),
        'synchronous_boost': BoostController(sync_boost_specs),
        'interleaved_boost': BoostController(interleaved_boost_specs),
        'micro_boost': BoostController(micro_boost_specs)
    }
    
    return controllers


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create standard controllers
    controllers = create_standard_boost_controllers()
    
    # Test synchronous boost controller
    sync_boost = controllers['synchronous_boost']
    sync_boost.set_control_mode(ControlMode.VOLTAGE_MODE, 5.0)  # 5V output
    
    # Simulate MFC voltage and load variations
    time_points = np.linspace(0, 100, 1001)  # 100 hours simulation
    mfc_voltages = 0.5 + 0.3 * np.sin(2 * np.pi * time_points / 24.0)  # Daily variation
    load_currents = 0.5 + 0.3 * np.sin(2 * np.pi * time_points / 12.0)  # Load variation
    
    measurements = []
    for i, t in enumerate(time_points):
        dt = 0.1 if i > 0 else 0.0  # 0.1 hour time step
        measurement = sync_boost.process(
            input_voltage=mfc_voltages[i],
            load_current=load_currents[i],
            dt=dt * 3600,  # Convert to seconds
            time=t
        )
        measurements.append(measurement)
        
        # Log every 10 hours
        if t % 10 < 0.1:
            logger.info(f"Hour {t:.1f}: Vin={measurement.input_voltage:.2f}V, "
                       f"Vout={measurement.output_voltage:.2f}V, "
                       f"Efficiency={measurement.efficiency:.1f}%, "
                       f"Duty={measurement.duty_cycle:.1f}%")
    
    # Analyze results
    efficiencies = [m.efficiency for m in measurements]
    output_voltages = [m.output_voltage for m in measurements]
    duty_cycles = [m.duty_cycle for m in measurements]
    
    print("\nSimulation Results:")
    print(f"Average efficiency: {np.mean(efficiencies):.1f}%")
    print(f"Output voltage regulation: {np.std(output_voltages):.3f}V RMS")
    print(f"Duty cycle range: {np.min(duty_cycles):.1f}% - {np.max(duty_cycles):.1f}%")
    
    # Cost analysis
    cost_analysis = sync_boost.get_cost_analysis()
    print(f"Total lifetime cost: ${cost_analysis['lifetime_cost']:.2f}")
    print(f"Operating cost per hour: ${cost_analysis['total_cost_per_hour']:.6f}")
    
    # Test MPPT mode
    print("\nTesting MPPT mode...")
    sync_boost.set_control_mode(ControlMode.MPPT, 0.0)
    
    # Simulate varying MFC characteristics
    for i in range(100):
        vin = 0.8 + 0.1 * np.random.random()  # Random MFC voltage
        iload = 0.8  # Fixed load
        measurement = sync_boost.process(vin, iload, dt=0.01, time=i*0.01)
        
        if i % 20 == 0:
            input_power = measurement.input_voltage * measurement.input_current
            print(f"Step {i}: Vin={vin:.2f}V, Pin={input_power:.3f}W, "
                  f"Duty={measurement.duty_cycle:.1f}%")
    
    # Test SPICE model generation
    print("\nTesting SPICE model generation...")
    
    # Generate SPICE model for current operating point
    operating_point = {
        'input_voltage': 2.5,
        'output_voltage': 5.0, 
        'output_current': 1.0,
        'duty_cycle': 0.6
    }
    
    spice_model = sync_boost.generate_spice_model(
        output_file="boost_converter_example.cir",
        operating_point=operating_point
    )
    
    print(f"Generated SPICE model ({len(spice_model)} characters)")
    print("First 500 characters:")
    print(spice_model[:500] + "...")
    
    # Generate testbench with multiple operating conditions
    test_conditions = [
        {'input_voltage': 1.0, 'output_voltage': 3.3, 'output_current': 0.5, 'duty_cycle': 0.7},
        {'input_voltage': 2.0, 'output_voltage': 5.0, 'output_current': 1.0, 'duty_cycle': 0.6},
        {'input_voltage': 3.0, 'output_voltage': 12.0, 'output_current': 0.8, 'duty_cycle': 0.75},
        {'input_voltage': 0.8, 'output_voltage': 5.0, 'output_current': 0.3, 'duty_cycle': 0.84}
    ]
    
    generated_files = sync_boost.export_spice_testbench("spice_testbench", test_conditions)
    print(f"\nGenerated {len(generated_files)} SPICE testbench files:")
    for file_path in generated_files:
        print(f"  - {file_path}")