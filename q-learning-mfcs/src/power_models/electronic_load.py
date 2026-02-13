"""
Electronic Load Models for MFC Power Testing

This module implements various electronic load modes for testing MFC power
systems including constant current, constant voltage, constant power, and
constant resistance modes with SPICE model generation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class LoadMode(Enum):
    """Electronic load operating modes"""
    CONSTANT_CURRENT = "constant_current"
    CONSTANT_VOLTAGE = "constant_voltage"
    CONSTANT_POWER = "constant_power"
    CONSTANT_RESISTANCE = "constant_resistance"
    BATTERY_DISCHARGE = "battery_discharge"
    SHORT_CIRCUIT = "short_circuit"
    OPEN_CIRCUIT = "open_circuit"


class LoadTopology(Enum):
    """Electronic load circuit topologies"""
    LINEAR_MOSFET = "linear_mosfet"
    SWITCHING_MOSFET = "switching_mosfet"
    RESISTIVE_ARRAY = "resistive_array"
    ACTIVE_RECTIFIER = "active_rectifier"


@dataclass
class ElectronicLoadSpecs:
    """Specifications for electronic load"""
    topology: LoadTopology
    voltage_range: Tuple[float, float]  # V (min, max)
    current_range: Tuple[float, float]  # A (min, max)
    power_rating: float  # W (maximum power dissipation)
    accuracy_current: float  # % of reading
    accuracy_voltage: float  # % of reading
    accuracy_power: float  # % of reading
    resolution_current: float  # A
    resolution_voltage: float  # V
    bandwidth: float  # Hz (control bandwidth)
    slew_rate: float  # A/s (current slew rate)
    thermal_resistance: float  # °C/W
    cooling_method: str  # air, liquid, passive
    cost: float  # USD
    weight: float  # kg
    efficiency: float  # % (power conversion efficiency)


@dataclass
class ElectronicLoadMeasurement:
    """Single electronic load measurement"""
    timestamp: float
    voltage: float  # V
    current: float  # A
    power: float  # W
    resistance: float  # Ω
    load_mode: LoadMode
    setpoint_value: float  # Target value for current mode
    temperature: float  # °C
    efficiency: float  # %
    ripple_current: float  # A RMS
    ripple_voltage: float  # V RMS
    control_error: float  # % error from setpoint


class ElectronicLoad:
    """Electronic load model with multiple operating modes"""
    
    def __init__(self, specs: ElectronicLoadSpecs):
        self.specs = specs
        self.load_mode = LoadMode.CONSTANT_CURRENT
        self.setpoint_current = 1.0  # A
        self.setpoint_voltage = 5.0  # V
        self.setpoint_power = 5.0  # W
        self.setpoint_resistance = 5.0  # Ω
        
        # Operating state
        self.temperature = 25.0  # °C
        self.operating_hours = 0.0
        self.enabled = True
        
        # Control loop parameters
        self.control_kp = 1.0
        self.control_ki = 10.0
        self.control_integral = 0.0
        
        # Initialize topology-specific parameters
        self._initialize_topology_parameters()
        
    def _initialize_topology_parameters(self):
        """Initialize topology-specific parameters"""
        if self.specs.topology == LoadTopology.LINEAR_MOSFET:
            # Linear MOSFET load parameters
            self.mosfet_gm = 10.0  # S (transconductance)
            self.mosfet_rds_on = 0.01  # Ω
            self.gate_capacitance = 1000e-12  # F
            self.thermal_time_constant = 10.0  # s
            
        elif self.specs.topology == LoadTopology.SWITCHING_MOSFET:
            # Switching MOSFET load parameters
            self.switching_frequency = 100000.0  # Hz
            self.duty_cycle = 0.5
            self.inductor_value = 100e-6  # H
            self.output_capacitance = 100e-6  # F
            
        elif self.specs.topology == LoadTopology.RESISTIVE_ARRAY:
            # Resistive array parameters
            self.num_resistors = 16
            self.base_resistance = 1.0  # Ω
            self.switch_resistance = 0.01  # Ω
            self.resistor_tolerance = 0.01  # 1%
            
        elif self.specs.topology == LoadTopology.ACTIVE_RECTIFIER:
            # Active rectifier parameters
            self.rectifier_efficiency = 0.95
            self.switching_frequency = 50000.0  # Hz
            self.output_inductance = 1000e-6  # H
            self.synchronous_rectification = True
    
    def set_mode(self, mode: LoadMode, setpoint: float):
        """Set load operating mode and setpoint value"""
        self.load_mode = mode
        
        if mode == LoadMode.CONSTANT_CURRENT:
            self.setpoint_current = min(setpoint, self.specs.current_range[1])
        elif mode == LoadMode.CONSTANT_VOLTAGE:
            self.setpoint_voltage = setpoint
        elif mode == LoadMode.CONSTANT_POWER:
            self.setpoint_power = min(setpoint, self.specs.power_rating)
        elif mode == LoadMode.CONSTANT_RESISTANCE:
            self.setpoint_resistance = setpoint
        elif mode == LoadMode.SHORT_CIRCUIT:
            self.setpoint_resistance = 0.001  # Very low resistance
        elif mode == LoadMode.OPEN_CIRCUIT:
            self.setpoint_current = 0.0
        # BATTERY_DISCHARGE mode uses CV-CC profile
    
    def process(self, input_voltage: float, dt: float = 1e-4, time: float = 0.0) -> ElectronicLoadMeasurement:
        """
        Process electronic load operation for one time step
        
        Args:
            input_voltage: Input voltage from source (V)
            dt: Time step (s)
            time: Current time (hours)
            
        Returns:
            ElectronicLoadMeasurement object
        """
        # Update operating hours
        self.operating_hours = time
        
        if not self.enabled:
            return self._create_measurement(input_voltage, 0.0, 0.0, time)
        
        # Check voltage limits
        if input_voltage < self.specs.voltage_range[0] or input_voltage > self.specs.voltage_range[1]:
            self.enabled = False
            return self._create_measurement(input_voltage, 0.0, 0.0, time)
        
        # Calculate load current based on mode
        target_current = self._calculate_target_current(input_voltage)
        
        # Apply control loop
        actual_current = self._apply_control_loop(target_current, dt)
        
        # Apply current limits
        actual_current = np.clip(actual_current, 
                               self.specs.current_range[0], 
                               self.specs.current_range[1])
        
        # Check power limits
        power = input_voltage * actual_current
        if power > self.specs.power_rating:
            actual_current = self.specs.power_rating / input_voltage
            power = self.specs.power_rating
        
        # Update temperature
        self._update_temperature(power, dt)
        
        # Calculate resistance
        resistance = input_voltage / actual_current if actual_current > 0 else float('inf')
        
        return self._create_measurement(input_voltage, actual_current, power, time, resistance)
    
    def _calculate_target_current(self, voltage: float) -> float:
        """Calculate target current based on load mode"""
        if self.load_mode == LoadMode.CONSTANT_CURRENT:
            return self.setpoint_current
            
        elif self.load_mode == LoadMode.CONSTANT_VOLTAGE:
            # For CV mode, we need to regulate the voltage across the load
            # This is more complex in practice, simplified here
            if voltage > self.setpoint_voltage:
                return self.specs.current_range[1]  # Maximum current to pull voltage down
            else:
                return 0.0  # No current if voltage is below setpoint
                
        elif self.load_mode == LoadMode.CONSTANT_POWER:
            return self.setpoint_power / voltage if voltage > 0 else 0.0
            
        elif self.load_mode == LoadMode.CONSTANT_RESISTANCE:
            return voltage / self.setpoint_resistance if self.setpoint_resistance > 0 else 0.0
            
        elif self.load_mode == LoadMode.BATTERY_DISCHARGE:
            # Battery discharge profile (CV-CC)
            if voltage > 3.0:  # Above cutoff voltage
                return min(self.setpoint_current, self.setpoint_power / voltage)
            else:
                return 0.0  # Stop discharge at cutoff
                
        elif self.load_mode == LoadMode.SHORT_CIRCUIT:
            return self.specs.current_range[1]  # Maximum current
            
        elif self.load_mode == LoadMode.OPEN_CIRCUIT:
            return 0.0
            
        else:
            return 0.0
    
    def _apply_control_loop(self, target_current: float, dt: float) -> float:
        """Apply control loop to reach target current"""
        # Simple PI controller
        error = target_current - getattr(self, '_last_current', 0.0)
        self.control_integral += error * dt
        
        # Anti-windup
        self.control_integral = np.clip(self.control_integral, -1.0, 1.0)
        
        # PI control output
        control_output = self.control_kp * error + self.control_ki * self.control_integral
        
        # Apply slew rate limiting
        if hasattr(self, '_last_current'):
            max_change = self.specs.slew_rate * dt
            control_output = np.clip(control_output, 
                                   self._last_current - max_change,
                                   self._last_current + max_change)
        
        actual_current = target_current + control_output * dt
        self._last_current = actual_current
        
        return actual_current
    
    def _update_temperature(self, power: float, dt: float):
        """Update junction temperature based on power dissipation"""
        ambient_temp = 25.0  # °C
        
        # Calculate efficiency based on topology
        if self.specs.topology == LoadTopology.LINEAR_MOSFET:
            # Linear regulation - lower efficiency
            efficiency = max(0.1, 1.0 - power / self.specs.power_rating * 0.5)
        elif self.specs.topology == LoadTopology.SWITCHING_MOSFET:
            # Switching regulation - higher efficiency
            efficiency = self.specs.efficiency / 100.0
        else:
            efficiency = 0.8  # Default efficiency
        
        # Power dissipated as heat
        power_dissipated = power * (1.0 - efficiency)
        
        # Thermal model
        temp_rise = power_dissipated * self.specs.thermal_resistance
        target_temp = ambient_temp + temp_rise
        
        # First-order thermal response
        thermal_time_constant = getattr(self, 'thermal_time_constant', 30.0)
        self.temperature += (target_temp - self.temperature) * dt / thermal_time_constant
        
        # Thermal protection
        if self.temperature > 85.0:  # Thermal shutdown
            self.enabled = False
    
    def _create_measurement(self, voltage: float, current: float, power: float, 
                          time: float, resistance: Optional[float] = None) -> ElectronicLoadMeasurement:
        """Create an electronic load measurement object"""
        if resistance is None:
            resistance = voltage / current if current > 0 else float('inf')
        
        # Calculate setpoint value for current mode
        if self.load_mode == LoadMode.CONSTANT_CURRENT:
            setpoint_value = self.setpoint_current
            control_error = abs(current - setpoint_value) / setpoint_value * 100 if setpoint_value > 0 else 0
        elif self.load_mode == LoadMode.CONSTANT_VOLTAGE:
            setpoint_value = self.setpoint_voltage
            control_error = abs(voltage - setpoint_value) / setpoint_value * 100 if setpoint_value > 0 else 0
        elif self.load_mode == LoadMode.CONSTANT_POWER:
            setpoint_value = self.setpoint_power
            control_error = abs(power - setpoint_value) / setpoint_value * 100 if setpoint_value > 0 else 0
        elif self.load_mode == LoadMode.CONSTANT_RESISTANCE:
            setpoint_value = self.setpoint_resistance
            control_error = abs(resistance - setpoint_value) / setpoint_value * 100 if setpoint_value > 0 else 0
        else:
            setpoint_value = 0.0
            control_error = 0.0
        
        # Calculate efficiency
        if self.specs.topology == LoadTopology.LINEAR_MOSFET:
            efficiency = max(10.0, 100.0 * (1.0 - power / self.specs.power_rating * 0.3))
        else:
            efficiency = self.specs.efficiency
        
        # Calculate ripple (simplified)
        if self.specs.topology == LoadTopology.SWITCHING_MOSFET:
            ripple_current = current * 0.05  # 5% ripple
            ripple_voltage = voltage * 0.01  # 1% ripple
        else:
            ripple_current = current * 0.01  # 1% ripple
            ripple_voltage = voltage * 0.005  # 0.5% ripple
        
        return ElectronicLoadMeasurement(
            timestamp=time,
            voltage=voltage,
            current=current,
            power=power,
            resistance=resistance,
            load_mode=self.load_mode,
            setpoint_value=setpoint_value,
            temperature=self.temperature,
            efficiency=efficiency,
            ripple_current=ripple_current,
            ripple_voltage=ripple_voltage,
            control_error=control_error
        )
    
    def get_power_consumption(self) -> float:
        """Get load control circuitry power consumption"""
        base_power = 0.1  # W for control circuits
        
        if self.specs.topology == LoadTopology.SWITCHING_MOSFET:
            # Additional power for switching control
            switching_power = 0.05  # W
        elif self.specs.topology == LoadTopology.RESISTIVE_ARRAY:
            # Power for relay/switch control
            switching_power = 0.02 * self.num_resistors  # W
        else:
            switching_power = 0.02  # W
        
        return base_power + switching_power
    
    def get_cost_analysis(self) -> Dict[str, float]:
        """Get comprehensive cost analysis"""
        initial_cost = self.specs.cost
        
        # Operating cost per hour
        control_power_cost = self.get_power_consumption() * 0.15 / 1000  # $0.15/kWh
        
        # Cooling cost (for high-power loads)
        if self.specs.power_rating > 100.0:
            cooling_cost_per_hour = 0.01  # Additional cooling cost
        else:
            cooling_cost_per_hour = 0.0
        
        # Maintenance cost
        if self.specs.topology == LoadTopology.RESISTIVE_ARRAY:
            maintenance_cost_per_hour = 0.005  # Resistor replacement
        else:
            maintenance_cost_per_hour = 0.002
        
        # Calibration cost
        calibration_cost_per_hour = initial_cost * 0.001 / 8760  # 0.1% per year
        
        total_cost_per_hour = (control_power_cost + cooling_cost_per_hour + 
                             maintenance_cost_per_hour + calibration_cost_per_hour)
        
        return {
            'initial_cost': initial_cost,
            'power_cost_per_hour': control_power_cost,
            'cooling_cost_per_hour': cooling_cost_per_hour,
            'maintenance_cost_per_hour': maintenance_cost_per_hour,
            'calibration_cost_per_hour': calibration_cost_per_hour,
            'total_cost_per_hour': total_cost_per_hour,
            'lifetime_cost': initial_cost + total_cost_per_hour * 87600  # 10 years
        }
    
    def generate_spice_model(self, output_file: Optional[str] = None,
                           operating_point: Optional[Dict[str, float]] = None) -> str:
        """
        Generate SPICE netlist model of the electronic load
        
        Args:
            output_file: Optional file path to save the SPICE model
            operating_point: Optional operating point for component values
            
        Returns:
            SPICE netlist as string
        """
        if operating_point is None:
            operating_point = {
                'input_voltage': 5.0,  # V
                'load_current': 1.0,  # A
                'load_power': 5.0,  # W
                'mode': self.load_mode.value
            }
        
        # Generate SPICE netlist
        spice_model = self._generate_load_spice_header()
        spice_model += self._generate_load_spice_parameters(operating_point)
        spice_model += self._generate_load_spice_circuit()
        spice_model += self._generate_load_spice_analysis()
        spice_model += self._generate_load_spice_footer()
        
        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(spice_model)
            logger.info(f"Electronic load SPICE model saved to {output_file}")
        
        return spice_model
    
    def _generate_load_spice_header(self) -> str:
        """Generate electronic load SPICE model header"""
        header = f"""* Electronic Load SPICE Model
* Topology: {self.specs.topology.value}
* Load Mode: {self.load_mode.value}
* Generated by MFC Power Models
* 
* Voltage Range: {self.specs.voltage_range[0]:.1f}V - {self.specs.voltage_range[1]:.1f}V
* Current Range: {self.specs.current_range[0]:.3f}A - {self.specs.current_range[1]:.1f}A
* Power Rating: {self.specs.power_rating:.1f}W
* Control Bandwidth: {self.specs.bandwidth/1000:.1f}kHz

"""
        return header
    
    def _generate_load_spice_parameters(self, operating_point: Dict[str, float]) -> str:
        """Generate electronic load SPICE parameter definitions"""
        vin = operating_point['input_voltage']
        iload = operating_point['load_current']
        pload = operating_point['load_power']
        
        params = f"""* Load Parameters
.param Vin={vin:.2f}
.param Iload={iload:.3f}
.param Pload={pload:.2f}
.param Rload={{Vin/Iload}}
.param Vmax={self.specs.voltage_range[1]:.1f}
.param Imax={self.specs.current_range[1]:.2f}
.param Pmax={self.specs.power_rating:.1f}

* Control Parameters
.param Kp={self.control_kp:.2f}
.param Ki={self.control_ki:.1f}
.param Bandwidth={self.specs.bandwidth:.0f}

"""
        return params
    
    def _generate_load_spice_circuit(self) -> str:
        """Generate electronic load SPICE circuit netlist"""
        if self.specs.topology == LoadTopology.LINEAR_MOSFET:
            return self._generate_linear_mosfet_load()
        elif self.specs.topology == LoadTopology.SWITCHING_MOSFET:
            return self._generate_switching_mosfet_load()
        elif self.specs.topology == LoadTopology.RESISTIVE_ARRAY:
            return self._generate_resistive_array_load()
        elif self.specs.topology == LoadTopology.ACTIVE_RECTIFIER:
            return self._generate_active_rectifier_load()
        else:
            return self._generate_linear_mosfet_load()  # Default
    
    def _generate_linear_mosfet_load(self) -> str:
        """Generate linear MOSFET electronic load circuit"""
        circuit = f"""* Linear MOSFET Electronic Load Circuit
* Input voltage source (DUT)
Vin input 0 DC {{Vin}}
Rin input n_input 0.01

* Load MOSFET (power device)
Mload n_input 0 gate 0 NMOS_LOAD
.model NMOS_LOAD NMOS(Kp=10 Vto=2 Lambda=0.01 Rd={self.mosfet_rds_on:.4f})

* Current sensing resistor
Rsense n_input load_pos {self.specs.current_range[1]/1000:.6f}

* Control amplifier and feedback
* Current feedback
Bisense vsense 0 V={{I(Rsense)*1000}}
Eisense isense 0 vsense 0 1

* Error amplifier (voltage controlled current source)
Gerror error_out 0 VCCS isense iref {{Kp}}

* Integrator for PI control
Cint error_out int_out 1u
Rint int_out 0 {{1/Ki}}

* Gate drive amplifier
Egate gate 0 VCVS error_out int_out 1

* Reference current source
Iref iref 0 DC {{Iload}}

* Temperature sensing (simplified)
Btemp vtemp 0 V={{25+I(Rsense)*I(Rsense)*{self.specs.thermal_resistance:.1f}}}

* Thermal protection comparator
Etp tp_out 0 value={{if(V(vtemp)>85,5,0)}}

* Protection shutdown switch
Sshutdown gate 0 tp_out 0 SWITCH
.model SWITCH SW(Vt=2.5 Vh=0.1)

"""
        return circuit
    
    def _generate_switching_mosfet_load(self) -> str:
        """Generate switching MOSFET electronic load circuit"""
        circuit = f"""* Switching MOSFET Electronic Load Circuit
* Input voltage source (DUT)
Vin input 0 DC {{Vin}}

* Input filter
Cin input n_input 100u
Rin n_input n_filt 0.01

* Main switching MOSFET
Mload n_filt n_sw gate_hs 0 NMOS_LOAD
.model NMOS_LOAD NMOS(Kp=50 Vto=2 Lambda=0.01)

* Synchronous rectifier MOSFET
Msync n_sw 0 gate_ls 0 NMOS_SYNC
.model NMOS_SYNC NMOS(Kp=100 Vto=2 Lambda=0.01)

* Output inductor and capacitor
Lout n_sw load_pos {self.inductor_value*1e6:.1f}u
Cout load_pos 0 {self.output_capacitance*1e6:.0f}u
Rload load_pos 0 {{Rload}}

* Current sensing
Rsense n_filt n_input 1m

* PWM Controller
Bpwm vpwm 0 V={{0.5*(1+sin(2*pi*{self.switching_frequency:.0f}*time))}}
Epwm_hs gate_hs 0 value={{if(V(vpwm)>V(vcontrol),5,0)}}
Epwm_ls gate_ls 0 value={{if(V(vpwm)<V(vcontrol),5,0)}}

* Control loop (simplified)
Bisense vsense 0 V={{I(Rsense)*1000}}
Eerror vcontrol 0 value={{2.5+Kp*(Iload-V(vsense))}}

* Output current measurement
Bout iout 0 V={{I(Lout)}}

"""
        return circuit
    
    def _generate_resistive_array_load(self) -> str:
        """Generate resistive array electronic load circuit"""
        circuit = """* Resistive Array Electronic Load Circuit
* Input voltage source (DUT)
Vin input 0 DC {Vin}

* Current sensing
Rsense input n_input 1m

* Resistor array (binary weighted)
"""
        
        # Generate binary-weighted resistor array
        for i in range(min(8, self.num_resistors)):  # Limit to 8 bits for SPICE
            resistance = self.base_resistance * (2 ** i)
            circuit += f"""
* Resistor {i+1}: {resistance:.2f}Ω
Sw{i+1} n_input n_r{i+1} vbit{i+1} 0 SWITCH
Rload{i+1} n_r{i+1} 0 {resistance:.2f}
Vbit{i+1} vbit{i+1} 0 PULSE(0 5 {i*0.1e-6:.1e} 1n 1n 0.5u 1u)
"""
        
        circuit += """
* Switch model
.model SWITCH SW(Vt=2.5 Vh=0.1 Ron=0.01 Roff=1Meg)

* Digital control logic (simplified)
Bcontrol vcontrol 0 V={{if(I(Rsense)<Iload,5,0)}}

* Current measurement
Bisense isense 0 V={{I(Rsense)}}

"""
        return circuit
    
    def _generate_active_rectifier_load(self) -> str:
        """Generate active rectifier electronic load circuit"""
        circuit = f"""* Active Rectifier Electronic Load Circuit
* AC Input voltage source (for AC loads)
Vin input 0 SIN(0 {{Vin*sqrt(2)}} 60)

* Bridge rectifier with synchronous MOSFETs
Mp1 input n_dc gate_p1 input PMOS_RECT
Mp2 0 n_dc gate_p2 0 PMOS_RECT
Mn1 n_dc input gate_n1 0 NMOS_RECT  
Mn2 n_dc 0 gate_n2 0 NMOS_RECT

.model PMOS_RECT PMOS(Kp=20 Vto=-2)
.model NMOS_RECT NMOS(Kp=20 Vto=2)

* DC link capacitor
Cdc n_dc 0 1000u

* Load regulation circuit
Mload n_dc load_pos gate_load 0 NMOS_LOAD
.model NMOS_LOAD NMOS(Kp=50 Vto=2)

* Output inductor
Lload load_pos load_neg {self.output_inductance*1e6:.0f}u
Rload load_neg 0 {{Rload}}

* Synchronous rectification control (simplified)
Egate_p1 gate_p1 input value={{if(V(input)>0,5,0)}}
Egate_p2 gate_p2 0 value={{if(V(input)<0,5,0)}}
Egate_n1 gate_n1 0 value={{if(V(input)<0,5,0)}}
Egate_n2 gate_n2 0 value={{if(V(input)>0,5,0)}}

* Load current control
Bisense vsense 0 V={{I(Lload)}}
Egate_load gate_load 0 value={{2.5+Kp*(Iload-V(vsense))}}

"""
        return circuit
    
    def _generate_load_spice_analysis(self) -> str:
        """Generate electronic load SPICE analysis commands"""
        if self.specs.topology == LoadTopology.SWITCHING_MOSFET:
            # Switching topology needs transient analysis
            period = 1.0 / self.switching_frequency
            sim_time = period * 10
            analysis = f"""* Analysis Commands for Switching Load
* Transient analysis
.tran 1n {sim_time*1e6:.0f}u

* Operating point
.op

* AC analysis for control loop stability
.ac dec 100 1 {self.specs.bandwidth*10:.0f}
"""
        elif self.specs.topology == LoadTopology.ACTIVE_RECTIFIER:
            # AC rectifier needs AC analysis
            analysis = """* Analysis Commands for Active Rectifier Load
* AC analysis
.tran 1m 50m

* Operating point
.op

* Harmonic analysis
.four 60 V(input) I(Vin)
"""
        else:
            # Linear topologies
            analysis = """* Analysis Commands for Linear Load
* Operating point analysis
.op

* DC sweep of input voltage
.dc Vin 0 Vmax 0.1

* AC analysis for stability
.ac dec 100 1 1Meg
"""
        
        # Common measurements
        analysis += """
* Measurements
.measure tran avg_current AVG I(Rsense) FROM 1m TO 10m
.measure tran avg_power AVG 'V(input)*I(Vin)' FROM 1m TO 10m
.measure dc load_regulation DERIV 'I(Rsense)' AT Vin={Vin}
.measure ac phase_margin FIND vp(error_out) WHEN vdb(error_out)=0

* Output commands
.print tran V(input) I(Rsense) V(gate)
.plot dc I(Rsense) V(gate)

"""
        return analysis
    
    def _generate_load_spice_footer(self) -> str:
        """Generate electronic load SPICE model footer"""
        footer = """* End of electronic load model
.end
"""
        return footer
    
    def export_load_testbench(self, output_dir: str, test_conditions: List[Dict[str, float]]) -> List[str]:
        """
        Export SPICE testbench with multiple load conditions
        
        Args:
            output_dir: Directory to save testbench files
            test_conditions: List of test condition dictionaries
            
        Returns:
            List of generated file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        for i, condition in enumerate(test_conditions):
            filename = f"electronic_load_test_{i+1:02d}.cir"
            filepath = output_path / filename
            
            # Generate SPICE model for this condition
            spice_model = self.generate_spice_model(operating_point=condition)
            
            # Add testbench-specific header
            testbench_header = f"""* Electronic Load Testbench #{i+1}
* Test Condition: Vin={condition.get('input_voltage', 5.0):.1f}V,
*                 Iload={condition.get('load_current', 1.0):.1f}A,
*                 Mode={condition.get('mode', 'constant_current')}

"""
            
            # Write complete testbench
            with open(filepath, 'w') as f:
                f.write(testbench_header)
                f.write(spice_model)
            
            generated_files.append(str(filepath))
            logger.info(f"Generated electronic load testbench: {filepath}")
        
        # Generate batch script
        batch_script = self._generate_load_batch_script(output_path, generated_files)
        script_path = output_path / "run_load_tests.sh"
        with open(script_path, 'w') as f:
            f.write(batch_script)
        os.chmod(script_path, 0o755)
        
        generated_files.append(str(script_path))
        
        return generated_files
    
    def _generate_load_batch_script(self, output_dir: Path, spice_files: List[str]) -> str:
        """Generate batch script for electronic load tests"""
        script = """#!/bin/bash
# Batch script to run electronic load SPICE simulations

echo "Running electronic load SPICE simulations..."

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
echo "All load simulations completed."
echo "Check individual .log files for results."
"""
        
        return script


def create_standard_electronic_loads() -> Dict[str, ElectronicLoad]:
    """Create standard electronic load configurations"""
    
    # Low-power linear load
    linear_load_specs = ElectronicLoadSpecs(
        topology=LoadTopology.LINEAR_MOSFET,
        voltage_range=(0.1, 30.0),  # V
        current_range=(0.001, 5.0),  # A
        power_rating=50.0,  # W
        accuracy_current=0.1,  # %
        accuracy_voltage=0.05,  # %
        accuracy_power=0.2,  # %
        resolution_current=0.0001,  # A
        resolution_voltage=0.001,  # V
        bandwidth=10000.0,  # Hz
        slew_rate=1.0,  # A/s
        thermal_resistance=2.5,  # °C/W
        cooling_method="air",
        cost=1500.0,  # USD
        weight=2.0,  # kg
        efficiency=60.0  # %
    )
    
    # High-power switching load
    switching_load_specs = ElectronicLoadSpecs(
        topology=LoadTopology.SWITCHING_MOSFET,
        voltage_range=(3.0, 80.0),  # V
        current_range=(0.01, 50.0),  # A
        power_rating=1000.0,  # W
        accuracy_current=0.05,  # %
        accuracy_voltage=0.02,  # %
        accuracy_power=0.1,  # %
        resolution_current=0.001,  # A
        resolution_voltage=0.01,  # V
        bandwidth=50000.0,  # Hz
        slew_rate=10.0,  # A/s
        thermal_resistance=0.5,  # °C/W
        cooling_method="liquid",
        cost=5000.0,  # USD
        weight=8.0,  # kg
        efficiency=90.0  # %
    )
    
    # Resistive array load (simple, robust)
    resistive_load_specs = ElectronicLoadSpecs(
        topology=LoadTopology.RESISTIVE_ARRAY,
        voltage_range=(1.0, 50.0),  # V
        current_range=(0.1, 20.0),  # A
        power_rating=250.0,  # W
        accuracy_current=1.0,  # %
        accuracy_voltage=0.1,  # %
        accuracy_power=1.5,  # %
        resolution_current=0.1,  # A
        resolution_voltage=0.1,  # V
        bandwidth=1000.0,  # Hz
        slew_rate=5.0,  # A/s
        thermal_resistance=1.0,  # °C/W
        cooling_method="air",
        cost=800.0,  # USD
        weight=5.0,  # kg
        efficiency=95.0  # %
    )
    
    # Micro load for low-power testing
    micro_load_specs = ElectronicLoadSpecs(
        topology=LoadTopology.LINEAR_MOSFET,
        voltage_range=(0.01, 5.0),  # V
        current_range=(0.0001, 0.1),  # A
        power_rating=0.5,  # W
        accuracy_current=0.5,  # %
        accuracy_voltage=0.1,  # %
        accuracy_power=1.0,  # %
        resolution_current=0.00001,  # A
        resolution_voltage=0.0001,  # V
        bandwidth=5000.0,  # Hz
        slew_rate=0.01,  # A/s
        thermal_resistance=50.0,  # °C/W
        cooling_method="passive",
        cost=500.0,  # USD
        weight=0.5,  # kg
        efficiency=40.0  # %
    )
    
    loads = {
        'linear_load': ElectronicLoad(linear_load_specs),
        'switching_load': ElectronicLoad(switching_load_specs),
        'resistive_load': ElectronicLoad(resistive_load_specs),
        'micro_load': ElectronicLoad(micro_load_specs)
    }
    
    return loads


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create standard electronic loads
    loads = create_standard_electronic_loads()
    
    # Test switching load
    switching_load = loads['switching_load']
    switching_load.set_mode(LoadMode.CONSTANT_CURRENT, 10.0)  # 10A constant current
    
    print("Electronic Load Testing:")
    print(f"Mode: {switching_load.load_mode.value}")
    print(f"Setpoint: {switching_load.setpoint_current:.1f}A")
    
    # Simulate load testing of a power source
    voltage_profile = [5.0, 10.0, 15.0, 20.0, 15.0, 10.0, 5.0]  # Voltage sweep
    measurements = []
    
    for i, voltage in enumerate(voltage_profile):
        measurement = switching_load.process(input_voltage=voltage, dt=0.001, time=i*0.1)
        measurements.append(measurement)
        
        print(f"Test {i+1}: Vin={voltage:.1f}V → "
              f"I={measurement.current:.2f}A, P={measurement.power:.1f}W, "
              f"Eff={measurement.efficiency:.1f}%, T={measurement.temperature:.1f}°C")
    
    # Test different load modes
    print("\nTesting different load modes at 12V:")
    test_voltage = 12.0
    
    # Constant current mode
    switching_load.set_mode(LoadMode.CONSTANT_CURRENT, 5.0)
    cc_measurement = switching_load.process(test_voltage, time=1.0)
    print(f"CC Mode (5A): I={cc_measurement.current:.2f}A, P={cc_measurement.power:.1f}W")
    
    # Constant power mode
    switching_load.set_mode(LoadMode.CONSTANT_POWER, 60.0)
    cp_measurement = switching_load.process(test_voltage, time=2.0)
    print(f"CP Mode (60W): I={cp_measurement.current:.2f}A, P={cp_measurement.power:.1f}W")
    
    # Constant resistance mode
    switching_load.set_mode(LoadMode.CONSTANT_RESISTANCE, 2.4)
    cr_measurement = switching_load.process(test_voltage, time=3.0)
    print(f"CR Mode (2.4Ω): I={cr_measurement.current:.2f}A, R={cr_measurement.resistance:.1f}Ω")
    
    # Cost analysis
    cost_analysis = switching_load.get_cost_analysis()
    print("\nCost Analysis:")
    print(f"Initial cost: ${cost_analysis['initial_cost']:.0f}")
    print(f"Operating cost per hour: ${cost_analysis['total_cost_per_hour']:.4f}")
    print(f"Lifetime cost: ${cost_analysis['lifetime_cost']:.0f}")
    
    # Generate SPICE models
    print("\nGenerating SPICE models...")
    
    # Test conditions for SPICE generation
    test_conditions = [
        {'input_voltage': 5.0, 'load_current': 1.0, 'load_power': 5.0, 'mode': 'constant_current'},
        {'input_voltage': 12.0, 'load_current': 5.0, 'load_power': 60.0, 'mode': 'constant_power'},
        {'input_voltage': 24.0, 'load_current': 2.0, 'load_power': 48.0, 'mode': 'constant_resistance'},
    ]
    
    generated_files = switching_load.export_load_testbench("spice_load_testbench", test_conditions)
    print(f"Generated {len(generated_files)} SPICE testbench files:")
    for file_path in generated_files:
        print(f"  - {file_path}")
    
    # Test all load types
    print("\nComparison of all load types:")
    for name, load in loads.items():
        specs = load.specs
        cost = load.get_cost_analysis()
        
        print(f"\n{name.upper()}:")
        print(f"  Topology: {specs.topology.value}")
        print(f"  Power rating: {specs.power_rating:.0f}W")
        print(f"  Current range: {specs.current_range[0]:.3f}A - {specs.current_range[1]:.1f}A")
        print(f"  Accuracy: {specs.accuracy_current:.2f}% (current)")
        print(f"  Cost: ${specs.cost:.0f}")
        print(f"  Efficiency: {specs.efficiency:.0f}%")