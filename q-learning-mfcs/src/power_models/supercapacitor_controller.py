"""
Supercapacitor Charger and Discharge Controller Models for MFC Systems

This module implements dedicated controllers for supercapacitor charging and discharging
with various topologies including constant current, constant voltage, and bidirectional
controllers with comprehensive SPICE model generation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ControllerTopology(Enum):
    """Supercapacitor controller topologies"""
    BUCK_BOOST = "buck_boost"  # Bidirectional buck-boost
    FLYBACK = "flyback"  # Isolated flyback
    SEPIC = "sepic"  # Single-ended primary inductor converter
    CUK = "cuk"  # Cuk converter
    FORWARD = "forward"  # Forward converter
    HALF_BRIDGE = "half_bridge"  # Half-bridge converter
    FULL_BRIDGE = "full_bridge"  # Full-bridge converter


class ControlMode(Enum):
    """Control modes for supercapacitor charging/discharging"""
    CC = "constant_current"  # Constant current
    CV = "constant_voltage"  # Constant voltage
    CC_CV = "cc_cv"  # Constant current then constant voltage
    CP = "constant_power"  # Constant power
    MPPT = "mppt"  # Maximum power point tracking
    LOAD_FOLLOWING = "load_following"  # Load following


@dataclass
class SupercapacitorControllerSpecs:
    """Specifications for supercapacitor controllers"""
    topology: ControllerTopology
    control_mode: ControlMode
    input_voltage_range: Tuple[float, float]  # V (min, max)
    output_voltage_range: Tuple[float, float]  # V (min, max)
    max_current: float  # A
    max_power: float  # W
    switching_frequency: float  # Hz
    efficiency: float  # 0-1
    ripple_current: float  # % of rated current
    ripple_voltage: float  # % of rated voltage
    response_time: float  # seconds
    power_consumption: float  # W (control circuitry)
    cost: float  # USD
    mass: float  # kg
    volume: float  # L


@dataclass
class ControllerMeasurement:
    """Single controller measurement"""
    timestamp: float
    input_voltage: float  # V
    output_voltage: float  # V
    input_current: float  # A
    output_current: float  # A
    input_power: float  # W
    output_power: float  # W
    efficiency: float  # 0-1
    switching_frequency: float  # Hz
    duty_cycle: float  # 0-1
    temperature: float  # °C
    control_mode: ControlMode
    fault_status: str  # 'ok', 'warning', 'fault'


class SupercapacitorController:
    """Comprehensive supercapacitor charge/discharge controller"""
    
    def __init__(self, specs: SupercapacitorControllerSpecs):
        self.specs = specs
        self.input_voltage = 0.0
        self.output_voltage = 0.0
        self.input_current = 0.0
        self.output_current = 0.0
        self.temperature = 25.0
        self.duty_cycle = 0.5
        self.operating_hours = 0.0
        
        # Control parameters
        self.setpoint_voltage = 0.0
        self.setpoint_current = 0.0
        self.setpoint_power = 0.0
        
        # PI controller parameters
        self.kp_voltage = 0.1
        self.ki_voltage = 10.0
        self.kp_current = 0.05
        self.ki_current = 5.0
        self.integrator_voltage = 0.0
        self.integrator_current = 0.0
        
        # Initialize topology-specific parameters
        self._initialize_topology_parameters()
        
        # Fault detection
        self.fault_conditions = {
            'overvoltage': False,
            'overcurrent': False,
            'overtemperature': False,
            'undervoltage': False
        }
        
    def _initialize_topology_parameters(self):
        """Initialize topology-specific parameters"""
        if self.specs.topology == ControllerTopology.BUCK_BOOST:
            # Buck-boost converter parameters
            self.inductor_value = 100e-6  # H
            self.capacitor_input = 100e-6  # F
            self.capacitor_output = 220e-6  # F
            self.mosfet_rdson = 0.01  # ohms
            self.diode_vf = 0.7  # V
            self.turns_ratio = 1.0
            
        elif self.specs.topology == ControllerTopology.FLYBACK:
            # Flyback converter parameters
            self.primary_inductance = 200e-6  # H
            self.secondary_inductance = 50e-6  # H
            self.turns_ratio = 2.0
            self.magnetizing_inductance = 1e-3  # H
            self.leakage_inductance = 5e-6  # H
            self.coupling_coefficient = 0.98
            
        elif self.specs.topology == ControllerTopology.SEPIC:
            # SEPIC converter parameters
            self.inductor1 = 100e-6  # H
            self.inductor2 = 100e-6  # H
            self.coupling_capacitor = 10e-6  # F
            self.output_capacitor = 220e-6  # F
            self.coupling_coefficient = 0.95
            
        elif self.specs.topology == ControllerTopology.CUK:
            # Cuk converter parameters
            self.inductor1 = 150e-6  # H
            self.inductor2 = 150e-6  # H
            self.coupling_capacitor = 47e-6  # F
            self.input_capacitor = 100e-6  # F
            self.output_capacitor = 220e-6  # F
            
        elif self.specs.topology == ControllerTopology.FORWARD:
            # Forward converter parameters
            self.transformer_ratio = 1.5
            self.primary_inductance = 300e-6  # H
            self.output_inductor = 50e-6  # H
            self.freewheeling_diode_vf = 0.5  # V
            self.reset_winding_ratio = 1.0
            
        elif self.specs.topology == ControllerTopology.HALF_BRIDGE:
            # Half-bridge converter parameters
            self.transformer_ratio = 2.0
            self.primary_inductance = 500e-6  # H
            self.output_inductor = 30e-6  # H
            self.blocking_capacitor = 10e-6  # F
            self.deadtime = 100e-9  # s
            
        elif self.specs.topology == ControllerTopology.FULL_BRIDGE:
            # Full-bridge converter parameters
            self.transformer_ratio = 1.0
            self.primary_inductance = 400e-6  # H
            self.output_inductor = 25e-6  # H
            self.phase_shift_angle = 45  # degrees
            self.deadtime = 200e-9  # s
    
    def control_step(self, input_voltage: float, load_current: float, 
                    dt: float, temperature: float = 25.0) -> ControllerMeasurement:
        """
        Execute one control step
        
        Args:
            input_voltage: Input voltage (V)
            load_current: Load current demand (A, positive for discharge)
            dt: Time step (seconds)
            temperature: Operating temperature (°C)
            
        Returns:
            ControllerMeasurement object
        """
        self.input_voltage = input_voltage
        self.temperature = temperature
        self.operating_hours += dt / 3600.0
        
        # Fault detection
        self._check_faults()
        
        # Control algorithm based on mode
        if self.specs.control_mode == ControlMode.CC:
            self._constant_current_control(load_current, dt)
        elif self.specs.control_mode == ControlMode.CV:
            self._constant_voltage_control(dt)
        elif self.specs.control_mode == ControlMode.CC_CV:
            self._cc_cv_control(load_current, dt)
        elif self.specs.control_mode == ControlMode.CP:
            self._constant_power_control(dt)
        elif self.specs.control_mode == ControlMode.MPPT:
            self._mppt_control(dt)
        elif self.specs.control_mode == ControlMode.LOAD_FOLLOWING:
            self._load_following_control(load_current, dt)
        
        # Calculate output based on topology
        self._calculate_converter_output()
        
        # Calculate efficiency
        efficiency = self._calculate_efficiency()
        
        # Determine fault status
        fault_status = self._get_fault_status()
        
        return ControllerMeasurement(
            timestamp=self.operating_hours * 3600,
            input_voltage=self.input_voltage,
            output_voltage=self.output_voltage,
            input_current=self.input_current,
            output_current=self.output_current,
            input_power=self.input_voltage * self.input_current,
            output_power=self.output_voltage * self.output_current,
            efficiency=efficiency,
            switching_frequency=self.specs.switching_frequency,
            duty_cycle=self.duty_cycle,
            temperature=temperature,
            control_mode=self.specs.control_mode,
            fault_status=fault_status
        )
    
    def _constant_current_control(self, target_current: float, dt: float):
        """Constant current control algorithm"""
        self.setpoint_current = np.clip(target_current, -self.specs.max_current, self.specs.max_current)
        
        # PI controller for current
        current_error = self.setpoint_current - self.output_current
        self.integrator_current += current_error * dt
        
        # Anti-windup
        self.integrator_current = np.clip(self.integrator_current, -10.0, 10.0)
        
        # PI output
        control_output = (self.kp_current * current_error + 
                         self.ki_current * self.integrator_current)
        
        # Convert to duty cycle
        self.duty_cycle = np.clip(0.5 + control_output * 0.1, 0.05, 0.95)
    
    def _constant_voltage_control(self, dt: float):
        """Constant voltage control algorithm"""
        # PI controller for voltage
        voltage_error = self.setpoint_voltage - self.output_voltage
        self.integrator_voltage += voltage_error * dt
        
        # Anti-windup
        self.integrator_voltage = np.clip(self.integrator_voltage, -10.0, 10.0)
        
        # PI output
        control_output = (self.kp_voltage * voltage_error + 
                         self.ki_voltage * self.integrator_voltage)
        
        # Convert to duty cycle
        self.duty_cycle = np.clip(0.5 + control_output * 0.01, 0.05, 0.95)
    
    def _cc_cv_control(self, target_current: float, dt: float):
        """Constant current / constant voltage control"""
        # Start with current control, switch to voltage control when voltage limit reached
        if self.output_voltage < self.setpoint_voltage * 0.95:
            self._constant_current_control(target_current, dt)
        else:
            self._constant_voltage_control(dt)
    
    def _constant_power_control(self, dt: float):
        """Constant power control algorithm"""
        if self.output_voltage > 0:
            target_current = self.setpoint_power / self.output_voltage
            self._constant_current_control(target_current, dt)
        else:
            self.duty_cycle = 0.1
    
    def _mppt_control(self, dt: float):
        """Maximum Power Point Tracking algorithm (Perturb and Observe)"""
        # Simple P&O MPPT algorithm
        power = self.input_voltage * self.input_current
        
        if not hasattr(self, 'prev_power'):
            self.prev_power = power
            self.prev_duty = self.duty_cycle
            self.mppt_step = 0.01
        
        # Perturb and observe
        if power > self.prev_power:
            # Moving in right direction
            if self.duty_cycle > self.prev_duty:
                self.duty_cycle += self.mppt_step
            else:
                self.duty_cycle -= self.mppt_step
        else:
            # Moving in wrong direction
            if self.duty_cycle > self.prev_duty:
                self.duty_cycle -= self.mppt_step
            else:
                self.duty_cycle += self.mppt_step
        
        # Limit duty cycle
        self.duty_cycle = np.clip(self.duty_cycle, 0.05, 0.95)
        
        # Store previous values
        self.prev_power = power
        self.prev_duty = self.duty_cycle
    
    def _load_following_control(self, load_current: float, dt: float):
        """Load following control algorithm"""
        # Adjust output to match load demand
        self._constant_current_control(load_current, dt)
    
    def _calculate_converter_output(self):
        """Calculate converter output based on topology and duty cycle"""
        if self.specs.topology == ControllerTopology.BUCK_BOOST:
            # Buck-boost: Vo = Vin * D / (1-D)
            if self.duty_cycle < 0.95:
                voltage_ratio = self.duty_cycle / (1.0 - self.duty_cycle)
                self.output_voltage = self.input_voltage * voltage_ratio
                # Current is inversely related for ideal converter
                if self.output_voltage > 0:
                    self.input_current = self.output_current / voltage_ratio
            else:
                self.output_voltage = self.input_voltage * 10  # High gain at high duty cycle
                self.input_current = self.output_current * 10
                
        elif self.specs.topology == ControllerTopology.FLYBACK:
            # Flyback: Vo = Vin * n * D / (1-D)
            if self.duty_cycle < 0.95:
                voltage_ratio = self.turns_ratio * self.duty_cycle / (1.0 - self.duty_cycle)
                self.output_voltage = self.input_voltage * voltage_ratio
                if self.output_voltage > 0:
                    self.input_current = self.output_current * voltage_ratio / self.turns_ratio
            
        elif self.specs.topology == ControllerTopology.SEPIC:
            # SEPIC: Vo = Vin * D / (1-D)
            if self.duty_cycle < 0.95:
                voltage_ratio = self.duty_cycle / (1.0 - self.duty_cycle)
                self.output_voltage = self.input_voltage * voltage_ratio
                if self.output_voltage > 0:
                    self.input_current = self.output_current * voltage_ratio
                    
        elif self.specs.topology == ControllerTopology.CUK:
            # Cuk: Vo = -Vin * D / (1-D) (inverting)
            if self.duty_cycle < 0.95:
                voltage_ratio = self.duty_cycle / (1.0 - self.duty_cycle)
                self.output_voltage = self.input_voltage * voltage_ratio  # Magnitude
                if self.output_voltage > 0:
                    self.input_current = self.output_current * voltage_ratio
                    
        elif self.specs.topology == ControllerTopology.FORWARD:
            # Forward: Vo = Vin * n * D
            voltage_ratio = self.transformer_ratio * self.duty_cycle
            self.output_voltage = self.input_voltage * voltage_ratio
            if self.output_voltage > 0:
                self.input_current = self.output_current * voltage_ratio / self.transformer_ratio
                
        elif self.specs.topology == ControllerTopology.HALF_BRIDGE:
            # Half-bridge: Vo = Vin * n * D / 2
            voltage_ratio = self.transformer_ratio * self.duty_cycle / 2.0
            self.output_voltage = self.input_voltage * voltage_ratio
            if self.output_voltage > 0:
                self.input_current = self.output_current * voltage_ratio / self.transformer_ratio
                
        elif self.specs.topology == ControllerTopology.FULL_BRIDGE:
            # Full-bridge: Vo = Vin * n * D
            voltage_ratio = self.transformer_ratio * self.duty_cycle
            self.output_voltage = self.input_voltage * voltage_ratio
            if self.output_voltage > 0:
                self.input_current = self.output_current * voltage_ratio / self.transformer_ratio
        
        # Apply limits
        self.output_voltage = np.clip(self.output_voltage, 
                                    self.specs.output_voltage_range[0], 
                                    self.specs.output_voltage_range[1])
        self.input_current = np.clip(self.input_current, 0, self.specs.max_current)
        self.output_current = np.clip(self.output_current, 0, self.specs.max_current)
    
    def _calculate_efficiency(self) -> float:
        """Calculate converter efficiency"""
        # Base efficiency from specs
        base_efficiency = self.specs.efficiency
        
        # Efficiency depends on load and operating conditions
        if self.output_current > 0:
            load_factor = self.output_current / self.specs.max_current
            # Efficiency curve: peak at 70% load, decreases at light and heavy loads
            efficiency_curve = 1.0 - 0.1 * (load_factor - 0.7)**2
            efficiency = base_efficiency * efficiency_curve
        else:
            efficiency = 0.5  # Light load efficiency
        
        # Temperature derating
        temp_factor = 1.0 - 0.002 * abs(self.temperature - 25.0)
        efficiency *= temp_factor
        
        # Input voltage derating
        vin_nominal = (self.specs.input_voltage_range[0] + self.specs.input_voltage_range[1]) / 2
        vin_factor = 1.0 - 0.001 * abs(self.input_voltage - vin_nominal)
        efficiency *= vin_factor
        
        return np.clip(efficiency, 0.1, 0.98)
    
    def _check_faults(self):
        """Check for fault conditions"""
        # Overvoltage
        self.fault_conditions['overvoltage'] = (
            self.input_voltage > self.specs.input_voltage_range[1] * 1.1 or
            self.output_voltage > self.specs.output_voltage_range[1] * 1.1
        )
        
        # Undervoltage
        self.fault_conditions['undervoltage'] = (
            self.input_voltage < self.specs.input_voltage_range[0] * 0.9
        )
        
        # Overcurrent
        self.fault_conditions['overcurrent'] = (
            self.input_current > self.specs.max_current * 1.2 or
            self.output_current > self.specs.max_current * 1.2
        )
        
        # Overtemperature
        self.fault_conditions['overtemperature'] = self.temperature > 85.0
    
    def _get_fault_status(self) -> str:
        """Get current fault status"""
        if any(self.fault_conditions.values()):
            return "fault"
        elif (self.temperature > 70.0 or 
              self.input_current > self.specs.max_current * 0.9 or
              self.output_current > self.specs.max_current * 0.9):
            return "warning"
        else:
            return "ok"
    
    def set_control_parameters(self, voltage_setpoint: float = None, 
                             current_setpoint: float = None,
                             power_setpoint: float = None):
        """Set control setpoints"""
        if voltage_setpoint is not None:
            self.setpoint_voltage = voltage_setpoint
        if current_setpoint is not None:
            self.setpoint_current = current_setpoint
        if power_setpoint is not None:
            self.setpoint_power = power_setpoint
    
    def generate_spice_model(self, output_file: Optional[str] = None,
                           operating_point: Optional[Dict[str, float]] = None) -> str:
        """
        Generate SPICE model for the supercapacitor controller
        
        Args:
            output_file: Optional file to write the SPICE model
            operating_point: Optional operating point for linearization
            
        Returns:
            SPICE netlist as string
        """
        if operating_point is None:
            operating_point = {
                'input_voltage': self.input_voltage,
                'output_voltage': self.output_voltage,
                'duty_cycle': self.duty_cycle,
                'temperature': self.temperature
            }
        
        # Generate topology-specific SPICE model
        if self.specs.topology == ControllerTopology.BUCK_BOOST:
            spice_model = self._generate_buck_boost_spice(operating_point)
        elif self.specs.topology == ControllerTopology.FLYBACK:
            spice_model = self._generate_flyback_spice(operating_point)
        elif self.specs.topology == ControllerTopology.SEPIC:
            spice_model = self._generate_sepic_spice(operating_point)
        elif self.specs.topology == ControllerTopology.CUK:
            spice_model = self._generate_cuk_spice(operating_point)
        elif self.specs.topology == ControllerTopology.FORWARD:
            spice_model = self._generate_forward_spice(operating_point)
        elif self.specs.topology == ControllerTopology.HALF_BRIDGE:
            spice_model = self._generate_half_bridge_spice(operating_point)
        elif self.specs.topology == ControllerTopology.FULL_BRIDGE:
            spice_model = self._generate_full_bridge_spice(operating_point)
        else:
            spice_model = self._generate_generic_spice(operating_point)
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(spice_model)
                logger.info(f"SPICE model written to {output_file}")
            except Exception as e:
                logger.error(f"Failed to write SPICE model to {output_file}: {e}")
        
        return spice_model
    
    def _generate_buck_boost_spice(self, op_point: Dict[str, float]) -> str:
        """Generate SPICE model for buck-boost topology"""
        return f"""* Buck-Boost Supercapacitor Controller
* Operating point: Vin={op_point['input_voltage']:.2f}V, Vout={op_point['output_voltage']:.2f}V, D={op_point['duty_cycle']:.3f}

.subckt buck_boost_controller vin vout gnd
* Main inductor
L_main vin n1 {self.inductor_value:.6e}

* Main switch (MOSFET)
M_main n1 n_gate gnd gnd NMOS_POWER
.model NMOS_POWER NMOS(VTO=2.0 KP=10 RDS={self.mosfet_rdson:.6f})

* Freewheeling diode
D_free gnd n1 POWER_DIODE
.model POWER_DIODE D(VF={self.diode_vf:.2f} RS=0.01)

* Output capacitor
C_out n1 vout {self.capacitor_output:.6e}
* Initial condition
.IC V(vout)={op_point['output_voltage']:.3f}

* Input capacitor
C_in vin gnd {self.capacitor_input:.6e}

* PWM control
V_pwm n_gate gnd PULSE(0 12 0 10n 10n {1/self.specs.switching_frequency*op_point['duty_cycle']:.9f} {1/self.specs.switching_frequency:.9f})

* ESR models
R_inductor_esr n1 n1_esr 0.01
R_cap_out_esr vout vout_esr 0.005
R_cap_in_esr vin vin_esr 0.002

* Control circuit power consumption
I_control vin gnd DC {self.specs.power_consumption / op_point['input_voltage']:.6f}

.ends buck_boost_controller

* Testbench
.include "buck_boost_controller.cir"
V_input vin_tb 0 DC {op_point['input_voltage']:.2f}
X_controller vin_tb vout_tb 0 buck_boost_controller
R_load vout_tb 0 10.0

* Analysis
.tran 0.1u 1m 0 0.1u
.probe V(vin_tb) V(vout_tb) I(X_controller.L_main)

* AC analysis for stability
.ac dec 100 10 100k
.probe V(vout_tb) I(X_controller.L_main)

* Efficiency calculation
.measure TRAN pin AVG P(V_input) FROM 100u TO 1m
.measure TRAN pout AVG {{V(vout_tb)*I(R_load)}} FROM 100u TO 1m
.measure TRAN efficiency PARAM {{pout/pin*100}}

.end
"""
    
    def _generate_flyback_spice(self, op_point: Dict[str, float]) -> str:
        """Generate SPICE model for flyback topology"""
        return f"""* Flyback Supercapacitor Controller
* Operating point: Vin={op_point['input_voltage']:.2f}V, Vout={op_point['output_voltage']:.2f}V, D={op_point['duty_cycle']:.3f}

.subckt flyback_controller vin vout gnd
* Primary inductor (magnetizing)
L_primary vin n_prim {self.primary_inductance:.6e}

* Secondary inductor (reflected)
L_secondary n_sec vout {self.secondary_inductance:.6e}

* Coupling between primary and secondary
K_couple L_primary L_secondary {self.coupling_coefficient:.3f}

* Primary switch
M_primary n_prim n_gate gnd gnd NMOS_POWER
.model NMOS_POWER NMOS(VTO=2.0 KP=10 RDS={self.mosfet_rdson:.6f})

* Secondary rectifier diode
D_rectifier n_sec vout POWER_DIODE
.model POWER_DIODE D(VF={self.diode_vf:.2f} RS=0.01)

* Output capacitor
C_output vout gnd {self.capacitor_output:.6e}
* Initial condition
.IC V(vout)={op_point['output_voltage']:.3f}

* Input capacitor
C_input vin gnd {self.capacitor_input:.6e}

* PWM control
V_pwm n_gate gnd PULSE(0 12 0 10n 10n {1/self.specs.switching_frequency*op_point['duty_cycle']:.9f} {1/self.specs.switching_frequency:.9f})

* Leakage inductance
L_leakage n_prim n_prim_leak {self.leakage_inductance:.6e}

* Transformer model
E_transformer n_sec 0 n_prim_leak 0 {self.turns_ratio:.2f}

.ends flyback_controller

* Testbench for flyback controller
.include "flyback_controller.cir"
V_input vin_tb 0 DC {op_point['input_voltage']:.2f}
X_controller vin_tb vout_tb 0 flyback_controller
R_load vout_tb 0 5.0

* Analysis
.tran 0.1u 1m 0 0.1u
.probe V(vin_tb) V(vout_tb) I(X_controller.L_primary) I(X_controller.D_rectifier)

.end
"""
    
    def _generate_sepic_spice(self, op_point: Dict[str, float]) -> str:
        """Generate SPICE model for SEPIC topology"""
        return f"""* SEPIC Supercapacitor Controller
* Operating point: Vin={op_point['input_voltage']:.2f}V, Vout={op_point['output_voltage']:.2f}V, D={op_point['duty_cycle']:.3f}

.subckt sepic_controller vin vout gnd
* Input inductor L1
L1 vin n1 {self.inductor1:.6e}

* Output inductor L2  
L2 n2 vout {self.inductor2:.6e}

* Coupling between inductors
K_couple L1 L2 {self.coupling_coefficient:.3f}

* Main switch
M_main n1 n_gate n2 n2 NMOS_POWER
.model NMOS_POWER NMOS(VTO=2.0 KP=10 RDS={self.mosfet_rdson:.6f})

* Output diode
D_output n2 vout POWER_DIODE
.model POWER_DIODE D(VF={self.diode_vf:.2f} RS=0.01)

* Coupling capacitor
C_coupling n1 n2 {self.coupling_capacitor:.6e}

* Output capacitor
C_output vout gnd {self.output_capacitor:.6e}
* Initial condition
.IC V(vout)={op_point['output_voltage']:.3f}

* Input capacitor
C_input vin gnd {self.capacitor_input:.6e}

* PWM control
V_pwm n_gate gnd PULSE(0 12 0 10n 10n {1/self.specs.switching_frequency*op_point['duty_cycle']:.9f} {1/self.specs.switching_frequency:.9f})

.ends sepic_controller

* Testbench
.include "sepic_controller.cir"
V_input vin_tb 0 DC {op_point['input_voltage']:.2f}
X_controller vin_tb vout_tb 0 sepic_controller
R_load vout_tb 0 8.0

* Analysis
.tran 0.1u 1m 0 0.1u
.probe V(vin_tb) V(vout_tb) I(X_controller.L1) I(X_controller.L2)

.end
"""
    
    def _generate_cuk_spice(self, op_point: Dict[str, float]) -> str:
        """Generate SPICE model for Cuk topology"""
        return f"""* Cuk Supercapacitor Controller
* Operating point: Vin={op_point['input_voltage']:.2f}V, Vout={op_point['output_voltage']:.2f}V, D={op_point['duty_cycle']:.3f}

.subckt cuk_controller vin vout gnd
* Input inductor L1
L1 vin n1 {self.inductor1:.6e}

* Output inductor L2
L2 n2 vout {self.inductor2:.6e}

* Main switch
M_main n1 n_gate gnd gnd NMOS_POWER
.model NMOS_POWER NMOS(VTO=2.0 KP=10 RDS={self.mosfet_rdson:.6f})

* Output diode
D_output gnd n2 POWER_DIODE
.model POWER_DIODE D(VF={self.diode_vf:.2f} RS=0.01)

* Coupling capacitor (energy transfer element)
C_coupling n1 n2 {self.coupling_capacitor:.6e}

* Input capacitor
C_input vin gnd {self.input_capacitor:.6e}

* Output capacitor
C_output vout gnd {self.output_capacitor:.6e}
* Initial condition
.IC V(vout)={op_point['output_voltage']:.3f}

* PWM control
V_pwm n_gate gnd PULSE(0 12 0 10n 10n {1/self.specs.switching_frequency*op_point['duty_cycle']:.9f} {1/self.specs.switching_frequency:.9f})

.ends cuk_controller

* Testbench
.include "cuk_controller.cir"
V_input vin_tb 0 DC {op_point['input_voltage']:.2f}
X_controller vin_tb vout_tb 0 cuk_controller
R_load vout_tb 0 6.0

* Analysis
.tran 0.1u 1m 0 0.1u
.probe V(vin_tb) V(vout_tb) I(X_controller.L1) I(X_controller.L2)

.end
"""
    
    def _generate_forward_spice(self, op_point: Dict[str, float]) -> str:
        """Generate SPICE model for forward topology"""
        return f"""* Forward Supercapacitor Controller
* Operating point: Vin={op_point['input_voltage']:.2f}V, Vout={op_point['output_voltage']:.2f}V, D={op_point['duty_cycle']:.3f}

.subckt forward_controller vin vout gnd
* Primary winding
L_primary vin n_prim {self.primary_inductance:.6e}

* Output inductor
L_output n_sec vout {self.output_inductor:.6e}

* Primary switch
M_primary n_prim n_gate gnd gnd NMOS_POWER
.model NMOS_POWER NMOS(VTO=2.0 KP=10 RDS={self.mosfet_rdson:.6f})

* Output rectifier diode
D_rectifier n_sec vout POWER_DIODE
.model POWER_DIODE D(VF={self.diode_vf:.2f} RS=0.01)

* Freewheeling diode
D_freewheeling gnd n_sec POWER_DIODE

* Output capacitor
C_output vout gnd {self.capacitor_output:.6e}
* Initial condition
.IC V(vout)={op_point['output_voltage']:.3f}

* Transformer model
E_transformer n_sec 0 n_prim 0 {self.transformer_ratio:.2f}

* Reset winding (for core reset)
E_reset n_reset 0 n_prim 0 {self.reset_winding_ratio:.2f}
D_reset n_reset vin POWER_DIODE

* PWM control
V_pwm n_gate gnd PULSE(0 12 0 10n 10n {1/self.specs.switching_frequency*op_point['duty_cycle']:.9f} {1/self.specs.switching_frequency:.9f})

.ends forward_controller

* Testbench
.include "forward_controller.cir"
V_input vin_tb 0 DC {op_point['input_voltage']:.2f}
X_controller vin_tb vout_tb 0 forward_controller
R_load vout_tb 0 4.0

* Analysis
.tran 0.1u 1m 0 0.1u
.probe V(vin_tb) V(vout_tb) I(X_controller.L_primary) I(X_controller.L_output)

.end
"""
    
    def _generate_half_bridge_spice(self, op_point: Dict[str, float]) -> str:
        """Generate SPICE model for half-bridge topology"""
        return f"""* Half-Bridge Supercapacitor Controller
* Operating point: Vin={op_point['input_voltage']:.2f}V, Vout={op_point['output_voltage']:.2f}V, D={op_point['duty_cycle']:.3f}

.subckt half_bridge_controller vin vout gnd
* Input voltage divider capacitors
C_div1 vin n_mid {self.blocking_capacitor * 2:.6e}
C_div2 n_mid gnd {self.blocking_capacitor * 2:.6e}

* Primary switches
M_high n_mid n_gate_high vin vin PMOS_POWER
M_low gnd n_gate_low n_mid n_mid NMOS_POWER
.model PMOS_POWER PMOS(VTO=-2.0 KP=10 RDS={self.mosfet_rdson:.6f})
.model NMOS_POWER NMOS(VTO=2.0 KP=10 RDS={self.mosfet_rdson:.6f})

* Primary winding
L_primary n_mid n_prim {self.primary_inductance:.6e}

* Output inductor
L_output n_sec vout {self.output_inductor:.6e}

* Output rectifier diodes
D_rect1 n_sec vout POWER_DIODE
D_rect2 gnd n_sec POWER_DIODE
.model POWER_DIODE D(VF={self.diode_vf:.2f} RS=0.01)

* Output capacitor
C_output vout gnd {self.capacitor_output:.6e}
* Initial condition
.IC V(vout)={op_point['output_voltage']:.3f}

* Transformer model
E_transformer n_sec 0 n_prim 0 {self.transformer_ratio:.2f}

* Blocking capacitor
C_blocking n_prim 0 {self.blocking_capacitor:.6e}

* PWM control with deadtime
V_pwm_high n_gate_high gnd PULSE(12 0 0 10n 10n {1/self.specs.switching_frequency*op_point['duty_cycle']-self.deadtime:.9f} {1/self.specs.switching_frequency:.9f})
V_pwm_low n_gate_low gnd PULSE(0 12 {self.deadtime:.9f} 10n 10n {1/self.specs.switching_frequency*op_point['duty_cycle']-self.deadtime:.9f} {1/self.specs.switching_frequency:.9f})

.ends half_bridge_controller

* Testbench
.include "half_bridge_controller.cir"
V_input vin_tb 0 DC {op_point['input_voltage']:.2f}
X_controller vin_tb vout_tb 0 half_bridge_controller
R_load vout_tb 0 3.0

* Analysis
.tran 0.1u 1m 0 0.1u
.probe V(vin_tb) V(vout_tb) V(X_controller.n_mid) I(X_controller.L_primary)

.end
"""
    
    def _generate_full_bridge_spice(self, op_point: Dict[str, float]) -> str:
        """Generate SPICE model for full-bridge topology"""
        return f"""* Full-Bridge Supercapacitor Controller
* Operating point: Vin={op_point['input_voltage']:.2f}V, Vout={op_point['output_voltage']:.2f}V, D={op_point['duty_cycle']:.3f}

.subckt full_bridge_controller vin vout gnd
* Primary switches (H-bridge)
M_q1 vin n_gate1 n_prim_pos n_prim_pos PMOS_POWER
M_q2 n_prim_pos n_gate2 gnd gnd NMOS_POWER
M_q3 vin n_gate3 n_prim_neg n_prim_neg PMOS_POWER
M_q4 n_prim_neg n_gate4 gnd gnd NMOS_POWER
.model PMOS_POWER PMOS(VTO=-2.0 KP=10 RDS={self.mosfet_rdson:.6f})
.model NMOS_POWER NMOS(VTO=2.0 KP=10 RDS={self.mosfet_rdson:.6f})

* Primary winding (differential)
L_primary n_prim_pos n_prim_neg {self.primary_inductance:.6e}

* Output inductor
L_output n_sec_ct vout {self.output_inductor:.6e}

* Output rectifier (center-tapped)
D_rect1 n_sec_pos n_sec_ct POWER_DIODE
D_rect2 n_sec_neg n_sec_ct POWER_DIODE
.model POWER_DIODE D(VF={self.diode_vf:.2f} RS=0.01)

* Output capacitor
C_output vout gnd {self.capacitor_output:.6e}
* Initial condition
.IC V(vout)={op_point['output_voltage']:.3f}

* Center-tapped transformer model
E_transformer1 n_sec_pos 0 n_prim_pos n_prim_neg {self.transformer_ratio:.2f}
E_transformer2 n_sec_neg 0 n_prim_neg n_prim_pos {self.transformer_ratio:.2f}

* PWM control with phase shift and deadtime
.param phase_shift={self.phase_shift_angle:.1f}
V_pwm1 n_gate1 gnd PULSE(12 0 0 10n 10n {1/self.specs.switching_frequency*op_point['duty_cycle']-self.deadtime:.9f} {1/self.specs.switching_frequency:.9f})
V_pwm2 n_gate2 gnd PULSE(0 12 {self.deadtime:.9f} 10n 10n {1/self.specs.switching_frequency*op_point['duty_cycle']-self.deadtime:.9f} {1/self.specs.switching_frequency:.9f})
V_pwm3 n_gate3 gnd PULSE(12 0 {1/self.specs.switching_frequency*self.phase_shift_angle/360:.9f} 10n 10n {1/self.specs.switching_frequency*op_point['duty_cycle']-self.deadtime:.9f} {1/self.specs.switching_frequency:.9f})
V_pwm4 n_gate4 gnd PULSE(0 12 {1/self.specs.switching_frequency*self.phase_shift_angle/360+self.deadtime:.9f} 10n 10n {1/self.specs.switching_frequency*op_point['duty_cycle']-self.deadtime:.9f} {1/self.specs.switching_frequency:.9f})

.ends full_bridge_controller

* Testbench
.include "full_bridge_controller.cir"
V_input vin_tb 0 DC {op_point['input_voltage']:.2f}
X_controller vin_tb vout_tb 0 full_bridge_controller
R_load vout_tb 0 2.0

* Analysis
.tran 0.1u 1m 0 0.1u
.probe V(vin_tb) V(vout_tb) V(X_controller.n_prim_pos,X_controller.n_prim_neg) I(X_controller.L_primary)

.end
"""
    
    def _generate_generic_spice(self, op_point: Dict[str, float]) -> str:
        """Generate generic SPICE model for unknown topology"""
        return f"""* Generic Supercapacitor Controller Model
* Operating point: Vin={op_point['input_voltage']:.2f}V, Vout={op_point['output_voltage']:.2f}V, D={op_point['duty_cycle']:.3f}

.subckt generic_controller vin vout gnd
* Controlled voltage source model
E_converter vout gnd vin gnd {{V(vin)*{op_point['output_voltage']/op_point['input_voltage']:.3f}}}

* Output resistance
R_output vout vout_internal 0.1

* Current limit
G_current_limit vout_internal gnd VALUE={{if(I(E_converter)<{self.specs.max_current:.2f}, I(E_converter), {self.specs.max_current:.2f})}}

* Efficiency model
R_efficiency vin gnd {{V(vin)/(I(E_converter)/{self.specs.efficiency:.3f})}}

.ends generic_controller

* Testbench
V_input vin_tb 0 DC {op_point['input_voltage']:.2f}
X_controller vin_tb vout_tb 0 generic_controller
R_load vout_tb 0 5.0

* Analysis
.tran 0.1u 1m
.probe V(vin_tb) V(vout_tb) I(X_controller.E_converter)

.end
"""
    
    def get_power_consumption(self) -> float:
        """Get controller power consumption"""
        base_power = self.specs.power_consumption
        
        # Additional power based on switching frequency
        switching_power = self.specs.switching_frequency * 1e-6  # W per kHz
        
        # Temperature-dependent power
        temp_power = 0.01 * abs(self.temperature - 25.0)  # W per °C deviation
        
        return base_power + switching_power + temp_power
    
    def get_cost_analysis(self) -> Dict[str, float]:
        """Get comprehensive cost analysis"""
        initial_cost = self.specs.cost
        
        # Operating cost per hour
        power_cost_per_hour = self.get_power_consumption() * 0.15 / 1000  # $0.15/kWh
        
        # Maintenance cost (minimal for solid-state controllers)
        maintenance_cost_per_hour = 0.002  # $0.002/hour
        
        # Replacement cost amortized over 10-year lifetime
        lifetime_hours = 10 * 365 * 24  # 10 years
        replacement_cost_per_hour = initial_cost / lifetime_hours
        
        total_cost_per_hour = power_cost_per_hour + maintenance_cost_per_hour + replacement_cost_per_hour
        
        return {
            'initial_cost': initial_cost,
            'power_cost_per_hour': power_cost_per_hour,
            'maintenance_cost_per_hour': maintenance_cost_per_hour,
            'replacement_cost_per_hour': replacement_cost_per_hour,
            'total_cost_per_hour': total_cost_per_hour,
            'lifetime_cost': initial_cost + total_cost_per_hour * lifetime_hours,
            'cost_per_watt': initial_cost / self.specs.max_power
        }


def create_standard_supercapacitor_controllers() -> Dict[str, SupercapacitorController]:
    """Create standard supercapacitor controller configurations"""
    
    # Buck-boost controller for general purpose
    buck_boost_specs = SupercapacitorControllerSpecs(
        topology=ControllerTopology.BUCK_BOOST,
        control_mode=ControlMode.CC_CV,
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
    
    # Flyback controller for isolated applications
    flyback_specs = SupercapacitorControllerSpecs(
        topology=ControllerTopology.FLYBACK,
        control_mode=ControlMode.CV,
        input_voltage_range=(10.0, 100.0),
        output_voltage_range=(3.0, 30.0),
        max_current=10.0,
        max_power=300.0,
        switching_frequency=50000.0,
        efficiency=0.88,
        ripple_current=10.0,
        ripple_voltage=5.0,
        response_time=0.002,
        power_consumption=1.5,
        cost=200.0,
        mass=0.5,
        volume=0.2
    )
    
    # High-power full-bridge controller
    full_bridge_specs = SupercapacitorControllerSpecs(
        topology=ControllerTopology.FULL_BRIDGE,
        control_mode=ControlMode.CC,
        input_voltage_range=(24.0, 400.0),
        output_voltage_range=(12.0, 300.0),
        max_current=100.0,
        max_power=5000.0,
        switching_frequency=20000.0,
        efficiency=0.95,
        ripple_current=3.0,
        ripple_voltage=1.0,
        response_time=0.0005,
        power_consumption=10.0,
        cost=800.0,
        mass=2.0,
        volume=1.0
    )
    
    # MPPT controller for energy harvesting
    mppt_specs = SupercapacitorControllerSpecs(
        topology=ControllerTopology.SEPIC,
        control_mode=ControlMode.MPPT,
        input_voltage_range=(0.5, 25.0),
        output_voltage_range=(2.5, 25.0),
        max_current=5.0,
        max_power=100.0,
        switching_frequency=200000.0,
        efficiency=0.90,
        ripple_current=8.0,
        ripple_voltage=3.0,
        response_time=0.01,
        power_consumption=0.5,
        cost=80.0,
        mass=0.1,
        volume=0.05
    )
    
    controllers = {
        'buck_boost_cc_cv': SupercapacitorController(buck_boost_specs),
        'flyback_isolated': SupercapacitorController(flyback_specs),
        'full_bridge_high_power': SupercapacitorController(full_bridge_specs),
        'sepic_mppt': SupercapacitorController(mppt_specs)
    }
    
    return controllers


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create standard controllers
    controllers = create_standard_supercapacitor_controllers()
    
    # Test buck-boost controller
    controller = controllers['buck_boost_cc_cv']
    
    print(f"Testing {controller.specs.topology.value} controller")
    print(f"Control mode: {controller.specs.control_mode.value}")
    print(f"Max power: {controller.specs.max_power:.0f}W")
    
    # Set control parameters
    controller.set_control_parameters(voltage_setpoint=12.0, current_setpoint=10.0)
    
    # Simulate operation
    print("\nControl simulation:")
    for i in range(10):
        measurement = controller.control_step(
            input_voltage=24.0, 
            load_current=5.0 + i,
            dt=0.001,
            temperature=25.0 + i
        )
        print(f"Step {i+1}: Vin={measurement.input_voltage:.1f}V, "
              f"Vout={measurement.output_voltage:.2f}V, "
              f"D={measurement.duty_cycle:.3f}, "
              f"Eff={measurement.efficiency:.1%}, "
              f"Status={measurement.fault_status}")
    
    # Cost analysis
    cost_analysis = controller.get_cost_analysis()
    print("\nCost analysis:")
    print(f"Initial cost: ${cost_analysis['initial_cost']:.2f}")
    print(f"Cost per watt: ${cost_analysis['cost_per_watt']:.2f}/W")
    print(f"Total lifetime cost: ${cost_analysis['lifetime_cost']:.2f}")
    
    # Generate SPICE model
    spice_model = controller.generate_spice_model()
    print(f"\nSPICE model generated ({len(spice_model)} characters)")
    print("Model includes buck-boost topology with PWM control")