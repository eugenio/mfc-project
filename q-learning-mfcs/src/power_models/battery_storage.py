"""
Battery Energy Storage Models for MFC Systems

This module implements various battery chemistries and energy storage models
including Li-ion, LiFePO4, and lead-acid batteries with SOC tracking,
degradation modeling, thermal effects, and Battery Management System (BMS)
with SPICE model generation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BatteryChemistry(Enum):
    """Battery chemistry types"""
    LITHIUM_ION = "li_ion"
    LITHIUM_IRON_PHOSPHATE = "lifepo4"
    LEAD_ACID = "lead_acid"
    NICKEL_METAL_HYDRIDE = "nimh"
    LITHIUM_POLYMER = "lipo"


class ChargeState(Enum):
    """Battery charging states"""
    CHARGING = "charging"
    DISCHARGING = "discharging"
    IDLE = "idle"
    CONSTANT_CURRENT = "cc"
    CONSTANT_VOLTAGE = "cv"
    FLOAT = "float"


class BMSTopology(Enum):
    """Battery Management System topologies"""
    CENTRALIZED = "centralized"
    DISTRIBUTED = "distributed"
    MODULAR = "modular"


class ProtectionFeature(Enum):
    """BMS protection features"""
    OVERVOLTAGE = "overvoltage"
    UNDERVOLTAGE = "undervoltage"
    OVERCURRENT_CHARGE = "overcurrent_charge"
    OVERCURRENT_DISCHARGE = "overcurrent_discharge"
    OVERTEMPERATURE = "overtemperature"
    UNDERTEMPERATURE = "undertemperature"
    SHORT_CIRCUIT = "short_circuit"


@dataclass
class BatterySpecs:
    """Specifications for battery system"""
    chemistry: BatteryChemistry
    nominal_capacity: float  # Ah
    nominal_voltage: float  # V
    max_charge_current: float  # A
    max_discharge_current: float  # A
    charge_voltage: float  # V (max charging voltage)
    cutoff_voltage: float  # V (min discharge voltage)
    internal_resistance: float  # Ω
    self_discharge_rate: float  # %/month
    cycle_life: int  # number of cycles
    temperature_range: Tuple[float, float]  # °C (min, max)
    cost: float  # USD
    weight: float  # kg
    energy_density: float  # Wh/kg
    power_density: float  # W/kg


@dataclass
class BatteryStateOfHealth:
    """Battery state of health tracking"""
    capacity_retention: float  # % of original capacity
    resistance_increase: float  # % increase from original
    cycle_count: int
    deep_discharge_count: int
    overcharge_count: int
    thermal_stress_hours: float
    calendar_age_days: float


@dataclass
class BatteryMeasurement:
    """Single battery measurement"""
    timestamp: float
    voltage: float  # V
    current: float  # A (positive = charging, negative = discharging)
    state_of_charge: float  # % (0-100)
    state_of_health: float  # % (0-100)
    temperature: float  # °C
    internal_resistance: float  # Ω
    power: float  # W
    energy_stored: float  # Wh
    charge_state: ChargeState
    time_to_full: Optional[float]  # hours
    time_to_empty: Optional[float]  # hours


@dataclass
class BMSSpecs:
    """Specifications for Battery Management System"""
    topology: BMSTopology
    num_cells: int
    max_cell_voltage: float  # V
    min_cell_voltage: float  # V
    max_charge_current: float  # A
    max_discharge_current: float  # A
    balancing_current: float  # A (passive/active balancing)
    protection_features: List[ProtectionFeature]
    communication_protocol: str  # CAN, I2C, SPI, etc.
    power_consumption: float  # W
    cost: float  # USD
    accuracy_voltage: float  # mV
    accuracy_current: float  # mA
    accuracy_temperature: float  # °C


@dataclass
class BMSMeasurement:
    """Single BMS measurement"""
    timestamp: float
    cell_voltages: List[float]  # V per cell
    pack_voltage: float  # V
    pack_current: float  # A
    cell_temperatures: List[float]  # °C per cell or group
    balancing_status: List[bool]  # Per cell balancing active
    protection_status: Dict[ProtectionFeature, bool]
    fault_codes: List[str]
    power_consumption: float  # W


class BatteryManagementSystem:
    """Battery Management System with protection and balancing"""
    
    def __init__(self, specs: BMSSpecs, battery_specs: BatterySpecs):
        self.specs = specs
        self.battery_specs = battery_specs
        
        # Cell voltages (start balanced at nominal)
        nominal_cell_voltage = battery_specs.nominal_voltage / specs.num_cells
        self.cell_voltages = [nominal_cell_voltage] * specs.num_cells
        
        # Cell temperatures
        self.cell_temperatures = [25.0] * specs.num_cells
        
        # Balancing state
        self.balancing_active = [False] * specs.num_cells
        self.balancing_time = [0.0] * specs.num_cells
        
        # Protection state
        self.protection_active = {feature: False for feature in ProtectionFeature}
        self.fault_codes = []
        
        # Operating state
        self.operating_hours = 0.0
        self.enabled = True
        
    def update(self, pack_voltage: float, pack_current: float, 
               soc: float, dt: float, time: float) -> BMSMeasurement:
        """
        Update BMS state and return measurement
        
        Args:
            pack_voltage: Battery pack voltage (V)
            pack_current: Pack current (A, positive = charging)
            soc: State of charge (%)
            dt: Time step (hours)
            time: Current time (hours)
            
        Returns:
            BMSMeasurement object
        """
        self.operating_hours = time
        
        # Update cell voltages based on pack voltage and imbalance
        self._update_cell_voltages(pack_voltage, soc)
        
        # Update cell temperatures
        self._update_cell_temperatures(pack_current, dt)
        
        # Check protection conditions
        self._check_protection_conditions(pack_current)
        
        # Update cell balancing
        self._update_balancing(dt)
        
        # Calculate power consumption
        power_consumption = self._calculate_power_consumption()
        
        return BMSMeasurement(
            timestamp=time,
            cell_voltages=self.cell_voltages.copy(),
            pack_voltage=pack_voltage,
            pack_current=pack_current,
            cell_temperatures=self.cell_temperatures.copy(),
            balancing_status=self.balancing_active.copy(),
            protection_status=self.protection_active.copy(),
            fault_codes=self.fault_codes.copy(),
            power_consumption=power_consumption
        )
    
    def _update_cell_voltages(self, pack_voltage: float, soc: float):
        """Update individual cell voltages with imbalance"""
        nominal_cell_voltage = pack_voltage / self.specs.num_cells
        
        # Create cell imbalance (varies with SOC and age)
        imbalance_factors = []
        for i in range(self.specs.num_cells):
            # Each cell has slightly different capacity and resistance
            base_factor = 1.0 + 0.02 * np.sin(i * 2 * np.pi / self.specs.num_cells)  # ±2% variation
            soc_factor = 1.0 + 0.01 * (soc - 50.0) / 50.0 * np.cos(i * np.pi / self.specs.num_cells)  # SOC-dependent
            aging_factor = 1.0 - 0.001 * self.operating_hours / 1000.0 * (i % 2)  # Aging difference
            
            total_factor = base_factor * soc_factor * aging_factor
            imbalance_factors.append(total_factor)
        
        # Normalize to maintain pack voltage
        factor_sum = sum(imbalance_factors)
        normalized_factors = [f * self.specs.num_cells / factor_sum for f in imbalance_factors]
        
        # Update cell voltages
        for i in range(self.specs.num_cells):
            self.cell_voltages[i] = nominal_cell_voltage * normalized_factors[i]
    
    def _update_cell_temperatures(self, current: float, dt: float):
        """Update cell temperatures based on current and heat generation"""
        ambient_temp = 25.0
        
        for i in range(self.specs.num_cells):
            # Heat generation from I²R losses
            cell_current = current  # Assume series connection
            cell_resistance = self.battery_specs.internal_resistance / self.specs.num_cells
            power_dissipation = cell_current**2 * cell_resistance
            
            # Thermal time constant (simplified)
            thermal_time_constant = 3600.0  # 1 hour
            temp_rise = power_dissipation * 50.0  # °C/W thermal resistance
            target_temp = ambient_temp + temp_rise
            
            # Temperature response
            self.cell_temperatures[i] += (target_temp - self.cell_temperatures[i]) * dt / thermal_time_constant
    
    def _check_protection_conditions(self, current: float):
        """Check all protection conditions and update status"""
        # Clear previous faults for rechecking
        self.fault_codes.clear()
        
        # Overvoltage protection
        max_cell_voltage = max(self.cell_voltages)
        if max_cell_voltage > self.specs.max_cell_voltage:
            self.protection_active[ProtectionFeature.OVERVOLTAGE] = True
            self.fault_codes.append(f"OVERVOLTAGE: {max_cell_voltage:.3f}V > {self.specs.max_cell_voltage:.3f}V")
        else:
            self.protection_active[ProtectionFeature.OVERVOLTAGE] = False
        
        # Undervoltage protection
        min_cell_voltage = min(self.cell_voltages)
        if min_cell_voltage < self.specs.min_cell_voltage:
            self.protection_active[ProtectionFeature.UNDERVOLTAGE] = True
            self.fault_codes.append(f"UNDERVOLTAGE: {min_cell_voltage:.3f}V < {self.specs.min_cell_voltage:.3f}V")
        else:
            self.protection_active[ProtectionFeature.UNDERVOLTAGE] = False
        
        # Overcurrent protection
        if current > 0 and current > self.specs.max_charge_current:
            self.protection_active[ProtectionFeature.OVERCURRENT_CHARGE] = True
            self.fault_codes.append(f"OVERCURRENT_CHARGE: {current:.2f}A > {self.specs.max_charge_current:.2f}A")
        else:
            self.protection_active[ProtectionFeature.OVERCURRENT_CHARGE] = False
            
        if current < 0 and abs(current) > self.specs.max_discharge_current:
            self.protection_active[ProtectionFeature.OVERCURRENT_DISCHARGE] = True
            self.fault_codes.append(f"OVERCURRENT_DISCHARGE: {abs(current):.2f}A > {self.specs.max_discharge_current:.2f}A")
        else:
            self.protection_active[ProtectionFeature.OVERCURRENT_DISCHARGE] = False
        
        # Temperature protection
        max_temp = max(self.cell_temperatures)
        min_temp = min(self.cell_temperatures)
        
        if max_temp > 60.0:  # Typical overtemperature threshold
            self.protection_active[ProtectionFeature.OVERTEMPERATURE] = True
            self.fault_codes.append(f"OVERTEMPERATURE: {max_temp:.1f}°C > 60.0°C")
        else:
            self.protection_active[ProtectionFeature.OVERTEMPERATURE] = False
            
        if min_temp < -10.0:  # Typical undertemperature threshold
            self.protection_active[ProtectionFeature.UNDERTEMPERATURE] = True
            self.fault_codes.append(f"UNDERTEMPERATURE: {min_temp:.1f}°C < -10.0°C")
        else:
            self.protection_active[ProtectionFeature.UNDERTEMPERATURE] = False
        
        # Update enabled status based on critical faults
        critical_faults = [
            ProtectionFeature.OVERVOLTAGE,
            ProtectionFeature.UNDERVOLTAGE,
            ProtectionFeature.OVERCURRENT_CHARGE,
            ProtectionFeature.OVERCURRENT_DISCHARGE,
            ProtectionFeature.OVERTEMPERATURE
        ]
        
        self.enabled = not any(self.protection_active[fault] for fault in critical_faults)
    
    def _update_balancing(self, dt: float):
        """Update cell balancing logic"""
        if not self.enabled:
            # Disable balancing if protection is active
            self.balancing_active = [False] * self.specs.num_cells
            return
        
        # Calculate voltage differences
        avg_voltage = np.mean(self.cell_voltages)
        voltage_threshold = 0.01  # 10mV balancing threshold
        
        for i in range(self.specs.num_cells):
            voltage_diff = self.cell_voltages[i] - avg_voltage
            
            if voltage_diff > voltage_threshold:
                # Cell voltage too high - activate balancing
                self.balancing_active[i] = True
                self.balancing_time[i] += dt
                
                # Simulate balancing effect (discharge high cell slightly)
                if self.specs.topology in [BMSTopology.CENTRALIZED, BMSTopology.MODULAR]:
                    # Passive balancing - resistor discharge
                    balancing_power = self.specs.balancing_current * self.cell_voltages[i]
                    energy_removed = balancing_power * dt  # Wh
                    voltage_reduction = energy_removed / 0.1  # Simplified: 0.1Wh per 1V
                    self.cell_voltages[i] -= min(voltage_reduction, voltage_diff)
                
            elif voltage_diff < -voltage_threshold and self.specs.topology == BMSTopology.DISTRIBUTED:
                # Active balancing can charge low cells
                self.balancing_active[i] = True
                # Charge transfer from high cells to this cell
                
            else:
                self.balancing_active[i] = False
    
    def _calculate_power_consumption(self) -> float:
        """Calculate BMS power consumption"""
        base_power = self.specs.power_consumption
        
        # Additional power for active balancing
        active_balancing_cells = sum(self.balancing_active)
        balancing_power = active_balancing_cells * 0.01  # 10mW per active balancing circuit
        
        # Communication and monitoring power
        comm_power = 0.05  # 50mW for communication
        
        return base_power + balancing_power + comm_power
    
    def generate_spice_model(self, output_file: Optional[str] = None,
                           operating_point: Optional[Dict[str, float]] = None) -> str:
        """
        Generate SPICE netlist model of the BMS
        
        Args:
            output_file: Optional file path to save the SPICE model
            operating_point: Optional operating point for component values
            
        Returns:
            SPICE netlist as string
        """
        if operating_point is None:
            operating_point = {
                'pack_voltage': self.battery_specs.nominal_voltage,
                'pack_current': 1.0,
                'cell_voltage': self.battery_specs.nominal_voltage / self.specs.num_cells,
                'balancing_current': self.specs.balancing_current
            }
        
        # Generate SPICE netlist
        spice_model = self._generate_bms_spice_header()
        spice_model += self._generate_bms_spice_parameters(operating_point)
        spice_model += self._generate_bms_spice_circuit()
        spice_model += self._generate_bms_spice_analysis()
        spice_model += self._generate_bms_spice_footer()
        
        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(spice_model)
            logger.info(f"BMS SPICE model saved to {output_file}")
        
        return spice_model
    
    def _generate_bms_spice_header(self) -> str:
        """Generate BMS SPICE model header"""
        header = f"""* Battery Management System SPICE Model
* Topology: {self.specs.topology.value}
* Number of cells: {self.specs.num_cells}
* Generated by MFC Power Models
* 
* Pack Voltage: {self.battery_specs.nominal_voltage:.1f}V
* Cell Voltage Range: {self.specs.min_cell_voltage:.1f}V - {self.specs.max_cell_voltage:.1f}V
* Max Charge Current: {self.specs.max_charge_current:.1f}A
* Max Discharge Current: {self.specs.max_discharge_current:.1f}A

"""
        return header
    
    def _generate_bms_spice_parameters(self, operating_point: Dict[str, float]) -> str:
        """Generate BMS SPICE parameter definitions"""
        pack_voltage = operating_point['pack_voltage']
        cell_voltage = operating_point['cell_voltage']
        
        params = f"""* BMS Parameters
.param Vpack={pack_voltage:.2f}
.param Vcell={cell_voltage:.2f}
.param Ncells={self.specs.num_cells}
.param Icharge_max={self.specs.max_charge_current:.2f}
.param Idischarge_max={self.specs.max_discharge_current:.2f}
.param Vovp={self.specs.max_cell_voltage:.3f}
.param Vuvp={self.specs.min_cell_voltage:.3f}
.param Ibal={self.specs.balancing_current:.3f}

* Component Values
.param Rbal=10
.param Cfilter=1u
.param Rshunt=1m
.param Vref=2.5

"""
        return params
    
    def _generate_bms_spice_circuit(self) -> str:
        """Generate BMS SPICE circuit netlist"""
        if self.specs.topology == BMSTopology.CENTRALIZED:
            return self._generate_centralized_bms_circuit()
        elif self.specs.topology == BMSTopology.DISTRIBUTED:
            return self._generate_distributed_bms_circuit()
        else:  # MODULAR
            return self._generate_modular_bms_circuit()
    
    def _generate_centralized_bms_circuit(self) -> str:
        """Generate centralized BMS SPICE circuit"""
        circuit = """* Centralized BMS Circuit
* Battery pack (series cells)
"""
        
        # Generate battery cells
        for i in range(self.specs.num_cells):
            if i == 0:
                pos_node = f"cell_{i+1}_pos"
                neg_node = "pack_neg"
            else:
                pos_node = f"cell_{i+1}_pos"
                neg_node = f"cell_{i}_pos"
            
            circuit += f"""
* Cell {i+1}
Vcell_{i+1} {pos_node} {neg_node} DC {{Vcell}}
Rcell_{i+1} {pos_node} n_cell_{i+1} {{Vcell/100}}
Ccell_{i+1} n_cell_{i+1} {neg_node} 10000u
"""
        
        # Main BMS controller
        circuit += f"""
* Main BMS Controller
Vcc vcc 0 DC 5.0
Vref vref 0 DC {{Vref}}

* Current sensing shunt
Rshunt pack_pos cell_{self.specs.num_cells}_pos {{Rshunt}}

* Pack terminals
Vpack_load pack_pos 0 DC {{Vpack}}

* Cell voltage monitoring (simplified)"""
        
        for i in range(self.specs.num_cells):
            pos_node = f"cell_{i+1}_pos" if i < self.specs.num_cells - 1 else "pack_pos"
            neg_node = f"cell_{i}_pos" if i > 0 else "pack_neg"
            
            circuit += f"""
* Cell {i+1} voltage monitor
Rmux_{i+1}_h {pos_node} vmux_{i+1}_h 1k
Rmux_{i+1}_l {neg_node} vmux_{i+1}_l 1k
Cfilter_{i+1} vmux_{i+1}_h vmux_{i+1}_l {{Cfilter}}

* Cell {i+1} balancing resistor
Sw_bal_{i+1} vmux_{i+1}_h n_bal_{i+1} vbal_{i+1} 0 SWITCH
Rbal_{i+1} n_bal_{i+1} vmux_{i+1}_l {{Rbal}}
Vbal_{i+1} vbal_{i+1} 0 PULSE(0 5 0 1u 1u 1m 2m)
"""
        
        circuit += """
* Overvoltage/Undervoltage comparators (simplified)
Eovp ovp_out 0 VCVS value={if(V(vmux_1_h,vmux_1_l)>Vovp,5,0)}
Euvp uvp_out 0 VCVS value={if(V(vmux_1_h,vmux_1_l)<Vuvp,5,0)}

* Switch model
.model SWITCH SW(Vt=2.5 Vh=0.1)

"""
        return circuit
    
    def _generate_distributed_bms_circuit(self) -> str:
        """Generate distributed BMS SPICE circuit"""
        circuit = """* Distributed BMS Circuit
* Each cell has its own monitoring and balancing circuit

"""
        for i in range(self.specs.num_cells):
            circuit += f"""
* Cell {i+1} with local BMS
Vcell_{i+1} cell_{i+1}_pos cell_{i+1}_neg DC {{Vcell}}
Rcell_{i+1} cell_{i+1}_pos n_cell_{i+1} {{Vcell/100}}

* Local BMS for cell {i+1}
Xlbms_{i+1} cell_{i+1}_pos cell_{i+1}_neg vcc 0 local_bms

"""
        
        circuit += """
* Local BMS subcircuit
.subckt local_bms cell_pos cell_neg vcc gnd
Rmux_h cell_pos vmux_h 1k
Rmux_l cell_neg vmux_l 1k
Cfilter vmux_h vmux_l 1u

* Voltage monitoring
Evmon vmon gnd vmux_h vmux_l 1

* Balancing circuit
Sw_bal vmux_h n_bal vbal gnd SWITCH
Rbal n_bal vmux_l 10
Vbal vbal gnd PULSE(0 5 0 1u 1u 1m 2m)

* Protection comparators
Eovp ovp gnd value={if(V(vmon)>Vovp,5,0)}
Euvp uvp gnd value={if(V(vmon)<Vuvp,5,0)}
.ends local_bms

* Switch model
.model SWITCH SW(Vt=2.5 Vh=0.1)

"""
        return circuit
    
    def _generate_modular_bms_circuit(self) -> str:
        """Generate modular BMS SPICE circuit"""
        circuit = """* Modular BMS Circuit
* Modules handle groups of cells

"""
        cells_per_module = max(1, self.specs.num_cells // 4)  # 4 modules max
        num_modules = (self.specs.num_cells + cells_per_module - 1) // cells_per_module
        
        for mod in range(num_modules):
            start_cell = mod * cells_per_module
            end_cell = min(start_cell + cells_per_module, self.specs.num_cells)
            
            circuit += f"""
* Module {mod+1} (cells {start_cell+1}-{end_cell})
Xbms_mod_{mod+1} mod_{mod+1}_pos mod_{mod+1}_neg vcc 0 bms_module

"""
            
            # Cells in this module
            for i in range(start_cell, end_cell):
                if i == start_cell:
                    pos_node = f"mod_{mod+1}_pos"
                    neg_node = f"cell_{i+1}_neg" if i < end_cell - 1 else f"mod_{mod+1}_neg"
                else:
                    pos_node = f"cell_{i}_neg"
                    neg_node = f"cell_{i+1}_neg" if i < end_cell - 1 else f"mod_{mod+1}_neg"
                
                circuit += f"""
Vcell_{i+1} {pos_node} {neg_node} DC {{Vcell}}
Rcell_{i+1} {pos_node} n_cell_{i+1} {{Vcell/100}}
"""
        
        circuit += """
* BMS Module subcircuit
.subckt bms_module mod_pos mod_neg vcc gnd
* Simplified module monitoring and balancing
Rmod_sense mod_pos mod_neg 1Meg
Cmod_filter mod_pos mod_neg 10u

* Module protection
Emod_ovp ovp gnd value={if(V(mod_pos,mod_neg)>Vovp*4,5,0)}
Emod_uvp uvp gnd value={if(V(mod_pos,mod_neg)<Vuvp*4,5,0)}
.ends bms_module

"""
        return circuit
    
    def _generate_bms_spice_analysis(self) -> str:
        """Generate BMS SPICE analysis commands"""
        analysis = """* Analysis Commands
* Operating point analysis
.op

* Transient analysis for balancing behavior
.tran 1m 10

* DC sweep of cell voltages
.dc Vcell_1 2.5 4.2 0.1

* Measurements
.measure tran avg_cell1_voltage AVG V(cell_1_pos,pack_neg) FROM 5 TO 10
.measure tran balancing_current AVG I(Rbal_1) FROM 5 TO 10
.measure dc ovp_threshold FIND V(ovp_out) WHEN V(cell_1_pos,pack_neg)=Vovp

* Output commands
.print tran V(cell_1_pos,pack_neg) V(ovp_out) V(uvp_out) I(Rbal_1)
.plot tran V(cell_1_pos,pack_neg) V(vbal_1)

"""
        return analysis
    
    def _generate_bms_spice_footer(self) -> str:
        """Generate BMS SPICE model footer"""
        footer = """* End of BMS model
.end
"""
        return footer


class BatterySystem:
    """Battery energy storage system model"""
    
    def __init__(self, specs: BatterySpecs):
        self.specs = specs
        
        # State variables
        self.state_of_charge = 50.0  # % (start at 50%)
        self.temperature = 25.0  # °C
        self.current = 0.0  # A
        self.voltage = specs.nominal_voltage  # V
        
        # State of health tracking
        self.soh = BatteryStateOfHealth(
            capacity_retention=100.0,
            resistance_increase=0.0,
            cycle_count=0,
            deep_discharge_count=0,
            overcharge_count=0,
            thermal_stress_hours=0.0,
            calendar_age_days=0.0
        )
        
        # Operating history
        self.operating_hours = 0.0
        self.total_energy_charged = 0.0  # Wh
        self.total_energy_discharged = 0.0  # Wh
        self.charge_state = ChargeState.IDLE
        
        # Initialize chemistry-specific parameters
        self._initialize_chemistry_parameters()
        
        # SOC tracking for cycle counting
        self.soc_history = [self.state_of_charge]
        self.last_soc_peak = self.state_of_charge
        self.last_soc_valley = self.state_of_charge
        
    def _initialize_chemistry_parameters(self):
        """Initialize chemistry-specific parameters"""
        if self.specs.chemistry == BatteryChemistry.LITHIUM_ION:
            # Li-ion parameters
            self.capacity_fade_rate = 0.02  # % per 100 cycles
            self.resistance_growth_rate = 0.005  # % per 100 cycles
            self.thermal_runaway_temp = 130.0  # °C
            self.optimal_temp_range = (15.0, 35.0)  # °C
            self.voltage_curve_coeffs = [3.2, 0.8, -0.1, 0.02]  # Polynomial coefficients
            
        elif self.specs.chemistry == BatteryChemistry.LITHIUM_IRON_PHOSPHATE:
            # LiFePO4 parameters
            self.capacity_fade_rate = 0.01  # % per 100 cycles (better than Li-ion)
            self.resistance_growth_rate = 0.003  # % per 100 cycles
            self.thermal_runaway_temp = 200.0  # °C (much safer)
            self.optimal_temp_range = (10.0, 45.0)  # °C
            self.voltage_curve_coeffs = [3.0, 0.4, -0.05, 0.01]  # Flatter curve
            
        elif self.specs.chemistry == BatteryChemistry.LEAD_ACID:
            # Lead-acid parameters
            self.capacity_fade_rate = 0.1  # % per 100 cycles (worse than Li-ion)
            self.resistance_growth_rate = 0.02  # % per 100 cycles
            self.thermal_runaway_temp = 60.0  # °C (lower)
            self.optimal_temp_range = (20.0, 25.0)  # °C (narrow range)
            self.voltage_curve_coeffs = [1.8, 0.4, -0.05, 0.01]  # Different voltage range
            
        elif self.specs.chemistry == BatteryChemistry.NICKEL_METAL_HYDRIDE:
            # NiMH parameters
            self.capacity_fade_rate = 0.05  # % per 100 cycles
            self.resistance_growth_rate = 0.01  # % per 100 cycles
            self.thermal_runaway_temp = 100.0  # °C
            self.optimal_temp_range = (0.0, 40.0)  # °C
            self.voltage_curve_coeffs = [1.0, 0.3, -0.02, 0.005]
            
        else:  # LITHIUM_POLYMER
            # LiPo parameters (similar to Li-ion but slightly different)
            self.capacity_fade_rate = 0.025  # % per 100 cycles
            self.resistance_growth_rate = 0.006  # % per 100 cycles
            self.thermal_runaway_temp = 120.0  # °C
            self.optimal_temp_range = (15.0, 35.0)  # °C
            self.voltage_curve_coeffs = [3.0, 0.9, -0.12, 0.025]
    
    def set_state_of_charge(self, soc: float):
        """Set the current state of charge"""
        self.state_of_charge = np.clip(soc, 0.0, 100.0)
        self.soc_history.append(self.state_of_charge)
        
        # Keep only last 1000 SOC readings for cycle counting
        if len(self.soc_history) > 1000:
            self.soc_history = self.soc_history[-1000:]
    
    def charge(self, current: float, dt: float, temperature: float = 25.0) -> BatteryMeasurement:
        """
        Charge the battery with specified current
        
        Args:
            current: Charging current (A, positive)
            dt: Time step (hours)
            temperature: Battery temperature (°C)
            
        Returns:
            BatteryMeasurement object
        """
        # Limit charging current
        current = min(abs(current), self.specs.max_charge_current)
        
        # Update temperature
        self.temperature = temperature
        
        # Calculate voltage based on SOC and current
        self.voltage = self._calculate_voltage(self.state_of_charge, current, is_charging=True)
        
        # Check if at maximum voltage (CV mode)
        if self.voltage >= self.specs.charge_voltage:
            self.charge_state = ChargeState.CONSTANT_VOLTAGE
            # In CV mode, current tapers off as SOC approaches 100%
            current = current * (100.0 - self.state_of_charge) / 20.0  # Taper current
        else:
            self.charge_state = ChargeState.CONSTANT_CURRENT
        
        # Update SOC based on charge added
        available_capacity = self._get_available_capacity()
        charge_added = current * dt  # Ah
        soc_increase = (charge_added / available_capacity) * 100.0
        
        # Apply charging efficiency
        charging_efficiency = self._get_charging_efficiency(current, temperature)
        effective_soc_increase = soc_increase * charging_efficiency
        
        self.set_state_of_charge(self.state_of_charge + effective_soc_increase)
        
        # Update current and power
        self.current = current
        power = self.voltage * current
        
        # Track energy
        energy_added = power * dt
        self.total_energy_charged += energy_added
        
        # Update aging
        self._update_aging(current, dt, temperature, is_charging=True)
        
        # Check for overcharge
        if self.state_of_charge >= 99.5:
            self.soh.overcharge_count += dt * 60  # Count minutes of overcharge
        
        return self._create_measurement(power, energy_added)
    
    def discharge(self, current: float, dt: float, temperature: float = 25.0) -> BatteryMeasurement:
        """
        Discharge the battery with specified current
        
        Args:
            current: Discharge current (A, positive)
            dt: Time step (hours)
            temperature: Battery temperature (°C)
            
        Returns:
            BatteryMeasurement object
        """
        # Limit discharge current
        current = min(abs(current), self.specs.max_discharge_current)
        
        # Update temperature
        self.temperature = temperature
        
        # Calculate voltage based on SOC and current
        self.voltage = self._calculate_voltage(self.state_of_charge, current, is_charging=False)
        
        # Check cutoff voltage
        if self.voltage <= self.specs.cutoff_voltage:
            current = 0.0  # Stop discharge at cutoff
            self.charge_state = ChargeState.IDLE
        else:
            self.charge_state = ChargeState.DISCHARGING
        
        # Update SOC based on charge removed
        available_capacity = self._get_available_capacity()
        charge_removed = current * dt  # Ah
        soc_decrease = (charge_removed / available_capacity) * 100.0
        
        # Apply discharge efficiency
        discharge_efficiency = self._get_discharge_efficiency(current, temperature)
        effective_soc_decrease = soc_decrease / discharge_efficiency
        
        self.set_state_of_charge(self.state_of_charge - effective_soc_decrease)
        
        # Update current and power (negative for discharge)
        self.current = -current
        power = self.voltage * current
        
        # Track energy
        energy_removed = power * dt
        self.total_energy_discharged += energy_removed
        
        # Update aging
        self._update_aging(current, dt, temperature, is_charging=False)
        
        # Check for deep discharge
        if self.state_of_charge <= 10.0:
            self.soh.deep_discharge_count += dt * 60  # Count minutes of deep discharge
        
        return self._create_measurement(-power, energy_removed)
    
    def idle(self, dt: float, temperature: float = 25.0) -> BatteryMeasurement:
        """
        Battery in idle state (no charging or discharging)
        
        Args:
            dt: Time step (hours)
            temperature: Battery temperature (°C)
            
        Returns:
            BatteryMeasurement object
        """
        # Update temperature
        self.temperature = temperature
        
        # Self-discharge
        self_discharge_rate_per_hour = self.specs.self_discharge_rate / (30.44 * 24)  # Convert from %/month to %/hour
        soc_decrease = self_discharge_rate_per_hour * dt
        self.set_state_of_charge(self.state_of_charge - soc_decrease)
        
        # Open circuit voltage
        self.voltage = self._calculate_voltage(self.state_of_charge, 0.0, is_charging=False)
        self.current = 0.0
        self.charge_state = ChargeState.IDLE
        
        # Update aging (calendar aging)
        self._update_aging(0.0, dt, temperature, is_charging=False)
        
        return self._create_measurement(0.0, 0.0)
    
    def _calculate_voltage(self, soc: float, current: float, is_charging: bool) -> float:
        """Calculate battery voltage based on SOC and current"""
        # Open circuit voltage from SOC curve
        soc_normalized = soc / 100.0
        
        # Polynomial voltage curve
        ocv = (self.voltage_curve_coeffs[0] + 
               self.voltage_curve_coeffs[1] * soc_normalized + 
               self.voltage_curve_coeffs[2] * soc_normalized**2 + 
               self.voltage_curve_coeffs[3] * soc_normalized**3)
        
        # Scale to nominal voltage
        ocv *= self.specs.nominal_voltage / 3.7  # Normalize to nominal voltage
        
        # Internal resistance voltage drop
        current_resistance = self._get_current_internal_resistance()
        
        if is_charging:
            voltage = ocv + current * current_resistance  # Voltage rises during charging
        else:
            voltage = ocv - current * current_resistance  # Voltage drops during discharge
        
        # Apply temperature correction
        temp_coeff = -0.003 if self.specs.chemistry == BatteryChemistry.LITHIUM_ION else -0.002  # V/°C
        voltage += temp_coeff * (self.temperature - 25.0)
        
        return max(voltage, 0.0)
    
    def _get_available_capacity(self) -> float:
        """Get current available capacity considering degradation"""
        return self.specs.nominal_capacity * (self.soh.capacity_retention / 100.0)
    
    def _get_current_internal_resistance(self) -> float:
        """Get current internal resistance considering degradation"""
        resistance_multiplier = 1.0 + (self.soh.resistance_increase / 100.0)
        return self.specs.internal_resistance * resistance_multiplier
    
    def _get_charging_efficiency(self, current: float, temperature: float) -> float:
        """Get charging efficiency based on current and temperature"""
        # Base efficiency
        if self.specs.chemistry == BatteryChemistry.LITHIUM_ION:
            base_efficiency = 0.95
        elif self.specs.chemistry == BatteryChemistry.LITHIUM_IRON_PHOSPHATE:
            base_efficiency = 0.98
        elif self.specs.chemistry == BatteryChemistry.LEAD_ACID:
            base_efficiency = 0.85
        else:
            base_efficiency = 0.90
        
        # Current rate effect (C-rate)
        c_rate = current / self.specs.nominal_capacity
        current_factor = 1.0 - 0.05 * max(0, c_rate - 0.5)  # Efficiency drops at high C-rates
        
        # Temperature effect
        temp_optimal = np.mean(self.optimal_temp_range)
        temp_deviation = abs(temperature - temp_optimal)
        temp_factor = 1.0 - 0.01 * temp_deviation
        
        return base_efficiency * current_factor * temp_factor
    
    def _get_discharge_efficiency(self, current: float, temperature: float) -> float:
        """Get discharge efficiency based on current and temperature"""
        # Discharge efficiency is typically higher than charge efficiency
        charge_eff = self._get_charging_efficiency(current, temperature)
        return min(0.99, charge_eff + 0.02)
    
    def _update_aging(self, current: float, dt: float, temperature: float, is_charging: bool):
        """Update battery aging parameters"""
        # Update operating hours
        self.operating_hours += dt
        
        # Calendar aging
        self.soh.calendar_age_days += dt / 24.0
        calendar_aging_rate = 0.001  # % capacity loss per day
        calendar_capacity_loss = calendar_aging_rate * dt / 24.0
        
        # Cycle aging (only if current > 0)
        cycle_capacity_loss = 0.0
        cycle_resistance_increase = 0.0
        
        if current > 0:
            # Calculate equivalent cycles
            c_rate = current / self.specs.nominal_capacity
            depth_factor = 1.0  # Simplified - real batteries have complex DoD effects
            
            equivalent_cycles = dt * c_rate * depth_factor / 2.0  # Half cycle per time step
            self.soh.cycle_count += equivalent_cycles
            
            # Capacity loss from cycling
            cycle_capacity_loss = (equivalent_cycles * self.capacity_fade_rate) / 100.0
            
            # Resistance increase from cycling  
            cycle_resistance_increase = (equivalent_cycles * self.resistance_growth_rate) / 100.0
        
        # Temperature aging
        temp_capacity_loss = 0.0
        if temperature > self.optimal_temp_range[1]:
            # High temperature stress
            stress_factor = (temperature - self.optimal_temp_range[1]) / 10.0
            temp_capacity_loss = stress_factor * 0.0001 * dt  # % per hour at elevated temp
            self.soh.thermal_stress_hours += dt
        
        # Apply aging
        total_capacity_loss = calendar_capacity_loss + cycle_capacity_loss + temp_capacity_loss
        self.soh.capacity_retention = max(50.0, self.soh.capacity_retention - total_capacity_loss)
        self.soh.resistance_increase += cycle_resistance_increase
        
        # Update cycle counting based on SOC history
        self._update_cycle_counting()
    
    def _update_cycle_counting(self):
        """Update cycle counting based on SOC peaks and valleys"""
        if len(self.soc_history) < 3:
            return
        
        current_soc = self.soc_history[-1]
        prev_soc = self.soc_history[-2]
        
        # Detect peaks and valleys
        if len(self.soc_history) >= 3:
            prev_prev_soc = self.soc_history[-3]
            
            # Peak detection (local maximum)
            if prev_soc > current_soc and prev_soc > prev_prev_soc:
                if prev_soc > self.last_soc_peak:
                    self.last_soc_peak = prev_soc
            
            # Valley detection (local minimum)
            if prev_soc < current_soc and prev_soc < prev_prev_soc:
                if prev_soc < self.last_soc_valley:
                    self.last_soc_valley = prev_soc
                    
                    # Calculate cycle depth when we hit a valley
                    cycle_depth = self.last_soc_peak - self.last_soc_valley
                    if cycle_depth > 10.0:  # Only count significant cycles
                        equivalent_cycles = cycle_depth / 100.0  # Full cycle = 100% DoD
                        self.soh.cycle_count += equivalent_cycles
    
    def _create_measurement(self, power: float, energy: float) -> BatteryMeasurement:
        """Create a battery measurement object"""
        # Calculate time to full/empty
        time_to_full = None
        time_to_empty = None
        
        if self.current > 0:  # Charging
            remaining_capacity = (100.0 - self.state_of_charge) / 100.0 * self._get_available_capacity()
            time_to_full = remaining_capacity / self.current if self.current > 0 else None
        elif self.current < 0:  # Discharging
            available_energy = self.state_of_charge / 100.0 * self._get_available_capacity()
            time_to_empty = available_energy / abs(self.current) if self.current < 0 else None
        
        # Calculate stored energy
        energy_stored = (self.state_of_charge / 100.0) * self._get_available_capacity() * self.specs.nominal_voltage
        
        # Calculate state of health
        state_of_health = self.soh.capacity_retention
        
        return BatteryMeasurement(
            timestamp=self.operating_hours,
            voltage=self.voltage,
            current=self.current,
            state_of_charge=self.state_of_charge,
            state_of_health=state_of_health,
            temperature=self.temperature,
            internal_resistance=self._get_current_internal_resistance(),
            power=power,
            energy_stored=energy_stored,
            charge_state=self.charge_state,
            time_to_full=time_to_full,
            time_to_empty=time_to_empty
        )
    
    def get_power_consumption(self) -> float:
        """Get battery management system power consumption"""
        # BMS power consumption varies by chemistry and features
        if self.specs.chemistry == BatteryChemistry.LITHIUM_ION:
            bms_power = 0.05  # W (typical for Li-ion BMS)
        elif self.specs.chemistry == BatteryChemistry.LEAD_ACID:
            bms_power = 0.02  # W (simpler BMS)
        else:
            bms_power = 0.03  # W
        
        # Additional power for active balancing
        if abs(self.current) > 0:
            balancing_power = 0.01  # W during active operation
        else:
            balancing_power = 0.005  # W for monitoring
        
        return bms_power + balancing_power
    
    def get_cost_analysis(self) -> Dict[str, float]:
        """Get comprehensive cost analysis"""
        initial_cost = self.specs.cost
        
        # Operating cost per hour
        bms_power_cost = self.get_power_consumption() * 0.15 / 1000  # $0.15/kWh
        
        # Replacement cost based on cycle life and current health
        expected_remaining_cycles = self.specs.cycle_life * (self.soh.capacity_retention / 100.0)
        replacement_cost_per_hour = initial_cost / (expected_remaining_cycles * 2)  # Assume 2h per cycle average
        
        # Maintenance cost (BMS updates, monitoring)
        maintenance_cost_per_hour = 0.001
        
        total_cost_per_hour = bms_power_cost + replacement_cost_per_hour + maintenance_cost_per_hour
        
        return {
            'initial_cost': initial_cost,
            'power_cost_per_hour': bms_power_cost,
            'maintenance_cost_per_hour': maintenance_cost_per_hour,
            'replacement_cost_per_hour': replacement_cost_per_hour,
            'total_cost_per_hour': total_cost_per_hour,
            'expected_lifetime_hours': expected_remaining_cycles * 2,
            'lifetime_cost': initial_cost + total_cost_per_hour * expected_remaining_cycles * 2
        }
    
    def get_state_summary(self) -> Dict[str, float]:
        """Get comprehensive state summary"""
        return {
            'state_of_charge': self.state_of_charge,
            'state_of_health': self.soh.capacity_retention,
            'voltage': self.voltage,
            'current': self.current,
            'temperature': self.temperature,
            'internal_resistance': self._get_current_internal_resistance(),
            'available_capacity': self._get_available_capacity(),
            'energy_stored': (self.state_of_charge / 100.0) * self._get_available_capacity() * self.specs.nominal_voltage,
            'cycle_count': self.soh.cycle_count,
            'operating_hours': self.operating_hours,
            'total_energy_charged': self.total_energy_charged,
            'total_energy_discharged': self.total_energy_discharged,
            'round_trip_efficiency': (self.total_energy_discharged / self.total_energy_charged * 100) if self.total_energy_charged > 0 else 0
        }


def create_standard_battery_systems() -> Dict[str, BatterySystem]:
    """Create standard battery system configurations"""
    
    # Li-ion battery (18650 cells in pack)
    li_ion_specs = BatterySpecs(
        chemistry=BatteryChemistry.LITHIUM_ION,
        nominal_capacity=10.0,  # Ah
        nominal_voltage=14.4,  # V (4S configuration)
        max_charge_current=5.0,  # A (0.5C)
        max_discharge_current=20.0,  # A (2C)
        charge_voltage=16.8,  # V (4.2V per cell)
        cutoff_voltage=11.2,  # V (2.8V per cell)
        internal_resistance=0.05,  # Ω
        self_discharge_rate=2.0,  # %/month
        cycle_life=2000,
        temperature_range=(-10.0, 60.0),  # °C
        cost=300.0,  # USD
        weight=1.5,  # kg
        energy_density=150.0,  # Wh/kg
        power_density=500.0  # W/kg
    )
    
    # LiFePO4 battery (safer, longer life)
    lifepo4_specs = BatterySpecs(
        chemistry=BatteryChemistry.LITHIUM_IRON_PHOSPHATE,
        nominal_capacity=20.0,  # Ah
        nominal_voltage=12.8,  # V (4S configuration)
        max_charge_current=10.0,  # A (0.5C)
        max_discharge_current=40.0,  # A (2C)
        charge_voltage=14.4,  # V (3.6V per cell)
        cutoff_voltage=10.0,  # V (2.5V per cell)
        internal_resistance=0.03,  # Ω
        self_discharge_rate=1.0,  # %/month
        cycle_life=5000,
        temperature_range=(-20.0, 70.0),  # °C
        cost=500.0,  # USD
        weight=2.5,  # kg
        energy_density=120.0,  # Wh/kg
        power_density=800.0  # W/kg
    )
    
    # Lead-acid battery (lower cost, proven technology)
    lead_acid_specs = BatterySpecs(
        chemistry=BatteryChemistry.LEAD_ACID,
        nominal_capacity=50.0,  # Ah
        nominal_voltage=12.0,  # V
        max_charge_current=5.0,  # A (0.1C)
        max_discharge_current=25.0,  # A (0.5C)
        charge_voltage=14.4,  # V
        cutoff_voltage=10.5,  # V
        internal_resistance=0.02,  # Ω
        self_discharge_rate=5.0,  # %/month
        cycle_life=500,
        temperature_range=(-20.0, 50.0),  # °C
        cost=150.0,  # USD
        weight=15.0,  # kg
        energy_density=35.0,  # Wh/kg
        power_density=150.0  # W/kg
    )
    
    # Small Li-ion for low-power applications
    small_li_ion_specs = BatterySpecs(
        chemistry=BatteryChemistry.LITHIUM_ION,
        nominal_capacity=2.5,  # Ah
        nominal_voltage=7.4,  # V (2S configuration)
        max_charge_current=1.25,  # A (0.5C)
        max_discharge_current=5.0,  # A (2C)
        charge_voltage=8.4,  # V
        cutoff_voltage=6.0,  # V
        internal_resistance=0.1,  # Ω
        self_discharge_rate=3.0,  # %/month
        cycle_life=1500,
        temperature_range=(-10.0, 60.0),  # °C
        cost=80.0,  # USD
        weight=0.3,  # kg
        energy_density=180.0,  # Wh/kg
        power_density=600.0  # W/kg
    )
    
    batteries = {
        'li_ion_pack': BatterySystem(li_ion_specs),
        'lifepo4_pack': BatterySystem(lifepo4_specs),
        'lead_acid': BatterySystem(lead_acid_specs),
        'small_li_ion': BatterySystem(small_li_ion_specs)
    }
    
    return batteries


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create standard battery systems
    batteries = create_standard_battery_systems()
    
    # Test LiFePO4 battery
    lifepo4 = batteries['lifepo4_pack']
    lifepo4.set_state_of_charge(30.0)  # Start at 30% SOC
    
    print("Initial battery state:")
    state = lifepo4.get_state_summary()
    for key, value in state.items():
        print(f"  {key}: {value:.2f}")
    
    # Simulate charging cycle
    print("\nCharging cycle simulation:")
    measurements = []
    
    # Charge at 5A for 2 hours
    for hour in range(20):  # 2 hours in 6-minute steps
        dt = 0.1  # 6 minutes
        measurement = lifepo4.charge(current=5.0, dt=dt, temperature=25.0)
        measurements.append(measurement)
        
        if hour % 5 == 0:
            print(f"  Hour {hour*dt:.1f}: SOC={measurement.state_of_charge:.1f}%, "
                  f"V={measurement.voltage:.2f}V, I={measurement.current:.1f}A, "
                  f"P={measurement.power:.1f}W")
        
        # Stop when nearly full
        if measurement.state_of_charge >= 95.0:
            break
    
    # Discharge cycle
    print("\nDischarge cycle simulation:")
    
    # Discharge at 10A until cutoff
    hour_counter = 0
    while lifepo4.state_of_charge > 20.0:
        dt = 0.1  # 6 minutes
        measurement = lifepo4.discharge(current=10.0, dt=dt, temperature=25.0)
        measurements.append(measurement)
        
        if hour_counter % 5 == 0:
            print(f"  Hour {hour_counter*dt:.1f}: SOC={measurement.state_of_charge:.1f}%, "
                  f"V={measurement.voltage:.2f}V, I={measurement.current:.1f}A, "
                  f"P={measurement.power:.1f}W")
        
        hour_counter += 1
        
        # Safety limit
        if hour_counter > 100:
            break
    
    # Final state
    print("\nFinal battery state:")
    final_state = lifepo4.get_state_summary()
    for key, value in final_state.items():
        print(f"  {key}: {value:.2f}")
    
    # Cost analysis
    cost_analysis = lifepo4.get_cost_analysis()
    print("\nCost Analysis:")
    print(f"Initial cost: ${cost_analysis['initial_cost']:.2f}")
    print(f"Total lifetime cost: ${cost_analysis['lifetime_cost']:.2f}")
    print(f"Operating cost per hour: ${cost_analysis['total_cost_per_hour']:.4f}")
    print(f"Expected lifetime: {cost_analysis['expected_lifetime_hours']:.0f} hours")
    
    # Test all battery types
    print("\nComparison of all battery types:")
    for name, battery in batteries.items():
        state = battery.get_state_summary()
        cost = battery.get_cost_analysis()
        specs = battery.specs
        
        print(f"\n{name.upper()}:")
        print(f"  Capacity: {specs.nominal_capacity:.1f}Ah, {specs.nominal_voltage:.1f}V")
        print(f"  Energy density: {specs.energy_density:.0f}Wh/kg")
        print(f"  Cycle life: {specs.cycle_life} cycles")
        print(f"  Cost: ${specs.cost:.0f} (${cost['total_cost_per_hour']:.4f}/hour)")
        print(f"  Weight: {specs.weight:.1f}kg")