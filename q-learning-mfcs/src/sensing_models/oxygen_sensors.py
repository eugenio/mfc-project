"""
Dissolved Oxygen Sensors for MFC Monitoring

This module implements various dissolved oxygen measurement technologies including
Clark electrodes, optical sensors, galvanic sensors, and paramagnetic sensors
for aerobic/anaerobic condition monitoring in MFC systems.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OxygenSensorType(Enum):
    """Types of oxygen sensors"""
    CLARK_ELECTRODE = "clark_electrode"
    OPTICAL_LUMINESCENCE = "optical_luminescence"
    GALVANIC = "galvanic"
    PARAMAGNETIC = "paramagnetic"
    ELECTROCHEMICAL = "electrochemical"
    FLUORESCENCE = "fluorescence"


@dataclass
class OxygenSensorSpecs:
    """Specifications for oxygen sensors"""
    sensor_type: OxygenSensorType
    measurement_range: Tuple[float, float]  # min, max DO concentration (mg/L)
    accuracy: float  # ±% of reading
    precision: float  # % of reading
    response_time: float  # seconds (90% response)
    power_consumption: float  # watts
    cost: float  # USD
    lifetime: float  # hours
    calibration_interval: float  # hours
    temperature_coefficient: float  # %/°C
    pressure_sensitivity: float  # %/bar
    salinity_sensitivity: float  # %/(g/L)
    fouling_sensitivity: float  # relative (0-1)


@dataclass
class OxygenMeasurement:
    """Single dissolved oxygen measurement"""
    timestamp: float
    do_concentration: float  # mg/L
    do_saturation: float  # % saturation
    temperature: float  # °C
    pressure: float  # bar
    salinity: float  # g/L
    sensor_type: OxygenSensorType
    uncertainty: float  # mg/L
    quality_flag: str  # 'good', 'warning', 'bad'
    raw_signal: float
    membrane_condition: float  # 0-1 (for membrane-based sensors)


class HenryLawModel:
    """Model dissolved oxygen equilibrium using Henry's Law"""
    
    def __init__(self):
        # Henry's law constants for O2 in water
        # H = c_aq / p_gas (mol/(L·atm))
        self.henry_constant_298k = 1.3e-3  # mol/(L·atm) at 25°C
        self.temperature_dependence = 1500.0  # K (van't Hoff equation)
        
        # Atmospheric O2 partial pressure
        self.atmospheric_o2_fraction = 0.21  # 21% O2 in air
        self.standard_pressure = 1.013  # bar
        
    def calculate_saturation_concentration(self, temperature: float, 
                                         pressure: float = 1.013, 
                                         salinity: float = 0.0) -> float:
        """
        Calculate O2 saturation concentration using Henry's Law
        
        Args:
            temperature: Water temperature (°C)
            pressure: Total pressure (bar)
            salinity: Water salinity (g/L)
            
        Returns:
            Saturation DO concentration (mg/L)
        """
        temp_k = temperature + 273.15
        
        # Temperature-corrected Henry's constant (for gas solubility, use van't Hoff equation)
        # Henry's constant increases with temperature, but solubility decreases
        henry_t = self.henry_constant_298k * np.exp(
            self.temperature_dependence * (1/temp_k - 1/298.15)
        )
        
        # O2 partial pressure
        p_o2 = self.atmospheric_o2_fraction * pressure  # bar
        p_o2_atm = p_o2 / 1.013  # Convert to atm
        
        # Dissolved O2 concentration (mol/L)
        c_o2_mol = henry_t * p_o2_atm
        
        # Convert to mg/L (MW of O2 = 32 g/mol)
        c_o2_mg_l = c_o2_mol * 32 * 1000
        
        # Salinity correction (Weiss equation approximation)
        salinity_factor = 1.0 - 0.00624 * salinity
        
        return c_o2_mg_l * salinity_factor
    
    def calculate_saturation_percentage(self, do_concentration: float, 
                                      temperature: float, 
                                      pressure: float = 1.013, 
                                      salinity: float = 0.0) -> float:
        """Calculate percent saturation"""
        saturation_conc = self.calculate_saturation_concentration(temperature, pressure, salinity)
        return (do_concentration / saturation_conc) * 100.0 if saturation_conc > 0 else 0.0


class OxygenSensor:
    """Base class for dissolved oxygen sensors"""
    
    def __init__(self, specs: OxygenSensorSpecs):
        self.specs = specs
        self.calibration_slope = 1.0
        self.calibration_offset = 0.0
        self.last_calibration = 0.0
        self.operating_hours = 0.0
        self.membrane_condition = 1.0  # For membrane-based sensors
        self.drift_rate = 0.5  # %/1000h
        
        # Initialize Henry's Law model
        self.henry_model = HenryLawModel()
        
        # Sensor-specific parameters
        self._initialize_sensor_parameters()
        
    def _initialize_sensor_parameters(self):
        """Initialize sensor-specific parameters"""
        if self.specs.sensor_type == OxygenSensorType.CLARK_ELECTRODE:
            # Clark electrode parameters
            self.cathode_potential = -0.8  # V vs Ag/AgCl
            self.membrane_permeability = 1e-8  # cm²/s
            self.membrane_thickness = 25e-4  # cm (25 μm)
            self.electrode_area = 0.1  # cm²
            self.electrolyte_concentration = 0.1  # M KCl
            
        elif self.specs.sensor_type == OxygenSensorType.OPTICAL_LUMINESCENCE:
            # Optical sensor parameters
            self.luminophore_type = "ruthenium_complex"
            self.excitation_wavelength = 470  # nm
            self.emission_wavelength = 600  # nm
            self.stern_volmer_constant = 0.002  # (mg/L)⁻¹
            self.lifetime_zero_oxygen = 5.0  # μs
            
        elif self.specs.sensor_type == OxygenSensorType.GALVANIC:
            # Galvanic sensor parameters
            self.anode_material = "lead"
            self.cathode_material = "gold"
            self.electrolyte_type = "KOH"
            self.current_density_factor = 2.0  # nA per mg/L O2
            
        elif self.specs.sensor_type == OxygenSensorType.PARAMAGNETIC:
            # Paramagnetic sensor parameters
            self.magnetic_susceptibility_o2 = 1.335e-6  # cgs units
            self.magnetic_field_strength = 0.1  # T
            self.flow_rate = 100  # mL/min
            
        elif self.specs.sensor_type == OxygenSensorType.ELECTROCHEMICAL:
            # General electrochemical sensor
            self.working_potential = -0.7  # V
            self.sensitivity = 100  # nA per mg/L
            self.background_current = 5  # nA
            
        elif self.specs.sensor_type == OxygenSensorType.FLUORESCENCE:
            # Fluorescence quenching sensor
            self.fluorophore_type = "pyrene"
            self.quenching_constant = 0.1  # (mg/L)⁻¹
            self.fluorescence_intensity_zero = 1000  # arbitrary units
    
    def measure(self, do_concentration: float, temperature: float = 25.0, 
                pressure: float = 1.013, salinity: float = 0.0, time: float = 0.0) -> OxygenMeasurement:
        """
        Simulate a dissolved oxygen measurement
        
        Args:
            do_concentration: True DO concentration (mg/L)
            temperature: Water temperature (°C)
            pressure: Pressure (bar)
            salinity: Water salinity (g/L)
            time: Current time (hours)
            
        Returns:
            OxygenMeasurement object
        """
        # Update operating hours
        self.operating_hours = time
        
        # Update membrane condition (for membrane-based sensors)
        self._update_membrane_condition(time)
        
        # Calculate drift
        drift_factor = 1.0 + (self.drift_rate * self.operating_hours / 100000.0)
        
        # Temperature correction
        temp_correction = 1.0 + self.specs.temperature_coefficient * (temperature - 25.0) / 100.0
        
        # Pressure correction
        pressure_correction = 1.0 + self.specs.pressure_sensitivity * (pressure - 1.013) / 100.0
        
        # Salinity correction
        salinity_correction = 1.0 - self.specs.salinity_sensitivity * salinity / 100.0
        
        # Calculate raw signal based on sensor type
        raw_signal = self._calculate_raw_signal(do_concentration, temperature, pressure)
        
        # Apply corrections and membrane condition
        corrected_signal = (raw_signal * temp_correction * pressure_correction * 
                          salinity_correction * drift_factor * self.membrane_condition)
        
        # Add noise
        noise_level = self.specs.precision / 100.0
        noise = np.random.normal(0, noise_level * corrected_signal)
        measured_signal = corrected_signal + noise
        
        # Convert to DO concentration
        measured_do = self._signal_to_concentration(measured_signal, temperature, pressure)
        
        # Calculate percent saturation
        do_saturation = self.henry_model.calculate_saturation_percentage(
            measured_do, temperature, pressure, salinity
        )
        
        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(measured_do, temperature, time)
        
        # Determine quality flag
        quality_flag = self._assess_quality(measured_do, uncertainty, time)
        
        return OxygenMeasurement(
            timestamp=time,
            do_concentration=measured_do,
            do_saturation=do_saturation,
            temperature=temperature,
            pressure=pressure,
            salinity=salinity,
            sensor_type=self.specs.sensor_type,
            uncertainty=uncertainty,
            quality_flag=quality_flag,
            raw_signal=measured_signal,
            membrane_condition=self.membrane_condition
        )
    
    def _calculate_raw_signal(self, do_concentration: float, temperature: float, pressure: float) -> float:
        """Calculate raw sensor signal based on DO concentration"""
        if self.specs.sensor_type == OxygenSensorType.CLARK_ELECTRODE:
            # Current proportional to O2 flux through membrane
            # i = n*F*A*P*c/d where n=electrons, F=Faraday, A=area, P=permeability, c=concentration, d=thickness
            faraday_constant = 96485  # C/mol
            n_electrons = 4  # O2 + 4H+ + 4e- → 2H2O
            flux = (self.membrane_permeability * do_concentration / 32000) / self.membrane_thickness  # mol/(cm²·s)
            current = n_electrons * faraday_constant * self.electrode_area * flux
            return current * 1e9  # Convert to nA
            
        elif self.specs.sensor_type == OxygenSensorType.OPTICAL_LUMINESCENCE:
            # Stern-Volmer quenching: I0/I = 1 + Ksv*[O2]
            quenching_factor = 1 + self.stern_volmer_constant * do_concentration
            intensity = 1000.0 / quenching_factor  # arbitrary units
            return intensity
            
        elif self.specs.sensor_type == OxygenSensorType.GALVANIC:
            # Current directly proportional to O2 concentration
            current = self.current_density_factor * do_concentration
            return current  # nA
            
        elif self.specs.sensor_type == OxygenSensorType.PARAMAGNETIC:
            # Magnetic force proportional to O2 concentration and magnetic susceptibility
            # Simplified model: signal ∝ χ*B²*c
            signal = (self.magnetic_susceptibility_o2 * self.magnetic_field_strength**2 * 
                     do_concentration * 1000)  # arbitrary units
            return signal
            
        elif self.specs.sensor_type == OxygenSensorType.ELECTROCHEMICAL:
            # Linear electrochemical response
            current = self.sensitivity * do_concentration + self.background_current
            return current  # nA
            
        elif self.specs.sensor_type == OxygenSensorType.FLUORESCENCE:
            # Fluorescence quenching
            quenched_intensity = (self.fluorescence_intensity_zero / 
                                (1 + self.quenching_constant * do_concentration))
            return quenched_intensity
            
        else:
            # Linear response for generic sensor
            return do_concentration * 100.0  # arbitrary signal units
    
    def _signal_to_concentration(self, signal: float, temperature: float, pressure: float) -> float:
        """Convert sensor signal to DO concentration"""
        if self.specs.sensor_type == OxygenSensorType.CLARK_ELECTRODE:
            # Reverse Clark electrode calculation
            faraday_constant = 96485
            n_electrons = 4
            current_a = signal * 1e-9  # Convert from nA
            flux = current_a / (n_electrons * faraday_constant * self.electrode_area)
            concentration_mol_cm3 = flux * self.membrane_thickness / self.membrane_permeability
            do_concentration = concentration_mol_cm3 * 32000  # Convert to mg/L
            
        elif self.specs.sensor_type == OxygenSensorType.OPTICAL_LUMINESCENCE:
            # Reverse Stern-Volmer equation
            i0_over_i = 1000.0 / signal
            do_concentration = (i0_over_i - 1) / self.stern_volmer_constant
            
        elif self.specs.sensor_type == OxygenSensorType.GALVANIC:
            # Reverse galvanic calculation
            do_concentration = signal / self.current_density_factor
            
        elif self.specs.sensor_type == OxygenSensorType.PARAMAGNETIC:
            # Reverse paramagnetic calculation
            do_concentration = signal / (self.magnetic_susceptibility_o2 * 
                                       self.magnetic_field_strength**2 * 1000)
            
        elif self.specs.sensor_type == OxygenSensorType.ELECTROCHEMICAL:
            # Reverse electrochemical calculation
            do_concentration = (signal - self.background_current) / self.sensitivity
            
        elif self.specs.sensor_type == OxygenSensorType.FLUORESCENCE:
            # Reverse fluorescence quenching
            do_concentration = ((self.fluorescence_intensity_zero / signal) - 1) / self.quenching_constant
            
        else:
            do_concentration = signal / 100.0
        
        # Apply calibration
        do_concentration = do_concentration * self.calibration_slope + self.calibration_offset
        
        # Ensure within measurement range and physical limits
        do_concentration = np.clip(do_concentration, 0.0, self.specs.measurement_range[1])
        
        return do_concentration
    
    def _update_membrane_condition(self, time: float):
        """Update membrane condition for membrane-based sensors"""
        if self.specs.sensor_type in [OxygenSensorType.CLARK_ELECTRODE, 
                                    OxygenSensorType.GALVANIC]:
            # Membrane degradation over time
            degradation_rate = 0.0001  # per hour
            fouling_rate = self.specs.fouling_sensitivity * 0.0005  # per hour
            
            total_degradation = (degradation_rate + fouling_rate) * self.operating_hours
            self.membrane_condition = max(0.3, 1.0 - total_degradation)  # Minimum 30% condition
        else:
            # Non-membrane sensors have minimal degradation
            self.membrane_condition = 1.0 - 0.00001 * self.operating_hours
            self.membrane_condition = max(0.8, self.membrane_condition)
    
    def _calculate_uncertainty(self, do_concentration: float, temperature: float, time: float) -> float:
        """Calculate measurement uncertainty"""
        # Base uncertainty from accuracy specification
        base_uncertainty = do_concentration * self.specs.accuracy / 100.0
        
        # Temperature uncertainty
        temp_uncertainty = abs(do_concentration * self.specs.temperature_coefficient * 
                             (temperature - 25.0) / 100.0)
        
        # Membrane degradation uncertainty
        membrane_uncertainty = do_concentration * (1.0 - self.membrane_condition) * 0.5
        
        # Calibration uncertainty increases with time since last calibration
        time_since_cal = time - self.last_calibration
        cal_uncertainty = do_concentration * 0.001 * time_since_cal  # 0.1% per 100 hours
        
        # Total uncertainty (root sum of squares)
        total_uncertainty = np.sqrt(base_uncertainty**2 + temp_uncertainty**2 + 
                                  membrane_uncertainty**2 + cal_uncertainty**2)
        
        return total_uncertainty
    
    def _assess_quality(self, do_concentration: float, uncertainty: float, time: float) -> str:
        """Assess measurement quality"""
        # Check if within range
        if do_concentration < 0 or do_concentration > self.specs.measurement_range[1]:
            return "bad"
        
        # Check relative uncertainty
        relative_uncertainty = uncertainty / do_concentration if do_concentration > 0 else float('inf')
        if relative_uncertainty > 0.3:  # 30%
            return "bad"
        elif relative_uncertainty > 0.15:  # 15%
            return "warning"
        
        # Check membrane condition
        if self.membrane_condition < 0.5:  # 50% degradation
            return "bad"
        elif self.membrane_condition < 0.7:  # 30% degradation
            return "warning"
        
        # Check calibration status
        time_since_cal = time - self.last_calibration
        if time_since_cal > self.specs.calibration_interval:
            return "warning"
        
        return "good"
    
    def calibrate(self, reference_concentrations: List[float], 
                  measured_values: List[float], time: float,
                  calibration_type: str = "two_point") -> bool:
        """Calibrate the sensor"""
        if len(reference_concentrations) != len(measured_values):
            logger.error("Reference concentrations and measured values must have same length")
            return False
        
        if calibration_type == "two_point" and len(reference_concentrations) != 2:
            logger.error("Two-point calibration requires exactly 2 points")
            return False
        
        if calibration_type == "two_point":
            # Two-point calibration (typically 0% and 100% saturation)
            ref_zero, ref_span = reference_concentrations
            meas_zero, meas_span = measured_values
            
            if meas_span != meas_zero:
                self.calibration_slope = (ref_span - ref_zero) / (meas_span - meas_zero)
                self.calibration_offset = ref_zero - self.calibration_slope * meas_zero
            else:
                logger.error("Measured span and zero values are identical")
                return False
        else:
            # Multi-point linear regression
            if len(reference_concentrations) < 2:
                logger.error("At least 2 calibration points required")
                return False
            
            slope, intercept = np.polyfit(measured_values, reference_concentrations, 1)
            self.calibration_slope = slope
            self.calibration_offset = intercept
        
        self.last_calibration = time
        logger.info(f"DO sensor calibrated: slope={self.calibration_slope:.4f}, "
                   f"offset={self.calibration_offset:.4f}")
        return True
    
    def replace_membrane(self):
        """Replace sensor membrane (for membrane-based sensors)"""
        if self.specs.sensor_type in [OxygenSensorType.CLARK_ELECTRODE, 
                                    OxygenSensorType.GALVANIC]:
            self.membrane_condition = 1.0
            logger.info("Sensor membrane replaced")
            return True
        else:
            logger.warning("This sensor type does not have a replaceable membrane")
            return False
    
    def get_power_consumption(self) -> float:
        """Get current power consumption"""
        base_power = self.specs.power_consumption
        
        # Additional power varies by sensor type
        if self.specs.sensor_type == OxygenSensorType.OPTICAL_LUMINESCENCE:
            # LED excitation and photodetector
            optical_power = 0.2  # W
        elif self.specs.sensor_type == OxygenSensorType.PARAMAGNETIC:
            # Magnet and mechanical components
            magnetic_power = 1.0  # W
        elif self.specs.sensor_type in [OxygenSensorType.CLARK_ELECTRODE, 
                                      OxygenSensorType.GALVANIC, 
                                      OxygenSensorType.ELECTROCHEMICAL]:
            # Potentiostat and signal conditioning
            electronic_power = 0.1  # W
        else:
            electronic_power = 0.05  # W
        
        additional_power = locals().get('optical_power', 0) + locals().get('magnetic_power', 0) + locals().get('electronic_power', 0)
        
        return base_power + additional_power
    
    def get_cost_analysis(self) -> Dict[str, float]:
        """Get comprehensive cost analysis"""
        initial_cost = self.specs.cost
        
        # Operating cost per hour
        power_cost_per_hour = self.get_power_consumption() * 0.15 / 1000  # $0.15/kWh
        
        # Maintenance cost varies by sensor type
        if self.specs.sensor_type in [OxygenSensorType.CLARK_ELECTRODE, 
                                    OxygenSensorType.GALVANIC]:
            # Membrane replacement every 6-12 months
            maintenance_cost_per_hour = 0.2  # includes membrane replacement
        elif self.specs.sensor_type == OxygenSensorType.OPTICAL_LUMINESCENCE:
            # Optical component cleaning
            maintenance_cost_per_hour = 0.05
        elif self.specs.sensor_type == OxygenSensorType.PARAMAGNETIC:
            # Mechanical maintenance
            maintenance_cost_per_hour = 0.15
        else:
            maintenance_cost_per_hour = 0.1
        
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


def create_standard_oxygen_sensors() -> Dict[str, OxygenSensor]:
    """Create standard oxygen sensor configurations"""
    
    # Clark electrode
    clark_specs = OxygenSensorSpecs(
        sensor_type=OxygenSensorType.CLARK_ELECTRODE,
        measurement_range=(0.0, 20.0),  # mg/L
        accuracy=2.0,  # ±2%
        precision=1.0,  # 1%
        response_time=30.0,  # seconds
        power_consumption=0.5,  # watts
        cost=800.0,  # USD
        lifetime=8760.0,  # 1 year
        calibration_interval=168.0,  # 1 week
        temperature_coefficient=4.0,  # %/°C
        pressure_sensitivity=4.0,  # %/bar
        salinity_sensitivity=2.0,  # %/(g/L)
        fouling_sensitivity=0.7  # high fouling sensitivity
    )
    
    # Optical luminescence sensor
    optical_specs = OxygenSensorSpecs(
        sensor_type=OxygenSensorType.OPTICAL_LUMINESCENCE,
        measurement_range=(0.0, 50.0),  # mg/L
        accuracy=1.0,  # ±1%
        precision=0.5,  # 0.5%
        response_time=10.0,  # seconds
        power_consumption=1.0,  # watts
        cost=2500.0,  # USD
        lifetime=17520.0,  # 2 years
        calibration_interval=720.0,  # 30 days
        temperature_coefficient=1.0,  # %/°C
        pressure_sensitivity=1.0,  # %/bar
        salinity_sensitivity=0.5,  # %/(g/L)
        fouling_sensitivity=0.2  # low fouling sensitivity
    )
    
    # Galvanic sensor
    galvanic_specs = OxygenSensorSpecs(
        sensor_type=OxygenSensorType.GALVANIC,
        measurement_range=(0.0, 15.0),  # mg/L
        accuracy=3.0,  # ±3%
        precision=1.5,  # 1.5%
        response_time=20.0,  # seconds
        power_consumption=0.01,  # watts (self-powered)
        cost=200.0,  # USD
        lifetime=4380.0,  # 6 months
        calibration_interval=336.0,  # 2 weeks
        temperature_coefficient=3.0,  # %/°C
        pressure_sensitivity=3.0,  # %/bar
        salinity_sensitivity=1.5,  # %/(g/L)
        fouling_sensitivity=0.8  # high fouling sensitivity
    )
    
    # Paramagnetic sensor
    paramagnetic_specs = OxygenSensorSpecs(
        sensor_type=OxygenSensorType.PARAMAGNETIC,
        measurement_range=(0.0, 100.0),  # mg/L (gas phase capable)
        accuracy=1.5,  # ±1.5%
        precision=0.8,  # 0.8%
        response_time=5.0,  # seconds
        power_consumption=5.0,  # watts
        cost=10000.0,  # USD
        lifetime=43800.0,  # 5 years
        calibration_interval=2160.0,  # 90 days
        temperature_coefficient=0.5,  # %/°C
        pressure_sensitivity=0.1,  # %/bar
        salinity_sensitivity=0.0,  # not affected by salinity
        fouling_sensitivity=0.1  # very low fouling sensitivity
    )
    
    sensors = {
        'clark_electrode': OxygenSensor(clark_specs),
        'optical_luminescence': OxygenSensor(optical_specs),
        'galvanic': OxygenSensor(galvanic_specs),
        'paramagnetic': OxygenSensor(paramagnetic_specs)
    }
    
    return sensors


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create standard sensors
    sensors = create_standard_oxygen_sensors()
    
    # Test optical sensor
    optical_sensor = sensors['optical_luminescence']
    
    # Test Henry's Law model
    henry_model = HenryLawModel()
    
    # Calculate saturation at different temperatures
    temperatures = [10, 15, 20, 25, 30, 35]  # °C
    for temp in temperatures:
        sat_conc = henry_model.calculate_saturation_concentration(temp)
        print(f"DO saturation at {temp}°C: {sat_conc:.2f} mg/L")
    
    # Simulate measurements with varying DO levels
    do_levels = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]  # mg/L
    temperature = 25.0
    
    measurements = []
    for i, do_level in enumerate(do_levels):
        measurement = optical_sensor.measure(do_level, temperature=temperature, time=i*10)
        measurements.append(measurement)
        
        print(f"DO: {do_level:.1f} mg/L → Measured: {measurement.do_concentration:.2f} ± "
              f"{measurement.uncertainty:.3f} mg/L ({measurement.do_saturation:.1f}% sat)")
    
    # Two-point calibration (0% and 100% saturation)
    sat_100_conc = henry_model.calculate_saturation_concentration(25.0)
    reference_concs = [0.0, sat_100_conc]
    measured_vals = [0.05, sat_100_conc * 0.98]  # Simulate slight measurement error
    
    success = optical_sensor.calibrate(reference_concs, measured_vals, time=100.0, 
                                     calibration_type="two_point")
    if success:
        print(f"Calibration successful: slope={optical_sensor.calibration_slope:.4f}")
    
    # Cost analysis
    cost_analysis = optical_sensor.get_cost_analysis()
    print(f"Total lifetime cost: ${cost_analysis['lifetime_cost']:.2f}")
    print(f"Operating cost per hour: ${cost_analysis['total_cost_per_hour']:.4f}")
    
    # Test membrane degradation (Clark electrode)
    clark_sensor = sensors['clark_electrode']
    print(f"Initial membrane condition: {clark_sensor.membrane_condition:.3f}")
    
    # Simulate 6 months of operation
    for month in range(6):
        time_hours = month * 720  # 720 hours per month
        measurement = clark_sensor.measure(5.0, temperature=25.0, time=time_hours)
        print(f"Month {month+1}: Membrane condition {clark_sensor.membrane_condition:.3f}, "
              f"DO: {measurement.do_concentration:.2f} mg/L")