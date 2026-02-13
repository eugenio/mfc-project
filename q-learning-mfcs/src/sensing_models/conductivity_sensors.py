"""
Conductivity Sensors for Electrolyte Monitoring in MFC Systems

This module implements various conductivity measurement technologies including
2-electrode, 4-electrode, inductive, and toroidal conductivity sensors for
electrolyte concentration and ionic strength monitoring.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConductivitySensorType(Enum):
    """Types of conductivity sensors"""
    TWO_ELECTRODE = "two_electrode"
    FOUR_ELECTRODE = "four_electrode"
    INDUCTIVE = "inductive"
    TOROIDAL = "toroidal"
    CAPACITIVE = "capacitive"


@dataclass
class ConductivitySensorSpecs:
    """Specifications for conductivity sensors"""
    sensor_type: ConductivitySensorType
    measurement_range: Tuple[float, float]  # min, max conductivity (mS/cm)
    accuracy: float  # ±% of reading
    precision: float  # % of reading
    response_time: float  # seconds
    power_consumption: float  # watts
    cost: float  # USD
    lifetime: float  # hours
    calibration_interval: float  # hours
    cell_constant: float  # cm⁻¹
    temperature_coefficient: float  # %/°C
    pressure_sensitivity: float  # %/bar
    fouling_sensitivity: float  # relative (0-1)


@dataclass
class ConductivityMeasurement:
    """Single conductivity measurement"""
    timestamp: float
    conductivity: float  # mS/cm
    temperature: float  # °C
    sensor_type: ConductivitySensorType
    uncertainty: float  # mS/cm
    quality_flag: str  # 'good', 'warning', 'bad'
    raw_signal: float
    cell_constant: float
    compensation_applied: bool
    fouling_factor: float


class ElectrolyteModel:
    """Model electrolyte properties and conductivity relationships"""
    
    def __init__(self):
        # Common electrolyte properties for MFC applications
        self.ion_mobilities = {
            'Na+': 5.19e-4,  # m²/(V·s)
            'K+': 7.62e-4,
            'Mg2+': 5.50e-4,
            'Ca2+': 6.17e-4,
            'Cl-': 7.91e-4,
            'SO42-': 8.29e-4,
            'PO43-': 6.94e-4,
            'acetate-': 4.24e-4,
            'lactate-': 3.86e-4
        }
        
        self.faraday_constant = 96485.3  # C/mol
        
    def calculate_theoretical_conductivity(self, ion_concentrations: Dict[str, float], 
                                         temperature: float = 25.0) -> float:
        """
        Calculate theoretical conductivity from ion concentrations
        
        Args:
            ion_concentrations: Dict of ion name -> concentration (mol/L)
            temperature: Temperature (°C)
            
        Returns:
            Conductivity (mS/cm)
        """
        # Temperature correction for ion mobility (approximately +2%/°C)
        temp_factor = 1.0 + 0.02 * (temperature - 25.0)
        
        total_conductivity = 0.0
        
        for ion, concentration in ion_concentrations.items():
            if ion in self.ion_mobilities:
                # Determine charge from ion name
                if '2+' in ion or '2-' in ion:
                    charge = 2
                elif '3+' in ion or '3-' in ion:
                    charge = 3
                else:
                    charge = 1
                
                # Conductivity contribution (S/m)
                lambda_i = self.ion_mobilities[ion] * self.faraday_constant * charge * temp_factor
                conductivity_contribution = lambda_i * concentration * 1000  # mol/L to mol/m³
                total_conductivity += conductivity_contribution
        
        # Convert S/m to mS/cm
        return total_conductivity * 0.1
    
    def estimate_ionic_strength(self, conductivity: float, temperature: float = 25.0) -> float:
        """
        Estimate ionic strength from conductivity (empirical relationship)
        
        Args:
            conductivity: Conductivity (mS/cm)
            temperature: Temperature (°C)
            
        Returns:
            Ionic strength (mol/L)
        """
        # Empirical relationship for typical MFC electrolytes
        # I ≈ 1.6 × 10⁻⁵ × κ¹·⁴ (where κ is in μS/cm)
        conductivity_us_cm = conductivity * 1000  # Convert to μS/cm
        ionic_strength = 1.6e-5 * (conductivity_us_cm ** 1.4)
        
        # Temperature correction
        temp_factor = 1.0 - 0.02 * (temperature - 25.0)
        
        return ionic_strength * temp_factor


class ConductivitySensor:
    """Base class for conductivity sensors"""
    
    def __init__(self, specs: ConductivitySensorSpecs):
        self.specs = specs
        self.calibration_factor = 1.0
        self.calibration_offset = 0.0
        self.last_calibration = 0.0
        self.operating_hours = 0.0
        self.fouling_factor = 1.0
        self.drift_rate = 0.1  # %/1000h
        
        # Initialize electrolyte model
        self.electrolyte_model = ElectrolyteModel()
        
        # Sensor-specific parameters
        self._initialize_sensor_parameters()
        
    def _initialize_sensor_parameters(self):
        """Initialize sensor-specific parameters"""
        if self.specs.sensor_type == ConductivitySensorType.TWO_ELECTRODE:
            # Two-electrode sensor parameters
            self.electrode_area = 1.0  # cm²
            self.electrode_spacing = 1.0  # cm
            self.polarization_factor = 1.05  # accounts for electrode polarization
            self.frequency = 1000.0  # Hz (AC excitation frequency)
            
        elif self.specs.sensor_type == ConductivitySensorType.FOUR_ELECTRODE:
            # Four-electrode sensor parameters
            self.current_electrodes_spacing = 2.0  # cm
            self.voltage_electrodes_spacing = 1.0  # cm
            self.excitation_current = 1e-3  # A
            self.input_impedance = 1e12  # Ω (high impedance voltage measurement)
            
        elif self.specs.sensor_type == ConductivitySensorType.INDUCTIVE:
            # Inductive sensor parameters
            self.primary_coil_turns = 100
            self.secondary_coil_turns = 100
            self.coil_area = 5.0  # cm²
            self.excitation_frequency = 10000.0  # Hz
            self.core_permeability = 1000  # relative permeability
            
        elif self.specs.sensor_type == ConductivitySensorType.TOROIDAL:
            # Toroidal sensor parameters
            self.toroid_diameter = 2.0  # cm
            self.wire_turns = 200
            self.core_material = "ferrite"
            self.excitation_voltage = 5.0  # V
            
        elif self.specs.sensor_type == ConductivitySensorType.CAPACITIVE:
            # Capacitive sensor parameters
            self.electrode_area = 2.0  # cm²
            self.electrode_gap = 0.5  # cm
            self.dielectric_constant = 80  # water
            self.measurement_frequency = 100000.0  # Hz
    
    def measure(self, true_conductivity: float, temperature: float = 25.0, 
                pressure: float = 1.0, time: float = 0.0) -> ConductivityMeasurement:
        """
        Simulate a conductivity measurement
        
        Args:
            true_conductivity: Actual conductivity (mS/cm)
            temperature: Solution temperature (°C)
            pressure: Pressure (bar)
            time: Current time (hours)
            
        Returns:
            ConductivityMeasurement object
        """
        # Update operating hours
        self.operating_hours = time
        
        # Calculate fouling effects
        self._update_fouling_factor(time)
        
        # Calculate drift
        drift_factor = 1.0 + (self.drift_rate * self.operating_hours / 100000.0)
        
        # Temperature correction
        temp_correction = 1.0 + self.specs.temperature_coefficient * (temperature - 25.0) / 100.0
        
        # Pressure correction
        pressure_correction = 1.0 + self.specs.pressure_sensitivity * (pressure - 1.0) / 100.0
        
        # Calculate raw signal based on sensor type
        raw_signal = self._calculate_raw_signal(true_conductivity, temperature)
        
        # Apply corrections and fouling
        corrected_signal = (raw_signal * temp_correction * pressure_correction * 
                          drift_factor * self.fouling_factor)
        
        # Add noise
        noise_level = self.specs.precision / 100.0
        noise = np.random.normal(0, noise_level * corrected_signal)
        measured_signal = corrected_signal + noise
        
        # Convert to conductivity
        measured_conductivity = self._signal_to_conductivity(measured_signal, temperature)
        
        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(measured_conductivity, temperature, time)
        
        # Determine quality flag
        quality_flag = self._assess_quality(measured_conductivity, uncertainty, time)
        
        return ConductivityMeasurement(
            timestamp=time,
            conductivity=measured_conductivity,
            temperature=temperature,
            sensor_type=self.specs.sensor_type,
            uncertainty=uncertainty,
            quality_flag=quality_flag,
            raw_signal=measured_signal,
            cell_constant=self.specs.cell_constant,
            compensation_applied=True,
            fouling_factor=self.fouling_factor
        )
    
    def _calculate_raw_signal(self, conductivity: float, temperature: float) -> float:
        """Calculate raw sensor signal based on conductivity"""
        if self.specs.sensor_type == ConductivitySensorType.TWO_ELECTRODE:
            # Ohm's law: G = σ/K where G is conductance, σ is conductivity, K is cell constant
            conductance = conductivity / self.specs.cell_constant  # S
            # Account for polarization effects
            apparent_conductance = conductance / self.polarization_factor
            return apparent_conductance * 1000  # Convert to mS for signal
            
        elif self.specs.sensor_type == ConductivitySensorType.FOUR_ELECTRODE:
            # Four-electrode eliminates polarization: V = I * R, where R = K/σ
            resistance = self.specs.cell_constant / conductivity  # Ω
            voltage = self.excitation_current * resistance
            return voltage * 1000  # Convert to mV
            
        elif self.specs.sensor_type == ConductivitySensorType.INDUCTIVE:
            # Inductive coupling proportional to conductivity
            # Signal ∝ σ * f * μ * A where f is frequency, μ is permeability, A is area
            signal = (conductivity * self.excitation_frequency * self.core_permeability * 
                     self.coil_area * 1e-6)  # Scaling factor
            return signal
            
        elif self.specs.sensor_type == ConductivitySensorType.TOROIDAL:
            # Toroidal: induced current proportional to conductivity and applied voltage
            induced_current = (conductivity * self.excitation_voltage * 
                             self.wire_turns * np.pi * (self.toroid_diameter/2)**2 * 1e-6)
            return induced_current * 1000  # Convert to mA
            
        elif self.specs.sensor_type == ConductivitySensorType.CAPACITIVE:
            # Capacitive: C = ε₀ * εᵣ * A / d, but conductivity affects loss tangent
            capacitance_base = 8.854e-12 * self.dielectric_constant * self.electrode_area / self.electrode_gap
            loss_tangent = conductivity / (2 * np.pi * self.measurement_frequency * 
                                         8.854e-12 * self.dielectric_constant)
            signal = capacitance_base * loss_tangent * 1e12  # Convert to pF·tan(δ)
            return signal
            
        else:
            # Linear response for generic sensor
            return conductivity * self.specs.cell_constant
    
    def _signal_to_conductivity(self, signal: float, temperature: float) -> float:
        """Convert sensor signal to conductivity"""
        if self.specs.sensor_type == ConductivitySensorType.TWO_ELECTRODE:
            # Reverse two-electrode calculation
            apparent_conductance = signal / 1000  # Convert from mS
            conductance = apparent_conductance * self.polarization_factor
            conductivity = conductance * self.specs.cell_constant
            
        elif self.specs.sensor_type == ConductivitySensorType.FOUR_ELECTRODE:
            # Reverse four-electrode calculation
            voltage = signal / 1000  # Convert from mV
            resistance = voltage / self.excitation_current
            conductivity = self.specs.cell_constant / resistance
            
        elif self.specs.sensor_type == ConductivitySensorType.INDUCTIVE:
            # Reverse inductive calculation
            conductivity = signal / (self.excitation_frequency * self.core_permeability * 
                                   self.coil_area * 1e-6)
            
        elif self.specs.sensor_type == ConductivitySensorType.TOROIDAL:
            # Reverse toroidal calculation
            induced_current = signal / 1000  # Convert from mA
            conductivity = induced_current / (self.excitation_voltage * self.wire_turns * 
                                            np.pi * (self.toroid_diameter/2)**2 * 1e-6)
            
        elif self.specs.sensor_type == ConductivitySensorType.CAPACITIVE:
            # Reverse capacitive calculation
            capacitance_base = 8.854e-12 * self.dielectric_constant * self.electrode_area / self.electrode_gap
            loss_tangent = signal * 1e-12 / capacitance_base
            conductivity = (loss_tangent * 2 * np.pi * self.measurement_frequency * 
                          8.854e-12 * self.dielectric_constant)
            
        else:
            conductivity = signal / self.specs.cell_constant
        
        # Apply calibration
        conductivity = conductivity * self.calibration_factor + self.calibration_offset
        
        # Ensure within measurement range
        conductivity = np.clip(conductivity, self.specs.measurement_range[0], 
                             self.specs.measurement_range[1])
        
        return conductivity
    
    def _update_fouling_factor(self, time: float):
        """Update fouling factor based on sensor type and operating time"""
        # Fouling accumulation depends on sensor type
        if self.specs.sensor_type in [ConductivitySensorType.TWO_ELECTRODE, 
                                    ConductivitySensorType.FOUR_ELECTRODE]:
            # Electrode sensors are more susceptible to fouling
            fouling_rate = self.specs.fouling_sensitivity * 0.001  # per hour
            self.fouling_factor = 1.0 - fouling_rate * self.operating_hours
            
        elif self.specs.sensor_type in [ConductivitySensorType.INDUCTIVE, 
                                      ConductivitySensorType.TOROIDAL]:
            # Non-contact sensors have minimal fouling
            fouling_rate = self.specs.fouling_sensitivity * 0.0001  # per hour
            self.fouling_factor = 1.0 - fouling_rate * self.operating_hours
            
        else:
            # Capacitive sensors have moderate fouling
            fouling_rate = self.specs.fouling_sensitivity * 0.0005  # per hour
            self.fouling_factor = 1.0 - fouling_rate * self.operating_hours
        
        # Ensure fouling factor doesn't go below 0.5 (50% signal loss maximum)
        self.fouling_factor = max(0.5, self.fouling_factor)
    
    def _calculate_uncertainty(self, conductivity: float, temperature: float, time: float) -> float:
        """Calculate measurement uncertainty"""
        # Base uncertainty from accuracy specification
        base_uncertainty = conductivity * self.specs.accuracy / 100.0
        
        # Temperature uncertainty
        temp_uncertainty = abs(conductivity * self.specs.temperature_coefficient * 
                             (temperature - 25.0) / 100.0)
        
        # Fouling uncertainty
        fouling_uncertainty = conductivity * (1.0 - self.fouling_factor)
        
        # Calibration uncertainty increases with time since last calibration
        time_since_cal = time - self.last_calibration
        cal_uncertainty = conductivity * 0.001 * time_since_cal  # 0.1% per 100 hours
        
        # Total uncertainty (root sum of squares)
        total_uncertainty = np.sqrt(base_uncertainty**2 + temp_uncertainty**2 + 
                                  fouling_uncertainty**2 + cal_uncertainty**2)
        
        return total_uncertainty
    
    def _assess_quality(self, conductivity: float, uncertainty: float, time: float) -> str:
        """Assess measurement quality"""
        # Check if within range
        if conductivity <= self.specs.measurement_range[0] or conductivity >= self.specs.measurement_range[1]:
            return "bad"
        
        # Check relative uncertainty
        relative_uncertainty = uncertainty / conductivity if conductivity > 0 else float('inf')
        if relative_uncertainty > 0.2:  # 20%
            return "bad"
        elif relative_uncertainty > 0.1:  # 10%
            return "warning"
        
        # Check fouling level
        if self.fouling_factor < 0.7:  # 30% signal loss
            return "bad"
        elif self.fouling_factor < 0.85:  # 15% signal loss
            return "warning"
        
        # Check calibration status
        time_since_cal = time - self.last_calibration
        if time_since_cal > self.specs.calibration_interval:
            return "warning"
        
        return "good"
    
    def calibrate(self, standard_conductivities: List[float], 
                  measured_values: List[float], time: float) -> bool:
        """Calibrate the sensor with standard solutions"""
        if len(standard_conductivities) != len(measured_values):
            logger.error("Standard conductivities and measured values must have same length")
            return False
        
        if len(standard_conductivities) < 2:
            logger.error("At least 2 calibration points required")
            return False
        
        # Linear regression for calibration curve
        standards = np.array(standard_conductivities)
        measurements = np.array(measured_values)
        
        # Calculate calibration factors
        slope, intercept = np.polyfit(measurements, standards, 1)
        
        self.calibration_factor = slope
        self.calibration_offset = intercept
        self.last_calibration = time
        
        logger.info(f"Conductivity sensor calibrated: factor={slope:.4f}, offset={intercept:.4f}")
        return True
    
    def clean_sensor(self, cleaning_effectiveness: float = 0.8):
        """Simulate sensor cleaning"""
        if cleaning_effectiveness < 0 or cleaning_effectiveness > 1:
            logger.error("Cleaning effectiveness must be between 0 and 1")
            return False
        
        # Restore fouling factor based on cleaning effectiveness
        fouling_removed = (1.0 - self.fouling_factor) * cleaning_effectiveness
        self.fouling_factor = min(1.0, self.fouling_factor + fouling_removed)
        
        logger.info(f"Sensor cleaned: fouling factor restored to {self.fouling_factor:.3f}")
        return True
    
    def get_power_consumption(self) -> float:
        """Get current power consumption"""
        base_power = self.specs.power_consumption
        
        # Additional power for signal conditioning varies by sensor type
        if self.specs.sensor_type in [ConductivitySensorType.INDUCTIVE, 
                                    ConductivitySensorType.TOROIDAL]:
            # AC excitation requires more power
            signal_conditioning_power = 0.5  # W
        elif self.specs.sensor_type == ConductivitySensorType.CAPACITIVE:
            # High frequency measurement
            signal_conditioning_power = 0.3  # W
        else:
            # Electrode-based sensors
            signal_conditioning_power = 0.1  # W
        
        return base_power + signal_conditioning_power
    
    def get_cost_analysis(self) -> Dict[str, float]:
        """Get comprehensive cost analysis"""
        initial_cost = self.specs.cost
        
        # Operating cost per hour
        power_cost_per_hour = self.get_power_consumption() * 0.15 / 1000  # $0.15/kWh
        
        # Maintenance cost varies by sensor type
        if self.specs.sensor_type in [ConductivitySensorType.TWO_ELECTRODE, 
                                    ConductivitySensorType.FOUR_ELECTRODE]:
            # Electrode sensors require more frequent cleaning/replacement
            maintenance_cost_per_hour = 0.1
        elif self.specs.sensor_type in [ConductivitySensorType.INDUCTIVE, 
                                      ConductivitySensorType.TOROIDAL]:
            # Non-contact sensors have lower maintenance
            maintenance_cost_per_hour = 0.02
        else:
            maintenance_cost_per_hour = 0.05
        
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


def create_standard_conductivity_sensors() -> Dict[str, ConductivitySensor]:
    """Create standard conductivity sensor configurations"""
    
    # Two-electrode sensor
    two_electrode_specs = ConductivitySensorSpecs(
        sensor_type=ConductivitySensorType.TWO_ELECTRODE,
        measurement_range=(0.1, 200.0),  # mS/cm
        accuracy=2.0,  # ±2%
        precision=1.0,  # 1%
        response_time=2.0,  # seconds
        power_consumption=0.5,  # watts
        cost=500.0,  # USD
        lifetime=8760.0,  # 1 year
        calibration_interval=720.0,  # 30 days
        cell_constant=1.0,  # cm⁻¹
        temperature_coefficient=2.0,  # %/°C
        pressure_sensitivity=0.1,  # %/bar
        fouling_sensitivity=0.8  # high fouling sensitivity
    )
    
    # Four-electrode sensor
    four_electrode_specs = ConductivitySensorSpecs(
        sensor_type=ConductivitySensorType.FOUR_ELECTRODE,
        measurement_range=(0.01, 1000.0),  # mS/cm
        accuracy=1.0,  # ±1%
        precision=0.5,  # 0.5%
        response_time=1.0,  # seconds
        power_consumption=1.0,  # watts
        cost=1500.0,  # USD
        lifetime=17520.0,  # 2 years
        calibration_interval=2160.0,  # 90 days
        cell_constant=1.0,  # cm⁻¹
        temperature_coefficient=2.0,  # %/°C
        pressure_sensitivity=0.1,  # %/bar
        fouling_sensitivity=0.6  # moderate fouling sensitivity
    )
    
    # Inductive sensor
    inductive_specs = ConductivitySensorSpecs(
        sensor_type=ConductivitySensorType.INDUCTIVE,
        measurement_range=(1.0, 2000.0),  # mS/cm
        accuracy=1.5,  # ±1.5%
        precision=0.8,  # 0.8%
        response_time=3.0,  # seconds
        power_consumption=2.0,  # watts
        cost=3000.0,  # USD
        lifetime=43800.0,  # 5 years
        calibration_interval=4380.0,  # 6 months
        cell_constant=0.1,  # cm⁻¹ (effective)
        temperature_coefficient=1.5,  # %/°C
        pressure_sensitivity=0.05,  # %/bar
        fouling_sensitivity=0.2  # low fouling sensitivity
    )
    
    # Toroidal sensor
    toroidal_specs = ConductivitySensorSpecs(
        sensor_type=ConductivitySensorType.TOROIDAL,
        measurement_range=(5.0, 2000.0),  # mS/cm
        accuracy=1.0,  # ±1%
        precision=0.5,  # 0.5%
        response_time=2.0,  # seconds
        power_consumption=3.0,  # watts
        cost=5000.0,  # USD
        lifetime=43800.0,  # 5 years
        calibration_interval=8760.0,  # 1 year
        cell_constant=0.05,  # cm⁻¹ (effective)
        temperature_coefficient=1.0,  # %/°C
        pressure_sensitivity=0.02,  # %/bar
        fouling_sensitivity=0.1  # very low fouling sensitivity
    )
    
    sensors = {
        'two_electrode': ConductivitySensor(two_electrode_specs),
        'four_electrode': ConductivitySensor(four_electrode_specs),
        'inductive': ConductivitySensor(inductive_specs),
        'toroidal': ConductivitySensor(toroidal_specs)
    }
    
    return sensors


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create standard sensors
    sensors = create_standard_conductivity_sensors()
    
    # Test four-electrode sensor
    four_electrode = sensors['four_electrode']
    
    # Simulate measurements over time with fouling
    time_points = np.linspace(0, 1000, 1001)  # 1000 hours
    true_conductivity = 50.0  # mS/cm
    
    measurements = []
    for t in time_points:
        measurement = four_electrode.measure(true_conductivity, temperature=25.0, time=t)
        measurements.append(measurement)
        
        # Clean sensor every 200 hours
        if t > 0 and t % 200 == 0:
            four_electrode.clean_sensor(cleaning_effectiveness=0.9)
    
    # Analyze results
    measured_conductivities = [m.conductivity for m in measurements]
    fouling_factors = [m.fouling_factor for m in measurements]
    uncertainties = [m.uncertainty for m in measurements]
    
    print(f"Initial conductivity: {measured_conductivities[0]:.2f} mS/cm")
    print(f"Final conductivity: {measured_conductivities[-1]:.2f} mS/cm")
    print(f"Final fouling factor: {fouling_factors[-1]:.3f}")
    print(f"Mean uncertainty: {np.mean(uncertainties):.3f} mS/cm")
    
    # Cost analysis
    cost_analysis = four_electrode.get_cost_analysis()
    print(f"Total lifetime cost: ${cost_analysis['lifetime_cost']:.2f}")
    print(f"Operating cost per hour: ${cost_analysis['total_cost_per_hour']:.4f}")
    
    # Test electrolyte model
    electrolyte = ElectrolyteModel()
    ion_conc = {'Na+': 0.1, 'Cl-': 0.1}  # 0.1 M NaCl
    theoretical_cond = electrolyte.calculate_theoretical_conductivity(ion_conc)
    estimated_strength = electrolyte.estimate_ionic_strength(theoretical_cond)
    
    print(f"Theoretical conductivity (0.1 M NaCl): {theoretical_cond:.2f} mS/cm")
    print(f"Estimated ionic strength: {estimated_strength:.4f} mol/L")