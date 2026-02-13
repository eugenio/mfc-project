"""
Substrate Concentration Sensors for MFC Monitoring

This module implements various substrate concentration measurement technologies
including UV-Vis spectroscopy, enzymatic biosensors, and fluorescence-based sensors.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SubstrateSensorType(Enum):
    """Types of substrate concentration sensors"""
    UV_VIS_SPECTROSCOPY = "uv_vis"
    ENZYMATIC_BIOSENSOR = "enzymatic"
    FLUORESCENCE = "fluorescence"
    CONDUCTOMETRIC = "conductometric"
    AMPEROMETRIC = "amperometric"


@dataclass
class SubstrateSensorSpecs:
    """Specifications for substrate concentration sensors"""
    sensor_type: SubstrateSensorType
    measurement_range: Tuple[float, float]  # min, max concentration (mM)
    accuracy: float  # ±% of reading
    precision: float  # % RSD
    response_time: float  # seconds
    power_consumption: float  # watts
    cost: float  # USD
    lifetime: float  # hours
    calibration_interval: float  # hours
    temperature_coefficient: float  # %/°C
    ph_sensitivity: float  # %/pH unit
    interference_compounds: List[str]


@dataclass
class SubstrateMeasurement:
    """Single substrate concentration measurement"""
    timestamp: float
    concentration: float  # mM
    sensor_type: SubstrateSensorType
    temperature: float  # °C
    ph: float
    uncertainty: float  # mM
    quality_flag: str  # 'good', 'warning', 'bad'
    raw_signal: float
    calibration_factor: float


class SubstrateSensor:
    """Base class for substrate concentration sensors"""
    
    def __init__(self, specs: SubstrateSensorSpecs, substrate_type: str = "lactate"):
        self.specs = specs
        self.substrate_type = substrate_type
        self.calibration_slope = 1.0
        self.calibration_offset = 0.0
        self.last_calibration = 0.0
        self.operating_hours = 0.0
        self.drift_coefficient = 0.001  # %/hour
        self.noise_level = specs.precision / 100.0
        
        # Sensor-specific parameters
        self._initialize_sensor_parameters()
        
    def _initialize_sensor_parameters(self):
        """Initialize sensor-specific parameters"""
        if self.specs.sensor_type == SubstrateSensorType.UV_VIS_SPECTROSCOPY:
            self.wavelength = 340 if self.substrate_type == "lactate" else 260  # nm
            self.path_length = 1.0  # cm
            self.extinction_coefficient = 6220 if self.substrate_type == "lactate" else 15000  # M⁻¹cm⁻¹
            
        elif self.specs.sensor_type == SubstrateSensorType.ENZYMATIC_BIOSENSOR:
            self.enzyme_activity = 100.0  # U/mL
            self.km_value = 0.5  # mM (Michaelis constant)
            self.enzyme_stability = 0.95  # daily retention
            
        elif self.specs.sensor_type == SubstrateSensorType.FLUORESCENCE:
            self.excitation_wavelength = 280  # nm
            self.emission_wavelength = 340  # nm
            self.quantum_yield = 0.15
            
        elif self.specs.sensor_type == SubstrateSensorType.AMPEROMETRIC:
            self.working_potential = -0.6  # V vs Ag/AgCl
            self.sensitivity = 50.0  # nA/mM
            self.background_current = 10.0  # nA
    
    def measure(self, true_concentration: float, temperature: float = 25.0, 
                ph: float = 7.0, time: float = 0.0) -> SubstrateMeasurement:
        """
        Simulate a substrate concentration measurement
        
        Args:
            true_concentration: Actual substrate concentration (mM)
            temperature: Solution temperature (°C)
            ph: Solution pH
            time: Current time (hours)
            
        Returns:
            SubstrateMeasurement object
        """
        # Update operating hours
        self.operating_hours = time
        
        # Calculate drift
        drift_factor = 1.0 + (self.drift_coefficient * self.operating_hours / 100.0)
        
        # Temperature correction
        temp_correction = 1.0 + self.specs.temperature_coefficient * (temperature - 25.0) / 100.0
        
        # pH correction
        ph_correction = 1.0 + self.specs.ph_sensitivity * (ph - 7.0) / 100.0
        
        # Sensor degradation
        degradation_factor = np.exp(-self.operating_hours / (self.specs.lifetime * 0.368))
        
        # Calculate raw signal based on sensor type
        raw_signal = self._calculate_raw_signal(true_concentration, temperature, ph)
        
        # Apply corrections and noise
        corrected_signal = raw_signal * temp_correction * ph_correction * drift_factor * degradation_factor
        
        # Add noise
        noise = np.random.normal(0, self.noise_level * corrected_signal)
        measured_signal = corrected_signal + noise
        
        # Convert to concentration
        measured_concentration = self._signal_to_concentration(measured_signal)
        
        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(measured_concentration, temperature, ph)
        
        # Determine quality flag
        quality_flag = self._assess_quality(measured_concentration, uncertainty, time)
        
        return SubstrateMeasurement(
            timestamp=time,
            concentration=measured_concentration,
            sensor_type=self.specs.sensor_type,
            temperature=temperature,
            ph=ph,
            uncertainty=uncertainty,
            quality_flag=quality_flag,
            raw_signal=measured_signal,
            calibration_factor=self.calibration_slope
        )
    
    def _calculate_raw_signal(self, concentration: float, temperature: float, ph: float) -> float:
        """Calculate raw sensor signal based on concentration"""
        if self.specs.sensor_type == SubstrateSensorType.UV_VIS_SPECTROSCOPY:
            # Beer-Lambert law: A = ε × c × l
            absorbance = self.extinction_coefficient * concentration * 1e-3 * self.path_length
            return absorbance
            
        elif self.specs.sensor_type == SubstrateSensorType.ENZYMATIC_BIOSENSOR:
            # Michaelis-Menten kinetics
            enzyme_activity_current = self.enzyme_activity * (self.enzyme_stability ** (self.operating_hours / 24.0))
            reaction_rate = enzyme_activity_current * concentration / (self.km_value + concentration)
            return reaction_rate
            
        elif self.specs.sensor_type == SubstrateSensorType.FLUORESCENCE:
            # Fluorescence intensity proportional to concentration
            intensity = self.quantum_yield * concentration * 1000  # arbitrary units
            return intensity
            
        elif self.specs.sensor_type == SubstrateSensorType.AMPEROMETRIC:
            # Current proportional to concentration
            current = self.sensitivity * concentration + self.background_current
            return current
            
        else:
            # Linear response for other sensors
            return concentration * 100.0  # arbitrary signal units
    
    def _signal_to_concentration(self, signal: float) -> float:
        """Convert sensor signal to concentration"""
        if self.specs.sensor_type == SubstrateSensorType.UV_VIS_SPECTROSCOPY:
            # Reverse Beer-Lambert law
            concentration = signal / (self.extinction_coefficient * self.path_length) * 1000
            
        elif self.specs.sensor_type == SubstrateSensorType.ENZYMATIC_BIOSENSOR:
            # Reverse Michaelis-Menten (approximate for low concentrations)
            enzyme_activity_current = self.enzyme_activity * (self.enzyme_stability ** (self.operating_hours / 24.0))
            concentration = signal * self.km_value / (enzyme_activity_current - signal)
            concentration = max(0, concentration)  # Ensure non-negative
            
        elif self.specs.sensor_type == SubstrateSensorType.FLUORESCENCE:
            concentration = signal / (self.quantum_yield * 1000)
            
        elif self.specs.sensor_type == SubstrateSensorType.AMPEROMETRIC:
            concentration = (signal - self.background_current) / self.sensitivity
            
        else:
            concentration = signal / 100.0
        
        # Apply calibration
        concentration = (concentration - self.calibration_offset) / self.calibration_slope
        
        # Ensure within measurement range
        concentration = np.clip(concentration, self.specs.measurement_range[0], 
                              self.specs.measurement_range[1])
        
        return concentration
    
    def _calculate_uncertainty(self, concentration: float, temperature: float, ph: float) -> float:
        """Calculate measurement uncertainty"""
        # Base uncertainty from accuracy specification
        base_uncertainty = concentration * self.specs.accuracy / 100.0
        
        # Additional uncertainty from temperature and pH effects
        temp_uncertainty = abs(concentration * self.specs.temperature_coefficient * (temperature - 25.0) / 100.0)
        ph_uncertainty = abs(concentration * self.specs.ph_sensitivity * (ph - 7.0) / 100.0)
        
        # Calibration uncertainty increases with time since last calibration
        time_since_cal = self.operating_hours - self.last_calibration
        cal_uncertainty = concentration * 0.001 * time_since_cal  # 0.1% per 100 hours
        
        # Total uncertainty (root sum of squares)
        total_uncertainty = np.sqrt(base_uncertainty**2 + temp_uncertainty**2 + 
                                  ph_uncertainty**2 + cal_uncertainty**2)
        
        return total_uncertainty
    
    def _assess_quality(self, concentration: float, uncertainty: float, time: float) -> str:
        """Assess measurement quality"""
        # Check if within range
        if concentration <= self.specs.measurement_range[0] or concentration >= self.specs.measurement_range[1]:
            return "bad"
        
        # Check relative uncertainty
        relative_uncertainty = uncertainty / concentration if concentration > 0 else float('inf')
        if relative_uncertainty > 0.2:  # 20%
            return "bad"
        elif relative_uncertainty > 0.1:  # 10%
            return "warning"
        
        # Check sensor lifetime
        if self.operating_hours > self.specs.lifetime * 0.9:
            return "warning"
        
        # Check calibration status
        time_since_cal = time - self.last_calibration
        if time_since_cal > self.specs.calibration_interval:
            return "warning"
        
        return "good"
    
    def calibrate(self, standard_concentrations: List[float], 
                  measured_signals: List[float], time: float) -> bool:
        """Calibrate the sensor with standard solutions"""
        if len(standard_concentrations) != len(measured_signals):
            logger.error("Standard concentrations and measured signals must have same length")
            return False
        
        if len(standard_concentrations) < 2:
            logger.error("At least 2 calibration points required")
            return False
        
        # Linear regression for calibration curve
        concentrations = np.array(standard_concentrations)
        signals = np.array(measured_signals)
        
        # Convert signals to raw concentrations
        raw_concentrations = [self._signal_to_concentration(s) for s in signals]
        
        # Calculate calibration factors
        slope, intercept = np.polyfit(raw_concentrations, concentrations, 1)
        
        self.calibration_slope = slope
        self.calibration_offset = intercept
        self.last_calibration = time
        
        logger.info(f"Sensor calibrated: slope={slope:.4f}, offset={intercept:.4f}")
        return True
    
    def get_power_consumption(self) -> float:
        """Get current power consumption"""
        # Base power consumption
        base_power = self.specs.power_consumption
        
        # Additional power for temperature compensation
        temp_power = 0.1 if hasattr(self, 'temp_compensation') else 0.0
        
        # Sensor aging increases power consumption
        aging_factor = 1.0 + (self.operating_hours / self.specs.lifetime) * 0.2
        
        return base_power * aging_factor + temp_power
    
    def get_cost_analysis(self) -> Dict[str, float]:
        """Get comprehensive cost analysis"""
        # Initial cost
        initial_cost = self.specs.cost
        
        # Operating cost per hour
        power_cost_per_hour = self.get_power_consumption() * 0.15  # $0.15/kWh
        
        # Maintenance cost (calibration, consumables)
        maintenance_cost_per_hour = 0.5 if self.specs.sensor_type == SubstrateSensorType.ENZYMATIC_BIOSENSOR else 0.1
        
        # Replacement cost amortized over lifetime
        replacement_cost_per_hour = initial_cost / self.specs.lifetime
        
        # Total operating cost
        total_cost_per_hour = power_cost_per_hour + maintenance_cost_per_hour + replacement_cost_per_hour
        
        return {
            'initial_cost': initial_cost,
            'power_cost_per_hour': power_cost_per_hour,
            'maintenance_cost_per_hour': maintenance_cost_per_hour,
            'replacement_cost_per_hour': replacement_cost_per_hour,
            'total_cost_per_hour': total_cost_per_hour,
            'lifetime_cost': initial_cost + total_cost_per_hour * self.specs.lifetime
        }


def create_standard_substrate_sensors() -> Dict[str, SubstrateSensor]:
    """Create standard substrate sensor configurations"""
    
    # UV-Vis Spectroscopy sensor
    uv_vis_specs = SubstrateSensorSpecs(
        sensor_type=SubstrateSensorType.UV_VIS_SPECTROSCOPY,
        measurement_range=(0.1, 50.0),  # mM
        accuracy=2.0,  # ±2%
        precision=1.0,  # 1% RSD
        response_time=5.0,  # seconds
        power_consumption=25.0,  # watts
        cost=15000.0,  # USD
        lifetime=8760.0,  # 1 year
        calibration_interval=168.0,  # 1 week
        temperature_coefficient=0.5,  # %/°C
        ph_sensitivity=1.0,  # %/pH unit
        interference_compounds=['acetate', 'formate', 'proteins']
    )
    
    # Enzymatic biosensor
    enzymatic_specs = SubstrateSensorSpecs(
        sensor_type=SubstrateSensorType.ENZYMATIC_BIOSENSOR,
        measurement_range=(0.01, 10.0),  # mM
        accuracy=5.0,  # ±5%
        precision=3.0,  # 3% RSD
        response_time=30.0,  # seconds
        power_consumption=2.0,  # watts
        cost=500.0,  # USD
        lifetime=720.0,  # 30 days
        calibration_interval=24.0,  # 1 day
        temperature_coefficient=3.0,  # %/°C
        ph_sensitivity=5.0,  # %/pH unit
        interference_compounds=['glucose', 'pyruvate']
    )
    
    # Amperometric sensor
    amperometric_specs = SubstrateSensorSpecs(
        sensor_type=SubstrateSensorType.AMPEROMETRIC,
        measurement_range=(0.1, 100.0),  # mM
        accuracy=3.0,  # ±3%
        precision=2.0,  # 2% RSD
        response_time=10.0,  # seconds
        power_consumption=1.0,  # watts
        cost=2000.0,  # USD
        lifetime=4380.0,  # 6 months
        calibration_interval=168.0,  # 1 week
        temperature_coefficient=2.0,  # %/°C
        ph_sensitivity=3.0,  # %/pH unit
        interference_compounds=['oxygen', 'hydrogen_peroxide']
    )
    
    sensors = {
        'uv_vis_lactate': SubstrateSensor(uv_vis_specs, 'lactate'),
        'enzymatic_lactate': SubstrateSensor(enzymatic_specs, 'lactate'),
        'amperometric_lactate': SubstrateSensor(amperometric_specs, 'lactate')
    }
    
    return sensors


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create standard sensors
    sensors = create_standard_substrate_sensors()
    
    # Test UV-Vis sensor
    uv_vis_sensor = sensors['uv_vis_lactate']
    
    # Simulate measurements over time
    time_points = np.linspace(0, 100, 101)  # 100 hours
    true_concentration = 10.0  # mM lactate
    
    measurements = []
    for t in time_points:
        measurement = uv_vis_sensor.measure(true_concentration, temperature=25.0, ph=7.0, time=t)
        measurements.append(measurement)
        
        if t == 50:  # Calibrate at 50 hours
            standards = [1.0, 5.0, 10.0, 20.0]  # mM
            signals = [uv_vis_sensor._calculate_raw_signal(c, 25.0, 7.0) for c in standards]
            uv_vis_sensor.calibrate(standards, signals, t)
    
    # Analyze results
    concentrations = [m.concentration for m in measurements]
    uncertainties = [m.uncertainty for m in measurements]
    
    print(f"Mean measured concentration: {np.mean(concentrations):.2f} ± {np.std(concentrations):.2f} mM")
    print(f"Mean uncertainty: {np.mean(uncertainties):.3f} mM")
    
    # Cost analysis
    cost_analysis = uv_vis_sensor.get_cost_analysis()
    print(f"Total lifetime cost: ${cost_analysis['lifetime_cost']:.2f}")
    print(f"Operating cost per hour: ${cost_analysis['total_cost_per_hour']:.4f}")