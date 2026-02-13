"""
Temperature Sensors with Thermal Dynamics for MFC Monitoring

This module implements various temperature measurement technologies including
thermocouples, RTDs, thermistors, and infrared sensors with thermal dynamics modeling.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TemperatureSensorType(Enum):
    """Types of temperature sensors"""
    THERMOCOUPLE_K = "thermocouple_k"
    THERMOCOUPLE_T = "thermocouple_t"
    RTD_PT100 = "rtd_pt100"
    RTD_PT1000 = "rtd_pt1000"
    THERMISTOR_NTC = "thermistor_ntc"
    THERMISTOR_PTC = "thermistor_ptc"
    IC_SENSOR = "ic_sensor"
    INFRARED = "infrared"


@dataclass
class TemperatureSensorSpecs:
    """Specifications for temperature sensors"""
    sensor_type: TemperatureSensorType
    measurement_range: Tuple[float, float]  # min, max temperature (°C)
    accuracy: float  # ±°C
    precision: float  # °C
    response_time: float  # seconds (63% response time constant)
    power_consumption: float  # watts
    cost: float  # USD
    lifetime: float  # hours
    calibration_interval: float  # hours
    thermal_mass: float  # J/K
    self_heating: float  # °C/mW
    stability: float  # °C/year


@dataclass
class TemperatureMeasurement:
    """Single temperature measurement"""
    timestamp: float
    temperature: float  # °C
    sensor_type: TemperatureSensorType
    uncertainty: float  # °C
    quality_flag: str  # 'good', 'warning', 'bad'
    raw_signal: float
    thermal_time_constant: float  # seconds
    self_heating_effect: float  # °C


class ThermalDynamicsModel:
    """Model thermal dynamics of the sensor and environment"""
    
    def __init__(self, thermal_mass: float, heat_transfer_coefficient: float = 10.0):
        self.thermal_mass = thermal_mass  # J/K
        self.heat_transfer_coefficient = heat_transfer_coefficient  # W/K
        self.sensor_temperature = 25.0  # °C
        self.environment_temperature = 25.0  # °C
        
    def update(self, environment_temp: float, dt: float, power_dissipation: float = 0.0):
        """
        Update sensor temperature based on thermal dynamics
        
        Args:
            environment_temp: Environment temperature (°C)
            dt: Time step (seconds)
            power_dissipation: Power dissipated in sensor (W)
        """
        self.environment_temperature = environment_temp
        
        # Heat transfer rate (W)
        heat_transfer = self.heat_transfer_coefficient * (environment_temp - self.sensor_temperature)
        
        # Temperature change due to heat transfer and self-heating
        dT_dt = (heat_transfer + power_dissipation) / self.thermal_mass
        
        # Update sensor temperature
        self.sensor_temperature += dT_dt * dt
        
    def get_time_constant(self) -> float:
        """Get thermal time constant (seconds)"""
        return self.thermal_mass / self.heat_transfer_coefficient


class TemperatureSensor:
    """Base class for temperature sensors with thermal dynamics"""
    
    def __init__(self, specs: TemperatureSensorSpecs):
        self.specs = specs
        self.calibration_offset = 0.0
        self.last_calibration = 0.0
        self.operating_hours = 0.0
        self.drift_rate = 0.01  # °C/1000h
        
        # Initialize thermal dynamics
        self.thermal_model = ThermalDynamicsModel(
            thermal_mass=specs.thermal_mass,
            heat_transfer_coefficient=specs.thermal_mass / specs.response_time
        )
        
        # Sensor-specific parameters
        self._initialize_sensor_parameters()
        
    def _initialize_sensor_parameters(self):
        """Initialize sensor-specific parameters"""
        if self.specs.sensor_type == TemperatureSensorType.THERMOCOUPLE_K:
            # K-type thermocouple (Chromel-Alumel)
            self.seebeck_coefficient = 41.3e-6  # V/°C at 25°C
            self.cold_junction_temp = 25.0  # °C
            self.reference_voltage = 0.0  # V
            
        elif self.specs.sensor_type == TemperatureSensorType.THERMOCOUPLE_T:
            # T-type thermocouple (Copper-Constantan)
            self.seebeck_coefficient = 42.9e-6  # V/°C at 25°C
            self.cold_junction_temp = 25.0  # °C
            self.reference_voltage = 0.0  # V
            
        elif self.specs.sensor_type == TemperatureSensorType.RTD_PT100:
            # PT100 RTD parameters
            self.r0 = 100.0  # Ω at 0°C
            self.alpha = 3.908e-3  # /°C
            self.beta = -5.775e-7  # /°C²
            self.excitation_current = 1e-3  # A
            
        elif self.specs.sensor_type == TemperatureSensorType.RTD_PT1000:
            # PT1000 RTD parameters
            self.r0 = 1000.0  # Ω at 0°C
            self.alpha = 3.908e-3  # /°C
            self.beta = -5.775e-7  # /°C²
            self.excitation_current = 0.5e-3  # A
            
        elif self.specs.sensor_type in [TemperatureSensorType.THERMISTOR_NTC, TemperatureSensorType.THERMISTOR_PTC]:
            # Thermistor parameters (Steinhart-Hart equation)
            self.r25 = 10000.0  # Ω at 25°C
            self.beta = 3380.0  # K
            self.a = 1.129148e-3
            self.b = 2.34125e-4
            self.c = 8.76741e-8
            
        elif self.specs.sensor_type == TemperatureSensorType.IC_SENSOR:
            # IC temperature sensor (LM35, etc.)
            self.sensitivity = 10.0e-3  # V/°C
            self.offset_voltage = 0.0  # V
            
        elif self.specs.sensor_type == TemperatureSensorType.INFRARED:
            # Infrared temperature sensor
            self.emissivity = 0.95
            self.stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
            self.detector_area = 1e-6  # m²
    
    def measure(self, environment_temp: float, time: float = 0.0, dt: float = 1.0) -> TemperatureMeasurement:
        """
        Simulate a temperature measurement with thermal dynamics
        
        Args:
            environment_temp: True environment temperature (°C)
            time: Current time (hours)
            dt: Time step since last measurement (seconds)
            
        Returns:
            TemperatureMeasurement object
        """
        # Update operating hours
        self.operating_hours = time
        
        # Calculate self-heating power
        power_dissipation = self._calculate_self_heating_power()
        
        # Update thermal dynamics
        self.thermal_model.update(environment_temp, dt, power_dissipation)
        
        # Get sensor temperature (may differ from environment due to thermal lag)
        sensor_temp = self.thermal_model.sensor_temperature
        
        # Calculate drift
        drift = self.drift_rate * self.operating_hours / 1000.0
        
        # Calculate raw signal
        raw_signal = self._calculate_raw_signal(sensor_temp)
        
        # Add noise
        noise_std = self.specs.precision / 3.0  # 3σ = precision
        noise = np.random.normal(0, noise_std)
        
        # Convert signal to temperature
        measured_temp = self._signal_to_temperature(raw_signal) + noise + drift + self.calibration_offset
        
        # Calculate self-heating effect
        self_heating_effect = power_dissipation * self.specs.self_heating
        
        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(measured_temp, time)
        
        # Determine quality flag
        quality_flag = self._assess_quality(measured_temp, uncertainty, time)
        
        return TemperatureMeasurement(
            timestamp=time,
            temperature=measured_temp,
            sensor_type=self.specs.sensor_type,
            uncertainty=uncertainty,
            quality_flag=quality_flag,
            raw_signal=raw_signal,
            thermal_time_constant=self.thermal_model.get_time_constant(),
            self_heating_effect=self_heating_effect
        )
    
    def _calculate_self_heating_power(self) -> float:
        """Calculate self-heating power dissipation"""
        if self.specs.sensor_type in [TemperatureSensorType.RTD_PT100, TemperatureSensorType.RTD_PT1000]:
            # RTD self-heating: P = I²R
            resistance = self.r0 * (1 + self.alpha * (self.thermal_model.sensor_temperature - 0))
            return self.excitation_current**2 * resistance
            
        elif self.specs.sensor_type in [TemperatureSensorType.THERMISTOR_NTC, TemperatureSensorType.THERMISTOR_PTC]:
            # Thermistor self-heating (assuming 1mA excitation)
            temp_k = self.thermal_model.sensor_temperature + 273.15
            resistance = self.r25 * np.exp(self.beta * (1/temp_k - 1/298.15))
            excitation_current = 1e-3  # A
            return excitation_current**2 * resistance
            
        else:
            # Minimal self-heating for other sensor types
            return 1e-6  # W
    
    def _calculate_raw_signal(self, temperature: float) -> float:
        """Calculate raw sensor signal based on temperature"""
        if self.specs.sensor_type == TemperatureSensorType.THERMOCOUPLE_K:
            # Thermocouple voltage (simplified linear approximation)
            voltage = self.seebeck_coefficient * (temperature - self.cold_junction_temp)
            return voltage
            
        elif self.specs.sensor_type == TemperatureSensorType.THERMOCOUPLE_T:
            voltage = self.seebeck_coefficient * (temperature - self.cold_junction_temp)
            return voltage
            
        elif self.specs.sensor_type in [TemperatureSensorType.RTD_PT100, TemperatureSensorType.RTD_PT1000]:
            # Callendar-Van Dusen equation (simplified)
            resistance = self.r0 * (1 + self.alpha * temperature + self.beta * temperature**2)
            return resistance
            
        elif self.specs.sensor_type == TemperatureSensorType.THERMISTOR_NTC:
            # Steinhart-Hart equation
            temp_k = temperature + 273.15
            resistance = self.r25 * np.exp(self.beta * (1/temp_k - 1/298.15))
            return resistance
            
        elif self.specs.sensor_type == TemperatureSensorType.THERMISTOR_PTC:
            # PTC thermistor (simplified model)
            temp_k = temperature + 273.15
            resistance = self.r25 * np.exp(-self.beta * (1/temp_k - 1/298.15))
            return resistance
            
        elif self.specs.sensor_type == TemperatureSensorType.IC_SENSOR:
            # Linear IC sensor
            voltage = self.sensitivity * temperature + self.offset_voltage
            return voltage
            
        elif self.specs.sensor_type == TemperatureSensorType.INFRARED:
            # Stefan-Boltzmann law for infrared
            temp_k = temperature + 273.15
            radiant_power = self.emissivity * self.stefan_boltzmann * self.detector_area * temp_k**4
            return radiant_power
            
        else:
            return temperature  # Direct reading
    
    def _signal_to_temperature(self, signal: float) -> float:
        """Convert sensor signal to temperature"""
        if self.specs.sensor_type in [TemperatureSensorType.THERMOCOUPLE_K, TemperatureSensorType.THERMOCOUPLE_T]:
            # Reverse thermocouple equation
            temperature = signal / self.seebeck_coefficient + self.cold_junction_temp
            
        elif self.specs.sensor_type in [TemperatureSensorType.RTD_PT100, TemperatureSensorType.RTD_PT1000]:
            # Reverse Callendar-Van Dusen (quadratic solution)
            a = self.beta
            b = self.alpha
            c = 1 - signal / self.r0
            
            if abs(a) < 1e-10:  # Linear case
                temperature = -c / b
            else:
                discriminant = b**2 - 4*a*c
                if discriminant >= 0:
                    temperature = (-b + np.sqrt(discriminant)) / (2*a)
                else:
                    temperature = -c / b  # Fallback to linear
                    
        elif self.specs.sensor_type == TemperatureSensorType.THERMISTOR_NTC:
            # Reverse Steinhart-Hart
            ln_r = np.log(signal / self.r25)
            temp_k = self.beta / (ln_r + self.beta / 298.15)
            temperature = temp_k - 273.15
            
        elif self.specs.sensor_type == TemperatureSensorType.THERMISTOR_PTC:
            # Reverse PTC equation
            ln_r = np.log(signal / self.r25)
            temp_k = self.beta / (-ln_r + self.beta / 298.15)
            temperature = temp_k - 273.15
            
        elif self.specs.sensor_type == TemperatureSensorType.IC_SENSOR:
            # Reverse linear IC sensor
            temperature = (signal - self.offset_voltage) / self.sensitivity
            
        elif self.specs.sensor_type == TemperatureSensorType.INFRARED:
            # Reverse Stefan-Boltzmann
            temp_k = (signal / (self.emissivity * self.stefan_boltzmann * self.detector_area))**(1/4)
            temperature = temp_k - 273.15
            
        else:
            temperature = signal
        
        # Ensure within measurement range
        temperature = np.clip(temperature, self.specs.measurement_range[0], 
                            self.specs.measurement_range[1])
        
        return temperature
    
    def _calculate_uncertainty(self, temperature: float, time: float) -> float:
        """Calculate measurement uncertainty"""
        # Base uncertainty from accuracy specification
        base_uncertainty = self.specs.accuracy
        
        # Drift uncertainty
        time_since_cal = time - self.last_calibration
        drift_uncertainty = self.drift_rate * time_since_cal / 1000.0
        
        # Self-heating uncertainty
        self_heating_uncertainty = self._calculate_self_heating_power() * self.specs.self_heating
        
        # Thermal lag uncertainty (during temperature changes)
        thermal_lag_uncertainty = 0.1  # °C (simplified)
        
        # Total uncertainty (root sum of squares)
        total_uncertainty = np.sqrt(base_uncertainty**2 + drift_uncertainty**2 + 
                                  self_heating_uncertainty**2 + thermal_lag_uncertainty**2)
        
        return total_uncertainty
    
    def _assess_quality(self, temperature: float, uncertainty: float, time: float) -> str:
        """Assess measurement quality"""
        # Check if within range
        if temperature <= self.specs.measurement_range[0] or temperature >= self.specs.measurement_range[1]:
            return "bad"
        
        # Check relative uncertainty
        if uncertainty > self.specs.accuracy * 2:
            return "bad"
        elif uncertainty > self.specs.accuracy * 1.5:
            return "warning"
        
        # Check sensor lifetime
        if self.operating_hours > self.specs.lifetime * 0.9:
            return "warning"
        
        # Check calibration status
        time_since_cal = time - self.last_calibration
        if time_since_cal > self.specs.calibration_interval:
            return "warning"
        
        return "good"
    
    def calibrate(self, reference_temperatures: List[float], 
                  measured_temperatures: List[float], time: float) -> bool:
        """Calibrate the sensor with reference temperatures"""
        if len(reference_temperatures) != len(measured_temperatures):
            logger.error("Reference and measured temperatures must have same length")
            return False
        
        if len(reference_temperatures) < 2:
            logger.error("At least 2 calibration points required")
            return False
        
        # Calculate average offset
        offsets = [ref - meas for ref, meas in zip(reference_temperatures, measured_temperatures)]
        self.calibration_offset = np.mean(offsets)
        self.last_calibration = time
        
        logger.info(f"Temperature sensor calibrated: offset={self.calibration_offset:.3f}°C")
        return True
    
    def get_power_consumption(self) -> float:
        """Get current power consumption"""
        base_power = self.specs.power_consumption
        
        # Additional power for signal conditioning
        if self.specs.sensor_type in [TemperatureSensorType.RTD_PT100, TemperatureSensorType.RTD_PT1000]:
            # RTD requires constant current source
            signal_conditioning_power = 0.1  # W
        elif self.specs.sensor_type == TemperatureSensorType.INFRARED:
            # IR sensor requires detector cooling
            signal_conditioning_power = 0.5  # W
        else:
            signal_conditioning_power = 0.05  # W
        
        return base_power + signal_conditioning_power
    
    def get_cost_analysis(self) -> Dict[str, float]:
        """Get comprehensive cost analysis"""
        initial_cost = self.specs.cost
        
        # Operating cost per hour
        power_cost_per_hour = self.get_power_consumption() * 0.15 / 1000  # $0.15/kWh
        
        # Maintenance cost (calibration, replacement)
        if self.specs.sensor_type in [TemperatureSensorType.THERMOCOUPLE_K, TemperatureSensorType.THERMOCOUPLE_T]:
            maintenance_cost_per_hour = 0.01  # Low maintenance
        elif self.specs.sensor_type == TemperatureSensorType.INFRARED:
            maintenance_cost_per_hour = 0.1  # Higher maintenance (detector cleaning)
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


def create_standard_temperature_sensors() -> Dict[str, TemperatureSensor]:
    """Create standard temperature sensor configurations"""
    
    # K-type thermocouple
    thermocouple_k_specs = TemperatureSensorSpecs(
        sensor_type=TemperatureSensorType.THERMOCOUPLE_K,
        measurement_range=(-40.0, 150.0),  # °C
        accuracy=0.5,  # ±0.5°C
        precision=0.1,  # 0.1°C
        response_time=0.5,  # seconds
        power_consumption=0.001,  # watts
        cost=50.0,  # USD
        lifetime=17520.0,  # 2 years
        calibration_interval=8760.0,  # 1 year
        thermal_mass=0.01,  # J/K
        self_heating=0.0,  # °C/mW
        stability=0.1  # °C/year
    )
    
    # PT100 RTD
    rtd_pt100_specs = TemperatureSensorSpecs(
        sensor_type=TemperatureSensorType.RTD_PT100,
        measurement_range=(-50.0, 200.0),  # °C
        accuracy=0.1,  # ±0.1°C
        precision=0.05,  # 0.05°C
        response_time=2.0,  # seconds
        power_consumption=0.01,  # watts
        cost=150.0,  # USD
        lifetime=43800.0,  # 5 years
        calibration_interval=17520.0,  # 2 years
        thermal_mass=0.05,  # J/K
        self_heating=0.1,  # °C/mW
        stability=0.05  # °C/year
    )
    
    # NTC Thermistor
    thermistor_ntc_specs = TemperatureSensorSpecs(
        sensor_type=TemperatureSensorType.THERMISTOR_NTC,
        measurement_range=(-10.0, 100.0),  # °C
        accuracy=0.2,  # ±0.2°C
        precision=0.1,  # 0.1°C
        response_time=1.0,  # seconds
        power_consumption=0.005,  # watts
        cost=10.0,  # USD
        lifetime=26280.0,  # 3 years
        calibration_interval=8760.0,  # 1 year
        thermal_mass=0.001,  # J/K
        self_heating=0.5,  # °C/mW
        stability=0.2  # °C/year
    )
    
    # IC Temperature Sensor
    ic_sensor_specs = TemperatureSensorSpecs(
        sensor_type=TemperatureSensorType.IC_SENSOR,
        measurement_range=(-10.0, 85.0),  # °C
        accuracy=0.5,  # ±0.5°C
        precision=0.2,  # 0.2°C
        response_time=5.0,  # seconds
        power_consumption=0.1,  # watts
        cost=25.0,  # USD
        lifetime=87600.0,  # 10 years
        calibration_interval=17520.0,  # 2 years
        thermal_mass=0.1,  # J/K
        self_heating=0.01,  # °C/mW
        stability=0.1  # °C/year
    )
    
    sensors = {
        'thermocouple_k': TemperatureSensor(thermocouple_k_specs),
        'rtd_pt100': TemperatureSensor(rtd_pt100_specs),
        'thermistor_ntc': TemperatureSensor(thermistor_ntc_specs),
        'ic_sensor': TemperatureSensor(ic_sensor_specs)
    }
    
    return sensors


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create standard sensors
    sensors = create_standard_temperature_sensors()
    
    # Test RTD sensor with thermal dynamics
    rtd_sensor = sensors['rtd_pt100']
    
    # Simulate temperature step response
    time_points = np.linspace(0, 60, 601)  # 60 seconds, 0.1s intervals
    environment_temps = [25.0] * 300 + [50.0] * 301  # Step from 25°C to 50°C at t=30s
    
    measurements = []
    for i, t in enumerate(time_points):
        dt = 0.1 if i > 0 else 0.0
        measurement = rtd_sensor.measure(environment_temps[i], time=t/3600, dt=dt)
        measurements.append(measurement)
    
    # Analyze thermal response
    measured_temps = [m.temperature for m in measurements]
    time_constants = [m.thermal_time_constant for m in measurements]
    
    print(f"Thermal time constant: {time_constants[0]:.2f} seconds")
    print(f"Final temperature: {measured_temps[-1]:.2f}°C")
    print(f"Step response settling time: ~{3*time_constants[0]:.1f} seconds")
    
    # Cost analysis
    cost_analysis = rtd_sensor.get_cost_analysis()
    print(f"Total lifetime cost: ${cost_analysis['lifetime_cost']:.2f}")
    print(f"Operating cost per hour: ${cost_analysis['total_cost_per_hour']:.6f}")