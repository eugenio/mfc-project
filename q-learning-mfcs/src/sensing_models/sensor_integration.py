"""
Comprehensive Sensor Integration Framework for MFC Systems

This module provides a unified interface for managing multiple sensor types,
data acquisition, calibration scheduling, power management, and cost analysis
for complete MFC monitoring systems.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import logging

from .substrate_sensors import SubstrateSensor, create_standard_substrate_sensors
from .temperature_sensors import TemperatureSensor, create_standard_temperature_sensors
from .conductivity_sensors import ConductivitySensor, create_standard_conductivity_sensors
from .oxygen_sensors import OxygenSensor, create_standard_oxygen_sensors

logger = logging.getLogger(__name__)


class SensorStatus(Enum):
    """Sensor operational status"""
    OPERATIONAL = "operational"
    WARNING = "warning"
    FAILED = "failed"
    MAINTENANCE = "maintenance"
    CALIBRATION = "calibration"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SensorAlert:
    """Sensor system alert"""
    timestamp: float
    sensor_id: str
    alert_level: AlertLevel
    message: str
    measurement_value: Optional[float] = None
    expected_range: Optional[Tuple[float, float]] = None
    auto_resolved: bool = False


@dataclass
class MFCEnvironmentalConditions:
    """Current environmental conditions in the MFC"""
    temperature: float = 25.0  # °C
    pressure: float = 1.013  # bar
    ph: float = 7.0
    salinity: float = 0.0  # g/L
    ionic_strength: float = 0.1  # mol/L
    substrate_concentration: float = 10.0  # mM
    do_concentration: float = 0.1  # mg/L (typically anaerobic)


@dataclass
class SensorConfiguration:
    """Configuration for a sensor in the MFC system"""
    sensor_id: str
    sensor_type: str
    location: str
    sampling_rate: float  # Hz
    enabled: bool = True
    auto_calibration: bool = True
    alert_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    maintenance_schedule: Dict[str, float] = field(default_factory=dict)


class SensorDataLogger:
    """Data logging and storage for sensor measurements"""
    
    def __init__(self, max_storage_hours: float = 8760.0):  # 1 year
        self.max_storage_hours = max_storage_hours
        self.data_buffer: Dict[str, List[Dict]] = {}
        self.start_time = 0.0
        
    def log_measurement(self, sensor_id: str, measurement: Dict):
        """Log a sensor measurement"""
        if sensor_id not in self.data_buffer:
            self.data_buffer[sensor_id] = []
        
        self.data_buffer[sensor_id].append(measurement)
        
        # Clean old data if buffer is too large
        self._cleanup_old_data(sensor_id)
    
    def _cleanup_old_data(self, sensor_id: str):
        """Remove data older than max_storage_hours"""
        if not self.data_buffer[sensor_id]:
            return
        
        current_time = self.data_buffer[sensor_id][-1]['timestamp']
        cutoff_time = current_time - self.max_storage_hours
        
        # Keep only recent data
        self.data_buffer[sensor_id] = [
            data for data in self.data_buffer[sensor_id] 
            if data['timestamp'] >= cutoff_time
        ]
    
    def get_recent_data(self, sensor_id: str, hours: float = 24.0) -> List[Dict]:
        """Get recent data for a sensor"""
        if sensor_id not in self.data_buffer:
            return []
        
        if not self.data_buffer[sensor_id]:
            return []
        
        current_time = self.data_buffer[sensor_id][-1]['timestamp']
        cutoff_time = current_time - hours
        
        return [
            data for data in self.data_buffer[sensor_id] 
            if data['timestamp'] >= cutoff_time
        ]
    
    def get_statistics(self, sensor_id: str, hours: float = 24.0) -> Dict[str, float]:
        """Get statistical summary of recent data"""
        recent_data = self.get_recent_data(sensor_id, hours)
        
        if not recent_data:
            return {}
        
        values = [data['value'] for data in recent_data if 'value' in data]
        
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values),
            'last_value': values[-1] if values else None
        }


class SensorSystem:
    """Comprehensive sensor system for MFC monitoring"""
    
    def __init__(self, system_id: str = "MFC_Sensor_System"):
        self.system_id = system_id
        self.sensors: Dict[str, Union[SubstrateSensor, TemperatureSensor, 
                                   ConductivitySensor, OxygenSensor]] = {}
        self.sensor_configs: Dict[str, SensorConfiguration] = {}
        self.sensor_status: Dict[str, SensorStatus] = {}
        self.environmental_conditions = MFCEnvironmentalConditions()
        self.data_logger = SensorDataLogger()
        self.alerts: List[SensorAlert] = []
        self.system_start_time = 0.0
        
        # Power and cost tracking
        self.total_power_consumption = 0.0
        self.total_cost_per_hour = 0.0
        
        # Initialize with standard sensors
        self._initialize_standard_sensors()
    
    def _initialize_standard_sensors(self):
        """Initialize with standard sensor configurations"""
        # Substrate sensors
        substrate_sensors = create_standard_substrate_sensors()
        for name, sensor in substrate_sensors.items():
            sensor_id = f"substrate_{name}"
            self.add_sensor(sensor_id, sensor, location="anode_chamber", 
                          sampling_rate=0.1)  # Every 10 seconds
        
        # Temperature sensors
        temp_sensors = create_standard_temperature_sensors()
        for name, sensor in temp_sensors.items():
            sensor_id = f"temperature_{name}"
            self.add_sensor(sensor_id, sensor, location="system_wide", 
                          sampling_rate=1.0)  # Every second
        
        # Conductivity sensors
        cond_sensors = create_standard_conductivity_sensors()
        for name, sensor in cond_sensors.items():
            sensor_id = f"conductivity_{name}"
            self.add_sensor(sensor_id, sensor, location="electrolyte", 
                          sampling_rate=0.2)  # Every 5 seconds
        
        # Oxygen sensors
        o2_sensors = create_standard_oxygen_sensors()
        for name, sensor in o2_sensors.items():
            sensor_id = f"oxygen_{name}"
            self.add_sensor(sensor_id, sensor, location="cathode_chamber", 
                          sampling_rate=0.5)  # Every 2 seconds
    
    def add_sensor(self, sensor_id: str, sensor: Any, location: str = "unknown", 
                   sampling_rate: float = 1.0, enabled: bool = True):
        """Add a sensor to the system"""
        self.sensors[sensor_id] = sensor
        self.sensor_status[sensor_id] = SensorStatus.OPERATIONAL
        
        # Default configuration
        config = SensorConfiguration(
            sensor_id=sensor_id,
            sensor_type=type(sensor).__name__,
            location=location,
            sampling_rate=sampling_rate,
            enabled=enabled,
            alert_thresholds={
                'measurement_range': sensor.specs.measurement_range,
                'uncertainty_threshold': (0.0, 20.0)  # Max 20% uncertainty
            },
            maintenance_schedule={
                'calibration_interval': sensor.specs.calibration_interval,
                'cleaning_interval': 168.0 if hasattr(sensor, 'clean_sensor') else 0.0  # Weekly
            }
        )
        
        self.sensor_configs[sensor_id] = config
        
        # Update power and cost tracking
        self._update_system_power_cost()
        
        logger.info(f"Added sensor {sensor_id} ({type(sensor).__name__}) at {location}")
    
    def remove_sensor(self, sensor_id: str):
        """Remove a sensor from the system"""
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]
            del self.sensor_configs[sensor_id]
            del self.sensor_status[sensor_id]
            self._update_system_power_cost()
            logger.info(f"Removed sensor {sensor_id}")
    
    def update_environmental_conditions(self, **kwargs):
        """Update environmental conditions"""
        for key, value in kwargs.items():
            if hasattr(self.environmental_conditions, key):
                setattr(self.environmental_conditions, key, value)
    
    def acquire_all_measurements(self, time: float) -> Dict[str, Dict]:
        """Acquire measurements from all enabled sensors"""
        measurements = {}
        
        for sensor_id, sensor in self.sensors.items():
            config = self.sensor_configs[sensor_id]
            
            if not config.enabled:
                continue
            
            # Check if it's time to sample this sensor
            if not self._should_sample_sensor(sensor_id, time):
                continue
            
            try:
                measurement = self._acquire_single_measurement(sensor_id, sensor, time)
                measurements[sensor_id] = measurement
                
                # Log the measurement
                self.data_logger.log_measurement(sensor_id, measurement)
                
                # Check for alerts
                self._check_sensor_alerts(sensor_id, measurement)
                
            except Exception as e:
                logger.error(f"Failed to acquire measurement from {sensor_id}: {e}")
                self.sensor_status[sensor_id] = SensorStatus.FAILED
                self._generate_alert(sensor_id, AlertLevel.CRITICAL, 
                                   f"Sensor acquisition failed: {e}")
        
        return measurements
    
    def _should_sample_sensor(self, sensor_id: str, time: float) -> bool:
        """Check if sensor should be sampled at this time"""
        config = self.sensor_configs[sensor_id]
        
        # Get last sampling time
        recent_data = self.data_logger.get_recent_data(sensor_id, hours=0.1)
        
        if not recent_data:
            return True  # First measurement
        
        last_sample_time = recent_data[-1]['timestamp']
        sampling_interval = 1.0 / config.sampling_rate  # Convert Hz to hours
        
        return (time - last_sample_time) >= sampling_interval
    
    def _acquire_single_measurement(self, sensor_id: str, sensor: Any, time: float) -> Dict[str, Any]:
        """Acquire measurement from a single sensor"""
        env = self.environmental_conditions
        
        if isinstance(sensor, SubstrateSensor):
            measurement = sensor.measure(
                true_concentration=env.substrate_concentration,
                temperature=env.temperature,
                ph=env.ph,
                time=time
            )
            return {
                'timestamp': time,
                'sensor_id': sensor_id,
                'value': measurement.concentration,
                'unit': 'mM',
                'uncertainty': measurement.uncertainty,
                'quality': measurement.quality_flag,
                'raw_signal': measurement.raw_signal
            }
            
        elif isinstance(sensor, TemperatureSensor):
            measurement = sensor.measure(
                environment_temp=env.temperature,
                time=time
            )
            return {
                'timestamp': time,
                'sensor_id': sensor_id,
                'value': measurement.temperature,
                'unit': '°C',
                'uncertainty': measurement.uncertainty,
                'quality': measurement.quality_flag,
                'thermal_time_constant': measurement.thermal_time_constant
            }
            
        elif isinstance(sensor, ConductivitySensor):
            # Estimate conductivity from ionic strength
            conductivity = env.ionic_strength * 100.0  # Rough approximation
            measurement = sensor.measure(
                true_conductivity=conductivity,
                temperature=env.temperature,
                pressure=env.pressure,
                time=time
            )
            return {
                'timestamp': time,
                'sensor_id': sensor_id,
                'value': measurement.conductivity,
                'unit': 'mS/cm',
                'uncertainty': measurement.uncertainty,
                'quality': measurement.quality_flag,
                'fouling_factor': measurement.fouling_factor
            }
            
        elif isinstance(sensor, OxygenSensor):
            measurement = sensor.measure(
                do_concentration=env.do_concentration,
                temperature=env.temperature,
                pressure=env.pressure,
                salinity=env.salinity,
                time=time
            )
            return {
                'timestamp': time,
                'sensor_id': sensor_id,
                'value': measurement.do_concentration,
                'unit': 'mg/L',
                'uncertainty': measurement.uncertainty,
                'quality': measurement.quality_flag,
                'saturation': measurement.do_saturation,
                'membrane_condition': measurement.membrane_condition
            }
        
        else:
            raise ValueError(f"Unknown sensor type: {type(sensor)}")
    
    def _check_sensor_alerts(self, sensor_id: str, measurement: Dict):
        """Check measurement against alert thresholds"""
        config = self.sensor_configs[sensor_id]
        value = measurement['value']
        uncertainty = measurement.get('uncertainty', 0.0)
        
        # Check measurement range
        if 'measurement_range' in config.alert_thresholds:
            min_val, max_val = config.alert_thresholds['measurement_range']
            if value < min_val or value > max_val:
                self._generate_alert(
                    sensor_id, AlertLevel.WARNING,
                    f"Measurement {value:.3f} outside range [{min_val}, {max_val}]",
                    measurement_value=value,
                    expected_range=(min_val, max_val)
                )
        
        # Check uncertainty threshold
        if 'uncertainty_threshold' in config.alert_thresholds:
            min_unc, max_unc = config.alert_thresholds['uncertainty_threshold']
            relative_uncertainty = (uncertainty / value * 100) if value != 0 else float('inf')
            
            if relative_uncertainty > max_unc:
                self._generate_alert(
                    sensor_id, AlertLevel.WARNING,
                    f"High uncertainty: {relative_uncertainty:.1f}% > {max_unc}%",
                    measurement_value=relative_uncertainty
                )
        
        # Check sensor quality
        quality = measurement.get('quality', 'unknown')
        if quality == 'bad':
            self._generate_alert(
                sensor_id, AlertLevel.CRITICAL,
                "Sensor quality degraded to 'bad'",
                measurement_value=value
            )
        elif quality == 'warning':
            self._generate_alert(
                sensor_id, AlertLevel.WARNING,
                "Sensor quality degraded to 'warning'",
                measurement_value=value
            )
    
    def _generate_alert(self, sensor_id: str, level: AlertLevel, message: str,
                       measurement_value: Optional[float] = None,
                       expected_range: Optional[Tuple[float, float]] = None):
        """Generate a sensor alert"""
        alert = SensorAlert(
            timestamp=self.data_logger.data_buffer.get(sensor_id, [{'timestamp': 0.0}])[-1]['timestamp'],
            sensor_id=sensor_id,
            alert_level=level,
            message=message,
            measurement_value=measurement_value,
            expected_range=expected_range
        )
        
        self.alerts.append(alert)
        
        # Update sensor status based on alert level
        if level == AlertLevel.CRITICAL:
            self.sensor_status[sensor_id] = SensorStatus.FAILED
        elif level == AlertLevel.WARNING:
            self.sensor_status[sensor_id] = SensorStatus.WARNING
        
        logger.log(
            logging.CRITICAL if level == AlertLevel.CRITICAL else logging.WARNING,
            f"ALERT [{level.value.upper()}] {sensor_id}: {message}"
        )
    
    def calibrate_sensor(self, sensor_id: str, calibration_data: Dict, time: float) -> bool:
        """Calibrate a specific sensor"""
        if sensor_id not in self.sensors:
            logger.error(f"Sensor {sensor_id} not found")
            return False
        
        sensor = self.sensors[sensor_id]
        self.sensor_status[sensor_id] = SensorStatus.CALIBRATION
        
        try:
            if hasattr(sensor, 'calibrate'):
                success = sensor.calibrate(
                    calibration_data.get('reference_values', []),
                    calibration_data.get('measured_values', []),
                    time
                )
                
                if success:
                    self.sensor_status[sensor_id] = SensorStatus.OPERATIONAL
                    logger.info(f"Successfully calibrated sensor {sensor_id}")
                else:
                    self.sensor_status[sensor_id] = SensorStatus.FAILED
                    logger.error(f"Failed to calibrate sensor {sensor_id}")
                
                return success
            else:
                logger.warning(f"Sensor {sensor_id} does not support calibration")
                self.sensor_status[sensor_id] = SensorStatus.OPERATIONAL
                return False
                
        except Exception as e:
            logger.error(f"Error calibrating sensor {sensor_id}: {e}")
            self.sensor_status[sensor_id] = SensorStatus.FAILED
            return False
    
    def perform_maintenance(self, sensor_id: str, maintenance_type: str) -> bool:
        """Perform maintenance on a sensor"""
        if sensor_id not in self.sensors:
            logger.error(f"Sensor {sensor_id} not found")
            return False
        
        sensor = self.sensors[sensor_id]
        self.sensor_status[sensor_id] = SensorStatus.MAINTENANCE
        
        try:
            if maintenance_type == "clean" and hasattr(sensor, 'clean_sensor'):
                success = sensor.clean_sensor()
            elif maintenance_type == "replace_membrane" and hasattr(sensor, 'replace_membrane'):
                success = sensor.replace_membrane()
            else:
                logger.warning(f"Maintenance type '{maintenance_type}' not supported for {sensor_id}")
                success = False
            
            self.sensor_status[sensor_id] = SensorStatus.OPERATIONAL
            return success
            
        except Exception as e:
            logger.error(f"Error during maintenance of {sensor_id}: {e}")
            self.sensor_status[sensor_id] = SensorStatus.FAILED
            return False
    
    def _update_system_power_cost(self):
        """Update total system power consumption and cost"""
        total_power = 0.0
        total_cost = 0.0
        
        for sensor_id, sensor in self.sensors.items():
            if hasattr(sensor, 'get_power_consumption'):
                total_power += sensor.get_power_consumption()
            
            if hasattr(sensor, 'get_cost_analysis'):
                cost_analysis = sensor.get_cost_analysis()
                total_cost += cost_analysis.get('total_cost_per_hour', 0.0)
        
        self.total_power_consumption = total_power
        self.total_cost_per_hour = total_cost
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        operational_count = sum(1 for status in self.sensor_status.values() 
                              if status == SensorStatus.OPERATIONAL)
        total_sensors = len(self.sensors)
        
        recent_alerts = [alert for alert in self.alerts[-10:]]  # Last 10 alerts
        
        return {
            'system_id': self.system_id,
            'total_sensors': total_sensors,
            'operational_sensors': operational_count,
            'system_health': operational_count / total_sensors if total_sensors > 0 else 0.0,
            'total_power_consumption': self.total_power_consumption,
            'total_cost_per_hour': self.total_cost_per_hour,
            'sensor_status': {sid: status.value for sid, status in self.sensor_status.items()},
            'recent_alerts': [
                {
                    'timestamp': alert.timestamp,
                    'sensor_id': alert.sensor_id,
                    'level': alert.alert_level.value,
                    'message': alert.message
                } for alert in recent_alerts
            ],
            'environmental_conditions': {
                'temperature': self.environmental_conditions.temperature,
                'pressure': self.environmental_conditions.pressure,
                'ph': self.environmental_conditions.ph,
                'salinity': self.environmental_conditions.salinity,
                'substrate_concentration': self.environmental_conditions.substrate_concentration,
                'do_concentration': self.environmental_conditions.do_concentration
            }
        }
    
    def get_sensor_summary(self, hours: float = 24.0) -> Dict[str, Dict]:
        """Get summary of all sensors over specified time period"""
        summary = {}
        
        for sensor_id in self.sensors:
            stats = self.data_logger.get_statistics(sensor_id, hours)
            config = self.sensor_configs[sensor_id]
            status = self.sensor_status[sensor_id]
            
            summary[sensor_id] = {
                'sensor_type': config.sensor_type,
                'location': config.location,
                'status': status.value,
                'sampling_rate': config.sampling_rate,
                'statistics': stats,
                'enabled': config.enabled
            }
        
        return summary
    
    def export_data(self, sensor_ids: Optional[List[str]] = None, 
                   hours: float = 24.0) -> Dict[str, List[Dict]]:
        """Export sensor data for analysis"""
        if sensor_ids is None:
            sensor_ids = list(self.sensors.keys())
        
        export_data = {}
        for sensor_id in sensor_ids:
            if sensor_id in self.sensors:
                export_data[sensor_id] = self.data_logger.get_recent_data(sensor_id, hours)
        
        return export_data


def create_standard_mfc_sensor_system(system_id: str = "MFC_Primary") -> SensorSystem:
    """Create a standard sensor system for MFC monitoring"""
    system = SensorSystem(system_id)
    
    # Configure alert thresholds for different sensor types
    for sensor_id, config in system.sensor_configs.items():
        if 'substrate' in sensor_id:
            config.alert_thresholds.update({
                'measurement_range': (0.1, 100.0),  # mM
                'uncertainty_threshold': (0.0, 15.0)  # %
            })
        elif 'temperature' in sensor_id:
            config.alert_thresholds.update({
                'measurement_range': (15.0, 45.0),  # °C
                'uncertainty_threshold': (0.0, 5.0)  # %
            })
        elif 'conductivity' in sensor_id:
            config.alert_thresholds.update({
                'measurement_range': (1.0, 100.0),  # mS/cm
                'uncertainty_threshold': (0.0, 10.0)  # %
            })
        elif 'oxygen' in sensor_id:
            config.alert_thresholds.update({
                'measurement_range': (0.0, 2.0),  # mg/L (anaerobic conditions)
                'uncertainty_threshold': (0.0, 20.0)  # %
            })
    
    return system


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sensor system
    sensor_system = create_standard_mfc_sensor_system("MFC_Test_System")
    
    print(f"Initialized sensor system with {len(sensor_system.sensors)} sensors")
    print(f"Total power consumption: {sensor_system.total_power_consumption:.2f} W")
    print(f"Total cost per hour: ${sensor_system.total_cost_per_hour:.4f}")
    
    # Simulate measurements over time
    simulation_hours = 48.0  # 48 hours
    time_step = 0.1  # hours
    
    for hour in np.arange(0, simulation_hours, time_step):
        # Update environmental conditions (simulate changes)
        sensor_system.update_environmental_conditions(
            temperature=25.0 + 5.0 * np.sin(2 * np.pi * hour / 24.0),  # Daily temperature cycle
            substrate_concentration=10.0 * (1.0 - 0.5 * hour / simulation_hours),  # Decreasing substrate
            do_concentration=0.1 + 0.5 * np.random.random()  # Random DO fluctuations
        )
        
        # Acquire measurements
        measurements = sensor_system.acquire_all_measurements(hour)
        
        if hour % 12.0 < time_step:  # Every 12 hours
            print(f"Hour {hour:.1f}: Acquired {len(measurements)} measurements")
    
    # Get system status
    status = sensor_system.get_system_status()
    print(f"\nSystem Health: {status['system_health']:.1%}")
    print(f"Recent Alerts: {len(status['recent_alerts'])}")
    
    # Get sensor summary
    summary = sensor_system.get_sensor_summary(hours=24.0)
    for sensor_id, info in summary.items():
        stats = info['statistics']
        if stats:
            print(f"{sensor_id}: {stats['mean']:.2f} ± {stats['std']:.2f} ({info['status']})")
    
    # Export data
    exported_data = sensor_system.export_data(hours=24.0)
    total_data_points = sum(len(data) for data in exported_data.values())
    print(f"\nExported {total_data_points} data points from {len(exported_data)} sensors")