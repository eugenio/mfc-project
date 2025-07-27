"""
Hardware Abstraction Layer for MFC Control

This module provides a hardware abstraction layer (HAL) that interfaces between
the high-level control algorithms and the physical hardware components of the
MFC system, enabling portability and maintainability.
"""

import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Protocol, Union
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Types of hardware devices"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    CONTROLLER = "controller"
    COMMUNICATION = "communication"
    POWER = "power"


class DeviceStatus(Enum):
    """Device operational status"""
    OFFLINE = "offline"
    ONLINE = "online"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    CALIBRATING = "calibrating"


@dataclass
class DeviceInfo:
    """Device information structure"""
    device_id: str
    device_type: DeviceType
    name: str
    manufacturer: str
    model: str
    firmware_version: str
    hardware_revision: str
    serial_number: str
    installation_date: str
    last_calibration: str


@dataclass
class DeviceConfiguration:
    """Device configuration parameters"""
    device_id: str
    parameters: Dict[str, Any]
    limits: Dict[str, Tuple[float, float]]
    calibration_data: Dict[str, float]
    maintenance_schedule: Dict[str, Any]


class DeviceDriver(Protocol):
    """Protocol for device drivers"""
    
    def initialize(self) -> bool:
        """Initialize the device"""
        ...
    
    def read(self) -> Any:
        """Read data from device"""
        ...
    
    def write(self, data: Any) -> bool:
        """Write data to device"""
        ...
    
    def calibrate(self) -> bool:
        """Calibrate the device"""
        ...
    
    def get_status(self) -> DeviceStatus:
        """Get device status"""
        ...
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get device diagnostics"""
        ...


class MFCDevice(ABC):
    """Abstract base class for MFC devices"""
    
    def __init__(self, device_info: DeviceInfo, config: DeviceConfiguration):
        self.device_info = device_info
        self.config = config
        self.status = DeviceStatus.OFFLINE
        self.last_update = 0.0
        self.error_count = 0
        self.calibration_drift = 0.0
        self.maintenance_hours = 0.0
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the device"""
        pass
    
    @abstractmethod
    def read(self) -> Any:
        """Read from device"""
        pass
    
    @abstractmethod
    def write(self, data: Any) -> bool:
        """Write to device"""
        pass
    
    def get_info(self) -> DeviceInfo:
        """Get device information"""
        return self.device_info
    
    def get_status(self) -> DeviceStatus:
        """Get current device status"""
        return self.status
    
    def set_status(self, status: DeviceStatus):
        """Set device status"""
        if self.status != status:
            logger.info(f"Device {self.device_info.device_id} status changed: {self.status.value} -> {status.value}")
            self.status = status
    
    def update_maintenance_hours(self, hours: float):
        """Update maintenance hour counter"""
        self.maintenance_hours += hours
        
        # Check maintenance schedule
        if 'maintenance_interval_hours' in self.config.maintenance_schedule:
            interval = self.config.maintenance_schedule['maintenance_interval_hours']
            if self.maintenance_hours >= interval:
                logger.warning(f"Device {self.device_info.device_id} requires maintenance")
                self.set_status(DeviceStatus.MAINTENANCE)


class SensorDevice(MFCDevice):
    """Generic sensor device implementation"""
    
    def __init__(self, device_info: DeviceInfo, config: DeviceConfiguration):
        super().__init__(device_info, config)
        self.sensor_value = 0.0
        self.noise_level = config.parameters.get('noise_level', 0.01)
        self.drift_rate = config.parameters.get('drift_rate', 0.001)  # %/hour
        
    def initialize(self) -> bool:
        """Initialize sensor"""
        try:
            # Simulate initialization sequence
            time.sleep(0.1)  # Startup delay
            self.set_status(DeviceStatus.ONLINE)
            logger.info(f"Sensor {self.device_info.device_id} initialized")
            return True
        except Exception as e:
            logger.error(f"Sensor initialization failed: {e}")
            self.set_status(DeviceStatus.ERROR)
            return False
    
    def read(self) -> float:
        """Read sensor value"""
        if self.status != DeviceStatus.ONLINE:
            logger.warning(f"Attempting to read from offline sensor {self.device_info.device_id}")
            return float('nan')
        
        try:
            # Apply calibration
            calibration_offset = self.config.calibration_data.get('offset', 0.0)
            calibration_gain = self.config.calibration_data.get('gain', 1.0)
            
            # Add noise
            noise = np.random.normal(0, self.noise_level)
            
            # Add drift over time
            hours_since_calibration = (time.time() - self.last_update) / 3600
            drift = self.calibration_drift + (self.drift_rate * hours_since_calibration)
            
            # Calculate final value
            raw_value = self.sensor_value + noise + drift
            calibrated_value = (raw_value + calibration_offset) * calibration_gain
            
            self.last_update = time.time()
            
            # Check limits
            if 'measurement_range' in self.config.limits:
                min_val, max_val = self.config.limits['measurement_range']
                if not (min_val <= calibrated_value <= max_val):
                    logger.warning(f"Sensor {self.device_info.device_id} reading out of range: {calibrated_value}")
                    self.error_count += 1
            
            return calibrated_value
            
        except Exception as e:
            logger.error(f"Sensor read error: {e}")
            self.error_count += 1
            self.set_status(DeviceStatus.ERROR)
            return float('nan')
    
    def write(self, data: Any) -> bool:
        """Set sensor value (for simulation)"""
        try:
            self.sensor_value = float(data)
            return True
        except (ValueError, TypeError):
            return False
    
    def calibrate(self) -> bool:
        """Calibrate sensor"""
        if self.status == DeviceStatus.ONLINE:
            self.set_status(DeviceStatus.CALIBRATING)
            
            # Simulate calibration process
            time.sleep(2.0)  # Calibration time
            
            # Reset drift
            self.calibration_drift = 0.0
            
            # Update calibration timestamp
            self.device_info.last_calibration = time.strftime('%Y-%m-%d %H:%M:%S')
            
            self.set_status(DeviceStatus.ONLINE)
            logger.info(f"Sensor {self.device_info.device_id} calibrated")
            return True
        
        return False


class ActuatorDevice(MFCDevice):
    """Generic actuator device implementation"""
    
    def __init__(self, device_info: DeviceInfo, config: DeviceConfiguration):
        super().__init__(device_info, config)
        self.output_value = 0.0
        self.target_value = 0.0
        self.slew_rate = config.parameters.get('slew_rate', 1.0)  # units/second
        self.deadband = config.parameters.get('deadband', 0.01)
        
    def initialize(self) -> bool:
        """Initialize actuator"""
        try:
            # Simulate initialization
            time.sleep(0.1)
            self.output_value = 0.0
            self.target_value = 0.0
            self.set_status(DeviceStatus.ONLINE)
            logger.info(f"Actuator {self.device_info.device_id} initialized")
            return True
        except Exception as e:
            logger.error(f"Actuator initialization failed: {e}")
            self.set_status(DeviceStatus.ERROR)
            return False
    
    def read(self) -> float:
        """Read current actuator position"""
        return self.output_value
    
    def write(self, data: float) -> bool:
        """Set actuator target value"""
        if self.status != DeviceStatus.ONLINE:
            logger.warning(f"Attempting to write to offline actuator {self.device_info.device_id}")
            return False
        
        try:
            # Check limits
            if 'output_range' in self.config.limits:
                min_val, max_val = self.config.limits['output_range']
                data = np.clip(data, min_val, max_val)
            
            # Apply deadband
            if abs(data - self.target_value) > self.deadband:
                self.target_value = data
                logger.debug(f"Actuator {self.device_info.device_id} target set to {data}")
            
            return True
            
        except Exception as e:
            logger.error(f"Actuator write error: {e}")
            self.error_count += 1
            return False
    
    def update(self, dt: float):
        """Update actuator position based on slew rate"""
        if self.target_value != self.output_value:
            # Calculate change based on slew rate
            max_change = self.slew_rate * dt
            error = self.target_value - self.output_value
            
            if abs(error) <= max_change:
                self.output_value = self.target_value
            else:
                self.output_value += np.sign(error) * max_change


class PowerDevice(MFCDevice):
    """Power supply/management device"""
    
    def __init__(self, device_info: DeviceInfo, config: DeviceConfiguration):
        super().__init__(device_info, config)
        self.voltage_output = 0.0
        self.current_output = 0.0
        self.power_limit = config.parameters.get('power_limit', 100.0)  # Watts
        self.efficiency = config.parameters.get('efficiency', 0.95)
        
    def initialize(self) -> bool:
        """Initialize power device"""
        try:
            self.voltage_output = self.config.parameters.get('nominal_voltage', 12.0)
            self.current_output = 0.0
            self.set_status(DeviceStatus.ONLINE)
            logger.info(f"Power device {self.device_info.device_id} initialized")
            return True
        except Exception as e:
            logger.error(f"Power device initialization failed: {e}")
            self.set_status(DeviceStatus.ERROR)
            return False
    
    def read(self) -> Dict[str, float]:
        """Read power parameters"""
        return {
            'voltage': self.voltage_output,
            'current': self.current_output,
            'power': self.voltage_output * self.current_output
        }
    
    def write(self, data: Dict[str, float]) -> bool:
        """Set power parameters"""
        try:
            if 'voltage' in data:
                voltage_range = self.config.limits.get('voltage_range', (0, 50))
                self.voltage_output = np.clip(data['voltage'], *voltage_range)
            
            if 'current' in data:
                current_range = self.config.limits.get('current_range', (0, 10))
                self.current_output = np.clip(data['current'], *current_range)
            
            # Check power limit
            power = self.voltage_output * self.current_output
            if power > self.power_limit:
                scale_factor = self.power_limit / power
                self.current_output *= scale_factor
                logger.warning(f"Power limited to {self.power_limit}W")
            
            return True
            
        except Exception as e:
            logger.error(f"Power device write error: {e}")
            return False


class MFCControlInterface:
    """High-level interface for MFC control operations"""
    
    def __init__(self):
        self.devices = {}
        self.device_groups = {}
        self.control_loops = {}
        
    def add_device(self, device: MFCDevice, group: Optional[str] = None):
        """Add a device to the interface"""
        self.devices[device.device_info.device_id] = device
        
        if group:
            if group not in self.device_groups:
                self.device_groups[group] = []
            self.device_groups[group].append(device.device_info.device_id)
        
        logger.info(f"Added device {device.device_info.device_id} to MFC interface")
    
    def get_device(self, device_id: str) -> Optional[MFCDevice]:
        """Get device by ID"""
        return self.devices.get(device_id)
    
    def get_devices_by_group(self, group: str) -> List[MFCDevice]:
        """Get all devices in a group"""
        if group not in self.device_groups:
            return []
        
        return [self.devices[device_id] for device_id in self.device_groups[group]
                if device_id in self.devices]
    
    def read_sensor(self, sensor_id: str) -> Optional[float]:
        """Read from a sensor"""
        device = self.get_device(sensor_id)
        if device and isinstance(device, SensorDevice):
            return device.read()
        return None
    
    def write_actuator(self, actuator_id: str, value: float) -> bool:
        """Write to an actuator"""
        device = self.get_device(actuator_id)
        if device and isinstance(device, ActuatorDevice):
            return device.write(value)
        return False
    
    def read_power_status(self, power_id: str) -> Optional[Dict[str, float]]:
        """Read power device status"""
        device = self.get_device(power_id)
        if device and isinstance(device, PowerDevice):
            return device.read()
        return None
    
    def set_power_output(self, power_id: str, voltage: float, current: float) -> bool:
        """Set power output"""
        device = self.get_device(power_id)
        if device and isinstance(device, PowerDevice):
            return device.write({'voltage': voltage, 'current': current})
        return False
    
    def initialize_all_devices(self) -> Dict[str, bool]:
        """Initialize all devices"""
        results = {}
        for device_id, device in self.devices.items():
            results[device_id] = device.initialize()
        
        online_count = sum(1 for success in results.values() if success)
        logger.info(f"Initialized {online_count}/{len(self.devices)} devices")
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        status_counts = {}
        device_statuses = {}
        
        for device_id, device in self.devices.items():
            status = device.get_status()
            device_statuses[device_id] = {
                'status': status.value,
                'type': device.device_info.device_type.value,
                'error_count': device.error_count,
                'maintenance_hours': device.maintenance_hours
            }
            
            status_counts[status.value] = status_counts.get(status.value, 0) + 1
        
        return {
            'device_count': len(self.devices),
            'status_summary': status_counts,
            'devices': device_statuses,
            'system_health': self._calculate_system_health()
        }
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health (0-1)"""
        if not self.devices:
            return 0.0
        
        online_devices = sum(1 for device in self.devices.values() 
                           if device.get_status() == DeviceStatus.ONLINE)
        
        return online_devices / len(self.devices)
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics"""
        diagnostics = {
            'timestamp': time.time(),
            'devices': {},
            'performance': {},
            'recommendations': []
        }
        
        for device_id, device in self.devices.items():
            device_diag = {
                'status': device.get_status().value,
                'error_count': device.error_count,
                'last_update': device.last_update,
                'maintenance_hours': device.maintenance_hours
            }
            
            # Add device-specific diagnostics
            if hasattr(device, 'get_diagnostics'):
                device_diag.update(device.get_diagnostics())
            
            diagnostics['devices'][device_id] = device_diag
            
            # Generate recommendations
            if device.error_count > 10:
                diagnostics['recommendations'].append(
                    f"Device {device_id} has high error count - check connections"
                )
            
            if device.get_status() == DeviceStatus.MAINTENANCE:
                diagnostics['recommendations'].append(
                    f"Device {device_id} requires maintenance"
                )
        
        # System-wide performance metrics
        diagnostics['performance'] = {
            'system_health': self._calculate_system_health(),
            'total_devices': len(self.devices),
            'online_devices': sum(1 for d in self.devices.values() 
                                if d.get_status() == DeviceStatus.ONLINE)
        }
        
        return diagnostics
    
    def calibrate_sensors(self, sensor_group: Optional[str] = None) -> Dict[str, bool]:
        """Calibrate sensors in a group or all sensors"""
        results = {}
        
        if sensor_group:
            devices_to_calibrate = self.get_devices_by_group(sensor_group)
        else:
            devices_to_calibrate = [d for d in self.devices.values() 
                                  if isinstance(d, SensorDevice)]
        
        for device in devices_to_calibrate:
            if hasattr(device, 'calibrate'):
                results[device.device_info.device_id] = device.calibrate()
            else:
                results[device.device_info.device_id] = False
        
        return results


class HardwareAbstractionLayer:
    """Main hardware abstraction layer for MFC systems"""
    
    def __init__(self):
        self.control_interface = MFCControlInterface()
        self.device_drivers = {}
        self.configuration_manager = ConfigurationManager()
        
    def register_driver(self, device_type: str, driver_class):
        """Register a device driver"""
        self.device_drivers[device_type] = driver_class
        logger.info(f"Registered driver for {device_type}")
    
    def create_device_from_config(self, config_path: str) -> Optional[MFCDevice]:
        """Create device from configuration file"""
        try:
            config = self.configuration_manager.load_device_config(config_path)
            device_type = config.get('device_type', 'unknown')
            
            if device_type in self.device_drivers:
                driver_class = self.device_drivers[device_type]
                device = driver_class(config)
                self.control_interface.add_device(device)
                return device
            else:
                logger.error(f"No driver registered for device type: {device_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create device from config {config_path}: {e}")
            return None
    
    def get_control_interface(self) -> MFCControlInterface:
        """Get the main control interface"""
        return self.control_interface


class ConfigurationManager:
    """Manage device configurations"""
    
    def __init__(self):
        self.configurations = {}
    
    def load_device_config(self, config_path: str) -> Dict[str, Any]:
        """Load device configuration from file"""
        # Placeholder implementation
        # In a real system, this would load from JSON/YAML files
        return {
            'device_type': 'sensor',
            'parameters': {},
            'limits': {},
            'calibration_data': {}
        }
    
    def save_device_config(self, device_id: str, config: Dict[str, Any]):
        """Save device configuration"""
        self.configurations[device_id] = config
    
    def get_device_config(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get device configuration"""
        return self.configurations.get(device_id)


def create_mfc_hardware_system() -> HardwareAbstractionLayer:
    """Create a complete MFC hardware system"""
    
    hal = HardwareAbstractionLayer()
    
    # Register standard device drivers
    hal.register_driver('pH_sensor', SensorDevice)
    hal.register_driver('temperature_sensor', SensorDevice)
    hal.register_driver('flow_sensor', SensorDevice)
    hal.register_driver('pump_actuator', ActuatorDevice)
    hal.register_driver('valve_actuator', ActuatorDevice)
    hal.register_driver('power_supply', PowerDevice)
    
    # Create standard MFC devices
    devices = []
    
    # pH sensor
    ph_info = DeviceInfo(
        device_id="pH_001",
        device_type=DeviceType.SENSOR,
        name="pH Sensor",
        manufacturer="Generic",
        model="pH-100",
        firmware_version="1.0.0",
        hardware_revision="A",
        serial_number="PH123456",
        installation_date="2024-01-01",
        last_calibration="2024-01-01"
    )
    
    ph_config = DeviceConfiguration(
        device_id="pH_001",
        parameters={'noise_level': 0.01, 'drift_rate': 0.001},
        limits={'measurement_range': (0.0, 14.0)},
        calibration_data={'offset': 0.0, 'gain': 1.0},
        maintenance_schedule={'maintenance_interval_hours': 8760}
    )
    
    ph_sensor = SensorDevice(ph_info, ph_config)
    devices.append(ph_sensor)
    
    # Temperature sensor
    temp_info = DeviceInfo(
        device_id="TEMP_001",
        device_type=DeviceType.SENSOR,
        name="Temperature Sensor",
        manufacturer="Generic",
        model="T-100",
        firmware_version="1.0.0",
        hardware_revision="A",
        serial_number="T123456",
        installation_date="2024-01-01",
        last_calibration="2024-01-01"
    )
    
    temp_config = DeviceConfiguration(
        device_id="TEMP_001",
        parameters={'noise_level': 0.1, 'drift_rate': 0.0005},
        limits={'measurement_range': (0.0, 100.0)},
        calibration_data={'offset': 0.0, 'gain': 1.0},
        maintenance_schedule={'maintenance_interval_hours': 8760}
    )
    
    temp_sensor = SensorDevice(temp_info, temp_config)
    devices.append(temp_sensor)
    
    # Pump actuator
    pump_info = DeviceInfo(
        device_id="PUMP_001",
        device_type=DeviceType.ACTUATOR,
        name="Peristaltic Pump",
        manufacturer="Generic",
        model="P-100",
        firmware_version="1.0.0",
        hardware_revision="A",
        serial_number="P123456",
        installation_date="2024-01-01",
        last_calibration="2024-01-01"
    )
    
    pump_config = DeviceConfiguration(
        device_id="PUMP_001",
        parameters={'slew_rate': 5.0, 'deadband': 0.1},
        limits={'output_range': (0.0, 100.0)},
        calibration_data={'offset': 0.0, 'gain': 1.0},
        maintenance_schedule={'maintenance_interval_hours': 4380}
    )
    
    pump_actuator = ActuatorDevice(pump_info, pump_config)
    devices.append(pump_actuator)
    
    # Power supply
    power_info = DeviceInfo(
        device_id="PSU_001",
        device_type=DeviceType.POWER,
        name="DC Power Supply",
        manufacturer="Generic",
        model="PSU-100",
        firmware_version="1.0.0",
        hardware_revision="A",
        serial_number="PSU123456",
        installation_date="2024-01-01",
        last_calibration="2024-01-01"
    )
    
    power_config = DeviceConfiguration(
        device_id="PSU_001",
        parameters={'power_limit': 100.0, 'efficiency': 0.95, 'nominal_voltage': 12.0},
        limits={'voltage_range': (0.0, 24.0), 'current_range': (0.0, 10.0)},
        calibration_data={'offset': 0.0, 'gain': 1.0},
        maintenance_schedule={'maintenance_interval_hours': 8760}
    )
    
    power_device = PowerDevice(power_info, power_config)
    devices.append(power_device)
    
    # Add all devices to the HAL
    for device in devices:
        hal.control_interface.add_device(device)
    
    # Create device groups
    hal.control_interface.device_groups['sensors'] = ['pH_001', 'TEMP_001']
    hal.control_interface.device_groups['actuators'] = ['PUMP_001']
    hal.control_interface.device_groups['power'] = ['PSU_001']
    
    return hal


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create MFC hardware system
    hal = create_mfc_hardware_system()
    
    print("MFC Hardware Abstraction Layer Test")
    print("=" * 40)
    
    # Initialize all devices
    init_results = hal.control_interface.initialize_all_devices()
    print(f"Device initialization results: {init_results}")
    
    # Get system status
    status = hal.control_interface.get_system_status()
    print(f"\nSystem Status:")
    print(f"Total devices: {status['device_count']}")
    print(f"System health: {status['system_health']:.1%}")
    print(f"Status summary: {status['status_summary']}")
    
    # Test sensor readings
    print(f"\nSensor Readings:")
    ph_value = hal.control_interface.read_sensor('pH_001')
    temp_value = hal.control_interface.read_sensor('TEMP_001')
    print(f"pH: {ph_value:.2f}")
    print(f"Temperature: {temp_value:.1f}°C")
    
    # Test actuator control
    print(f"\nActuator Control:")
    success = hal.control_interface.write_actuator('PUMP_001', 50.0)
    print(f"Pump set to 50%: {success}")
    
    # Test power management
    print(f"\nPower Management:")
    power_success = hal.control_interface.set_power_output('PSU_001', 12.0, 2.0)
    power_status = hal.control_interface.read_power_status('PSU_001')
    print(f"Power set to 12V, 2A: {power_success}")
    print(f"Power status: {power_status}")
    
    # Run diagnostics
    print(f"\nSystem Diagnostics:")
    diagnostics = hal.control_interface.run_diagnostics()
    print(f"System health: {diagnostics['performance']['system_health']:.1%}")
    if diagnostics['recommendations']:
        print("Recommendations:")
        for rec in diagnostics['recommendations']:
            print(f"  - {rec}")
    else:
        print("No recommendations - system operating normally")
    
    # Test calibration
    print(f"\nSensor Calibration:")
    cal_results = hal.control_interface.calibrate_sensors('sensors')
    print(f"Calibration results: {cal_results}")