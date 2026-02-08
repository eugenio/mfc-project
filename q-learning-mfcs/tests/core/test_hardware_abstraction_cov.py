"""Tests for hardware_abstraction.py - 98%+ coverage target.

Covers HardwareAbstractionLayer, MFCControlInterface, SensorDevice,
ActuatorDevice, PowerDevice, ConfigurationManager, and helpers.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from controller_models.hardware_abstraction import (
    ActuatorDevice,
    ConfigurationManager,
    DeviceConfiguration,
    DeviceInfo,
    DeviceStatus,
    DeviceType,
    HardwareAbstractionLayer,
    MFCControlInterface,
    MFCDevice,
    PowerDevice,
    SensorDevice,
    create_mfc_hardware_system,
)


def _make_device_info(device_id="DEV_001", dtype=DeviceType.SENSOR):
    return DeviceInfo(
        device_id=device_id, device_type=dtype, name="Test",
        manufacturer="Test", model="T-1", firmware_version="1.0",
        hardware_revision="A", serial_number="SN001",
        installation_date="2024-01-01", last_calibration="2024-01-01",
    )


def _make_device_config(device_id="DEV_001", params=None, limits=None):
    return DeviceConfiguration(
        device_id=device_id,
        parameters=params or {"noise_level": 0.01, "drift_rate": 0.001},
        limits=limits or {"measurement_range": (0.0, 100.0)},
        calibration_data={"offset": 0.0, "gain": 1.0},
        maintenance_schedule={"maintenance_interval_hours": 100},
    )


class TestEnums:
    def test_device_type(self):
        assert DeviceType.SENSOR.value == "sensor"
        assert DeviceType.ACTUATOR.value == "actuator"
        assert DeviceType.CONTROLLER.value == "controller"
        assert DeviceType.COMMUNICATION.value == "communication"
        assert DeviceType.POWER.value == "power"

    def test_device_status(self):
        assert DeviceStatus.OFFLINE.value == "offline"
        assert DeviceStatus.ONLINE.value == "online"
        assert DeviceStatus.ERROR.value == "error"
        assert DeviceStatus.MAINTENANCE.value == "maintenance"
        assert DeviceStatus.CALIBRATING.value == "calibrating"


class TestMFCDeviceBase:
    def test_get_info(self):
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        assert sensor.get_info() == info

    def test_get_status(self):
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        assert sensor.get_status() == DeviceStatus.OFFLINE

    def test_set_status(self):
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        sensor.set_status(DeviceStatus.ONLINE)
        assert sensor.status == DeviceStatus.ONLINE

    def test_set_status_same(self):
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        sensor.set_status(DeviceStatus.OFFLINE)
        assert sensor.status == DeviceStatus.OFFLINE

    def test_update_maintenance_hours(self):
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        sensor.update_maintenance_hours(50)
        assert sensor.maintenance_hours == 50
        sensor.update_maintenance_hours(60)
        assert sensor.status == DeviceStatus.MAINTENANCE

    def test_update_maintenance_no_interval(self):
        info = _make_device_info()
        config = DeviceConfiguration(
            device_id="DEV_001", parameters={},
            limits={}, calibration_data={}, maintenance_schedule={},
        )
        sensor = SensorDevice(info, config)
        sensor.update_maintenance_hours(1000)
        assert sensor.maintenance_hours == 1000


class TestSensorDevice:
    @patch("time.sleep")
    def test_initialize(self, mock_sleep):
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        result = sensor.initialize()
        assert result is True
        assert sensor.status == DeviceStatus.ONLINE

    def test_read_offline(self):
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        result = sensor.read()
        assert np.isnan(result)

    def test_read_online(self):
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        sensor.status = DeviceStatus.ONLINE
        sensor.sensor_value = 50.0
        sensor.last_update = time.time()
        result = sensor.read()
        assert isinstance(result, float)

    def test_read_out_of_range(self):
        info = _make_device_info()
        config = _make_device_config(
            limits={"measurement_range": (0.0, 10.0)},
        )
        sensor = SensorDevice(info, config)
        sensor.status = DeviceStatus.ONLINE
        sensor.sensor_value = 20.0
        sensor.last_update = time.time()
        result = sensor.read()
        assert sensor.error_count > 0

    def test_read_no_limits(self):
        info = _make_device_info()
        config = DeviceConfiguration(
            device_id="DEV_001", parameters={"noise_level": 0.001},
            limits={}, calibration_data={"offset": 0.0, "gain": 1.0},
            maintenance_schedule={},
        )
        sensor = SensorDevice(info, config)
        sensor.status = DeviceStatus.ONLINE
        sensor.sensor_value = 5.0
        sensor.last_update = time.time()
        result = sensor.read()
        assert isinstance(result, float)

    def test_write(self):
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        assert sensor.write(42.0) is True
        assert sensor.sensor_value == 42.0

    def test_write_invalid(self):
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        assert sensor.write("not_a_number") is False

    @patch("time.sleep")
    def test_calibrate_online(self, mock_sleep):
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        sensor.status = DeviceStatus.ONLINE
        result = sensor.calibrate()
        assert result is True
        assert sensor.calibration_drift == 0.0

    def test_calibrate_offline(self):
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        result = sensor.calibrate()
        assert result is False

    def test_initialize_exception(self):
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        with patch("time.sleep", side_effect=RuntimeError("fail")):
            result = sensor.initialize()
            assert result is False
            assert sensor.status == DeviceStatus.ERROR

    def test_read_exception(self):
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        sensor.status = DeviceStatus.ONLINE
        sensor.last_update = time.time()
        # Force exception in read via patching config calibration_data as object
        original_cal = sensor.config.calibration_data
        sensor.config.calibration_data = "not_a_dict"
        result = sensor.read()
        assert np.isnan(result)
        assert sensor.status == DeviceStatus.ERROR
        sensor.config.calibration_data = original_cal


class TestActuatorDevice:
    @patch("time.sleep")
    def test_initialize(self, mock_sleep):
        info = _make_device_info(dtype=DeviceType.ACTUATOR)
        config = _make_device_config(params={"slew_rate": 5.0, "deadband": 0.1})
        act = ActuatorDevice(info, config)
        assert act.initialize() is True
        assert act.status == DeviceStatus.ONLINE

    def test_read(self):
        info = _make_device_info(dtype=DeviceType.ACTUATOR)
        config = _make_device_config(params={"slew_rate": 5.0, "deadband": 0.1})
        act = ActuatorDevice(info, config)
        act.output_value = 10.0
        assert act.read() == 10.0

    def test_write_online(self):
        info = _make_device_info(dtype=DeviceType.ACTUATOR)
        config = _make_device_config(
            params={"slew_rate": 5.0, "deadband": 0.1},
            limits={"output_range": (0.0, 100.0)},
        )
        act = ActuatorDevice(info, config)
        act.status = DeviceStatus.ONLINE
        assert act.write(50.0) is True
        assert act.target_value == 50.0

    def test_write_offline(self):
        info = _make_device_info(dtype=DeviceType.ACTUATOR)
        config = _make_device_config(params={"slew_rate": 5.0, "deadband": 0.1})
        act = ActuatorDevice(info, config)
        assert act.write(50.0) is False

    def test_write_deadband(self):
        info = _make_device_info(dtype=DeviceType.ACTUATOR)
        config = _make_device_config(params={"slew_rate": 5.0, "deadband": 1.0})
        act = ActuatorDevice(info, config)
        act.status = DeviceStatus.ONLINE
        act.target_value = 50.0
        act.write(50.5)
        assert act.target_value == 50.0

    def test_write_clamp(self):
        info = _make_device_info(dtype=DeviceType.ACTUATOR)
        config = _make_device_config(
            params={"slew_rate": 5.0, "deadband": 0.1},
            limits={"output_range": (0.0, 100.0)},
        )
        act = ActuatorDevice(info, config)
        act.status = DeviceStatus.ONLINE
        act.write(200.0)

    def test_write_no_limits(self):
        info = _make_device_info(dtype=DeviceType.ACTUATOR)
        config = _make_device_config(
            params={"slew_rate": 5.0, "deadband": 0.1},
            limits={},
        )
        act = ActuatorDevice(info, config)
        act.status = DeviceStatus.ONLINE
        assert act.write(50.0) is True

    @patch("time.sleep")
    def test_initialize_exception(self, mock_sleep):
        info = _make_device_info(dtype=DeviceType.ACTUATOR)
        config = _make_device_config(params={"slew_rate": 5.0, "deadband": 0.1})
        act = ActuatorDevice(info, config)
        with patch("time.sleep", side_effect=RuntimeError("fail")):
            result = act.initialize()
            assert result is False
            assert act.status == DeviceStatus.ERROR

    def test_write_exception(self):
        info = _make_device_info(dtype=DeviceType.ACTUATOR)
        config = _make_device_config(
            params={"slew_rate": 5.0, "deadband": 0.1},
            limits={"output_range": "invalid"},
        )
        act = ActuatorDevice(info, config)
        act.status = DeviceStatus.ONLINE
        result = act.write(50.0)
        assert result is False

    def test_update_reach_target(self):
        info = _make_device_info(dtype=DeviceType.ACTUATOR)
        config = _make_device_config(params={"slew_rate": 100.0, "deadband": 0.1})
        act = ActuatorDevice(info, config)
        act.target_value = 10.0
        act.output_value = 9.5
        act.update(dt=1.0)
        assert act.output_value == 10.0

    def test_update_slew_limited(self):
        info = _make_device_info(dtype=DeviceType.ACTUATOR)
        config = _make_device_config(params={"slew_rate": 1.0, "deadband": 0.1})
        act = ActuatorDevice(info, config)
        act.target_value = 10.0
        act.output_value = 0.0
        act.update(dt=1.0)
        assert act.output_value == 1.0

    def test_update_negative_direction(self):
        info = _make_device_info(dtype=DeviceType.ACTUATOR)
        config = _make_device_config(params={"slew_rate": 1.0, "deadband": 0.1})
        act = ActuatorDevice(info, config)
        act.target_value = 0.0
        act.output_value = 5.0
        act.update(dt=1.0)
        assert act.output_value == 4.0

    def test_update_no_change(self):
        info = _make_device_info(dtype=DeviceType.ACTUATOR)
        config = _make_device_config(params={"slew_rate": 1.0, "deadband": 0.1})
        act = ActuatorDevice(info, config)
        act.target_value = 5.0
        act.output_value = 5.0
        act.update(dt=1.0)
        assert act.output_value == 5.0


class TestPowerDevice:
    def test_initialize(self):
        info = _make_device_info(dtype=DeviceType.POWER)
        config = _make_device_config(
            params={"power_limit": 100.0, "efficiency": 0.95, "nominal_voltage": 12.0},
            limits={"voltage_range": (0.0, 24.0), "current_range": (0.0, 10.0)},
        )
        pd = PowerDevice(info, config)
        assert pd.initialize() is True
        assert pd.voltage_output == 12.0

    def test_read(self):
        info = _make_device_info(dtype=DeviceType.POWER)
        config = _make_device_config(
            params={"power_limit": 100.0, "efficiency": 0.95, "nominal_voltage": 12.0},
        )
        pd = PowerDevice(info, config)
        pd.voltage_output = 12.0
        pd.current_output = 2.0
        result = pd.read()
        assert result["voltage"] == 12.0
        assert result["current"] == 2.0
        assert result["power"] == 24.0

    def test_write_voltage_and_current(self):
        info = _make_device_info(dtype=DeviceType.POWER)
        config = _make_device_config(
            params={"power_limit": 100.0, "efficiency": 0.95},
            limits={"voltage_range": (0.0, 24.0), "current_range": (0.0, 10.0)},
        )
        pd = PowerDevice(info, config)
        assert pd.write({"voltage": 12.0, "current": 2.0}) is True

    def test_write_power_limited(self):
        info = _make_device_info(dtype=DeviceType.POWER)
        config = _make_device_config(
            params={"power_limit": 50.0, "efficiency": 0.95},
            limits={"voltage_range": (0.0, 24.0), "current_range": (0.0, 10.0)},
        )
        pd = PowerDevice(info, config)
        pd.write({"voltage": 20.0, "current": 10.0})
        assert pd.voltage_output * pd.current_output <= 50.1

    def test_write_no_limits(self):
        info = _make_device_info(dtype=DeviceType.POWER)
        config = _make_device_config(
            params={"power_limit": 100.0},
            limits={},
        )
        pd = PowerDevice(info, config)
        assert pd.write({"voltage": 12.0}) is True

    def test_initialize_exception(self):
        info = _make_device_info(dtype=DeviceType.POWER)
        config = _make_device_config(params={})
        pd = PowerDevice(info, config)
        # Replace parameters with a broken object to cause exception
        pd.config.parameters = None
        result = pd.initialize()
        assert result is False
        assert pd.status == DeviceStatus.ERROR

    def test_write_exception(self):
        info = _make_device_info(dtype=DeviceType.POWER)
        config = _make_device_config(
            params={"power_limit": 100.0},
            limits={"voltage_range": "invalid"},
        )
        pd = PowerDevice(info, config)
        result = pd.write({"voltage": 12.0})
        assert result is False

    def test_run_diagnostics_device_with_get_diagnostics(self):
        iface = MFCControlInterface()
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        sensor.get_diagnostics = lambda: {"custom": True}
        iface.add_device(sensor)
        diag = iface.run_diagnostics()
        assert diag["devices"]["DEV_001"].get("custom") is True

    def test_calibrate_no_calibrate_method(self):
        iface = MFCControlInterface()
        info = _make_device_info(dtype=DeviceType.ACTUATOR)
        config = _make_device_config(params={"slew_rate": 1.0, "deadband": 0.1})
        act = ActuatorDevice(info, config)
        iface.add_device(act, group="other")
        results = iface.calibrate_sensors("other")
        assert results["DEV_001"] is False


class TestMFCControlInterface:
    def test_add_device(self):
        iface = MFCControlInterface()
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        iface.add_device(sensor, group="sensors")
        assert "DEV_001" in iface.devices
        assert "DEV_001" in iface.device_groups["sensors"]

    def test_add_device_no_group(self):
        iface = MFCControlInterface()
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        iface.add_device(sensor)
        assert "DEV_001" in iface.devices

    def test_get_device(self):
        iface = MFCControlInterface()
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        iface.add_device(sensor)
        assert iface.get_device("DEV_001") is sensor
        assert iface.get_device("NONEXISTENT") is None

    def test_get_devices_by_group(self):
        iface = MFCControlInterface()
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        iface.add_device(sensor, group="sensors")
        result = iface.get_devices_by_group("sensors")
        assert len(result) == 1
        assert iface.get_devices_by_group("other") == []

    def test_read_sensor(self):
        iface = MFCControlInterface()
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        sensor.status = DeviceStatus.ONLINE
        sensor.last_update = time.time()
        iface.add_device(sensor)
        result = iface.read_sensor("DEV_001")
        assert isinstance(result, float)

    def test_read_sensor_not_found(self):
        iface = MFCControlInterface()
        assert iface.read_sensor("NONEXISTENT") is None

    def test_read_sensor_wrong_type(self):
        iface = MFCControlInterface()
        info = _make_device_info(dtype=DeviceType.ACTUATOR)
        config = _make_device_config(params={"slew_rate": 1.0, "deadband": 0.1})
        act = ActuatorDevice(info, config)
        iface.add_device(act)
        assert iface.read_sensor("DEV_001") is None

    def test_write_actuator(self):
        iface = MFCControlInterface()
        info = _make_device_info(dtype=DeviceType.ACTUATOR)
        config = _make_device_config(params={"slew_rate": 1.0, "deadband": 0.1})
        act = ActuatorDevice(info, config)
        act.status = DeviceStatus.ONLINE
        iface.add_device(act)
        assert iface.write_actuator("DEV_001", 50.0) is True

    def test_write_actuator_not_found(self):
        iface = MFCControlInterface()
        assert iface.write_actuator("NONEXISTENT", 50.0) is False

    def test_read_power_status(self):
        iface = MFCControlInterface()
        info = _make_device_info(dtype=DeviceType.POWER)
        config = _make_device_config(
            params={"power_limit": 100.0, "efficiency": 0.95, "nominal_voltage": 12.0},
        )
        pd = PowerDevice(info, config)
        pd.voltage_output = 12.0
        iface.add_device(pd)
        result = iface.read_power_status("DEV_001")
        assert result is not None
        assert "voltage" in result

    def test_read_power_status_not_found(self):
        iface = MFCControlInterface()
        assert iface.read_power_status("NONEXISTENT") is None

    def test_set_power_output(self):
        iface = MFCControlInterface()
        info = _make_device_info(dtype=DeviceType.POWER)
        config = _make_device_config(
            params={"power_limit": 100.0, "efficiency": 0.95},
            limits={"voltage_range": (0.0, 24.0), "current_range": (0.0, 10.0)},
        )
        pd = PowerDevice(info, config)
        iface.add_device(pd)
        assert iface.set_power_output("DEV_001", 12.0, 2.0) is True

    def test_set_power_output_not_found(self):
        iface = MFCControlInterface()
        assert iface.set_power_output("NONEXISTENT", 12.0, 2.0) is False

    @patch("time.sleep")
    def test_initialize_all_devices(self, mock_sleep):
        iface = MFCControlInterface()
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        iface.add_device(sensor)
        results = iface.initialize_all_devices()
        assert results["DEV_001"] is True

    def test_get_system_status(self):
        iface = MFCControlInterface()
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        iface.add_device(sensor)
        status = iface.get_system_status()
        assert "device_count" in status
        assert "system_health" in status
        assert status["device_count"] == 1

    def test_system_health_empty(self):
        iface = MFCControlInterface()
        assert iface._calculate_system_health() == 0.0

    def test_system_health_online(self):
        iface = MFCControlInterface()
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        sensor.status = DeviceStatus.ONLINE
        iface.add_device(sensor)
        assert iface._calculate_system_health() == 1.0

    def test_run_diagnostics(self):
        iface = MFCControlInterface()
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        sensor.error_count = 15
        iface.add_device(sensor)
        diag = iface.run_diagnostics()
        assert "recommendations" in diag
        assert any("high error count" in r for r in diag["recommendations"])

    def test_run_diagnostics_maintenance(self):
        iface = MFCControlInterface()
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        sensor.status = DeviceStatus.MAINTENANCE
        iface.add_device(sensor)
        diag = iface.run_diagnostics()
        assert any("requires maintenance" in r for r in diag["recommendations"])

    @patch("time.sleep")
    def test_calibrate_sensors_group(self, mock_sleep):
        iface = MFCControlInterface()
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        sensor.status = DeviceStatus.ONLINE
        iface.add_device(sensor, group="sensors")
        results = iface.calibrate_sensors("sensors")
        assert results["DEV_001"] is True

    @patch("time.sleep")
    def test_calibrate_sensors_all(self, mock_sleep):
        iface = MFCControlInterface()
        info = _make_device_info()
        config = _make_device_config()
        sensor = SensorDevice(info, config)
        sensor.status = DeviceStatus.ONLINE
        iface.add_device(sensor)
        results = iface.calibrate_sensors()
        assert results["DEV_001"] is True


class TestHardwareAbstractionLayer:
    def test_init(self):
        hal = HardwareAbstractionLayer()
        assert hal.control_interface is not None
        assert isinstance(hal.configuration_manager, ConfigurationManager)

    def test_register_driver(self):
        hal = HardwareAbstractionLayer()
        hal.register_driver("test_sensor", SensorDevice)
        assert "test_sensor" in hal.device_drivers

    def test_get_control_interface(self):
        hal = HardwareAbstractionLayer()
        iface = hal.get_control_interface()
        assert isinstance(iface, MFCControlInterface)

    def test_create_device_from_config_success(self):
        hal = HardwareAbstractionLayer()

        class FakeDriver:
            def __init__(self, config):
                self.device_info = _make_device_info("FAKE_001")

        hal.register_driver("sensor", FakeDriver)
        device = hal.create_device_from_config("dummy.json")
        assert device is not None

    def test_create_device_from_config_failure(self):
        hal = HardwareAbstractionLayer()
        hal.register_driver("sensor", SensorDevice)
        device = hal.create_device_from_config("dummy.json")
        # SensorDevice expects DeviceInfo, so it will fail
        assert device is None

    def test_create_device_unknown_type(self):
        hal = HardwareAbstractionLayer()
        device = hal.create_device_from_config("dummy.json")
        assert device is None


class TestConfigurationManager:
    def test_load_device_config(self):
        cm = ConfigurationManager()
        config = cm.load_device_config("test.json")
        assert "device_type" in config

    def test_save_and_get(self):
        cm = ConfigurationManager()
        cm.save_device_config("dev1", {"param": 1})
        result = cm.get_device_config("dev1")
        assert result == {"param": 1}

    def test_get_nonexistent(self):
        cm = ConfigurationManager()
        assert cm.get_device_config("nope") is None


class TestCreateMFCHardwareSystem:
    def test_creates_system(self):
        hal = create_mfc_hardware_system()
        assert isinstance(hal, HardwareAbstractionLayer)
        iface = hal.get_control_interface()
        assert "pH_001" in iface.devices
        assert "TEMP_001" in iface.devices
        assert "PUMP_001" in iface.devices
        assert "PSU_001" in iface.devices

    def test_device_groups(self):
        hal = create_mfc_hardware_system()
        iface = hal.get_control_interface()
        assert "sensors" in iface.device_groups
        assert "actuators" in iface.device_groups
        assert "power" in iface.device_groups

    def test_registered_drivers(self):
        hal = create_mfc_hardware_system()
        assert "pH_sensor" in hal.device_drivers
        assert "pump_actuator" in hal.device_drivers
        assert "power_supply" in hal.device_drivers
