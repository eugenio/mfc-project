"""Tests for control_electronics.py - 98%+ coverage target.

Covers ControlElectronics, MCUSpecs, ADCSpecs, DACSpecs, GPIOSpecs,
CommunicationInterface, ElectronicsMeasurement, and helper functions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from controller_models.control_electronics import (
    ADCSpecs,
    CommunicationInterface,
    ControlElectronics,
    DACSpecs,
    ElectronicsMeasurement,
    GPIOSpecs,
    InterfaceType,
    MCUArchitecture,
    MCUSpecs,
    create_standard_control_electronics,
)


@pytest.fixture
def mcu_specs():
    return MCUSpecs(
        architecture=MCUArchitecture.ARM_CORTEX_M4,
        clock_frequency_mhz=168.0,
        cores=1,
        ram_kb=256.0,
        flash_kb=512.0,
        cache_kb=8.0,
        fpu=True,
        dsp_extensions=True,
        power_consumption_active_mw=100.0,
        power_consumption_sleep_mw=5.0,
        power_consumption_deep_sleep_mw=0.05,
        operating_voltage_v=3.3,
        temperature_range=(-40, 85),
        package_type="LQFP100",
        cost=10.0,
    )


@pytest.fixture
def adc_specs():
    return ADCSpecs(
        resolution_bits=12,
        sampling_rate_ksps=500.0,
        channels=4,
        reference_voltage_v=3.3,
        input_range_v=(0.0, 3.3),
        differential_inputs=False,
        internal_reference=True,
        noise_lsb=1.0,
        linearity_error_lsb=1.5,
        power_consumption_mw=5.0,
        interface_type=InterfaceType.SPI,
        cost=3.0,
    )


@pytest.fixture
def dac_specs():
    return DACSpecs(
        resolution_bits=12,
        update_rate_ksps=100.0,
        channels=2,
        reference_voltage_v=3.3,
        output_range_v=(0.0, 3.3),
        current_drive_ma=10.0,
        settling_time_us=5.0,
        linearity_error_lsb=1.0,
        power_consumption_mw=4.0,
        interface_type=InterfaceType.SPI,
        cost=2.0,
    )


@pytest.fixture
def gpio_specs():
    return GPIOSpecs(
        pins=16,
        voltage_levels_v=[3.3],
        current_drive_ma=10.0,
        input_leakage_ua=0.05,
        switching_frequency_mhz=50.0,
        pull_up_down_available=True,
        interrupt_capable=True,
        analog_capable=False,
        power_consumption_per_pin_uw=5.0,
        cost=0.0,
    )


@pytest.fixture
def comm_interfaces():
    return [
        CommunicationInterface(InterfaceType.SPI, 20.0, 3.0, 1.0, 40.0, 1.0),
        CommunicationInterface(InterfaceType.UART, 1.0, 1.5, 5.0, 30.0, 0.5),
    ]


@pytest.fixture
def electronics(mcu_specs, adc_specs, dac_specs, gpio_specs, comm_interfaces):
    return ControlElectronics(
        mcu_specs, adc_specs, dac_specs, gpio_specs, comm_interfaces,
    )


class TestEnums:
    def test_mcu_architecture_values(self):
        assert MCUArchitecture.ARM_CORTEX_M0.value == "arm_cortex_m0"
        assert MCUArchitecture.ARM_CORTEX_M7.value == "arm_cortex_m7"
        assert MCUArchitecture.RISC_V.value == "risc_v"
        assert MCUArchitecture.X86.value == "x86"
        assert MCUArchitecture.DSP.value == "dsp"
        assert MCUArchitecture.ARM_CORTEX_A.value == "arm_cortex_a"
        assert MCUArchitecture.ARM_CORTEX_M4.value == "arm_cortex_m4"

    def test_interface_type_values(self):
        assert InterfaceType.SPI.value == "spi"
        assert InterfaceType.I2C.value == "i2c"
        assert InterfaceType.UART.value == "uart"
        assert InterfaceType.CAN.value == "can"
        assert InterfaceType.ETHERNET.value == "ethernet"
        assert InterfaceType.USB.value == "usb"
        assert InterfaceType.GPIO.value == "gpio"


class TestDataclasses:
    def test_mcu_specs(self, mcu_specs):
        assert mcu_specs.architecture == MCUArchitecture.ARM_CORTEX_M4
        assert mcu_specs.clock_frequency_mhz == 168.0
        assert mcu_specs.cost == 10.0

    def test_adc_specs(self, adc_specs):
        assert adc_specs.resolution_bits == 12
        assert adc_specs.channels == 4

    def test_dac_specs(self, dac_specs):
        assert dac_specs.resolution_bits == 12
        assert dac_specs.channels == 2

    def test_gpio_specs(self, gpio_specs):
        assert gpio_specs.pins == 16

    def test_comm_interface(self):
        ci = CommunicationInterface(InterfaceType.CAN, 1.0, 8.0, 100.0, 45.0, 5.0)
        assert ci.interface_type == InterfaceType.CAN
        assert ci.speed_mbps == 1.0

    def test_electronics_measurement(self):
        m = ElectronicsMeasurement(
            timestamp=1000.0,
            mcu_temperature_c=30.0,
            power_consumption_mw=120.0,
            cpu_utilization_pct=50.0,
            memory_usage_pct=40.0,
            adc_readings={0: 1.5},
            dac_outputs={0: 2.0},
            gpio_states={0: True},
            communication_activity={InterfaceType.SPI: 10.0},
            fault_flags=[],
        )
        assert m.mcu_temperature_c == 30.0
        assert m.power_consumption_mw == 120.0


class TestControlElectronicsInit:
    def test_init_state(self, electronics):
        assert electronics.mcu_temperature == 25.0
        assert electronics.cpu_utilization == 0.0
        assert electronics.memory_usage == 0.0
        assert electronics.power_mode == "active"
        assert len(electronics.adc_readings) == 4
        assert len(electronics.dac_outputs) == 2
        assert len(electronics.gpio_states) == 16
        assert len(electronics.comm_activity) == 2
        assert len(electronics.comm_error_counts) == 2
        assert electronics.fault_flags == []
        assert electronics.uptime_hours == 0.0
        assert electronics.boot_count == 0
        assert electronics.thermal_time_constant == 300.0
        assert electronics.ambient_temperature == 25.0


class TestReadADC:
    @patch("time.sleep")
    def test_read_adc_valid_channel(self, mock_sleep, electronics):
        electronics.adc_readings[0] = 1.5
        reading = electronics.read_adc(0, samples=1)
        assert isinstance(reading, (float, np.floating))
        assert electronics.cpu_utilization > 0.0

    @patch("time.sleep")
    def test_read_adc_invalid_channel(self, mock_sleep, electronics):
        with pytest.raises(ValueError, match="Invalid ADC channel"):
            electronics.read_adc(10)

    @patch("time.sleep")
    def test_read_adc_clamping(self, mock_sleep, electronics):
        electronics.adc_readings[0] = 5.0
        reading = electronics.read_adc(0)
        assert reading <= 3.3

    @patch("time.sleep")
    def test_read_adc_noise_history_limit(self, mock_sleep, electronics):
        electronics.adc_readings[0] = 1.0
        for _ in range(1005):
            electronics.read_adc(0)
        assert len(electronics.adc_noise_history) <= 1000


class TestWriteDAC:
    def test_write_dac_valid(self, electronics):
        result = electronics.write_dac(0, 2.0)
        assert result is True
        assert electronics.dac_outputs[0] != 0.0

    def test_write_dac_invalid_channel(self, electronics):
        result = electronics.write_dac(10, 2.0)
        assert result is False

    def test_write_dac_out_of_range_clamped(self, electronics):
        result = electronics.write_dac(0, 5.0)
        assert result is True

    def test_write_dac_negative_out_of_range(self, electronics):
        result = electronics.write_dac(0, -1.0)
        assert result is True

    def test_write_dac_cpu_utilization_increase(self, electronics):
        initial_cpu = electronics.cpu_utilization
        electronics.write_dac(0, 1.5)
        assert electronics.cpu_utilization > initial_cpu


class TestGPIO:
    def test_set_gpio_output(self, electronics):
        result = electronics.set_gpio(0, True, "output")
        assert result is True
        assert electronics.gpio_states[0] is True
        assert electronics.gpio_directions[0] == "output"

    def test_set_gpio_input(self, electronics):
        result = electronics.set_gpio(0, True, "input")
        assert result is True
        assert electronics.gpio_directions[0] == "input"

    def test_set_gpio_invalid_pin(self, electronics):
        result = electronics.set_gpio(20, True)
        assert result is False

    def test_set_gpio_invalid_direction(self, electronics):
        result = electronics.set_gpio(0, True, "bidirectional")
        assert result is False

    def test_set_gpio_switching_power(self, electronics):
        electronics.set_gpio(0, True, "output")
        electronics.set_gpio(0, False, "output")
        assert hasattr(electronics, "_previous_gpio_states")

    def test_read_gpio_valid_output(self, electronics):
        electronics.set_gpio(0, True, "output")
        result = electronics.read_gpio(0)
        assert result is True

    def test_read_gpio_invalid_pin(self, electronics):
        result = electronics.read_gpio(20)
        assert result is False

    def test_read_gpio_input_direction(self, electronics):
        electronics.set_gpio(0, False, "input")
        result = electronics.read_gpio(0)
        assert isinstance(result, bool)

    def test_read_gpio_input_with_external_state(self, electronics):
        electronics.set_gpio(0, False, "input")
        electronics._external_gpio_states = {0: True}
        with patch("numpy.random.random", return_value=0.5):
            result = electronics.read_gpio(0)
            assert result is True

    def test_read_gpio_input_read_error(self, electronics):
        electronics.set_gpio(0, False, "input")
        with patch("numpy.random.random", return_value=0.0001):
            result = electronics.read_gpio(0)
            assert isinstance(result, bool)


class TestCommunication:
    @patch("time.sleep")
    def test_communicate_success(self, mock_sleep, electronics):
        with patch("numpy.random.random", return_value=0.99):
            success, response = electronics.communicate(
                InterfaceType.SPI, b"test",
            )
            assert success is True
            assert response == b"test"

    @patch("time.sleep")
    def test_communicate_unavailable_interface(self, mock_sleep, electronics):
        success, response = electronics.communicate(InterfaceType.CAN, b"test")
        assert success is False
        assert response == b""

    @patch("time.sleep")
    def test_communicate_error(self, mock_sleep, electronics):
        with patch("numpy.random.random", return_value=0.0):
            success, response = electronics.communicate(
                InterfaceType.SPI, b"test",
            )
            assert success is False

    @patch("time.sleep")
    def test_communicate_cpu_utilization(self, mock_sleep, electronics):
        initial_cpu = electronics.cpu_utilization
        with patch("numpy.random.random", return_value=0.99):
            electronics.communicate(InterfaceType.SPI, b"testdata")
        assert electronics.cpu_utilization > initial_cpu


class TestThermalModel:
    def test_update_thermal_model_basic(self, electronics):
        electronics.update_thermal_model(dt=1.0)
        assert electronics.mcu_temperature >= 25.0

    def test_update_thermal_model_with_ambient(self, electronics):
        electronics.update_thermal_model(dt=1.0, ambient_temp=40.0)
        assert electronics.ambient_temperature == 40.0

    def test_update_thermal_model_high_temp(self, electronics):
        electronics.mcu_temperature = 90.0
        electronics.update_thermal_model(dt=1.0)
        assert "TEMP_HIGH" in electronics.fault_flags

    def test_update_thermal_model_low_temp(self, electronics):
        electronics.mcu_temperature = -50.0
        electronics.update_thermal_model(dt=1.0)
        assert "TEMP_LOW" in electronics.fault_flags


class TestPowerMode:
    def test_set_power_mode_active(self, electronics):
        assert electronics.set_power_mode("active") is True
        assert electronics.power_mode == "active"

    def test_set_power_mode_sleep(self, electronics):
        electronics.cpu_utilization = 50.0
        electronics.comm_activity[InterfaceType.SPI] = 20.0
        assert electronics.set_power_mode("sleep") is True
        assert electronics.cpu_utilization == 5.0

    def test_set_power_mode_deep_sleep(self, electronics):
        electronics.cpu_utilization = 50.0
        assert electronics.set_power_mode("deep_sleep") is True
        assert electronics.cpu_utilization == 0.0

    def test_set_power_mode_invalid(self, electronics):
        assert electronics.set_power_mode("turbo") is False


class TestGetPowerConsumption:
    def test_active_power(self, electronics):
        power = electronics.get_power_consumption()
        assert power > 0.0

    def test_sleep_power(self, electronics):
        electronics.set_power_mode("sleep")
        power = electronics.get_power_consumption()
        assert power > 0.0

    def test_deep_sleep_power(self, electronics):
        electronics.set_power_mode("deep_sleep")
        power = electronics.get_power_consumption()
        assert power > 0.0

    def test_power_with_active_gpio(self, electronics):
        electronics.set_gpio(0, True, "output")
        electronics.set_gpio(1, True, "output")
        power = electronics.get_power_consumption()
        assert power > 0.0

    def test_power_with_switching_power(self, electronics):
        electronics._gpio_switching_power = 10.0
        power = electronics.get_power_consumption()
        assert power > 0.0
        assert electronics._gpio_switching_power < 10.0


class TestGetMeasurement:
    def test_get_measurement_returns_valid(self, electronics):
        m = electronics.get_measurement()
        assert isinstance(m, ElectronicsMeasurement)
        assert m.timestamp > 0
        assert isinstance(m.adc_readings, dict)
        assert isinstance(m.dac_outputs, dict)
        assert isinstance(m.gpio_states, dict)

    def test_get_measurement_decays_cpu(self, electronics):
        electronics.cpu_utilization = 50.0
        electronics.get_measurement()
        assert electronics.cpu_utilization < 50.0

    def test_get_measurement_decays_comm(self, electronics):
        electronics.comm_activity[InterfaceType.SPI] = 50.0
        electronics.get_measurement()
        assert electronics.comm_activity[InterfaceType.SPI] < 50.0

    def test_get_measurement_temp_fault_high(self, electronics):
        electronics.mcu_temperature = 100.0
        m = electronics.get_measurement()
        assert "TEMP_HIGH" in m.fault_flags

    def test_get_measurement_temp_fault_low(self, electronics):
        electronics.mcu_temperature = -50.0
        m = electronics.get_measurement()
        assert "TEMP_LOW" in m.fault_flags

    def test_get_measurement_no_fault(self, electronics):
        electronics.mcu_temperature = 25.0
        m = electronics.get_measurement()
        assert len(m.fault_flags) == 0


class TestGetCostAnalysis:
    def test_cost_analysis_returns_dict(self, electronics):
        cost = electronics.get_cost_analysis()
        assert isinstance(cost, dict)
        assert "initial_cost" in cost
        assert "total_cost_per_hour" in cost
        assert cost["initial_cost"] > 0

    def test_cost_analysis_keys(self, electronics):
        cost = electronics.get_cost_analysis()
        expected_keys = [
            "initial_cost", "mcu_cost", "adc_cost", "dac_cost",
            "gpio_cost", "communication_cost", "power_cost_per_hour",
            "development_cost_per_hour", "maintenance_cost_per_hour",
            "total_cost_per_hour",
        ]
        for key in expected_keys:
            assert key in cost


class TestCreateStandardElectronics:
    def test_creates_two_configs(self):
        systems = create_standard_control_electronics()
        assert "high_performance" in systems
        assert "low_power" in systems

    def test_high_performance_config(self):
        systems = create_standard_control_electronics()
        hp = systems["high_performance"]
        assert isinstance(hp, ControlElectronics)
        assert hp.mcu_specs.architecture == MCUArchitecture.ARM_CORTEX_M7

    def test_low_power_config(self):
        systems = create_standard_control_electronics()
        lp = systems["low_power"]
        assert isinstance(lp, ControlElectronics)
        assert lp.mcu_specs.architecture == MCUArchitecture.ARM_CORTEX_M0
