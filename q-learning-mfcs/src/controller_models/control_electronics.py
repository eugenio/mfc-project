"""Control Electronics Models for MFC Systems.

This module implements comprehensive models for control electronics including
microcontrollers (MCU), analog-to-digital converters (ADC), digital-to-analog
converters (DAC), GPIO interfaces, and communication systems.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class MCUArchitecture(Enum):
    """Supported MCU architectures."""

    ARM_CORTEX_M0 = "arm_cortex_m0"
    ARM_CORTEX_M4 = "arm_cortex_m4"
    ARM_CORTEX_M7 = "arm_cortex_m7"
    ARM_CORTEX_A = "arm_cortex_a"
    RISC_V = "risc_v"
    X86 = "x86"
    DSP = "dsp"


class InterfaceType(Enum):
    """Communication interface types."""

    SPI = "spi"
    I2C = "i2c"
    UART = "uart"
    CAN = "can"
    ETHERNET = "ethernet"
    USB = "usb"
    GPIO = "gpio"


@dataclass
class MCUSpecs:
    """Specifications for microcontroller unit."""

    architecture: MCUArchitecture
    clock_frequency_mhz: float
    cores: int
    ram_kb: float
    flash_kb: float
    cache_kb: float
    fpu: bool  # Floating point unit
    dsp_extensions: bool
    power_consumption_active_mw: float
    power_consumption_sleep_mw: float
    power_consumption_deep_sleep_mw: float
    operating_voltage_v: float
    temperature_range: tuple[float, float]  # °C
    package_type: str
    cost: float  # USD


@dataclass
class ADCSpecs:
    """Specifications for analog-to-digital converter."""

    resolution_bits: int
    sampling_rate_ksps: float  # kilo-samples per second
    channels: int
    reference_voltage_v: float
    input_range_v: tuple[float, float]
    differential_inputs: bool
    internal_reference: bool
    noise_lsb: float  # Noise in LSB
    linearity_error_lsb: float
    power_consumption_mw: float
    interface_type: InterfaceType
    cost: float  # USD


@dataclass
class DACSpecs:
    """Specifications for digital-to-analog converter."""

    resolution_bits: int
    update_rate_ksps: float
    channels: int
    reference_voltage_v: float
    output_range_v: tuple[float, float]
    current_drive_ma: float
    settling_time_us: float
    linearity_error_lsb: float
    power_consumption_mw: float
    interface_type: InterfaceType
    cost: float  # USD


@dataclass
class GPIOSpecs:
    """Specifications for general-purpose I/O."""

    pins: int
    voltage_levels_v: list[float]  # Supported voltage levels
    current_drive_ma: float
    input_leakage_ua: float
    switching_frequency_mhz: float
    pull_up_down_available: bool
    interrupt_capable: bool
    analog_capable: bool
    power_consumption_per_pin_uw: float
    cost: float  # USD


@dataclass
class CommunicationInterface:
    """Communication interface specifications."""

    interface_type: InterfaceType
    speed_mbps: float
    power_consumption_mw: float
    cable_length_m: float
    noise_immunity_db: float
    cost: float  # USD


@dataclass
class ElectronicsMeasurement:
    """Single electronics measurement."""

    timestamp: float
    mcu_temperature_c: float
    power_consumption_mw: float
    cpu_utilization_pct: float
    memory_usage_pct: float
    adc_readings: dict[int, float]  # Channel -> voltage
    dac_outputs: dict[int, float]  # Channel -> voltage
    gpio_states: dict[int, bool]  # Pin -> state
    communication_activity: dict[InterfaceType, float]  # Interface -> utilization %
    fault_flags: list[str]


class ControlElectronics:
    """Comprehensive control electronics system model."""

    def __init__(
        self,
        mcu_specs: MCUSpecs,
        adc_specs: ADCSpecs,
        dac_specs: DACSpecs,
        gpio_specs: GPIOSpecs,
        communication_interfaces: list[CommunicationInterface],
    ) -> None:
        self.mcu_specs = mcu_specs
        self.adc_specs = adc_specs
        self.dac_specs = dac_specs
        self.gpio_specs = gpio_specs
        self.comm_interfaces = {
            iface.interface_type: iface for iface in communication_interfaces
        }

        # Runtime state
        self.mcu_temperature = 25.0  # °C
        self.cpu_utilization = 0.0  # %
        self.memory_usage = 0.0  # %
        self.power_mode = "active"  # active, sleep, deep_sleep

        # ADC state
        self.adc_readings = dict.fromkeys(range(adc_specs.channels), 0.0)
        self.adc_noise_history = []

        # DAC state
        self.dac_outputs = dict.fromkeys(range(dac_specs.channels), 0.0)
        self.dac_settling_time = 0.0

        # GPIO state
        self.gpio_states = dict.fromkeys(range(gpio_specs.pins), False)
        self.gpio_directions = dict.fromkeys(
            range(gpio_specs.pins),
            "input",
        )  # input/output

        # Communication state
        self.comm_activity = dict.fromkeys(self.comm_interfaces.keys(), 0.0)
        self.comm_error_counts = dict.fromkeys(self.comm_interfaces.keys(), 0)

        # Performance tracking
        self.fault_flags = []
        self.uptime_hours = 0.0
        self.boot_count = 0

        # Initialize thermal model
        self.thermal_time_constant = 300.0  # seconds
        self.ambient_temperature = 25.0  # °C

    def read_adc(self, channel: int, samples: int = 1) -> float:
        """Read analog value from ADC channel.

        Args:
            channel: ADC channel number
            samples: Number of samples to average

        Returns:
            Voltage reading in volts

        """
        if channel >= self.adc_specs.channels:
            msg = f"Invalid ADC channel: {channel}"
            raise ValueError(msg)

        # Simulate ADC conversion time
        conversion_time = samples / (self.adc_specs.sampling_rate_ksps * 1000)
        time.sleep(conversion_time / 1000)  # Convert to seconds

        # Get base reading
        base_reading = self.adc_readings[channel]

        # Add noise based on ADC specifications
        noise_voltage = (
            self.adc_specs.noise_lsb / (2**self.adc_specs.resolution_bits)
        ) * self.adc_specs.reference_voltage_v
        noise = np.random.normal(0, noise_voltage)

        # Add quantization noise
        lsb_voltage = self.adc_specs.reference_voltage_v / (
            2**self.adc_specs.resolution_bits
        )
        quantized_reading = np.round(base_reading / lsb_voltage) * lsb_voltage

        # Apply linearity error
        linearity_error = (
            self.adc_specs.linearity_error_lsb / (2**self.adc_specs.resolution_bits)
        ) * self.adc_specs.reference_voltage_v
        linearity_noise = np.random.uniform(-linearity_error, linearity_error)

        final_reading = quantized_reading + noise + linearity_noise

        # Clamp to input range
        min_v, max_v = self.adc_specs.input_range_v
        final_reading = np.clip(final_reading, min_v, max_v)

        # Update CPU utilization
        self.cpu_utilization += 0.1  # Small increase per ADC read

        # Store noise for analysis
        self.adc_noise_history.append(abs(final_reading - base_reading))
        if len(self.adc_noise_history) > 1000:
            self.adc_noise_history.pop(0)

        return final_reading

    def write_dac(self, channel: int, voltage: float) -> bool:
        """Write analog value to DAC channel.

        Args:
            channel: DAC channel number
            voltage: Voltage to output

        Returns:
            True if successful

        """
        if channel >= self.dac_specs.channels:
            logger.error(f"Invalid DAC channel: {channel}")
            return False

        # Check output range
        min_v, max_v = self.dac_specs.output_range_v
        if not (min_v <= voltage <= max_v):
            logger.warning(f"DAC voltage {voltage}V out of range [{min_v}, {max_v}]")
            voltage = np.clip(voltage, min_v, max_v)

        # Quantize to DAC resolution
        lsb_voltage = self.dac_specs.reference_voltage_v / (
            2**self.dac_specs.resolution_bits
        )
        quantized_voltage = np.round(voltage / lsb_voltage) * lsb_voltage

        # Apply linearity error
        linearity_error = (
            self.dac_specs.linearity_error_lsb / (2**self.dac_specs.resolution_bits)
        ) * self.dac_specs.reference_voltage_v
        linearity_noise = np.random.uniform(-linearity_error, linearity_error)

        output_voltage = quantized_voltage + linearity_noise

        # Store output with settling time
        self.dac_outputs[channel] = output_voltage
        self.dac_settling_time = (
            self.dac_specs.settling_time_us / 1000000
        )  # Convert to seconds

        # Update CPU utilization
        self.cpu_utilization += 0.05  # Small increase per DAC write

        return True

    def set_gpio(self, pin: int, state: bool, direction: str = "output") -> bool:
        """Set GPIO pin state.

        Args:
            pin: GPIO pin number
            state: Pin state (True/False)
            direction: Pin direction ("input"/"output")

        Returns:
            True if successful

        """
        if pin >= self.gpio_specs.pins:
            logger.error(f"Invalid GPIO pin: {pin}")
            return False

        if direction not in ["input", "output"]:
            logger.error(f"Invalid GPIO direction: {direction}")
            return False

        self.gpio_directions[pin] = direction

        if direction == "output":
            self.gpio_states[pin] = state

            # Update power consumption based on switching
            if hasattr(self, "_previous_gpio_states"):
                if self._previous_gpio_states.get(pin) != state:
                    # Pin switched state
                    switching_power = (
                        self.gpio_specs.current_drive_ma
                        * self.mcu_specs.operating_voltage_v
                    )
                    self._gpio_switching_power = (
                        getattr(self, "_gpio_switching_power", 0) + switching_power
                    )

            self._previous_gpio_states = self.gpio_states.copy()

        # Update CPU utilization
        self.cpu_utilization += 0.01

        return True

    def read_gpio(self, pin: int) -> bool:
        """Read GPIO pin state.

        Args:
            pin: GPIO pin number

        Returns:
            Pin state

        """
        if pin >= self.gpio_specs.pins:
            logger.error(f"Invalid GPIO pin: {pin}")
            return False

        if self.gpio_directions[pin] == "input":
            # Simulate input reading with some noise
            if hasattr(self, "_external_gpio_states"):
                base_state = self._external_gpio_states.get(pin, False)
            else:
                base_state = self.gpio_states[pin]

            # Add small probability of read error
            if np.random.random() < 0.001:  # 0.1% error rate
                return not base_state

            return base_state
        return self.gpio_states[pin]

    def communicate(
        self,
        interface_type: InterfaceType,
        data: bytes,
        target_address: int | None = None,
    ) -> tuple[bool, bytes]:
        """Send data via communication interface.

        Args:
            interface_type: Communication interface to use
            data: Data to send
            target_address: Target address (for multi-device buses)

        Returns:
            (success, response_data)

        """
        if interface_type not in self.comm_interfaces:
            logger.error(f"Interface {interface_type} not available")
            return False, b""

        interface = self.comm_interfaces[interface_type]

        # Calculate transmission time
        bits_per_byte = 8 + 1 + 1  # Data + start + stop bits (simplified)
        total_bits = len(data) * bits_per_byte
        transmission_time = total_bits / (interface.speed_mbps * 1000000)

        # Simulate transmission delay
        time.sleep(transmission_time)

        # Update communication activity
        current_activity = self.comm_activity[interface_type]
        self.comm_activity[interface_type] = min(
            100.0,
            current_activity + (transmission_time * 100),
        )

        # Simulate communication errors based on noise immunity
        error_probability = 1.0 / (10 ** (interface.noise_immunity_db / 20))
        if np.random.random() < error_probability:
            self.comm_error_counts[interface_type] += 1
            logger.warning(f"Communication error on {interface_type}")
            return False, b""

        # Update CPU utilization
        self.cpu_utilization += len(data) * 0.001  # Scale with data size

        # Simulate successful transmission with echo response
        response = data  # Simple echo for testing
        return True, response

    def update_thermal_model(
        self,
        dt: float,
        ambient_temp: float | None = None,
    ) -> None:
        """Update thermal model of electronics."""
        if ambient_temp is not None:
            self.ambient_temperature = ambient_temp

        # Calculate power dissipation
        total_power = self.get_power_consumption()
        power_watts = total_power / 1000.0

        # Simple thermal model: T = T_ambient + thermal_resistance * power
        thermal_resistance = 50.0  # °C/W (typical for small electronics)
        target_temperature = self.ambient_temperature + thermal_resistance * power_watts

        # First-order thermal response
        temp_diff = target_temperature - self.mcu_temperature
        self.mcu_temperature += temp_diff * (
            1 - np.exp(-dt / self.thermal_time_constant)
        )

        # Check thermal limits
        min_temp, max_temp = self.mcu_specs.temperature_range
        if self.mcu_temperature < min_temp:
            self.fault_flags.append("TEMP_LOW")
            logger.warning(f"Temperature below minimum: {self.mcu_temperature:.1f}°C")
        elif self.mcu_temperature > max_temp:
            self.fault_flags.append("TEMP_HIGH")
            logger.error(f"Temperature above maximum: {self.mcu_temperature:.1f}°C")

    def set_power_mode(self, mode: str) -> bool:
        """Set MCU power mode.

        Args:
            mode: Power mode ("active", "sleep", "deep_sleep")

        Returns:
            True if successful

        """
        valid_modes = ["active", "sleep", "deep_sleep"]
        if mode not in valid_modes:
            logger.error(f"Invalid power mode: {mode}")
            return False

        self.power_mode = mode

        # Reset activity levels based on power mode
        if mode == "sleep":
            self.cpu_utilization *= 0.1  # Reduce utilization
            self.comm_activity = {k: v * 0.1 for k, v in self.comm_activity.items()}
        elif mode == "deep_sleep":
            self.cpu_utilization = 0.0
            self.comm_activity = {k: 0.0 for k, v in self.comm_activity.items()}

        return True

    def get_power_consumption(self) -> float:
        """Get total power consumption in milliwatts.

        Returns:
            Total power consumption (mW)

        """
        # MCU power based on mode
        if self.power_mode == "active":
            mcu_power = self.mcu_specs.power_consumption_active_mw
        elif self.power_mode == "sleep":
            mcu_power = self.mcu_specs.power_consumption_sleep_mw
        else:  # deep_sleep
            mcu_power = self.mcu_specs.power_consumption_deep_sleep_mw

        # Scale MCU power by utilization
        mcu_power *= (
            0.3 + 0.7 * self.cpu_utilization / 100.0
        )  # 30% baseline + 70% variable

        # ADC power
        adc_power = self.adc_specs.power_consumption_mw

        # DAC power
        dac_power = self.dac_specs.power_consumption_mw

        # GPIO power
        active_pins = sum(1 for state in self.gpio_states.values() if state)
        gpio_power = active_pins * self.gpio_specs.power_consumption_per_pin_uw / 1000.0

        # Communication interface power
        comm_power = sum(
            interface.power_consumption_mw * (self.comm_activity[iface_type] / 100.0)
            for iface_type, interface in self.comm_interfaces.items()
        )

        # GPIO switching power (transient)
        switching_power = getattr(self, "_gpio_switching_power", 0.0)
        self._gpio_switching_power = max(
            0.0,
            switching_power * 0.9,
        )  # Decay switching power

        return (
            mcu_power
            + adc_power
            + dac_power
            + gpio_power
            + comm_power
            + switching_power
        )

    def get_measurement(self) -> ElectronicsMeasurement:
        """Get comprehensive electronics measurement."""
        # Decay CPU utilization over time
        self.cpu_utilization = max(0.0, self.cpu_utilization * 0.95)

        # Decay communication activity
        for interface_type in self.comm_activity:
            self.comm_activity[interface_type] = max(
                0.0,
                self.comm_activity[interface_type] * 0.9,
            )

        # Calculate memory usage (simplified model)
        base_memory = 30.0  # Base OS usage
        variable_memory = (
            self.cpu_utilization / 100.0
        ) * 40.0  # Variable based on activity
        self.memory_usage = base_memory + variable_memory

        # Clear fault flags that are no longer active
        current_faults = []
        min_temp, max_temp = self.mcu_specs.temperature_range
        if self.mcu_temperature < min_temp:
            current_faults.append("TEMP_LOW")
        elif self.mcu_temperature > max_temp:
            current_faults.append("TEMP_HIGH")

        self.fault_flags = current_faults

        return ElectronicsMeasurement(
            timestamp=time.time(),
            mcu_temperature_c=self.mcu_temperature,
            power_consumption_mw=self.get_power_consumption(),
            cpu_utilization_pct=self.cpu_utilization,
            memory_usage_pct=self.memory_usage,
            adc_readings=self.adc_readings.copy(),
            dac_outputs=self.dac_outputs.copy(),
            gpio_states=self.gpio_states.copy(),
            communication_activity=self.comm_activity.copy(),
            fault_flags=self.fault_flags.copy(),
        )

    def get_cost_analysis(self) -> dict[str, float]:
        """Get comprehensive cost analysis."""
        # Component costs
        mcu_cost = self.mcu_specs.cost
        adc_cost = self.adc_specs.cost
        dac_cost = self.dac_specs.cost
        gpio_cost = self.gpio_specs.cost
        comm_cost = sum(interface.cost for interface in self.comm_interfaces.values())

        initial_cost = mcu_cost + adc_cost + dac_cost + gpio_cost + comm_cost

        # Operating costs
        power_cost_per_hour = self.get_power_consumption() * 0.15 / 1000000  # $0.15/kWh

        # Development and maintenance costs
        development_cost_amortized = 1000.0 / 8760  # $1000 spread over 1 year
        maintenance_cost_per_hour = 0.01  # $0.01/hour

        total_cost_per_hour = (
            power_cost_per_hour + development_cost_amortized + maintenance_cost_per_hour
        )

        return {
            "initial_cost": initial_cost,
            "mcu_cost": mcu_cost,
            "adc_cost": adc_cost,
            "dac_cost": dac_cost,
            "gpio_cost": gpio_cost,
            "communication_cost": comm_cost,
            "power_cost_per_hour": power_cost_per_hour,
            "development_cost_per_hour": development_cost_amortized,
            "maintenance_cost_per_hour": maintenance_cost_per_hour,
            "total_cost_per_hour": total_cost_per_hour,
        }


def create_standard_control_electronics() -> dict[str, ControlElectronics]:
    """Create standard control electronics configurations."""
    # High-performance configuration
    hp_mcu_specs = MCUSpecs(
        architecture=MCUArchitecture.ARM_CORTEX_M7,
        clock_frequency_mhz=480.0,
        cores=1,
        ram_kb=1024.0,
        flash_kb=2048.0,
        cache_kb=32.0,
        fpu=True,
        dsp_extensions=True,
        power_consumption_active_mw=150.0,
        power_consumption_sleep_mw=10.0,
        power_consumption_deep_sleep_mw=0.1,
        operating_voltage_v=3.3,
        temperature_range=(-40, 85),
        package_type="LQFP144",
        cost=25.0,
    )

    hp_adc_specs = ADCSpecs(
        resolution_bits=16,
        sampling_rate_ksps=1000.0,
        channels=16,
        reference_voltage_v=3.3,
        input_range_v=(0.0, 3.3),
        differential_inputs=True,
        internal_reference=True,
        noise_lsb=0.5,
        linearity_error_lsb=1.0,
        power_consumption_mw=15.0,
        interface_type=InterfaceType.SPI,
        cost=8.0,
    )

    hp_dac_specs = DACSpecs(
        resolution_bits=16,
        update_rate_ksps=500.0,
        channels=8,
        reference_voltage_v=3.3,
        output_range_v=(0.0, 3.3),
        current_drive_ma=20.0,
        settling_time_us=1.0,
        linearity_error_lsb=0.5,
        power_consumption_mw=12.0,
        interface_type=InterfaceType.SPI,
        cost=6.0,
    )

    hp_gpio_specs = GPIOSpecs(
        pins=100,
        voltage_levels_v=[3.3, 5.0],
        current_drive_ma=25.0,
        input_leakage_ua=0.1,
        switching_frequency_mhz=100.0,
        pull_up_down_available=True,
        interrupt_capable=True,
        analog_capable=True,
        power_consumption_per_pin_uw=10.0,
        cost=0.0,  # Included in MCU
    )

    hp_comm_interfaces = [
        CommunicationInterface(InterfaceType.SPI, 50.0, 5.0, 1.0, 40.0, 2.0),
        CommunicationInterface(InterfaceType.I2C, 1.0, 2.0, 1.0, 30.0, 1.0),
        CommunicationInterface(InterfaceType.UART, 10.0, 3.0, 10.0, 35.0, 1.5),
        CommunicationInterface(InterfaceType.CAN, 1.0, 8.0, 100.0, 45.0, 5.0),
        CommunicationInterface(InterfaceType.ETHERNET, 100.0, 50.0, 100.0, 50.0, 15.0),
    ]

    # Low-power configuration
    lp_mcu_specs = MCUSpecs(
        architecture=MCUArchitecture.ARM_CORTEX_M0,
        clock_frequency_mhz=48.0,
        cores=1,
        ram_kb=64.0,
        flash_kb=256.0,
        cache_kb=0.0,
        fpu=False,
        dsp_extensions=False,
        power_consumption_active_mw=15.0,
        power_consumption_sleep_mw=1.0,
        power_consumption_deep_sleep_mw=0.01,
        operating_voltage_v=3.3,
        temperature_range=(-40, 85),
        package_type="QFN48",
        cost=3.0,
    )

    lp_adc_specs = ADCSpecs(
        resolution_bits=12,
        sampling_rate_ksps=100.0,
        channels=8,
        reference_voltage_v=3.3,
        input_range_v=(0.0, 3.3),
        differential_inputs=False,
        internal_reference=True,
        noise_lsb=1.0,
        linearity_error_lsb=2.0,
        power_consumption_mw=2.0,
        interface_type=InterfaceType.I2C,
        cost=2.0,
    )

    lp_dac_specs = DACSpecs(
        resolution_bits=12,
        update_rate_ksps=50.0,
        channels=2,
        reference_voltage_v=3.3,
        output_range_v=(0.0, 3.3),
        current_drive_ma=5.0,
        settling_time_us=10.0,
        linearity_error_lsb=1.0,
        power_consumption_mw=3.0,
        interface_type=InterfaceType.I2C,
        cost=1.5,
    )

    lp_gpio_specs = GPIOSpecs(
        pins=32,
        voltage_levels_v=[3.3],
        current_drive_ma=5.0,
        input_leakage_ua=0.01,
        switching_frequency_mhz=10.0,
        pull_up_down_available=True,
        interrupt_capable=True,
        analog_capable=False,
        power_consumption_per_pin_uw=1.0,
        cost=0.0,
    )

    lp_comm_interfaces = [
        CommunicationInterface(InterfaceType.SPI, 10.0, 2.0, 0.5, 35.0, 1.0),
        CommunicationInterface(InterfaceType.I2C, 0.4, 1.0, 0.5, 25.0, 0.5),
        CommunicationInterface(InterfaceType.UART, 1.0, 1.5, 5.0, 30.0, 0.5),
    ]

    return {
        "high_performance": ControlElectronics(
            hp_mcu_specs,
            hp_adc_specs,
            hp_dac_specs,
            hp_gpio_specs,
            hp_comm_interfaces,
        ),
        "low_power": ControlElectronics(
            lp_mcu_specs,
            lp_adc_specs,
            lp_dac_specs,
            lp_gpio_specs,
            lp_comm_interfaces,
        ),
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create control electronics systems
    systems = create_standard_control_electronics()

    # Test high-performance system
    hp_system = systems["high_performance"]

    # Simulate some operations

    # Set up some ADC readings
    hp_system.adc_readings[0] = 1.5  # Sensor input
    hp_system.adc_readings[1] = 2.8  # Another sensor

    # Read ADC values
    for ch in range(2):
        reading = hp_system.read_adc(ch, samples=10)

    # Control outputs via DAC
    hp_system.write_dac(0, 2.0)  # Control signal 1
    hp_system.write_dac(1, 1.5)  # Control signal 2

    # GPIO operations
    hp_system.set_gpio(0, True, "output")  # Enable signal
    hp_system.set_gpio(1, False, "output")  # Disable signal

    # Communication test
    success, response = hp_system.communicate(InterfaceType.SPI, b"test_data")

    # Update thermal model
    hp_system.update_thermal_model(dt=1.0, ambient_temp=30.0)

    # Get measurement
    measurement = hp_system.get_measurement()

    # Cost analysis
    cost_analysis = hp_system.get_cost_analysis()
