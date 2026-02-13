"""Control System Configuration Classes.

This module provides comprehensive configuration classes for MFC control systems,
including PID controllers, flow control, substrate control, and recirculation systems.

Classes:
- PIDConfig: PID controller parameters
- FlowControlConfig: Flow rate control parameters
- SubstrateControlConfig: Substrate addition control parameters
- RecirculationConfig: Anolyte recirculation system parameters
- AdvancedControlConfig: Advanced Q-learning control parameters
- ControlSystemConfig: Complete control system configuration

Literature References:
1. Srinivasan, R., & Rengaswamy, R. (2012). "Fault-tolerant process control: Methods and applications"
2. Astrom, K. J., & Hagglund, T. (2006). "Advanced PID Control"
3. Logan, B. E. (2008). "Microbial fuel cells"
4. Pinto, R. P., et al. (2011). "Multi-population model of a microbial electrolysis cell"
"""

from dataclasses import dataclass, field
from enum import Enum


class ControlMode(Enum):
    """Control system operation modes."""

    MANUAL = "manual"
    AUTOMATIC = "automatic"
    CASCADE = "cascade"
    FEEDFORWARD = "feedforward"
    ADAPTIVE = "adaptive"


class SubstrateControlStrategy(Enum):
    """Substrate control strategies."""

    CONCENTRATION_BASED = "concentration_based"
    UTILIZATION_BASED = "utilization_based"
    POWER_BASED = "power_based"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class PIDConfig:
    """PID controller configuration parameters."""

    # Core PID parameters (Astrom & Hagglund, 2006)
    kp: float = 2.0  # Proportional gain
    ki: float = 0.1  # Integral gain
    kd: float = 0.5  # Derivative gain

    # Controller limits and constraints
    output_min: float = 0.0  # Minimum controller output
    output_max: float = 100.0  # Maximum controller output
    integral_min: float = -50.0  # Anti-windup integral minimum
    integral_max: float = 50.0  # Anti-windup integral maximum

    # Tuning parameters
    setpoint_weight: float = 1.0  # Setpoint weighting (0-1)
    derivative_on_measurement: bool = True  # Derivative on measurement vs error
    integral_time_constant: float = 10.0  # Integral time constant (s)
    derivative_time_constant: float = 1.0  # Derivative time constant (s)

    # Advanced features
    enable_anti_windup: bool = True  # Enable integral anti-windup
    enable_bumpless_transfer: bool = True  # Bumpless manual-auto transfer
    dead_band: float = 0.0  # Controller dead band

    # Adaptive parameters
    enable_gain_scheduling: bool = False  # Enable gain scheduling
    operating_point_gains: dict[float, tuple[float, float, float]] = field(
        default_factory=dict,
    )

    def __post_init__(self):
        """Validate PID configuration parameters."""
        if self.kp < 0:
            msg = "Proportional gain must be non-negative"
            raise ValueError(msg)
        if self.ki < 0:
            msg = "Integral gain must be non-negative"
            raise ValueError(msg)
        if self.kd < 0:
            msg = "Derivative gain must be non-negative"
            raise ValueError(msg)
        if self.output_min >= self.output_max:
            msg = "Output minimum must be less than maximum"
            raise ValueError(msg)


@dataclass
class FlowControlConfig:
    """Flow rate control system configuration."""

    # Flow rate parameters (Logan, 2008)
    min_flow_rate: float = 5.0  # Minimum flow rate (mL/h)
    max_flow_rate: float = 50.0  # Maximum flow rate (mL/h)
    nominal_flow_rate: float = 15.0  # Nominal operating flow rate (mL/h)

    # Flow rate constraints
    max_flow_rate_change: float = 10.0  # Maximum flow rate change per step (mL/h)
    flow_rate_tolerance: float = 0.5  # Flow rate control tolerance (mL/h)

    # Pump characteristics
    pump_efficiency: float = 0.95  # Pump efficiency (0-1)
    pump_response_time: float = 2.0  # Pump response time constant (s)
    pipe_dead_volume: float = 0.05  # Dead volume in pipes (L)

    # Control parameters
    control_mode: ControlMode = ControlMode.AUTOMATIC
    flow_pid: PIDConfig = field(
        default_factory=lambda: PIDConfig(kp=1.5, ki=0.2, kd=0.1),
    )

    # Safety parameters
    emergency_shutdown_flow: float = 0.0  # Emergency shutdown flow rate (mL/h)
    flow_alarm_high: float = 45.0  # High flow alarm threshold (mL/h)
    flow_alarm_low: float = 2.0  # Low flow alarm threshold (mL/h)


@dataclass
class SubstrateControlConfig:
    """Substrate addition control system configuration."""

    # Target concentrations (Pinto et al., 2011)
    target_outlet_concentration: float = 12.0  # Target outlet concentration (mmol/L)
    target_reservoir_concentration: float = (
        20.0  # Target reservoir concentration (mmol/L)
    )

    # Substrate addition parameters
    min_addition_rate: float = 0.0  # Minimum addition rate (mmol/h)
    max_addition_rate: float = 50.0  # Maximum addition rate (mmol/h)
    nominal_addition_rate: float = 10.0  # Nominal addition rate (mmol/h)

    # Control strategy
    control_strategy: SubstrateControlStrategy = (
        SubstrateControlStrategy.CONCENTRATION_BASED
    )

    # PID controllers for cascade control
    outlet_pid: PIDConfig = field(
        default_factory=lambda: PIDConfig(kp=2.0, ki=0.1, kd=0.5),
    )
    reservoir_pid: PIDConfig = field(
        default_factory=lambda: PIDConfig(kp=1.0, ki=0.05, kd=0.2),
    )

    # Substrate monitoring thresholds
    starvation_threshold_warning: float = 5.0  # Warning threshold (mmol/L)
    starvation_threshold_critical: float = 2.0  # Critical threshold (mmol/L)
    excess_threshold: float = 25.0  # Excess substrate threshold (mmol/L)
    halt_threshold: float = 0.5  # Substrate decline halt threshold (mmol/L)

    # Adaptive control parameters
    starvation_boost_factor: float = 2.0  # Boost factor for starvation conditions
    critical_boost_factor: float = 5.0  # Emergency boost factor
    mixing_efficiency_factor: float = 0.8  # Factor for poor mixing conditions

    # Multi-objective weights (for MULTI_OBJECTIVE strategy)
    concentration_weight: float = 0.6  # Weight for concentration control
    utilization_weight: float = 0.3  # Weight for utilization efficiency
    power_weight: float = 0.1  # Weight for power optimization


@dataclass
class RecirculationConfig:
    """Anolyte recirculation system configuration."""

    # Reservoir parameters
    reservoir_volume: float = 1.0  # Reservoir volume (L)
    initial_substrate_concentration: float = 20.0  # Initial concentration (mmol/L)

    # Mixing dynamics (based on fluid mechanics principles)
    mixing_time_constant: float = 0.1  # Mixing time constant (h)
    heat_loss_coefficient: float = 0.02  # Heat loss coefficient

    # Recirculation system parameters
    pump_efficiency: float = 0.95  # Pump efficiency
    pipe_dead_volume: float = 0.05  # Dead volume in pipes (L)

    # Flow characteristics
    min_recirculation_rate: float = 10.0  # Minimum recirculation rate (mL/h)
    max_recirculation_rate: float = 100.0  # Maximum recirculation rate (mL/h)

    # Monitoring parameters
    mixing_efficiency_threshold: float = 0.7  # Poor mixing threshold
    excellent_mixing_threshold: float = 0.95  # Excellent mixing threshold

    # Control adjustments based on mixing
    poor_mixing_factor: float = 0.8  # Rate reduction for poor mixing
    excellent_mixing_factor: float = 1.1  # Rate boost for excellent mixing
    circulation_boost_factor: float = 0.1  # Boost factor based on circulation cycles


@dataclass
class AdvancedControlConfig:
    """Advanced Q-learning and adaptive control configuration."""

    # Q-learning parameters
    learning_rate: float = 0.0987  # Learning rate
    discount_factor: float = 0.9517  # Discount factor
    epsilon: float = 0.3702  # Exploration rate
    epsilon_decay: float = 0.9978  # Epsilon decay rate
    epsilon_min: float = 0.08  # Minimum epsilon

    # State discretization
    power_bins: int = 8  # Number of power state bins
    power_max: float = 2.0  # Maximum power for discretization
    biofilm_deviation_bins: int = 6  # Number of biofilm deviation bins
    biofilm_max_deviation: float = 1.0  # Maximum biofilm deviation
    substrate_utilization_bins: int = 8  # Number of substrate utilization bins
    substrate_utilization_max: float = 50.0  # Maximum substrate utilization

    # Action discretization
    flow_rate_adjustments: list[float] = field(
        default_factory=lambda: [
            -12.0,
            -10.0,
            -5.0,
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
            5.0,
            6.0,
        ],
    )
    substrate_actions: list[float] = field(
        default_factory=lambda: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    )

    # Multi-objective weights
    power_objective_weight: float = 0.4  # Power optimization weight
    biofilm_health_weight: float = 0.3  # Biofilm health weight
    sensor_agreement_weight: float = 0.2  # Sensor agreement weight
    stability_weight: float = 0.1  # System stability weight

    # Reward structure
    power_reward_weight: float = 10.0  # Power reward weight
    power_penalty_threshold: float = 50.0  # Power penalty threshold
    power_penalty_multiplier: float = 100.0  # Power penalty multiplier
    biofilm_reward_weight: float = 50.0  # Biofilm reward weight
    biofilm_optimal_thickness: float = 30.0  # Optimal biofilm thickness (μm)
    substrate_consumption_weight: float = 5.0  # Substrate consumption weight

    # Stability parameters
    stability_target_flow_rate: float = 15.0  # Target flow rate for stability
    stability_target_outlet_concentration: float = 12.0  # Target outlet concentration

    # Time phase discretization (hours)
    time_phase_hours: list[int] = field(default_factory=lambda: [200, 500, 800, 1000])


@dataclass
class ControlSystemConfig:
    """Complete control system configuration."""

    # Individual control subsystem configurations
    flow_control: FlowControlConfig = field(default_factory=FlowControlConfig)
    substrate_control: SubstrateControlConfig = field(
        default_factory=SubstrateControlConfig,
    )
    recirculation: RecirculationConfig = field(default_factory=RecirculationConfig)
    advanced_control: AdvancedControlConfig = field(
        default_factory=AdvancedControlConfig,
    )

    # Global control parameters
    control_update_interval: float = 60.0  # Control update interval (s)
    data_logging_interval: float = 10.0  # Data logging interval (s)

    # System-wide safety parameters
    emergency_shutdown_enabled: bool = True  # Enable emergency shutdown
    fault_tolerance_enabled: bool = True  # Enable fault tolerance

    # Performance monitoring
    enable_performance_monitoring: bool = True  # Enable performance monitoring
    performance_metrics_interval: float = (
        300.0  # Performance metrics update interval (s)
    )

    # Configuration metadata
    configuration_name: str = "default_control"  # Configuration name
    configuration_version: str = "1.0.0"  # Configuration version
    created_by: str = "system"  # Configuration creator
    description: str = "Default control system configuration"  # Description


# Pre-defined control configurations for different scenarios


def get_conservative_control_config() -> ControlSystemConfig:
    """Get conservative control configuration for stable operation."""
    config = ControlSystemConfig()

    # Conservative PID tuning
    config.flow_control.flow_pid.kp = 1.0
    config.flow_control.flow_pid.ki = 0.05
    config.flow_control.flow_pid.kd = 0.1

    config.substrate_control.outlet_pid.kp = 1.5
    config.substrate_control.outlet_pid.ki = 0.05
    config.substrate_control.outlet_pid.kd = 0.2

    # Conservative flow rates
    config.flow_control.max_flow_rate_change = 5.0
    config.substrate_control.max_addition_rate = 30.0

    # Conservative Q-learning parameters
    config.advanced_control.learning_rate = 0.05
    config.advanced_control.epsilon = 0.2

    config.configuration_name = "conservative_control"
    config.description = "Conservative control settings for stable operation"

    return config


def get_aggressive_control_config() -> ControlSystemConfig:
    """Get aggressive control configuration for rapid optimization."""
    config = ControlSystemConfig()

    # Aggressive PID tuning
    config.flow_control.flow_pid.kp = 3.0
    config.flow_control.flow_pid.ki = 0.5
    config.flow_control.flow_pid.kd = 0.8

    config.substrate_control.outlet_pid.kp = 3.0
    config.substrate_control.outlet_pid.ki = 0.2
    config.substrate_control.outlet_pid.kd = 1.0

    # Aggressive flow rates
    config.flow_control.max_flow_rate_change = 15.0
    config.substrate_control.max_addition_rate = 80.0

    # Aggressive Q-learning parameters
    config.advanced_control.learning_rate = 0.2
    config.advanced_control.epsilon = 0.5
    config.advanced_control.power_objective_weight = 0.5

    config.configuration_name = "aggressive_control"
    config.description = "Aggressive control settings for rapid optimization"

    return config


def get_precision_control_config() -> ControlSystemConfig:
    """Get precision control configuration for high-accuracy applications."""
    config = ControlSystemConfig()

    # Precision PID tuning with derivative on measurement
    config.flow_control.flow_pid.kp = 2.5
    config.flow_control.flow_pid.ki = 0.15
    config.flow_control.flow_pid.kd = 0.6
    config.flow_control.flow_pid.derivative_on_measurement = True

    # Tight tolerances
    config.flow_control.flow_rate_tolerance = 0.1
    config.substrate_control.target_outlet_concentration = 12.0

    # Higher update rates
    config.control_update_interval = 30.0
    config.data_logging_interval = 5.0

    config.configuration_name = "precision_control"
    config.description = "Precision control settings for high-accuracy applications"

    return config


# Configuration validation functions
def validate_control_config(config: ControlSystemConfig) -> bool:
    """Validate control system configuration.

    Args:
        config: Control system configuration to validate

    Returns:
        bool: True if configuration is valid

    Raises:
        ValueError: If configuration is invalid

    """
    # Validate flow control parameters
    if config.flow_control.min_flow_rate >= config.flow_control.max_flow_rate:
        msg = "Minimum flow rate must be less than maximum flow rate"
        raise ValueError(msg)

    if (
        config.flow_control.nominal_flow_rate < config.flow_control.min_flow_rate
        or config.flow_control.nominal_flow_rate > config.flow_control.max_flow_rate
    ):
        msg = "Nominal flow rate must be within min/max bounds"
        raise ValueError(msg)

    # Validate substrate control parameters
    if (
        config.substrate_control.min_addition_rate
        >= config.substrate_control.max_addition_rate
    ):
        msg = "Minimum addition rate must be less than maximum addition rate"
        raise ValueError(
            msg,
        )

    if (
        config.substrate_control.starvation_threshold_critical
        >= config.substrate_control.starvation_threshold_warning
    ):
        msg = "Critical starvation threshold must be less than warning threshold"
        raise ValueError(
            msg,
        )

    # Validate recirculation parameters
    if config.recirculation.reservoir_volume <= 0:
        msg = "Reservoir volume must be positive"
        raise ValueError(msg)

    if (
        config.recirculation.pump_efficiency <= 0
        or config.recirculation.pump_efficiency > 1
    ):
        msg = "Pump efficiency must be between 0 and 1"
        raise ValueError(msg)

    # Validate Q-learning parameters
    if (
        config.advanced_control.learning_rate <= 0
        or config.advanced_control.learning_rate > 1
    ):
        msg = "Learning rate must be between 0 and 1"
        raise ValueError(msg)

    if (
        config.advanced_control.discount_factor <= 0
        or config.advanced_control.discount_factor > 1
    ):
        msg = "Discount factor must be between 0 and 1"
        raise ValueError(msg)

    # Validate weights sum to reasonable values
    total_weight = (
        config.advanced_control.power_objective_weight
        + config.advanced_control.biofilm_health_weight
        + config.advanced_control.sensor_agreement_weight
        + config.advanced_control.stability_weight
    )

    if not (0.8 <= total_weight <= 1.2):  # Allow some tolerance
        msg = f"Multi-objective weights should sum to approximately 1.0, got {total_weight}"
        raise ValueError(
            msg,
        )

    return True
