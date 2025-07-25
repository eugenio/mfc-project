"""
Parameter validation framework for MFC configuration classes.
Provides comprehensive validation for Q-learning and sensor parameters.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
from .qlearning_config import QLearningConfig, QLearningRewardWeights
from .sensor_config import SensorConfig, EISConfig, QCMConfig, SensorFusionConfig


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    
    def __init__(self, parameter: str, value: Any, message: str):
        self.parameter = parameter
        self.value = value
        self.message = message
        super().__init__(f"Validation failed for '{parameter}' = {value}: {message}")


def validate_range(value: float, min_val: float, max_val: float, 
                  parameter: str, inclusive_min: bool = True, inclusive_max: bool = True) -> None:
    """Validate that a value is within the specified range."""
    if inclusive_min and inclusive_max:
        if not (min_val <= value <= max_val):
            raise ConfigValidationError(
                parameter, value, 
                f"must be in range [{min_val}, {max_val}]"
            )
    elif inclusive_min and not inclusive_max:
        if not (min_val <= value < max_val):
            raise ConfigValidationError(
                parameter, value,
                f"must be in range [{min_val}, {max_val})"
            )
    elif not inclusive_min and inclusive_max:
        if not (min_val < value <= max_val):
            raise ConfigValidationError(
                parameter, value,
                f"must be in range ({min_val}, {max_val}]"
            )
    else:
        if not (min_val < value < max_val):
            raise ConfigValidationError(
                parameter, value,
                f"must be in range ({min_val}, {max_val})"
            )


def validate_positive(value: float, parameter: str, allow_zero: bool = False) -> None:
    """Validate that a value is positive (optionally allowing zero)."""
    if allow_zero:
        if value < 0:
            raise ConfigValidationError(parameter, value, "must be non-negative")
    else:
        if value <= 0:
            raise ConfigValidationError(parameter, value, "must be positive")


def validate_probability(value: float, parameter: str) -> None:
    """Validate that a value is a valid probability [0, 1]."""
    validate_range(value, 0.0, 1.0, parameter)


def validate_qlearning_config(config: QLearningConfig) -> bool:
    """
    Validate Q-learning configuration parameters.
    
    Args:
        config: QLearningConfig instance to validate
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ConfigValidationError: If any parameter is invalid
    """
    
    # Core Q-learning parameters
    validate_range(config.learning_rate, 0.0, 1.0, "learning_rate", inclusive_min=False)
    validate_probability(config.discount_factor, "discount_factor")
    validate_probability(config.epsilon, "epsilon")
    
    # Epsilon decay parameters
    validate_range(config.epsilon_decay, 0.0, 1.0, "epsilon_decay", inclusive_min=False)
    validate_probability(config.epsilon_min, "epsilon_min")
    
    # Alternative configurations
    validate_range(config.enhanced_learning_rate, 0.0, 1.0, "enhanced_learning_rate", inclusive_min=False)
    validate_probability(config.enhanced_discount_factor, "enhanced_discount_factor")
    validate_probability(config.enhanced_epsilon, "enhanced_epsilon")
    
    # Advanced decay parameters
    validate_range(config.advanced_epsilon_decay, 0.0, 1.0, "advanced_epsilon_decay", inclusive_min=False)
    validate_probability(config.advanced_epsilon_min, "advanced_epsilon_min")
    
    # Validate epsilon relationships
    # Only check if epsilon > 0 (when epsilon is 0, epsilon_min doesn't matter)
    if config.epsilon > 0 and config.epsilon_min >= config.epsilon:
        raise ConfigValidationError(
            "epsilon_min", config.epsilon_min,
            f"must be less than epsilon ({config.epsilon})"
        )
    
    # Only check if enhanced_epsilon > 0
    if config.enhanced_epsilon > 0 and config.advanced_epsilon_min >= config.enhanced_epsilon:
        raise ConfigValidationError(
            "advanced_epsilon_min", config.advanced_epsilon_min,
            f"must be less than enhanced_epsilon ({config.enhanced_epsilon})"
        )
    
    # State space parameters
    validate_positive(config.power_bins, "power_bins", allow_zero=False)
    validate_positive(config.power_max, "power_max", allow_zero=False)
    validate_positive(config.biofilm_bins, "biofilm_bins", allow_zero=False)
    validate_positive(config.biofilm_max, "biofilm_max", allow_zero=False)
    validate_positive(config.substrate_bins, "substrate_bins", allow_zero=False)
    validate_positive(config.substrate_max, "substrate_max", allow_zero=False)
    
    # Enhanced state space parameters
    validate_positive(config.eis_thickness_bins, "eis_thickness_bins", allow_zero=False)
    validate_positive(config.eis_thickness_max, "eis_thickness_max", allow_zero=False)
    validate_positive(config.eis_conductivity_bins, "eis_conductivity_bins", allow_zero=False)
    validate_positive(config.eis_conductivity_max, "eis_conductivity_max", allow_zero=False)
    
    validate_positive(config.qcm_mass_bins, "qcm_mass_bins", allow_zero=False)
    validate_positive(config.qcm_mass_max, "qcm_mass_max", allow_zero=False)
    validate_positive(config.qcm_frequency_bins, "qcm_frequency_bins", allow_zero=False)
    validate_positive(config.qcm_frequency_max, "qcm_frequency_max", allow_zero=False)
    
    # Sensor integration parameters
    validate_range(config.sensor_weight, 0.0, 1.0, "sensor_weight")
    validate_range(config.sensor_confidence_threshold, 0.0, 1.0, "sensor_confidence_threshold")
    validate_positive(config.exploration_boost_factor, "exploration_boost_factor", allow_zero=False)
    
    # Flow rate bounds
    validate_positive(config.flow_rate_min, "flow_rate_min", allow_zero=False)
    validate_positive(config.flow_rate_max, "flow_rate_max", allow_zero=False)
    
    if config.flow_rate_min >= config.flow_rate_max:
        raise ConfigValidationError(
            "flow_rate_min", config.flow_rate_min,
            f"must be less than flow_rate_max ({config.flow_rate_max})"
        )
    
    # Multi-objective weights should sum to 1.0 (approximately)
    total_weight = (config.power_objective_weight + 
                   config.biofilm_health_weight + 
                   config.sensor_agreement_weight + 
                   config.stability_weight)
    
    if not np.isclose(total_weight, 1.0, rtol=1e-3):
        raise ConfigValidationError(
            "multi_objective_weights", total_weight,
            f"should sum to 1.0, got {total_weight:.4f}"
        )
    
    # Validate reward weights
    validate_qlearning_reward_weights(config.reward_weights)
    
    return True


def validate_qlearning_reward_weights(weights: QLearningRewardWeights) -> bool:
    """
    Validate Q-learning reward weights.
    
    Args:
        weights: QLearningRewardWeights instance to validate
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ConfigValidationError: If any weight is invalid
    """
    
    # All weights should be positive (penalties can be negative but should be reasonable)
    validate_positive(weights.power_weight, "power_weight", allow_zero=False)
    validate_positive(weights.consumption_weight, "consumption_weight", allow_zero=False)
    validate_positive(weights.efficiency_weight, "efficiency_weight", allow_zero=False)
    validate_positive(weights.biofilm_weight, "biofilm_weight", allow_zero=False)
    
    # Penalty multipliers should be positive
    validate_positive(weights.power_penalty_multiplier, "power_penalty_multiplier", allow_zero=False)
    validate_positive(weights.substrate_penalty_multiplier, "substrate_penalty_multiplier", allow_zero=False)
    validate_positive(weights.efficiency_penalty_multiplier, "efficiency_penalty_multiplier", allow_zero=False)
    
    # Thresholds should be in reasonable ranges
    validate_range(weights.efficiency_threshold, 0.0, 1.0, "efficiency_threshold")
    
    # Biofilm penalties should be negative (they are penalties)
    if weights.biofilm_penalty >= 0:
        raise ConfigValidationError(
            "biofilm_penalty", weights.biofilm_penalty,
            "should be negative (it's a penalty)"
        )
    
    if weights.combined_penalty >= 0:
        raise ConfigValidationError(
            "combined_penalty", weights.combined_penalty,
            "should be negative (it's a penalty)"
        )
    
    return True


def validate_sensor_config(config: SensorConfig) -> bool:
    """
    Validate sensor configuration parameters.
    
    Args:
        config: SensorConfig instance to validate
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ConfigValidationError: If any parameter is invalid
    """
    
    # Validate sub-configurations
    validate_eis_config(config.eis)
    validate_qcm_config(config.qcm)
    validate_sensor_fusion_config(config.fusion)
    
    # Validate update intervals
    validate_positive(config.eis_update_interval, "eis_update_interval", allow_zero=False)
    validate_positive(config.qcm_update_interval, "qcm_update_interval", allow_zero=False)
    validate_positive(config.fusion_update_interval, "fusion_update_interval", allow_zero=False)
    
    # Validate timeouts
    validate_positive(config.sensor_timeout, "sensor_timeout", allow_zero=False)
    validate_positive(config.calibration_interval, "calibration_interval", allow_zero=False)
    
    return True


def validate_eis_config(config: EISConfig) -> bool:
    """Validate EIS configuration parameters."""
    
    # Frequency range validation
    if config.frequency_range[0] >= config.frequency_range[1]:
        raise ConfigValidationError(
            "frequency_range", config.frequency_range,
            "minimum frequency must be less than maximum frequency"
        )
    
    validate_positive(config.frequency_range[0], "frequency_range[0]", allow_zero=False)
    validate_positive(config.frequency_range[1], "frequency_range[1]", allow_zero=False)
    
    # Measurement parameters
    validate_positive(config.n_frequency_points, "n_frequency_points", allow_zero=False)
    validate_positive(config.measurement_amplitude, "measurement_amplitude", allow_zero=False)
    
    # Species parameters
    validate_positive(config.geobacter_base_resistivity, "geobacter_base_resistivity", allow_zero=False)
    validate_positive(config.shewanella_base_resistivity, "shewanella_base_resistivity", allow_zero=False)
    validate_positive(config.mixed_base_resistivity, "mixed_base_resistivity", allow_zero=False)
    
    # Noise and drift parameters
    validate_range(config.noise_level, 0.0, 1.0, "noise_level")
    validate_positive(config.drift_rate, "drift_rate", allow_zero=True)
    
    return True


def validate_qcm_config(config: QCMConfig) -> bool:
    """Validate QCM configuration parameters."""
    
    # Sensitivity parameters
    validate_positive(config.sensitivity_5mhz, "sensitivity_5mhz", allow_zero=False)
    validate_positive(config.sensitivity_10mhz, "sensitivity_10mhz", allow_zero=False)
    validate_positive(config.default_sensitivity, "default_sensitivity", allow_zero=False)
    
    # Biofilm properties
    validate_positive(config.biofilm_density, "biofilm_density", allow_zero=False)
    validate_positive(config.biofilm_viscosity, "biofilm_viscosity", allow_zero=False)
    validate_positive(config.thickness_limit, "thickness_limit", allow_zero=False)
    
    # Measurement ranges
    if config.mass_range[0] >= config.mass_range[1]:
        raise ConfigValidationError(
            "mass_range", config.mass_range,
            "minimum mass must be less than maximum mass"
        )
    
    if config.frequency_shift_range[0] >= config.frequency_shift_range[1]:
        raise ConfigValidationError(
            "frequency_shift_range", config.frequency_shift_range,
            "minimum frequency shift must be less than maximum frequency shift"
        )
    
    # Electrode parameters
    validate_positive(config.electrode_area, "electrode_area", allow_zero=False)
    
    return True


def validate_sensor_fusion_config(config: SensorFusionConfig) -> bool:
    """Validate sensor fusion configuration parameters."""
    
    # Kalman filter parameters
    validate_positive(config.kalman_initial_uncertainty, "kalman_initial_uncertainty", allow_zero=False)
    
    # Process noise parameters (should be positive)
    validate_positive(config.process_noise_thickness, "process_noise_thickness", allow_zero=False)
    validate_positive(config.process_noise_biomass, "process_noise_biomass", allow_zero=False)
    validate_positive(config.process_noise_conductivity, "process_noise_conductivity", allow_zero=False)
    
    # Measurement noise parameters (should be positive)
    validate_positive(config.measurement_noise_eis_thickness, "measurement_noise_eis_thickness", allow_zero=False)
    validate_positive(config.measurement_noise_qcm_thickness, "measurement_noise_qcm_thickness", allow_zero=False)
    validate_positive(config.measurement_noise_conductivity, "measurement_noise_conductivity", allow_zero=False)
    
    # Reliability parameters
    validate_range(config.initial_eis_reliability, 0.0, 1.0, "initial_eis_reliability")
    validate_range(config.initial_qcm_reliability, 0.0, 1.0, "initial_qcm_reliability")
    
    # Weight and threshold parameters
    validate_range(config.minimum_sensor_weight, 0.0, 1.0, "minimum_sensor_weight")
    validate_positive(config.max_disagreement_threshold, "max_disagreement_threshold", allow_zero=False)
    validate_range(config.fault_threshold, 0.0, 1.0, "fault_threshold")
    
    # Decay factors should be in (0, 1]
    validate_range(config.eis_reliability_decay, 0.0, 1.0, "eis_reliability_decay", inclusive_min=False)
    validate_range(config.qcm_reliability_decay, 0.0, 1.0, "qcm_reliability_decay", inclusive_min=False)
    
    # Update weights should be small positive values
    validate_range(config.eis_reliability_update_weight, 0.0, 1.0, "eis_reliability_update_weight")
    validate_range(config.qcm_reliability_update_weight, 0.0, 1.0, "qcm_reliability_update_weight")
    
    return True


def validate_all_configurations(qlearning_config: Optional[QLearningConfig] = None,
                               sensor_config: Optional[SensorConfig] = None) -> Dict[str, bool]:
    """
    Validate all provided configurations.
    
    Args:
        qlearning_config: Optional Q-learning configuration to validate
        sensor_config: Optional sensor configuration to validate
        
    Returns:
        Dict[str, bool]: Validation results for each configuration
        
    Raises:
        ConfigValidationError: If any validation fails
    """
    
    results = {}
    
    if qlearning_config is not None:
        results['qlearning'] = validate_qlearning_config(qlearning_config)
    
    if sensor_config is not None:
        results['sensor'] = validate_sensor_config(sensor_config)
    
    return results


# Convenience function for quick validation
def quick_validate(config: Union[QLearningConfig, SensorConfig]) -> bool:
    """
    Quick validation for any supported configuration type.
    
    Args:
        config: Configuration instance to validate
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ConfigValidationError: If validation fails
        TypeError: If configuration type is not supported
    """
    
    if isinstance(config, QLearningConfig):
        return validate_qlearning_config(config)
    elif isinstance(config, SensorConfig):
        return validate_sensor_config(config)
    else:
        raise TypeError(f"Unsupported configuration type: {type(config)}")