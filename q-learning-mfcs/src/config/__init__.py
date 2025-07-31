"""
Configuration module for MFC simulation parameters.
This module provides configuration classes, validation, and I/O functionality for all major subsystems.
"""

from .qlearning_config import QLearningConfig, QLearningRewardWeights, StateSpaceConfig, DEFAULT_QLEARNING_CONFIG
from .sensor_config import SensorConfig, EISConfig, QCMConfig, SensorFusionConfig, FusionMethod, HIGH_ACCURACY_SENSOR_CONFIG
from .parameter_validation import (
    validate_qlearning_config,
    validate_sensor_config,
    ConfigValidationError
)
from .config_io import save_config, load_config, merge_configs

__all__ = [
    'QLearningConfig',
    'QLearningRewardWeights',
    'StateSpaceConfig',
    'DEFAULT_QLEARNING_CONFIG',
    'SensorConfig',
    'EISConfig',
    'QCMConfig',
    'SensorFusionConfig',
    'FusionMethod',
    'HIGH_ACCURACY_SENSOR_CONFIG',
    'validate_qlearning_config',
    'validate_sensor_config',
    'ConfigValidationError',
    'save_config',
    'load_config',
    'merge_configs'
]
