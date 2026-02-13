"""
Configuration module for MFC simulation parameters.
This module provides configuration classes, validation, and I/O functionality for all major subsystems.
"""

from .config_io import load_config, merge_configs, save_config
from .parameter_validation import (
    ConfigValidationError,
    validate_qlearning_config,
    validate_sensor_config,
)
from .qlearning_config import (
    DEFAULT_QLEARNING_CONFIG,
    QLearningConfig,
    QLearningRewardWeights,
    StateSpaceConfig,
)
from .sensor_config import (
    HIGH_ACCURACY_SENSOR_CONFIG,
    EISConfig,
    FusionMethod,
    QCMConfig,
    SensorConfig,
    SensorFusionConfig,
)

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
