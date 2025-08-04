"""Configuration validation test module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from typing import Any

# Import validation modules
from q_learning_mfcs.src.config.parameter_validation import (
    ConfigValidationError,
    validate_range,
    validate_positive,
    validate_probability,
    validate_qlearning_config,
    validate_qlearning_reward_weights,
    validate_sensor_config,
    validate_eis_config,
    validate_qcm_config,
    validate_sensor_fusion_config,
    validate_all_configurations,
    quick_validate
)

# Import config classes for testing
from q_learning_mfcs.src.config.qlearning_config import (
    QLearningConfig,
    QLearningRewardWeights,
    SensorConfig,
    EISConfig,
    QCMConfig,
    SensorFusionConfig
)
