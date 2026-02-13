"""Configuration module for MFC simulation parameters.

This module provides configuration classes, validation, and I/O functionality
for all major subsystems.

Validation Modules
------------------
The config package provides three types of validation:

1. **Parameter Validation** (``parameter_validation.py``):
   - Generic validation utilities (``validate_range``, ``validate_positive``, etc.)
   - Q-learning config validation (``validate_qlearning_config``)
   - Sensor config validation (``validate_sensor_config``)

2. **Biological Validation** (``biological_validation.py``):
   - Kinetic parameters validation (``validate_kinetic_parameters``)
   - Metabolic reaction validation (``validate_metabolic_reaction``)
   - Species metabolic config validation (``validate_species_metabolic_config``)
   - Biofilm kinetics validation (``validate_biofilm_kinetics_config``)
   - Substrate and electrochemical validation

3. **Model Validation** (``model_validation.py``):
   - ML/statistical model validation (cross-validation, metrics)
   - Not for config validation - use for predictive model evaluation
"""

from .biological_validation import (
    validate_all_biological_configs,
    validate_biofilm_kinetics_config,
    validate_comprehensive_substrate_config,
    validate_electrochemical_config,
    validate_kinetic_parameters,
    validate_metabolic_reaction,
    validate_species_metabolic_config,
    validate_substrate_degradation_pathway,
    validate_substrate_kinetics_config,
)
from .config_io import load_config, merge_configs, save_config
from .parameter_validation import (
    ConfigValidationError,
    validate_positive,
    validate_probability,
    validate_qlearning_config,
    validate_range,
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
    "DEFAULT_QLEARNING_CONFIG",
    "HIGH_ACCURACY_SENSOR_CONFIG",
    "ConfigValidationError",
    "EISConfig",
    "FusionMethod",
    "QCMConfig",
    "QLearningConfig",
    "QLearningRewardWeights",
    "SensorConfig",
    "SensorFusionConfig",
    "StateSpaceConfig",
    "load_config",
    "merge_configs",
    "save_config",
    "validate_all_biological_configs",
    "validate_biofilm_kinetics_config",
    "validate_comprehensive_substrate_config",
    "validate_electrochemical_config",
    "validate_kinetic_parameters",
    "validate_metabolic_reaction",
    "validate_positive",
    "validate_probability",
    "validate_qlearning_config",
    "validate_range",
    "validate_sensor_config",
    "validate_species_metabolic_config",
    "validate_substrate_degradation_pathway",
    "validate_substrate_kinetics_config",
]
