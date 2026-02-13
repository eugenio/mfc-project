"""
Configuration I/O utilities for loading and saving configuration files.
Supports YAML and JSON formats with validation.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union
from dataclasses import asdict, is_dataclass
import logging

from .qlearning_config import QLearningConfig, QLearningRewardWeights, StateSpaceConfig
from .sensor_config import SensorConfig, EISConfig, QCMConfig, SensorFusionConfig, FusionMethod
from .parameter_validation import validate_qlearning_config, validate_sensor_config

# Set up logging
logger = logging.getLogger(__name__)


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert dataclass to dictionary recursively.
    
    Args:
        obj: Dataclass object or nested structure
        
    Returns:
        Dictionary representation
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        result: Dict[str, Any] = {}
        for field_name, field_value in asdict(obj).items():
            if isinstance(field_value, dict):
                # Handle nested dictionaries
                result[field_name] = field_value
            elif hasattr(field_value, '__dataclass_fields__'):
                # Handle nested dataclasses
                result[field_name] = dataclass_to_dict(field_value)
            elif isinstance(field_value, FusionMethod):
                # Handle enum values
                result[field_name] = field_value.value
            else:
                result[field_name] = field_value
        return result
    return obj


def dict_to_dataclass(data: Dict[str, Any], dataclass_type: type) -> Any:
    """
    Convert dictionary to dataclass instance.
    
    Args:
        data: Dictionary with configuration data
        dataclass_type: Target dataclass type
        
    Returns:
        Dataclass instance
    """
    # Handle nested dataclasses
    if dataclass_type == QLearningConfig:
        if 'rewards' in data and isinstance(data['rewards'], dict):
            data['rewards'] = QLearningRewardWeights(**data['rewards'])
        if 'state_space' in data and isinstance(data['state_space'], dict):
            data['state_space'] = StateSpaceConfig(**data['state_space'])
    elif dataclass_type == SensorConfig:
        if 'eis' in data and isinstance(data['eis'], dict):
            data['eis'] = EISConfig(**data['eis'])
        if 'qcm' in data and isinstance(data['qcm'], dict):
            data['qcm'] = QCMConfig(**data['qcm'])
        if 'fusion' in data and isinstance(data['fusion'], dict):
            # Handle FusionMethod enum
            if 'method' in data['fusion'] and isinstance(data['fusion']['method'], str):
                data['fusion']['method'] = FusionMethod(data['fusion']['method'])
            data['fusion'] = SensorFusionConfig(**data['fusion'])
    
    return dataclass_type(**data)


def save_config(config: Union[QLearningConfig, SensorConfig], 
                filepath: Union[str, Path],
                format: str = 'yaml') -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration object to save
        filepath: Output file path
        format: File format ('yaml' or 'json')
        
    Raises:
        ValueError: If format is not supported
        ConfigValidationError: If configuration is invalid
    """
    # Validate configuration
    if isinstance(config, QLearningConfig):
        validate_qlearning_config(config)
    elif isinstance(config, SensorConfig):
        validate_sensor_config(config)
    else:
        raise ValueError(f"Unsupported configuration type: {type(config)}")
    
    # Convert to dictionary
    config_dict = dataclass_to_dict(config)
    
    # Ensure path exists
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save based on format
    if format.lower() == 'yaml':
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    elif format.lower() == 'json':
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")
    
    logger.info(f"Configuration saved to {filepath}")


def load_config(filepath: Union[str, Path], 
                config_type: type) -> Union[QLearningConfig, SensorConfig]:
    """
    Load configuration from file.
    
    Args:
        filepath: Configuration file path
        config_type: Expected configuration type (QLearningConfig or SensorConfig)
        
    Returns:
        Configuration object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
        ConfigValidationError: If configuration is invalid
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    # Load based on file extension
    if filepath.suffix.lower() in ['.yaml', '.yml']:
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif filepath.suffix.lower() == '.json':
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    # Convert to dataclass
    config = dict_to_dataclass(config_dict, config_type)
    
    # Validate configuration
    if isinstance(config, QLearningConfig):
        validate_qlearning_config(config)
    elif isinstance(config, SensorConfig):
        validate_sensor_config(config)
    
    logger.info(f"Configuration loaded from {filepath}")
    return config


def merge_configs(base_config: Union[QLearningConfig, SensorConfig],
                  override_dict: Dict[str, Any]) -> Union[QLearningConfig, SensorConfig]:
    """
    Merge configuration with override values.
    
    Args:
        base_config: Base configuration object
        override_dict: Dictionary with values to override
        
    Returns:
        New configuration object with merged values
    """
    # Convert base config to dict
    config_dict = dataclass_to_dict(base_config)
    
    # Deep merge override values
    def deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    merged_dict = deep_merge(config_dict, override_dict)
    
    # Convert back to dataclass
    config_type = type(base_config)
    merged_config = dict_to_dataclass(merged_dict, config_type)
    
    # Validate merged configuration
    if isinstance(merged_config, QLearningConfig):
        validate_qlearning_config(merged_config)
    elif isinstance(merged_config, SensorConfig):
        validate_sensor_config(merged_config)
    
    return merged_config