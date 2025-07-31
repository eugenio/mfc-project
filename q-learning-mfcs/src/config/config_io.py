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


def convert_values_for_serialization(obj: Any) -> Any:
    """
    Convert values recursively for serialization compatibility.
    
    Args:
        obj: Object to convert
        
    Returns:
        Converted object
    """
    if isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_values_for_serialization(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_values_for_serialization(item) for item in obj]
    elif isinstance(obj, FusionMethod):
        return obj.value
    return obj


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert dataclass to dictionary recursively.
    
    Args:
        obj: Dataclass object or nested structure
        
    Returns:
        Dictionary representation
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        result = asdict(obj)
        # Recursively convert all values for serialization
        return convert_values_for_serialization(result)
    return obj


def convert_lists_to_tuples_for_dataclass(data: Dict[str, Any], dataclass_type: type) -> Dict[str, Any]:
    """
    Convert lists back to tuples for fields that expect tuples.
    
    Args:
        data: Dictionary with configuration data
        dataclass_type: Target dataclass type
        
    Returns:
        Dictionary with tuples restored
    """
    # Fields that should be tuples
    tuple_fields = {
        'QLearningConfig': [],
        'StateSpaceConfig': [
            'power_range', 'biofilm_range', 'substrate_range', 
            'reservoir_substrate_range', 'cell_substrate_range', 'outlet_substrate_range',
            'outlet_error_range', 'flow_rate_range'
        ],
        'EISConfig': ['frequency_range'],
        'QCMConfig': ['mass_range', 'frequency_shift_range', 'dissipation_range'],
        'SensorFusionConfig': []
    }
    
    class_name = dataclass_type.__name__
    if class_name in tuple_fields:
        for field in tuple_fields[class_name]:
            if field in data and isinstance(data[field], list):
                data[field] = tuple(data[field])
    
    return data


def dict_to_dataclass(data: Dict[str, Any], dataclass_type: type) -> Any:
    """
    Convert dictionary to dataclass instance.
    
    Args:
        data: Dictionary with configuration data
        dataclass_type: Target dataclass type
        
    Returns:
        Dataclass instance
    """
    # Convert lists back to tuples where needed
    data = convert_lists_to_tuples_for_dataclass(data, dataclass_type)
    
    # Filter out fields that have init=False
    if hasattr(dataclass_type, '__dataclass_fields__'):
        init_fields = {
            name: field for name, field in dataclass_type.__dataclass_fields__.items()
            if field.init
        }
        data = {key: value for key, value in data.items() if key in init_fields}
    
    # Handle nested dataclasses
    if dataclass_type == QLearningConfig:
        if 'reward_weights' in data and isinstance(data['reward_weights'], dict):
            data['reward_weights'] = QLearningRewardWeights(**data['reward_weights'])
        if 'state_space' in data and isinstance(data['state_space'], dict):
            state_data = convert_lists_to_tuples_for_dataclass(data['state_space'], StateSpaceConfig)
            data['state_space'] = StateSpaceConfig(**state_data)
    elif dataclass_type == SensorConfig:
        # Handle FusionMethod enum at the SensorConfig level
        if 'fusion_method' in data and isinstance(data['fusion_method'], str):
            data['fusion_method'] = FusionMethod(data['fusion_method'])
        if 'eis' in data and isinstance(data['eis'], dict):
            eis_data = convert_lists_to_tuples_for_dataclass(data['eis'], EISConfig)
            data['eis'] = EISConfig(**eis_data)
        if 'qcm' in data and isinstance(data['qcm'], dict):
            qcm_data = convert_lists_to_tuples_for_dataclass(data['qcm'], QCMConfig)
            data['qcm'] = QCMConfig(**qcm_data)
        if 'fusion' in data and isinstance(data['fusion'], dict):
            fusion_data = convert_lists_to_tuples_for_dataclass(data['fusion'], SensorFusionConfig)
            data['fusion'] = SensorFusionConfig(**fusion_data)
    
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