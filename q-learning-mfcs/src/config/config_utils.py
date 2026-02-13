"""
Configuration Validation and Loading Utilities

This module provides utilities for loading, validating, and converting
configuration files and objects for the MFC system.

Functions:
- load_yaml_config: Load YAML configuration files
- load_json_config: Load JSON configuration files
- validate_config_schema: Validate configuration against schema
- convert_legacy_config: Convert legacy configuration formats
- merge_config_files: Merge multiple configuration files
- environment_substitution: Substitute environment variables in configs

Classes:
- ConfigLoader: Comprehensive configuration loader
- ConfigValidator: Schema-based configuration validator
- ConfigConverter: Configuration format converter

Features:
- Schema validation with JSON Schema
- Environment variable substitution
- Legacy configuration migration
- Configuration file merging and inheritance
- Type conversion and validation
- Error reporting and debugging
"""

import json
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import yaml

if TYPE_CHECKING:
    pass
import logging
import warnings
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime

from jsonschema import ValidationError, validate

# Import configuration classes
from .config_manager import ConfigurationError


class ConfigurationFormatError(ConfigurationError):
    """Exception for configuration format errors."""
    pass


class ConfigurationSchemaError(ConfigurationError):
    """Exception for configuration schema validation errors."""
    pass


@dataclass
class ValidationResult:
    """Results of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    schema_version: Optional[str] = None
    validation_time: float = 0.0


def load_yaml_config(file_path: Union[str, Path],
                    validate_schema: bool = True,
                    substitute_env: bool = True) -> Dict[str, Any]:
    """
    Load YAML configuration file with optional validation and environment substitution.
    
    Args:
        file_path: Path to YAML file
        validate_schema: Whether to validate against schema
        substitute_env: Whether to substitute environment variables
        
    Returns:
        Loaded configuration dictionary
        
    Raises:
        ConfigurationFormatError: If file format is invalid
        ConfigurationSchemaError: If schema validation fails
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ConfigurationFormatError(f"Configuration file not found: {file_path}")

    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()

        # Substitute environment variables if requested
        if substitute_env:
            content = substitute_environment_variables(content)

        # Load YAML
        config = yaml.safe_load(content)

        if config is None:
            config = {}

        # Validate schema if requested
        if validate_schema:
            validation_result = validate_config_schema(config)
            if not validation_result.is_valid:
                raise ConfigurationSchemaError(
                    f"Schema validation failed: {', '.join(validation_result.errors)}")

        return config

    except yaml.YAMLError as e:
        raise ConfigurationFormatError(f"Invalid YAML format in {file_path}: {e}")
    except Exception as e:
        raise ConfigurationFormatError(f"Failed to load configuration from {file_path}: {e}")


def load_json_config(file_path: Union[str, Path],
                    validate_schema: bool = True,
                    substitute_env: bool = True) -> Dict[str, Any]:
    """
    Load JSON configuration file with optional validation and environment substitution.
    
    Args:
        file_path: Path to JSON file
        validate_schema: Whether to validate against schema
        substitute_env: Whether to substitute environment variables
        
    Returns:
        Loaded configuration dictionary
        
    Raises:
        ConfigurationFormatError: If file format is invalid
        ConfigurationSchemaError: If schema validation fails
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ConfigurationFormatError(f"Configuration file not found: {file_path}")

    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()

        # Substitute environment variables if requested
        if substitute_env:
            content = substitute_environment_variables(content)

        # Load JSON
        config = json.loads(content)

        # Validate schema if requested
        if validate_schema:
            validation_result = validate_config_schema(config)
            if not validation_result.is_valid:
                raise ConfigurationSchemaError(
                    f"Schema validation failed: {', '.join(validation_result.errors)}")

        return config

    except json.JSONDecodeError as e:
        raise ConfigurationFormatError(f"Invalid JSON format in {file_path}: {e}")
    except Exception as e:
        raise ConfigurationFormatError(f"Failed to load configuration from {file_path}: {e}")


def substitute_environment_variables(content: str) -> str:
    """
    Substitute environment variables in configuration content.
    
    Supports formats:
    - ${VAR_NAME}
    - ${VAR_NAME:default_value}
    - $VAR_NAME (simple form)
    
    Args:
        content: Configuration file content
        
    Returns:
        Content with environment variables substituted
    """
    # Pattern for ${VAR_NAME:default} format
    pattern1 = re.compile(r'\$\{([^}:]+):([^}]*)\}')
    # Pattern for ${VAR_NAME} format
    pattern2 = re.compile(r'\$\{([^}]+)\}')
    # Pattern for $VAR_NAME format (word boundaries)
    pattern3 = re.compile(r'\$([A-Za-z_][A-Za-z0-9_]*)')

    def replace_with_default(match):
        var_name = match.group(1)
        default_value = match.group(2)
        return os.environ.get(var_name, default_value)

    def replace_simple(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))  # Return original if not found

    # Apply substitutions in order
    content = pattern1.sub(replace_with_default, content)
    content = pattern2.sub(replace_simple, content)
    content = pattern3.sub(replace_simple, content)

    return content


def get_config_schema() -> Dict[str, Any]:
    """
    Get JSON schema for MFC configuration validation.
    
    Returns:
        JSON schema dictionary
    """
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "MFC Configuration Schema",
        "type": "object",
        "properties": {
            "biological": {
                "type": "object",
                "properties": {
                    "species_configs": {
                        "type": "object",
                        "patternProperties": {
                            "^[a-zA-Z_][a-zA-Z0-9_]*$": {
                                "type": "object",
                                "properties": {
                                    "max_growth_rate": {"type": "number", "minimum": 0},
                                    "electron_transport_efficiency": {"type": "number", "minimum": 0, "maximum": 1},
                                    "cytochrome_content": {"type": "number", "minimum": 0},
                                    "metabolite_concentrations": {"type": "object"}
                                },
                                "required": ["max_growth_rate", "electron_transport_efficiency"]
                            }
                        }
                    },
                    "substrate_configs": {
                        "type": "object",
                        "patternProperties": {
                            "^[a-zA-Z_][a-zA-Z0-9_]*$": {
                                "type": "object",
                                "properties": {
                                    "molecular_weight": {"type": "number", "minimum": 0},
                                    "chemical_formula": {"type": "string"},
                                    "species_kinetics": {"type": "object"}
                                }
                            }
                        }
                    }
                }
            },
            "control": {
                "type": "object",
                "properties": {
                    "flow_control": {
                        "type": "object",
                        "properties": {
                            "min_flow_rate": {"type": "number", "minimum": 0},
                            "max_flow_rate": {"type": "number", "minimum": 0},
                            "nominal_flow_rate": {"type": "number", "minimum": 0}
                        }
                    },
                    "substrate_control": {
                        "type": "object",
                        "properties": {
                            "target_outlet_concentration": {"type": "number", "minimum": 0},
                            "target_reservoir_concentration": {"type": "number", "minimum": 0},
                            "min_addition_rate": {"type": "number", "minimum": 0},
                            "max_addition_rate": {"type": "number", "minimum": 0}
                        }
                    },
                    "advanced_control": {
                        "type": "object",
                        "properties": {
                            "learning_rate": {"type": "number", "minimum": 0, "maximum": 1},
                            "discount_factor": {"type": "number", "minimum": 0, "maximum": 1},
                            "epsilon": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                }
            },
            "visualization": {
                "type": "object",
                "properties": {
                    "plot_style": {
                        "type": "object",
                        "properties": {
                            "figure_width": {"type": "number", "minimum": 0},
                            "figure_height": {"type": "number", "minimum": 0},
                            "dpi": {"type": "integer", "minimum": 1},
                            "line_width": {"type": "number", "minimum": 0}
                        }
                    },
                    "color_scheme_type": {
                        "type": "string",
                        "enum": ["default", "scientific", "colorblind_friendly", "high_contrast", "grayscale", "publication"]
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["png", "pdf", "svg", "eps"]
                    }
                }
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "version": {"type": "string"},
                    "description": {"type": "string"},
                    "environment": {"type": "string", "enum": ["development", "testing", "production"]},
                    "created_at": {"type": "string", "format": "date-time"},
                    "updated_at": {"type": "string", "format": "date-time"}
                }
            }
        },
        "additionalProperties": True
    }

    return schema


def validate_config_schema(config: Dict[str, Any],
                          schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """
    Validate configuration against JSON schema.
    
    Args:
        config: Configuration dictionary to validate
        schema: JSON schema (uses default if None)
        
    Returns:
        Validation result
    """
    import time
    start_time = time.time()

    if schema is None:
        schema = get_config_schema()

    result = ValidationResult(is_valid=False)

    try:
        validate(instance=config, schema=schema)
        result.is_valid = True

    except ValidationError as e:
        result.errors.append(f"Schema validation error: {e.message}")
        if e.path:
            result.errors.append(f"Error path: {' -> '.join(str(p) for p in e.path)}")

    except Exception as e:
        result.errors.append(f"Validation failed: {str(e)}")

    result.validation_time = time.time() - start_time
    return result


def dataclass_to_dict(obj: Any) -> Any:
    """
    Convert dataclass instance to dictionary recursively.
    
    Args:
        obj: Dataclass instance or other object
        
    Returns:
        Dictionary representation
    """
    if is_dataclass(obj):
        result = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = dataclass_to_dict(value)
        return result
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    else:
        return obj


def dict_to_dataclass(data: Dict[str, Any], dataclass_type: Type) -> Any:
    """
    Convert dictionary to dataclass instance.
    
    Args:
        data: Dictionary data
        dataclass_type: Target dataclass type
        
    Returns:
        Dataclass instance
    """
    if not is_dataclass(dataclass_type):
        raise ValueError(f"Target type {dataclass_type} is not a dataclass")

    # Get field information
    field_types = {f.name: f.type for f in fields(dataclass_type)}
    kwargs = {}

    for field_name, field_type in field_types.items():
        if field_name in data:
            value = data[field_name]

            # Handle nested dataclasses
            if is_dataclass(field_type) and isinstance(value, dict):
                kwargs[field_name] = dict_to_dataclass(value, field_type)
            else:
                kwargs[field_name] = value

    return dataclass_type(**kwargs)


def merge_config_files(*file_paths: Union[str, Path],
                      output_path: Optional[Union[str, Path]] = None,
                      format: str = "yaml") -> Dict[str, Any]:
    """
    Merge multiple configuration files.
    
    Args:
        file_paths: Paths to configuration files to merge
        output_path: Path to save merged configuration
        format: Output format ("yaml" or "json")
        
    Returns:
        Merged configuration dictionary
    """
    merged_config: Dict[str, Any] = {}

    for file_path in file_paths:
        file_path = Path(file_path)

        if file_path.suffix.lower() in ['.yaml', '.yml']:
            config = load_yaml_config(file_path, validate_schema=False)
        elif file_path.suffix.lower() == '.json':
            config = load_json_config(file_path, validate_schema=False)
        else:
            raise ConfigurationFormatError(f"Unsupported file format: {file_path.suffix}")

        # Deep merge configurations
        merged_config = deep_merge_dicts(merged_config, config)

    # Save merged configuration if output path provided
    if output_path:
        save_config_file(merged_config, output_path, format)

    return merged_config


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge into base
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def save_config_file(config: Dict[str, Any],
                    file_path: Union[str, Path],
                    format: str = "yaml") -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        file_path: Output file path
        format: File format ("yaml" or "json")
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "yaml":
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    elif format.lower() == "json":
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, sort_keys=False, default=str)
    else:
        raise ValueError(f"Unsupported format: {format}")


class ConfigLoader:
    """Comprehensive configuration loader with caching and validation."""

    def __init__(self, cache_enabled: bool = True,
                 validate_schema: bool = True):
        """
        Initialize configuration loader.
        
        Args:
            cache_enabled: Enable configuration caching
            validate_schema: Enable schema validation
        """
        self.cache_enabled = cache_enabled
        self.validate_schema = validate_schema
        self.cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self.logger = logging.getLogger(__name__)

    def load_config(self, file_path: Union[str, Path],
                   force_reload: bool = False) -> Dict[str, Any]:
        """
        Load configuration with caching support.
        
        Args:
            file_path: Path to configuration file
            force_reload: Force reload even if cached
            
        Returns:
            Configuration dictionary
        """
        file_path = Path(file_path)
        cache_key = str(file_path.absolute())

        # Check cache
        if (self.cache_enabled and not force_reload and
            cache_key in self.cache):

            cached_config, cached_time = self.cache[cache_key]
            file_mtime = file_path.stat().st_mtime

            if cached_time >= file_mtime:
                self.logger.debug(f"Using cached configuration for {file_path}")
                return cached_config

        # Load configuration
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            config = load_yaml_config(file_path, self.validate_schema)
        elif file_path.suffix.lower() == '.json':
            config = load_json_config(file_path, self.validate_schema)
        else:
            raise ConfigurationFormatError(f"Unsupported file format: {file_path.suffix}")

        # Cache configuration
        if self.cache_enabled:
            self.cache[cache_key] = (config, time.time())

        self.logger.info(f"Loaded configuration from {file_path}")
        return config

    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self.cache.clear()
        self.logger.info("Configuration cache cleared")


class ConfigConverter:
    """Configuration format converter and migrator."""

    def __init__(self):
        """Initialize configuration converter."""
        self.logger = logging.getLogger(__name__)

    def convert_legacy_qlearning_config(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert legacy Q-learning configuration to new format.
        
        Args:
            legacy_config: Legacy configuration dictionary
            
        Returns:
            Converted configuration
        """
        converted = {
            "control": {
                "advanced_control": {}
            },
            "metadata": {
                "version": "2.0.0",
                "description": "Converted from legacy format",
                "created_at": datetime.now().isoformat()
            }
        }

        # Map legacy fields to new structure
        field_mapping = {
            "learning_rate": ["control", "advanced_control", "learning_rate"],
            "discount_factor": ["control", "advanced_control", "discount_factor"],
            "epsilon": ["control", "advanced_control", "epsilon"],
            "epsilon_decay": ["control", "advanced_control", "epsilon_decay"],
            "epsilon_min": ["control", "advanced_control", "epsilon_min"],
            "power_bins": ["control", "advanced_control", "power_bins"],
            "power_max": ["control", "advanced_control", "power_max"],
            "flow_rate_min": ["control", "flow_control", "min_flow_rate"],
            "flow_rate_max": ["control", "flow_control", "max_flow_rate"]
        }

        for legacy_key, new_path in field_mapping.items():
            if legacy_key in legacy_config:
                self._set_nested_value(converted, new_path, legacy_config[legacy_key])

        # Handle rewards structure
        if "rewards" in legacy_config:
            rewards = legacy_config["rewards"]
            reward_mapping = {
                "power_weight": ["control", "advanced_control", "power_reward_weight"],
                "biofilm_weight": ["control", "advanced_control", "biofilm_reward_weight"],
                "consumption_weight": ["control", "advanced_control", "substrate_consumption_weight"]
            }

            for legacy_key, new_path in reward_mapping.items():
                if legacy_key in rewards:
                    self._set_nested_value(converted, new_path, rewards[legacy_key])

        self.logger.info("Converted legacy Q-learning configuration")
        return converted

    def _set_nested_value(self, dictionary: Dict[str, Any],
                         path: List[str], value: Any) -> None:
        """Set nested dictionary value using path."""
        current = dictionary
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def migrate_config_version(self, config: Dict[str, Any],
                              target_version: str = "2.0.0") -> Dict[str, Any]:
        """
        Migrate configuration to target version.
        
        Args:
            config: Configuration to migrate
            target_version: Target version string
            
        Returns:
            Migrated configuration
        """
        current_version = config.get("metadata", {}).get("version", "1.0.0")

        if current_version == target_version:
            return config

        # Version-specific migrations
        if current_version == "1.0.0" and target_version == "2.0.0":
            return self._migrate_v1_to_v2(config)

        warnings.warn(f"No migration path from {current_version} to {target_version}")
        return config

    def _migrate_v1_to_v2(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from version 1.0.0 to 2.0.0."""
        migrated = config.copy()

        # Update metadata
        if "metadata" not in migrated:
            migrated["metadata"] = {}

        migrated["metadata"]["version"] = "2.0.0"
        migrated["metadata"]["migrated_at"] = datetime.now().isoformat()

        # Restructure configuration sections
        # (Add specific migration logic as needed)

        self.logger.info("Migrated configuration from v1.0.0 to v2.0.0")
        return migrated


# Utility functions for common operations
def get_config_value(config: Dict[str, Any], path: str,
                    default: Any = None, separator: str = ".") -> Any:
    """
    Get nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., "control.flow_control.max_flow_rate")
        default: Default value if path not found
        separator: Path separator
        
    Returns:
        Configuration value or default
    """
    keys = path.split(separator)
    current = config

    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def set_config_value(config: Dict[str, Any], path: str,
                    value: Any, separator: str = ".") -> None:
    """
    Set nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path
        value: Value to set
        separator: Path separator
    """
    keys = path.split(separator)
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def validate_config_types(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration data types.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of type validation errors
    """
    errors = []

    # Define expected types for common configuration paths
    type_checks: Dict[str, Union[Type, Tuple[Type, ...]]] = {
        "control.flow_control.min_flow_rate": (int, float),
        "control.flow_control.max_flow_rate": (int, float),
        "control.advanced_control.learning_rate": (int, float),
        "control.advanced_control.discount_factor": (int, float),
        "visualization.plot_style.figure_width": (int, float),
        "visualization.plot_style.figure_height": (int, float),
        "visualization.plot_style.dpi": int,
    }

    for path, expected_type in type_checks.items():
        value = get_config_value(config, path)
        if value is not None and not isinstance(value, expected_type):
            errors.append(f"Invalid type for {path}: expected {expected_type}, got {type(value)}")

    return errors
