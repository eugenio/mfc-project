"""Configuration I/O utilities for loading and saving configuration files.

Supports YAML and JSON formats with validation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from .electrode_config import (
    ElectrodeConfiguration,
    ElectrodeGeometry,
    ElectrodeMaterial,
    MaterialProperties,
)
from .parameter_validation import validate_qlearning_config, validate_sensor_config
from .qlearning_config import QLearningConfig, QLearningRewardWeights, StateSpaceConfig
from .sensor_config import (
    EISConfig,
    FusionMethod,
    QCMConfig,
    SensorConfig,
    SensorFusionConfig,
)

# Set up logging
logger = logging.getLogger(__name__)


def convert_values_for_serialization(obj: Any) -> Any:
    """Convert values recursively for serialization compatibility.

    Handles enums, tuples, and nested structures.

    Args:
        obj: Object to convert

    Returns:
        Converted object

    """
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, tuple):
        return [convert_values_for_serialization(item) for item in obj]
    if isinstance(obj, dict):
        return {
            key: convert_values_for_serialization(value) for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [convert_values_for_serialization(item) for item in obj]
    return obj


def _enum_dict_factory(field_list: list[tuple[str, Any]]) -> dict[str, Any]:
    """Convert Enum values recursively for asdict dict_factory.

    Handle enums nested inside dicts, lists, and tuples, converting
    them to their string/value representation for serialization.

    Args:
        field_list: List of (field_name, value) tuples from asdict

    Returns:
        Dictionary with enum values converted to their value representation

    """
    return {k: convert_values_for_serialization(v) for k, v in field_list}


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert dataclass to dictionary recursively.

    Args:
        obj: Dataclass object or nested structure

    Returns:
        Dictionary representation

    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj, dict_factory=_enum_dict_factory)
    return obj


def convert_lists_to_tuples_for_dataclass(
    data: dict[str, Any],
    dataclass_type: type,
) -> dict[str, Any]:
    """Convert lists back to tuples for fields that expect tuples.

    Args:
        data: Dictionary with configuration data
        dataclass_type: Target dataclass type

    Returns:
        Dictionary with tuples restored

    """
    # Fields that should be tuples
    tuple_fields = {
        "QLearningConfig": [],
        "StateSpaceConfig": [
            "power_range",
            "biofilm_range",
            "substrate_range",
            "reservoir_substrate_range",
            "cell_substrate_range",
            "outlet_substrate_range",
            "outlet_error_range",
            "flow_rate_range",
        ],
        "EISConfig": ["frequency_range"],
        "QCMConfig": ["mass_range", "frequency_shift_range", "dissipation_range"],
        "SensorFusionConfig": [],
    }

    def convert_recursive(obj: Any) -> Any:
        """Recursively convert lists to tuples in nested structures."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if isinstance(value, dict):
                    result[key] = convert_recursive(value)
                else:
                    result[key] = value

            # Apply tuple conversions to the current level
            class_name = dataclass_type.__name__
            if class_name in tuple_fields:
                for field in tuple_fields[class_name]:
                    if field in result and isinstance(result[field], list):
                        result[field] = tuple(result[field])

            return result
        return obj

    return convert_recursive(data)


def dict_to_dataclass(data: dict[str, Any], dataclass_type: type) -> Any:
    """Convert dictionary to dataclass instance.

    Args:
        data: Dictionary with configuration data
        dataclass_type: Target dataclass type

    Returns:
        Dataclass instance

    """
    # Convert lists back to tuples where needed
    data = convert_lists_to_tuples_for_dataclass(data, dataclass_type)

    # Filter out fields that have init=False
    if hasattr(dataclass_type, "__dataclass_fields__"):
        init_fields = {
            name: field
            for name, field in dataclass_type.__dataclass_fields__.items()
            if field.init
        }
        data = {key: value for key, value in data.items() if key in init_fields}

    # Handle nested dataclasses
    if dataclass_type == QLearningConfig:
        if "reward_weights" in data and isinstance(data["reward_weights"], dict):
            data["reward_weights"] = QLearningRewardWeights(**data["reward_weights"])
        if "state_space" in data and isinstance(data["state_space"], dict):
            state_data = convert_lists_to_tuples_for_dataclass(
                data["state_space"],
                StateSpaceConfig,
            )
            data["state_space"] = StateSpaceConfig(**state_data)

        # Handle electrode configurations
        if "anode_config" in data and isinstance(data["anode_config"], dict):
            anode = data["anode_config"]
            if "material" in anode and isinstance(anode["material"], str):
                anode["material"] = ElectrodeMaterial(anode["material"])
            if "geometry" in anode and isinstance(anode["geometry"], dict):
                geom = anode["geometry"]
                if "geometry_type" in geom and isinstance(geom["geometry_type"], str):
                    geom["geometry_type"] = ElectrodeGeometry(geom["geometry_type"])
                from .electrode_config import ElectrodeGeometrySpec

                anode["geometry"] = ElectrodeGeometrySpec(**geom)
            if "material_properties" in anode and isinstance(
                anode["material_properties"],
                dict,
            ):
                anode["material_properties"] = MaterialProperties(
                    **anode["material_properties"],
                )
            data["anode_config"] = ElectrodeConfiguration(**anode)

        if "cathode_config" in data and isinstance(data["cathode_config"], dict):
            cathode = data["cathode_config"]
            if "material" in cathode and isinstance(cathode["material"], str):
                cathode["material"] = ElectrodeMaterial(cathode["material"])
            if "geometry" in cathode and isinstance(cathode["geometry"], dict):
                geom = cathode["geometry"]
                if "geometry_type" in geom and isinstance(geom["geometry_type"], str):
                    geom["geometry_type"] = ElectrodeGeometry(geom["geometry_type"])
                from .electrode_config import ElectrodeGeometrySpec

                cathode["geometry"] = ElectrodeGeometrySpec(**geom)
            if "material_properties" in cathode and isinstance(
                cathode["material_properties"],
                dict,
            ):
                cathode["material_properties"] = MaterialProperties(
                    **cathode["material_properties"],
                )
            data["cathode_config"] = ElectrodeConfiguration(**cathode)
    elif dataclass_type == SensorConfig:
        # Handle FusionMethod enum at the SensorConfig level
        if "fusion_method" in data and isinstance(data["fusion_method"], str):
            data["fusion_method"] = FusionMethod(data["fusion_method"])
        if "eis" in data and isinstance(data["eis"], dict):
            eis_data = convert_lists_to_tuples_for_dataclass(data["eis"], EISConfig)
            data["eis"] = EISConfig(**eis_data)
        if "qcm" in data and isinstance(data["qcm"], dict):
            qcm_data = convert_lists_to_tuples_for_dataclass(data["qcm"], QCMConfig)
            data["qcm"] = QCMConfig(**qcm_data)
        if "fusion" in data and isinstance(data["fusion"], dict):
            fusion_data = convert_lists_to_tuples_for_dataclass(
                data["fusion"],
                SensorFusionConfig,
            )
            data["fusion"] = SensorFusionConfig(**fusion_data)

    return dataclass_type(**data)


def save_config(
    config: QLearningConfig | SensorConfig,
    filepath: str | Path,
    format: str = "yaml",
) -> None:
    """Save configuration to file.

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
        msg = f"Unsupported configuration type: {type(config)}"
        raise ValueError(msg)

    # Convert to dictionary
    config_dict = dataclass_to_dict(config)

    # Ensure path exists
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save based on format
    if format.lower() == "yaml":
        with open(filepath, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    elif format.lower() == "json":
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)
    else:
        msg = f"Unsupported format: {format}. Use 'yaml' or 'json'"
        raise ValueError(msg)

    logger.info(f"Configuration saved to {filepath}")


def load_config(
    filepath: str | Path,
    config_type: type,
) -> QLearningConfig | SensorConfig:
    """Load configuration from file.

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
        msg = f"Configuration file not found: {filepath}"
        raise FileNotFoundError(msg)

    # Load based on file extension
    if filepath.suffix.lower() in [".yaml", ".yml"]:
        with open(filepath) as f:
            config_dict = yaml.safe_load(f)
    elif filepath.suffix.lower() == ".json":
        with open(filepath) as f:
            config_dict = json.load(f)
    else:
        msg = f"Unsupported file format: {filepath.suffix}"
        raise ValueError(msg)

    # Convert to dataclass
    config = dict_to_dataclass(config_dict, config_type)

    # Validate configuration
    if isinstance(config, QLearningConfig):
        validate_qlearning_config(config)
    elif isinstance(config, SensorConfig):
        validate_sensor_config(config)

    logger.info(f"Configuration loaded from {filepath}")
    return config


def merge_configs(
    base_config: QLearningConfig | SensorConfig,
    override_dict: dict[str, Any],
) -> QLearningConfig | SensorConfig:
    """Merge configuration with override values.

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
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
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
