"""Fix for enum serialization in config_io.py."""

from enum import Enum


def enum_dict_factory(field_list):
    """Custom dict factory for asdict that converts Enum values to their string values."""

    def convert_value(obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, dict):
            return {k: convert_value(v) for k, v in obj.items()}
        if isinstance(obj, list | tuple):
            return type(obj)(convert_value(v) for v in obj)
        return obj

    return {k: convert_value(v) for k, v in field_list}
