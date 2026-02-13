"""
Fix for enum serialization in config_io.py
"""
from enum import Enum


def enum_dict_factory(field_list):
    """
    Custom dict factory for asdict that converts Enum values to their string values.
    """
    def convert_value(obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: convert_value(v) for k, v in obj.items()}
        elif isinstance(obj, list | tuple):
            return type(obj)(convert_value(v) for v in obj)
        else:
            return obj

    return {k: convert_value(v) for k, v in field_list}
