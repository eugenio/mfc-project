def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert dataclass to dictionary recursively.

    Args:
        obj: Dataclass object or nested structure

    Returns:
        Dictionary representation

    """
    if is_dataclass(obj) and not isinstance(obj, type):
        # Custom dict factory to handle enums
        def enum_dict_factory(field_list):
            result = {}
            for k, v in field_list:
                if isinstance(v, Enum):
                    result[k] = v.value
                else:
                    result[k] = v
            return result

        result = asdict(obj, dict_factory=enum_dict_factory)
        # Recursively convert all values for serialization
        return convert_values_for_serialization(result)
    return obj
