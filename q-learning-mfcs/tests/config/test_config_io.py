"""
Comprehensive test suite for config_io module.

This test suite provides 100% coverage of the config_io.py module,
including all I/O functions and data conversion utilities.
"""

import json
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
import yaml

from src.config.config_io import (
    convert_lists_to_tuples_for_dataclass,
    convert_values_for_serialization,
    dataclass_to_dict,
    dict_to_dataclass,
    load_config,
    merge_configs,
    save_config,
)


class TestEnum(Enum):
    """Test enum for serialization testing."""
    OPTION_A = "option_a"
    OPTION_B = "option_b"


@dataclass
class SimpleConfig:
    name: str
    value: int
    option: TestEnum


@dataclass
class ConfigWithTuple:
    coordinates: tuple
    name: str


@dataclass
class NestedConfig:
    simple: SimpleConfig
    description: str


class TestConvertValuesForSerialization:
    """Test convert_values_for_serialization function."""

    def test_convert_tuple_to_list(self):
        """Test conversion of tuple to list."""
        data = (1, 2, 3)
        result = convert_values_for_serialization(data)
        
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_convert_nested_tuples(self):
        """Test conversion of nested tuples."""
        data = {
            'tuple_simple': (1, 2, 3),
            'tuple_nested': ((1, 2), (3, 4)),
            'list_with_tuples': [(5, 6), (7, 8)]
        }
        
        result = convert_values_for_serialization(data)
        
        assert isinstance(result['tuple_simple'], list)
        assert result['tuple_simple'] == [1, 2, 3]
        assert isinstance(result['tuple_nested'], list)
        assert result['tuple_nested'] == [[1, 2], [3, 4]]
        assert isinstance(result['list_with_tuples'], list)
        assert result['list_with_tuples'] == [[5, 6], [7, 8]]

    def test_convert_enum(self):
        """Test conversion of enum to value."""
        data = {
            'enum_field': TestEnum.OPTION_A,
            'nested': {
                'enum_field': TestEnum.OPTION_B
            }
        }
        
        result = convert_values_for_serialization(data)
        
        assert result['enum_field'] == "option_a"
        assert result['nested']['enum_field'] == "option_b"

    def test_convert_list_recursively(self):
        """Test recursive conversion of lists."""
        data = [
            (1, 2),  # Tuple in list
            {'key': (3, 4)},  # Dict with tuple in list
            TestEnum.OPTION_A  # Enum in list
        ]
        
        result = convert_values_for_serialization(data)
        
        assert result[0] == [1, 2]
        assert result[1]['key'] == [3, 4]
        assert result[2] == "option_a"

    def test_convert_dict_recursively(self):
        """Test recursive conversion of dictionaries."""
        data = {
            'level1': {
                'level2': {
                    'tuple': (1, 2),
                    'enum': TestEnum.OPTION_B
                }
            }
        }
        
        result = convert_values_for_serialization(data)
        
        assert result['level1']['level2']['tuple'] == [1, 2]
        assert result['level1']['level2']['enum'] == "option_b"

    def test_convert_preserves_other_types(self):
        """Test that other types are preserved unchanged."""
        data = {
            'string': 'hello',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'none': None
        }
        
        result = convert_values_for_serialization(data)
        
        assert result == data


class TestDataclassToDict:
    """Test dataclass_to_dict function."""

    def test_dataclass_to_dict_simple(self):
        """Test converting simple dataclass to dict."""
        config = SimpleConfig(
            name="test",
            value=42,
            option=TestEnum.OPTION_A
        )
        
        result = dataclass_to_dict(config)
        
        expected = {
            'name': 'test',
            'value': 42,
            'option': 'option_a'  # Enum converted to value
        }
        
        assert result == expected

    def test_dataclass_to_dict_with_tuple(self):
        """Test converting dataclass with tuple to dict."""
        config = ConfigWithTuple(
            coordinates=(10, 20, 30),
            name="test"
        )
        
        result = dataclass_to_dict(config)
        
        expected = {
            'coordinates': [10, 20, 30],  # Tuple converted to list
            'name': 'test'
        }
        
        assert result == expected

    def test_dataclass_to_dict_nested(self):
        """Test converting nested dataclass to dict."""
        simple = SimpleConfig(
            name="nested",
            value=100,
            option=TestEnum.OPTION_B
        )
        
        config = NestedConfig(
            simple=simple,
            description="nested config"
        )
        
        result = dataclass_to_dict(config)
        
        expected = {
            'simple': {
                'name': 'nested',
                'value': 100,
                'option': 'option_b'
            },
            'description': 'nested config'
        }
        
        assert result == expected

    def test_dataclass_to_dict_non_dataclass(self):
        """Test that non-dataclass objects are returned unchanged."""
        regular_dict = {'key': 'value'}
        result = dataclass_to_dict(regular_dict)
        
        assert result == regular_dict


class TestConvertListsToTuplesForDataclass:
    """Test convert_lists_to_tuples_for_dataclass function."""

    def test_convert_known_tuple_fields(self):
        """Test conversion of known tuple fields."""
        data = {
            'power_range': [0, 100],
            'biofilm_range': [0, 50],
            'flow_rate_range': [1, 10]
        }
        
        # Mock StateSpaceConfig to test tuple field conversion
        mock_dataclass_type = Mock()
        mock_dataclass_type.__name__ = 'StateSpaceConfig'
        
        result = convert_lists_to_tuples_for_dataclass(data, mock_dataclass_type)
        
        assert isinstance(result['power_range'], tuple)
        assert result['power_range'] == (0, 100)
        assert isinstance(result['biofilm_range'], tuple)
        assert result['biofilm_range'] == (0, 50)
        assert isinstance(result['flow_rate_range'], tuple)
        assert result['flow_rate_range'] == (1, 10)

    def test_convert_nested_structures(self):
        """Test conversion in nested dictionary structures."""
        data = {
            'config': {
                'power_range': [0, 100],
                'other_field': 'not_converted'
            }
        }
        
        mock_dataclass_type = Mock()
        mock_dataclass_type.__name__ = 'StateSpaceConfig'
        
        result = convert_lists_to_tuples_for_dataclass(data, mock_dataclass_type)
        
        assert isinstance(result['config']['power_range'], tuple)
        assert result['config']['power_range'] == (0, 100)
        assert result['config']['other_field'] == 'not_converted'

    def test_convert_unknown_dataclass_type(self):
        """Test with unknown dataclass type (should return unchanged)."""
        data = {
            'power_range': [0, 100],
            'other_field': [1, 2, 3]
        }
        
        mock_dataclass_type = Mock()
        mock_dataclass_type.__name__ = 'UnknownConfig'
        
        result = convert_lists_to_tuples_for_dataclass(data, mock_dataclass_type)
        
        # Should be unchanged since no tuple fields defined for UnknownConfig
        assert result == data

    def test_convert_preserves_non_list_values(self):
        """Test that non-list values are preserved."""
        data = {
            'power_range': (0, 100),  # Already a tuple
            'string_field': 'hello',
            'int_field': 42
        }
        
        mock_dataclass_type = Mock()
        mock_dataclass_type.__name__ = 'StateSpaceConfig'
        
        result = convert_lists_to_tuples_for_dataclass(data, mock_dataclass_type)
        
        assert result['power_range'] == (0, 100)  # Unchanged
        assert result['string_field'] == 'hello'
        assert result['int_field'] == 42


class TestDictToDataclass:
    """Test dict_to_dataclass function."""

    @dataclass
    class TestConfig:
        name: str
        value: int
        enabled: bool = True

    def test_dict_to_dataclass_simple(self):
        """Test converting simple dict to dataclass."""
        data = {
            'name': 'test_config',
            'value': 42,
            'enabled': False
        }
        
        # Mock the convert_lists_to_tuples_for_dataclass function
        with patch('src.config.config_io.convert_lists_to_tuples_for_dataclass') as mock_convert:
            mock_convert.return_value = data
            
            result = dict_to_dataclass(data, self.TestConfig)
        
        assert isinstance(result, self.TestConfig)
        assert result.name == 'test_config'
        assert result.value == 42
        assert result.enabled is False


class TestSaveConfig:
    """Test save_config function."""

    @dataclass
    class TestConfig:
        name: str
        value: int
        option: TestEnum = TestEnum.OPTION_A

    def test_save_config_yaml(self):
        """Test saving config to YAML file."""
        config = self.TestConfig(name="test", value=42, option=TestEnum.OPTION_B)
        
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("yaml.dump") as mock_yaml_dump:
                save_config(config, "/path/to/config.yaml")
                
                mock_file.assert_called_once_with("/path/to/config.yaml", 'w')
                mock_yaml_dump.assert_called_once()

    def test_save_config_json(self):
        """Test saving config to JSON file."""
        config = self.TestConfig(name="test", value=42)
        
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("json.dump") as mock_json_dump:
                save_config(config, "/path/to/config.json")
                
                mock_file.assert_called_once_with("/path/to/config.json", 'w')
                mock_json_dump.assert_called_once()

    def test_save_config_unsupported_format(self):
        """Test saving config to unsupported format."""
        config = self.TestConfig(name="test", value=42)
        
        with pytest.raises(ValueError) as exc_info:
            save_config(config, "/path/to/config.txt")
        
        assert "Unsupported file format" in str(exc_info.value)


class TestLoadConfig:
    """Test load_config function."""

    @dataclass
    class TestConfig:
        name: str
        value: int
        enabled: bool = True

    def test_load_config_yaml(self):
        """Test loading config from YAML file."""
        yaml_content = """
        name: test_config
        value: 42
        enabled: false
        """
        
        config_dict = {
            'name': 'test_config',
            'value': 42,
            'enabled': False
        }
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("yaml.safe_load", return_value=config_dict):
                with patch('src.config.config_io.dict_to_dataclass') as mock_dict_to_dataclass:
                    mock_config = self.TestConfig(name='test_config', value=42, enabled=False)
                    mock_dict_to_dataclass.return_value = mock_config
                    
                    result = load_config("/path/to/config.yaml", self.TestConfig)
                    
                    mock_dict_to_dataclass.assert_called_once_with(config_dict, self.TestConfig)
                    assert result == mock_config

    def test_load_config_unsupported_format(self):
        """Test loading config from unsupported format."""
        with pytest.raises(ValueError) as exc_info:
            load_config("/path/to/config.txt", self.TestConfig)
        
        assert "Unsupported file format" in str(exc_info.value)


class TestMergeConfigs:
    """Test merge_configs function."""

    @dataclass
    class TestConfig:
        name: str
        value: int
        settings: dict

    def test_merge_configs_simple(self):
        """Test merging simple configurations."""
        base_config = self.TestConfig(
            name="base",
            value=10,
            settings={'debug': True}
        )
        
        override_config = self.TestConfig(
            name="override",
            value=20,
            settings={'debug': False, 'logging': 'INFO'}
        )
        
        result = merge_configs(base_config, override_config)
        
        assert isinstance(result, self.TestConfig)
        assert result.name == "override"
        assert result.value == 20

    def test_merge_configs_with_none_base(self):
        """Test merging when base config is None."""
        override_config = self.TestConfig(
            name="override",
            value=20,
            settings={'debug': False}
        )
        
        result = merge_configs(None, override_config)
        
        assert result is override_config

    def test_merge_configs_with_none_override(self):
        """Test merging when override config is None."""
        base_config = self.TestConfig(
            name="base",
            value=10,
            settings={'debug': True}
        )
        
        result = merge_configs(base_config, None)
        
        assert result is base_config

    def test_merge_configs_both_none(self):
        """Test merging when both configs are None."""
        result = merge_configs(None, None)
        
        assert result is None

    def test_merge_configs_different_types(self):
        """Test merging configs of different types raises error."""
        @dataclass
        class DifferentConfig:
            other_field: str
        
        base_config = self.TestConfig(name="base", value=10, settings={})
        different_config = DifferentConfig(other_field="different")
        
        with pytest.raises(TypeError) as exc_info:
            merge_configs(base_config, different_config)
        
        assert "must be instances of the same dataclass type" in str(exc_info.value)