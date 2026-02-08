"""Tests for config_utils module - 98%+ coverage target."""
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.config_utils import (
    ConfigConverter,
    ConfigLoader,
    ConfigurationFormatError,
    ConfigurationSchemaError,
    ValidationResult,
    dataclass_to_dict,
    deep_merge_dicts,
    dict_to_dataclass,
    get_config_schema,
    get_config_value,
    load_json_config,
    load_yaml_config,
    merge_config_files,
    save_config_file,
    set_config_value,
    substitute_environment_variables,
    validate_config_schema,
    validate_config_types,
)


class TestExceptions:
    def test_format_error(self):
        with pytest.raises(ConfigurationFormatError):
            raise ConfigurationFormatError("bad format")

    def test_schema_error(self):
        with pytest.raises(ConfigurationSchemaError):
            raise ConfigurationSchemaError("bad schema")


class TestValidationResult:
    def test_defaults(self):
        vr = ValidationResult(is_valid=True)
        assert vr.is_valid is True
        assert vr.errors == []
        assert vr.warnings == []


class TestLoadYamlConfig:
    def test_load_valid(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump({"biological": {"key": "val"}}))
        cfg = load_yaml_config(str(f))
        assert cfg["biological"]["key"] == "val"

    def test_not_found(self):
        with pytest.raises(ConfigurationFormatError, match="not found"):
            load_yaml_config("/nonexistent/file.yaml")

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        cfg = load_yaml_config(str(f))
        assert cfg == {}

    def test_no_validation(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump({"a": 1}))
        cfg = load_yaml_config(str(f), validate_schema=False)
        assert cfg["a"] == 1

    def test_no_env_substitution(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text("key: $HOME")
        cfg = load_yaml_config(str(f), substitute_env=False, validate_schema=False)
        assert cfg["key"] == "$HOME"

    def test_invalid_yaml(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("{{invalid yaml::::")
        with pytest.raises(ConfigurationFormatError):
            load_yaml_config(str(f), validate_schema=False)

    def test_schema_validation_failure(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump({"biological": {"species_configs": {"sp1": {"max_growth_rate": "not_a_number"}}}}))
        with pytest.raises(ConfigurationFormatError):
            load_yaml_config(str(f), validate_schema=True)


class TestLoadJsonConfig:
    def test_load_valid(self, tmp_path):
        f = tmp_path / "cfg.json"
        f.write_text(json.dumps({"biological": {"key": "val"}}))
        cfg = load_json_config(str(f))
        assert cfg["biological"]["key"] == "val"

    def test_not_found(self):
        with pytest.raises(ConfigurationFormatError, match="not found"):
            load_json_config("/nonexistent/file.json")

    def test_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{invalid json")
        with pytest.raises(ConfigurationFormatError):
            load_json_config(str(f), validate_schema=False)

    def test_no_validation(self, tmp_path):
        f = tmp_path / "cfg.json"
        f.write_text(json.dumps({"a": 1}))
        cfg = load_json_config(str(f), validate_schema=False)
        assert cfg["a"] == 1

    def test_schema_validation_failure(self, tmp_path):
        f = tmp_path / "cfg.json"
        f.write_text(json.dumps({"biological": {"species_configs": {"sp1": {"max_growth_rate": "bad"}}}}))
        with pytest.raises(ConfigurationFormatError):
            load_json_config(str(f), validate_schema=True)


class TestSubstituteEnvironmentVariables:
    def test_default_value(self):
        result = substitute_environment_variables("${NONEXISTENT_VAR_12345:default_val}")
        assert result == "default_val"

    def test_env_var_set(self, monkeypatch):
        monkeypatch.setenv("TEST_CFG_VAR_XYZ", "hello")
        result = substitute_environment_variables("${TEST_CFG_VAR_XYZ}")
        assert result == "hello"

    def test_simple_dollar(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR_ABC", "world")
        result = substitute_environment_variables("$MY_TEST_VAR_ABC")
        assert result == "world"

    def test_no_env_var(self):
        result = substitute_environment_variables("$NONEXISTENT_VAR_99999")
        assert result == "$NONEXISTENT_VAR_99999"

    def test_with_default_env_exists(self, monkeypatch):
        monkeypatch.setenv("EXISTS_VAR_99", "exists")
        result = substitute_environment_variables("${EXISTS_VAR_99:fallback}")
        assert result == "exists"


class TestGetConfigSchema:
    def test_returns_dict(self):
        schema = get_config_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "biological" in schema["properties"]


class TestValidateConfigSchema:
    def test_valid_config(self):
        result = validate_config_schema({"biological": {"key": "val"}})
        assert result.is_valid is True

    def test_invalid_config(self):
        result = validate_config_schema({"biological": {"species_configs": {"sp1": {"max_growth_rate": "not_num"}}}})
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_custom_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        result = validate_config_schema({"x": 5}, schema=schema)
        assert result.is_valid is True

    def test_custom_schema_failure(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "additionalProperties": False}
        result = validate_config_schema({"x": "bad", "y": 1}, schema=schema)
        assert result.is_valid is False

    def test_general_exception(self):
        with patch("config.config_utils.validate", side_effect=RuntimeError("fail")):
            result = validate_config_schema({})
        assert result.is_valid is False
        assert any("fail" in e for e in result.errors)

    def test_error_path(self):
        schema = {
            "type": "object",
            "properties": {"nested": {"type": "object", "properties": {"x": {"type": "integer"}}, "additionalProperties": False}},
        }
        result = validate_config_schema({"nested": {"x": "bad"}}, schema=schema)
        assert result.is_valid is False


class TestDataclassToDict:
    def test_simple_dataclass(self):
        @dataclass
        class Simple:
            a: int = 1
            b: str = "hello"

        obj = Simple()
        d = dataclass_to_dict(obj)
        assert d == {"a": 1, "b": "hello"}

    def test_nested_dataclass(self):
        @dataclass
        class Inner:
            x: int = 10

        @dataclass
        class Outer:
            inner: Inner = None

            def __post_init__(self):
                if self.inner is None:
                    self.inner = Inner()

        obj = Outer()
        d = dataclass_to_dict(obj)
        assert d["inner"] == {"x": 10}

    def test_non_dataclass(self):
        assert dataclass_to_dict(42) == 42

    def test_list_tuple(self):
        @dataclass
        class WithList:
            items: list = None

            def __post_init__(self):
                if self.items is None:
                    self.items = [1, 2, 3]

        d = dataclass_to_dict(WithList())
        assert d["items"] == [1, 2, 3]

    def test_dict_field(self):
        @dataclass
        class WithDict:
            data: dict = None

            def __post_init__(self):
                if self.data is None:
                    self.data = {"a": 1}

        d = dataclass_to_dict(WithDict())
        assert d["data"]["a"] == 1

    def test_tuple_values(self):
        @dataclass
        class WithTuple:
            tup: tuple = (1, 2, 3)

        d = dataclass_to_dict(WithTuple())
        assert d["tup"] == [1, 2, 3]


class TestDictToDataclass:
    def test_simple(self):
        @dataclass
        class Simple:
            a: int = 0
            b: str = ""

        obj = dict_to_dataclass({"a": 5, "b": "hi"}, Simple)
        assert obj.a == 5
        assert obj.b == "hi"

    def test_not_dataclass(self):
        with pytest.raises(ValueError, match="not a dataclass"):
            dict_to_dataclass({}, int)

    def test_missing_fields(self):
        @dataclass
        class WithDefaults:
            a: int = 0
            b: str = "default"

        obj = dict_to_dataclass({"a": 3}, WithDefaults)
        assert obj.a == 3
        assert obj.b == "default"


class TestDeepMergeDicts:
    def test_basic(self):
        result = deep_merge_dicts({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_nested(self):
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = deep_merge_dicts(base, override)
        assert result["a"] == {"x": 1, "y": 3, "z": 4}

    def test_override(self):
        result = deep_merge_dicts({"a": 1}, {"a": 2})
        assert result["a"] == 2


class TestMergeConfigFiles:
    def test_merge_yaml(self, tmp_path):
        f1 = tmp_path / "a.yaml"
        f2 = tmp_path / "b.yaml"
        f1.write_text(yaml.dump({"a": 1}))
        f2.write_text(yaml.dump({"b": 2}))
        result = merge_config_files(str(f1), str(f2))
        assert result["a"] == 1
        assert result["b"] == 2

    def test_merge_json(self, tmp_path):
        f1 = tmp_path / "a.json"
        f1.write_text(json.dumps({"c": 3}))
        result = merge_config_files(str(f1))
        assert result["c"] == 3

    def test_unsupported_format(self, tmp_path):
        f = tmp_path / "a.xml"
        f.write_text("<xml/>")
        with pytest.raises(ConfigurationFormatError):
            merge_config_files(str(f))

    def test_merge_with_output(self, tmp_path):
        f1 = tmp_path / "a.yaml"
        f1.write_text(yaml.dump({"x": 1}))
        out = tmp_path / "out.yaml"
        result = merge_config_files(str(f1), output_path=str(out))
        assert out.exists()
        assert result["x"] == 1


class TestSaveConfigFile:
    def test_save_yaml(self, tmp_path):
        out = tmp_path / "out.yaml"
        save_config_file({"a": 1}, str(out), format="yaml")
        assert out.exists()

    def test_save_json(self, tmp_path):
        out = tmp_path / "out.json"
        save_config_file({"a": 1}, str(out), format="json")
        assert out.exists()

    def test_unsupported_format(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported"):
            save_config_file({}, str(tmp_path / "out.xml"), format="xml")


class TestConfigLoader:
    def test_load_yaml_cached(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump({"a": 1}))
        loader = ConfigLoader(validate_schema=False)
        cfg1 = loader.load_config(str(f))
        cfg2 = loader.load_config(str(f))
        assert cfg1 == cfg2

    def test_load_json(self, tmp_path):
        f = tmp_path / "cfg.json"
        f.write_text(json.dumps({"a": 1}))
        loader = ConfigLoader(validate_schema=False)
        cfg = loader.load_config(str(f))
        assert cfg["a"] == 1

    def test_force_reload(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump({"a": 1}))
        loader = ConfigLoader(validate_schema=False)
        loader.load_config(str(f))
        cfg = loader.load_config(str(f), force_reload=True)
        assert cfg["a"] == 1

    def test_unsupported_format(self, tmp_path):
        f = tmp_path / "cfg.xml"
        f.write_text("<xml/>")
        loader = ConfigLoader()
        with pytest.raises(ConfigurationFormatError):
            loader.load_config(str(f))

    def test_clear_cache(self):
        loader = ConfigLoader()
        loader.cache["key"] = ({}, 0.0)
        loader.clear_cache()
        assert len(loader.cache) == 0

    def test_cache_disabled(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump({"a": 1}))
        loader = ConfigLoader(cache_enabled=False, validate_schema=False)
        cfg = loader.load_config(str(f))
        assert cfg["a"] == 1
        assert len(loader.cache) == 0

    def test_yml_extension(self, tmp_path):
        f = tmp_path / "cfg.yml"
        f.write_text(yaml.dump({"a": 1}))
        loader = ConfigLoader(validate_schema=False)
        cfg = loader.load_config(str(f))
        assert cfg["a"] == 1


class TestConfigConverter:
    def test_convert_legacy_basic(self):
        converter = ConfigConverter()
        legacy = {"learning_rate": 0.1, "epsilon": 0.3}
        result = converter.convert_legacy_qlearning_config(legacy)
        assert result["control"]["advanced_control"]["learning_rate"] == 0.1
        assert result["control"]["advanced_control"]["epsilon"] == 0.3

    def test_convert_legacy_with_rewards(self):
        converter = ConfigConverter()
        legacy = {"rewards": {"power_weight": 10.0, "biofilm_weight": 5.0}}
        result = converter.convert_legacy_qlearning_config(legacy)
        assert result["control"]["advanced_control"]["power_reward_weight"] == 10.0

    def test_convert_legacy_flow_rates(self):
        converter = ConfigConverter()
        legacy = {"flow_rate_min": 0.001, "flow_rate_max": 0.01}
        result = converter.convert_legacy_qlearning_config(legacy)
        assert "flow_control" in result["control"]

    def test_migrate_same_version(self):
        converter = ConfigConverter()
        config = {"metadata": {"version": "2.0.0"}}
        result = converter.migrate_config_version(config, "2.0.0")
        assert result is config

    def test_migrate_v1_to_v2(self):
        converter = ConfigConverter()
        config = {"metadata": {"version": "1.0.0"}, "data": 1}
        result = converter.migrate_config_version(config, "2.0.0")
        assert result["metadata"]["version"] == "2.0.0"

    def test_migrate_no_path(self):
        converter = ConfigConverter()
        config = {"metadata": {"version": "3.0.0"}}
        with pytest.warns(UserWarning, match="No migration path"):
            result = converter.migrate_config_version(config, "4.0.0")
        assert result is config

    def test_migrate_no_metadata(self):
        converter = ConfigConverter()
        config = {}
        result = converter.migrate_config_version(config, "2.0.0")
        assert result["metadata"]["version"] == "2.0.0"

    def test_set_nested_value_creates_keys(self):
        converter = ConfigConverter()
        d = {}
        converter._set_nested_value(d, ["a", "b", "c"], 42)
        assert d["a"]["b"]["c"] == 42


class TestGetSetConfigValue:
    def test_get_simple(self):
        cfg = {"a": {"b": {"c": 5}}}
        assert get_config_value(cfg, "a.b.c") == 5

    def test_get_default(self):
        assert get_config_value({}, "a.b.c", default="missing") == "missing"

    def test_get_type_error(self):
        cfg = {"a": 5}
        assert get_config_value(cfg, "a.b.c", default=None) is None

    def test_set_simple(self):
        cfg = {}
        set_config_value(cfg, "a.b.c", 10)
        assert cfg["a"]["b"]["c"] == 10

    def test_set_existing(self):
        cfg = {"a": {"b": 1}}
        set_config_value(cfg, "a.b", 2)
        assert cfg["a"]["b"] == 2


class TestValidateConfigTypes:
    def test_valid(self):
        cfg = {"control": {"flow_control": {"min_flow_rate": 5.0}}}
        errors = validate_config_types(cfg)
        assert len(errors) == 0

    def test_invalid_type(self):
        cfg = {"control": {"flow_control": {"min_flow_rate": "bad"}}}
        errors = validate_config_types(cfg)
        assert len(errors) > 0

    def test_missing_path(self):
        cfg = {}
        errors = validate_config_types(cfg)
        assert len(errors) == 0
