"""Coverage boost tests for config_utils.py."""
import json
import os
import sys
import warnings
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.config_utils import (
    ConfigConverter,
    ConfigLoader,
    ConfigurationFormatError,
    dataclass_to_dict,
    dict_to_dataclass,
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


class TestLoadYaml:
    def test_nonexistent(self, tmp_path):
        with pytest.raises(ConfigurationFormatError):
            load_yaml_config(tmp_path / "nope.yaml")

    def test_empty(self, tmp_path):
        f = tmp_path / "e.yaml"
        f.write_text("")
        r = load_yaml_config(str(f), validate_schema=False)
        assert r == {}

    def test_env_sub(self, tmp_path):
        os.environ["TEST_U_V"] = "sv"
        f = tmp_path / "e.yaml"
        f.write_text("k: ${TEST_U_V}")
        r = load_yaml_config(str(f), validate_schema=False)
        assert r["k"] == "sv"
        del os.environ["TEST_U_V"]

    def test_invalid(self, tmp_path):
        f = tmp_path / "b.yaml"
        f.write_text("{{bad:: [")
        with pytest.raises(ConfigurationFormatError):
            load_yaml_config(str(f), validate_schema=False)


class TestLoadJson:
    def test_nonexistent(self, tmp_path):
        with pytest.raises(ConfigurationFormatError):
            load_json_config(tmp_path / "n.json")

    def test_invalid(self, tmp_path):
        f = tmp_path / "b.json"
        f.write_text("{bad")
        with pytest.raises(ConfigurationFormatError):
            load_json_config(str(f), validate_schema=False)

    def test_valid(self, tmp_path):
        f = tmp_path / "o.json"
        f.write_text(json.dumps({"k": "v"}))
        r = load_json_config(str(f), validate_schema=False)
        assert r["k"] == "v"


class TestEnvSub:
    def test_default(self):
        r = substitute_environment_variables("${NX:def}")
        assert r == "def"

    def test_simple(self):
        os.environ["TSU"] = "hi"
        r = substitute_environment_variables("$TSU")
        assert r == "hi"
        del os.environ["TSU"]

    def test_unset(self):
        r = substitute_environment_variables("$NX_S")
        assert "$NX_S" in r


class TestSchema:
    def test_valid(self):
        r = validate_config_schema({"biological": {}})
        assert r.is_valid is True

    def test_invalid(self):
        s = {"type": "object", "required": ["x"]}
        r = validate_config_schema({"y": 1}, s)
        assert r.is_valid is False


class TestDC:
    def test_to_dict(self):
        @dataclass
        class I:
            x: int = 1
        @dataclass
        class O:
            i: I = None
            l: list = None
        o = O(i=I(x=5), l=[I(x=10)])
        r = dataclass_to_dict(o)
        assert r["i"]["x"] == 5

    def test_from_dict(self):
        @dataclass
        class S:
            n: str = ""
            v: int = 0
        r = dict_to_dataclass({"n": "t", "v": 42}, S)
        assert r.n == "t"

    def test_not_dc(self):
        with pytest.raises(ValueError):
            dict_to_dataclass({"a": 1}, dict)

    def test_dict_field(self):
        @dataclass
        class D:
            d: dict = None
        r = dataclass_to_dict(D(d={"a": {"b": 1}}))
        assert r["d"]["a"]["b"] == 1


class TestMerge:
    def test_yaml(self, tmp_path):
        f1 = tmp_path / "a.yaml"
        f1.write_text(yaml.dump({"x": 1, "n": {"a": 1}}))
        f2 = tmp_path / "b.yaml"
        f2.write_text(yaml.dump({"y": 2, "n": {"b": 2}}))
        r = merge_config_files(str(f1), str(f2))
        assert r["n"]["a"] == 1

    def test_output(self, tmp_path):
        f = tmp_path / "a.yaml"
        f.write_text(yaml.dump({"x": 1}))
        o = tmp_path / "o.yaml"
        merge_config_files(str(f), output_path=str(o))
        assert o.exists()

    def test_unsupported(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("d")
        with pytest.raises(ConfigurationFormatError):
            merge_config_files(str(f))


class TestSave:
    def test_json(self, tmp_path):
        o = tmp_path / "o.json"
        save_config_file({"k": "v"}, str(o), format="json")
        assert o.exists()

    def test_unsupported(self, tmp_path):
        with pytest.raises(ValueError):
            save_config_file({}, str(tmp_path / "o.xml"), format="xml")


class TestLoader:
    def test_cache(self, tmp_path):
        f = tmp_path / "t.yaml"
        f.write_text(yaml.dump({"k": "v"}))
        l = ConfigLoader(cache_enabled=True, validate_schema=False)
        r1 = l.load_config(str(f))
        r2 = l.load_config(str(f))
        assert r1 == r2

    def test_force(self, tmp_path):
        f = tmp_path / "t.yaml"
        f.write_text(yaml.dump({"k": "v"}))
        l = ConfigLoader(cache_enabled=True, validate_schema=False)
        l.load_config(str(f))
        r = l.load_config(str(f), force_reload=True)
        assert r["k"] == "v"

    def test_unsup(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("d")
        l = ConfigLoader(validate_schema=False)
        with pytest.raises(ConfigurationFormatError):
            l.load_config(str(f))

    def test_clear(self):
        l = ConfigLoader()
        l.cache["k"] = ({"a": 1}, 0)
        l.clear_cache()
        assert len(l.cache) == 0

    def test_json(self, tmp_path):
        f = tmp_path / "t.json"
        f.write_text(json.dumps({"k": "v"}))
        l = ConfigLoader(validate_schema=False)
        r = l.load_config(str(f))
        assert r["k"] == "v"


class TestConverter:
    def test_legacy(self):
        c = ConfigConverter()
        lg = {"learning_rate": 0.1, "rewards": {"power_weight": 0.5}}
        r = c.convert_legacy_qlearning_config(lg)
        assert r["control"]["advanced_control"]["learning_rate"] == 0.1

    def test_same_ver(self):
        c = ConfigConverter()
        cfg = {"metadata": {"version": "2.0.0"}}
        assert c.migrate_config_version(cfg, "2.0.0") is cfg

    def test_v1_v2(self):
        c = ConfigConverter()
        cfg = {"metadata": {"version": "1.0.0"}}
        r = c.migrate_config_version(cfg, "2.0.0")
        assert r["metadata"]["version"] == "2.0.0"

    def test_unknown(self):
        c = ConfigConverter()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c.migrate_config_version({"metadata": {"version": "0.5.0"}}, "2.0.0")
            assert len(w) == 1


class TestUtils:
    def test_get_nested(self):
        assert get_config_value({"a": {"b": {"c": 42}}}, "a.b.c") == 42

    def test_get_missing(self):
        assert get_config_value({}, "a.b.c", default="d") == "d"

    def test_set(self):
        cfg = {}
        set_config_value(cfg, "a.b.c", 42)
        assert cfg["a"]["b"]["c"] == 42

    def test_types_valid(self):
        cfg = {"control": {"flow_control": {"min_flow_rate": 1.0}}}
        assert len(validate_config_types(cfg)) == 0

    def test_types_invalid(self):
        cfg = {"control": {"flow_control": {"min_flow_rate": "bad"}}}
        assert len(validate_config_types(cfg)) > 0
