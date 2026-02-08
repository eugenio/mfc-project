"""Tests for config/config_io.py - coverage target 98%+."""
import sys
import os
import json
import tempfile
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.config_io import (
    convert_values_for_serialization,
    dataclass_to_dict,
    convert_lists_to_tuples_for_dataclass,
    dict_to_dataclass,
    save_config,
    load_config,
    merge_configs,
    _enum_dict_factory,
)
from config.qlearning_config import (
    QLearningConfig,
    QLearningRewardWeights,
    StateSpaceConfig,
)
from config.sensor_config import (
    SensorConfig,
    EISConfig,
    QCMConfig,
    SensorFusionConfig,
    FusionMethod,
)


class TestConvertValuesForSerialization:
    def test_enum(self):
        class Color(Enum):
            RED = "red"
        assert convert_values_for_serialization(Color.RED) == "red"

    def test_tuple(self):
        result = convert_values_for_serialization((1, 2, 3))
        assert result == [1, 2, 3]

    def test_dict(self):
        class C(Enum):
            A = "a"
        result = convert_values_for_serialization({"key": C.A})
        assert result == {"key": "a"}

    def test_list(self):
        class C(Enum):
            A = "a"
        result = convert_values_for_serialization([C.A, 1])
        assert result == ["a", 1]

    def test_plain_value(self):
        assert convert_values_for_serialization(42) == 42


class TestEnumDictFactory:
    def test_basic(self):
        class C(Enum):
            X = "x"
        result = _enum_dict_factory([("a", C.X), ("b", 5)])
        assert result == {"a": "x", "b": 5}


class TestDataclassToDict:
    def test_non_dataclass(self):
        assert dataclass_to_dict("hello") == "hello"

    def test_dataclass(self):
        cfg = QLearningRewardWeights()
        result = dataclass_to_dict(cfg)
        assert isinstance(result, dict)


class TestConvertListsToTuples:
    def test_state_space_config(self):
        data = {"power_range": [0.0, 1.0], "other": "val"}
        result = convert_lists_to_tuples_for_dataclass(data, StateSpaceConfig)
        assert isinstance(result["power_range"], tuple)

    def test_no_matching_fields(self):
        data = {"some_field": [1, 2]}
        result = convert_lists_to_tuples_for_dataclass(data, QLearningConfig)
        assert isinstance(result["some_field"], list)


class TestDictToDataclass:
    def test_qlearning_config(self):
        cfg = QLearningConfig()
        d = dataclass_to_dict(cfg)
        restored = dict_to_dataclass(d, QLearningConfig)
        assert isinstance(restored, QLearningConfig)

    def test_sensor_config(self):
        cfg = SensorConfig()
        d = dataclass_to_dict(cfg)
        restored = dict_to_dataclass(d, SensorConfig)
        assert isinstance(restored, SensorConfig)

    def test_sensor_config_fusion_method_str(self):
        cfg = SensorConfig()
        d = dataclass_to_dict(cfg)
        d["fusion_method"] = "kalman_filter"
        restored = dict_to_dataclass(d, SensorConfig)
        assert isinstance(restored, SensorConfig)


class TestSaveConfig:
    def test_save_yaml(self):
        cfg = QLearningConfig()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            save_config(cfg, path, format="yaml")
            assert Path(path).exists()
        finally:
            os.unlink(path)

    def test_save_json(self):
        cfg = QLearningConfig()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_config(cfg, path, format="json")
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
        finally:
            os.unlink(path)

    def test_save_sensor_config(self):
        cfg = SensorConfig()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            save_config(cfg, path, format="yaml")
            assert Path(path).exists()
        finally:
            os.unlink(path)

    def test_save_unsupported_format(self):
        cfg = QLearningConfig()
        with pytest.raises(ValueError, match="Unsupported format"):
            save_config(cfg, "/tmp/test.txt", format="xml")

    def test_save_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported configuration"):
            save_config("not_a_config", "/tmp/test.yaml", format="yaml")


class TestLoadConfig:
    def test_load_yaml(self):
        cfg = QLearningConfig()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            save_config(cfg, path, format="yaml")
            loaded = load_config(path, QLearningConfig)
            assert isinstance(loaded, QLearningConfig)
        finally:
            os.unlink(path)

    def test_load_json(self):
        cfg = QLearningConfig()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_config(cfg, path, format="json")
            loaded = load_config(path, QLearningConfig)
            assert isinstance(loaded, QLearningConfig)
        finally:
            os.unlink(path)

    def test_load_sensor_config(self):
        cfg = SensorConfig()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            save_config(cfg, path, format="yaml")
            loaded = load_config(path, SensorConfig)
            assert isinstance(loaded, SensorConfig)
        finally:
            os.unlink(path)

    def test_load_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/tmp/nonexistent_xyz.yaml", QLearningConfig)

    def test_load_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            f.write(b"<config/>")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                load_config(path, QLearningConfig)
        finally:
            os.unlink(path)


class TestMergeConfigs:
    def test_merge_qlearning(self):
        cfg = QLearningConfig()
        merged = merge_configs(cfg, {"learning_rate": 0.5})
        assert isinstance(merged, QLearningConfig)

    def test_merge_sensor(self):
        cfg = SensorConfig()
        merged = merge_configs(cfg, {"noise_threshold": 0.1})
        assert isinstance(merged, SensorConfig)

    def test_deep_merge(self):
        cfg = QLearningConfig()
        merged = merge_configs(
            cfg, {"reward_weights": {"power_weight": 0.9}}
        )
        assert isinstance(merged, QLearningConfig)
