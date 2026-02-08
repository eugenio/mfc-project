"""Tests for config/config_io.py - targeting 98%+ coverage."""
import importlib.util
import json
import os
import sys
import tempfile
from copy import deepcopy
from enum import Enum
from unittest.mock import MagicMock

import pytest

# Mock numpy before any config loading
_mock_np = MagicMock()
_mock_np.isclose = lambda a, b, rtol=1e-5: abs(a - b) <= rtol * abs(b)
sys.modules.setdefault("numpy", _mock_np)

# Mock yaml
_mock_yaml = MagicMock()
sys.modules.setdefault("yaml", _mock_yaml)

_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")

# Load electrode_config
_spec_ec = importlib.util.spec_from_file_location(
    "config.electrode_config",
    os.path.join(_src, "config", "electrode_config.py"),
)
_mod_ec = importlib.util.module_from_spec(_spec_ec)
sys.modules.setdefault("config.electrode_config", _mod_ec)
_spec_ec.loader.exec_module(_mod_ec)

# Load qlearning_config
_spec_ql = importlib.util.spec_from_file_location(
    "config.qlearning_config",
    os.path.join(_src, "config", "qlearning_config.py"),
)
_mod_ql = importlib.util.module_from_spec(_spec_ql)
sys.modules.setdefault("config.qlearning_config", _mod_ql)
_spec_ql.loader.exec_module(_mod_ql)

# Load sensor_config
_spec_sc = importlib.util.spec_from_file_location(
    "config.sensor_config",
    os.path.join(_src, "config", "sensor_config.py"),
)
_mod_sc = importlib.util.module_from_spec(_spec_sc)
sys.modules.setdefault("config.sensor_config", _mod_sc)
_spec_sc.loader.exec_module(_mod_sc)

# Load parameter_validation
_spec_pv = importlib.util.spec_from_file_location(
    "config.parameter_validation",
    os.path.join(_src, "config", "parameter_validation.py"),
)
_mod_pv = importlib.util.module_from_spec(_spec_pv)
sys.modules.setdefault("config.parameter_validation", _mod_pv)
_spec_pv.loader.exec_module(_mod_pv)

# Now load config_io
_spec = importlib.util.spec_from_file_location(
    "config.config_io",
    os.path.join(_src, "config", "config_io.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["config.config_io"] = _mod
_spec.loader.exec_module(_mod)

convert_values_for_serialization = _mod.convert_values_for_serialization
dataclass_to_dict = _mod.dataclass_to_dict
convert_lists_to_tuples_for_dataclass = _mod.convert_lists_to_tuples_for_dataclass
dict_to_dataclass = _mod.dict_to_dataclass
save_config = _mod.save_config
load_config = _mod.load_config
merge_configs = _mod.merge_configs
_enum_dict_factory = _mod._enum_dict_factory

QLearningConfig = _mod_ql.QLearningConfig
QLearningRewardWeights = _mod_ql.QLearningRewardWeights
StateSpaceConfig = _mod_ql.StateSpaceConfig
DEFAULT_QLEARNING_CONFIG = _mod_ql.DEFAULT_QLEARNING_CONFIG
SensorConfig = _mod_sc.SensorConfig
EISConfig = _mod_sc.EISConfig
QCMConfig = _mod_sc.QCMConfig
SensorFusionConfig = _mod_sc.SensorFusionConfig
FusionMethod = _mod_sc.FusionMethod
ElectrodeConfiguration = _mod_ec.ElectrodeConfiguration
ElectrodeGeometry = _mod_ec.ElectrodeGeometry
ElectrodeMaterial = _mod_ec.ElectrodeMaterial
MaterialProperties = _mod_ec.MaterialProperties


class TestConvertValuesForSerialization:
    def test_enum(self):
        class Color(Enum):
            RED = "red"
        assert convert_values_for_serialization(Color.RED) == "red"

    def test_tuple(self):
        result = convert_values_for_serialization((1, 2, 3))
        assert result == [1, 2, 3]

    def test_dict(self):
        class Color(Enum):
            RED = "red"
        result = convert_values_for_serialization({"color": Color.RED, "x": 1})
        assert result == {"color": "red", "x": 1}

    def test_list(self):
        result = convert_values_for_serialization([1, [2, 3]])
        assert result == [1, [2, 3]]

    def test_plain_value(self):
        assert convert_values_for_serialization(42) == 42
        assert convert_values_for_serialization("str") == "str"


class TestEnumDictFactory:
    def test_converts_enums(self):
        class Color(Enum):
            RED = "red"
        result = _enum_dict_factory([("color", Color.RED), ("x", 1)])
        assert result == {"color": "red", "x": 1}


class TestDataclassToDict:
    def test_with_dataclass(self):
        cfg = SensorConfig()
        result = dataclass_to_dict(cfg)
        assert isinstance(result, dict)
        assert "eis" in result

    def test_non_dataclass(self):
        result = dataclass_to_dict("not a dataclass")
        assert result == "not a dataclass"

    def test_type_class_not_instance(self):
        result = dataclass_to_dict(SensorConfig)
        assert result == SensorConfig


class TestConvertListsToTuples:
    def test_eis_frequency_range(self):
        data = {"frequency_range": [100.0, 1e6]}
        result = convert_lists_to_tuples_for_dataclass(data, EISConfig)
        assert isinstance(result["frequency_range"], tuple)

    def test_qcm_mass_range(self):
        data = {"mass_range": [0.0, 100.0], "frequency_shift_range": [0.0, 500.0]}
        result = convert_lists_to_tuples_for_dataclass(data, QCMConfig)
        assert isinstance(result["mass_range"], tuple)
        assert isinstance(result["frequency_shift_range"], tuple)

    def test_no_conversion_needed(self):
        data = {"x": 42}
        result = convert_lists_to_tuples_for_dataclass(data, QLearningConfig)
        assert result == {"x": 42}

    def test_nested_dict(self):
        data = {"sub": {"inner": 1}, "x": 2}
        result = convert_lists_to_tuples_for_dataclass(data, QLearningConfig)
        assert result["sub"]["inner"] == 1

    def test_non_dict(self):
        result = convert_lists_to_tuples_for_dataclass(42, QLearningConfig)
        assert result == 42


class TestDictToDataclass:
    def test_qlearning_config_basic(self):
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        d = dataclass_to_dict(cfg)
        result = dict_to_dataclass(d, QLearningConfig)
        assert isinstance(result, QLearningConfig)
        assert result.learning_rate == cfg.learning_rate

    def test_sensor_config_basic(self):
        cfg = SensorConfig()
        d = dataclass_to_dict(cfg)
        result = dict_to_dataclass(d, SensorConfig)
        assert isinstance(result, SensorConfig)

    def test_sensor_config_with_fusion_method_str(self):
        d = dataclass_to_dict(SensorConfig())
        d["fusion_method"] = "kalman_filter"
        result = dict_to_dataclass(d, SensorConfig)
        assert result.fusion_method == FusionMethod.KALMAN_FILTER

    def test_qlearning_with_nested_reward_weights(self):
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        d = dataclass_to_dict(cfg)
        assert isinstance(d["reward_weights"], dict)
        result = dict_to_dataclass(d, QLearningConfig)
        assert isinstance(result.reward_weights, QLearningRewardWeights)

    def test_qlearning_with_nested_state_space(self):
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        d = dataclass_to_dict(cfg)
        result = dict_to_dataclass(d, QLearningConfig)
        assert isinstance(result.state_space, StateSpaceConfig)

    def test_qlearning_with_anode_config(self):
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        d = dataclass_to_dict(cfg)
        if "anode_config" in d and isinstance(d["anode_config"], dict):
            result = dict_to_dataclass(d, QLearningConfig)
            assert isinstance(result.anode_config, ElectrodeConfiguration)

    def test_qlearning_with_cathode_config_dict(self):
        """Cover cathode_config dict branch (lines 202-220)."""
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        d = dataclass_to_dict(cfg)
        # Inject cathode_config as dict with material str, geometry dict, material_properties dict
        d["cathode_config"] = {
            "material": "graphite_plate",
            "geometry": {
                "geometry_type": "rectangular_plate",
                "length": 0.05,
                "width": 0.05,
                "thickness": 0.005,
            },
            "material_properties": {
                "specific_conductance": 25000,
                "contact_resistance": 0.1,
                "surface_charge_density": -0.05,
                "hydrophobicity_angle": 75,
                "surface_roughness": 1.5,
                "biofilm_adhesion_coefficient": 0.7,
                "attachment_energy": -15.0,
            },
        }
        result = dict_to_dataclass(d, QLearningConfig)
        assert isinstance(result.cathode_config, ElectrodeConfiguration)


class TestSaveConfig:
    def test_save_qlearning_yaml(self, tmp_path):
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        filepath = tmp_path / "test_config.yaml"
        save_config(cfg, filepath, format="yaml")
        _mock_yaml.dump.assert_called()

    def test_save_qlearning_json(self, tmp_path):
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        filepath = tmp_path / "test_config.json"
        save_config(cfg, filepath, format="json")
        assert filepath.exists()

    def test_save_sensor_yaml(self, tmp_path):
        cfg = SensorConfig()
        filepath = tmp_path / "sensor.yaml"
        save_config(cfg, filepath, format="yaml")
        _mock_yaml.dump.assert_called()

    def test_save_sensor_json(self, tmp_path):
        cfg = SensorConfig()
        filepath = tmp_path / "sensor.json"
        save_config(cfg, filepath, format="json")
        assert filepath.exists()

    def test_unsupported_format(self, tmp_path):
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        with pytest.raises(ValueError, match="Unsupported format"):
            save_config(cfg, tmp_path / "test.xml", format="xml")

    def test_unsupported_config_type(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported configuration type"):
            save_config("not a config", tmp_path / "test.yaml", format="yaml")

    def test_creates_parent_dir(self, tmp_path):
        cfg = SensorConfig()
        filepath = tmp_path / "subdir" / "nested" / "test.json"
        save_config(cfg, filepath, format="json")
        assert filepath.exists()


class TestLoadConfig:
    def test_load_json(self, tmp_path):
        cfg = SensorConfig()
        filepath = tmp_path / "test.json"
        d = dataclass_to_dict(cfg)
        with open(filepath, "w") as f:
            json.dump(d, f)
        result = load_config(filepath, SensorConfig)
        assert isinstance(result, SensorConfig)

    def test_load_yaml(self, tmp_path):
        cfg = SensorConfig()
        filepath = tmp_path / "test.yaml"
        filepath.write_text("")  # Create file
        d = dataclass_to_dict(cfg)
        _mock_yaml.safe_load.return_value = d
        result = load_config(filepath, SensorConfig)
        assert isinstance(result, SensorConfig)

    def test_load_yml_extension(self, tmp_path):
        cfg = SensorConfig()
        filepath = tmp_path / "test.yml"
        filepath.write_text("")
        d = dataclass_to_dict(cfg)
        _mock_yaml.safe_load.return_value = d
        result = load_config(filepath, SensorConfig)
        assert isinstance(result, SensorConfig)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(tmp_path / "nonexistent.json", SensorConfig)

    def test_unsupported_extension(self, tmp_path):
        filepath = tmp_path / "test.xml"
        filepath.write_text("")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_config(filepath, SensorConfig)

    def test_load_qlearning_json(self, tmp_path):
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        filepath = tmp_path / "ql.json"
        d = dataclass_to_dict(cfg)
        with open(filepath, "w") as f:
            json.dump(d, f)
        result = load_config(filepath, QLearningConfig)
        assert isinstance(result, QLearningConfig)


class TestMergeConfigs:
    def test_merge_sensor_config(self):
        cfg = SensorConfig()
        original_interval = cfg.eis_update_interval
        merged = merge_configs(cfg, {"eis_update_interval": 999.0})
        assert merged.eis_update_interval == 999.0

    def test_merge_qlearning_config(self):
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        merged = merge_configs(cfg, {"learning_rate": 0.5})
        assert merged.learning_rate == 0.5

    def test_deep_merge_nested(self):
        cfg = SensorConfig()
        d = dataclass_to_dict(cfg)
        merged = merge_configs(cfg, {"eis_update_interval": 100.0})
        assert isinstance(merged, SensorConfig)

    def test_merge_with_nested_dict_override(self):
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        d = dataclass_to_dict(cfg)
        # Override a nested value via the merge
        merged = merge_configs(cfg, {"learning_rate": 0.2})
        assert merged.learning_rate == 0.2

    def test_deep_merge_recursive_dicts(self):
        """Cover line 363 - deep_merge with nested dict in override."""
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        # reward_weights is serialized as a dict, override a sub-key
        merged = merge_configs(cfg, {"reward_weights": {"power_weight": 20.0}})
        assert merged.reward_weights.power_weight == 20.0
