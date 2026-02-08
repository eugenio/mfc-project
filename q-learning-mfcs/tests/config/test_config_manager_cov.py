"""Tests for config_manager module - 98%+ coverage target."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.config_manager import (
    BiologicalConfigValidator,
    ConfigManager,
    ConfigMerger,
    ConfigProfile,
    ConfigurationError,
    ConfigurationLoadError,
    ConfigurationValidationError,
    ControlConfigValidator,
    VisualizationConfigValidator,
    get_config_manager,
    reset_config_manager,
)


class TestExceptions:
    def test_configuration_error(self):
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("test")

    def test_validation_error(self):
        with pytest.raises(ConfigurationValidationError):
            raise ConfigurationValidationError("test")

    def test_load_error(self):
        with pytest.raises(ConfigurationLoadError):
            raise ConfigurationLoadError("test")


class TestConfigProfile:
    def test_defaults(self):
        p = ConfigProfile()
        assert p.profile_name == "default"
        assert p.profile_version == "1.0.0"
        assert p.environment == "development"
        assert p.biological is None
        assert p.control is None
        assert p.visualization is None

    def test_update_timestamp(self):
        p = ConfigProfile()
        old = p.updated_at
        p.update_timestamp()
        assert p.updated_at >= old


class TestBiologicalConfigValidator:
    def test_valid_config(self):
        v = BiologicalConfigValidator()
        assert v.validate({"species_configs": {"sp1": {"max_growth_rate": 0.5, "electron_transport_efficiency": 0.8}}}) is True

    def test_empty_dict(self):
        v = BiologicalConfigValidator()
        assert v.validate({}) is True

    def test_not_dict(self):
        v = BiologicalConfigValidator()
        with pytest.raises(ConfigurationValidationError, match="must be a dictionary"):
            v.validate("not a dict")

    def test_species_not_dict(self):
        v = BiologicalConfigValidator()
        with pytest.raises(ConfigurationValidationError, match="must be a dictionary"):
            v.validate({"species_configs": {"sp1": "bad"}})

    def test_missing_required_field(self):
        v = BiologicalConfigValidator()
        with pytest.raises(ConfigurationValidationError, match="Missing required"):
            v.validate({"species_configs": {"sp1": {"max_growth_rate": 0.5}}})

    def test_substrate_configs_valid(self):
        v = BiologicalConfigValidator()
        assert v.validate({"substrate_configs": {"s1": {"molecular_weight": 180}}}) is True

    def test_substrate_not_dict(self):
        v = BiologicalConfigValidator()
        with pytest.raises(ConfigurationValidationError, match="must be a dictionary"):
            v.validate({"substrate_configs": {"s1": "bad"}})


class TestControlConfigValidator:
    def test_validate_calls_module(self):
        v = ControlConfigValidator()
        with patch("config.control_config.validate_control_config", return_value=True) as mock_fn:
            mock_config = MagicMock()
            result = v.validate(mock_config)
            assert result is True
            mock_fn.assert_called_once_with(mock_config)


class TestVisualizationConfigValidator:
    def test_validate_calls_module(self):
        v = VisualizationConfigValidator()
        with patch("config.visualization_config.validate_visualization_config", return_value=True) as mock_fn:
            mock_config = MagicMock()
            result = v.validate(mock_config)
            assert result is True
            mock_fn.assert_called_once_with(mock_config)


class TestConfigMerger:
    def test_shallow_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = ConfigMerger.merge_dicts(base, override, deep_merge=False)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 5, "z": 6}, "c": 7}
        result = ConfigMerger.merge_dicts(base, override, deep_merge=True)
        assert result["a"] == {"x": 1, "y": 5, "z": 6}
        assert result["b"] == 3
        assert result["c"] == 7

    def test_merge_profiles_both_biological(self):
        base = ConfigProfile(biological={"a": 1}, profile_name="base")
        override = ConfigProfile(biological={"b": 2}, profile_name="override", profile_version="2.0")
        merged = ConfigMerger.merge_profiles(base, override)
        assert merged.biological == {"a": 1, "b": 2}
        assert merged.profile_name == "override"

    def test_merge_profiles_one_biological(self):
        base = ConfigProfile(biological={"a": 1})
        override = ConfigProfile(biological=None)
        merged = ConfigMerger.merge_profiles(base, override)
        assert merged.biological == {"a": 1}

    def test_merge_profiles_override_biological(self):
        base = ConfigProfile(biological=None)
        override = ConfigProfile(biological={"b": 2})
        merged = ConfigMerger.merge_profiles(base, override)
        assert merged.biological == {"b": 2}

    def test_merge_profiles_control_viz(self):
        mock_ctrl = MagicMock()
        mock_viz = MagicMock()
        base = ConfigProfile()
        override = ConfigProfile(control=mock_ctrl, visualization=mock_viz)
        merged = ConfigMerger.merge_profiles(base, override)
        assert merged.control is mock_ctrl
        assert merged.visualization is mock_viz

    def test_merge_profiles_context(self):
        base = ConfigProfile(context={"a": 1})
        override = ConfigProfile(context={"b": 2})
        merged = ConfigMerger.merge_profiles(base, override)
        assert merged.context == {"a": 1, "b": 2}


class TestConfigManager:
    @pytest.fixture
    def manager(self, tmp_path):
        return ConfigManager(config_directory=str(tmp_path / "configs"))

    def test_init(self, manager):
        assert manager.config_directory.exists()
        assert "biological" in manager.validators
        assert manager.current_profile is None

    def test_create_profile(self, manager):
        p = manager.create_profile("test", biological={"key": "val"})
        assert p.profile_name == "test"
        assert "test" in manager.profiles

    def test_create_profile_with_inheritance(self, manager):
        manager.create_profile("base", biological={"a": 1})
        p = manager.create_profile("child", inherits_from="base")
        assert "child" in manager.profiles

    def test_get_profile(self, manager):
        manager.create_profile("test")
        p = manager.get_profile("test")
        assert p.profile_name == "test"

    def test_get_profile_not_found(self, manager):
        with pytest.raises(ConfigurationError, match="Profile not found"):
            manager.get_profile("nope")

    def test_get_profile_with_inheritance(self, manager):
        manager.create_profile("base", biological={"a": 1})
        manager.create_profile("child", biological={"b": 2}, inherits_from="base")
        p = manager.get_profile("child")
        assert p.biological == {"a": 1, "b": 2}

    def test_set_current_profile(self, manager):
        manager.create_profile("test")
        manager.set_current_profile("test")
        assert manager.current_profile == "test"

    def test_set_current_profile_not_found(self, manager):
        with pytest.raises(ConfigurationError):
            manager.set_current_profile("nope")

    def test_get_current_profile(self, manager):
        manager.create_profile("test")
        manager.set_current_profile("test")
        p = manager.get_current_profile()
        assert p.profile_name == "test"

    def test_get_current_profile_none(self, manager):
        assert manager.get_current_profile() is None

    def test_delete_profile(self, manager):
        manager.create_profile("test")
        manager.delete_profile("test")
        assert "test" not in manager.profiles

    def test_delete_profile_not_found(self, manager):
        with pytest.raises(ConfigurationError):
            manager.delete_profile("nope")

    def test_delete_active_profile(self, manager):
        manager.create_profile("test")
        manager.set_current_profile("test")
        with pytest.raises(ConfigurationError, match="Cannot delete active"):
            manager.delete_profile("test")

    def test_list_profiles(self, manager):
        manager.create_profile("a")
        manager.create_profile("b")
        names = manager.list_profiles()
        assert "a" in names and "b" in names

    def test_get_configuration(self, manager):
        manager.create_profile("test", biological={"x": 1})
        manager.set_current_profile("test")
        bio = manager.get_configuration("biological")
        assert bio == {"x": 1}

    def test_get_configuration_explicit_profile(self, manager):
        manager.create_profile("test", biological={"x": 1})
        bio = manager.get_configuration("biological", "test")
        assert bio == {"x": 1}

    def test_get_configuration_no_current(self, manager):
        with pytest.raises(ConfigurationError, match="No current profile"):
            manager.get_configuration("biological")

    def test_get_configuration_unknown_type(self, manager):
        manager.create_profile("test")
        manager.set_current_profile("test")
        with pytest.raises(ConfigurationError, match="Unknown configuration"):
            manager.get_configuration("unknown")

    def test_get_configuration_control_viz(self, manager):
        manager.create_profile("test")
        manager.set_current_profile("test")
        assert manager.get_configuration("control") is None
        assert manager.get_configuration("visualization") is None

    def test_update_configuration(self, manager):
        manager.create_profile("test")
        manager.set_current_profile("test")
        manager.update_configuration("biological", {"new": "data"})
        assert manager.profiles["test"].biological == {"new": "data"}

    def test_update_configuration_control(self, manager):
        manager.create_profile("test")
        manager.set_current_profile("test")
        from config.control_config import ControlSystemConfig
        ctrl = ControlSystemConfig()
        manager.update_configuration("control", ctrl)
        assert manager.profiles["test"].control is ctrl

    def test_update_configuration_visualization(self, manager):
        manager.create_profile("test")
        manager.set_current_profile("test")
        from config.visualization_config import VisualizationConfig
        viz = VisualizationConfig()
        manager.update_configuration("visualization", viz)
        assert manager.profiles["test"].visualization is viz

    def test_update_configuration_unknown(self, manager):
        manager.create_profile("test")
        manager.set_current_profile("test")
        with pytest.raises(ConfigurationError, match="Unknown configuration"):
            manager.update_configuration("unknown", {})

    def test_update_configuration_no_current(self, manager):
        with pytest.raises(ConfigurationError, match="No current profile"):
            manager.update_configuration("biological", {})

    def test_update_configuration_profile_not_found(self, manager):
        manager.current_profile = "nonexistent"
        with pytest.raises(ConfigurationError, match="Profile not found"):
            manager.update_configuration("biological", {})

    def test_validate_profile_with_bio(self, manager):
        p = ConfigProfile(biological={"species_configs": {"s": {"max_growth_rate": 1.0, "electron_transport_efficiency": 0.5}}})
        assert manager.validate_profile(p) is True

    def test_validate_profile_invalid_bio(self, manager):
        p = ConfigProfile(biological="bad")
        with pytest.raises(ConfigurationValidationError):
            manager.validate_profile(p)

    def test_load_profile_from_yaml(self, manager, tmp_path):
        cfg = {"biological": {"key": "val"}, "metadata": {"description": "test", "version": "2.0", "environment": "testing", "inherits_from": "base"}}
        f = tmp_path / "configs" / "test.yaml"
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "w") as fh:
            yaml.dump(cfg, fh)
        p = manager.load_profile_from_file("test", str(f))
        assert p.biological == {"key": "val"}
        assert p.profile_version == "2.0"

    def test_load_profile_from_json(self, manager, tmp_path):
        cfg = {"biological": {"key": "val"}}
        f = tmp_path / "configs" / "test.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "w") as fh:
            json.dump(cfg, fh)
        p = manager.load_profile_from_file("test", str(f))
        assert p.biological == {"key": "val"}

    def test_load_profile_not_found(self, manager):
        with pytest.raises(ConfigurationLoadError):
            manager.load_profile_from_file("test", "/nonexistent/path.yaml")

    def test_load_profile_unsupported_format(self, manager, tmp_path):
        f = tmp_path / "test.xml"
        f.write_text("<xml/>")
        with pytest.raises(ConfigurationLoadError):
            manager.load_profile_from_file("test", str(f))

    def test_load_profile_with_control_dict(self, manager, tmp_path):
        cfg = {"control": {"control_update_interval": 30.0}}
        f = tmp_path / "configs" / "ctrl.yaml"
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "w") as fh:
            yaml.dump(cfg, fh)
        p = manager.load_profile_from_file("ctrl", str(f))
        assert p.control is not None

    def test_load_profile_with_viz_dict(self, manager, tmp_path):
        cfg = {"visualization": {"output_format": "svg"}}
        f = tmp_path / "configs" / "viz.yaml"
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "w") as fh:
            yaml.dump(cfg, fh)
        p = manager.load_profile_from_file("viz", str(f))
        assert p.visualization is not None

    def test_load_profile_with_control_non_dict(self, manager, tmp_path):
        cfg = {"control": "raw_value"}
        f = tmp_path / "configs" / "ctrl2.yaml"
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "w") as fh:
            yaml.dump(cfg, fh)
        with pytest.raises(ConfigurationLoadError):
            manager.load_profile_from_file("ctrl2", str(f))

    def test_load_profile_with_viz_non_dict(self, manager, tmp_path):
        cfg = {"visualization": "raw_value"}
        f = tmp_path / "configs" / "viz2.yaml"
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "w") as fh:
            yaml.dump(cfg, fh)
        with pytest.raises(ConfigurationLoadError):
            manager.load_profile_from_file("viz2", str(f))

    def test_save_profile_yaml(self, manager, tmp_path):
        manager.create_profile("test", biological={"a": 1})
        out = str(tmp_path / "out.yaml")
        manager.save_profile_to_file("test", out, format="yaml")
        assert Path(out).exists()

    def test_save_profile_json(self, manager, tmp_path):
        manager.create_profile("test", biological={"a": 1})
        out = str(tmp_path / "out.json")
        manager.save_profile_to_file("test", out, format="json")
        assert Path(out).exists()

    def test_save_profile_not_found(self, manager, tmp_path):
        with pytest.raises(ConfigurationError, match="Profile not found"):
            manager.save_profile_to_file("nope", str(tmp_path / "out.yaml"))

    def test_save_profile_unsupported_format(self, manager, tmp_path):
        manager.create_profile("test")
        with pytest.raises(ConfigurationError):
            manager.save_profile_to_file("test", str(tmp_path / "out.xml"), format="xml")

    def test_load_default_profiles(self, tmp_path):
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        f = cfg_dir / "default.yaml"
        with open(f, "w") as fh:
            yaml.dump({"biological": {"a": 1}}, fh)
        mgr = ConfigManager(config_directory=str(cfg_dir))
        assert "default" in mgr.profiles

    def test_load_default_profiles_json(self, tmp_path):
        cfg_dir = tmp_path / "configs_json"
        cfg_dir.mkdir()
        f = cfg_dir / "myjson.json"
        with open(f, "w") as fh:
            json.dump({"biological": {"b": 2}}, fh)
        mgr = ConfigManager(config_directory=str(cfg_dir))
        assert "myjson" in mgr.profiles

    def test_load_default_profiles_failure(self, tmp_path):
        cfg_dir = tmp_path / "configs_bad"
        cfg_dir.mkdir()
        f = cfg_dir / "bad.yaml"
        f.write_text("{{invalid yaml::::")
        mgr = ConfigManager(config_directory=str(cfg_dir))
        assert "bad" not in mgr.profiles


class TestGlobalFunctions:
    def test_get_config_manager(self, tmp_path):
        import config.config_manager as mod
        mod._config_manager = None
        mgr = get_config_manager(str(tmp_path / "cfgs"))
        assert mgr is not None
        mgr2 = get_config_manager()
        assert mgr2 is mgr
        mod._config_manager = None

    def test_reset_config_manager(self):
        import config.config_manager as mod
        mod._config_manager = MagicMock()
        reset_config_manager()
        assert mod._config_manager is None
