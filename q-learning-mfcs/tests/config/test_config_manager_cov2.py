"""Coverage boost tests for config_manager.py - targeting uncovered lines."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.config_manager import (
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


@pytest.mark.coverage_extra
class TestConfigManagerLoadProfile:
    """Test load_profile_from_file uncovered branches."""

    def test_load_json_profile(self, tmp_path):
        config_data = {
            "biological": {"species_configs": {}},
            "metadata": {
                "description": "Test profile",
                "version": "2.0.0",
                "environment": "testing",
                "inherits_from": "base",
            },
        }
        json_file = tmp_path / "test_profile.json"
        json_file.write_text(json.dumps(config_data))
        mgr = ConfigManager(str(tmp_path / "configs"))
        profile = mgr.load_profile_from_file("json_test", str(json_file))
        assert profile.profile_name == "json_test"
        assert profile.description == "Test profile"
        assert profile.profile_version == "2.0.0"
        assert profile.environment == "testing"
        assert profile.inherits_from == "base"

    def test_load_unsupported_format(self, tmp_path):
        txt_file = tmp_path / "bad.txt"
        txt_file.write_text("data")
        mgr = ConfigManager(str(tmp_path / "configs"))
        with pytest.raises(ConfigurationLoadError):
            mgr.load_profile_from_file("bad", str(txt_file))

    def test_load_nonexistent_file(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        with pytest.raises(ConfigurationLoadError):
            mgr.load_profile_from_file("missing", str(tmp_path / "nope.yaml"))

    def test_load_profile_with_control_dict(self, tmp_path):
        config_data = {"control": {"some_setting": 42}}
        yaml_file = tmp_path / "ctrl.yaml"
        yaml_file.write_text(yaml.dump(config_data))
        mgr = ConfigManager(str(tmp_path / "configs"))
        with patch.object(mgr, "_dict_to_control_config") as mock_ctrl, \
             patch.object(mgr, "validate_profile", return_value=True):
            mock_ctrl.return_value = MagicMock()
            mgr.load_profile_from_file("ctrl", str(yaml_file))
            mock_ctrl.assert_called_once()

    def test_load_profile_with_viz_dict(self, tmp_path):
        config_data = {"visualization": {"color_scheme": "dark"}}
        yaml_file = tmp_path / "viz.yaml"
        yaml_file.write_text(yaml.dump(config_data))
        mgr = ConfigManager(str(tmp_path / "configs"))
        with patch.object(mgr, "_dict_to_visualization_config") as mock_viz, \
             patch.object(mgr, "validate_profile", return_value=True):
            mock_viz.return_value = MagicMock()
            mgr.load_profile_from_file("viz", str(yaml_file))
            mock_viz.assert_called_once()

    def test_load_profile_with_non_dict_control(self, tmp_path):
        config_data = {"control": "string_value"}
        yaml_file = tmp_path / "ctrl_str.yaml"
        yaml_file.write_text(yaml.dump(config_data))
        mgr = ConfigManager(str(tmp_path / "configs"))
        with patch.object(mgr, "validate_profile", return_value=True):
            profile = mgr.load_profile_from_file("ctrl_str", str(yaml_file))
            assert profile.control == "string_value"

    def test_load_profile_with_non_dict_viz(self, tmp_path):
        config_data = {"visualization": "string_viz"}
        yaml_file = tmp_path / "viz_str.yaml"
        yaml_file.write_text(yaml.dump(config_data))
        mgr = ConfigManager(str(tmp_path / "configs"))
        with patch.object(mgr, "validate_profile", return_value=True):
            profile = mgr.load_profile_from_file("viz_str", str(yaml_file))
            assert profile.visualization == "string_viz"


@pytest.mark.coverage_extra
class TestConfigManagerSaveProfile:
    def test_save_profile_json(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        mgr.create_profile("save_test", biological={"key": "val"})
        out_file = tmp_path / "output.json"
        mgr.save_profile_to_file("save_test", str(out_file), format="json")
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert "biological" in data

    def test_save_profile_unsupported_format(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        mgr.create_profile("fmt_test")
        with pytest.raises(ConfigurationError):
            mgr.save_profile_to_file("fmt_test", str(tmp_path / "out.xml"), format="xml")

    def test_save_nonexistent_profile(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        with pytest.raises(ConfigurationError):
            mgr.save_profile_to_file("nope", str(tmp_path / "out.yaml"))


@pytest.mark.coverage_extra
class TestConfigManagerGetConfiguration:
    def test_get_configuration_no_current_profile(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        with pytest.raises(ConfigurationError, match="No current profile set"):
            mgr.get_configuration("biological")

    def test_get_configuration_unknown_type(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        mgr.create_profile("test_p")
        with pytest.raises(ConfigurationError, match="Unknown configuration type"):
            mgr.get_configuration("unknown_type", "test_p")

    def test_get_configuration_control(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        mgr.create_profile("test_p")
        result = mgr.get_configuration("control", "test_p")
        assert result is None

    def test_get_configuration_visualization(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        mgr.create_profile("test_p")
        result = mgr.get_configuration("visualization", "test_p")
        assert result is None

    def test_update_configuration_no_current(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        with pytest.raises(ConfigurationError, match="No current profile set"):
            mgr.update_configuration("biological", {"key": "val"})

    def test_update_configuration_unknown_type(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        mgr.create_profile("test_p")
        mgr.set_current_profile("test_p")
        with pytest.raises(ConfigurationError):
            mgr.update_configuration("nonexistent", {})

    def test_update_configuration_control(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        mgr.create_profile("test_p")
        mgr.set_current_profile("test_p")
        with patch.object(mgr, "validate_profile", return_value=True):
            mgr.update_configuration("control", MagicMock())

    def test_update_configuration_visualization(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        mgr.create_profile("test_p")
        mgr.set_current_profile("test_p")
        with patch.object(mgr, "validate_profile", return_value=True):
            mgr.update_configuration("visualization", MagicMock())

    def test_update_nonexistent_profile(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        mgr.create_profile("test_p")
        mgr.current_profile = "test_p"
        with pytest.raises(ConfigurationError):
            mgr.update_configuration("biological", {}, profile_name="nope")


@pytest.mark.coverage_extra
class TestConfigProfileInheritance:
    def test_get_profile_with_inheritance(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        mgr.create_profile("base", biological={"base_key": "base_val"})
        mgr.create_profile("child", biological={"child_key": "child_val"}, inherits_from="base")
        result = mgr.get_profile("child")
        assert result.biological["base_key"] == "base_val"
        assert result.biological["child_key"] == "child_val"


@pytest.mark.coverage_extra
class TestDeleteProfile:
    def test_delete_active_profile(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        mgr.create_profile("active")
        mgr.set_current_profile("active")
        with pytest.raises(ConfigurationError, match="Cannot delete active"):
            mgr.delete_profile("active")


@pytest.mark.coverage_extra
class TestValidateProfile:
    def test_validate_profile_with_control(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        profile = ConfigProfile()
        profile.control = MagicMock()
        with patch.object(mgr.validators["control"], "validate", return_value=True):
            assert mgr.validate_profile(profile) is True

    def test_validate_profile_with_viz(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        profile = ConfigProfile()
        profile.visualization = MagicMock()
        with patch.object(mgr.validators["visualization"], "validate", return_value=True):
            assert mgr.validate_profile(profile) is True

    def test_validate_profile_exception(self, tmp_path):
        mgr = ConfigManager(str(tmp_path / "configs"))
        profile = ConfigProfile()
        profile.biological = "not_a_dict"
        with pytest.raises(ConfigurationValidationError):
            mgr.validate_profile(profile)


@pytest.mark.coverage_extra
class TestGetConfigManagerSingleton:
    def test_reset_and_get(self, tmp_path):
        reset_config_manager()
        mgr = get_config_manager(str(tmp_path / "singleton_configs"))
        assert mgr is not None
        mgr2 = get_config_manager()
        assert mgr is mgr2
        reset_config_manager()


@pytest.mark.coverage_extra
class TestConfigMergerProfiles:
    def test_merge_profiles_base_only_biological(self):
        base = ConfigProfile(biological={"a": 1})
        override = ConfigProfile(biological=None)
        merged = ConfigMerger.merge_profiles(base, override)
        assert merged.biological == {"a": 1}

    def test_merge_profiles_override_only_biological(self):
        base = ConfigProfile(biological=None)
        override = ConfigProfile(biological={"b": 2})
        merged = ConfigMerger.merge_profiles(base, override)
        assert merged.biological == {"b": 2}

    def test_merge_dicts_shallow(self):
        result = ConfigMerger.merge_dicts({"a": 1, "b": 2}, {"b": 3, "c": 4}, deep_merge=False)
        assert result == {"a": 1, "b": 3, "c": 4}
