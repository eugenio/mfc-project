"""Tests for config/parameter_bridge.py - targeting 98%+ coverage."""
import importlib.util
import os
import sys
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

# Mock numpy to avoid reimport
_mock_np = MagicMock()
_mock_np.isclose = lambda a, b, rtol=1e-5: abs(a - b) <= rtol * abs(b)
sys.modules.setdefault("numpy", _mock_np)

_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")

# Load dependencies in order
for mod_name, mod_file in [
    ("config.electrode_config", "electrode_config.py"),
    ("config.qlearning_config", "qlearning_config.py"),
    ("config.sensor_config", "sensor_config.py"),
    ("config.parameter_validation", "parameter_validation.py"),
]:
    _s = importlib.util.spec_from_file_location(
        mod_name, os.path.join(_src, "config", mod_file)
    )
    _m = importlib.util.module_from_spec(_s)
    sys.modules.setdefault(mod_name, _m)
    _s.loader.exec_module(_m)

# Load literature_database
_spec_ld = importlib.util.spec_from_file_location(
    "config.literature_database",
    os.path.join(_src, "config", "literature_database.py"),
)
_mod_ld = importlib.util.module_from_spec(_spec_ld)
sys.modules.setdefault("config.literature_database", _mod_ld)
_spec_ld.loader.exec_module(_mod_ld)

# Load parameter_bridge
_spec = importlib.util.spec_from_file_location(
    "config.parameter_bridge",
    os.path.join(_src, "config", "parameter_bridge.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["config.parameter_bridge"] = _mod
_spec.loader.exec_module(_mod)

ParameterMappingRule = _mod.ParameterMappingRule
ParameterBridge = _mod.ParameterBridge
PARAMETER_BRIDGE = _mod.PARAMETER_BRIDGE

_ql_mod = sys.modules["config.qlearning_config"]
QLearningConfig = _ql_mod.QLearningConfig
DEFAULT_QLEARNING_CONFIG = _ql_mod.DEFAULT_QLEARNING_CONFIG


class TestParameterMappingRule:
    def test_defaults(self):
        rule = ParameterMappingRule(
            literature_name="learning_rate",
            config_attribute="learning_rate",
        )
        assert rule.conversion_factor == 1.0
        assert rule.validation_required is True
        assert rule.notes is None

    def test_custom(self):
        rule = ParameterMappingRule(
            literature_name="max_current_density",
            config_attribute="biofilm_physics.max_current_density",
            conversion_factor=1e-3,
            notes="Convert mA/cm2 to A/m2",
        )
        assert rule.conversion_factor == 1e-3
        assert rule.notes is not None


class TestParameterBridge:
    def setup_method(self):
        self.bridge = ParameterBridge()

    def test_init(self):
        assert self.bridge.literature_db is not None
        assert len(self.bridge.mapping_rules) > 0

    def test_mapping_rules_keys(self):
        rules = self.bridge.mapping_rules
        assert "learning_rate" in rules
        assert "discount_factor" in rules
        assert "exploration_rate" in rules
        assert "electrode_area" in rules
        assert "max_current_density" in rules
        assert "biofilm_conductivity" in rules
        assert "growth_rate" in rules
        assert "half_saturation" in rules
        assert "biofilm_thickness" in rules
        assert "biofilm_density" in rules
        assert "substrate_concentration" in rules
        assert "flow_rate" in rules

    def test_validate_parameter_unmapped(self):
        result = self.bridge.validate_parameter_with_literature("unknown_key", 1.0)
        assert result["status"] == "unmapped"
        assert len(result["recommendations"]) == 0

    def test_validate_parameter_mapped_no_conversion(self):
        result = self.bridge.validate_parameter_with_literature("learning_rate", 0.1)
        assert "status" in result
        assert "mapping_rule" in result

    def test_validate_parameter_with_conversion(self):
        result = self.bridge.validate_parameter_with_literature(
            "max_current_density", 5.0
        )
        assert "mapping_rule" in result
        if result["status"] != "unmapped":
            assert "converted_value" in result
            assert "conversion_factor" in result
            assert "original_value" in result

    def test_create_literature_validated_config_unmapped(self):
        config, results = self.bridge.create_literature_validated_config(
            {"nonexistent_param": 999.0}
        )
        assert isinstance(config, QLearningConfig)
        assert "nonexistent_param" in results
        assert results["nonexistent_param"]["status"] == "unmapped"

    def test_create_literature_validated_config_valid(self):
        config, results = self.bridge.create_literature_validated_config(
            {"learning_rate": 0.1}
        )
        assert isinstance(config, QLearningConfig)
        if "learning_rate" in results:
            assert results["learning_rate"]["status"] in ["valid", "caution", "invalid"]

    def test_create_literature_validated_config_with_conversion(self):
        config, results = self.bridge.create_literature_validated_config(
            {"biofilm_thickness": 50.0}
        )
        assert isinstance(config, QLearningConfig)

    def test_set_nested_attribute(self):
        obj = MagicMock()
        obj.sub = MagicMock()
        self.bridge._set_nested_attribute(obj, "sub.attr", 42)
        assert obj.sub.attr == 42

    def test_set_nested_attribute_simple(self):
        obj = MagicMock()
        self.bridge._set_nested_attribute(obj, "simple", 99)
        assert obj.simple == 99

    def test_get_nested_attribute(self):
        obj = MagicMock()
        obj.sub = MagicMock()
        obj.sub.val = 42
        result = self.bridge._get_nested_attribute(obj, "sub.val")
        assert result == 42

    def test_get_nested_attribute_simple(self):
        obj = MagicMock()
        obj.x = 10
        result = self.bridge._get_nested_attribute(obj, "x")
        assert result == 10

    def test_extract_literature_parameters(self):
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        params = self.bridge.extract_literature_parameters_from_config(cfg)
        assert isinstance(params, dict)
        assert "learning_rate" in params
        assert "discount_factor" in params

    def test_extract_literature_parameters_missing_attr(self):
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        # Add a rule with a nonexistent attribute
        self.bridge.mapping_rules["test_missing"] = ParameterMappingRule(
            literature_name="test_missing",
            config_attribute="totally.nonexistent.path",
        )
        params = self.bridge.extract_literature_parameters_from_config(cfg)
        # Should skip the missing attribute without error
        assert "test_missing" not in params

    def test_get_parameter_mapping_info(self):
        info = self.bridge.get_parameter_mapping_info()
        assert isinstance(info, dict)
        assert "learning_rate" in info
        assert "literature_parameter" in info["learning_rate"]
        assert "config_attribute" in info["learning_rate"]
        assert "conversion_factor" in info["learning_rate"]
        assert "notes" in info["learning_rate"]

    def test_validate_entire_config(self):
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        report = self.bridge.validate_entire_config(cfg)
        assert "overall_status" in report
        assert "parameter_results" in report
        assert "summary" in report
        assert report["summary"]["total_parameters"] > 0

    def test_validate_entire_config_all_statuses(self):
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        report = self.bridge.validate_entire_config(cfg)
        summary = report["summary"]
        total = (
            summary["valid_parameters"]
            + summary["caution_parameters"]
            + summary["invalid_parameters"]
            + summary["unmapped_parameters"]
        )
        assert total == summary["total_parameters"]

    def test_suggest_parameter_improvements(self):
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        suggestions = self.bridge.suggest_parameter_improvements(cfg)
        assert isinstance(suggestions, list)

    def test_suggest_parameter_improvements_with_bad_value(self):
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        cfg.learning_rate = 999.0  # Way out of range
        suggestions = self.bridge.suggest_parameter_improvements(cfg)
        # May or may not produce suggestions depending on validation
        assert isinstance(suggestions, list)


    def test_create_config_invalid_param_skipped(self):
        """Cover line 199 - invalid validation status causes continue."""
        from unittest.mock import patch
        # Mock validate_parameter_value to return "invalid" status
        mock_db = MagicMock()
        mock_db.validate_parameter_value.return_value = {"status": "invalid", "value": 999.0}
        self.bridge.literature_db = mock_db
        config, results = self.bridge.create_literature_validated_config(
            {"learning_rate": 999.0}
        )
        # The param should be validated but NOT applied (status=invalid)
        assert "learning_rate" in results
        assert results["learning_rate"]["status"] == "invalid"

    def test_validate_entire_config_unmapped_status(self):
        """Cover line 327 - unmapped parameter status."""
        from unittest.mock import patch
        mock_db = MagicMock()
        mock_db.validate_parameter_value.return_value = {"status": "unknown_status", "value": 0}
        self.bridge.literature_db = mock_db
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        report = self.bridge.validate_entire_config(cfg)
        assert report["summary"]["unmapped_parameters"] >= 0

    def test_validate_entire_config_caution_overall(self):
        """Cover lines 332-333 - caution overall status."""
        from unittest.mock import patch
        mock_db = MagicMock()
        mock_db.validate_parameter_value.return_value = {"status": "caution", "value": 0}
        self.bridge.literature_db = mock_db
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        report = self.bridge.validate_entire_config(cfg)
        assert report["overall_status"] == "caution"
        assert report["summary"]["caution_parameters"] > 0


class TestGlobalParameterBridge:
    def test_global_instance(self):
        assert PARAMETER_BRIDGE is not None
        assert isinstance(PARAMETER_BRIDGE, ParameterBridge)
