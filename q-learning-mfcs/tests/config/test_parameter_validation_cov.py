"""Tests for config/parameter_validation.py - targeting 98%+ coverage."""
import importlib.util
import os
import sys
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

# Mock numpy before loading the module to avoid reimport conflicts
_mock_np = MagicMock()
_mock_np.isclose = lambda a, b, rtol=1e-5: abs(a - b) <= rtol * abs(b)
sys.modules.setdefault("numpy", _mock_np)

# Load the module via importlib to avoid config __init__ heavy imports
_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")

# First load electrode_config (dependency of qlearning_config)
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

# Now load parameter_validation
_spec = importlib.util.spec_from_file_location(
    "config.parameter_validation",
    os.path.join(_src, "config", "parameter_validation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["config.parameter_validation"] = _mod
_spec.loader.exec_module(_mod)

ConfigValidationError = _mod.ConfigValidationError
validate_range = _mod.validate_range
validate_positive = _mod.validate_positive
validate_probability = _mod.validate_probability
validate_qlearning_config = _mod.validate_qlearning_config
validate_qlearning_reward_weights = _mod.validate_qlearning_reward_weights
validate_sensor_config = _mod.validate_sensor_config
validate_eis_config = _mod.validate_eis_config
validate_qcm_config = _mod.validate_qcm_config
validate_sensor_fusion_config = _mod.validate_sensor_fusion_config
validate_all_configurations = _mod.validate_all_configurations
quick_validate = _mod.quick_validate

QLearningConfig = _mod_ql.QLearningConfig
QLearningRewardWeights = _mod_ql.QLearningRewardWeights
DEFAULT_QLEARNING_CONFIG = _mod_ql.DEFAULT_QLEARNING_CONFIG
SensorConfig = _mod_sc.SensorConfig
EISConfig = _mod_sc.EISConfig
QCMConfig = _mod_sc.QCMConfig
SensorFusionConfig = _mod_sc.SensorFusionConfig


class TestConfigValidationError:
    def test_attributes(self):
        err = ConfigValidationError("param", 42, "bad value")
        assert err.parameter == "param"
        assert err.value == 42
        assert err.message == "bad value"

    def test_str(self):
        err = ConfigValidationError("param", 42, "bad value")
        assert "param" in str(err)
        assert "42" in str(err)
        assert "bad value" in str(err)


class TestValidateRange:
    def test_inclusive_both(self):
        validate_range(5.0, 0.0, 10.0, "x")

    def test_inclusive_both_fail(self):
        with pytest.raises(ConfigValidationError):
            validate_range(11.0, 0.0, 10.0, "x")

    def test_inclusive_min_exclusive_max(self):
        validate_range(5.0, 0.0, 10.0, "x", True, False)
        with pytest.raises(ConfigValidationError):
            validate_range(10.0, 0.0, 10.0, "x", True, False)

    def test_exclusive_min_inclusive_max(self):
        validate_range(5.0, 0.0, 10.0, "x", False, True)
        with pytest.raises(ConfigValidationError):
            validate_range(0.0, 0.0, 10.0, "x", False, True)

    def test_exclusive_both(self):
        validate_range(5.0, 0.0, 10.0, "x", False, False)
        with pytest.raises(ConfigValidationError):
            validate_range(0.0, 0.0, 10.0, "x", False, False)
        with pytest.raises(ConfigValidationError):
            validate_range(10.0, 0.0, 10.0, "x", False, False)

    def test_custom_message(self):
        with pytest.raises(ConfigValidationError, match="custom msg"):
            validate_range(11.0, 0.0, 10.0, "x", "custom msg")

    def test_custom_message_in_range(self):
        validate_range(5.0, 0.0, 10.0, "x", "custom msg")

    def test_default_error_message(self):
        with pytest.raises(ConfigValidationError, match="must be in range"):
            validate_range(-1.0, 0.0, 10.0, "x")


class TestValidatePositive:
    def test_positive_ok(self):
        validate_positive(1.0, "x")

    def test_zero_not_ok_default(self):
        with pytest.raises(ConfigValidationError, match="must be positive"):
            validate_positive(0.0, "x")

    def test_negative_not_ok(self):
        with pytest.raises(ConfigValidationError, match="must be positive"):
            validate_positive(-1.0, "x")

    def test_allow_zero(self):
        validate_positive(0.0, "x", allow_zero=True)

    def test_negative_with_allow_zero(self):
        with pytest.raises(ConfigValidationError, match="must be non-negative"):
            validate_positive(-1.0, "x", allow_zero=True)


class TestValidateProbability:
    def test_valid(self):
        validate_probability(0.5, "p")
        validate_probability(0.0, "p")
        validate_probability(1.0, "p")

    def test_invalid(self):
        with pytest.raises(ConfigValidationError):
            validate_probability(-0.1, "p")
        with pytest.raises(ConfigValidationError):
            validate_probability(1.1, "p")


class TestValidateQLearningConfig:
    def _default(self):
        return deepcopy(DEFAULT_QLEARNING_CONFIG)

    def test_default_config_valid(self):
        assert validate_qlearning_config(self._default()) is True

    def test_invalid_learning_rate(self):
        cfg = self._default()
        cfg.learning_rate = 0.0
        with pytest.raises(ConfigValidationError):
            validate_qlearning_config(cfg)

    def test_invalid_discount_factor(self):
        cfg = self._default()
        cfg.discount_factor = 1.5
        with pytest.raises(ConfigValidationError):
            validate_qlearning_config(cfg)

    def test_invalid_epsilon(self):
        cfg = self._default()
        cfg.epsilon = -0.1
        with pytest.raises(ConfigValidationError):
            validate_qlearning_config(cfg)

    def test_epsilon_min_ge_epsilon(self):
        cfg = self._default()
        cfg.epsilon = 0.3
        cfg.epsilon_min = 0.5
        with pytest.raises(ConfigValidationError, match="must be less than epsilon"):
            validate_qlearning_config(cfg)

    def test_epsilon_zero_skips_min_check(self):
        cfg = self._default()
        cfg.epsilon = 0.0
        cfg.epsilon_min = 0.5
        # Should not raise for epsilon relationship when epsilon == 0
        # But may fail for other reasons, so just check it doesn't fail on this specific check
        # Actually epsilon=0 is boundary - validate_probability accepts 0
        # The epsilon_min >= epsilon check is skipped when epsilon == 0
        # But validate_probability(0.0) passes, so epsilon=0 is ok
        # However epsilon_decay validation may fail
        try:
            validate_qlearning_config(cfg)
        except ConfigValidationError as e:
            assert "epsilon_min" not in e.parameter or "must be less than" not in e.message

    def test_advanced_epsilon_min_ge_enhanced_epsilon(self):
        cfg = self._default()
        cfg.enhanced_epsilon = 0.3
        cfg.advanced_epsilon_min = 0.5
        with pytest.raises(ConfigValidationError, match="must be less than enhanced_epsilon"):
            validate_qlearning_config(cfg)

    def test_enhanced_epsilon_zero_skips_check(self):
        cfg = self._default()
        cfg.enhanced_epsilon = 0.0
        cfg.advanced_epsilon_min = 0.5
        # Should skip the check
        try:
            validate_qlearning_config(cfg)
        except ConfigValidationError as e:
            assert "advanced_epsilon_min" not in e.parameter or "must be less than" not in e.message

    def test_invalid_power_bins(self):
        cfg = self._default()
        cfg.power_bins = 0
        with pytest.raises(ConfigValidationError):
            validate_qlearning_config(cfg)

    def test_flow_rate_min_ge_max(self):
        cfg = self._default()
        cfg.flow_rate_min = 60.0
        cfg.flow_rate_max = 50.0
        with pytest.raises(ConfigValidationError, match="must be less than flow_rate_max"):
            validate_qlearning_config(cfg)

    def test_weights_not_sum_to_one(self):
        cfg = self._default()
        cfg.power_objective_weight = 0.5
        cfg.biofilm_health_weight = 0.5
        cfg.sensor_agreement_weight = 0.5
        cfg.stability_weight = 0.5
        with pytest.raises(ConfigValidationError, match="should sum to 1.0"):
            validate_qlearning_config(cfg)


class TestValidateQLearningRewardWeights:
    def _default_weights(self):
        return deepcopy(DEFAULT_QLEARNING_CONFIG.reward_weights)

    def test_default_valid(self):
        assert validate_qlearning_reward_weights(self._default_weights()) is True

    def test_invalid_power_weight(self):
        w = self._default_weights()
        w.power_weight = 0.0
        with pytest.raises(ConfigValidationError):
            validate_qlearning_reward_weights(w)

    def test_invalid_consumption_weight(self):
        w = self._default_weights()
        w.consumption_weight = -1.0
        with pytest.raises(ConfigValidationError):
            validate_qlearning_reward_weights(w)

    def test_biofilm_penalty_positive(self):
        w = self._default_weights()
        w.biofilm_penalty = 1.0
        with pytest.raises(ConfigValidationError, match="should be negative"):
            validate_qlearning_reward_weights(w)

    def test_combined_penalty_positive(self):
        w = self._default_weights()
        w.combined_penalty = 1.0
        with pytest.raises(ConfigValidationError, match="should be negative"):
            validate_qlearning_reward_weights(w)

    def test_efficiency_threshold_out_of_range(self):
        w = self._default_weights()
        w.efficiency_threshold = 1.5
        with pytest.raises(ConfigValidationError):
            validate_qlearning_reward_weights(w)


class TestValidateSensorConfig:
    def test_default_valid(self):
        cfg = SensorConfig()
        assert validate_sensor_config(cfg) is True

    def test_invalid_eis_update_interval(self):
        cfg = SensorConfig()
        cfg.eis_update_interval = 0.0
        with pytest.raises(ConfigValidationError):
            validate_sensor_config(cfg)

    def test_invalid_qcm_update_interval(self):
        cfg = SensorConfig()
        cfg.qcm_update_interval = -1.0
        with pytest.raises(ConfigValidationError):
            validate_sensor_config(cfg)

    def test_invalid_sensor_timeout(self):
        cfg = SensorConfig()
        cfg.sensor_timeout = 0.0
        with pytest.raises(ConfigValidationError):
            validate_sensor_config(cfg)


class TestValidateEISConfig:
    def test_default_valid(self):
        cfg = EISConfig()
        assert validate_eis_config(cfg) is True

    def test_invalid_frequency_range_inverted(self):
        cfg = EISConfig()
        cfg.frequency_range = (1e6, 100.0)
        with pytest.raises(ConfigValidationError, match="minimum frequency"):
            validate_eis_config(cfg)

    def test_invalid_noise_level(self):
        cfg = EISConfig()
        cfg.noise_level = 2.0
        with pytest.raises(ConfigValidationError):
            validate_eis_config(cfg)


class TestValidateQCMConfig:
    def test_default_valid(self):
        cfg = QCMConfig()
        assert validate_qcm_config(cfg) is True

    def test_invalid_mass_range(self):
        cfg = QCMConfig()
        cfg.mass_range = (1000.0, 0.0)
        with pytest.raises(ConfigValidationError, match="minimum mass"):
            validate_qcm_config(cfg)

    def test_invalid_frequency_shift_range(self):
        cfg = QCMConfig()
        cfg.frequency_shift_range = (500.0, 0.0)
        with pytest.raises(ConfigValidationError, match="minimum frequency shift"):
            validate_qcm_config(cfg)


class TestValidateSensorFusionConfig:
    def test_default_valid(self):
        cfg = SensorFusionConfig()
        assert validate_sensor_fusion_config(cfg) is True

    def test_invalid_kalman_uncertainty(self):
        cfg = SensorFusionConfig()
        cfg.kalman_initial_uncertainty = 0.0
        with pytest.raises(ConfigValidationError):
            validate_sensor_fusion_config(cfg)


class TestValidateAllConfigurations:
    def test_both_none(self):
        result = validate_all_configurations()
        assert result == {}

    def test_qlearning_only(self):
        result = validate_all_configurations(qlearning_config=deepcopy(DEFAULT_QLEARNING_CONFIG))
        assert result["qlearning"] is True

    def test_sensor_only(self):
        result = validate_all_configurations(sensor_config=SensorConfig())
        assert result["sensor"] is True

    def test_both(self):
        result = validate_all_configurations(
            qlearning_config=deepcopy(DEFAULT_QLEARNING_CONFIG),
            sensor_config=SensorConfig(),
        )
        assert result["qlearning"] is True
        assert result["sensor"] is True


class TestQuickValidate:
    def test_qlearning(self):
        assert quick_validate(deepcopy(DEFAULT_QLEARNING_CONFIG)) is True

    def test_sensor(self):
        assert quick_validate(SensorConfig()) is True

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported configuration type"):
            quick_validate("not a config")
