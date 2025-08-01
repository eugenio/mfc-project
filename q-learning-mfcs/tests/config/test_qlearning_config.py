"""
Unit tests for Q-learning configuration classes.
Tests parameter validation, default values, and configuration relationships.
"""

import pytest

# Import the configuration classes to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from config.qlearning_config import (
    QLearningConfig, QLearningRewardWeights, StateSpaceConfig,
    DEFAULT_QLEARNING_CONFIG, ENHANCED_QLEARNING_CONFIG,
    CONSERVATIVE_QLEARNING_CONFIG, AGGRESSIVE_QLEARNING_CONFIG
)
from config.parameter_validation import (
    validate_qlearning_config, validate_qlearning_reward_weights,
    ConfigValidationError
)


class TestQLearningRewardWeights:
    """Test suite for QLearningRewardWeights dataclass."""

    def test_default_initialization(self):
        """Test default initialization of reward weights."""
        weights = QLearningRewardWeights()

        # Test Bayesian optimized default values
        assert weights.power_weight == 18.450199366779497
        assert weights.consumption_weight == 5.0
        assert weights.efficiency_weight == 10.45562341841683
        assert weights.biofilm_weight == 76.25366956196135
        assert weights.efficiency_threshold == 0.5
        assert weights.biofilm_penalty < 0  # Should be negative penalty
        assert weights.combined_penalty < 0  # Should be negative penalty

    def test_custom_initialization(self):
        """Test custom initialization with specific values."""
        weights = QLearningRewardWeights(
            power_weight=15.0,
            efficiency_threshold=0.7,
            biofilm_penalty=-75.0
        )

        assert weights.power_weight == 15.0
        assert weights.efficiency_threshold == 0.7
        assert weights.biofilm_penalty == -75.0
        # Other values should be defaults
        assert weights.consumption_weight == 5.0

    def test_validation_passes_for_valid_weights(self):
        """Test that validation passes for valid reward weights."""
        weights = QLearningRewardWeights()
        assert validate_qlearning_reward_weights(weights) is True

    def test_validation_fails_for_negative_positive_weights(self):
        """Test that validation fails for negative positive weights."""
        weights = QLearningRewardWeights(power_weight=-1.0)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_qlearning_reward_weights(weights)
        assert "power_weight" in str(exc_info.value)

    def test_validation_fails_for_positive_penalties(self):
        """Test that validation fails for positive penalty values."""
        weights = QLearningRewardWeights(biofilm_penalty=50.0)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_qlearning_reward_weights(weights)
        assert "biofilm_penalty" in str(exc_info.value)
        assert "negative" in str(exc_info.value)

    def test_validation_fails_for_invalid_threshold(self):
        """Test that validation fails for invalid efficiency threshold."""
        weights = QLearningRewardWeights(efficiency_threshold=1.5)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_qlearning_reward_weights(weights)
        assert "efficiency_threshold" in str(exc_info.value)


class TestQLearningConfig:
    """Test suite for QLearningConfig dataclass."""

    def test_default_initialization(self):
        """Test default initialization of Q-learning config."""
        config = QLearningConfig()

        # Test core Q-learning parameters
        assert config.learning_rate == 0.1
        assert config.discount_factor == 0.95
        assert config.epsilon == 0.3
        assert config.epsilon_decay == 0.9995
        assert config.epsilon_min == 0.01

        # Test enhanced parameters (Bayesian optimized)
        assert config.enhanced_learning_rate == 0.010698320638589937
        assert config.enhanced_discount_factor == 0.9609992557261428
        assert config.enhanced_epsilon == 0.40064536992631566

        # Test state space parameters
        assert config.power_bins == 10
        assert config.biofilm_bins == 10
        assert config.substrate_bins == 10

        # Test sensor parameters
        assert config.sensor_weight == 0.3
        assert config.sensor_confidence_threshold == 0.3
        assert config.exploration_boost_factor == 1.5

    def test_custom_initialization(self):
        """Test custom initialization with specific values."""
        config = QLearningConfig(
            learning_rate=0.2,
            discount_factor=0.9,
            epsilon=0.4,
            power_bins=8,
            sensor_weight=0.5
        )

        assert config.learning_rate == 0.2
        assert config.discount_factor == 0.9
        assert config.epsilon == 0.4
        assert config.power_bins == 8
        assert config.sensor_weight == 0.5
        # Other values should be defaults
        assert config.epsilon_decay == 0.9995

    def test_reward_weights_initialization(self):
        """Test that reward weights are properly initialized."""
        config = QLearningConfig()
        assert isinstance(config.reward_weights, QLearningRewardWeights)
        assert config.reward_weights.power_weight == 18.450199366779497

    def test_action_space_lists(self):
        """Test action space list initialization."""
        config = QLearningConfig()

        # Test flow rate actions
        expected_flow_actions = [-12, -10, -5, -2, -1, 0, 1, 2, 5, 6]
        assert config.flow_rate_actions == expected_flow_actions

        # Test substrate actions
        expected_substrate_actions = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        assert config.substrate_actions == expected_substrate_actions

        # Test unified actions
        expected_unified_actions = [-8, -4, -2, -1, 0, 1, 2, 3, 4]
        assert config.unified_flow_actions == expected_unified_actions

    def test_time_bins_initialization(self):
        """Test time bins list initialization."""
        config = QLearningConfig()
        expected_time_bins = [200, 500, 800, 1000]
        assert config.time_bins == expected_time_bins

    def test_validation_passes_for_valid_config(self):
        """Test that validation passes for valid configuration."""
        config = QLearningConfig()
        assert validate_qlearning_config(config) is True

    def test_validation_fails_for_invalid_learning_rate(self):
        """Test validation fails for invalid learning rate."""
        config = QLearningConfig(learning_rate=0.0)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_qlearning_config(config)
        assert "learning_rate" in str(exc_info.value)

        config = QLearningConfig(learning_rate=1.5)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_qlearning_config(config)
        assert "learning_rate" in str(exc_info.value)

    def test_validation_fails_for_invalid_probability_values(self):
        """Test validation fails for invalid probability values."""
        config = QLearningConfig(discount_factor=1.5)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_qlearning_config(config)
        assert "discount_factor" in str(exc_info.value)

        config = QLearningConfig(epsilon=-0.1)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_qlearning_config(config)
        assert "epsilon" in str(exc_info.value)

    def test_validation_fails_for_negative_bins(self):
        """Test validation fails for negative bin counts."""
        config = QLearningConfig(power_bins=0)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_qlearning_config(config)
        assert "power_bins" in str(exc_info.value)

    def test_validation_fails_for_invalid_epsilon_relationships(self):
        """Test validation fails when epsilon_min >= epsilon."""
        config = QLearningConfig(epsilon=0.2, epsilon_min=0.3)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_qlearning_config(config)
        assert "epsilon_min" in str(exc_info.value)

    def test_validation_fails_for_invalid_flow_rate_bounds(self):
        """Test validation fails when flow_rate_min >= flow_rate_max."""
        config = QLearningConfig(flow_rate_min=1e-4, flow_rate_max=5e-5)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_qlearning_config(config)
        assert "flow_rate_min" in str(exc_info.value)

    def test_validation_fails_for_unbalanced_objective_weights(self):
        """Test validation fails when multi-objective weights don't sum to 1."""
        config = QLearningConfig(
            power_objective_weight=0.5,
            biofilm_health_weight=0.5,
            sensor_agreement_weight=0.5,  # This makes total > 1
            stability_weight=0.1
        )
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_qlearning_config(config)
        assert "multi_objective_weights" in str(exc_info.value)


class TestStateSpaceConfig:
    """Test suite for StateSpaceConfig dataclass."""

    def test_default_initialization(self):
        """Test default initialization of state space config."""
        config = StateSpaceConfig()

        assert config.power_bins == 8
        assert config.power_range == (0.0, 2.0)
        assert config.biofilm_bins == 6
        assert config.biofilm_range == (0.0, 1.0)
        assert config.substrate_bins == 8
        assert config.substrate_range == (0.0, 50.0)
        assert config.sensor_state_bins == 12

    def test_total_states_calculation_without_sensors(self):
        """Test total states calculation without sensors."""
        config = StateSpaceConfig()
        expected_states = config.power_bins * config.biofilm_bins * config.substrate_bins
        assert config.get_total_states(include_sensors=False) == expected_states
        assert config.get_total_states(include_sensors=False) == 8 * 6 * 8  # 384

    def test_total_states_calculation_with_sensors(self):
        """Test total states calculation with sensors."""
        config = StateSpaceConfig()
        base_states = config.power_bins * config.biofilm_bins * config.substrate_bins
        expected_states = base_states * config.sensor_state_bins
        assert config.get_total_states(include_sensors=True) == expected_states
        assert config.get_total_states(include_sensors=True) == 384 * 12  # 4608

    def test_custom_bins_affect_total_states(self):
        """Test that custom bin counts affect total state calculation."""
        config = StateSpaceConfig(
            power_bins=4,
            biofilm_bins=3,
            substrate_bins=5,
            sensor_state_bins=6
        )

        expected_base = 4 * 3 * 5  # 60
        expected_with_sensors = 60 * 6  # 360

        assert config.get_total_states(include_sensors=False) == expected_base
        assert config.get_total_states(include_sensors=True) == expected_with_sensors


class TestPredefinedConfigurations:
    """Test suite for predefined configuration instances."""

    def test_default_config_validation(self):
        """Test that default configuration is valid."""
        config = DEFAULT_QLEARNING_CONFIG
        assert validate_qlearning_config(config) is True

    def test_enhanced_config_validation(self):
        """Test that enhanced configuration is valid."""
        config = ENHANCED_QLEARNING_CONFIG
        assert validate_qlearning_config(config) is True

        # Test specific enhanced values
        assert config.learning_rate == 0.0987
        assert config.discount_factor == 0.9517
        assert config.epsilon == 0.3702
        assert config.sensor_weight == 0.3
        assert config.sensor_confidence_threshold == 0.3

    def test_conservative_config_validation(self):
        """Test that conservative configuration is valid."""
        config = CONSERVATIVE_QLEARNING_CONFIG
        assert validate_qlearning_config(config) is True

        # Test conservative characteristics
        assert config.learning_rate == 0.05  # Lower learning rate
        assert config.discount_factor == 0.99  # Higher discount factor
        assert config.epsilon == 0.1  # Lower exploration
        assert config.epsilon_min == 0.01  # Very low minimum exploration

    def test_aggressive_config_validation(self):
        """Test that aggressive configuration is valid."""
        config = AGGRESSIVE_QLEARNING_CONFIG
        assert validate_qlearning_config(config) is True

        # Test aggressive characteristics
        assert config.learning_rate == 0.3  # Higher learning rate
        assert config.discount_factor == 0.9  # Lower discount factor
        assert config.epsilon == 0.5  # Higher exploration
        assert config.epsilon_min == 0.2  # Higher minimum exploration

    def test_config_differences(self):
        """Test that different configurations have meaningful differences."""
        default_config = DEFAULT_QLEARNING_CONFIG
        enhanced_config = ENHANCED_QLEARNING_CONFIG
        conservative_config = CONSERVATIVE_QLEARNING_CONFIG
        aggressive_config = AGGRESSIVE_QLEARNING_CONFIG

        # Learning rates should be different
        learning_rates = [
            default_config.learning_rate,
            enhanced_config.learning_rate,
            conservative_config.learning_rate,
            aggressive_config.learning_rate
        ]
        assert len(set(learning_rates)) > 1  # At least 2 different values

        # Epsilon values should be different
        epsilons = [
            default_config.epsilon,
            enhanced_config.epsilon,
            conservative_config.epsilon,
            aggressive_config.epsilon
        ]
        assert len(set(epsilons)) > 1  # At least 2 different values


class TestConfigurationEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_zero_bins_validation(self):
        """Test that zero bins raise validation errors."""
        with pytest.raises(ConfigValidationError):
            config = QLearningConfig(power_bins=0)
            validate_qlearning_config(config)

    def test_negative_bins_validation(self):
        """Test that negative bins raise validation errors."""
        with pytest.raises(ConfigValidationError):
            config = QLearningConfig(biofilm_bins=-1)
            validate_qlearning_config(config)

    def test_boundary_probability_values(self):
        """Test boundary probability values (0.0 and 1.0)."""
        # These should be valid
        config = QLearningConfig(
            discount_factor=0.0,
            epsilon=1.0,
            sensor_weight=0.0
        )
        assert validate_qlearning_config(config) is True

        config = QLearningConfig(
            discount_factor=1.0,
            epsilon=0.0,
            sensor_weight=1.0
        )
        assert validate_qlearning_config(config) is True

    def test_learning_rate_boundary_values(self):
        """Test learning rate boundary values."""
        # Learning rate should be > 0, so 0.0 should fail
        with pytest.raises(ConfigValidationError):
            config = QLearningConfig(learning_rate=0.0)
            validate_qlearning_config(config)

        # Learning rate = 1.0 should be valid
        config = QLearningConfig(learning_rate=1.0)
        assert validate_qlearning_config(config) is True

    def test_very_small_positive_values(self):
        """Test very small positive values are accepted."""
        config = QLearningConfig(
            learning_rate=1e-10,
            epsilon_decay=1e-10,
            flow_rate_min=1e-20,
            flow_rate_max=1e-19
        )
        assert validate_qlearning_config(config) is True

    def test_action_space_empty_lists(self):
        """Test behavior with empty action space lists."""
        config = QLearningConfig(
            flow_rate_actions=[],
            substrate_actions=[],
            unified_flow_actions=[]
        )
        # Configuration should still be valid (validation doesn't check list contents)
        assert validate_qlearning_config(config) is True


class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_config_modification_preserves_validity(self):
        """Test that modifying a valid config can preserve validity."""
        config = QLearningConfig()

        # Modify some parameters
        config.learning_rate = 0.05
        config.epsilon = 0.2
        config.power_bins = 12

        # Should still be valid
        assert validate_qlearning_config(config) is True

    def test_config_serialization_roundtrip(self):
        """Test that configuration can be serialized and deserialized."""

        config = QLearningConfig(
            learning_rate=0.15,
            discount_factor=0.92,
            epsilon=0.25
        )

        # Convert to dict (simulate serialization)
        config_dict = {
            'learning_rate': config.learning_rate,
            'discount_factor': config.discount_factor,
            'epsilon': config.epsilon,
            'epsilon_decay': config.epsilon_decay,
            'epsilon_min': config.epsilon_min
        }

        # Create new config from dict (simulate deserialization)
        new_config = QLearningConfig(
            learning_rate=config_dict['learning_rate'],
            discount_factor=config_dict['discount_factor'],
            epsilon=config_dict['epsilon'],
            epsilon_decay=config_dict['epsilon_decay'],
            epsilon_min=config_dict['epsilon_min']
        )

        # Verify values match
        assert new_config.learning_rate == config.learning_rate
        assert new_config.discount_factor == config.discount_factor
        assert new_config.epsilon == config.epsilon

        # Both should be valid
        assert validate_qlearning_config(config) is True
        assert validate_qlearning_config(new_config) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
