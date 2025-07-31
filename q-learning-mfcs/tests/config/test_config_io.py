"""
Tests for configuration I/O functionality.
"""

import pytest
import tempfile
from pathlib import Path
import json
import yaml

from src.config import (
    QLearningConfig, SensorConfig, 
    FusionMethod, QLearningRewardWeights,
    DEFAULT_QLEARNING_CONFIG, HIGH_ACCURACY_SENSOR_CONFIG
)
from src.config.config_io import (
    save_config, load_config, merge_configs,
    dataclass_to_dict, dict_to_dataclass
)
from src.config.parameter_validation import ConfigValidationError


class TestConfigSaveLoad:
    """Test configuration saving and loading functionality."""
    
    def test_save_load_qlearning_yaml(self):
        """Test saving and loading Q-learning config as YAML."""
        config = DEFAULT_QLEARNING_CONFIG
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            filepath = Path(f.name)
        
        try:
            # Save config
            save_config(config, filepath, format='yaml')
            assert filepath.exists()
            
            # Load config
            loaded_config = load_config(filepath, QLearningConfig)
            
            # Verify key parameters
            assert loaded_config.learning_rate == config.learning_rate
            assert loaded_config.discount_factor == config.discount_factor
            assert loaded_config.epsilon == config.epsilon
            assert loaded_config.reward_weights.power_weight == config.reward_weights.power_weight
            
        finally:
            filepath.unlink()
    
    def test_save_load_qlearning_json(self):
        """Test saving and loading Q-learning config as JSON."""
        config = QLearningConfig(
            learning_rate=0.2,
            discount_factor=0.9,
            epsilon=0.4
        )
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = Path(f.name)
        
        try:
            # Save config
            save_config(config, filepath, format='json')
            assert filepath.exists()
            
            # Load config
            loaded_config = load_config(filepath, QLearningConfig)
            
            # Verify
            assert loaded_config.learning_rate == 0.2
            assert loaded_config.discount_factor == 0.9
            assert loaded_config.epsilon == 0.4
            
        finally:
            filepath.unlink()
    
    def test_save_load_sensor_yaml(self):
        """Test saving and loading sensor config as YAML."""
        config = HIGH_ACCURACY_SENSOR_CONFIG
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            filepath = Path(f.name)
        
        try:
            # Save config
            save_config(config, filepath, format='yaml')
            assert filepath.exists()
            
            # Load config
            loaded_config = load_config(filepath, SensorConfig)
            
            # Verify
            assert loaded_config.eis.frequency_range == config.eis.frequency_range
            assert loaded_config.qcm.sensitivity_5mhz == config.qcm.sensitivity_5mhz
            assert loaded_config.fusion_method == config.fusion_method
            
        finally:
            filepath.unlink()
    
    def test_fusion_method_enum_preservation(self):
        """Test that FusionMethod enum is preserved through save/load."""
        config = SensorConfig()
        config.fusion_method = FusionMethod.WEIGHTED_AVERAGE
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            filepath = Path(f.name)
        
        try:
            save_config(config, filepath)
            loaded_config = load_config(filepath, SensorConfig)
            
            assert loaded_config.fusion_method == FusionMethod.WEIGHTED_AVERAGE
            assert isinstance(loaded_config.fusion_method, FusionMethod)
            
        finally:
            filepath.unlink()
    
    def test_invalid_config_raises_error(self):
        """Test that saving invalid config raises error."""
        config = QLearningConfig(learning_rate=-0.5)  # Invalid
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            filepath = Path(f.name)
        
        try:
            with pytest.raises(ConfigValidationError):
                save_config(config, filepath)
        finally:
            if filepath.exists():
                filepath.unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("/path/that/does/not/exist.yaml", QLearningConfig)
    
    def test_unsupported_format(self):
        """Test unsupported file format."""
        config = QLearningConfig()
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            filepath = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                save_config(config, filepath, format='txt')
        finally:
            filepath.unlink()


class TestConfigMerge:
    """Test configuration merging functionality."""
    
    def test_merge_qlearning_simple(self):
        """Test simple parameter override."""
        base_config = QLearningConfig()
        override = {'learning_rate': 0.5, 'epsilon': 0.7}
        
        merged = merge_configs(base_config, override)
        
        assert merged.learning_rate == 0.5
        assert merged.epsilon == 0.7
        assert merged.discount_factor == base_config.discount_factor
    
    def test_merge_nested_rewards(self):
        """Test merging nested reward weights."""
        base_config = QLearningConfig()
        override = {
            'reward_weights': {
                'power_weight': 20.0,
                'biofilm_weight': 60.0
            }
        }
        
        merged = merge_configs(base_config, override)
        
        assert merged.reward_weights.power_weight == 20.0
        assert merged.reward_weights.biofilm_weight == 60.0
        # Other reward weights should remain unchanged
        assert merged.reward_weights.consumption_weight == base_config.reward_weights.consumption_weight
    
    def test_merge_sensor_config(self):
        """Test merging sensor configurations."""
        base_config = SensorConfig()
        override = {
            'eis': {'noise_level': 0.001},
            'qcm': {'frequency_noise': 0.001},
            'fusion_method': 'weighted_average'
        }
        
        merged = merge_configs(base_config, override)
        
        assert merged.eis.noise_level == 0.001
        assert merged.qcm.frequency_noise == 0.001
        assert merged.fusion_method == FusionMethod.WEIGHTED_AVERAGE
    
    def test_merge_invalid_raises_error(self):
        """Test that merging with invalid values raises error."""
        base_config = QLearningConfig()
        override = {'learning_rate': 2.0}  # Invalid (> 1.0)
        
        with pytest.raises(ConfigValidationError):
            merge_configs(base_config, override)


class TestDataclassConversion:
    """Test dataclass to dict conversion utilities."""
    
    def test_dataclass_to_dict_simple(self):
        """Test simple dataclass conversion."""
        config = QLearningRewardWeights(power_weight=15.0)
        result = dataclass_to_dict(config)
        
        assert isinstance(result, dict)
        assert result['power_weight'] == 15.0
        assert 'biofilm_weight' in result
    
    def test_dataclass_to_dict_nested(self):
        """Test nested dataclass conversion."""
        config = QLearningConfig()
        result = dataclass_to_dict(config)
        
        assert isinstance(result, dict)
        assert isinstance(result['reward_weights'], dict)
        assert result['reward_weights']['power_weight'] == config.reward_weights.power_weight
    
    def test_dict_to_dataclass_qlearning(self):
        """Test dictionary to Q-learning config conversion."""
        data = {
            'learning_rate': 0.3,
            'discount_factor': 0.8,
            'epsilon': 0.5,
            'reward_weights': {
                'power_weight': 25.0,
                'biofilm_weight': 75.0
            }
        }
        
        config = dict_to_dataclass(data, QLearningConfig)
        
        assert config.learning_rate == 0.3
        assert config.discount_factor == 0.8
        assert config.reward_weights.power_weight == 25.0
        assert isinstance(config.reward_weights, QLearningRewardWeights)
    
    def test_dict_to_dataclass_sensor(self):
        """Test dictionary to sensor config conversion."""
        data = {
            'eis': {'noise_level': 0.005},
            'qcm': {'sensitivity_5mhz': 20.0},
            'fusion_method': 'kalman_filter'
        }
        
        config = dict_to_dataclass(data, SensorConfig)
        
        assert config.eis.noise_level == 0.005
        assert config.qcm.sensitivity_5mhz == 20.0
        assert config.fusion_method == FusionMethod.KALMAN_FILTER