"""
Unit tests for sensor configuration classes.
Tests EIS, QCM, and sensor fusion configuration validation.
"""

import pytest

# Import the configuration classes to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from config.sensor_config import (
    SensorConfig, EISConfig, QCMConfig, SensorFusionConfig, FusionMethod,
    DEFAULT_SENSOR_CONFIG, HIGH_ACCURACY_SENSOR_CONFIG, ROBUST_SENSOR_CONFIG
)
from config.parameter_validation import (
    validate_sensor_config, validate_eis_config, validate_qcm_config,
    validate_sensor_fusion_config, ConfigValidationError
)


class TestFusionMethod:
    """Test suite for FusionMethod enum."""
    
    def test_fusion_method_values(self):
        """Test that all fusion methods have correct string values."""
        assert FusionMethod.KALMAN_FILTER.value == "kalman_filter"
        assert FusionMethod.WEIGHTED_AVERAGE.value == "weighted_average"
        assert FusionMethod.MAXIMUM_LIKELIHOOD.value == "maximum_likelihood"
        assert FusionMethod.BAYESIAN_FUSION.value == "bayesian_fusion"
        
    def test_fusion_method_enumeration(self):
        """Test that all expected fusion methods are present."""
        expected_methods = {
            "KALMAN_FILTER", "WEIGHTED_AVERAGE", 
            "MAXIMUM_LIKELIHOOD", "BAYESIAN_FUSION"
        }
        actual_methods = {method.name for method in FusionMethod}
        assert actual_methods == expected_methods


class TestEISConfig:
    """Test suite for EISConfig dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of EIS config."""
        config = EISConfig()
        
        # Test frequency range
        assert config.frequency_range == (100.0, 1e6)
        assert config.n_frequency_points == 50
        assert config.measurement_amplitude == 0.010
        
        # Test species parameters
        assert config.geobacter_base_resistivity == 100.0
        assert config.shewanella_base_resistivity == 500.0
        assert config.mixed_base_resistivity == 250.0
        
        # Test circuit parameters
        assert config.solution_resistance == 50.0
        assert config.double_layer_capacitance == 50e-6
        assert config.biofilm_resistance == 1000.0
        
        # Test noise parameters
        assert config.noise_level == 0.02
        assert config.drift_rate == 0.001
        
    def test_custom_initialization(self):
        """Test custom initialization with specific values."""
        config = EISConfig(
            frequency_range=(50.0, 5e5),
            n_frequency_points=100,
            measurement_amplitude=0.005,
            geobacter_base_resistivity=150.0
        )
        
        assert config.frequency_range == (50.0, 5e5)
        assert config.n_frequency_points == 100
        assert config.measurement_amplitude == 0.005
        assert config.geobacter_base_resistivity == 150.0
        # Other values should be defaults
        assert config.shewanella_base_resistivity == 500.0
        
    def test_species_calibration_dict(self):
        """Test species calibration dictionary structure."""
        config = EISConfig()
        
        # Check that all expected species are present
        expected_species = {'geobacter', 'shewanella', 'mixed'}
        assert set(config.species_calibration.keys()) == expected_species
        
        # Check structure of calibration data
        for species, cal_data in config.species_calibration.items():
            assert 'thickness_slope' in cal_data
            assert 'thickness_intercept' in cal_data
            assert 'max_thickness' in cal_data
            assert 'sensitivity_range' in cal_data
            assert isinstance(cal_data['sensitivity_range'], tuple)
            assert len(cal_data['sensitivity_range']) == 2
            
    def test_validation_passes_for_valid_config(self):
        """Test that validation passes for valid EIS configuration."""
        config = EISConfig()
        assert validate_eis_config(config) is True
        
    def test_validation_fails_for_invalid_frequency_range(self):
        """Test validation fails for invalid frequency range."""
        config = EISConfig(frequency_range=(1e6, 100.0))  # min > max
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_eis_config(config)
        assert "frequency_range" in str(exc_info.value)
        
    def test_validation_fails_for_zero_frequency(self):
        """Test validation fails for zero frequency."""
        config = EISConfig(frequency_range=(0.0, 1e6))
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_eis_config(config)
        assert "frequency_range[0]" in str(exc_info.value)
        
    def test_validation_fails_for_negative_parameters(self):
        """Test validation fails for negative parameters."""
        config = EISConfig(n_frequency_points=-1)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_eis_config(config)
        assert "n_frequency_points" in str(exc_info.value)
        
        config = EISConfig(measurement_amplitude=-0.001)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_eis_config(config)
        assert "measurement_amplitude" in str(exc_info.value)
        
    def test_validation_fails_for_invalid_noise_level(self):
        """Test validation fails for invalid noise level (should be 0-1)."""
        config = EISConfig(noise_level=1.5)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_eis_config(config)
        assert "noise_level" in str(exc_info.value)


class TestQCMConfig:
    """Test suite for QCMConfig dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of QCM config."""
        config = QCMConfig()
        
        # Test sensitivity parameters
        assert config.sensitivity_5mhz == 17.7
        assert config.sensitivity_10mhz == 4.4
        assert config.default_sensitivity == 17.7
        
        # Test biofilm properties
        assert config.biofilm_density == 1.1
        assert config.biofilm_viscosity == 0.01
        assert config.thickness_limit == 1e-6
        
        # Test measurement ranges
        assert config.mass_range == (0.0, 1000.0)
        assert config.frequency_shift_range == (0.0, 500.0)
        assert config.dissipation_range == (0.0, 0.01)
        
        # Test electrode configuration
        assert config.electrode_area == 0.196
        
    def test_custom_initialization(self):
        """Test custom initialization with specific values."""
        config = QCMConfig(
            sensitivity_5mhz=20.0,
            biofilm_density=1.2,
            mass_range=(0.0, 2000.0),
            electrode_area=0.3
        )
        
        assert config.sensitivity_5mhz == 20.0
        assert config.biofilm_density == 1.2
        assert config.mass_range == (0.0, 2000.0)
        assert config.electrode_area == 0.3
        # Other values should be defaults
        assert config.sensitivity_10mhz == 4.4
        
    def test_validation_passes_for_valid_config(self):
        """Test that validation passes for valid QCM configuration."""
        config = QCMConfig()
        assert validate_qcm_config(config) is True
        
    def test_validation_fails_for_negative_sensitivity(self):
        """Test validation fails for negative sensitivity values."""
        config = QCMConfig(sensitivity_5mhz=-1.0)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_qcm_config(config)
        assert "sensitivity_5mhz" in str(exc_info.value)
        
    def test_validation_fails_for_invalid_mass_range(self):
        """Test validation fails for invalid mass range."""
        config = QCMConfig(mass_range=(1000.0, 500.0))  # min > max
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_qcm_config(config)
        assert "mass_range" in str(exc_info.value)
        
    def test_validation_fails_for_invalid_frequency_shift_range(self):
        """Test validation fails for invalid frequency shift range."""
        config = QCMConfig(frequency_shift_range=(500.0, 100.0))  # min > max
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_qcm_config(config)
        assert "frequency_shift_range" in str(exc_info.value)
        
    def test_validation_fails_for_zero_electrode_area(self):
        """Test validation fails for zero electrode area."""
        config = QCMConfig(electrode_area=0.0)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_qcm_config(config)
        assert "electrode_area" in str(exc_info.value)


class TestSensorFusionConfig:
    """Test suite for SensorFusionConfig dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of sensor fusion config."""
        config = SensorFusionConfig()
        
        # Test Kalman filter parameters
        assert config.kalman_initial_uncertainty == 100.0
        assert config.process_noise_thickness == 0.1
        assert config.process_noise_biomass == 0.5
        assert config.measurement_noise_eis_thickness == 2.0
        assert config.measurement_noise_qcm_thickness == 1.0
        
        # Test calibration parameters
        assert config.eis_thickness_slope == -125.0
        assert config.eis_thickness_intercept == 1750.0
        assert config.qcm_density_factor == 1.0
        
        # Test reliability parameters
        assert config.initial_eis_reliability == 1.0
        assert config.initial_qcm_reliability == 1.0
        
        # Test weights and thresholds
        assert config.minimum_sensor_weight == 0.1
        assert config.max_disagreement_threshold == 10.0
        assert config.fault_threshold == 0.3
        
    def test_species_properties_dict(self):
        """Test species properties dictionary structure."""
        config = SensorFusionConfig()
        
        # Check that all expected species are present
        expected_species = {'geobacter', 'shewanella', 'mixed'}
        assert set(config.species_properties.keys()) == expected_species
        
        # Check structure of species data
        for species, props in config.species_properties.items():
            assert 'density' in props
            assert 'porosity' in props
            assert props['density'] > 0
            assert 0 < props['porosity'] < 1
            
    def test_status_scores_dict(self):
        """Test status scores dictionary structure."""
        config = SensorFusionConfig()
        
        expected_statuses = {'good', 'degraded', 'failed'}
        assert set(config.status_scores.keys()) == expected_statuses
        
        # Check that scores are in valid range
        for status, score in config.status_scores.items():
            assert 0 <= score <= 1
            
        # Check relative ordering
        assert config.status_scores['good'] > config.status_scores['degraded']
        assert config.status_scores['degraded'] > config.status_scores['failed']
        
    def test_confidence_weights_dict(self):
        """Test confidence weights dictionary structure."""
        config = SensorFusionConfig()
        
        expected_weights = {'agreement', 'balance', 'status'}
        assert set(config.confidence_weights.keys()) == expected_weights
        
        # Check that weights are positive
        for weight_name, weight_value in config.confidence_weights.items():
            assert weight_value > 0
            
    def test_validation_passes_for_valid_config(self):
        """Test that validation passes for valid sensor fusion configuration."""
        config = SensorFusionConfig()
        assert validate_sensor_fusion_config(config) is True
        
    def test_validation_fails_for_negative_process_noise(self):
        """Test validation fails for negative process noise."""
        config = SensorFusionConfig(process_noise_thickness=-0.1)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_sensor_fusion_config(config)
        assert "process_noise_thickness" in str(exc_info.value)
        
    def test_validation_fails_for_invalid_reliability_values(self):
        """Test validation fails for invalid reliability values."""
        config = SensorFusionConfig(initial_eis_reliability=1.5)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_sensor_fusion_config(config)
        assert "initial_eis_reliability" in str(exc_info.value)
        
        config = SensorFusionConfig(initial_qcm_reliability=-0.1)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_sensor_fusion_config(config)
        assert "initial_qcm_reliability" in str(exc_info.value)
        
    def test_validation_fails_for_invalid_decay_factors(self):
        """Test validation fails for invalid decay factors."""
        config = SensorFusionConfig(eis_reliability_decay=0.0)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_sensor_fusion_config(config)
        assert "eis_reliability_decay" in str(exc_info.value)
        
        config = SensorFusionConfig(qcm_reliability_decay=1.1)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_sensor_fusion_config(config)
        assert "qcm_reliability_decay" in str(exc_info.value)


class TestSensorConfig:
    """Test suite for master SensorConfig dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of sensor config."""
        config = SensorConfig()
        
        # Test sub-configurations
        assert isinstance(config.eis, EISConfig)
        assert isinstance(config.qcm, QCMConfig)
        assert isinstance(config.fusion, SensorFusionConfig)
        
        # Test global parameters
        assert config.fusion_method == FusionMethod.KALMAN_FILTER
        assert config.enable_eis is True
        assert config.enable_qcm is True
        
        # Test update intervals
        assert config.eis_update_interval == 60.0
        assert config.qcm_update_interval == 30.0
        assert config.fusion_update_interval == 10.0
        
        # Test timeouts
        assert config.sensor_timeout == 300.0
        assert config.calibration_interval == 3600.0
        
        # Test logging parameters
        assert config.log_sensor_data is True
        assert config.log_fusion_results is True
        assert config.log_calibration_events is True
        
    def test_custom_initialization(self):
        """Test custom initialization with specific values."""
        custom_eis = EISConfig(n_frequency_points=100)
        custom_qcm = QCMConfig(sensitivity_5mhz=20.0)
        custom_fusion = SensorFusionConfig(kalman_initial_uncertainty=50.0)
        
        config = SensorConfig(
            eis=custom_eis,
            qcm=custom_qcm,
            fusion=custom_fusion,
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
            eis_update_interval=45.0
        )
        
        assert config.eis.n_frequency_points == 100
        assert config.qcm.sensitivity_5mhz == 20.0
        assert config.fusion.kalman_initial_uncertainty == 50.0
        assert config.fusion_method == FusionMethod.WEIGHTED_AVERAGE
        assert config.eis_update_interval == 45.0
        # Other values should be defaults
        assert config.qcm_update_interval == 30.0
        
    def test_validation_passes_for_valid_config(self):
        """Test that validation passes for valid sensor configuration."""
        config = SensorConfig()
        assert validate_sensor_config(config) is True
        
    def test_validation_fails_for_negative_intervals(self):
        """Test validation fails for negative update intervals."""
        config = SensorConfig(eis_update_interval=-1.0)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_sensor_config(config)
        assert "eis_update_interval" in str(exc_info.value)
        
    def test_validation_fails_for_invalid_sub_configs(self):
        """Test validation fails when sub-configurations are invalid."""
        invalid_eis = EISConfig(frequency_range=(1e6, 100.0))  # Invalid range
        config = SensorConfig(eis=invalid_eis)
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_sensor_config(config)
        assert "frequency_range" in str(exc_info.value)


class TestPredefinedSensorConfigurations:
    """Test suite for predefined sensor configuration instances."""
    
    def test_default_sensor_config_validation(self):
        """Test that default sensor configuration is valid."""
        config = DEFAULT_SENSOR_CONFIG
        assert validate_sensor_config(config) is True
        
    def test_high_accuracy_sensor_config_validation(self):
        """Test that high accuracy sensor configuration is valid."""
        config = HIGH_ACCURACY_SENSOR_CONFIG
        assert validate_sensor_config(config) is True
        
        # Test high accuracy characteristics
        assert config.fusion.kalman_initial_uncertainty == 50.0  # Lower uncertainty
        assert config.fusion.process_noise_thickness == 0.05  # Lower noise
        assert config.fusion.measurement_noise_eis_thickness == 1.0  # Lower noise
        assert config.eis_update_interval == 30.0  # Faster updates
        assert config.qcm_update_interval == 15.0  # Faster updates
        assert config.fusion_update_interval == 5.0  # Faster updates
        
    def test_robust_sensor_config_validation(self):
        """Test that robust sensor configuration is valid."""
        config = ROBUST_SENSOR_CONFIG
        assert validate_sensor_config(config) is True
        
        # Test robust characteristics
        assert config.fusion.fault_threshold == 0.2  # More sensitive fault detection
        assert config.fusion.max_disagreement_threshold == 5.0  # Stricter disagreement
        assert config.fusion.minimum_sensor_weight == 0.2  # Higher minimum weight
        
    def test_config_differences(self):
        """Test that different configurations have meaningful differences."""
        default_config = DEFAULT_SENSOR_CONFIG
        high_accuracy_config = HIGH_ACCURACY_SENSOR_CONFIG
        robust_config = ROBUST_SENSOR_CONFIG
        
        # Update intervals should be different
        update_intervals = [
            default_config.fusion_update_interval,
            high_accuracy_config.fusion_update_interval,
            robust_config.fusion_update_interval
        ]
        assert len(set(update_intervals)) >= 2  # At least 2 different values
        
        # Fault thresholds should be different
        fault_thresholds = [
            default_config.fusion.fault_threshold,
            high_accuracy_config.fusion.fault_threshold,
            robust_config.fusion.fault_threshold
        ]
        assert len(set(fault_thresholds)) >= 2  # At least 2 different values


class TestSensorConfigurationEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_zero_update_intervals_validation(self):
        """Test that zero update intervals raise validation errors."""
        with pytest.raises(ConfigValidationError):
            config = SensorConfig(eis_update_interval=0.0)
            validate_sensor_config(config)
            
    def test_boundary_probability_values(self):
        """Test boundary probability values in fusion config."""
        config = SensorConfig(
            fusion=SensorFusionConfig(
                initial_eis_reliability=0.0,
                initial_qcm_reliability=1.0,
                fault_threshold=0.0
            )
        )
        assert validate_sensor_config(config) is True
        
        config = SensorConfig(
            fusion=SensorFusionConfig(
                initial_eis_reliability=1.0,
                initial_qcm_reliability=0.0,
                fault_threshold=1.0
            )
        )
        assert validate_sensor_config(config) is True
        
    def test_very_small_positive_values(self):
        """Test very small positive values are accepted."""
        config = SensorConfig(
            eis_update_interval=1e-10,
            qcm_update_interval=1e-10,
            fusion_update_interval=1e-10,
            sensor_timeout=1e-10
        )
        assert validate_sensor_config(config) is True
        
    def test_extreme_noise_values(self):
        """Test extreme but valid noise values."""
        config = SensorConfig(
            eis=EISConfig(noise_level=0.0),  # No noise
            fusion=SensorFusionConfig(
                process_noise_thickness=1e-10,  # Very low noise
                measurement_noise_eis_thickness=1e10  # Very high noise
            )
        )
        assert validate_sensor_config(config) is True


class TestSensorConfigurationIntegration:
    """Integration tests for sensor configuration system."""
    
    def test_config_modification_preserves_validity(self):
        """Test that modifying a valid config can preserve validity."""
        config = SensorConfig()
        
        # Modify some parameters
        config.eis_update_interval = 120.0
        config.fusion.fault_threshold = 0.5
        config.eis.n_frequency_points = 25
        
        # Should still be valid
        assert validate_sensor_config(config) is True
        
    def test_nested_config_validation(self):
        """Test that validation properly checks nested configurations."""
        # Create valid config
        config = SensorConfig()
        assert validate_sensor_config(config) is True
        
        # Make nested config invalid
        config.eis.frequency_range = (1e6, 100.0)  # Invalid range
        
        # Should now fail validation
        with pytest.raises(ConfigValidationError):
            validate_sensor_config(config)
            
    def test_fusion_method_enum_integration(self):
        """Test that fusion method enum integrates properly."""
        for method in FusionMethod:
            config = SensorConfig(fusion_method=method)
            assert validate_sensor_config(config) is True
            assert config.fusion_method == method


if __name__ == "__main__":
    pytest.main([__file__, "-v"])