"""
Sensor configuration classes for MFC sensor integration.
Replaces hardcoded values in EIS, QCM, and sensor fusion models.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Union
from enum import Enum


class FusionMethod(Enum):
    """Available sensor fusion methods."""
    KALMAN_FILTER = "kalman_filter"
    WEIGHTED_AVERAGE = "weighted_average"  
    MAXIMUM_LIKELIHOOD = "maximum_likelihood"
    BAYESIAN_FUSION = "bayesian_fusion"


@dataclass
class EISConfig:
    """Electrochemical Impedance Spectroscopy (EIS) sensor configuration."""
    
    # Frequency range and measurement parameters
    frequency_range: Tuple[float, float] = (100.0, 1e6)  # Hz
    n_frequency_points: int = 50  # Number of frequency points
    measurement_amplitude: float = 0.010  # V, measurement amplitude
    
    # Species-specific parameters
    geobacter_base_resistivity: float = 100.0  # Ohm⋅m
    geobacter_capacitance_factor: float = 1e-6  # F/m²
    geobacter_thickness_sensitivity: float = 0.8  # Sensitivity factor
    geobacter_max_thickness: float = 80.0  # μm
    geobacter_conductivity: float = 1e-3  # S/m
    
    shewanella_base_resistivity: float = 500.0  # Ohm⋅m
    shewanella_capacitance_factor: float = 2e-6  # F/m²
    shewanella_thickness_sensitivity: float = 0.6  # Sensitivity factor
    shewanella_max_thickness: float = 60.0  # μm
    shewanella_conductivity: float = 5e-4  # S/m
    
    mixed_base_resistivity: float = 250.0  # Ohm⋅m
    mixed_capacitance_factor: float = 1.5e-6  # F/m²
    mixed_thickness_sensitivity: float = 0.7  # Sensitivity factor
    mixed_max_thickness: float = 70.0  # μm
    mixed_conductivity: float = 7.5e-4  # S/m
    
    # Circuit element parameters
    solution_resistance: float = 50.0  # Ohm
    double_layer_capacitance: float = 50e-6  # F
    biofilm_resistance: float = 1000.0  # Ohm
    biofilm_capacitance: float = 1e-6  # F
    charge_transfer_resistance: float = 500.0  # Ohm
    
    # Measurement noise and drift
    noise_level: float = 0.02  # 2% noise
    drift_rate: float = 0.001  # per measurement
    
    # Calibration parameters by species
    species_calibration: Dict[str, Dict[str, Union[float, Tuple[float, float]]]] = field(default_factory=lambda: {
        'geobacter': {
            'thickness_slope': 150.0,
            'thickness_intercept': 2000.0,
            'max_thickness': 80.0,
            'sensitivity_range': (5.0, 60.0)
        },
        'shewanella': {
            'thickness_slope': 100.0,
            'thickness_intercept': 1500.0,
            'max_thickness': 60.0,
            'sensitivity_range': (3.0, 45.0)
        },
        'mixed': {
            'thickness_slope': 125.0,
            'thickness_intercept': 1750.0,
            'max_thickness': 70.0,
            'sensitivity_range': (4.0, 52.5)
        }
    })
    
    # Model update parameters
    density_normalization_factor: float = 10.0
    thickness_resistance_factor: float = 10.0
    capacitance_thickness_factor: float = 10.0
    minimum_thickness_factor: float = 0.1
    thickness_impact_factor: float = 0.8
    charge_transfer_base_resistance: float = 500.0


@dataclass
class QCMConfig:
    """Quartz Crystal Microbalance (QCM) sensor configuration."""
    
    # Fundamental QCM parameters
    sensitivity_5mhz: float = 17.7  # ng⋅cm⁻²⋅Hz⁻¹ for 5 MHz crystal
    sensitivity_10mhz: float = 4.4  # ng⋅cm⁻²⋅Hz⁻¹ for 10 MHz crystal
    default_sensitivity: float = 17.7  # Default sensitivity
    
    # Biofilm property parameters
    biofilm_density: float = 1.1  # g/cm³
    biofilm_viscosity: float = 0.01  # Pa⋅s
    thickness_limit: float = 1e-6  # m, minimum detectable thickness
    
    # Viscoelastic correction parameters
    viscosity_factor_max: float = 0.3  # Maximum viscosity factor
    viscosity_scaling_factor: float = 10.0  # Viscosity scaling
    viscosity_correction_scaling: float = 5.0  # Correction scaling
    
    # Mass measurement parameters
    mass_range: Tuple[float, float] = (0.0, 1000.0)  # ng/cm²
    frequency_shift_range: Tuple[float, float] = (0.0, 500.0)  # Hz
    dissipation_range: Tuple[float, float] = (0.0, 0.01)  # Dissipation factor
    
    # Electrode configuration
    electrode_area: float = 0.196e-4  # m² (5mm diameter electrode, 0.196 cm²)
    
    # Measurement noise parameters
    frequency_noise: float = 0.1  # Hz, frequency measurement noise
    dissipation_noise: float = 1e-6  # Dissipation measurement noise


@dataclass
class SensorFusionConfig:
    """Multi-sensor fusion configuration."""
    
    # Kalman filter parameters
    kalman_initial_uncertainty: float = 100.0  # Initial state uncertainty
    
    # Process noise covariance (Q matrix)
    process_noise_thickness: float = 0.1  # Thickness process noise
    process_noise_biomass: float = 0.5  # Biomass process noise
    process_noise_conductivity: float = 1e-6  # Conductivity process noise
    process_noise_velocity_thickness: float = 0.01  # Thickness velocity noise
    process_noise_velocity_biomass: float = 0.05  # Biomass velocity noise
    
    # Measurement noise covariance (R matrix)
    measurement_noise_eis_thickness: float = 2.0  # EIS thickness noise
    measurement_noise_qcm_thickness: float = 1.0  # QCM thickness noise
    measurement_noise_conductivity: float = 1e-5  # Conductivity noise
    
    # Calibration parameters
    eis_thickness_slope: float = -125.0  # EIS calibration slope
    eis_thickness_intercept: float = 1750.0  # EIS calibration intercept
    eis_conductivity_factor: float = 1.0  # EIS conductivity factor
    eis_baseline_uncertainty: float = 2.0  # EIS uncertainty
    
    qcm_density_factor: float = 1.0  # QCM density factor
    qcm_viscosity_correction: float = 1.0  # QCM viscosity correction
    qcm_sensitivity_factor: float = 1.0  # QCM sensitivity factor
    qcm_baseline_uncertainty: float = 1.0  # QCM uncertainty
    
    # Reliability parameters
    initial_eis_reliability: float = 1.0  # Initial EIS reliability
    initial_qcm_reliability: float = 1.0  # Initial QCM reliability
    last_calibration_time: float = 0.0  # Last calibration timestamp
    
    # Fusion weights and thresholds
    minimum_sensor_weight: float = 0.1  # Minimum sensor weight
    max_disagreement_threshold: float = 10.0  # Maximum disagreement
    fault_threshold: float = 0.3  # Sensor fault threshold
    
    # Calibration update parameters
    min_calibration_measurements: int = 3  # Minimum measurements for calibration
    eis_error_threshold: float = 5.0  # EIS error threshold for calibration
    qcm_error_threshold: float = 2.0  # QCM error threshold for calibration
    
    # Reliability update parameters
    minimum_agreement_factor: float = 0.1  # Minimum agreement factor
    eis_reliability_decay: float = 0.99  # EIS reliability decay
    eis_reliability_update_weight: float = 0.01  # EIS update weight
    qcm_reliability_decay: float = 0.995  # QCM reliability decay
    qcm_reliability_update_weight: float = 0.005  # QCM update weight
    
    # Uncertainty calculation parameters
    eis_uncertainty_factor: float = 1.0  # EIS uncertainty factor
    eis_uncertainty_normalization: float = 50.0  # EIS normalization
    eis_uncertainty_quadratic_factor: float = 50.0  # EIS quadratic factor
    minimum_eis_reliability: float = 0.1  # Minimum EIS reliability
    
    qcm_uncertainty_factor: float = 1.0  # QCM uncertainty factor
    qcm_uncertainty_normalization: float = 20.0  # QCM normalization
    qcm_uncertainty_linear_factor: float = 20.0  # QCM linear factor
    minimum_qcm_reliability: float = 0.1  # Minimum QCM reliability
    
    # Biomass estimation parameters
    biomass_uncertainty_percentage: float = 0.2  # 20% biomass uncertainty
    conductivity_uncertainty_factor: float = 0.1  # Conductivity uncertainty
    
    # Fusion method parameters
    weight_calculation_epsilon: float = 1e-6  # Small value for weight calculation
    likelihood_range_multiplier: float = 3.0  # Likelihood range multiplier
    theta_range_points: int = 1000  # Number of theta points for Bayesian
    likelihood_threshold: float = 0.5  # Likelihood threshold
    fwhm_to_sigma_conversion: float = 2.355  # FWHM to sigma conversion
    
    # Species-specific density and porosity
    species_properties: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'geobacter': {
            'density': 1.15,  # g/cm³
            'porosity': 0.7
        },
        'shewanella': {
            'density': 1.08,  # g/cm³  
            'porosity': 0.8
        },
        'mixed': {
            'density': 1.12,  # g/cm³
            'porosity': 0.75
        }
    })
    
    volume_conversion_factor: float = 100.0  # Volume conversion factor
    
    # Confidence calculation parameters
    confidence_weight_balance: float = 1.0  # Weight balance factor
    status_scores: Dict[str, float] = field(default_factory=lambda: {
        'good': 1.0,
        'degraded': 0.6,
        'failed': 0.1
    })
    
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        'agreement': 0.5,
        'balance': 0.2,
        'status': 0.3
    })
    
    # Fault detection parameters
    failure_count_divisor: float = 2.0  # Failure count divisor
    poor_agreement_threshold: float = 0.3  # Poor agreement threshold
    low_confidence_threshold: float = 0.4  # Low confidence threshold


@dataclass
class SensorConfig:
    """Master sensor configuration combining all sensor types."""
    
    # Sub-configurations
    eis: EISConfig = field(default_factory=EISConfig)
    qcm: QCMConfig = field(default_factory=QCMConfig)
    fusion: SensorFusionConfig = field(default_factory=SensorFusionConfig)
    
    # Global sensor parameters
    fusion_method: FusionMethod = FusionMethod.KALMAN_FILTER
    enable_eis: bool = True
    enable_qcm: bool = True
    
    # Sensor update intervals
    eis_update_interval: float = 60.0  # seconds
    qcm_update_interval: float = 30.0  # seconds
    fusion_update_interval: float = 10.0  # seconds
    
    # Sensor status thresholds
    sensor_timeout: float = 300.0  # seconds, sensor timeout
    calibration_interval: float = 3600.0  # seconds, 1 hour
    
    # Data logging parameters
    log_sensor_data: bool = True
    log_fusion_results: bool = True
    log_calibration_events: bool = True


# Default configurations for different scenarios
DEFAULT_SENSOR_CONFIG = SensorConfig()

HIGH_ACCURACY_SENSOR_CONFIG = SensorConfig(
    fusion=SensorFusionConfig(
        kalman_initial_uncertainty=50.0,
        process_noise_thickness=0.05,
        process_noise_biomass=0.25,
        measurement_noise_eis_thickness=1.0,
        measurement_noise_qcm_thickness=0.5
    ),
    eis_update_interval=30.0,
    qcm_update_interval=15.0,
    fusion_update_interval=5.0
)

ROBUST_SENSOR_CONFIG = SensorConfig(
    fusion=SensorFusionConfig(
        fault_threshold=0.2,  # More sensitive fault detection
        max_disagreement_threshold=5.0,  # Stricter disagreement limit
        minimum_sensor_weight=0.2  # Higher minimum weight
    )
)