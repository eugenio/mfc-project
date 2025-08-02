"""
Sensor Fusion Model for EIS and QCM Integration

This module implements sensor fusion algorithms to combine EIS and QCM
measurements for improved biofilm characterization accuracy and robustness.

Key Features:
- Kalman filter for state estimation
- Cross-validation between EIS and QCM measurements
- Adaptive calibration based on sensor agreement
- Real-time biofilm property estimation
- Uncertainty quantification

Implementation includes:
- Multi-sensor state estimation
- Sensor reliability assessment
- Adaptive fusion weights
- Biofilm property reconciliation
- Real-time calibration updates
"""

import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

# Import sensing models
from .eis_model import BacterialSpecies, EISMeasurement
from .qcm_model import QCMMeasurement

# Import configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from config import FusionMethod as ConfigFusionMethod
    from config import SensorConfig
except ImportError:
    SensorConfig = None
    ConfigFusionMethod = None

# Add GPU acceleration
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from gpu_acceleration import get_gpu_accelerator
except ImportError:
    get_gpu_accelerator = None


# Use configuration FusionMethod if available, otherwise define locally
if ConfigFusionMethod is not None:
    FusionMethod = ConfigFusionMethod
else:
    class FusionMethod(Enum):
        """Sensor fusion methods."""
        KALMAN_FILTER = "kalman_filter"
        WEIGHTED_AVERAGE = "weighted_average"
        MAXIMUM_LIKELIHOOD = "maximum_likelihood"
        BAYESIAN_FUSION = "bayesian_fusion"


@dataclass
class FusedMeasurement:
    """Container for fused sensor measurement data."""

    timestamp: float  # hours

    # Biofilm properties with uncertainties
    thickness_um: float
    thickness_uncertainty: float

    biomass_density_g_per_L: float
    biomass_uncertainty: float

    conductivity_S_per_m: float
    conductivity_uncertainty: float

    # Individual sensor contributions
    eis_thickness: float
    qcm_thickness: float

    eis_weight: float
    qcm_weight: float

    # Quality metrics
    sensor_agreement: float  # 0-1, higher is better
    fusion_confidence: float  # 0-1, overall confidence
    cross_validation_error: float  # Residual between sensors

    # Sensor status
    eis_status: str  # "good", "degraded", "failed"
    qcm_status: str  # "good", "degraded", "failed"


class KalmanFilter:
    """
    Kalman filter for biofilm state estimation using EIS and QCM sensors.

    State vector: [thickness, biomass_density, conductivity, velocity_thickness, velocity_biomass]

    Measurement vector: [eis_thickness, qcm_thickness, eis_conductivity]
    """

    def __init__(self, dt: float = 0.1, config: SensorConfig | None = None):
        """
        Initialize Kalman filter.

        Args:
            dt: Time step (hours)
            config: Optional sensor configuration
        """
        self.dt = dt
        self.config = config
        self.n_states = 5  # [thickness, biomass, conductivity, d_thickness/dt, d_biomass/dt]
        self.n_measurements = 3  # [eis_thickness, qcm_thickness, eis_conductivity]

        # Initialize state vector and covariance
        self.state = np.zeros(self.n_states)
        self.covariance = np.eye(self.n_states) * 100.0  # Initial uncertainty

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, dt, 0],    # thickness
            [0, 1, 0, 0, dt],    # biomass_density
            [0, 0, 1, 0, 0],     # conductivity (slowly varying)
            [0, 0, 0, 1, 0],     # d_thickness/dt
            [0, 0, 0, 0, 1]      # d_biomass/dt
        ])

        # Process noise covariance from configuration or defaults
        if self.config and self.config.fusion:
            process_noise = self.config.fusion.process_noise_covariance
            measurement_noise = self.config.fusion.measurement_noise_covariance
            self.Q = np.diag(process_noise)
        else:
            self.Q = np.diag([0.1, 0.5, 1e-6, 0.01, 0.05])  # Tuned for biofilm dynamics

        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0],  # EIS thickness measurement
            [1, 0, 0, 0, 0],  # QCM thickness measurement
            [0, 0, 1, 0, 0]   # EIS conductivity measurement
        ])

        # Measurement noise covariance (will be updated adaptively)
        if self.config and self.config.fusion:
            measurement_noise = self.config.fusion.measurement_noise_covariance[:3]  # Take first 3 values
            self.R = np.diag(measurement_noise)
        else:
            self.R = np.diag([2.0, 1.0, 1e-5])  # [eis_thick, qcm_thick, conductivity]

        # Filter status
        self.initialized = False
        self.innovation_history = []
        self.residual_history = []

    def initialize_state(self, initial_thickness: float, initial_biomass: float,
                        initial_conductivity: float):
        """Initialize filter state with first measurements."""
        self.state = np.array([
            initial_thickness,    # thickness
            initial_biomass,      # biomass_density
            initial_conductivity, # conductivity
            0.0,                  # d_thickness/dt
            0.0                   # d_biomass/dt
        ])
        self.initialized = True

    def predict(self):
        """Prediction step of Kalman filter."""
        # Predict state
        self.state = np.dot(self.F, self.state)

        # Predict covariance
        self.covariance = np.dot(np.dot(self.F, self.covariance), self.F.T) + self.Q

        # Ensure non-negative biofilm properties
        self.state[0] = max(0, self.state[0])  # thickness
        self.state[1] = max(0, self.state[1])  # biomass
        self.state[2] = max(0, self.state[2])  # conductivity

    def update(self, measurements: np.ndarray, measurement_uncertainties: np.ndarray):
        """
        Update step of Kalman filter.

        Args:
            measurements: [eis_thickness, qcm_thickness, eis_conductivity]
            measurement_uncertainties: Uncertainties for each measurement
        """
        # Update measurement noise covariance
        self.R = np.diag(measurement_uncertainties**2)

        # Innovation (measurement residual)
        innovation = measurements - np.dot(self.H, self.state)
        self.innovation_history.append(innovation.copy())

        # Innovation covariance
        S = np.dot(np.dot(self.H, self.covariance), self.H.T) + self.R

        # Kalman gain
        K = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(S))

        # Update state
        self.state = self.state + np.dot(K, innovation)

        # Update covariance
        identity_matrix = np.eye(self.n_states)
        self.covariance = np.dot((identity_matrix - np.dot(K, self.H)), self.covariance)

        # Ensure non-negative values
        self.state[0] = max(0, self.state[0])  # thickness
        self.state[1] = max(0, self.state[1])  # biomass
        self.state[2] = max(0, self.state[2])  # conductivity

        # Store residuals for validation
        post_fit_residual = measurements - np.dot(self.H, self.state)
        self.residual_history.append(post_fit_residual.copy())

    def get_state_estimate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get current state estimate and uncertainties.

        Returns:
            Tuple of (state_vector, state_uncertainties)
        """
        state_uncertainties = np.sqrt(np.diag(self.covariance))
        return self.state.copy(), state_uncertainties

    def assess_filter_performance(self) -> dict[str, float]:
        """Assess Kalman filter performance metrics."""
        if len(self.innovation_history) < 5:
            return {'insufficient_data': True}

        # Recent innovation statistics
        recent_innovations = np.array(self.innovation_history[-10:])

        # Normalized innovation squared (should follow chi-squared distribution)
        nis_values = []
        for innovation in recent_innovations:
            try:
                S_inv = np.linalg.inv(np.dot(np.dot(self.H, self.covariance), self.H.T) + self.R)
                nis = np.dot(np.dot(innovation.T, S_inv), innovation)
                nis_values.append(nis)
            except np.linalg.LinAlgError:
                continue

        if not nis_values:
            return {'numerical_issues': True}

        # Chi-squared test for filter consistency
        mean_nis = np.mean(nis_values)
        expected_nis = self.n_measurements  # Expected value for chi-squared

        # Filter consistency (should be close to 1.0)
        consistency = expected_nis / mean_nis if mean_nis > 0 else 0

        return {
            'mean_normalized_innovation': mean_nis,
            'expected_innovation': expected_nis,
            'filter_consistency': consistency,
            'innovation_variance': np.var(nis_values),
            'sample_count': len(nis_values)
        }


class SensorCalibration:
    """
    Adaptive calibration system for EIS and QCM sensors.

    Maintains calibration parameters and updates them based on
    sensor agreement and external validation data.
    """

    def __init__(self, config: SensorConfig | None = None):
        """Initialize calibration system."""
        self.config = config

        # EIS calibration parameters from config or defaults
        if self.config and self.config.eis:
            species_cal = self.config.eis.species_calibration.get('mixed', {})
            self.eis_calibration = {
                'thickness_slope': species_cal.get('thickness_slope', -125.0),
                'thickness_intercept': species_cal.get('thickness_intercept', 1750.0),
                'conductivity_factor': 1.0,  # Scaling factor
                'uncertainty_model': 'linear',  # 'linear' or 'quadratic'
                'baseline_uncertainty': self.config.eis.baseline_uncertainty,
            }
        else:
            self.eis_calibration = {
                'thickness_slope': -125.0,  # Ohm per μm
                'thickness_intercept': 1750.0,  # Ohm
                'conductivity_factor': 1.0,  # Scaling factor
                'uncertainty_model': 'linear',  # 'linear' or 'quadratic'
                'baseline_uncertainty': 2.0,  # μm
            }

        # QCM calibration parameters from config or defaults
        if self.config and self.config.qcm:
            self.qcm_calibration = {
                'density_factor': self.config.qcm.biofilm_density,
                'viscosity_correction': 1.0,  # Viscoelastic correction
                'sensitivity_factor': self.config.qcm.mass_sensitivity_factor,
                'uncertainty_model': 'sqrt',  # 'linear' or 'sqrt'
                'baseline_uncertainty': self.config.qcm.baseline_uncertainty,
            }
        else:
            self.qcm_calibration = {
                'density_factor': 1.0,  # Biofilm density correction
                'viscosity_correction': 1.0,  # Viscoelastic correction
                'sensitivity_factor': 1.0,  # Mass sensitivity correction
                'uncertainty_model': 'sqrt',  # 'linear' or 'sqrt'
                'baseline_uncertainty': 1.0,  # μm
            }

        # Cross-calibration data
        self.calibration_history = []
        self.agreement_history = []

        # Calibration quality metrics from config or defaults
        if self.config and self.config.fusion:
            self.eis_reliability = self.config.fusion.eis_reliability
            self.qcm_reliability = self.config.fusion.qcm_reliability
        else:
            self.eis_reliability = 1.0  # 0-1
            self.qcm_reliability = 1.0  # 0-1
        self.last_calibration_time = 0.0

    def update_calibration(self, eis_measurements: list[EISMeasurement],
                          qcm_measurements: list[QCMMeasurement],
                          reference_thickness: float | None = None):
        """
        Update calibration parameters based on sensor agreement.

        Args:
            eis_measurements: Recent EIS measurements
            qcm_measurements: Recent QCM measurements
            reference_thickness: External reference measurement (optional)
        """
        if len(eis_measurements) < 3 or len(qcm_measurements) < 3:
            return

        # Extract thickness estimates
        eis_thicknesses = [m.impedance_magnitude for m in eis_measurements]
        qcm_thicknesses = [m.thickness_estimate for m in qcm_measurements]

        # Calculate current agreement
        if len(eis_thicknesses) == len(qcm_thicknesses):
            agreements = []
            for eis_thick, qcm_thick in zip(eis_thicknesses, qcm_thicknesses, strict=False):
                if max(eis_thick, qcm_thick) > 0:
                    agreement = 1.0 - abs(eis_thick - qcm_thick) / max(eis_thick, qcm_thick)
                    agreements.append(max(0, agreement))

            current_agreement = np.mean(agreements) if agreements else 0.5
            self.agreement_history.append(current_agreement)

        # Update calibration if reference is available
        if reference_thickness is not None:
            self._update_with_reference(eis_measurements, qcm_measurements, reference_thickness)

        # Update reliability estimates
        self._update_reliability()

    def _update_with_reference(self, eis_measurements: list[EISMeasurement],
                             qcm_measurements: list[QCMMeasurement],
                             reference_thickness: float):
        """Update calibration using external reference."""
        # Calculate current sensor estimates
        eis_estimate = np.mean([m.impedance_magnitude for m in eis_measurements])
        qcm_estimate = np.mean([m.thickness_estimate for m in qcm_measurements])

        # Update EIS calibration
        if abs(eis_estimate - reference_thickness) > 5.0:  # μm threshold
            error_ratio = reference_thickness / eis_estimate if eis_estimate > 0 else 1.0
            self.eis_calibration['thickness_slope'] *= error_ratio

        # Update QCM calibration
        if abs(qcm_estimate - reference_thickness) > 2.0:  # μm threshold
            error_ratio = reference_thickness / qcm_estimate if qcm_estimate > 0 else 1.0
            self.qcm_calibration['density_factor'] *= error_ratio

        # Store calibration event
        self.calibration_history.append({
            'timestamp': len(self.calibration_history),
            'reference_thickness': reference_thickness,
            'eis_estimate': eis_estimate,
            'qcm_estimate': qcm_estimate,
            'eis_correction': error_ratio if 'error_ratio' in locals() else 1.0,
            'qcm_correction': error_ratio if abs(qcm_estimate - reference_thickness) > 2.0 else 1.0
        })

    def _update_reliability(self):
        """Update sensor reliability estimates."""
        if len(self.agreement_history) < 5:
            return

        # Recent agreement trend
        recent_agreement = np.mean(self.agreement_history[-10:])

        # Update reliability based on agreement
        agreement_factor = max(0.1, recent_agreement)

        # EIS reliability (affected by electrode fouling)
        self.eis_reliability = min(1.0, self.eis_reliability * 0.99 + agreement_factor * 0.01)

        # QCM reliability (affected by crystal aging)
        self.qcm_reliability = min(1.0, self.qcm_reliability * 0.995 + agreement_factor * 0.005)

    def get_measurement_uncertainty(self, sensor_type: str, measured_value: float) -> float:
        """
        Calculate measurement uncertainty for given sensor and value.

        Args:
            sensor_type: 'eis' or 'qcm'
            measured_value: Measured thickness value

        Returns:
            Estimated uncertainty (same units as measured_value)
        """
        if sensor_type.lower() == 'eis':
            base_uncertainty = self.eis_calibration['baseline_uncertainty']
            if self.eis_calibration['uncertainty_model'] == 'linear':
                uncertainty = base_uncertainty * (1.0 + measured_value / 50.0)
            else:  # quadratic
                uncertainty = base_uncertainty * (1.0 + (measured_value / 50.0)**2)

            # Scale by reliability
            uncertainty /= max(0.1, self.eis_reliability)

        else:  # QCM
            base_uncertainty = self.qcm_calibration['baseline_uncertainty']
            if self.qcm_calibration['uncertainty_model'] == 'sqrt':
                uncertainty = base_uncertainty * np.sqrt(1.0 + measured_value / 20.0)
            else:  # linear
                uncertainty = base_uncertainty * (1.0 + measured_value / 20.0)

            # Scale by reliability
            uncertainty /= max(0.1, self.qcm_reliability)

        return uncertainty

    def get_calibration_status(self) -> dict[str, Any]:
        """Get current calibration status."""
        return {
            'eis_reliability': self.eis_reliability,
            'qcm_reliability': self.qcm_reliability,
            'recent_agreement': np.mean(self.agreement_history[-5:]) if self.agreement_history else 0.5,
            'calibration_events': len(self.calibration_history),
            'eis_calibration': self.eis_calibration.copy(),
            'qcm_calibration': self.qcm_calibration.copy(),
            'needs_calibration': (self.eis_reliability < 0.8 or self.qcm_reliability < 0.8)
        }


class SensorFusion:
    """
    Complete sensor fusion system for EIS and QCM integration.

    Features:
    - Multiple fusion algorithms
    - Adaptive sensor weighting
    - Real-time calibration updates
    - Uncertainty quantification
    - Fault detection and isolation
    """

    def __init__(self, method: FusionMethod = FusionMethod.KALMAN_FILTER,
                 species: BacterialSpecies = BacterialSpecies.MIXED,
                 use_gpu: bool = True,
                 config: SensorConfig | None = None):
        """
        Initialize sensor fusion system.

        Args:
            method: Fusion algorithm to use
            species: Bacterial species for calibration
            use_gpu: Enable GPU acceleration
            config: Optional sensor configuration
        """
        self.method = method
        self.species = species
        self.use_gpu = use_gpu
        self.config = config

        # Initialize GPU if available
        self.gpu_acc = None
        if use_gpu and get_gpu_accelerator:
            self.gpu_acc = get_gpu_accelerator()
            self.gpu_available = self.gpu_acc.is_gpu_available()
        else:
            self.gpu_available = False

        # Initialize components
        self.kalman_filter = KalmanFilter(config=self.config) if method == FusionMethod.KALMAN_FILTER else None
        self.calibration = SensorCalibration(config=self.config)

        # Fusion parameters from configuration or defaults
        if self.config and self.config.fusion:
            fusion_config = self.config.fusion
            self.min_sensor_weight = fusion_config.min_sensor_weight
            self.max_disagreement = fusion_config.max_sensor_disagreement
            self.fault_threshold = fusion_config.sensor_fault_threshold
        else:
            self.min_sensor_weight = 0.1  # Minimum weight for any sensor
            self.max_disagreement = 10.0  # μm, maximum acceptable disagreement
            self.fault_threshold = 0.3   # Reliability threshold for fault detection

        # Historical data
        self.fusion_history = []
        self.sensor_weights_history = []
        self.agreement_history = []

        # Current state
        self.current_thickness = 0.0
        self.current_biomass = 0.0
        self.current_conductivity = 0.0
        self.measurement_time = 0.0

    def fuse_measurements(self, eis_measurement: EISMeasurement,
                         qcm_measurement: QCMMeasurement,
                         eis_properties: dict[str, float],
                         qcm_properties: dict[str, float],
                         time_hours: float = 0.0) -> FusedMeasurement:
        """
        Fuse EIS and QCM measurements into unified biofilm state estimate.

        Args:
            eis_measurement: EIS measurement data
            qcm_measurement: QCM measurement data
            eis_properties: Biofilm properties from EIS
            qcm_properties: Biofilm properties from QCM
            time_hours: Measurement timestamp

        Returns:
            Fused measurement with uncertainties
        """
        # Extract individual estimates
        eis_thickness = eis_properties.get('thickness_um', 0.0)
        qcm_thickness = qcm_properties.get('thickness_um', 0.0)
        eis_conductivity = eis_properties.get('conductivity_S_per_m', 0.0)

        # Calculate measurement uncertainties
        eis_uncertainty = self.calibration.get_measurement_uncertainty('eis', eis_thickness)
        qcm_uncertainty = self.calibration.get_measurement_uncertainty('qcm', qcm_thickness)

        # Assess sensor status
        eis_status = self._assess_sensor_status('eis', eis_measurement, eis_properties)
        qcm_status = self._assess_sensor_status('qcm', qcm_measurement, qcm_properties)

        # Calculate sensor agreement
        thickness_agreement = self._calculate_agreement(eis_thickness, qcm_thickness)

        # Apply fusion method
        if self.method == FusionMethod.KALMAN_FILTER:
            fused_result = self._kalman_fusion(
                eis_thickness, qcm_thickness, eis_conductivity,
                eis_uncertainty, qcm_uncertainty, time_hours
            )
        elif self.method == FusionMethod.WEIGHTED_AVERAGE:
            fused_result = self._weighted_average_fusion(
                eis_thickness, qcm_thickness, eis_conductivity,
                eis_uncertainty, qcm_uncertainty
            )
        elif self.method == FusionMethod.MAXIMUM_LIKELIHOOD:
            fused_result = self._maximum_likelihood_fusion(
                eis_thickness, qcm_thickness, eis_conductivity,
                eis_uncertainty, qcm_uncertainty
            )
        else:  # Bayesian inference
            fused_result = self._bayesian_fusion(
                eis_thickness, qcm_thickness, eis_conductivity,
                eis_uncertainty, qcm_uncertainty
            )

        # Extract fused values
        fused_thickness = fused_result['thickness']
        fused_thickness_uncertainty = fused_result['thickness_uncertainty']
        fused_conductivity = fused_result['conductivity']
        fused_conductivity_uncertainty = fused_result['conductivity_uncertainty']
        eis_weight = fused_result['eis_weight']
        qcm_weight = fused_result['qcm_weight']

        # Estimate biomass from fused thickness and species properties
        fused_biomass = self._estimate_biomass_from_thickness(fused_thickness)
        fused_biomass_uncertainty = fused_biomass * 0.2  # 20% relative uncertainty

        # Calculate overall confidence
        fusion_confidence = self._calculate_fusion_confidence(
            thickness_agreement, eis_weight, qcm_weight, eis_status, qcm_status
        )

        # Calculate cross-validation error
        cross_validation_error = abs(eis_thickness - qcm_thickness)

        # Create fused measurement
        fused_measurement = FusedMeasurement(
            timestamp=time_hours,
            thickness_um=fused_thickness,
            thickness_uncertainty=fused_thickness_uncertainty,
            biomass_density_g_per_L=fused_biomass,
            biomass_uncertainty=fused_biomass_uncertainty,
            conductivity_S_per_m=fused_conductivity,
            conductivity_uncertainty=fused_conductivity_uncertainty,
            eis_thickness=eis_thickness,
            qcm_thickness=qcm_thickness,
            eis_weight=eis_weight,
            qcm_weight=qcm_weight,
            sensor_agreement=thickness_agreement,
            fusion_confidence=fusion_confidence,
            cross_validation_error=cross_validation_error,
            eis_status=eis_status,
            qcm_status=qcm_status
        )

        # Update historical data
        self.fusion_history.append(fused_measurement)
        self.sensor_weights_history.append((eis_weight, qcm_weight))
        self.agreement_history.append(thickness_agreement)

        # Update current state
        self.current_thickness = fused_thickness
        self.current_biomass = fused_biomass
        self.current_conductivity = fused_conductivity
        self.measurement_time = time_hours

        # Update calibration
        self.calibration.update_calibration([eis_measurement], [qcm_measurement])

        return fused_measurement

    def _kalman_fusion(self, eis_thickness: float, qcm_thickness: float,
                      eis_conductivity: float, eis_uncertainty: float,
                      qcm_uncertainty: float, time_hours: float) -> dict[str, float]:
        """Apply Kalman filter fusion."""
        if not self.kalman_filter.initialized:
            # Initialize with first measurements
            biomass_estimate = self._estimate_biomass_from_thickness((eis_thickness + qcm_thickness) / 2)
            self.kalman_filter.initialize_state(
                (eis_thickness + qcm_thickness) / 2,
                biomass_estimate,
                eis_conductivity
            )

        # Prediction step
        self.kalman_filter.predict()

        # Update step with measurements
        measurements = np.array([eis_thickness, qcm_thickness, eis_conductivity])
        uncertainties = np.array([eis_uncertainty, qcm_uncertainty, eis_conductivity * 0.1])

        self.kalman_filter.update(measurements, uncertainties)

        # Get fused estimates
        state, state_uncertainties = self.kalman_filter.get_state_estimate()

        # Calculate sensor weights based on uncertainties
        total_thickness_uncertainty = eis_uncertainty + qcm_uncertainty
        eis_weight = qcm_uncertainty / total_thickness_uncertainty if total_thickness_uncertainty > 0 else 0.5
        qcm_weight = eis_uncertainty / total_thickness_uncertainty if total_thickness_uncertainty > 0 else 0.5

        return {
            'thickness': state[0],
            'thickness_uncertainty': state_uncertainties[0],
            'conductivity': state[2],
            'conductivity_uncertainty': state_uncertainties[2],
            'eis_weight': eis_weight,
            'qcm_weight': qcm_weight
        }

    def _weighted_average_fusion(self, eis_thickness: float, qcm_thickness: float,
                               eis_conductivity: float, eis_uncertainty: float,
                               qcm_uncertainty: float) -> dict[str, float]:
        """Apply weighted average fusion."""
        # Calculate weights inversely proportional to uncertainties
        eis_weight = 1.0 / (eis_uncertainty**2 + 1e-6)
        qcm_weight = 1.0 / (qcm_uncertainty**2 + 1e-6)

        # Normalize weights
        total_weight = eis_weight + qcm_weight
        eis_weight /= total_weight
        qcm_weight /= total_weight

        # Apply reliability factors
        eis_weight *= self.calibration.eis_reliability
        qcm_weight *= self.calibration.qcm_reliability

        # Renormalize
        total_weight = eis_weight + qcm_weight
        if total_weight > 0:
            eis_weight /= total_weight
            qcm_weight /= total_weight
        else:
            eis_weight = qcm_weight = 0.5

        # Ensure minimum weights
        eis_weight = max(self.min_sensor_weight, eis_weight)
        qcm_weight = max(self.min_sensor_weight, qcm_weight)

        # Renormalize final weights
        total_weight = eis_weight + qcm_weight
        eis_weight /= total_weight
        qcm_weight /= total_weight

        # Calculate fused thickness
        fused_thickness = eis_weight * eis_thickness + qcm_weight * qcm_thickness

        # Calculate fused uncertainty
        fused_uncertainty = np.sqrt(
            (eis_weight * eis_uncertainty)**2 + (qcm_weight * qcm_uncertainty)**2
        )

        return {
            'thickness': fused_thickness,
            'thickness_uncertainty': fused_uncertainty,
            'conductivity': eis_conductivity,
            'conductivity_uncertainty': eis_conductivity * 0.1,
            'eis_weight': eis_weight,
            'qcm_weight': qcm_weight
        }

    def _maximum_likelihood_fusion(self, eis_thickness: float, qcm_thickness: float,
                                 eis_conductivity: float, eis_uncertainty: float,
                                 qcm_uncertainty: float) -> dict[str, float]:
        """Apply maximum likelihood fusion."""
        # Maximum likelihood estimate for Gaussian distributions
        # θ_ML = (σ₂²θ₁ + σ₁²θ₂) / (σ₁² + σ₂²)

        var_eis = eis_uncertainty**2
        var_qcm = qcm_uncertainty**2

        # MLE thickness estimate
        fused_thickness = (var_qcm * eis_thickness + var_eis * qcm_thickness) / (var_eis + var_qcm)

        # MLE uncertainty
        fused_uncertainty = np.sqrt((var_eis * var_qcm) / (var_eis + var_qcm))

        # Calculate effective weights
        eis_weight = var_qcm / (var_eis + var_qcm)
        qcm_weight = var_eis / (var_eis + var_qcm)

        return {
            'thickness': fused_thickness,
            'thickness_uncertainty': fused_uncertainty,
            'conductivity': eis_conductivity,
            'conductivity_uncertainty': eis_conductivity * 0.1,
            'eis_weight': eis_weight,
            'qcm_weight': qcm_weight
        }

    def _bayesian_fusion(self, eis_thickness: float, qcm_thickness: float,
                        eis_conductivity: float, eis_uncertainty: float,
                        qcm_uncertainty: float) -> dict[str, float]:
        """Apply Bayesian inference fusion."""
        # Simplified Bayesian fusion with uniform prior
        # Posterior is proportional to product of likelihoods

        # Define likelihood functions (Gaussian)
        def eis_likelihood(theta):
            return np.exp(-0.5 * ((theta - eis_thickness) / eis_uncertainty)**2)

        def qcm_likelihood(theta):
            return np.exp(-0.5 * ((theta - qcm_thickness) / qcm_uncertainty)**2)

        # Combined likelihood
        def combined_likelihood(theta):
            return eis_likelihood(theta) * qcm_likelihood(theta)

        # Find MAP estimate (maximum a posteriori)
        theta_range = np.linspace(
            min(eis_thickness, qcm_thickness) - 3 * max(eis_uncertainty, qcm_uncertainty),
            max(eis_thickness, qcm_thickness) + 3 * max(eis_uncertainty, qcm_uncertainty),
            1000
        )

        likelihoods = [combined_likelihood(theta) for theta in theta_range]
        map_idx = np.argmax(likelihoods)
        fused_thickness = theta_range[map_idx]

        # Estimate uncertainty from likelihood width
        likelihood_normalized = np.array(likelihoods) / max(likelihoods)
        above_half_max = likelihood_normalized > 0.5

        if np.any(above_half_max):
            width_indices = np.where(above_half_max)[0]
            width = theta_range[width_indices[-1]] - theta_range[width_indices[0]]
            fused_uncertainty = width / 2.355  # FWHM to sigma conversion
        else:
            fused_uncertainty = min(eis_uncertainty, qcm_uncertainty)

        # Calculate effective weights based on relative likelihoods
        eis_likelihood_val = eis_likelihood(fused_thickness)
        qcm_likelihood_val = qcm_likelihood(fused_thickness)
        total_likelihood = eis_likelihood_val + qcm_likelihood_val

        if total_likelihood > 0:
            eis_weight = eis_likelihood_val / total_likelihood
            qcm_weight = qcm_likelihood_val / total_likelihood
        else:
            eis_weight = qcm_weight = 0.5

        return {
            'thickness': fused_thickness,
            'thickness_uncertainty': fused_uncertainty,
            'conductivity': eis_conductivity,
            'conductivity_uncertainty': eis_conductivity * 0.1,
            'eis_weight': eis_weight,
            'qcm_weight': qcm_weight
        }

    def _assess_sensor_status(self, sensor_type: str, measurement: EISMeasurement | QCMMeasurement,
                            properties: dict[str, float]) -> str:
        """Assess individual sensor status."""
        if sensor_type == 'eis':
            # EIS sensor assessment
            quality = properties.get('measurement_quality', 0.5)
            reliability = self.calibration.eis_reliability

            if quality > 0.8 and reliability > 0.8:
                return "good"
            elif quality > 0.5 and reliability > 0.5:
                return "degraded"
            else:
                return "failed"

        else:  # QCM
            # QCM sensor assessment
            quality = properties.get('measurement_quality', 0.5)
            reliability = self.calibration.qcm_reliability
            q_factor = getattr(measurement, 'quality_factor', 10000)

            if quality > 0.8 and reliability > 0.8 and q_factor > 5000:
                return "good"
            elif quality > 0.5 and reliability > 0.5 and q_factor > 1000:
                return "degraded"
            else:
                return "failed"

    def _calculate_agreement(self, value1: float, value2: float) -> float:
        """Calculate agreement between two measurements (0-1)."""
        if max(value1, value2) == 0:
            return 1.0  # Both zero is perfect agreement

        relative_diff = abs(value1 - value2) / max(value1, value2)
        agreement = max(0.0, 1.0 - relative_diff / 0.5)  # 50% diff gives 0 agreement

        return agreement

    def _estimate_biomass_from_thickness(self, thickness: float) -> float:
        """Estimate biomass density from thickness using species properties."""
        if self.species == BacterialSpecies.GEOBACTER:
            # G. sulfurreducens: denser biofilms
            density = 1.15  # g/cm³
            porosity = 0.7
        elif self.species == BacterialSpecies.SHEWANELLA:
            # S. oneidensis: less dense biofilms
            density = 1.08  # g/cm³
            porosity = 0.8
        else:  # Mixed
            density = 1.12  # g/cm³
            porosity = 0.75

        # Convert thickness to biomass density
        # biomass_density = thickness * density * (1 - porosity) / volume_factor
        volume_factor = 100.0  # Empirical factor for μm to g/L conversion
        biomass_density = thickness * density * (1 - porosity) / volume_factor

        return max(0.0, biomass_density)

    def _calculate_fusion_confidence(self, agreement: float, eis_weight: float,
                                   qcm_weight: float, eis_status: str, qcm_status: str) -> float:
        """Calculate overall fusion confidence."""
        # Agreement component (0-1)
        agreement_factor = agreement

        # Weight balance component (penalize extreme weights)
        weight_balance = 1.0 - abs(eis_weight - qcm_weight)

        # Sensor status component
        status_scores = {'good': 1.0, 'degraded': 0.6, 'failed': 0.1}
        status_factor = (status_scores[eis_status] + status_scores[qcm_status]) / 2

        # Combined confidence
        confidence = agreement_factor * 0.5 + weight_balance * 0.2 + status_factor * 0.3

        return np.clip(confidence, 0.0, 1.0)

    def get_fusion_summary(self) -> dict[str, Any]:
        """Get comprehensive fusion system summary."""
        return {
            'fusion_method': self.method.value,
            'species': self.species.value,
            'current_thickness_um': self.current_thickness,
            'current_biomass_g_per_L': self.current_biomass,
            'current_conductivity_S_per_m': self.current_conductivity,
            'measurement_time_hours': self.measurement_time,
            'fusion_history_length': len(self.fusion_history),
            'recent_agreement': np.mean(self.agreement_history[-5:]) if self.agreement_history else 0.5,
            'calibration_status': self.calibration.get_calibration_status(),
            'kalman_filter_performance': (self.kalman_filter.assess_filter_performance()
                                        if self.kalman_filter else {}),
            'gpu_available': self.gpu_available,
            'sensor_weights': {
                'recent_eis_weight': np.mean([w[0] for w in self.sensor_weights_history[-5:]])
                                   if self.sensor_weights_history else 0.5,
                'recent_qcm_weight': np.mean([w[1] for w in self.sensor_weights_history[-5:]])
                                   if self.sensor_weights_history else 0.5
            }
        }

    def detect_sensor_faults(self) -> dict[str, Any]:
        """Detect and diagnose sensor faults."""
        faults = {
            'eis_faults': [],
            'qcm_faults': [],
            'fusion_faults': []
        }

        # Check recent fusion history
        if len(self.fusion_history) < 5:
            return faults

        recent_measurements = self.fusion_history[-10:]

        # EIS fault detection
        eis_qualities = [m.eis_status for m in recent_measurements]
        if eis_qualities.count('failed') > len(eis_qualities) / 2:
            faults['eis_faults'].append('frequent_failures')

        if self.calibration.eis_reliability < self.fault_threshold:
            faults['eis_faults'].append('low_reliability')

        # QCM fault detection
        qcm_qualities = [m.qcm_status for m in recent_measurements]
        if qcm_qualities.count('failed') > len(qcm_qualities) / 2:
            faults['qcm_faults'].append('frequent_failures')

        if self.calibration.qcm_reliability < self.fault_threshold:
            faults['qcm_faults'].append('low_reliability')

        # Fusion fault detection
        agreements = [m.sensor_agreement for m in recent_measurements]
        if np.mean(agreements) < 0.3:
            faults['fusion_faults'].append('poor_sensor_agreement')

        confidences = [m.fusion_confidence for m in recent_measurements]
        if np.mean(confidences) < 0.4:
            faults['fusion_faults'].append('low_fusion_confidence')

        cross_val_errors = [m.cross_validation_error for m in recent_measurements]
        if np.mean(cross_val_errors) > self.max_disagreement:
            faults['fusion_faults'].append('excessive_sensor_disagreement')

        return faults
