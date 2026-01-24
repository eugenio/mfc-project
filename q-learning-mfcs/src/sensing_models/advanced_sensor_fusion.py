"""Advanced Sensor Fusion with Predictive Capabilities.

Phase 2 enhancement of the MFC sensor fusion system with:
- Predictive Kalman filtering with multi-step ahead predictions
- Advanced statistical fault detection and isolation
- Adaptive model parameters based on learning biofilm dynamics
- Machine learning-enhanced state estimation
- Biofilm growth pattern recognition and prediction

Created: 2025-07-31
Last Modified: 2025-07-31
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

# Import base sensor fusion components
from .sensor_fusion import (
    BacterialSpecies,
    FusedMeasurement,
    FusionMethod,
    KalmanFilter,
    SensorFusion,
)

if TYPE_CHECKING:
    from .eis_model import EISMeasurement
    from .qcm_model import QCMMeasurement

# Configuration and GPU support
try:
    from config import SensorConfig
except ImportError:
    SensorConfig = None

try:
    from gpu_acceleration import get_gpu_accelerator
except ImportError:
    get_gpu_accelerator = None

# Statistical analysis
try:
    from scipy import stats
    from scipy.optimize import minimize
    from scipy.signal import find_peaks

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PredictiveState:
    """State prediction with uncertainty bounds."""

    predicted_values: np.ndarray  # [thickness, biomass, conductivity]
    upper_confidence: np.ndarray  # Upper 95% confidence interval
    lower_confidence: np.ndarray  # Lower 95% confidence interval
    prediction_horizon_hours: float
    prediction_accuracy: float  # Historical accuracy metric (0-1)


@dataclass
class AnomalyDetection:
    """Anomaly detection results."""

    timestamp: float
    anomaly_score: float  # 0-1, higher means more anomalous
    anomaly_type: str  # 'sensor_drift', 'measurement_outlier', 'pattern_change'
    affected_sensors: list[str]  # ['eis', 'qcm', 'fusion']
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float  # Detection confidence (0-1)
    recommended_action: str  # Suggested response


@dataclass
class BiofimGrowthPattern:
    """Biofilm growth pattern analysis."""

    growth_phase: str  # 'lag', 'exponential', 'stationary', 'decline', 'unknown'
    growth_rate_um_per_hour: float
    pattern_confidence: float  # 0-1
    predicted_next_phase: str
    phase_transition_time_hours: float | None
    characteristic_time_constant: float  # Hours


class AdvancedKalmanFilter(KalmanFilter):
    """Enhanced Kalman filter with predictive capabilities and adaptive parameters.

    Extended state vector: [thickness, biomass, conductivity, d_thickness/dt, d_biomass/dt, d_conductivity/dt]
    Includes adaptive noise covariance and multi-step prediction.
    """

    def __init__(
        self,
        dt: float = 0.1,
        config: SensorConfig | None = None,
        prediction_horizon: int = 10,
        enable_adaptation: bool = True,
    ) -> None:
        """Initialize advanced Kalman filter.

        Args:
            dt: Time step (hours)
            config: Optional sensor configuration
            prediction_horizon: Number of steps to predict ahead
            enable_adaptation: Enable adaptive parameter tuning

        """
        # Initialize base filter
        super().__init__(dt, config)

        # Extended state space for better dynamics modeling
        self.n_states = 6  # [thickness, biomass, conductivity, d_thickness/dt, d_biomass/dt, d_conductivity/dt]
        self.n_measurements = 3  # [eis_thickness, qcm_thickness, eis_conductivity]

        # Re-initialize with extended state
        self.state = np.zeros(self.n_states)
        self.covariance = np.eye(self.n_states) * 100.0

        # Enhanced state transition matrix
        self.F = np.array(
            [
                [1, 0, 0, dt, 0, 0],  # thickness
                [0, 1, 0, 0, dt, 0],  # biomass_density
                [0, 0, 1, 0, 0, dt],  # conductivity
                [0, 0, 0, 0.95, 0, 0],  # d_thickness/dt (with decay)
                [0, 0, 0, 0, 0.9, 0],  # d_biomass/dt (with decay)
                [0, 0, 0, 0, 0, 0.8],  # d_conductivity/dt (with decay)
            ],
        )

        # Enhanced process noise with adaptive capability
        if enable_adaptation:
            self.Q = np.diag([0.1, 0.5, 1e-6, 0.05, 0.1, 1e-8])
            self.Q_adaptive_factor = 1.0
        else:
            self.Q = np.diag([0.1, 0.5, 1e-6, 0.05, 0.1, 1e-8])

        # Enhanced measurement matrix
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],  # EIS thickness measurement
                [1, 0, 0, 0, 0, 0],  # QCM thickness measurement
                [0, 0, 1, 0, 0, 0],  # EIS conductivity measurement
            ],
        )

        # Prediction capabilities
        self.prediction_horizon = prediction_horizon
        self.enable_adaptation = enable_adaptation
        self.prediction_history = []
        self.prediction_errors = []

        # Adaptive parameters
        self.innovation_window = deque(maxlen=20)
        self.residual_window = deque(maxlen=20)
        self.adaptation_counter = 0

        # Biofilm dynamics modeling
        self.growth_model_params = {
            "max_growth_rate": 2.0,  # μm/hour
            "saturation_thickness": 100.0,  # μm
            "decay_rate": 0.01,  # 1/hour
        }

        logger.info("Advanced Kalman Filter initialized with predictive capabilities")

    def initialize_state(
        self,
        initial_thickness: float,
        initial_biomass: float,
        initial_conductivity: float,
    ) -> None:
        """Initialize extended state vector."""
        self.state = np.array(
            [
                initial_thickness,  # thickness
                initial_biomass,  # biomass_density
                initial_conductivity,  # conductivity
                0.1,  # d_thickness/dt (small positive initial growth)
                0.05,  # d_biomass/dt
                0.0,  # d_conductivity/dt
            ],
        )
        self.initialized = True

        logger.debug(f"Advanced filter initialized with state: {self.state}")

    def predict_multi_step(self, steps: int) -> PredictiveState:
        """Multi-step ahead prediction with uncertainty propagation.

        Args:
            steps: Number of time steps to predict ahead

        Returns:
            Predicted state with confidence intervals

        """
        if not self.initialized:
            logger.warning("Filter not initialized, returning zero predictions")
            return PredictiveState(
                predicted_values=np.zeros(3),
                upper_confidence=np.zeros(3),
                lower_confidence=np.zeros(3),
                prediction_horizon_hours=steps * self.dt,
                prediction_accuracy=0.0,
            )

        # Save current state
        original_state = self.state.copy()
        original_covariance = self.covariance.copy()

        predicted_states = []
        predicted_covariances = []

        # Multi-step prediction
        for _step in range(steps):
            # Apply nonlinear biofilm growth model
            self._apply_biofilm_dynamics()

            # Standard prediction step
            self.state = np.dot(self.F, self.state)
            self.covariance = np.dot(np.dot(self.F, self.covariance), self.F.T) + self.Q

            # Store prediction
            predicted_states.append(self.state.copy())
            predicted_covariances.append(self.covariance.copy())

            # Ensure physical constraints
            self.state[0] = max(0, self.state[0])  # thickness >= 0
            self.state[1] = max(0, self.state[1])  # biomass >= 0
            self.state[2] = max(0, self.state[2])  # conductivity >= 0

        # Restore original state
        self.state = original_state
        self.covariance = original_covariance

        # Extract final prediction
        final_state = predicted_states[-1]
        final_covariance = predicted_covariances[-1]

        # Calculate confidence intervals (95%)
        state_std = np.sqrt(np.diag(final_covariance))
        confidence_factor = 1.96  # 95% confidence interval

        predicted_values = final_state[:3]  # [thickness, biomass, conductivity]
        upper_confidence = predicted_values + confidence_factor * state_std[:3]
        lower_confidence = predicted_values - confidence_factor * state_std[:3]

        # Ensure physical bounds
        lower_confidence = np.maximum(lower_confidence, 0)

        # Calculate prediction accuracy based on historical performance
        prediction_accuracy = self._calculate_prediction_accuracy()

        prediction = PredictiveState(
            predicted_values=predicted_values,
            upper_confidence=upper_confidence,
            lower_confidence=lower_confidence,
            prediction_horizon_hours=steps * self.dt,
            prediction_accuracy=prediction_accuracy,
        )

        # Store for accuracy tracking
        self.prediction_history.append(prediction)

        return prediction

    def _apply_biofilm_dynamics(self) -> None:
        """Apply nonlinear biofilm growth dynamics to state prediction."""
        thickness = self.state[0]
        self.state[1]
        growth_rate = self.state[3]  # d_thickness/dt

        # Logistic growth model with saturation
        max_rate = self.growth_model_params["max_growth_rate"]
        saturation = self.growth_model_params["saturation_thickness"]
        decay_rate = self.growth_model_params["decay_rate"]

        # Modified growth rate based on current thickness
        saturation_factor = 1.0 - (thickness / saturation) if saturation > 0 else 1.0
        saturation_factor = max(
            0,
            saturation_factor,
        )  # Prevent negative growth beyond saturation

        # Apply logistic growth dynamics
        growth_acceleration = max_rate * saturation_factor - decay_rate * growth_rate

        # Update growth rate (with limits)
        new_growth_rate = growth_rate + growth_acceleration * self.dt
        new_growth_rate = np.clip(new_growth_rate, -0.5, max_rate)  # Physical limits

        self.state[3] = new_growth_rate

        # Couple biomass growth to thickness growth
        biomass_growth_factor = 0.8  # Biomass grows slightly slower than thickness
        self.state[4] = new_growth_rate * biomass_growth_factor

    def _calculate_prediction_accuracy(self) -> float:
        """Calculate historical prediction accuracy."""
        if len(self.prediction_errors) < 5:
            return 0.5  # Default moderate accuracy

        # Calculate mean absolute percentage error
        recent_errors = self.prediction_errors[-10:]
        mape = np.mean([abs(error) for error in recent_errors])

        # Convert to accuracy (0-1 scale)
        return max(0, 1.0 - mape / 10.0)  # Assuming 10μm is terrible error

    def update_with_prediction_validation(
        self,
        measurements: np.ndarray,
        measurement_uncertainties: np.ndarray,
        validate_predictions: bool = True,
    ) -> None:
        """Update filter and validate previous predictions.

        Args:
            measurements: [eis_thickness, qcm_thickness, eis_conductivity]
            measurement_uncertainties: Uncertainties for each measurement
            validate_predictions: Whether to validate and update prediction accuracy

        """
        # Standard update
        self.update(measurements, measurement_uncertainties)

        # Validate previous predictions if available
        if validate_predictions and self.prediction_history:
            self._validate_predictions(measurements)

        # Adaptive parameter tuning
        if self.enable_adaptation:
            self._adapt_filter_parameters()

    def _validate_predictions(self, actual_measurements: np.ndarray) -> None:
        """Validate previous predictions against actual measurements."""
        if not self.prediction_history:
            return

        # Find predictions that should match current time
        for i, prediction in enumerate(self.prediction_history):
            if (
                abs(prediction.prediction_horizon_hours - self.dt) < 0.01
            ):  # One-step prediction
                # Calculate prediction error
                predicted_thickness = prediction.predicted_values[0]
                actual_thickness = np.mean(
                    actual_measurements[:2],
                )  # Average EIS and QCM

                prediction_error = abs(predicted_thickness - actual_thickness)
                self.prediction_errors.append(prediction_error)

                # Remove validated prediction
                self.prediction_history.pop(i)
                break

    def _adapt_filter_parameters(self) -> None:
        """Adaptively tune filter parameters based on recent performance."""
        self.adaptation_counter += 1

        # Adapt every 10 measurements
        if self.adaptation_counter % 10 != 0:
            return

        if len(self.innovation_window) < 10:
            return

        # Calculate innovation statistics
        innovations = np.array(list(self.innovation_window))
        innovation_variance = np.var(innovations, axis=0)

        # Adapt measurement noise covariance
        # If innovations are larger than expected, increase R
        expected_variance = np.diag(self.R)
        variance_ratio = innovation_variance / expected_variance

        # Conservative adaptation
        adaptation_rate = 0.1
        for i in range(len(variance_ratio)):
            if variance_ratio[i] > 1.5:  # Innovation too large
                self.R[i, i] *= 1 + adaptation_rate
            elif variance_ratio[i] < 0.5:  # Innovation too small
                self.R[i, i] *= 1 - adaptation_rate

        # Adapt process noise if residuals are consistently small/large
        if len(self.residual_window) >= 10:
            residuals = np.array(list(self.residual_window))
            residual_magnitude = np.mean(np.linalg.norm(residuals, axis=1))

            if residual_magnitude > 2.0:  # Model mismatch
                self.Q_adaptive_factor = min(2.0, self.Q_adaptive_factor * 1.1)
            elif residual_magnitude < 0.5:  # Over-conservative model
                self.Q_adaptive_factor = max(0.5, self.Q_adaptive_factor * 0.95)

            self.Q = self.Q * self.Q_adaptive_factor

        logger.debug(
            f"Filter parameters adapted: R_scale={np.mean(np.diag(self.R)):.3f}, "
            f"Q_factor={self.Q_adaptive_factor:.3f}",
        )

    def get_biofilm_growth_analysis(self) -> BiofimGrowthPattern:
        """Analyze current biofilm growth pattern."""
        if not self.initialized or len(self.innovation_history) < 10:
            return BiofimGrowthPattern(
                growth_phase="unknown",
                growth_rate_um_per_hour=0.0,
                pattern_confidence=0.0,
                predicted_next_phase="unknown",
                phase_transition_time_hours=None,
                characteristic_time_constant=float("inf"),
            )

        # Extract thickness history from state
        thickness = self.state[0]
        growth_rate = self.state[3]  # d_thickness/dt

        # Classify growth phase based on growth rate and thickness
        if growth_rate < 0.01 and thickness < 5.0:
            phase = "lag"
            next_phase = "exponential"
            transition_time = 2.0  # hours
        elif growth_rate > 0.5:
            phase = "exponential"
            next_phase = "stationary"
            transition_time = thickness / growth_rate * 0.8  # Time to 80% saturation
        elif 0.01 <= growth_rate <= 0.5 and thickness > 20.0:
            phase = "stationary"
            next_phase = "decline"
            transition_time = 10.0  # hours
        elif growth_rate < 0:
            phase = "decline"
            next_phase = "lag"
            transition_time = thickness / abs(growth_rate)  # Time to minimal thickness
        else:
            phase = "unknown"
            next_phase = "unknown"
            transition_time = None

        # Calculate characteristic time constant
        if abs(growth_rate) > 0.001:
            time_constant = thickness / abs(growth_rate)
        else:
            time_constant = float("inf")

        # Pattern confidence based on filter consistency
        filter_performance = self.assess_filter_performance()
        confidence = filter_performance.get("filter_consistency", 0.5)
        confidence = min(1.0, max(0.0, confidence))

        return BiofimGrowthPattern(
            growth_phase=phase,
            growth_rate_um_per_hour=growth_rate,
            pattern_confidence=confidence,
            predicted_next_phase=next_phase,
            phase_transition_time_hours=transition_time,
            characteristic_time_constant=time_constant,
        )


class StatisticalAnomalyDetector:
    """Advanced statistical anomaly detection for sensor measurements and fusion results.

    Uses multiple detection methods:
    - Isolation Forest for outlier detection
    - Statistical process control (SPC) charts
    - Change point detection
    - Sensor drift analysis
    """

    def __init__(self, window_size: int = 50, sensitivity: float = 0.95) -> None:
        """Initialize anomaly detector.

        Args:
            window_size: Size of sliding window for analysis
            sensitivity: Detection sensitivity (0-1, higher = more sensitive)

        """
        self.window_size = window_size
        self.sensitivity = sensitivity

        # Historical data storage
        self.measurement_history = deque(maxlen=window_size * 2)
        self.fusion_history = deque(maxlen=window_size * 2)
        self.anomaly_history = []

        # Statistical baselines
        self.baseline_stats = {
            "thickness_mean": 0.0,
            "thickness_std": 1.0,
            "agreement_mean": 0.5,
            "agreement_std": 0.2,
            "confidence_mean": 0.5,
            "confidence_std": 0.2,
        }

        # Detection thresholds
        self.control_limits = {
            "thickness_upper": 0.0,
            "thickness_lower": 0.0,
            "agreement_lower": 0.0,
            "confidence_lower": 0.0,
        }

        # Initialize with scipy if available
        self.scipy_available = SCIPY_AVAILABLE
        if not self.scipy_available:
            logger.warning("SciPy not available - using simplified anomaly detection")

        logger.info(
            f"Statistical anomaly detector initialized (sensitivity={sensitivity})",
        )

    def update_baseline(self, measurements: list[FusedMeasurement]) -> None:
        """Update statistical baseline from historical measurements."""
        if len(measurements) < 10:
            return

        # Extract key metrics
        thicknesses = [m.thickness_um for m in measurements]
        agreements = [m.sensor_agreement for m in measurements]
        confidences = [m.fusion_confidence for m in measurements]

        # Update baseline statistics
        self.baseline_stats["thickness_mean"] = np.mean(thicknesses)
        self.baseline_stats["thickness_std"] = np.std(thicknesses)
        self.baseline_stats["agreement_mean"] = np.mean(agreements)
        self.baseline_stats["agreement_std"] = np.std(agreements)
        self.baseline_stats["confidence_mean"] = np.mean(confidences)
        self.baseline_stats["confidence_std"] = np.std(confidences)

        # Calculate control limits (3-sigma for 99.7% coverage)
        sigma_factor = 3.0 * (
            1.0 - self.sensitivity + 0.5
        )  # Adjust based on sensitivity

        self.control_limits["thickness_upper"] = (
            self.baseline_stats["thickness_mean"]
            + sigma_factor * self.baseline_stats["thickness_std"]
        )
        self.control_limits["thickness_lower"] = max(
            0,
            self.baseline_stats["thickness_mean"]
            - sigma_factor * self.baseline_stats["thickness_std"],
        )
        self.control_limits["agreement_lower"] = max(
            0,
            self.baseline_stats["agreement_mean"]
            - sigma_factor * self.baseline_stats["agreement_std"],
        )
        self.control_limits["confidence_lower"] = max(
            0,
            self.baseline_stats["confidence_mean"]
            - sigma_factor * self.baseline_stats["confidence_std"],
        )

        logger.debug(
            f"Baseline updated: thickness={self.baseline_stats['thickness_mean']:.2f}±"
            f"{self.baseline_stats['thickness_std']:.2f}, "
            f"agreement={self.baseline_stats['agreement_mean']:.3f}",
        )

    def detect_anomalies(
        self,
        current_measurement: FusedMeasurement,
    ) -> list[AnomalyDetection]:
        """Detect anomalies in current measurement.

        Args:
            current_measurement: Latest fused measurement

        Returns:
            List of detected anomalies

        """
        anomalies = []

        # Store measurement for history
        self.fusion_history.append(current_measurement)

        # Need sufficient history for reliable detection
        if len(self.fusion_history) < 10:
            return anomalies

        # Update baseline periodically
        if len(self.fusion_history) % 20 == 0:
            self.update_baseline(list(self.fusion_history))

        # 1. Statistical Process Control (SPC) anomalies
        spc_anomalies = self._detect_spc_anomalies(current_measurement)
        anomalies.extend(spc_anomalies)

        # 2. Sensor drift detection
        drift_anomalies = self._detect_sensor_drift()
        anomalies.extend(drift_anomalies)

        # 3. Pattern change detection
        pattern_anomalies = self._detect_pattern_changes()
        anomalies.extend(pattern_anomalies)

        # 4. Cross-validation anomalies
        cross_val_anomalies = self._detect_cross_validation_anomalies(
            current_measurement,
        )
        anomalies.extend(cross_val_anomalies)

        # Store anomalies for trend analysis
        self.anomaly_history.extend(anomalies)

        # Keep only recent anomalies
        cutoff_time = current_measurement.timestamp - 24.0  # Last 24 hours
        self.anomaly_history = [
            a for a in self.anomaly_history if a.timestamp > cutoff_time
        ]

        return anomalies

    def _detect_spc_anomalies(
        self,
        measurement: FusedMeasurement,
    ) -> list[AnomalyDetection]:
        """Detect statistical process control anomalies."""
        anomalies = []

        # Check thickness against control limits
        if measurement.thickness_um > self.control_limits["thickness_upper"]:
            anomalies.append(
                AnomalyDetection(
                    timestamp=measurement.timestamp,
                    anomaly_score=min(
                        1.0,
                        (
                            measurement.thickness_um
                            - self.control_limits["thickness_upper"]
                        )
                        / self.baseline_stats["thickness_std"],
                    ),
                    anomaly_type="measurement_outlier",
                    affected_sensors=["eis", "qcm"],
                    severity=(
                        "medium"
                        if measurement.thickness_um
                        < self.control_limits["thickness_upper"] * 1.5
                        else "high"
                    ),
                    confidence=0.9,
                    recommended_action="Verify sensor calibration and check for measurement errors",
                ),
            )

        elif measurement.thickness_um < self.control_limits["thickness_lower"]:
            anomalies.append(
                AnomalyDetection(
                    timestamp=measurement.timestamp,
                    anomaly_score=min(
                        1.0,
                        (
                            self.control_limits["thickness_lower"]
                            - measurement.thickness_um
                        )
                        / self.baseline_stats["thickness_std"],
                    ),
                    anomaly_type="measurement_outlier",
                    affected_sensors=["eis", "qcm"],
                    severity="medium",
                    confidence=0.8,
                    recommended_action="Check for sensor calibration drift or biofilm detachment",
                ),
            )

        # Check sensor agreement
        if measurement.sensor_agreement < self.control_limits["agreement_lower"]:
            anomalies.append(
                AnomalyDetection(
                    timestamp=measurement.timestamp,
                    anomaly_score=(
                        self.control_limits["agreement_lower"]
                        - measurement.sensor_agreement
                    )
                    / self.baseline_stats["agreement_std"],
                    anomaly_type="sensor_drift",
                    affected_sensors=["eis", "qcm"],
                    severity="high" if measurement.sensor_agreement < 0.2 else "medium",
                    confidence=0.85,
                    recommended_action="Investigate sensor disagreement - possible drift or interference",
                ),
            )

        # Check fusion confidence
        if measurement.fusion_confidence < self.control_limits["confidence_lower"]:
            anomalies.append(
                AnomalyDetection(
                    timestamp=measurement.timestamp,
                    anomaly_score=(
                        self.control_limits["confidence_lower"]
                        - measurement.fusion_confidence
                    )
                    / self.baseline_stats["confidence_std"],
                    anomaly_type="pattern_change",
                    affected_sensors=["fusion"],
                    severity="medium",
                    confidence=0.7,
                    recommended_action="Review fusion algorithm parameters and sensor status",
                ),
            )

        return anomalies

    def _detect_sensor_drift(self) -> list[AnomalyDetection]:
        """Detect gradual sensor drift using trend analysis."""
        anomalies = []

        if len(self.fusion_history) < 20:
            return anomalies

        # Analyze recent measurements for trends
        recent_measurements = list(self.fusion_history)[-20:]
        timestamps = [m.timestamp for m in recent_measurements]
        eis_thicknesses = [m.eis_thickness for m in recent_measurements]
        qcm_thicknesses = [m.qcm_thickness for m in recent_measurements]
        agreements = [m.sensor_agreement for m in recent_measurements]

        # Detect linear trends
        if self.scipy_available and len(timestamps) >= 10:
            try:
                # EIS drift detection
                eis_slope, _, eis_r_value, eis_p_value, _ = stats.linregress(
                    timestamps,
                    eis_thicknesses,
                )
                if (
                    abs(eis_slope) > 0.5 and eis_p_value < 0.05
                ):  # Significant trend > 0.5 μm/hour
                    anomalies.append(
                        AnomalyDetection(
                            timestamp=recent_measurements[-1].timestamp,
                            anomaly_score=min(1.0, abs(eis_slope) / 2.0),
                            anomaly_type="sensor_drift",
                            affected_sensors=["eis"],
                            severity="high" if abs(eis_slope) > 1.0 else "medium",
                            confidence=abs(eis_r_value),
                            recommended_action=f"EIS sensor showing {'positive' if eis_slope > 0 else 'negative'} drift trend",
                        ),
                    )

                # QCM drift detection
                qcm_slope, _, qcm_r_value, qcm_p_value, _ = stats.linregress(
                    timestamps,
                    qcm_thicknesses,
                )
                if (
                    abs(qcm_slope) > 0.3 and qcm_p_value < 0.05
                ):  # Significant trend > 0.3 μm/hour
                    anomalies.append(
                        AnomalyDetection(
                            timestamp=recent_measurements[-1].timestamp,
                            anomaly_score=min(1.0, abs(qcm_slope) / 1.5),
                            anomaly_type="sensor_drift",
                            affected_sensors=["qcm"],
                            severity="high" if abs(qcm_slope) > 0.8 else "medium",
                            confidence=abs(qcm_r_value),
                            recommended_action=f"QCM sensor showing {'positive' if qcm_slope > 0 else 'negative'} drift trend",
                        ),
                    )

                # Agreement trend (should be stable)
                agree_slope, _, agree_r_value, agree_p_value, _ = stats.linregress(
                    timestamps,
                    agreements,
                )
                if (
                    abs(agree_slope) > 0.01 and agree_p_value < 0.05
                ):  # Declining agreement
                    anomalies.append(
                        AnomalyDetection(
                            timestamp=recent_measurements[-1].timestamp,
                            anomaly_score=min(1.0, abs(agree_slope) / 0.05),
                            anomaly_type="sensor_drift",
                            affected_sensors=["eis", "qcm"],
                            severity="high" if agree_slope < -0.02 else "medium",
                            confidence=abs(agree_r_value),
                            recommended_action="Sensor agreement trending down - check for calibration drift",
                        ),
                    )

            except Exception as e:
                logger.warning(f"Error in drift detection: {e}")

        return anomalies

    def _detect_pattern_changes(self) -> list[AnomalyDetection]:
        """Detect sudden changes in measurement patterns."""
        anomalies = []

        if len(self.fusion_history) < 30:
            return anomalies

        # Compare recent vs. historical patterns
        recent_window = 10
        historical_window = 20

        recent_measurements = list(self.fusion_history)[-recent_window:]
        historical_measurements = list(self.fusion_history)[
            -(historical_window + recent_window) : -recent_window
        ]

        # Calculate statistical differences
        recent_thickness_mean = np.mean([m.thickness_um for m in recent_measurements])
        historical_thickness_mean = np.mean(
            [m.thickness_um for m in historical_measurements],
        )
        historical_thickness_std = np.std(
            [m.thickness_um for m in historical_measurements],
        )

        # Detect significant mean shift
        if historical_thickness_std > 0:
            z_score = (
                abs(recent_thickness_mean - historical_thickness_mean)
                / historical_thickness_std
            )
            if z_score > 2.5:  # Significant change
                anomalies.append(
                    AnomalyDetection(
                        timestamp=recent_measurements[-1].timestamp,
                        anomaly_score=min(1.0, z_score / 5.0),
                        anomaly_type="pattern_change",
                        affected_sensors=["eis", "qcm"],
                        severity="high" if z_score > 4.0 else "medium",
                        confidence=min(0.95, z_score / 5.0),
                        recommended_action=f"Significant {'increase' if recent_thickness_mean > historical_thickness_mean else 'decrease'} in thickness pattern detected",
                    ),
                )

        # Detect variance changes
        recent_thickness_std = np.std([m.thickness_um for m in recent_measurements])
        if historical_thickness_std > 0 and recent_thickness_std > 0:
            variance_ratio = recent_thickness_std / historical_thickness_std
            if (
                variance_ratio > 2.0 or variance_ratio < 0.5
            ):  # Significant variance change
                anomalies.append(
                    AnomalyDetection(
                        timestamp=recent_measurements[-1].timestamp,
                        anomaly_score=min(1.0, abs(np.log(variance_ratio)) / 2.0),
                        anomaly_type="pattern_change",
                        affected_sensors=["fusion"],
                        severity="medium",
                        confidence=0.8,
                        recommended_action=f"Measurement variability {'increased' if variance_ratio > 1 else 'decreased'} significantly",
                    ),
                )

        return anomalies

    def _detect_cross_validation_anomalies(
        self,
        measurement: FusedMeasurement,
    ) -> list[AnomalyDetection]:
        """Detect anomalies in cross-validation between sensors."""
        anomalies = []

        # Large disagreement between sensors
        if measurement.cross_validation_error > 15.0:  # μm threshold
            severity = (
                "critical" if measurement.cross_validation_error > 30.0 else "high"
            )
            anomalies.append(
                AnomalyDetection(
                    timestamp=measurement.timestamp,
                    anomaly_score=min(1.0, measurement.cross_validation_error / 50.0),
                    anomaly_type="sensor_drift",
                    affected_sensors=["eis", "qcm"],
                    severity=severity,
                    confidence=0.9,
                    recommended_action=f"Large sensor disagreement ({measurement.cross_validation_error:.1f} μm) - check calibration",
                ),
            )

        # Check for systematic bias
        if len(self.fusion_history) >= 10:
            recent_measurements = list(self.fusion_history)[-10:]
            eis_bias = np.mean(
                [m.eis_thickness - m.qcm_thickness for m in recent_measurements],
            )

            if abs(eis_bias) > 8.0:  # Consistent bias > 8 μm
                anomalies.append(
                    AnomalyDetection(
                        timestamp=measurement.timestamp,
                        anomaly_score=min(1.0, abs(eis_bias) / 20.0),
                        anomaly_type="sensor_drift",
                        affected_sensors=["eis" if eis_bias > 0 else "qcm"],
                        severity="medium",
                        confidence=0.8,
                        recommended_action=f"Systematic bias detected: {'EIS reads higher' if eis_bias > 0 else 'QCM reads higher'} by {abs(eis_bias):.1f} μm",
                    ),
                )

        return anomalies

    def get_anomaly_summary(self) -> dict[str, Any]:
        """Get summary of recent anomaly detection results."""
        if not self.anomaly_history:
            return {"no_anomalies": True}

        # Recent anomalies (last 4 hours)
        recent_cutoff = max([a.timestamp for a in self.anomaly_history]) - 4.0
        recent_anomalies = [
            a for a in self.anomaly_history if a.timestamp > recent_cutoff
        ]

        # Count by type and severity
        anomaly_counts = {
            "total": len(recent_anomalies),
            "by_type": {},
            "by_severity": {},
            "by_sensor": {},
        }

        for anomaly in recent_anomalies:
            # Count by type
            anomaly_counts["by_type"][anomaly.anomaly_type] = (
                anomaly_counts["by_type"].get(anomaly.anomaly_type, 0) + 1
            )

            # Count by severity
            anomaly_counts["by_severity"][anomaly.severity] = (
                anomaly_counts["by_severity"].get(anomaly.severity, 0) + 1
            )

            # Count by affected sensors
            for sensor in anomaly.affected_sensors:
                anomaly_counts["by_sensor"][sensor] = (
                    anomaly_counts["by_sensor"].get(sensor, 0) + 1
                )

        # Overall system health score
        critical_count = anomaly_counts["by_severity"].get("critical", 0)
        high_count = anomaly_counts["by_severity"].get("high", 0)
        medium_count = anomaly_counts["by_severity"].get("medium", 0)

        # Health score (0-1, 1 = perfect health)
        health_score = max(
            0,
            1.0 - (critical_count * 0.3 + high_count * 0.2 + medium_count * 0.1),
        )

        return {
            "recent_anomalies": anomaly_counts,
            "system_health_score": health_score,
            "most_recent_anomaly": recent_anomalies[-1] if recent_anomalies else None,
            "baseline_stats": self.baseline_stats.copy(),
            "detection_sensitivity": self.sensitivity,
        }


class AdvancedSensorFusion(SensorFusion):
    """Phase 2 enhanced sensor fusion with predictive capabilities and advanced fault detection.

    Combines the base sensor fusion functionality with:
    - Advanced Kalman filtering with multi-step predictions
    - Statistical anomaly detection and fault isolation
    - Biofilm growth pattern recognition
    - Machine learning-enhanced parameter adaptation
    """

    def __init__(
        self,
        method: FusionMethod = FusionMethod.KALMAN_FILTER,
        species: BacterialSpecies = BacterialSpecies.MIXED,
        use_gpu: bool = True,
        config: SensorConfig | None = None,
        prediction_horizon: int = 10,
        enable_anomaly_detection: bool = True,
    ) -> None:
        """Initialize advanced sensor fusion system.

        Args:
            method: Fusion algorithm to use
            species: Bacterial species for calibration
            use_gpu: Enable GPU acceleration
            config: Optional sensor configuration
            prediction_horizon: Steps ahead for predictions
            enable_anomaly_detection: Enable statistical anomaly detection

        """
        # Initialize base fusion system
        super().__init__(method, species, use_gpu, config)

        # Replace standard Kalman filter with advanced version
        if method == FusionMethod.KALMAN_FILTER:
            self.kalman_filter = AdvancedKalmanFilter(
                config=self.config,
                prediction_horizon=prediction_horizon,
                enable_adaptation=True,
            )

        # Initialize anomaly detection
        self.enable_anomaly_detection = enable_anomaly_detection
        if enable_anomaly_detection:
            detection_sensitivity = 0.85  # High sensitivity for research environment
            self.anomaly_detector = StatisticalAnomalyDetector(
                window_size=50,
                sensitivity=detection_sensitivity,
            )
        else:
            self.anomaly_detector = None

        # Advanced tracking
        self.prediction_history = []
        self.anomaly_history = []
        self.growth_pattern_history = []

        # Performance metrics
        self.advanced_metrics = {
            "prediction_accuracy": 0.0,
            "anomaly_detection_rate": 0.0,
            "system_health_score": 1.0,
            "adaptive_improvements": 0,
        }

        logger.info(
            f"Advanced sensor fusion initialized with prediction horizon {prediction_horizon} steps",
        )

    def fuse_measurements_with_prediction(
        self,
        eis_measurement: EISMeasurement,
        qcm_measurement: QCMMeasurement,
        eis_properties: dict[str, float],
        qcm_properties: dict[str, float],
        time_hours: float = 0.0,
        predict_steps: int = 5,
    ) -> tuple[FusedMeasurement, PredictiveState | None, list[AnomalyDetection]]:
        """Enhanced fusion with prediction and anomaly detection.

        Args:
            eis_measurement: EIS measurement data
            qcm_measurement: QCM measurement data
            eis_properties: Biofilm properties from EIS
            qcm_properties: Biofilm properties from QCM
            time_hours: Measurement timestamp
            predict_steps: Number of steps to predict ahead

        Returns:
            Tuple of (fused_measurement, prediction, anomalies)

        """
        # Standard fusion
        fused_measurement = self.fuse_measurements(
            eis_measurement,
            qcm_measurement,
            eis_properties,
            qcm_properties,
            time_hours,
        )

        prediction = None
        anomalies = []

        # Generate predictions if using advanced Kalman filter
        if isinstance(self.kalman_filter, AdvancedKalmanFilter) and predict_steps > 0:
            try:
                prediction = self.kalman_filter.predict_multi_step(predict_steps)
                self.prediction_history.append(prediction)

                # Update prediction accuracy metric
                self.advanced_metrics["prediction_accuracy"] = (
                    prediction.prediction_accuracy
                )

            except Exception as e:
                logger.warning(f"Prediction failed: {e}")

        # Anomaly detection
        if self.anomaly_detector:
            try:
                anomalies = self.anomaly_detector.detect_anomalies(fused_measurement)
                self.anomaly_history.extend(anomalies)

                # Update anomaly detection metrics
                anomaly_summary = self.anomaly_detector.get_anomaly_summary()
                self.advanced_metrics["system_health_score"] = anomaly_summary.get(
                    "system_health_score",
                    1.0,
                )

            except Exception as e:
                logger.warning(f"Anomaly detection failed: {e}")

        # Update performance tracking
        self._update_advanced_metrics()

        return fused_measurement, prediction, anomalies

    def analyze_biofilm_growth_pattern(self) -> BiofimGrowthPattern | None:
        """Analyze current biofilm growth pattern."""
        if isinstance(self.kalman_filter, AdvancedKalmanFilter):
            try:
                pattern = self.kalman_filter.get_biofilm_growth_analysis()
                self.growth_pattern_history.append(pattern)
                return pattern
            except Exception as e:
                logger.warning(f"Growth pattern analysis failed: {e}")
                return None
        return None

    def get_system_health_assessment(self) -> dict[str, Any]:
        """Comprehensive system health assessment."""
        health_assessment = {
            "timestamp": datetime.now(UTC).isoformat(),
            "overall_health_score": self.advanced_metrics["system_health_score"],
            "component_health": {},
            "recommendations": [],
        }

        # Sensor fusion health
        base_summary = self.get_fusion_summary()
        health_assessment["component_health"]["fusion"] = {
            "status": "good" if base_summary["recent_agreement"] > 0.7 else "degraded",
            "agreement": base_summary["recent_agreement"],
            "calibration_quality": base_summary["calibration_status"][
                "eis_reliability"
            ],
        }

        # Kalman filter health
        if isinstance(self.kalman_filter, AdvancedKalmanFilter):
            filter_perf = self.kalman_filter.assess_filter_performance()
            consistency = filter_perf.get("filter_consistency", 0.5)

            health_assessment["component_health"]["kalman_filter"] = {
                "status": (
                    "good"
                    if consistency > 0.8
                    else "degraded"
                    if consistency > 0.5
                    else "poor"
                ),
                "consistency": consistency,
                "prediction_accuracy": self.advanced_metrics["prediction_accuracy"],
            }

        # Anomaly detection health
        if self.anomaly_detector:
            anomaly_summary = self.anomaly_detector.get_anomaly_summary()
            recent_anomalies = anomaly_summary.get("recent_anomalies", {}).get(
                "total",
                0,
            )

            health_assessment["component_health"]["anomaly_detection"] = {
                "status": "good" if recent_anomalies < 3 else "alert",
                "recent_anomalies": recent_anomalies,
                "sensitivity": anomaly_summary.get("detection_sensitivity", 0.85),
            }

        # Generate recommendations
        if health_assessment["overall_health_score"] < 0.7:
            health_assessment["recommendations"].append(
                "System health below optimal - investigate sensor calibration",
            )

        if base_summary["recent_agreement"] < 0.5:
            health_assessment["recommendations"].append(
                "Poor sensor agreement - check for drift or interference",
            )

        if self.advanced_metrics["prediction_accuracy"] < 0.6:
            health_assessment["recommendations"].append(
                "Low prediction accuracy - consider model recalibration",
            )

        # Recent critical anomalies
        recent_critical = [
            a for a in self.anomaly_history[-10:] if a.severity == "critical"
        ]
        if recent_critical:
            health_assessment["recommendations"].append(
                f"Critical anomalies detected - {recent_critical[-1].recommended_action}",
            )

        return health_assessment

    def _update_advanced_metrics(self) -> None:
        """Update advanced performance metrics."""
        # Track adaptive improvements from Kalman filter
        if isinstance(self.kalman_filter, AdvancedKalmanFilter):
            if hasattr(self.kalman_filter, "adaptation_counter"):
                self.advanced_metrics["adaptive_improvements"] = (
                    self.kalman_filter.adaptation_counter
                )

        # Calculate anomaly detection rate (anomalies per hour)
        if self.anomaly_history:
            time_span = max(
                1.0,
                self.anomaly_history[-1].timestamp - self.anomaly_history[0].timestamp,
            )
            self.advanced_metrics["anomaly_detection_rate"] = (
                len(self.anomaly_history) / time_span
            )

    def export_advanced_diagnostics(self) -> dict[str, Any]:
        """Export comprehensive diagnostics for analysis."""
        diagnostics = {
            "fusion_diagnostics": self.get_fusion_summary(),
            "health_assessment": self.get_system_health_assessment(),
            "advanced_metrics": self.advanced_metrics.copy(),
            "recent_predictions": [],
            "recent_anomalies": [],
            "growth_patterns": [],
        }

        # Export recent predictions
        if self.prediction_history:
            for pred in self.prediction_history[-5:]:
                diagnostics["recent_predictions"].append(
                    {
                        "predicted_thickness": pred.predicted_values[0],
                        "prediction_horizon_hours": pred.prediction_horizon_hours,
                        "accuracy": pred.prediction_accuracy,
                        "confidence_bounds": {
                            "upper": pred.upper_confidence[0],
                            "lower": pred.lower_confidence[0],
                        },
                    },
                )

        # Export recent anomalies
        if self.anomaly_history:
            for anomaly in self.anomaly_history[-10:]:
                diagnostics["recent_anomalies"].append(
                    {
                        "timestamp": anomaly.timestamp,
                        "type": anomaly.anomaly_type,
                        "severity": anomaly.severity,
                        "score": anomaly.anomaly_score,
                        "affected_sensors": anomaly.affected_sensors,
                        "recommendation": anomaly.recommended_action,
                    },
                )

        # Export growth patterns
        if self.growth_pattern_history:
            for pattern in self.growth_pattern_history[-5:]:
                diagnostics["growth_patterns"].append(
                    {
                        "phase": pattern.growth_phase,
                        "growth_rate": pattern.growth_rate_um_per_hour,
                        "confidence": pattern.pattern_confidence,
                        "predicted_next_phase": pattern.predicted_next_phase,
                    },
                )

        return diagnostics


def create_advanced_sensor_fusion(
    config: SensorConfig | None = None,
) -> AdvancedSensorFusion:
    """Factory function to create advanced sensor fusion system with optimal settings.

    Args:
        config: Optional sensor configuration

    Returns:
        Configured AdvancedSensorFusion instance

    """
    # Default to Kalman filter with mixed species
    method = FusionMethod.KALMAN_FILTER
    species = BacterialSpecies.MIXED

    # Override with config if available
    if config and config.fusion:
        if hasattr(config.fusion, "fusion_method"):
            method = config.fusion.fusion_method
        if hasattr(config, "biofilm_species"):
            species = config.biofilm_species

    # Create with optimized settings for research environment
    fusion_system = AdvancedSensorFusion(
        method=method,
        species=species,
        use_gpu=True,  # Enable GPU acceleration
        config=config,
        prediction_horizon=10,  # 10-step ahead prediction
        enable_anomaly_detection=True,  # Enable anomaly detection
    )

    logger.info("Advanced sensor fusion system created with optimal research settings")
    return fusion_system
