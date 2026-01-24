#!/usr/bin/env python3
"""Degradation Detection Module for MFC Systems.

This module provides comprehensive degradation detection and analysis for
Microbial Fuel Cell (MFC) systems, including early warning detection,
degradation trend analysis, component health monitoring, and predictive
degradation modeling.

Author: MFC Analysis Team
Created: 2025-07-31
Last Modified: 2025-07-31
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
TimeSeriesData = Union[pd.Series, np.ndarray, list[float]]
Timestamp = Union[datetime, pd.Timestamp, float]
DegradationValue = Union[float, np.floating]
HealthScore = Union[float, np.floating]

# Generic types
T = TypeVar("T")
DegradationMetricType = TypeVar("DegradationMetricType", bound="DegradationMetrics")


class DegradationType(Enum):
    """Types of degradation in MFC systems."""

    BIOFILM_AGING = auto()
    ELECTRODE_CORROSION = auto()
    MEMBRANE_FOULING = auto()
    CATALYST_POISONING = auto()
    SUBSTRATE_DEPLETION = auto()
    MICROBIAL_DECLINE = auto()
    CONDUCTIVITY_LOSS = auto()
    STRUCTURAL_DAMAGE = auto()
    TEMPERATURE_STRESS = auto()
    PH_DRIFT = auto()

    def __str__(self) -> str:
        return self.name.lower().replace("_", " ")


class DegradationSeverity(Enum):
    """Severity levels for degradation."""

    MINIMAL = auto()
    MINOR = auto()
    MODERATE = auto()
    SEVERE = auto()
    CRITICAL = auto()

    @property
    def numeric_value(self) -> float:
        """Return numeric representation of severity level."""
        return {
            DegradationSeverity.MINIMAL: 0.1,
            DegradationSeverity.MINOR: 0.3,
            DegradationSeverity.MODERATE: 0.5,
            DegradationSeverity.SEVERE: 0.8,
            DegradationSeverity.CRITICAL: 1.0,
        }[self]


class DetectionMethod(Enum):
    """Methods for degradation detection."""

    STATISTICAL_PROCESS_CONTROL = auto()
    TREND_ANALYSIS = auto()
    CHANGE_POINT_DETECTION = auto()
    ANOMALY_DETECTION = auto()
    MACHINE_LEARNING = auto()
    SPECTRAL_ANALYSIS = auto()
    WAVELET_ANALYSIS = auto()


@dataclass(frozen=True)
class DegradationThresholds:
    """Thresholds for degradation detection."""

    power_degradation_threshold: float = 0.05  # 5% power loss
    efficiency_degradation_threshold: float = 0.10  # 10% efficiency loss
    biofilm_health_threshold: float = 0.80  # 80% biofilm health
    electrode_performance_threshold: float = 0.85  # 85% electrode performance
    early_warning_threshold: float = 0.02  # 2% early warning
    critical_degradation_threshold: float = 0.25  # 25% critical degradation
    trend_detection_window: int = 50  # Data points for trend analysis
    anomaly_detection_sensitivity: float = 2.0  # Standard deviations for anomaly

    def __post_init__(self) -> None:
        """Validate threshold values."""
        thresholds_to_validate = [
            ("power_degradation_threshold", 0.0, 1.0),
            ("efficiency_degradation_threshold", 0.0, 1.0),
            ("biofilm_health_threshold", 0.0, 1.0),
            ("electrode_performance_threshold", 0.0, 1.0),
        ]

        for threshold_name, min_val, max_val in thresholds_to_validate:
            value = getattr(self, threshold_name)
            if not (min_val <= value <= max_val):
                msg = f"{threshold_name} must be between {min_val} and {max_val}"
                raise ValueError(
                    msg,
                )


@dataclass
class DegradationEvent:
    """Represents a detected degradation event."""

    detection_time: datetime
    degradation_type: DegradationType
    severity: DegradationSeverity
    affected_component: str
    degradation_rate: float  # Rate of degradation per hour
    confidence: float  # Detection confidence (0-1)
    baseline_value: float  # Baseline performance value
    current_value: float  # Current degraded value
    trend_slope: float  # Trend slope indicating degradation direction
    detection_method: DetectionMethod
    additional_metrics: dict[str, float] = field(default_factory=dict)
    root_cause_analysis: str | None = None
    recommended_actions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate degradation event data."""
        if not (0.0 <= self.confidence <= 1.0):
            msg = "Confidence must be between 0.0 and 1.0"
            raise ValueError(msg)
        if self.degradation_rate < 0:
            msg = "Degradation rate must be non-negative"
            raise ValueError(msg)

    @property
    def degradation_percentage(self) -> float:
        """Calculate degradation as percentage of baseline."""
        if self.baseline_value == 0:
            return 0.0
        return abs(self.baseline_value - self.current_value) / self.baseline_value

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary format."""
        return {
            "detection_time": self.detection_time.isoformat(),
            "degradation_type": self.degradation_type.name,
            "severity": self.severity.name,
            "affected_component": self.affected_component,
            "degradation_rate": self.degradation_rate,
            "confidence": self.confidence,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "trend_slope": self.trend_slope,
            "detection_method": self.detection_method.name,
            "degradation_percentage": self.degradation_percentage,
            "additional_metrics": self.additional_metrics,
            "root_cause_analysis": self.root_cause_analysis,
            "recommended_actions": self.recommended_actions,
        }


@dataclass
class DegradationMetrics:
    """Comprehensive degradation analysis metrics."""

    # Overall degradation indicators
    overall_health_score: float = 1.0  # 0-1 scale, 1 = perfect health
    degradation_rate: float = 0.0  # Rate of health decline per hour
    time_to_critical_degradation: float = float("inf")  # Hours until critical state

    # Component-specific health scores
    biofilm_health_score: float = 1.0
    electrode_health_score: float = 1.0
    membrane_health_score: float = 1.0
    system_health_score: float = 1.0

    # Performance degradation metrics
    power_degradation_percentage: float = 0.0
    efficiency_degradation_percentage: float = 0.0
    output_stability_degradation: float = 0.0

    # Trend analysis results
    short_term_trend: float = 0.0  # Trend over last 24 hours
    medium_term_trend: float = 0.0  # Trend over last week
    long_term_trend: float = 0.0  # Trend over last month
    trend_acceleration: float = 0.0  # Rate of change in trend

    # Anomaly detection results
    anomaly_score: float = 0.0  # Anomaly detection score
    anomaly_count: int = 0  # Number of anomalies detected
    anomaly_severity: float = 0.0  # Average severity of anomalies

    # Change point detection
    change_points_detected: list[tuple[float, float]] = field(
        default_factory=list,
    )  # (time, magnitude)
    significant_changes: int = 0
    last_significant_change: datetime | None = None

    # Predictive metrics
    predicted_failure_time: datetime | None = None
    prediction_confidence: float = 0.0
    remaining_useful_life: float = float("inf")  # Hours

    # Detection quality metrics
    detection_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    detection_latency: float = 0.0  # Hours from actual to detected degradation

    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    analysis_duration: timedelta = field(default=timedelta(hours=1))
    data_quality_score: float = 1.0
    sample_size: int = 0
    method_used: DetectionMethod = DetectionMethod.STATISTICAL_PROCESS_CONTROL

    def __post_init__(self) -> None:
        """Validate and compute derived metrics."""
        self._validate_metrics()
        self._compute_derived_metrics()

    def _validate_metrics(self) -> None:
        """Validate metric values."""
        health_scores = [
            "overall_health_score",
            "biofilm_health_score",
            "electrode_health_score",
            "membrane_health_score",
            "system_health_score",
        ]

        for score_name in health_scores:
            value = getattr(self, score_name)
            if not (0.0 <= value <= 1.0):
                logger.warning(
                    f"Health score {score_name} value {value} outside expected range [0, 1]",
                )

    def _compute_derived_metrics(self) -> None:
        """Compute derived degradation metrics."""
        # Compute overall system health from component scores
        component_scores = [
            self.biofilm_health_score,
            self.electrode_health_score,
            self.membrane_health_score,
        ]

        # Use weighted average (biofilm is most critical)
        weights = [0.4, 0.3, 0.3]
        self.system_health_score = sum(
            score * weight
            for score, weight in zip(component_scores, weights, strict=False)
        )

        # Update overall health score
        self.overall_health_score = min(
            self.system_health_score,
            self.overall_health_score,
        )

    def get_severity_level(self) -> DegradationSeverity:
        """Determine degradation severity based on health score."""
        if self.overall_health_score >= 0.9:
            return DegradationSeverity.MINIMAL
        if self.overall_health_score >= 0.8:
            return DegradationSeverity.MINOR
        if self.overall_health_score >= 0.6:
            return DegradationSeverity.MODERATE
        if self.overall_health_score >= 0.4:
            return DegradationSeverity.SEVERE
        return DegradationSeverity.CRITICAL

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "overall_health_score": self.overall_health_score,
            "degradation_rate": self.degradation_rate,
            "time_to_critical_degradation": self.time_to_critical_degradation,
            "biofilm_health_score": self.biofilm_health_score,
            "electrode_health_score": self.electrode_health_score,
            "membrane_health_score": self.membrane_health_score,
            "system_health_score": self.system_health_score,
            "power_degradation_percentage": self.power_degradation_percentage,
            "efficiency_degradation_percentage": self.efficiency_degradation_percentage,
            "output_stability_degradation": self.output_stability_degradation,
            "short_term_trend": self.short_term_trend,
            "medium_term_trend": self.medium_term_trend,
            "long_term_trend": self.long_term_trend,
            "trend_acceleration": self.trend_acceleration,
            "anomaly_score": self.anomaly_score,
            "anomaly_count": self.anomaly_count,
            "anomaly_severity": self.anomaly_severity,
            "change_points_detected": [
                (float(t), float(m)) for t, m in self.change_points_detected
            ],
            "significant_changes": self.significant_changes,
            "last_significant_change": (
                self.last_significant_change.isoformat()
                if self.last_significant_change
                else None
            ),
            "predicted_failure_time": (
                self.predicted_failure_time.isoformat()
                if self.predicted_failure_time
                else None
            ),
            "prediction_confidence": self.prediction_confidence,
            "remaining_useful_life": self.remaining_useful_life,
            "detection_accuracy": self.detection_accuracy,
            "false_positive_rate": self.false_positive_rate,
            "detection_latency": self.detection_latency,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "analysis_duration": self.analysis_duration.total_seconds(),
            "data_quality_score": self.data_quality_score,
            "sample_size": self.sample_size,
            "method_used": self.method_used.name,
            "severity_level": self.get_severity_level().name,
        }


class DegradationDetector(Protocol):
    """Protocol for degradation detector implementations."""

    def detect_degradation(
        self,
        data: TimeSeriesData,
        timestamps: Sequence[Timestamp] | None = None,
        **kwargs: Any,
    ) -> tuple[DegradationMetrics, list[DegradationEvent]]:
        """Detect degradation in time series data."""
        ...

    def predict_degradation(
        self,
        current_time: float,
        prediction_horizon: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Predict future degradation."""
        ...


class BaseDegradationDetector(ABC):
    """Base class for degradation detectors."""

    def __init__(
        self,
        thresholds: DegradationThresholds | None = None,
        min_data_points: int = 30,
        detection_confidence_threshold: float = 0.7,
    ) -> None:
        """Initialize degradation detector.

        Args:
            thresholds: Custom degradation thresholds
            min_data_points: Minimum data points required for analysis
            detection_confidence_threshold: Minimum confidence for degradation detection

        """
        self.thresholds = thresholds or DegradationThresholds()
        self.min_data_points = min_data_points
        self.detection_confidence_threshold = detection_confidence_threshold
        self.baseline_established = False
        self.baseline_stats: dict[str, float] = {}
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging for the detector."""
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def detect_degradation(
        self,
        data: TimeSeriesData,
        timestamps: Sequence[Timestamp] | None = None,
        **kwargs: Any,
    ) -> tuple[DegradationMetrics, list[DegradationEvent]]:
        """Detect degradation in time series data."""

    @abstractmethod
    def predict_degradation(
        self,
        current_time: float,
        prediction_horizon: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Predict future degradation."""

    def validate_data(self, data: TimeSeriesData) -> bool:
        """Validate input data quality.

        Args:
            data: Time series data to validate

        Returns:
            True if data is valid, False otherwise

        """
        try:
            data_array = np.asarray(data, dtype=np.float64)

            if len(data_array) < self.min_data_points:
                self.logger.warning(
                    f"Insufficient data points: {len(data_array)} < {self.min_data_points}",
                )
                return False

            if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
                self.logger.warning("Data contains NaN or infinite values")
                return False

            return True

        except Exception as e:
            self.logger.exception(f"Data validation error: {e!s}")
            return False

    def establish_baseline(
        self,
        baseline_data: TimeSeriesData,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Establish baseline statistics for degradation detection.

        Args:
            baseline_data: Baseline time series data
            **kwargs: Additional parameters

        Returns:
            Dictionary of baseline statistics

        """
        try:
            data_array = np.asarray(baseline_data, dtype=np.float64)

            baseline_stats = {
                "mean": float(np.mean(data_array)),
                "std": float(np.std(data_array)),
                "median": float(np.median(data_array)),
                "min": float(np.min(data_array)),
                "max": float(np.max(data_array)),
                "q25": float(np.percentile(data_array, 25)),
                "q75": float(np.percentile(data_array, 75)),
                "iqr": float(
                    np.percentile(data_array, 75) - np.percentile(data_array, 25),
                ),
                "trend": float(
                    np.polyfit(np.arange(len(data_array)), data_array, 1)[0],
                ),
            }

            self.baseline_stats = baseline_stats
            self.baseline_established = True

            self.logger.info("Baseline established successfully")
            return baseline_stats

        except Exception as e:
            self.logger.exception(f"Baseline establishment failed: {e!s}")
            return {}

    def _calculate_health_score(
        self,
        current_value: float,
        baseline_value: float,
        component_type: str = "general",
    ) -> float:
        """Calculate health score based on current vs baseline performance.

        Args:
            current_value: Current performance value
            baseline_value: Baseline performance value
            component_type: Type of component for specific calculations

        Returns:
            Health score between 0 and 1

        """
        if baseline_value == 0:
            return 1.0

        # Calculate relative performance
        relative_performance = current_value / baseline_value

        # Different components may have different degradation patterns
        if component_type == "biofilm":
            # Biofilm health decreases more sensitively
            health_score = min(1.0, relative_performance**0.5)
        elif component_type == "electrode":
            # Electrode degradation is more linear
            health_score = min(1.0, relative_performance)
        else:
            # General case
            health_score = min(1.0, relative_performance)

        return max(0.0, health_score)


class StatisticalDegradationDetector(BaseDegradationDetector):
    """Statistical degradation detector using control charts and trend analysis."""

    def __init__(
        self,
        thresholds: DegradationThresholds | None = None,
        min_data_points: int = 50,
        detection_confidence_threshold: float = 0.7,
        control_chart_type: str = "cusum",  # Options: cusum, ewma, shewhart
    ) -> None:
        """Initialize statistical degradation detector.

        Args:
            thresholds: Custom degradation thresholds
            min_data_points: Minimum data points required
            detection_confidence_threshold: Minimum confidence for detection
            control_chart_type: Type of control chart to use

        """
        super().__init__(thresholds, min_data_points, detection_confidence_threshold)
        self.control_chart_type = control_chart_type
        self.control_limits: dict[str, float] = {}
        self.trend_window = self.thresholds.trend_detection_window

    def detect_degradation(
        self,
        data: TimeSeriesData,
        timestamps: Sequence[Timestamp] | None = None,
        **kwargs: Any,
    ) -> tuple[DegradationMetrics, list[DegradationEvent]]:
        """Perform statistical degradation detection.

        Args:
            data: Time series data to analyze
            timestamps: Optional timestamps for data points
            **kwargs: Additional analysis parameters

        Returns:
            Tuple of (DegradationMetrics, List[DegradationEvent])

        """
        if not self.validate_data(data):
            msg = "Invalid input data for degradation detection"
            raise ValueError(msg)

        data_array = np.asarray(data, dtype=np.float64)
        n_samples = len(data_array)

        # Establish baseline if not already done
        if not self.baseline_established:
            baseline_length = min(n_samples // 3, 100)  # Use first third or 100 points
            self.establish_baseline(data_array[:baseline_length])

        # Initialize metrics
        metrics = DegradationMetrics(
            analysis_timestamp=datetime.now(),
            sample_size=n_samples,
            method_used=DetectionMethod.STATISTICAL_PROCESS_CONTROL,
        )

        degradation_events: list[DegradationEvent] = []

        # Perform trend analysis
        trend_results = self._analyze_trends(data_array, timestamps)
        metrics.short_term_trend = trend_results["short_term"]
        metrics.medium_term_trend = trend_results["medium_term"]
        metrics.long_term_trend = trend_results["long_term"]
        metrics.trend_acceleration = trend_results["acceleration"]

        # Perform anomaly detection
        anomaly_results = self._detect_anomalies(data_array)
        metrics.anomaly_score = anomaly_results["score"]
        metrics.anomaly_count = int(anomaly_results["count"])
        metrics.anomaly_severity = anomaly_results["severity"]

        # Perform change point detection
        change_point_results = self._detect_change_points(data_array, timestamps)
        metrics.change_points_detected = change_point_results["change_points"]
        metrics.significant_changes = change_point_results["significant_count"]
        metrics.last_significant_change = change_point_results["last_change"]

        # Calculate health scores
        health_scores = self._calculate_component_health_scores(data_array)
        metrics.biofilm_health_score = health_scores["biofilm"]
        metrics.electrode_health_score = health_scores["electrode"]
        metrics.membrane_health_score = health_scores["membrane"]

        # Calculate degradation percentages
        degradation_percentages = self._calculate_degradation_percentages(data_array)
        metrics.power_degradation_percentage = degradation_percentages["power"]
        metrics.efficiency_degradation_percentage = degradation_percentages[
            "efficiency"
        ]
        metrics.output_stability_degradation = degradation_percentages["stability"]

        # Calculate degradation rate
        metrics.degradation_rate = self._calculate_degradation_rate(data_array)

        # Predict failure and remaining useful life
        failure_prediction = self._predict_failure(data_array)
        metrics.predicted_failure_time = failure_prediction["failure_time"]
        metrics.prediction_confidence = failure_prediction["confidence"]
        metrics.remaining_useful_life = failure_prediction["remaining_life"]
        metrics.time_to_critical_degradation = failure_prediction["time_to_critical"]

        # Calculate data quality score
        metrics.data_quality_score = self._assess_data_quality(data_array)

        # Generate degradation events based on analysis
        degradation_events = self._generate_degradation_events(
            data_array,
            metrics,
            timestamps,
        )

        return metrics, degradation_events

    def predict_degradation(
        self,
        current_time: float,
        prediction_horizon: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Predict future degradation using statistical models.

        Args:
            current_time: Current system time
            prediction_horizon: Time horizon for prediction
            **kwargs: Additional prediction parameters

        Returns:
            Dictionary with prediction results

        """
        if not self.baseline_established:
            return {
                "predicted_health_scores": [],
                "predicted_degradation_events": [],
                "confidence": 0.0,
                "prediction_horizon": prediction_horizon,
            }

        # Use trend extrapolation for prediction
        current_trend = self.baseline_stats.get("trend", 0.0)
        current_mean = self.baseline_stats.get("mean", 1.0)

        # Generate prediction time points
        time_points = np.linspace(current_time, current_time + prediction_horizon, 100)
        predicted_values = current_mean + current_trend * (time_points - current_time)

        # Calculate predicted health scores
        predicted_health_scores = [
            self._calculate_health_score(value, current_mean)
            for value in predicted_values
        ]

        # Identify predicted degradation events
        predicted_events = []
        for i, (time_point, health_score) in enumerate(
            zip(time_points, predicted_health_scores, strict=False),
        ):
            if health_score < 0.8 and i > 0:  # Degradation threshold
                if predicted_health_scores[i - 1] >= 0.8:  # First time below threshold
                    severity = self._determine_severity_from_health_score(health_score)
                    predicted_events.append(
                        {
                            "time": time_point,
                            "health_score": health_score,
                            "severity": severity.name,
                            "type": "PREDICTED_DEGRADATION",
                        },
                    )

        # Calculate overall prediction confidence
        trend_consistency = 1.0 - abs(current_trend) / max(current_mean, 1.0)
        prediction_confidence = max(0.0, min(1.0, trend_consistency))

        return {
            "predicted_health_scores": list(
                zip(time_points.tolist(), predicted_health_scores, strict=False),
            ),
            "predicted_degradation_events": predicted_events,
            "confidence": prediction_confidence,
            "prediction_horizon": prediction_horizon,
            "baseline_trend": current_trend,
            "current_health_estimate": (
                predicted_health_scores[0] if predicted_health_scores else 1.0
            ),
        }

    def _analyze_trends(
        self,
        data: np.ndarray,
        timestamps: Sequence[Timestamp] | None = None,
    ) -> dict[str, float]:
        """Analyze trends at different time scales."""
        n = len(data)

        # Short-term trend (last 24 data points or 10% of data)
        short_term_window = min(24, max(10, n // 10))
        short_term_data = data[-short_term_window:]
        short_term_trend = float(
            np.polyfit(np.arange(len(short_term_data)), short_term_data, 1)[0],
        )

        # Medium-term trend (last week equivalent or 30% of data)
        medium_term_window = min(168, max(30, n // 3))  # Assuming hourly data
        medium_term_data = data[-medium_term_window:]
        medium_term_trend = float(
            np.polyfit(np.arange(len(medium_term_data)), medium_term_data, 1)[0],
        )

        # Long-term trend (all data)
        long_term_trend = float(np.polyfit(np.arange(n), data, 1)[0])

        # Trend acceleration (change in trend)
        if n > 50:
            first_half = data[: n // 2]
            second_half = data[n // 2 :]
            first_trend = np.polyfit(np.arange(len(first_half)), first_half, 1)[0]
            second_trend = np.polyfit(np.arange(len(second_half)), second_half, 1)[0]
            acceleration = float(second_trend - first_trend)
        else:
            acceleration = 0.0

        return {
            "short_term": short_term_trend,
            "medium_term": medium_term_trend,
            "long_term": long_term_trend,
            "acceleration": acceleration,
        }

    def _detect_anomalies(self, data: np.ndarray) -> dict[str, float]:
        """Detect anomalies using statistical methods."""
        # Calculate z-scores
        mean_val = np.mean(data)
        std_val = np.std(data)

        if std_val == 0:
            return {"score": 0.0, "count": 0, "severity": 0.0}

        z_scores = np.abs((data - mean_val) / std_val)

        # Identify anomalies (beyond threshold standard deviations)
        threshold = self.thresholds.anomaly_detection_sensitivity
        anomalies = z_scores > threshold

        anomaly_count = int(np.sum(anomalies))
        anomaly_score = float(np.mean(z_scores))
        anomaly_severity = (
            float(np.mean(z_scores[anomalies])) if anomaly_count > 0 else 0.0
        )

        return {
            "score": anomaly_score,
            "count": anomaly_count,
            "severity": anomaly_severity,
        }

    def _detect_change_points(
        self,
        data: np.ndarray,
        timestamps: Sequence[Timestamp] | None = None,
    ) -> dict[str, Any]:
        """Detect change points in the data."""
        # Simple change point detection using CUSUM
        try:
            # Calculate cumulative sum of deviations from mean
            mean_val = np.mean(data)
            deviations = data - mean_val
            cusum_pos = np.maximum.accumulate(
                np.maximum(0, np.cumsum(deviations - 0.5 * np.std(data))),
            )
            cusum_neg = np.maximum.accumulate(
                np.maximum(0, np.cumsum(-deviations - 0.5 * np.std(data))),
            )

            # Threshold for change detection
            threshold = 4 * np.std(data)

            # Find change points
            change_points = []
            pos_changes = np.where(cusum_pos > threshold)[0]
            neg_changes = np.where(cusum_neg > threshold)[0]

            for change_idx in np.concatenate([pos_changes, neg_changes]):
                if change_idx < len(data):
                    magnitude = abs(data[change_idx] - mean_val)
                    change_points.append((float(change_idx), float(magnitude)))

            # Remove duplicate nearby change points
            if change_points:
                change_points = sorted(set(change_points), key=lambda x: x[0])
                filtered_points = [change_points[0]]
                for point in change_points[1:]:
                    if (
                        point[0] - filtered_points[-1][0] > 10
                    ):  # Minimum distance between changes
                        filtered_points.append(point)
                change_points = filtered_points

            # Count significant changes (above certain magnitude)
            significant_threshold = 2 * np.std(data)
            significant_changes = sum(
                1 for _, magnitude in change_points if magnitude > significant_threshold
            )

            # Find last significant change
            last_change = None
            if change_points and timestamps:
                for time_idx, magnitude in reversed(change_points):
                    if magnitude > significant_threshold and int(time_idx) < len(
                        timestamps,
                    ):
                        if isinstance(timestamps[int(time_idx)], datetime):
                            last_change = timestamps[int(time_idx)]
                        else:
                            last_change = datetime.now() - timedelta(
                                hours=len(data) - time_idx,
                            )
                        break

            return {
                "change_points": change_points,
                "significant_count": significant_changes,
                "last_change": last_change,
            }

        except Exception as e:
            self.logger.exception(f"Change point detection failed: {e!s}")
            return {"change_points": [], "significant_count": 0, "last_change": None}

    def _calculate_component_health_scores(self, data: np.ndarray) -> dict[str, float]:
        """Calculate health scores for different MFC components."""
        if not self.baseline_established:
            return {"biofilm": 1.0, "electrode": 1.0, "membrane": 1.0}

        current_mean = np.mean(data[-min(24, len(data)) :])  # Recent average
        baseline_mean = self.baseline_stats["mean"]

        # Calculate component-specific health scores
        biofilm_health = self._calculate_health_score(
            current_mean,
            baseline_mean,
            "biofilm",
        )
        electrode_health = self._calculate_health_score(
            current_mean,
            baseline_mean,
            "electrode",
        )
        membrane_health = self._calculate_health_score(
            current_mean,
            baseline_mean,
            "general",
        )

        # Add noise factors based on data variability
        current_std = np.std(data[-min(24, len(data)) :])
        baseline_std = self.baseline_stats["std"]

        stability_factor = min(1.0, baseline_std / max(current_std, 0.001))

        return {
            "biofilm": float(biofilm_health * stability_factor),
            "electrode": float(electrode_health * stability_factor),
            "membrane": float(membrane_health * stability_factor),
        }

    def _calculate_degradation_percentages(self, data: np.ndarray) -> dict[str, float]:
        """Calculate degradation percentages for different metrics."""
        if not self.baseline_established:
            return {"power": 0.0, "efficiency": 0.0, "stability": 0.0}

        current_mean = np.mean(data[-min(24, len(data)) :])
        baseline_mean = self.baseline_stats["mean"]

        current_std = np.std(data[-min(24, len(data)) :])
        baseline_std = self.baseline_stats["std"]

        # Power degradation (assuming data represents power output)
        power_degradation = (
            max(0.0, (baseline_mean - current_mean) / baseline_mean)
            if baseline_mean > 0
            else 0.0
        )

        # Efficiency degradation (similar to power)
        efficiency_degradation = power_degradation

        # Stability degradation (increased variability)
        stability_degradation = (
            max(0.0, (current_std - baseline_std) / baseline_std)
            if baseline_std > 0
            else 0.0
        )

        return {
            "power": float(power_degradation),
            "efficiency": float(efficiency_degradation),
            "stability": float(stability_degradation),
        }

    def _calculate_degradation_rate(self, data: np.ndarray) -> float:
        """Calculate overall degradation rate."""
        if len(data) < 10:
            return 0.0

        # Use trend slope as degradation rate
        trend_slope = np.polyfit(np.arange(len(data)), data, 1)[0]

        # Normalize by baseline value
        if self.baseline_established:
            baseline_mean = self.baseline_stats["mean"]
            normalized_rate = -trend_slope / baseline_mean if baseline_mean > 0 else 0.0
        else:
            data_mean = np.mean(data)
            normalized_rate = -trend_slope / data_mean if data_mean > 0 else 0.0

        return max(0.0, float(normalized_rate))  # Only positive degradation rates

    def _predict_failure(self, data: np.ndarray) -> dict[str, Any]:
        """Predict failure time and remaining useful life."""
        try:
            current_trend = np.polyfit(np.arange(len(data)), data, 1)[0]
            current_value = data[-1]

            # Critical threshold (20% of baseline or minimum operational level)
            if self.baseline_established:
                critical_threshold = 0.2 * self.baseline_stats["mean"]
            else:
                critical_threshold = 0.2 * np.mean(data[: min(50, len(data))])

            # Calculate time to reach critical threshold
            if current_trend < 0 and current_value > critical_threshold:
                time_to_critical = (current_value - critical_threshold) / abs(
                    current_trend,
                )
                failure_time = datetime.now() + timedelta(hours=time_to_critical)
                remaining_life = time_to_critical
                confidence = min(1.0, abs(current_trend) / (0.01 * current_value))
            else:
                time_to_critical = float("inf")
                failure_time = None
                remaining_life = float("inf")
                confidence = 0.0

            return {
                "failure_time": failure_time,
                "remaining_life": float(remaining_life),
                "time_to_critical": float(time_to_critical),
                "confidence": float(confidence),
            }

        except Exception as e:
            self.logger.exception(f"Failure prediction failed: {e!s}")
            return {
                "failure_time": None,
                "remaining_life": float("inf"),
                "time_to_critical": float("inf"),
                "confidence": 0.0,
            }

    def _assess_data_quality(self, data: np.ndarray) -> float:
        """Assess quality of input data."""
        quality_factors = []

        # Completeness (no missing values)
        completeness = 1.0 - (np.sum(np.isnan(data)) / len(data))
        quality_factors.append(completeness)

        # Consistency (low number of extreme outliers)
        z_scores = np.abs((data - np.mean(data)) / (np.std(data) + 1e-10))
        outlier_ratio = np.sum(z_scores > 3) / len(data)
        consistency = 1.0 - min(outlier_ratio, 1.0)
        quality_factors.append(consistency)

        # Temporal stability (reasonable rate of change)
        if len(data) > 1:
            rate_of_change = np.abs(np.diff(data))
            mean_change = np.mean(rate_of_change)
            data_range = np.max(data) - np.min(data)
            stability = 1.0 - min(mean_change / (data_range + 1e-10), 1.0)
            quality_factors.append(stability)

        return float(np.mean(quality_factors))

    def _generate_degradation_events(
        self,
        data: np.ndarray,
        metrics: DegradationMetrics,
        timestamps: Sequence[Timestamp] | None = None,
    ) -> list[DegradationEvent]:
        """Generate degradation events based on analysis results."""
        events = []

        try:
            # Check if overall degradation is significant
            if metrics.overall_health_score < 0.9:
                severity = self._determine_severity_from_health_score(
                    metrics.overall_health_score,
                )

                event = DegradationEvent(
                    detection_time=datetime.now(),
                    degradation_type=DegradationType.BIOFILM_AGING,  # Default primary concern
                    severity=severity,
                    affected_component="MFC System",
                    degradation_rate=metrics.degradation_rate,
                    confidence=0.8,
                    baseline_value=self.baseline_stats.get("mean", 1.0),
                    current_value=float(np.mean(data[-10:])),
                    trend_slope=metrics.long_term_trend,
                    detection_method=DetectionMethod.STATISTICAL_PROCESS_CONTROL,
                    additional_metrics={
                        "health_score": metrics.overall_health_score,
                        "anomaly_score": metrics.anomaly_score,
                        "trend_acceleration": metrics.trend_acceleration,
                    },
                    recommended_actions=self._generate_recommendations(metrics),
                )
                events.append(event)

            # Check for component-specific degradation
            if metrics.biofilm_health_score < 0.8:
                events.append(
                    self._create_component_degradation_event(
                        "Biofilm",
                        DegradationType.BIOFILM_AGING,
                        metrics.biofilm_health_score,
                        data,
                        metrics,
                    ),
                )

            if metrics.electrode_health_score < 0.8:
                events.append(
                    self._create_component_degradation_event(
                        "Electrode",
                        DegradationType.ELECTRODE_CORROSION,
                        metrics.electrode_health_score,
                        data,
                        metrics,
                    ),
                )

            # Check for anomalies
            if metrics.anomaly_count > 5:
                events.append(self._create_anomaly_event(data, metrics))

        except Exception as e:
            self.logger.exception(f"Event generation failed: {e!s}")

        return events

    def _determine_severity_from_health_score(
        self,
        health_score: float,
    ) -> DegradationSeverity:
        """Determine degradation severity from health score."""
        if health_score >= 0.9:
            return DegradationSeverity.MINIMAL
        if health_score >= 0.8:
            return DegradationSeverity.MINOR
        if health_score >= 0.6:
            return DegradationSeverity.MODERATE
        if health_score >= 0.4:
            return DegradationSeverity.SEVERE
        return DegradationSeverity.CRITICAL

    def _create_component_degradation_event(
        self,
        component_name: str,
        degradation_type: DegradationType,
        health_score: float,
        data: np.ndarray,
        metrics: DegradationMetrics,
    ) -> DegradationEvent:
        """Create degradation event for specific component."""
        severity = self._determine_severity_from_health_score(health_score)

        return DegradationEvent(
            detection_time=datetime.now(),
            degradation_type=degradation_type,
            severity=severity,
            affected_component=component_name,
            degradation_rate=metrics.degradation_rate,
            confidence=0.75,
            baseline_value=self.baseline_stats.get("mean", 1.0),
            current_value=float(np.mean(data[-10:])),
            trend_slope=metrics.long_term_trend,
            detection_method=DetectionMethod.STATISTICAL_PROCESS_CONTROL,
            additional_metrics={"health_score": health_score},
            recommended_actions=[
                f"Inspect {component_name.lower()}",
                f"Consider {component_name.lower()} maintenance",
            ],
        )

    def _create_anomaly_event(
        self,
        data: np.ndarray,
        metrics: DegradationMetrics,
    ) -> DegradationEvent:
        """Create event for anomaly detection."""
        return DegradationEvent(
            detection_time=datetime.now(),
            degradation_type=DegradationType.MICROBIAL_DECLINE,
            severity=DegradationSeverity.MODERATE,
            affected_component="System Performance",
            degradation_rate=0.0,
            confidence=metrics.anomaly_score / 3.0,  # Normalize to 0-1
            baseline_value=self.baseline_stats.get("mean", 1.0),
            current_value=float(np.mean(data[-10:])),
            trend_slope=0.0,
            detection_method=DetectionMethod.ANOMALY_DETECTION,
            additional_metrics={
                "anomaly_count": metrics.anomaly_count,
                "anomaly_severity": metrics.anomaly_severity,
            },
            recommended_actions=[
                "Investigate anomalous behavior",
                "Check system parameters",
            ],
        )

    def _generate_recommendations(self, metrics: DegradationMetrics) -> list[str]:
        """Generate maintenance recommendations based on degradation metrics."""
        recommendations = []

        if metrics.overall_health_score < 0.7:
            recommendations.append("Schedule comprehensive system inspection")

        if metrics.biofilm_health_score < 0.8:
            recommendations.append("Consider biofilm cleaning or replacement")

        if metrics.electrode_health_score < 0.8:
            recommendations.append("Inspect electrodes for corrosion or fouling")

        if metrics.degradation_rate > 0.01:  # 1% per hour
            recommendations.append("Investigate rapid degradation causes")

        if metrics.anomaly_count > 10:
            recommendations.append("Review operational parameters for anomalies")

        if not recommendations:
            recommendations.append("Continue normal monitoring")

        return recommendations


# Factory functions
def create_statistical_degradation_detector(
    thresholds: DegradationThresholds | None = None,
    **kwargs: Any,
) -> StatisticalDegradationDetector:
    """Create a statistical degradation detector."""
    return StatisticalDegradationDetector(thresholds=thresholds, **kwargs)


# Example usage and testing
def run_example_degradation_detection() -> None:
    """Run example degradation detection."""
    # Generate synthetic MFC data with degradation
    np.random.seed(42)

    # Create baseline data (healthy operation)
    baseline_hours = 100
    baseline_data = 20.0 + np.random.normal(0, 0.5, baseline_hours)  # Stable ~20W

    # Create degrading data
    degradation_hours = 200
    time_points = np.arange(degradation_hours)

    # Add gradual degradation
    degradation_trend = -0.02 * time_points  # 2% degradation per 100 hours

    # Add random events
    random_events = np.zeros(degradation_hours)
    random_events[50:60] = -2.0  # Sudden drop at hour 50-60
    random_events[120:125] = -1.5  # Another drop

    # Add noise
    noise = np.random.normal(0, 0.3, degradation_hours)

    degraded_data = 20.0 + degradation_trend + random_events + noise

    # Combine baseline and degraded data
    full_data = np.concatenate([baseline_data, degraded_data])

    # Create detector and analyze
    detector = create_statistical_degradation_detector()

    # Establish baseline
    detector.establish_baseline(baseline_data)

    # Detect degradation
    metrics, events = detector.detect_degradation(full_data)

    for _i, _event in enumerate(events):
        pass

    # Test prediction
    detector.predict_degradation(
        current_time=len(full_data),
        prediction_horizon=100.0,
    )


if __name__ == "__main__":
    run_example_degradation_detection()
