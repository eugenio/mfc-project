#!/usr/bin/env python3
"""Reliability Analysis Module for MFC Systems.

This module provides comprehensive reliability analysis for Microbial Fuel Cell (MFC)
systems, including failure rate analysis, Mean Time Between Failures (MTBF) calculation,
system availability assessment, and predictive reliability modeling.

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
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Sequence

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
TimeSeriesData = Union[pd.Series, np.ndarray, list[float]]
Timestamp = Union[datetime, pd.Timestamp, float]
ReliabilityValue = Union[float, np.floating]
FailureTime = Union[float, int, np.number]

# Generic types
T = TypeVar("T")


class FailureMode(Enum):
    """Enumeration of MFC failure modes."""

    BIOFILM_DEGRADATION = auto()
    MEMBRANE_FOULING = auto()
    ELECTRODE_CORROSION = auto()
    SYSTEM_CONTAMINATION = auto()
    TEMPERATURE_SHOCK = auto()
    PH_IMBALANCE = auto()
    POWER_FLUCTUATION = auto()
    MECHANICAL_FAILURE = auto()
    UNKNOWN = auto()

    def __str__(self) -> str:
        return self.name.lower().replace("_", " ")


class ReliabilityModel(Enum):
    """Enumeration of reliability models."""

    EXPONENTIAL = auto()
    WEIBULL = auto()
    LOGNORMAL = auto()
    GAMMA = auto()
    BETA = auto()


class MaintenanceStrategy(Enum):
    """Enumeration of maintenance strategies."""

    REACTIVE = auto()
    PREVENTIVE = auto()
    PREDICTIVE = auto()
    CONDITION_BASED = auto()


@dataclass(frozen=True)
class FailureEvent:
    """Represents a failure event in the MFC system."""

    failure_time: float
    failure_mode: FailureMode
    severity: float  # 0.0 to 1.0
    recovery_time: float  # Hours to recovery
    root_cause: str = ""
    operator_notes: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate failure event data."""
        if not (0.0 <= self.severity <= 1.0):
            msg = "Severity must be between 0.0 and 1.0"
            raise ValueError(msg)
        if self.failure_time < 0:
            msg = "Failure time must be non-negative"
            raise ValueError(msg)
        if self.recovery_time < 0:
            msg = "Recovery time must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class ReliabilityParameters:
    """Parameters for reliability distribution models."""

    model_type: ReliabilityModel
    parameters: dict[str, float]
    confidence_level: float = 0.95
    goodness_of_fit: float = 0.0

    def __post_init__(self) -> None:
        """Validate reliability parameters."""
        if not (0.0 < self.confidence_level < 1.0):
            msg = "Confidence level must be between 0 and 1"
            raise ValueError(msg)


@dataclass
class ReliabilityMetrics:
    """Reliability metrics for MFC system components."""

    # Core reliability metrics
    failure_rate: float = 0.0  # λ (failures per unit time)
    mean_time_to_failure: float = 0.0  # MTTF in hours
    reliability_coefficient: float = 0.0  # R(t) at specific time
    availability: float = 0.0  # System availability (0-1)

    # Additional metrics
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    sample_size: int = 0
    observation_period: float = 0.0  # hours

    def __post_init__(self) -> None:
        """Validate reliability metrics."""
        if self.failure_rate < 0:
            msg = "Failure rate must be non-negative"
            raise ValueError(msg)
        if self.mean_time_to_failure < 0:
            msg = "MTTF must be non-negative"
            raise ValueError(msg)
        if not (0.0 <= self.availability <= 1.0):
            msg = "Availability must be between 0.0 and 1.0"
            raise ValueError(msg)


@dataclass
class ReliabilityPrediction:
    """Comprehensive reliability prediction results."""

    # Basic reliability metrics
    mean_time_to_failure: float = 0.0  # MTTF in hours
    mean_time_between_failures: float = 0.0  # MTBF in hours
    mean_time_to_repair: float = 0.0  # MTTR in hours
    availability: float = 0.0  # System availability (0-1)

    # Failure rate metrics
    instantaneous_failure_rate: float = 0.0  # λ(t) at current time
    average_failure_rate: float = 0.0  # Average λ
    failure_rate_trend: float = 0.0  # Trend in failure rate

    # Reliability functions
    reliability_at_time: dict[float, float] = field(default_factory=dict)  # R(t)
    hazard_function: dict[float, float] = field(default_factory=dict)  # h(t)
    cumulative_hazard: dict[float, float] = field(default_factory=dict)  # H(t)

    # Model parameters
    distribution_parameters: ReliabilityParameters | None = None
    model_confidence: float = 0.0
    prediction_horizon: float = 8760.0  # Default: 1 year in hours

    # Failure mode analysis
    dominant_failure_modes: list[tuple[FailureMode, float]] = field(
        default_factory=list,
    )
    failure_mode_probabilities: dict[FailureMode, float] = field(default_factory=dict)

    # Maintenance recommendations
    recommended_maintenance_interval: float = 0.0  # Hours
    maintenance_strategy: MaintenanceStrategy = MaintenanceStrategy.PREVENTIVE
    cost_benefit_ratio: float = 0.0

    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    data_period: timedelta = field(default=timedelta(hours=1))
    sample_size: int = 0
    prediction_accuracy: float = 0.0

    def __post_init__(self) -> None:
        """Validate and compute derived metrics."""
        self._validate_metrics()
        self._compute_derived_metrics()

    def _validate_metrics(self) -> None:
        """Validate reliability metrics."""
        if not (0.0 <= self.availability <= 1.0):
            logger.warning(
                f"Availability {self.availability} outside expected range [0, 1]",
            )

        if self.mean_time_to_failure < 0:
            logger.warning("MTTF cannot be negative")

        if self.mean_time_between_failures < 0:
            logger.warning("MTBF cannot be negative")

    def _compute_derived_metrics(self) -> None:
        """Compute derived reliability metrics."""
        # Calculate availability if MTBF and MTTR are available
        if self.mean_time_between_failures > 0 and self.mean_time_to_repair >= 0:
            mtbf = self.mean_time_between_failures
            mttr = self.mean_time_to_repair
            self.availability = mtbf / (mtbf + mttr)

    def get_reliability_at_time(self, time: float) -> float:
        """Get reliability value at specific time."""
        if time in self.reliability_at_time:
            return self.reliability_at_time[time]

        # Interpolate if exact time not available
        times = sorted(self.reliability_at_time.keys())
        if not times:
            return 1.0  # Default reliability

        if time <= times[0]:
            return self.reliability_at_time[times[0]]
        if time >= times[-1]:
            return self.reliability_at_time[times[-1]]

        # Linear interpolation
        for i in range(len(times) - 1):
            if times[i] <= time <= times[i + 1]:
                t1, t2 = times[i], times[i + 1]
                r1, r2 = self.reliability_at_time[t1], self.reliability_at_time[t2]
                return r1 + (r2 - r1) * (time - t1) / (t2 - t1)

        return 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction to dictionary format."""
        return {
            "mean_time_to_failure": self.mean_time_to_failure,
            "mean_time_between_failures": self.mean_time_between_failures,
            "mean_time_to_repair": self.mean_time_to_repair,
            "availability": self.availability,
            "instantaneous_failure_rate": self.instantaneous_failure_rate,
            "average_failure_rate": self.average_failure_rate,
            "failure_rate_trend": self.failure_rate_trend,
            "model_confidence": self.model_confidence,
            "prediction_horizon": self.prediction_horizon,
            "dominant_failure_modes": [
                (mode.name, prob) for mode, prob in self.dominant_failure_modes
            ],
            "failure_mode_probabilities": {
                mode.name: prob
                for mode, prob in self.failure_mode_probabilities.items()
            },
            "recommended_maintenance_interval": self.recommended_maintenance_interval,
            "maintenance_strategy": self.maintenance_strategy.name,
            "cost_benefit_ratio": self.cost_benefit_ratio,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "data_period": self.data_period.total_seconds(),
            "sample_size": self.sample_size,
            "prediction_accuracy": self.prediction_accuracy,
        }


class ReliabilityAnalyzer(Protocol):
    """Protocol for reliability analyzer implementations."""

    def analyze_reliability(
        self,
        failure_data: Sequence[FailureEvent],
        operational_data: TimeSeriesData | None = None,
        **kwargs: Any,
    ) -> ReliabilityPrediction:
        """Analyze system reliability."""
        ...

    def predict_next_failure(
        self,
        current_time: float,
        **kwargs: Any,
    ) -> tuple[float, float]:  # (predicted_time, confidence)
        """Predict next failure time."""
        ...


class BaseReliabilityAnalyzer(ABC):
    """Base class for reliability analyzers."""

    def __init__(
        self,
        confidence_level: float = 0.95,
        prediction_horizon: float = 8760.0,  # 1 year in hours
        min_failures_for_analysis: int = 3,
    ) -> None:
        """Initialize reliability analyzer.

        Args:
            confidence_level: Confidence level for statistical analysis
            prediction_horizon: Time horizon for predictions (hours)
            min_failures_for_analysis: Minimum failures needed for analysis

        """
        self.confidence_level = confidence_level
        self.prediction_horizon = prediction_horizon
        self.min_failures_for_analysis = min_failures_for_analysis
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging for the analyzer."""
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def analyze_reliability(
        self,
        failure_data: Sequence[FailureEvent],
        operational_data: TimeSeriesData | None = None,
        **kwargs: Any,
    ) -> ReliabilityPrediction:
        """Analyze system reliability."""

    @abstractmethod
    def predict_next_failure(
        self,
        current_time: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Predict next failure time."""

    def validate_failure_data(self, failure_data: Sequence[FailureEvent]) -> bool:
        """Validate failure data quality.

        Args:
            failure_data: Sequence of failure events

        Returns:
            True if data is valid, False otherwise

        """
        try:
            if len(failure_data) < self.min_failures_for_analysis:
                self.logger.warning(
                    f"Insufficient failure data: {len(failure_data)} < {self.min_failures_for_analysis}",
                )
                return False

            # Check for chronological order
            failure_times = [event.failure_time for event in failure_data]
            if failure_times != sorted(failure_times):
                self.logger.warning("Failure data not in chronological order")

            # Check for reasonable failure times
            if any(time < 0 for time in failure_times):
                self.logger.error("Negative failure times detected")
                return False

            return True

        except Exception as e:
            self.logger.exception(f"Failure data validation error: {e!s}")
            return False

    def _calculate_time_between_failures(
        self,
        failure_data: Sequence[FailureEvent],
    ) -> np.ndarray:
        """Calculate time between consecutive failures."""
        failure_times = sorted([event.failure_time for event in failure_data])

        if len(failure_times) < 2:
            return np.array([])

        return np.diff(failure_times)

    def _fit_reliability_distribution(
        self,
        failure_times: np.ndarray,
        distribution_type: ReliabilityModel = ReliabilityModel.WEIBULL,
    ) -> ReliabilityParameters:
        """Fit reliability distribution to failure data."""
        if len(failure_times) < 2:
            return ReliabilityParameters(
                model_type=distribution_type,
                parameters={},
                goodness_of_fit=0.0,
            )

        try:
            if distribution_type == ReliabilityModel.EXPONENTIAL:
                # Exponential distribution: single parameter (lambda)
                rate = 1.0 / np.mean(failure_times)
                params = {"lambda": rate}

                # Kolmogorov-Smirnov test for goodness of fit
                ks_stat, p_value = stats.kstest(
                    failure_times,
                    stats.expon(scale=1 / rate).cdf,
                )
                goodness_of_fit = 1.0 - ks_stat

            elif distribution_type == ReliabilityModel.WEIBULL:
                # Weibull distribution: shape (k) and scale (lambda) parameters
                params_fit = stats.weibull_min.fit(failure_times, floc=0)
                shape, loc, scale = params_fit
                params = {"shape": shape, "scale": scale}

                # Goodness of fit test
                ks_stat, p_value = stats.kstest(
                    failure_times,
                    lambda x: stats.weibull_min.cdf(x, shape, loc, scale),
                )
                goodness_of_fit = 1.0 - ks_stat

            elif distribution_type == ReliabilityModel.LOGNORMAL:
                # Lognormal distribution
                params_fit = stats.lognorm.fit(failure_times, floc=0)
                s, loc, scale = params_fit
                params = {"sigma": s, "mu": np.log(scale)}

                # Goodness of fit test
                ks_stat, p_value = stats.kstest(
                    failure_times,
                    lambda x: stats.lognorm.cdf(x, s, loc, scale),
                )
                goodness_of_fit = 1.0 - ks_stat

            else:
                # Default to exponential if unsupported distribution
                rate = 1.0 / np.mean(failure_times)
                params = {"lambda": rate}
                goodness_of_fit = 0.5

            return ReliabilityParameters(
                model_type=distribution_type,
                parameters=params,
                confidence_level=self.confidence_level,
                goodness_of_fit=max(0.0, goodness_of_fit),
            )

        except Exception as e:
            self.logger.exception(f"Distribution fitting failed: {e!s}")
            # Return default exponential parameters
            rate = 1.0 / np.mean(failure_times) if len(failure_times) > 0 else 0.001
            return ReliabilityParameters(
                model_type=ReliabilityModel.EXPONENTIAL,
                parameters={"lambda": rate},
                goodness_of_fit=0.0,
            )


class StatisticalReliabilityAnalyzer(BaseReliabilityAnalyzer):
    """Statistical reliability analyzer using classical reliability engineering methods."""

    def __init__(
        self,
        confidence_level: float = 0.95,
        prediction_horizon: float = 8760.0,
        min_failures_for_analysis: int = 3,
        default_distribution: ReliabilityModel = ReliabilityModel.WEIBULL,
    ) -> None:
        """Initialize statistical reliability analyzer.

        Args:
            confidence_level: Confidence level for statistical analysis
            prediction_horizon: Time horizon for predictions (hours)
            min_failures_for_analysis: Minimum failures needed for analysis
            default_distribution: Default reliability distribution model

        """
        super().__init__(
            confidence_level,
            prediction_horizon,
            min_failures_for_analysis,
        )
        self.default_distribution = default_distribution
        self.fitted_distribution: ReliabilityParameters | None = None

    def analyze_reliability(
        self,
        failure_data: Sequence[FailureEvent],
        operational_data: TimeSeriesData | None = None,
        **kwargs: Any,
    ) -> ReliabilityPrediction:
        """Perform statistical reliability analysis.

        Args:
            failure_data: Sequence of failure events
            operational_data: Optional operational performance data
            **kwargs: Additional analysis parameters

        Returns:
            ReliabilityPrediction with comprehensive reliability metrics

        """
        if not self.validate_failure_data(failure_data):
            msg = "Invalid failure data for reliability analysis"
            raise ValueError(msg)

        # Extract failure times and calculate time between failures
        failure_times = np.array([event.failure_time for event in failure_data])
        time_between_failures = self._calculate_time_between_failures(failure_data)

        # Fit reliability distribution
        self.fitted_distribution = self._fit_reliability_distribution(
            time_between_failures,
            kwargs.get("distribution_type", self.default_distribution),
        )

        # Calculate basic reliability metrics
        mttf = self._calculate_mttf(time_between_failures)
        mtbf = self._calculate_mtbf(time_between_failures)
        mttr = self._calculate_mttr(failure_data)
        availability = self._calculate_availability(mtbf, mttr)

        # Calculate failure rate metrics
        avg_failure_rate = self._calculate_average_failure_rate(time_between_failures)
        instantaneous_failure_rate = self._calculate_instantaneous_failure_rate(
            failure_times,
        )
        failure_rate_trend = self._calculate_failure_rate_trend(failure_times)

        # Generate reliability functions
        time_points = np.linspace(0, self.prediction_horizon, 100)
        reliability_function = self._calculate_reliability_function(time_points)
        hazard_function = self._calculate_hazard_function(time_points)
        cumulative_hazard = self._calculate_cumulative_hazard_function(time_points)

        # Analyze failure modes
        failure_mode_analysis = self._analyze_failure_modes(failure_data)

        # Generate maintenance recommendations
        maintenance_interval = self._recommend_maintenance_interval(mtbf, mttr)
        maintenance_strategy = self._recommend_maintenance_strategy(failure_data)
        cost_benefit_ratio = self._calculate_cost_benefit_ratio(
            mtbf,
            mttr,
            maintenance_interval,
        )

        # Calculate prediction accuracy
        prediction_accuracy = self._assess_prediction_accuracy(failure_data)

        return ReliabilityPrediction(
            mean_time_to_failure=mttf,
            mean_time_between_failures=mtbf,
            mean_time_to_repair=mttr,
            availability=availability,
            instantaneous_failure_rate=instantaneous_failure_rate,
            average_failure_rate=avg_failure_rate,
            failure_rate_trend=failure_rate_trend,
            reliability_at_time=dict(
                zip(time_points, reliability_function, strict=False),
            ),
            hazard_function=dict(zip(time_points, hazard_function, strict=False)),
            cumulative_hazard=dict(zip(time_points, cumulative_hazard, strict=False)),
            distribution_parameters=self.fitted_distribution,
            model_confidence=(
                self.fitted_distribution.goodness_of_fit
                if self.fitted_distribution
                else 0.0
            ),
            prediction_horizon=self.prediction_horizon,
            dominant_failure_modes=failure_mode_analysis["dominant_modes"],
            failure_mode_probabilities=failure_mode_analysis["mode_probabilities"],
            recommended_maintenance_interval=maintenance_interval,
            maintenance_strategy=maintenance_strategy,
            cost_benefit_ratio=cost_benefit_ratio,
            analysis_timestamp=datetime.now(),
            data_period=(
                timedelta(hours=failure_times[-1] - failure_times[0])
                if len(failure_times) > 1
                else timedelta(hours=1)
            ),
            sample_size=len(failure_data),
            prediction_accuracy=prediction_accuracy,
        )

    def predict_next_failure(
        self,
        current_time: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Predict next failure time using fitted distribution.

        Args:
            current_time: Current system time
            **kwargs: Additional prediction parameters

        Returns:
            Tuple of (predicted_failure_time, confidence_level)

        """
        if not self.fitted_distribution or not self.fitted_distribution.parameters:
            return (current_time + 1000.0, 0.0)  # Default prediction

        try:
            params = self.fitted_distribution.parameters

            if self.fitted_distribution.model_type == ReliabilityModel.EXPONENTIAL:
                # Exponential distribution prediction
                lambda_rate = params["lambda"]
                mean_time_to_next_failure = 1.0 / lambda_rate
                predicted_time = current_time + mean_time_to_next_failure
                confidence = self.fitted_distribution.goodness_of_fit

            elif self.fitted_distribution.model_type == ReliabilityModel.WEIBULL:
                # Weibull distribution prediction
                shape = params["shape"]
                scale = params["scale"]

                # Calculate conditional reliability and predict next failure
                current_reliability = self._weibull_reliability(
                    current_time,
                    shape,
                    scale,
                )
                if current_reliability > 0.01:  # Avoid numerical issues
                    # Use median time to next failure
                    median_ttf = scale * (np.log(2)) ** (1.0 / shape)
                    predicted_time = current_time + median_ttf
                else:
                    predicted_time = current_time + scale

                confidence = self.fitted_distribution.goodness_of_fit

            else:
                # Default prediction for other distributions
                avg_tbf = 1.0 / params.get("lambda", 0.001)
                predicted_time = current_time + avg_tbf
                confidence = 0.5

            return (float(predicted_time), float(confidence))

        except Exception as e:
            self.logger.exception(f"Failure prediction failed: {e!s}")
            return (current_time + 1000.0, 0.0)

    # Helper methods for reliability calculations
    def _calculate_mttf(self, time_between_failures: np.ndarray) -> float:
        """Calculate Mean Time To Failure."""
        if len(time_between_failures) == 0:
            return 0.0
        return float(np.mean(time_between_failures))

    def _calculate_mtbf(self, time_between_failures: np.ndarray) -> float:
        """Calculate Mean Time Between Failures."""
        return self._calculate_mttf(time_between_failures)

    def _calculate_mttr(self, failure_data: Sequence[FailureEvent]) -> float:
        """Calculate Mean Time To Repair."""
        recovery_times = [event.recovery_time for event in failure_data]
        return float(np.mean(recovery_times)) if recovery_times else 0.0

    def _calculate_availability(self, mtbf: float, mttr: float) -> float:
        """Calculate system availability."""
        if mtbf + mttr == 0:
            return 1.0
        return mtbf / (mtbf + mttr)

    def _calculate_average_failure_rate(
        self,
        time_between_failures: np.ndarray,
    ) -> float:
        """Calculate average failure rate."""
        if len(time_between_failures) == 0:
            return 0.0
        return 1.0 / np.mean(time_between_failures)

    def _calculate_instantaneous_failure_rate(self, failure_times: np.ndarray) -> float:
        """Calculate instantaneous failure rate."""
        if len(failure_times) < 2:
            return 0.0

        # Use recent failure data for instantaneous rate
        recent_failures = failure_times[-min(5, len(failure_times)) :]
        if len(recent_failures) < 2:
            return 0.0

        time_span = recent_failures[-1] - recent_failures[0]
        failure_count = len(recent_failures) - 1

        return failure_count / time_span if time_span > 0 else 0.0

    def _calculate_failure_rate_trend(self, failure_times: np.ndarray) -> float:
        """Calculate trend in failure rate over time."""
        if len(failure_times) < 3:
            return 0.0

        # Calculate failure rates in sliding windows
        window_size = max(3, len(failure_times) // 3)
        rates = []
        times = []

        for i in range(window_size, len(failure_times)):
            window_failures = failure_times[i - window_size : i]
            time_span = window_failures[-1] - window_failures[0]
            rate = (window_size - 1) / time_span if time_span > 0 else 0.0
            rates.append(rate)
            times.append(window_failures[-1])

        if len(rates) < 2:
            return 0.0

        # Linear regression to find trend
        return float(np.polyfit(times, rates, 1)[0])

    def _calculate_reliability_function(self, time_points: np.ndarray) -> np.ndarray:
        """Calculate reliability function R(t) for given time points."""
        if not self.fitted_distribution or not self.fitted_distribution.parameters:
            # Default exponential reliability
            return np.exp(-0.001 * time_points)

        params = self.fitted_distribution.parameters

        if self.fitted_distribution.model_type == ReliabilityModel.EXPONENTIAL:
            lambda_rate = params["lambda"]
            return np.exp(-lambda_rate * time_points)

        if self.fitted_distribution.model_type == ReliabilityModel.WEIBULL:
            shape = params["shape"]
            scale = params["scale"]
            return np.exp(-((time_points / scale) ** shape))

        # Default exponential
        return np.exp(-0.001 * time_points)

    def _calculate_hazard_function(self, time_points: np.ndarray) -> np.ndarray:
        """Calculate hazard function h(t) for given time points."""
        if not self.fitted_distribution or not self.fitted_distribution.parameters:
            # Default exponential hazard (constant)
            return np.full_like(time_points, 0.001)

        params = self.fitted_distribution.parameters

        if self.fitted_distribution.model_type == ReliabilityModel.EXPONENTIAL:
            lambda_rate = params["lambda"]
            return np.full_like(time_points, lambda_rate)

        if self.fitted_distribution.model_type == ReliabilityModel.WEIBULL:
            shape = params["shape"]
            scale = params["scale"]
            return (shape / scale) * (time_points / scale) ** (shape - 1)

        # Default constant hazard
        return np.full_like(time_points, 0.001)

    def _calculate_cumulative_hazard_function(
        self,
        time_points: np.ndarray,
    ) -> np.ndarray:
        """Calculate cumulative hazard function H(t)."""
        reliability = self._calculate_reliability_function(time_points)
        # H(t) = -ln(R(t))
        return -np.log(np.maximum(reliability, 1e-10))  # Avoid log(0)

    def _weibull_reliability(self, time: float, shape: float, scale: float) -> float:
        """Calculate Weibull reliability at given time."""
        return float(np.exp(-((time / scale) ** shape)))

    def _analyze_failure_modes(
        self,
        failure_data: Sequence[FailureEvent],
    ) -> dict[str, Any]:
        """Analyze failure modes and their probabilities."""
        failure_modes = [event.failure_mode for event in failure_data]
        mode_counts: dict[FailureMode, int] = {}

        for mode in failure_modes:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        total_failures = len(failure_data)
        mode_probabilities = {
            mode: count / total_failures for mode, count in mode_counts.items()
        }

        # Sort by probability
        dominant_modes = sorted(
            mode_probabilities.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]  # Top 3 failure modes

        return {
            "mode_probabilities": mode_probabilities,
            "dominant_modes": dominant_modes,
        }

    def _recommend_maintenance_interval(self, mtbf: float, mttr: float) -> float:
        """Recommend optimal maintenance interval."""
        if mtbf <= 0:
            return 168.0  # Default: 1 week

        # Optimal interval is typically 0.7-0.8 of MTBF for preventive maintenance
        optimal_interval = 0.75 * mtbf
        return float(max(24.0, optimal_interval))  # At least 24 hours

    def _recommend_maintenance_strategy(
        self,
        failure_data: Sequence[FailureEvent],
    ) -> MaintenanceStrategy:
        """Recommend maintenance strategy based on failure patterns."""
        if len(failure_data) < 3:
            return MaintenanceStrategy.PREVENTIVE

        # Analyze failure severity and frequency
        avg_severity = np.mean([event.severity for event in failure_data])
        failure_times = np.array([event.failure_time for event in failure_data])

        # Check for increasing failure rate (wear-out)
        if len(failure_times) >= 5:
            recent_interval = np.mean(np.diff(failure_times[-3:]))
            early_interval = np.mean(np.diff(failure_times[:3]))

            if (
                recent_interval < 0.8 * early_interval
            ):  # Failures becoming more frequent
                return MaintenanceStrategy.PREDICTIVE

        # High severity failures suggest condition-based maintenance
        if avg_severity > 0.7:
            return MaintenanceStrategy.CONDITION_BASED

        return MaintenanceStrategy.PREVENTIVE

    def _calculate_cost_benefit_ratio(
        self,
        mtbf: float,
        mttr: float,
        maintenance_interval: float,
    ) -> float:
        """Calculate cost-benefit ratio for maintenance strategy."""
        if mtbf <= 0 or maintenance_interval <= 0:
            return 0.0

        # Simplified cost-benefit calculation
        # Benefit: Reduced downtime cost
        # Cost: Maintenance cost

        downtime_reduction = mttr * (1.0 - maintenance_interval / mtbf)
        maintenance_frequency = 8760.0 / maintenance_interval  # Annual frequency

        # Assume maintenance takes 10% of MTTR
        maintenance_downtime = 0.1 * mttr * maintenance_frequency

        net_benefit = downtime_reduction - maintenance_downtime
        maintenance_cost_factor = maintenance_frequency * 0.1  # Normalized cost

        return (
            net_benefit / maintenance_cost_factor
            if maintenance_cost_factor > 0
            else 0.0
        )

    def _assess_prediction_accuracy(
        self,
        failure_data: Sequence[FailureEvent],
    ) -> float:
        """Assess accuracy of reliability predictions."""
        if not self.fitted_distribution or len(failure_data) < 3:
            return 0.0

        # Use cross-validation approach
        # Use first 70% of data for training, remaining for validation
        split_point = int(0.7 * len(failure_data))
        training_data = failure_data[:split_point]
        validation_data = failure_data[split_point:]

        if len(training_data) < 2 or len(validation_data) < 1:
            return self.fitted_distribution.goodness_of_fit

        # Fit distribution on training data
        training_tbf = self._calculate_time_between_failures(training_data)
        temp_distribution = self._fit_reliability_distribution(training_tbf)

        # Predict validation failures
        validation_times = [event.failure_time for event in validation_data]
        prediction_errors = []

        for i, actual_time in enumerate(validation_times):
            if i == 0:
                last_failure_time = training_data[-1].failure_time
            else:
                last_failure_time = validation_times[i - 1]

            predicted_time, _ = self._predict_failure_with_distribution(
                last_failure_time,
                temp_distribution,
            )

            error = abs(predicted_time - actual_time) / actual_time
            prediction_errors.append(error)

        # Calculate accuracy as 1 - mean_relative_error
        mean_error = float(np.mean(prediction_errors))
        accuracy = max(0.0, 1.0 - mean_error)

        return float(accuracy)

    def _predict_failure_with_distribution(
        self,
        current_time: float,
        distribution: ReliabilityParameters,
    ) -> tuple[float, float]:
        """Predict failure using specific distribution parameters."""
        if not distribution.parameters:
            return (current_time + 1000.0, 0.0)

        params = distribution.parameters

        if distribution.model_type == ReliabilityModel.EXPONENTIAL:
            lambda_rate = params["lambda"]
            predicted_time = current_time + (1.0 / lambda_rate)

        elif distribution.model_type == ReliabilityModel.WEIBULL:
            shape = params["shape"]
            scale = params["scale"]
            median_ttf = scale * (np.log(2)) ** (1.0 / shape)
            predicted_time = current_time + median_ttf

        else:
            predicted_time = current_time + 1000.0

        return (predicted_time, distribution.goodness_of_fit)


# Factory functions
def create_statistical_reliability_analyzer(
    confidence_level: float = 0.95,
    prediction_horizon: float = 8760.0,
    **kwargs: Any,
) -> StatisticalReliabilityAnalyzer:
    """Create a statistical reliability analyzer."""
    return StatisticalReliabilityAnalyzer(
        confidence_level=confidence_level,
        prediction_horizon=prediction_horizon,
        **kwargs,
    )


# Example usage and testing
def run_example_reliability_analysis() -> None:
    """Run example reliability analysis."""
    # Generate synthetic failure data
    np.random.seed(42)

    # Create synthetic failure events
    failure_events = []
    current_time = 0.0

    for i in range(15):  # 15 failure events
        # Exponential inter-arrival times with increasing failure rate
        time_to_failure = np.random.exponential(500.0 - i * 20)  # Degradation over time
        current_time += time_to_failure

        failure_modes_list = list(FailureMode)
        failure_mode = failure_modes_list[np.random.randint(len(failure_modes_list))]
        severity = np.random.beta(2, 5)  # Most failures are minor
        recovery_time = np.random.lognormal(2.0, 0.5)  # Log-normal recovery times

        event = FailureEvent(
            failure_time=current_time,
            failure_mode=failure_mode,
            severity=severity,
            recovery_time=recovery_time,
            root_cause=f"Synthetic cause {i + 1}",
            operator_notes=f"Test failure event {i + 1}",
        )
        failure_events.append(event)

    # Create analyzer and perform analysis
    analyzer = create_statistical_reliability_analyzer()
    analyzer.analyze_reliability(failure_events)

    # Predict next failure
    next_failure_time, confidence = analyzer.predict_next_failure(current_time)


if __name__ == "__main__":
    run_example_reliability_analysis()
