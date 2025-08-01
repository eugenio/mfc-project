#!/usr/bin/env python3
"""
Stability Analysis Framework for MFC Systems

This module provides a comprehensive framework for analyzing the stability
of Microbial Fuel Cell (MFC) systems, including power output stability,
biofilm health monitoring, and system performance degradation analysis.

Author: MFC Analysis Team
Created: 2025-07-31
Last Modified: 2025-07-31
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Protocol,
    TypeVar, Sequence
)

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases for better readability
TimeSeriesData = Union[pd.Series, np.ndarray, List[float]]
Timestamp = Union[datetime, pd.Timestamp, float]
MetricValue = Union[float, int, np.floating, np.integer]
ParameterDict = Dict[str, Any]

# Generic types
T = TypeVar('T')
StabilityMetricType = TypeVar('StabilityMetricType', bound='StabilityMetrics')


class StabilityLevel(Enum):
    """Enumeration of stability levels for MFC systems."""
    CRITICAL = auto()
    POOR = auto()
    FAIR = auto()
    GOOD = auto()
    EXCELLENT = auto()

    def __str__(self) -> str:
        return self.name.lower()

    @property
    def numeric_value(self) -> float:
        """Return numeric representation of stability level."""
        return {
            StabilityLevel.CRITICAL: 0.0,
            StabilityLevel.POOR: 0.25,
            StabilityLevel.FAIR: 0.5,
            StabilityLevel.GOOD: 0.75,
            StabilityLevel.EXCELLENT: 1.0
        }[self]


class AnalysisMethod(Enum):
    """Enumeration of stability analysis methods."""
    STATISTICAL = auto()
    SPECTRAL = auto()
    TREND_ANALYSIS = auto()
    WAVELET = auto()
    MACHINE_LEARNING = auto()


@dataclass(frozen=True)
class StabilityThresholds:
    """Thresholds for stability assessment."""
    power_variation_threshold: float = 0.05  # 5% variation threshold
    efficiency_degradation_threshold: float = 0.10  # 10% degradation threshold
    biofilm_health_threshold: float = 0.70  # 70% health threshold
    system_reliability_threshold: float = 0.95  # 95% reliability threshold
    temperature_stability_threshold: float = 2.0  # ±2°C temperature variation
    ph_stability_threshold: float = 0.5  # ±0.5 pH variation

    def __post_init__(self) -> None:
        """Validate threshold values."""
        if not (0.0 <= self.power_variation_threshold <= 1.0):
            raise ValueError("Power variation threshold must be between 0 and 1")
        if not (0.0 <= self.efficiency_degradation_threshold <= 1.0):
            raise ValueError("Efficiency degradation threshold must be between 0 and 1")


@dataclass
class StabilityMetrics:
    """Comprehensive stability metrics for MFC systems."""

    # Power stability metrics
    power_stability_coefficient: float = 0.0
    power_variation: float = 0.0
    power_trend_slope: float = 0.0
    power_autocorrelation: float = 0.0

    # Efficiency metrics
    efficiency_stability: float = 0.0
    efficiency_degradation_rate: float = 0.0
    efficiency_recovery_factor: float = 0.0

    # Biofilm stability metrics
    biofilm_health_index: float = 0.0
    biofilm_thickness_stability: float = 0.0
    biofilm_conductivity_stability: float = 0.0

    # System-wide metrics
    overall_stability_score: float = 0.0
    stability_level: StabilityLevel = StabilityLevel.FAIR
    confidence_interval: Tuple[float, float] = (0.0, 0.0)

    # Temporal metrics
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    analysis_duration: timedelta = field(default=timedelta(hours=1))

    # Metadata
    data_quality_score: float = 1.0
    analysis_method: AnalysisMethod = AnalysisMethod.STATISTICAL
    sample_size: int = 0

    def __post_init__(self) -> None:
        """Validate and compute derived metrics."""
        self._validate_metrics()
        self._compute_overall_stability()

    def _validate_metrics(self) -> None:
        """Validate metric values are within expected ranges."""
        metrics_to_validate = [
            ('power_stability_coefficient', 0.0, 1.0),
            ('efficiency_stability', 0.0, 1.0),
            ('biofilm_health_index', 0.0, 1.0),
            ('data_quality_score', 0.0, 1.0),
        ]

        for metric_name, min_val, max_val in metrics_to_validate:
            value = getattr(self, metric_name)
            if not (min_val <= value <= max_val):
                logger.warning(
                    f"Metric {metric_name} value {value} outside expected range "
                    f"[{min_val}, {max_val}]"
                )

    def _compute_overall_stability(self) -> None:
        """Compute overall stability score from individual metrics."""
        weights = {
            'power_stability_coefficient': 0.30,
            'efficiency_stability': 0.25,
            'biofilm_health_index': 0.20,
            'power_autocorrelation': 0.15,
            'biofilm_thickness_stability': 0.10
        }

        score = sum(
            getattr(self, metric) * weight
            for metric, weight in weights.items()
        )

        self.overall_stability_score = max(0.0, min(1.0, score))
        self.stability_level = self._determine_stability_level(score)

    def _determine_stability_level(self, score: float) -> StabilityLevel:
        """Determine stability level based on overall score."""
        if score >= 0.9:
            return StabilityLevel.EXCELLENT
        elif score >= 0.75:
            return StabilityLevel.GOOD
        elif score >= 0.5:
            return StabilityLevel.FAIR
        elif score >= 0.25:
            return StabilityLevel.POOR
        else:
            return StabilityLevel.CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'power_stability_coefficient': self.power_stability_coefficient,
            'power_variation': self.power_variation,
            'power_trend_slope': self.power_trend_slope,
            'power_autocorrelation': self.power_autocorrelation,
            'efficiency_stability': self.efficiency_stability,
            'efficiency_degradation_rate': self.efficiency_degradation_rate,
            'efficiency_recovery_factor': self.efficiency_recovery_factor,
            'biofilm_health_index': self.biofilm_health_index,
            'biofilm_thickness_stability': self.biofilm_thickness_stability,
            'biofilm_conductivity_stability': self.biofilm_conductivity_stability,
            'overall_stability_score': self.overall_stability_score,
            'stability_level': self.stability_level.name,
            'confidence_interval': self.confidence_interval,
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'analysis_duration': self.analysis_duration.total_seconds(),
            'data_quality_score': self.data_quality_score,
            'analysis_method': self.analysis_method.name,
            'sample_size': self.sample_size
        }


class StabilityAnalyzer(Protocol):
    """Protocol for stability analyzer implementations."""

    def analyze(
        self,
        data: TimeSeriesData,
        timestamps: Optional[Sequence[Timestamp]] = None,
        **kwargs: Any
    ) -> StabilityMetrics:
        """Analyze stability of time series data."""
        ...

    def validate_data(self, data: TimeSeriesData) -> bool:
        """Validate input data quality."""
        ...


class BaseStabilityAnalyzer(ABC):
    """Base class for stability analyzers."""

    def __init__(
        self,
        thresholds: Optional[StabilityThresholds] = None,
        min_data_points: int = 10,
        confidence_level: float = 0.95
    ) -> None:
        """Initialize the stability analyzer.
        
        Args:
            thresholds: Custom stability thresholds
            min_data_points: Minimum number of data points required
            confidence_level: Confidence level for statistical analysis
        """
        self.thresholds = thresholds or StabilityThresholds()
        self.min_data_points = min_data_points
        self.confidence_level = confidence_level
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging for the analyzer."""
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def analyze(
        self,
        data: TimeSeriesData,
        timestamps: Optional[Sequence[Timestamp]] = None,
        **kwargs: Any
    ) -> StabilityMetrics:
        """Analyze stability of time series data."""
        pass

    def validate_data(self, data: TimeSeriesData) -> bool:
        """Validate input data quality.
        
        Args:
            data: Time series data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Convert data to numpy array for validation
            data_array = np.asarray(data)

            # Check for minimum data points
            if len(data_array) < self.min_data_points:
                self.logger.warning(
                    f"Insufficient data points: {len(data_array)} < {self.min_data_points}"
                )
                return False

            # Check for NaN or infinite values
            if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
                self.logger.warning("Data contains NaN or infinite values")
                return False

            # Check for constant data (no variation)
            if np.all(data_array == data_array[0]):
                self.logger.warning("Data has no variation (all values identical)")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}")
            return False

    def _calculate_confidence_interval(
        self,
        values: np.ndarray,
        confidence_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """Calculate confidence interval for values.
        
        Args:
            values: Array of values
            confidence_level: Confidence level (defaults to instance level)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        mean_val = np.mean(values)
        std_err = np.std(values) / np.sqrt(len(values))

        # Use t-distribution for small samples
        from scipy import stats
        df = len(values) - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df)

        margin_error = t_critical * std_err
        return (mean_val - margin_error, mean_val + margin_error)


class StatisticalStabilityAnalyzer(BaseStabilityAnalyzer):
    """Statistical stability analyzer using traditional statistical methods."""

    def __init__(
        self,
        thresholds: Optional[StabilityThresholds] = None,
        min_data_points: int = 30,
        confidence_level: float = 0.95,
        window_size: Optional[int] = None
    ) -> None:
        """Initialize statistical analyzer.
        
        Args:
            thresholds: Custom stability thresholds
            min_data_points: Minimum number of data points
            confidence_level: Confidence level for analysis
            window_size: Rolling window size for analysis
        """
        super().__init__(thresholds, min_data_points, confidence_level)
        self.window_size = window_size or max(10, min_data_points // 3)

    def analyze(
        self,
        data: TimeSeriesData,
        timestamps: Optional[Sequence[Timestamp]] = None,
        **kwargs: Any
    ) -> StabilityMetrics:
        """Perform statistical stability analysis.
        
        Args:
            data: Time series data to analyze
            timestamps: Optional timestamps for data points
            **kwargs: Additional analysis parameters
            
        Returns:
            StabilityMetrics with computed stability measures
        """
        if not self.validate_data(data):
            raise ValueError("Invalid input data for stability analysis")

        data_array = np.asarray(data, dtype=np.float64)
        n_samples = len(data_array)

        # Calculate basic stability metrics
        power_stability = self._calculate_power_stability(data_array)
        power_variation = self._calculate_power_variation(data_array)
        power_trend = self._calculate_trend_slope(data_array)
        power_autocorr = self._calculate_autocorrelation(data_array)

        # Calculate efficiency metrics (assuming data represents power output)
        efficiency_stability = self._calculate_efficiency_stability(data_array)
        efficiency_degradation = self._calculate_degradation_rate(data_array)
        efficiency_recovery = self._calculate_recovery_factor(data_array)

        # Calculate biofilm metrics (synthetic for now)
        biofilm_health = self._estimate_biofilm_health(data_array)
        biofilm_thickness_stability = self._calculate_biofilm_thickness_stability(data_array)
        biofilm_conductivity_stability = self._calculate_biofilm_conductivity_stability(data_array)

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(data_array)

        # Data quality assessment
        data_quality = self._assess_data_quality(data_array)

        return StabilityMetrics(
            power_stability_coefficient=power_stability,
            power_variation=power_variation,
            power_trend_slope=power_trend,
            power_autocorrelation=power_autocorr,
            efficiency_stability=efficiency_stability,
            efficiency_degradation_rate=efficiency_degradation,
            efficiency_recovery_factor=efficiency_recovery,
            biofilm_health_index=biofilm_health,
            biofilm_thickness_stability=biofilm_thickness_stability,
            biofilm_conductivity_stability=biofilm_conductivity_stability,
            confidence_interval=confidence_interval,
            analysis_timestamp=datetime.now(),
            analysis_duration=timedelta(seconds=1),  # Placeholder
            data_quality_score=data_quality,
            analysis_method=AnalysisMethod.STATISTICAL,
            sample_size=n_samples
        )

    def _calculate_power_stability(self, data: np.ndarray) -> float:
        """Calculate power stability coefficient."""
        mean_power = np.mean(data)
        std_power = np.std(data)

        if mean_power == 0:
            return 0.0

        # Stability coefficient: inverse of coefficient of variation
        cv = std_power / mean_power
        return max(0.0, 1.0 - cv)

    def _calculate_power_variation(self, data: np.ndarray) -> float:
        """Calculate power variation metric."""
        return float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else 0.0

    def _calculate_trend_slope(self, data: np.ndarray) -> float:
        """Calculate trend slope using linear regression."""
        x = np.arange(len(data))
        slope = float(np.polyfit(x, data, 1)[0])
        return slope

    def _calculate_autocorrelation(self, data: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation at specified lag."""
        if len(data) <= lag:
            return 0.0

        return float(np.corrcoef(data[:-lag], data[lag:])[0, 1])

    def _calculate_efficiency_stability(self, data: np.ndarray) -> float:
        """Calculate efficiency stability (simplified)."""
        # Rolling efficiency calculation
        rolling_mean = pd.Series(data).rolling(window=self.window_size).mean()
        rolling_std = pd.Series(data).rolling(window=self.window_size).std()

        efficiency_cv = rolling_std / rolling_mean
        mean_efficiency_cv = efficiency_cv.dropna().mean()

        return max(0.0, 1.0 - mean_efficiency_cv)

    def _calculate_degradation_rate(self, data: np.ndarray) -> float:
        """Calculate degradation rate."""
        if len(data) < 2:
            return 0.0

        # Linear trend as degradation indicator
        slope = self._calculate_trend_slope(data)
        mean_value = np.mean(data)

        if mean_value == 0:
            return 0.0

        # Normalize slope by mean value
        degradation_rate = -slope / mean_value if slope < 0 else 0.0
        return float(np.clip(degradation_rate, 0.0, 1.0))

    def _calculate_recovery_factor(self, data: np.ndarray) -> float:
        """Calculate recovery factor from dips in performance."""
        # Find local minima and measure recovery
        from scipy.signal import argrelextrema

        if len(data) < 5:
            return 1.0

        minima_indices = argrelextrema(data, np.less, order=2)[0]

        if len(minima_indices) == 0:
            return 1.0

        recovery_factors = []
        for min_idx in minima_indices:
            if min_idx + 5 < len(data):  # Ensure enough data for recovery analysis
                min_value = data[min_idx]
                recovery_window = data[min_idx:min_idx + 5]
                max_recovery = np.max(recovery_window)

                recovery_factor = (max_recovery - min_value) / min_value if min_value != 0 else 1.0
                recovery_factors.append(recovery_factor)

        return float(np.mean(recovery_factors)) if recovery_factors else 1.0

    def _estimate_biofilm_health(self, data: np.ndarray) -> float:
        """Estimate biofilm health from power data (correlation-based)."""
        # Simplified biofilm health estimation based on power stability
        stability = self._calculate_power_stability(data)
        variability = np.std(data) / np.mean(data) if np.mean(data) != 0 else 1.0

        # Biofilm health correlates with stable power and low variability
        health_score = stability * (1.0 - min(variability, 1.0))
        return float(np.clip(health_score, 0.0, 1.0))

    def _calculate_biofilm_thickness_stability(self, data: np.ndarray) -> float:
        """Calculate biofilm thickness stability (synthetic)."""
        # Synthetic calculation based on power data characteristics
        trend_magnitude = abs(self._calculate_trend_slope(data))
        max_power = np.max(data)
        normalized_trend = trend_magnitude / max_power if max_power != 0 else 0.0

        thickness_stability = 1.0 - min(normalized_trend, 1.0)
        return float(thickness_stability)

    def _calculate_biofilm_conductivity_stability(self, data: np.ndarray) -> float:
        """Calculate biofilm conductivity stability (synthetic)."""
        # Synthetic calculation based on autocorrelation and variance
        autocorr = abs(self._calculate_autocorrelation(data))
        stability = self._calculate_power_stability(data)

        conductivity_stability = (autocorr + stability) / 2.0
        return float(np.clip(conductivity_stability, 0.0, 1.0))

    def _assess_data_quality(self, data: np.ndarray) -> float:
        """Assess overall data quality."""
        quality_factors = []

        # Check for completeness (no NaN values)
        completeness = 1.0 - (np.sum(np.isnan(data)) / len(data))
        quality_factors.append(completeness)

        # Check for reasonable value range (no extreme outliers)
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        outliers = np.sum((data < q1 - 3 * iqr) | (data > q3 + 3 * iqr))
        outlier_ratio = outliers / len(data)
        outlier_quality = 1.0 - min(outlier_ratio, 1.0)
        quality_factors.append(outlier_quality)

        # Check for temporal consistency (smooth transitions)
        diff_data = np.abs(np.diff(data))
        mean_diff = np.mean(diff_data)
        std_diff = np.std(diff_data)
        smoothness = 1.0 - min(std_diff / mean_diff if mean_diff != 0 else 1.0, 1.0)
        quality_factors.append(smoothness)

        return float(np.mean(quality_factors))


class StabilityFramework:
    """Main framework class for MFC stability analysis."""

    def __init__(
        self,
        analyzer: Optional[BaseStabilityAnalyzer] = None,
        thresholds: Optional[StabilityThresholds] = None
    ) -> None:
        """Initialize the stability framework.
        
        Args:
            analyzer: Custom stability analyzer
            thresholds: Custom stability thresholds
        """
        self.thresholds = thresholds or StabilityThresholds()
        self.analyzer = analyzer or StatisticalStabilityAnalyzer(self.thresholds)
        self.analysis_history: List[StabilityMetrics] = []
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging for the framework."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_stability(
        self,
        data: TimeSeriesData,
        timestamps: Optional[Sequence[Timestamp]] = None,
        **kwargs: Any
    ) -> StabilityMetrics:
        """Perform comprehensive stability analysis.
        
        Args:
            data: Time series data to analyze
            timestamps: Optional timestamps for data points
            **kwargs: Additional analysis parameters
            
        Returns:
            StabilityMetrics with computed stability measures
        """
        try:
            self.logger.info("Starting stability analysis...")

            # Perform analysis using configured analyzer
            metrics = self.analyzer.analyze(data, timestamps, **kwargs)

            # Store analysis in history
            self.analysis_history.append(metrics)

            # Log results
            self.logger.info(
                f"Stability analysis complete. Overall score: {metrics.overall_stability_score:.3f}, "
                f"Level: {metrics.stability_level}"
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Stability analysis failed: {str(e)}")
            raise

    def get_stability_trends(
        self,
        window_size: int = 10
    ) -> Optional[Dict[str, List[float]]]:
        """Get stability trends from analysis history.
        
        Args:
            window_size: Size of rolling window for trend analysis
            
        Returns:
            Dictionary with trend data or None if insufficient history
        """
        if len(self.analysis_history) < 2:
            return None

        trends = {
            'timestamps': [m.analysis_timestamp.timestamp() for m in self.analysis_history],
            'overall_stability': [m.overall_stability_score for m in self.analysis_history],
            'power_stability': [m.power_stability_coefficient for m in self.analysis_history],
            'efficiency_stability': [m.efficiency_stability for m in self.analysis_history],
            'biofilm_health': [m.biofilm_health_index for m in self.analysis_history]
        }

        return trends

    def export_analysis_results(
        self,
        filepath: Union[str, Path],
        format_type: str = 'json'
    ) -> None:
        """Export analysis results to file.
        
        Args:
            filepath: Output file path
            format_type: Export format ('json' or 'csv')
        """
        if not self.analysis_history:
            raise ValueError("No analysis results to export")

        filepath = Path(filepath)

        if format_type.lower() == 'json':
            import json
            data = [metrics.to_dict() for metrics in self.analysis_history]
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        elif format_type.lower() == 'csv':
            data = [metrics.to_dict() for metrics in self.analysis_history]
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)

        else:
            raise ValueError(f"Unsupported export format: {format_type}")

        self.logger.info(f"Analysis results exported to {filepath}")


# Factory functions for creating analyzers
def create_statistical_analyzer(
    thresholds: Optional[StabilityThresholds] = None,
    **kwargs: Any
) -> StatisticalStabilityAnalyzer:
    """Create a statistical stability analyzer."""
    return StatisticalStabilityAnalyzer(thresholds, **kwargs)


def create_default_framework(
    analyzer_type: str = 'statistical',
    **kwargs: Any
) -> StabilityFramework:
    """Create a default stability framework."""
    thresholds = StabilityThresholds()

    if analyzer_type == 'statistical':
        analyzer = create_statistical_analyzer(thresholds, **kwargs)
    else:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")

    return StabilityFramework(analyzer, thresholds)


# Example usage and testing functions
def run_example_analysis() -> None:
    """Run example stability analysis."""
    # Generate synthetic MFC power data
    np.random.seed(42)
    time_points = np.linspace(0, 24, 1000)  # 24 hours of data
    base_power = 20.0

    # Add realistic variations
    seasonal_variation = 2.0 * np.sin(2 * np.pi * time_points / 24)  # Daily cycle
    random_noise = np.random.normal(0, 0.5, len(time_points))
    degradation_trend = -0.1 * time_points / 24  # Slight degradation

    synthetic_data = base_power + seasonal_variation + random_noise + degradation_trend

    # Create framework and analyze
    framework = create_default_framework()
    metrics = framework.analyze_stability(synthetic_data)

    print("Stability Analysis Results:")
    print(f"Overall Stability Score: {metrics.overall_stability_score:.3f}")
    print(f"Stability Level: {metrics.stability_level}")
    print(f"Power Stability Coefficient: {metrics.power_stability_coefficient:.3f}")
    print(f"Biofilm Health Index: {metrics.biofilm_health_index:.3f}")
    print(f"Data Quality Score: {metrics.data_quality_score:.3f}")


if __name__ == "__main__":
    run_example_analysis()
