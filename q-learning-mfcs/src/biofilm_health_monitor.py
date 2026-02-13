"""
Predictive Biofilm Health Monitoring System

Phase 2 enhancement implementing advanced health monitoring with:
- Real-time biofilm health assessment using multi-sensor data
- Predictive health degradation detection with early warning
- Health trend analysis and intervention recommendations
- Integration with advanced sensor fusion and Q-learning control

Created: 2025-07-31
Last Modified: 2025-07-31
"""

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

# Import sensor fusion components
from sensing_models.advanced_sensor_fusion import (
    AnomalyDetection,
    BiofimGrowthPattern,
    FusedMeasurement,
)
from sensing_models.sensor_fusion import BacterialSpecies

# Configuration
try:
    from config import QLearningConfig, SensorConfig
except ImportError:
    SensorConfig = None
    QLearningConfig = None

# Statistical analysis
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Biofilm health status levels."""
    EXCELLENT = "excellent"  # >90% optimal performance
    GOOD = "good"           # 70-90% optimal performance
    FAIR = "fair"           # 50-70% optimal performance
    POOR = "poor"           # 30-50% optimal performance
    CRITICAL = "critical"   # <30% optimal performance
    UNKNOWN = "unknown"     # Insufficient data


class HealthTrend(Enum):
    """Health trend directions."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class HealthMetrics:
    """Comprehensive biofilm health metrics."""

    # Core health indicators (0-1 scale)
    overall_health_score: float
    thickness_health: float  # Relative to optimal thickness
    conductivity_health: float  # Electron transfer efficiency
    growth_health: float  # Growth pattern quality
    stability_health: float  # Measurement consistency

    # Derived indicators
    health_status: HealthStatus
    health_trend: HealthTrend

    # Predictive metrics
    predicted_health_24h: float  # Predicted health in 24 hours
    predicted_intervention_time: float | None  # Hours until intervention needed

    # Component contributions
    thickness_contribution: float  # Contribution to overall health
    conductivity_contribution: float
    growth_contribution: float
    stability_contribution: float

    # Risk assessments
    fouling_risk: float  # Risk of electrode fouling (0-1)
    detachment_risk: float  # Risk of biofilm detachment (0-1)
    stagnation_risk: float  # Risk of growth stagnation (0-1)

    # Confidence metrics
    assessment_confidence: float  # Confidence in health assessment (0-1)
    prediction_confidence: float  # Confidence in predictions (0-1)


@dataclass
class HealthAlert:
    """Health monitoring alert."""

    timestamp: float
    alert_type: str  # 'health_decline', 'intervention_needed', 'anomaly_detected'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    affected_components: list[str]
    recommended_actions: list[str]
    predicted_consequences: str
    confidence: float


@dataclass
class InterventionRecommendation:
    """Recommended intervention based on health assessment."""

    intervention_type: str  # 'flow_adjustment', 'substrate_change', 'cleaning', 'monitoring'
    urgency: str  # 'immediate', 'within_hours', 'within_days', 'routine'
    description: str
    expected_benefit: str
    implementation_difficulty: str  # 'easy', 'moderate', 'difficult'
    estimated_duration: float  # Hours
    success_probability: float  # 0-1


class BiofimHealthOptimizer:
    """
    Biofilm health optimization using physiological models and operational parameters.

    Determines optimal operating conditions for biofilm health based on:
    - Species-specific growth parameters
    - Environmental conditions
    - Historical performance data
    """

    def __init__(self, species: BacterialSpecies = BacterialSpecies.MIXED, config: SensorConfig | None = None):
        """
        Initialize health optimizer.

        Args:
            species: Target bacterial species
            config: Optional sensor configuration
        """
        self.species = species
        self.config = config

        # Species-specific optimal parameters
        self.optimal_parameters = self._initialize_optimal_parameters()

        # Health thresholds from config or defaults
        if config and hasattr(config, 'health_thresholds'):
            self.health_thresholds = config.health_thresholds
        else:
            self.health_thresholds = {
                'thickness_optimal_um': 15.0,
                'thickness_tolerance_um': 5.0,
                'conductivity_optimal_S_per_m': 0.05,
                'conductivity_tolerance': 0.02,
                'growth_rate_optimal_um_per_h': 0.5,
                'growth_rate_tolerance': 0.3
            }

        logger.info(f"Biofilm health optimizer initialized for {species.value}")

    def _initialize_optimal_parameters(self) -> dict[str, float]:
        """Initialize species-specific optimal parameters."""
        if self.species == BacterialSpecies.GEOBACTER:
            return {
                'optimal_thickness_um': 12.0,
                'max_thickness_um': 25.0,
                'optimal_conductivity_S_per_m': 0.08,
                'optimal_growth_rate_um_per_h': 0.6,
                'ph_optimal': 7.0,
                'temperature_optimal_C': 30.0,
                'substrate_concentration_optimal_mM': 10.0
            }
        elif self.species == BacterialSpecies.SHEWANELLA:
            return {
                'optimal_thickness_um': 18.0,
                'max_thickness_um': 35.0,
                'optimal_conductivity_S_per_m': 0.03,
                'optimal_growth_rate_um_per_h': 0.4,
                'ph_optimal': 7.5,
                'temperature_optimal_C': 25.0,
                'substrate_concentration_optimal_mM': 15.0
            }
        else:  # Mixed culture
            return {
                'optimal_thickness_um': 15.0,
                'max_thickness_um': 30.0,
                'optimal_conductivity_S_per_m': 0.05,
                'optimal_growth_rate_um_per_h': 0.5,
                'ph_optimal': 7.2,
                'temperature_optimal_C': 28.0,
                'substrate_concentration_optimal_mM': 12.0
            }

    def calculate_thickness_health(self, current_thickness: float, thickness_trend: float) -> float:
        """Calculate thickness health score (0-1)."""
        optimal = self.optimal_parameters['optimal_thickness_um']
        max_thickness = self.optimal_parameters['max_thickness_um']

        # Distance from optimal
        deviation = abs(current_thickness - optimal)
        relative_deviation = deviation / optimal if optimal > 0 else 1.0

        # Base health score (Gaussian-like)
        base_health = np.exp(-0.5 * (relative_deviation / 0.3)**2)

        # Penalty for excessive thickness
        if current_thickness > max_thickness:
            excess_penalty = (current_thickness - max_thickness) / max_thickness
            base_health *= max(0.1, 1.0 - excess_penalty)

        # Trend adjustment
        if thickness_trend > 0.1:  # Healthy growth
            base_health = min(1.0, base_health * 1.1)
        elif thickness_trend < -0.1:  # Decline
            base_health = max(0.0, base_health * 0.9)

        return np.clip(base_health, 0.0, 1.0)

    def calculate_conductivity_health(self, current_conductivity: float, conductivity_trend: float) -> float:
        """Calculate conductivity health score (0-1)."""
        optimal = self.optimal_parameters['optimal_conductivity_S_per_m']

        if optimal <= 0:
            return 0.5  # Neutral score if no optimal defined

        # Logarithmic health function (conductivity spans orders of magnitude)
        if current_conductivity > 0:
            log_ratio = np.log10(current_conductivity / optimal)
            health = np.exp(-0.5 * (log_ratio / 0.5)**2)  # Narrower tolerance for conductivity
        else:
            health = 0.0

        # Trend adjustment
        if conductivity_trend > 0.001:  # Improving electron transfer
            health = min(1.0, health * 1.05)
        elif conductivity_trend < -0.001:  # Declining
            health = max(0.0, health * 0.95)

        return np.clip(health, 0.0, 1.0)

    def calculate_growth_health(self, growth_pattern: BiofimGrowthPattern) -> float:
        """Calculate growth pattern health score (0-1)."""
        if not growth_pattern or growth_pattern.pattern_confidence < 0.3:
            return 0.5  # Neutral score for unknown patterns

        optimal_rate = self.optimal_parameters['optimal_growth_rate_um_per_h']
        current_rate = growth_pattern.growth_rate_um_per_hour

        # Phase-based scoring
        phase_scores = {
            'lag': 0.6,      # Acceptable during startup
            'exponential': 1.0 if abs(current_rate - optimal_rate) < 0.2 else 0.8,
            'stationary': 0.8,  # Good steady state
            'decline': 0.3,   # Concerning
            'unknown': 0.5
        }

        base_score = phase_scores.get(growth_pattern.growth_phase, 0.5)

        # Rate deviation penalty
        if optimal_rate > 0:
            rate_deviation = abs(current_rate - optimal_rate) / optimal_rate
            rate_penalty = max(0, rate_deviation - 0.3)  # Tolerance of 30%
            base_score *= max(0.2, 1.0 - rate_penalty)

        # Confidence weighting
        final_score = base_score * growth_pattern.pattern_confidence + 0.5 * (1 - growth_pattern.pattern_confidence)

        return np.clip(final_score, 0.0, 1.0)

    def assess_risks(self, thickness: float, conductivity: float, growth_rate: float,
                    measurement_stability: float) -> tuple[float, float, float]:
        """
        Assess specific risk factors.

        Returns:
            Tuple of (fouling_risk, detachment_risk, stagnation_risk)
        """
        # Electrode fouling risk (high thickness, low conductivity)
        max_thickness = self.optimal_parameters['max_thickness_um']
        min_conductivity = self.optimal_parameters['optimal_conductivity_S_per_m'] * 0.3

        fouling_risk = 0.0
        if thickness > max_thickness * 0.8:
            fouling_risk += (thickness - max_thickness * 0.8) / (max_thickness * 0.2)
        if conductivity < min_conductivity:
            fouling_risk += (min_conductivity - conductivity) / min_conductivity
        fouling_risk = min(1.0, fouling_risk / 2.0)  # Average of thickness and conductivity factors

        # Detachment risk (rapid growth changes, poor stability)
        detachment_risk = 0.0
        if abs(growth_rate) > 1.0:  # Rapid changes
            detachment_risk += min(1.0, abs(growth_rate) / 2.0)
        if measurement_stability < 0.3:  # Poor stability
            detachment_risk += (0.3 - measurement_stability) / 0.3
        detachment_risk = min(1.0, detachment_risk / 2.0)

        # Stagnation risk (very low growth, declining conductivity)
        stagnation_risk = 0.0
        if abs(growth_rate) < 0.05:  # Very slow growth
            stagnation_risk += 0.5
        if conductivity < self.optimal_parameters['optimal_conductivity_S_per_m'] * 0.5:
            stagnation_risk += 0.3
        if measurement_stability > 0.95:  # Too stable might indicate stagnation
            stagnation_risk += 0.2
        stagnation_risk = min(1.0, stagnation_risk)

        return fouling_risk, detachment_risk, stagnation_risk


class PredictiveBiofimHealthMonitor:
    """
    Comprehensive predictive biofilm health monitoring system.

    Integrates sensor fusion data, growth patterns, and predictive models
    to provide real-time health assessment and early intervention recommendations.
    """

    def __init__(self, species: BacterialSpecies = BacterialSpecies.MIXED,
                 config: SensorConfig | None = None,
                 history_window: int = 100):
        """
        Initialize predictive health monitor.

        Args:
            species: Target bacterial species
            config: Optional sensor configuration
            history_window: Size of historical data window
        """
        self.species = species
        self.config = config
        self.history_window = history_window

        # Initialize health optimizer
        self.health_optimizer = BiofimHealthOptimizer(species, config)

        # Historical data storage
        self.health_history = deque(maxlen=history_window)
        self.measurement_history = deque(maxlen=history_window)
        self.alert_history = []
        self.intervention_history = []

        # Predictive model parameters
        self.prediction_models = {
            'health_decay_rate': 0.02,  # Per hour
            'intervention_threshold': 0.4,  # Health score triggering intervention
            'critical_threshold': 0.3,  # Health score for critical alerts
        }

        # Alert thresholds
        self.alert_thresholds = {
            'health_decline_rate': -0.05,  # Per measurement
            'risk_threshold': 0.7,  # Individual risk factor threshold
            'stability_threshold': 0.3  # Minimum acceptable stability
        }

        # Current state
        self.current_health = None
        self.last_assessment_time = 0.0

        logger.info(f"Predictive biofilm health monitor initialized for {species.value}")

    def assess_health(self, fused_measurement: FusedMeasurement,
                     growth_pattern: BiofimGrowthPattern | None = None,
                     anomalies: list[AnomalyDetection] | None = None) -> HealthMetrics:
        """
        Comprehensive health assessment from sensor data.

        Args:
            fused_measurement: Latest fused sensor measurement
            growth_pattern: Optional growth pattern analysis
            anomalies: Optional detected anomalies

        Returns:
            Complete health metrics
        """
        timestamp = fused_measurement.timestamp

        # Calculate measurement stability from recent history
        stability_score = self._calculate_measurement_stability()

        # Calculate individual health components
        thickness_health = self.health_optimizer.calculate_thickness_health(
            fused_measurement.thickness_um,
            self._calculate_thickness_trend()
        )

        conductivity_health = self.health_optimizer.calculate_conductivity_health(
            fused_measurement.conductivity_S_per_m,
            self._calculate_conductivity_trend()
        )

        growth_health = 0.5  # Default neutral
        if growth_pattern:
            growth_health = self.health_optimizer.calculate_growth_health(growth_pattern)

        stability_health = stability_score

        # Calculate component contributions (weighted importance)
        weights = {
            'thickness': 0.3,
            'conductivity': 0.3,
            'growth': 0.25,
            'stability': 0.15
        }

        # Overall health score
        overall_health = (
            weights['thickness'] * thickness_health +
            weights['conductivity'] * conductivity_health +
            weights['growth'] * growth_health +
            weights['stability'] * stability_health
        )

        # Anomaly penalty
        if anomalies:
            anomaly_penalty = min(0.3, len([a for a in anomalies if a.severity in ['high', 'critical']]) * 0.1)
            overall_health = max(0.0, overall_health - anomaly_penalty)

        # Determine health status
        health_status = self._categorize_health_status(overall_health)

        # Calculate health trend
        health_trend = self._calculate_health_trend()

        # Risk assessments
        growth_rate = growth_pattern.growth_rate_um_per_hour if growth_pattern else 0.0
        fouling_risk, detachment_risk, stagnation_risk = self.health_optimizer.assess_risks(
            fused_measurement.thickness_um,
            fused_measurement.conductivity_S_per_m,
            growth_rate,
            stability_score
        )

        # Predictive health assessment
        predicted_health_24h = self._predict_health_24h(overall_health, health_trend)
        predicted_intervention_time = self._predict_intervention_time(overall_health, health_trend)

        # Assessment confidence
        assessment_confidence = min(
            fused_measurement.fusion_confidence,
            growth_pattern.pattern_confidence if growth_pattern else 0.5,
            stability_score
        )

        prediction_confidence = assessment_confidence * 0.8  # Predictions less certain

        # Create health metrics
        health_metrics = HealthMetrics(
            overall_health_score=overall_health,
            thickness_health=thickness_health,
            conductivity_health=conductivity_health,
            growth_health=growth_health,
            stability_health=stability_health,
            health_status=health_status,
            health_trend=health_trend,
            predicted_health_24h=predicted_health_24h,
            predicted_intervention_time=predicted_intervention_time,
            thickness_contribution=weights['thickness'] * thickness_health,
            conductivity_contribution=weights['conductivity'] * conductivity_health,
            growth_contribution=weights['growth'] * growth_health,
            stability_contribution=weights['stability'] * stability_health,
            fouling_risk=fouling_risk,
            detachment_risk=detachment_risk,
            stagnation_risk=stagnation_risk,
            assessment_confidence=assessment_confidence,
            prediction_confidence=prediction_confidence
        )

        # Store for history
        self.health_history.append(health_metrics)
        self.measurement_history.append(fused_measurement)
        self.current_health = health_metrics
        self.last_assessment_time = timestamp

        return health_metrics

    def _calculate_measurement_stability(self) -> float:
        """Calculate measurement stability from recent history."""
        if len(self.measurement_history) < 5:
            return 0.5  # Neutral stability for insufficient data

        recent_measurements = list(self.measurement_history)[-10:]

        # Coefficient of variation for thickness
        thicknesses = [m.thickness_um for m in recent_measurements]
        thickness_cv = np.std(thicknesses) / np.mean(thicknesses) if np.mean(thicknesses) > 0 else 1.0

        # Sensor agreement stability
        agreements = [m.sensor_agreement for m in recent_measurements]
        agreement_stability = 1.0 - np.std(agreements)  # Lower std = higher stability

        # Fusion confidence stability
        confidences = [m.fusion_confidence for m in recent_measurements]
        confidence_stability = 1.0 - np.std(confidences)

        # Combined stability score
        stability = (
            max(0, 1.0 - thickness_cv * 2.0) * 0.4 +  # Thickness stability
            max(0, agreement_stability) * 0.3 +        # Agreement stability
            max(0, confidence_stability) * 0.3         # Confidence stability
        )

        return np.clip(stability, 0.0, 1.0)

    def _calculate_thickness_trend(self) -> float:
        """Calculate thickness trend (μm/hour)."""
        if len(self.measurement_history) < 5:
            return 0.0

        recent_measurements = list(self.measurement_history)[-10:]
        times = [m.timestamp for m in recent_measurements]
        thicknesses = [m.thickness_um for m in recent_measurements]

        if len(set(times)) < 2:  # Need different timestamps
            return 0.0

        # Simple linear regression
        if SCIPY_AVAILABLE:
            try:
                slope, _, _, _, _ = stats.linregress(times, thicknesses)
                return slope
            except Exception:
                pass

        # Fallback: simple difference
        if len(thicknesses) >= 2:
            dt = times[-1] - times[-2]
            if dt > 0:
                return (thicknesses[-1] - thicknesses[-2]) / dt

        return 0.0

    def _calculate_conductivity_trend(self) -> float:
        """Calculate conductivity trend (S/m/hour)."""
        if len(self.measurement_history) < 5:
            return 0.0

        recent_measurements = list(self.measurement_history)[-10:]
        times = [m.timestamp for m in recent_measurements]
        conductivities = [m.conductivity_S_per_m for m in recent_measurements]

        if len(set(times)) < 2:
            return 0.0

        # Simple linear regression
        if SCIPY_AVAILABLE:
            try:
                slope, _, _, _, _ = stats.linregress(times, conductivities)
                return slope
            except Exception:
                pass

        # Fallback: simple difference
        if len(conductivities) >= 2:
            dt = times[-1] - times[-2]
            if dt > 0:
                return (conductivities[-1] - conductivities[-2]) / dt

        return 0.0

    def _categorize_health_status(self, health_score: float) -> HealthStatus:
        """Categorize health score into status level."""
        if health_score >= 0.9:
            return HealthStatus.EXCELLENT
        elif health_score >= 0.7:
            return HealthStatus.GOOD
        elif health_score >= 0.5:
            return HealthStatus.FAIR
        elif health_score >= 0.3:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _calculate_health_trend(self) -> HealthTrend:
        """Calculate health trend from recent history."""
        if len(self.health_history) < 5:
            return HealthTrend.UNKNOWN

        recent_health = [h.overall_health_score for h in list(self.health_history)[-10:]]

        # Simple trend analysis
        if len(recent_health) >= 3:
            recent_avg = np.mean(recent_health[-3:])
            earlier_avg = np.mean(recent_health[-6:-3]) if len(recent_health) >= 6 else recent_avg

            health_change = recent_avg - earlier_avg
            health_variance = np.var(recent_health)

            if health_variance > 0.05:  # High variance
                return HealthTrend.VOLATILE
            elif health_change > 0.05:
                return HealthTrend.IMPROVING
            elif health_change < -0.05:
                return HealthTrend.DECLINING
            else:
                return HealthTrend.STABLE

        return HealthTrend.UNKNOWN

    def _predict_health_24h(self, current_health: float, trend: HealthTrend) -> float:
        """Predict health score 24 hours ahead."""
        # Base prediction assumes gradual decay without intervention
        decay_rate = self.prediction_models['health_decay_rate']
        predicted_health = current_health * (1 - decay_rate * 24)

        # Trend adjustments
        if trend == HealthTrend.IMPROVING:
            predicted_health = min(1.0, predicted_health * 1.1)
        elif trend == HealthTrend.DECLINING:
            predicted_health = max(0.0, predicted_health * 0.9)
        elif trend == HealthTrend.VOLATILE:
            predicted_health = max(0.0, predicted_health * 0.95)  # Volatility generally negative

        return np.clip(predicted_health, 0.0, 1.0)

    def _predict_intervention_time(self, current_health: float, trend: HealthTrend) -> float | None:
        """Predict time until intervention needed (hours)."""
        threshold = self.prediction_models['intervention_threshold']

        if current_health <= threshold:
            return 0.0  # Immediate intervention needed

        # Simple linear extrapolation based on trend
        if trend == HealthTrend.DECLINING:
            # Estimate decline rate from recent history
            if len(self.health_history) >= 5:
                recent_scores = [h.overall_health_score for h in list(self.health_history)[-5:]]

                # Estimate decline rate based on score differences
                decline_rate = max(0.01, (recent_scores[0] - recent_scores[-1]) / max(1, len(recent_scores) - 1))

                # Time to reach threshold
                time_to_threshold = (current_health - threshold) / decline_rate
                return max(0, time_to_threshold)

        elif trend == HealthTrend.VOLATILE:
            # Conservative estimate for volatile conditions
            return 12.0  # Check within 12 hours

        # For stable or improving trends, no immediate intervention predicted
        return None

    def generate_alerts(self, health_metrics: HealthMetrics,
                       anomalies: list[AnomalyDetection] | None = None) -> list[HealthAlert]:
        """Generate health alerts based on current assessment."""
        alerts = []
        timestamp = self.last_assessment_time

        # Critical health alert
        if health_metrics.health_status == HealthStatus.CRITICAL:
            alerts.append(HealthAlert(
                timestamp=timestamp,
                alert_type='health_decline',
                severity='critical',
                message=f'Biofilm health critical: {health_metrics.overall_health_score:.2f}',
                affected_components=['biofilm'],
                recommended_actions=['Immediate intervention required', 'Check all sensor readings', 'Consider system restart'],
                predicted_consequences='Biofilm failure, loss of power generation',
                confidence=health_metrics.assessment_confidence
            ))

        # Health decline alert
        elif health_metrics.health_trend == HealthTrend.DECLINING and health_metrics.overall_health_score < 0.6:
            alerts.append(HealthAlert(
                timestamp=timestamp,
                alert_type='health_decline',
                severity='high' if health_metrics.overall_health_score < 0.4 else 'medium',
                message=f'Biofilm health declining: {health_metrics.overall_health_score:.2f}',
                affected_components=['biofilm'],
                recommended_actions=['Monitor closely', 'Consider parameter adjustment', 'Check environmental conditions'],
                predicted_consequences='Continued decline may require intervention',
                confidence=health_metrics.assessment_confidence
            ))

        # Intervention needed alert
        if health_metrics.predicted_intervention_time is not None and health_metrics.predicted_intervention_time < 6.0:
            severity = 'critical' if health_metrics.predicted_intervention_time < 1.0 else 'high'
            alerts.append(HealthAlert(
                timestamp=timestamp,
                alert_type='intervention_needed',
                severity=severity,
                message=f'Intervention predicted in {health_metrics.predicted_intervention_time:.1f} hours',
                affected_components=['biofilm'],
                recommended_actions=['Prepare intervention strategy', 'Review recent parameter changes'],
                predicted_consequences='Health will drop below acceptable levels',
                confidence=health_metrics.prediction_confidence
            ))

        # Risk-based alerts
        high_risks = []
        if health_metrics.fouling_risk > self.alert_thresholds['risk_threshold']:
            high_risks.append('electrode fouling')
        if health_metrics.detachment_risk > self.alert_thresholds['risk_threshold']:
            high_risks.append('biofilm detachment')
        if health_metrics.stagnation_risk > self.alert_thresholds['risk_threshold']:
            high_risks.append('growth stagnation')

        if high_risks:
            alerts.append(HealthAlert(
                timestamp=timestamp,
                alert_type='risk_assessment',
                severity='medium',
                message=f'High risk factors detected: {", ".join(high_risks)}',
                affected_components=['biofilm'],
                recommended_actions=[f'Address {risk} risk' for risk in high_risks],
                predicted_consequences='Risk factors may lead to system degradation',
                confidence=0.8
            ))

        # Anomaly-based alerts
        if anomalies:
            critical_anomalies = [a for a in anomalies if a.severity == 'critical']
            if critical_anomalies:
                alerts.append(HealthAlert(
                    timestamp=timestamp,
                    alert_type='anomaly_detected',
                    severity='critical',
                    message=f'{len(critical_anomalies)} critical anomalies detected',
                    affected_components=list({sensor for a in critical_anomalies for sensor in a.affected_sensors}),
                    recommended_actions=[a.recommended_action for a in critical_anomalies[:3]],  # Top 3
                    predicted_consequences='System reliability compromised',
                    confidence=np.mean([a.confidence for a in critical_anomalies])
                ))

        # Store alerts
        self.alert_history.extend(alerts)

        # Clean old alerts (keep last 24 hours)
        cutoff_time = timestamp - 24.0
        self.alert_history = [a for a in self.alert_history if a.timestamp > cutoff_time]

        return alerts

    def generate_intervention_recommendations(self, health_metrics: HealthMetrics) -> list[InterventionRecommendation]:
        """Generate specific intervention recommendations."""
        recommendations = []

        # Health score based recommendations
        if health_metrics.overall_health_score < 0.4:
            recommendations.append(InterventionRecommendation(
                intervention_type='flow_adjustment',
                urgency='immediate' if health_metrics.overall_health_score < 0.3 else 'within_hours',
                description='Adjust flow rate to optimize nutrient delivery and waste removal',
                expected_benefit='Improve biofilm health and stability',
                implementation_difficulty='easy',
                estimated_duration=1.0,
                success_probability=0.7
            ))

        # Component-specific recommendations
        if health_metrics.thickness_health < 0.5:
            if health_metrics.fouling_risk > 0.6:
                recommendations.append(InterventionRecommendation(
                    intervention_type='cleaning',
                    urgency='within_hours',
                    description='Gentle electrode cleaning to remove excess biomass',
                    expected_benefit='Reduce fouling and improve electron transfer',
                    implementation_difficulty='moderate',
                    estimated_duration=2.0,
                    success_probability=0.8
                ))
            else:
                recommendations.append(InterventionRecommendation(
                    intervention_type='substrate_change',
                    urgency='within_days',
                    description='Adjust substrate concentration to promote healthy growth',
                    expected_benefit='Optimize biofilm thickness',
                    implementation_difficulty='easy',
                    estimated_duration=0.5,
                    success_probability=0.6
                ))

        if health_metrics.conductivity_health < 0.5:
            recommendations.append(InterventionRecommendation(
                intervention_type='substrate_change',
                urgency='within_hours',
                description='Optimize substrate type and concentration for electron transfer',
                expected_benefit='Improve biofilm conductivity and power output',
                implementation_difficulty='moderate',
                estimated_duration=4.0,
                success_probability=0.65
            ))

        if health_metrics.growth_health < 0.5:
            recommendations.append(InterventionRecommendation(
                intervention_type='monitoring',
                urgency='within_hours',
                description='Increase monitoring frequency and check environmental parameters',
                expected_benefit='Early detection of growth issues',
                implementation_difficulty='easy',
                estimated_duration=0.2,
                success_probability=0.9
            ))

        # Risk-based recommendations
        if health_metrics.detachment_risk > 0.7:
            recommendations.append(InterventionRecommendation(
                intervention_type='flow_adjustment',
                urgency='immediate',
                description='Reduce flow rate to minimize shear stress on biofilm',
                expected_benefit='Prevent biofilm detachment',
                implementation_difficulty='easy',
                estimated_duration=0.5,
                success_probability=0.8
            ))

        if health_metrics.stagnation_risk > 0.7:
            recommendations.append(InterventionRecommendation(
                intervention_type='substrate_change',
                urgency='within_days',
                description='Add growth stimulants or change nutrient composition',
                expected_benefit='Stimulate biofilm growth and activity',
                implementation_difficulty='moderate',
                estimated_duration=6.0,
                success_probability=0.7
            ))

        # Sort by urgency and success probability
        urgency_order = {'immediate': 0, 'within_hours': 1, 'within_days': 2, 'routine': 3}
        recommendations.sort(key=lambda r: (urgency_order.get(r.urgency, 3), -r.success_probability))

        return recommendations[:5]  # Return top 5 recommendations

    def get_health_dashboard_data(self) -> dict[str, Any]:
        """Get comprehensive data for health monitoring dashboard."""
        if not self.current_health:
            return {'error': 'No health data available'}

        # Recent health trend
        recent_scores = [h.overall_health_score for h in list(self.health_history)[-20:]]

        dashboard_data = {
            'current_health': {
                'overall_score': self.current_health.overall_health_score,
                'status': self.current_health.health_status.value,
                'trend': self.current_health.health_trend.value,
                'last_update': self.last_assessment_time
            },
            'component_health': {
                'thickness': self.current_health.thickness_health,
                'conductivity': self.current_health.conductivity_health,
                'growth': self.current_health.growth_health,
                'stability': self.current_health.stability_health
            },
            'risk_assessment': {
                'fouling_risk': self.current_health.fouling_risk,
                'detachment_risk': self.current_health.detachment_risk,
                'stagnation_risk': self.current_health.stagnation_risk
            },
            'predictions': {
                'health_24h': self.current_health.predicted_health_24h,
                'intervention_time': self.current_health.predicted_intervention_time,
                'prediction_confidence': self.current_health.prediction_confidence
            },
            'health_history': recent_scores,
            'active_alerts': len([a for a in self.alert_history if a.timestamp > self.last_assessment_time - 1.0]),
            'system_confidence': self.current_health.assessment_confidence
        }

        return dashboard_data


def create_predictive_health_monitor(species: BacterialSpecies = BacterialSpecies.MIXED,
                                   config: SensorConfig | None = None) -> PredictiveBiofimHealthMonitor:
    """
    Factory function to create predictive health monitor with optimal settings.

    Args:
        species: Target bacterial species
        config: Optional sensor configuration

    Returns:
        Configured PredictiveBiofimHealthMonitor instance
    """
    monitor = PredictiveBiofimHealthMonitor(
        species=species,
        config=config,
        history_window=100  # Store last 100 measurements
    )

    logger.info(f"Predictive biofilm health monitor created for {species.value}")
    return monitor
