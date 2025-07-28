"""
Degradation Pattern Detection for MFC Long-term Stability

Advanced algorithms for detecting and characterizing degradation patterns
in MFC components and system performance. Uses machine learning and statistical
analysis to identify early warning signs of component failure.

Created: 2025-07-28
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import warnings
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
import logging

class DegradationType(Enum):
    """Types of degradation patterns that can be detected."""
    MEMBRANE_FOULING = "membrane_fouling"
    ELECTRODE_CORROSION = "electrode_corrosion"
    BIOFILM_AGING = "biofilm_aging"
    CATALYST_DEACTIVATION = "catalyst_deactivation"
    STRUCTURAL_FATIGUE = "structural_fatigue"
    CHEMICAL_POISONING = "chemical_poisoning"
    THERMAL_DAMAGE = "thermal_damage"
    MECHANICAL_WEAR = "mechanical_wear"
    UNKNOWN = "unknown"


class DegradationSeverity(Enum):
    """Severity levels for degradation patterns."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    FAILURE = "failure"

class DegradationPattern:
    """Data structure for a detected degradation pattern."""
    pattern_id: str
    degradation_type: DegradationType
    severity: DegradationSeverity
    confidence: float  # 0-1
    start_time: datetime
    end_time: Optional[datetime] = None
    affected_components: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    trend_rate: float = 0.0  # Rate of degradation per hour
    predicted_failure_time: Optional[datetime] = None
    root_cause_analysis: Dict[str, Any] = field(default_factory=dict)
    mitigation_recommendations: List[str] = field(default_factory=list)

class TimeSeriesMetrics:
    """Statistical metrics for time series analysis."""
    mean: float
    std: float
    trend_slope: float
    trend_r2: float
    autocorrelation: float
    stationarity_p_value: float
    change_points: List[int]
    anomaly_score: float
    seasonality_strength: float
    noise_level: float

class DegradationDetector:
    """
    Advanced degradation pattern detection system.
    
    Uses multiple statistical and machine learning approaches to detect
    degradation patterns in MFC system data:
    - Statistical process control
    - Change point detection
    - Anomaly detection
    - Trend analysis
    - Pattern recognition
    """
    
    def __init__(self, 
                 detection_window_hours: int = 168,  # 1 week
                 min_pattern_duration_hours: int = 24,  # 1 day
                 confidence_threshold: float = 0.7,
                 anomaly_contamination: float = 0.1):
        """
        Initialize the degradation detector.
        
        Args:
            detection_window_hours: Sliding window size for analysis
            min_pattern_duration_hours: Minimum duration to consider a pattern
            confidence_threshold: Minimum confidence to report a pattern
            anomaly_contamination: Expected fraction of anomalies
        """
        self.detection_window_hours = detection_window_hours
        self.min_pattern_duration_hours = min_pattern_duration_hours
        self.confidence_threshold = confidence_threshold
        self.anomaly_contamination = anomaly_contamination
        
        # Detection models
        self.anomaly_detector = IsolationForest(
            contamination=anomaly_contamination,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        
        # Historical data storage
        self.data_history: List[Dict[str, Any]] = []
        self.detected_patterns: List[DegradationPattern] = []
        self.baseline_metrics: Dict[str, TimeSeriesMetrics] = {}
        
        # Component mapping for degradation types
        self.component_degradation_map = {
            'membrane': [DegradationType.MEMBRANE_FOULING, DegradationType.CHEMICAL_POISONING],
            'anode': [DegradationType.ELECTRODE_CORROSION, DegradationType.BIOFILM_AGING],
            'cathode': [DegradationType.ELECTRODE_CORROSION, DegradationType.CATALYST_DEACTIVATION],
            'separator': [DegradationType.STRUCTURAL_FATIGUE, DegradationType.CHEMICAL_POISONING],
            'housing': [DegradationType.MECHANICAL_WEAR, DegradationType.THERMAL_DAMAGE]
        }
        
        # Pattern signatures (thresholds and indicators)
        self.pattern_signatures = self._initialize_pattern_signatures()
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _initialize_pattern_signatures(self) -> Dict[DegradationType, Dict[str, Any]]:
        """Initialize pattern signatures for different degradation types."""
        return {
            DegradationType.MEMBRANE_FOULING: {
                'resistance_increase_rate': 0.1,  # Ω·h⁻¹
                'power_decline_rate': -0.01,      # W·h⁻¹
                'voltage_drop_threshold': 0.05,   # V
                'efficiency_decline_rate': -0.001  # %·h⁻¹
            },
            DegradationType.ELECTRODE_CORROSION: {
                'voltage_decline_rate': -0.02,    # V·h⁻¹
                'current_instability': 0.1,       # Coefficient of variation
                'power_decline_rate': -0.015,     # W·h⁻¹
                'impedance_increase_rate': 0.05   # Ω·h⁻¹
            },
            DegradationType.BIOFILM_AGING: {
                'biofilm_thickness_plateau': 100e-6,  # m
                'current_density_decline': -0.1,       # A/m²·h⁻¹
                'substrate_efficiency_drop': -0.005,   # %·h⁻¹
                'pH_instability': 0.2                  # pH units std
            },
            DegradationType.CATALYST_DEACTIVATION: {
                'cathode_potential_drop': -0.01,  # V·h⁻¹
                'oxygen_reduction_decline': -0.02, # A·h⁻¹
                'activation_overpotential': 0.1,   # V increase
                'temperature_sensitivity': 1.5     # Performance change per °C
            },
            DegradationType.STRUCTURAL_FATIGUE: {
                'mechanical_stress_cycles': 1000,  # Cycles
                'deformation_rate': 1e-6,          # m·cycle⁻¹
                'seal_integrity_loss': 0.01,       # Pressure drop rate
                'vibration_increase': 0.1          # Amplitude increase rate
            },
            DegradationType.CHEMICAL_POISONING: {
                'sudden_performance_drop': 0.2,    # Fraction
                'recovery_time_constant': 48,      # Hours
                'selectivity_loss': 0.1,           # Efficiency drop
                'byproduct_accumulation': 0.01     # Concentration increase
            },
            DegradationType.THERMAL_DAMAGE: {
                'temperature_excursion': 5.0,      # °C above normal
                'thermal_runaway_rate': 2.0,       # °C·h⁻¹
                'material_property_change': 0.05,  # Fraction
                'thermal_cycling_damage': 100      # Temperature cycles
            },
            DegradationType.MECHANICAL_WEAR: {
                'wear_rate': 1e-9,                 # m·h⁻¹
                'friction_increase': 0.01,         # Coefficient increase
                'alignment_drift': 1e-6,           # m·h⁻¹
                'component_loosening': 0.1         # Torque loss rate
            }
        }
    
    def add_data_point(self, data: Dict[str, Any]):
        """Add a new data point for analysis."""
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now()
        
        self.data_history.append(data)
        
        # Keep only recent data within detection window
        cutoff_time = datetime.now() - timedelta(hours=self.detection_window_hours)
        self.data_history = [
            d for d in self.data_history 
            if d['timestamp'] >= cutoff_time
        ]
    
    def calculate_time_series_metrics(self, values: np.ndarray, 
                                    timestamps: np.ndarray) -> TimeSeriesMetrics:
        """Calculate comprehensive time series metrics."""
        if len(values) < 10:
            # Not enough data for meaningful analysis
            return TimeSeriesMetrics(
                mean=np.mean(values), std=np.std(values),
                trend_slope=0.0, trend_r2=0.0, autocorrelation=0.0,
                stationarity_p_value=1.0, change_points=[], anomaly_score=0.0,
                seasonality_strength=0.0, noise_level=np.std(values)
            )
        
        # Basic statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Trend analysis
        time_numeric = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
        
        # Autocorrelation (lag-1)
        if len(values) > 1:
            autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0
        
        # Stationarity test (Augmented Dickey-Fuller)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(values)
            stationarity_p = adf_result[1]
        except ImportError:
            # Fallback: use variance stability test
            if len(values) >= 20:
                mid = len(values) // 2
                var1 = np.var(values[:mid])
                var2 = np.var(values[mid:])
                f_stat = max(var1, var2) / (min(var1, var2) + 1e-10)
                stationarity_p = 1.0 / (1.0 + f_stat)  # Rough approximation
            else:
                stationarity_p = 1.0
        
        # Change point detection (simple method)
        change_points = self._detect_change_points(values)
        
        # Anomaly score using isolation forest
        try:
            values_reshaped = values.reshape(-1, 1)
            anomaly_scores = self.anomaly_detector.fit(values_reshaped).decision_function(values_reshaped)
            anomaly_score = np.mean(np.abs(anomaly_scores))
        except:
            anomaly_score = 0.0
        
        # Seasonality strength (simplified)
        seasonality_strength = self._estimate_seasonality(values)
        
        # Noise level estimation
        if len(values) >= 3:
            # High-frequency component as noise estimate
            diff2 = np.diff(values, n=2)
            noise_level = np.std(diff2) / np.sqrt(6)  # Normalized
        else:
            noise_level = std_val
        
        return TimeSeriesMetrics(
            mean=mean_val,
            std=std_val,
            trend_slope=slope,
            trend_r2=r_value**2,
            autocorrelation=autocorr,
            stationarity_p_value=stationarity_p,
            change_points=change_points,
            anomaly_score=anomaly_score,
            seasonality_strength=seasonality_strength,
            noise_level=noise_level
        )
    
    def _detect_change_points(self, values: np.ndarray, 
                             min_segment_length: int = 10) -> List[int]:
        """Detect change points in time series using PELT algorithm approximation."""
        if len(values) < 2 * min_segment_length:
            return []
        
        change_points = []
        
        # Simple variance-based change point detection
        window_size = max(min_segment_length, len(values) // 10)
        
        for i in range(window_size, len(values) - window_size):
            # Calculate variance before and after potential change point
            before = values[max(0, i - window_size):i]
            after = values[i:min(len(values), i + window_size)]
            
            if len(before) < 3 or len(after) < 3:
                continue
            
            var_before = np.var(before)
            var_after = np.var(after)
            
            # F-test for variance change
            if var_before > 0 and var_after > 0:
                f_stat = max(var_before, var_after) / min(var_before, var_after)
                if f_stat > 4.0:  # Threshold for significant change
                    change_points.append(i)
        
        # Remove nearby change points
        if change_points:
            filtered_points = [change_points[0]]
            for cp in change_points[1:]:
                if cp - filtered_points[-1] >= min_segment_length:
                    filtered_points.append(cp)
            change_points = filtered_points
        
        return change_points
    
    def _estimate_seasonality(self, values: np.ndarray, 
                            max_period: int = 24) -> float:
        """Estimate seasonality strength in time series."""
        if len(values) < 2 * max_period:
            return 0.0
        
        # Try different periods and find strongest correlation
        max_correlation = 0.0
        
        for period in range(2, min(max_period + 1, len(values) // 2)):
            # Split into segments of this period
            n_segments = len(values) // period
            if n_segments < 2:
                continue
            
            segments = values[:n_segments * period].reshape(n_segments, period)
            
            # Calculate correlation between segments
            correlations = []
            for i in range(n_segments - 1):
                corr = np.corrcoef(segments[i], segments[i + 1])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            
            if correlations:
                avg_corr = np.mean(correlations)
                max_correlation = max(max_correlation, avg_corr)
        
        return max_correlation
    
    def detect_degradation_patterns(self) -> List[DegradationPattern]:
        """Detect degradation patterns in the current data."""
        if len(self.data_history) < self.min_pattern_duration_hours:
            return []
        
        # Convert data to DataFrame for easier analysis
        df = pd.DataFrame(self.data_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        detected_patterns = []
        
        # Analyze each metric for degradation patterns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in ['timestamp']:
                continue
            
            values = df[column].values
            timestamps = df['timestamp'].values
            
            # Skip if insufficient data or all NaN
            if len(values) < 10 or np.all(np.isnan(values)):
                continue
            
            # Calculate metrics
            metrics = self.calculate_time_series_metrics(values, timestamps)
            
            # Check for each degradation type
            for degradation_type in DegradationType:
                pattern = self._check_degradation_signature(
                    column, values, timestamps, metrics, degradation_type
                )
                
                if pattern and pattern.confidence >= self.confidence_threshold:
                    detected_patterns.append(pattern)
        
        # Remove duplicate and overlapping patterns
        filtered_patterns = self._filter_duplicate_patterns(detected_patterns)
        
        # Update internal pattern list
        self.detected_patterns.extend(filtered_patterns)
        
        return filtered_patterns
    
    def _check_degradation_signature(self, 
                                   metric_name: str,
                                   values: np.ndarray,
                                   timestamps: np.ndarray,
                                   metrics: TimeSeriesMetrics,
                                   degradation_type: DegradationType) -> Optional[DegradationPattern]:
        """Check if a metric shows signs of a specific degradation type."""
        signature = self.pattern_signatures.get(degradation_type, {})
        if not signature:
            return None
        
        confidence = 0.0
        evidence = {}
        
        # Check trend-based indicators
        if degradation_type == DegradationType.MEMBRANE_FOULING:
            if 'resistance' in metric_name.lower():
                if metrics.trend_slope > signature.get('resistance_increase_rate', 0):
                    confidence += 0.3
                    evidence['resistance_trend'] = metrics.trend_slope
            
            elif 'power' in metric_name.lower():
                if metrics.trend_slope < signature.get('power_decline_rate', 0):
                    confidence += 0.3
                    evidence['power_trend'] = metrics.trend_slope
            
            elif 'voltage' in metric_name.lower():
                recent_drop = values[-1] - values[0] if len(values) > 1 else 0
                if recent_drop < -signature.get('voltage_drop_threshold', 0):
                    confidence += 0.2
                    evidence['voltage_drop'] = recent_drop
        
        elif degradation_type == DegradationType.ELECTRODE_CORROSION:
            if 'voltage' in metric_name.lower():
                if metrics.trend_slope < signature.get('voltage_decline_rate', 0):
                    confidence += 0.4
                    evidence['voltage_decline'] = metrics.trend_slope
            
            elif 'current' in metric_name.lower():
                cv = metrics.std / (abs(metrics.mean) + 1e-10)
                if cv > signature.get('current_instability', 0):
                    confidence += 0.3
                    evidence['current_instability'] = cv
        
        elif degradation_type == DegradationType.BIOFILM_AGING:
            if 'biofilm' in metric_name.lower() and 'thickness' in metric_name.lower():
                if metrics.mean > signature.get('biofilm_thickness_plateau', 0):
                    confidence += 0.3
                    evidence['biofilm_plateau'] = metrics.mean
            
            elif 'current_density' in metric_name.lower():
                if metrics.trend_slope < signature.get('current_density_decline', 0):
                    confidence += 0.3
                    evidence['current_density_trend'] = metrics.trend_slope
        
        # Add statistical evidence
        if metrics.stationarity_p_value < 0.05:  # Non-stationary (changing mean/variance)
            confidence += 0.1
            evidence['non_stationary'] = True
        
        if len(metrics.change_points) > 0:
            confidence += 0.1
            evidence['change_points'] = len(metrics.change_points)
        
        if metrics.anomaly_score > 0.5:
            confidence += 0.1
            evidence['high_anomaly_score'] = metrics.anomaly_score
        
        # Determine severity based on trend magnitude and confidence
        severity = self._determine_severity(metrics, evidence, degradation_type)
        
        if confidence < 0.3:  # Minimum threshold for detection
            return None
        
        # Create pattern object
        pattern = DegradationPattern(
            pattern_id=f"{degradation_type.value}_{metric_name}_{len(self.detected_patterns)}",
            degradation_type=degradation_type,
            severity=severity,
            confidence=min(confidence, 1.0),
            start_time=pd.to_datetime(timestamps[0]) if len(timestamps) > 0 else datetime.now(),
            end_time=pd.to_datetime(timestamps[-1]) if len(timestamps) > 0 else None,
            affected_components=self._identify_affected_components(metric_name, degradation_type),
            metrics={
                'trend_slope': metrics.trend_slope,
                'trend_r2': metrics.trend_r2,
                'anomaly_score': metrics.anomaly_score,
                'change_points': len(metrics.change_points),
                'stationarity_p': metrics.stationarity_p_value,
                **evidence
            },
            trend_rate=metrics.trend_slope,
            predicted_failure_time=self._predict_failure_time(metrics, degradation_type),
            root_cause_analysis=self._analyze_root_cause(metrics, evidence, degradation_type),
            mitigation_recommendations=self._generate_mitigation_recommendations(degradation_type, severity)
        )
        
        return pattern
    
    def _determine_severity(self, 
                          metrics: TimeSeriesMetrics,
                          evidence: Dict[str, Any],
                          degradation_type: DegradationType) -> DegradationSeverity:
        """Determine severity level based on metrics and evidence."""
        # Base severity on trend magnitude and confidence
        trend_magnitude = abs(metrics.trend_slope)
        anomaly_level = metrics.anomaly_score
        
        # Normalize based on degradation type
        severity_score = 0.0
        
        if degradation_type == DegradationType.MEMBRANE_FOULING:
            severity_score = trend_magnitude * 10 + anomaly_level * 0.5
        elif degradation_type == DegradationType.ELECTRODE_CORROSION:
            severity_score = trend_magnitude * 5 + anomaly_level * 0.3
        else:
            severity_score = trend_magnitude * 2 + anomaly_level * 0.2
        
        # Add evidence-based adjustments
        if 'change_points' in evidence and evidence['change_points'] > 2:
            severity_score += 0.2
        
        if evidence.get('non_stationary', False):
            severity_score += 0.1
        
        # Map to severity levels
        if severity_score < 0.1:
            return DegradationSeverity.MINIMAL
        elif severity_score < 0.3:
            return DegradationSeverity.LOW
        elif severity_score < 0.6:
            return DegradationSeverity.MODERATE
        elif severity_score < 1.0:
            return DegradationSeverity.HIGH
        elif severity_score < 2.0:
            return DegradationSeverity.CRITICAL
        else:
            return DegradationSeverity.FAILURE
    
    def _identify_affected_components(self, 
                                    metric_name: str,
                                    degradation_type: DegradationType) -> List[str]:
        """Identify which components are affected by the degradation."""
        affected = []
        
        # Map metric names to components
        if any(term in metric_name.lower() for term in ['membrane', 'resistance', 'conductivity']):
            affected.append('membrane')
        
        if any(term in metric_name.lower() for term in ['anode', 'biofilm']):
            affected.append('anode')
        
        if any(term in metric_name.lower() for term in ['cathode', 'oxygen']):
            affected.append('cathode')
        
        if any(term in metric_name.lower() for term in ['separator', 'electrolyte']):
            affected.append('separator')
        
        if any(term in metric_name.lower() for term in ['housing', 'structure', 'mechanical']):
            affected.append('housing')
        
        # Add based on degradation type
        for component, types in self.component_degradation_map.items():
            if degradation_type in types and component not in affected:
                affected.append(component)
        
        return affected if affected else ['unknown']
    
    def _predict_failure_time(self, 
                            metrics: TimeSeriesMetrics,
                            degradation_type: DegradationType) -> Optional[datetime]:
        """Predict when failure might occur based on current trends."""
        if abs(metrics.trend_slope) < 1e-10:  # No significant trend
            return None
        
        # Define failure thresholds for different degradation types
        failure_thresholds = {
            DegradationType.MEMBRANE_FOULING: {'resistance': 10.0, 'power': 0.1},
            DegradationType.ELECTRODE_CORROSION: {'voltage': 0.1, 'current': 0.01},
            DegradationType.BIOFILM_AGING: {'thickness': 200e-6, 'current_density': 0.1},
            DegradationType.CATALYST_DEACTIVATION: {'potential': 0.05, 'current': 0.1},
            DegradationType.STRUCTURAL_FATIGUE: {'stress': 1000, 'deformation': 1e-4},
        }
        
        threshold = failure_thresholds.get(degradation_type, {}).get('default', 1.0)
        
        # Simple linear extrapolation
        current_value = metrics.mean
        slope = metrics.trend_slope
        
        if degradation_type in [DegradationType.MEMBRANE_FOULING, DegradationType.ELECTRODE_CORROSION]:
            # For degradation that increases values
            if slope > 0:
                hours_to_failure = (threshold - current_value) / slope
            else:
                return None
        else:
            # For degradation that decreases values
            if slope < 0:
                hours_to_failure = (current_value - threshold) / abs(slope)
            else:
                return None
        
        if hours_to_failure > 0 and hours_to_failure < 8760:  # Within 1 year
            return datetime.now() + timedelta(hours=hours_to_failure)
        
        return None
    
    def _analyze_root_cause(self, 
                          metrics: TimeSeriesMetrics,
                          evidence: Dict[str, Any],
                          degradation_type: DegradationType) -> Dict[str, Any]:
        """Analyze potential root causes of degradation."""
        root_cause = {
            'degradation_type': degradation_type.value,
            'primary_indicators': [],
            'contributing_factors': [],
            'confidence_factors': []
        }
        
        if degradation_type == DegradationType.MEMBRANE_FOULING:
            root_cause['primary_indicators'] = [
                'Increasing membrane resistance',
                'Declining power output',
                'Voltage drop across membrane'
            ]
            root_cause['contributing_factors'] = [
                'High substrate concentration',
                'Low flow velocity',
                'Inadequate membrane cleaning',
                'Biofilm accumulation'
            ]
        
        elif degradation_type == DegradationType.ELECTRODE_CORROSION:
            root_cause['primary_indicators'] = [
                'Voltage decline over time',
                'Current instability',
                'Increased impedance'
            ]
            root_cause['contributing_factors'] = [
                'Low pH conditions',
                'High oxygen exposure',
                'Material incompatibility',
                'Electrochemical stress'
            ]
        
        elif degradation_type == DegradationType.BIOFILM_AGING:
            root_cause['primary_indicators'] = [
                'Biofilm thickness plateau',
                'Declining current density',
                'Reduced substrate efficiency'
            ]
            root_cause['contributing_factors'] = [
                'Nutrient depletion',
                'Toxic metabolite accumulation',
                'pH drift',
                'Temperature fluctuations'
            ]
        
        # Add statistical evidence
        if metrics.stationarity_p_value < 0.05:
            root_cause['confidence_factors'].append('Non-stationary behavior detected')
        
        if len(metrics.change_points) > 0:
            root_cause['confidence_factors'].append(f'{len(metrics.change_points)} change points detected')
        
        if metrics.anomaly_score > 0.5:
            root_cause['confidence_factors'].append(f'High anomaly score: {metrics.anomaly_score:.2f}')
        
        return root_cause
    
    def _generate_mitigation_recommendations(self, 
                                           degradation_type: DegradationType,
                                           severity: DegradationSeverity) -> List[str]:
        """Generate mitigation recommendations based on degradation type and severity."""
        recommendations = []
        
        base_recommendations = {
            DegradationType.MEMBRANE_FOULING: [
                "Increase membrane cleaning frequency",
                "Reduce substrate concentration",
                "Optimize flow velocity",
                "Consider membrane replacement",
                "Implement backwash procedures"
            ],
            DegradationType.ELECTRODE_CORROSION: [
                "Check and adjust pH levels",
                "Reduce oxygen exposure",
                "Inspect electrode materials",
                "Consider protective coatings",
                "Monitor electrochemical conditions"
            ],
            DegradationType.BIOFILM_AGING: [
                "Refresh biofilm culture",
                "Optimize nutrient supply",
                "Monitor and control pH",
                "Remove toxic metabolites",
                "Consider biofilm rejuvenation"
            ],
            DegradationType.CATALYST_DEACTIVATION: [
                "Check catalyst loading",
                "Remove catalyst poisons",
                "Optimize reaction conditions",
                "Consider catalyst regeneration",
                "Monitor temperature control"
            ],
            DegradationType.STRUCTURAL_FATIGUE: [
                "Inspect structural components",
                "Reduce mechanical stress",
                "Check fastener tightness",
                "Monitor vibration levels",
                "Schedule preventive maintenance"
            ]
        }
        
        base_recs = base_recommendations.get(degradation_type, [
            "Monitor system closely",
            "Investigate root cause",
            "Consult maintenance manual"
        ])
        
        # Add severity-based urgency
        if severity in [DegradationSeverity.CRITICAL, DegradationSeverity.FAILURE]:
            recommendations.append("URGENT: Immediate attention required")
            recommendations.append("Consider system shutdown if safety risk")
        
        elif severity == DegradationSeverity.HIGH:
            recommendations.append("HIGH PRIORITY: Schedule maintenance within 24 hours")
        
        elif severity == DegradationSeverity.MODERATE:
            recommendations.append("Schedule maintenance within one week")
        
        recommendations.extend(base_recs[:3])  # Top 3 recommendations
        
        return recommendations
    
    def _filter_duplicate_patterns(self, 
                                 patterns: List[DegradationPattern]) -> List[DegradationPattern]:
        """Remove duplicate and overlapping patterns."""
        if not patterns:
            return []
        
        # Sort by confidence (highest first)
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        filtered = []
        
        for pattern in patterns:
            # Check if similar pattern already exists
            is_duplicate = False
            
            for existing in filtered:
                # Same degradation type and overlapping components
                if (pattern.degradation_type == existing.degradation_type and
                    any(comp in existing.affected_components for comp in pattern.affected_components)):
                    
                    # Check time overlap (allow some tolerance)
                    time_overlap = (
                        pattern.start_time <= existing.end_time and
                        existing.start_time <= pattern.end_time
                    ) if existing.end_time and pattern.end_time else True
                    
                    if time_overlap:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(pattern)
        
        return filtered
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get a summary of all detected patterns."""
        if not self.detected_patterns:
            return {
                'total_patterns': 0,
                'by_type': {},
                'by_severity': {},
                'active_patterns': 0,
                'predictions': []
            }
        
        # Count by type
        by_type = {}
        for pattern in self.detected_patterns:
            pattern_type = pattern.degradation_type.value
            by_type[pattern_type] = by_type.get(pattern_type, 0) + 1
        
        # Count by severity
        by_severity = {}
        for pattern in self.detected_patterns:
            severity = pattern.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        # Active patterns (no end time or recent end time)
        now = datetime.now()
        active_patterns = 0
        predictions = []
        
        for pattern in self.detected_patterns:
            is_active = (
                pattern.end_time is None or
                (now - pattern.end_time).total_seconds() < 3600  # Within last hour
            )
            
            if is_active:
                active_patterns += 1
                
                if pattern.predicted_failure_time:
                    predictions.append({
                        'pattern_id': pattern.pattern_id,
                        'degradation_type': pattern.degradation_type.value,
                        'predicted_failure': pattern.predicted_failure_time.isoformat(),
                        'confidence': pattern.confidence,
                        'severity': pattern.severity.value
                    })
        
        return {
            'total_patterns': len(self.detected_patterns),
            'by_type': by_type,
            'by_severity': by_severity,
            'active_patterns': active_patterns,
            'predictions': sorted(predictions, key=lambda x: x['predicted_failure'])
        }
    
    def export_patterns_report(self, filepath: str):
        """Export detailed patterns report to file."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'detector_config': {
                'detection_window_hours': self.detection_window_hours,
                'min_pattern_duration_hours': self.min_pattern_duration_hours,
                'confidence_threshold': self.confidence_threshold,
                'anomaly_contamination': self.anomaly_contamination
            },
            'summary': self.get_pattern_summary(),
            'detailed_patterns': []
        }
        
        for pattern in self.detected_patterns:
            pattern_dict = {
                'pattern_id': pattern.pattern_id,
                'degradation_type': pattern.degradation_type.value,
                'severity': pattern.severity.value,
                'confidence': pattern.confidence,
                'start_time': pattern.start_time.isoformat(),
                'end_time': pattern.end_time.isoformat() if pattern.end_time else None,
                'affected_components': pattern.affected_components,
                'metrics': pattern.metrics,
                'trend_rate': pattern.trend_rate,
                'predicted_failure_time': pattern.predicted_failure_time.isoformat() if pattern.predicted_failure_time else None,
                'root_cause_analysis': pattern.root_cause_analysis,
                'mitigation_recommendations': pattern.mitigation_recommendations
            }
            report['detailed_patterns'].append(pattern_dict)
        
        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Degradation patterns report exported to {filepath}")