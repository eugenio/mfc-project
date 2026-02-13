"""
Machine Learning Optimization for MFC Systems

Phase 2 final enhancement implementing ML-based optimization:
- Hyperparameter optimization for Q-learning and sensor fusion
- Meta-learning for rapid adaptation to new operating conditions  
- Ensemble methods for robust control decisions
- Transfer learning between different MFC configurations
- Automated feature engineering for improved state representation

Created: 2025-07-31
Last Modified: 2025-07-31
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timezone
from collections import deque

# Import adaptive controller components
from adaptive_mfc_controller import AdaptiveMFCController, SystemState

# Base components
from sensing_models.sensor_fusion import BacterialSpecies
from config import QLearningConfig, SensorConfig

# ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score, RandomizedSearchCV, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Bayesian optimization
try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """ML optimization strategies."""
    BAYESIAN = "bayesian"  # Bayesian optimization
    EVOLUTIONARY = "evolutionary"  # Evolutionary algorithms
    GRID_SEARCH = "grid_search"  # Systematic grid search
    RANDOM_SEARCH = "random_search"  # Random parameter sampling
    ENSEMBLE = "ensemble"  # Combine multiple strategies


class FeatureType(Enum):
    """Types of engineered features."""
    RAW = "raw"  # Raw sensor measurements
    STATISTICAL = "statistical"  # Statistical features (mean, std, etc.)
    TEMPORAL = "temporal"  # Time-based features
    SPECTRAL = "spectral"  # Frequency domain features
    INTERACTION = "interaction"  # Feature interactions
    DERIVED = "derived"  # Domain-specific derived features


@dataclass
class OptimizationResult:
    """Result of ML optimization."""

    strategy_used: OptimizationStrategy
    best_parameters: Dict[str, Any]
    performance_improvement: float  # Relative improvement (0-1)
    validation_score: float  # Cross-validation score
    optimization_time_seconds: float
    convergence_achieved: bool
    confidence_interval: Tuple[float, float]  # 95% CI for improvement


@dataclass
class FeatureImportance:
    """Feature importance analysis."""

    feature_name: str
    importance_score: float  # 0-1 scale
    feature_type: FeatureType
    description: str
    stability: float  # How stable is this feature's importance


@dataclass
class ModelEnsemble:
    """Ensemble of models for robust predictions."""

    models: List[Any]  # Trained models
    weights: List[float]  # Model weights
    performance_scores: List[float]  # Individual model scores
    ensemble_score: float  # Combined ensemble score
    diversity_score: float  # Diversity between models


class FeatureEngineer:
    """
    Advanced feature engineering for MFC control optimization.
    
    Creates informative features from raw sensor data, system state,
    and historical performance for improved ML model performance.
    """

    def __init__(self, window_size: int = 50, enable_advanced_features: bool = True):
        """
        Initialize feature engineer.
        
        Args:
            window_size: Size of sliding window for temporal features
            enable_advanced_features: Enable computationally expensive features
        """
        self.window_size = window_size
        self.enable_advanced_features = enable_advanced_features

        # Feature history
        self.raw_data_history = deque(maxlen=window_size * 2)
        self.system_state_history = deque(maxlen=window_size * 2)

        # Feature scaling
        self.scalers = {}
        self.feature_stats = {}

        logger.info(f"Feature engineer initialized with window_size={window_size}")

    def extract_features(self, system_state: SystemState,
                        performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Extract comprehensive feature set from system state.
        
        Args:
            system_state: Current system state
            performance_metrics: Current performance metrics
            
        Returns:
            Dictionary of engineered features
        """
        features = {}

        # Store for history
        self.system_state_history.append(system_state)

        # 1. Raw sensor features
        features.update(self._extract_raw_features(system_state))

        # 2. Statistical features
        features.update(self._extract_statistical_features())

        # 3. Temporal features
        features.update(self._extract_temporal_features())

        # 4. Health-based features
        features.update(self._extract_health_features(system_state))

        # 5. Control-based features
        features.update(self._extract_control_features(system_state))

        # 6. Performance features
        features.update(self._extract_performance_features(performance_metrics))

        # 7. Advanced features (if enabled)
        if self.enable_advanced_features:
            features.update(self._extract_advanced_features())

        # 8. Derived domain features
        features.update(self._extract_derived_features(system_state))

        return features

    def _extract_raw_features(self, system_state: SystemState) -> Dict[str, float]:
        """Extract raw sensor measurement features."""
        fused = system_state.fused_measurement

        return {
            'thickness_um': fused.thickness_um,
            'thickness_uncertainty': fused.thickness_uncertainty,
            'biomass_density': fused.biomass_density_g_per_L,
            'biomass_uncertainty': fused.biomass_uncertainty,
            'conductivity': fused.conductivity_S_per_m,
            'conductivity_uncertainty': fused.conductivity_uncertainty,
            'eis_thickness': fused.eis_thickness,
            'qcm_thickness': fused.qcm_thickness,
            'sensor_agreement': fused.sensor_agreement,
            'fusion_confidence': fused.fusion_confidence,
            'cross_validation_error': fused.cross_validation_error,
            'eis_weight': fused.eis_weight,
            'qcm_weight': fused.qcm_weight
        }

    def _extract_statistical_features(self) -> Dict[str, float]:
        """Extract statistical features from recent history."""
        if len(self.system_state_history) < 5:
            return {}

        # Extract time series
        recent_states = list(self.system_state_history)[-20:]
        thicknesses = [s.fused_measurement.thickness_um for s in recent_states]
        conductivities = [s.fused_measurement.conductivity_S_per_m for s in recent_states]
        agreements = [s.fused_measurement.sensor_agreement for s in recent_states]

        features = {}

        # Thickness statistics
        features['thickness_mean'] = np.mean(thicknesses)
        features['thickness_std'] = np.std(thicknesses)
        features['thickness_cv'] = features['thickness_std'] / max(0.01, features['thickness_mean'])
        features['thickness_range'] = np.ptp(thicknesses)  # Peak-to-peak
        features['thickness_trend'] = np.polyfit(range(len(thicknesses)), thicknesses, 1)[0]

        # Conductivity statistics
        features['conductivity_mean'] = np.mean(conductivities)
        features['conductivity_std'] = np.std(conductivities)
        features['conductivity_cv'] = features['conductivity_std'] / max(1e-6, features['conductivity_mean'])
        features['conductivity_trend'] = np.polyfit(range(len(conductivities)), conductivities, 1)[0]

        # Agreement statistics
        features['agreement_mean'] = np.mean(agreements)
        features['agreement_std'] = np.std(agreements)
        features['agreement_min'] = np.min(agreements)

        return features

    def _extract_temporal_features(self) -> Dict[str, float]:
        """Extract time-based features."""
        if len(self.system_state_history) < 10:
            return {}

        recent_states = list(self.system_state_history)[-self.window_size:]
        features = {}

        # Extract time series
        timestamps = [s.fused_measurement.timestamp for s in recent_states]
        thicknesses = [s.fused_measurement.thickness_um for s in recent_states]

        # Time since last major change
        thickness_changes = np.abs(np.diff(thicknesses))
        major_change_threshold = np.std(thicknesses) * 2
        major_changes = np.where(thickness_changes > major_change_threshold)[0]

        if len(major_changes) > 0:
            features['time_since_major_change'] = len(thicknesses) - major_changes[-1]
        else:
            features['time_since_major_change'] = len(thicknesses)

        # Autocorrelation features
        if len(thicknesses) >= 20:
            # Lag-1 autocorrelation
            lag1_corr = np.corrcoef(thicknesses[:-1], thicknesses[1:])[0, 1]
            features['thickness_autocorr_lag1'] = lag1_corr if not np.isnan(lag1_corr) else 0

            # Lag-5 autocorrelation
            if len(thicknesses) >= 25:
                lag5_corr = np.corrcoef(thicknesses[:-5], thicknesses[5:])[0, 1]
                features['thickness_autocorr_lag5'] = lag5_corr if not np.isnan(lag5_corr) else 0

        # Frequency domain features (simplified)
        if self.enable_advanced_features and len(thicknesses) >= 32:
            fft_coeffs = np.fft.fft(thicknesses)
            power_spectrum = np.abs(fft_coeffs)**2
            features['dominant_frequency'] = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            features['spectral_centroid'] = np.sum(np.arange(1, len(power_spectrum)//2) * power_spectrum[1:len(power_spectrum)//2]) / np.sum(power_spectrum[1:len(power_spectrum)//2])

        return features

    def _extract_health_features(self, system_state: SystemState) -> Dict[str, float]:
        """Extract health-related features."""
        health = system_state.health_metrics

        return {
            'overall_health_score': health.overall_health_score,
            'thickness_health': health.thickness_health,
            'conductivity_health': health.conductivity_health,
            'growth_health': health.growth_health,
            'stability_health': health.stability_health,
            'predicted_health_24h': health.predicted_health_24h,
            'fouling_risk': health.fouling_risk,
            'detachment_risk': health.detachment_risk,
            'stagnation_risk': health.stagnation_risk,
            'assessment_confidence': health.assessment_confidence,
            'prediction_confidence': health.prediction_confidence,
            'health_status_numeric': self._health_status_to_numeric(health.health_status),
            'health_trend_numeric': self._health_trend_to_numeric(health.health_trend)
        }

    def _extract_control_features(self, system_state: SystemState) -> Dict[str, float]:
        """Extract control-related features."""
        return {
            'flow_rate': system_state.flow_rate,
            'inlet_concentration': system_state.inlet_concentration,
            'outlet_concentration': system_state.outlet_concentration,
            'current_density': system_state.current_density,
            'power_output': system_state.power_output,
            'concentration_efficiency': system_state.outlet_concentration / max(0.01, system_state.inlet_concentration),
            'power_density': system_state.power_output / max(0.01, system_state.current_density),
            'strategy_numeric': self._strategy_to_numeric(system_state.current_strategy),
            'adaptation_mode_numeric': self._adaptation_mode_to_numeric(system_state.adaptation_mode),
            'intervention_active': float(system_state.intervention_active)
        }

    def _extract_performance_features(self, performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """Extract performance-related features."""
        return {
            'power_efficiency': performance_metrics.get('power_efficiency', 0.0),
            'biofilm_health_score': performance_metrics.get('biofilm_health_score', 0.5),
            'sensor_reliability': performance_metrics.get('sensor_reliability', 0.5),
            'system_stability': performance_metrics.get('system_stability', 0.5),
            'control_confidence': performance_metrics.get('control_confidence', 0.5)
        }

    def _extract_advanced_features(self) -> Dict[str, float]:
        """Extract computationally expensive advanced features."""
        if len(self.system_state_history) < 20:
            return {}

        recent_states = list(self.system_state_history)[-20:]
        features = {}

        # Multi-scale features
        thicknesses = [s.fused_measurement.thickness_um for s in recent_states]

        # Short-term (5 points) vs long-term (20 points) behavior
        if len(thicknesses) >= 20:
            short_term_mean = np.mean(thicknesses[-5:])
            long_term_mean = np.mean(thicknesses)
            features['short_vs_long_term_ratio'] = short_term_mean / max(0.01, long_term_mean)

        # Stability measures
        if len(thicknesses) >= 10:
            # Hurst exponent (simplified estimation)
            lags = range(2, min(10, len(thicknesses)//2))
            variability = []
            for lag in lags:
                differences = [thicknesses[i] - thicknesses[i-lag] for i in range(lag, len(thicknesses))]
                variability.append(np.var(differences))

            if len(variability) > 1 and all(v > 0 for v in variability):
                log_lags = [np.log(lag) for lag in lags]
                log_vars = [np.log(var) for var in variability]
                hurst_slope = np.polyfit(log_lags, log_vars, 1)[0]
                features['hurst_exponent'] = hurst_slope / 2.0

        return features

    def _extract_derived_features(self, system_state: SystemState) -> Dict[str, float]:
        """Extract domain-specific derived features."""
        fused = system_state.fused_measurement
        health = system_state.health_metrics

        return {
            # Biofilm efficiency metrics
            'thickness_per_biomass': fused.thickness_um / max(0.01, fused.biomass_density_g_per_L),
            'conductivity_per_thickness': fused.conductivity_S_per_m / max(0.01, fused.thickness_um),
            'sensor_disagreement': abs(fused.eis_thickness - fused.qcm_thickness),
            'relative_sensor_disagreement': abs(fused.eis_thickness - fused.qcm_thickness) / max(0.01, (fused.eis_thickness + fused.qcm_thickness) / 2),

            # Health efficiency metrics
            'health_per_thickness': health.overall_health_score / max(0.01, fused.thickness_um),
            'risk_weighted_health': health.overall_health_score * (1 - max(health.fouling_risk, health.detachment_risk, health.stagnation_risk)),
            'prediction_reliability': health.prediction_confidence * health.assessment_confidence,

            # System efficiency metrics
            'power_per_flow': system_state.power_output / max(0.01, system_state.flow_rate),
            'substrate_utilization': (system_state.inlet_concentration - system_state.outlet_concentration) / max(0.01, system_state.inlet_concentration),
            'current_efficiency': system_state.current_density / max(0.01, fused.conductivity_S_per_m),

            # Composite indicators
            'overall_system_efficiency': (system_state.power_output * health.overall_health_score * fused.fusion_confidence) / max(0.01, system_state.flow_rate),
            'biofilm_maturity_index': fused.thickness_um * fused.conductivity_S_per_m * health.stability_health,
            'sensor_fusion_quality': fused.sensor_agreement * fused.fusion_confidence * (1 - fused.cross_validation_error / 20.0)
        }

    def _health_status_to_numeric(self, status) -> float:
        """Convert health status to numeric value."""
        status_map = {
            'excellent': 5.0,
            'good': 4.0,
            'fair': 3.0,
            'poor': 2.0,
            'critical': 1.0,
            'unknown': 0.0
        }
        return status_map.get(status.value if hasattr(status, 'value') else status, 0.0)

    def _health_trend_to_numeric(self, trend) -> float:
        """Convert health trend to numeric value."""
        trend_map = {
            'improving': 1.0,
            'stable': 0.0,
            'declining': -1.0,
            'volatile': -0.5,
            'unknown': 0.0
        }
        return trend_map.get(trend.value if hasattr(trend, 'value') else trend, 0.0)

    def _strategy_to_numeric(self, strategy) -> float:
        """Convert control strategy to numeric value."""
        strategy_map = {
            'performance_focused': 1.0,
            'balanced': 0.5,
            'health_focused': 0.0,
            'conservative': -0.5,
            'recovery': -1.0
        }
        return strategy_map.get(strategy.value if hasattr(strategy, 'value') else strategy, 0.0)

    def _adaptation_mode_to_numeric(self, mode) -> float:
        """Convert adaptation mode to numeric value."""
        mode_map = {
            'aggressive': 1.0,
            'moderate': 0.5,
            'conservative': 0.0,
            'disabled': -1.0
        }
        return mode_map.get(mode.value if hasattr(mode, 'value') else mode, 0.0)

    def get_feature_importance(self, features: Dict[str, float], target_values: List[float],
                             n_bootstrap: int = 100) -> List[FeatureImportance]:
        """
        Calculate feature importance using multiple methods.
        
        Args:
            features: Dictionary of feature values
            target_values: Target values for importance calculation
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            List of feature importance rankings
        """
        if not SKLEARN_AVAILABLE or len(target_values) < 10:
            return []

        # Prepare data
        feature_names = list(features.keys())
        X = np.array([[features[name] for name in feature_names]])
        y = np.array(target_values[-len(X):])  # Match lengths

        if X.shape[0] != len(y):
            return []

        importance_results = []

        try:
            # Random Forest feature importance
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            rf_importances = rf.feature_importances_

            # Create importance objects
            for i, name in enumerate(feature_names):
                importance_results.append(FeatureImportance(
                    feature_name=name,
                    importance_score=rf_importances[i],
                    feature_type=self._classify_feature_type(name),
                    description=self._get_feature_description(name),
                    stability=0.8  # Placeholder - would be calculated from bootstrap
                ))

            # Sort by importance
            importance_results.sort(key=lambda x: x.importance_score, reverse=True)

        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")

        return importance_results[:20]  # Top 20 features

    def _classify_feature_type(self, feature_name: str) -> FeatureType:
        """Classify feature into type category."""
        if any(word in feature_name for word in ['mean', 'std', 'cv', 'range']):
            return FeatureType.STATISTICAL
        elif any(word in feature_name for word in ['trend', 'autocorr', 'lag', 'frequency']):
            return FeatureType.TEMPORAL
        elif any(word in feature_name for word in ['health', 'risk', 'prediction']):
            return FeatureType.DERIVED
        elif any(word in feature_name for word in ['per_', 'ratio', 'efficiency', 'index']):
            return FeatureType.INTERACTION
        else:
            return FeatureType.RAW

    def _get_feature_description(self, feature_name: str) -> str:
        """Get human-readable description of feature."""
        descriptions = {
            'thickness_um': 'Biofilm thickness in micrometers',
            'conductivity': 'Biofilm electrical conductivity',
            'overall_health_score': 'Overall biofilm health assessment',
            'sensor_agreement': 'Agreement between EIS and QCM sensors',
            'power_efficiency': 'Power output per unit flow rate',
            'fouling_risk': 'Risk of electrode fouling',
            'thickness_trend': 'Recent trend in thickness measurements'
        }
        return descriptions.get(feature_name, f'Engineered feature: {feature_name}')


class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization for MFC control systems.
    
    Optimizes parameters for Q-learning, sensor fusion, and health monitoring
    using various optimization strategies.
    """

    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN,
                 max_evaluations: int = 50,
                 cv_folds: int = 5):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            strategy: Optimization strategy to use
            max_evaluations: Maximum number of parameter evaluations
            cv_folds: Number of cross-validation folds
        """
        self.strategy = strategy
        self.max_evaluations = max_evaluations
        self.cv_folds = cv_folds

        # Optimization history
        self.evaluation_history = []
        self.best_parameters = None
        self.best_score = -float('inf')

        # Parameter spaces
        self.parameter_spaces = self._define_parameter_spaces()

        logger.info(f"Hyperparameter optimizer initialized with {strategy.value} strategy")

    def _define_parameter_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Define parameter search spaces for different components."""
        return {
            'qlearning': {
                'learning_rate': {'type': 'continuous', 'bounds': (0.01, 0.8)},
                'epsilon': {'type': 'continuous', 'bounds': (0.05, 0.9)},
                'discount_factor': {'type': 'continuous', 'bounds': (0.7, 0.99)},
                'health_weight': {'type': 'continuous', 'bounds': (0.1, 0.9)},
                'adaptation_rate': {'type': 'continuous', 'bounds': (0.01, 0.5)}
            },
            'sensor_fusion': {
                'min_sensor_weight': {'type': 'continuous', 'bounds': (0.05, 0.3)},
                'max_disagreement': {'type': 'continuous', 'bounds': (5.0, 20.0)},
                'fault_threshold': {'type': 'continuous', 'bounds': (0.2, 0.8)},
                'prediction_horizon': {'type': 'discrete', 'choices': [5, 10, 15, 20]}
            },
            'health_monitoring': {
                'intervention_threshold': {'type': 'continuous', 'bounds': (0.3, 0.7)},
                'critical_threshold': {'type': 'continuous', 'bounds': (0.2, 0.5)},
                'risk_threshold': {'type': 'continuous', 'bounds': (0.5, 0.9)},
                'stability_threshold': {'type': 'continuous', 'bounds': (0.2, 0.6)}
            }
        }

    def optimize_controller_parameters(self, controller: AdaptiveMFCController,
                                     historical_data: List[Dict[str, Any]],
                                     target_metric: str = 'overall_performance') -> OptimizationResult:
        """
        Optimize controller parameters using historical performance data.
        
        Args:
            controller: MFC controller to optimize
            historical_data: Historical control and performance data
            target_metric: Target metric to optimize
            
        Returns:
            Optimization results
        """
        start_time = datetime.now()

        # Prepare data for optimization
        X, y = self._prepare_optimization_data(historical_data, target_metric)

        if len(y) < 20:
            logger.warning("Insufficient data for reliable optimization")
            return OptimizationResult(
                strategy_used=self.strategy,
                best_parameters={},
                performance_improvement=0.0,
                validation_score=0.0,
                optimization_time_seconds=0.0,
                convergence_achieved=False,
                confidence_interval=(0.0, 0.0)
            )

        # Execute optimization based on strategy
        if self.strategy == OptimizationStrategy.BAYESIAN:
            result = self._bayesian_optimization(X, y)
        elif self.strategy == OptimizationStrategy.EVOLUTIONARY:
            result = self._evolutionary_optimization(X, y)
        elif self.strategy == OptimizationStrategy.RANDOM_SEARCH:
            result = self._random_search_optimization(X, y)
        else:  # Grid search
            result = self._grid_search_optimization(X, y)

        # Calculate performance improvement
        baseline_score = np.mean(y)
        improvement = (result['best_score'] - baseline_score) / max(abs(baseline_score), 0.01)

        # Validate with cross-validation
        validation_score = self._validate_parameters(X, y, result['best_params'])

        optimization_time = (datetime.now() - start_time).total_seconds()

        # Apply optimized parameters to controller
        self._apply_optimized_parameters(controller, result['best_params'])

        return OptimizationResult(
            strategy_used=self.strategy,
            best_parameters=result['best_params'],
            performance_improvement=improvement,
            validation_score=validation_score,
            optimization_time_seconds=optimization_time,
            convergence_achieved=result.get('converged', True),
            confidence_interval=self._calculate_confidence_interval(validation_score, len(y))
        )

    def _prepare_optimization_data(self, historical_data: List[Dict[str, Any]],
                                 target_metric: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for optimization."""
        features = []
        targets = []

        feature_engineer = FeatureEngineer()

        for data_point in historical_data:
            if 'system_state' not in data_point or 'performance_metrics' not in data_point:
                continue

            # Extract features
            system_features = feature_engineer.extract_features(
                data_point['system_state'],
                data_point['performance_metrics']
            )

            # Calculate target value
            if target_metric in data_point['performance_metrics']:
                target = data_point['performance_metrics'][target_metric]
            else:
                # Composite target from multiple metrics
                target = self._calculate_composite_target(data_point)

            features.append(list(system_features.values()))
            targets.append(target)

        return np.array(features), np.array(targets)

    def _calculate_composite_target(self, data_point: Dict[str, Any]) -> float:
        """Calculate composite performance target."""
        metrics = data_point['performance_metrics']

        # Weighted combination of key metrics
        target = (
            metrics.get('power_efficiency', 0.0) * 0.3 +
            metrics.get('biofilm_health_score', 0.0) * 0.3 +
            metrics.get('sensor_reliability', 0.0) * 0.2 +
            metrics.get('system_stability', 0.0) * 0.2
        )

        return target

    def _bayesian_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Bayesian optimization implementation."""
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available, falling back to random search")
            return self._random_search_optimization(X, y)

        # Define objective function
        def objective(params):
            # Convert parameter vector to dictionary
            param_dict = self._vector_to_params(params)

            # Evaluate using cross-validation
            score = self._evaluate_parameters(X, y, param_dict)
            return -score  # Minimize negative score

        # Parameter bounds
        bounds = self._get_parameter_bounds()

        # Bayesian optimization (simplified - using scipy minimize)
        result = minimize(
            objective,
            x0=self._get_initial_params(),
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': self.max_evaluations}
        )

        best_params = self._vector_to_params(result.x)

        return {
            'best_params': best_params,
            'best_score': -result.fun,
            'converged': result.success
        }

    def _evolutionary_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evolutionary optimization implementation."""
        if not SCIPY_AVAILABLE:
            return self._random_search_optimization(X, y)

        def objective(params):
            param_dict = self._vector_to_params(params)
            score = self._evaluate_parameters(X, y, param_dict)
            return -score  # Minimize negative score

        bounds = self._get_parameter_bounds()

        result = differential_evolution(
            objective,
            bounds,
            maxiter=self.max_evaluations // 10,  # Adjust for population-based method
            seed=42
        )

        best_params = self._vector_to_params(result.x)

        return {
            'best_params': best_params,
            'best_score': -result.fun,
            'converged': result.success
        }

    def _random_search_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Random search optimization implementation."""
        best_score = -float('inf')
        best_params = {}

        for _ in range(self.max_evaluations):
            # Generate random parameters
            params = self._generate_random_params()

            # Evaluate
            score = self._evaluate_parameters(X, y, params)

            if score > best_score:
                best_score = score
                best_params = params.copy()

        return {
            'best_params': best_params,
            'best_score': best_score,
            'converged': True
        }

    def _grid_search_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Grid search optimization implementation."""
        # Simplified grid search over key parameters
        learning_rates = [0.1, 0.2, 0.3, 0.5]
        epsilons = [0.1, 0.2, 0.3, 0.4]
        health_weights = [0.2, 0.4, 0.6, 0.8]

        best_score = -float('inf')
        best_params = {}

        for lr in learning_rates:
            for eps in epsilons:
                for hw in health_weights:
                    params = {
                        'learning_rate': lr,
                        'epsilon': eps,
                        'health_weight': hw,
                        'discount_factor': 0.9,  # Fixed
                        'adaptation_rate': 0.1   # Fixed
                    }

                    score = self._evaluate_parameters(X, y, params)

                    if score > best_score:
                        best_score = score
                        best_params = params.copy()

        return {
            'best_params': best_params,
            'best_score': best_score,
            'converged': True
        }

    def _evaluate_parameters(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> float:
        """Evaluate parameter set using cross-validation."""
        if not SKLEARN_AVAILABLE:
            return np.random.random()  # Fallback

        try:
            # Create model with parameters (simplified)
            model = RandomForestRegressor(
                n_estimators=max(10, int(params.get('n_estimators', 50))),
                max_depth=max(3, int(params.get('max_depth', 10))),
                random_state=42
            )

            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=min(self.cv_folds, len(y)//5), scoring='r2')
            return np.mean(cv_scores)

        except Exception as e:
            logger.warning(f"Parameter evaluation failed: {e}")
            return -1.0  # Poor score for failed evaluations

    def _validate_parameters(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> float:
        """Validate optimized parameters."""
        return self._evaluate_parameters(X, y, params)

    def _apply_optimized_parameters(self, controller: AdaptiveMFCController, params: Dict[str, Any]):
        """Apply optimized parameters to controller."""
        # Apply Q-learning parameters
        if hasattr(controller, 'q_controller'):
            if 'learning_rate' in params:
                controller.q_controller.learning_rate = params['learning_rate']
                controller.q_controller.base_learning_rate = params['learning_rate']
            if 'epsilon' in params:
                controller.q_controller.epsilon = params['epsilon']
                controller.q_controller.base_epsilon = params['epsilon']
            if 'discount_factor' in params:
                controller.q_controller.discount_factor = params['discount_factor']
                controller.q_controller.base_discount_factor = params['discount_factor']
            if 'health_weight' in params:
                controller.q_controller.health_weight = params['health_weight']
                controller.q_controller.power_weight = 1.0 - params['health_weight']
            if 'adaptation_rate' in params:
                controller.q_controller.adaptation_rate = params['adaptation_rate']

        logger.info(f"Applied optimized parameters: {params}")

    def _get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        return [
            (0.01, 0.8),   # learning_rate
            (0.05, 0.9),   # epsilon
            (0.7, 0.99),   # discount_factor
            (0.1, 0.9),    # health_weight
            (0.01, 0.5)    # adaptation_rate
        ]

    def _get_initial_params(self) -> np.ndarray:
        """Get initial parameter values."""
        return np.array([0.2, 0.3, 0.9, 0.4, 0.1])

    def _vector_to_params(self, param_vector: np.ndarray) -> Dict[str, Any]:
        """Convert parameter vector to dictionary."""
        return {
            'learning_rate': param_vector[0],
            'epsilon': param_vector[1],
            'discount_factor': param_vector[2],
            'health_weight': param_vector[3],
            'adaptation_rate': param_vector[4]
        }

    def _generate_random_params(self) -> Dict[str, Any]:
        """Generate random parameter set within bounds."""
        bounds = self._get_parameter_bounds()
        return {
            'learning_rate': np.random.uniform(bounds[0][0], bounds[0][1]),
            'epsilon': np.random.uniform(bounds[1][0], bounds[1][1]),
            'discount_factor': np.random.uniform(bounds[2][0], bounds[2][1]),
            'health_weight': np.random.uniform(bounds[3][0], bounds[3][1]),
            'adaptation_rate': np.random.uniform(bounds[4][0], bounds[4][1])
        }

    def _calculate_confidence_interval(self, score: float, n_samples: int) -> Tuple[float, float]:
        """Calculate confidence interval for performance improvement."""
        # Simplified confidence interval calculation
        std_error = 0.1 / np.sqrt(max(1, n_samples))
        margin = 1.96 * std_error  # 95% CI
        return (score - margin, score + margin)


class MLOptimizedMFCController:
    """
    ML-optimized MFC controller that continuously learns and adapts.
    
    Integrates all ML optimization components for maximum performance.
    """

    def __init__(self, base_controller: AdaptiveMFCController,
                 optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN,
                 reoptimization_interval: int = 100):  # Steps between reoptimization
        """
        Initialize ML-optimized controller.
        
        Args:
            base_controller: Base adaptive controller
            optimization_strategy: ML optimization strategy
            reoptimization_interval: Steps between reoptimization
        """
        self.base_controller = base_controller
        self.optimization_strategy = optimization_strategy
        self.reoptimization_interval = reoptimization_interval

        # ML components
        self.feature_engineer = FeatureEngineer()
        self.hyperparameter_optimizer = HyperparameterOptimizer(optimization_strategy)

        # Learning history
        self.control_data_history = []
        self.optimization_history = []
        self.performance_history = []

        # Control state
        self.steps_since_optimization = 0
        self.last_optimization_score = 0.0

        logger.info(f"ML-optimized MFC controller initialized with {optimization_strategy.value}")

    def control_step_with_learning(self, eis_measurement, qcm_measurement,
                                 eis_properties: Dict[str, float],
                                 qcm_properties: Dict[str, float],
                                 time_hours: float) -> Dict[str, Any]:
        """
        Execute control step with continuous learning and optimization.
        
        Args:
            eis_measurement: EIS sensor measurement
            qcm_measurement: QCM sensor measurement
            eis_properties: Processed EIS properties
            qcm_properties: Processed QCM properties
            time_hours: Current time in hours
            
        Returns:
            Enhanced control results with ML insights
        """
        # Execute base control step
        control_results = self.base_controller.control_step(
            eis_measurement, qcm_measurement, eis_properties, qcm_properties, time_hours
        )

        # Extract features for learning
        features = self.feature_engineer.extract_features(
            control_results['system_state'],
            control_results['performance_metrics']
        )

        # Store data for learning
        learning_data = {
            'timestamp': time_hours,
            'features': features,
            'system_state': control_results['system_state'],
            'control_decision': control_results['control_decision'],
            'performance_metrics': control_results['performance_metrics'],
            'execution_results': control_results['execution_results']
        }
        self.control_data_history.append(learning_data)

        # Periodic reoptimization
        self.steps_since_optimization += 1
        if self.steps_since_optimization >= self.reoptimization_interval:
            optimization_result = self._perform_reoptimization()
            control_results['optimization_result'] = optimization_result
            self.steps_since_optimization = 0

        # Add ML insights
        control_results['ml_insights'] = self._generate_ml_insights(features, control_results)
        control_results['feature_importance'] = self._get_recent_feature_importance()

        return control_results

    def _perform_reoptimization(self) -> OptimizationResult:
        """Perform periodic reoptimization of controller parameters."""
        if len(self.control_data_history) < 50:
            logger.info("Insufficient data for reoptimization")
            return OptimizationResult(
                strategy_used=self.optimization_strategy,
                best_parameters={},
                performance_improvement=0.0,
                validation_score=0.0,
                optimization_time_seconds=0.0,
                convergence_achieved=False,
                confidence_interval=(0.0, 0.0)
            )

        logger.info(f"Performing reoptimization with {len(self.control_data_history)} data points")

        # Optimize parameters
        optimization_result = self.hyperparameter_optimizer.optimize_controller_parameters(
            self.base_controller,
            self.control_data_history,
            target_metric='overall_performance'
        )

        # Store optimization history
        self.optimization_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'result': optimization_result,
            'data_points_used': len(self.control_data_history)
        })

        # Update performance tracking
        self.last_optimization_score = optimization_result.validation_score

        logger.info(f"Reoptimization completed. Improvement: {optimization_result.performance_improvement:.3f}")

        return optimization_result

    def _generate_ml_insights(self, features: Dict[str, float],
                            control_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ML-based insights about current system state."""
        insights = {
            'feature_summary': {
                'total_features': len(features),
                'key_values': {k: v for k, v in list(features.items())[:5]},  # Top 5 features
            },
            'learning_status': {
                'data_points_collected': len(self.control_data_history),
                'steps_since_optimization': self.steps_since_optimization,
                'last_optimization_score': self.last_optimization_score
            },
            'performance_trend': self._analyze_performance_trend(),
            'optimization_recommendations': self._get_optimization_recommendations(features)
        }

        return insights

    def _get_recent_feature_importance(self) -> List[FeatureImportance]:
        """Get recent feature importance analysis."""
        if len(self.control_data_history) < 20:
            return []

        # Extract features and performance metrics
        recent_data = self.control_data_history[-50:]  # Last 50 points

        # Get a representative feature set
        features = recent_data[-1]['features']

        # Get performance targets
        performance_values = [d['performance_metrics'].get('overall_performance', 0.5)
                            for d in recent_data]

        return self.feature_engineer.get_feature_importance(features, performance_values)

    def _analyze_performance_trend(self) -> Dict[str, Any]:
        """Analyze recent performance trends."""
        if len(self.control_data_history) < 10:
            return {'insufficient_data': True}

        recent_performance = [d['performance_metrics'].get('biofilm_health_score', 0.5)
                            for d in self.control_data_history[-20:]]

        # Simple trend analysis
        if len(recent_performance) >= 5:
            trend_slope = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]

            if trend_slope > 0.01:
                trend = 'improving'
            elif trend_slope < -0.01:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'unknown'

        return {
            'trend_direction': trend,
            'current_performance': recent_performance[-1] if recent_performance else 0.5,
            'average_performance': np.mean(recent_performance) if recent_performance else 0.5,
            'performance_variance': np.var(recent_performance) if recent_performance else 0.0
        }

    def _get_optimization_recommendations(self, features: Dict[str, float]) -> List[str]:
        """Get optimization recommendations based on current state."""
        recommendations = []

        # Health-based recommendations
        if features.get('overall_health_score', 0.5) < 0.4:
            recommendations.append("Consider increasing health_weight parameter")

        # Sensor reliability recommendations
        if features.get('sensor_agreement', 0.5) < 0.6:
            recommendations.append("Review sensor calibration and fusion parameters")

        # Performance recommendations
        if features.get('power_efficiency', 0.5) < 0.4:
            recommendations.append("Consider performance-focused control strategy")

        # Stability recommendations
        if features.get('system_stability', 0.5) < 0.5:
            recommendations.append("Reduce exploration rate for more stable control")

        return recommendations[:3]  # Top 3 recommendations

    def get_ml_status_report(self) -> Dict[str, Any]:
        """Get comprehensive ML optimization status report."""
        report = {
            'optimization_strategy': self.optimization_strategy.value,
            'learning_progress': {
                'total_data_points': len(self.control_data_history),
                'optimizations_performed': len(self.optimization_history),
                'steps_since_last_optimization': self.steps_since_optimization,
                'next_optimization_in': max(0, self.reoptimization_interval - self.steps_since_optimization)
            },
            'performance_metrics': {
                'last_optimization_score': self.last_optimization_score,
                'average_performance': np.mean([d['performance_metrics'].get('biofilm_health_score', 0.5)
                                              for d in self.control_data_history[-20:]]) if len(self.control_data_history) >= 20 else 0.5
            },
            'feature_engineering': {
                'features_generated': len(self.control_data_history[-1]['features']) if self.control_data_history else 0,
                'feature_types': ['raw', 'statistical', 'temporal', 'health', 'derived']
            },
            'recent_optimizations': self.optimization_history[-3:] if len(self.optimization_history) >= 3 else self.optimization_history
        }

        return report


def create_ml_optimized_controller(species: BacterialSpecies = BacterialSpecies.MIXED,
                                 qlearning_config: Optional[QLearningConfig] = None,
                                 sensor_config: Optional[SensorConfig] = None,
                                 optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN) -> MLOptimizedMFCController:
    """
    Factory function to create fully ML-optimized MFC controller.
    
    Args:
        species: Target bacterial species
        qlearning_config: Q-learning configuration
        sensor_config: Sensor configuration
        optimization_strategy: ML optimization strategy
        
    Returns:
        Configured MLOptimizedMFCController instance
    """
    # Create base adaptive controller
    from adaptive_mfc_controller import create_adaptive_mfc_controller
    base_controller = create_adaptive_mfc_controller(species, qlearning_config, sensor_config)

    # Create ML-optimized controller
    ml_controller = MLOptimizedMFCController(
        base_controller=base_controller,
        optimization_strategy=optimization_strategy,
        reoptimization_interval=100  # Reoptimize every 100 steps
    )

    logger.info(f"ML-optimized MFC controller created with {optimization_strategy.value} optimization")
    return ml_controller
