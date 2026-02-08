"""Coverage tests for ml_optimization.py (98%+ target)."""
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock sensing_models and adaptive_mfc_controller before import
_mock_sensor_fusion = MagicMock()
_mock_sensor_fusion.BacterialSpecies = MagicMock()
_mock_sensor_fusion.BacterialSpecies.MIXED = "MIXED"
sys.modules.setdefault("sensing_models", MagicMock())
sys.modules.setdefault("sensing_models.sensor_fusion", _mock_sensor_fusion)
sys.modules.setdefault("adaptive_mfc_controller", MagicMock())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ml_optimization import (
    FeatureEngineer,
    FeatureImportance,
    FeatureType,
    HyperparameterOptimizer,
    MLOptimizedMFCController,
    ModelEnsemble,
    OptimizationResult,
    OptimizationStrategy,
    create_ml_optimized_controller,
)


def _make_system_state(**overrides):
    """Create a mock SystemState object."""
    fused = MagicMock()
    fused.thickness_um = overrides.get("thickness_um", 100.0)
    fused.thickness_uncertainty = 2.0
    fused.biomass_density_g_per_L = overrides.get("biomass_density", 5.0)
    fused.biomass_uncertainty = 0.5
    fused.conductivity_S_per_m = overrides.get("conductivity", 0.01)
    fused.conductivity_uncertainty = 0.001
    fused.eis_thickness = overrides.get("eis_thickness", 98.0)
    fused.qcm_thickness = overrides.get("qcm_thickness", 102.0)
    fused.sensor_agreement = overrides.get("sensor_agreement", 0.9)
    fused.fusion_confidence = overrides.get("fusion_confidence", 0.85)
    fused.cross_validation_error = overrides.get("cross_validation_error", 1.0)
    fused.eis_weight = 0.5
    fused.qcm_weight = 0.5
    fused.timestamp = overrides.get("timestamp", 0.0)

    health = MagicMock()
    health.overall_health_score = overrides.get("overall_health_score", 0.8)
    health.thickness_health = 0.9
    health.conductivity_health = 0.85
    health.growth_health = 0.7
    health.stability_health = 0.8
    health.predicted_health_24h = 0.75
    health.fouling_risk = overrides.get("fouling_risk", 0.1)
    health.detachment_risk = 0.05
    health.stagnation_risk = 0.08
    health.assessment_confidence = 0.9
    health.prediction_confidence = 0.8
    health.health_status = overrides.get("health_status", "good")
    health.health_trend = overrides.get("health_trend", "stable")

    state = MagicMock()
    state.fused_measurement = fused
    state.health_metrics = health
    state.flow_rate = overrides.get("flow_rate", 0.01)
    state.inlet_concentration = overrides.get("inlet_concentration", 20.0)
    state.outlet_concentration = overrides.get("outlet_concentration", 12.0)
    state.current_density = overrides.get("current_density", 1.0)
    state.power_output = overrides.get("power_output", 0.5)
    state.current_strategy = overrides.get("current_strategy", "balanced")
    state.adaptation_mode = overrides.get("adaptation_mode", "moderate")
    state.intervention_active = overrides.get("intervention_active", False)
    return state


def _make_performance_metrics(**overrides):
    return {
        "power_efficiency": overrides.get("power_efficiency", 0.6),
        "biofilm_health_score": overrides.get("biofilm_health_score", 0.7),
        "sensor_reliability": overrides.get("sensor_reliability", 0.8),
        "system_stability": overrides.get("system_stability", 0.75),
        "control_confidence": overrides.get("control_confidence", 0.65),
    }


# ------------------------------------------------------------------ #
# FeatureEngineer
# ------------------------------------------------------------------ #
class TestFeatureEngineer:
    def test_init_default(self):
        fe = FeatureEngineer()
        assert fe.window_size == 50
        assert fe.enable_advanced_features is True

    def test_init_custom(self):
        fe = FeatureEngineer(window_size=10, enable_advanced_features=False)
        assert fe.window_size == 10
        assert fe.enable_advanced_features is False

    def test_extract_features_minimal(self):
        fe = FeatureEngineer()
        state = _make_system_state()
        metrics = _make_performance_metrics()
        features = fe.extract_features(state, metrics)
        assert "thickness_um" in features
        assert "power_efficiency" in features
        assert len(fe.system_state_history) == 1

    def test_extract_raw_features(self):
        fe = FeatureEngineer()
        state = _make_system_state()
        raw = fe._extract_raw_features(state)
        assert raw["thickness_um"] == 100.0
        assert raw["sensor_agreement"] == 0.9

    def test_extract_statistical_features_empty(self):
        fe = FeatureEngineer()
        result = fe._extract_statistical_features()
        assert result == {}

    def test_extract_statistical_features_sufficient_history(self):
        fe = FeatureEngineer()
        for i in range(20):
            state = _make_system_state(
                thickness_um=100.0 + i * 0.5,
                conductivity=0.01 + i * 0.001,
                sensor_agreement=0.85 + i * 0.005,
            )
            fe.system_state_history.append(state)
        result = fe._extract_statistical_features()
        assert "thickness_mean" in result
        assert "conductivity_trend" in result
        assert "agreement_min" in result

    def test_extract_temporal_features_empty(self):
        fe = FeatureEngineer()
        result = fe._extract_temporal_features()
        assert result == {}

    def test_extract_temporal_features_sufficient(self):
        fe = FeatureEngineer()
        for i in range(50):
            state = _make_system_state(
                thickness_um=100.0 + np.sin(i * 0.2) * 5,
                timestamp=float(i),
            )
            fe.system_state_history.append(state)
        result = fe._extract_temporal_features()
        assert "time_since_major_change" in result

    def test_extract_temporal_features_with_autocorrelation(self):
        fe = FeatureEngineer()
        for i in range(50):
            state = _make_system_state(
                thickness_um=100.0 + i * 0.1,
                timestamp=float(i),
            )
            fe.system_state_history.append(state)
        result = fe._extract_temporal_features()
        assert "thickness_autocorr_lag1" in result
        assert "thickness_autocorr_lag5" in result

    def test_extract_temporal_features_with_fft(self):
        fe = FeatureEngineer(enable_advanced_features=True)
        for i in range(50):
            state = _make_system_state(
                thickness_um=100.0 + np.sin(i * 0.5) * 10,
                timestamp=float(i),
            )
            fe.system_state_history.append(state)
        result = fe._extract_temporal_features()
        assert "dominant_frequency" in result
        assert "spectral_centroid" in result

    def test_extract_temporal_features_no_major_change(self):
        """All thicknesses identical = no major change."""
        fe = FeatureEngineer()
        for i in range(50):
            state = _make_system_state(thickness_um=100.0, timestamp=float(i))
            fe.system_state_history.append(state)
        result = fe._extract_temporal_features()
        assert result["time_since_major_change"] == len(
            list(fe.system_state_history)[-fe.window_size:]
        )

    def test_extract_health_features(self):
        fe = FeatureEngineer()
        state = _make_system_state()
        result = fe._extract_health_features(state)
        assert "overall_health_score" in result
        assert "fouling_risk" in result

    def test_extract_control_features(self):
        fe = FeatureEngineer()
        state = _make_system_state()
        result = fe._extract_control_features(state)
        assert "flow_rate" in result
        assert "power_density" in result

    def test_extract_performance_features(self):
        fe = FeatureEngineer()
        result = fe._extract_performance_features(_make_performance_metrics())
        assert result["power_efficiency"] == 0.6

    def test_extract_performance_features_empty(self):
        fe = FeatureEngineer()
        result = fe._extract_performance_features({})
        assert result["power_efficiency"] == 0.0
        assert result["control_confidence"] == 0.5

    def test_extract_advanced_features_empty(self):
        fe = FeatureEngineer()
        result = fe._extract_advanced_features()
        assert result == {}

    def test_extract_advanced_features_sufficient(self):
        fe = FeatureEngineer()
        for i in range(25):
            state = _make_system_state(thickness_um=100.0 + i * 0.5)
            fe.system_state_history.append(state)
        result = fe._extract_advanced_features()
        assert "short_vs_long_term_ratio" in result

    def test_extract_advanced_features_hurst(self):
        fe = FeatureEngineer()
        np.random.seed(42)
        for i in range(25):
            state = _make_system_state(thickness_um=100.0 + np.random.randn() * 5)
            fe.system_state_history.append(state)
        result = fe._extract_advanced_features()
        assert "hurst_exponent" in result

    def test_extract_advanced_features_disabled(self):
        fe = FeatureEngineer(enable_advanced_features=False)
        state = _make_system_state()
        metrics = _make_performance_metrics()
        features = fe.extract_features(state, metrics)
        # Advanced features should not be extracted
        assert "short_vs_long_term_ratio" not in features

    def test_extract_derived_features(self):
        fe = FeatureEngineer()
        state = _make_system_state()
        result = fe._extract_derived_features(state)
        assert "thickness_per_biomass" in result
        assert "sensor_fusion_quality" in result

    def test_health_status_to_numeric_enum(self):
        fe = FeatureEngineer()
        mock_status = MagicMock()
        mock_status.value = "excellent"
        assert fe._health_status_to_numeric(mock_status) == 5.0

    def test_health_status_to_numeric_string(self):
        fe = FeatureEngineer()
        assert fe._health_status_to_numeric("poor") == 2.0

    def test_health_status_to_numeric_unknown(self):
        fe = FeatureEngineer()
        assert fe._health_status_to_numeric("unknown_status") == 0.0

    def test_health_trend_to_numeric(self):
        fe = FeatureEngineer()
        assert fe._health_trend_to_numeric("improving") == 1.0
        assert fe._health_trend_to_numeric("declining") == -1.0

    def test_health_trend_to_numeric_enum(self):
        fe = FeatureEngineer()
        mock_trend = MagicMock()
        mock_trend.value = "volatile"
        assert fe._health_trend_to_numeric(mock_trend) == -0.5

    def test_strategy_to_numeric(self):
        fe = FeatureEngineer()
        assert fe._strategy_to_numeric("performance_focused") == 1.0
        assert fe._strategy_to_numeric("recovery") == -1.0

    def test_strategy_to_numeric_unknown(self):
        fe = FeatureEngineer()
        assert fe._strategy_to_numeric("unknown") == 0.0

    def test_adaptation_mode_to_numeric(self):
        fe = FeatureEngineer()
        assert fe._adaptation_mode_to_numeric("aggressive") == 1.0
        assert fe._adaptation_mode_to_numeric("disabled") == -1.0

    def test_adaptation_mode_to_numeric_enum(self):
        fe = FeatureEngineer()
        mock_mode = MagicMock()
        mock_mode.value = "conservative"
        assert fe._adaptation_mode_to_numeric(mock_mode) == 0.0

    def test_get_feature_importance_insufficient_data(self):
        fe = FeatureEngineer()
        result = fe.get_feature_importance({"a": 1.0}, [1.0, 2.0])
        assert result == []

    def test_get_feature_importance_sklearn_single_sample(self):
        """With single sample X and target trimmed to match, sklearn fits."""
        fe = FeatureEngineer()
        features = {"a": 1.0, "b": 2.0}
        targets = list(range(20))
        result = fe.get_feature_importance(features, targets)
        # X has 1 row, y[-1:] has 1 element, so sklearn fits with 1 sample
        assert isinstance(result, list)

    def test_get_feature_importance_sklearn_no_available(self):
        fe = FeatureEngineer()
        features = {"a": 1.0}
        targets = list(range(15))
        with patch("ml_optimization.SKLEARN_AVAILABLE", False):
            result = fe.get_feature_importance(features, targets)
        assert result == []

    def test_classify_feature_type(self):
        fe = FeatureEngineer()
        assert fe._classify_feature_type("thickness_mean") == FeatureType.STATISTICAL
        assert fe._classify_feature_type("thickness_trend") == FeatureType.TEMPORAL
        assert fe._classify_feature_type("fouling_risk") == FeatureType.DERIVED
        assert fe._classify_feature_type("power_per_flow") == FeatureType.INTERACTION
        assert fe._classify_feature_type("thickness_um") == FeatureType.RAW

    def test_get_feature_description_known(self):
        fe = FeatureEngineer()
        assert "thickness" in fe._get_feature_description("thickness_um").lower()

    def test_get_feature_description_unknown(self):
        fe = FeatureEngineer()
        desc = fe._get_feature_description("unknown_feature")
        assert "Engineered feature" in desc


# ------------------------------------------------------------------ #
# HyperparameterOptimizer
# ------------------------------------------------------------------ #
class TestHyperparameterOptimizer:
    def test_init_default(self):
        opt = HyperparameterOptimizer()
        assert opt.strategy == OptimizationStrategy.BAYESIAN
        assert opt.max_evaluations == 50
        assert opt.cv_folds == 5

    def test_init_custom(self):
        opt = HyperparameterOptimizer(
            strategy=OptimizationStrategy.GRID_SEARCH,
            max_evaluations=10,
            cv_folds=3,
        )
        assert opt.strategy == OptimizationStrategy.GRID_SEARCH

    def test_define_parameter_spaces(self):
        opt = HyperparameterOptimizer()
        spaces = opt._define_parameter_spaces()
        assert "qlearning" in spaces
        assert "sensor_fusion" in spaces
        assert "health_monitoring" in spaces

    def test_optimize_insufficient_data(self):
        opt = HyperparameterOptimizer()
        controller = MagicMock()
        result = opt.optimize_controller_parameters(controller, [])
        assert isinstance(result, OptimizationResult)
        assert result.convergence_achieved is False

    def test_calculate_composite_target(self):
        opt = HyperparameterOptimizer()
        data_point = {
            "performance_metrics": {
                "power_efficiency": 0.8,
                "biofilm_health_score": 0.7,
                "sensor_reliability": 0.6,
                "system_stability": 0.5,
            }
        }
        target = opt._calculate_composite_target(data_point)
        expected = 0.8 * 0.3 + 0.7 * 0.3 + 0.6 * 0.2 + 0.5 * 0.2
        assert abs(target - expected) < 1e-6

    def test_get_parameter_bounds(self):
        opt = HyperparameterOptimizer()
        bounds = opt._get_parameter_bounds()
        assert len(bounds) == 5
        assert bounds[0] == (0.01, 0.8)

    def test_get_initial_params(self):
        opt = HyperparameterOptimizer()
        params = opt._get_initial_params()
        assert len(params) == 5

    def test_vector_to_params(self):
        opt = HyperparameterOptimizer()
        vec = np.array([0.2, 0.3, 0.9, 0.4, 0.1])
        params = opt._vector_to_params(vec)
        assert params["learning_rate"] == 0.2
        assert params["health_weight"] == 0.4

    def test_generate_random_params(self):
        opt = HyperparameterOptimizer()
        np.random.seed(42)
        params = opt._generate_random_params()
        assert 0.01 <= params["learning_rate"] <= 0.8
        assert 0.7 <= params["discount_factor"] <= 0.99

    def test_calculate_confidence_interval(self):
        opt = HyperparameterOptimizer()
        lo, hi = opt._calculate_confidence_interval(0.5, 100)
        assert lo < 0.5 < hi

    def test_evaluate_parameters_sklearn(self):
        opt = HyperparameterOptimizer()
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.array([1, 2, 3]) + np.random.randn(50) * 0.1
        params = {"n_estimators": 20, "max_depth": 5}
        score = opt._evaluate_parameters(X, y, params)
        assert isinstance(score, float)

    def test_validate_parameters(self):
        opt = HyperparameterOptimizer()
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.array([1, 2, 3]) + np.random.randn(50) * 0.1
        score = opt._validate_parameters(X, y, {"n_estimators": 20})
        assert isinstance(score, float)

    def test_apply_optimized_parameters(self):
        opt = HyperparameterOptimizer()
        controller = MagicMock()
        controller.q_controller = MagicMock()
        params = {
            "learning_rate": 0.2,
            "epsilon": 0.3,
            "discount_factor": 0.9,
            "health_weight": 0.6,
            "adaptation_rate": 0.15,
        }
        opt._apply_optimized_parameters(controller, params)
        assert controller.q_controller.learning_rate == 0.2
        assert controller.q_controller.epsilon == 0.3
        assert controller.q_controller.health_weight == 0.6
        assert controller.q_controller.power_weight == 0.4

    def test_apply_optimized_parameters_no_q_controller(self):
        opt = HyperparameterOptimizer()
        controller = MagicMock(spec=[])
        params = {"learning_rate": 0.2}
        opt._apply_optimized_parameters(controller, params)

    def test_random_search_optimization(self):
        opt = HyperparameterOptimizer(
            strategy=OptimizationStrategy.RANDOM_SEARCH,
            max_evaluations=3,
        )
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        result = opt._random_search_optimization(X, y)
        assert "best_params" in result
        assert "best_score" in result

    def test_grid_search_optimization(self):
        opt = HyperparameterOptimizer(
            strategy=OptimizationStrategy.GRID_SEARCH,
            max_evaluations=5,
        )
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        result = opt._grid_search_optimization(X, y)
        assert "best_params" in result

    def test_bayesian_optimization(self):
        opt = HyperparameterOptimizer(
            strategy=OptimizationStrategy.BAYESIAN,
            max_evaluations=3,
        )
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        result = opt._bayesian_optimization(X, y)
        assert "best_params" in result

    def test_evolutionary_optimization(self):
        opt = HyperparameterOptimizer(
            strategy=OptimizationStrategy.EVOLUTIONARY,
            max_evaluations=10,
        )
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        result = opt._evolutionary_optimization(X, y)
        assert "best_params" in result

    def test_prepare_optimization_data_with_target_metric(self):
        opt = HyperparameterOptimizer()
        state = _make_system_state()
        data = [
            {
                "system_state": state,
                "performance_metrics": {
                    "overall_performance": 0.8,
                    "power_efficiency": 0.6,
                },
            }
        ]
        X, y = opt._prepare_optimization_data(data, "overall_performance")
        assert len(y) == 1
        assert y[0] == 0.8

    def test_prepare_optimization_data_composite_target(self):
        opt = HyperparameterOptimizer()
        state = _make_system_state()
        data = [
            {
                "system_state": state,
                "performance_metrics": {
                    "power_efficiency": 0.6,
                    "biofilm_health_score": 0.7,
                },
            }
        ]
        X, y = opt._prepare_optimization_data(data, "nonexistent_metric")
        assert len(y) == 1

    def test_prepare_optimization_data_skip_incomplete(self):
        opt = HyperparameterOptimizer()
        data = [{"some_other_key": True}]
        X, y = opt._prepare_optimization_data(data, "overall_performance")
        assert len(y) == 0

    def _make_consistent_data(self, n=25):
        """Make historical data with consistent feature dimension."""
        controller = MagicMock()
        controller.q_controller = MagicMock()
        # Pre-built feature dicts with fixed keys to avoid inhomogeneous arrays
        fixed_features = {"f1": 0.5, "f2": 0.3, "f3": 0.8}
        data = []
        for i in range(n):
            state = _make_system_state(thickness_um=100.0 + i)
            data.append({
                "system_state": state,
                "performance_metrics": {
                    "overall_performance": 0.5 + i * 0.01,
                    "power_efficiency": 0.6,
                    "biofilm_health_score": 0.7,
                },
            })
        return controller, data

    def test_optimize_with_bayesian_strategy(self):
        opt = HyperparameterOptimizer(
            strategy=OptimizationStrategy.BAYESIAN,
            max_evaluations=2,
        )
        controller, data = self._make_consistent_data()
        # Patch _prepare_optimization_data to return consistent arrays
        np.random.seed(42)
        X = np.random.randn(25, 5)
        y = np.random.randn(25)
        with patch.object(opt, "_prepare_optimization_data", return_value=(X, y)):
            result = opt.optimize_controller_parameters(controller, data)
        assert isinstance(result, OptimizationResult)

    def test_optimize_with_random_search_strategy(self):
        opt = HyperparameterOptimizer(
            strategy=OptimizationStrategy.RANDOM_SEARCH,
            max_evaluations=2,
        )
        controller, data = self._make_consistent_data()
        np.random.seed(42)
        X = np.random.randn(25, 5)
        y = np.random.randn(25)
        with patch.object(opt, "_prepare_optimization_data", return_value=(X, y)):
            result = opt.optimize_controller_parameters(controller, data)
        assert isinstance(result, OptimizationResult)

    def test_optimize_with_grid_search_strategy(self):
        opt = HyperparameterOptimizer(
            strategy=OptimizationStrategy.GRID_SEARCH,
            max_evaluations=2,
        )
        controller, data = self._make_consistent_data()
        np.random.seed(42)
        X = np.random.randn(25, 5)
        y = np.random.randn(25)
        with patch.object(opt, "_prepare_optimization_data", return_value=(X, y)):
            result = opt.optimize_controller_parameters(controller, data)
        assert isinstance(result, OptimizationResult)

    def test_optimize_with_evolutionary_strategy(self):
        opt = HyperparameterOptimizer(
            strategy=OptimizationStrategy.EVOLUTIONARY,
            max_evaluations=10,
        )
        controller, data = self._make_consistent_data()
        np.random.seed(42)
        X = np.random.randn(25, 5)
        y = np.random.randn(25)
        with patch.object(opt, "_prepare_optimization_data", return_value=(X, y)):
            result = opt.optimize_controller_parameters(controller, data)
        assert isinstance(result, OptimizationResult)

    def test_bayesian_fallback_no_scipy(self):
        opt = HyperparameterOptimizer(max_evaluations=2)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        with patch("ml_optimization.SCIPY_AVAILABLE", False):
            result = opt._bayesian_optimization(X, y)
        assert "best_params" in result

    def test_evolutionary_fallback_no_scipy(self):
        opt = HyperparameterOptimizer(max_evaluations=2)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        with patch("ml_optimization.SCIPY_AVAILABLE", False):
            result = opt._evolutionary_optimization(X, y)
        assert "best_params" in result

    def test_evaluate_parameters_no_sklearn(self):
        opt = HyperparameterOptimizer()
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        with patch("ml_optimization.SKLEARN_AVAILABLE", False):
            score = opt._evaluate_parameters(X, y, {})
        assert isinstance(score, float)

    def test_evaluate_parameters_exception(self):
        opt = HyperparameterOptimizer()
        # Pass data that causes an error (e.g., 1 sample only)
        X = np.random.randn(1, 3)
        y = np.random.randn(1)
        score = opt._evaluate_parameters(X, y, {})
        assert score == -1.0


# ------------------------------------------------------------------ #
# MLOptimizedMFCController
# ------------------------------------------------------------------ #
class TestMLOptimizedMFCController:
    def _make_controller(self):
        base = MagicMock()
        base.q_controller = MagicMock()
        return MLOptimizedMFCController(
            base_controller=base,
            optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
            reoptimization_interval=5,
        )

    def test_init(self):
        ctrl = self._make_controller()
        assert ctrl.reoptimization_interval == 5
        assert ctrl.steps_since_optimization == 0

    def test_control_step_with_learning_basic(self):
        ctrl = self._make_controller()
        ctrl.base_controller.control_step.return_value = {
            "system_state": _make_system_state(),
            "performance_metrics": _make_performance_metrics(),
            "control_decision": {"action": "none"},
            "execution_results": {"success": True},
        }
        result = ctrl.control_step_with_learning(
            MagicMock(), MagicMock(), {}, {}, 1.0,
        )
        assert "ml_insights" in result
        assert "feature_importance" in result

    def test_control_step_triggers_reoptimization(self):
        ctrl = self._make_controller()
        ctrl.steps_since_optimization = 4  # Will trigger at 5
        ctrl.base_controller.control_step.return_value = {
            "system_state": _make_system_state(),
            "performance_metrics": _make_performance_metrics(),
            "control_decision": {"action": "none"},
            "execution_results": {"success": True},
        }
        result = ctrl.control_step_with_learning(
            MagicMock(), MagicMock(), {}, {}, 1.0,
        )
        assert "optimization_result" in result
        assert ctrl.steps_since_optimization == 0

    def test_perform_reoptimization_insufficient_data(self):
        ctrl = self._make_controller()
        result = ctrl._perform_reoptimization()
        assert isinstance(result, OptimizationResult)
        assert result.convergence_achieved is False

    def test_perform_reoptimization_sufficient_data(self):
        ctrl = self._make_controller()
        for i in range(60):
            ctrl.control_data_history.append({
                "timestamp": float(i),
                "features": {"a": 1.0},
                "system_state": _make_system_state(),
                "control_decision": {},
                "performance_metrics": {"overall_performance": 0.5},
                "execution_results": {},
            })
        # Patch _prepare_optimization_data to avoid inhomogeneous array issue
        np.random.seed(42)
        X = np.random.randn(60, 5)
        y = np.random.randn(60)
        with patch.object(
            ctrl.hyperparameter_optimizer,
            "_prepare_optimization_data",
            return_value=(X, y),
        ):
            result = ctrl._perform_reoptimization()
        assert isinstance(result, OptimizationResult)
        assert len(ctrl.optimization_history) > 0

    def test_generate_ml_insights(self):
        ctrl = self._make_controller()
        features = {"a": 1.0, "b": 2.0}
        control_results = {}
        insights = ctrl._generate_ml_insights(features, control_results)
        assert "feature_summary" in insights
        assert "learning_status" in insights

    def test_get_recent_feature_importance_insufficient(self):
        ctrl = self._make_controller()
        result = ctrl._get_recent_feature_importance()
        assert result == []

    def test_get_recent_feature_importance_sufficient(self):
        ctrl = self._make_controller()
        for i in range(25):
            ctrl.control_data_history.append({
                "features": {"a": float(i)},
                "performance_metrics": {"overall_performance": 0.5 + i * 0.01},
            })
        result = ctrl._get_recent_feature_importance()
        assert isinstance(result, list)

    def test_analyze_performance_trend_insufficient(self):
        ctrl = self._make_controller()
        result = ctrl._analyze_performance_trend()
        assert result == {"insufficient_data": True}

    def test_analyze_performance_trend_improving(self):
        ctrl = self._make_controller()
        for i in range(20):
            ctrl.control_data_history.append({
                "performance_metrics": {"biofilm_health_score": 0.3 + i * 0.03},
            })
        result = ctrl._analyze_performance_trend()
        assert result["trend_direction"] == "improving"

    def test_analyze_performance_trend_declining(self):
        ctrl = self._make_controller()
        for i in range(20):
            ctrl.control_data_history.append({
                "performance_metrics": {"biofilm_health_score": 0.9 - i * 0.03},
            })
        result = ctrl._analyze_performance_trend()
        assert result["trend_direction"] == "declining"

    def test_analyze_performance_trend_stable(self):
        ctrl = self._make_controller()
        for i in range(20):
            ctrl.control_data_history.append({
                "performance_metrics": {"biofilm_health_score": 0.5},
            })
        result = ctrl._analyze_performance_trend()
        assert result["trend_direction"] == "stable"

    def test_get_optimization_recommendations_all(self):
        ctrl = self._make_controller()
        features = {
            "overall_health_score": 0.3,
            "sensor_agreement": 0.4,
            "power_efficiency": 0.3,
            "system_stability": 0.3,
        }
        recs = ctrl._get_optimization_recommendations(features)
        assert len(recs) == 3  # Max 3 recommendations

    def test_get_optimization_recommendations_none(self):
        ctrl = self._make_controller()
        features = {
            "overall_health_score": 0.9,
            "sensor_agreement": 0.9,
            "power_efficiency": 0.9,
            "system_stability": 0.9,
        }
        recs = ctrl._get_optimization_recommendations(features)
        assert len(recs) == 0

    def test_get_ml_status_report_empty(self):
        ctrl = self._make_controller()
        report = ctrl.get_ml_status_report()
        assert report["optimization_strategy"] == "random_search"
        assert report["learning_progress"]["total_data_points"] == 0

    def test_get_ml_status_report_with_data(self):
        ctrl = self._make_controller()
        for i in range(25):
            ctrl.control_data_history.append({
                "features": {"a": 1.0},
                "performance_metrics": {"biofilm_health_score": 0.5},
            })
        ctrl.optimization_history.append({"result": "test"})
        report = ctrl.get_ml_status_report()
        assert report["learning_progress"]["total_data_points"] == 25
        assert report["feature_engineering"]["features_generated"] == 1


# ------------------------------------------------------------------ #
# Factory function
# ------------------------------------------------------------------ #
class TestCreateMLOptimizedController:
    def test_create_factory(self):
        # The import is inside the function so we need to patch in adaptive_mfc_controller module
        mock_module = MagicMock()
        mock_module.create_adaptive_mfc_controller.return_value = MagicMock()
        with patch.dict("sys.modules", {"adaptive_mfc_controller": mock_module}):
            ctrl = create_ml_optimized_controller(
                species=_mock_sensor_fusion.BacterialSpecies.MIXED,
                optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
            )
            assert isinstance(ctrl, MLOptimizedMFCController)
            mock_module.create_adaptive_mfc_controller.assert_called_once()


# ------------------------------------------------------------------ #
# Dataclasses and Enums
# ------------------------------------------------------------------ #
class TestDataclassesAndEnums:
    def test_optimization_strategy_values(self):
        assert OptimizationStrategy.BAYESIAN.value == "bayesian"
        assert OptimizationStrategy.ENSEMBLE.value == "ensemble"

    def test_feature_type_values(self):
        assert FeatureType.RAW.value == "raw"
        assert FeatureType.SPECTRAL.value == "spectral"

    def test_optimization_result_creation(self):
        result = OptimizationResult(
            strategy_used=OptimizationStrategy.BAYESIAN,
            best_parameters={"lr": 0.1},
            performance_improvement=0.5,
            validation_score=0.8,
            optimization_time_seconds=1.0,
            convergence_achieved=True,
            confidence_interval=(0.7, 0.9),
        )
        assert result.performance_improvement == 0.5

    def test_feature_importance_creation(self):
        fi = FeatureImportance(
            feature_name="test",
            importance_score=0.5,
            feature_type=FeatureType.RAW,
            description="Test feature",
            stability=0.8,
        )
        assert fi.feature_name == "test"

    def test_model_ensemble_creation(self):
        me = ModelEnsemble(
            models=[MagicMock()],
            weights=[1.0],
            performance_scores=[0.8],
            ensemble_score=0.85,
            diversity_score=0.3,
        )
        assert me.ensemble_score == 0.85
