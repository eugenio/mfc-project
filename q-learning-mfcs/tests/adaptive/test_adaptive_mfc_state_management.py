"""Tests for Adaptive MFC Controller State Management.

US-012: Test Adaptive Controller State Management
Target: 90%+ coverage for state management

Tests cover:
- State discretization with configurable bins
- State bounds validation
- State transition tracking
- State history buffer
- Edge cases: out-of-bounds, NaN values
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class MockEnumValue:
    """Mock enum value with .value attribute."""

    def __init__(self, value: str):
        self.value = value

    def __eq__(self, other):
        if isinstance(other, MockEnumValue):
            return self.value == other.value
        return self.value == other

    def __hash__(self):
        return hash(self.value)


class MockHealthStatus:
    """Mock HealthStatus enum."""

    EXCELLENT = MockEnumValue("excellent")
    GOOD = MockEnumValue("good")
    FAIR = MockEnumValue("fair")
    POOR = MockEnumValue("poor")
    CRITICAL = MockEnumValue("critical")
    UNKNOWN = MockEnumValue("unknown")


class MockHealthTrend:
    """Mock HealthTrend enum."""

    IMPROVING = MockEnumValue("improving")
    STABLE = MockEnumValue("stable")
    DECLINING = MockEnumValue("declining")
    VOLATILE = MockEnumValue("volatile")
    UNKNOWN = MockEnumValue("unknown")


class MockBacterialSpecies:
    """Mock BacterialSpecies enum."""

    MIXED = MockEnumValue("mixed")
    GEOBACTER = MockEnumValue("geobacter")
    SHEWANELLA = MockEnumValue("shewanella")
    PSEUDOMONAS = MockEnumValue("pseudomonas")


@dataclass
class MockHealthMetrics:
    """Mock HealthMetrics dataclass."""

    overall_health_score: float = 0.8
    thickness_health: float = 0.85
    conductivity_health: float = 0.9
    growth_health: float = 0.75
    stability_health: float = 0.8
    health_status: Any = field(default_factory=lambda: MockHealthStatus.GOOD)
    health_trend: Any = field(default_factory=lambda: MockHealthTrend.STABLE)
    predicted_health_24h: float = 0.78
    predicted_intervention_time: float | None = None
    thickness_contribution: float = 0.25
    conductivity_contribution: float = 0.35
    growth_contribution: float = 0.2
    stability_contribution: float = 0.2
    fouling_risk: float = 0.2
    detachment_risk: float = 0.15
    stagnation_risk: float = 0.1
    assessment_confidence: float = 0.9
    prediction_confidence: float = 0.85


@dataclass
class MockFusedMeasurement:
    """Mock FusedMeasurement dataclass."""

    timestamp: float = 0.0
    thickness_um: float = 15.0
    eis_thickness: float = 14.5
    qcm_thickness: float = 15.5
    biomass_density_g_per_L: float = 2.5
    conductivity_S_per_m: float = 0.01
    active_fraction: float = 0.9
    eis_status: str = "healthy"
    qcm_status: str = "healthy"
    fusion_method: str = "weighted_average"
    fusion_confidence: float = 0.9
    sensor_agreement: float = 0.85
    cross_validation_error: float = 0.02


@dataclass
class MockPredictiveState:
    """Mock PredictiveState dataclass."""

    predicted_values: np.ndarray = field(
        default_factory=lambda: np.array([15.0, 2.5, 0.01]),
    )
    upper_confidence: np.ndarray = field(
        default_factory=lambda: np.array([16.0, 2.7, 0.012]),
    )
    lower_confidence: np.ndarray = field(
        default_factory=lambda: np.array([14.0, 2.3, 0.008]),
    )
    prediction_horizon_hours: float = 1.0
    prediction_accuracy: float = 0.85


@dataclass
class MockAnomalyDetection:
    """Mock AnomalyDetection dataclass."""

    timestamp: float = 0.0
    anomaly_score: float = 0.1
    anomaly_type: str = "sensor_drift"
    affected_sensors: list = field(default_factory=list)
    severity: str = "low"
    confidence: float = 0.8
    recommended_action: str = "Monitor"


@dataclass
class MockQLearningConfig:
    """Mock QLearningConfig dataclass."""

    enhanced_learning_rate: float = 0.1
    enhanced_discount_factor: float = 0.95
    enhanced_epsilon: float = 0.3
    sensor_weight: float = 0.5
    power_objective_weight: float = 0.3
    biofilm_health_weight: float = 0.4
    sensor_agreement_weight: float = 0.15
    stability_weight: float = 0.15
    sensor_confidence_threshold: float = 0.6
    exploration_boost_factor: float = 1.5
    state_space: MagicMock = field(default_factory=MagicMock)


@dataclass
class MockSensorConfig:
    """Mock SensorConfig dataclass."""

    eis_enabled: bool = True
    qcm_enabled: bool = True


class MockAdvancedQLearningFlowController:
    """Mock base controller class."""

    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.3):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        self.flow_actions = [-10, -5, 0, 5, 10]
        self.substrate_actions = [-5, 0, 5]

    def setup_enhanced_state_action_spaces(self):
        pass

    def get_state_hash(self, *args):
        return hash(args)


class MockSensingEnhancedQLearningController(MockAdvancedQLearningFlowController):
    """Mock sensing enhanced controller."""

    def __init__(
        self,
        qlearning_config=None,
        sensor_config=None,
        enable_sensor_state=True,
        fault_tolerance=True,
    ):
        lr = qlearning_config.enhanced_learning_rate if qlearning_config else 0.1
        df = qlearning_config.enhanced_discount_factor if qlearning_config else 0.95
        eps = qlearning_config.enhanced_epsilon if qlearning_config else 0.3
        super().__init__(lr, df, eps)
        self.qlearning_config = qlearning_config or MockQLearningConfig()
        self.sensor_config = sensor_config or MockSensorConfig()
        self.enable_sensor_state = enable_sensor_state
        self.fault_tolerance = fault_tolerance
        self.sensor_weight = qlearning_config.sensor_weight if qlearning_config else 0.5
        self.power_weight = qlearning_config.power_objective_weight if qlearning_config else 0.3
        self.biofilm_health_weight = qlearning_config.biofilm_health_weight if qlearning_config else 0.4
        self.sensor_agreement_weight = qlearning_config.sensor_agreement_weight if qlearning_config else 0.15
        self.stability_weight = qlearning_config.stability_weight if qlearning_config else 0.15
        self.min_sensor_confidence = qlearning_config.sensor_confidence_threshold if qlearning_config else 0.6
        self.exploration_boost_factor = qlearning_config.exploration_boost_factor if qlearning_config else 1.5
        self.sensor_confidence_history = []
        self.sensor_fault_count = 0
        self.sensor_degradation_factor = 1.0
        self.sensor_guided_decisions = 0
        self.model_guided_decisions = 0
        self.total_reward_components = {"power": 0.0, "biofilm_health": 0.0, "sensor_agreement": 0.0, "stability": 0.0}
        self.state_predictions = []
        self.sensor_validations = []
        self.prediction_errors = []
        self.setup_sensor_enhanced_state_space()

    def setup_sensor_enhanced_state_space(self):
        self.eis_thickness_bins = np.linspace(0, 50, 10)
        self.qcm_thickness_bins = np.linspace(0, 50, 10)
        self.conductivity_bins = np.linspace(0.001, 0.1, 10)

    def choose_action_with_sensors(self, state, sensor_data, available_actions=None):
        return np.random.randint(0, 10)

    def update_q_value_with_sensors(self, state, action, reward, next_state, sensor_data):
        pass

    def get_controller_performance_summary(self):
        return {"performance": "good"}


def create_mock_health_monitor(*args, **kwargs):
    """Create a mock health monitor."""
    monitor = MagicMock()
    monitor.assess_health.return_value = MockHealthMetrics()
    monitor.generate_alerts.return_value = []
    monitor.generate_intervention_recommendations.return_value = []
    monitor.get_health_dashboard_data.return_value = {}
    return monitor


def create_mock_sensor_fusion(*args, **kwargs):
    """Create a mock sensor fusion."""
    fusion = MagicMock()
    fusion.fuse_measurements_with_prediction.return_value = (
        MockFusedMeasurement(),
        MockPredictiveState(),
        [],
    )
    fusion.analyze_biofilm_growth_pattern.return_value = MagicMock()
    fusion.get_system_health_assessment.return_value = {}
    return fusion


@pytest.fixture(scope="module")
def mock_modules():
    """Set up mock modules for testing."""
    mock_biofilm = MagicMock()
    mock_advanced_fusion = MagicMock()
    mock_sensor_fusion = MagicMock()
    mock_config = MagicMock()
    mock_sensing_enhanced = MagicMock()
    mock_mfc_recirculation = MagicMock()

    # Set up mock classes
    mock_biofilm.HealthStatus = MockHealthStatus
    mock_biofilm.HealthTrend = MockHealthTrend
    mock_biofilm.HealthMetrics = MockHealthMetrics
    mock_biofilm.HealthAlert = MagicMock()
    mock_biofilm.InterventionRecommendation = MagicMock()
    mock_biofilm.create_predictive_health_monitor = create_mock_health_monitor

    mock_advanced_fusion.FusedMeasurement = MockFusedMeasurement
    mock_advanced_fusion.PredictiveState = MockPredictiveState
    mock_advanced_fusion.AnomalyDetection = MockAnomalyDetection
    mock_advanced_fusion.create_advanced_sensor_fusion = create_mock_sensor_fusion

    mock_sensor_fusion.BacterialSpecies = MockBacterialSpecies

    mock_config.QLearningConfig = MockQLearningConfig
    mock_config.SensorConfig = MockSensorConfig
    mock_config.validate_qlearning_config = MagicMock()
    mock_config.validate_sensor_config = MagicMock()

    mock_mfc_recirculation.AdvancedQLearningFlowController = MockAdvancedQLearningFlowController
    mock_sensing_enhanced.SensingEnhancedQLearningController = MockSensingEnhancedQLearningController

    return {
        "biofilm": mock_biofilm,
        "advanced_fusion": mock_advanced_fusion,
        "sensor_fusion": mock_sensor_fusion,
        "config": mock_config,
        "sensing_enhanced": mock_sensing_enhanced,
        "mfc_recirculation": mock_mfc_recirculation,
    }


@pytest.fixture
def adaptive_module(mock_modules):
    """Import adaptive_mfc_controller with mocks."""
    # Clear cached import
    if "adaptive_mfc_controller" in sys.modules:
        del sys.modules["adaptive_mfc_controller"]

    with patch.dict(
        "sys.modules",
        {
            "biofilm_health_monitor": mock_modules["biofilm"],
            "sensing_models.advanced_sensor_fusion": mock_modules["advanced_fusion"],
            "sensing_models.sensor_fusion": mock_modules["sensor_fusion"],
            "config": mock_modules["config"],
            "sensing_enhanced_q_controller": mock_modules["sensing_enhanced"],
            "mfc_recirculation_control": mock_modules["mfc_recirculation"],
        },
    ):
        from adaptive_mfc_controller import (
            AdaptationMode,
            AdaptiveMFCController,
            ControlDecision,
            ControlStrategy,
            HealthAwareQLearning,
            SystemState,
            create_adaptive_mfc_controller,
        )

        yield {
            "AdaptiveMFCController": AdaptiveMFCController,
            "HealthAwareQLearning": HealthAwareQLearning,
            "ControlStrategy": ControlStrategy,
            "AdaptationMode": AdaptationMode,
            "ControlDecision": ControlDecision,
            "SystemState": SystemState,
            "create_adaptive_mfc_controller": create_adaptive_mfc_controller,
            "mocks": mock_modules,
        }


@pytest.fixture
def health_controller(adaptive_module):
    """Create a HealthAwareQLearning controller."""
    return adaptive_module["HealthAwareQLearning"](
        qlearning_config=MockQLearningConfig(),
        sensor_config=MockSensorConfig(),
    )


@pytest.fixture
def mock_system_state(adaptive_module):
    """Create a mock system state with default values."""
    SystemState = adaptive_module["SystemState"]
    return SystemState(
        fused_measurement=MockFusedMeasurement(),
        prediction=MockPredictiveState(),
        anomalies=[],
        health_metrics=MockHealthMetrics(),
        health_alerts=[],
        flow_rate=15.0,
        inlet_concentration=10.0,
        outlet_concentration=8.0,
        current_density=0.5,
        power_output=0.1,
        current_strategy=adaptive_module["ControlStrategy"].BALANCED,
        adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
        intervention_active=False,
    )


# =============================================================================
# Test State Discretization with Configurable Bins
# =============================================================================


class TestStateDiscretization:
    """Tests for state discretization functionality."""

    def test_state_conversion_returns_tuple(self, health_controller, mock_system_state):
        """Test state conversion returns a tuple."""
        state = health_controller._system_state_to_qlearning_state(mock_system_state)
        assert isinstance(state, tuple)
        assert len(state) == 3

    def test_state_discretization_produces_integers(self, health_controller, mock_system_state):
        """Test state values are discretized to integers."""
        state = health_controller._system_state_to_qlearning_state(mock_system_state)
        for value in state:
            assert isinstance(value, int)

    def test_state_discretization_non_negative(self, health_controller, mock_system_state):
        """Test state values are non-negative."""
        state = health_controller._system_state_to_qlearning_state(mock_system_state)
        for value in state:
            assert value >= 0

    def test_inlet_concentration_binning(self, health_controller, adaptive_module):
        """Test inlet concentration is discretized to bins."""
        SystemState = adaptive_module["SystemState"]

        # Test different inlet concentrations
        for inlet_conc in [0.0, 5.0, 10.0, 20.0, 50.0]:
            system_state = SystemState(
                fused_measurement=MockFusedMeasurement(),
                prediction=MockPredictiveState(),
                anomalies=[],
                health_metrics=MockHealthMetrics(),
                health_alerts=[],
                flow_rate=15.0,
                inlet_concentration=inlet_conc,
                outlet_concentration=8.0,
                current_density=0.5,
                power_output=0.1,
                current_strategy=adaptive_module["ControlStrategy"].BALANCED,
                adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
                intervention_active=False,
            )
            state = health_controller._system_state_to_qlearning_state(system_state)
            expected_bin = int(inlet_conc / 5.0)
            assert state[0] == expected_bin

    def test_outlet_concentration_binning(self, health_controller, adaptive_module):
        """Test outlet concentration is discretized to bins."""
        SystemState = adaptive_module["SystemState"]

        for outlet_conc in [0.0, 5.0, 10.0, 15.0, 25.0]:
            system_state = SystemState(
                fused_measurement=MockFusedMeasurement(),
                prediction=MockPredictiveState(),
                anomalies=[],
                health_metrics=MockHealthMetrics(),
                health_alerts=[],
                flow_rate=15.0,
                inlet_concentration=10.0,
                outlet_concentration=outlet_conc,
                current_density=0.5,
                power_output=0.1,
                current_strategy=adaptive_module["ControlStrategy"].BALANCED,
                adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
                intervention_active=False,
            )
            state = health_controller._system_state_to_qlearning_state(system_state)
            expected_bin = int(outlet_conc / 5.0)
            assert state[1] == expected_bin

    def test_current_density_binning(self, health_controller, adaptive_module):
        """Test current density is discretized to bins."""
        SystemState = adaptive_module["SystemState"]

        for current_density in [0.0, 0.1, 0.5, 1.0, 2.0]:
            system_state = SystemState(
                fused_measurement=MockFusedMeasurement(),
                prediction=MockPredictiveState(),
                anomalies=[],
                health_metrics=MockHealthMetrics(),
                health_alerts=[],
                flow_rate=15.0,
                inlet_concentration=10.0,
                outlet_concentration=8.0,
                current_density=current_density,
                power_output=0.1,
                current_strategy=adaptive_module["ControlStrategy"].BALANCED,
                adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
                intervention_active=False,
            )
            state = health_controller._system_state_to_qlearning_state(system_state)
            expected_bin = int(current_density / 0.1)
            assert state[2] == expected_bin

    def test_different_states_produce_different_discretizations(self, health_controller, adaptive_module):
        """Test different system states produce different discretized states."""
        SystemState = adaptive_module["SystemState"]

        state1 = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=5.0,
            outlet_concentration=4.0,
            current_density=0.2,
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )

        state2 = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=20.0,
            outlet_concentration=16.0,
            current_density=1.0,
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )

        disc_state1 = health_controller._system_state_to_qlearning_state(state1)
        disc_state2 = health_controller._system_state_to_qlearning_state(state2)
        assert disc_state1 != disc_state2

    def test_state_bin_consistency(self, health_controller, mock_system_state):
        """Test same state produces consistent discretization."""
        state1 = health_controller._system_state_to_qlearning_state(mock_system_state)
        state2 = health_controller._system_state_to_qlearning_state(mock_system_state)
        assert state1 == state2


# =============================================================================
# Test State Bounds Validation
# =============================================================================


class TestStateBoundsValidation:
    """Tests for state bounds validation."""

    def test_zero_values_handled(self, health_controller, adaptive_module):
        """Test zero values are handled correctly."""
        SystemState = adaptive_module["SystemState"]

        system_state = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=0.0,
            inlet_concentration=0.0,
            outlet_concentration=0.0,
            current_density=0.0,
            power_output=0.0,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )
        state = health_controller._system_state_to_qlearning_state(system_state)
        assert all(v >= 0 for v in state)
        assert state == (0, 0, 0)

    def test_small_positive_values(self, health_controller, adaptive_module):
        """Test small positive values near zero."""
        SystemState = adaptive_module["SystemState"]

        system_state = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=0.01,
            inlet_concentration=0.01,
            outlet_concentration=0.01,
            current_density=0.01,
            power_output=0.001,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )
        state = health_controller._system_state_to_qlearning_state(system_state)
        # Small values should map to bin 0
        assert all(v >= 0 for v in state)

    def test_large_values_produce_large_bins(self, health_controller, adaptive_module):
        """Test large values produce appropriate bin indices."""
        SystemState = adaptive_module["SystemState"]

        system_state = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=100.0,
            inlet_concentration=100.0,
            outlet_concentration=100.0,
            current_density=10.0,
            power_output=1.0,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )
        state = health_controller._system_state_to_qlearning_state(system_state)
        # Large values should produce large bin indices
        assert state[0] == 20  # 100 / 5.0
        assert state[1] == 20  # 100 / 5.0
        assert state[2] == 100  # 10.0 / 0.1

    def test_negative_concentration_values(self, health_controller, adaptive_module):
        """Test negative concentration values (edge case)."""
        SystemState = adaptive_module["SystemState"]

        system_state = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=-5.0,
            outlet_concentration=-5.0,
            current_density=-0.5,
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )
        state = health_controller._system_state_to_qlearning_state(system_state)
        # Negative values produce negative bins (implementation dependent)
        assert isinstance(state[0], int)
        assert isinstance(state[1], int)
        assert isinstance(state[2], int)

    def test_fractional_values(self, health_controller, adaptive_module):
        """Test fractional values are handled with int conversion."""
        SystemState = adaptive_module["SystemState"]

        system_state = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=7.5,  # Should map to bin 1 (7.5/5.0 = 1.5 -> 1)
            outlet_concentration=3.5,  # Should map to bin 0 (3.5/5.0 = 0.7 -> 0)
            current_density=0.35,  # Should map to bin 3 (0.35/0.1 = 3.5 -> 3)
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )
        state = health_controller._system_state_to_qlearning_state(system_state)
        assert state[0] == 1
        assert state[1] == 0
        assert state[2] == 3


# =============================================================================
# Test State Transition Tracking
# =============================================================================


class TestStateTransitionTracking:
    """Tests for state transition tracking."""

    def test_adaptation_history_initialized_empty(self, health_controller):
        """Test adaptation history starts empty."""
        assert health_controller.adaptation_history == []

    def test_adaptation_records_state_transition(self, health_controller, mock_system_state):
        """Test adaptation records state transition."""
        initial_len = len(health_controller.adaptation_history)
        health_controller.adapt_parameters(MockHealthMetrics(), mock_system_state)
        assert len(health_controller.adaptation_history) == initial_len + 1

    def test_adaptation_history_contains_timestamp(self, health_controller, mock_system_state):
        """Test adaptation history contains timestamp."""
        health_controller.adapt_parameters(MockHealthMetrics(), mock_system_state)
        latest = health_controller.adaptation_history[-1]
        assert "timestamp" in latest

    def test_adaptation_history_contains_learning_rate(self, health_controller, mock_system_state):
        """Test adaptation history contains learning rate."""
        health_controller.adapt_parameters(MockHealthMetrics(), mock_system_state)
        latest = health_controller.adaptation_history[-1]
        assert "learning_rate" in latest
        assert isinstance(latest["learning_rate"], float)

    def test_adaptation_history_contains_epsilon(self, health_controller, mock_system_state):
        """Test adaptation history contains epsilon."""
        health_controller.adapt_parameters(MockHealthMetrics(), mock_system_state)
        latest = health_controller.adaptation_history[-1]
        assert "epsilon" in latest
        assert isinstance(latest["epsilon"], float)

    def test_adaptation_history_contains_discount_factor(self, health_controller, mock_system_state):
        """Test adaptation history contains discount factor."""
        health_controller.adapt_parameters(MockHealthMetrics(), mock_system_state)
        latest = health_controller.adaptation_history[-1]
        assert "discount_factor" in latest
        assert isinstance(latest["discount_factor"], float)

    def test_adaptation_history_contains_health_score(self, health_controller, mock_system_state):
        """Test adaptation history contains health score."""
        health_controller.adapt_parameters(MockHealthMetrics(), mock_system_state)
        latest = health_controller.adaptation_history[-1]
        assert "health_score" in latest

    def test_adaptation_history_contains_trigger(self, health_controller, mock_system_state):
        """Test adaptation history contains trigger."""
        health_controller.adapt_parameters(MockHealthMetrics(), mock_system_state)
        latest = health_controller.adaptation_history[-1]
        assert "adaptation_trigger" in latest

    def test_multiple_adaptations_tracked(self, health_controller, mock_system_state):
        """Test multiple adaptations are tracked."""
        for i in range(5):
            health_metrics = MockHealthMetrics()
            health_metrics.overall_health_score = 0.5 + i * 0.1
            health_controller.adapt_parameters(health_metrics, mock_system_state)
        assert len(health_controller.adaptation_history) == 5

    def test_transition_trigger_critical_health(self, health_controller, mock_system_state):
        """Test trigger identification for critical health."""
        health_metrics = MockHealthMetrics()
        health_metrics.health_status = MockHealthStatus.CRITICAL
        health_controller.adapt_parameters(health_metrics, mock_system_state)
        latest = health_controller.adaptation_history[-1]
        assert "critical_health" in latest["adaptation_trigger"]

    def test_transition_trigger_volatile_trend(self, health_controller, mock_system_state):
        """Test trigger identification for volatile trend."""
        health_metrics = MockHealthMetrics()
        health_metrics.health_trend = MockHealthTrend.VOLATILE
        health_controller.adapt_parameters(health_metrics, mock_system_state)
        latest = health_controller.adaptation_history[-1]
        assert "volatile_trend" in latest["adaptation_trigger"]

    def test_transition_trigger_high_risk(self, health_controller, mock_system_state):
        """Test trigger identification for high risk."""
        health_metrics = MockHealthMetrics()
        health_metrics.fouling_risk = 0.9
        health_controller.adapt_parameters(health_metrics, mock_system_state)
        latest = health_controller.adaptation_history[-1]
        assert "high_risk" in latest["adaptation_trigger"]

    def test_transition_trigger_routine(self, health_controller, mock_system_state):
        """Test trigger identification for routine adaptation."""
        health_metrics = MockHealthMetrics()  # Default healthy state
        health_controller.adapt_parameters(health_metrics, mock_system_state)
        latest = health_controller.adaptation_history[-1]
        assert latest["adaptation_trigger"] == "routine_adaptation"


# =============================================================================
# Test State History Buffer
# =============================================================================


class TestStateHistoryBuffer:
    """Tests for state history buffer functionality."""

    def test_health_reward_history_initialized_empty(self, health_controller):
        """Test health reward history starts empty."""
        assert health_controller.health_reward_history == []

    def test_power_reward_history_initialized_empty(self, health_controller):
        """Test power reward history starts empty."""
        assert health_controller.power_reward_history == []

    def test_intervention_history_initialized_empty(self, health_controller):
        """Test intervention history starts empty."""
        assert health_controller.intervention_history == []

    def test_reward_calculation_updates_health_history(self, health_controller, mock_system_state):
        """Test reward calculation updates health reward history."""
        initial_len = len(health_controller.health_reward_history)
        health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=MockHealthMetrics(),
            system_state=mock_system_state,
        )
        assert len(health_controller.health_reward_history) == initial_len + 1

    def test_reward_calculation_updates_power_history(self, health_controller, mock_system_state):
        """Test reward calculation updates power reward history."""
        initial_len = len(health_controller.power_reward_history)
        health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=MockHealthMetrics(),
            system_state=mock_system_state,
        )
        assert len(health_controller.power_reward_history) == initial_len + 1

    def test_multiple_rewards_tracked(self, health_controller, mock_system_state):
        """Test multiple reward calculations are tracked."""
        for i in range(10):
            health_controller.calculate_health_aware_reward(
                base_reward=float(i),
                health_metrics=MockHealthMetrics(),
                system_state=mock_system_state,
            )
        assert len(health_controller.health_reward_history) == 10
        assert len(health_controller.power_reward_history) == 10

    def test_health_reward_values_stored(self, health_controller, mock_system_state):
        """Test health reward values are properly stored."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 0.9
        health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
        )
        assert len(health_controller.health_reward_history) > 0
        assert isinstance(health_controller.health_reward_history[-1], float)

    def test_power_reward_values_stored(self, health_controller, mock_system_state):
        """Test power reward values are properly stored."""
        health_controller.calculate_health_aware_reward(
            base_reward=2.0,
            health_metrics=MockHealthMetrics(),
            system_state=mock_system_state,
        )
        assert len(health_controller.power_reward_history) > 0
        # Power reward = base_reward * power_weight = 2.0 * 0.6 = 1.2
        assert isinstance(health_controller.power_reward_history[-1], float)

    def test_controller_control_history_initialized(self, adaptive_module):
        """Test AdaptiveMFCController control history is initialized."""
        controller = adaptive_module["AdaptiveMFCController"]()
        assert hasattr(controller, "control_history")
        assert controller.control_history == []

    def test_controller_strategy_changes_tracked(self, adaptive_module):
        """Test AdaptiveMFCController strategy changes are tracked."""
        controller = adaptive_module["AdaptiveMFCController"]()
        assert hasattr(controller, "strategy_changes")
        assert controller.strategy_changes == []

    def test_controller_intervention_outcomes_tracked(self, adaptive_module):
        """Test AdaptiveMFCController intervention outcomes are tracked."""
        controller = adaptive_module["AdaptiveMFCController"]()
        assert hasattr(controller, "intervention_outcomes")
        assert controller.intervention_outcomes == []


# =============================================================================
# Test Edge Cases: Out-of-Bounds and NaN Values
# =============================================================================


class TestEdgeCasesOutOfBounds:
    """Tests for out-of-bounds edge cases."""

    def test_very_large_inlet_concentration(self, health_controller, adaptive_module):
        """Test very large inlet concentration."""
        SystemState = adaptive_module["SystemState"]

        system_state = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=1000.0,
            outlet_concentration=8.0,
            current_density=0.5,
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )
        state = health_controller._system_state_to_qlearning_state(system_state)
        assert isinstance(state[0], int)
        assert state[0] == 200  # 1000 / 5.0

    def test_very_large_current_density(self, health_controller, adaptive_module):
        """Test very large current density."""
        SystemState = adaptive_module["SystemState"]

        system_state = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=10.0,
            outlet_concentration=8.0,
            current_density=100.0,
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )
        state = health_controller._system_state_to_qlearning_state(system_state)
        assert isinstance(state[2], int)
        assert state[2] == 1000  # 100.0 / 0.1

    def test_extreme_health_score_zero(self, health_controller, mock_system_state):
        """Test extreme health score of zero."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 0.0
        reward = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
        )
        assert isinstance(reward, float)
        assert not math.isnan(reward)

    def test_extreme_health_score_one(self, health_controller, mock_system_state):
        """Test extreme health score of 1.0."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 1.0
        reward = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
        )
        assert isinstance(reward, float)
        assert not math.isnan(reward)

    def test_all_risks_at_maximum(self, health_controller, mock_system_state):
        """Test all risks at maximum values."""
        health_metrics = MockHealthMetrics()
        health_metrics.fouling_risk = 1.0
        health_metrics.detachment_risk = 1.0
        health_metrics.stagnation_risk = 1.0
        reward = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
        )
        assert isinstance(reward, float)
        assert not math.isnan(reward)


class TestEdgeCasesNaNValues:
    """Tests for NaN value edge cases."""

    def test_nan_inlet_concentration(self, health_controller, adaptive_module):
        """Test NaN inlet concentration handling."""
        SystemState = adaptive_module["SystemState"]

        system_state = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=float("nan"),
            outlet_concentration=8.0,
            current_density=0.5,
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )
        # NaN handling - int(nan) raises ValueError in Python
        with pytest.raises((ValueError, TypeError)):
            health_controller._system_state_to_qlearning_state(system_state)

    def test_nan_current_density(self, health_controller, adaptive_module):
        """Test NaN current density handling."""
        SystemState = adaptive_module["SystemState"]

        system_state = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=10.0,
            outlet_concentration=8.0,
            current_density=float("nan"),
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )
        with pytest.raises((ValueError, TypeError)):
            health_controller._system_state_to_qlearning_state(system_state)

    def test_nan_base_reward_handling(self, health_controller, mock_system_state):
        """Test NaN base reward handling."""
        reward = health_controller.calculate_health_aware_reward(
            base_reward=float("nan"),
            health_metrics=MockHealthMetrics(),
            system_state=mock_system_state,
        )
        # NaN propagates through calculations
        assert math.isnan(reward)

    def test_inf_values_in_concentration(self, health_controller, adaptive_module):
        """Test infinity values in concentration."""
        SystemState = adaptive_module["SystemState"]

        system_state = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=float("inf"),
            outlet_concentration=8.0,
            current_density=0.5,
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )
        # Infinity may cause overflow or special handling
        with pytest.raises((OverflowError, ValueError)):
            health_controller._system_state_to_qlearning_state(system_state)


class TestEdgeCasesAnomalies:
    """Tests for anomaly edge cases."""

    def test_single_anomaly_in_state(self, health_controller, adaptive_module):
        """Test single anomaly in system state."""
        SystemState = adaptive_module["SystemState"]

        system_state = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[MockAnomalyDetection()],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=10.0,
            outlet_concentration=8.0,
            current_density=0.5,
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )
        decision = health_controller.choose_health_aware_action(system_state)
        assert decision is not None

    def test_multiple_anomalies_in_state(self, health_controller, adaptive_module):
        """Test multiple anomalies in system state."""
        SystemState = adaptive_module["SystemState"]

        anomalies = [
            MockAnomalyDetection(severity="low"),
            MockAnomalyDetection(severity="medium"),
            MockAnomalyDetection(severity="high"),
        ]
        system_state = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=anomalies,
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=10.0,
            outlet_concentration=8.0,
            current_density=0.5,
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )
        decision = health_controller.choose_health_aware_action(system_state)
        assert decision is not None
        # Confidence should be reduced due to anomalies
        assert decision.confidence < 1.0

    def test_critical_anomaly_affects_decision(self, health_controller, adaptive_module):
        """Test critical anomaly affects decision confidence."""
        SystemState = adaptive_module["SystemState"]

        critical_anomaly = MockAnomalyDetection(severity="critical")
        system_state = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[critical_anomaly],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=10.0,
            outlet_concentration=8.0,
            current_density=0.5,
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )
        # Trigger adaptation
        health_controller.adapt_parameters(MockHealthMetrics(), system_state)
        latest = health_controller.adaptation_history[-1]
        assert "critical_anomalies" in latest["adaptation_trigger"]


class TestEdgeCasesPrediction:
    """Tests for prediction edge cases."""

    def test_null_prediction_handled(self, health_controller, adaptive_module):
        """Test null prediction is handled."""
        SystemState = adaptive_module["SystemState"]

        system_state = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=None,  # No prediction available
            anomalies=[],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=10.0,
            outlet_concentration=8.0,
            current_density=0.5,
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )
        # Should not raise an error
        state = health_controller._system_state_to_qlearning_state(system_state)
        assert isinstance(state, tuple)


class TestControllerStateManagement:
    """Tests for AdaptiveMFCController state management."""

    def test_system_parameters_accessible(self, adaptive_module):
        """Test system parameters are accessible."""
        controller = adaptive_module["AdaptiveMFCController"]()
        assert hasattr(controller, "system_parameters")
        assert "flow_rate" in controller.system_parameters
        assert "inlet_concentration" in controller.system_parameters

    def test_system_parameters_modifiable(self, adaptive_module):
        """Test system parameters can be modified."""
        controller = adaptive_module["AdaptiveMFCController"]()
        original_flow = controller.system_parameters["flow_rate"]
        controller.system_parameters["flow_rate"] = 20.0
        assert controller.system_parameters["flow_rate"] == 20.0
        assert controller.system_parameters["flow_rate"] != original_flow

    def test_current_strategy_state(self, adaptive_module):
        """Test current strategy state is maintained."""
        controller = adaptive_module["AdaptiveMFCController"]()
        assert controller.current_strategy == adaptive_module["ControlStrategy"].BALANCED
        controller.current_strategy = adaptive_module["ControlStrategy"].HEALTH_FOCUSED
        assert controller.current_strategy == adaptive_module["ControlStrategy"].HEALTH_FOCUSED

    def test_adaptation_mode_state(self, adaptive_module):
        """Test adaptation mode state is maintained."""
        controller = adaptive_module["AdaptiveMFCController"]()
        assert controller.adaptation_mode == adaptive_module["AdaptationMode"].MODERATE
        controller.adaptation_mode = adaptive_module["AdaptationMode"].AGGRESSIVE
        assert controller.adaptation_mode == adaptive_module["AdaptationMode"].AGGRESSIVE

    def test_intervention_active_state(self, adaptive_module):
        """Test intervention active state is maintained."""
        controller = adaptive_module["AdaptiveMFCController"]()
        assert controller.intervention_active is False
        controller.intervention_active = True
        assert controller.intervention_active is True

    def test_last_intervention_time_tracked(self, adaptive_module):
        """Test last intervention time is tracked."""
        controller = adaptive_module["AdaptiveMFCController"]()
        assert hasattr(controller, "last_intervention_time")
        assert controller.last_intervention_time == 0.0


class TestHealthRewardBounds:
    """Tests for health reward calculation bounds."""

    def test_health_reward_clipped_upper(self, health_controller, mock_system_state):
        """Test health reward is clipped at upper bound."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 1.0
        health_metrics.thickness_health = 1.0
        health_metrics.conductivity_health = 1.0
        health_metrics.growth_health = 1.0
        health_metrics.health_trend = MockHealthTrend.IMPROVING
        health_metrics.predicted_health_24h = 1.5

        health_reward = health_controller._calculate_health_reward(
            health_metrics,
            mock_system_state,
        )
        assert health_reward <= 1.5

    def test_health_reward_clipped_lower(self, health_controller, mock_system_state):
        """Test health reward is clipped at lower bound."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 0.0
        health_metrics.health_trend = MockHealthTrend.DECLINING

        health_reward = health_controller._calculate_health_reward(
            health_metrics,
            mock_system_state,
        )
        assert health_reward >= -0.5

    def test_risk_penalty_accumulates(self, health_controller):
        """Test risk penalties accumulate correctly."""
        health_metrics = MockHealthMetrics()
        health_metrics.fouling_risk = 0.9
        health_metrics.detachment_risk = 0.9
        health_metrics.stagnation_risk = 0.9
        health_metrics.health_status = MockHealthStatus.CRITICAL

        penalty = health_controller._calculate_risk_penalty(health_metrics)
        # Each risk above 0.7 contributes, plus critical health penalty
        assert penalty > 0.5


class TestDecisionConfidenceCalculation:
    """Tests for decision confidence calculation."""

    def test_confidence_reduced_by_risk(self, health_controller, adaptive_module):
        """Test confidence is reduced by risk factors."""
        SystemState = adaptive_module["SystemState"]

        health_metrics = MockHealthMetrics()
        health_metrics.fouling_risk = 0.9

        system_state = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[],
            health_metrics=health_metrics,
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=10.0,
            outlet_concentration=8.0,
            current_density=0.5,
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )

        confidence = health_controller._calculate_decision_confidence(0, system_state)
        # High risk should reduce confidence
        assert confidence < 0.9

    def test_confidence_reduced_by_critical_health(self, health_controller, adaptive_module):
        """Test confidence is reduced by critical health."""
        SystemState = adaptive_module["SystemState"]

        health_metrics = MockHealthMetrics()
        health_metrics.health_status = MockHealthStatus.CRITICAL

        system_state = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[],
            health_metrics=health_metrics,
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=10.0,
            outlet_concentration=8.0,
            current_density=0.5,
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )

        confidence = health_controller._calculate_decision_confidence(0, system_state)
        assert confidence < 0.9

    def test_confidence_in_valid_range(self, health_controller, mock_system_state):
        """Test confidence is always in valid range."""
        for action_idx in range(10):
            confidence = health_controller._calculate_decision_confidence(
                action_idx,
                mock_system_state,
            )
            assert 0.1 <= confidence <= 1.0


class TestGetAdaptationTrigger:
    """Tests for adaptation trigger identification."""

    def test_trigger_multiple_conditions(self, health_controller, adaptive_module):
        """Test trigger with multiple conditions."""
        SystemState = adaptive_module["SystemState"]

        health_metrics = MockHealthMetrics()
        health_metrics.health_status = MockHealthStatus.CRITICAL
        health_metrics.health_trend = MockHealthTrend.VOLATILE
        health_metrics.fouling_risk = 0.9

        system_state = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[MockAnomalyDetection(severity="critical")],
            health_metrics=health_metrics,
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=10.0,
            outlet_concentration=8.0,
            current_density=0.5,
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )

        trigger = health_controller._get_adaptation_trigger(health_metrics, system_state)
        assert "critical_health" in trigger
        assert "volatile_trend" in trigger
        assert "high_risk" in trigger
        assert "critical_anomalies" in trigger

    def test_trigger_with_no_special_conditions(self, health_controller, mock_system_state):
        """Test trigger with normal conditions returns routine."""
        trigger = health_controller._get_adaptation_trigger(
            MockHealthMetrics(),
            mock_system_state,
        )
        assert trigger == "routine_adaptation"


class TestStabilityBonus:
    """Tests for stability bonus calculation."""

    def test_stability_bonus_high_confidence(self, health_controller, adaptive_module):
        """Test stability bonus with high fusion confidence."""
        SystemState = adaptive_module["SystemState"]

        fused = MockFusedMeasurement()
        fused.fusion_confidence = 0.95
        fused.sensor_agreement = 0.95

        system_state = SystemState(
            fused_measurement=fused,
            prediction=MockPredictiveState(),
            anomalies=[],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=10.0,
            outlet_concentration=8.0,
            current_density=0.5,
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )

        bonus = health_controller._calculate_stability_bonus(system_state)
        # High confidence should give bonus
        assert bonus > 0.0

    def test_stability_bonus_no_anomalies(self, health_controller, mock_system_state):
        """Test stability bonus with no anomalies."""
        bonus = health_controller._calculate_stability_bonus(mock_system_state)
        # No anomalies should give bonus
        assert bonus > 0.0

    def test_stability_bonus_with_anomalies(self, health_controller, adaptive_module):
        """Test stability bonus reduced with anomalies."""
        SystemState = adaptive_module["SystemState"]

        system_state_no_anomalies = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=10.0,
            outlet_concentration=8.0,
            current_density=0.5,
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )

        system_state_with_anomalies = SystemState(
            fused_measurement=MockFusedMeasurement(),
            prediction=MockPredictiveState(),
            anomalies=[MockAnomalyDetection(severity="high")],
            health_metrics=MockHealthMetrics(),
            health_alerts=[],
            flow_rate=15.0,
            inlet_concentration=10.0,
            outlet_concentration=8.0,
            current_density=0.5,
            power_output=0.1,
            current_strategy=adaptive_module["ControlStrategy"].BALANCED,
            adaptation_mode=adaptive_module["AdaptationMode"].MODERATE,
            intervention_active=False,
        )

        bonus_no = health_controller._calculate_stability_bonus(system_state_no_anomalies)
        bonus_with = health_controller._calculate_stability_bonus(system_state_with_anomalies)
        # Bonus should be less with anomalies
        assert bonus_no >= bonus_with
