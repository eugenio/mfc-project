"""Tests for Adaptive MFC Controller Core Logic.

US-010: Test Adaptive Controller Core Logic
Target: 50%+ coverage for core logic

Tests cover:
- AdaptiveMFCController initialization
- State observation and action selection
- Q-table updates with learning rate decay
- Exploration/exploitation balance
- Action space discretization
"""

from __future__ import annotations

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
    mock_sensing = MagicMock()
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


class TestControlStrategy:
    """Tests for control strategy enumeration."""

    def test_all_strategies_exist(self, adaptive_module):
        """Test all control strategies are defined."""
        ControlStrategy = adaptive_module["ControlStrategy"]
        assert hasattr(ControlStrategy, "PERFORMANCE_FOCUSED")
        assert hasattr(ControlStrategy, "HEALTH_FOCUSED")
        assert hasattr(ControlStrategy, "BALANCED")
        assert hasattr(ControlStrategy, "CONSERVATIVE")
        assert hasattr(ControlStrategy, "RECOVERY")

    def test_strategy_values(self, adaptive_module):
        """Test strategy values are correct."""
        ControlStrategy = adaptive_module["ControlStrategy"]
        assert ControlStrategy.PERFORMANCE_FOCUSED.value == "performance_focused"
        assert ControlStrategy.HEALTH_FOCUSED.value == "health_focused"
        assert ControlStrategy.BALANCED.value == "balanced"


class TestAdaptationMode:
    """Tests for adaptation mode enumeration."""

    def test_all_modes_exist(self, adaptive_module):
        """Test all adaptation modes are defined."""
        AdaptationMode = adaptive_module["AdaptationMode"]
        assert hasattr(AdaptationMode, "AGGRESSIVE")
        assert hasattr(AdaptationMode, "MODERATE")
        assert hasattr(AdaptationMode, "CONSERVATIVE")
        assert hasattr(AdaptationMode, "DISABLED")

    def test_mode_values(self, adaptive_module):
        """Test mode values are correct."""
        AdaptationMode = adaptive_module["AdaptationMode"]
        assert AdaptationMode.AGGRESSIVE.value == "aggressive"
        assert AdaptationMode.MODERATE.value == "moderate"


class TestAdaptiveMFCControllerInitialization:
    """Tests for AdaptiveMFCController initialization."""

    def test_default_initialization(self, adaptive_module):
        """Test controller initializes with defaults."""
        controller = adaptive_module["AdaptiveMFCController"]()
        assert controller is not None
        assert controller.current_strategy == adaptive_module["ControlStrategy"].BALANCED
        assert controller.adaptation_mode == adaptive_module["AdaptationMode"].MODERATE
        assert controller.intervention_active is False

    def test_initialization_with_species(self, adaptive_module):
        """Test controller initializes with specific species."""
        controller = adaptive_module["AdaptiveMFCController"](species=MockBacterialSpecies.GEOBACTER)
        assert controller.species == MockBacterialSpecies.GEOBACTER

    def test_initialization_with_strategy(self, adaptive_module):
        """Test controller initializes with custom strategy."""
        controller = adaptive_module["AdaptiveMFCController"](
            initial_strategy=adaptive_module["ControlStrategy"].HEALTH_FOCUSED,
        )
        assert controller.current_strategy == adaptive_module["ControlStrategy"].HEALTH_FOCUSED

    def test_subsystems_initialized(self, adaptive_module):
        """Test subsystems are initialized."""
        controller = adaptive_module["AdaptiveMFCController"]()
        assert controller.sensor_fusion is not None
        assert controller.health_monitor is not None
        assert controller.q_controller is not None

    def test_system_parameters_defaults(self, adaptive_module):
        """Test default system parameters."""
        controller = adaptive_module["AdaptiveMFCController"]()
        assert "flow_rate" in controller.system_parameters
        assert "inlet_concentration" in controller.system_parameters
        assert controller.system_parameters["flow_rate"] == 15.0
        assert controller.system_parameters["temperature"] == 25.0

    def test_history_tracking_initialized(self, adaptive_module):
        """Test history lists are initialized."""
        controller = adaptive_module["AdaptiveMFCController"]()
        assert controller.control_history == []
        assert controller.strategy_changes == []
        assert controller.intervention_outcomes == []


class TestHealthAwareQLearningInitialization:
    """Tests for HealthAwareQLearning initialization."""

    def test_default_initialization(self, adaptive_module):
        """Test HealthAwareQLearning initializes with defaults."""
        controller = adaptive_module["HealthAwareQLearning"](
            qlearning_config=MockQLearningConfig(),
            sensor_config=MockSensorConfig(),
        )
        assert controller is not None
        assert controller.health_weight == 0.4
        assert controller.power_weight == pytest.approx(0.6, abs=0.01)

    def test_custom_health_weight(self, adaptive_module):
        """Test custom health weight."""
        controller = adaptive_module["HealthAwareQLearning"](
            qlearning_config=MockQLearningConfig(),
            sensor_config=MockSensorConfig(),
            health_weight=0.7,
        )
        assert controller.health_weight == 0.7
        assert controller.power_weight == pytest.approx(0.3, abs=0.01)

    def test_adaptation_rate(self, adaptive_module):
        """Test adaptation rate setting."""
        controller = adaptive_module["HealthAwareQLearning"](
            qlearning_config=MockQLearningConfig(),
            sensor_config=MockSensorConfig(),
            adaptation_rate=0.2,
        )
        assert controller.adaptation_rate == 0.2

    def test_base_parameters_stored(self, adaptive_module):
        """Test base parameters are stored."""
        controller = adaptive_module["HealthAwareQLearning"](
            qlearning_config=MockQLearningConfig(),
            sensor_config=MockSensorConfig(),
        )
        assert hasattr(controller, "base_learning_rate")
        assert hasattr(controller, "base_epsilon")
        assert hasattr(controller, "base_discount_factor")

    def test_risk_thresholds_defined(self, adaptive_module):
        """Test risk thresholds are defined."""
        controller = adaptive_module["HealthAwareQLearning"](
            qlearning_config=MockQLearningConfig(),
            sensor_config=MockSensorConfig(),
        )
        assert "high_risk_epsilon_boost" in controller.risk_thresholds
        assert "health_critical_epsilon" in controller.risk_thresholds

    def test_action_space_initialized(self, adaptive_module):
        """Test action space is initialized."""
        controller = adaptive_module["HealthAwareQLearning"](
            qlearning_config=MockQLearningConfig(),
            sensor_config=MockSensorConfig(),
        )
        assert hasattr(controller, "actions")
        assert len(controller.actions) > 0


@pytest.fixture
def health_controller(adaptive_module):
    """Create a HealthAwareQLearning controller."""
    return adaptive_module["HealthAwareQLearning"](
        qlearning_config=MockQLearningConfig(),
        sensor_config=MockSensorConfig(),
    )


@pytest.fixture
def mock_system_state(adaptive_module):
    """Create a mock system state."""
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


class TestHealthAwareReward:
    """Tests for health-aware reward calculation."""

    def test_reward_calculation_basic(self, health_controller, mock_system_state):
        """Test basic reward calculation."""
        health_metrics = MockHealthMetrics()
        reward = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
            intervention_active=False,
        )
        assert isinstance(reward, float)

    def test_reward_with_intervention(self, health_controller, mock_system_state):
        """Test reward with intervention active."""
        health_metrics = MockHealthMetrics()
        reward_no = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
            intervention_active=False,
        )
        reward_yes = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
            intervention_active=True,
        )
        assert reward_yes != reward_no

    def test_reward_history_tracked(self, health_controller, mock_system_state):
        """Test reward history is tracked."""
        initial_len = len(health_controller.health_reward_history)
        health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=MockHealthMetrics(),
            system_state=mock_system_state,
        )
        assert len(health_controller.health_reward_history) == initial_len + 1


class TestParameterAdaptation:
    """Tests for Q-learning parameter adaptation."""

    def test_adapt_parameters_called(self, health_controller, mock_system_state):
        """Test parameter adaptation is called."""
        initial_len = len(health_controller.adaptation_history)
        health_controller.adapt_parameters(MockHealthMetrics(), mock_system_state)
        assert len(health_controller.adaptation_history) == initial_len + 1

    def test_learning_rate_adaptation_critical(self, health_controller, mock_system_state):
        """Test learning rate increases for critical health."""
        health_metrics = MockHealthMetrics()
        health_metrics.health_status = MockHealthStatus.CRITICAL
        original_lr = health_controller.base_learning_rate
        health_controller.adapt_parameters(health_metrics, mock_system_state)
        assert health_controller.learning_rate >= original_lr

    def test_epsilon_adaptation_high_risk(self, health_controller, mock_system_state):
        """Test epsilon increases for high risk."""
        health_metrics = MockHealthMetrics()
        health_metrics.fouling_risk = 0.9
        original_epsilon = health_controller.base_epsilon
        health_controller.adapt_parameters(health_metrics, mock_system_state)
        assert health_controller.epsilon >= original_epsilon

    def test_adaptation_history_recorded(self, health_controller, mock_system_state):
        """Test adaptation events are recorded."""
        health_controller.adapt_parameters(MockHealthMetrics(), mock_system_state)
        assert len(health_controller.adaptation_history) > 0
        latest = health_controller.adaptation_history[-1]
        assert "timestamp" in latest
        assert "learning_rate" in latest
        assert "epsilon" in latest


class TestActionSelection:
    """Tests for health-aware action selection."""

    def test_choose_action_returns_decision(self, health_controller, mock_system_state, adaptive_module):
        """Test action selection returns ControlDecision."""
        decision = health_controller.choose_health_aware_action(mock_system_state)
        ControlDecision = adaptive_module["ControlDecision"]
        assert isinstance(decision, ControlDecision)

    def test_decision_has_required_fields(self, health_controller, mock_system_state):
        """Test decision has required fields."""
        decision = health_controller.choose_health_aware_action(mock_system_state)
        assert hasattr(decision, "action_index")
        assert hasattr(decision, "action_description")
        assert hasattr(decision, "expected_outcome")
        assert hasattr(decision, "confidence")
        assert hasattr(decision, "rationale")

    def test_decision_confidence_in_range(self, health_controller, mock_system_state):
        """Test decision confidence is 0-1."""
        decision = health_controller.choose_health_aware_action(mock_system_state)
        assert 0.0 <= decision.confidence <= 1.0


class TestStateConversion:
    """Tests for state to Q-learning state conversion."""

    def test_state_conversion_returns_tuple(self, health_controller, mock_system_state):
        """Test state conversion returns tuple."""
        state = health_controller._system_state_to_qlearning_state(mock_system_state)
        assert isinstance(state, tuple)
        assert len(state) == 3

    def test_state_discretization(self, health_controller, mock_system_state):
        """Test state values are discretized."""
        state = health_controller._system_state_to_qlearning_state(mock_system_state)
        assert all(isinstance(s, int) and s >= 0 for s in state)


class TestSensorDataPreparation:
    """Tests for sensor data preparation."""

    def test_sensor_data_structure(self, health_controller, mock_system_state):
        """Test sensor data structure."""
        sensor_data = health_controller._prepare_sensor_data(mock_system_state)
        assert "eis" in sensor_data
        assert "qcm" in sensor_data
        assert "fusion" in sensor_data

    def test_eis_data_fields(self, health_controller, mock_system_state):
        """Test EIS data fields."""
        sensor_data = health_controller._prepare_sensor_data(mock_system_state)
        eis = sensor_data["eis"]
        assert "thickness_um" in eis
        assert "conductivity_S_per_m" in eis

    def test_qcm_data_fields(self, health_controller, mock_system_state):
        """Test QCM data fields."""
        sensor_data = health_controller._prepare_sensor_data(mock_system_state)
        qcm = sensor_data["qcm"]
        assert "thickness_um" in qcm
        assert "measurement_quality" in qcm


class TestRiskAssessment:
    """Tests for decision risk assessment."""

    def test_risk_assessment_structure(self, health_controller, mock_system_state):
        """Test risk assessment structure."""
        risks = health_controller._assess_decision_risks(0, mock_system_state)
        assert "biofilm_damage" in risks
        assert "performance_loss" in risks
        assert "system_instability" in risks

    def test_risk_values_non_negative(self, health_controller, mock_system_state):
        """Test all risk values are non-negative."""
        for action_idx in range(10):
            risks = health_controller._assess_decision_risks(action_idx, mock_system_state)
            for risk_value in risks.values():
                assert risk_value >= 0.0


class TestActionOutcomePrediction:
    """Tests for action outcome prediction."""

    def test_outcome_prediction_structure(self, health_controller, mock_system_state):
        """Test outcome prediction structure."""
        outcomes = health_controller._predict_action_outcomes(1, mock_system_state)
        assert "thickness_change" in outcomes
        assert "health_change" in outcomes
        assert "power_change" in outcomes

    def test_different_actions_different_outcomes(self, health_controller, mock_system_state):
        """Test different actions produce different outcomes."""
        outcomes_0 = health_controller._predict_action_outcomes(0, mock_system_state)
        outcomes_1 = health_controller._predict_action_outcomes(1, mock_system_state)
        assert outcomes_0 != outcomes_1


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_controller_default(self, adaptive_module):
        """Test factory creates controller."""
        controller = adaptive_module["create_adaptive_mfc_controller"]()
        assert controller is not None
        assert isinstance(controller, adaptive_module["AdaptiveMFCController"])

    def test_create_controller_with_species(self, adaptive_module):
        """Test factory with species."""
        controller = adaptive_module["create_adaptive_mfc_controller"](
            species=MockBacterialSpecies.SHEWANELLA,
        )
        assert controller.species == MockBacterialSpecies.SHEWANELLA


class TestComprehensiveStatus:
    """Tests for comprehensive status retrieval."""

    def test_status_without_history(self, adaptive_module):
        """Test status returns error when no history."""
        controller = adaptive_module["AdaptiveMFCController"]()
        status = controller.get_comprehensive_status()
        assert "error" in status

    def test_status_structure_with_history(self, adaptive_module):
        """Test status structure with history."""
        controller = adaptive_module["AdaptiveMFCController"]()
        controller.control_history.append({
            "timestamp": 0.0,
            "system_health_score": 0.85,
            "health_alerts": [],
            "performance_metrics": {"efficiency": 0.9},
            "system_state": MagicMock(health_metrics=MockHealthMetrics()),
        })
        status = controller.get_comprehensive_status()
        assert "timestamp" in status
        assert "control_strategy" in status
        assert "system_health" in status


class TestExplorationExploitationBalance:
    """Tests for exploration/exploitation balance."""

    def test_epsilon_bounds(self, health_controller, mock_system_state):
        """Test epsilon stays within bounds."""
        for _ in range(10):
            health_controller.adapt_parameters(MockHealthMetrics(), mock_system_state)
        assert 0.0 <= health_controller.epsilon <= 1.0

    def test_high_risk_increases_exploration(self, health_controller, mock_system_state):
        """Test high risk increases exploration."""
        initial_epsilon = health_controller.epsilon
        health_metrics = MockHealthMetrics()
        health_metrics.fouling_risk = 0.9
        health_metrics.detachment_risk = 0.9
        health_controller.adapt_parameters(health_metrics, mock_system_state)
        assert health_controller.epsilon >= initial_epsilon


class TestActionSpaceDiscretization:
    """Tests for action space discretization."""

    def test_action_space_not_empty(self, health_controller):
        """Test action space is not empty."""
        assert len(health_controller.actions) > 0

    def test_actions_are_tuples(self, health_controller):
        """Test actions are tuples."""
        for action in health_controller.actions:
            assert isinstance(action, tuple)
            assert len(action) == 2  # (flow_change, substrate_change)


class TestLearningRateDecay:
    """Tests for Q-table updates with learning rate decay."""

    def test_learning_rate_changes_with_health(self, health_controller, mock_system_state):
        """Test learning rate adapts to health."""
        initial_lr = health_controller.learning_rate
        health_metrics = MockHealthMetrics()
        health_metrics.health_status = MockHealthStatus.CRITICAL
        health_controller.adapt_parameters(health_metrics, mock_system_state)
        assert health_controller.learning_rate != initial_lr

    def test_discount_factor_adapts(self, health_controller, mock_system_state):
        """Test discount factor adapts to critical health."""
        initial_df = health_controller.discount_factor
        health_metrics = MockHealthMetrics()
        health_metrics.health_status = MockHealthStatus.CRITICAL
        health_controller.adapt_parameters(health_metrics, mock_system_state)
        assert health_controller.discount_factor != initial_df
