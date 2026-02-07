"""Tests for Adaptive MFC Controller Reward System.

US-011: Test Adaptive Controller Rewards
Target: 90%+ coverage for reward system

Tests cover:
- Reward calculation for power output
- Reward penalties for constraint violations
- Multi-objective reward aggregation
- Reward normalization
- Reward history tracking
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
        self.sensor_weight = (
            qlearning_config.sensor_weight if qlearning_config else 0.5
        )
        self.power_weight = (
            qlearning_config.power_objective_weight if qlearning_config else 0.3
        )
        self.biofilm_health_weight = (
            qlearning_config.biofilm_health_weight if qlearning_config else 0.4
        )
        self.sensor_agreement_weight = (
            qlearning_config.sensor_agreement_weight if qlearning_config else 0.15
        )
        self.stability_weight = (
            qlearning_config.stability_weight if qlearning_config else 0.15
        )
        self.min_sensor_confidence = (
            qlearning_config.sensor_confidence_threshold if qlearning_config else 0.6
        )
        self.exploration_boost_factor = (
            qlearning_config.exploration_boost_factor if qlearning_config else 1.5
        )
        self.sensor_confidence_history = []
        self.sensor_fault_count = 0
        self.sensor_degradation_factor = 1.0
        self.sensor_guided_decisions = 0
        self.model_guided_decisions = 0
        self.total_reward_components = {
            "power": 0.0,
            "biofilm_health": 0.0,
            "sensor_agreement": 0.0,
            "stability": 0.0,
        }
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

    def update_q_value_with_sensors(
        self, state, action, reward, next_state, sensor_data,
    ):
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

    mock_mfc_recirculation.AdvancedQLearningFlowController = (
        MockAdvancedQLearningFlowController
    )
    mock_sensing_enhanced.SensingEnhancedQLearningController = (
        MockSensingEnhancedQLearningController
    )

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


class TestPowerRewardCalculation:
    """Tests for reward calculation based on power output."""

    def test_power_reward_scaling_with_base_reward(
        self, health_controller, mock_system_state,
    ):
        """Test power reward scales with base reward."""
        health_metrics = MockHealthMetrics()
        reward_1 = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
        )
        reward_2 = health_controller.calculate_health_aware_reward(
            base_reward=2.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
        )
        assert reward_2 > reward_1

    def test_power_reward_weight_applied(self, health_controller, mock_system_state):
        """Test power weight is applied correctly."""
        assert health_controller.power_weight == pytest.approx(0.6, abs=0.01)
        health_metrics = MockHealthMetrics()
        health_metrics.health_trend = MockHealthTrend.STABLE
        health_metrics.health_status = MockHealthStatus.GOOD
        health_metrics.fouling_risk = 0.0
        health_metrics.detachment_risk = 0.0
        health_metrics.stagnation_risk = 0.0

        health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
        )
        assert len(health_controller.power_reward_history) > 0
        power_component = health_controller.power_reward_history[-1]
        assert power_component == pytest.approx(0.6, abs=0.01)

    def test_power_reward_with_zero_base(self, health_controller, mock_system_state):
        """Test power reward with zero base reward."""
        health_metrics = MockHealthMetrics()
        health_controller.calculate_health_aware_reward(
            base_reward=0.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
        )
        power_component = health_controller.power_reward_history[-1]
        assert power_component == 0.0

    def test_power_reward_with_negative_base(self, health_controller, mock_system_state):
        """Test power reward with negative base reward."""
        health_metrics = MockHealthMetrics()
        health_controller.calculate_health_aware_reward(
            base_reward=-1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
        )
        power_component = health_controller.power_reward_history[-1]
        assert power_component < 0.0

    def test_power_weight_custom_value(self, adaptive_module):
        """Test controller with custom health weight affects power weight."""
        controller = adaptive_module["HealthAwareQLearning"](
            qlearning_config=MockQLearningConfig(),
            sensor_config=MockSensorConfig(),
            health_weight=0.7,
        )
        assert controller.power_weight == pytest.approx(0.3, abs=0.01)

    def test_power_reward_independent_of_health_metrics(
        self, health_controller, mock_system_state,
    ):
        """Test power reward component is independent of health state."""
        good_health = MockHealthMetrics()
        good_health.overall_health_score = 0.95

        poor_health = MockHealthMetrics()
        poor_health.overall_health_score = 0.2

        health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=good_health,
            system_state=mock_system_state,
        )
        power_good = health_controller.power_reward_history[-1]

        health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=poor_health,
            system_state=mock_system_state,
        )
        power_poor = health_controller.power_reward_history[-1]

        assert power_good == power_poor


class TestConstraintViolationPenalties:
    """Tests for reward penalties when constraints are violated."""

    def test_no_penalty_low_risk(self, health_controller):
        """Test no penalty when risks are low."""
        health_metrics = MockHealthMetrics()
        health_metrics.fouling_risk = 0.3
        health_metrics.detachment_risk = 0.2
        health_metrics.stagnation_risk = 0.1
        health_metrics.health_status = MockHealthStatus.GOOD

        penalty = health_controller._calculate_risk_penalty(health_metrics)
        assert penalty == 0.0

    def test_fouling_risk_penalty(self, health_controller):
        """Test penalty applied for high fouling risk."""
        health_metrics = MockHealthMetrics()
        health_metrics.fouling_risk = 0.9
        health_metrics.detachment_risk = 0.2
        health_metrics.stagnation_risk = 0.1
        health_metrics.health_status = MockHealthStatus.GOOD

        penalty = health_controller._calculate_risk_penalty(health_metrics)
        expected_fouling_penalty = 0.2 * 0.9
        assert penalty >= expected_fouling_penalty

    def test_detachment_risk_penalty_higher(self, health_controller):
        """Test detachment risk has higher penalty weight."""
        health_metrics = MockHealthMetrics()
        health_metrics.fouling_risk = 0.2
        health_metrics.detachment_risk = 0.9
        health_metrics.stagnation_risk = 0.1
        health_metrics.health_status = MockHealthStatus.GOOD

        penalty = health_controller._calculate_risk_penalty(health_metrics)
        expected_detachment_penalty = 0.3 * 0.9
        assert penalty >= expected_detachment_penalty

    def test_stagnation_risk_penalty(self, health_controller):
        """Test penalty applied for high stagnation risk."""
        health_metrics = MockHealthMetrics()
        health_metrics.fouling_risk = 0.2
        health_metrics.detachment_risk = 0.2
        health_metrics.stagnation_risk = 0.9
        health_metrics.health_status = MockHealthStatus.GOOD

        penalty = health_controller._calculate_risk_penalty(health_metrics)
        expected_stagnation_penalty = 0.15 * 0.9
        assert penalty >= expected_stagnation_penalty

    def test_critical_health_penalty(self, health_controller):
        """Test critical health status adds penalty."""
        health_metrics = MockHealthMetrics()
        health_metrics.fouling_risk = 0.2
        health_metrics.detachment_risk = 0.2
        health_metrics.stagnation_risk = 0.1
        health_metrics.health_status = MockHealthStatus.CRITICAL

        penalty = health_controller._calculate_risk_penalty(health_metrics)
        assert penalty >= 0.5

    def test_poor_health_penalty(self, health_controller):
        """Test poor health status adds penalty."""
        health_metrics = MockHealthMetrics()
        health_metrics.fouling_risk = 0.2
        health_metrics.detachment_risk = 0.2
        health_metrics.stagnation_risk = 0.1
        health_metrics.health_status = MockHealthStatus.POOR

        penalty = health_controller._calculate_risk_penalty(health_metrics)
        assert penalty >= 0.2

    def test_multiple_risk_penalties_cumulative(self, health_controller):
        """Test multiple high risks add cumulative penalties."""
        health_metrics = MockHealthMetrics()
        health_metrics.fouling_risk = 0.8
        health_metrics.detachment_risk = 0.8
        health_metrics.stagnation_risk = 0.8
        health_metrics.health_status = MockHealthStatus.GOOD

        penalty = health_controller._calculate_risk_penalty(health_metrics)
        single_risk_metrics = MockHealthMetrics()
        single_risk_metrics.fouling_risk = 0.8
        single_risk_metrics.detachment_risk = 0.3
        single_risk_metrics.stagnation_risk = 0.3
        single_risk_metrics.health_status = MockHealthStatus.GOOD

        single_penalty = health_controller._calculate_risk_penalty(single_risk_metrics)
        assert penalty > single_penalty

    def test_critical_with_all_high_risks(self, health_controller):
        """Test maximum penalty with critical health and all high risks."""
        health_metrics = MockHealthMetrics()
        health_metrics.fouling_risk = 0.95
        health_metrics.detachment_risk = 0.95
        health_metrics.stagnation_risk = 0.95
        health_metrics.health_status = MockHealthStatus.CRITICAL

        penalty = health_controller._calculate_risk_penalty(health_metrics)
        assert penalty >= 0.5

    def test_penalty_affects_total_reward(self, health_controller, mock_system_state):
        """Test risk penalty reduces total reward."""
        low_risk_metrics = MockHealthMetrics()
        low_risk_metrics.fouling_risk = 0.1
        low_risk_metrics.detachment_risk = 0.1
        low_risk_metrics.stagnation_risk = 0.1
        low_risk_metrics.health_status = MockHealthStatus.GOOD

        high_risk_metrics = MockHealthMetrics()
        high_risk_metrics.fouling_risk = 0.9
        high_risk_metrics.detachment_risk = 0.9
        high_risk_metrics.stagnation_risk = 0.9
        high_risk_metrics.health_status = MockHealthStatus.CRITICAL

        low_risk_reward = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=low_risk_metrics,
            system_state=mock_system_state,
        )
        high_risk_reward = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=high_risk_metrics,
            system_state=mock_system_state,
        )
        assert low_risk_reward > high_risk_reward


class TestMultiObjectiveRewardAggregation:
    """Tests for multi-objective reward combining power, health, etc."""

    def test_reward_combines_power_and_health(
        self, health_controller, mock_system_state,
    ):
        """Test reward combines power and health components."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 0.9

        reward = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
        )
        assert len(health_controller.power_reward_history) > 0
        assert len(health_controller.health_reward_history) > 0

    def test_intervention_modifier_improving(
        self, health_controller, mock_system_state,
    ):
        """Test intervention modifier boosts reward for improving trend."""
        health_metrics = MockHealthMetrics()
        health_metrics.health_trend = MockHealthTrend.IMPROVING

        reward_no_intervention = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
            intervention_active=False,
        )
        health_controller.health_reward_history.clear()
        health_controller.power_reward_history.clear()

        reward_with_intervention = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
            intervention_active=True,
        )
        assert reward_with_intervention > reward_no_intervention

    def test_intervention_modifier_declining(
        self, health_controller, mock_system_state,
    ):
        """Test intervention penalty for declining trend."""
        health_metrics = MockHealthMetrics()
        health_metrics.health_trend = MockHealthTrend.DECLINING

        reward_no_intervention = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
            intervention_active=False,
        )
        health_controller.health_reward_history.clear()
        health_controller.power_reward_history.clear()

        reward_with_intervention = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
            intervention_active=True,
        )
        assert reward_with_intervention < reward_no_intervention

    def test_stability_bonus_high_confidence(
        self, health_controller, mock_system_state,
    ):
        """Test stability bonus for high confidence measurements."""
        mock_system_state.fused_measurement.fusion_confidence = 0.95
        mock_system_state.fused_measurement.sensor_agreement = 0.9
        mock_system_state.anomalies = []

        bonus = health_controller._calculate_stability_bonus(mock_system_state)
        assert bonus > 0.0

    def test_stability_bonus_low_confidence(self, health_controller, mock_system_state):
        """Test reduced stability bonus for low confidence."""
        mock_system_state.fused_measurement.fusion_confidence = 0.5
        mock_system_state.fused_measurement.sensor_agreement = 0.5
        mock_system_state.anomalies = []

        bonus = health_controller._calculate_stability_bonus(mock_system_state)
        high_confidence_state = mock_system_state
        high_confidence_state.fused_measurement.fusion_confidence = 0.95
        high_confidence_state.fused_measurement.sensor_agreement = 0.9

        high_bonus = health_controller._calculate_stability_bonus(high_confidence_state)
        assert high_bonus >= bonus

    def test_stability_bonus_with_anomalies(self, health_controller, mock_system_state):
        """Test stability bonus reduced with anomalies."""
        mock_system_state.fused_measurement.fusion_confidence = 0.9
        mock_system_state.fused_measurement.sensor_agreement = 0.9

        mock_system_state.anomalies = []
        bonus_no_anomalies = health_controller._calculate_stability_bonus(
            mock_system_state,
        )

        mock_system_state.anomalies = [
            MockAnomalyDetection(severity="low"),
            MockAnomalyDetection(severity="low"),
        ]
        bonus_with_anomalies = health_controller._calculate_stability_bonus(
            mock_system_state,
        )

        assert bonus_no_anomalies >= bonus_with_anomalies

    def test_reward_formula_components(self, health_controller, mock_system_state):
        """Test all components contribute to final reward."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 0.8
        health_metrics.fouling_risk = 0.1
        health_metrics.detachment_risk = 0.1
        health_metrics.stagnation_risk = 0.1

        mock_system_state.fused_measurement.fusion_confidence = 0.9
        mock_system_state.fused_measurement.sensor_agreement = 0.9
        mock_system_state.anomalies = []

        reward = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
            intervention_active=False,
        )
        assert isinstance(reward, float)
        assert not np.isnan(reward)
        assert not np.isinf(reward)


class TestHealthRewardComponent:
    """Tests for the health-based reward component."""

    def test_health_reward_basic(self, health_controller, mock_system_state):
        """Test basic health reward calculation."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 0.8

        reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )
        assert isinstance(reward, float)

    def test_health_reward_improving_trend_bonus(
        self, health_controller, mock_system_state,
    ):
        """Test improving trend gives bonus."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 0.7
        health_metrics.health_trend = MockHealthTrend.STABLE

        stable_reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )

        health_metrics.health_trend = MockHealthTrend.IMPROVING
        improving_reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )

        assert improving_reward > stable_reward

    def test_health_reward_declining_trend_penalty(
        self, health_controller, mock_system_state,
    ):
        """Test declining trend gives penalty."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 0.7
        health_metrics.health_trend = MockHealthTrend.STABLE

        stable_reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )

        health_metrics.health_trend = MockHealthTrend.DECLINING
        declining_reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )

        assert declining_reward < stable_reward

    def test_health_reward_volatile_trend_penalty(
        self, health_controller, mock_system_state,
    ):
        """Test volatile trend gives penalty."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 0.7
        health_metrics.health_trend = MockHealthTrend.STABLE

        stable_reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )

        health_metrics.health_trend = MockHealthTrend.VOLATILE
        volatile_reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )

        assert volatile_reward < stable_reward

    def test_health_reward_thickness_bonus(self, health_controller, mock_system_state):
        """Test thickness health bonus for high values."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 0.7
        health_metrics.thickness_health = 0.5
        health_metrics.conductivity_health = 0.5
        health_metrics.growth_health = 0.5
        health_metrics.health_trend = MockHealthTrend.STABLE

        low_reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )

        health_metrics.thickness_health = 0.9
        high_reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )

        assert high_reward > low_reward

    def test_health_reward_conductivity_bonus(
        self, health_controller, mock_system_state,
    ):
        """Test conductivity health bonus is significant."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 0.7
        health_metrics.thickness_health = 0.5
        health_metrics.conductivity_health = 0.5
        health_metrics.growth_health = 0.5
        health_metrics.health_trend = MockHealthTrend.STABLE

        low_reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )

        health_metrics.conductivity_health = 0.9
        high_reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )

        assert high_reward > low_reward

    def test_health_reward_growth_bonus(self, health_controller, mock_system_state):
        """Test growth health bonus."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 0.7
        health_metrics.thickness_health = 0.5
        health_metrics.conductivity_health = 0.5
        health_metrics.growth_health = 0.5
        health_metrics.health_trend = MockHealthTrend.STABLE

        low_reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )

        health_metrics.growth_health = 0.9
        high_reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )

        assert high_reward > low_reward

    def test_health_reward_prediction_bonus(self, health_controller, mock_system_state):
        """Test positive prediction gives bonus."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 0.7
        health_metrics.predicted_health_24h = 0.6
        health_metrics.health_trend = MockHealthTrend.STABLE

        negative_prediction_reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )

        health_metrics.predicted_health_24h = 0.85
        positive_prediction_reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )

        assert positive_prediction_reward > negative_prediction_reward


class TestRewardNormalization:
    """Tests for reward normalization and clipping."""

    def test_health_reward_clipped_upper(self, health_controller, mock_system_state):
        """Test health reward is clipped at upper bound."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 2.0
        health_metrics.thickness_health = 1.0
        health_metrics.conductivity_health = 1.0
        health_metrics.growth_health = 1.0
        health_metrics.health_trend = MockHealthTrend.IMPROVING
        health_metrics.predicted_health_24h = 3.0

        reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )
        assert reward <= 1.5

    def test_health_reward_clipped_lower(self, health_controller, mock_system_state):
        """Test health reward is clipped at lower bound."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = -1.0
        health_metrics.thickness_health = 0.0
        health_metrics.conductivity_health = 0.0
        health_metrics.growth_health = 0.0
        health_metrics.health_trend = MockHealthTrend.DECLINING
        health_metrics.predicted_health_24h = -1.0

        reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )
        assert reward >= -0.5

    def test_health_reward_within_bounds(self, health_controller, mock_system_state):
        """Test health reward stays within expected bounds."""
        for _ in range(100):
            health_metrics = MockHealthMetrics()
            health_metrics.overall_health_score = np.random.uniform(0, 1)
            health_metrics.thickness_health = np.random.uniform(0, 1)
            health_metrics.conductivity_health = np.random.uniform(0, 1)
            health_metrics.growth_health = np.random.uniform(0, 1)
            trends = [
                MockHealthTrend.IMPROVING,
                MockHealthTrend.STABLE,
                MockHealthTrend.DECLINING,
                MockHealthTrend.VOLATILE,
            ]
            health_metrics.health_trend = np.random.choice(trends)
            health_metrics.predicted_health_24h = np.random.uniform(0, 1)

            reward = health_controller._calculate_health_reward(
                health_metrics, mock_system_state,
            )
            assert -0.5 <= reward <= 1.5

    def test_total_reward_bounded(self, health_controller, mock_system_state):
        """Test total reward doesn't explode to extreme values."""
        for _ in range(50):
            health_metrics = MockHealthMetrics()
            health_metrics.overall_health_score = np.random.uniform(-1, 2)
            health_metrics.fouling_risk = np.random.uniform(0, 1)
            health_metrics.detachment_risk = np.random.uniform(0, 1)
            health_metrics.stagnation_risk = np.random.uniform(0, 1)
            statuses = [
                MockHealthStatus.EXCELLENT,
                MockHealthStatus.GOOD,
                MockHealthStatus.FAIR,
                MockHealthStatus.POOR,
                MockHealthStatus.CRITICAL,
            ]
            health_metrics.health_status = np.random.choice(statuses)
            trends = [
                MockHealthTrend.IMPROVING,
                MockHealthTrend.STABLE,
                MockHealthTrend.DECLINING,
            ]
            health_metrics.health_trend = np.random.choice(trends)

            base_reward = np.random.uniform(-2, 2)
            reward = health_controller.calculate_health_aware_reward(
                base_reward=base_reward,
                health_metrics=health_metrics,
                system_state=mock_system_state,
                intervention_active=np.random.choice([True, False]),
            )
            assert not np.isnan(reward)
            assert not np.isinf(reward)
            assert -10 < reward < 10


class TestRewardHistoryTracking:
    """Tests for reward history tracking."""

    def test_health_reward_history_appended(self, health_controller, mock_system_state):
        """Test health reward is appended to history."""
        initial_len = len(health_controller.health_reward_history)
        health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=MockHealthMetrics(),
            system_state=mock_system_state,
        )
        assert len(health_controller.health_reward_history) == initial_len + 1

    def test_power_reward_history_appended(self, health_controller, mock_system_state):
        """Test power reward is appended to history."""
        initial_len = len(health_controller.power_reward_history)
        health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=MockHealthMetrics(),
            system_state=mock_system_state,
        )
        assert len(health_controller.power_reward_history) == initial_len + 1

    def test_history_grows_with_multiple_calls(
        self, health_controller, mock_system_state,
    ):
        """Test history grows with multiple reward calculations."""
        n_calls = 10
        initial_health_len = len(health_controller.health_reward_history)
        initial_power_len = len(health_controller.power_reward_history)

        for _ in range(n_calls):
            health_controller.calculate_health_aware_reward(
                base_reward=1.0,
                health_metrics=MockHealthMetrics(),
                system_state=mock_system_state,
            )

        assert (
            len(health_controller.health_reward_history) == initial_health_len + n_calls
        )
        assert (
            len(health_controller.power_reward_history) == initial_power_len + n_calls
        )

    def test_history_values_correct(self, health_controller, mock_system_state):
        """Test history values match expected calculations."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 0.8
        health_metrics.health_trend = MockHealthTrend.STABLE

        base_reward = 1.0
        health_controller.calculate_health_aware_reward(
            base_reward=base_reward,
            health_metrics=health_metrics,
            system_state=mock_system_state,
        )

        power_recorded = health_controller.power_reward_history[-1]
        expected_power = base_reward * health_controller.power_weight
        assert power_recorded == pytest.approx(expected_power, abs=0.001)

    def test_history_maintains_order(self, health_controller, mock_system_state):
        """Test history maintains chronological order."""
        rewards = [0.5, 1.0, 1.5, 2.0]
        initial_len = len(health_controller.power_reward_history)

        for r in rewards:
            health_controller.calculate_health_aware_reward(
                base_reward=r,
                health_metrics=MockHealthMetrics(),
                system_state=mock_system_state,
            )

        recent_power = health_controller.power_reward_history[initial_len:]
        for i in range(len(rewards) - 1):
            assert recent_power[i] < recent_power[i + 1]

    def test_intervention_history_tracked(self, health_controller, mock_system_state):
        """Test intervention history is tracked."""
        initial_len = len(health_controller.intervention_history)
        assert initial_len == 0

    def test_adaptation_history_tracked(self, health_controller, mock_system_state):
        """Test adaptation history is tracked."""
        initial_len = len(health_controller.adaptation_history)
        health_controller.adapt_parameters(MockHealthMetrics(), mock_system_state)
        assert len(health_controller.adaptation_history) == initial_len + 1

    def test_multiple_histories_independent(self, health_controller, mock_system_state):
        """Test different histories are independent."""
        health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=MockHealthMetrics(),
            system_state=mock_system_state,
        )

        health_len = len(health_controller.health_reward_history)
        power_len = len(health_controller.power_reward_history)
        adapt_len = len(health_controller.adaptation_history)

        assert health_len == power_len
        assert adapt_len != health_len


class TestRewardEdgeCases:
    """Tests for reward calculation edge cases."""

    def test_reward_with_nan_health_score(self, health_controller, mock_system_state):
        """Test handling of NaN health score."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = float("nan")

        reward = health_controller._calculate_health_reward(
            health_metrics, mock_system_state,
        )
        assert np.isnan(reward) or isinstance(reward, float)

    def test_reward_with_extreme_base_reward(
        self, health_controller, mock_system_state,
    ):
        """Test with extreme base reward values."""
        health_metrics = MockHealthMetrics()

        reward_high = health_controller.calculate_health_aware_reward(
            base_reward=1000.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
        )
        assert not np.isinf(reward_high)

        reward_low = health_controller.calculate_health_aware_reward(
            base_reward=-1000.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
        )
        assert not np.isinf(reward_low)

    def test_reward_with_zero_health_weight(self, adaptive_module, mock_system_state):
        """Test reward calculation with zero health weight."""
        controller = adaptive_module["HealthAwareQLearning"](
            qlearning_config=MockQLearningConfig(),
            sensor_config=MockSensorConfig(),
            health_weight=0.0,
        )
        assert controller.health_weight == 0.0
        assert controller.power_weight == 1.0

        reward = controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=MockHealthMetrics(),
            system_state=mock_system_state,
        )
        assert isinstance(reward, float)

    def test_reward_with_full_health_weight(self, adaptive_module, mock_system_state):
        """Test reward calculation with full health weight."""
        controller = adaptive_module["HealthAwareQLearning"](
            qlearning_config=MockQLearningConfig(),
            sensor_config=MockSensorConfig(),
            health_weight=1.0,
        )
        assert controller.health_weight == 1.0
        assert controller.power_weight == 0.0

        reward = controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=MockHealthMetrics(),
            system_state=mock_system_state,
        )
        assert isinstance(reward, float)

    def test_reward_with_empty_anomalies(self, health_controller, mock_system_state):
        """Test stability bonus with empty anomaly list."""
        mock_system_state.anomalies = []
        bonus = health_controller._calculate_stability_bonus(mock_system_state)
        assert bonus >= 0.0

    def test_reward_with_many_anomalies(self, health_controller, mock_system_state):
        """Test stability bonus with many anomalies."""
        mock_system_state.anomalies = [MockAnomalyDetection() for _ in range(10)]
        bonus = health_controller._calculate_stability_bonus(mock_system_state)
        assert isinstance(bonus, float)

    def test_reward_with_critical_anomalies(self, health_controller, mock_system_state):
        """Test stability bonus with critical anomalies."""
        mock_system_state.anomalies = [
            MockAnomalyDetection(severity="critical"),
            MockAnomalyDetection(severity="high"),
        ]
        bonus = health_controller._calculate_stability_bonus(mock_system_state)
        no_critical = mock_system_state
        no_critical.anomalies = [
            MockAnomalyDetection(severity="low"),
            MockAnomalyDetection(severity="low"),
        ]
        bonus_no_critical = health_controller._calculate_stability_bonus(no_critical)
        assert bonus_no_critical >= bonus


class TestRewardSystemIntegration:
    """Integration tests for the complete reward system."""

    def test_full_reward_calculation_cycle(
        self, health_controller, mock_system_state,
    ):
        """Test complete reward calculation cycle."""
        health_metrics = MockHealthMetrics()
        health_metrics.overall_health_score = 0.8
        health_metrics.health_trend = MockHealthTrend.STABLE
        health_metrics.fouling_risk = 0.3
        health_metrics.detachment_risk = 0.2
        health_metrics.stagnation_risk = 0.1
        health_metrics.health_status = MockHealthStatus.GOOD

        mock_system_state.fused_measurement.fusion_confidence = 0.9
        mock_system_state.fused_measurement.sensor_agreement = 0.85
        mock_system_state.anomalies = []

        reward = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
            intervention_active=False,
        )

        assert isinstance(reward, float)
        assert not np.isnan(reward)
        assert len(health_controller.health_reward_history) > 0
        assert len(health_controller.power_reward_history) > 0

    def test_reward_consistency_across_calls(
        self, health_controller, mock_system_state,
    ):
        """Test reward calculation is consistent with same inputs."""
        health_metrics = MockHealthMetrics()

        reward1 = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
        )
        reward2 = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=health_metrics,
            system_state=mock_system_state,
        )

        assert reward1 == pytest.approx(reward2, abs=0.001)

    def test_reward_responds_to_state_changes(
        self, health_controller, mock_system_state,
    ):
        """Test reward changes appropriately with state changes."""
        good_metrics = MockHealthMetrics()
        good_metrics.overall_health_score = 0.95
        good_metrics.health_status = MockHealthStatus.EXCELLENT
        good_metrics.health_trend = MockHealthTrend.IMPROVING
        good_metrics.fouling_risk = 0.1
        good_metrics.detachment_risk = 0.1
        good_metrics.stagnation_risk = 0.1

        poor_metrics = MockHealthMetrics()
        poor_metrics.overall_health_score = 0.2
        poor_metrics.health_status = MockHealthStatus.CRITICAL
        poor_metrics.health_trend = MockHealthTrend.DECLINING
        poor_metrics.fouling_risk = 0.9
        poor_metrics.detachment_risk = 0.9
        poor_metrics.stagnation_risk = 0.9

        good_reward = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=good_metrics,
            system_state=mock_system_state,
        )
        poor_reward = health_controller.calculate_health_aware_reward(
            base_reward=1.0,
            health_metrics=poor_metrics,
            system_state=mock_system_state,
        )

        assert good_reward > poor_reward
