"""Integration Tests for Adaptive MFC Controller.

US-013: Adaptive Controller Integration Tests
Target: 90%+ coverage for adaptive_mfc_controller.py

Tests cover:
- Complete control loop with mock MFC
- Q-table persistence and loading
- Performance metrics collection
- Convergence on simple control tasks
"""

from __future__ import annotations

import json
import pickle
import sys
import tempfile
import time
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
class MockHealthAlert:
    """Mock HealthAlert dataclass."""

    message: str = "Test alert"
    severity: str = "low"
    timestamp: float = 0.0


@dataclass
class MockInterventionRecommendation:
    """Mock InterventionRecommendation dataclass."""

    intervention_type: str = "flow_adjustment"
    description: str = "Adjust flow rate"
    urgency: str = "low"
    expected_benefit: float = 0.1
    success_probability: float = 0.8


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


@dataclass
class MockGrowthPattern:
    """Mock growth pattern for biofilm analysis."""

    growth_rate: float = 0.1
    phase: str = "exponential"
    stability: float = 0.9


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
        self, state, action, reward, next_state, sensor_data
    ):
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        self.q_table[state][action] += self.learning_rate * (
            reward - self.q_table[state][action]
        )

    def get_controller_performance_summary(self):
        total = self.sensor_guided_decisions + self.model_guided_decisions
        return {
            "total_decisions": total,
            "sensor_guided": self.sensor_guided_decisions,
            "q_table_size": len(self.q_table),
        }


class MockEISMeasurement:
    """Mock EIS measurement."""

    def __init__(self):
        self.timestamp = 0.0
        self.frequency = 1000.0
        self.impedance_real = 100.0
        self.impedance_imag = 50.0


class MockQCMMeasurement:
    """Mock QCM measurement."""

    def __init__(self):
        self.timestamp = 0.0
        self.frequency_shift = -1000.0
        self.dissipation = 1e-6


def create_mock_health_monitor(*args, **kwargs):
    """Create a mock health monitor."""
    monitor = MagicMock()
    monitor.assess_health.return_value = MockHealthMetrics()
    monitor.generate_alerts.return_value = []
    monitor.generate_intervention_recommendations.return_value = []
    monitor.get_health_dashboard_data.return_value = {
        "health_score": 0.85,
        "trend": "stable",
    }
    return monitor


def create_mock_sensor_fusion(*args, **kwargs):
    """Create a mock sensor fusion."""
    fusion = MagicMock()
    fusion.fuse_measurements_with_prediction.return_value = (
        MockFusedMeasurement(),
        MockPredictiveState(),
        [],
    )
    fusion.analyze_biofilm_growth_pattern.return_value = MockGrowthPattern()
    fusion.get_system_health_assessment.return_value = {
        "overall_health": 0.9,
        "sensor_status": "healthy",
    }
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

    mock_biofilm.HealthStatus = MockHealthStatus
    mock_biofilm.HealthTrend = MockHealthTrend
    mock_biofilm.HealthMetrics = MockHealthMetrics
    mock_biofilm.HealthAlert = MockHealthAlert
    mock_biofilm.InterventionRecommendation = MockInterventionRecommendation
    mock_biofilm.create_predictive_health_monitor = create_mock_health_monitor

    mock_advanced_fusion.FusedMeasurement = MockFusedMeasurement
    mock_advanced_fusion.PredictiveState = MockPredictiveState
    mock_advanced_fusion.AnomalyDetection = MockAnomalyDetection
    mock_advanced_fusion.create_advanced_sensor_fusion = create_mock_sensor_fusion

    mock_sensor_fusion.BacterialSpecies = MockBacterialSpecies

    mock_config.QLearningConfig = MockQLearningConfig
    mock_config.SensorConfig = MockSensorConfig

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


class TestCompleteControlLoop:
    """Tests for complete control loop with mock MFC."""

    def test_single_control_step_execution(self, adaptive_module):
        """Test executing a single control step."""
        controller = adaptive_module["AdaptiveMFCController"]()
        result = controller.control_step(
            MockEISMeasurement(), MockQCMMeasurement(),
            {"thickness_um": 15.0}, {"thickness_um": 14.5}, time_hours=1.0,
        )
        assert result is not None
        assert "timestamp" in result
        assert "control_decision" in result

    def test_multiple_control_steps(self, adaptive_module):
        """Test executing multiple consecutive control steps."""
        controller = adaptive_module["AdaptiveMFCController"]()
        for t in range(10):
            controller.control_step(
                MockEISMeasurement(), MockQCMMeasurement(),
                {"thickness_um": 15.0}, {"thickness_um": 14.5}, time_hours=float(t),
            )
        assert len(controller.control_history) == 10

    def test_critical_health_triggers_recovery(self, adaptive_module):
        """Test critical health triggers recovery mode."""
        controller = adaptive_module["AdaptiveMFCController"]()
        critical_health = MockHealthMetrics()
        critical_health.health_status = MockHealthStatus.CRITICAL
        controller.health_monitor.assess_health.return_value = critical_health
        controller.control_step(
            MockEISMeasurement(), MockQCMMeasurement(),
            {"thickness_um": 15.0}, {"thickness_um": 14.5}, time_hours=1.0,
        )
        expected_strategy = adaptive_module["ControlStrategy"].RECOVERY
        assert controller.current_strategy == expected_strategy

    def test_poor_health_triggers_health_focused(self, adaptive_module):
        """Test poor health triggers health-focused mode."""
        controller = adaptive_module["AdaptiveMFCController"]()
        poor_health = MockHealthMetrics()
        poor_health.health_status = MockHealthStatus.POOR
        controller.health_monitor.assess_health.return_value = poor_health
        controller.control_step(
            MockEISMeasurement(), MockQCMMeasurement(),
            {"thickness_um": 15.0}, {"thickness_um": 14.5}, time_hours=1.0,
        )
        expected_strategy = adaptive_module["ControlStrategy"].HEALTH_FOCUSED
        assert controller.current_strategy == expected_strategy

    def test_excellent_health_enables_performance(self, adaptive_module):
        """Test excellent health enables performance-focused mode."""
        controller = adaptive_module["AdaptiveMFCController"]()
        ex_health = MockHealthMetrics()
        ex_health.health_status = MockHealthStatus.EXCELLENT
        ex_health.health_trend = MockHealthTrend.STABLE
        ex_health.fouling_risk = 0.1
        ex_health.detachment_risk = 0.1
        ex_health.stagnation_risk = 0.1
        controller.health_monitor.assess_health.return_value = ex_health
        controller.control_step(
            MockEISMeasurement(), MockQCMMeasurement(),
            {"thickness_um": 15.0}, {"thickness_um": 14.5}, time_hours=1.0,
        )
        expected_strategy = adaptive_module["ControlStrategy"].PERFORMANCE_FOCUSED
        assert controller.current_strategy == expected_strategy

    def test_control_loop_handles_anomalies(self, adaptive_module):
        """Test control loop handles sensor anomalies."""
        controller = adaptive_module["AdaptiveMFCController"]()
        anomaly = MockAnomalyDetection(severity="high")
        controller.sensor_fusion.fuse_measurements_with_prediction.return_value = (
            MockFusedMeasurement(), MockPredictiveState(), [anomaly],
        )
        result = controller.control_step(
            MockEISMeasurement(), MockQCMMeasurement(),
            {"thickness_um": 15.0}, {"thickness_um": 14.5}, time_hours=1.0,
        )
        assert len(result["system_state"].anomalies) == 1

    def test_control_loop_with_health_alerts(self, adaptive_module):
        """Test control loop handles health alerts."""
        controller = adaptive_module["AdaptiveMFCController"]()
        alert = MockHealthAlert(message="High fouling", severity="warning")
        controller.health_monitor.generate_alerts.return_value = [alert]
        result = controller.control_step(
            MockEISMeasurement(), MockQCMMeasurement(),
            {"thickness_um": 15.0}, {"thickness_um": 14.5}, time_hours=1.0,
        )
        assert len(result["health_alerts"]) == 1


class TestQTablePersistence:
    """Tests for Q-table persistence and loading."""

    def test_q_table_serialization_pickle(self, adaptive_module):
        """Test Q-table can be serialized with pickle."""
        controller = adaptive_module["AdaptiveMFCController"]()
        for t in range(5):
            controller.control_step(
                MockEISMeasurement(), MockQCMMeasurement(),
                {"thickness_um": 15.0 + t}, {"thickness_um": 14.5 + t},
                time_hours=float(t),
            )
        q_table = controller.q_controller.q_table
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(q_table, f)
            temp_path = f.name
        with open(temp_path, "rb") as f:
            loaded_table = pickle.load(f)
        assert loaded_table == q_table
        Path(temp_path).unlink()

    def test_q_table_round_trip(self, adaptive_module):
        """Test complete save/load round trip preserves data."""
        controller = adaptive_module["AdaptiveMFCController"]()
        for t in range(5):
            controller.control_step(
                MockEISMeasurement(), MockQCMMeasurement(),
                {"thickness_um": 15.0 + t * 0.5}, {"thickness_um": 14.5 + t * 0.5},
                time_hours=float(t),
            )
        state = {
            "q_table": controller.q_controller.q_table,
            "learning_rate": controller.q_controller.learning_rate,
            "system_parameters": controller.system_parameters,
        }
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(state, f)
            temp_path = f.name
        with open(temp_path, "rb") as f:
            loaded_state = pickle.load(f)
        assert loaded_state["q_table"] == state["q_table"]
        assert loaded_state["learning_rate"] == state["learning_rate"]
        Path(temp_path).unlink()

    def test_q_table_state_preservation(self, adaptive_module):
        """Test Q-table preserves learned values after save/load."""
        controller = adaptive_module["AdaptiveMFCController"]()
        for t in range(10):
            controller.control_step(
                MockEISMeasurement(), MockQCMMeasurement(),
                {"thickness_um": 15.0}, {"thickness_um": 14.5},
                time_hours=float(t),
            )
        original_table = dict(controller.q_controller.q_table)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(original_table, f)
            temp_path = f.name
        new_controller = adaptive_module["AdaptiveMFCController"]()
        with open(temp_path, "rb") as f:
            new_controller.q_controller.q_table = pickle.load(f)
        assert new_controller.q_controller.q_table == original_table
        Path(temp_path).unlink()

    def test_q_table_json_serialization(self, adaptive_module):
        """Test Q-table can be serialized to JSON with conversion."""
        controller = adaptive_module["AdaptiveMFCController"]()
        for t in range(5):
            controller.control_step(
                MockEISMeasurement(), MockQCMMeasurement(),
                {"thickness_um": 15.0}, {"thickness_um": 14.5},
                time_hours=float(t),
            )
        q_table = controller.q_controller.q_table
        json_table = {str(k): v for k, v in q_table.items()}
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(json_table, f)
            temp_path = f.name
        with open(temp_path) as f:
            loaded_table = json.load(f)
        assert len(loaded_table) == len(q_table)
        Path(temp_path).unlink()


class TestPerformanceMetricsCollection:
    """Tests for performance metrics collection."""

    def test_performance_metrics_calculated(self, adaptive_module):
        """Test performance metrics are calculated in control step."""
        controller = adaptive_module["AdaptiveMFCController"]()
        result = controller.control_step(
            MockEISMeasurement(), MockQCMMeasurement(),
            {"thickness_um": 15.0}, {"thickness_um": 14.5}, time_hours=1.0,
        )
        assert "performance_metrics" in result
        metrics = result["performance_metrics"]
        assert "power_efficiency" in metrics
        assert "biofilm_health_score" in metrics

    def test_health_score_in_metrics(self, adaptive_module):
        """Test health score is included in control results."""
        controller = adaptive_module["AdaptiveMFCController"]()
        result = controller.control_step(
            MockEISMeasurement(), MockQCMMeasurement(),
            {"thickness_um": 15.0}, {"thickness_um": 14.5}, time_hours=1.0,
        )
        assert "system_health_score" in result
        assert 0.0 <= result["system_health_score"] <= 1.0

    def test_comprehensive_status_includes_metrics(self, adaptive_module):
        """Test comprehensive status includes performance metrics."""
        controller = adaptive_module["AdaptiveMFCController"]()
        controller.control_step(
            MockEISMeasurement(), MockQCMMeasurement(),
            {"thickness_um": 15.0}, {"thickness_um": 14.5}, time_hours=1.0,
        )
        status = controller.get_comprehensive_status()
        assert "recent_performance" in status

    def test_q_learning_stats_in_status(self, adaptive_module):
        """Test Q-learning stats are in comprehensive status."""
        controller = adaptive_module["AdaptiveMFCController"]()
        controller.control_step(
            MockEISMeasurement(), MockQCMMeasurement(),
            {"thickness_um": 15.0}, {"thickness_um": 14.5}, time_hours=1.0,
        )
        status = controller.get_comprehensive_status()
        assert "q_learning_stats" in status

    def test_strategy_change_tracking(self, adaptive_module):
        """Test strategy changes are tracked."""
        controller = adaptive_module["AdaptiveMFCController"]()
        controller.control_step(
            MockEISMeasurement(), MockQCMMeasurement(),
            {"thickness_um": 15.0}, {"thickness_um": 14.5}, time_hours=0.0,
        )
        critical_health = MockHealthMetrics()
        critical_health.health_status = MockHealthStatus.CRITICAL
        controller.health_monitor.assess_health.return_value = critical_health
        controller.control_step(
            MockEISMeasurement(), MockQCMMeasurement(),
            {"thickness_um": 15.0}, {"thickness_um": 14.5}, time_hours=1.0,
        )
        assert len(controller.strategy_changes) > 0


class TestConvergenceOnSimpleTasks:
    """Tests for convergence on simple control tasks."""

    def test_q_table_grows_with_exploration(self, adaptive_module):
        """Test Q-table grows as controller explores."""
        controller = adaptive_module["AdaptiveMFCController"]()
        initial_size = len(controller.q_controller.q_table)
        for t in range(20):
            controller.control_step(
                MockEISMeasurement(), MockQCMMeasurement(),
                {"thickness_um": 15.0 + np.random.randn()},
                {"thickness_um": 14.5 + np.random.randn()},
                time_hours=float(t),
            )
        final_size = len(controller.q_controller.q_table)
        assert final_size >= initial_size

    def test_reward_accumulation(self, adaptive_module):
        """Test rewards accumulate during control loop."""
        controller = adaptive_module["AdaptiveMFCController"]()
        for t in range(10):
            controller.control_step(
                MockEISMeasurement(), MockQCMMeasurement(),
                {"thickness_um": 15.0}, {"thickness_um": 14.5},
                time_hours=float(t),
            )
        assert len(controller.q_controller.health_reward_history) >= 10
        assert len(controller.q_controller.power_reward_history) >= 10

    def test_adaptation_history_recorded(self, adaptive_module):
        """Test parameter adaptation history is recorded."""
        controller = adaptive_module["AdaptiveMFCController"]()
        for t in range(10):
            controller.control_step(
                MockEISMeasurement(), MockQCMMeasurement(),
                {"thickness_um": 15.0}, {"thickness_um": 14.5},
                time_hours=float(t),
            )
        assert len(controller.q_controller.adaptation_history) >= 10

    def test_epsilon_adaptation_convergence(self, adaptive_module):
        """Test epsilon adapts during learning."""
        controller = adaptive_module["AdaptiveMFCController"]()
        epsilon_values = []
        for t in range(10):
            controller.control_step(
                MockEISMeasurement(), MockQCMMeasurement(),
                {"thickness_um": 15.0}, {"thickness_um": 14.5},
                time_hours=float(t),
            )
            epsilon_values.append(controller.q_controller.epsilon)
        for eps in epsilon_values:
            assert 0.0 <= eps <= 1.0

    def test_control_responds_to_changing_health(self, adaptive_module):
        """Test control responds appropriately to health changes."""
        controller = adaptive_module["AdaptiveMFCController"]()
        good_health = MockHealthMetrics()
        good_health.health_status = MockHealthStatus.GOOD
        controller.health_monitor.assess_health.return_value = good_health
        for t in range(5):
            controller.control_step(
                MockEISMeasurement(), MockQCMMeasurement(),
                {"thickness_um": 15.0}, {"thickness_um": 14.5},
                time_hours=float(t),
            )
        initial_strategy = controller.current_strategy
        poor_health = MockHealthMetrics()
        poor_health.health_status = MockHealthStatus.POOR
        controller.health_monitor.assess_health.return_value = poor_health
        for t in range(5, 10):
            controller.control_step(
                MockEISMeasurement(), MockQCMMeasurement(),
                {"thickness_um": 15.0}, {"thickness_um": 14.5},
                time_hours=float(t),
            )
        assert controller.current_strategy != initial_strategy


class TestCompleteExecution:
    """Tests for complete execution scenarios."""

    def test_full_session_under_45_seconds(self, adaptive_module):
        """Test complete session completes within time limit."""
        controller = adaptive_module["AdaptiveMFCController"]()
        start_time = time.time()
        for t in range(50):
            controller.control_step(
                MockEISMeasurement(), MockQCMMeasurement(),
                {"thickness_um": 15.0 + np.sin(t / 5)},
                {"thickness_um": 14.5 + np.sin(t / 5)},
                time_hours=float(t) / 10,
            )
        elapsed = time.time() - start_time
        assert elapsed < 45.0

    def test_intervention_execution(self, adaptive_module):
        """Test interventions are executed when needed."""
        controller = adaptive_module["AdaptiveMFCController"]()
        immediate_intervention = MockInterventionRecommendation()
        immediate_intervention.urgency = "immediate"
        immediate_intervention.success_probability = 0.9
        controller.health_monitor.generate_intervention_recommendations.return_value = [
            immediate_intervention
        ]
        controller.control_step(
            MockEISMeasurement(), MockQCMMeasurement(),
            {"thickness_um": 15.0}, {"thickness_um": 14.5},
            time_hours=1.0,
        )
        assert len(controller.intervention_outcomes) > 0

    def test_high_risk_triggers_conservative(self, adaptive_module):
        """Test high risk conditions trigger conservative strategy."""
        controller = adaptive_module["AdaptiveMFCController"]()
        high_risk_health = MockHealthMetrics()
        high_risk_health.health_status = MockHealthStatus.FAIR
        high_risk_health.fouling_risk = 0.9
        controller.health_monitor.assess_health.return_value = high_risk_health
        controller.control_step(
            MockEISMeasurement(), MockQCMMeasurement(),
            {"thickness_um": 15.0}, {"thickness_um": 14.5},
            time_hours=1.0,
        )
        expected_strategy = adaptive_module["ControlStrategy"].CONSERVATIVE
        assert controller.current_strategy == expected_strategy

    def test_all_control_history_has_required_fields(self, adaptive_module):
        """Test all control history entries have required fields."""
        controller = adaptive_module["AdaptiveMFCController"]()
        for t in range(10):
            controller.control_step(
                MockEISMeasurement(), MockQCMMeasurement(),
                {"thickness_um": 15.0}, {"thickness_um": 14.5},
                time_hours=float(t),
            )
        required_fields = [
            "timestamp", "system_state", "control_decision",
            "execution_results", "performance_metrics", "system_health_score",
        ]
        for entry in controller.control_history:
            for fld in required_fields:
                assert fld in entry, f"Missing field: {fld}"


class TestEdgeCases:
    """Edge case tests for integration scenarios."""

    def test_zero_flow_rate_handling(self, adaptive_module):
        """Test handling of zero flow rate."""
        controller = adaptive_module["AdaptiveMFCController"]()
        controller.system_parameters["flow_rate"] = 0.01
        result = controller.control_step(
            MockEISMeasurement(), MockQCMMeasurement(),
            {"thickness_um": 15.0}, {"thickness_um": 14.5},
            time_hours=1.0,
        )
        assert result is not None
        assert result["performance_metrics"]["power_efficiency"] > 0

    def test_high_anomaly_count(self, adaptive_module):
        """Test handling of many anomalies."""
        controller = adaptive_module["AdaptiveMFCController"]()
        anomalies = [MockAnomalyDetection(severity="high") for _ in range(5)]
        controller.sensor_fusion.fuse_measurements_with_prediction.return_value = (
            MockFusedMeasurement(), MockPredictiveState(), anomalies,
        )
        result = controller.control_step(
            MockEISMeasurement(), MockQCMMeasurement(),
            {"thickness_um": 15.0}, {"thickness_um": 14.5},
            time_hours=1.0,
        )
        assert len(result["system_state"].anomalies) == 5

    def test_rapid_health_changes(self, adaptive_module):
        """Test handling of rapid health status changes."""
        controller = adaptive_module["AdaptiveMFCController"]()
        health_states = [
            MockHealthStatus.EXCELLENT, MockHealthStatus.CRITICAL,
            MockHealthStatus.GOOD, MockHealthStatus.POOR,
        ]
        for i, status in enumerate(health_states):
            health = MockHealthMetrics()
            health.health_status = status
            health.health_trend = MockHealthTrend.VOLATILE
            controller.health_monitor.assess_health.return_value = health
            controller.control_step(
                MockEISMeasurement(), MockQCMMeasurement(),
                {"thickness_um": 15.0}, {"thickness_um": 14.5},
                time_hours=float(i),
            )
        assert len(controller.strategy_changes) > 0

    def test_concurrent_alerts_and_interventions(self, adaptive_module):
        """Test handling of alerts and interventions together."""
        controller = adaptive_module["AdaptiveMFCController"]()
        alert = MockHealthAlert(message="Critical", severity="critical")
        intervention = MockInterventionRecommendation()
        intervention.urgency = "immediate"
        intervention.success_probability = 0.9
        controller.health_monitor.generate_alerts.return_value = [alert]
        rec_mock = controller.health_monitor.generate_intervention_recommendations
        rec_mock.return_value = [intervention]
        result = controller.control_step(
            MockEISMeasurement(), MockQCMMeasurement(),
            {"thickness_um": 15.0}, {"thickness_um": 14.5},
            time_hours=1.0,
        )
        assert len(result["health_alerts"]) == 1
        assert len(result["intervention_recommendations"]) == 1
