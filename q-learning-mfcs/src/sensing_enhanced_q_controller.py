"""
Sensing-Enhanced Q-Learning Controller

This module implements an advanced Q-learning controller that incorporates
EIS and QCM sensor data for improved biofilm monitoring and control decisions.

Key Features:
- Extended state space with sensor measurements
- Sensor-guided reward function
- Adaptive action selection based on sensor confidence
- Multi-objective optimization (power, biofilm health, sensor agreement)
- Fault-tolerant control with sensor degradation handling

The controller uses real-time sensor feedback to make more informed
control decisions, leading to better biofilm management and MFC performance.
"""

import os
import sys
from typing import Any

import numpy as np

# Import configuration classes
from config import (
    QLearningConfig,
    SensorConfig,
    validate_qlearning_config,
    validate_sensor_config,
)

# Import base controller
sys.path.append(os.path.dirname(__file__))
from mfc_recirculation_control import AdvancedQLearningFlowController

# Import sensing models
sys.path.append(os.path.join(os.path.dirname(__file__), 'sensing_models'))
try:
    from sensing_models.eis_model import EISModel
    from sensing_models.qcm_model import QCMModel
    from sensing_models.sensor_fusion import SensorFusion
except ImportError as e:
    print(f"Warning: Sensing models not available: {e}")
    EISModel = QCMModel = SensorFusion = None

# Import GPU acceleration
try:
    from gpu_acceleration import get_gpu_accelerator
except ImportError:
    get_gpu_accelerator = None


class SensingEnhancedQLearningController(AdvancedQLearningFlowController):
    """
    Enhanced Q-learning controller with integrated EIS/QCM sensor feedback.

    Extends the base Q-learning controller with:
    - Sensor-informed state representation
    - Multi-objective reward function
    - Adaptive exploration based on sensor confidence
    - Sensor fault handling and degradation compensation
    """

    def __init__(self, qlearning_config: QLearningConfig | None = None,
                 sensor_config: SensorConfig | None = None,
                 enable_sensor_state: bool = True, fault_tolerance: bool = True):
        """
        Initialize sensing-enhanced Q-learning controller.

        Args:
            qlearning_config: Q-learning configuration parameters
            sensor_config: Sensor configuration parameters
            enable_sensor_state: Include sensor data in state representation
            fault_tolerance: Enable fault-tolerant operation
        """
        # Use default configurations if not provided
        if qlearning_config is None:
            qlearning_config = QLearningConfig()
        if sensor_config is None:
            sensor_config = SensorConfig()

        # Validate configurations
        validate_qlearning_config(qlearning_config)
        validate_sensor_config(sensor_config)

        # Store configurations
        self.qlearning_config = qlearning_config
        self.sensor_config = sensor_config

        # Initialize base controller with enhanced parameters
        super().__init__(
            qlearning_config.enhanced_learning_rate,
            qlearning_config.enhanced_discount_factor,
            qlearning_config.enhanced_epsilon
        )

        # Sensor configuration
        self.enable_sensor_state = enable_sensor_state
        self.sensor_weight = qlearning_config.sensor_weight
        self.fault_tolerance = fault_tolerance

        # Enhanced state space setup
        self.setup_sensor_enhanced_state_space()

        # Multi-objective parameters from config
        self.power_weight = qlearning_config.power_objective_weight
        self.biofilm_health_weight = qlearning_config.biofilm_health_weight
        self.sensor_agreement_weight = qlearning_config.sensor_agreement_weight
        self.stability_weight = qlearning_config.stability_weight

        # Sensor confidence tracking
        self.sensor_confidence_history = []
        self.sensor_fault_count = 0
        self.sensor_degradation_factor = 1.0

        # Adaptive exploration parameters from config
        self.min_sensor_confidence = qlearning_config.sensor_confidence_threshold
        self.exploration_boost_factor = qlearning_config.exploration_boost_factor

        # Performance tracking
        self.sensor_guided_decisions = 0
        self.model_guided_decisions = 0
        self.total_reward_components = {
            'power': 0.0,
            'biofilm_health': 0.0,
            'sensor_agreement': 0.0,
            'stability': 0.0
        }

        # State prediction and validation
        self.state_predictions = []
        self.sensor_validations = []
        self.prediction_errors = []

    def setup_sensor_enhanced_state_space(self):
        """Setup enhanced state space including sensor measurements."""
        # Call parent setup first
        super().setup_enhanced_state_action_spaces()

        if self.enable_sensor_state:
            # Additional sensor-based state variables using configuration
            # Get state space config
            state_config = self.qlearning_config.state_space

            # EIS-related states
            self.eis_thickness_bins = np.linspace(
                0, state_config.eis_thickness_max,
                state_config.eis_thickness_bins
            )
            self.eis_conductivity_bins = np.linspace(
                0, state_config.eis_conductivity_max,
                state_config.eis_conductivity_bins
            )
            self.eis_quality_bins = np.linspace(
                0, 1, state_config.eis_confidence_bins
            )

            # QCM-related states
            self.qcm_mass_bins = np.linspace(
                0, state_config.qcm_mass_max,
                state_config.qcm_mass_bins
            )
            self.qcm_frequency_shift_bins = np.linspace(
                0, state_config.qcm_frequency_max,
                state_config.qcm_frequency_bins
            )
            self.qcm_dissipation_bins = np.linspace(
                0, state_config.qcm_dissipation_max,
                state_config.qcm_dissipation_bins
            )

            # Sensor fusion states
            self.sensor_agreement_bins = np.linspace(
                0, 1, state_config.sensor_agreement_bins
            )
            self.fusion_confidence_bins = np.linspace(
                0, 1, state_config.fusion_confidence_bins
            )
            self.sensor_status_bins = np.array([0, 1, 2, 3])  # good, degraded, failed, unavailable

            print("Sensor-enhanced state space initialized")
        else:
            print("Using base state space without sensor integration")

    def discretize_sensor_enhanced_state(self, base_state: tuple, sensor_data: dict | None = None) -> tuple:
        """
        Convert continuous state to discrete state including sensor data.

        Args:
            base_state: Base state tuple from parent class
            sensor_data: Optional sensor measurement data

        Returns:
            Enhanced discrete state tuple
        """
        if not self.enable_sensor_state or sensor_data is None:
            return base_state

        # Extract sensor information
        eis_data = sensor_data.get('eis', {})
        qcm_data = sensor_data.get('qcm', {})
        fusion_data = sensor_data.get('fusion', {})

        # EIS state discretization
        eis_thickness = eis_data.get('thickness_um', 0.0)
        eis_conductivity = eis_data.get('conductivity_S_per_m', 0.0)
        eis_quality = eis_data.get('measurement_quality', 0.0)

        eis_thick_idx = np.clip(np.digitize(eis_thickness, self.eis_thickness_bins) - 1,
                               0, len(self.eis_thickness_bins) - 2)
        eis_cond_idx = np.clip(np.digitize(eis_conductivity, self.eis_conductivity_bins) - 1,
                              0, len(self.eis_conductivity_bins) - 2)
        eis_qual_idx = np.clip(np.digitize(eis_quality, self.eis_quality_bins) - 1,
                              0, len(self.eis_quality_bins) - 2)

        # QCM state discretization
        qcm_mass = qcm_data.get('mass_per_area_ng_per_cm2', 0.0)
        qcm_freq_shift = abs(qcm_data.get('frequency_shift_Hz', 0.0))
        qcm_dissipation = qcm_data.get('dissipation', 0.0)

        qcm_mass_idx = np.clip(np.digitize(qcm_mass, self.qcm_mass_bins) - 1,
                              0, len(self.qcm_mass_bins) - 2)
        qcm_freq_idx = np.clip(np.digitize(qcm_freq_shift, self.qcm_frequency_shift_bins) - 1,
                              0, len(self.qcm_frequency_shift_bins) - 2)
        qcm_diss_idx = np.clip(np.digitize(qcm_dissipation, self.qcm_dissipation_bins) - 1,
                              0, len(self.qcm_dissipation_bins) - 2)

        # Sensor fusion state discretization
        sensor_agreement = fusion_data.get('sensor_agreement', 0.5)
        fusion_confidence = fusion_data.get('fusion_confidence', 0.5)
        sensor_status = self._encode_sensor_status(
            eis_data.get('status', 'unavailable'),
            qcm_data.get('status', 'unavailable')
        )

        agreement_idx = np.clip(np.digitize(sensor_agreement, self.sensor_agreement_bins) - 1,
                               0, len(self.sensor_agreement_bins) - 2)
        confidence_idx = np.clip(np.digitize(fusion_confidence, self.fusion_confidence_bins) - 1,
                                0, len(self.fusion_confidence_bins) - 2)
        status_idx = np.clip(sensor_status, 0, len(self.sensor_status_bins) - 1)

        # Combine base state with sensor state
        enhanced_state = base_state + (
            eis_thick_idx, eis_cond_idx, eis_qual_idx,
            qcm_mass_idx, qcm_freq_idx, qcm_diss_idx,
            agreement_idx, confidence_idx, status_idx
        )

        return enhanced_state

    def _encode_sensor_status(self, eis_status: str, qcm_status: str) -> int:
        """Encode sensor status into discrete value."""
        status_map = {'good': 3, 'degraded': 2, 'failed': 1, 'unavailable': 0}

        eis_score = status_map.get(eis_status, 0)
        qcm_score = status_map.get(qcm_status, 0)

        # Combined status (average)
        combined_score = (eis_score + qcm_score) / 2

        if combined_score >= 2.5:
            return 3  # good
        elif combined_score >= 1.5:
            return 2  # degraded
        elif combined_score >= 0.5:
            return 1  # failed
        else:
            return 0  # unavailable

    def choose_action_with_sensors(self, base_state: tuple, sensor_data: dict | None = None,
                                 available_actions: list | None = None) -> int:
        """
        Choose action using sensor-enhanced state and adaptive exploration.

        Args:
            base_state: Base Q-learning state
            sensor_data: Sensor measurement data
            available_actions: Available actions (if constrained)

        Returns:
            Selected action index
        """
        # Get enhanced state
        enhanced_state = self.discretize_sensor_enhanced_state(base_state, sensor_data)

        # Calculate sensor confidence
        sensor_confidence = self._calculate_sensor_confidence(sensor_data)
        self.sensor_confidence_history.append(sensor_confidence)

        # Adaptive exploration based on sensor confidence
        effective_epsilon = self._calculate_adaptive_epsilon(sensor_confidence)

        # Choose action with adaptive exploration
        if np.random.random() < effective_epsilon:
            # Exploration: choose random action
            if available_actions:
                action_idx = np.random.choice(len(available_actions))
            else:
                action_idx = np.random.choice(len(self.actions))
        else:
            # Exploitation: choose best action based on Q-values
            if available_actions:
                # Evaluate Q-values for available actions only
                action_idx = self._choose_best_available_action(enhanced_state, available_actions)
            else:
                # Standard Q-value maximization
                q_values = [self.q_table[enhanced_state][a] for a in range(len(self.actions))]
                action_idx = np.argmax(q_values)

        # Track decision type
        if sensor_confidence > self.min_sensor_confidence:
            self.sensor_guided_decisions += 1
        else:
            self.model_guided_decisions += 1

        return action_idx

    def _calculate_sensor_confidence(self, sensor_data: dict | None) -> float:
        """Calculate overall sensor confidence."""
        if sensor_data is None:
            return 0.0

        confidences = []

        # EIS confidence
        eis_data = sensor_data.get('eis', {})
        if eis_data:
            eis_quality = eis_data.get('measurement_quality', 0.0)
            eis_status = eis_data.get('status', 'unavailable')
            status_factor = self.sensor_config.fusion.status_scores.copy()
            status_factor['unavailable'] = 0.0
            eis_confidence = eis_quality * status_factor.get(eis_status, 0.0)
            confidences.append(eis_confidence)

        # QCM confidence
        qcm_data = sensor_data.get('qcm', {})
        if qcm_data:
            qcm_quality = qcm_data.get('measurement_quality', 0.0)
            qcm_status = qcm_data.get('status', 'unavailable')
            status_factor = self.sensor_config.fusion.status_scores.copy()
            status_factor['unavailable'] = 0.0
            qcm_confidence = qcm_quality * status_factor.get(qcm_status, 0.0)
            confidences.append(qcm_confidence)

        # Fusion confidence
        fusion_data = sensor_data.get('fusion', {})
        if fusion_data:
            fusion_confidence = fusion_data.get('fusion_confidence', 0.0)
            confidences.append(fusion_confidence)

        # Return weighted average confidence
        if confidences:
            return np.mean(confidences) * self.sensor_degradation_factor
        else:
            return 0.0

    def _calculate_adaptive_epsilon(self, sensor_confidence: float) -> float:
        """Calculate adaptive exploration rate based on sensor confidence."""
        base_epsilon = self.epsilon

        if sensor_confidence < self.min_sensor_confidence:
            # Boost exploration when sensors are unreliable
            adaptive_epsilon = min(1.0, base_epsilon * self.exploration_boost_factor)
        else:
            # Normal exploration with sensor guidance
            confidence_factor = (1.0 - sensor_confidence) * 0.5  # 0-0.5 range
            adaptive_epsilon = base_epsilon * (1.0 + confidence_factor)

        return adaptive_epsilon

    def _choose_best_available_action(self, state: tuple, available_actions: list) -> int:
        """Choose best action from available actions only."""
        if not available_actions:
            return 0

        # Get Q-values for available actions
        q_values = []
        for action_idx in available_actions:
            if action_idx < len(self.actions):
                q_value = self.q_table[state][action_idx]
                q_values.append((q_value, action_idx))

        if q_values:
            # Return action index with highest Q-value
            best_q_value, best_action_idx = max(q_values, key=lambda x: x[0])
            return available_actions.index(best_action_idx)
        else:
            return 0

    def calculate_sensor_enhanced_reward(self, base_reward: float, sensor_data: dict | None = None,
                                       system_state: dict | None = None) -> float:
        """
        Calculate enhanced reward incorporating sensor feedback.

        Args:
            base_reward: Base reward from system performance
            sensor_data: Sensor measurement data
            system_state: Current system state

        Returns:
            Enhanced reward value
        """
        if sensor_data is None or system_state is None:
            return base_reward

        # Base power reward component
        power_reward = base_reward * self.power_weight

        # Biofilm health reward component
        biofilm_reward = self._calculate_biofilm_health_reward(sensor_data, system_state)

        # Sensor agreement reward component
        agreement_reward = self._calculate_sensor_agreement_reward(sensor_data)

        # System stability reward component
        stability_reward = self._calculate_stability_reward(system_state)

        # Combine rewards
        total_reward = (power_reward +
                       biofilm_reward * self.biofilm_health_weight +
                       agreement_reward * self.sensor_agreement_weight +
                       stability_reward * self.stability_weight)

        # Update reward component tracking
        self.total_reward_components['power'] += power_reward
        self.total_reward_components['biofilm_health'] += biofilm_reward
        self.total_reward_components['sensor_agreement'] += agreement_reward
        self.total_reward_components['stability'] += stability_reward

        return total_reward

    def _calculate_biofilm_health_reward(self, sensor_data: dict, system_state: dict) -> float:
        """Calculate reward based on biofilm health indicators."""
        biofilm_reward = 0.0

        # EIS-based biofilm health
        eis_data = sensor_data.get('eis', {})
        if eis_data:
            eis_thickness = eis_data.get('thickness_um', 0.0)
            eis_conductivity = eis_data.get('conductivity_S_per_m', 0.0)

            # Optimal thickness reward using configuration
            rewards_config = self.qlearning_config.reward_weights
            optimal_thickness = rewards_config.biofilm_optimal_thickness_um
            thickness_deviation = abs(eis_thickness - optimal_thickness) / optimal_thickness
            thickness_reward = max(0, 1.0 - thickness_deviation)

            # Conductivity reward (higher is better for electron transfer)
            conductivity_reward = min(1.0, eis_conductivity / rewards_config.conductivity_normalization_S_per_m)

            biofilm_reward += (thickness_reward + conductivity_reward) / 2

        # QCM-based biofilm health
        qcm_data = sensor_data.get('qcm', {})
        if qcm_data:
            qcm_mass = qcm_data.get('mass_per_area_ng_per_cm2', 0.0)
            qcm_quality = qcm_data.get('measurement_quality', 0.0)

            # Mass accumulation reward (steady growth is good)
            if len(self.sensor_confidence_history) > 1:
                previous_mass = getattr(self, '_previous_qcm_mass', 0.0)
                mass_growth_rate = (qcm_mass - previous_mass) / max(1.0, qcm_mass)
                growth_reward = max(0, min(1.0, mass_growth_rate * self.qlearning_config.reward_weights.mass_growth_rate_factor))
                biofilm_reward += growth_reward * qcm_quality

            self._previous_qcm_mass = qcm_mass

        return biofilm_reward

    def _calculate_sensor_agreement_reward(self, sensor_data: dict) -> float:
        """Calculate reward based on sensor agreement."""
        fusion_data = sensor_data.get('fusion', {})
        if not fusion_data:
            return 0.0

        sensor_agreement = fusion_data.get('sensor_agreement', 0.5)
        fusion_confidence = fusion_data.get('fusion_confidence', 0.5)

        # Higher agreement and confidence yield higher reward
        agreement_reward = sensor_agreement * 0.6 + fusion_confidence * 0.4

        return agreement_reward

    def _calculate_stability_reward(self, system_state: dict) -> float:
        """Calculate reward based on system stability."""
        # Flow rate stability
        current_flow = system_state.get('flow_rate', 10.0)
        target_flow = self.qlearning_config.stability_target_flow_rate
        flow_stability = max(0, 1.0 - abs(current_flow - target_flow) / target_flow)

        # Substrate concentration stability
        outlet_conc = system_state.get('outlet_concentration', 10.0)
        target_outlet = self.qlearning_config.stability_target_outlet_concentration
        conc_stability = max(0, 1.0 - abs(outlet_conc - target_outlet) / target_outlet)

        # Combined stability reward
        stability_reward = (flow_stability + conc_stability) / 2

        return stability_reward

    def update_q_value_with_sensors(self, state: tuple, action: int, reward: float,
                                   next_state: tuple, sensor_data: dict | None = None):
        """
        Update Q-value with sensor-enhanced learning.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            sensor_data: Sensor data for learning enhancement
        """
        # Enhanced reward calculation
        enhanced_reward = reward
        if sensor_data:
            sensor_confidence = self._calculate_sensor_confidence(sensor_data)

            # Adjust learning rate based on sensor confidence
            adaptive_learning_rate = self.learning_rate * (0.5 + 0.5 * sensor_confidence)

            # Apply sensor-guided learning
            if sensor_confidence > self.min_sensor_confidence:
                enhanced_reward = self.calculate_sensor_enhanced_reward(
                    reward, sensor_data, {'flow_rate': 15.0, 'outlet_concentration': 12.0}
                )
        else:
            adaptive_learning_rate = self.learning_rate

        # Standard Q-learning update with adaptive learning rate
        current_q = self.q_table[state][action]
        max_next_q = max([self.q_table[next_state][a] for a in range(len(self.actions))])

        new_q = current_q + adaptive_learning_rate * (
            enhanced_reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state][action] = new_q

        # Update statistics
        self.total_rewards += enhanced_reward

    def handle_sensor_fault(self, fault_type: str, affected_sensor: str):
        """
        Handle sensor faults by adapting control strategy.

        Args:
            fault_type: Type of fault ('failed', 'degraded', 'noisy')
            affected_sensor: Which sensor is affected ('eis', 'qcm', 'both')
        """
        self.sensor_fault_count += 1

        if fault_type == 'failed':
            # Reduce sensor weight for failed sensors
            if affected_sensor == 'both':
                self.sensor_degradation_factor = 0.1
            else:
                self.sensor_degradation_factor = 0.5

        elif fault_type == 'degraded':
            # Moderate reduction in sensor weight
            self.sensor_degradation_factor = max(0.3, self.sensor_degradation_factor * 0.8)

        elif fault_type == 'noisy':
            # Slight reduction in sensor weight
            self.sensor_degradation_factor = max(0.6, self.sensor_degradation_factor * 0.9)

        # Increase exploration when sensors are unreliable
        if self.sensor_degradation_factor < 0.5:
            self.epsilon = min(0.8, self.epsilon * 1.2)

        print(f"Sensor fault handled: {fault_type} in {affected_sensor}, "
              f"degradation factor: {self.sensor_degradation_factor:.2f}")

    def get_controller_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive controller performance summary."""
        total_decisions = self.sensor_guided_decisions + self.model_guided_decisions

        summary = {
            'basic_performance': {
                'total_episodes': self.episode_count,
                'total_rewards': self.total_rewards,
                'current_epsilon': self.epsilon,
                'q_table_size': len(self.q_table)
            },
            'sensor_integration': {
                'sensor_enabled': self.enable_sensor_state,
                'sensor_weight': self.sensor_weight,
                'sensor_degradation_factor': self.sensor_degradation_factor,
                'sensor_fault_count': self.sensor_fault_count
            },
            'decision_statistics': {
                'total_decisions': total_decisions,
                'sensor_guided_decisions': self.sensor_guided_decisions,
                'model_guided_decisions': self.model_guided_decisions,
                'sensor_guidance_ratio': (self.sensor_guided_decisions / total_decisions
                                        if total_decisions > 0 else 0)
            },
            'reward_components': self.total_reward_components.copy(),
            'sensor_confidence': {
                'recent_confidence': (np.mean(self.sensor_confidence_history[-10:])
                                    if len(self.sensor_confidence_history) >= 10 else 0),
                'confidence_history_length': len(self.sensor_confidence_history),
                'min_confidence_threshold': self.min_sensor_confidence
            },
            'multi_objective_weights': {
                'power_weight': self.power_weight,
                'biofilm_health_weight': self.biofilm_health_weight,
                'sensor_agreement_weight': self.sensor_agreement_weight,
                'stability_weight': self.stability_weight
            }
        }

        return summary

    def reset_sensor_tracking(self):
        """Reset sensor-related tracking variables."""
        self.sensor_confidence_history.clear()
        self.sensor_fault_count = 0
        self.sensor_degradation_factor = 1.0
        self.sensor_guided_decisions = 0
        self.model_guided_decisions = 0

        # Reset reward components
        for key in self.total_reward_components:
            self.total_reward_components[key] = 0.0

        print("Sensor tracking variables reset")

    def adapt_to_sensor_availability(self, eis_available: bool, qcm_available: bool):
        """
        Adapt controller behavior based on sensor availability.

        Args:
            eis_available: Whether EIS sensor is available
            qcm_available: Whether QCM sensor is available
        """
        if not eis_available and not qcm_available:
            # No sensors available - use base controller
            self.enable_sensor_state = False
            self.sensor_weight = 0.0
            print("No sensors available - using base Q-learning controller")

        elif eis_available and qcm_available:
            # Both sensors available - full sensor integration
            self.enable_sensor_state = True
            self.sensor_weight = 0.3
            print("Both sensors available - full sensor integration enabled")

        else:
            # Single sensor available - partial integration
            self.enable_sensor_state = True
            self.sensor_weight = 0.2  # Reduced weight for single sensor
            sensor_type = "EIS" if eis_available else "QCM"
            print(f"Single sensor ({sensor_type}) available - partial integration enabled")

    def validate_sensor_enhanced_operation(self) -> dict[str, bool]:
        """Validate that sensor-enhanced operation is working correctly."""
        validation_results = {
            'sensor_state_enabled': self.enable_sensor_state,
            'state_space_extended': len(self.eis_thickness_bins) > 0 if hasattr(self, 'eis_thickness_bins') else False,
            'sensor_confidence_tracking': len(self.sensor_confidence_history) > 0,
            'multi_objective_rewards': any(v > 0 for v in self.total_reward_components.values()),
            'adaptive_exploration': hasattr(self, 'exploration_boost_factor'),
            'fault_tolerance': self.fault_tolerance
        }

        return validation_results

    def get_state_hash(self, inlet_conc: float, outlet_conc: float, total_current: float) -> str:
        """Generate a state hash for Q-learning state lookup."""
        # Simple discretization for state encoding
        inlet_bin = int(inlet_conc / 5.0)  # 5 mM bins
        outlet_bin = int(outlet_conc / 5.0)  # 5 mM bins
        current_bin = int(total_current / 0.1)  # 0.1 A bins

        return f"{inlet_bin}_{outlet_bin}_{current_bin}"

    def choose_action(self, state: str) -> int:
        """Choose an action using epsilon-greedy policy."""
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, 10)  # 10 action choices
        else:
            # Exploit: best known action
            q_values = self.q_table[state]
            if q_values:
                return max(q_values, key=q_values.get)
            else:
                return np.random.randint(0, 10)  # Random if no Q-values yet
