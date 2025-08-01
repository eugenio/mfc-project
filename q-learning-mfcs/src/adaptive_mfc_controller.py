"""
Adaptive MFC Control System

Phase 2 enhancement implementing adaptive control algorithms that integrate:
- Health-aware Q-learning with predictive biofilm monitoring
- Multi-objective optimization (power, health, stability)
- Real-time parameter adaptation based on system state
- Intelligent intervention strategies with risk assessment

Created: 2025-07-31
Last Modified: 2025-07-31
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timezone

# Import base control components
from sensing_enhanced_q_controller import SensingEnhancedQLearningController

# Import health monitoring
from biofilm_health_monitor import (
    HealthMetrics, HealthAlert, HealthStatus,
    InterventionRecommendation, create_predictive_health_monitor
)

# Import advanced sensor fusion
from sensing_models.advanced_sensor_fusion import (
    FusedMeasurement, PredictiveState, AnomalyDetection,
    create_advanced_sensor_fusion
)

# Import base sensor components
from sensing_models.sensor_fusion import BacterialSpecies
from sensing_models.eis_model import EISMeasurement
from sensing_models.qcm_model import QCMMeasurement

# Configuration
try:
    from config import QLearningConfig, SensorConfig
except ImportError:
    QLearningConfig = None
    SensorConfig = None

logger = logging.getLogger(__name__)


class ControlStrategy(Enum):
    """Control strategy modes."""
    PERFORMANCE_FOCUSED = "performance_focused"  # Maximize power output
    HEALTH_FOCUSED = "health_focused"           # Prioritize biofilm health
    BALANCED = "balanced"                       # Balance power and health
    CONSERVATIVE = "conservative"               # Minimize risks
    RECOVERY = "recovery"                       # Recover from poor health


class AdaptationMode(Enum):
    """Parameter adaptation modes."""
    AGGRESSIVE = "aggressive"  # Fast adaptation, higher risk
    MODERATE = "moderate"      # Balanced adaptation
    CONSERVATIVE = "conservative"  # Slow, safe adaptation
    DISABLED = "disabled"      # No adaptation


@dataclass
class ControlDecision:
    """Control decision with rationale."""

    action_index: int
    action_description: str
    expected_outcome: Dict[str, float]  # Expected changes in key metrics
    confidence: float  # Decision confidence (0-1)
    rationale: str  # Explanation of decision
    risk_assessment: Dict[str, float]  # Assessed risks
    intervention_type: Optional[str] = None  # If this is an intervention


@dataclass
class SystemState:
    """Complete system state for control decisions."""

    # Sensor measurements
    fused_measurement: FusedMeasurement
    prediction: Optional[PredictiveState]
    anomalies: List[AnomalyDetection]

    # Health assessment
    health_metrics: HealthMetrics
    health_alerts: List[HealthAlert]

    # System parameters
    flow_rate: float
    inlet_concentration: float
    outlet_concentration: float
    current_density: float
    power_output: float

    # Control state
    current_strategy: ControlStrategy
    adaptation_mode: AdaptationMode
    intervention_active: bool


class HealthAwareQLearning(SensingEnhancedQLearningController):
    """
    Enhanced Q-learning controller with health-aware reward function and adaptive parameters.
    
    Extends the sensor-enhanced controller with:
    - Health-weighted reward functions
    - Adaptive exploration based on system health
    - Multi-objective optimization
    - Risk-aware action selection
    """

    def __init__(self, qlearning_config: Optional[QLearningConfig] = None,
                 sensor_config: Optional[SensorConfig] = None,
                 health_weight: float = 0.4,
                 adaptation_rate: float = 0.1):
        """
        Initialize health-aware Q-learning controller.
        
        Args:
            qlearning_config: Q-learning configuration
            sensor_config: Sensor configuration
            health_weight: Weight for health in reward function (0-1)
            adaptation_rate: Rate of parameter adaptation
        """
        # Initialize base controller
        super().__init__(qlearning_config, sensor_config, enable_sensor_state=True, fault_tolerance=True)

        # Health-aware parameters
        self.health_weight = health_weight
        self.power_weight = 1.0 - health_weight
        self.adaptation_rate = adaptation_rate

        # Adaptive parameters
        self.base_learning_rate = self.learning_rate
        self.base_epsilon = self.epsilon
        self.base_discount_factor = self.discount_factor

        # Performance tracking
        self.health_reward_history = []
        self.power_reward_history = []
        self.intervention_history = []
        self.adaptation_history = []

        # Risk thresholds
        self.risk_thresholds = {
            'high_risk_epsilon_boost': 0.3,  # Increase exploration when risks are high
            'health_critical_epsilon': 0.8,  # Very high exploration for critical health
            'intervention_learning_boost': 2.0  # Learning rate multiplier during interventions
        }

        # Initialize actions list for compatibility with sensing enhanced controller
        # Create a combined action space from flow and substrate actions
        if hasattr(self, 'flow_actions') and hasattr(self, 'substrate_actions'):
            self.actions = []
            for flow_idx, flow_val in enumerate(self.flow_actions):
                for substr_idx, substr_val in enumerate(self.substrate_actions):
                    self.actions.append((flow_val, substr_val))
        else:
            # Fallback action space
            self.actions = [
                (-10, -5), (-10, 0), (-10, 5),  # Decrease flow, vary substrate
                (-5, -5), (-5, 0), (-5, 5),    # Small flow decrease
                (0, -5), (0, 0), (0, 5),       # Maintain flow, vary substrate
                (5, -5), (5, 0), (5, 5),       # Small flow increase
                (10, -5), (10, 0), (10, 5)     # Increase flow, vary substrate
            ]

        logger.info(f"Health-aware Q-learning initialized with health_weight={health_weight}")

    def calculate_health_aware_reward(self, base_reward: float, health_metrics: HealthMetrics,
                                    system_state: SystemState, intervention_active: bool = False) -> float:
        """
        Calculate reward incorporating health metrics and system state.
        
        Args:
            base_reward: Base power/performance reward
            health_metrics: Current health assessment
            system_state: Complete system state
            intervention_active: Whether an intervention is active
            
        Returns:
            Health-aware reward value
        """
        # Base power reward component
        power_reward = base_reward * self.power_weight

        # Health reward component
        health_reward = self._calculate_health_reward(health_metrics, system_state)

        # Intervention bonus/penalty
        intervention_modifier = 1.0
        if intervention_active:
            # Bonus for successful interventions
            if health_metrics.health_trend.value in ['improving', 'stable']:
                intervention_modifier = 1.2
            else:
                intervention_modifier = 0.8  # Penalty for unsuccessful interventions

        # Risk penalty
        risk_penalty = self._calculate_risk_penalty(health_metrics)

        # Stability bonus
        stability_bonus = self._calculate_stability_bonus(system_state)

        # Combined reward
        total_reward = (
            (power_reward + health_reward * self.health_weight) * intervention_modifier
            - risk_penalty + stability_bonus
        )

        # Store for analysis
        self.health_reward_history.append(health_reward)
        self.power_reward_history.append(power_reward)

        return total_reward

    def _calculate_health_reward(self, health_metrics: HealthMetrics, system_state: SystemState) -> float:
        """Calculate health-based reward component."""
        # Base health reward
        health_reward = health_metrics.overall_health_score

        # Trend bonus/penalty
        if health_metrics.health_trend.value == 'improving':
            health_reward *= 1.3
        elif health_metrics.health_trend.value == 'declining':
            health_reward *= 0.7
        elif health_metrics.health_trend.value == 'volatile':
            health_reward *= 0.8

        # Component-specific bonuses
        if health_metrics.thickness_health > 0.8:
            health_reward += 0.1
        if health_metrics.conductivity_health > 0.8:
            health_reward += 0.15  # Conductivity very important for power
        if health_metrics.growth_health > 0.8:
            health_reward += 0.1

        # Prediction bonus (reward good predictions)
        if health_metrics.predicted_health_24h > health_metrics.overall_health_score:
            health_reward += 0.05  # Small bonus for positive predictions

        return np.clip(health_reward, -0.5, 1.5)  # Allow some negative rewards

    def _calculate_risk_penalty(self, health_metrics: HealthMetrics) -> float:
        """Calculate penalty based on risk factors."""
        penalty = 0.0

        # Individual risk penalties
        if health_metrics.fouling_risk > 0.7:
            penalty += 0.2 * health_metrics.fouling_risk
        if health_metrics.detachment_risk > 0.7:
            penalty += 0.3 * health_metrics.detachment_risk  # Detachment is worse
        if health_metrics.stagnation_risk > 0.7:
            penalty += 0.15 * health_metrics.stagnation_risk

        # Critical health penalty
        if health_metrics.health_status == HealthStatus.CRITICAL:
            penalty += 0.5
        elif health_metrics.health_status == HealthStatus.POOR:
            penalty += 0.2

        return penalty

    def _calculate_stability_bonus(self, system_state: SystemState) -> float:
        """Calculate bonus for system stability."""
        bonus = 0.0

        # Measurement stability
        if system_state.fused_measurement.fusion_confidence > 0.8:
            bonus += 0.05
        if system_state.fused_measurement.sensor_agreement > 0.8:
            bonus += 0.05

        # Low anomaly bonus
        if len(system_state.anomalies) == 0:
            bonus += 0.05
        elif len([a for a in system_state.anomalies if a.severity in ['high', 'critical']]) == 0:
            bonus += 0.02

        return bonus

    def adapt_parameters(self, health_metrics: HealthMetrics, system_state: SystemState):
        """Adapt learning parameters based on system state."""
        # Base parameter adaptation
        confidence_factor = health_metrics.assessment_confidence

        # Learning rate adaptation
        if health_metrics.health_status == HealthStatus.CRITICAL:
            # Increase learning for critical situations
            self.learning_rate = min(0.8, self.base_learning_rate * 2.0)
        elif health_metrics.health_trend.value == 'volatile':
            # Moderate learning for volatile conditions
            self.learning_rate = self.base_learning_rate * 1.2
        else:
            # Gradual adaptation toward base rate
            target_rate = self.base_learning_rate * (0.5 + confidence_factor)
            self.learning_rate += self.adaptation_rate * (target_rate - self.learning_rate)

        # Epsilon (exploration) adaptation
        base_epsilon_adjustment = 0.0

        # Health-based exploration
        if health_metrics.health_status == HealthStatus.CRITICAL:
            base_epsilon_adjustment += self.risk_thresholds['health_critical_epsilon']
        elif health_metrics.overall_health_score < 0.5:
            base_epsilon_adjustment += 0.3

        # Risk-based exploration
        max_risk = max(health_metrics.fouling_risk, health_metrics.detachment_risk, health_metrics.stagnation_risk)
        if max_risk > 0.7:
            base_epsilon_adjustment += self.risk_thresholds['high_risk_epsilon_boost']

        # Anomaly-based exploration
        critical_anomalies = len([a for a in system_state.anomalies if a.severity == 'critical'])
        if critical_anomalies > 0:
            base_epsilon_adjustment += 0.2 * critical_anomalies

        # Apply epsilon adjustment with decay
        self.epsilon = min(0.9, self.base_epsilon + base_epsilon_adjustment * 0.5)

        # Discount factor adaptation (less important for immediate rewards in critical situations)
        if health_metrics.health_status == HealthStatus.CRITICAL:
            self.discount_factor = max(0.7, self.base_discount_factor * 0.8)
        else:
            self.discount_factor = self.base_discount_factor

        # Store adaptation event
        self.adaptation_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'discount_factor': self.discount_factor,
            'health_score': health_metrics.overall_health_score,
            'adaptation_trigger': self._get_adaptation_trigger(health_metrics, system_state)
        })

        logger.debug(f"Parameters adapted: lr={self.learning_rate:.3f}, ε={self.epsilon:.3f}, "
                    f"γ={self.discount_factor:.3f}")

    def _get_adaptation_trigger(self, health_metrics: HealthMetrics, system_state: SystemState) -> str:
        """Identify what triggered parameter adaptation."""
        triggers = []

        if health_metrics.health_status == HealthStatus.CRITICAL:
            triggers.append('critical_health')
        if health_metrics.health_trend.value == 'volatile':
            triggers.append('volatile_trend')
        if len([a for a in system_state.anomalies if a.severity == 'critical']) > 0:
            triggers.append('critical_anomalies')
        if max(health_metrics.fouling_risk, health_metrics.detachment_risk, health_metrics.stagnation_risk) > 0.7:
            triggers.append('high_risk')

        return ', '.join(triggers) if triggers else 'routine_adaptation'

    def choose_health_aware_action(self, system_state: SystemState,
                                 available_actions: Optional[List] = None) -> ControlDecision:
        """
        Choose action with health awareness and full decision rationale.
        
        Args:
            system_state: Complete system state
            available_actions: Optional list of available actions
            
        Returns:
            Control decision with rationale
        """
        # Adapt parameters based on current state
        self.adapt_parameters(system_state.health_metrics, system_state)

        # Get base state for Q-learning
        base_state = self._system_state_to_qlearning_state(system_state)

        # Prepare sensor data
        sensor_data = self._prepare_sensor_data(system_state)

        # Choose action using enhanced method
        action_idx = self.choose_action_with_sensors(base_state, sensor_data, available_actions)

        # Generate decision rationale
        decision = self._generate_decision_rationale(action_idx, system_state, available_actions)

        return decision

    def _system_state_to_qlearning_state(self, system_state: SystemState) -> Tuple:
        """Convert system state to Q-learning state tuple."""
        # Extract key parameters for state representation
        inlet_conc = system_state.inlet_concentration
        outlet_conc = system_state.outlet_concentration
        current_density = system_state.current_density

        # Discretize state (using base controller method)
        # state_hash = self.get_state_hash(inlet_conc, outlet_conc, current_density)  # Unused

        # Convert to tuple format expected by enhanced controller
        # This is a simplified mapping - in practice would be more sophisticated
        inlet_bin = int(inlet_conc / 5.0)
        outlet_bin = int(outlet_conc / 5.0)
        current_bin = int(current_density / 0.1)

        return (inlet_bin, outlet_bin, current_bin)

    def _prepare_sensor_data(self, system_state: SystemState) -> Dict[str, Any]:
        """Prepare sensor data dictionary for enhanced controller."""
        fused = system_state.fused_measurement

        sensor_data = {
            'eis': {
                'thickness_um': fused.eis_thickness,
                'conductivity_S_per_m': fused.conductivity_S_per_m,
                'measurement_quality': fused.fusion_confidence,
                'status': fused.eis_status
            },
            'qcm': {
                'thickness_um': fused.qcm_thickness,
                'mass_per_area_ng_per_cm2': fused.biomass_density_g_per_L * 1000,  # Rough conversion
                'measurement_quality': fused.fusion_confidence,
                'status': fused.qcm_status
            },
            'fusion': {
                'sensor_agreement': fused.sensor_agreement,
                'fusion_confidence': fused.fusion_confidence,
                'cross_validation_error': fused.cross_validation_error
            }
        }

        return sensor_data

    def _generate_decision_rationale(self, action_idx: int, system_state: SystemState,
                                   available_actions: Optional[List] = None) -> ControlDecision:
        """Generate comprehensive decision rationale."""
        # Map action index to description
        action_descriptions = {
            0: "Maintain current flow rate",
            1: "Increase flow rate by 10%",
            2: "Decrease flow rate by 10%",
            3: "Increase flow rate by 20%",
            4: "Decrease flow rate by 20%",
            5: "Optimize for power output",
            6: "Optimize for biofilm health",
            7: "Emergency flow reduction",
            8: "Substrate concentration adjustment",
            9: "System monitoring mode"
        }

        action_description = action_descriptions.get(action_idx, f"Action {action_idx}")

        # Expected outcomes based on current state and action
        expected_outcome = self._predict_action_outcomes(action_idx, system_state)

        # Decision confidence based on Q-values and system state
        confidence = self._calculate_decision_confidence(action_idx, system_state)

        # Generate rationale
        rationale_parts = []

        # Health-based rationale
        if system_state.health_metrics.health_status == HealthStatus.CRITICAL:
            rationale_parts.append("Critical health status requires immediate intervention")
        elif system_state.health_metrics.health_trend.value == 'declining':
            rationale_parts.append("Declining health trend necessitates corrective action")

        # Risk-based rationale
        risks = []
        if system_state.health_metrics.fouling_risk > 0.7:
            risks.append("fouling")
        if system_state.health_metrics.detachment_risk > 0.7:
            risks.append("detachment")
        if system_state.health_metrics.stagnation_risk > 0.7:
            risks.append("stagnation")

        if risks:
            rationale_parts.append(f"High {', '.join(risks)} risk detected")

        # Sensor-based rationale
        if len(system_state.anomalies) > 0:
            rationale_parts.append(f"{len(system_state.anomalies)} sensor anomalies influence decision")

        # Strategy-based rationale
        if system_state.current_strategy == ControlStrategy.HEALTH_FOCUSED:
            rationale_parts.append("Health-focused strategy prioritizes biofilm wellness")
        elif system_state.current_strategy == ControlStrategy.PERFORMANCE_FOCUSED:
            rationale_parts.append("Performance-focused strategy maximizes power output")

        rationale = ". ".join(rationale_parts) if rationale_parts else "Standard operational decision"

        # Risk assessment for this decision
        risk_assessment = self._assess_decision_risks(action_idx, system_state)

        # Determine if this is an intervention
        intervention_type = None
        if system_state.health_metrics.health_status in [HealthStatus.CRITICAL, HealthStatus.POOR]:
            if action_idx in [7, 8]:  # Emergency actions
                intervention_type = "emergency_intervention"
            elif action_idx in [6]:  # Health optimization
                intervention_type = "health_intervention"

        return ControlDecision(
            action_index=action_idx,
            action_description=action_description,
            expected_outcome=expected_outcome,
            confidence=confidence,
            rationale=rationale,
            risk_assessment=risk_assessment,
            intervention_type=intervention_type
        )

    def _predict_action_outcomes(self, action_idx: int, system_state: SystemState) -> Dict[str, float]:
        """Predict expected outcomes of action."""
        # Simplified outcome prediction based on action type
        base_thickness = system_state.fused_measurement.thickness_um
        # base_conductivity = system_state.fused_measurement.conductivity_S_per_m  # Unused
        base_health = system_state.health_metrics.overall_health_score

        outcomes = {
            'thickness_change': 0.0,
            'conductivity_change': 0.0,
            'health_change': 0.0,
            'power_change': 0.0
        }

        # Action-specific predictions
        if action_idx == 1:  # Increase flow 10%
            outcomes['thickness_change'] = -0.5  # Slight erosion
            outcomes['conductivity_change'] = 0.002  # Better mass transfer
            outcomes['health_change'] = 0.1 if base_health < 0.7 else -0.05
            outcomes['power_change'] = 0.05
        elif action_idx == 2:  # Decrease flow 10%
            outcomes['thickness_change'] = 0.3  # More growth
            outcomes['conductivity_change'] = -0.001  # Poorer mass transfer
            outcomes['health_change'] = 0.05 if base_thickness < 10 else -0.1
            outcomes['power_change'] = -0.02
        elif action_idx == 6:  # Optimize for health
            outcomes['health_change'] = 0.15
            outcomes['power_change'] = -0.05  # May sacrifice some power
        elif action_idx == 5:  # Optimize for power
            outcomes['power_change'] = 0.1
            outcomes['health_change'] = -0.05  # May stress biofilm

        return outcomes

    def _calculate_decision_confidence(self, action_idx: int, system_state: SystemState) -> float:
        """Calculate confidence in decision."""
        # Base confidence from measurement quality
        base_confidence = system_state.fused_measurement.fusion_confidence

        # Reduce confidence for high-risk situations
        max_risk = max(
            system_state.health_metrics.fouling_risk,
            system_state.health_metrics.detachment_risk,
            system_state.health_metrics.stagnation_risk
        )
        risk_penalty = max_risk * 0.3

        # Reduce confidence for critical health
        if system_state.health_metrics.health_status == HealthStatus.CRITICAL:
            health_penalty = 0.2
        elif system_state.health_metrics.health_status == HealthStatus.POOR:
            health_penalty = 0.1
        else:
            health_penalty = 0.0

        # Anomaly penalty
        anomaly_penalty = len(system_state.anomalies) * 0.05

        # Combine factors
        confidence = base_confidence - risk_penalty - health_penalty - anomaly_penalty

        return np.clip(confidence, 0.1, 1.0)

    def _assess_decision_risks(self, action_idx: int, system_state: SystemState) -> Dict[str, float]:
        """Assess risks associated with decision."""
        risks = {
            'biofilm_damage': 0.0,
            'performance_loss': 0.0,
            'system_instability': 0.0,
            'sensor_interference': 0.0
        }

        # Action-specific risks
        if action_idx in [3, 4]:  # Large flow changes
            risks['biofilm_damage'] = 0.3
            risks['system_instability'] = 0.2
        elif action_idx == 7:  # Emergency flow reduction
            risks['performance_loss'] = 0.4
        elif action_idx == 8:  # Substrate adjustment
            risks['system_instability'] = 0.3

        # State-dependent risk modifiers
        if system_state.health_metrics.detachment_risk > 0.7:
            risks['biofilm_damage'] *= 1.5  # Higher damage risk

        if len(system_state.anomalies) > 2:
            risks['sensor_interference'] = 0.2

        return risks


class AdaptiveMFCController:
    """
    Master adaptive MFC controller integrating all Phase 2 enhancements.
    
    Coordinates:
    - Advanced sensor fusion with predictive capabilities
    - Predictive biofilm health monitoring
    - Health-aware Q-learning control
    - Intelligent intervention strategies
    - Multi-objective optimization
    """

    def __init__(self, species: BacterialSpecies = BacterialSpecies.MIXED,
                 qlearning_config: Optional[QLearningConfig] = None,
                 sensor_config: Optional[SensorConfig] = None,
                 initial_strategy: ControlStrategy = ControlStrategy.BALANCED):
        """
        Initialize adaptive MFC controller.
        
        Args:
            species: Target bacterial species
            qlearning_config: Q-learning configuration
            sensor_config: Sensor configuration
            initial_strategy: Initial control strategy
        """
        self.species = species
        self.qlearning_config = qlearning_config
        self.sensor_config = sensor_config

        # Initialize subsystems
        self.sensor_fusion = create_advanced_sensor_fusion(sensor_config)
        self.health_monitor = create_predictive_health_monitor(species, sensor_config)
        self.q_controller = HealthAwareQLearning(qlearning_config, sensor_config)

        # Control state
        self.current_strategy = initial_strategy
        self.adaptation_mode = AdaptationMode.MODERATE
        self.intervention_active = False
        self.last_intervention_time = 0.0

        # Performance tracking
        self.control_history = []
        self.strategy_changes = []
        self.intervention_outcomes = []

        # System parameters (would be connected to actual MFC in practice)
        self.system_parameters = {
            'flow_rate': 15.0,  # mL/min
            'inlet_concentration': 10.0,  # mM
            'outlet_concentration': 8.0,  # mM
            'current_density': 0.5,  # A/m²
            'power_output': 0.1,  # W
            'temperature': 25.0,  # °C
            'ph': 7.0
        }

        logger.info(f"Adaptive MFC controller initialized for {species.value} with {initial_strategy.value} strategy")

    def control_step(self, eis_measurement: EISMeasurement, qcm_measurement: QCMMeasurement,
                    eis_properties: Dict[str, float], qcm_properties: Dict[str, float],
                    time_hours: float) -> Dict[str, Any]:
        """
        Execute one control step with full system integration.
        
        Args:
            eis_measurement: EIS sensor measurement
            qcm_measurement: QCM sensor measurement  
            eis_properties: Processed EIS properties
            qcm_properties: Processed QCM properties
            time_hours: Current time in hours
            
        Returns:
            Complete control step results
        """
        # 1. Sensor fusion with prediction
        fused_measurement, prediction, anomalies = self.sensor_fusion.fuse_measurements_with_prediction(
            eis_measurement, qcm_measurement, eis_properties, qcm_properties, time_hours, predict_steps=5
        )

        # 2. Biofilm growth pattern analysis
        growth_pattern = self.sensor_fusion.analyze_biofilm_growth_pattern()

        # 3. Health assessment
        health_metrics = self.health_monitor.assess_health(fused_measurement, growth_pattern, anomalies)
        health_alerts = self.health_monitor.generate_alerts(health_metrics, anomalies)

        # 4. Create system state
        system_state = SystemState(
            fused_measurement=fused_measurement,
            prediction=prediction,
            anomalies=anomalies,
            health_metrics=health_metrics,
            health_alerts=health_alerts,
            flow_rate=self.system_parameters['flow_rate'],
            inlet_concentration=self.system_parameters['inlet_concentration'],
            outlet_concentration=self.system_parameters['outlet_concentration'],
            current_density=self.system_parameters['current_density'],
            power_output=self.system_parameters['power_output'],
            current_strategy=self.current_strategy,
            adaptation_mode=self.adaptation_mode,
            intervention_active=self.intervention_active
        )

        # 5. Strategy adaptation
        self._adapt_control_strategy(system_state)

        # 6. Check for interventions
        intervention_recommendations = self._evaluate_interventions(system_state)

        # 7. Control decision
        control_decision = self.q_controller.choose_health_aware_action(system_state)

        # 8. Execute control action
        execution_results = self._execute_control_action(control_decision, system_state)

        # 9. Update Q-learning
        self._update_qlearning(system_state, control_decision, execution_results)

        # 10. Compile results
        control_results = {
            'timestamp': time_hours,
            'system_state': system_state,
            'control_decision': control_decision,
            'execution_results': execution_results,
            'intervention_recommendations': intervention_recommendations,
            'health_alerts': health_alerts,
            'prediction': prediction,
            'performance_metrics': self._calculate_performance_metrics(system_state),
            'system_health_score': health_metrics.overall_health_score
        }

        # Store for history
        self.control_history.append(control_results)

        # Log important events
        if health_alerts:
            logger.warning(f"Health alerts generated: {[a.message for a in health_alerts]}")
        if control_decision.intervention_type:
            logger.info(f"Intervention executed: {control_decision.intervention_type}")

        return control_results

    def _adapt_control_strategy(self, system_state: SystemState):
        """Adapt control strategy based on system state."""
        previous_strategy = self.current_strategy

        # Critical health - switch to health focus
        if system_state.health_metrics.health_status == HealthStatus.CRITICAL:
            self.current_strategy = ControlStrategy.RECOVERY
        elif system_state.health_metrics.health_status == HealthStatus.POOR:
            self.current_strategy = ControlStrategy.HEALTH_FOCUSED

        # High risks - conservative approach
        elif max(
            system_state.health_metrics.fouling_risk,
            system_state.health_metrics.detachment_risk,
            system_state.health_metrics.stagnation_risk
        ) > 0.8:
            self.current_strategy = ControlStrategy.CONSERVATIVE

        # Excellent health - optimize performance
        elif (system_state.health_metrics.health_status == HealthStatus.EXCELLENT and
              system_state.health_metrics.health_trend.value in ['stable', 'improving']):
            self.current_strategy = ControlStrategy.PERFORMANCE_FOCUSED

        # Default to balanced
        else:
            self.current_strategy = ControlStrategy.BALANCED

        # Log strategy changes
        if self.current_strategy != previous_strategy:
            change_event = {
                'timestamp': system_state.fused_measurement.timestamp,
                'from_strategy': previous_strategy.value,
                'to_strategy': self.current_strategy.value,
                'trigger': self._identify_strategy_trigger(system_state),
                'health_score': system_state.health_metrics.overall_health_score
            }
            self.strategy_changes.append(change_event)
            logger.info(f"Strategy changed: {previous_strategy.value} → {self.current_strategy.value}")

    def _identify_strategy_trigger(self, system_state: SystemState) -> str:
        """Identify what triggered strategy change."""
        if system_state.health_metrics.health_status == HealthStatus.CRITICAL:
            return 'critical_health'
        elif system_state.health_metrics.health_status == HealthStatus.POOR:
            return 'poor_health'
        elif len([a for a in system_state.health_alerts if a.severity == 'critical']) > 0:
            return 'critical_alerts'
        elif max(system_state.health_metrics.fouling_risk,
                system_state.health_metrics.detachment_risk,
                system_state.health_metrics.stagnation_risk) > 0.8:
            return 'high_risk'
        elif system_state.health_metrics.health_status == HealthStatus.EXCELLENT:
            return 'excellent_health'
        else:
            return 'operational_optimization'

    def _evaluate_interventions(self, system_state: SystemState) -> List[InterventionRecommendation]:
        """Evaluate and potentially execute interventions."""
        recommendations = self.health_monitor.generate_intervention_recommendations(system_state.health_metrics)

        # Check if intervention should be executed automatically
        for rec in recommendations:
            if rec.urgency == 'immediate' and rec.success_probability > 0.7:
                logger.info(f"Executing immediate intervention: {rec.description}")
                self._execute_intervention(rec, system_state)
                break

        return recommendations

    def _execute_intervention(self, intervention: InterventionRecommendation, system_state: SystemState):
        """Execute an intervention."""
        self.intervention_active = True
        self.last_intervention_time = system_state.fused_measurement.timestamp

        # Log intervention
        intervention_event = {
            'timestamp': system_state.fused_measurement.timestamp,
            'type': intervention.intervention_type,
            'description': intervention.description,
            'urgency': intervention.urgency,
            'expected_benefit': intervention.expected_benefit,
            'success_probability': intervention.success_probability
        }
        self.intervention_outcomes.append(intervention_event)

        # In practice, this would interface with actual MFC hardware
        logger.info(f"Intervention executed: {intervention.intervention_type}")

    def _execute_control_action(self, decision: ControlDecision, system_state: SystemState) -> Dict[str, Any]:
        """Execute control action and return results."""
        # Simulate action execution (in practice would control actual MFC)
        action_idx = decision.action_index

        # Apply action to system parameters (simplified simulation)
        if action_idx == 1:  # Increase flow 10%
            self.system_parameters['flow_rate'] *= 1.1
        elif action_idx == 2:  # Decrease flow 10%
            self.system_parameters['flow_rate'] *= 0.9
        elif action_idx == 3:  # Increase flow 20%
            self.system_parameters['flow_rate'] *= 1.2
        elif action_idx == 4:  # Decrease flow 20%
            self.system_parameters['flow_rate'] *= 0.8
        elif action_idx == 7:  # Emergency flow reduction
            self.system_parameters['flow_rate'] *= 0.5
        elif action_idx == 8:  # Substrate adjustment
            self.system_parameters['inlet_concentration'] *= 1.1

        # Calculate execution success
        execution_success = decision.confidence * (1.0 - sum(decision.risk_assessment.values()) / 4.0)

        return {
            'action_executed': decision.action_description,
            'parameter_changes': self._get_parameter_changes(),
            'execution_success': execution_success,
            'estimated_outcome': decision.expected_outcome
        }

    def _get_parameter_changes(self) -> Dict[str, float]:
        """Get recent parameter changes."""
        # Simplified - would track actual changes
        return {
            'flow_rate': self.system_parameters['flow_rate'],
            'inlet_concentration': self.system_parameters['inlet_concentration']
        }

    def _update_qlearning(self, system_state: SystemState, decision: ControlDecision, execution_results: Dict[str, Any]):
        """Update Q-learning based on results."""
        # Calculate reward
        base_reward = 0.1  # Simplified power reward
        reward = self.q_controller.calculate_health_aware_reward(
            base_reward, system_state.health_metrics, system_state, self.intervention_active
        )

        # Create next state (simplified)
        current_state = self.q_controller._system_state_to_qlearning_state(system_state)
        next_state = current_state  # Would be updated based on new measurements

        # Update Q-values
        sensor_data = self.q_controller._prepare_sensor_data(system_state)
        self.q_controller.update_q_value_with_sensors(
            current_state, decision.action_index, reward, next_state, sensor_data
        )

    def _calculate_performance_metrics(self, system_state: SystemState) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        return {
            'power_efficiency': self.system_parameters['power_output'] / max(0.01, self.system_parameters['flow_rate']),
            'biofilm_health_score': system_state.health_metrics.overall_health_score,
            'sensor_reliability': system_state.fused_measurement.fusion_confidence,
            'system_stability': 1.0 - len(system_state.anomalies) / 10.0,
            'control_confidence': system_state.health_metrics.prediction_confidence
        }

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self.control_history:
            return {'error': 'No control history available'}

        latest = self.control_history[-1]

        status = {
            'timestamp': latest['timestamp'],
            'control_strategy': self.current_strategy.value,
            'adaptation_mode': self.adaptation_mode.value,
            'intervention_active': self.intervention_active,
            'system_health': {
                'overall_score': latest['system_health_score'],
                'status': latest['system_state'].health_metrics.health_status.value,
                'trend': latest['system_state'].health_metrics.health_trend.value
            },
            'sensor_fusion': self.sensor_fusion.get_system_health_assessment(),
            'biofilm_health': self.health_monitor.get_health_dashboard_data(),
            'q_learning_stats': self.q_controller.get_controller_performance_summary(),
            'recent_performance': latest['performance_metrics'],
            'active_alerts': len(latest['health_alerts']),
            'strategy_changes_24h': len([s for s in self.strategy_changes
                                       if s['timestamp'] > latest['timestamp'] - 24.0]),
            'interventions_24h': len([i for i in self.intervention_outcomes
                                    if i['timestamp'] > latest['timestamp'] - 24.0])
        }

        return status


def create_adaptive_mfc_controller(species: BacterialSpecies = BacterialSpecies.MIXED,
                                 qlearning_config: Optional[QLearningConfig] = None,
                                 sensor_config: Optional[SensorConfig] = None) -> AdaptiveMFCController:
    """
    Factory function to create fully integrated adaptive MFC controller.
    
    Args:
        species: Target bacterial species
        qlearning_config: Q-learning configuration
        sensor_config: Sensor configuration
        
    Returns:
        Configured AdaptiveMFCController instance
    """
    controller = AdaptiveMFCController(
        species=species,
        qlearning_config=qlearning_config,
        sensor_config=sensor_config,
        initial_strategy=ControlStrategy.BALANCED
    )

    logger.info(f"Adaptive MFC controller created for {species.value} with integrated Phase 2 enhancements")
    return controller
