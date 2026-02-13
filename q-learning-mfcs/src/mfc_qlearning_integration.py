#!/usr/bin/env python3
"""
Q-Learning Integration for Complete MFC System
==============================================

Advanced Q-learning controller for integrated MFC system optimization.
Manages multi-objective optimization across:
- Power generation (maximize)
- Coulombic efficiency (maximize) 
- System lifetime (maximize via fouling/degradation management)
- Operational costs (minimize)

Features:
- Multi-dimensional state space (biofilm, membrane, cathode, economics)
- Hierarchical action space (flow, substrate, temperature, cleaning)
- Reward function with weighted objectives
- Dynamic exploration with performance feedback
- Long-term planning with degradation awareness

Created: 2025-07-27 (Phase 5)
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import pickle

from mfc_system_integration import MFCSystemState, IntegratedMFCSystem


class OptimizationObjective(Enum):
    """Multi-objective optimization targets."""
    POWER_DENSITY = "power_density"
    COULOMBIC_EFFICIENCY = "coulombic_efficiency" 
    SYSTEM_LIFETIME = "system_lifetime"
    OPERATIONAL_COST = "operational_cost"
    SUBSTRATE_UTILIZATION = "substrate_utilization"


@dataclass
class QLearningConfig:
    """Configuration for Q-learning controller."""
    
    # Learning parameters
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon_initial: float = 0.3
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    
    # State space discretization
    state_bins: int = 8
    memory_length: int = 10  # Steps to remember for state encoding
    
    # Action space
    flow_actions: List[float] = field(default_factory=lambda: [-10, -5, -2, 0, 2, 5, 10])  # mL/h changes
    substrate_actions: List[float] = field(default_factory=lambda: [-2.0, -1.0, -0.5, 0, 0.5, 1.0, 2.0])  # mmol/L changes
    cleaning_actions: List[str] = field(default_factory=lambda: ["none", "backwash", "chemical"])
    temperature_actions: List[float] = field(default_factory=lambda: [-2, -1, 0, 1, 2])  # °C changes
    
    # Reward weights (sum should equal 1.0)
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "power_density": 0.35,
        "coulombic_efficiency": 0.25, 
        "system_lifetime": 0.25,
        "operational_cost": 0.15
    })
    
    # Performance thresholds
    power_target: float = 0.5      # W/m² target
    efficiency_target: float = 0.8  # 80% CE target
    fouling_threshold: float = 50.0  # μm before cleaning
    cost_limit: float = 0.1        # $/h operational cost limit


class SystemStateEncoder:
    """Encodes multi-dimensional MFC system state for Q-learning."""
    
    def __init__(self, config: QLearningConfig):
        self.config = config
        self.state_history = deque(maxlen=config.memory_length)
        
        # Define state feature ranges for normalization
        self.feature_ranges = {
            'power_density': (0, 2.0),        # W/m²
            'cell_voltage': (0, 1.2),         # V
            'current_density': (0, 1000),     # A/m²
            'coulombic_efficiency': (0, 1),   # fraction
            'biofilm_thickness': (0, 100),    # μm
            'substrate_concentration': (0, 50), # mmol/L
            'fouling_thickness': (0, 200),    # μm
            'degradation_fraction': (0, 1),   # fraction
            'membrane_conductivity': (0, 100), # S/m
            'oxygen_concentration': (0, 0.5), # mol/m³
            'operational_cost': (0, 1.0),     # $/h
            'time_hours': (0, 8760)           # hours (1 year)
        }
    
    def encode_state(self, states: List[MFCSystemState], system_time: float) -> Tuple[int, ...]:
        """
        Encode system state into discrete Q-learning state.
        
        Args:
            states: Current cell states
            system_time: Current system time
            
        Returns:
            Discrete state tuple for Q-table indexing
        """
        if not states:
            return tuple([0] * 12)  # Empty state
        
        # Calculate system-level metrics
        avg_power = np.mean([s.power_density for s in states])
        avg_voltage = np.mean([s.cell_voltage for s in states])
        avg_current = np.mean([s.current_density for s in states])
        avg_efficiency = np.mean([s.coulombic_efficiency for s in states])
        avg_biofilm = np.mean([s.biofilm_thickness for s in states])
        avg_substrate = np.mean([s.substrate_concentration for s in states])
        avg_fouling = np.mean([s.fouling_thickness for s in states])
        avg_degradation = np.mean([s.degradation_fraction for s in states])
        avg_conductivity = np.mean([s.membrane_conductivity for s in states])
        avg_oxygen = np.mean([s.oxygen_concentration for s in states])
        avg_cost = np.mean([s.operational_cost for s in states])
        
        # Create feature vector
        features = {
            'power_density': avg_power,
            'cell_voltage': avg_voltage,
            'current_density': avg_current,
            'coulombic_efficiency': avg_efficiency,
            'biofilm_thickness': avg_biofilm,
            'substrate_concentration': avg_substrate,
            'fouling_thickness': avg_fouling,
            'degradation_fraction': avg_degradation,
            'membrane_conductivity': avg_conductivity,
            'oxygen_concentration': avg_oxygen,
            'operational_cost': avg_cost,
            'time_hours': system_time
        }
        
        # Discretize features
        discrete_state = []
        for feature, value in features.items():
            min_val, max_val = self.feature_ranges[feature]
            normalized = (value - min_val) / (max_val - min_val)
            normalized = np.clip(normalized, 0, 1)
            bin_idx = int(normalized * (self.config.state_bins - 1))
            discrete_state.append(bin_idx)
        
        # Store for trend analysis
        self.state_history.append(features)
        
        return tuple(discrete_state)
    
    def get_state_trends(self) -> Dict[str, float]:
        """Calculate trends in key state variables."""
        if len(self.state_history) < 2:
            return {}
        
        trends = {}
        for feature in ['power_density', 'coulombic_efficiency', 'fouling_thickness']:
            if feature in self.state_history[-1] and feature in self.state_history[0]:
                current = self.state_history[-1][feature]
                past = self.state_history[0][feature]
                trend = (current - past) / max(abs(past), 1e-6)
                trends[f"{feature}_trend"] = trend
        
        return trends


class MultiObjectiveRewardCalculator:
    """Calculates multi-objective rewards for MFC system optimization."""
    
    def __init__(self, config: QLearningConfig):
        self.config = config
        self.baseline_metrics = None
        self.performance_history = deque(maxlen=100)
    
    def calculate_reward(self, current_states: List[MFCSystemState], 
                        previous_states: List[MFCSystemState],
                        action_taken: Dict[str, Any],
                        system_time: float) -> float:
        """
        Calculate multi-objective reward.
        
        Args:
            current_states: Current cell states
            previous_states: Previous cell states  
            action_taken: Action that was executed
            system_time: Current system time
            
        Returns:
            Combined reward value
        """
        if not current_states or not previous_states:
            return 0.0
        
        # Calculate individual objective rewards
        power_reward = self._calculate_power_reward(current_states, previous_states)
        efficiency_reward = self._calculate_efficiency_reward(current_states, previous_states)
        lifetime_reward = self._calculate_lifetime_reward(current_states, previous_states)
        cost_reward = self._calculate_cost_reward(current_states, previous_states)
        
        # Weight and combine objectives
        weights = self.config.objective_weights
        total_reward = (
            weights["power_density"] * power_reward +
            weights["coulombic_efficiency"] * efficiency_reward +
            weights["system_lifetime"] * lifetime_reward +
            weights["operational_cost"] * cost_reward
        )
        
        # Add action-specific bonuses/penalties
        action_reward = self._calculate_action_reward(action_taken, current_states)
        total_reward += action_reward
        
        # Store performance for adaptive learning
        performance_metrics = {
            'power': np.mean([s.power_density for s in current_states]),
            'efficiency': np.mean([s.coulombic_efficiency for s in current_states]),
            'fouling': np.mean([s.fouling_thickness for s in current_states]),
            'cost': np.mean([s.operational_cost for s in current_states])
        }
        self.performance_history.append(performance_metrics)
        
        return total_reward
    
    def _calculate_power_reward(self, current: List[MFCSystemState], 
                               previous: List[MFCSystemState]) -> float:
        """Calculate power generation reward."""
        current_power = np.mean([s.power_density for s in current])
        previous_power = np.mean([s.power_density for s in previous])
        
        # Reward power increase
        power_change = current_power - previous_power
        change_reward = power_change * 100  # Scale factor
        
        # Reward achieving target power
        target_reward = 0
        if current_power >= self.config.power_target:
            target_reward = 50  # Bonus for meeting target
        
        # Penalty for very low power
        if current_power < 0.1:
            penalty = -20
        else:
            penalty = 0
        
        return change_reward + target_reward + penalty
    
    def _calculate_efficiency_reward(self, current: List[MFCSystemState],
                                   previous: List[MFCSystemState]) -> float:
        """Calculate coulombic efficiency reward.""" 
        current_eff = np.mean([s.coulombic_efficiency for s in current])
        previous_eff = np.mean([s.coulombic_efficiency for s in previous])
        
        # Reward efficiency improvement
        eff_change = current_eff - previous_eff
        change_reward = eff_change * 200  # Scale factor
        
        # Reward high efficiency
        if current_eff >= self.config.efficiency_target:
            target_reward = 30
        elif current_eff >= 0.6:
            target_reward = 10
        else:
            target_reward = -10  # Penalty for low efficiency
        
        return change_reward + target_reward
    
    def _calculate_lifetime_reward(self, current: List[MFCSystemState],
                                 previous: List[MFCSystemState]) -> float:
        """Calculate system lifetime reward (anti-fouling/degradation)."""
        current_fouling = np.mean([s.fouling_thickness for s in current])
        previous_fouling = np.mean([s.fouling_thickness for s in previous])
        current_degradation = np.mean([s.degradation_fraction for s in current])
        previous_degradation = np.mean([s.degradation_fraction for s in previous])
        
        # Reward fouling reduction/slow growth
        fouling_change = current_fouling - previous_fouling
        fouling_reward = -fouling_change * 2  # Penalty for fouling increase
        
        # Reward degradation prevention
        degradation_change = current_degradation - previous_degradation
        degradation_reward = -degradation_change * 100  # Strong penalty for degradation
        
        # Penalty for excessive fouling
        if current_fouling > self.config.fouling_threshold:
            fouling_penalty = -50
        else:
            fouling_penalty = 0
        
        return fouling_reward + degradation_reward + fouling_penalty
    
    def _calculate_cost_reward(self, current: List[MFCSystemState],
                             previous: List[MFCSystemState]) -> float:
        """Calculate operational cost reward."""
        current_cost = np.mean([s.operational_cost for s in current])
        previous_cost = np.mean([s.operational_cost for s in previous])
        
        # Reward cost reduction
        cost_change = previous_cost - current_cost  # Positive for cost reduction
        change_reward = cost_change * 100
        
        # Penalty for exceeding cost limit
        if current_cost > self.config.cost_limit:
            cost_penalty = -30
        else:
            cost_penalty = 0
        
        return change_reward + cost_penalty
    
    def _calculate_action_reward(self, action: Dict[str, Any], 
                               states: List[MFCSystemState]) -> float:
        """Calculate action-specific rewards."""
        action_reward = 0
        
        # Cleaning action rewards
        if action.get('cleaning_action') == 'chemical':
            avg_fouling = np.mean([s.fouling_thickness for s in states])
            if avg_fouling > 40:  # Appropriate cleaning
                action_reward += 20
            else:  # Unnecessary cleaning
                action_reward -= 10
        
        # Flow rate optimization
        flow_change = action.get('flow_change', 0)
        if abs(flow_change) > 8:  # Large flow changes
            action_reward -= 5  # Prefer stability
        
        # Temperature optimization  
        temp_change = action.get('temperature_change', 0)
        if abs(temp_change) > 3:  # Large temperature changes
            action_reward -= 3  # Prefer stability
        
        return action_reward


class IntegratedQLearningController:
    """
    Advanced Q-learning controller for integrated MFC system.
    
    Manages multi-objective optimization with hierarchical actions
    and long-term planning capabilities.
    """
    
    def __init__(self, config: QLearningConfig):
        self.config = config
        self.epsilon = config.epsilon_initial
        
        # Initialize components
        self.state_encoder = SystemStateEncoder(config)
        self.reward_calculator = MultiObjectiveRewardCalculator(config)
        
        # Q-learning components
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.action_space = self._create_action_space()
        self.learning_stats = {
            'total_steps': 0,
            'total_reward': 0.0,
            'epsilon_updates': 0,
            'actions_taken': defaultdict(int)
        }
        
        print("🤖 Integrated Q-Learning Controller Initialized")
        print(f"   State space: {config.state_bins}^12 discrete states")
        print(f"   Action space: {len(self.action_space)} combined actions")
        print(f"   Objectives: {list(config.objective_weights.keys())}")
    
    def _create_action_space(self) -> List[Dict[str, Any]]:
        """Create combined action space."""
        actions = []
        
        # Combine flow, substrate, cleaning, and temperature actions
        for flow_change in self.config.flow_actions:
            for substrate_change in self.config.substrate_actions:
                for cleaning_action in self.config.cleaning_actions:
                    for temp_change in self.config.temperature_actions:
                        # Skip some combinations to reduce action space
                        if abs(flow_change) > 5 and abs(substrate_change) > 1:
                            continue  # Avoid extreme combined actions
                        if cleaning_action != "none" and (abs(flow_change) > 2 or abs(substrate_change) > 1):
                            continue  # Cleaning with minimal other changes
                        
                        action = {
                            'flow_change': flow_change,
                            'substrate_change': substrate_change,
                            'cleaning_action': cleaning_action,
                            'temperature_change': temp_change
                        }
                        actions.append(action)
        
        print(f"   Generated {len(actions)} actions from combinations")
        return actions
    
    def select_action(self, states: List[MFCSystemState], system_time: float) -> Dict[str, Any]:
        """
        Select optimal action using epsilon-greedy strategy.
        
        Args:
            states: Current system states
            system_time: Current time
            
        Returns:
            Selected action dictionary
        """
        # Encode current state
        state_key = self.state_encoder.encode_state(states, system_time)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Exploration: random action with intelligent bias
            action_idx = self._intelligent_exploration(states)
        else:
            # Exploitation: best known action
            q_values = [self.q_table[state_key][i] for i in range(len(self.action_space))]
            if max(q_values) == 0:  # Unvisited state
                action_idx = self._intelligent_exploration(states)
            else:
                action_idx = np.argmax(q_values)
        
        selected_action = self.action_space[action_idx]
        
        # Track action selection
        self.learning_stats['actions_taken'][action_idx] += 1
        
        return selected_action, action_idx, state_key
    
    def _intelligent_exploration(self, states: List[MFCSystemState]) -> int:
        """Intelligent exploration based on current system state."""
        avg_fouling = np.mean([s.fouling_thickness for s in states])
        avg_power = np.mean([s.power_density for s in states])
        avg_efficiency = np.mean([s.coulombic_efficiency for s in states])
        
        # Bias exploration based on current needs
        action_weights = np.ones(len(self.action_space))
        
        for i, action in enumerate(self.action_space):
            # Favor cleaning if fouling is high
            if avg_fouling > 40 and action['cleaning_action'] != 'none':
                action_weights[i] *= 3
            
            # Favor substrate increase if efficiency is low
            if avg_efficiency < 0.6 and action['substrate_change'] > 0:
                action_weights[i] *= 2
            
            # Favor flow increase if power is low
            if avg_power < 0.3 and action['flow_change'] > 0:
                action_weights[i] *= 2
            
            # Avoid extreme actions during exploration
            if abs(action['flow_change']) > 8 or abs(action['substrate_change']) > 1.5:
                action_weights[i] *= 0.5
        
        # Weighted random selection
        probabilities = action_weights / np.sum(action_weights)
        return np.random.choice(len(self.action_space), p=probabilities)
    
    def update_q_table(self, state: Tuple, action_idx: int, reward: float, 
                      next_state: Tuple):
        """Update Q-table using Q-learning algorithm."""
        current_q = self.q_table[state][action_idx]
        
        # Calculate max Q-value for next state
        next_q_values = [self.q_table[next_state][i] for i in range(len(self.action_space))]
        max_next_q = max(next_q_values) if next_q_values else 0
        
        # Q-learning update
        new_q = current_q + self.config.learning_rate * (
            reward + self.config.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action_idx] = new_q
        
        # Update learning statistics
        self.learning_stats['total_steps'] += 1
        self.learning_stats['total_reward'] += reward
        
        # Update exploration rate
        self._update_epsilon()
    
    def _update_epsilon(self):
        """Update exploration rate with adaptive decay."""
        # Standard decay
        self.epsilon = max(self.config.epsilon_min, 
                          self.epsilon * self.config.epsilon_decay)
        
        # Adaptive adjustments based on performance
        if len(self.reward_calculator.performance_history) >= 20:
            recent_performance = list(self.reward_calculator.performance_history)[-20:]
            
            # Increase exploration if performance is stagnating
            power_trend = np.polyfit(range(20), [p['power'] for p in recent_performance], 1)[0]
            if abs(power_trend) < 0.001:  # Stagnating
                self.epsilon = min(self.epsilon * 1.1, self.config.epsilon_initial)
                self.learning_stats['epsilon_updates'] += 1
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get learning progress and statistics."""
        q_table_size = len(self.q_table)
        total_q_values = sum(len(actions) for actions in self.q_table.values())
        
        # Calculate action diversity
        action_counts = self.learning_stats['actions_taken']
        if action_counts:
            action_entropy = -sum((count/sum(action_counts.values())) * 
                                np.log(count/sum(action_counts.values()) + 1e-10) 
                                for count in action_counts.values())
        else:
            action_entropy = 0
        
        return {
            'q_table_states': q_table_size,
            'total_q_values': total_q_values,
            'epsilon': self.epsilon,
            'total_steps': self.learning_stats['total_steps'],
            'average_reward': (self.learning_stats['total_reward'] / 
                             max(self.learning_stats['total_steps'], 1)),
            'action_entropy': action_entropy,
            'most_used_actions': dict(sorted(action_counts.items(), 
                                           key=lambda x: x[1], reverse=True)[:5])
        }
    
    def save_model(self, filepath: str):
        """Save Q-learning model."""
        model_data = {
            'config': self.config,
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'learning_stats': self.learning_stats,
            'action_space': self.action_space
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load Q-learning model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
        self.epsilon = model_data['epsilon']
        self.learning_stats = model_data['learning_stats']
        self.action_space = model_data['action_space']


def create_default_qlearning_config() -> QLearningConfig:
    """Create default Q-learning configuration for MFC systems."""
    return QLearningConfig(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_initial=0.3,
        epsilon_decay=0.995,
        objective_weights={
            "power_density": 0.3,
            "coulombic_efficiency": 0.25,
            "system_lifetime": 0.3,
            "operational_cost": 0.15
        }
    )


def demonstrate_qlearning_integration():
    """Demonstrate Q-learning integration with MFC system."""
    print("🤖 Q-Learning Integration Demonstration")
    
    # Create Q-learning configuration
    qconfig = create_default_qlearning_config()
    controller = IntegratedQLearningController(qconfig)
    
    # Create simple MFC system for demo
    from mfc_system_integration import MFCStackParameters
    
    mfc_config = MFCStackParameters(
        n_cells=3,
        enable_qlearning=False  # We'll manage it manually
    )
    
    system = IntegratedMFCSystem(mfc_config)
    
    print("\n📊 Running Q-Learning Demo...")
    
    # Run simulation with Q-learning control
    for step in range(10):
        # Get current state
        current_states = system.cell_states
        
        # Select action
        action, action_idx, state_key = controller.select_action(current_states, system.time)
        
        print(f"Step {step}: Action = {action}")
        
        # Step system (simplified - would apply action in real implementation)
        next_states = system.step_system_dynamics(1.0)
        
        # Calculate reward
        reward = controller.reward_calculator.calculate_reward(
            next_states, current_states, action, system.time
        )
        
        # Encode next state
        next_state_key = controller.state_encoder.encode_state(next_states, system.time)
        
        # Update Q-table
        controller.update_q_table(state_key, action_idx, reward, next_state_key)
        
        print(f"  Reward: {reward:.2f}, Epsilon: {controller.epsilon:.3f}")
    
    # Show learning progress
    progress = controller.get_learning_progress()
    print("\n📈 Learning Progress:")
    for key, value in progress.items():
        print(f"  {key}: {value}")
    
    return controller


if __name__ == "__main__":
    demonstrate_qlearning_integration()