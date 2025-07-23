#!/usr/bin/env python3
"""
MFC Unified Q-Learning Control with Optimized Parameters from Optuna
===================================================================

This is the optimized version of the MFC unified Q-learning controller using
the best parameters found through Optuna hyperparameter optimization.

Optimization Results:
- Trial #37 achieved best objective: 8.991467
- Control RMSE: 4.528 mmol/L
- Energy: 4.952 Wh
- Optimized for 600-hour simulations

Author: Claude & User
Date: 2025-07-23
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
from datetime import datetime
from collections import defaultdict
from typing import Tuple, Dict

# GPU configuration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np  # Fallback to numpy


class OptimizedUnifiedQController:
    """
    Optimized Unified Q-Learning Controller with Optuna-tuned parameters
    Controls both flow rate and substrate concentration simultaneously
    """
    def __init__(self, target_outlet_conc=12.0):
        # Optimized Q-learning parameters from Optuna
        self.learning_rate = 0.0987
        self.discount_factor = 0.9517
        self.epsilon = 0.3702  # Initial exploration rate
        self.epsilon_decay = 0.9978
        self.epsilon_min = 0.1020
        
        # Target concentration
        self.target_outlet_conc = target_outlet_conc
        
        # Initialize Q-table
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Performance tracking
        self.performance_history = []
        self.action_history = []
        
        # OPTIMIZED ACTION SPACE from Optuna
        flow_actions = list(range(-12, 7))  # -12 to +6 mL/h
        substrate_actions = []
        # Coarse substrate adjustments as determined by optimization
        for val in np.arange(-1.046, 1.196, 1.0):
            substrate_actions.append(round(val, 3))
        
        # Create combined action space
        self.actions = []
        for flow_adj in flow_actions:
            for substr_adj in substrate_actions:
                self.actions.append((flow_adj, substr_adj))
        
        # State space discretization (same structure as before)
        self.state_bins = {
            'power': np.linspace(0, 0.03, 10),
            'biofilm_deviation': np.linspace(0, 2.0, 10),
            'substrate_utilization': np.linspace(0, 100, 10),
            'outlet_conc_error': np.linspace(-10, 10, 10),
            'flow_rate': np.linspace(5, 50, 10),
            'time_phase': np.linspace(0, 1000, 10)
        }
        
        # Control bounds
        self.min_substrate = 5.0
        self.max_substrate = 50.0
        
        # Optimized reward parameters from Optuna
        self.reward_params = {
            'biofilm_base_reward': 56.69,
            'biofilm_steady_bonus': 33.12,
            'biofilm_penalty_multiplier': 92.79,
            'power_increase_multiplier': 57.48,
            'power_decrease_multiplier': 80.38,
            'substrate_increase_multiplier': 38.87,
            'substrate_decrease_multiplier': 65.11,
            'conc_precise_reward': 15.60,
            'conc_acceptable_reward': 5.15,
            'conc_poor_penalty': -8.66,
            'flow_penalty_threshold': 22.81,
            'flow_penalty_multiplier': 24.77,
            'biofilm_threshold_ratio': 0.884
        }
        
        print(f"Optimized Unified Q-learning initialized:")
        print(f"- State space dimensions: 6 (extended)")
        print(f"- Action space size: {len(self.actions)} dual actions")
        print(f"- Target outlet concentration: {self.target_outlet_conc:.1f} mmol/L")
        print(f"- Optimization source: Optuna Trial #37")
        
    def discretize_state(self, power, biofilm_deviation, substrate_utilization, 
                        outlet_conc_error, flow_rate, time_hours):
        """Discretize continuous state into bins"""
        def get_bin(value, bins):
            return np.clip(np.digitize(value, bins) - 1, 0, len(bins) - 2)
        
        power_idx = get_bin(power, self.state_bins['power'])
        biofilm_idx = get_bin(biofilm_deviation, self.state_bins['biofilm_deviation'])
        substrate_idx = get_bin(substrate_utilization, self.state_bins['substrate_utilization'])
        error_idx = get_bin(outlet_conc_error, self.state_bins['outlet_conc_error'])
        flow_idx = get_bin(flow_rate, self.state_bins['flow_rate'])
        time_idx = get_bin(time_hours, self.state_bins['time_phase'])
        
        return (power_idx, biofilm_idx, substrate_idx, error_idx, flow_idx, time_idx)
    
    def select_action(self, state, current_flow_rate, current_inlet_conc):
        """Select dual action using epsilon-greedy with optimized parameters"""
        if np.random.random() < self.epsilon:
            # Exploration with bias towards reasonable actions
            if np.random.random() < 0.7:  # 70% chance of reasonable exploration
                # Prefer small adjustments during exploration
                flow_options = [i for i, (f, s) in enumerate(self.actions) if abs(f) <= 4]
                if flow_options:
                    action_idx = np.random.choice(flow_options)
                else:
                    action_idx = np.random.randint(len(self.actions))
            else:
                action_idx = np.random.randint(len(self.actions))
        else:
            # Exploitation: best known action
            q_values = [self.q_table[state][i] for i in range(len(self.actions))]
            action_idx = np.argmax(q_values)
        
        # Extract dual action
        flow_adjustment, substrate_adjustment = self.actions[action_idx]
        
        # Apply flow rate change with bounds (convert to L/h)
        flow_change_lh = flow_adjustment * 1e-3  # mL/h to L/h
        new_flow_rate = np.clip(current_flow_rate + flow_change_lh, 0.005, 0.050)
        
        # Track current flow rate for reward calculation (in mL/h)
        self.current_flow_rate = new_flow_rate * 1000  # L/h to mL/h
        
        # Apply substrate concentration change with bounds
        new_inlet_conc = np.clip(current_inlet_conc + substrate_adjustment, 
                                self.min_substrate, self.max_substrate)
        
        # Store action in history
        self.action_history.append((flow_adjustment, substrate_adjustment))
        if len(self.action_history) > 1000:  # Keep last 1000 actions
            self.action_history.pop(0)
        
        return action_idx, new_flow_rate, new_inlet_conc
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using optimized Q-learning parameters"""
        current_q = self.q_table[state][action]
        next_max_q = max([self.q_table[next_state][a] for a in range(len(self.actions))]) if next_state in self.q_table else 0
        
        # Q-learning update with optimized parameters
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state][action] = new_q
        
        # Update exploration rate with optimized decay
        if len(self.performance_history) > 0 and len(self.performance_history) % 100 == 0:
            avg_recent_reward = np.mean(self.performance_history[-100:])
            
            # Adaptive decay based on performance
            if avg_recent_reward > -50:  # Good performance
                decay_factor = self.epsilon_decay * 0.995
            else:  # Poor performance
                decay_factor = self.epsilon_decay * 1.005  # Slower decay
            
            self.epsilon = max(self.epsilon_min, self.epsilon * decay_factor)
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def calculate_optimized_reward(self, power, biofilm_deviation, substrate_utilization,
                                  outlet_conc, prev_power, prev_biofilm_dev, prev_substrate_util,
                                  prev_outlet_conc, biofilm_thickness_history=None):
        """
        Calculate reward using Optuna-optimized parameters
        """
        # Calculate changes
        power_change = power - prev_power
        substrate_change = substrate_utilization - prev_substrate_util
        error_improvement = abs(prev_outlet_conc - self.target_outlet_conc) - abs(outlet_conc - self.target_outlet_conc)
        
        # 1. POWER COMPONENT with optimized multipliers
        if power_change > 0:
            power_reward = power_change * self.reward_params['power_increase_multiplier']
        else:
            power_reward = power_change * self.reward_params['power_decrease_multiplier']
        
        power_base = 10.0 if power > 0.005 else -5.0
        
        # 2. SUBSTRATE COMPONENT with optimized multipliers
        if substrate_change > 0:
            substrate_reward = substrate_change * self.reward_params['substrate_increase_multiplier']
        else:
            substrate_reward = substrate_change * self.reward_params['substrate_decrease_multiplier']
        
        substrate_base = 5.0 if substrate_utilization > 10.0 else -2.0
        
        # 3. BIOFILM COMPONENT with optimized parameters
        optimal_thickness = 1.3
        deviation_threshold = 0.05 * optimal_thickness
        
        if biofilm_deviation <= deviation_threshold:
            biofilm_reward = self.reward_params['biofilm_base_reward'] - (biofilm_deviation / deviation_threshold) * 15.0
            
            if biofilm_thickness_history is not None and len(biofilm_thickness_history) >= 3:
                recent_thickness = biofilm_thickness_history[-3:]
                if len(recent_thickness) >= 2:
                    growth_rate = abs(recent_thickness[-1] - recent_thickness[-2])
                    if growth_rate < 0.01:
                        biofilm_reward += self.reward_params['biofilm_steady_bonus']
        else:
            excess_deviation = biofilm_deviation - deviation_threshold
            biofilm_reward = -self.reward_params['biofilm_penalty_multiplier'] * (excess_deviation / deviation_threshold)
        
        # 4. CONCENTRATION CONTROL with optimized parameters
        outlet_error = abs(outlet_conc - self.target_outlet_conc)
        
        if outlet_error <= 0.5:
            concentration_reward = self.reward_params['conc_precise_reward'] - (outlet_error * 10.0)
        elif outlet_error <= 2.0:
            concentration_reward = self.reward_params['conc_acceptable_reward'] - (outlet_error * 2.5)
        else:
            concentration_reward = self.reward_params['conc_poor_penalty'] - (outlet_error * 5.0)
        
        concentration_base = 3.0 if error_improvement > 0 else -1.0
        
        # 5. STABILITY BONUS
        stability_bonus = 0
        if (abs(power_change) < 0.001 and abs(substrate_change) < 0.5 and 
            outlet_error < 1.0 and biofilm_deviation <= deviation_threshold):
            stability_bonus = 30.0
        
        # 6. FLOW PENALTY with optimized thresholds
        flow_penalty = 0
        current_flow_rate = getattr(self, 'current_flow_rate', 10.0)
        if biofilm_thickness_history is not None and len(biofilm_thickness_history) > 0:
            avg_biofilm = np.mean(biofilm_thickness_history[-5:])
            if avg_biofilm < optimal_thickness * self.reward_params['biofilm_threshold_ratio']:
                if current_flow_rate > self.reward_params['flow_penalty_threshold']:
                    flow_penalty = -self.reward_params['flow_penalty_multiplier'] * \
                                  (current_flow_rate - self.reward_params['flow_penalty_threshold']) / 10.0
        
        # 7. COMBINED PENALTY
        combined_penalty = 0
        if (power_change < 0 and substrate_change < 0 and 
            error_improvement < 0 and biofilm_deviation > deviation_threshold):
            combined_penalty = -200.0
        
        # Total reward
        total_reward = (power_reward + power_base + 
                       substrate_reward + substrate_base + 
                       biofilm_reward + 
                       concentration_reward + concentration_base +
                       stability_bonus + flow_penalty + combined_penalty)
        
        # Track performance
        self.performance_history.append(total_reward)
        if len(self.performance_history) > 1000:
            self.performance_history.pop(0)
        
        return total_reward
    
    def get_control_statistics(self):
        """Get comprehensive control performance statistics"""
        if len(self.performance_history) == 0:
            return {'avg_reward': 0, 'reward_trend': 0, 'exploration_rate': self.epsilon}
        
        recent_rewards = self.performance_history[-50:] if len(self.performance_history) >= 50 else self.performance_history
        
        # Calculate trend
        if len(recent_rewards) >= 10:
            x = np.arange(len(recent_rewards))
            z = np.polyfit(x, recent_rewards, 1)
            trend = z[0]  # Slope of the trend line
        else:
            trend = 0
        
        return {
            'avg_reward': np.mean(recent_rewards),
            'reward_trend': trend,
            'exploration_rate': self.epsilon,
            'q_table_size': len(self.q_table),
            'total_actions': len(self.action_history)
        }


# Import the base simulation class
from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation


class OptimizedMFCSimulation(MFCUnifiedQLearningSimulation):
    """
    MFC Simulation with Optuna-optimized Q-learning controller
    """
    def __init__(self, use_gpu=True, target_outlet_conc=12.0):
        """Initialize simulation with optimized controller"""
        # Initialize base class
        super().__init__(use_gpu, target_outlet_conc)
        
        # Replace the controller with optimized version
        self.unified_controller = OptimizedUnifiedQController(target_outlet_conc)
        
        # Override the reward calculation method
        self.unified_controller.calculate_unified_reward = self.unified_controller.calculate_optimized_reward
        
        print("\n=== MFC Simulation with Optuna-Optimized Parameters ===")
        print("Based on optimization results from Trial #37")
        print("Optimized for minimal control error and stable operation")
        print("=======================================================\n")


def main():
    """Run optimized MFC simulation"""
    print("MFC Unified Q-Learning with Optuna-Optimized Parameters")
    print("=" * 60)
    
    # Create and run optimized simulation
    sim = OptimizedMFCSimulation(use_gpu=False, target_outlet_conc=12.0)
    sim.run_simulation()
    
    print("\n=== OPTIMIZED SIMULATION COMPLETE ===")
    print("Results saved with optimized parameters from Optuna")
    print("Compare with baseline to see improvements!")


if __name__ == "__main__":
    main()