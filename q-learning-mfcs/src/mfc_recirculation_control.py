#!/usr/bin/env python3
"""
MFC Stack with Anolyte Recirculation and Advanced Substrate Control
Enhanced Q-learning agent with:
1. Individual cell substrate concentration monitoring
2. Feedback loop for reservoir substrate concentration control
3. 1L anolyte reservoir simulation with recirculation
4. Multi-sensor feedback control system
"""

import numpy as np
import pandas as pd
import json
import argparse
import logging
import sys
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from path_config import get_simulation_data_path

class AnolytereservoirSystem:
    """Simulates 1L anolyte reservoir with recirculation and substrate control"""
    
    def __init__(self, initial_substrate_conc=20.0, volume_liters=1.0):
        self.volume = volume_liters  # L
        self.substrate_concentration = initial_substrate_conc  # mmol/L
        self.total_substrate_added = 0.0  # mmol
        self.total_volume_circulated = 0.0  # L
        
        # Substrate addition control
        self.substrate_addition_rate = 0.0  # mmol/h
        self.max_substrate_addition = 100.0  # mmol/h
        self.substrate_halt = False
        
        # Mixing dynamics
        self.mixing_time_constant = 0.1  # hours for complete mixing
        
        # Recirculation system parameters
        self.pump_efficiency = 0.95  # Pump efficiency
        self.pipe_dead_volume = 0.05  # L dead volume in pipes
        self.heat_loss_coefficient = 0.02  # Temperature effects
        
        # Advanced tracking
        self.circulation_cycles = 0
        self.total_pump_time = 0.0  # hours
        self.substrate_balance_history = []
        self.mixing_efficiency_history = []
        
    def add_substrate(self, amount_mmol, dt_hours):
        """Add substrate to reservoir"""
        if not self.substrate_halt:
            substrate_added = amount_mmol * dt_hours
            self.total_substrate_added += substrate_added
            
            # Update concentration considering mixing
            new_total_substrate = (self.substrate_concentration * self.volume + substrate_added)
            self.substrate_concentration = new_total_substrate / self.volume
    
    def circulate_anolyte(self, flow_rate_ml_h, stack_outlet_conc, dt_hours):
        """Enhanced anolyte recirculation simulation with detailed system modeling"""
        flow_rate_l_h = flow_rate_ml_h / 1000.0
        
        # Account for pump efficiency and dead volume effects
        effective_flow_rate = flow_rate_l_h * self.pump_efficiency
        volume_returned = effective_flow_rate * dt_hours
        
        # Track pump operation time
        if flow_rate_ml_h > 0:
            self.total_pump_time += dt_hours
            self.circulation_cycles += 1
        
        # Track circulation
        self.total_volume_circulated += volume_returned
        
        # Enhanced mixing model with dead volume effects
        if volume_returned > 0:
            # Calculate effective mixing considering dead volume
            effective_volume_fraction = volume_returned / (self.volume + self.pipe_dead_volume)
            
            # Multi-stage mixing model (more realistic)
            # Stage 1: Initial contact mixing
            initial_mixing_factor = min(1.0, effective_volume_fraction * 2.0)
            
            # Stage 2: Exponential approach to equilibrium
            time_mixing_factor = 1.0 - np.exp(-dt_hours / self.mixing_time_constant)
            
            # Stage 3: Temperature and density effects
            density_factor = 1.0 - self.heat_loss_coefficient * dt_hours
            
            # Combined mixing efficiency
            overall_mixing_efficiency = initial_mixing_factor * time_mixing_factor * density_factor
            
            # Calculate concentration change
            concentration_difference = stack_outlet_conc - self.substrate_concentration
            concentration_change = concentration_difference * overall_mixing_efficiency * effective_volume_fraction
            
            # Apply concentration change with bounds checking
            old_concentration = self.substrate_concentration
            self.substrate_concentration = max(0.0, self.substrate_concentration + concentration_change)
            
            # Track substrate balance and mixing efficiency
            substrate_balance = {
                'time': self.total_pump_time,
                'inlet_conc': stack_outlet_conc,
                'reservoir_conc_before': old_concentration,
                'reservoir_conc_after': self.substrate_concentration,
                'volume_returned': volume_returned,
                'concentration_change': concentration_change
            }
            self.substrate_balance_history.append(substrate_balance)
            
            mixing_efficiency = {
                'time': self.total_pump_time,
                'initial_mixing': initial_mixing_factor,
                'time_mixing': time_mixing_factor,
                'density_factor': density_factor,
                'overall_efficiency': overall_mixing_efficiency
            }
            self.mixing_efficiency_history.append(mixing_efficiency)
    
    def get_inlet_concentration(self):
        """Get current substrate concentration for stack inlet"""
        return self.substrate_concentration
    
    def get_sensor_readings(self):
        """Get comprehensive sensor readings from reservoir system"""
        return {
            'substrate_concentration': self.substrate_concentration,
            'total_substrate_added': self.total_substrate_added,
            'total_volume_circulated': self.total_volume_circulated,
            'circulation_cycles': self.circulation_cycles,
            'pump_operation_time': self.total_pump_time,
            'current_mixing_efficiency': (self.mixing_efficiency_history[-1]['overall_efficiency'] 
                                        if self.mixing_efficiency_history else 1.0),
            'substrate_addition_active': not self.substrate_halt
        }

class SubstrateConcentrationController:
    """Advanced feedback controller for substrate concentration management"""
    
    def __init__(self, target_outlet_conc=12.0, target_reservoir_conc=20.0):
        self.target_outlet_conc = target_outlet_conc
        self.target_reservoir_conc = target_reservoir_conc
        
        # PID parameters for outlet concentration control (reduced gains for lactate)
        self.kp_outlet = 0.5  # Reduced from 2.0
        self.ki_outlet = 0.01  # Reduced from 0.1
        self.kd_outlet = 0.1  # Reduced from 0.5
        self.outlet_error_integral = 0.0
        self.previous_outlet_error = 0.0
        
        # PID parameters for reservoir concentration control (reduced gains)
        self.kp_reservoir = 0.2  # Reduced from 1.0
        self.ki_reservoir = 0.005  # Reduced from 0.05
        self.kd_reservoir = 0.05  # Reduced from 0.2
        self.reservoir_error_integral = 0.0
        self.previous_reservoir_error = 0.0
        
        # Control limits (reduced maximum addition rate)
        self.min_addition_rate = 0.0  # mmol/h
        self.max_addition_rate = 5.0  # Reduced from 50.0 mmol/h
        
        # Substrate halt conditions (tighter control)
        self.halt_threshold = 0.2  # Reduced from 0.5 mmol/L decline threshold
        self.previous_outlet_conc = None
        
        # Enhanced feedback control parameters (tighter thresholds)
        self.starvation_threshold_critical = 2.0  # mmol/L
        self.starvation_threshold_warning = 5.0  # mmol/L
        self.excess_threshold = 22.0  # Reduced from 25.0 mmol/L - halt addition if too high
        
        # Adaptive control gains
        self.control_mode = "normal"  # normal, emergency, conservation
        self.control_history = []
        
    def calculate_substrate_addition(self, outlet_conc, reservoir_conc, cell_concentrations, reservoir_sensors, dt_hours):
        """Calculate substrate addition rate based on multi-sensor feedback"""
        
        # 1. Outlet concentration PID control
        outlet_error = self.target_outlet_conc - outlet_conc
        self.outlet_error_integral += outlet_error * dt_hours
        
        if self.previous_outlet_error is not None:
            outlet_error_derivative = (outlet_error - self.previous_outlet_error) / dt_hours
        else:
            outlet_error_derivative = 0.0
        
        outlet_control = (self.kp_outlet * outlet_error + 
                         self.ki_outlet * self.outlet_error_integral + 
                         self.kd_outlet * outlet_error_derivative)
        
        # 2. Reservoir concentration PID control
        reservoir_error = self.target_reservoir_conc - reservoir_conc
        self.reservoir_error_integral += reservoir_error * dt_hours
        
        if self.previous_reservoir_error is not None:
            reservoir_error_derivative = (reservoir_error - self.previous_reservoir_error) / dt_hours
        else:
            reservoir_error_derivative = 0.0
        
        reservoir_control = (self.kp_reservoir * reservoir_error + 
                            self.ki_reservoir * self.reservoir_error_integral + 
                            self.kd_reservoir * reservoir_error_derivative)
        
        # 3. Cell concentration monitoring (prevent starvation)
        min_cell_conc = min(cell_concentrations)
        starvation_factor = 1.0
        if min_cell_conc < 5.0:  # Starvation threshold
            starvation_factor = 2.0  # Boost addition
        elif min_cell_conc < 2.0:  # Critical starvation
            starvation_factor = 5.0  # Emergency boost
        
        # 4. Enhanced halt conditions and adaptive control mode
        halt_addition = False
        
        # Check for declining outlet concentration
        if self.previous_outlet_conc is not None:
            outlet_decline = self.previous_outlet_conc - outlet_conc
            if outlet_decline > self.halt_threshold:
                halt_addition = True
        
        # Check for excess concentration (prevent waste)
        if reservoir_conc > self.excess_threshold:
            halt_addition = True
            self.control_mode = "conservation"
        
        # Adaptive control mode selection
        if min_cell_conc < self.starvation_threshold_critical:
            self.control_mode = "emergency"
            starvation_factor *= 3.0  # Triple the addition rate
        elif min_cell_conc < self.starvation_threshold_warning:
            self.control_mode = "warning"
            starvation_factor *= 1.5  # Increase addition rate
        else:
            self.control_mode = "normal"
        
        # 5. Combine control signals with adaptive gains
        if self.control_mode == "emergency":
            # Emergency mode: prioritize cell feeding
            base_addition_rate = max(outlet_control, reservoir_control) * starvation_factor
        elif self.control_mode == "conservation":
            # Conservation mode: reduce addition
            base_addition_rate = min(outlet_control, reservoir_control) * 0.5
        else:
            # Normal mode: balanced control
            base_addition_rate = (outlet_control + reservoir_control) * starvation_factor
        
        # Apply limits and halt condition
        if halt_addition:
            addition_rate = 0.0
        else:
            addition_rate = np.clip(base_addition_rate, self.min_addition_rate, self.max_addition_rate)
        
        # 6. Integrate reservoir sensor feedback for advanced control
        mixing_efficiency = reservoir_sensors['current_mixing_efficiency']
        circulation_cycles = reservoir_sensors['circulation_cycles']
        pump_time = reservoir_sensors['pump_operation_time']
        
        # Adjust addition rate based on mixing efficiency
        if mixing_efficiency < 0.7:  # Poor mixing detected
            addition_rate *= 0.8  # Reduce rate to prevent stratification
        elif mixing_efficiency > 0.95:  # Excellent mixing
            addition_rate *= 1.1  # Can increase rate safely
        
        # Consider circulation dynamics
        if circulation_cycles > 0:
            circulation_factor = min(1.2, 1.0 + 0.1 * np.log(circulation_cycles + 1))
            addition_rate *= circulation_factor
        
        # Log comprehensive control decision with sensor data
        control_decision = {
            'mode': self.control_mode,
            'outlet_error': outlet_error,
            'reservoir_error': reservoir_error,
            'min_cell_conc': min_cell_conc,
            'addition_rate': addition_rate,
            'halt': halt_addition,
            'reservoir_sensors': reservoir_sensors.copy(),
            'mixing_efficiency': mixing_efficiency,
            'circulation_factor': circulation_cycles,
            'pump_operation_time': pump_time
        }
        self.control_history.append(control_decision)
        
        # Update previous values
        self.previous_outlet_error = outlet_error
        self.previous_reservoir_error = reservoir_error
        self.previous_outlet_conc = outlet_conc
        
        return addition_rate, halt_addition

class AdvancedQLearningFlowController:
    """Enhanced Q-Learning controller with recirculation and substrate monitoring"""
    
    def __init__(self, learning_rate=0.0987, discount_factor=0.9517, epsilon=0.3702, config=None):
        """Enhanced Q-Learning controller with configurable substrate control"""
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = config.epsilon_decay if config else 0.9995
        self.epsilon_min = config.epsilon_min if config else 0.01
        
        # Configuration for substrate control
        self.config = config or self._default_config()
        
        # Q-table stored as nested dictionary
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Enhanced state and action spaces
        self.setup_enhanced_state_action_spaces()
        
        # Statistics
        self.total_rewards = 0
        self.episode_count = 0
        
        # Previous state for learning
        self.previous_state = None
        self.previous_action = None
        
    def _default_config(self):
        """Default configuration if none provided"""
        from config.qlearning_config import DEFAULT_QLEARNING_CONFIG
        return DEFAULT_QLEARNING_CONFIG
        
    def setup_enhanced_state_action_spaces(self):
        """Enhanced state space including reservoir and cell concentrations with substrate control"""
        # Enhanced state variables: [power, biofilm_deviation, substrate_utilization, 
        #                           reservoir_conc, min_cell_conc, outlet_conc_error, time_phase]
        self.power_bins = np.linspace(0, 2.0, 8)
        self.biofilm_bins = np.linspace(0, 1.0, 8)
        self.substrate_bins = np.linspace(0, 50, 8)
        self.reservoir_conc_bins = np.linspace(5, 30, 6)
        self.cell_conc_bins = np.linspace(0, 25, 6)
        self.outlet_error_bins = np.linspace(0, 15, 6)
        self.time_bins = np.array([200, 500, 800, 1000])
        
        # Enhanced action space: combined flow rate and substrate addition
        self.flow_actions = np.array(self.config.flow_rate_actions)
        self.substrate_actions = np.array(self.config.substrate_actions)
        
        # Combined action space: (flow_action_idx, substrate_action_idx)
        self.total_actions = len(self.flow_actions) * len(self.substrate_actions)
        
    def discretize_enhanced_state(self, power, biofilm_deviation, substrate_utilization, 
                                 reservoir_conc, min_cell_conc, outlet_error, time_hours):
        """Convert continuous enhanced state to discrete state key"""
        power_idx = np.clip(np.digitize(power, self.power_bins) - 1, 0, len(self.power_bins) - 2)
        biofilm_idx = np.clip(np.digitize(biofilm_deviation, self.biofilm_bins) - 1, 0, len(self.biofilm_bins) - 2)
        substrate_idx = np.clip(np.digitize(substrate_utilization, self.substrate_bins) - 1, 0, len(self.substrate_bins) - 2)
        reservoir_idx = np.clip(np.digitize(reservoir_conc, self.reservoir_conc_bins) - 1, 0, len(self.reservoir_conc_bins) - 2)
        cell_idx = np.clip(np.digitize(min_cell_conc, self.cell_conc_bins) - 1, 0, len(self.cell_conc_bins) - 2)
        outlet_idx = np.clip(np.digitize(outlet_error, self.outlet_error_bins) - 1, 0, len(self.outlet_error_bins) - 2)
        time_idx = np.clip(np.digitize(time_hours, self.time_bins) - 1, 0, len(self.time_bins) - 2)
        
        return (power_idx, biofilm_idx, substrate_idx, reservoir_idx, cell_idx, outlet_idx, time_idx)
    
    def calculate_substrate_reward(self, reservoir_conc, cell_concentrations, outlet_conc, substrate_addition, inlet_conc=None):
        """Calculate reward for substrate control based on sensor readings with enhanced outlet sensor logic"""
        reward = 0.0
        
        # Reward for maintaining target concentrations
        reservoir_error = abs(reservoir_conc - self.config.substrate_target_reservoir)
        if reservoir_error < 2.0:  # Within 2 mM of target
            reward += self.config.reward_weights.substrate_target_reward * (1.0 - reservoir_error / 2.0)
        
        # Enhanced outlet sensor reward/penalty system
        outlet_error = abs(outlet_conc - self.config.outlet_reward_threshold)
        base_outlet_reward = 0.0
        
        if outlet_error < 2.0:  # Within 2 mM of user-configurable threshold
            # Proportional reward when approaching threshold
            proximity_factor = (1.0 - outlet_error / 2.0)
            base_outlet_reward = self.config.reward_weights.substrate_target_reward * proximity_factor * self.config.outlet_reward_scaling
        
        # Check for outlet sensor penalty condition (outlet equals inlet)
        if inlet_conc is not None:
            concentration_difference = abs(inlet_conc - outlet_conc)
            if concentration_difference < 0.5:  # Outlet approximately equals inlet (< 0.5 mM difference)
                # Apply 15% penalty increase
                if base_outlet_reward > 0:
                    base_outlet_reward *= (1.0 - (self.config.outlet_penalty_multiplier - 1.0))  # Reduce reward
                else:
                    # Apply penalty
                    penalty = self.config.reward_weights.substrate_target_reward * 0.5 * self.config.outlet_penalty_multiplier
                    base_outlet_reward = -penalty
        
        reward += base_outlet_reward
        
        # Reward for each cell maintaining target concentration
        for cell_conc in cell_concentrations:
            cell_error = abs(cell_conc - self.config.substrate_target_cell)
            if cell_error < 3.0:  # Within 3 mM of target
                reward += self.config.reward_weights.substrate_target_reward * 0.2 * (1.0 - cell_error / 3.0)
        
        # Configurable exponential penalties for exceeding thresholds
        if reservoir_conc > self.config.substrate_max_threshold:
            excess = reservoir_conc - self.config.substrate_max_threshold
            # Exponential penalty that grows with configurable exponent
            penalty_multiplier = self.config.substrate_penalty_base_multiplier + (excess / self.config.substrate_max_threshold)**self.config.substrate_excess_penalty_exponent
            reward += self.config.reward_weights.substrate_excess_penalty * excess * penalty_multiplier
        
        if outlet_conc > self.config.substrate_max_threshold:
            excess = outlet_conc - self.config.substrate_max_threshold
            penalty_multiplier = self.config.substrate_penalty_base_multiplier + (excess / self.config.substrate_max_threshold)**self.config.substrate_excess_penalty_exponent
            reward += self.config.reward_weights.substrate_excess_penalty * excess * penalty_multiplier
        
        # Configurable severe penalty for very high concentrations
        if reservoir_conc > self.config.substrate_severe_threshold:
            severe_excess = reservoir_conc - self.config.substrate_severe_threshold
            reward += -self.config.substrate_severe_penalty_multiplier * severe_excess
        
        if outlet_conc > self.config.substrate_severe_threshold:
            severe_excess = outlet_conc - self.config.substrate_severe_threshold
            reward += -self.config.substrate_severe_penalty_multiplier * severe_excess
        
        # Penalties for starvation
        min_cell_conc = min(cell_concentrations) if cell_concentrations else outlet_conc
        if min_cell_conc < self.config.substrate_min_threshold:
            starvation = self.config.substrate_min_threshold - min_cell_conc
            reward += self.config.reward_weights.substrate_starvation_penalty * starvation
        
        # Penalty for substrate addition (encourages efficiency)
        if substrate_addition > 0:
            reward += self.config.reward_weights.substrate_addition_penalty * substrate_addition
        
        return reward
    
    def choose_combined_action(self, state):
        """Choose combined flow and substrate action using epsilon-greedy"""
        if np.random.random() < self.epsilon:
            # Random action
            action_idx = np.random.randint(0, self.total_actions)
        else:
            # Greedy action
            q_values = [self.q_table[state][a] for a in range(self.total_actions)]
            action_idx = np.argmax(q_values)
        
        # Convert single action index to flow and substrate actions
        flow_idx = action_idx // len(self.substrate_actions)
        substrate_idx = action_idx % len(self.substrate_actions)
        
        return action_idx, flow_idx, substrate_idx
    
    def update_q_value(self, state, action_idx, reward, next_state):
        """Update Q-value using Q-learning rule"""
        if next_state is not None:
            max_next_q = max([self.q_table[next_state][a] for a in range(self.total_actions)])
        else:
            max_next_q = 0.0
        
        current_q = self.q_table[state][action_idx]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action_idx] = new_q
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_checkpoint(self, filepath):
        """Save Q-learning model checkpoint to file"""
        # Convert defaultdict Q-table to regular dict for JSON serialization
        q_table_dict = {}
        for state_key, actions in self.q_table.items():
            # Convert state tuple to string for JSON compatibility
            state_str = str(state_key)
            q_table_dict[state_str] = dict(actions)
        
        checkpoint = {
            'model_info': {
                'checkpoint_type': 'Q-Learning Controller',
                'created_at': datetime.now().isoformat(),
                'algorithm': 'Q-Learning with Epsilon-Greedy',
                'version': '1.0'
            },
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'initial_epsilon': 0.3702,  # Store initial value
                'current_epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min
            },
            'state_action_space': {
                'power_bins': len(self.power_bins),
                'biofilm_bins': len(self.biofilm_bins),
                'substrate_bins': len(self.substrate_bins),
                'reservoir_conc_bins': len(self.reservoir_conc_bins),
                'cell_conc_bins': len(self.cell_conc_bins),
                'outlet_error_bins': len(self.outlet_error_bins),
                'time_bins': len(self.time_bins),
                'total_actions': self.total_actions,
                'flow_actions': self.flow_actions.tolist(),
                'substrate_actions': self.substrate_actions.tolist()
            },
            'training_statistics': {
                'total_rewards': self.total_rewards,
                'episode_count': self.episode_count,
                'q_table_size': len(self.q_table),
                'states_explored': len([state for state in self.q_table if any(self.q_table[state].values())]),
                'total_q_updates': sum(len(actions) for actions in self.q_table.values())
            },
            'q_table': q_table_dict,
            'configuration': {
                'substrate_target_reservoir': self.config.substrate_target_reservoir,
                'substrate_target_outlet': self.config.substrate_target_outlet,
                'substrate_target_cell': self.config.substrate_target_cell,
                'substrate_max_threshold': self.config.substrate_max_threshold,
                'substrate_min_threshold': self.config.substrate_min_threshold,
                'substrate_addition_max': self.config.substrate_addition_max
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        return checkpoint
    
    def load_checkpoint(self, filepath):
        """Load Q-learning model checkpoint from file"""
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        # Restore hyperparameters
        self.learning_rate = checkpoint['hyperparameters']['learning_rate']
        self.discount_factor = checkpoint['hyperparameters']['discount_factor']
        self.epsilon = checkpoint['hyperparameters']['current_epsilon']
        self.epsilon_decay = checkpoint['hyperparameters']['epsilon_decay']
        self.epsilon_min = checkpoint['hyperparameters']['epsilon_min']
        
        # Restore training statistics
        self.total_rewards = checkpoint['training_statistics']['total_rewards']
        self.episode_count = checkpoint['training_statistics']['episode_count']
        
        # Restore Q-table
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state_str, actions in checkpoint['q_table'].items():
            # Convert string back to tuple
            state_key = eval(state_str)  # Note: eval is used here for tuple conversion
            for action_idx, q_value in actions.items():
                self.q_table[state_key][int(action_idx)] = q_value
        
        return checkpoint

class MFCCellWithMonitoring:
    """Individual MFC cell with enhanced substrate monitoring"""
    
    def __init__(self, cell_id, initial_biofilm=1.0):
        self.cell_id = cell_id
        self.biofilm_thickness = initial_biofilm
        self.substrate_concentration = 0.0  # Current cell substrate concentration
        self.inlet_concentration = 0.0      # Concentration entering this cell
        self.outlet_concentration = 0.0     # Concentration leaving this cell
        
        # Cell-specific parameters (LITERATURE-VALIDATED)
        self.max_reaction_rate = 0.15  # Increased based on literature
        self.biofilm_growth_rate = 0.05  # Increased 50x based on Shewanella literature (0.825 h⁻¹ max)
        self.optimal_biofilm_thickness = 1.3
        self.anode_overpotential = 0.1  # Default anode overpotential (V)
        self.current = 0.0  # Cell current (A)
        self.voltage = 0.7  # Cell voltage (V)
        
        # Monitoring data
        self.concentration_history = []
        self.consumption_rate_history = []
        
    def update_concentrations(self, inlet_conc, flow_rate_ml_h, dt_hours):
        """Update substrate concentrations through this cell"""
        self.inlet_concentration = inlet_conc
        
        # Calculate residence time
        cell_volume_ml = 50.0  # ml per cell
        if flow_rate_ml_h > 0:
            residence_time_h = cell_volume_ml / flow_rate_ml_h
        else:
            residence_time_h = 100.0  # Very long residence time for no flow
        
        # Substrate consumption with literature-based diffusion limitations
        # Get biofilm physics parameters from configuration
        from config.qlearning_config import DEFAULT_QLEARNING_CONFIG
        biofilm_params = DEFAULT_QLEARNING_CONFIG.biofilm_physics
        
        # Monod kinetics with diffusion limitation (Stewart & Franklin 2008)
        # Convert thickness from μm to m for calculations
        thickness_m = self.biofilm_thickness * 1e-6
        
        # Effective diffusivity in biofilm
        D_eff = biofilm_params.effective_diffusivity  # m²/s
        
        # Thiele modulus: ratio of reaction rate to diffusion rate
        # φ = L * sqrt(k_max / (D_eff * (Ks + S)))
        k_max = biofilm_params.max_specific_growth_rate / 3600  # Convert h⁻¹ to s⁻¹
        Ks = biofilm_params.half_saturation_constant  # mM
        substrate_conc_mol_m3 = inlet_conc  # Approximate conversion for calculation
        
        thiele_modulus = thickness_m * (k_max / (D_eff * (Ks + substrate_conc_mol_m3)))**0.5
        
        # Effectiveness factor (Fogler 2006, Chapter 12)
        if thiele_modulus < 0.3:
            effectiveness_factor = 1.0 - thiele_modulus**2 / 6.0
        else:
            effectiveness_factor = (3.0 / thiele_modulus) * (1.0 / np.tanh(thiele_modulus) - 1.0 / thiele_modulus)
        
        # Biofilm activity considering both thickness and diffusion limitations
        biofilm_activity = (self.biofilm_thickness / biofilm_params.diffusion_length_scale) * effectiveness_factor
        
        # Maximum consumption rate with Monod kinetics
        max_consumption = (self.max_reaction_rate * biofilm_activity * inlet_conc * residence_time_h) / (Ks + inlet_conc)
        
        # Actual consumption limited by substrate availability
        substrate_consumed = min(max_consumption, inlet_conc * 0.95)  # Max 95% consumption per cell
        
        # Update outlet concentration
        self.outlet_concentration = max(0.0, inlet_conc - substrate_consumed)
        self.substrate_concentration = (inlet_conc + self.outlet_concentration) / 2.0
        
        # Update biofilm based on substrate availability using Monod kinetics
        # Monod growth kinetics: μ = μ_max * S / (Ks + S)
        # Limit substrate concentration for numerical stability (max 100 mM for calculation)
        limited_substrate_conc = min(self.substrate_concentration, 100.0)
        monod_growth_rate = (biofilm_params.max_specific_growth_rate * limited_substrate_conc) / (biofilm_params.half_saturation_constant + limited_substrate_conc)
        
        # Growth and decay with literature parameters - add numerical stability checks
        net_growth_rate = monod_growth_rate - biofilm_params.decay_rate
        
        # Prevent exponential explosion by limiting growth rate per time step
        max_change_per_step = 0.1  # Maximum 10% change per time step
        if abs(net_growth_rate * dt_hours) > max_change_per_step:
            net_growth_rate = np.sign(net_growth_rate) * max_change_per_step / dt_hours
        
        biofilm_change = net_growth_rate * self.biofilm_thickness * dt_hours
        
        # Natural biofilm growth/decay with diffusion-limited equilibrium
        # Add maximum reasonable thickness based on diffusion limitations
        max_diffusion_thickness = biofilm_params.diffusion_length_scale * 2.0  # 200 μm max
        
        new_thickness = self.biofilm_thickness + biofilm_change
        self.biofilm_thickness = np.clip(new_thickness, biofilm_params.minimum_thickness, max_diffusion_thickness)
        
        # Store monitoring data
        self.concentration_history.append(self.substrate_concentration)
        self.consumption_rate_history.append(substrate_consumed / dt_hours if dt_hours > 0 else 0)
        
        return self.outlet_concentration
    
    def process_with_monitoring(self, inlet_conc, flow_rate_ml_h, dt_hours):
        """Process cell with monitoring - alias for update_concentrations"""
        return self.update_concentrations(inlet_conc, flow_rate_ml_h, dt_hours)

def simulate_mfc_with_recirculation(duration_hours=100, config=None, checkpoint_path=None):
    """Main simulation function with recirculation and advanced substrate control
    
    Args:
        duration_hours: Simulation duration in hours
        config: Optional configuration object
        checkpoint_path: Optional path to Q-learning checkpoint to load
    
    Returns:
        results: Dictionary containing time series data
        cells: List of MFC cell objects
        reservoir: Reservoir system object
        controller: Substrate controller object
        q_controller: Q-learning controller object
    """
    
    # Initialize components with configurable parameters
    if config is None:
        from config.qlearning_config import DEFAULT_QLEARNING_CONFIG
        config = DEFAULT_QLEARNING_CONFIG
    
    reservoir = AnolytereservoirSystem(
        initial_substrate_conc=config.initial_substrate_concentration, 
        volume_liters=config.reservoir_volume_liters
    )
    controller = SubstrateConcentrationController(
        target_outlet_conc=config.substrate_target_outlet, 
        target_reservoir_conc=config.substrate_target_reservoir
    )
    q_controller = AdvancedQLearningFlowController(config=config)
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading Q-learning checkpoint from: {checkpoint_path}")
        checkpoint_info = q_controller.load_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint with {checkpoint_info['training_statistics']['q_table_size']} states")
        print(f"Continuing from epsilon: {q_controller.epsilon:.4f}")
    
    # Initialize MFC cells
    n_cells = 5
    cells = [MFCCellWithMonitoring(i+1, initial_biofilm=1.0) for i in range(n_cells)]
    
    # Simulation parameters (configurable duration)
    dt_hours = 10.0 / 3600.0  # 10 seconds
    n_steps = int(duration_hours / dt_hours)
    
    # Initial conditions
    flow_rate_ml_h = 10.0
    
    # Data storage
    results = {
        'time_hours': [],
        'reservoir_concentration': [],
        'cell_concentrations': [],
        'outlet_concentration': [],
        'flow_rate': [],
        'substrate_addition_rate': [],
        'total_power': [],
        'biofilm_thicknesses': [],
        'substrate_halt': [],
        'q_value': [],
        'epsilon': [],
        'q_action': [],
        'substrate_conc_cell_0': [],
        'substrate_conc_cell_1': [],
        'substrate_conc_cell_2': [],
        'substrate_conc_cell_3': [],
        'substrate_conc_cell_4': [],
        'biofilm_thickness_cell_0': [],
        'biofilm_thickness_cell_1': [],
        'biofilm_thickness_cell_2': [],
        'biofilm_thickness_cell_3': [],
        'biofilm_thickness_cell_4': [],
        'power_cell_0': [],
        'power_cell_1': [],
        'power_cell_2': [],
        'power_cell_3': [],
        'power_cell_4': []
    }
    
    print("=== MFC Simulation with Recirculation and Advanced Substrate Control ===")
    print(f"Duration: {duration_hours} hours")
    print(f"Reservoir volume: {reservoir.volume} L")
    print(f"Number of cells: {n_cells}")
    print(f"Target outlet concentration: {controller.target_outlet_conc} mmol/L")
    print("=" * 80)
    
    for step in range(n_steps):
        time_hours = step * dt_hours
        
        # Get current reservoir concentration for inlet
        inlet_concentration = reservoir.get_inlet_concentration()
        
        # Process substrate through cell stack
        current_conc = inlet_concentration
        cell_concentrations = []
        
        for cell in cells:
            current_conc = cell.update_concentrations(current_conc, flow_rate_ml_h, dt_hours)
            cell_concentrations.append(cell.substrate_concentration)
        
        outlet_concentration = current_conc
        
        # Get reservoir sensor readings
        reservoir_sensors = reservoir.get_sensor_readings()
        
        # Q-learning control for substrate addition (replaced PID controller)
        # Get current system state
        avg_biofilm = np.mean([cell.biofilm_thickness for cell in cells])
        biofilm_deviation = abs(avg_biofilm - cells[0].optimal_biofilm_thickness)
        substrate_utilization = ((inlet_concentration - outlet_concentration) / 
                               inlet_concentration * 100 if inlet_concentration > 0 else 0)
        min_cell_conc = min(cell_concentrations)
        outlet_error = abs(outlet_concentration - 12.0)  # Target outlet concentration
        
        # Discretize state for Q-learning
        state = q_controller.discretize_enhanced_state(
            0.001,  # Simplified power for this context
            biofilm_deviation, 
            substrate_utilization,
            reservoir.substrate_concentration, 
            min_cell_conc, 
            outlet_error, 
            time_hours
        )
        
        # Choose combined action (flow + substrate)
        action_idx, flow_idx, substrate_idx = q_controller.choose_combined_action(state)
        
        # Apply flow action
        flow_rate_ml_h += q_controller.flow_actions[flow_idx]
        flow_rate_ml_h = np.clip(flow_rate_ml_h, 5.0, 50.0)  # Keep within bounds
        
        # Apply substrate action
        substrate_addition = q_controller.substrate_actions[substrate_idx]
        addition_rate = max(0.0, substrate_addition)  # No negative additions
        addition_rate = min(addition_rate, q_controller.config.substrate_addition_max)
        
        # Calculate reward for Q-learning
        substrate_reward = q_controller.calculate_substrate_reward(
            reservoir.substrate_concentration,
            cell_concentrations,
            outlet_concentration,
            addition_rate,
            inlet_concentration  # Pass inlet concentration for outlet sensor penalty logic
        )
        
        # Update Q-value if we have previous state
        if q_controller.previous_state is not None:
            q_controller.update_q_value(
                q_controller.previous_state,
                q_controller.previous_action,
                substrate_reward,
                state
            )
        
        # Store current state and action for next iteration
        q_controller.previous_state = state
        q_controller.previous_action = action_idx
        
        # Add substrate to reservoir (no halt flag needed - Q-learning handles this)
        if addition_rate > 0:
            reservoir.add_substrate(addition_rate, dt_hours)
        
        # Simulate recirculation
        reservoir.circulate_anolyte(flow_rate_ml_h, outlet_concentration, dt_hours)
        
        # Calculate system performance metrics (LITERATURE-VALIDATED)
        # Updated power calculation with acetate-specific potential (0.35V vs 0.77V)
        total_power = sum(0.35 * cell.biofilm_thickness * cell.substrate_concentration * 0.002
                         for cell in cells)  # Acetate-specific potential with enhanced efficiency
        
        avg_biofilm = np.mean([cell.biofilm_thickness for cell in cells])
        biofilm_deviation = abs(avg_biofilm - cells[0].optimal_biofilm_thickness)
        substrate_utilization = ((inlet_concentration - outlet_concentration) / 
                               inlet_concentration * 100 if inlet_concentration > 0 else 0)
        
        # Q-learning control (flow rate adjustment)
        min_cell_conc = min(cell_concentrations)
        outlet_error = abs(outlet_concentration - controller.target_outlet_conc)
        
        state = q_controller.discretize_enhanced_state(
            total_power, biofilm_deviation, substrate_utilization,
            reservoir.substrate_concentration, min_cell_conc, outlet_error, time_hours
        )
        
        # Store results
        results['time_hours'].append(time_hours)
        results['reservoir_concentration'].append(reservoir.substrate_concentration)
        results['cell_concentrations'].append(cell_concentrations.copy())
        results['outlet_concentration'].append(outlet_concentration)
        results['flow_rate'].append(flow_rate_ml_h)
        results['substrate_addition_rate'].append(addition_rate)
        results['total_power'].append(total_power)
        results['biofilm_thicknesses'].append([cell.biofilm_thickness for cell in cells])
        results['substrate_halt'].append(int(addition_rate == 0.0))  # 1 if no substrate added, 0 otherwise
        
        # Store Q-learning metrics
        current_q_value = q_controller.q_table[state][action_idx] if state in q_controller.q_table else 0.0
        results['q_value'].append(current_q_value)
        results['epsilon'].append(q_controller.epsilon)
        results['q_action'].append(action_idx)
        
        # Store individual cell data
        for i, cell in enumerate(cells):
            results[f'substrate_conc_cell_{i}'].append(cell.substrate_concentration)
            results[f'biofilm_thickness_cell_{i}'].append(cell.biofilm_thickness)
            cell_power = 0.35 * cell.biofilm_thickness * cell.substrate_concentration * 0.002
            results[f'power_cell_{i}'].append(cell_power)
        
        # Progress reporting
        if step % (n_steps // 10) == 0:
            print(f"Time: {time_hours:.1f}h, "
                  f"Reservoir: {reservoir.substrate_concentration:.2f} mmol/L, "
                  f"Outlet: {outlet_concentration:.2f} mmol/L, "
                  f"Min Cell: {min_cell_conc:.2f} mmol/L, "
                  f"Addition: {addition_rate:.1f} mmol/h, "
                  f"Q-Action: {action_idx}")
    
    return results, cells, reservoir, controller, q_controller

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='MFC Recirculation Control Simulation with Literature-Validated Parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=100,
        help='Simulation duration in hours'
    )
    parser.add_argument(
        '--prefix', '-p',
        type=str,
        default='mfc_simulation',
        help='Prefix for output directory and files'
    )
    parser.add_argument(
        '--user-suffix', '-u',
        type=str,
        default='',
        help='Optional user-defined suffix for filenames'
    )
    parser.add_argument(
        '--no-timestamp',
        action='store_true',
        help='Suppress timestamp in filenames (directory will still have timestamp)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Suppress generation of plots'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='../data/simulation_data',
        help='Base output directory for simulation results'
    )
    parser.add_argument(
        '--load-checkpoint',
        type=str,
        default=None,
        help='Path to existing Q-learning checkpoint to load and continue training'
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    print("🔬 Running MFC Simulation with Literature-Validated Parameters")
    print(f"📊 Duration: {args.duration} hours")
    print(f"📁 Output prefix: '{args.prefix}'")
    print(f"📝 User suffix: '{args.user_suffix}'" if args.user_suffix else "")
    print(f"⏰ Timestamp in filenames: {'No' if args.no_timestamp else 'Yes'}")
    print(f"📊 Generate plots: {'No' if args.no_plots else 'Yes'}")
    print("=" * 60)
    
    # Load configuration
    from config.qlearning_config import DEFAULT_QLEARNING_CONFIG
    config = DEFAULT_QLEARNING_CONFIG
    
    # Run simulation with specified duration
    results, cells, reservoir, controller, q_controller = simulate_mfc_with_recirculation(
        args.duration, config, args.load_checkpoint)
    
    # Create timestamp (always used for directory, optionally for files)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build filename components
    base_name = args.prefix
    duration_str = f"{args.duration}h"
    timestamp_str = f"_{timestamp}" if not args.no_timestamp else ""
    user_suffix = f"_{args.user_suffix}" if args.user_suffix else ""
    
    # Generate filenames
    data_filename = f"{base_name}_{duration_str}{timestamp_str}{user_suffix}_data.csv"
    data_json_filename = f"{base_name}_{duration_str}{timestamp_str}{user_suffix}_data.json"
    metadata_filename = f"{base_name}_{duration_str}{timestamp_str}{user_suffix}_metadata.json"
    log_filename = f"{base_name}_{duration_str}{timestamp_str}{user_suffix}_simulation.log"
    model_checkpoint_filename = f"{base_name}_{duration_str}{timestamp_str}{user_suffix}_model_checkpoint.json"
    
    # Create output directory
    dir_suffix = f"_{args.user_suffix}" if args.user_suffix else ""
    output_dir = Path(args.output_dir) / f"{args.prefix}_{timestamp}{dir_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file and console
    log_file = output_dir / log_filename
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Log simulation start
    logger.info("=== MFC SIMULATION STARTED ===")
    logger.info(f"Duration: {args.duration} hours")
    logger.info(f"Output directory: {output_dir}")
    if args.load_checkpoint:
        logger.info(f"Loading Q-learning checkpoint: {args.load_checkpoint}")
    logger.info(f"Configuration: {config.__dict__ if hasattr(config, '__dict__') else 'DEFAULT_CONFIG'}")
    
    # Prepare DataFrame (keep all columns from results dict)
    df = pd.DataFrame(results)
    
    # Save CSV data file
    csv_file = output_dir / data_filename
    df.to_csv(csv_file, index=False)
    logger.info(f"CSV data saved to: {data_filename}")
    
    # Save complete data in JSON format
    json_data_file = output_dir / data_json_filename
    data_dict = df.to_dict('records')  # Convert DataFrame to list of dictionaries
    with open(json_data_file, 'w') as f:
        json.dump({
            'simulation_data': data_dict,
            'data_info': {
                'total_records': len(data_dict),
                'columns': list(df.columns),
                'time_range': {
                    'start_hours': float(df['time_hours'].min()),
                    'end_hours': float(df['time_hours'].max()),
                    'time_step': float(df['time_hours'].iloc[1] - df['time_hours'].iloc[0]) if len(df) > 1 else 0
                }
            }
        }, f, indent=2)
    logger.info(f"JSON data saved to: {data_json_filename}")
    
    # Save Q-learning model checkpoint
    model_checkpoint_file = output_dir / model_checkpoint_filename
    checkpoint_info = q_controller.save_checkpoint(model_checkpoint_file)
    logger.info(f"Q-learning checkpoint saved to: {model_checkpoint_filename}")
    logger.info(f"Checkpoint contains {checkpoint_info['training_statistics']['q_table_size']} states")
    logger.info(f"States with learned values: {checkpoint_info['training_statistics']['states_explored']}")
    
    # Prepare metadata
    metadata = {
        'simulation_info': {
            'timestamp': timestamp,
            'duration_hours': args.duration,
            'prefix': args.prefix,
            'user_suffix': args.user_suffix,
            'output_directory': str(output_dir),
            'checkpoint_loaded': args.load_checkpoint is not None,
            'checkpoint_source': args.load_checkpoint if args.load_checkpoint else None,
            'files_generated': {
                'data_csv': data_filename,
                'data_json': data_json_filename,
                'metadata': metadata_filename,
                'simulation_log': log_filename,
                'model_checkpoint': model_checkpoint_filename,
                'plots': [] if args.no_plots else []  # Will be populated below
            }
        },
        'simulation_parameters': {
            'n_cells': 5,
            'dt_hours': 10.0 / 3600.0,
            'n_steps': len(results['time_hours']),
            'reservoir_volume_L': reservoir.volume,
            'initial_substrate_concentration': config.initial_substrate_concentration,
            'substrate_target_reservoir': config.substrate_target_reservoir,
            'substrate_target_outlet': config.substrate_target_outlet,
            'substrate_target_cell': config.substrate_target_cell,
            'substrate_max_threshold': config.substrate_max_threshold,
            'substrate_min_threshold': config.substrate_min_threshold,
            'substrate_addition_max': config.substrate_addition_max,
            'q_learning_parameters': {
                'learning_rate': q_controller.learning_rate,
                'discount_factor': q_controller.discount_factor,
                'epsilon_initial': 0.3702,
                'epsilon_final': q_controller.epsilon,
                'epsilon_decay': q_controller.epsilon_decay,
                'epsilon_min': q_controller.epsilon_min,
                'flow_rate_actions': list(q_controller.flow_actions),
                'substrate_actions': list(q_controller.substrate_actions)
            }
        },
        'performance_summary': {
            'initial_reservoir_concentration': config.initial_substrate_concentration,
            'final_reservoir_concentration': reservoir.substrate_concentration,
            'initial_outlet_concentration': results['outlet_concentration'][0] if results['outlet_concentration'] else 0,
            'final_outlet_concentration': results['outlet_concentration'][-1] if results['outlet_concentration'] else 0,
            'total_substrate_added': reservoir.total_substrate_added,
            'average_substrate_addition_rate': np.mean(results['substrate_addition_rate']),
            'initial_power_output': results['total_power'][0] if results['total_power'] else 0,
            'final_power_output': results['total_power'][-1] if results['total_power'] else 0,
            'average_power_output': np.mean(results['total_power']),
            'initial_biofilm_thicknesses': results['biofilm_thicknesses'][0] if results['biofilm_thicknesses'] else [],
            'final_biofilm_thicknesses': [cell.biofilm_thickness for cell in cells],
            'average_biofilm_thickness': np.mean([cell.biofilm_thickness for cell in cells]),
            'total_circulation_cycles': reservoir.circulation_cycles,
            'total_pump_time': reservoir.total_pump_time,
            'final_q_value': results['q_value'][-1] if results['q_value'] else 0,
            'final_epsilon': results['epsilon'][-1] if results['epsilon'] else 0
        },
        'model_checkpoint_info': {
            'q_table_size': checkpoint_info['training_statistics']['q_table_size'],
            'states_explored': checkpoint_info['training_statistics']['states_explored'],
            'total_q_updates': checkpoint_info['training_statistics']['total_q_updates'],
            'training_episodes': checkpoint_info['training_statistics']['episode_count'],
            'total_rewards': checkpoint_info['training_statistics']['total_rewards'],
            'final_epsilon': checkpoint_info['hyperparameters']['current_epsilon'],
            'checkpoint_created_at': checkpoint_info['model_info']['created_at']
        },
        'controller_statistics': {
            'substrate_halt_percentage': np.mean(results['substrate_halt']) * 100 if results['substrate_halt'] else 0,
            'control_modes_used': list(set(d['mode'] for d in controller.control_history)) if controller.control_history else [],
            'average_outlet_error': np.mean([abs(d['outlet_error']) for d in controller.control_history]) if controller.control_history else 0,
            'average_reservoir_error': np.mean([abs(d['reservoir_error']) for d in controller.control_history]) if controller.control_history else 0
        },
        'cell_performance': {
            f'cell_{i}': {
                'initial_substrate_conc': results[f'substrate_conc_cell_{i}'][0] if results[f'substrate_conc_cell_{i}'] else 0,
                'final_substrate_conc': results[f'substrate_conc_cell_{i}'][-1] if results[f'substrate_conc_cell_{i}'] else 0,
                'average_substrate_conc': np.mean(results[f'substrate_conc_cell_{i}']),
                'initial_biofilm': results[f'biofilm_thickness_cell_{i}'][0] if results[f'biofilm_thickness_cell_{i}'] else 0,
                'final_biofilm': results[f'biofilm_thickness_cell_{i}'][-1] if results[f'biofilm_thickness_cell_{i}'] else 0,
                'average_power': np.mean(results[f'power_cell_{i}']),
                'final_power': results[f'power_cell_{i}'][-1] if results[f'power_cell_{i}'] else 0
            } for i in range(5)
        }
    }
    
    # Generate plots if not suppressed
    if not args.no_plots:
        logger.info("Generating plots...")
        import sys
        sys.path.append('.')
        from plotting_system import plot_mfc_simulation_results
        
        plot_prefix = f"{base_name}_{duration_str}{timestamp_str}{user_suffix}"
        plot_timestamp = plot_mfc_simulation_results(csv_file, output_prefix=str(output_dir / plot_prefix))
        
        # Update metadata with generated plot filenames
        plot_files = [
            f"{plot_prefix}_overview_{plot_timestamp}.png",
            f"{plot_prefix}_cells_{plot_timestamp}.png"
        ]
        if args.duration > 100:
            plot_files.append(f"{plot_prefix}_dynamics_{plot_timestamp}.png")
        
        metadata['simulation_info']['files_generated']['plots'] = plot_files
        logger.info(f"Generated {len(plot_files)} plot files")
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    # Save metadata
    metadata_file = output_dir / metadata_filename
    metadata_converted = convert_numpy_types(metadata)
    with open(metadata_file, 'w') as f:
        json.dump(metadata_converted, f, indent=2)
    logger.info(f"Metadata saved to: {metadata_filename}")
    
    # Log simulation completion
    logger.info("=== MFC SIMULATION COMPLETE ===")
    logger.info(f"Duration: {args.duration} hours")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Files generated:")
    logger.info(f"  - Data CSV: {data_filename}")
    logger.info(f"  - Data JSON: {data_json_filename}")
    logger.info(f"  - Metadata: {metadata_filename}")
    logger.info(f"  - Simulation log: {log_filename}")
    logger.info(f"  - Model checkpoint: {model_checkpoint_filename}")
    if not args.no_plots:
        logger.info(f"  - Plots: {len(metadata['simulation_info']['files_generated']['plots'])} files")
    
    logger.info("PERFORMANCE SUMMARY:")
    logger.info(f"   Reservoir concentration: {reservoir.substrate_concentration:.3f} mmol/L")
    logger.info(f"   Final outlet concentration: {results['outlet_concentration'][-1]:.3f} mmol/L")
    logger.info(f"   Total substrate added: {reservoir.total_substrate_added:.3f} mmol")
    logger.info(f"   Average biofilm thickness: {np.mean([cell.biofilm_thickness for cell in cells]):.3f}")
    logger.info(f"   Final power output: {results['total_power'][-1]:.6f} W")
    
    print("\n=== LITERATURE-VALIDATED SIMULATION COMPLETE ===")
    print(f"📊 Duration: {args.duration} hours")
    print(f"📁 Output directory: {output_dir}")
    print(f"📁 Data files: {data_filename} (CSV), {data_json_filename} (JSON)")
    print(f"📁 Metadata file: {metadata_filename}")
    print(f"📁 Simulation log: {log_filename}")
    print(f"🤖 Model checkpoint: {model_checkpoint_filename}")
    if not args.no_plots:
        print(f"📊 Plots generated: {len(metadata['simulation_info']['files_generated']['plots'])} files")
    print("\n🔬 PERFORMANCE SUMMARY:")
    print(f"   Reservoir concentration: {reservoir.substrate_concentration:.3f} mmol/L")
    print(f"   Final outlet concentration: {results['outlet_concentration'][-1]:.3f} mmol/L")
    print(f"   Total substrate added: {reservoir.total_substrate_added:.3f} mmol")
    print(f"   Average biofilm thickness: {np.mean([cell.biofilm_thickness for cell in cells]):.3f}")
    print(f"   Final power output: {results['total_power'][-1]:.6f} W")
    
    print("\n🧬 LITERATURE IMPROVEMENTS APPLIED:")
    print("   ✅ Biofilm growth rate: 0.05 h⁻¹ (50x increase from 0.001)")
    print("   ✅ Acetate-specific potential: 0.35 V (reduced from 0.77 V)")
    print("   ✅ Enhanced reaction rate: 0.15 (increased substrate consumption)")
    print("   ✅ Balanced decay rate: 0.01 h⁻¹ (50x increase to match growth)")