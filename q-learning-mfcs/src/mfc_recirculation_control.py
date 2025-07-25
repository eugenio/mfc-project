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
from datetime import datetime
from collections import defaultdict
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
    
    def calculate_substrate_reward(self, reservoir_conc, cell_concentrations, outlet_conc, substrate_addition):
        """Calculate reward for substrate control based on sensor readings"""
        reward = 0.0
        
        # Reward for maintaining target concentrations
        reservoir_error = abs(reservoir_conc - self.config.substrate_target_reservoir)
        if reservoir_error < 2.0:  # Within 2 mM of target
            reward += self.config.reward_weights.substrate_target_reward * (1.0 - reservoir_error / 2.0)
        
        outlet_error = abs(outlet_conc - self.config.substrate_target_outlet)
        if outlet_error < 2.0:  # Within 2 mM of target
            reward += self.config.reward_weights.substrate_target_reward * (1.0 - outlet_error / 2.0)
        
        # Reward for each cell maintaining target concentration
        for cell_conc in cell_concentrations:
            cell_error = abs(cell_conc - self.config.substrate_target_cell)
            if cell_error < 3.0:  # Within 3 mM of target
                reward += self.config.reward_weights.substrate_target_reward * 0.2 * (1.0 - cell_error / 3.0)
        
        # Penalties for exceeding thresholds
        if reservoir_conc > self.config.substrate_max_threshold:
            excess = reservoir_conc - self.config.substrate_max_threshold
            reward += self.config.reward_weights.substrate_excess_penalty * excess
        
        if outlet_conc > self.config.substrate_max_threshold:
            excess = outlet_conc - self.config.substrate_max_threshold
            reward += self.config.reward_weights.substrate_excess_penalty * excess
        
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
        
        # Substrate consumption based on biofilm activity
        biofilm_efficiency = min(1.0, self.biofilm_thickness / self.optimal_biofilm_thickness)
        max_consumption = self.max_reaction_rate * biofilm_efficiency * residence_time_h
        
        # Actual consumption limited by available substrate
        substrate_consumed = min(max_consumption, inlet_conc * 0.9)  # Max 90% consumption per cell
        
        # Update outlet concentration
        self.outlet_concentration = max(0.0, inlet_conc - substrate_consumed)
        self.substrate_concentration = (inlet_conc + self.outlet_concentration) / 2.0
        
        # Update biofilm based on substrate availability
        if self.substrate_concentration > 5.0:  # Sufficient substrate
            growth_factor = 1.0
        elif self.substrate_concentration > 2.0:  # Limited substrate
            growth_factor = 0.5
        else:  # Starved conditions
            growth_factor = 0.1
        
        biofilm_growth = self.biofilm_growth_rate * growth_factor * dt_hours
        biofilm_decay = 0.01 * self.biofilm_thickness * dt_hours  # Increased decay to balance higher growth
        
        self.biofilm_thickness = np.clip(
            self.biofilm_thickness + biofilm_growth - biofilm_decay,
            0.5, 3.0
        )
        
        # Store monitoring data
        self.concentration_history.append(self.substrate_concentration)
        self.consumption_rate_history.append(substrate_consumed / dt_hours if dt_hours > 0 else 0)
        
        return self.outlet_concentration
    
    def process_with_monitoring(self, inlet_conc, flow_rate_ml_h, dt_hours):
        """Process cell with monitoring - alias for update_concentrations"""
        return self.update_concentrations(inlet_conc, flow_rate_ml_h, dt_hours)

def simulate_mfc_with_recirculation(duration_hours=100):
    """Main simulation function with recirculation and advanced substrate control"""
    
    # Initialize components
    reservoir = AnolytereservoirSystem(initial_substrate_conc=20.0, volume_liters=1.0)
    controller = SubstrateConcentrationController(target_outlet_conc=12.0, target_reservoir_conc=20.0)
    q_controller = AdvancedQLearningFlowController()
    
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
        'substrate_halt': []
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
            addition_rate
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
        
        # Progress reporting
        if step % (n_steps // 10) == 0:
            print(f"Time: {time_hours:.1f}h, "
                  f"Reservoir: {reservoir.substrate_concentration:.2f} mmol/L, "
                  f"Outlet: {outlet_concentration:.2f} mmol/L, "
                  f"Min Cell: {min_cell_conc:.2f} mmol/L, "
                  f"Addition: {addition_rate:.1f} mmol/h, "
                  f"Q-Action: {action_idx}")
    
    return results, cells, reservoir, controller

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
        '--suffix', '-s',
        type=str,
        default='',
        help='Suffix to add to output filenames (e.g., "_validated", "_100h")'
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    print("🔬 Running MFC Simulation with Literature-Validated Parameters")
    print(f"📊 Duration: {args.duration} hours")
    print(f"📁 Output suffix: '{args.suffix}'")
    print("=" * 60)
    
    # Run simulation with specified duration
    results, cells, reservoir, controller = simulate_mfc_with_recirculation(args.duration)
    
    # Save results with configurable suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = args.suffix if args.suffix else f"_{args.duration}h_validated"
    
    # Convert to DataFrame for easier analysis
    df_data = {
        'time_hours': results['time_hours'],
        'reservoir_concentration': results['reservoir_concentration'],
        'outlet_concentration': results['outlet_concentration'],
        'flow_rate': results['flow_rate'],
        'substrate_addition_rate': results['substrate_addition_rate'],
        'total_power': results['total_power'],
        'substrate_halt': results['substrate_halt']
    }
    
    # Add individual cell data
    for i in range(len(cells)):
        df_data[f'cell_{i+1}_concentration'] = [concs[i] for concs in results['cell_concentrations']]
        df_data[f'cell_{i+1}_biofilm'] = [thick[i] for thick in results['biofilm_thicknesses']]
    
    df = pd.DataFrame(df_data)
    csv_file = get_simulation_data_path(f"mfc_recirculation_control{suffix}_{timestamp}.csv")
    json_file = get_simulation_data_path(f"mfc_recirculation_control{suffix}_{timestamp}.json")
    
    # Save CSV
    df.to_csv(csv_file, index=False)
    
    # Save JSON with comprehensive data
    json_data = {
        'simulation_metadata': {
            'timestamp': timestamp,
            'duration_hours': args.duration,
            'n_cells': 5,
            'target_outlet_conc': controller.target_outlet_conc,
            'reservoir_volume_L': reservoir.volume,
            'dt_hours': 10.0 / 3600.0,
            'n_steps': len(results['time_hours'])
        },
        'performance_summary': {
            'final_reservoir_concentration': reservoir.substrate_concentration,
            'final_outlet_concentration': results['outlet_concentration'][-1],
            'total_substrate_added': reservoir.total_substrate_added,
            'average_biofilm_thickness': np.mean([cell.biofilm_thickness for cell in cells]),
            'final_biofilm_thicknesses': [cell.biofilm_thickness for cell in cells],
            'total_circulation_cycles': reservoir.circulation_cycles,
            'total_pump_time': reservoir.total_pump_time
        },
        'time_series_data': df_data,
        'controller_history': controller.control_history[-100:] if len(controller.control_history) > 100 else controller.control_history,  # Last 100 entries
        'reservoir_mixing_history': reservoir.mixing_efficiency_history[-50:] if len(reservoir.mixing_efficiency_history) > 50 else reservoir.mixing_efficiency_history  # Last 50 entries
    }
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print("\n=== LITERATURE-VALIDATED SIMULATION COMPLETE ===")
    print(f"📊 Duration: {args.duration} hours")
    print(f"📁 CSV saved to: {csv_file}")
    print(f"📁 JSON saved to: {json_file}")
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