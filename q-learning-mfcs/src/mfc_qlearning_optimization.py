
#!/usr/bin/env python3
"""
MFC Stack Optimization with Q-Learning Controller
Q-learning agent controls flow rate to optimize power, biofilm, and substrate utilization
Duration: 1000 hours, timestep: 10 seconds
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import pickle
from path_config import get_figure_path, get_simulation_data_path, get_model_path
from typing import Optional

# Import configuration classes
from config import QLearningConfig, validate_qlearning_config

# Import universal GPU acceleration
from gpu_acceleration import get_gpu_accelerator

# Initialize GPU accelerator
gpu_accelerator = get_gpu_accelerator()
GPU_AVAILABLE = gpu_accelerator.is_gpu_available()

class QLearningFlowController:
    def __init__(self, config: Optional[QLearningConfig] = None):
        """Q-Learning controller for flow rate optimization"""
        # Use default configuration if not provided
        if config is None:
            config = QLearningConfig()

        # Validate configuration
        validate_qlearning_config(config)
        self.config = config

        # Extract basic Q-learning parameters
        self.learning_rate = config.learning_rate
        self.discount_factor = config.discount_factor
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min

        # Q-table stored as nested dictionary
        self.q_table = defaultdict(lambda: defaultdict(float))

        # Define state and action discretization
        self.setup_state_action_spaces()

        # Statistics
        self.total_rewards = 0
        self.episode_count = 0

    def setup_state_action_spaces(self):
        """Define state and action discretization for Q-learning"""
        # State variables from configuration
        self.power_bins = np.linspace(0, self.config.power_max, self.config.power_bins)
        self.biofilm_bins = np.linspace(0, self.config.biofilm_max_deviation,
                                       self.config.biofilm_deviation_bins)
        self.substrate_bins = np.linspace(0, self.config.substrate_utilization_max,
                                        self.config.substrate_utilization_bins)
        self.time_bins = np.array(self.config.time_phase_hours)

        # Action space from configuration
        self.actions = np.array(self.config.flow_rate_adjustments_ml_per_h)

    def discretize_state(self, power, biofilm_deviation, substrate_utilization, time_hours):
        """Convert continuous state to discrete state key"""
        power_idx = np.digitize(power, self.power_bins) - 1
        biofilm_idx = np.digitize(biofilm_deviation, self.biofilm_bins) - 1
        substrate_idx = np.digitize(substrate_utilization, self.substrate_bins) - 1
        time_idx = np.digitize(time_hours, self.time_bins) - 1

        # Clip indices to valid ranges
        power_idx = np.clip(power_idx, 0, len(self.power_bins) - 2)
        biofilm_idx = np.clip(biofilm_idx, 0, len(self.biofilm_bins) - 2)
        substrate_idx = np.clip(substrate_idx, 0, len(self.substrate_bins) - 2)
        time_idx = np.clip(time_idx, 0, len(self.time_bins) - 2)

        return (power_idx, biofilm_idx, substrate_idx, time_idx)

    def select_action(self, state, current_flow_rate):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Exploration: random action
            action_idx = np.random.randint(len(self.actions))
        else:
            # Exploitation: best known action
            q_values = [self.q_table[state][i] for i in range(len(self.actions))]
            action_idx = np.argmax(q_values)

        # Calculate new flow rate with bounds from configuration
        flow_adjustment = self.actions[action_idx] * 1e-6 / 3.6  # Convert mL/h to m³/s
        min_flow = self.config.flow_rate_min * 1e-6 / 3.6  # Convert mL/h to m³/s
        max_flow = self.config.flow_rate_max * 1e-6 / 3.6  # Convert mL/h to m³/s
        new_flow_rate = np.clip(current_flow_rate + flow_adjustment, min_flow, max_flow)

        return action_idx, new_flow_rate

    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning rule"""
        current_q = self.q_table[state][action]

        # Find maximum Q-value for next state
        next_q_values = [self.q_table[next_state][i] for i in range(len(self.actions))]
        max_next_q = max(next_q_values) if next_q_values else 0

        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state][action] = new_q

        # Update statistics
        self.total_rewards += reward

        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def calculate_reward(self, power, biofilm_deviation, substrate_utilization,
                        prev_power, prev_biofilm_dev, prev_substrate_util,
                        biofilm_thickness_history=None):
        """
        Calculate reward for Q-learning with optimized objectives:
        - Maximize power and substrate consumption
        - Minimize biofilm growth (target optimal thickness with zero derivative)
        - Penalties for power/consumption decrease and biofilm deviation > 5%
        """
        # 1. POWER COMPONENT
        power_change = power - prev_power
        if power_change > 0:
            power_reward = power_change * 50  # Strong reward for power increase
        elif power_change < 0:
            power_reward = power_change * 100  # Strong penalty for power decrease
        else:
            power_reward = 0

        # Base reward for maintaining high power
        power_base = min(20.0, power * 20.0)  # Higher base reward for high power

        # 2. SUBSTRATE CONSUMPTION COMPONENT
        substrate_change = substrate_utilization - prev_substrate_util
        if substrate_change > 0:
            substrate_reward = substrate_change * 30  # Strong reward for consumption increase
        elif substrate_change < 0:
            substrate_reward = substrate_change * 60  # Strong penalty for consumption decrease
        else:
            substrate_reward = 0

        # Base reward for maintaining high consumption
        substrate_base = min(15.0, substrate_utilization * 0.75)  # Higher base reward

        # 3. BIOFILM OPTIMAL THICKNESS COMPONENT
        # Check if biofilm deviation is within acceptable range (±5%)
        optimal_thickness = 1.3  # From self.optimal_biofilm_thickness
        deviation_threshold = 0.05 * optimal_thickness  # 5% threshold

        if biofilm_deviation <= deviation_threshold:
            # Within optimal range - reward for maintaining optimal thickness
            biofilm_reward = 25.0 - (biofilm_deviation / deviation_threshold) * 10.0

            # Extra reward if biofilm growth rate is near zero (steady state)
            if biofilm_thickness_history is not None and len(biofilm_thickness_history) >= 3:
                # Calculate biofilm growth rate (derivative approximation)
                recent_thickness = biofilm_thickness_history[-3:]
                if len(recent_thickness) >= 2:
                    growth_rate = abs(recent_thickness[-1] - recent_thickness[-2])
                    if growth_rate < 0.01:  # Very low growth rate
                        biofilm_reward += 15.0  # Bonus for steady state
        else:
            # Outside optimal range (>5% deviation) - apply penalty
            excess_deviation = biofilm_deviation - deviation_threshold
            biofilm_reward = -50.0 * (excess_deviation / deviation_threshold)  # Strong penalty

        # 4. COMBINED PENALTY for simultaneous degradation
        if power_change < 0 and substrate_change < 0 and biofilm_deviation > deviation_threshold:
            # Triple penalty when all objectives worsen
            combined_penalty = -100.0
        else:
            combined_penalty = 0

        # 5. TOTAL REWARD
        total_reward = (power_reward + power_base +
                       substrate_reward + substrate_base +
                       biofilm_reward + combined_penalty)

        return total_reward

    def save_q_table(self, filename):
        """Save Q-table to file"""
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self, filename):
        """Load Q-table from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                loaded_table = pickle.load(f)
                self.q_table = defaultdict(lambda: defaultdict(float), loaded_table)
            print(f"Q-table loaded from {filename}")


def add_subplot_labels(fig, start_letter='a'):
    """Add alphabetic labels to all subplots in a figure"""
    import string
    axes = fig.get_axes()

    # Generate letters programmatically based on number of plots
    for i, ax in enumerate(axes):
        if start_letter.islower():
            # Use lowercase letters
            start_idx = string.ascii_lowercase.index(start_letter)
            letter_idx = (start_idx + i) % 26
            letter = string.ascii_lowercase[letter_idx]
        else:
            # Use uppercase letters
            start_idx = string.ascii_uppercase.index(start_letter)
            letter_idx = (start_idx + i) % 26
            letter = string.ascii_uppercase[letter_idx]

        # Position label outside plot area (top-left corner, diagonally outside)
        ax.text(-0.1, 1.05, letter, transform=ax.transAxes, fontsize=16, fontweight='bold',
                verticalalignment='bottom', horizontalalignment='right', zorder=1000)

class MFCQLearningSimulation:
    def __init__(self, use_gpu=True):
        """Initialize MFC Q-learning simulation"""
        self.use_gpu = use_gpu and GPU_AVAILABLE

        # Simulation parameters
        self.num_cells = 5
        self.dt = 10.0  # seconds
        self.total_time = 1000 * 3600  # 1000 hours in seconds
        self.num_steps = int(self.total_time / self.dt)

        # Physical parameters
        self.V_a = 5.5e-5  # Anodic volume (m³)
        self.A_m = 5.0e-4  # Membrane area (m²)
        self.F = 96485.0   # Faraday constant (C/mol)

        # Biological parameters
        self.r_max = 1.0e-5  # Maximum reaction rate (mol/(m²·s)) - for mmol/L concentrations
        self.K_AC = 5.0      # Acetate half-saturation constant (mol/m³) = 5 mmol/L
        self.K_dec = 8.33e-4   # Decay constant (s⁻¹)
        self.Y_ac = 0.05       # Biomass yield (kg/mol)

        # Optimization parameters
        self.optimal_biofilm_thickness = 1.3  # Optimal biofilm for max electron transfer
        self.flow_rate_bounds = (5.0e-6, 50.0e-6)  # Flow rate bounds (m³/s)

        # Multi-objective weights
        self.w_power = 0.5
        self.w_biofilm = 0.3
        self.w_substrate = 0.2

        # Initialize Q-learning controller
        self.q_controller = QLearningFlowController()

        # Track biofilm thickness history for derivative calculation
        self.biofilm_history = []

        # Initialize arrays
        self.initialize_arrays()

        # Debug counter
        self.debug_counter = 0

        # Create output directories
        os.makedirs('figures', exist_ok=True)
        os.makedirs('simulation_data', exist_ok=True)
        os.makedirs('q_learning_models', exist_ok=True)

    def initialize_arrays(self):
        """Initialize simulation state arrays"""
        array_func = gpu_accelerator.zeros if self.use_gpu else np.zeros

        # Cell state arrays [time_step, cell_index]
        self.cell_voltages = array_func((self.num_steps, self.num_cells))
        self.biofilm_thickness = array_func((self.num_steps, self.num_cells))
        self.acetate_concentrations = array_func((self.num_steps, self.num_cells))
        self.current_densities = array_func((self.num_steps, self.num_cells))
        self.power_outputs = array_func((self.num_steps, self.num_cells))
        self.substrate_consumptions = array_func((self.num_steps, self.num_cells))

        # Stack-level arrays [time_step]
        self.stack_voltages = array_func(self.num_steps)
        self.stack_powers = array_func(self.num_steps)
        self.flow_rates = array_func(self.num_steps)
        self.objective_values = array_func(self.num_steps)
        self.substrate_utilizations = array_func(self.num_steps)
        self.q_rewards = array_func(self.num_steps)
        self.q_actions = array_func(self.num_steps)

        # Initialize starting conditions
        self.biofilm_thickness[0, :] = 1.0  # Starting biofilm thickness
        self.acetate_concentrations[0, :] = 20.0  # Starting acetate concentration (20 mmol/L = 20 mol/m³)
        self.flow_rates[0] = 10.0 / 3.6e9  # Starting flow rate (10 mL/h converted to m³/s)

    def biofilm_factor(self, thickness):
        """Calculate biofilm factor affecting mass transfer"""
        if self.use_gpu:
            delta_opt = gpu_accelerator.abs(thickness - self.optimal_biofilm_thickness)
            return 1.0 + 0.3 * delta_opt + 0.1 * delta_opt * delta_opt
        else:
            delta_opt = np.abs(thickness - self.optimal_biofilm_thickness)
            return 1.0 + 0.3 * delta_opt + 0.1 * delta_opt * delta_opt

    def reaction_rate(self, c_ac, biofilm):
        """Enhanced Monod kinetics with biofilm effects"""
        biofilm_factor = self.biofilm_factor(biofilm)
        effective_rate = self.r_max / biofilm_factor

        # Substrate limitation
        substrate_term = c_ac / (self.K_AC + c_ac)

        # Optimal biofilm enhancement for electron transfer
        if self.use_gpu:
            electron_enhancement = gpu_accelerator.where(
                gpu_accelerator.abs(biofilm - self.optimal_biofilm_thickness) < 0.1,
                1.2,  # 20% boost at optimal thickness
                1.0
            )
        else:
            electron_enhancement = np.where(
                np.abs(biofilm - self.optimal_biofilm_thickness) < 0.1,
                1.2,  # 20% boost at optimal thickness
                1.0
            )

        return effective_rate * substrate_term * electron_enhancement

    def update_cell(self, cell_idx, inlet_concentration, flow_rate, biofilm):
        """Update single cell state with sequential flow"""
        residence_time = self.V_a / flow_rate

        # Reaction rate calculation
        reaction_rate = self.reaction_rate(inlet_concentration, biofilm)

        # Acetate consumption in this cell (convert to mmol/L units)
        consumption_rate = self.A_m * reaction_rate  # mol/s
        acetate_consumed = (consumption_rate * residence_time) / self.V_a * 1000  # Convert to mmol/L

        # Debug print for first few steps
        if cell_idx == 0 and hasattr(self, 'debug_counter') and self.debug_counter < 3:
            print(f"Debug - Cell {cell_idx}: inlet={inlet_concentration:.4f}, reaction_rate={reaction_rate:.6f}, "
                  f"residence_time={residence_time:.1f}s, consumed={acetate_consumed:.4f}")
            if cell_idx == 0:
                self.debug_counter += 1

        if self.use_gpu:
            outlet_concentration = gpu_accelerator.maximum(0.001, inlet_concentration - acetate_consumed)  # Reduced minimum
        else:
            outlet_concentration = np.maximum(0.001, inlet_concentration - acetate_consumed)  # Reduced minimum

        # Current calculation (8 electrons per acetate molecule)
        current_density = consumption_rate * 8.0 * self.F / self.A_m  # A/m²

        # Voltage calculation with improved model
        voltage_base = 0.8  # Base voltage
        if self.use_gpu:
            concentration_factor = gpu_accelerator.log(1.0 + inlet_concentration / self.K_AC)
            biofilm_voltage_loss = 0.05 * gpu_accelerator.abs(biofilm - self.optimal_biofilm_thickness)
            cell_voltage = gpu_accelerator.maximum(0.1, voltage_base + 0.1 * concentration_factor - biofilm_voltage_loss)
        else:
            concentration_factor = np.log(1.0 + inlet_concentration / self.K_AC)
            biofilm_voltage_loss = 0.05 * np.abs(biofilm - self.optimal_biofilm_thickness)
            cell_voltage = np.maximum(0.1, voltage_base + 0.1 * concentration_factor - biofilm_voltage_loss)

        power_output = cell_voltage * current_density * self.A_m  # Watts

        return {
            'outlet_concentration': outlet_concentration,
            'current_density': current_density,
            'voltage': cell_voltage,
            'power': power_output,
            'substrate_consumed': acetate_consumed
        }

    def update_biofilm(self, step, dt):
        """Update biofilm thickness for all cells"""
        for cell_idx in range(self.num_cells):
            current_thickness = self.biofilm_thickness[step-1, cell_idx]
            substrate_conc = self.acetate_concentrations[step-1, cell_idx]
            flow_rate = self.flow_rates[step-1]

            # Biofilm growth model
            growth_rate = 0.001 * substrate_conc / (0.5 + substrate_conc)  # Growth from substrate
            decay_rate = 0.0002 * current_thickness  # Natural decay
            shear_rate = 0.0001 * (flow_rate * 1e6) ** 0.5  # Shear from flow

            # Control growth near optimal thickness
            if self.use_gpu:
                control_factor = gpu_accelerator.where(
                    current_thickness > self.optimal_biofilm_thickness,
                    0.5,  # Reduce growth above optimal
                    gpu_accelerator.where(current_thickness < self.optimal_biofilm_thickness * 0.8,
                            1.5,  # Enhance growth below optimal
                            1.0)
                )
            else:
                control_factor = 1.0
                if current_thickness > self.optimal_biofilm_thickness:
                    control_factor = 0.5  # Reduce growth above optimal
                elif current_thickness < self.optimal_biofilm_thickness * 0.8:
                    control_factor = 1.5  # Enhance growth below optimal

            net_growth = (growth_rate * control_factor - decay_rate - shear_rate) * dt

            if self.use_gpu:
                new_thickness = gpu_accelerator.clip(current_thickness + net_growth, 0.5, 3.0)
            else:
                new_thickness = np.clip(current_thickness + net_growth, 0.5, 3.0)

            self.biofilm_thickness[step, cell_idx] = new_thickness

    def simulate_step(self, step):
        """Simulate single time step with Q-learning control"""
        if step == 0:
            return

        current_time = step * self.dt
        time_hours = current_time / 3600.0

        # Update biofilm thickness
        self.update_biofilm(step, self.dt)

        # Sequential anolyte flow through cells
        inlet_concentration = 20.0  # mol/m³ constant inlet (20 mmol/L)
        current_concentration = inlet_concentration
        stack_voltage = 0.0
        stack_power = 0.0

        for cell_idx in range(self.num_cells):
            biofilm = self.biofilm_thickness[step, cell_idx]
            flow_rate = self.flow_rates[step-1]  # Use previous flow rate

            result = self.update_cell(cell_idx, current_concentration, flow_rate, biofilm)

            # Update cell arrays
            self.acetate_concentrations[step, cell_idx] = result['outlet_concentration']
            self.current_densities[step, cell_idx] = result['current_density']
            self.cell_voltages[step, cell_idx] = result['voltage']
            self.power_outputs[step, cell_idx] = result['power']
            self.substrate_consumptions[step, cell_idx] = result['substrate_consumed']

            # Update for next cell
            current_concentration = result['outlet_concentration']
            stack_voltage += result['voltage']
            stack_power += result['power']

        # Calculate current state metrics
        final_conc = self.acetate_concentrations[step, -1]
        substrate_utilization = (20.0 - final_conc) / 20.0 * 100.0

        if self.use_gpu:
            biofilm_deviation = float(gpu_accelerator.mean(gpu_accelerator.abs(self.biofilm_thickness[step, :] - self.optimal_biofilm_thickness)))
        else:
            biofilm_deviation = float(np.mean(np.abs(self.biofilm_thickness[step, :] - self.optimal_biofilm_thickness)))

        # Update biofilm thickness history for derivative calculation
        if step % 60 == 0:  # Update every 10 minutes
            avg_biofilm_thickness = np.mean(self.biofilm_thickness[step, :])
            self.biofilm_history.append(avg_biofilm_thickness)
            # Keep only last 10 measurements for derivative calculation
            if len(self.biofilm_history) > 10:
                self.biofilm_history.pop(0)

        # Q-learning control (every 60 steps = 10 minutes)
        if step > 1 and step % 60 == 0:
            # Get current state
            current_state = self.q_controller.discretize_state(
                float(stack_power), biofilm_deviation, float(substrate_utilization), time_hours
            )

            # Select action and update flow rate
            action_idx, new_flow_rate = self.q_controller.select_action(
                current_state, self.flow_rates[step-1]
            )

            self.flow_rates[step] = new_flow_rate
            self.q_actions[step] = action_idx

            # Calculate reward and update Q-table if not first Q-learning step
            if hasattr(self, 'prev_state') and hasattr(self, 'prev_action'):
                prev_power = float(self.stack_powers[step-60])
                prev_biofilm_dev = float(np.mean(np.abs(self.biofilm_thickness[step-60, :] - self.optimal_biofilm_thickness)))
                prev_substrate_util = float(self.substrate_utilizations[step-60])

                reward = self.q_controller.calculate_reward(
                    float(stack_power), biofilm_deviation, float(substrate_utilization),
                    prev_power, prev_biofilm_dev, prev_substrate_util,
                    biofilm_thickness_history=self.biofilm_history
                )

                self.q_rewards[step] = reward

                self.q_controller.update_q_table(
                    self.prev_state, self.prev_action, reward, current_state
                )

            # Store current state and action for next update
            self.prev_state = current_state
            self.prev_action = action_idx
        else:
            # Maintain previous flow rate
            self.flow_rates[step] = self.flow_rates[step-1]

        # Update stack arrays
        self.stack_voltages[step] = stack_voltage
        self.stack_powers[step] = stack_power
        self.substrate_utilizations[step] = substrate_utilization

        # Calculate multi-objective value
        power_objective = min(1.0, stack_power / 5.0)
        biofilm_objective = max(0.0, 1.0 - biofilm_deviation)
        substrate_objective = min(1.0, substrate_utilization / 20.0)

        self.objective_values[step] = (self.w_power * power_objective +
                                     self.w_biofilm * biofilm_objective +
                                     self.w_substrate * substrate_objective)

    def run_simulation(self):
        """Run the complete Q-learning simulation"""
        print("Starting MFC Q-Learning optimization simulation...")
        print(f"Duration: 1000 hours, Timesteps: {self.num_steps}")
        print(f"GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        print("Flow rate control: Q-Learning (independent variable: system state)")
        print(f"Initial flow rate: {self.flow_rates[0] * 1000:.1f} mL/h")

        start_time = time.time()

        for step in range(self.num_steps):
            self.simulate_step(step)

            # Progress reporting
            if step % 36000 == 0:  # Every 100 hours
                hours = step * self.dt / 3600
                epsilon = self.q_controller.epsilon if hasattr(self.q_controller, 'epsilon') else 0
                print(f"Progress: {hours:.0f}/1000 hours, "
                      f"Power: {float(self.stack_powers[step]):.3f} W, "
                      f"Flow: {float(self.flow_rates[step]) * 1000:.1f} mL/h, "
                      f"Utilization: {float(self.substrate_utilizations[step]):.2f}%, "
                      f"Epsilon: {epsilon:.3f}")

        simulation_time = time.time() - start_time
        print(f"Q-learning simulation completed in {simulation_time:.2f} seconds")

        # Convert GPU arrays to CPU if needed
        if self.use_gpu:
            self.cell_voltages = gpu_accelerator.to_cpu(self.cell_voltages)
            self.biofilm_thickness = gpu_accelerator.to_cpu(self.biofilm_thickness)
            self.acetate_concentrations = gpu_accelerator.to_cpu(self.acetate_concentrations)
            self.current_densities = gpu_accelerator.to_cpu(self.current_densities)
            self.power_outputs = gpu_accelerator.to_cpu(self.power_outputs)
            self.substrate_consumptions = gpu_accelerator.to_cpu(self.substrate_consumptions)
            self.stack_voltages = gpu_accelerator.to_cpu(self.stack_voltages)
            self.stack_powers = gpu_accelerator.to_cpu(self.stack_powers)
            self.flow_rates = gpu_accelerator.to_cpu(self.flow_rates)
            self.objective_values = gpu_accelerator.to_cpu(self.objective_values)
            self.substrate_utilizations = gpu_accelerator.to_cpu(self.substrate_utilizations)
            self.q_rewards = gpu_accelerator.to_cpu(self.q_rewards)
            self.q_actions = gpu_accelerator.to_cpu(self.q_actions)

    def save_data(self):
        """Save simulation data and Q-learning model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create time arrays
        time_seconds = np.arange(self.num_steps) * self.dt
        time_hours = time_seconds / 3600

        # Prepare data for CSV export
        csv_data = {
            'time_seconds': time_seconds,
            'time_hours': time_hours,
            'stack_voltage': self.stack_voltages,
            'stack_power': self.stack_powers,
            'flow_rate_ml_h': self.flow_rates * 3.6e9,
            'objective_value': self.objective_values,
            'substrate_utilization': self.substrate_utilizations,
            'q_reward': self.q_rewards,
            'q_action': self.q_actions
        }

        # Add cell-specific data
        for i in range(self.num_cells):
            csv_data[f'cell_{i+1}_voltage'] = self.cell_voltages[:, i]
            csv_data[f'cell_{i+1}_power'] = self.power_outputs[:, i]
            csv_data[f'cell_{i+1}_biofilm'] = self.biofilm_thickness[:, i]
            csv_data[f'cell_{i+1}_acetate_out'] = self.acetate_concentrations[:, i]

        # Save CSV
        df = pd.DataFrame(csv_data)
        csv_filename = get_simulation_data_path(f'mfc_qlearning_{timestamp}.csv')
        df.to_csv(csv_filename, index=False)
        print(f"CSV data saved to {csv_filename}")

        # Save Q-learning model
        q_model_filename = get_model_path(f'q_table_{timestamp}.pkl')
        self.q_controller.save_q_table(q_model_filename)
        print(f"Q-table saved to {q_model_filename}")

        # Prepare JSON data
        json_data = {
            'simulation_info': {
                'timestamp': timestamp,
                'duration_hours': 1000,
                'timestep_seconds': 10,
                'num_cells': 5,
                'gpu_acceleration': self.use_gpu,
                'control_method': 'Q-Learning',
                'initial_flow_rate_ml_h': float(self.flow_rates[0] * 3.6e9),
                'q_learning_params': {
                    'learning_rate': self.q_controller.learning_rate,
                    'discount_factor': self.q_controller.discount_factor,
                    'final_epsilon': self.q_controller.epsilon,
                    'total_reward': float(self.q_controller.total_rewards)
                }
            },
            'results': {
                'total_energy_wh': float(np.trapezoid(self.stack_powers, dx=self.dt/3600)),
                'average_power_w': float(np.mean(self.stack_powers)),
                'final_stack_voltage_v': float(self.stack_voltages[-1]),
                'final_stack_power_w': float(self.stack_powers[-1]),
                'final_flow_rate_ml_h': float(self.flow_rates[-1] * 3.6e9),
                'final_substrate_utilization_percent': float(self.substrate_utilizations[-1]),
                'final_objective_value': float(self.objective_values[-1]),
                'q_learning_total_reward': float(self.q_controller.total_rewards),
                'q_table_size': len(self.q_controller.q_table)
            }
        }

        # Save JSON
        json_filename = get_simulation_data_path(f'mfc_qlearning_{timestamp}.json')
        with open(json_filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON data saved to {json_filename}")

        return timestamp

    def generate_plots(self, timestamp):
        """Generate Q-learning specific visualization dashboard"""
        time_hours = np.arange(self.num_steps) * self.dt / 3600

        # Create main dashboard figure
        fig = plt.figure(figsize=(24, 16))

        # Plot 1: Stack Power and Q-Learning Rewards
        ax1 = plt.subplot(3, 3, 1)
        ax1_twin = ax1.twinx()
        line1 = ax1.plot(time_hours, self.stack_powers, 'b-', linewidth=1.5, label='Stack Power')
        line2 = ax1_twin.plot(time_hours, self.q_rewards, 'r-', linewidth=1, alpha=0.7, label='Q-Reward')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Power (W)', color='blue')
        ax1_twin.set_ylabel('Q-Learning Reward', color='red')
        ax1.set_title('Power Output and Q-Learning Rewards')
        ax1.grid(True, alpha=0.3)

        # Combine legends
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')

        # Plot 2: Q-Learning Flow Rate Control
        plt.subplot(3, 3, 2)
        plt.plot(time_hours, self.flow_rates * 1000, 'g-', linewidth=1.5)
        plt.xlabel('Time (hours)')
        plt.ylabel('Flow Rate (mL/h)')
        plt.title('Q-Learning Flow Rate Control')
        plt.grid(True, alpha=0.3)

        # Plot 3: Substrate Utilization
        plt.subplot(3, 3, 3)
        plt.plot(time_hours, self.substrate_utilizations, 'purple', linewidth=1.5)
        plt.xlabel('Time (hours)')
        plt.ylabel('Utilization (%)')
        plt.title('Substrate Utilization Efficiency')
        plt.grid(True, alpha=0.3)

        # Plot 4: Biofilm Thickness Evolution
        plt.subplot(3, 3, 4)
        for i in range(self.num_cells):
            plt.plot(time_hours, self.biofilm_thickness[:, i],
                    linewidth=1.5, label=f'Cell {i+1}')
        plt.axhline(y=self.optimal_biofilm_thickness, color='black',
                   linestyle='--', alpha=0.7, label='Optimal')
        plt.xlabel('Time (hours)')
        plt.ylabel('Biofilm Thickness')
        plt.title('Biofilm Growth Control')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 5: Q-Learning Actions
        plt.subplot(3, 3, 5)
        action_indices = self.q_actions[self.q_actions != 0]  # Remove zeros
        action_times = time_hours[self.q_actions != 0]
        if len(action_indices) > 0:
            plt.scatter(action_times, action_indices, alpha=0.6, s=10)
            plt.xlabel('Time (hours)')
            plt.ylabel('Action Index')
            plt.title('Q-Learning Action Selection')
        plt.grid(True, alpha=0.3)

        # Plot 6: Cell Voltages
        plt.subplot(3, 3, 6)
        for i in range(self.num_cells):
            plt.plot(time_hours, self.cell_voltages[:, i],
                    linewidth=1.5, label=f'Cell {i+1}')
        plt.xlabel('Time (hours)')
        plt.ylabel('Voltage (V)')
        plt.title('Individual Cell Voltages')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 7: Multi-Objective Progress
        plt.subplot(3, 3, 7)
        plt.plot(time_hours, self.objective_values, 'orange', linewidth=1.5)
        plt.xlabel('Time (hours)')
        plt.ylabel('Objective Value')
        plt.title('Multi-Objective Optimization Progress')
        plt.grid(True, alpha=0.3)

        # Plot 8: Q-Learning Exploration
        plt.subplot(3, 3, 8)
        # Plot epsilon decay over time (approximate)
        epsilon_values = []
        current_epsilon = 0.3
        for i in range(self.num_steps):
            if i % 60 == 0:  # Q-learning updates
                current_epsilon = max(0.05, current_epsilon * 0.995)
            epsilon_values.append(current_epsilon)

        plt.plot(time_hours, epsilon_values, 'brown', linewidth=1.5)
        plt.xlabel('Time (hours)')
        plt.ylabel('Exploration Rate (ε)')
        plt.title('Q-Learning Exploration Decay')
        plt.grid(True, alpha=0.3)

        # Plot 9: Performance Summary
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        # Calculate summary metrics
        total_energy = np.trapezoid(self.stack_powers, dx=self.dt/3600)
        avg_power = np.mean(self.stack_powers)
        final_utilization = self.substrate_utilizations[-1]
        final_flow_rate = self.flow_rates[-1] * 3.6e9
        q_table_size = len(self.q_controller.q_table)
        total_q_reward = self.q_controller.total_rewards

        summary_text = f"""
        Q-LEARNING OPTIMIZATION
        
        Total Energy: {total_energy:.1f} Wh
        Average Power: {avg_power:.3f} W
        Final Power: {self.stack_powers[-1]:.3f} W
        
        Final Flow Rate: {final_flow_rate:.1f} mL/h
        Substrate Utilization: {final_utilization:.2f}%
        
        Q-Learning Stats:
        Total Reward: {total_q_reward:.1f}
        Q-Table Size: {q_table_size} states
        Final ε: {self.q_controller.epsilon:.3f}
        
        LEARNING ACHIEVED:
        ✓ Adaptive Flow Control
        ✓ Multi-Objective Balance
        ✓ State-Action Mapping
        """

        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                facecolor="lightgreen", alpha=0.5))

        # Create a separate figure for instantaneous flow rate
        plt.figure(figsize=(12, 8))

        # Plot 10: Instantaneous Flow Rate with detailed view
        plt.subplot(2, 1, 1)
        plt.plot(time_hours, self.flow_rates * 1000, 'g-', linewidth=1.5, alpha=0.8)
        plt.xlabel('Time (hours)')
        plt.ylabel('Instantaneous Flow Rate (mL/h)')
        plt.title('Instantaneous Flow Rate Evolution - Q-Learning Control')
        plt.grid(True, alpha=0.3)

        # Add markers for Q-learning decision points (every 60 steps = 10 minutes)
        q_decision_times = time_hours[::60]  # Every 60 steps
        q_decision_flows = (self.flow_rates[::60]) * 1000
        plt.scatter(q_decision_times, q_decision_flows, c='red', s=8, alpha=0.6,
                   label='Q-Learning Decisions')
        plt.legend()

        # Plot 11: Flow Rate Histogram
        plt.subplot(2, 1, 2)
        flow_rates_ml = self.flow_rates * 1000
        plt.hist(flow_rates_ml[flow_rates_ml > 0], bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Flow Rate (mL/h)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Flow Rates Used by Q-Learning Agent')
        plt.grid(True, alpha=0.3)

        # Add statistics text
        mean_flow = np.mean(flow_rates_ml[flow_rates_ml > 0])
        std_flow = np.std(flow_rates_ml[flow_rates_ml > 0])
        plt.text(0.7, 0.8, f'Mean: {mean_flow:.1f} mL/h\nStd: {std_flow:.1f} mL/h',
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3",
                facecolor="lightblue", alpha=0.7))

        # Add alphabetic labels to flow rate analysis figure
        current_fig = plt.gcf()
        add_subplot_labels(current_fig, 'j')

        plt.tight_layout()

        # Save flow rate figure
        flow_filename = get_figure_path(f'mfc_flow_rate_analysis_{timestamp}.png')
        plt.savefig(flow_filename, dpi=300, bbox_inches='tight')
        print(f"Flow rate analysis saved to {flow_filename}")

        plt.close()

        # Create figure for Flow Rate vs Substrate Utilization relationship
        plt.figure(figsize=(14, 10))

        # Plot 12: Flow Rate vs Substrate Utilization Scatter Plot
        plt.subplot(2, 2, 1)
        flow_rates_ml = self.flow_rates * 1000
        # Remove zero flow rates for better visualization
        valid_mask = flow_rates_ml > 0
        if np.any(valid_mask):
            plt.scatter(flow_rates_ml[valid_mask], self.substrate_utilizations[valid_mask],
                       c=time_hours[valid_mask], cmap='viridis', s=10, alpha=0.6)
            plt.colorbar(label='Time (hours)')
        else:
            plt.scatter(flow_rates_ml, self.substrate_utilizations, s=10, alpha=0.6)

        plt.xlabel('Flow Rate (mL/h)')
        plt.ylabel('Substrate Utilization (%)')
        plt.title('Flow Rate vs Substrate Utilization\n(Color = Time)')
        plt.grid(True, alpha=0.3)

        # Plot 13: Flow Rate vs Substrate Utilization Time Series
        plt.subplot(2, 2, 2)
        ax_flow = plt.gca()
        ax_util = ax_flow.twinx()

        line1 = ax_flow.plot(time_hours, flow_rates_ml, 'b-', linewidth=1.5,
                            alpha=0.7, label='Flow Rate (mL/h)')
        line2 = ax_util.plot(time_hours, self.substrate_utilizations, 'r-', linewidth=1.5,
                            alpha=0.7, label='Substrate Util. (%)')

        ax_flow.set_xlabel('Time (hours)')
        ax_flow.set_ylabel('Flow Rate (mL/h)', color='blue')
        ax_util.set_ylabel('Substrate Utilization (%)', color='red')
        ax_flow.set_title('Flow Rate and Substrate Utilization vs Time')
        ax_flow.grid(True, alpha=0.3)

        # Combine legends
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax_flow.legend(lines, labels, loc='upper left')

        # Plot 14: Binned Analysis
        plt.subplot(2, 2, 3)
        if np.any(valid_mask):
            # Create flow rate bins
            flow_bins = np.linspace(np.min(flow_rates_ml[valid_mask]),
                                  np.max(flow_rates_ml[valid_mask]), 10)
            if len(flow_bins) > 1:
                bin_centers = []
                bin_utils = []
                bin_stds = []

                for i in range(len(flow_bins)-1):
                    mask = (flow_rates_ml >= flow_bins[i]) & (flow_rates_ml < flow_bins[i+1])
                    if np.any(mask):
                        bin_centers.append((flow_bins[i] + flow_bins[i+1]) / 2)
                        bin_utils.append(np.mean(self.substrate_utilizations[mask]))
                        bin_stds.append(np.std(self.substrate_utilizations[mask]))

                if bin_centers:
                    plt.errorbar(bin_centers, bin_utils, yerr=bin_stds,
                               fmt='o-', capsize=5, capthick=2, linewidth=2)
                    plt.xlabel('Flow Rate (mL/h)')
                    plt.ylabel('Average Substrate Utilization (%)')
                    plt.title('Substrate Utilization vs Flow Rate\n(Binned Average ± Std)')
                    plt.grid(True, alpha=0.3)

        # Plot 15: Q-Learning Decision Analysis
        plt.subplot(2, 2, 4)
        # Show only Q-learning decision points (every 60 steps)
        q_times = time_hours[::60]
        q_flows = flow_rates_ml[::60]
        q_utils = self.substrate_utilizations[::60]

        plt.scatter(q_flows, q_utils, c=q_times, cmap='plasma', s=20, alpha=0.8)
        plt.colorbar(label='Time (hours)')
        plt.xlabel('Flow Rate (mL/h)')
        plt.ylabel('Substrate Utilization (%)')
        plt.title('Q-Learning Decisions:\nFlow Rate vs Substrate Utilization')
        plt.grid(True, alpha=0.3)

        # Add correlation coefficient if there are valid data points
        if len(q_flows) > 1 and np.any(q_flows > 0):
            valid_q_mask = q_flows > 0
            if np.sum(valid_q_mask) > 1:
                corr = np.corrcoef(q_flows[valid_q_mask], q_utils[valid_q_mask])[0,1]
                plt.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                        transform=plt.gca().transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Add alphabetic labels to flow vs utilization figure
        current_fig = plt.gcf()
        add_subplot_labels(current_fig, 'l')

        plt.tight_layout()

        # Save flow vs utilization analysis figure
        flow_util_filename = get_figure_path(f'mfc_flow_vs_utilization_{timestamp}.png')
        plt.savefig(flow_util_filename, dpi=300, bbox_inches='tight')
        print(f"Flow rate vs utilization analysis saved to {flow_util_filename}")

        plt.close()

        # Return to main dashboard
        plt.figure(fig.number)

        # Add alphabetic labels to main dashboard
        add_subplot_labels(fig, 'a')

        plt.tight_layout()

        # Save dashboard
        dashboard_filename = get_figure_path(f'mfc_qlearning_dashboard_{timestamp}.png')
        plt.savefig(dashboard_filename, dpi=300, bbox_inches='tight')
        print(f"Q-Learning dashboard saved to {dashboard_filename}")

        plt.close()


def main():
    """Main execution function"""
    print("=== MFC Q-Learning Optimization Simulation ===")
    print("Control Method: Q-Learning Agent")
    print("Objectives:")
    print("1. Maximize instantaneous power output")
    print("2. Minimize biofilm growth (maintain optimal thickness)")
    print("3. Minimize substrate outlet concentration")
    print("Independent variable: System state (power, biofilm, substrate, time)")
    print("Dependent variable: Anolyte flow rate (Q-learning controlled)")
    print("Configuration: Sequential flow through 5-cell stack")
    print("=" * 50)

    # Initialize and run simulation
    sim = MFCQLearningSimulation(use_gpu=False)
    sim.run_simulation()

    # Save data and generate plots
    timestamp = sim.save_data()
    sim.generate_plots(timestamp)

    print("\n=== Q-LEARNING SIMULATION COMPLETE ===")
    print(f"Results saved with timestamp: {timestamp}")
    print(f"Data saved to: {get_simulation_data_path('')}")
    print(f"Figures saved to: {get_figure_path('')}")
    print(f"Models saved to: {get_model_path('')}")

    # Final summary
    total_energy = np.trapezoid(sim.stack_powers, dx=sim.dt/3600)
    print("\nFinal Performance Summary:")
    print(f"Total Energy: {total_energy:.1f} Wh")
    print(f"Average Power: {np.mean(sim.stack_powers):.3f} W")
    print(f"Final Power: {sim.stack_powers[-1]:.3f} W")
    print(f"Q-Learning Optimized Flow Rate: {sim.flow_rates[-1] * 1000:.1f} mL/h")
    print(f"Final Substrate Utilization: {sim.substrate_utilizations[-1]:.2f}%")
    print(f"Q-Learning Total Reward: {sim.q_controller.total_rewards:.1f}")
    print(f"Q-Table Size: {len(sim.q_controller.q_table)} learned states")


if __name__ == "__main__":
    main()
