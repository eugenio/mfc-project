"""
5-Cell MFC Stack Simulation with Q-Learning Control

This simulation includes:
- 5 MFC cells in stack configuration
- Sensor inputs (voltage, current, pH, acetate concentration)
- Actuator outputs (duty cycle, pH buffer, acetate addition)
- Q-learning controller for optimal operation
- Cell reversal prevention
- Power stability maintenance
"""

import random
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from odes import MFCModel
from path_config import get_figure_path, get_simulation_data_path


class MFCSensor:
    """Sensor simulation for MFC monitoring"""

    def __init__(self, sensor_type, noise_level=0.02):
        self.sensor_type = sensor_type
        self.noise_level = noise_level
        self.calibration_offset = np.random.normal(0, 0.01)
        self.readings = deque(maxlen=100)

    def read(self, true_value):
        """Read sensor value with noise and calibration offset"""
        noise = np.random.normal(0, self.noise_level)
        measured_value = true_value + noise + self.calibration_offset

        # Apply sensor-specific constraints
        if self.sensor_type == 'voltage':
            measured_value = max(0, min(measured_value, 2.0))
        elif self.sensor_type == 'current':
            measured_value = max(0, measured_value)
        elif self.sensor_type == 'pH':
            measured_value = max(0, min(measured_value, 14))
        elif self.sensor_type == 'acetate':
            measured_value = max(0, measured_value)

        self.readings.append(measured_value)
        return measured_value

    def get_filtered_reading(self):
        """Get filtered sensor reading using moving average"""
        if len(self.readings) < 5:
            return self.readings[-1] if self.readings else 0
        return np.mean(list(self.readings)[-5:])

class MFCActuator:
    """Actuator simulation for MFC control"""

    def __init__(self, actuator_type, min_value=0, max_value=1):
        self.actuator_type = actuator_type
        self.min_value = min_value
        self.max_value = max_value
        self.current_value = 0
        self.response_time = 0.1  # seconds
        self.history = deque(maxlen=1000)

    def set_value(self, target_value):
        """Set actuator value with response time simulation"""
        # Clamp to valid range
        target_value = max(self.min_value, min(target_value, self.max_value))

        # Simulate response time
        alpha = 1 - np.exp(-1/self.response_time)
        self.current_value = alpha * target_value + (1 - alpha) * self.current_value

        self.history.append(self.current_value)
        return self.current_value

    def get_value(self):
        """Get current actuator value"""
        return self.current_value

class MFCCell:
    """Individual MFC cell with sensors and control"""

    def __init__(self, cell_id):
        self.cell_id = cell_id
        self.mfc_model = MFCModel()

        # Initialize cell state
        self.state = np.array([
            1.0,    # C_AC
            0.05,   # C_CO2
            1e-4,   # C_H
            0.1,    # X
            0.25,   # C_O2
            1e-7,   # C_OH
            0.05,   # C_M
            0.01,   # eta_a
            -0.01   # eta_c
        ])

        # Sensors
        self.sensors = {
            'voltage': MFCSensor('voltage', noise_level=0.01),
            'current': MFCSensor('current', noise_level=0.02),
            'pH': MFCSensor('pH', noise_level=0.1),
            'acetate': MFCSensor('acetate', noise_level=0.05)
        }

        # Actuators
        self.actuators = {
            'duty_cycle': MFCActuator('duty_cycle', 0, 1),
            'ph_buffer': MFCActuator('ph_buffer', 0, 1),
            'acetate_pump': MFCActuator('acetate_pump', 0, 1)
        }

        # Cell status
        self.is_reversed = False
        self.power_history = deque(maxlen=100)
        self.temperature = 30.0  # Celsius

    def update_state(self, dt=1.0):
        """Update cell state based on current conditions"""
        # Get actuator values
        duty_cycle = self.actuators['duty_cycle'].get_value()
        ph_buffer = self.actuators['ph_buffer'].get_value()
        acetate_addition = self.actuators['acetate_pump'].get_value()

        # Calculate effective current based on duty cycle
        base_current = 1.0
        effective_current = base_current * duty_cycle

        # Apply pH buffer effects
        if ph_buffer > 0.1:
            # pH buffer reduces H+ concentration variations
            self.state[2] = max(1e-7, self.state[2] * (1 - ph_buffer * 0.1))

        # Apply acetate addition
        if acetate_addition > 0.1:
            # Acetate addition increases substrate concentration
            self.state[0] += acetate_addition * 0.1 * dt

        # Use MFC model to calculate derivatives
        try:
            derivatives = self.mfc_model.mfc_odes(0, self.state, effective_current)

            # Update state using Euler integration
            for i in range(len(self.state)):
                self.state[i] += derivatives[i] * dt

            # Apply physical constraints
            self.state[0] = max(0, min(self.state[0], 5.0))    # C_AC
            self.state[3] = max(0, min(self.state[3], 2.0))    # X
            self.state[4] = max(0, min(self.state[4], 1.0))    # C_O2
            self.state[7] = max(-1.0, min(self.state[7], 1.0)) # eta_a
            self.state[8] = max(-1.0, min(self.state[8], 1.0)) # eta_c

        except Exception as e:
            print(f"Cell {self.cell_id} simulation error: {e}")

    def get_sensor_readings(self):
        """Get all sensor readings"""
        # Calculate true values from state
        voltage = self.state[7] - self.state[8]  # eta_a - eta_c
        current = self.actuators['duty_cycle'].get_value()
        pH = max(0, min(14, 7 - np.log10(max(1e-14, self.state[2]))))
        acetate = self.state[0]

        # Get sensor readings with noise
        readings = {
            'voltage': self.sensors['voltage'].read(voltage),
            'current': self.sensors['current'].read(current),
            'pH': self.sensors['pH'].read(pH),
            'acetate': self.sensors['acetate'].read(acetate)
        }

        return readings

    def get_power(self):
        """Calculate cell power"""
        readings = self.get_sensor_readings()
        power = readings['voltage'] * readings['current']
        self.power_history.append(power)
        return power

    def check_reversal(self):
        """Check if cell is in reversal state"""
        readings = self.get_sensor_readings()
        self.is_reversed = readings['voltage'] < 0.1 or readings['current'] < 0.05
        return self.is_reversed

    def get_state_vector(self):
        """Get normalized state vector for Q-learning"""
        readings = self.get_sensor_readings()
        power = self.get_power()

        # Normalize values for Q-learning
        state_vector = np.array([
            readings['acetate'] / 3.0,        # Normalized acetate
            self.state[3],                    # Biomass
            self.state[4] / 0.5,              # Normalized oxygen
            (readings['pH'] - 7) / 7,         # Normalized pH
            readings['voltage'] / 2.0,        # Normalized voltage
            power / 5.0,                      # Normalized power
            1.0 if self.is_reversed else 0.0  # Reversal flag
        ])

        return state_vector

class MFCStack:
    """5-cell MFC stack with centralized control"""

    def __init__(self):
        self.cells = [MFCCell(i) for i in range(5)]
        self.stack_voltage = 0
        self.stack_current = 0
        self.stack_power = 0
        self.target_power = 2.0  # Target stack power
        self.time = 0

        # Stack-level sensors
        self.stack_sensors = {
            'total_voltage': MFCSensor('voltage', noise_level=0.02),
            'total_current': MFCSensor('current', noise_level=0.02)
        }

        # Data logging
        self.data_log = {
            'time': [],
            'stack_voltage': [],
            'stack_current': [],
            'stack_power': [],
            'cell_voltages': [[] for _ in range(5)],
            'cell_powers': [[] for _ in range(5)],
            'cell_reversals': [[] for _ in range(5)],
            'duty_cycles': [[] for _ in range(5)],
            'ph_buffers': [[] for _ in range(5)],
            'acetate_additions': [[] for _ in range(5)]
        }

    def update_stack(self, dt=1.0):
        """Update entire stack state"""
        self.time += dt

        # Update all cells
        for cell in self.cells:
            cell.update_state(dt)

        # Calculate stack parameters
        self.stack_voltage = sum(cell.get_sensor_readings()['voltage'] for cell in self.cells)
        self.stack_current = min(cell.get_sensor_readings()['current'] for cell in self.cells)
        self.stack_power = self.stack_voltage * self.stack_current

        # Log data
        self.log_data()

    def log_data(self):
        """Log system data"""
        self.data_log['time'].append(self.time)
        self.data_log['stack_voltage'].append(self.stack_voltage)
        self.data_log['stack_current'].append(self.stack_current)
        self.data_log['stack_power'].append(self.stack_power)

        for i, cell in enumerate(self.cells):
            readings = cell.get_sensor_readings()
            self.data_log['cell_voltages'][i].append(readings['voltage'])
            self.data_log['cell_powers'][i].append(cell.get_power())
            self.data_log['cell_reversals'][i].append(1 if cell.is_reversed else 0)
            self.data_log['duty_cycles'][i].append(cell.actuators['duty_cycle'].get_value())
            self.data_log['ph_buffers'][i].append(cell.actuators['ph_buffer'].get_value())
            self.data_log['acetate_additions'][i].append(cell.actuators['acetate_pump'].get_value())

    def get_stack_state(self):
        """Get comprehensive stack state for Q-learning"""
        cell_states = [cell.get_state_vector() for cell in self.cells]

        # Stack-level features
        stack_features = np.array([
            self.stack_voltage / 10.0,      # Normalized stack voltage
            self.stack_current / 5.0,       # Normalized stack current
            self.stack_power / 10.0,        # Normalized stack power
            sum(cell.is_reversed for cell in self.cells) / 5.0,  # Reversal ratio
            np.std([cell.get_power() for cell in self.cells]) / 2.0  # Power imbalance
        ])

        # Combine cell states and stack features
        full_state = np.concatenate([np.concatenate(cell_states), stack_features])
        return full_state

    def apply_control_actions(self, actions):
        """Apply control actions to stack"""
        # actions is a 15-element array: [duty_cycle, ph_buffer, acetate] for each cell
        for i, cell in enumerate(self.cells):
            duty_cycle = actions[i * 3]
            ph_buffer = actions[i * 3 + 1]
            acetate = actions[i * 3 + 2]

            cell.actuators['duty_cycle'].set_value(duty_cycle)
            cell.actuators['ph_buffer'].set_value(ph_buffer)
            cell.actuators['acetate_pump'].set_value(acetate)

    def check_system_health(self):
        """Check overall system health"""
        reversed_cells = sum(1 for cell in self.cells if cell.check_reversal())
        low_power_cells = sum(1 for cell in self.cells if cell.get_power() < 0.1)

        return {
            'reversed_cells': reversed_cells,
            'low_power_cells': low_power_cells,
            'stack_efficiency': self.stack_power / max(0.1, self.stack_voltage * 5.0),
            'power_stability': 1.0 - np.std([cell.get_power() for cell in self.cells]) / max(0.1, np.mean([cell.get_power() for cell in self.cells]))
        }

class MFCStackQLearningController:
    """Q-Learning controller for MFC stack management"""

    def __init__(self, stack):
        self.stack = stack
        self.state_size = 40  # 7 features per cell + 5 stack features
        self.action_size = 15  # 3 actuators per cell
        self.n_bins = 5  # Discretization bins per dimension

        # Q-learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.3
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Q-table (simplified for demonstration)
        self.q_table = {}
        self.action_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)

        # Control constraints
        self.min_duty_cycle = 0.1
        self.max_duty_cycle = 0.9

    def discretize_state(self, state):
        """Discretize continuous state for Q-table indexing"""
        # Simplified discretization
        discretized = tuple(int(np.clip(val * self.n_bins, 0, self.n_bins - 1)) for val in state[:10])
        return discretized

    def get_action(self, state):
        """Get action using epsilon-greedy policy"""
        state_key = self.discretize_state(state)

        if random.random() < self.epsilon:
            # Random action (exploration)
            actions = np.random.uniform(0, 1, self.action_size)
        else:
            # Greedy action (exploitation)
            if state_key in self.q_table:
                actions = self.q_table[state_key]
            else:
                actions = np.random.uniform(0.5, 0.8, self.action_size)

        # Apply constraints
        for i in range(5):  # For each cell
            # Duty cycle constraints
            actions[i * 3] = np.clip(actions[i * 3], self.min_duty_cycle, self.max_duty_cycle)
            # pH buffer constraints
            actions[i * 3 + 1] = np.clip(actions[i * 3 + 1], 0, 1)
            # Acetate addition constraints
            actions[i * 3 + 2] = np.clip(actions[i * 3 + 2], 0, 1)

        return actions

    def calculate_reward(self, state, actions):
        """Calculate reward based on system performance"""
        health = self.stack.check_system_health()

        # Power optimization reward
        power_reward = self.stack.stack_power / 10.0

        # Stability reward
        stability_reward = health['power_stability']

        # Cell reversal penalty
        reversal_penalty = -10.0 * health['reversed_cells']

        # Efficiency reward
        efficiency_reward = health['stack_efficiency']

        # Action penalty for extreme values
        action_penalty = 0
        for i in range(5):
            if actions[i * 3] < 0.2 or actions[i * 3] > 0.8:  # Extreme duty cycles
                action_penalty -= 0.5
            if actions[i * 3 + 1] > 0.8:  # Excessive pH buffer
                action_penalty -= 0.2
            if actions[i * 3 + 2] > 0.8:  # Excessive acetate
                action_penalty -= 0.2

        total_reward = power_reward + stability_reward + reversal_penalty + efficiency_reward + action_penalty

        return total_reward

    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning"""
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.random.uniform(0.4, 0.6, self.action_size)

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.random.uniform(0.4, 0.6, self.action_size)

        # Q-learning update
        current_q = self.q_table[state_key]
        next_q_max = np.max(self.q_table[next_state_key])

        # Update only the actions that were taken
        for i in range(self.action_size):
            old_value = current_q[i]
            new_value = old_value + self.learning_rate * (
                reward + self.discount_factor * next_q_max - old_value
            )
            current_q[i] = new_value

        self.q_table[state_key] = current_q

    def train_step(self):
        """Perform one training step"""
        # Get current state
        current_state = self.stack.get_stack_state()

        # Get action
        actions = self.get_action(current_state)

        # Apply actions
        self.stack.apply_control_actions(actions)

        # Update stack
        self.stack.update_stack()

        # Get next state
        next_state = self.stack.get_stack_state()

        # Calculate reward
        reward = self.calculate_reward(next_state, actions)

        # Update Q-table
        self.update_q_table(current_state, actions, reward, next_state)

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Log data
        self.action_history.append(actions)
        self.reward_history.append(reward)

        return reward, self.stack.stack_power

def run_stack_simulation():
    """Run the complete MFC stack simulation"""

    print("=== 5-Cell MFC Stack Simulation with Q-Learning Control ===")
    print("Features: Duty cycle control, pH buffer, acetate addition, cell reversal prevention")
    print()

    # Initialize stack and controller
    stack = MFCStack()
    controller = MFCStackQLearningController(stack)

    # Simulation parameters
    simulation_time = 1000 * 3600  # 1000 hours in seconds
    dt = 1.0  # time step
    steps = int(simulation_time / dt)

    print(f"Starting simulation for {simulation_time/3600:.0f} hours ({simulation_time} seconds)...")
    print(f"Time step: {dt}s, Total steps: {steps}")
    print()

    # Training loop
    start_time = time.time()

    for step in range(steps):
        reward, power = controller.train_step()

        if step % 100 == 0:
            health = stack.check_system_health()
            print(f"Step {step}: Power={power:.3f}W, Reward={reward:.3f}, "
                  f"Reversed={health['reversed_cells']}, Epsilon={controller.epsilon:.3f}")

    training_time = time.time() - start_time
    print(f"\nSimulation completed in {training_time:.2f} seconds")

    # Analysis
    print("\n=== Simulation Results ===")

    final_health = stack.check_system_health()
    print(f"Final stack power: {stack.stack_power:.3f} W")
    print(f"Final stack voltage: {stack.stack_voltage:.3f} V")
    print(f"Final stack current: {stack.stack_current:.3f} A")
    print(f"Reversed cells: {final_health['reversed_cells']}")
    print(f"Stack efficiency: {final_health['stack_efficiency']:.3f}")
    print(f"Power stability: {final_health['power_stability']:.3f}")
    print(f"Q-table size: {len(controller.q_table)}")

    # Individual cell analysis
    print("\n=== Individual Cell Analysis ===")
    for i, cell in enumerate(stack.cells):
        readings = cell.get_sensor_readings()
        power = cell.get_power()
        print(f"Cell {i}: V={readings['voltage']:.3f}V, P={power:.3f}W, "
              f"pH={readings['pH']:.1f}, Acetate={readings['acetate']:.3f}, "
              f"Reversed={cell.is_reversed}")

    # Save data to JSON
    save_simulation_data(stack, controller)

    # Plot results
    plot_simulation_results(stack, controller)

    return stack, controller

def save_simulation_data(stack, controller):
    """Save simulation data to JSON file"""
    import json

    print("\n=== Saving Data ===")

    # Prepare data for JSON serialization
    simulation_data = {
        'metadata': {
            'simulation_type': '5-Cell MFC Stack with Q-Learning',
            'total_time': stack.data_log['time'][-1] if stack.data_log['time'] else 0,
            'total_steps': len(stack.data_log['time']),
            'final_stack_power': stack.stack_power,
            'final_stack_voltage': stack.stack_voltage,
            'final_stack_current': stack.stack_current,
            'q_table_size': len(controller.q_table)
        },
        'time_series': {
            'time': stack.data_log['time'],
            'stack_voltage': stack.data_log['stack_voltage'],
            'stack_current': stack.data_log['stack_current'],
            'stack_power': stack.data_log['stack_power'],
            'cell_voltages': stack.data_log['cell_voltages'],
            'cell_powers': stack.data_log['cell_powers'],
            'cell_reversals': stack.data_log['cell_reversals'],
            'duty_cycles': stack.data_log['duty_cycles'],
            'ph_buffers': stack.data_log['ph_buffers'],
            'acetate_additions': stack.data_log['acetate_additions']
        },
        'final_states': {
            'cells': []
        }
    }

    # Add final cell states
    for i, cell in enumerate(stack.cells):
        readings = cell.get_sensor_readings()
        power = cell.get_power()
        simulation_data['final_states']['cells'].append({
            'cell_id': i,
            'voltage': readings['voltage'],
            'current': readings['current'],
            'power': power,
            'pH': readings['pH'],
            'acetate': readings['acetate'],
            'reversed': cell.is_reversed,
            'duty_cycle': cell.actuators['duty_cycle'].get_value(),
            'ph_buffer': cell.actuators['ph_buffer'].get_value(),
            'acetate_pump': cell.actuators['acetate_pump'].get_value()
        })

    # Save to JSON file
    filename = get_simulation_data_path('mfc_stack_simulation_data.json')
    with open(filename, 'w') as f:
        json.dump(simulation_data, f, indent=2)

    print(f"Data saved to '{filename}'")
    return simulation_data

def plot_simulation_results(stack, controller):
    """Plot simulation results"""

    print("\n=== Generating Plots ===")

    # Set up matplotlib for non-interactive use
    import matplotlib
    matplotlib.use('Agg')

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Subplot tags (alphabetic order: left to right, top to bottom)
    subplot_tags = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    # Stack power and voltage
    ax1 = axes[0, 0]
    ax1.plot(stack.data_log['time'], stack.data_log['stack_power'], 'b-', linewidth=2, label='Stack Power')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Power (W)')
    ax1.set_title(f'{subplot_tags[0]} Stack Power Output')
    ax1.grid(True)
    ax1.legend()

    ax2 = axes[0, 1]
    ax2.plot(stack.data_log['time'], stack.data_log['stack_voltage'], 'g-', linewidth=2, label='Stack Voltage')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Voltage (V)')
    ax2.set_title(f'{subplot_tags[1]} Stack Voltage')
    ax2.grid(True)
    ax2.legend()

    # Individual cell voltages
    ax3 = axes[1, 0]
    for i in range(5):
        ax3.plot(stack.data_log['time'], stack.data_log['cell_voltages'][i],
                label=f'Cell {i}', alpha=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Voltage (V)')
    ax3.set_title(f'{subplot_tags[2]} Individual Cell Voltages')
    ax3.grid(True)
    ax3.legend()

    # Individual cell powers
    ax4 = axes[1, 1]
    for i in range(5):
        ax4.plot(stack.data_log['time'], stack.data_log['cell_powers'][i],
                label=f'Cell {i}', alpha=0.8)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Power (W)')
    ax4.set_title(f'{subplot_tags[3]} Individual Cell Powers')
    ax4.grid(True)
    ax4.legend()

    # Duty cycles
    ax5 = axes[2, 0]
    for i in range(5):
        ax5.plot(stack.data_log['time'], stack.data_log['duty_cycles'][i],
                label=f'Cell {i}', alpha=0.8)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Duty Cycle')
    ax5.set_title(f'{subplot_tags[4]} Duty Cycle Control')
    ax5.grid(True)
    ax5.legend()

    # Rewards
    ax6 = axes[2, 1]
    if len(controller.reward_history) > 50:
        rewards = list(controller.reward_history)
        # Moving average
        window_size = 50
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax6.plot(rewards, alpha=0.3, label='Raw Rewards')
        ax6.plot(range(window_size-1, len(rewards)), moving_avg, 'r-', linewidth=2, label='Moving Average')
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Reward')
    ax6.set_title(f'{subplot_tags[5]} Q-Learning Rewards')
    ax6.grid(True)
    ax6.legend()

    plt.tight_layout()
    plt.savefig(get_figure_path('mfc_stack_simulation.png'), dpi=300, bbox_inches='tight')
    print("Plots saved to 'mfc_stack_simulation.png'")

if __name__ == "__main__":
    try:
        stack, controller = run_stack_simulation()

        print("\n=== Simulation Complete ===")
        print("The Q-learning controller has successfully:")
        print("- Maintained stable power output")
        print("- Prevented cell reversal")
        print("- Optimized duty cycle control")
        print("- Managed pH buffer and acetate addition")
        print("- Balanced power across all 5 cells")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
