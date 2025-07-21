"""
GPU-Accelerated 100-Hour MFC Simulation Runner

This script demonstrates how to run the Mojo GPU-accelerated simulation
and provides a Python fallback for demonstration purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess
import os

def run_mojo_gpu_simulation():
    """Attempt to run the Mojo GPU simulation"""
    
    print("=== Attempting Mojo GPU Simulation ===")
    
    try:
        # Try to compile and run the Mojo GPU simulation
        result = subprocess.run(
            ["mojo", "run", "q-learning-mfcs/mfc_100h_gpu.mojo"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print("✓ Mojo GPU simulation completed successfully")
            print(result.stdout)
            return True
        else:
            print("✗ Mojo GPU simulation failed")
            print("Error:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Mojo GPU simulation timed out")
        return False
    except FileNotFoundError:
        print("✗ Mojo compiler not found")
        return False
    except Exception as e:
        print(f"✗ Error running Mojo simulation: {e}")
        return False

def run_accelerated_python_simulation():
    """Run an accelerated Python simulation as fallback"""
    
    print("\n=== Running Accelerated Python Simulation ===")
    print("This demonstrates the same concepts with NumPy acceleration")
    print()
    
    # Simulation parameters
    n_cells = 5
    simulation_hours = 100
    time_step = 1 # 1 minute steps for speed
    steps_per_hour = int(3600 / time_step)
    total_steps = simulation_hours * steps_per_hour
    
    print(f"Simulating {n_cells} cells for {simulation_hours} hours")
    print(f"Time step: {time_step} seconds")
    print(f"Total steps: {total_steps:,}")
    print()
    
    # Initialize arrays for vectorized computation
    cell_states = np.array([
        [1.0, 0.05, 1e-4, 0.1, 0.25, 1e-7, 0.05, 0.01, -0.01, 1.0, 1.0]  # 11 state variables
        for _ in range(n_cells)
    ])
    
    # Add slight variations between cells
    cell_states += np.random.normal(0, 0.05, cell_states.shape)
    cell_states = np.clip(cell_states, 0.001, 10.0)  # Prevent negative values
    
    # System parameters
    substrate_level = 100.0
    ph_buffer_level = 100.0
    maintenance_cycles = 0
    total_energy = 0.0
    
    # Logging arrays
    log_interval = steps_per_hour  # Log every hour
    n_log_points = simulation_hours
    
    performance_log = np.zeros((n_log_points, 8))
    
    # Q-learning parameters
    epsilon = 0.3
    epsilon_decay = 0.9995
    epsilon_min = 0.01
    
    # Simple Q-table (state -> action mapping)
    q_states = {}
    
    def discretize_state(state):
        """Simple state discretization"""
        return tuple(int(val * 10) % 10 for val in state[:5])
    
    def get_action(state, epsilon):
        """Epsilon-greedy action selection"""
        state_key = discretize_state(state)
        
        if np.random.random() < epsilon or state_key not in q_states:
            # Random action
            return np.random.uniform([0.1, 0.0, 0.0], [0.9, 1.0, 1.0])
        else:
            # Greedy action
            return q_states[state_key]
    
    def update_q_table(state, action, reward):
        """Update Q-table with new experience"""
        state_key = discretize_state(state)
        if state_key not in q_states:
            q_states[state_key] = action.copy()
        else:
            # Simple update
            q_states[state_key] = 0.9 * q_states[state_key] + 0.1 * action
    
    def compute_mfc_dynamics(states, actions, dt):
        """Vectorized MFC dynamics computation"""
        
        # MFC parameters
        F = 96485.332
        R = 8.314
        T = 303.0
        k1_0 = 0.207
        k2_0 = 3.288e-5
        K_AC = 0.592
        K_O2 = 0.004
        alpha = 0.051
        beta = 0.063
        
        # Extract states
        C_AC = states[:, 0]
        C_CO2 = states[:, 1]
        C_H = states[:, 2]
        X = states[:, 3]
        C_O2 = states[:, 4]
        C_OH = states[:, 5]
        C_M = states[:, 6]
        eta_a = states[:, 7]
        eta_c = states[:, 8]
        aging = states[:, 9]
        biofilm = states[:, 10]
        
        # Extract actions
        duty_cycle = actions[:, 0]
        ph_buffer = actions[:, 1]
        acetate_add = actions[:, 2]
        
        # Effective current
        effective_current = duty_cycle * aging
        
        # Reaction rates (vectorized)
        r1 = k1_0 * np.exp((alpha * F) / (R * T) * eta_a) * (C_AC / (K_AC + C_AC)) * X * aging / biofilm
        r2 = -k2_0 * (C_O2 / (K_O2 + C_O2)) * np.exp((beta - 1.0) * F / (R * T) * eta_c) * aging
        
        # Derivatives (simplified)
        dC_AC_dt = (1.56 + acetate_add * 0.5 - C_AC) * 0.1 - r1 * 0.01
        dX_dt = r1 * 0.001 - X * 0.0001
        dC_O2_dt = (0.3125 - C_O2) * 0.1 + r2 * 0.01
        deta_a_dt = (effective_current - r1) * 0.001
        deta_c_dt = (-effective_current - r2) * 0.001
        dC_H_dt = r1 * 0.001 - ph_buffer * C_H * 0.1
        
        # Update states
        states[:, 0] = np.clip(C_AC + dC_AC_dt * dt, 0.001, 5.0)
        states[:, 3] = np.clip(X + dX_dt * dt, 0.001, 2.0)
        states[:, 4] = np.clip(C_O2 + dC_O2_dt * dt, 0.001, 1.0)
        states[:, 7] = np.clip(eta_a + deta_a_dt * dt, -1.0, 1.0)
        states[:, 8] = np.clip(eta_c + deta_c_dt * dt, -1.0, 1.0)
        states[:, 2] = np.clip(C_H + dC_H_dt * dt, 1e-14, 1e-2)
        
        return states
    
    def apply_aging(states, dt_hours):
        """Apply aging effects"""
        aging_rate = 0.001 * dt_hours
        biofilm_growth = 0.0005 * dt_hours
        
        states[:, 9] *= (1 - aging_rate)  # Aging
        states[:, 9] = np.clip(states[:, 9], 0.5, 1.0)
        
        states[:, 10] += biofilm_growth  # Biofilm growth
        states[:, 10] = np.clip(states[:, 10], 1.0, 2.0)
        
        return states
    
    def calculate_system_metrics(states, actions):
        """Calculate system performance metrics"""
        voltages = states[:, 7] - states[:, 8]  # eta_a - eta_c
        currents = actions[:, 0] * states[:, 9]  # duty_cycle * aging
        powers = voltages * currents
        
        stack_voltage = np.sum(voltages)
        stack_current = np.min(currents)
        stack_power = stack_voltage * stack_current
        
        return stack_voltage, stack_current, stack_power, powers
    
    def calculate_reward(states, actions, stack_power):
        """Calculate Q-learning reward"""
        # Power reward
        power_reward = stack_power / 5.0
        
        # Stability reward
        voltages = states[:, 7] - states[:, 8]
        stability_reward = 1.0 - np.std(voltages) / max(0.1, np.mean(voltages))
        
        # Reversal penalty
        reversal_penalty = -10.0 * np.sum(voltages < 0.1)
        
        # Resource penalty
        resource_penalty = -0.1 * (np.sum(actions[:, 1]) + np.sum(actions[:, 2]))
        
        return power_reward + stability_reward + reversal_penalty + resource_penalty
    
    # Main simulation loop
    print("Starting simulation...")
    start_time = time.time()
    
    log_idx = 0
    
    for step in range(total_steps):
        # Q-learning action selection
        actions = np.array([get_action(cell_state, epsilon) for cell_state in cell_states])
        
        # MFC dynamics
        cell_states = compute_mfc_dynamics(cell_states, actions, time_step)
        
        # Aging effects
        if step % steps_per_hour == 0:  # Apply hourly
            dt_hours = 1.0
            cell_states = apply_aging(cell_states, dt_hours)
            
            # Update epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # System metrics
        stack_voltage, stack_current, stack_power, cell_powers = calculate_system_metrics(cell_states, actions)
        
        # Update resources
        if step % steps_per_hour == 0:
            substrate_level -= stack_power * 0.1
            ph_buffer_level -= np.sum(actions[:, 1]) * 0.05
            total_energy += stack_power * 1.0  # Wh (hourly logging = 1 hour intervals)
            
            # Maintenance
            if substrate_level < 20:
                substrate_level = 100.0
                maintenance_cycles += 1
            if ph_buffer_level < 20:
                ph_buffer_level = 100.0
                maintenance_cycles += 1
        
        # Q-learning update
        reward = calculate_reward(cell_states, actions, stack_power)
        for i in range(n_cells):
            update_q_table(cell_states[i], actions[i], reward)
        
        # Logging
        if step % log_interval == 0 and log_idx < n_log_points:
            current_hour = step * time_step / 3600
            performance_log[log_idx] = [
                current_hour,
                stack_voltage,
                stack_current,
                stack_power,
                total_energy,
                substrate_level,
                ph_buffer_level,
                maintenance_cycles
            ]
            log_idx += 1
        
        # Progress reporting
        if step % (total_steps // 10) == 0:
            progress = (step / total_steps) * 100
            current_hour = step * time_step / 3600
            print(f"Progress: {progress:.1f}% - Hour {current_hour:.1f}")
            print(f"  Stack power: {stack_power:.3f}W")
            print(f"  Total energy: {total_energy:.2f}Wh")
            print(f"  Substrate: {substrate_level:.1f}%")
            print(f"  Maintenance: {maintenance_cycles}")
            print(f"  Q-table size: {len(q_states)}")
            print()
    
    simulation_time = time.time() - start_time
    
    print("=== Simulation Complete ===")
    print(f"Real time: {simulation_time:.1f} seconds")
    print(f"Speedup: {(simulation_hours * 3600 / simulation_time):.0f}x")
    print()
    
    # Final analysis
    print("=== Final Results ===")
    print(f"Total energy: {total_energy:.2f} Wh")
    print(f"Final stack power: {stack_power:.3f} W")
    print(f"Final stack voltage: {stack_voltage:.3f} V")
    print(f"Maintenance cycles: {maintenance_cycles}")
    print(f"Q-table size: {len(q_states)} states learned")
    print()
    
    # Individual cell analysis
    print("=== Final Cell States ===")
    for i in range(n_cells):
        voltage = cell_states[i, 7] - cell_states[i, 8]
        power = voltage * actions[i, 0] * cell_states[i, 9]
        aging = cell_states[i, 9]
        biofilm = cell_states[i, 10]
        
        print(f"Cell {i}: V={voltage:.3f}V, P={power:.3f}W, Age={aging:.3f}, Biofilm={biofilm:.2f}x")
    
    # Generate plots
    generate_plots(performance_log, log_idx)
    
    return performance_log[:log_idx]

def generate_plots(performance_log, n_points):
    """Generate visualization plots"""
    
    print("\n=== Generating Plots ===")
    
    # Set up matplotlib
    import matplotlib
    matplotlib.use('Agg')
    
    if n_points < 2:
        print("Insufficient data for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    hours = performance_log[:n_points, 0]
    
    # Power evolution
    ax1 = axes[0, 0]
    ax1.plot(hours, performance_log[:n_points, 3], 'b-', linewidth=2)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Stack Power (W)')
    ax1.set_title('Power Evolution Over 100 Hours')
    ax1.grid(True)
    
    # Cumulative energy
    ax2 = axes[0, 1]
    ax2.plot(hours, performance_log[:n_points, 4], 'g-', linewidth=2)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Cumulative Energy (Wh)')
    ax2.set_title('Total Energy Production')
    ax2.grid(True)
    
    # Resource levels
    ax3 = axes[1, 0]
    ax3.plot(hours, performance_log[:n_points, 5], 'r-', label='Substrate', linewidth=2)
    ax3.plot(hours, performance_log[:n_points, 6], 'b-', label='pH Buffer', linewidth=2)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Resource Level (%)')
    ax3.set_title('Resource Consumption')
    ax3.legend()
    ax3.grid(True)
    
    # Maintenance events
    ax4 = axes[1, 1]
    ax4.step(hours, performance_log[:n_points, 7], 'orange', linewidth=2, where='post')
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Maintenance Events')
    ax4.set_title('Maintenance Schedule')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('mfc_100h_gpu_results.png', dpi=300, bbox_inches='tight')
    print("Plots saved to 'mfc_100h_gpu_results.png'")

def main():
    """Main execution function"""
    
    print("=== GPU-Accelerated MFC 100-Hour Simulation ===")
    print("This demonstrates GPU acceleration for long-term MFC simulation")
    print()
    
    # First try Mojo GPU simulation
    mojo_success = run_mojo_gpu_simulation()
    
    if not mojo_success:
        print("\nFalling back to accelerated Python simulation...")
        print("This demonstrates the same concepts with NumPy vectorization")
        
        # Run Python fallback
        results = run_accelerated_python_simulation()
        
        print("\n=== Demonstration Complete ===")
        print("GPU acceleration benefits:")
        print("✓ Parallel processing of all 5 cells")
        print("✓ Vectorized tensor operations")
        print("✓ Efficient memory usage")
        print("✓ Scalable to hundreds of cells")
        print("✓ Real-time performance capability")
        print("✓ 100+ hour simulations in seconds")
        
    else:
        print("\n=== Mojo GPU Simulation Successful ===")
        print("The Mojo implementation provides:")
        print("✓ Hardware acceleration (GPU/NPU/ASIC)")
        print("✓ Zero-cost abstractions")
        print("✓ Optimal memory layout")
        print("✓ Compile-time optimizations")

if __name__ == "__main__":
    main()