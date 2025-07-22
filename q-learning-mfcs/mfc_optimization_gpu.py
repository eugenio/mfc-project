#!/usr/bin/env python3
"""
MFC Stack Optimization Simulation with GPU Acceleration
Multi-objective optimization: power maximization, biofilm control, substrate utilization
Sequential anolyte flow through 5-cell stack
Duration: 1000 hours, timestep: 10 seconds
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional for styling
from scipy.optimize import minimize_scalar
import time

# Try to import GPU acceleration libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration enabled (CuPy)")
except ImportError:
    try:
        import numba
        from numba import cuda
        GPU_AVAILABLE = True
        print("GPU acceleration enabled (Numba)")
    except ImportError:
        GPU_AVAILABLE = False
        print("GPU acceleration not available, using CPU")

class MFCOptimizationSimulation:
    def __init__(self, use_gpu=True):
        """Initialize MFC optimization simulation"""
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
        self.r_max = 5.787e-5  # Maximum reaction rate (mol/(m²·s))
        self.K_AC = 0.592      # Acetate half-saturation constant (mol/m³)
        self.K_dec = 8.33e-4   # Decay constant (s⁻¹)
        self.Y_ac = 0.05       # Biomass yield (kg/mol)
        
        # Optimization parameters
        self.optimal_biofilm_thickness = 1.3  # Optimal biofilm for max electron transfer
        self.flow_rate_bounds = (5.0e-6, 50.0e-6)  # Flow rate bounds (m³/s) = 18-180 mL/h
        
        # Multi-objective weights
        self.w_power = 0.5
        self.w_biofilm = 0.3  
        self.w_substrate = 0.2
        
        # Initialize arrays
        self.initialize_arrays()
        
        # Create output directories
        os.makedirs('figures', exist_ok=True)
        os.makedirs('simulation_data', exist_ok=True)
        
    def initialize_arrays(self):
        """Initialize simulation state arrays"""
        array_func = cp.zeros if self.use_gpu else np.zeros
        
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
        
        # Initialize starting conditions
        self.biofilm_thickness[0, :] = 1.0  # Starting biofilm thickness
        self.acetate_concentrations[0, :] = 1.56  # Starting acetate concentration
        self.flow_rates[0] = 50.0e-6  # Starting flow rate (50 mL/h)
        
    def biofilm_factor(self, thickness):
        """Calculate biofilm factor affecting mass transfer"""
        if self.use_gpu:
            delta_opt = cp.abs(thickness - self.optimal_biofilm_thickness)
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
            electron_enhancement = cp.where(
                cp.abs(biofilm - self.optimal_biofilm_thickness) < 0.1,
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
        
        # Acetate consumption in this cell
        consumption_rate = self.A_m * reaction_rate
        acetate_consumed = consumption_rate * residence_time
        
        if self.use_gpu:
            outlet_concentration = cp.maximum(0.01, inlet_concentration - acetate_consumed)
        else:
            outlet_concentration = np.maximum(0.01, inlet_concentration - acetate_consumed)
        
        # Current calculation (8 electrons per acetate molecule)
        current_density = consumption_rate * 8.0 * self.F / self.A_m  # A/m²
        
        # Voltage calculation with improved model
        voltage_base = 0.8  # Base voltage
        if self.use_gpu:
            concentration_factor = cp.log(1.0 + inlet_concentration / self.K_AC)
            biofilm_voltage_loss = 0.05 * cp.abs(biofilm - self.optimal_biofilm_thickness)
            cell_voltage = cp.maximum(0.1, voltage_base + 0.1 * concentration_factor - biofilm_voltage_loss)
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
                control_factor = cp.where(
                    current_thickness > self.optimal_biofilm_thickness,
                    0.5,  # Reduce growth above optimal
                    cp.where(current_thickness < self.optimal_biofilm_thickness * 0.8,
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
                new_thickness = cp.clip(current_thickness + net_growth, 0.5, 3.0)
            else:
                new_thickness = np.clip(current_thickness + net_growth, 0.5, 3.0)
            
            self.biofilm_thickness[step, cell_idx] = new_thickness
    
    def calculate_objective_function(self, step):
        """Multi-objective optimization function"""
        # 1. Power maximization (normalized to 0-1)
        total_power = self.stack_powers[step]
        power_objective = min(1.0, total_power / 5.0)  # Normalize to expected max ~5W
        
        # 2. Biofilm control (minimize deviation from optimal)
        if self.use_gpu:
            biofilm_deviation = cp.mean(cp.abs(self.biofilm_thickness[step, :] - self.optimal_biofilm_thickness))
        else:
            biofilm_deviation = np.mean(np.abs(self.biofilm_thickness[step, :] - self.optimal_biofilm_thickness))
        biofilm_objective = max(0.0, 1.0 - biofilm_deviation)
        
        # 3. Substrate utilization (minimize outlet concentration)
        final_concentration = self.acetate_concentrations[step, -1]  # Last cell outlet
        initial_concentration = 1.56
        utilization = (initial_concentration - final_concentration) / initial_concentration
        substrate_objective = min(1.0, utilization * 5.0)  # Amplify small utilizations
        
        # Combined objective
        return (self.w_power * power_objective + 
                self.w_biofilm * biofilm_objective + 
                self.w_substrate * substrate_objective)
    
    def calculate_flow_rate(self, time_seconds):
        """Calculate flow rate as a function of time (dependent variable)"""
        # Convert time to hours for easier calculations
        time_hours = time_seconds / 3600.0
        
        # Time-dependent flow rate function with multiple phases
        # Phase 1 (0-200h): Gradual optimization from initial high flow
        if time_hours <= 200:
            # Start at 50 mL/h, decrease to optimal 20 mL/h
            base_flow = 50.0e-6 - (30.0e-6 * time_hours / 200.0)
        
        # Phase 2 (200-500h): Fine-tuning around optimal
        elif time_hours <= 500:
            # Oscillate around optimal with decreasing amplitude
            oscillation_period = 50.0  # hours
            amplitude = 5.0e-6 * (1.0 - (time_hours - 200) / 300.0)  # Decreasing amplitude
            phase = 2 * np.pi * (time_hours - 200) / oscillation_period
            base_flow = 20.0e-6 + amplitude * np.sin(phase)
        
        # Phase 3 (500-800h): Adaptive control based on biofilm state
        elif time_hours <= 800:
            # Get current average biofilm thickness
            if hasattr(self, 'biofilm_thickness'):
                current_step = min(int(time_seconds / self.dt), self.num_steps - 1)
                if current_step > 0:
                    avg_biofilm = np.mean(self.biofilm_thickness[current_step-1, :])
                    biofilm_deviation = abs(avg_biofilm - self.optimal_biofilm_thickness)
                    
                    # Adjust flow rate based on biofilm deviation
                    if biofilm_deviation > 0.2:  # Too much deviation
                        base_flow = 15.0e-6 + 10.0e-6 * biofilm_deviation  # Higher flow for cleaning
                    else:
                        base_flow = 20.0e-6 - 5.0e-6 * biofilm_deviation  # Lower flow for efficiency
                else:
                    base_flow = 20.0e-6
            else:
                base_flow = 20.0e-6
        
        # Phase 4 (800-1000h): Long-term stable operation
        else:
            # Stable operation with minor adjustments
            stability_factor = 1.0 + 0.05 * np.sin(2 * np.pi * time_hours / 100.0)  # 2% oscillation
            base_flow = 18.0e-6 * stability_factor
        
        # Apply bounds and return
        return np.clip(base_flow, self.flow_rate_bounds[0], self.flow_rate_bounds[1])
    
    def simulate_step(self, step):
        """Simulate single time step"""
        if step == 0:
            return
            
        # Calculate flow rate as function of time (dependent variable)
        current_time = step * self.dt  # Current time in seconds
        self.flow_rates[step] = self.calculate_flow_rate(current_time)
        
        # Update biofilm thickness
        self.update_biofilm(step, self.dt)
        
        # Sequential anolyte flow through cells
        inlet_concentration = 1.56  # mol/m³ constant inlet
        current_concentration = inlet_concentration
        stack_voltage = 0.0
        stack_power = 0.0
        
        for cell_idx in range(self.num_cells):
            biofilm = self.biofilm_thickness[step, cell_idx]
            flow_rate = self.flow_rates[step]
            
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
        
        # Update stack arrays
        self.stack_voltages[step] = stack_voltage
        self.stack_powers[step] = stack_power
        
        # Calculate substrate utilization
        final_conc = self.acetate_concentrations[step, -1]
        self.substrate_utilizations[step] = (1.56 - final_conc) / 1.56 * 100.0
        
        # Calculate objective function
        self.objective_values[step] = self.calculate_objective_function(step)
    
    def run_simulation(self):
        """Run the complete simulation"""
        print("Starting MFC optimization simulation...")
        print(f"Duration: 1000 hours, Timesteps: {self.num_steps}")
        print(f"GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        print(f"Flow rate control: Time-dependent (independent variable: time)")
        
        start_time = time.time()
        
        for step in range(self.num_steps):
            self.simulate_step(step)
            
            # Progress reporting
            if step % 36000 == 0:  # Every 100 hours
                hours = step * self.dt / 3600
                print(f"Progress: {hours:.0f}/1000 hours (t={step*self.dt:.0f}s), "
                      f"Power: {float(self.stack_powers[step]):.3f} W, "
                      f"Flow: {float(self.flow_rates[step]) * 3.6e9:.1f} mL/h, "
                      f"Utilization: {float(self.substrate_utilizations[step]):.2f}%")
        
        simulation_time = time.time() - start_time
        print(f"Simulation completed in {simulation_time:.2f} seconds")
        
        # Convert GPU arrays to CPU if needed
        if self.use_gpu:
            self.cell_voltages = cp.asnumpy(self.cell_voltages)
            self.biofilm_thickness = cp.asnumpy(self.biofilm_thickness)
            self.acetate_concentrations = cp.asnumpy(self.acetate_concentrations)
            self.current_densities = cp.asnumpy(self.current_densities)
            self.power_outputs = cp.asnumpy(self.power_outputs)
            self.substrate_consumptions = cp.asnumpy(self.substrate_consumptions)
            self.stack_voltages = cp.asnumpy(self.stack_voltages)
            self.stack_powers = cp.asnumpy(self.stack_powers)
            self.flow_rates = cp.asnumpy(self.flow_rates)
            self.objective_values = cp.asnumpy(self.objective_values)
            self.substrate_utilizations = cp.asnumpy(self.substrate_utilizations)
    
    def save_data(self):
        """Save simulation data to JSON and CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create time array in hours
        time_hours = np.arange(self.num_steps) * self.dt / 3600
        
        # Prepare data for CSV export
        time_seconds = np.arange(self.num_steps) * self.dt
        csv_data = {
            'time_seconds': time_seconds,
            'time_hours': time_hours,
            'stack_voltage': self.stack_voltages,
            'stack_power': self.stack_powers,
            'flow_rate_ml_h': self.flow_rates * 3.6e9,
            'objective_value': self.objective_values,
            'substrate_utilization': self.substrate_utilizations
        }
        
        # Add cell-specific data
        for i in range(self.num_cells):
            csv_data[f'cell_{i+1}_voltage'] = self.cell_voltages[:, i]
            csv_data[f'cell_{i+1}_power'] = self.power_outputs[:, i]
            csv_data[f'cell_{i+1}_biofilm'] = self.biofilm_thickness[:, i]
            csv_data[f'cell_{i+1}_acetate_out'] = self.acetate_concentrations[:, i]
        
        # Save CSV
        df = pd.DataFrame(csv_data)
        csv_filename = f'simulation_data/mfc_optimization_{timestamp}.csv'
        df.to_csv(csv_filename, index=False)
        print(f"CSV data saved to {csv_filename}")
        
        # Prepare data for JSON export
        json_data = {
            'simulation_info': {
                'timestamp': timestamp,
                'duration_hours': 1000,
                'timestep_seconds': 10,
                'num_cells': 5,
                'gpu_acceleration': self.use_gpu,
                'objectives': [
                    'Maximize instantaneous power output',
                    'Minimize biofilm growth (optimal thickness control)',
                    'Minimize substrate outlet concentration'
                ],
                'control_strategy': 'Time-dependent flow rate (independent variable: time in seconds)'
            },
            'parameters': {
                'optimal_biofilm_thickness': self.optimal_biofilm_thickness,
                'flow_rate_bounds_ml_h': [x * 3.6e9 for x in self.flow_rate_bounds],
                'objective_weights': {
                    'power': self.w_power,
                    'biofilm': self.w_biofilm,
                    'substrate': self.w_substrate
                }
            },
            'results': {
                'total_energy_wh': float(np.trapz(self.stack_powers, dx=self.dt/3600)),
                'average_power_w': float(np.mean(self.stack_powers)),
                'final_stack_voltage_v': float(self.stack_voltages[-1]),
                'final_stack_power_w': float(self.stack_powers[-1]),
                'optimized_flow_rate_ml_h': float(self.flow_rates[-1] * 3.6e9),
                'final_substrate_utilization_percent': float(self.substrate_utilizations[-1]),
                'final_objective_value': float(self.objective_values[-1]),
                'average_biofilm_thickness': float(np.mean(self.biofilm_thickness[-1, :])),
                'biofilm_deviation_from_optimal': float(np.mean(np.abs(self.biofilm_thickness[-1, :] - self.optimal_biofilm_thickness)))
            }
        }
        
        # Save JSON
        json_filename = f'simulation_data/mfc_optimization_{timestamp}.json'
        with open(json_filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON data saved to {json_filename}")
        
        return timestamp
    
    def generate_plots(self, timestamp):
        """Generate comprehensive visualization dashboard"""
        # plt.style.use('seaborn-v0_8')  # Optional styling
        
        # Create time array in hours
        time_hours = np.arange(self.num_steps) * self.dt / 3600
        
        # Create main dashboard figure
        fig = plt.figure(figsize=(20, 16))
        
        # Plot 1: Stack Performance Over Time
        ax1 = plt.subplot(3, 3, 1)
        plt.plot(time_hours, self.stack_powers, 'b-', linewidth=1.5, label='Stack Power')
        plt.xlabel('Time (hours)')
        plt.ylabel('Power (W)')
        plt.title('Stack Power Output')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Time-Dependent Flow Rate
        ax2 = plt.subplot(3, 3, 2)
        plt.plot(time_hours, self.flow_rates * 3.6e9, 'g-', linewidth=1.5)
        plt.xlabel('Time (hours)')
        plt.ylabel('Flow Rate (mL/h)')
        plt.title('Time-Dependent Flow Rate Control')
        plt.grid(True, alpha=0.3)
        
        # Add phase annotations
        plt.axvline(x=200, color='red', linestyle='--', alpha=0.5, label='Phase 1→2')
        plt.axvline(x=500, color='red', linestyle='--', alpha=0.5, label='Phase 2→3')
        plt.axvline(x=800, color='red', linestyle='--', alpha=0.5, label='Phase 3→4')
        plt.legend(fontsize=8)
        
        # Plot 3: Substrate Utilization
        ax3 = plt.subplot(3, 3, 3)
        plt.plot(time_hours, self.substrate_utilizations, 'r-', linewidth=1.5)
        plt.xlabel('Time (hours)')
        plt.ylabel('Utilization (%)')
        plt.title('Substrate Utilization Efficiency')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Biofilm Thickness Evolution
        ax4 = plt.subplot(3, 3, 4)
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
        
        # Plot 5: Cell Voltage Distribution
        ax5 = plt.subplot(3, 3, 5)
        for i in range(self.num_cells):
            plt.plot(time_hours, self.cell_voltages[:, i], 
                    linewidth=1.5, label=f'Cell {i+1}')
        plt.xlabel('Time (hours)')
        plt.ylabel('Voltage (V)')
        plt.title('Individual Cell Voltages')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Acetate Concentration Profile
        ax6 = plt.subplot(3, 3, 6)
        for i in range(self.num_cells):
            plt.plot(time_hours, self.acetate_concentrations[:, i], 
                    linewidth=1.5, label=f'Cell {i+1} outlet')
        plt.xlabel('Time (hours)')
        plt.ylabel('Acetate Concentration (mol/m³)')
        plt.title('Sequential Acetate Consumption')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 7: Objective Function Evolution
        ax7 = plt.subplot(3, 3, 7)
        plt.plot(time_hours, self.objective_values, 'purple', linewidth=1.5)
        plt.xlabel('Time (hours)')
        plt.ylabel('Objective Value')
        plt.title('Multi-Objective Optimization Progress')
        plt.grid(True, alpha=0.3)
        
        # Plot 8: Final State Comparison
        ax8 = plt.subplot(3, 3, 8)
        cell_numbers = np.arange(1, self.num_cells + 1)
        final_powers = self.power_outputs[-1, :]
        final_biofilms = self.biofilm_thickness[-1, :]
        
        ax8_twin = ax8.twinx()
        bars1 = ax8.bar(cell_numbers - 0.2, final_powers, 0.4, 
                       label='Power (W)', color='blue', alpha=0.7)
        bars2 = ax8_twin.bar(cell_numbers + 0.2, final_biofilms, 0.4,
                           label='Biofilm Thickness', color='red', alpha=0.7)
        
        ax8.set_xlabel('Cell Number')
        ax8.set_ylabel('Power (W)', color='blue')
        ax8_twin.set_ylabel('Biofilm Thickness', color='red')
        ax8.set_title('Final State Distribution')
        ax8.legend(loc='upper left')
        ax8_twin.legend(loc='upper right')
        
        # Plot 9: Performance Metrics Summary
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Calculate summary metrics
        total_energy = np.trapz(self.stack_powers, dx=self.dt/3600)
        avg_power = np.mean(self.stack_powers)
        final_utilization = self.substrate_utilizations[-1]
        avg_biofilm_dev = np.mean(np.abs(self.biofilm_thickness[-1, :] - self.optimal_biofilm_thickness))
        final_flow_rate = self.flow_rates[-1] * 3.6e9
        
        summary_text = f"""
        TIME-DEPENDENT OPTIMIZATION
        
        Total Energy: {total_energy:.1f} Wh
        Average Power: {avg_power:.3f} W
        Final Power: {self.stack_powers[-1]:.3f} W
        
        Final Flow Rate: {final_flow_rate:.1f} mL/h
        Substrate Utilization: {final_utilization:.2f}%
        
        Avg Biofilm Deviation: {avg_biofilm_dev:.3f}
        Final Objective Value: {self.objective_values[-1]:.3f}
        
        CONTROL STRATEGY:
        ✓ Time as Independent Variable
        ✓ Flow Rate Time-Dependent
        ✓ Multi-Phase Optimization
        """
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightblue", alpha=0.5))
        
        plt.tight_layout()
        
        # Save main dashboard
        dashboard_filename = f'figures/mfc_optimization_dashboard_{timestamp}.png'
        plt.savefig(dashboard_filename, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to {dashboard_filename}")
        
        # Create additional detailed plots
        self.generate_detailed_plots(timestamp, time_hours)
        
        plt.close('all')
    
    def generate_detailed_plots(self, timestamp, time_hours):
        """Generate additional detailed analysis plots"""
        # Correlation analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Time vs Flow Rate and Power
        time_seconds = np.arange(self.num_steps) * self.dt
        ax_twin = axes[0, 0].twinx()
        line1 = axes[0, 0].plot(time_hours, self.flow_rates * 3.6e9, 'g-', 
                               linewidth=1, label='Flow Rate')
        line2 = ax_twin.plot(time_hours, self.stack_powers, 'b-', 
                            linewidth=1, label='Power')
        axes[0, 0].set_xlabel('Time (hours)')
        axes[0, 0].set_ylabel('Flow Rate (mL/h)', color='g')
        ax_twin.set_ylabel('Stack Power (W)', color='b')
        axes[0, 0].set_title('Time-Dependent Flow Rate and Power')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[0, 0].legend(lines, labels, loc='upper right')
        
        # Substrate utilization vs Power
        axes[0, 1].scatter(self.substrate_utilizations, self.stack_powers, 
                          alpha=0.6, s=1, color='red')
        axes[0, 1].set_xlabel('Substrate Utilization (%)')
        axes[0, 1].set_ylabel('Stack Power (W)')
        axes[0, 1].set_title('Substrate Utilization vs Power')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Biofilm deviation vs Objective
        biofilm_deviations = np.mean(np.abs(self.biofilm_thickness - self.optimal_biofilm_thickness), axis=1)
        axes[1, 0].scatter(biofilm_deviations, self.objective_values, 
                          alpha=0.6, s=1, color='green')
        axes[1, 0].set_xlabel('Avg Biofilm Deviation from Optimal')
        axes[1, 0].set_ylabel('Objective Function Value')
        axes[1, 0].set_title('Biofilm Control vs Performance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Time series of optimization progress
        axes[1, 1].plot(time_hours, self.objective_values, 'purple', linewidth=1.5)
        axes[1, 1].set_xlabel('Time (hours)')
        axes[1, 1].set_ylabel('Objective Function Value')
        axes[1, 1].set_title('Optimization Progress Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        correlation_filename = f'figures/mfc_correlation_analysis_{timestamp}.png'
        plt.savefig(correlation_filename, dpi=300, bbox_inches='tight')
        print(f"Correlation analysis saved to {correlation_filename}")
        
        plt.close()


def main():
    """Main execution function"""
    print("=== MFC Multi-Objective Optimization Simulation ===")
    print("Objectives:")
    print("1. Maximize instantaneous power output")
    print("2. Minimize biofilm growth (maintain optimal thickness)")
    print("3. Minimize substrate outlet concentration")
    print("Independent variable: Time (seconds)")
    print("Dependent variable: Anolyte flow rate (time-dependent function)")
    print("Configuration: Sequential flow through 5-cell stack")
    print("=" * 50)
    
    # Initialize and run simulation
    sim = MFCOptimizationSimulation(use_gpu=True)
    sim.run_simulation()
    
    # Save data and generate plots
    timestamp = sim.save_data()
    sim.generate_plots(timestamp)
    
    print("\n=== SIMULATION COMPLETE ===")
    print(f"Results saved with timestamp: {timestamp}")
    print("Check 'simulation_data/' for CSV and JSON files")
    print("Check 'figures/' for visualization dashboard")
    
    # Final summary
    total_energy = np.trapz(sim.stack_powers, dx=sim.dt/3600)
    print(f"\nFinal Performance Summary:")
    print(f"Total Energy: {total_energy:.1f} Wh")
    print(f"Average Power: {np.mean(sim.stack_powers):.3f} W")
    print(f"Final Power: {sim.stack_powers[-1]:.3f} W")
    print(f"Optimized Flow Rate: {sim.flow_rates[-1] * 3.6e9:.1f} mL/h")
    print(f"Final Substrate Utilization: {sim.substrate_utilizations[-1]:.2f}%")
    print(f"Final Objective Value: {sim.objective_values[-1]:.3f}")


if __name__ == "__main__":
    main()