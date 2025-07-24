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
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import time
from collections import defaultdict
import pickle

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
        
        # PID parameters for outlet concentration control
        self.kp_outlet = 2.0
        self.ki_outlet = 0.1
        self.kd_outlet = 0.5
        self.outlet_error_integral = 0.0
        self.previous_outlet_error = 0.0
        
        # PID parameters for reservoir concentration control
        self.kp_reservoir = 1.0
        self.ki_reservoir = 0.05
        self.kd_reservoir = 0.2
        self.reservoir_error_integral = 0.0
        self.previous_reservoir_error = 0.0
        
        # Control limits
        self.min_addition_rate = 0.0  # mmol/h
        self.max_addition_rate = 50.0  # mmol/h
        
        # Substrate halt conditions
        self.halt_threshold = 0.5  # mmol/L decline threshold
        self.previous_outlet_conc = None
        
        # Enhanced feedback control parameters
        self.starvation_threshold_critical = 2.0  # mmol/L
        self.starvation_threshold_warning = 5.0  # mmol/L
        self.excess_threshold = 25.0  # mmol/L - halt addition if too high
        
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
    
    def __init__(self, learning_rate=0.0987, discount_factor=0.9517, epsilon=0.3702):
        """Enhanced Q-Learning controller"""
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.9978
        self.epsilon_min = 0.1020
        
        # Q-table stored as nested dictionary
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Enhanced state and action spaces
        self.setup_enhanced_state_action_spaces()
        
        # Statistics
        self.total_rewards = 0
        self.episode_count = 0
        
    def setup_enhanced_state_action_spaces(self):
        """Enhanced state space including reservoir and cell concentrations"""
        # Enhanced state variables: [power, biofilm_deviation, substrate_utilization, 
        #                           reservoir_conc, min_cell_conc, outlet_conc_error, time_phase]
        self.power_bins = np.linspace(0, 2.0, 8)
        self.biofilm_bins = np.linspace(0, 1.0, 8)
        self.substrate_bins = np.linspace(0, 50, 8)
        self.reservoir_conc_bins = np.linspace(5, 30, 6)
        self.cell_conc_bins = np.linspace(0, 25, 6)
        self.outlet_error_bins = np.linspace(0, 15, 6)
        self.time_bins = np.array([200, 500, 800, 1000])
        
        # Enhanced action space: flow rate adjustments
        self.actions = np.array([-12, -10, -5, -2, -1, 0, 1, 2, 5, 6])
        
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

class MFCCellWithMonitoring:
    """Individual MFC cell with enhanced substrate monitoring"""
    
    def __init__(self, cell_id, initial_biofilm=1.0):
        self.cell_id = cell_id
        self.biofilm_thickness = initial_biofilm
        self.substrate_concentration = 0.0  # Current cell substrate concentration
        self.inlet_concentration = 0.0      # Concentration entering this cell
        self.outlet_concentration = 0.0     # Concentration leaving this cell
        
        # Cell-specific parameters
        self.max_reaction_rate = 0.1  # max substrate consumption rate
        self.biofilm_growth_rate = 0.001
        self.optimal_biofilm_thickness = 1.3
        
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
        biofilm_decay = 0.0002 * self.biofilm_thickness * dt_hours
        
        self.biofilm_thickness = np.clip(
            self.biofilm_thickness + biofilm_growth - biofilm_decay,
            0.5, 3.0
        )
        
        # Store monitoring data
        self.concentration_history.append(self.substrate_concentration)
        self.consumption_rate_history.append(substrate_consumed / dt_hours if dt_hours > 0 else 0)
        
        return self.outlet_concentration

def simulate_mfc_with_recirculation():
    """Main simulation function with recirculation and advanced substrate control"""
    
    # Initialize components
    reservoir = AnolytereservoirSystem(initial_substrate_conc=20.0, volume_liters=1.0)
    controller = SubstrateConcentrationController(target_outlet_conc=12.0, target_reservoir_conc=20.0)
    q_controller = AdvancedQLearningFlowController()
    
    # Initialize MFC cells
    n_cells = 5
    cells = [MFCCellWithMonitoring(i+1, initial_biofilm=1.0) for i in range(n_cells)]
    
    # Simulation parameters
    duration_hours = 100
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
        
        # Calculate substrate addition using feedback controller with sensor data
        addition_rate, halt_flag = controller.calculate_substrate_addition(
            outlet_concentration, 
            reservoir.substrate_concentration,
            cell_concentrations,
            reservoir_sensors,
            dt_hours
        )
        
        # Add substrate to reservoir
        reservoir.substrate_halt = halt_flag
        reservoir.add_substrate(addition_rate, dt_hours)
        
        # Simulate recirculation
        reservoir.circulate_anolyte(flow_rate_ml_h, outlet_concentration, dt_hours)
        
        # Calculate system performance metrics
        total_power = sum(0.8 * cell.biofilm_thickness * cell.substrate_concentration * 0.001 
                         for cell in cells)  # Simplified power calculation
        
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
        results['substrate_halt'].append(halt_flag)
        
        # Progress reporting
        if step % (n_steps // 10) == 0:
            print(f"Time: {time_hours:.1f}h, "
                  f"Reservoir: {reservoir.substrate_concentration:.2f} mmol/L, "
                  f"Outlet: {outlet_concentration:.2f} mmol/L, "
                  f"Min Cell: {min_cell_conc:.2f} mmol/L, "
                  f"Addition: {addition_rate:.1f} mmol/h, "
                  f"Halt: {halt_flag}")
    
    return results, cells, reservoir, controller

if __name__ == "__main__":
    # Run simulation
    results, cells, reservoir, controller = simulate_mfc_with_recirculation()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    csv_file = f"simulation_data/mfc_recirculation_control_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"\n=== SIMULATION COMPLETE ===")
    print(f"Results saved to: {csv_file}")
    print(f"Final reservoir concentration: {reservoir.substrate_concentration:.2f} mmol/L")
    print(f"Final outlet concentration: {results['outlet_concentration'][-1]:.2f} mmol/L")
    print(f"Total substrate added: {reservoir.total_substrate_added:.2f} mmol")
    print(f"Average biofilm thickness: {np.mean([cell.biofilm_thickness for cell in cells]):.2f}")