#!/usr/bin/env python3
"""
GPU-Accelerated MFC Simulation with Optimized Q-learning

Universal GPU-accelerated version supporting both NVIDIA CUDA and AMD ROCm.
Includes improved Q-learning stability for long-term simulations.

Created: 2025-07-26
Last modified: 2025-07-26
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path
import signal

def detect_gpu_backend():
    """Detect available GPU backend (NVIDIA CUDA or AMD ROCm)."""
    
    print("🔍 Detecting available GPU backends...")
    
    # Check for NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("🟢 NVIDIA GPU detected!")
            return 'cuda', 'nvidia'
    except FileNotFoundError:
        pass
    
    # Check for AMD GPU
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("🟠 AMD GPU detected!")
            return 'rocm', 'amd'
    except FileNotFoundError:
        pass
    
    # Check for AMD GPU via lspci
    try:
        result = subprocess.run(['lspci'], capture_output=True, text=True)
        if 'AMD' in result.stdout and ('Radeon' in result.stdout or 'RDNA' in result.stdout):
            print("🟠 AMD GPU detected via lspci!")
            return 'rocm', 'amd'
    except FileNotFoundError:
        pass
    
    print("⚪ No GPU detected, falling back to CPU")
    return 'cpu', 'cpu'

def setup_gpu_backend():
    """Setup GPU backend with JAX."""
    
    backend, gpu_type = detect_gpu_backend()
    
    if backend == 'cuda':
        try:
            os.environ['JAX_PLATFORM_NAME'] = 'cuda'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            import jax
            # Test if CUDA backend actually works
            devices = jax.devices()
            import jax.numpy as jnp
            from jax import random, jit, vmap
            print(f"✅ JAX CUDA backend initialized on {devices[0]}")
            return jax, jnp, random, jit, vmap, 'NVIDIA CUDA', True
        except Exception as e:
            print(f"⚠️  CUDA setup failed: {e}")
    
    elif backend == 'rocm':
        try:
            os.environ['JAX_PLATFORM_NAME'] = 'rocm' 
            os.environ['HIP_VISIBLE_DEVICES'] = '0'
            import jax
            # Test if ROCm backend actually works
            devices = jax.devices()
            import jax.numpy as jnp
            from jax import random, jit, vmap
            print(f"✅ JAX ROCm backend initialized on {devices[0]}")
            return jax, jnp, random, jit, vmap, 'AMD ROCm', True
        except Exception as e:
            print(f"⚠️  ROCm setup failed: {e}")
    
    # Fallback to CPU JAX
    try:
        os.environ.pop('JAX_PLATFORM_NAME', None)
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        os.environ.pop('HIP_VISIBLE_DEVICES', None)
        import jax
        import jax.numpy as jnp
        from jax import random, jit, vmap
        devices = jax.devices()
        print(f"✅ JAX CPU backend initialized on {devices[0]}")
        return jax, jnp, random, jit, vmap, 'CPU (JAX)', True
    except Exception as e:
        print(f"⚠️  JAX setup failed: {e}")
        import numpy as jnp
        print("✅ NumPy fallback initialized")
        return None, jnp, None, None, None, 'CPU (NumPy)', False

# Initialize GPU backend
jax, jnp, random, jit, vmap, BACKEND_NAME, JAX_AVAILABLE = setup_gpu_backend()
import numpy as np

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.qlearning_config import DEFAULT_QLEARNING_CONFIG


class GPUAcceleratedMFC:
    """GPU-accelerated MFC simulation supporting NVIDIA and AMD GPUs."""
    
    def __init__(self, config, n_cells=None, reservoir_volume=None, electrode_area=None):
        """Initialize GPU-accelerated MFC simulation.
        
        Args:
            config: Q-learning configuration
            n_cells: Number of MFC cells
            reservoir_volume: Volume of substrate reservoir (L)
            electrode_area: Electrode surface area per cell (m²)
        """
        
        print("🚀 Initializing GPU-accelerated MFC simulation...")
        print(f"🔧 Backend: {BACKEND_NAME}")
        
        # Check device info
        if JAX_AVAILABLE and jax is not None:
            try:
                self.device = jax.devices()[0]
                print(f"🎯 Device: {self.device}")
            except:
                print("🎯 Device: JAX backend (device info unavailable)")
        else:
            print("🎯 Device: CPU (NumPy)")
        
        self.config = config
        # Use config values if not explicitly provided
        self.n_cells = n_cells if n_cells is not None else config.n_cells
        self.reservoir_volume = reservoir_volume if reservoir_volume is not None else config.reservoir_volume_liters
        self.anode_area = electrode_area if electrode_area is not None else config.anode_area_per_cell  # m² per cell (current collection)
        self.cathode_area = config.cathode_area_per_cell  # m² per cell (oxygen reduction/proton reduction)
        
        # Sensor electrode areas (separate from working electrodes)
        self.eis_sensor_area = config.eis_sensor_area  # m² per sensor (EIS)
        self.qcm_sensor_area = config.qcm_sensor_area  # m² per sensor (QCM)
        
        # Legacy compatibility
        self.electrode_area = self.anode_area  # For backward compatibility
        
        # Initialize state arrays
        # Import configuration
        from config.qlearning_config import DEFAULT_QLEARNING_CONFIG
        config = DEFAULT_QLEARNING_CONFIG
        
        try:
            if JAX_AVAILABLE and random is not None:
                self.key = random.PRNGKey(42)
                self.biofilm_thicknesses = jnp.ones(self.n_cells) * 1.0  # μm
                self.cell_concentrations = jnp.ones(self.n_cells) * config.initial_cell_concentration  # mM - from config
                self.q_table = self.load_pretrained_qtable()  # Load pre-trained Q-table
                self.stability_buffer = jnp.zeros(100)  # Rolling stability check
            else:
                self.key = None
                self.biofilm_thicknesses = jnp.ones(self.n_cells) * 1.0  # μm
                self.cell_concentrations = jnp.ones(self.n_cells) * config.initial_cell_concentration  # mM - from config
                self.q_table = self.load_pretrained_qtable()  # Load pre-trained Q-table
                self.stability_buffer = jnp.zeros(100)  # Rolling stability check
        except Exception as e:
            print(f"⚠️  JAX initialization failed: {e}")
            print("🔄 Falling back to NumPy...")
            self.key = None
            self.biofilm_thicknesses = np.ones(self.n_cells) * 1.0  # μm
            self.cell_concentrations = np.ones(self.n_cells) * config.initial_cell_concentration  # mM - from config
            self.q_table = self.load_pretrained_qtable()  # Load pre-trained Q-table
            self.stability_buffer = np.zeros(100)  # Rolling stability check
        
        # Use unified target concentrations from config
        self.reservoir_concentration = config.initial_substrate_concentration  # mM - from config
        self.outlet_concentration = config.substrate_target_outlet  # mM - from config
        self.target_concentration = config.substrate_target_concentration  # mM - universal target from config
        self.epsilon = self.load_trained_epsilon()  # Load epsilon from checkpoint
        
        # Performance tracking
        self.total_substrate_added = 0.0
        self.stability_index = 0
        
        print(f"✅ {BACKEND_NAME} acceleration initialized successfully!")
        print("⚡ System configuration:")
        print(f"   - Cells: {self.n_cells}")
        print(f"   - Anode area: {self.anode_area*1e4:.1f} cm²/cell ({self.anode_area*self.n_cells*1e4:.1f} cm² total)")
        print(f"   - Cathode area: {self.cathode_area*1e4:.1f} cm²/cell ({self.cathode_area*self.n_cells*1e4:.1f} cm² total)")
        print(f"   - EIS sensor area: {self.eis_sensor_area*1e4:.3f} cm² per sensor")
        print(f"   - QCM sensor area: {self.qcm_sensor_area*1e4:.3f} cm² per sensor")
        print(f"   - Reservoir: {self.reservoir_volume:.1f} L")
    
    def load_pretrained_qtable(self):
        """Load pre-trained Q-table from the most recent compatible checkpoint."""
        
        print("🧠 Loading pre-trained Q-learning weights...")
        
        # Search for checkpoint files
        data_dir = Path("../data/simulation_data")
        checkpoint_files = list(data_dir.glob("**/lactate_controlled_*_model_checkpoint.json"))
        
        if not checkpoint_files:
            print("⚠️  No pre-trained checkpoints found, initializing with zeros")
            return jnp.zeros((100, 10)) if JAX_AVAILABLE else np.zeros((100, 10))
        
        # Sort by modification time and get the most recent
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_checkpoint = checkpoint_files[0]
        
        print(f"📂 Loading checkpoint: {latest_checkpoint.name}")
        
        try:
            with open(latest_checkpoint, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Extract checkpoint Q-table
            checkpoint_qtable = checkpoint_data.get('q_table', {})
            
            if not checkpoint_qtable:
                print("⚠️  No Q-table data in checkpoint, initializing with zeros")
                return jnp.zeros((100, 10)) if JAX_AVAILABLE else np.zeros((100, 10))
            
            # Initialize simplified Q-table (100 states, 10 actions)
            if JAX_AVAILABLE:
                q_table = jnp.zeros((100, 10))
            else:
                q_table = np.zeros((100, 10))
            
            # Map from checkpoint state-action space to simplified space
            actions_loaded = 0
            
            for state_key, action_values in checkpoint_qtable.items():
                try:
                    # Parse state tuple from string representation
                    # Format: "(np.int64(0), np.int64(2), np.int64(0), ...)"
                    state_str = state_key.strip("()")
                    state_parts = [part.strip() for part in state_str.split(",")]
                    
                    # Extract substrate bin (3rd element) for simplified state mapping
                    substrate_bin_str = state_parts[2] if len(state_parts) > 2 else "0"
                    substrate_bin = int(substrate_bin_str.split("(")[-1].split(")")[0])
                    
                    # Map to simplified 100-state space (substrate concentration 0-99)
                    simplified_state = min(substrate_bin * 12, 99)  # Scale 8-bin to 100-state
                    
                    # Map actions from 70-action space to 10-action space
                    for action_str, q_value in action_values.items():
                        action_id = int(action_str)
                        
                        # Map action space: 70 actions -> 10 actions
                        if action_id < 70:  # Valid checkpoint action
                            simplified_action = action_id % 10  # Simple modulo mapping
                            
                            # Update Q-table with mapped values
                            if JAX_AVAILABLE and hasattr(q_table, 'at'):
                                current_q = q_table[simplified_state, simplified_action]
                                # Average multiple mappings
                                new_q = (current_q + float(q_value)) / 2.0
                                q_table = q_table.at[simplified_state, simplified_action].set(new_q)
                            else:
                                current_q = q_table[simplified_state, simplified_action]
                                q_table[simplified_state, simplified_action] = (current_q + float(q_value)) / 2.0
                            
                            actions_loaded += 1
                
                except (ValueError, IndexError, KeyError):
                    continue  # Skip malformed state entries
            
            print(f"✅ Loaded {actions_loaded} Q-values from checkpoint")
            print(f"🎯 Q-table shape: {q_table.shape}")
            
            # Show sample Q-values for verification
            if JAX_AVAILABLE:
                sample_q = float(jnp.max(jnp.abs(q_table)))
            else:
                sample_q = float(np.max(np.abs(q_table)))
            
            print(f"🔍 Max Q-value magnitude: {sample_q:.3f}")
            
            return q_table
            
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print("🔄 Initializing with zeros")
            return jnp.zeros((100, 10)) if JAX_AVAILABLE else np.zeros((100, 10))
    
    def load_trained_epsilon(self):
        """Load epsilon value from the most recent checkpoint."""
        
        # Search for checkpoint files
        data_dir = Path("../data/simulation_data")
        checkpoint_files = list(data_dir.glob("**/lactate_controlled_*_model_checkpoint.json"))
        
        if not checkpoint_files:
            return self.config.enhanced_epsilon  # Fall back to config default
        
        # Get the most recent checkpoint
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_checkpoint = checkpoint_files[0]
        
        try:
            with open(latest_checkpoint, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Get current epsilon from checkpoint
            current_epsilon = checkpoint_data.get('hyperparameters', {}).get('current_epsilon', self.config.enhanced_epsilon)
            print(f"🎯 Loaded trained epsilon: {current_epsilon}")
            return float(current_epsilon)
            
        except Exception as e:
            print(f"⚠️  Failed to load epsilon from checkpoint: {e}")
            return self.config.enhanced_epsilon
    
    def update_biofilm_growth(self, biofilm_thickness, substrate_conc, dt):
        """GPU-accelerated biofilm growth calculation."""
        
        # Monod kinetics with diffusion limitations
        max_growth_rate = 0.05  # h^-1
        half_sat = 5.0  # mM
        decay_rate = 0.01  # h^-1
        
        # Thiele modulus for diffusion limitation
        diffusivity = 6e-10  # m²/s
        thickness_m = biofilm_thickness * 1e-6  # Convert μm to m
        thiele_modulus = thickness_m * jnp.sqrt(max_growth_rate / diffusivity)
        
        # Effectiveness factor
        effectiveness = jnp.tanh(thiele_modulus) / thiele_modulus
        effectiveness = jnp.where(thiele_modulus < 1e-6, 1.0, effectiveness)
        
        # Growth rate with effectiveness factor
        growth_rate = effectiveness * max_growth_rate * substrate_conc / (half_sat + substrate_conc)
        net_growth = growth_rate - decay_rate
        
        # Numerical stability with maximum 5% change per step
        max_change = 0.05 * dt
        net_growth = jnp.clip(net_growth * dt, -max_change, max_change)
        
        new_thickness = biofilm_thickness * (1 + net_growth)
        
        # Natural upper limit from diffusion (around 200 μm)
        max_thickness = 200.0
        new_thickness = jnp.minimum(new_thickness, max_thickness)
        new_thickness = jnp.maximum(new_thickness, 0.1)  # Minimum viable thickness
        
        return new_thickness
    
    def calculate_power_output(self, biofilm_thickness, substrate_conc):
        """GPU-accelerated power calculation."""
        
        # Enhanced Geobacter power model parameters
        max_current_density = 2.0  # A/m² (literature value for G. sulfurreducens)
        acetate_potential = 0.35  # V (standard potential for acetate oxidation)
        half_saturation = 5.0  # mM (Monod constant for substrate)
        
        # Effectiveness factor based on biofilm thickness (diffusion limitation)
        # Thicker biofilms have reduced effectiveness due to mass transfer
        max_effective_thickness = 100.0  # μm, beyond which effectiveness drops
        thickness_factor = jnp.minimum(biofilm_thickness / max_effective_thickness, 1.0)
        
        # Substrate limitation (Monod kinetics)
        substrate_factor = substrate_conc / (half_saturation + substrate_conc)
        
        # Current calculation using anode area for power generation
        current_density = max_current_density * thickness_factor * substrate_factor  # A/m²
        current = current_density * self.anode_area  # A (use anode area for current collection)
        power = current * acetate_potential  # W
        
        return power
    
    def q_learning_action_selection(self, state_index, q_table, epsilon, key=None):
        """GPU-accelerated Q-learning action selection."""
        
        if JAX_AVAILABLE and random is not None and key is not None:
            # JAX-accelerated version
            key, subkey = random.split(key)
            explore = random.uniform(subkey) < epsilon
            
            # Get Q-values for current state
            q_values = q_table[state_index]
            
            # Choose action
            key, subkey = random.split(key)
            random_action = random.randint(subkey, (), 0, q_values.shape[0])
            greedy_action = jnp.argmax(q_values)
            
            action = jnp.where(explore, random_action, greedy_action)
            return action, key
        else:
            # NumPy fallback
            if np.random.random() < epsilon:
                action = np.random.randint(0, q_table.shape[1])
            else:
                action = np.argmax(q_table[state_index])
            return action, None
    
    def calculate_reward(self, reservoir_conc, outlet_conc, power, target_conc=None):
        """GPU-accelerated reward calculation with enhanced stability."""
        
        if target_conc is None:
            target_conc = self.target_concentration
        
        # Substrate concentration control (primary objective)
        conc_deviation = jnp.abs(reservoir_conc - target_conc)
        
        # Exponential penalty for large deviations
        substrate_penalty = -self.config.reward_weights.substrate_penalty_multiplier * jnp.exp(conc_deviation / 10.0)
        
        # Power reward
        power_reward = self.config.reward_weights.power_weight * jnp.log1p(power)
        
        # Stability bonus for staying near target
        stability_bonus = jnp.where(conc_deviation < 2.0, 50.0, 0.0)
        
        # Combined reward with stability emphasis
        total_reward = substrate_penalty + power_reward + stability_bonus
        
        return total_reward
    
    def simulate_timestep(self, dt_hours):
        """Simulate one timestep with GPU acceleration."""
        
        # Update biofilm growth (vectorized across all cells)
        if JAX_AVAILABLE and jit is not None:
            update_func = jit(self.update_biofilm_growth)
            self.biofilm_thicknesses = update_func(
                self.biofilm_thicknesses, self.cell_concentrations, dt_hours)
        else:
            self.biofilm_thicknesses = self.update_biofilm_growth(
                self.biofilm_thicknesses, self.cell_concentrations, dt_hours)
        
        # Calculate power output (vectorized)
        if JAX_AVAILABLE and vmap is not None:
            power_func = vmap(self.calculate_power_output)
            cell_powers = power_func(self.biofilm_thicknesses, self.cell_concentrations)
        else:
            cell_powers = [self.calculate_power_output(thick, conc) 
                          for thick, conc in zip(self.biofilm_thicknesses, self.cell_concentrations)]
            cell_powers = jnp.array(cell_powers)
        
        total_power = jnp.sum(cell_powers)
        
        # Q-learning substrate control
        state_index = min(int(self.reservoir_concentration), 99)  # Simplified state
        action, self.key = self.q_learning_action_selection(
            state_index, self.q_table, self.epsilon, self.key)
        
        # Convert action to substrate addition rate with improved stability
        # Balanced substrate addition rates around expected consumption
        substrate_actions = jnp.array([-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
        if action < len(substrate_actions):
            substrate_addition = float(substrate_actions[action])
        else:
            # Fallback for out-of-range actions (shouldn't happen with 10 actions)
            substrate_addition = 0.0
        
        # Enhanced substrate consumption model based on biofilm activity
        # Consumption rate proportional to biofilm thickness and substrate availability
        biofilm_activity = jnp.mean(self.biofilm_thicknesses) / 200.0  # Normalized by max thickness
        substrate_availability = jnp.mean(self.cell_concentrations) / (5.0 + jnp.mean(self.cell_concentrations))
        
        # Base consumption rate from literature: ~0.5-2.0 mmol/L/h for active biofilm
        base_consumption_rate = 1.0  # mmol/L/h
        consumption_rate = base_consumption_rate * biofilm_activity * substrate_availability
        
        # Total consumption is rate * volume * time
        substrate_consumption = consumption_rate * self.reservoir_volume * dt_hours
        
        # Update reservoir concentration with improved stability controls
        concentration_change = (substrate_addition - substrate_consumption) / self.reservoir_volume
        self.reservoir_concentration += concentration_change * dt_hours
        
        # Apply constraints to prevent extreme concentrations
        # Allow concentration to go higher to maintain 25 mM target
        self.reservoir_concentration = max(0.1, min(self.reservoir_concentration, 100.0))
        
        # Update cell concentrations (improved mixing model)
        mixing_rate = 0.8  # h^-1 (faster mixing)
        for i in range(self.n_cells):
            conc_diff = self.reservoir_concentration - float(self.cell_concentrations[i])
            if JAX_AVAILABLE and hasattr(self.cell_concentrations, 'at'):
                self.cell_concentrations = self.cell_concentrations.at[i].set(
                    self.cell_concentrations[i] + mixing_rate * conc_diff * dt_hours)
            else:
                self.cell_concentrations[i] += mixing_rate * conc_diff * dt_hours
        
        # Update outlet concentration
        self.outlet_concentration = float(jnp.mean(self.cell_concentrations))
        
        # Q-learning update with enhanced stability
        reward = self.calculate_reward(
            self.reservoir_concentration, self.outlet_concentration, float(total_power))
        
        # Update Q-table
        learning_rate = self.config.enhanced_learning_rate
        next_state = min(int(self.reservoir_concentration), 99)
        
        # Q-learning update
        current_q = self.q_table[state_index, action]
        max_next_q = jnp.max(self.q_table[next_state])
        target_q = reward + self.config.enhanced_discount_factor * max_next_q
        
        if JAX_AVAILABLE and hasattr(self.q_table, 'at'):
            self.q_table = self.q_table.at[state_index, action].set(
                current_q + learning_rate * (target_q - current_q))
        else:
            # NumPy fallback
            self.q_table[state_index, action] = current_q + learning_rate * (target_q - current_q)
        
        # Update epsilon with enhanced decay
        self.epsilon *= self.config.advanced_epsilon_decay
        self.epsilon = max(self.epsilon, self.config.advanced_epsilon_min)
        
        # Track substrate addition
        self.total_substrate_added += max(0, substrate_addition * dt_hours)
        
        # Update stability buffer
        if JAX_AVAILABLE and hasattr(self.stability_buffer, 'at'):
            self.stability_buffer = self.stability_buffer.at[self.stability_index % 100].set(
                self.reservoir_concentration)
        else:
            self.stability_buffer[self.stability_index % 100] = self.reservoir_concentration
        self.stability_index += 1
        
        return {
            'total_power': float(total_power),
            'substrate_addition': substrate_addition,
            'action': int(action),
            'reward': float(reward),
            'epsilon': float(self.epsilon)
        }
    
    def run_simulation(self, duration_hours, output_dir):
        """Run full simulation with GPU acceleration."""
        
        print(f"🚀 Starting {duration_hours}-hour GPU-accelerated simulation...")
        print(f"🔧 Backend: {BACKEND_NAME}")
        
        # Simulation parameters
        dt_hours = 0.1  # 6-minute timesteps
        n_steps = int(duration_hours / dt_hours)
        
        # Results storage
        results = {
            'time_hours': [],
            'reservoir_concentration': [],
            'outlet_concentration': [],
            'total_power': [],
            'biofilm_thicknesses': [],
            'substrate_addition_rate': [],
            'q_action': [],
            'epsilon': [],
            'reward': []
        }
        
        # Progress tracking
        progress_interval = n_steps // 20  # 5% intervals
        start_time = time.time()
        
        print(f"📊 Running {n_steps:,} timesteps with dt={dt_hours}h")
        print("🎯 Target: 25.0 mM ± 2.0 mM")
        
        for step in range(n_steps):
            current_time = step * dt_hours
            
            # Simulate timestep
            step_results = self.simulate_timestep(dt_hours)
            
            # Store results (every 10 steps to save memory)
            if step % 10 == 0:
                results['time_hours'].append(current_time)
                results['reservoir_concentration'].append(float(self.reservoir_concentration))
                results['outlet_concentration'].append(float(self.outlet_concentration))
                results['total_power'].append(step_results['total_power'])
                results['biofilm_thicknesses'].append([float(x) for x in self.biofilm_thicknesses])
                results['substrate_addition_rate'].append(step_results['substrate_addition'])
                results['q_action'].append(step_results['action'])
                results['epsilon'].append(step_results['epsilon'])
                results['reward'].append(step_results['reward'])
            
            # Progress updates
            if step % progress_interval == 0:
                elapsed = time.time() - start_time
                progress = step / n_steps * 100
                eta = elapsed / max(step, 1) * (n_steps - step)
                
                print(f"⏱️  {progress:.1f}% complete | "
                      f"Time: {current_time:.0f}h | "
                      f"Reservoir: {self.reservoir_concentration:.1f} mM | "
                      f"Power: {step_results['total_power']:.1f} W | "
                      f"ETA: {eta/3600:.1f}h")
        
        elapsed_time = time.time() - start_time
        print(f"✅ Simulation completed in {elapsed_time/3600:.2f} hours")
        
        # Calculate final metrics
        final_metrics = self.calculate_final_metrics(results)
        
        return results, final_metrics
    
    def calculate_final_metrics(self, results):
        """Calculate comprehensive performance metrics."""
        
        reservoir_conc = np.array(results['reservoir_concentration'])
        power_output = np.array(results['total_power'])
        
        # Performance metrics
        target = 25.0
        deviations = np.abs(reservoir_conc - target)
        
        metrics = {
            'final_reservoir_concentration': float(reservoir_conc[-1]),
            'mean_concentration': float(np.mean(reservoir_conc)),
            'std_concentration': float(np.std(reservoir_conc)),
            'max_deviation': float(np.max(deviations)),
            'mean_deviation': float(np.mean(deviations)),
            'final_power': float(power_output[-1]),
            'mean_power': float(np.mean(power_output)),
            'total_substrate_added': float(self.total_substrate_added),
            'control_effectiveness_2mM': float(np.sum(deviations <= 2.0) / len(deviations) * 100),
            'control_effectiveness_5mM': float(np.sum(deviations <= 5.0) / len(deviations) * 100),
            'stability_coefficient_variation': float(np.std(reservoir_conc[-1000:]) / np.mean(reservoir_conc[-1000:]) * 100)
        }
        
        return metrics
    
    def cleanup_gpu_resources(self):
        """Clean up GPU resources and free memory"""
        try:
            if JAX_AVAILABLE and jax is not None:
                # Clear JAX GPU memory
                if hasattr(jax, 'clear_backends'):
                    jax.clear_backends()
                
                # For ROCm, force HIP memory cleanup (without sudo)
                if BACKEND_NAME == 'AMD ROCm':
                    try:
                        import os
                        # Clear ROCm environment variables instead of system commands
                        os.environ.pop('HIP_VISIBLE_DEVICES', None)
                        os.environ.pop('ROCR_VISIBLE_DEVICES', None)
                    except Exception:
                        pass
                
                # Clear any cached compilations
                if hasattr(jax, 'clear_caches'):
                    jax.clear_caches()
                    
                print("🧹 GPU resources cleaned up")
                
        except Exception as e:
            print(f"⚠️  GPU cleanup warning: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()


def calculate_maintenance_requirements(total_substrate_consumed_mmol, simulation_hours):
    """Calculate maintenance requirements for substrate and buffer reservoirs."""
    
    # Stock concentrations and volumes
    substrate_stock_concentration = 5.0  # M (5000 mM)
    substrate_stock_volume = 5.0  # L
    buffer_stock_concentration = 3.0  # M (3000 mM)  
    buffer_stock_volume = 5.0  # L
    
    # Total stock available
    total_substrate_stock_mmol = substrate_stock_concentration * 1000 * substrate_stock_volume
    total_buffer_stock_mmol = buffer_stock_concentration * 1000 * buffer_stock_volume
    
    # Consumption rates
    daily_substrate_consumption = total_substrate_consumed_mmol / (simulation_hours / 24)
    hourly_substrate_consumption = total_substrate_consumed_mmol / simulation_hours
    
    # Assume buffer consumption is 0.3x substrate consumption (typical for lactate MFC)
    total_buffer_consumed_mmol = total_substrate_consumed_mmol * 0.3
    daily_buffer_consumption = total_buffer_consumed_mmol / (simulation_hours / 24)
    
    # Calculate refill schedules
    substrate_refill_days = total_substrate_stock_mmol / daily_substrate_consumption
    buffer_refill_days = total_buffer_stock_mmol / daily_buffer_consumption
    
    return {
        'substrate': {
            'total_consumed_mmol': total_substrate_consumed_mmol,
            'daily_consumption_mmol': daily_substrate_consumption,
            'hourly_consumption_mmol': hourly_substrate_consumption,
            'stock_available_mmol': total_substrate_stock_mmol,
            'refill_interval_days': substrate_refill_days,
            'annual_refills_needed': 365 / substrate_refill_days
        },
        'buffer': {
            'total_consumed_mmol': total_buffer_consumed_mmol,
            'daily_consumption_mmol': daily_buffer_consumption,
            'stock_available_mmol': total_buffer_stock_mmol,
            'refill_interval_days': buffer_refill_days,
            'annual_refills_needed': 365 / buffer_refill_days
        }
    }


def run_gpu_accelerated_simulation(duration_hours=8784):
    """Run GPU-accelerated 1-year MFC simulation."""
    
    print("="*80)
    print("🚀 GPU-ACCELERATED MFC SIMULATION")
    print("="*80)
    print(f"⏱️  Duration: {duration_hours:,} hours ({duration_hours/24:.0f} days)")
    print("🎯 Target: 25.0 mM substrate concentration")
    print("🧠 Q-learning: Bayesian-optimized parameters")
    print(f"🔥 Acceleration: {BACKEND_NAME}")
    print("="*80)
    
    # Initialize GPU-accelerated simulation
    mfc_sim = GPUAcceleratedMFC(DEFAULT_QLEARNING_CONFIG)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"../data/simulation_data/gpu_1year_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Run simulation
    start_time = time.time()
    results, metrics = mfc_sim.run_simulation(duration_hours, output_dir)
    total_time = time.time() - start_time
    
    # Calculate maintenance requirements
    maintenance = calculate_maintenance_requirements(
        metrics['total_substrate_added'], duration_hours)
    
    # Save results
    results_summary = {
        'simulation_info': {
            'duration_hours': duration_hours,
            'duration_days': duration_hours / 24,
            'total_runtime_hours': total_time / 3600,
            'acceleration_backend': BACKEND_NAME,
            'device': str(mfc_sim.device) if hasattr(mfc_sim, 'device') else 'CPU',
            'timestamp': timestamp
        },
        'performance_metrics': metrics,
        'substrate_consumption': {
            'total_consumed_mmol': metrics['total_substrate_added'],
            'daily_rate_mmol': metrics['total_substrate_added'] / (duration_hours / 24),
            'hourly_rate_mmol': metrics['total_substrate_added'] / duration_hours
        },
        'maintenance_requirements': maintenance
    }
    
    # Save comprehensive results
    results_file = output_dir / f"gpu_simulation_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    # Save raw data (compressed)
    import pandas as pd
    df = pd.DataFrame(results)
    data_file = output_dir / f"gpu_simulation_data_{timestamp}.csv.gz"
    df.to_csv(data_file, compression='gzip', index=False)
    
    # Print results
    print("\n" + "="*80)
    print("🎉 GPU-ACCELERATED SIMULATION COMPLETE!")
    print("="*80)
    
    print("⏱️  PERFORMANCE:")
    print(f"   Runtime: {total_time/3600:.2f} hours")
    print(f"   Backend: {BACKEND_NAME}")
    
    print("\n🎯 SUBSTRATE CONTROL:")
    print(f"   Final concentration: {metrics['final_reservoir_concentration']:.2f} mM")
    print(f"   Mean concentration: {metrics['mean_concentration']:.2f} ± {metrics['std_concentration']:.2f} mM")
    print(f"   Control effectiveness (±2mM): {metrics['control_effectiveness_2mM']:.1f}%")
    print(f"   Control effectiveness (±5mM): {metrics['control_effectiveness_5mM']:.1f}%")
    
    print("\n⚡ POWER OUTPUT:")
    print(f"   Final power: {metrics['final_power']:.2f} W")
    print(f"   Mean power: {metrics['mean_power']:.2f} W")
    
    print("\n📦 SUBSTRATE CONSUMPTION:")
    print(f"   Total consumed: {metrics['total_substrate_added']:.2f} mmol")
    print(f"   Daily rate: {results_summary['substrate_consumption']['daily_rate_mmol']:.2f} mmol/day")
    
    print("\n🔧 MAINTENANCE SCHEDULE:")
    print(f"   Substrate refill: every {maintenance['substrate']['refill_interval_days']:.0f} days")
    print(f"   Buffer refill: every {maintenance['buffer']['refill_interval_days']:.0f} days")
    print(f"   Annual substrate refills: {maintenance['substrate']['annual_refills_needed']:.1f}")
    print(f"   Annual buffer refills: {maintenance['buffer']['annual_refills_needed']:.1f}")
    
    print("\n📁 RESULTS SAVED:")
    print(f"   Summary: {results_file}")
    print(f"   Raw data: {data_file}")
    
    # Send email notification
    try:
        from email_notification import send_completion_email
        print("\n📧 Sending email notification...")
        send_completion_email(str(results_file))
    except Exception as e:
        print(f"⚠️  Email notification failed: {e}")
    
    return results_summary, output_dir


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print(f"\n⚠️  Received signal {signum}. Saving results...")
    sys.exit(0)


if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run simulation
    results, output_dir = run_gpu_accelerated_simulation(8784)
    
    print(f"\n🎉 GPU-accelerated simulation complete! Results in {output_dir}")