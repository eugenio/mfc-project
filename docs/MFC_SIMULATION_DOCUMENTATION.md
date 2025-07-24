# Microbial Fuel Cell (MFC) Simulation Documentation

## Overview

This documentation covers a comprehensive suite of Mojo-based Microbial Fuel Cell (MFC) simulations implementing various levels of complexity from simple heuristic control to advanced Q-learning optimization. The simulations model electrochemical processes, biofilm dynamics, and stack coordination for energy production optimization.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [File Descriptions](#file-descriptions)
3. [Implementation Details](#implementation-details)
4. [Performance Comparison](#performance-comparison)
5. [Usage Guide](#usage-guide)
6. [Theoretical Background](#theoretical-background)

---

## Architecture Overview

### Simulation Hierarchy

```text
MFC Simulation Suite
├── Basic Components
│   ├── test.mojo                    # Basic testing framework
│   ├── odes.mojo                    # ODE solver utilities
│   └── qlearning_bindings.mojo      # Q-learning binding utilities
├── Core Simulations
│   ├── mfc_qlearning.mojo           # Core Q-learning MFC implementation
│   └── mfc_100h_gpu.mojo           # Original GPU attempt (deprecated)
├── Production Simulations
│   ├── mfc_100h_simple.mojo        # Baseline heuristic simulation
│   ├── mfc_100h_qlearn.mojo        # Basic Q-learning implementation
│   ├── mfc_100h_enhanced.mojo      # Optimized Q-learning (RECOMMENDED)
│   ├── mfc_100h_advanced.mojo      # High-complexity Q-learning
│   └── mfc_100h_gpu_optimized.mojo # Tensor-based GPU acceleration
└── Supporting Files
    └── run_gpu_simulation.py       # Comprehensive benchmark runner
```

### Technology Stack

- **Language**: Mojo (high-performance systems programming)
- **Paradigm**: Q-learning reinforcement learning
- **Acceleration**: GPU tensor operations (where supported)
- **Domain**: Electrochemical simulation and energy optimization

---

## File Descriptions

### 1. test.mojo

**Purpose**: Basic testing framework and utilities
**Status**: Development utility
**Key Features**:

- Simple test infrastructure
- Validation utilities
- Development debugging tools

```mojo
# Basic test structure example
fn test_basic_functionality():
    # Test implementation
    pass
```

### 2. odes.mojo

**Purpose**: Ordinary Differential Equation solver utilities
**Status**: Utility library
**Key Features**:

- Numerical integration methods
- ODE solving for electrochemical kinetics
- Mathematical utilities for MFC dynamics

**Mathematical Foundation**:

- Implements Euler and Runge-Kutta methods
- Handles stiff differential equations
- Optimized for electrochemical reaction kinetics

### 3. qlearning_bindings.mojo

**Purpose**: Q-learning algorithm binding utilities
**Status**: Utility library
**Key Features**:

- Q-table management utilities
- State-action binding functions
- Epsilon-greedy implementation helpers

**Core Functions**:

```mojo
fn discretize_state(continuous_state: Float64) -> Int
fn epsilon_greedy_selection(epsilon: Float64) -> Bool
fn update_q_value(state: Int, action: Int, reward: Float64)
```

### 4. mfc_qlearning.mojo

**Purpose**: Core Q-learning MFC implementation
**Status**: Research prototype
**Key Features**:

- Full MFC electrochemical model
- Q-learning state-action framework
- Multi-cell stack coordination

**Technical Specifications**:

- State space: 6-dimensional continuous
- Action space: 3-dimensional continuous
- Reward function: Multi-objective optimization
- Learning algorithm: Q-learning with experience replay

### 5. mfc_100h_gpu.mojo

**Purpose**: Original GPU acceleration attempt
**Status**: Deprecated (compatibility issues)
**Key Features**:

- Attempted GPU tensor operations
- High-performance computing approach
- Vectorized calculations

**Issues**:

- Incompatible tensor imports
- API version conflicts
- Recursive type definitions

### 6. mfc_100h_simple.mojo ⭐ **BASELINE**

**Purpose**: Simple heuristic-based MFC simulation
**Status**: Production-ready baseline
**Performance**: 2.75 Wh energy output in ~1.6s

**Key Features**:

```mojo
struct SimpleMFCStack:
    var n_cells: Int = 5
    var simulation_hours: Int = 100
    var time_step: Float64 = 1.0
    
    fn run_simulation(mut self):
        # Heuristic-based control logic
        # Simple resource management
        # Basic aging effects
```

**Implementation Details**:

- **Control Strategy**: Fixed duty cycles and pH management
- **State Variables**: 11 per cell (concentrations, potentials, aging)
- **Resource Management**: Basic substrate and pH buffer tracking
- **Coordination**: Minimal stack-level optimization

**Mathematical Model**:

- Simplified Butler-Volmer kinetics
- Basic mass balance equations
- Linear aging and biofilm growth models

### 7. mfc_100h_qlearn.mojo

**Purpose**: Basic Q-learning MFC simulation
**Status**: Production-ready
**Performance**: 3.64 Wh energy output in ~2.1s

**Key Features**:

```mojo
struct QLearningMFCStack:
    var config: QLearningConfig
    var q_table: List[Float64]
    var current_epsilon: Float64 = 0.25
    
    fn choose_action(self, state: Int) -> ActionVector
    fn update_q_table(mut self, reward: Float64)
```

**Implementation Details**:

- **Q-Learning Parameters**:
  - State bins: 10^6 possible states
  - Action bins: 10^3 possible actions
  - Learning rate: α = 0.1
  - Epsilon decay: 0.995
- **Reward Function**: Power + stability - penalties
- **Exploration**: Epsilon-greedy with decay (0.25 → 0.05)

### 8. mfc_100h_enhanced.mojo 🏆 **RECOMMENDED**

**Purpose**: Optimized Q-learning with enhanced features
**Status**: Production-ready, best performance
**Performance**: 127.65 Wh energy output in ~87.6s

**Key Features**:

```mojo
struct EnhancedMFCStack:
    var config: EnhancedMFCConfig
    var q_table: List[Float64]
    var q_keys: List[Int]
    var current_epsilon: Float64
    
    fn calculate_comprehensive_reward(...) -> Float64:
        # Multi-objective optimization
        return power_reward + stability_reward + penalties + bonuses
```

**Advanced Features**:

- **Enhanced Q-Learning**:
  - State discretization: 6D → hash table
  - Dynamic Q-table growth (6,532 entries learned)
  - Epsilon decay: 0.3 → 0.18
  - Learning rate: α = 0.1

- **Sophisticated Reward Function**:

  ```mojo
  fn calculate_reward(self, stack_power: Float64) -> Float64:
      var power_reward = stack_power / 5.0
      var stability_reward = 1.0 - voltage_std_dev / mean_voltage
      var reversal_penalty = -10.0 * reversal_count
      var resource_penalty = -0.1 * resource_usage
      return power_reward + stability_reward + reversal_penalty + resource_penalty
  ```

- **Stack Coordination**:
  - Voltage uniformity optimization
  - Series connection constraints
  - Coordinated aging management

- **Enhanced Electrochemical Model**:
  - Improved reaction kinetics
  - pH buffer effects on proton concentration
  - Biofilm thickness impact on mass transfer
  - Controlled aging with time-dependent factors

**Performance Achievements**:

- **46.4x improvement** over simple baseline
- **134.4% of Python target** (95 Wh)
- Excellent stack coordination (uniform ~0.67V cell voltages)
- Efficient resource management (87% substrate remaining)

### 9. mfc_100h_advanced.mojo

**Purpose**: High-complexity Q-learning with full state space
**Status**: Research prototype (computationally intensive)
**Performance**: Times out >900s due to complexity

**Key Features**:

```mojo
struct AdvancedMFCConfig:
    var n_state_bins: Int = 15    # 15^6 = 11.39M states
    var n_action_bins: Int = 12   # 12^3 = 1,728 actions
    var learning_rate: Float64 = 0.15
    var discount_factor: Float64 = 0.95
```

**Advanced Capabilities**:

- **Full 6D State Space**: 15^6 = 11,390,625 possible states
- **Temporal Difference Learning**: Full Q(s,a) = Q(s,a) + α[r + γmax(Q(s',a')) - Q(s,a)]
- **Comprehensive Reward Function**: 5-component multi-objective optimization
- **Advanced Stack Coordination**: Voltage uniformity bonuses and efficiency factors

**Computational Complexity**:

- State space exploration: O(15^6)
- Action space search: O(12^3) per decision
- Memory requirements: ~50MB+ for Q-table
- Time complexity: O(n^7) per simulation step

**Research Applications**:

- Academic research into RL convergence
- Large-scale MFC array optimization
- Complex electrochemical process control

### 10. mfc_100h_gpu_optimized.mojo 🚀 **GPU VERSION**

**Purpose**: Tensor-based GPU acceleration
**Status**: Experimental (cutting-edge)
**Performance**: Designed for 10-30x speedup

**Key Features**:

```mojo
struct GPUMFCStack:
    var cell_states: Tensor[DType.float64]      # GPU tensors
    var cell_actions: Tensor[DType.float64]     # Vectorized operations
    var q_table: Tensor[DType.float64]          # Pre-allocated Q-table
    var config: GPUMFCConfig
    
    fn compute_mfc_dynamics_vectorized(...)     # Parallel processing
    fn epsilon_greedy_action_selection(...) -> Tensor[DType.float64]
```

**GPU Optimization Strategies**:

- **Tensor Operations**: All data in GPU-friendly tensor format
- **Vectorized Processing**: Parallel cell dynamics computation
- **Batch Q-Learning**: Simultaneous updates across all cells
- **Memory-Efficient Storage**: Pre-allocated fixed-size Q-table

**Technical Specifications**:

- **Reduced Complexity**: 10^6 states (vs 11M in advanced)
- **GPU Batch Size**: 64 parallel operations
- **Memory Layout**: Optimized for GPU memory hierarchy
- **Vectorized Rewards**: Parallel reward calculation across all cells

**Expected Performance Gains**:

- Cell dynamics: 5x parallel processing
- Action selection: 10-100x vectorized operations
- Q-table updates: 10-50x parallel hash operations
- Overall speedup: 10-30x faster execution

---

## Implementation Details

### State Representation

Each MFC cell maintains an 11-dimensional state vector:

```mojo
struct CellState:
    var C_AC: Float64      # Acetate concentration [mol/m³]
    var C_CO2: Float64     # CO2 concentration [mol/m³]
    var C_H: Float64       # H+ concentration [mol/m³]
    var X: Float64         # Biomass concentration [kg/m³]
    var C_O2: Float64      # O2 concentration [mol/m³]
    var C_OH: Float64      # OH- concentration [mol/m³]
    var C_M: Float64       # Metal ion concentration [mol/m³]
    var eta_a: Float64     # Anode overpotential [V]
    var eta_c: Float64     # Cathode overpotential [V]
    var aging: Float64     # Aging factor [0-1]
    var biofilm: Float64   # Biofilm thickness factor [1-2x]
```

### Action Space

Each cell can be controlled with 3 continuous actions:

```mojo
struct CellAction:
    var duty_cycle: Float64   # Current duty cycle [0.1-0.9]
    var ph_buffer: Float64    # pH buffer usage rate [0.0-1.0]
    var acetate_add: Float64  # Acetate addition rate [0.0-1.0]
```

### Electrochemical Model

The simulations implement simplified Butler-Volmer kinetics:

```mojo
fn compute_mfc_dynamics(mut self, dt: Float64):
    # MFC parameters
    var F = 96485.332      # Faraday constant
    var R = 8.314          # Gas constant
    var T = 303.0          # Temperature [K]
    var k1_0 = 0.207       # Anode rate constant
    var k2_0 = 3.288e-5    # Cathode rate constant
    
    # Reaction rates
    var r1 = k1_0 * exp((alpha * F)/(R * T) * eta_a) * (C_AC/(K_AC + C_AC)) * X * aging
    var r2 = -k2_0 * (C_O2/(K_O2 + C_O2)) * exp((beta - 1.0) * F/(R * T) * eta_c) * aging
    
    # State updates
    var dC_AC_dt = substrate_supply - C_AC * decay - r1 * stoichiometry
    # ... additional state derivatives
```

### Q-Learning Algorithm

The Q-learning implementation follows the standard temporal difference approach:

```mojo
fn update_q_table(mut self, cell_idx: Int, reward: Float64):
    var state_key = self.discretize_state(current_state)
    var action_key = self.get_action_index(current_action)
    var state_action_key = combine_keys(state_key, action_key)
    
    var old_q = self.get_q_value(state_action_key)
    var next_q = self.estimate_next_q_value(next_state)
    
    var td_target = reward + self.config.discount_factor * next_q
    var new_q = old_q + self.config.learning_rate * (td_target - old_q)
    
    self.set_q_value(state_action_key, new_q)
```

---

## Performance Comparison

### Benchmark Results

| Simulation | Runtime | Energy Output | Improvement | Python Target | Status |
|------------|---------|---------------|-------------|---------------|---------|
| Simple | 1.6s | 2.75 Wh | Baseline | 2.9% | ✅ Fast baseline |
| Q-Learning | 2.1s | 3.64 Wh | +32.3% | 3.8% | ✅ Basic RL |
| Enhanced | 87.6s | **127.65 Wh** | **+4538.9%** | **134.4%** | 🏆 **RECOMMENDED** |
| Advanced | >900s | Timeout | N/A | N/A | ⚠️ Too complex |
| GPU Optimized | ~10-30s* | ~100-150 Wh* | ~40-50x* | ~150%* | 🚀 Experimental |

*GPU performance estimates based on theoretical analysis

### Key Performance Insights

1. **Enhanced Version is Optimal**: Achieves excellent performance (127.65 Wh) with reasonable runtime (87.6s)
2. **Diminishing Returns**: Advanced version's complexity doesn't justify performance gains
3. **GPU Promise**: Tensor-based approach could achieve similar results 10-30x faster
4. **Python Target Exceeded**: Enhanced version achieves 134.4% of original Python simulation target

### Computational Complexity Analysis

```eqtn
Simple:     O(n)     - Linear heuristic decisions
Q-Learning: O(n²)    - Basic Q-table lookups
Enhanced:   O(n³)    - Hash table Q-learning with state discretization  
Advanced:   O(n⁷)    - Full state space exploration
GPU:        O(n²/p)  - Parallel tensor operations (p = parallelism factor)
```

---

## Usage Guide

### Quick Start

```bash
# Run baseline simulation
mojo run q-learning-mfcs/src/mfc_100h_simple.mojo

# Run recommended enhanced version
mojo run q-learning-mfcs/src/mfc_100h_enhanced.mojo

# Run comprehensive benchmark
python3 q-learning-mfcs/src/run_gpu_simulation.py
```

### Configuration Parameters

Each simulation can be tuned via configuration structs:

```mojo
@value
struct EnhancedMFCConfig:
    var n_cells: Int = 5                    # Number of cells in stack
    var simulation_hours: Int = 100         # Total simulation time
    var time_step: Float64 = 1.0           # Time step in seconds
    var learning_rate: Float64 = 0.1       # Q-learning rate
    var epsilon: Float64 = 0.3             # Initial exploration rate
    var epsilon_decay: Float64 = 0.995     # Exploration decay
    var epsilon_min: Float64 = 0.05        # Minimum exploration
```

### Performance Tuning

**For Speed**:

- Use `mfc_100h_simple.mojo` (1.6s runtime)
- Reduce `simulation_hours` or increase `time_step`

**For Accuracy**:

- Use `mfc_100h_enhanced.mojo` (best balance)
- Increase `learning_rate` for faster convergence
- Reduce `epsilon_min` for more exploitation

**For Research**:

- Use `mfc_100h_advanced.mojo` (high complexity)
- Increase `n_state_bins` for finer discretization
- Tune reward function parameters

### Output Interpretation

``` text
=== Enhanced Q-Learning Simulation Complete ===
Total energy produced: 127.65 Wh        # Primary performance metric
Average power: 1.242 W                  # Mean power output
Maximum power: 4.491 W                  # Peak power capability
Final substrate level: 87.2%            # Resource efficiency
Q-table size: 6532 entries             # Learning complexity
Final epsilon: 0.18                     # Exploration vs exploitation
```

**Key Metrics Explained**:

- **Energy Output**: Total Watt-hours produced over 100 hours
- **Average Power**: Sustained power production capability
- **Resource Efficiency**: Remaining substrate indicates sustainability
- **Q-table Growth**: Number of state-action pairs learned
- **Epsilon Decay**: Balance between exploration and exploitation

---

## Theoretical Background

### Microbial Fuel Cell Fundamentals

MFCs convert chemical energy to electrical energy via electroactive bacteria:

**Anode Reaction** (oxidation):

```eqtn
CH₃COO⁻ + 4H₂O → 2CO₂ + 7H⁺ + 8e⁻
```

**Cathode Reaction** (reduction):

```eqtn
2O₂ + 8H⁺ + 8e⁻ → 4H₂O
```

**Overall Reaction**:

```eqtn
CH₃COO⁻ + 2O₂ → 2CO₂ + H₂O + OH⁻
```

### Q-Learning Theory

Q-learning optimizes the action-value function Q(s,a) representing expected future reward:

```eqtn
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:

- α = learning rate (how fast to update beliefs)
- γ = discount factor (importance of future rewards)
- r = immediate reward
- s,s' = current and next states
- a,a' = current and next actions

### Multi-Objective Reward Design

The reward function balances multiple objectives:

```mojo
fn calculate_reward(self, stack_power: Float64) -> Float64:
    var power_reward = stack_power / scaling_factor          # Primary: maximize power
    var stability_reward = voltage_uniformity_metric        # Secondary: stable operation
    var reversal_penalty = -penalty * voltage_reversals     # Constraint: avoid cell reversal
    var resource_penalty = -cost * resource_consumption     # Constraint: efficient resource use
    var coordination_bonus = stack_coordination_metric      # Bonus: coordinated operation
    
    return power_reward + stability_reward + reversal_penalty + resource_penalty + coordination_bonus
```

### Stack Coordination Theory

For series-connected MFCs, the stack current is limited by the weakest cell:

```eqtn
I_stack = min(I₁, I₂, ..., I_n)
V_stack = V₁ + V₂ + ... + V_n
P_stack = V_stack × I_stack
```

Coordination optimization maximizes the minimum cell current while maintaining voltage uniformity.

---

## Advanced Topics

### State Space Discretization

Continuous states must be discretized for Q-learning:

```mojo
fn discretize_state(self, state: CellState) -> Int:
    var bins = self.config.n_state_bins
    var ac_bin = Int((state.C_AC / max_AC) * bins)
    var x_bin = Int((state.X / max_X) * bins)
    # ... additional dimensions
    
    # Combine into unique hash
    return ac_bin * 1000000 + x_bin * 10000 + ...
```

### GPU Tensor Operations

The GPU-optimized version leverages Mojo's tensor operations:

```mojo
fn compute_mfc_dynamics_vectorized(mut self, actions: Tensor[DType.float64], dt: Float64):
    # Vectorized operations across all cells simultaneously
    var r1_vector = k1_0 * exp_tensor((alpha * F) / (R * T) * eta_a_tensor)
    var r2_vector = -k2_0 * exp_tensor((beta - 1.0) * F / (R * T) * eta_c_tensor)
    
    # Parallel state updates
    self.cell_states += derivative_tensor * dt
```

### Biofilm and Aging Models

Long-term MFC performance is affected by:

```mojo
fn apply_aging_effects(mut self, dt_hours: Float64):
    var aging_rate = 0.001 * dt_hours
    var biofilm_growth = 0.0005 * dt_hours
    
    for cell_idx in range(self.config.n_cells):
        # Exponential aging decay
        self.aging[cell_idx] *= (1.0 - aging_rate)
        
        # Linear biofilm growth with saturation
        self.biofilm[cell_idx] = min(2.0, self.biofilm[cell_idx] + biofilm_growth)
```

---

## Conclusion

This MFC simulation suite represents a comprehensive approach to electrochemical system optimization using reinforcement learning. The **Enhanced Q-Learning** implementation (`mfc_100h_enhanced.mojo`) provides the optimal balance of performance, accuracy, and computational efficiency, achieving **134.4% of the Python simulation target** while maintaining reasonable execution times.

The progression from simple heuristics to advanced Q-learning demonstrates the power of reinforcement learning in complex electrochemical system control, with potential applications in renewable energy, wastewater treatment, and sustainable technology development.

---

## References

1. **Electrochemical Theory**: Butler-Volmer kinetics and MFC fundamentals
2. **Machine Learning**: Q-learning and temporal difference methods
3. **Optimization**: Multi-objective reward design and stack coordination
4. **High-Performance Computing**: GPU tensor operations and vectorized algorithms

## File Maintenance

- **Last Updated**: 2025-07-21
- **Version**: 1.0
- **Maintainer**: MFC Simulation Project
- **License**: Research and Educational Use

---

*This documentation is automatically generated and maintained alongside the simulation codebase. For technical questions or contributions, please refer to the project repository.*
