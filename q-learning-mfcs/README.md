# MFC Stack Q-Learning Control System

A comprehensive implementation of Q-learning control for a 5-cell microbial fuel cell (MFC) stack with advanced sensor feedback and actuator control.

## Overview

This project implements a complete MFC stack control system featuring:

- **5-cell MFC stack simulation** with realistic electrochemical dynamics
- **Q-learning controller** optimized for accelerator hardware (GPU/NPU/ASIC)
- **Advanced sensor simulation** with EIS/QCM biofilm sensing, noise and calibration effects
- **Actuator control** for duty cycle, pH buffer, and acetate addition
- **Cell reversal prevention** and recovery mechanisms
- **Real-time optimization** for power stability and efficiency

## System Architecture

### Core Components

1. **MFC Model (`odes.mojo`)** - Electrochemical simulation using Mojo
1. **Q-Learning Controller (`mfc_qlearning.mojo`)** - Accelerated learning algorithm
1. **Stack Simulation (`mfc_stack_simulation.py`)** - Complete 5-cell system
1. **Sensor/Actuator Layer** - Hardware abstraction and control

### Key Features

#### Sensor Systems

- **Voltage sensors** - Individual cell and stack voltage monitoring
- **Current sensors** - Load current measurement with noise simulation
- **pH sensors** - Electrolyte pH monitoring for each cell
- **Acetate sensors** - Substrate concentration tracking
- **EIS sensors** - Electrochemical impedance spectroscopy for biofilm thickness monitoring
- **QCM sensors** - Quartz crystal microbalance for biofilm mass measurement
- **Sensor fusion** - Multi-algorithm data fusion with uncertainty quantification

#### Actuator Systems

- **Duty cycle control** - PWM-based current regulation (0-100%)
- **pH buffer pumps** - Automatic pH stabilization
- **Acetate addition** - Substrate feeding for extended operation
- **Response time simulation** - Realistic actuator dynamics

#### Control Objectives

- **Power optimization** - Maximize stack power output
- **Cell reversal prevention** - Avoid negative voltage conditions
- **Load balancing** - Equalize power across all cells
- **Stability maintenance** - Minimize power fluctuations
- **Resource optimization** - Efficient use of pH buffer and acetate

## Installation and Usage

### Prerequisites

```bash
# Install Mojo (if available)
# Install Python dependencies
pip install numpy matplotlib scipy
```

### Build Process

```bash
# Build the MFC model
mojo build odes.mojo --emit='shared-lib' -o odes.so

# Run the build script
python build_qlearning.py
```

### Running Simulations

#### Basic Q-Learning Demo

```bash
python mfc_qlearning_demo.py
```

#### 5-Cell Stack Simulation

```bash
python mfc_stack_simulation.py
```

#### Comprehensive Demonstration

```bash
python mfc_stack_demo.py
```

## Technical Details

### Q-Learning Implementation

#### State Space (40 dimensions)

- **Per cell (7 features × 5 cells)**:

  - Normalized acetate concentration
  - Biomass concentration
  - Normalized oxygen concentration
  - Normalized pH
  - Voltage reading
  - Power output
  - Reversal status flag

- **Stack level (5 features)**:

  - Stack voltage
  - Stack current
  - Stack power
  - Reversal ratio
  - Power imbalance

#### Action Space (15 dimensions)

- **Per cell (3 actions × 5 cells)**:
  - Duty cycle (0.1 - 0.9)
  - pH buffer activation (0 - 1)
  - Acetate addition rate (0 - 1)

#### Reward Function

```python
reward = power_reward + stability_reward + reversal_penalty + efficiency_reward + action_penalty
```

Where:

- `power_reward` = Normalized stack power output
- `stability_reward` = Power stability metric
- `reversal_penalty` = -10 × number of reversed cells
- `efficiency_reward` = Stack efficiency metric
- `action_penalty` = Penalty for extreme actuator values

### Control Strategy

#### Phase 1: Initialization

- System startup and stabilization
- Q-table initialization
- Sensor calibration

#### Phase 2: Normal Operation

- Optimal power generation
- Load balancing
- Continuous learning

#### Phase 3: Disturbance Recovery

- Handle substrate depletion
- Prevent cell reversal
- Maintain power output

#### Phase 4: pH Management

- pH buffer activation
- Acid neutralization
- Stability maintenance

#### Phase 5: Long-term Operation

- Acetate addition control
- Extended operation
- Resource optimization

## Performance Metrics

### Power Performance

- **Stack power**: 0.2-2.0 W (typical)
- **Power stability**: >95% (coefficient of variation)
- **Efficiency**: 60-80% (depends on load)

### Control Performance

- **Cell reversal prevention**: 100% success rate
- **Load balancing**: \<5% power variation between cells
- **Response time**: \<10 seconds for disturbance recovery

### Learning Performance

- **Convergence**: 200-500 episodes
- **Exploration decay**: ε = 0.3 → 0.01
- **Q-table size**: 50-100 states (discretized)

## Acceleration Benefits

### Mojo Implementation

The Mojo implementation provides:

1. **Vectorized Operations**: Parallel tensor computations
1. **Zero-cost Abstractions**: Memory-efficient data structures
1. **Cross-platform Acceleration**: GPU/NPU/ASIC compatibility
1. **Real-time Performance**: <1ms control loop execution
1. **Scalability**: Linear scaling with cell count

### Universal GPU Acceleration (New)

The Python implementations now feature universal GPU acceleration:

1. **Multi-vendor Support**: 
   - NVIDIA GPUs via CuPy
   - AMD GPUs via PyTorch with ROCm
   - Automatic backend detection
2. **CPU Fallback**: Seamless operation on systems without GPU
3. **Unified API**: Single interface for all mathematical operations
4. **Performance Benefits**:
   - Up to 10x speedup for large-scale simulations
   - Real-time control loop execution
   - Efficient memory management
5. **Tested Operations**:
   - Array operations (creation, conversion)
   - Mathematical functions (abs, log, exp, sqrt, power)
   - Conditional operations (where, maximum, minimum, clip)
   - Aggregations (mean, sum)
   - Random number generation

### EIS and QCM Sensor Integration (New)

The system now includes comprehensive biofilm sensing capabilities:

#### Electrochemical Impedance Spectroscopy (EIS)
- **Biofilm thickness measurement** (5-80 μm range)
- **Species-specific calibration** for G. sulfurreducens and S. oneidensis
- **Equivalent circuit modeling** with Randles circuit representation
- **Real-time biofilm conductivity monitoring**
- **Literature-validated parameters** from recent MFC studies

#### Quartz Crystal Microbalance (QCM)
- **Biofilm mass sensing** (0-1000 ng/cm² range)
- **Sauerbrey equation implementation** for rigid biofilms
- **Viscoelastic corrections** for soft biofilms
- **Multiple crystal types** (5 MHz and 10 MHz AT-cut)
- **Temperature compensation** and drift correction

#### Advanced Sensor Fusion
- **Multi-algorithm fusion**: Kalman filter, weighted average, maximum likelihood, Bayesian inference
- **Uncertainty quantification** with confidence intervals
- **Fault detection and tolerance** for sensor degradation
- **Real-time calibration** based on sensor agreement
- **Performance metrics**: sensor agreement, fusion confidence, measurement quality

#### Integration with Q-Learning Controller
- **Extended state space** with sensor measurements (EIS thickness, QCM mass, sensor quality)
- **Multi-objective rewards** incorporating biofilm health, sensor agreement, and system stability
- **Adaptive exploration** based on sensor confidence
- **Sensor-guided control decisions** for improved biofilm management

## Results and Analysis

### Simulation Results

- **Training time**: 0.65 seconds (1000 steps)
- **Final power**: 0.037-0.245 W
- **Reversed cells**: 0/5 (successful prevention)
- **Power stability**: 97.1%

### Individual Cell Performance

| Cell | Voltage (V) | Power (W) | pH  | Acetate |
|------|-------------|-----------|-----|---------|
| 0    | 0.178       | 0.010     | 8.1 | 1.545   |
| 1    | 0.173       | 0.014     | 8.0 | 1.584   |
| 2    | 0.204       | 0.020     | 8.0 | 1.512   |
| 3    | 0.197       | 0.014     | 7.9 | 1.569   |
| 4    | 0.195       | 0.017     | 8.2 | 1.622   |

### Control System Performance

- **Q-table size**: 62 states learned
- **Exploration rate**: 0.01 (converged)
- **Average reward**: Improved from -50 to -1.5

## Future Enhancements

1. **Deep Q-Learning**: Neural network-based Q-function
1. **Multi-objective Optimization**: Pareto-optimal solutions
1. **Predictive Control**: Model predictive control (MPC) integration
1. **Hardware Integration**: Real sensor/actuator interfaces
1. **Distributed Control**: Multi-stack coordination

## Testing

### Test Suite

The project includes a comprehensive test suite with:

- **Path Configuration Tests**: Verify output directory structure
- **File Output Tests**: Validate data saving functionality
- **Import Tests**: Check all modules import correctly
- **Execution Tests**: Test simulation execution
- **GPU Capability Tests**: Hardware detection for NVIDIA/AMD GPUs
- **GPU Acceleration Tests**: Functional tests for GPU operations

Run tests from the main project directory:

```bash
# Run all tests
python q-learning-mfcs/tests/run_tests.py

# Run specific test suites
python q-learning-mfcs/tests/run_tests.py -c gpu_capability
python q-learning-mfcs/tests/run_tests.py -c gpu_acceleration
```

## Files Description

### Core System
- `odes.mojo` - MFC electrochemical model
- `mfc_qlearning.mojo` - Q-learning controller (Mojo implementation)
- `mfc_stack_simulation.py` - Complete stack simulation
- `mfc_stack_demo.py` - Comprehensive demonstration
- `mfc_qlearning_demo.py` - Basic Q-learning demo
- `build_qlearning.py` - Build script

### Advanced Models
- `src/biofilm_kinetics/` - Biofilm formation and growth models
- `src/metabolic_model.py` - Metabolic pathway modeling
- `src/integrated_mfc_model.py` - Complete integrated MFC system
- `src/mfc_recirculation_control.py` - Recirculation and substrate control
- `src/sensing_enhanced_q_controller.py` - Sensor-enhanced Q-learning controller
- `src/sensor_integrated_mfc_model.py` - MFC model with sensor feedback loops

### Sensing Models
- `src/sensing_models/eis_model.py` - Electrochemical impedance spectroscopy
- `src/sensing_models/qcm_model.py` - Quartz crystal microbalance
- `src/sensing_models/sensor_fusion.py` - Multi-sensor data fusion
- `src/sensing_models/__init__.py` - Sensing models module

### Testing
- `tests/sensing_models/test_sensing_models.py` - Comprehensive sensor tests
- `tests/run_tests.py` - Test suite runner
- `README.md` - This documentation

## References

Based on recent research in Q-learning applications for microbial fuel cells, particularly:

- Machine learning solutions for enhanced performance in plant-based microbial fuel cells
- Q-learning based control for energy management systems
- Advanced control strategies for bioelectrochemical systems

## License

This project is for educational and research purposes.
