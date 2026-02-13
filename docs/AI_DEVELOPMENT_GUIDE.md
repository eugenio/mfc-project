# AI Development Guide for MFC Q-Learning Control System

*Last Updated: July 29, 2025*

*Last Updated: July 29, 2025*

## Executive Summary

This document provides comprehensive guidance for AI development agents working on the Microbial Fuel Cell (MFC) Q-Learning Control System project. The system implements advanced reinforcement learning for optimizing bioelectrochemical processes in a 5-cell MFC stack, featuring universal GPU acceleration (NVIDIA/AMD/CPU), sophisticated sensor integration, and literature-validated biological parameters.

**Current Version**: 2.2.0 (July 29, 2025)\
**Python Version**: 3.12.11+\
**Primary Dependencies**: JAX 0.6.0, JAXlib 0.6.0.dev20250728, Modular 25.4+, Streamlit 1.47+

## Project Overview

### Core Purpose

The MFC project is a research and development platform for optimizing microbial fuel cell performance through:

- Q-learning based control algorithms optimized for accelerator hardware
- Real-time sensor fusion (EIS, QCM) for biofilm monitoring
- GPU-accelerated simulations supporting both NVIDIA CUDA and AMD ROCm
- Literature-validated biological and electrochemical parameters

### Key Achievements

- **Power Optimization**: 132.5% improvement through literature validation
- **Control Performance**: 100% cell reversal prevention, 97.1% power stability
- **Computational Efficiency**: 0.65 seconds training time for 1000 steps
- **Biofilm Management**: Maximum sustainable thickness (3.0) achieved

## Architecture Overview

### System Components

```
mfc-project/
├── q-learning-mfcs/          # Core Q-learning MFC control system
│   ├── src/                 # Source code
│   │   ├── *.py            # Python implementations
│   │   ├── *.mojo          # High-performance Mojo implementations
│   │   ├── config/         # Configuration management system
│   │   ├── sensing_models/ # EIS, QCM, and sensor fusion
│   │   ├── biofilm_kinetics/ # Biofilm growth models
│   │   └── metabolic_model/  # Species-specific metabolism
│   ├── tests/              # Comprehensive test suite
│   ├── examples/           # Usage examples
│   └── data/              # Simulation outputs
├── conf-repo/             # Infrastructure configuration
├── docs/                  # Technical documentation
└── pixi.toml             # Dependency management
```

### Technology Stack

**Languages & Frameworks:**

- Python 3.12.11+ (primary implementation language with conda-forge support)
- Mojo 25.4+ (high-performance computing with ROCm/CUDA acceleration)
- JAX 0.6.0 + JAXlib 0.6.0.dev20250728 (universal GPU acceleration)
- PyTorch 2.4+ (fallback GPU acceleration and neural networks)
- Streamlit 1.47+ (web-based GUI interface)

**Key Dependencies:**

- NumPy 2.3.1+, SciPy 1.16.0+, Pandas 2.3.1+ (scientific computing)
- Matplotlib 3.10.3+, Seaborn 0.13.2+, Plotly 6.2.0+ (visualization)
- Optuna 4.4.0+ (hyperparameter optimization)
- pytest 8.4.1+ (testing framework)
- PyYAML 6.0.2+ (configuration management)

**Development Tools:**

- Pixi (conda-based dependency and environment management)
- Ruff 0.12.4+, MyPy 1.17.0+ (code quality and type checking)
- GitLab CI/CD with automated testing and security scanning
- Pandoc 3.7.0+ + Tectonic 0.15.0+ (documentation compilation)
- Git LFS 3.7.0+ (large file support for simulation data)

## Core Modules and Their Functions

### 1. MFC Model (`odes.mojo`, `integrated_mfc_model.py`)

**Purpose**: Electrochemical simulation of MFC behavior
**Key Features**:

- Monod kinetics with biofilm effects
- Multi-substrate support (acetate, lactate, pyruvate, glucose)
- Temperature and pH compensation
- Real-time mass balance calculations

### 2. Q-Learning Controller (`mfc_qlearning.mojo`, `sensing_enhanced_q_controller.py`)

**Purpose**: Adaptive control using reinforcement learning
**State Space** (40 dimensions):

- Per-cell features: substrate, biomass, oxygen, pH, voltage, power, reversal status
- Stack-level features: total voltage, current, power, reversal ratio, imbalance

**Action Space** (15 dimensions):

- Per-cell controls: duty cycle, pH buffer, substrate addition

**Reward Function**:

```python
reward = power_reward + stability_reward + reversal_penalty + efficiency_reward + action_penalty
```

### 3. Sensor Systems (`sensing_models/`)

**EIS (Electrochemical Impedance Spectroscopy)**:

- Biofilm thickness measurement (5-80 μm range)
- Equivalent circuit modeling
- Species-specific calibration

**QCM (Quartz Crystal Microbalance)**:

- Biofilm mass sensing (0-1000 ng/cm²)
- Sauerbrey equation implementation
- Viscoelastic corrections

**Sensor Fusion**:

- Multi-algorithm fusion (Kalman, weighted average, Bayesian)
- Uncertainty quantification
- Fault detection and tolerance

### 4. GPU Acceleration (`gpu_acceleration.py`)

**Universal GPU Support**:

- NVIDIA CUDA via CuPy
- AMD ROCm via PyTorch
- Automatic backend detection
- CPU fallback

**Optimized Operations**:

- Vectorized array operations
- Mathematical functions (exp, log, sqrt)
- Conditional operations (where, clip)
- Random number generation

### 5. Configuration System (`config/`)

**Biological Configuration**:

- Species-specific parameters (Geobacter, Shewanella)
- Substrate kinetics (Vmax, Km, Ki)
- Literature-referenced values

**Control Configuration**:

- PID tuning parameters
- Q-learning hyperparameters
- Flow control settings

**Profiles**:

- Conservative (stable, long-term operation)
- Research (aggressive optimization)
- Precision (high-accuracy laboratory use)

## Development Workflows

### 1. Running Simulations

**Basic Q-Learning Demo**:

```bash
cd q-learning-mfcs
python mfc_qlearning_demo.py
```

**Full Stack Simulation**:

```bash
python mfc_stack_simulation.py
```

**Comprehensive Simulation with Sensors**:

```bash
python run_comprehensive_simulation.py
```

### 2. Testing

**Run All Tests**:

```bash
python q-learning-mfcs/tests/run_tests.py
```

**Specific Test Categories**:

```bash
# GPU capability detection
python q-learning-mfcs/tests/run_tests.py -c gpu_capability

# GPU acceleration functionality
python q-learning-mfcs/tests/run_tests.py -c gpu_acceleration
```

### 3. Code Quality

**Before Committing**:

```bash
# Linting
ruff check .

# Type checking
mypy .

# YAML validation (for config files)
yamllint conf-repo/
```

### 4. Building Mojo Components

```bash
# Build the MFC model
mojo build odes.mojo --emit='shared-lib' -o odes.so

# Run build script
python build_qlearning.py
```

## Key Algorithms and Models

### 1. Biofilm Growth Model

```python
growth_rate = mu_max * (S / (Ks + S)) * (1 - X/X_max)
decay_rate = kd * X
net_growth = growth_rate - decay_rate
```

### 2. Electrochemical Model

```python
E_cell = E0 - (RT/nF) * ln(Q) - eta_activation - eta_concentration
I = n * F * A * k * C_substrate * (1 - exp(-alpha*n*F*eta/RT))
```

### 3. Q-Learning Update

```python
Q[s,a] = Q[s,a] + alpha * (reward + gamma * max(Q[s']) - Q[s,a])
```

### 4. Sensor Fusion (Kalman Filter)

```python
# Prediction
x_pred = F @ x + B @ u
P_pred = F @ P @ F.T + Q

# Update
K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)
x = x_pred + K @ (z - H @ x_pred)
P = (I - K @ H) @ P_pred
```

## Data Structures and Formats

### 1. Simulation Output CSV

```csv
time_hours,stack_power,biofilm_thickness,substrate_concentration,coulombic_efficiency
0.0,0.001,0.1,20.0,0.65
...
```

### 2. Configuration YAML

```yaml
biological:
  species: "geobacter"
  max_growth_rate: 0.05
  electron_transport_efficiency: 0.85
control:
  flow_control:
    kp: 0.5
    ki: 0.1
    kd: 0.05
```

### 3. Q-Table Storage (Pickle)

```python
{
    'q_table': np.array(...),
    'state_bins': {...},
    'action_space': [...],
    'metadata': {...}
}
```

## Performance Optimization Guidelines

### 1. GPU Utilization

- Use batch operations whenever possible
- Minimize CPU-GPU memory transfers
- Leverage `gpu_acceleration.py` unified interface

### 2. Memory Management

- Use generators for large datasets
- Implement checkpointing for long simulations
- Clear matplotlib figures after saving

### 3. Computational Efficiency

- Vectorize operations using NumPy/CuPy
- Use Mojo for performance-critical loops
- Profile code to identify bottlenecks

## Common Development Tasks

### 1. Adding a New Species

1. Create species config in `biological_config.py`
1. Add metabolic pathways in `metabolic_model/`
1. Update validation in `biological_validation.py`
1. Add example in `examples/`

### 2. Implementing New Control Strategy

1. Extend `AdvancedQLearningController`
1. Define new state/action spaces
1. Implement reward function
1. Add to `control_config.py`

### 3. Adding Sensor Type

1. Create model in `sensing_models/`
1. Add to sensor fusion in `sensor_fusion.py`
1. Update `SensorIntegratedMFCModel`
1. Write tests in `tests/sensing_models/`

## Debugging and Troubleshooting

### Common Issues

**GPU Not Detected**:

```python
# Check GPU availability
from gpu_acceleration import GPUAccelerator
gpu = GPUAccelerator()
print(gpu.get_device_info())
```

**Biofilm Starvation**:

- Check substrate concentration thresholds
- Verify flow rates and residence times
- Review substrate addition control logic

**Convergence Issues**:

- Adjust learning rate and exploration parameters
- Check reward function scaling
- Verify state space discretization

### Logging and Monitoring

**Enable Debug Logging**:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Monitor Simulation Progress**:

- Check `simulation.log` in output directory
- Review `simulation_progress.json` for real-time stats
- Use visualization tools for analysis

## Integration Points

### 1. GitLab CI/CD

- Automated testing on push
- Security scanning (SAST, secret detection)
- Performance benchmarking

### 2. MinIO Storage

- Large file handling via Git LFS
- S3-compatible API
- Configured in `.gitlab-ci.yml`

### 3. External APIs

- GitLab API for issue tracking
- Potential IoT sensor integration
- Cloud deployment readiness

## Best Practices

### 1. Code Style

- Follow PEP 8 for Python code
- Use type hints for clarity
- Document complex algorithms

### 2. Testing

- Write unit tests for new features
- Include integration tests
- Test GPU and CPU code paths

### 3. Documentation

- Update docstrings for API changes
- Keep configuration examples current
- Document performance characteristics

### 4. Version Control

- One file per commit (per CLAUDE.md)
- Descriptive commit messages
- Tag releases appropriately

## Future Development Opportunities

### 1. Advanced ML Integration

- Deep Q-Learning with neural networks
- Multi-agent reinforcement learning
- Transfer learning between species

### 2. System Extensions

- Multi-stack coordination
- Real-time hardware integration
- Cloud-based monitoring dashboard

### 3. Research Directions

- Novel sensor technologies
- Hybrid biological systems
- Economic optimization models

## Resources and References

### Key Documentation Files

- `q-learning-mfcs/README.md` - Detailed system overview
- `docs/MFC_SIMULATION_DOCUMENTATION.md` - Technical specifications
- `docs/GPU_ACCELERATION_GUIDE.md` - GPU programming guide
- `docs/LITERATURE_VALIDATION_ANALYSIS.md` - Parameter validation

### Literature References

- Lovley (2003) - Geobacter metabolism
- Marsili et al. (2008) - Shewanella electron transfer
- Torres et al. (2010) - Kinetic modeling
- Recent 2024-2025 MFC optimization studies

### Support Channels

- GitLab Issues for bug reports
- Project documentation in `/docs`
- Configuration examples in `/examples`

## Conclusion

This guide provides a comprehensive overview for AI agents working on the MFC project. The system represents cutting-edge research in bioelectrochemical systems control, combining biological modeling, machine learning, and high-performance computing. Follow the guidelines, leverage the existing infrastructure, and contribute to advancing sustainable energy technology.
