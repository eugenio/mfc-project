# MFC Project

A comprehensive microbial fuel cell (MFC) research and development project featuring Q-learning control systems, advanced simulation capabilities, and infrastructure automation.

## Overview

This project implements a complete ecosystem for MFC research, including:

- **Q-Learning Control System**: Advanced reinforcement learning for real-time MFC optimization
- **High-Performance Computing**: Mojo-accelerated simulations with GPU/NPU/ASIC support
- **Infrastructure Automation**: Ansible-based deployment and configuration management
- **Comprehensive Analysis**: Multiple simulation models, performance analysis, and validation tools

## Project Status

### Q-Learning MFC Control System

**Status**: ✅ Complete and functional

The `q-learning-mfcs/` directory contains a fully implemented Q-learning control system for a 5-cell MFC stack featuring:

- **Advanced Control**: Q-learning algorithm optimized for accelerator hardware (GPU/NPU/ASIC)
- **Complete Simulation**: 5-cell MFC stack with realistic electrochemical dynamics
- **Advanced Sensor Integration**: EIS and QCM biofilm sensing, voltage, current, pH, and substrate concentration monitoring
- **Actuator Control**: PWM duty cycle, pH buffer, and acetate addition systems
- **Performance**: 97.1% power stability, 100% cell reversal prevention
- **Documentation**: Comprehensive LaTeX reports with analysis and figures

**Key Achievements**:

- Training time: 0.65 seconds (1000 steps)
- Power output: 0.037-0.245 W
- Zero cell reversals during operation
- Complete sensor/actuator simulation with realistic noise and dynamics

For detailed technical information, implementation details, and usage instructions, see the [complete documentation](q-learning-mfcs/README.md).

## Development Environment

This project uses [Pixi](https://pixi.sh) for dependency management and development environment setup:

```bash
# Install dependencies
pixi install

# Install development tools
pixi install -e dev

# Run security checks
pixi run -e dev bandit -r . --exclude ./.pixi
pixi run -e dev detect-secrets scan --exclude-files '.pixi/.*'

# Format markdown files
pixi run -e dev mdformat .

# Run linting and type checking
ruff check .
mypy .
```

### Prerequisites

- Python 3.8+
- Mojo SDK (for high-performance simulations)
- Ansible (for infrastructure deployment)
- GPU support (optional, for accelerated simulations)
  - NVIDIA CUDA (via CuPy) or AMD ROCm (via PyTorch)
  - Automatic CPU fallback if no GPU available

## Project Structure

```text
├── q-learning-mfcs/          # Q-learning MFC control system
│   ├── src/                 # Source code for simulations and control
│   │   ├── *.py            # Python implementations
│   │   └── *.mojo          # High-performance Mojo implementations
│   ├── data/               # Simulation outputs and datasets
│   │   └── figures/        # Generated analysis graphs
│   ├── reports/            # LaTeX research reports and analysis
│   ├── simulation_data/    # Raw simulation results (CSV/JSON)
│   ├── q_learning_models/  # Trained Q-learning models (.pkl)
│   └── bibliography/       # Research papers and references
├── conf-repo/               # Infrastructure configuration
│   ├── ansible-minio-setup.yml  # MinIO deployment playbook
│   ├── inventory.yml       # Ansible inventory
│   └── templates/          # Configuration templates
├── docs/                    # Project documentation
│   ├── MFC_SIMULATION_DOCUMENTATION.md
│   ├── LITERATURE_VALIDATION_ANALYSIS.md
│   └── ...                 # Additional documentation files
├── logs/                    # Application and session logs
├── pixi.toml               # Dependency configuration
├── pixi.lock               # Locked dependencies
└── README.md               # This file
```

## Key Components

### Q-Learning Control System (`q-learning-mfcs/src/`)

- **Mojo Implementations**: High-performance simulations leveraging GPU acceleration
  - `mfc_100h_gpu.mojo`, `mfc_100h_gpu_optimized.mojo`: GPU-accelerated 100-hour simulations
  - `mfc_qlearning.mojo`: Core Q-learning algorithm implementation
  - `odes.mojo`: Differential equation solvers

- **Python Implementations**: Analysis and visualization tools
  - `mfc_unified_qlearning_control.py`: Unified control system with universal GPU acceleration
  - `mfc_qlearning_optimization.py`: Q-learning flow control with GPU support
  - `mfc_dynamic_substrate_control.py`: Dynamic substrate control with GPU acceleration
  - `mfc_optimization_gpu.py`: Multi-objective optimization with GPU support
  - `energy_sustainability_analysis.py`: Energy efficiency analysis
  - `literature_validation_comparison_plots.py`: Validation against published data

- **GPU Acceleration Module**: Universal GPU support
  - `gpu_acceleration.py`: Unified interface for NVIDIA CUDA and AMD ROCm
  - Automatic CPU fallback for systems without GPU
  - Comprehensive mathematical operations with hardware acceleration

- **Stack Simulation**: Multi-cell MFC stack modeling
  - `mfc_stack_simulation.py`: Complete 5-cell stack simulation
  - `stack_physical_specs.py`: Physical specifications and parameters

### Infrastructure Automation (`conf-repo/`)

- **Ansible Playbooks**: Automated deployment and configuration
- **MinIO Setup**: Object storage for simulation data
- **Monitoring Scripts**: Health checks and backup automation

### Documentation (`docs/`)

- Comprehensive simulation documentation
- Literature validation and analysis reports
- Energy balance and sustainability studies
- EIS-QCM biofilm correlation models

## Features

### Simulation Capabilities

- **Multi-timescale Simulations**: 100-hour and 1000-hour continuous operation
- **Universal GPU Acceleration**: 
  - Support for both NVIDIA CUDA and AMD ROCm
  - Automatic backend detection and selection
  - CPU fallback for systems without GPU
  - Unified API for all mathematical operations
- **Comprehensive Modeling**: Electrochemical, biofilm, and substrate dynamics
- **Noise Modeling**: Realistic sensor noise and environmental variations

### Control Strategies

- **Q-Learning Optimization**: Reinforcement learning for optimal control
- **Dynamic Substrate Control**: Adaptive feeding strategies
- **Flow Rate Optimization**: Hydraulic retention time optimization
- **Cell Reversal Prevention**: Active monitoring and intervention

### Analysis Tools

- **Performance Visualization**: Comprehensive plotting and analysis
- **Energy Sustainability**: Lifecycle analysis and efficiency metrics
- **Literature Validation**: Comparison with published experimental data
- **Economic Analysis**: Cost-benefit and ROI calculations

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mfc-project
   ```

2. **Install dependencies**:
   ```bash
   pixi install
   ```

3. **Run a simulation**:
   ```bash
   cd q-learning-mfcs/src
   python mfc_unified_qlearning_control.py
   ```

4. **Generate analysis plots**:
   ```bash
   python generate_all_figures.py
   ```

5. **Run tests** (including GPU capability tests):
   ```bash
   python q-learning-mfcs/tests/run_tests.py
   # Or run specific test suites:
   python q-learning-mfcs/tests/run_tests.py -c gpu_capability
   python q-learning-mfcs/tests/run_tests.py -c gpu_acceleration
   ```

## Contributing

Contributions are welcome! Please ensure:

- Code passes `ruff` and `mypy` checks
- Mojo files are formatted with `mojo format`
- Tests are included for new functionality
- Documentation is updated accordingly

## License

This project is for educational and research purposes.
