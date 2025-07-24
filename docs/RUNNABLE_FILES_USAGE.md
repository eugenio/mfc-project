# Runnable Files Usage Documentation

This document provides comprehensive usage instructions for all runnable files in the MFC simulation repository.

> **Note**: All Python simulation files now feature universal GPU acceleration with automatic backend detection (NVIDIA CUDA, AMD ROCm) and CPU fallback. See [GPU_ACCELERATION_GUIDE.md](GPU_ACCELERATION_GUIDE.md) for details.

## Table of Contents

1. [Overview](#overview)
1. [Mojo Simulation Files](#mojo-simulation-files)
1. [Python Analysis and Utilities](#python-analysis-and-utilities)
1. [Development and Testing Files](#development-and-testing-files)
1. [Quick Reference Guide](#quick-reference-guide)

______________________________________________________________________

## Overview

The repository contains multiple types of runnable files:

- **🔥 Mojo Simulations**: High-performance MFC simulations (`.mojo` files)
- **🐍 Python Analysis**: Data processing, visualization, and reports (`.py` files)
- **🧪 Development Tools**: Testing and development utilities

______________________________________________________________________

## Mojo Simulation Files

### 1. `q-learning-mfcs/src/mfc_100h_simple.mojo` ⭐ **BASELINE**

**Purpose**: Simple heuristic-based 100-hour MFC simulation
**Performance**: 2.75 Wh in ~1.6 seconds
**Status**: ✅ Production-ready baseline

**Usage**:

```bash
# Run the simulation
mojo run q-learning-mfcs/src/mfc_100h_simple.mojo

# Format before running
mojo format q-learning-mfcs/src/mfc_100h_simple.mojo
mojo run q-learning-mfcs/src/mfc_100h_simple.mojo
```

**Output Example**:

```
=== Simple 100-Hour MFC Simulation ===
Simulating 5 cells for 100 hours
...
Final total energy: 2.75 Wh
Average power: 0.028 W
Simulation completed successfully
```

**When to Use**:

- Quick baseline testing
- Performance comparison reference
- Educational demonstrations
- Fast prototyping

______________________________________________________________________

### 2. `q-learning-mfcs/src/mfc_100h_qlearn.mojo`

**Purpose**: Basic Q-learning MFC simulation with reinforcement learning
**Performance**: 3.64 Wh in ~2.1 seconds (+32.3% vs simple)
**Status**: ✅ Production-ready

**Usage**:

```bash
# Run basic Q-learning simulation
mojo run q-learning-mfcs/src/mfc_100h_qlearn.mojo
```

**Configuration**:

- Learning rate: α = 0.1
- Epsilon decay: 0.995
- State space: 10^6 states
- Action space: 10^3 actions

**Output Example**:

```
=== Q-Learning 100-Hour MFC Simulation ===
Q-learning parameters: ε=0.25, α=0.1
...
Total energy produced: 3.64 Wh
Q-table size: 1,247 entries
Final epsilon: 0.05
```

______________________________________________________________________

### 3. `q-learning-mfcs/src/mfc_100h_enhanced.mojo` 🏆 **RECOMMENDED**

**Purpose**: Enhanced Q-learning with advanced optimization features
**Performance**: 127.65 Wh in ~87.6 seconds (+4538.9% vs simple)
**Status**: ✅ Production-ready, best performance

**Usage**:

```bash
# Run enhanced Q-learning simulation
mojo run q-learning-mfcs/src/mfc_100h_enhanced.mojo

# Monitor progress (takes ~1.5 minutes)
timeout 300 mojo run q-learning-mfcs/src/mfc_100h_enhanced.mojo
```

**Key Features**:

- Dynamic Q-table growth (6,532+ learned entries)
- Multi-objective reward function
- Stack coordination optimization
- Epsilon decay: 0.3 → 0.18
- Advanced biofilm and aging models

**Output Example**:

```
=== Enhanced Q-Learning 100-Hour MFC Simulation ===
Matching Python simulation characteristics
...
Total energy produced: 127.65 Wh
Average power: 1.242 W
Python target achievement: 134.4%
Stack coordination achieved
```

**Performance Metrics**:

- Energy efficiency: 28.4%
- Power density: 0.25 W/cell
- Resource efficiency: 87% substrate remaining
- Learning progress: 6,532 state-action pairs

______________________________________________________________________

### 4. `q-learning-mfcs/src/mfc_100h_advanced.mojo` ⚠️ **RESEARCH**

**Purpose**: High-complexity Q-learning with full state space exploration
**Performance**: Times out >900 seconds (too complex for practical use)
**Status**: 🔬 Research prototype

**Usage**:

```bash
# WARNING: This will likely timeout
timeout 1800 mojo run q-learning-mfcs/src/mfc_100h_advanced.mojo

# For research purposes only - not recommended for production
```

**Configuration**:

- State bins: 15^6 = 11,390,625 possible states
- Action bins: 12^3 = 1,728 actions
- Full temporal difference learning
- Comprehensive 5-component reward function

**Computational Requirements**:

- Memory: >50MB for Q-table
- Time complexity: O(n^7) per step
- Suitable for: Academic research, algorithm validation

______________________________________________________________________

### 5. `q-learning-mfcs/src/mfc_100h_gpu_optimized.mojo` 🚀 **EXPERIMENTAL**

**Purpose**: Tensor-based GPU acceleration experiment
**Performance**: Designed for 10-30x speedup (experimental)
**Status**: 🧪 Experimental

**Usage**:

```bash
# GPU-optimized simulation (requires compatible hardware)
mojo run q-learning-mfcs/src/mfc_100h_gpu_optimized.mojo

# Check GPU availability first
nvidia-smi  # For NVIDIA GPUs
```

**Features**:

- GPU tensor operations
- Vectorized processing
- Pre-allocated Q-table (50,000 entries)
- Batch processing (64 operations)
- Reduced complexity: 10^6 states

**Requirements**:

- Compatible GPU hardware
- Mojo GPU support
- Sufficient GPU memory (>1GB)

______________________________________________________________________

### 6. `q-learning-mfcs/src/mfc_100h_gpu.mojo` ❌ **DEPRECATED**

**Purpose**: Original GPU acceleration attempt
**Status**: 💀 Deprecated (compatibility issues)

**Issues**:

- Incompatible tensor imports
- API version conflicts
- Recursive type definitions

**Do Not Use**: This file has known issues and should not be executed.

______________________________________________________________________

### 7. Development Mojo Files

#### `q-learning-mfcs/src/mfc_qlearning.mojo`

**Purpose**: Core Q-learning implementation prototype
**Usage**: `mojo run q-learning-mfcs/src/mfc_qlearning.mojo`
**Status**: Research prototype

#### `q-learning-mfcs/src/odes.mojo`

**Purpose**: ODE solver utilities for electrochemical kinetics
**Usage**: Library file (not directly runnable)
**Status**: Utility library

#### `q-learning-mfcs/src/qlearning_bindings.mojo`

**Purpose**: Q-learning algorithm bindings
**Usage**: Library file (not directly runnable)\
**Status**: Utility library

#### `q-learning-mfcs/src/test.mojo`

**Purpose**: Basic testing framework
**Usage**: `mojo run q-learning-mfcs/src/test.mojo`
**Status**: Development utility

______________________________________________________________________

## Python Analysis and Utilities

### 1. `q-learning-mfcs/src/run_gpu_simulation.py` 🏆 **BENCHMARK RUNNER**

**Purpose**: Comprehensive benchmark runner for all Mojo simulations
**Usage**:

```bash
# Run comprehensive benchmark comparison
python3 q-learning-mfcs/src/run_gpu_simulation.py

# With pixi environment
pixi run python q-learning-mfcs/src/run_gpu_simulation.py
```

**Features**:

- Parallel simulation execution
- Performance comparison
- Automatic result analysis
- Visualization generation
- Detailed benchmarking reports

**Output Files**:

- `mfc_simulation_comparison.png` - Performance comparison plots
- Console output with detailed analysis

**Example Output**:

```
=== MFC 100-Hour Simulation Comprehensive Benchmark ===
Running all Mojo implementations with parallel execution

✓ Simple 100h MFC completed in 1.6s - Energy: 2.75 Wh
✓ Q-Learning MFC completed in 2.1s - Energy: 3.64 Wh
✓ Enhanced Q-Learning MFC completed in 87.6s - Energy: 127.65 Wh

🏆 Best Performer: Enhanced Q-Learning MFC (134.4% of Python target)
```

______________________________________________________________________

### 2. Report Generation Scripts

#### `q-learning-mfcs/src/generate_pdf_report.py`

**Purpose**: Generate PDF analysis reports
**Usage**:

```bash
cd q-learning-mfcs/src
python3 generate_pdf_report.py

# Requires: matplotlib, reportlab, numpy
```

#### `q-learning-mfcs/src/generate_enhanced_pdf_report.py`

**Purpose**: Generate enhanced PDF reports with advanced analysis
**Usage**:

```bash
cd q-learning-mfcs/src
python3 generate_enhanced_pdf_report.py
```

#### `q-learning-mfcs/src/generate_performance_graphs.py`

**Purpose**: Generate performance visualization graphs
**Usage**:

```bash
cd q-learning-mfcs/src
python3 generate_performance_graphs.py
```

______________________________________________________________________

### 3. Analysis and Modeling Scripts

#### `q-learning-mfcs/src/energy_sustainability_analysis.py`

**Purpose**: Energy sustainability and efficiency analysis
**Usage**:

```bash
cd q-learning-mfcs/src
python3 energy_sustainability_analysis.py
```

#### `q-learning-mfcs/src/create_summary_plots.py`

**Purpose**: Create summary visualization plots
**Usage**:

```bash
cd q-learning-mfcs/src
python3 create_summary_plots.py
```

#### `q-learning-mfcs/src/stack_physical_specs.py`

**Purpose**: Physical specifications and modeling for MFC stacks
**Usage**:

```bash
cd q-learning-mfcs/src
python3 stack_physical_specs.py
```

______________________________________________________________________

### 4. Simulation and Demo Scripts

#### `q-learning-mfcs/src/mfc_100h_simulation.py`

**Purpose**: Python reference implementation of 100h simulation
**Usage**:

```bash
cd q-learning-mfcs/src
python3 mfc_100h_simulation.py
```

#### `q-learning-mfcs/src/mfc_stack_simulation.py`

**Purpose**: MFC stack simulation in Python
**Usage**:

```bash
cd q-learning-mfcs/src
python3 mfc_stack_simulation.py
```

#### `q-learning-mfcs/src/mfc_qlearning_demo.py`

**Purpose**: Q-learning demonstration and tutorial
**Usage**:

```bash
cd q-learning-mfcs/src
python3 mfc_qlearning_demo.py
```

#### `q-learning-mfcs/src/mfc_stack_demo.py`

**Purpose**: MFC stack demonstration
**Usage**:

```bash
cd q-learning-mfcs/src
python3 mfc_stack_demo.py
```

______________________________________________________________________

### 5. Model and Build Scripts

#### `q-learning-mfcs/src/mfc_model.py`

**Purpose**: Core MFC modeling functions and utilities
**Usage**: Import as library module

```python
from mfc_model import MFCModel
```

#### `q-learning-mfcs/src/build_qlearning.py`

**Purpose**: Build and compile Q-learning components
**Usage**:

```bash
cd q-learning-mfcs/src
python3 build_qlearning.py
```

______________________________________________________________________

### 6. External Analysis

#### `q-learning-mfcs/src/analyze_pdf_comments.py`

**Purpose**: Analyze comments and annotations in PDF reports
**Usage**:

```bash
python3 q-learning-mfcs/src/analyze_pdf_comments.py [pdf_file]
```

#### `q-learning-mfcs/src/test.py`

**Purpose**: Python testing framework
**Usage**:

```bash
cd q-learning-mfcs/src
python3 test.py
```

______________________________________________________________________

## Development and Testing Files

### Testing Framework

- `q-learning-mfcs/src/test.mojo` - Mojo testing utilities
- `q-learning-mfcs/src/test.py` - Python testing framework

### Build and Configuration

- `pixi.toml` - Package management configuration
- `pixi.lock` - Lock file for reproducible environments

______________________________________________________________________

## Quick Reference Guide

### 🚀 **For Quick Testing**

```bash
# Fastest baseline (1.6s)
mojo run q-learning-mfcs/src/mfc_100h_simple.mojo
```

### 🏆 **For Best Performance**

```bash
# Recommended enhanced version (87.6s)
mojo run q-learning-mfcs/src/mfc_100h_enhanced.mojo
```

### 📊 **For Comprehensive Analysis**

```bash
# Full benchmark comparison
python3 q-learning-mfcs/src/run_gpu_simulation.py
```

### 🔬 **For Research**

```bash
# Advanced Q-learning (warning: very slow)
timeout 1800 mojo run q-learning-mfcs/src/mfc_100h_advanced.mojo
```

### 🚀 **For GPU Acceleration (Experimental)**

```bash
# GPU-optimized version
mojo run q-learning-mfcs/src/mfc_100h_gpu_optimized.mojo
```

______________________________________________________________________

## Performance Summary

| File | Runtime | Energy Output | Use Case |
|------|---------|---------------|----------|
| `mfc_100h_simple.mojo` | 1.6s | 2.75 Wh | Quick testing |
| `mfc_100h_qlearn.mojo` | 2.1s | 3.64 Wh | Basic Q-learning |
| `mfc_100h_enhanced.mojo` 🏆 | 87.6s | **127.65 Wh** | **Production** |
| `mfc_100h_advanced.mojo` | >900s | Timeout | Research |
| `mfc_100h_gpu_optimized.mojo` | ~10-30s\* | ~100-150 Wh\* | Experimental |

\*GPU performance estimates

______________________________________________________________________

## Environment Setup

### Prerequisites

```bash
# Install Mojo (follow official instructions)
curl https://get.modular.com | sh -
modular install mojo

# Install Python dependencies with pixi
pixi install

# Or with pip
pip install matplotlib numpy reportlab
```

### Running Simulations

```bash
# Format Mojo files before running
mojo format q-learning-mfcs/src/*.mojo

# Run individual simulations
mojo run q-learning-mfcs/src/[simulation_file].mojo

# Run Python analysis
python3 q-learning-mfcs/src/[script_name].py
```

______________________________________________________________________

## Troubleshooting

### Common Issues

1. **Mojo Compilation Errors**:

   ```bash
   mojo format [file].mojo  # Format first
   mojo run [file].mojo     # Then run
   ```

1. **Python Dependencies**:

   ```bash
   pixi install            # Install with pixi
   # or
   pip install -r requirements.txt
   ```

1. **GPU Issues**:

   ```bash
   nvidia-smi              # Check GPU availability
   # Fallback to CPU versions if needed
   ```

1. **Performance Issues**:

   - Use `mfc_100h_simple.mojo` for quick tests
   - Avoid `mfc_100h_advanced.mojo` for production
   - Use `timeout` command for long-running simulations

______________________________________________________________________

## File Maintenance

- **Last Updated**: 2025-07-21
- **Version**: 1.0
- **Maintainer**: MFC Simulation Project

______________________________________________________________________

*This documentation covers all runnable files in the repository. For detailed implementation information, see `MFC_SIMULATION_DOCUMENTATION.md`.*
