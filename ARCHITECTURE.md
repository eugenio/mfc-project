# MFC Q-Learning Control System - Architecture Documentation

## Overview

The MFC (Microbial Fuel Cell) Q-Learning Control System is a sophisticated research platform that combines biological modeling, machine learning, and high-performance computing to optimize bioelectrochemical processes. The system features a modular architecture designed for scalability, extensibility, and performance.

This document serves as the single source of truth for the system architecture.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Architecture Layers](#architecture-layers)
3. [Component Interaction](#component-interaction)
4. [Data Flow](#data-flow)
5. [Directory Structure](#directory-structure)
6. [Technology Stack](#technology-stack)
7. [Configuration System](#configuration-system)
8. [Simulation Engine](#simulation-engine)
9. [Performance Characteristics](#performance-characteristics)
10. [Development Workflow](#development-workflow)
11. [Integration Points](#integration-points)
12. [Troubleshooting](#troubleshooting)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MFC Control System                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   Q-Learning    │    │  Physical Model  │                │
│  │   Controller    │◄───│   (MFC Stack)    │                │
│  │                 │    │                  │                │
│  │ • State: 40D    │    │ • 5 cells       │                │
│  │ • Action: 15D   │    │ • Electrochemical│                │
│  │ • Multi-objective│    │ • Biofilm       │                │
│  └────────┬────────┘    └────────▲────────┘                │
│           │                       │                         │
│           ▼                       │                         │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  Sensor Fusion  │    │   Actuators     │                │
│  │                 │    │                 │                │
│  │ • EIS sensors   │    │ • Flow pumps    │                │
│  │ • QCM sensors   │    │ • pH buffer     │                │
│  │ • Kalman filter │    │ • Substrate add │                │
│  └─────────────────┘    └─────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture Layers

### 1. Hardware Abstraction Layer

```
┌─────────────────────────────────────────────────────────────┐
│                 HARDWARE ABSTRACTION LAYER                 │
├─────────────────────────────────────────────────────────────┤
│ GPU Acceleration    │ CPU Processing    │ Storage Systems   │
│ ================    │ ==============    │ ===============   │
│ • NVIDIA CUDA       │ • Multi-core      │ • MinIO S3        │
│ • AMD ROCm          │ • SIMD            │ • GitLab LFS      │
│ • CPU fallback      │ • Threading       │ • Local files    │
└─────────────────────────────────────────────────────────────┘
```

### 2. Computational Core

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPUTATIONAL CORE                      │
├─────────────────────────────────────────────────────────────┤
│ Mojo Engine         │ Python Runtime    │ GPU Acceleration  │
│ ===========         │ ==============    │ ================  │
│ • odes.mojo         │ • NumPy/SciPy     │ • CuPy/PyTorch    │
│ • mfc_qlearning.mojo│ • Pandas          │ • JAX             │
│ • High-perf loops   │ • Matplotlib      │ • Auto backend    │
└─────────────────────────────────────────────────────────────┘
```

### 3. Control and Intelligence Layer

```
┌─────────────────────────────────────────────────────────────┐
│               CONTROL & INTELLIGENCE LAYER                 │
├─────────────────────────────────────────────────────────────┤
│ Q-Learning Controller │ Sensor Fusion   │ Config Management │
│ ==================== │ =============   │ ================= │
│ • State space (40D)   │ • EIS sensors   │ • Profile system  │
│ • Action space (15D)  │ • QCM sensors   │ • Validation      │
│ • Multi-objective     │ • Kalman filter │ • Inheritance     │
│ • ε-greedy policy     │ • Fault tolerance│ • YAML/JSON      │
└─────────────────────────────────────────────────────────────┘
```

### 4. Biological Modeling Layer

```
┌─────────────────────────────────────────────────────────────┐
│                 BIOLOGICAL MODELING LAYER                  │
├─────────────────────────────────────────────────────────────┤
│ Species Models      │ Substrate Models  │ Biofilm Dynamics │
│ ==============      │ ===============   │ ================= │
│ • Geobacter        │ • Acetate         │ • Growth kinetics │
│ • Shewanella       │ • Lactate         │ • Decay processes │
│ • Mixed cultures   │ • Pyruvate        │ • Thickness model │
│ • Literature refs  │ • Glucose         │ • Mass transfer   │
└─────────────────────────────────────────────────────────────┘
```

### 5. Physical Simulation Layer

```
┌─────────────────────────────────────────────────────────────┐
│                PHYSICAL SIMULATION LAYER                   │
├─────────────────────────────────────────────────────────────┤
│ Electrochemical     │ Flow Dynamics     │ Mass Balance      │
│ ===============     │ =============     │ ============      │
│ • Nernst equation   │ • Pump models     │ • Substrate       │
│ • Butler-Volmer     │ • Recirculation   │ • Biomass         │
│ • Multi-cell stack  │ • Residence time  │ • Products        │
│ • Cell reversal     │ • Mixing          │ • Conservation    │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Interaction

```
                         ┌─────────────────────────────────────┐
                         │        Configuration Manager        │
                         │  ┌─────────────────────────────────┐│
                         │  │ • Profile Management           ││
                         │  │ • Validation & Type Checking   ││
                         │  │ • Environment Variable Subst.  ││
                         │  │ • Inheritance & Overrides      ││
                         │  └─────────────────────────────────┘│
                         └─────┬───────────────────────────────┘
                               │ Configuration
                               ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   EIS Sensor    │    │   QCM Sensor    │    │  Standard       │
│   Model         │    │   Model         │    │  Sensors        │
│                 │    │                 │    │                 │
│ • Thickness     │    │ • Mass          │    │ • Voltage       │
│ • Conductivity  │    │ • Frequency     │    │ • Current       │
│ • Calibration   │    │ • Viscoelastic  │    │ • pH            │
│ • Noise model   │    │ • Temperature   │    │ • Substrate     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │ Sensor Data
                                 ▼
                   ┌─────────────────────────────────────┐
                   │        Sensor Fusion Engine        │
                   │  ┌─────────────────────────────────┐│
                   │  │ • Kalman Filter                ││
                   │  │ • Weighted Average             ││
                   │  │ • Maximum Likelihood           ││
                   │  │ • Bayesian Inference           ││
                   │  │ • Uncertainty Quantification   ││
                   │  │ • Fault Detection              ││
                   │  └─────────────────────────────────┘│
                   └─────┬───────────────────────────────┘
                         │ Fused State
                         ▼
         ┌─────────────────────────────────────────────────────────┐
         │              Q-Learning Controller                      │
         │  ┌─────────────────────────────────────────────────────┐│
         │  │ State Space (40D):                                 ││
         │  │ • Per-cell: substrate, biomass, O2, pH, V, P, rev  ││
         │  │ • Stack: voltage, current, power, reversal ratio   ││
         │  │                                                    ││
         │  │ Action Space (15D):                                ││
         │  │ • Per-cell: duty cycle, pH buffer, substrate add  ││
         │  │                                                    ││
         │  │ Reward Function:                                   ││
         │  │ • Power optimization                               ││
         │  │ • Stability maintenance                            ││
         │  │ • Reversal prevention                              ││
         │  │ • Multi-objective balancing                        ││
         │  └─────────────────────────────────────────────────────┘│
         └─────┬───────────────────────────────────────────────────┘
               │ Control Actions
               ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Flow Pumps    │    │   pH Buffer     │    │   Substrate     │
│                 │    │   System        │    │   Addition      │
│ • PWM control   │    │                 │    │                 │
│ • Flow rate     │    │ • Buffer pumps  │    │ • Concentration │
│ • Response time │    │ • pH monitoring │    │ • Feed rate     │
│ • Dynamics      │    │ • Neutralization│    │ • Timing        │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │ Physical Actions
                                 ▼
                   ┌─────────────────────────────────────┐
                   │           MFC Physical Model        │
                   │  ┌─────────────────────────────────┐│
                   │  │ Biological Layer:              ││
                   │  │ • Species metabolism           ││
                   │  │ • Biofilm dynamics             ││
                   │  │ • Substrate kinetics           ││
                   │  │                                ││
                   │  │ Electrochemical Layer:         ││
                   │  │ • Butler-Volmer kinetics       ││
                   │  │ • Nernst potential             ││
                   │  │ • Ohmic losses                 ││
                   │  │                                ││
                   │  │ Transport Layer:               ││
                   │  │ • Mass transfer                ││
                   │  │ • Flow dynamics                ││
                   │  │ • Mixing effects               ││
                   │  └─────────────────────────────────┘│
                   └─────┬───────────────────────────────┘
                         │ System State Update
                         ▼
                   ┌─────────────────────────────────────┐
                   │          Data & Analytics           │
                   │  ┌─────────────────────────────────┐│
                   │  │ • Real-time logging            ││
                   │  │ • Performance metrics          ││
                   │  │ • Visualization                ││
                   │  │ • Data export (CSV/JSON)       ││
                   │  │ • Statistical analysis         ││
                   │  │ • Report generation            ││
                   │  └─────────────────────────────────┘│
                   └─────────────────────────────────────┘
```

---

## Data Flow

### Primary Data Pathways

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Sensors   │───▶│ Sensor Fusion│───▶│ Q-Learning  │───▶│   Actuators  │
│             │    │              │    │ Controller  │    │              │
│ • EIS       │    │ • Kalman     │    │             │    │ • Flow pumps │
│ • QCM       │    │ • Weighted   │    │ • State     │    │ • pH buffer  │
│ • Voltage   │    │ • Bayesian   │    │ • Action    │    │ • Substrate  │
│ • Current   │    │ • ML         │    │ • Reward    │    │   addition   │
│ • pH        │    │              │    │ • Policy    │    │              │
│ • Substrate │    │              │    │             │    │              │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

### Configuration Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Configuration  │───▶│  Validation &   │───▶│   Runtime       │
│  Files          │    │  Processing     │    │   System        │
│                 │    │                 │    │                 │
│ • YAML profiles │    │ • Schema check  │    │ • Live config   │
│ • JSON configs  │    │ • Type safety   │    │ • Hot reload    │
│ • Environment   │    │ • Inheritance   │    │ • Override      │
│   variables     │    │ • Defaults      │    │   capability    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## Directory Structure

```
mfc-project/
├── .claude/                      # Claude Code automation
│   ├── hooks/                   # Development hooks
│   │   ├── enhanced_file_chunking.py  # Smart chunking
│   │   ├── pre_tool_use.py     # Tool monitoring
│   │   ├── gitlab_issue_manager.py    # Issue automation
│   │   └── send_event.py       # Event logging
│   ├── settings.json           # Hook configuration
│   └── CLAUDE.md              # User instructions
│
├── q-learning-mfcs/            # Core Q-learning system
│   ├── src/                   # Source code
│   │   ├── config/           # Configuration system
│   │   │   ├── qlearning_config.py    # Q-learning parameters
│   │   │   ├── sensor_config.py       # Sensor settings
│   │   │   ├── biological_validation.py # Parameter validation
│   │   │   └── config_io.py           # Config serialization
│   │   ├── sensing_models/   # Sensor models
│   │   │   ├── advanced_sensor_fusion.py  # Fusion algorithms
│   │   │   ├── eis_sensor.py  # EIS implementation
│   │   │   └── qcm_sensor.py  # QCM implementation
│   │   ├── monitoring/       # Dashboard and monitoring
│   │   │   ├── dashboard_api.py  # REST API backend
│   │   │   └── ssl_config.py     # SSL configuration
│   │   ├── gui/              # Streamlit GUI
│   │   │   ├── pages/        # GUI page modules
│   │   │   └── plots/        # Visualization modules
│   │   ├── utils/            # Utilities
│   │   │   └── gitlab_issue_manager.py  # GitLab integration
│   │   ├── base_controller.py    # Controller base classes
│   │   ├── run_simulation.py     # Unified simulation CLI
│   │   └── *.py/*.mojo          # Core implementations
│   ├── tests/                # Test suite
│   └── README.md             # Detailed documentation
│
├── scripts/                  # Utility scripts
│   ├── detect_gpu.py        # GPU detection
│   ├── install_gpu_deps.py  # Dependency installer
│   └── ralph/               # Autonomous agent workflow
│
├── pixi.toml                # Dependency management
├── setup_environment.sh     # Environment setup
├── ARCHITECTURE.md         # This file
└── README.md               # Project overview
```

---

## Technology Stack

### Languages & Frameworks
- **Python 3.12+**: Primary implementation language
- **Mojo**: High-performance numerical computations
- **JAX/CuPy/PyTorch**: GPU acceleration backends
- **Streamlit**: Web-based GUI interface

### Development Tools
- **Pixi**: Dependency and environment management
- **Claude Code**: AI-assisted development
- **GitLab**: Version control and CI/CD
- **Ruff/MyPy**: Code quality and type checking

### Scientific Libraries
- **NumPy/SciPy**: Numerical computations
- **Pandas**: Data analysis
- **Matplotlib/Plotly**: Visualization
- **Optuna**: Hyperparameter optimization

---

## Configuration System

### Hierarchical Configuration

```yaml
# Example: research_optimization.yaml
biological:
  species: geobacter
  max_growth_rate: 0.46
  half_saturation: 2.5

control:
  q_learning:
    learning_rate: 0.1
    discount_factor: 0.95
    epsilon: 0.1

visualization:
  plot_interval: 10
  save_format: png
```

### Profile System
- **Conservative**: Stable long-term operation
- **Research**: Aggressive optimization
- **Precision**: High-accuracy laboratory use

### Configuration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Configuration Architecture                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐                              ┌───────────────┐ │
│  │  YAML Files     │                              │ Environment   │ │
│  │                 │                              │ Variables     │ │
│  │ • Profile configs│         ┌─────────────┐     │               │ │
│  │ • Species params │────────▶│ Config      │◀────│ • Overrides   │ │
│  │ • Control params │         │ Manager     │     │ • Secrets     │ │
│  │ • Viz settings  │         │             │     │ • Runtime     │ │
│  └─────────────────┘         │ • Load      │     │   settings    │ │
│                              │ • Validate  │     └───────────────┘ │
│  ┌─────────────────┐         │ • Merge     │                       │
│  │  JSON Schemas   │────────▶│ • Inherit   │                       │
│  │                 │         │ • Override  │                       │
│  │ • Type checking │         └─────────────┘                       │
│  │ • Validation    │                  │                            │
│  │ • Constraints   │                  ▼                            │
│  └─────────────────┘         Runtime Configuration                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Simulation Engine

### Unified Simulation CLI

The unified simulation CLI (`run_simulation.py`) consolidates multiple simulation modes:

```bash
# Quick demonstration
python run_simulation.py demo

# Standard 100-hour simulation
python run_simulation.py 100h --cells 5

# GPU-accelerated simulation
python run_simulation.py gpu --duration 500

# Full sensor-integrated simulation
python run_simulation.py comprehensive --output ./results

# List available modes
python run_simulation.py --list-modes
```

Available modes:
- **demo**: Quick 1-hour demonstration (fast, for testing)
- **100h**: Standard 100-hour Q-learning simulation
- **1year**: Extended 1000-hour simulation
- **gpu**: GPU-accelerated simulation
- **stack**: 5-cell MFC stack with Q-learning
- **comprehensive**: Full sensor-integrated simulation

### Time Integration Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Simulation Engine Core                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 Time Integration Loop                       │   │
│  │                                                             │   │
│  │  t = 0 ──┬─→ Physical Model Update ──┬─→ Control Update ─┐  │   │
│  │          │                          │                   │  │   │
│  │          │  ┌─────────────────────┐  │  ┌─────────────┐  │  │   │
│  │          │  │ • Biofilm growth    │  │  │ • Sensor    │  │  │   │
│  │          │  │ • Substrate kinetics│  │  │   fusion    │  │  │   │
│  │          │  │ • Electrochemistry  │  │  │ • Q-learning│  │  │   │
│  │          │  │ • Mass transport    │  │  │   decision  │  │  │   │
│  │          │  └─────────────────────┘  │  │ • Actuator  │  │  │   │
│  │          │                          │  │   commands  │  │  │   │
│  │          └──────────────────────────┘  └─────────────┘  │  │   │
│  │                                                         │  │   │
│  │  ┌─────────────────────────────────────────────────────┘  │   │
│  │  │                                                        │   │
│  │  ▼                                                        │   │
│  │  Data Logging & Checkpoint ──────────────────────────────┘   │
│  │  • State variables                                            │
│  │  • Performance metrics                                       │
│  │  • Control actions                                           │
│  │  • Periodic saves                                            │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### State Variable Organization

```python
Global State = {
    'time': float,
    'cells': [
        {
            'id': int,
            'biological': {
                'biomass': float,           # g/L
                'biofilm_thickness': float, # μm
                'substrate_conc': float,    # mM
                'growth_rate': float,       # h⁻¹
            },
            'electrochemical': {
                'voltage': float,           # V
                'current': float,           # A
                'power': float,             # W
            },
            'sensors': {
                'eis': {'thickness': float, 'quality': float},
                'qcm': {'mass': float, 'quality': float}
            }
        } for cell in range(5)
    ],
    'stack': {
        'total_voltage': float,
        'total_current': float,
        'total_power': float,
        'efficiency': float,
    },
    'control': {
        'q_learning': {
            'state': array,
            'action': array,
            'reward': float,
            'epsilon': float,
        }
    }
}
```

---

## Performance Characteristics

### Computational Performance
- **Training Time**: 0.65 seconds (1000 steps)
- **Simulation Speed**: ~1000 hours in minutes
- **GPU Speedup**: 10x for matrix operations
- **Memory Usage**: 50-500 MB depending on configuration

### Control Performance
- **Power Output**: 0.190 W (optimized) vs 0.081 W (baseline)
- **Stability**: 97.1% power consistency
- **Cell Reversal**: 100% prevention rate
- **Biofilm Health**: Maintained at optimal 3.0 μm

### Computational Complexity

| Component | Time Complexity | Space Complexity | Scaling Factor |
|-----------|----------------|------------------|----------------|
| Physical Model | O(n) per timestep | O(n) | Linear with cells |
| Q-Learning | O(s×a) per decision | O(s×a) | State-action space |
| Sensor Fusion | O(m²) per sensor | O(m) | Quadratic with sensors |
| GPU Operations | O(1) dispatch | O(n) | Parallel efficiency |

### Memory Usage Patterns

```
Peak Memory Usage:
├── Base System: ~50 MB
├── Q-Table: ~10-100 MB (depends on discretization)
├── Simulation History: ~1-10 MB per hour
├── GPU Arrays: ~100-500 MB (depends on batch size)
└── Plotting/Analysis: ~50-200 MB per figure
```

---

## Development Workflow

### 1. Environment Setup

```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Setup environment
./setup_environment.sh

# Activate environment
pixi shell
```

### 2. Running Simulations

```bash
# Using unified CLI (recommended)
python q-learning-mfcs/src/run_simulation.py demo
python q-learning-mfcs/src/run_simulation.py 100h --cells 5

# Legacy scripts (still available)
python q-learning-mfcs/src/mfc_recirculation_control.py
python q-learning-mfcs/src/mfc_unified_qlearning_control.py --profile research
```

### 3. Development Tools

```bash
# Run tests
pixi run pytest q-learning-mfcs/tests

# Code quality
pixi run ruff check
pixi run mypy q-learning-mfcs/src

# Git operations
git add <files>
git commit -m "message"
```

---

## Integration Points

### GPU Backend Selection

```python
# Automatic detection
from gpu_acceleration import GPUAccelerator
gpu = GPUAccelerator()  # Auto-selects CUDA/ROCm/CPU

# Manual selection
gpu = GPUAccelerator(backend='cuda')
```

### GPU Acceleration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GPU Acceleration Layer                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │ NVIDIA CUDA │    │  AMD ROCm   │    │ CPU Fallback│             │
│  │   (CuPy)    │    │ (PyTorch)   │    │  (NumPy)    │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│         │                   │                   │                  │
│         └───────────────────┼───────────────────┘                  │
│                             │                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │            Unified GPU Interface                            │   │
│  │  • Automatic backend detection                             │   │
│  │  • Device capability querying                              │   │
│  │  • Memory management                                       │   │
│  │  • Operation mapping                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### GitLab Integration

Automatic issue creation for:
- Test failures
- Build failures
- Performance regressions
- Documentation gaps
- Hook failures

### Configuration Loading

```python
from config.config_manager import ConfigurationManager

config = ConfigurationManager()
config.load_profile('research')
bio_params = config.get('biological')
```

---

## Troubleshooting

### Common Issues

1. **GPU Not Detected**: Check CUDA/ROCm installation
2. **Import Errors**: Verify PYTHONPATH and pixi environment
3. **Hook Failures**: Check .claude/settings.json configuration
4. **Performance Issues**: Verify GPU acceleration is active

### Debug Commands

```bash
# Check GPU status
python scripts/detect_gpu.py

# Verify environment
pixi info

# Test hooks
cd .claude/hooks && python pre_tool_use.py

# Check logs
tail -f logs/session_*.log
```

### Security and Reliability

**Data Protection:**
- No hardcoded secrets (uses environment variables)
- Git secrets scanning enabled
- Secure random number generation
- Input validation and sanitization

**Error Handling:**
- Graceful degradation for hardware failures
- Automatic fallback mechanisms
- Comprehensive logging
- Recovery from checkpoints

---

## Conclusion

The MFC Q-Learning Control System represents a sophisticated integration of biological modeling, machine learning, and software engineering best practices. The modular architecture, comprehensive automation, and performance optimizations make it suitable for both research and production deployments.

This architecture provides a robust foundation for understanding, maintaining, and extending the system.
