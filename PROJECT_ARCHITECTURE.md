# MFC Project Architecture Documentation
## Project Overview

The MFC (Microbial Fuel Cell) project is a sophisticated research platform that combines biological modeling, machine learning, and high-performance computing to optimize bioelectrochemical processes. The system features advanced Q-learning control, GPU acceleration, comprehensive sensor integration, and intelligent development automation.
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

### Core Components

#### 1. Q-Learning Control System (`q-learning-mfcs/`)
- **Purpose**: Reinforcement learning for optimal MFC control
- **Key Features**:
  - 40-dimensional state space (per-cell metrics)
  - 15-dimensional action space (flow control, pH, substrate)
  - Multi-objective reward function
  - ε-greedy exploration policy

#### 2. GPU Acceleration Layer
- **Universal GPU Support**:
  - NVIDIA CUDA (via CuPy)
  - AMD ROCm (via PyTorch/JAX)
  - Automatic CPU fallback
- **Performance**: Up to 10x speedup for large simulations

#### 3. Sensor Integration
- **EIS (Electrochemical Impedance Spectroscopy)**:
  - Biofilm thickness measurement
  - Conductivity analysis
- **QCM (Quartz Crystal Microbalance)**:
  - Biofilm mass detection
  - Viscoelastic properties
- **Sensor Fusion**: Kalman filter for robust state estimation

#### 4. Development Automation (`.claude/`)
- **Intelligent Hooks System**:
  - Pre/post tool use monitoring
  - Automatic file chunking for large commits
  - GitLab integration for issue tracking
  - Event logging and notification system
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
│   │   ├── sensing_models/   # Sensor models
│   │   ├── utils/           # Utilities
│   │   └── *.py/*.mojo      # Core implementations
│   ├── data/                # Simulation outputs
│   ├── tests/              # Test suite
│   └── README.md           # Detailed documentation
│
├── scripts/                 # Utility scripts
│   ├── detect_gpu.py       # GPU detection
│   └── install_gpu_deps.py # Dependency installer
│
├── pixi.toml               # Dependency management
├── setup_environment.sh    # Environment setup
└── README.md              # Project overview
```
