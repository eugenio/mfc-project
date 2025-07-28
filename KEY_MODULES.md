# Key Modules Documentation
## Overview

This document provides detailed documentation of the key modules in the MFC project, their responsibilities, interfaces, and usage patterns.
## Core Q-Learning Modules

### 1. `mfc_recirculation_control.py`
**Purpose**: Main optimized MFC control system with recirculation  
**Location**: `q-learning-mfcs/src/mfc_recirculation_control.py`

**Key Features**:
- Literature-validated parameters for 132.5% power improvement
- Advanced recirculation control
- Multi-objective optimization
- Comprehensive logging and visualization

**Main Classes**:
```python
class RecirculationMFCModel:
    """Enhanced MFC model with recirculation dynamics"""
    def __init__(self, n_cells=5, use_literature_params=True):
        # Initialize with optimized parameters
        pass
    
    def step_dynamics(self, dt=1.0):
        # Simulate one timestep with recirculation
        pass
```

### 2. `mfc_unified_qlearning_control.py`
**Purpose**: Unified control system with universal GPU acceleration  
**Location**: `q-learning-mfcs/src/mfc_unified_qlearning_control.py`

**Key Features**:
- Automatic GPU backend detection (CUDA/ROCm/CPU)
- Unified API for all operations
- High-performance matrix computations

**Usage Example**:
```python
model = UnifiedMFCModel(use_gpu=True)
controller = UnifiedQController(learning_rate=0.1)

for step in range(1000):
    state = model.get_state()
    action = controller.select_action(state)
    model.apply_action(action)
    reward = model.calculate_reward()
    controller.update(state, action, reward)
```

### 3. `sensor_integrated_mfc_model.py`
**Purpose**: MFC model with advanced sensor integration  
**Location**: `q-learning-mfcs/src/sensor_integrated_mfc_model.py`

**Key Components**:
- EIS sensor integration
- QCM sensor integration
- Sensor fusion with Kalman filtering
- Noise modeling and uncertainty quantification

**Interfaces**:
```python
class SensorIntegratedMFCModel:
    def measure_biofilm_state(self) -> Dict[str, float]:
        """Get fused biofilm measurements"""
        
    def get_sensor_confidence(self) -> float:
        """Get sensor fusion confidence level"""
```
## Sensor Models

### 4. `sensing_models/eis_model.py`
**Purpose**: Electrochemical Impedance Spectroscopy modeling  
**Location**: `q-learning-mfcs/src/sensing_models/eis_model.py`

**Capabilities**:
- Biofilm thickness measurement (0.1-10 μm range)
- Conductivity analysis
- Species-specific calibration
- Realistic noise modeling

### 5. `sensing_models/qcm_model.py`
**Purpose**: Quartz Crystal Microbalance modeling  
**Location**: `q-learning-mfcs/src/sensing_models/qcm_model.py`

**Features**:
- Mass detection (ng/cm² resolution)
- Viscoelastic property analysis
- Temperature compensation
- Sauerbrey equation implementation

### 6. `sensing_models/sensor_fusion.py`
**Purpose**: Multi-sensor data fusion  
**Location**: `q-learning-mfcs/src/sensing_models/sensor_fusion.py`

**Fusion Methods**:
- Kalman filter (default)
- Weighted average
- Maximum likelihood
- Bayesian inference
- Machine learning fusion
