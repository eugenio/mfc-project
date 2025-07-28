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
## Configuration Management

### 7. `config/config_manager.py`
**Purpose**: Hierarchical configuration management  
**Location**: `q-learning-mfcs/src/config/config_manager.py`

**Key Features**:
- YAML-based configuration
- Profile system (conservative, research, precision)
- Schema validation
- Environment variable substitution
- Runtime overrides

**Usage**:
```python
config = ConfigurationManager()
config.load_profile('research')
bio_params = config.get('biological')
```

### 8. `config/validation_schemas.py`
**Purpose**: Configuration validation schemas  
**Location**: `q-learning-mfcs/src/config/validation_schemas.py`

**Validates**:
- Biological parameters (growth rates, kinetics)
- Control parameters (PID gains, Q-learning)
- Visualization settings
- Hardware configurations
## GPU Acceleration

### 9. `gpu_acceleration.py`
**Purpose**: Universal GPU acceleration interface  
**Location**: `q-learning-mfcs/src/gpu_acceleration.py`

**Supported Backends**:
- NVIDIA CUDA (via CuPy)
- AMD ROCm (via PyTorch/JAX)
- CPU fallback (NumPy)

**Key Methods**:
```python
class GPUAccelerator:
    def __init__(self, backend='auto'):
        """Auto-detect best available backend"""
        
    def array(self, data):
        """Create GPU array"""
        
    def exp(self, x):
        """Exponential function"""
        
    def matmul(self, a, b):
        """Matrix multiplication"""
```
## Automation Hooks

### 10. `enhanced_file_chunking.py`
**Purpose**: Intelligent file chunking for version control  
**Location**: `.claude/hooks/enhanced_file_chunking.py`

**Features**:
- Language-aware parsing (Python, JS, TS, Mojo, Markdown)
- Logical boundary detection
- Automatic commit generation
- Size-aware chunking

**Supported Patterns**:
- Python: imports, classes, functions
- JavaScript/TypeScript: imports, classes, functions
- Markdown: headers, sections, code blocks
- Mojo: structs, functions

### 11. `gitlab_issue_manager.py`
**Purpose**: Automated GitLab issue management  
**Location**: `.claude/hooks/gitlab_issue_manager.py`

**Capabilities**:
- Automatic issue creation
- Issue updates and closure
- Label management
- Milestone tracking

**Trigger Events**:
- Test failures
- Build failures
- Performance regressions
- Documentation gaps

### 12. `pre_tool_use.py`
**Purpose**: Pre-execution hook for tool monitoring  
**Location**: `.claude/hooks/pre_tool_use.py`

**Functions**:
- File operation monitoring
- Threshold checking
- Dangerous operation prevention
- Event logging
## Utilities

### 13. `utils/data_export.py`
**Purpose**: Data export utilities  
**Location**: `q-learning-mfcs/src/utils/data_export.py`

**Export Formats**:
- CSV for time series data
- JSON for configuration and metadata
- HDF5 for large datasets
- LaTeX tables for reports

### 14. `utils/visualization.py`
**Purpose**: Advanced plotting utilities  
**Location**: `q-learning-mfcs/src/utils/visualization.py`

**Capabilities**:
- Real-time plotting
- Multi-panel dashboards
- 3D visualizations
- Statistical plots
- Performance metrics

### 15. `utils/literature_params.py`
**Purpose**: Literature-validated parameter database  
**Location**: `q-learning-mfcs/src/utils/literature_params.py`

**Contents**:
- Species-specific parameters
- Substrate kinetics
- Electrochemical constants
- Reference citations
## Testing Modules

### 16. `tests/test_gpu_acceleration.py`
**Purpose**: GPU functionality testing  
**Location**: `q-learning-mfcs/tests/test_gpu_acceleration.py`

**Tests**:
- Backend detection
- Mathematical operations
- Memory management
- Performance benchmarks

### 17. `tests/test_sensor_fusion.py`
**Purpose**: Sensor fusion validation  
**Location**: `q-learning-mfcs/tests/test_sensor_fusion.py`

**Validates**:
- Fusion accuracy
- Uncertainty propagation
- Edge cases
- Performance
## Module Dependencies

```
┌─────────────────────────────────────────────────────┐
│                  Module Dependency Graph             │
├─────────────────────────────────────────────────────┤
│                                                      │
│   mfc_recirculation_control                          │
│            │                                         │
│            ├──► sensor_integrated_mfc_model          │
│            │            │                            │
│            │            ├──► eis_model               │
│            │            ├──► qcm_model               │
│            │            └──► sensor_fusion           │
│            │                                         │
│            ├──► gpu_acceleration                     │
│            │                                         │
│            ├──► config_manager                       │
│            │         │                              │
│            │         └──► validation_schemas         │
│            │                                         │
│            └──► utils/                              │
│                  ├──► data_export                   │
│                  ├──► visualization                 │
│                  └──► literature_params             │
│                                                      │
└─────────────────────────────────────────────────────┘
```
## Best Practices for Module Usage

### 1. Configuration Loading
Always load configuration before initializing models:
```python
config = ConfigurationManager()
config.load_profile('research')
model = SensorIntegratedMFCModel(config=config)
```

### 2. GPU Acceleration
Check GPU availability and handle fallback:
```python
gpu = GPUAccelerator()
if gpu.backend == 'numpy':
    print("Warning: Running on CPU")
```

### 3. Sensor Integration
Always check sensor confidence:
```python
biofilm_state = model.measure_biofilm_state()
confidence = model.get_sensor_confidence()
if confidence < 0.8:
    print("Warning: Low sensor confidence")
```

### 4. Error Handling
Use proper exception handling:
```python
try:
    result = model.step_dynamics()
except ConvergenceError as e:
    print(f"Simulation failed: {e}")
    model.reset_to_checkpoint()
```
