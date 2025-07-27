# Quick Start Guide for AI Agents

## Getting Started in 5 Minutes

This guide provides a rapid introduction to the MFC Q-Learning Control System for AI development agents who need to quickly understand and start working with the codebase.

## Project Overview - 30 Second Summary

**What it is**: A sophisticated microbial fuel cell control system using reinforcement learning (Q-learning) to optimize bioelectrochemical processes, featuring GPU acceleration, sensor fusion, and literature-validated biological parameters.

**What it does**: 
- Controls a 5-cell MFC stack for maximum power output
- Uses Q-learning to adaptively optimize flow rates, pH buffering, and substrate addition
- Integrates EIS and QCM sensors for real-time biofilm monitoring
- Achieves 132.5% power improvements through literature validation

**Key achievement**: 97.1% power stability with 100% cell reversal prevention

## Essential File Map

```
mfc-project/
├── 📋 AI_DEVELOPMENT_GUIDE.md     ← Comprehensive guide (read this)
├── 🏗️  SYSTEM_ARCHITECTURE.md     ← Technical architecture
├── 📚 API_REFERENCE.md           ← Complete API documentation
├── ⚡ QUICK_START_GUIDE.md        ← This file
├── 📖 README.md                   ← Project overview
├── 🔧 pixi.toml                   ← Dependencies (use `pixi install`)
├── q-learning-mfcs/              ← Core Q-learning system
│   ├── src/                      ← Source code
│   │   ├── 🧠 mfc_recirculation_control.py   ← Main optimized model
│   │   ├── 🎯 mfc_unified_qlearning_control.py ← Unified control system
│   │   ├── 🔬 sensor_integrated_mfc_model.py ← Sensor integration
│   │   ├── ⚙️  config/                        ← Configuration system
│   │   └── 📡 sensing_models/                 ← EIS/QCM sensors
│   ├── 📊 data/                  ← Simulation outputs
│   ├── 🧪 tests/                 ← Test suite
│   └── 📖 README.md              ← Detailed system documentation
└── 📄 docs/                     ← Technical documentation
```

## Quick Actions

### 1. Install Dependencies (1 minute)
```bash
# Install pixi if not available
curl -fsSL https://pixi.sh/install.sh | bash

# Install project dependencies
cd /home/uge/mfc-project
pixi install
```

### 2. Run a Basic Simulation (2 minutes)
```bash
cd q-learning-mfcs

# Quick demo of Q-learning controller
python src/mfc_qlearning_demo.py

# Full 100-hour optimized simulation
python src/mfc_recirculation_control.py --hours 100
```

### 3. Run Tests (1 minute)
```bash
# Run all tests
python tests/run_tests.py

# Check GPU capabilities
python tests/run_tests.py -c gpu_capability
```

### 4. Check System Status (30 seconds)
```bash
# Check git status
git status

# Check recent simulation results
ls -la q-learning-mfcs/data/simulation_data/ | head -10
```

## Essential Code Patterns

### Pattern 1: Basic Simulation Setup
```python
from sensor_integrated_mfc_model import SensorIntegratedMFCModel
from sensing_enhanced_q_controller import SensingEnhancedQController

# Initialize system
model = SensorIntegratedMFCModel(
    n_cells=5,
    species='geobacter', 
    substrate='acetate',
    use_gpu=True,
    simulation_hours=100
)

# Initialize controller
controller = SensingEnhancedQController(learning_rate=0.1)

# Run simulation
for hour in range(100):
    state = model.get_normalized_state()
    action = controller.select_action(state)
    model.apply_control_actions(action)
    new_state = model.step_dynamics(dt=1.0)
    
    reward = controller.compute_reward(prev_state, new_state, action)
    controller.update_q_table(state, action, reward, new_state)
```

### Pattern 2: Configuration Management
```python
from config.config_manager import ConfigurationManager

# Load configuration
config_mgr = ConfigurationManager()
config_mgr.load_profile_from_file('research', 'configs/research_optimization.yaml')

# Get specific configurations
bio_config = config_mgr.get_configuration('biological')
control_config = config_mgr.get_configuration('control')
```

### Pattern 3: GPU Acceleration
```python
from gpu_acceleration import GPUAccelerator

# Initialize GPU
gpu = GPUAccelerator()
print(f"Using backend: {gpu.backend}")

# GPU-accelerated operations
data = gpu.array([1.0, 2.0, 3.0])
result = gpu.exp(data)
cpu_result = gpu.to_cpu(result)
```

### Pattern 4: Sensor Integration
```python
from sensing_models.eis_model import EISModel
from sensing_models.qcm_model import QCMModel
from sensing_models.sensor_fusion import SensorFusion, FusionMethod

# Initialize sensors
eis = EISModel(species='geobacter')
qcm = QCMModel(crystal_frequency=5e6)
fusion = SensorFusion(fusion_method=FusionMethod.KALMAN_FILTER)

# Take measurements
eis_data = eis.measure_biofilm_thickness(1.5, 0.1)
qcm_data = qcm.measure_biofilm_mass(1.5, 1.1)
fused_result = fusion.fuse_measurements(eis_data, qcm_data)
```

## Current System Status (Based on Latest Sync)

### ✅ What's Working Well
- **Literature-validated parameters**: 132.5% power improvement achieved
- **GPU acceleration**: Universal support (NVIDIA/AMD/CPU fallback)
- **Sensor integration**: EIS and QCM with Kalman filter fusion
- **Configuration system**: Hierarchical YAML-based with validation
- **Testing framework**: Comprehensive test suite with hardware detection

### 🔧 Key Configuration Profiles
- **Conservative**: `configs/conservative_control.yaml` - Stable long-term operation
- **Research**: `configs/research_optimization.yaml` - Aggressive optimization
- **Precision**: `configs/precision_control.yaml` - High-accuracy laboratory use

### 📊 Performance Benchmarks
- **Training time**: 0.65 seconds (1000 steps)
- **Power output**: 0.190 W (optimized) vs 0.081 W (original)
- **Biofilm health**: 3.0 μm (maximum sustainable thickness)
- **Cell reversal prevention**: 100% success rate
- **GPU speedup**: Up to 10x for large simulations

## Common Development Tasks

### Task 1: Run Different Simulation Durations
```bash
# Short test (10 hours)
python src/mfc_recirculation_control.py --hours 10

# Standard simulation (100 hours)  
python src/mfc_recirculation_control.py --hours 100

# Long-term study (1000 hours)
python src/mfc_recirculation_control.py --hours 1000
```

### Task 2: Generate Analysis Plots
```python
from sensor_simulation_plotter import create_all_sensor_plots

# Load simulation data
import pandas as pd
data = pd.read_csv('data/simulation_data/latest_simulation.csv')

# Generate comprehensive plots
plot_files = create_all_sensor_plots(data, "analysis_20250127")
print(f"Dashboard: {plot_files['comprehensive_dashboard']}")
```

### Task 3: Test Different Species/Substrates
```python
# Test Shewanella with lactate
model = SensorIntegratedMFCModel(
    species='shewanella',
    substrate='lactate',
    simulation_hours=100
)

# Test mixed culture
model = SensorIntegratedMFCModel(
    species='mixed',
    substrate='acetate',
    simulation_hours=100
)
```

### Task 4: Debug GPU Issues
```python
from gpu_acceleration import GPUAccelerator

gpu = GPUAccelerator()
device_info = gpu.get_device_info()
print(f"Backend: {device_info['backend']}")
print(f"Device: {device_info['device_name']}")
print(f"Memory: {device_info['memory_available']} bytes")

# Test GPU functionality
test_array = gpu.array([1.0, 2.0, 3.0])
result = gpu.exp(test_array)
print(f"GPU test successful: {gpu.to_cpu(result)}")
```

## Troubleshooting Quick Fixes

### Issue: Import Errors
```bash
# Check Python path
export PYTHONPATH="/home/uge/mfc-project/q-learning-mfcs/src:$PYTHONPATH"

# Verify pixi environment
pixi shell
```

### Issue: GPU Not Detected
```python
# Check what's available
from gpu_acceleration import GPUAccelerator
gpu = GPUAccelerator()
print(gpu.get_device_info())

# Force CPU fallback
model = SensorIntegratedMFCModel(use_gpu=False)
```

### Issue: Simulation Convergence Problems
```python
# Reset Q-learning parameters
controller.epsilon = 0.3  # Increase exploration
controller.learning_rate = 0.05  # Reduce learning rate

# Check reward function scaling
reward = controller.compute_reward(prev_state, current_state, action)
print(f"Reward components: {controller.get_reward_breakdown()}")
```

### Issue: Configuration Validation Errors
```python
# Load default configuration
config_mgr.load_default_profile()

# Check validation errors
try:
    config_mgr.validate_configuration(config_dict)
except ConfigurationError as e:
    print(f"Validation error: {e}")
```

## Next Steps After Quick Start

1. **Read the full documentation**: Start with `AI_DEVELOPMENT_GUIDE.md`
2. **Explore configuration options**: Check `q-learning-mfcs/src/config/`
3. **Run comprehensive tests**: Use `python tests/run_tests.py`
4. **Examine simulation results**: Look in `q-learning-mfcs/data/`
5. **Review literature validation**: Read `docs/LITERATURE_VALIDATION_ANALYSIS.md`

## Key Insights for AI Agents

### 🧠 System Intelligence
- The Q-learning controller learns optimal policies through exploration
- Multi-objective reward function balances power, stability, and biofilm health
- Sensor fusion provides robust state estimation under uncertainty

### ⚡ Performance Optimizations
- GPU acceleration provides significant speedups (10x+)
- Literature-validated parameters dramatically improve performance
- Pixi dependency management ensures reproducible environments

### 🔬 Scientific Rigor
- All biological parameters are literature-referenced
- Comprehensive validation against experimental data
- Statistical analysis of uncertainty and confidence intervals

### 🛠️ Development Best Practices
- Modular architecture enables easy extension
- Comprehensive testing covers unit, integration, and hardware tests
- Configuration system supports multiple operational profiles

This quick start guide should get you productive with the MFC system in minutes. For deeper understanding, refer to the comprehensive documentation files.