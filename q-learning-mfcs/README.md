# MFC Biological Configuration System

A comprehensive, advanced configuration management system for Microbial Fuel Cell (MFC) biological systems with integrated parameter optimization, uncertainty quantification, and real-time analytics.

## Overview

This system provides a robust framework for managing complex biological configurations in MFC systems, featuring:

- **Comprehensive Configuration Management**: Hierarchical, validated configuration profiles with inheritance
- **Parameter Optimization**: Bayesian, genetic, and gradient-based optimization algorithms
- **Uncertainty Quantification**: Monte Carlo methods, polynomial chaos expansion, and Bayesian inference
- **Sensitivity Analysis**: Sobol indices, Morris elementary effects, and advanced sensitivity methods
- **Real-time Processing**: Live data streaming, processing, and analytics
- **Advanced Visualization**: Multi-dimensional plotting, interactive analysis, and statistical visualization
- **Model Validation**: Cross-validation, performance metrics, and statistical testing
- **Statistical Analysis**: Comprehensive hypothesis testing and experimental design tools

## Features

### 🔧 Configuration Management
- **Multi-format Support**: YAML, JSON, TOML configuration files
- **Profile-based Management**: Environment-specific configurations (development, production, research)
- **Configuration Inheritance**: Hierarchical configuration with override capabilities
- **Validation & Type Checking**: Comprehensive parameter validation with biological constraints
- **Hot Reloading**: Dynamic configuration updates without system restart

### 📊 Advanced Analytics
- **Parameter Optimization**: 
  - Bayesian optimization with Gaussian processes
  - Genetic algorithms with multi-objective support
  - Gradient-based optimization methods
- **Uncertainty Quantification**:
  - Monte Carlo and Latin Hypercube sampling
  - Polynomial Chaos Expansion
  - Bayesian parameter estimation
- **Sensitivity Analysis**:
  - Sobol sensitivity indices
  - Morris elementary effects
  - Variance-based methods

### 🔬 Biological System Support
- **Species Configuration**: Geobacter, Shewanella, and custom microbial species
- **Substrate Management**: Multi-substrate kinetics and concentration control
- **Environmental Parameters**: Temperature, pH, conductivity, and dissolved oxygen
- **Growth Models**: Monod kinetics, inhibition models, and biofilm dynamics

### 🎛️ Control Systems
- **PID Controllers**: Flow rate, substrate concentration, and environmental control
- **Q-Learning Integration**: Reinforcement learning for adaptive control
- **Multi-objective Optimization**: Power output, biofilm health, and stability
- **Fault Tolerance**: Emergency shutdown and error recovery

### 📈 Real-time Analytics
- **Data Streaming**: High-frequency sensor data acquisition
- **Signal Processing**: Filtering, smoothing, and outlier detection
- **Anomaly Detection**: Statistical and machine learning-based detection
- **Alert System**: Configurable thresholds and notification system

### 📊 Visualization
- **Multi-dimensional Plotting**: 3D scatter plots, surface plots, parallel coordinates
- **Interactive Analysis**: Zoom, pan, selection, and real-time updates
- **Statistical Plots**: Distribution comparisons, correlation matrices, uncertainty bands
- **Publication-quality Figures**: Vector graphics with customizable styling

## Installation

### Prerequisites
- Python 3.8+
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn (for visualization)
- Scikit-learn (for machine learning features)
- Optional: Plotly (for interactive plots), Statsmodels (for advanced statistics)

### Basic Installation
```bash
git clone https://github.com/your-repo/mfc-project.git
cd mfc-project/q-learning-mfcs
pip install -r requirements.txt
```

### Development Installation
```bash
git clone https://github.com/your-repo/mfc-project.git
cd mfc-project/q-learning-mfcs
pip install -e .
pip install -r requirements-dev.txt
```

## Quick Start

### Basic Configuration
```python
from config.config_manager import ConfigurationManager
from config.biological_config import BiologicalConfig

# Load configuration
config_manager = ConfigurationManager()
config = config_manager.load_configuration('configs/conservative_control.yaml')

# Validate configuration
config_manager.validate_configuration(config)
```

### Parameter Optimization
```python
from config.parameter_optimization import BayesianOptimizer
from config.sensitivity_analysis import ParameterSpace, ParameterDefinition, ParameterBounds

# Define parameter space
parameter_space = ParameterSpace([
    ParameterDefinition("flow_rate", ParameterBounds(5.0, 30.0)),
    ParameterDefinition("substrate_conc", ParameterBounds(10.0, 25.0))
])

# Create optimizer
optimizer = BayesianOptimizer(parameter_space, objectives)

# Run optimization
result = optimizer.optimize(objective_function, max_evaluations=50)
```

### Uncertainty Quantification
```python
from config.uncertainty_quantification import MonteCarloAnalyzer, UncertainParameter

# Define uncertain parameters
uncertain_params = [
    UncertainParameter("flow_rate", DistributionType.NORMAL, 
                      {'mean': 15.0, 'std': 1.0}),
    UncertainParameter("temperature", DistributionType.UNIFORM,
                      {'low': 25.0, 'high': 35.0})
]

# Run Monte Carlo analysis
analyzer = MonteCarloAnalyzer(uncertain_params)
result = analyzer.propagate_uncertainty(model_function, n_samples=1000)
```

### Real-time Processing
```python
from config.real_time_processing import MFCDataStream, StreamProcessor

# Create data stream
stream = MFCDataStream("sensors", sensor_config, sampling_rate=1.0)

# Process data
processor = StreamProcessor(processing_config)
stream.add_callback(lambda data: processor.process([data]))

# Start streaming
stream.start()
```

## Configuration Files

### Directory Structure
```
configs/
├── conservative_control.yaml    # Conservative operation settings
├── precision_control.yaml       # High-precision laboratory settings
├── research_control.yaml        # Research-oriented configuration
└── biological/
    ├── geobacter_sulfur.yaml    # Geobacter species configuration
    ├── shewanella_oneid.yaml    # Shewanella species configuration
    └── mixed_culture.yaml       # Mixed culture configuration
```

### Example Configuration
```yaml
metadata:
  version: "2.0.0"
  description: "Conservative control settings"
  environment: "production"
  created_at: "2025-07-25T08:52:53+02:00"

biological:
  species_configs:
    geobacter:
      max_growth_rate: 0.08  # 1/h
      electron_transport_efficiency: 0.80
      cytochrome_content: 0.08
      metabolite_concentrations:
        acetate: 15.0  # mmol/L
        lactate: 8.0
        pyruvate: 5.0

control:
  flow_control:
    min_flow_rate: 8.0  # mL/h
    max_flow_rate: 35.0
    nominal_flow_rate: 15.0
    flow_pid:
      kp: 1.2
      ki: 0.05
      kd: 0.08

visualization:
  plot_style:
    figure_width: 10.0
    figure_height: 7.0
    dpi: 300
  color_scheme_type: "scientific"
```

## API Reference

### Configuration Management
- `ConfigurationManager`: Main configuration management class
- `BiologicalConfig`: Biological system configuration
- `ControlConfig`: Control system configuration
- `VisualizationConfig`: Visualization settings

### Optimization & Analysis
- `BayesianOptimizer`: Gaussian process-based optimization
- `GeneticOptimizer`: Multi-objective genetic algorithm
- `SensitivityAnalyzer`: Parameter sensitivity analysis
- `MonteCarloAnalyzer`: Uncertainty quantification
- `StatisticalAnalyzer`: Hypothesis testing and statistical analysis

### Real-time Processing
- `MFCDataStream`: Real-time data acquisition
- `StreamProcessor`: Data processing pipeline
- `RealTimeAnalyzer`: Live analytics and monitoring
- `AlertSystem`: Configurable alerting system

### Visualization
- `MultiDimensionalPlotter`: Advanced plotting capabilities
- `InteractiveAnalyzer`: Interactive visualization tools
- `StatisticalVisualizer`: Statistical plots and analysis
- `ComparisonAnalyzer`: Model and result comparison

## Examples

### Comprehensive Example
```python
# Run the comprehensive example
python examples/comprehensive_example.py
```

This example demonstrates:
- Complete system configuration and validation
- Parameter optimization with multiple objectives
- Uncertainty quantification with Monte Carlo methods
- Sensitivity analysis using Sobol indices
- Real-time data processing and analytics
- Advanced statistical analysis and hypothesis testing
- Model validation with cross-validation
- Interactive visualization and reporting

### Specific Use Cases
```python
# Parameter optimization
python examples/optimization_example.py

# Uncertainty analysis
python examples/uncertainty_example.py

# Real-time monitoring
python examples/realtime_example.py

# Statistical analysis
python examples/statistical_example.py
```

## Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_configuration.py

# Run with coverage
python -m pytest --cov=config tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings to all public functions
- Include unit tests for new features
- Update documentation for API changes
- Use type hints throughout the codebase

## Performance Considerations

### Optimization Tips
- Use Latin Hypercube sampling for better parameter space coverage
- Enable parallel processing for Monte Carlo analyses
- Configure appropriate buffer sizes for real-time streams
- Use vector operations where possible for large datasets

### Scalability
- The system is designed to handle:
  - 10,000+ parameter samples for uncertainty quantification
  - Real-time data streams up to 1000 Hz
  - Multi-objective optimization with 10+ parameters
  - Concurrent analysis of multiple MFC systems

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Configuration Validation Failures**: Check parameter ranges and types
3. **Optimization Convergence**: Adjust algorithm parameters or increase evaluations
4. **Memory Issues**: Reduce sample sizes or enable parallel processing

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Roadmap

### Version 2.1 (Q3 2025)
- [ ] Advanced machine learning models for parameter prediction
- [ ] Integration with IoT sensor networks
- [ ] Cloud-based configuration management
- [ ] Enhanced web-based visualization dashboard

### Version 2.2 (Q4 2025)
- [ ] Multi-MFC system coordination
- [ ] Advanced process control algorithms
- [ ] Machine learning-based fault detection
- [ ] Integration with laboratory information systems

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{mfc_bio_config_2025,
  title={MFC Biological Configuration System},
  author={Development Team},
  year={2025},
  url={https://github.com/your-repo/mfc-project},
  version={2.0.0}
}
```

## Acknowledgments

- Biological parameter values derived from peer-reviewed literature
- Statistical methods based on established scientific practices
- Control algorithms adapted from industrial control systems
- Visualization techniques inspired by modern data science practices

## Support

- **Documentation**: [Full documentation](https://your-docs-site.com)
- **Issues**: [GitHub Issues](https://github.com/your-repo/mfc-project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/mfc-project/discussions)
- **Email**: support@your-organization.com

---

**Note**: This system is designed for research and educational purposes. Always validate results with experimental data and consult domain experts for production deployments.

## Original Q-Learning Overview

This project implements a complete MFC stack control system featuring:

- **5-cell MFC stack simulation** with realistic electrochemical dynamics
- **Q-learning controller** optimized for accelerator hardware (GPU/NPU/ASIC)
- **Advanced sensor simulation** with EIS/QCM biofilm sensing, noise and calibration effects
- **Comprehensive configuration system** with literature-referenced biological parameters
- **Species-specific modeling** for Geobacter, Shewanella, and mixed cultures  
- **Substrate-specific kinetics** for acetate, lactate, pyruvate, and glucose
- **Control system parameterization** with PID tuning and Q-learning optimization
- **Visualization configuration** with publication-ready plotting and analysis
- **Parameter sensitivity analysis** framework with global and local methods
- **Actuator control** for duty cycle, pH buffer, and acetate addition
- **Cell reversal prevention** and recovery mechanisms
- **Real-time optimization** for power stability and efficiency

## System Architecture

### Core Components

1. **MFC Model (`odes.mojo`)** - Electrochemical simulation using Mojo
1. **Q-Learning Controller (`mfc_qlearning.mojo`)** - Accelerated learning algorithm
1. **Stack Simulation (`mfc_stack_simulation.py`)** - Complete 5-cell system
1. **Biological Configuration System** - Literature-referenced parameter management
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

#### Biological Configuration Examples

```bash
# Run Geobacter-acetate configuration example
python examples/example_geobacter_acetate_config.py

# Run Shewanella-lactate configuration example
python examples/example_shewanella_lactate_config.py

# Run mixed culture configuration example
python examples/example_mixed_culture_config.py
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

## Configuration System

### Overview

The MFC system features a comprehensive configuration management framework that replaces hardcoded parameters with literature-backed, configurable values. This system enables:

- **Biological parameter management** with species-specific and substrate-specific configurations
- **Control system parameterization** for PID controllers, Q-learning parameters, and flow control
- **Visualization configuration** for publication-ready plots and analysis
- **Parameter sensitivity analysis** to identify critical system parameters

### Configuration Components

#### 1. Biological Configuration (`src/config/biological_config.py`)

**Species-Specific Parameters**:
- Metabolic rates and electron transport efficiency
- Cytochrome content and growth characteristics
- Literature-referenced parameters for Geobacter and Shewanella

**Substrate-Specific Parameters**:
- Michaelis-Menten kinetics (Vmax, Km, Ki)
- Molecular properties and chemical formulas
- Species-substrate interaction parameters

**Example**:
```python
from config.biological_config import get_geobacter_config
config = get_geobacter_config()
# Access: config.max_growth_rate, config.electron_transport_efficiency
```

#### 2. Control System Configuration (`src/config/control_config.py`)

**PID Controller Parameters**:
- Proportional, integral, and derivative gains
- Anti-windup and bumpless transfer settings
- Setpoint weighting and derivative filtering

**Flow Control Parameters**:
- Flow rate bounds and tolerance settings
- Pump characteristics and response times
- Safety parameters and alarm thresholds

**Q-Learning Parameters**:
- Learning rate, discount factor, exploration parameters
- State and action space discretization
- Multi-objective reward weights

**Example**:
```python
from config.control_config import get_precision_control_config
config = get_precision_control_config()
# Access: config.flow_control.flow_pid.kp, config.advanced_control.learning_rate
```

#### 3. Visualization Configuration (`src/config/visualization_config.py`)

**Plot Styling**:
- Figure dimensions, DPI, and font settings
- Line widths, marker sizes, and transparency
- Grid styling and color schemes

**Data Processing**:
- Sampling rates and smoothing parameters
- Outlier detection and missing data handling
- Statistical analysis settings

**Layout Configuration**:
- Subplot arrangements and spacing
- Legend positioning and styling
- Axis scaling and limits

#### 4. Configuration Management (`src/config/config_manager.py`)

**Profile Management**:
- Multiple configuration profiles (conservative, aggressive, precision)
- Profile inheritance and overrides
- Version control and migration

**Validation and Loading**:
- JSON Schema validation
- YAML/JSON file support
- Environment variable substitution

### Configuration Profiles

#### Conservative Control (`configs/conservative_control.yaml`)
- Stable operation with conservative PID tuning
- Higher safety margins and slower response times
- Suitable for long-term continuous operation

#### Research Optimization (`configs/research_optimization.yaml`)
- Aggressive optimization for maximum performance
- High exploration rates and rapid adaptation
- Ideal for research and development scenarios

#### Precision Control (`configs/precision_control.yaml`)
- High-accuracy control for laboratory applications
- Tight tolerances and precise parameter tuning
- Publication-quality measurement requirements

### Parameter Sensitivity Analysis

#### Framework Features (`src/config/sensitivity_analysis.py`)

**Local Sensitivity Analysis**:
- One-at-a-time parameter perturbation
- Gradient-based sensitivity calculation
- Local parameter importance ranking

**Global Sensitivity Analysis**:
- Sobol global sensitivity indices
- Morris elementary effects method
- Variance-based importance measures

**Sampling Methods**:
- Latin Hypercube sampling
- Sobol sequence sampling
- Random and grid-based sampling

#### Usage Example

```python
from config.sensitivity_analysis import ParameterSpace, SensitivityAnalyzer
from config.sensitivity_analysis import ParameterDefinition, ParameterBounds

# Define parameter space
parameters = [
    ParameterDefinition(
        name="learning_rate",
        bounds=ParameterBounds(0.01, 0.3, nominal_value=0.1),
        config_path=["control", "advanced_control", "learning_rate"]
    )
]

param_space = ParameterSpace(parameters)
analyzer = SensitivityAnalyzer(param_space, model_function, output_names)

# Perform Sobol analysis
result = analyzer.analyze_sensitivity(
    method=SensitivityMethod.SOBOL,
    n_samples=1000
)

# Rank parameters by importance
ranking = analyzer.rank_parameters(result, "power_output", "total_order")
```

### Configuration Usage

#### Loading Configurations

```python
from config.config_manager import get_config_manager

# Initialize configuration manager
config_mgr = get_config_manager("configs/")

# Load specific profile
config_mgr.load_profile_from_file("conservative", "configs/conservative_control.yaml")
config_mgr.set_current_profile("conservative")

# Access configurations
biological_config = config_mgr.get_configuration("biological")
control_config = config_mgr.get_configuration("control")
```

#### Creating Custom Profiles

```python
# Create new profile
profile = config_mgr.create_profile(
    profile_name="custom_research",
    biological=custom_biological_config,
    control=custom_control_config,
    inherits_from="research_optimization"
)

# Save to file
config_mgr.save_profile_to_file("custom_research", "configs/custom_research.yaml")
```

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
1. **Real-time Performance**: \<1ms control loop execution
1. **Scalability**: Linear scaling with cell count

### Universal GPU Acceleration (New)

The Python implementations now feature universal GPU acceleration:

1. **Multi-vendor Support**:
   - NVIDIA GPUs via CuPy
   - AMD GPUs via PyTorch with ROCm
   - Automatic backend detection
1. **CPU Fallback**: Seamless operation on systems without GPU
1. **Unified API**: Single interface for all mathematical operations
1. **Performance Benefits**:
   - Up to 10x speedup for large-scale simulations
   - Real-time control loop execution
   - Efficient memory management
1. **Tested Operations**:
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

### Biological Configuration System (New)

The system now includes a comprehensive biological parameter management framework:

#### Species-Specific Configurations

- **Geobacter sulfurreducens**: Optimized for direct electron transfer and acetate utilization
- **Shewanella oneidensis**: Enhanced for flavin-mediated electron transfer and lactate metabolism
- **Mixed cultures**: Dynamic species ratio management with synergistic interactions
- **Literature-referenced parameters**: All values backed by peer-reviewed research

#### Substrate-Specific Modeling

- **Acetate**: Primary substrate for Geobacter with complete oxidation pathway
- **Lactate**: Preferred substrate for Shewanella with pyruvate intermediate
- **Pyruvate**: Universal substrate for both species with enhanced kinetics
- **Glucose**: Complex substrate with fermentation pathways

#### Key Configuration Features

- **Metabolic reaction definitions** with enzyme kinetics and thermodynamics
- **Biofilm formation parameters** with species-specific growth characteristics
- **Environmental compensation** for temperature and pH effects
- **Comprehensive validation** ensuring biological plausibility
- **Modular design** for easy extension with new species and substrates

#### Literature References

All parameters are referenced to key publications:

- Lovley (2003): Geobacter metabolism and electron transfer
- Bond et al. (2002): Electrode-reducing microorganisms
- Marsili et al. (2008): Shewanella flavin-mediated electron transfer
- Torres et al. (2010): Kinetic perspective on extracellular electron transfer
- Marcus et al. (2007): Biofilm anode modeling
- Reguera et al. (2005): Microbial nanowires

## Results and Analysis

### Simulation Results

- **Training time**: 0.65 seconds (1000 steps)
- **Final power**: 0.037-0.245 W
- **Reversed cells**: 0/5 (successful prevention)
- **Power stability**: 97.1%

### Individual Cell Performance

| Cell | Voltage (V) | Power (W) | pH | Acetate |
|------|-------------|-----------|-----|---------|
| 0 | 0.178 | 0.010 | 8.1 | 1.545 |
| 1 | 0.173 | 0.014 | 8.0 | 1.584 |
| 2 | 0.204 | 0.020 | 8.0 | 1.512 |
| 3 | 0.197 | 0.014 | 7.9 | 1.569 |
| 4 | 0.195 | 0.017 | 8.2 | 1.622 |

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
- `src/metabolic_model/` - Metabolic pathway modeling with species-specific parameters
- `src/integrated_mfc_model.py` - Complete integrated MFC system
- `src/mfc_recirculation_control.py` - Recirculation and substrate control
- `src/sensing_enhanced_q_controller.py` - Sensor-enhanced Q-learning controller
- `src/sensor_integrated_mfc_model.py` - MFC model with sensor feedback loops

### Biological Configuration System

- `src/config/biological_config.py` - Species-specific metabolic and biofilm parameters
- `src/config/substrate_config.py` - Substrate-specific kinetic and thermodynamic properties
- `src/config/biological_validation.py` - Parameter validation and biological plausibility checks
- `src/config/parameter_validation.py` - Core validation functions and error handling

### Sensing Models

- `src/sensing_models/eis_model.py` - Electrochemical impedance spectroscopy
- `src/sensing_models/qcm_model.py` - Quartz crystal microbalance
- `src/sensing_models/sensor_fusion.py` - Multi-sensor data fusion
- `src/sensing_models/__init__.py` - Sensing models module

### Configuration Examples

- `examples/example_geobacter_acetate_config.py` - Geobacter with acetate configuration
- `examples/example_shewanella_lactate_config.py` - Enhanced Shewanella with lactate
- `examples/example_mixed_culture_config.py` - Mixed culture systems with competition
- `examples/README.md` - Comprehensive examples documentation

### Testing

- `tests/sensing_models/test_sensing_models.py` - Comprehensive sensor tests
- `tests/config/` - Configuration system tests
- `tests/run_tests.py` - Test suite runner
- `README.md` - This documentation

## References

Based on recent research in Q-learning applications for microbial fuel cells, particularly:

- Machine learning solutions for enhanced performance in plant-based microbial fuel cells
- Q-learning based control for energy management systems
- Advanced control strategies for bioelectrochemical systems

## License

This project is for educational and research purposes.
