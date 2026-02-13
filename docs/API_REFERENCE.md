# MFC Q-Learning Control System - API Reference

*Last Updated: July 29, 2025*

## Overview

This document provides comprehensive API documentation for the MFC Q-Learning Control System. All classes, functions, and modules are documented with usage examples, parameter descriptions, and return value specifications.

## Core APIs

### 1. MFC Model APIs

#### `IntegratedMFCModel`

The main simulation model combining biological, electrochemical, and control systems.

```python
class IntegratedMFCModel:
    """
    Integrated MFC model with biological dynamics, electrochemistry, and control.
    
    Attributes:
        n_cells (int): Number of MFC cells in the stack
        species (str): Bacterial species ('geobacter', 'shewanella', 'mixed')
        substrate (str): Primary substrate ('acetate', 'lactate', 'pyruvate', 'glucose')
        use_gpu (bool): Enable GPU acceleration
        history (List[State]): Simulation state history
    """
    
    def __init__(self, n_cells: int = 5, species: str = 'geobacter', 
                 substrate: str = 'acetate', use_gpu: bool = True,
                 simulation_hours: float = 100.0) -> None:
        """
        Initialize the integrated MFC model.
        
        Args:
            n_cells: Number of cells in the MFC stack (default: 5)
            species: Bacterial species configuration
            substrate: Primary substrate type
            use_gpu: Enable GPU acceleration if available
            simulation_hours: Total simulation duration
            
        Example:
            >>> model = IntegratedMFCModel(n_cells=5, species='geobacter')
            >>> model.initialize_system()
        """
    
    def step_dynamics(self, dt: float = 1.0) -> 'SystemState':
        """
        Advance the system by one time step.
        
        Args:
            dt: Time step size in hours
            
        Returns:
            SystemState: Current system state after time step
            
        Example:
            >>> state = model.step_dynamics(dt=0.1)
            >>> print(f"Power: {state.total_power:.3f} W")
        """
    
    def set_control_action(self, cell_id: int, action: Dict[str, float]) -> None:
        """
        Apply control action to specific cell.
        
        Args:
            cell_id: Target cell index (0-based)
            action: Control parameters
                - 'duty_cycle': PWM duty cycle (0.0-1.0)
                - 'ph_buffer': pH buffer activation (0.0-1.0)
                - 'substrate_addition': Substrate addition rate (mM/h)
                
        Example:
            >>> model.set_control_action(0, {
            ...     'duty_cycle': 0.8,
            ...     'ph_buffer': 0.0,
            ...     'substrate_addition': 0.5
            ... })
        """
```

#### `SensorIntegratedMFCModel`

Extended MFC model with sensor integration and fusion capabilities.

```python
class SensorIntegratedMFCModel(IntegratedMFCModel):
    """
    MFC model with integrated EIS and QCM sensor systems.
    
    Additional Attributes:
        enable_eis (bool): Enable EIS sensors
        enable_qcm (bool): Enable QCM sensors
        sensor_fusion_method (FusionMethod): Fusion algorithm
        sensor_states (List[Dict]): Real-time sensor data
    """
    
    def __init__(self, enable_eis: bool = True, enable_qcm: bool = True,
                 sensor_fusion_method: 'FusionMethod' = FusionMethod.KALMAN_FILTER,
                 **kwargs) -> None:
        """
        Initialize sensor-integrated MFC model.
        
        Args:
            enable_eis: Enable electrochemical impedance spectroscopy
            enable_qcm: Enable quartz crystal microbalance
            sensor_fusion_method: Algorithm for multi-sensor fusion
            **kwargs: Passed to parent IntegratedMFCModel
            
        Example:
            >>> model = SensorIntegratedMFCModel(
            ...     enable_eis=True,
            ...     enable_qcm=True,
            ...     sensor_fusion_method=FusionMethod.KALMAN_FILTER
            ... )
        """
    
    def get_sensor_readings(self, cell_id: int) -> Dict[str, Any]:
        """
        Get current sensor readings for a specific cell.
        
        Args:
            cell_id: Target cell index
            
        Returns:
            Dict containing sensor data:
                - 'eis_thickness': Biofilm thickness from EIS (μm)
                - 'qcm_mass': Biofilm mass from QCM (ng/cm²)
                - 'fusion_confidence': Sensor agreement (0-1)
                - 'sensor_quality': Overall sensor health (0-1)
                
        Example:
            >>> sensors = model.get_sensor_readings(0)
            >>> print(f"Biofilm thickness: {sensors['eis_thickness']:.2f} μm")
        """
```

### 2. Q-Learning Controller APIs

#### `AdvancedQLearningController`

Main reinforcement learning controller for MFC optimization.

```python
class AdvancedQLearningController:
    """
    Advanced Q-Learning controller with multi-objective optimization.
    
    Attributes:
        state_bins (Dict[str, np.ndarray]): State space discretization
        action_space (np.ndarray): Available actions
        q_table (np.ndarray): Q-value lookup table
        epsilon (float): Current exploration rate
        learning_stats (Dict): Training statistics
    """
    
    def __init__(self, state_dim: int = 40, action_dim: int = 15,
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 epsilon_start: float = 0.3, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995) -> None:
        """
        Initialize Q-Learning controller.
        
        Args:
            state_dim: Dimensionality of state space
            action_dim: Dimensionality of action space
            learning_rate: Q-learning step size (alpha)
            discount_factor: Future reward discount (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate per episode
            
        Example:
            >>> controller = AdvancedQLearningController(
            ...     learning_rate=0.1,
            ...     epsilon_start=0.3
            ... )
        """
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current system state (normalized)
            
        Returns:
            Selected action vector
            
        Example:
            >>> state = model.get_normalized_state()
            >>> action = controller.select_action(state)
            >>> model.apply_control_action(action)
        """
    
    def update_q_table(self, state: np.ndarray, action: np.ndarray,
                      reward: float, next_state: np.ndarray) -> None:
        """
        Update Q-table using Q-learning algorithm.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            
        Example:
            >>> controller.update_q_table(
            ...     prev_state, action, reward, current_state
            ... )
        """
    
    def compute_reward(self, prev_state: 'SystemState', 
                      current_state: 'SystemState',
                      action: np.ndarray) -> float:
        """
        Compute multi-objective reward signal.
        
        Args:
            prev_state: Previous system state
            current_state: Current system state
            action: Control action taken
            
        Returns:
            Computed reward value
            
        Reward Components:
            - Power optimization: +50x increases, -100x decreases
            - Substrate utilization: +30x increases, -60x decreases
            - Biofilm health: +25 optimal, -50x deviation
            - Cell reversal: -10 per reversed cell
            - Action penalty: Discourages extreme actions
            
        Example:
            >>> reward = controller.compute_reward(
            ...     prev_state, current_state, action
            ... )
        """
```

#### `SensingEnhancedQController`

Q-Learning controller enhanced with sensor feedback.

```python
class SensingEnhancedQController(AdvancedQLearningController):
    """
    Q-Learning controller with sensor-enhanced state space.
    
    Extended state space includes:
    - EIS biofilm thickness measurements
    - QCM mass measurements
    - Sensor quality indicators
    - Fusion confidence metrics
    """
    
    def get_sensor_enhanced_state(self, model: 'SensorIntegratedMFCModel') -> np.ndarray:
        """
        Extract sensor-enhanced state vector.
        
        Args:
            model: Sensor-integrated MFC model
            
        Returns:
            Enhanced state vector including sensor data
            
        State Vector Components (45D):
            - Base state (40D): Standard MFC state
            - Sensor state (5D): EIS thickness, QCM mass, fusion confidence
            
        Example:
            >>> state = controller.get_sensor_enhanced_state(model)
            >>> action = controller.select_action(state)
        """
```

### 3. Sensor System APIs

#### `EISModel`

Electrochemical Impedance Spectroscopy sensor model.

```python
class EISModel:
    """
    EIS sensor model for biofilm thickness measurement.
    
    Attributes:
        frequency_range (Tuple[float, float]): EIS frequency range (Hz)
        calibration_params (Dict): Species-specific calibration
        noise_level (float): Measurement noise standard deviation
    """
    
    def __init__(self, species: str = 'geobacter', 
                 frequency_range: Tuple[float, float] = (0.1, 100000),
                 noise_level: float = 0.05) -> None:
        """
        Initialize EIS sensor model.
        
        Args:
            species: Bacterial species for calibration
            frequency_range: Frequency sweep range (Hz)
            noise_level: Relative noise level (0-1)
            
        Example:
            >>> eis = EISModel(species='geobacter', noise_level=0.03)
        """
    
    def measure_biofilm_thickness(self, true_thickness: float, 
                                conductivity: float) -> Dict[str, float]:
        """
        Simulate EIS measurement of biofilm thickness.
        
        Args:
            true_thickness: Actual biofilm thickness (μm)
            conductivity: Solution conductivity (S/m)
            
        Returns:
            Dict containing:
                - 'measured_thickness': EIS thickness estimate (μm)
                - 'impedance_magnitude': |Z| at characteristic frequency
                - 'phase_angle': Phase angle (degrees)
                - 'measurement_quality': Quality indicator (0-1)
                
        Example:
            >>> result = eis.measure_biofilm_thickness(1.5, 0.1)
            >>> thickness = result['measured_thickness']
        """
```

#### `QCMModel`

Quartz Crystal Microbalance sensor model.

```python
class QCMModel:
    """
    QCM sensor model for biofilm mass measurement.
    
    Attributes:
        crystal_frequency (float): Base crystal frequency (Hz)
        sensitivity_factor (float): Mass sensitivity (Hz⋅cm²/ng)
        viscoelastic_correction (bool): Enable viscoelastic effects
    """
    
    def __init__(self, crystal_frequency: float = 5e6,
                 crystal_type: str = 'AT_cut',
                 viscoelastic_correction: bool = True) -> None:
        """
        Initialize QCM sensor model.
        
        Args:
            crystal_frequency: Base resonance frequency (Hz)
            crystal_type: Crystal cut type ('AT_cut', 'BT_cut')
            viscoelastic_correction: Account for soft biofilm effects
            
        Example:
            >>> qcm = QCMModel(crystal_frequency=5e6, crystal_type='AT_cut')
        """
    
    def measure_biofilm_mass(self, thickness: float, density: float,
                           viscosity: float = None) -> Dict[str, float]:
        """
        Simulate QCM measurement of biofilm mass.
        
        Args:
            thickness: Biofilm thickness (μm)
            density: Biofilm density (g/cm³)
            viscosity: Biofilm viscosity (Pa⋅s, optional)
            
        Returns:
            Dict containing:
                - 'mass_per_area': Areal mass density (ng/cm²)
                - 'frequency_shift': Resonance frequency change (Hz)
                - 'dissipation': Energy dissipation factor
                - 'measurement_quality': Quality indicator (0-1)
                
        Example:
            >>> result = qcm.measure_biofilm_mass(1.5, 1.1)
            >>> mass = result['mass_per_area']
        """
```

#### `SensorFusion`

Multi-sensor data fusion engine.

```python
class SensorFusion:
    """
    Multi-algorithm sensor fusion for biofilm characterization.
    
    Supported fusion methods:
    - Kalman Filter: Optimal for Gaussian noise
    - Weighted Average: Simple but robust
    - Maximum Likelihood: Statistical optimization
    - Bayesian Inference: Full uncertainty quantification
    """
    
    def __init__(self, fusion_method: 'FusionMethod' = FusionMethod.KALMAN_FILTER,
                 uncertainty_quantification: bool = True) -> None:
        """
        Initialize sensor fusion engine.
        
        Args:
            fusion_method: Fusion algorithm selection
            uncertainty_quantification: Enable confidence intervals
            
        Example:
            >>> fusion = SensorFusion(
            ...     fusion_method=FusionMethod.KALMAN_FILTER
            ... )
        """
    
    def fuse_measurements(self, eis_data: Dict, qcm_data: Dict,
                         previous_estimate: Dict = None) -> Dict[str, Any]:
        """
        Fuse EIS and QCM measurements for optimal biofilm estimate.
        
        Args:
            eis_data: EIS sensor measurements
            qcm_data: QCM sensor measurements
            previous_estimate: Prior state estimate (for Kalman filter)
            
        Returns:
            Dict containing fused estimates:
                - 'thickness': Optimal thickness estimate (μm)
                - 'mass': Optimal mass estimate (ng/cm²)
                - 'density': Derived density (g/cm³)
                - 'confidence': Fusion confidence (0-1)
                - 'uncertainty': Confidence intervals
                
        Example:
            >>> fused = fusion.fuse_measurements(eis_result, qcm_result)
            >>> thickness = fused['thickness']
            >>> confidence = fused['confidence']
        """
```

### 4. Configuration APIs

#### `ConfigurationManager`

Central configuration management system.

```python
class ConfigurationManager:
    """
    Hierarchical configuration management with validation and inheritance.
    
    Attributes:
        profiles (Dict): Loaded configuration profiles
        current_profile (str): Active profile name
        config_directory (str): Configuration file directory
    """
    
    def __init__(self, config_directory: str = 'configs/') -> None:
        """
        Initialize configuration manager.
        
        Args:
            config_directory: Directory containing configuration files
            
        Example:
            >>> config_mgr = ConfigurationManager('configs/')
        """
    
    def load_profile_from_file(self, profile_name: str, 
                              file_path: str) -> 'ConfigurationProfile':
        """
        Load configuration profile from YAML/JSON file.
        
        Args:
            profile_name: Unique profile identifier
            file_path: Path to configuration file
            
        Returns:
            Loaded and validated configuration profile
            
        Example:
            >>> profile = config_mgr.load_profile_from_file(
            ...     'research', 'configs/research_optimization.yaml'
            ... )
        """
    
    def get_configuration(self, config_type: str) -> Any:
        """
        Get specific configuration section from current profile.
        
        Args:
            config_type: Configuration section ('biological', 'control', 'visualization')
            
        Returns:
            Configuration object for specified type
            
        Example:
            >>> bio_config = config_mgr.get_configuration('biological')
            >>> max_growth = bio_config.max_growth_rate
        """
    
    def validate_configuration(self, config: Dict) -> bool:
        """
        Validate configuration against schema and biological constraints.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, raises ValidationError if invalid
            
        Example:
            >>> is_valid = config_mgr.validate_configuration(config_dict)
        """
```

#### `BiologicalConfig`

Species and substrate-specific biological parameters.

```python
@dataclass
class BiologicalConfig:
    """
    Biological configuration for specific species-substrate combination.
    
    All parameters are literature-referenced and validated.
    """
    
    # Species identification
    species: str                    # 'geobacter', 'shewanella', 'mixed'
    substrate: str                  # 'acetate', 'lactate', 'pyruvate', 'glucose'
    
    # Growth kinetics
    max_growth_rate: float          # μmax (h⁻¹)
    substrate_affinity: float       # Ks (mM)
    substrate_inhibition: float     # Ki (mM)
    yield_coefficient: float        # Y (g biomass/g substrate)
    
    # Biofilm parameters
    max_biofilm_thickness: float    # Lmax (μm)
    biofilm_growth_rate: float      # kb (h⁻¹)
    biofilm_decay_rate: float       # kd (h⁻¹)
    adhesion_strength: float        # τ (Pa)
    
    # Electrochemical properties
    electron_transport_efficiency: float    # dimensionless (0-1)
    standard_potential: float              # E° (V vs SHE)
    exchange_current_density: float        # i0 (A/m²)
    charge_transfer_coefficient: float     # α (dimensionless)
    
    # Environmental factors
    optimal_temperature: float      # T (°C)
    optimal_ph: float              # pH units
    temperature_sensitivity: float # Q10 factor
    ph_tolerance_range: Tuple[float, float]  # pH range
    
    def get_species_config(species: str) -> 'BiologicalConfig':
        """
        Get predefined configuration for specific species.
        
        Args:
            species: Target species name
            
        Returns:
            Species-specific biological configuration
            
        Example:
            >>> config = BiologicalConfig.get_species_config('geobacter')
            >>> print(f"Max growth rate: {config.max_growth_rate} h⁻¹")
        """
```

### 5. GPU Acceleration APIs

#### `GPUAccelerator`

Universal GPU acceleration interface.

```python
class GPUAccelerator:
    """
    Universal GPU acceleration with automatic backend detection.
    
    Supported backends:
    - NVIDIA CUDA (via CuPy)
    - AMD ROCm (via PyTorch)
    - CPU fallback (via NumPy)
    
    Attributes:
        backend (str): Active backend ('cuda', 'rocm', 'cpu')
        device_info (Dict): Hardware information
        available_memory (int): Available GPU memory (bytes)
    """
    
    def __init__(self, preferred_backend: str = 'auto') -> None:
        """
        Initialize GPU accelerator with automatic detection.
        
        Args:
            preferred_backend: Preferred backend ('cuda', 'rocm', 'cpu', 'auto')
            
        Example:
            >>> gpu = GPUAccelerator()
            >>> print(f"Using backend: {gpu.backend}")
        """
    
    def array(self, data: Union[List, np.ndarray], dtype: str = 'float32') -> Any:
        """
        Create GPU array from input data.
        
        Args:
            data: Input data (list, numpy array)
            dtype: Data type specification
            
        Returns:
            GPU array in appropriate backend format
            
        Example:
            >>> gpu_array = gpu.array([1.0, 2.0, 3.0])
            >>> result = gpu.exp(gpu_array)
        """
    
    def to_cpu(self, gpu_array: Any) -> np.ndarray:
        """
        Transfer GPU array to CPU memory.
        
        Args:
            gpu_array: GPU array in backend format
            
        Returns:
            NumPy array on CPU
            
        Example:
            >>> cpu_result = gpu.to_cpu(gpu_array)
        """
    
    def exp(self, x: Any) -> Any:
        """Compute element-wise exponential."""
    
    def log(self, x: Any) -> Any:
        """Compute element-wise natural logarithm."""
    
    def sqrt(self, x: Any) -> Any:
        """Compute element-wise square root."""
    
    def power(self, x: Any, exponent: float) -> Any:
        """Compute element-wise power."""
    
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """Conditional element selection."""
    
    def mean(self, x: Any, axis: int = None) -> Any:
        """Compute mean along specified axis."""
    
    def sum(self, x: Any, axis: int = None) -> Any:
        """Compute sum along specified axis."""
    
    def maximum(self, x: Any, y: Any) -> Any:
        """Element-wise maximum."""
    
    def minimum(self, x: Any, y: Any) -> Any:
        """Element-wise minimum."""
    
    def clip(self, x: Any, min_val: float, max_val: float) -> Any:
        """Clip values to specified range."""
    
    def random_normal(self, shape: Tuple[int, ...], 
                     mean: float = 0.0, std: float = 1.0) -> Any:
        """Generate random numbers from normal distribution."""
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get detailed device information.
        
        Returns:
            Dict containing:
                - 'backend': Active backend name
                - 'device_name': Hardware device name
                - 'memory_total': Total device memory (bytes)
                - 'memory_available': Available memory (bytes)
                - 'compute_capability': Compute version (CUDA only)
                
        Example:
            >>> info = gpu.get_device_info()
            >>> print(f"Device: {info['device_name']}")
        """
```

### 6. Visualization APIs

#### `create_all_sensor_plots`

Comprehensive visualization function for sensor-integrated simulations.

```python
def create_all_sensor_plots(data: Dict[str, Any], timestamp: str,
                           output_dir: str = None) -> Dict[str, str]:
    """
    Generate comprehensive sensor analysis plots.
    
    Args:
        data: Simulation data dictionary containing:
            - 'time_hours': Time series (array)
            - 'stack_power': Power output (array)
            - 'biofilm_thickness': Biofilm thickness series (array)
            - 'coulombic_efficiency_series': Efficiency time series (array)
            - 'cell_voltages': Individual cell voltages (2D array)
            - 'substrate_concentrations': Substrate levels (array)
            - 'sensor_enabled': Boolean sensor status
            - 'fusion_method': Sensor fusion algorithm name
        timestamp: Timestamp string for file naming
        output_dir: Output directory (optional, uses default if None)
        
    Returns:
        Dict mapping plot names to file paths:
            - 'comprehensive_dashboard': Main analysis dashboard
            - 'sensor_analysis': Sensor-specific plots
            - 'performance_summary': Performance metrics
            - 'csv_data': Exported CSV data file
            - 'json_data': Exported JSON data file
            
    Example:
        >>> plot_files = create_all_sensor_plots(
        ...     simulation_data, "20250127_143022"
        ... )
        >>> dashboard_path = plot_files['comprehensive_dashboard']
    """
```

## Usage Examples

### Complete Simulation Workflow

```python
# Initialize system with sensor integration
model = SensorIntegratedMFCModel(
    n_cells=5,
    species='geobacter',
    substrate='acetate',
    enable_eis=True,
    enable_qcm=True,
    use_gpu=True,
    simulation_hours=100.0
)

# Initialize Q-learning controller
controller = SensingEnhancedQController(
    learning_rate=0.1,
    epsilon_start=0.3
)

# Load configuration
config_mgr = ConfigurationManager()
config_mgr.load_profile_from_file('research', 'configs/research_optimization.yaml')
bio_config = config_mgr.get_configuration('biological')

# Run simulation loop
for hour in range(100):
    # Get sensor-enhanced state
    state = controller.get_sensor_enhanced_state(model)
    
    # Select and apply control action
    action = controller.select_action(state)
    model.apply_control_actions(action)
    
    # Step system dynamics
    new_state = model.step_dynamics(dt=1.0)
    
    # Compute reward and update Q-table
    reward = controller.compute_reward(prev_state, new_state, action)
    controller.update_q_table(state, action, reward, 
                             controller.get_sensor_enhanced_state(model))
    
    # Log progress
    if hour % 10 == 0:
        print(f"Hour {hour}: Power={new_state.total_power:.3f}W, "
              f"Reward={reward:.2f}, ε={controller.epsilon:.3f}")

# Generate analysis plots
results = model.get_simulation_results()
plot_files = create_all_sensor_plots(results, "simulation_20250127")
print(f"Results saved to: {plot_files['comprehensive_dashboard']}")
```

### Custom Species Configuration

```python
# Create custom species configuration
custom_config = BiologicalConfig(
    species='custom_bacteria',
    substrate='lactate',
    max_growth_rate=0.08,  # h⁻¹
    substrate_affinity=2.5,  # mM
    yield_coefficient=0.45,  # g/g
    electron_transport_efficiency=0.90,
    standard_potential=0.35,  # V vs SHE
    optimal_temperature=35.0,  # °C
    optimal_ph=7.2
)

# Validate and use configuration
config_mgr.validate_configuration(custom_config.__dict__)
model = IntegratedMFCModel(biological_config=custom_config)
```

### GPU Acceleration Usage

```python
# Initialize GPU acceleration
gpu = GPUAccelerator()
print(f"Using {gpu.backend} backend")

# Create GPU arrays for computation
biofilm_data = gpu.array(model.get_biofilm_thickness_array())
substrate_data = gpu.array(model.get_substrate_concentrations())

# Perform GPU-accelerated calculations
growth_rate = gpu.exp(-substrate_data / gpu.array([2.5])) * gpu.array([0.05])
new_biofilm = biofilm_data + growth_rate * gpu.array([1.0])  # dt = 1.0

# Transfer results back to CPU
cpu_result = gpu.to_cpu(new_biofilm)
```

## Error Handling and Exceptions

### Common Exceptions

```python
class MFCModelError(Exception):
    """Base exception for MFC model errors."""

class ConfigurationError(MFCModelError):
    """Configuration validation or loading errors."""

class ConvergenceError(MFCModelError):
    """Q-learning convergence failures."""

class SensorError(MFCModelError):
    """Sensor measurement or fusion errors."""

class GPUError(MFCModelError):
    """GPU acceleration errors."""
```

### Error Handling Patterns

```python
try:
    model = SensorIntegratedMFCModel(enable_eis=True)
    controller = SensingEnhancedQController()
    
    # Simulation loop with error handling
    for step in range(1000):
        try:
            state = controller.get_sensor_enhanced_state(model)
            action = controller.select_action(state)
            new_state = model.step_dynamics()
            
        except SensorError as e:
            print(f"Sensor error at step {step}: {e}")
            # Fallback to standard (non-sensor) state
            state = model.get_standard_state()
            continue
            
        except ConvergenceError as e:
            print(f"Convergence issue at step {step}: {e}")
            # Reset exploration rate
            controller.epsilon = 0.5
            continue
            
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Load default configuration
    config_mgr.load_default_profile()
    
except GPUError as e:
    print(f"GPU error: {e}")
    # Fall back to CPU computation
    model.use_gpu = False
```

This API reference provides comprehensive documentation for all major components of the MFC Q-Learning Control System, enabling AI development agents to effectively understand, use, and extend the system capabilities.
