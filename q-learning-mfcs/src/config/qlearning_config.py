"""
Q-Learning configuration classes for MFC optimization.
Replaces hardcoded values in Q-learning controllers and optimization algorithms.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class QLearningRewardWeights:
    """Reward function weights for Q-learning optimization."""
    
    # Power optimization weights
    power_weight: float = 10.0  # Weight for power output maximization
    power_penalty_threshold: float = 50.0  # Penalty threshold for low power
    power_penalty_multiplier: float = 100.0  # Penalty multiplier for very low power
    power_base_reward_multiplier: float = 20.0  # Base reward multiplier
    
    # Substrate utilization weights  
    consumption_weight: float = 5.0  # Weight for substrate consumption
    substrate_reward_multiplier: float = 30.0  # Substrate utilization reward
    substrate_penalty_multiplier: float = 60.0  # Substrate waste penalty
    substrate_base_reward: float = 15.0  # Base substrate reward
    substrate_multiplier: float = 0.75  # Substrate efficiency multiplier
    
    # Substrate concentration control rewards
    substrate_target_reward: float = 50.0  # Reward for maintaining target concentrations
    substrate_excess_penalty: float = -100.0  # Penalty for exceeding max threshold
    substrate_starvation_penalty: float = -75.0  # Penalty for starvation conditions
    substrate_addition_penalty: float = -15.0  # Higher penalty per unit of substrate added (was -5.0)
    
    # Efficiency optimization weights
    efficiency_weight: float = 20.0  # Weight for substrate efficiency
    efficiency_threshold: float = 0.5  # Minimum efficiency threshold (50%)
    efficiency_penalty_multiplier: float = 100.0  # Penalty for low efficiency
    
    # Biofilm control weights
    biofilm_weight: float = 50.0  # Weight for biofilm penalty
    biofilm_reward: float = 25.0  # Reward for optimal biofilm
    biofilm_penalty_factor: float = 10.0  # Penalty factor for poor biofilm
    biofilm_steady_state_bonus: float = 15.0  # Bonus for steady state
    biofilm_penalty: float = -50.0  # Direct biofilm penalty
    
    # Combined penalties
    combined_penalty: float = -100.0  # Combined system failure penalty
    
    # Biofilm specific parameters for sensors
    biofilm_optimal_thickness_um: float = 30.0  # Optimal thickness for G. sulfurreducens
    conductivity_normalization_S_per_m: float = 0.005  # Normalization factor
    mass_growth_rate_factor: float = 10.0  # Growth rate normalization


@dataclass 
class QLearningConfig:
    """Comprehensive Q-learning configuration for MFC optimization."""
    
    # Core Q-learning parameters
    learning_rate: float = 0.1  # Alpha: learning rate (0 < α ≤ 1)
    discount_factor: float = 0.95  # Gamma: discount factor (0 ≤ γ ≤ 1)  
    epsilon: float = 0.3  # Exploration rate (0 ≤ ε ≤ 1)
    
    # Epsilon decay parameters (faster decay for substrate control)
    epsilon_decay: float = 0.9995  # Decay rate per step (faster decay)
    epsilon_min: float = 0.01  # Much lower minimum epsilon (1% vs 10%)
    
    # Alternative configurations for different controllers
    enhanced_learning_rate: float = 0.0987  # Enhanced controller specific
    enhanced_discount_factor: float = 0.9517  # Enhanced controller specific  
    enhanced_epsilon: float = 0.3702  # Enhanced controller specific
    
    # Advanced epsilon decay configurations
    advanced_epsilon_decay: float = 0.9978  # Advanced decay rate
    advanced_epsilon_min: float = 0.1020  # Advanced minimum epsilon
    
    # Reward weights configuration
    reward_weights: QLearningRewardWeights = field(default_factory=QLearningRewardWeights)
    
    # State space discretization
    power_bins: int = 10  # Number of bins for power state
    power_max: float = 2.0  # Maximum power for binning (W)
    
    biofilm_bins: int = 10  # Number of bins for biofilm state  
    biofilm_max: float = 1.0  # Maximum biofilm factor for binning
    
    substrate_bins: int = 10  # Number of bins for substrate state
    substrate_max: float = 50.0  # Maximum substrate concentration (mM)
    
    # Time-based state bins
    time_bins: List[int] = field(default_factory=lambda: [200, 500, 800, 1000])
    
    # Enhanced controller state space (sensor-integrated)
    eis_thickness_bins: int = 8  # EIS biofilm thickness bins
    eis_thickness_max: float = 80.0  # Maximum EIS thickness (μm)
    
    eis_conductivity_bins: int = 6  # EIS conductivity bins
    eis_conductivity_max: float = 0.01  # Maximum conductivity (S/m)
    
    eis_confidence_bins: int = 4  # EIS confidence level bins
    
    qcm_mass_bins: int = 8  # QCM mass bins
    qcm_mass_max: float = 1000.0  # Maximum QCM mass (ng/cm²)
    
    qcm_frequency_bins: int = 6  # QCM frequency shift bins
    qcm_frequency_max: float = 500.0  # Maximum frequency shift (Hz)
    
    qcm_dissipation_bins: int = 4  # QCM dissipation bins
    qcm_dissipation_max: float = 0.01  # Maximum dissipation
    
    fusion_confidence_bins: int = 5  # Sensor fusion confidence bins
    sensor_agreement_bins: int = 4  # Sensor agreement bins
    
    # Action space configuration
    flow_rate_actions: List[int] = field(default_factory=lambda: [-12, -10, -5, -2, -1, 0, 1, 2, 5, 6])
    substrate_actions: List[float] = field(default_factory=lambda: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    
    # Substrate control thresholds (configurable)
    substrate_target_reservoir: float = 20.0  # Target reservoir concentration (mM)
    substrate_target_outlet: float = 12.0  # Target outlet concentration (mM)
    substrate_target_cell: float = 15.0  # Target per-cell concentration (mM)
    substrate_max_threshold: float = 25.0  # Maximum allowed concentration (mM)
    substrate_min_threshold: float = 2.0  # Minimum starvation threshold (mM)
    substrate_addition_max: float = 5.0  # Maximum addition rate (mmol/h)
    
    # Unified controller action spaces
    unified_flow_actions: List[int] = field(default_factory=lambda: [-8, -4, -2, -1, 0, 1, 2, 3, 4])
    
    # Advanced state space configurations
    outlet_error_bins: int = 8  # Outlet error bins
    outlet_error_range: Tuple[float, float] = (-10.0, 10.0)  # Error range (mM)
    
    flow_rate_bins: int = 6  # Flow rate bins  
    flow_rate_range: Tuple[float, float] = (5.0, 50.0)  # Flow rate range (mL/h)
    
    # Sensor integration parameters
    sensor_weight: float = 0.3  # Weight for sensor data in decision making
    sensor_confidence_threshold: float = 0.3  # Minimum sensor confidence
    exploration_boost_factor: float = 1.5  # Boost exploration when confidence low
    
    # Performance thresholds and targets
    optimal_biofilm_thickness: float = 1.3  # Optimal biofilm thickness target
    biofilm_deviation_threshold: float = 0.05  # Allowable deviation from optimal
    growth_rate_threshold: float = 0.01  # Minimum growth rate for steady state
    
    # Flow rate optimization bounds
    flow_rate_min: float = 5.0  # Minimum flow rate (mL/h)
    flow_rate_max: float = 50.0  # Maximum flow rate (mL/h)
    
    # Stability targets
    stability_target_flow_rate: float = 15.0  # Target flow rate for stability (mL/h)
    stability_target_outlet_concentration: float = 12.0  # Target outlet concentration (mM)
    
    # Substrate concentration bounds
    substrate_concentration_min: float = 5.0  # mmol/L minimum
    substrate_concentration_max: float = 45.0  # mmol/L maximum
    
    # Action selection parameters
    flow_adjustment_unit: float = 1e-6  # Flow rate adjustment unit (m³/s)
    flow_conversion_factor: float = 3.6  # Conversion factor for flow units
    
    # Multi-objective optimization weights
    power_objective_weight: float = 0.4  # Power weight in multi-objective
    biofilm_health_weight: float = 0.3  # Biofilm health weight
    sensor_agreement_weight: float = 0.2  # Sensor agreement weight
    stability_weight: float = 0.1  # System stability weight
    
    # Reward weights configuration
    rewards: QLearningRewardWeights = field(default_factory=QLearningRewardWeights)
    
    # State space configuration
    state_space: 'StateSpaceConfig' = field(default_factory=lambda: StateSpaceConfig())


@dataclass
class StateSpaceConfig:
    """Configuration for Q-learning state space discretization."""
    
    # Basic state space
    power_bins: int = 8
    power_range: Tuple[float, float] = (0.0, 2.0)
    
    biofilm_bins: int = 6  
    biofilm_range: Tuple[float, float] = (0.0, 1.0)
    
    substrate_bins: int = 8
    substrate_range: Tuple[float, float] = (0.0, 50.0)
    
    # Enhanced state space with sensors
    sensor_state_bins: int = 12  # Combined sensor state bins
    
    # Substrate sensor state configuration  
    reservoir_substrate_bins: int = 8  # Reservoir substrate concentration bins
    reservoir_substrate_range: Tuple[float, float] = (0.0, 50.0)  # Range (mM)
    
    cell_substrate_bins: int = 6  # Per-cell substrate concentration bins
    cell_substrate_range: Tuple[float, float] = (0.0, 30.0)  # Range (mM)
    
    outlet_substrate_bins: int = 6  # Outlet substrate concentration bins
    outlet_substrate_range: Tuple[float, float] = (0.0, 25.0)  # Range (mM)
    
    # EIS sensor state configuration
    eis_thickness_bins: int = 8  # EIS biofilm thickness bins
    eis_thickness_max: float = 80.0  # Maximum EIS thickness (μm)
    eis_conductivity_bins: int = 6  # EIS conductivity bins
    eis_conductivity_max: float = 0.01  # Maximum conductivity (S/m)
    eis_confidence_bins: int = 4  # EIS confidence level bins
    
    # QCM sensor state configuration
    qcm_mass_bins: int = 8  # QCM mass bins
    qcm_mass_max: float = 1000.0  # Maximum QCM mass (ng/cm²)
    qcm_frequency_bins: int = 6  # QCM frequency bins
    qcm_frequency_max: float = 500.0  # Maximum frequency shift (Hz)
    qcm_dissipation_bins: int = 4  # QCM dissipation bins (from existing code)
    qcm_dissipation_max: float = 0.01  # Maximum dissipation factor (from sensor_config.py)
    
    # Sensor fusion state configuration
    sensor_agreement_bins: int = 4  # Sensor agreement bins (from existing code)
    fusion_confidence_bins: int = 5  # Sensor fusion confidence bins (from existing code)
    
    def get_total_states(self, include_sensors: bool = False) -> int:
        """Calculate total number of discrete states."""
        base_states = self.power_bins * self.biofilm_bins * self.substrate_bins
        if include_sensors:
            return base_states * self.sensor_state_bins
        return base_states


# Default configurations for different use cases
DEFAULT_QLEARNING_CONFIG = QLearningConfig()

ENHANCED_QLEARNING_CONFIG = QLearningConfig(
    learning_rate=0.0987,
    discount_factor=0.9517, 
    epsilon=0.3702,
    sensor_weight=0.3,
    sensor_confidence_threshold=0.3
)

CONSERVATIVE_QLEARNING_CONFIG = QLearningConfig(
    learning_rate=0.05,
    discount_factor=0.99,
    epsilon=0.1,
    epsilon_decay=0.999,
    epsilon_min=0.01
)

AGGRESSIVE_QLEARNING_CONFIG = QLearningConfig(
    learning_rate=0.3,
    discount_factor=0.9,
    epsilon=0.5, 
    epsilon_decay=0.99,
    epsilon_min=0.2
)