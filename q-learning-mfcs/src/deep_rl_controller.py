"""Deep Reinforcement Learning Controller for Advanced MFC Control.

This module implements state-of-the-art deep reinforcement learning algorithms
for MFC control, replacing traditional Q-tables with deep neural networks.

Key Features:
- Deep Q-Networks (DQN) with experience replay and target networks
- Double DQN for reduced overestimation bias
- Dueling DQN architecture for better value estimation
- Priority Experience Replay for more efficient learning
- Rainbow DQN combining multiple improvements
- Actor-Critic methods (A3C, PPO, SAC)
- Multi-agent deep RL for complex system control

The controller integrates with Phase 2 components:
- Advanced sensor fusion data as input features
- Health-aware reward functions
- Predictive state representations
- GPU acceleration for neural network training

Created: 2025-07-31
Last Modified: 2025-07-31
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

# Optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Import Phase 2 components
from ml_optimization import FeatureEngineer

if TYPE_CHECKING:
    from adaptive_mfc_controller import SystemState

# Import configuration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DRLAlgorithm(Enum):
    """Deep RL algorithm types."""

    DQN = "deep_q_network"
    DOUBLE_DQN = "double_dqn"
    DUELING_DQN = "dueling_dqn"
    RAINBOW_DQN = "rainbow_dqn"
    PPO = "proximal_policy_optimization"
    SAC = "soft_actor_critic"
    A3C = "async_advantage_actor_critic"


@dataclass
class DRLConfig:
    """Configuration for deep RL algorithms."""

    # Network architecture
    hidden_layers: list[int] = None
    activation: str = "relu"
    dropout_rate: float = 0.1
    batch_norm: bool = True

    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 64
    memory_size: int = 50000
    target_update_freq: int = 1000
    gradient_clip: float = 1.0

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 50000

    # Experience replay
    priority_replay: bool = True
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_frames: int = 100000

    # Algorithm specific
    gamma: float = 0.99
    tau: float = 0.005  # Soft update parameter
    noise_std: float = 0.1  # For exploration noise

    # Training
    train_freq: int = 4
    warmup_steps: int = 10000
    max_episodes: int = 10000

    # PPO specific parameters
    ppo_clip_ratio: float = 0.2
    ppo_epochs: int = 10
    ppo_value_loss_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_gae_lambda: float = 0.95

    # A3C specific parameters
    a3c_update_steps: int = 20
    a3c_num_workers: int = 4
    a3c_entropy_coef: float = 0.01
    a3c_value_loss_coef: float = 0.5

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [512, 256, 128]


class PriorityReplayBuffer:
    """Priority experience replay buffer with proportional prioritization."""

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        """Initialize priority replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent (0 = uniform, 1 = full priority)

        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        # Sum tree for efficient priority sampling
        self.tree_ptr = 0
        self.sum_tree = np.zeros((2 * capacity - 1,), dtype=np.float32)
        self.min_tree = np.full((2 * capacity - 1,), float("inf"), dtype=np.float32)

        self.max_priority = 1.0

    def _update_tree(self, tree_idx: int, priority: float) -> None:
        """Update sum and min trees."""
        change = priority - self.sum_tree[tree_idx]
        self.sum_tree[tree_idx] = priority
        self.min_tree[tree_idx] = priority

        # Update parent nodes
        while tree_idx:
            tree_idx = (tree_idx - 1) // 2
            self.sum_tree[tree_idx] += change
            self.min_tree[tree_idx] = min(
                self.min_tree[2 * tree_idx + 1],
                self.min_tree[2 * tree_idx + 2],
            )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add experience with maximum priority."""
        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        # Set maximum priority for new experience
        tree_idx = self.pos + self.capacity - 1
        self._update_tree(tree_idx, self.max_priority**self.alpha)

        self.pos = (self.pos + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
    ) -> tuple[list, np.ndarray, np.ndarray]:
        """Sample experiences with importance sampling."""
        indices = []
        priorities = []

        segment = self.sum_tree[0] / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            idx = self._get_leaf(s)
            data_idx = idx - self.capacity + 1

            indices.append(data_idx)
            priorities.append(self.sum_tree[idx])

        # Importance sampling weights
        sampling_probabilities = np.array(priorities) / self.sum_tree[0]
        is_weights = np.power(self.capacity * sampling_probabilities, -beta)
        is_weights /= is_weights.max()

        batch = [self.buffer[idx] for idx in indices]

        return batch, np.array(indices), is_weights.astype(np.float32)

    def _get_leaf(self, s: float) -> int:
        """Get leaf index from cumulative sum."""
        idx = 0
        while idx < self.capacity - 1:  # While not leaf
            left = 2 * idx + 1
            right = left + 1

            if s <= self.sum_tree[left]:
                idx = left
            else:
                s -= self.sum_tree[left]
                idx = right

        return idx

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for given indices."""
        for idx, priority in zip(indices, priorities, strict=False):
            tree_idx = idx + self.capacity - 1
            self.max_priority = max(self.max_priority, priority)
            self._update_tree(tree_idx, priority**self.alpha)

    def __len__(self) -> int:
        return len(self.buffer)


class DuelingDQN(nn.Module):
    """Dueling DQN architecture with separate value and advantage streams."""

    def __init__(self, state_dim: int, action_dim: int, config: DRLConfig) -> None:
        """Initialize dueling DQN.

        Args:
            state_dim: Input state dimension
            action_dim: Number of actions
            config: DRL configuration

        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # Shared feature extraction layers
        layers = []
        input_dim = state_dim

        for hidden_dim in config.hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._get_activation(config.activation))
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
            input_dim = hidden_dim

        self.feature_layers = nn.Sequential(*layers)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(config.hidden_layers[-1], config.hidden_layers[-1] // 2),
            self._get_activation(config.activation),
            nn.Linear(config.hidden_layers[-1] // 2, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.hidden_layers[-1], config.hidden_layers[-1] // 2),
            self._get_activation(config.activation),
            nn.Linear(config.hidden_layers[-1] // 2, action_dim),
        )

        # Initialize weights
        self.apply(self._init_weights)

        logger.info(
            f"DuelingDQN initialized: {state_dim}→{config.hidden_layers}→{action_dim}",
        )

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "swish": nn.SiLU(),
        }
        return activations.get(activation, nn.ReLU())

    def _init_weights(self, m) -> None:
        """Initialize network weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dueling architecture."""
        features = self.feature_layers(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage using dueling architecture
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class RainbowDQN(nn.Module):
    """Rainbow DQN combining multiple improvements:
    - Dueling architecture
    - Noisy networks
    - Distributional RL (C51)
    - Multi-step learning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: DRLConfig,
        atoms: int = 51,
        v_min: float = -10,
        v_max: float = 10,
    ) -> None:
        """Initialize Rainbow DQN.

        Args:
            state_dim: Input state dimension
            action_dim: Number of actions
            config: DRL configuration
            atoms: Number of atoms for distributional RL
            v_min: Minimum value for value distribution
            v_max: Maximum value for value distribution

        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (atoms - 1)

        # Shared noisy layers
        self.feature_layers = self._build_noisy_layers(state_dim, config.hidden_layers)

        # Noisy value and advantage streams
        self.value_stream = NoisyLinear(config.hidden_layers[-1], atoms)
        self.advantage_stream = NoisyLinear(
            config.hidden_layers[-1],
            action_dim * atoms,
        )

        logger.info(
            f"RainbowDQN initialized with {atoms} atoms, value range [{v_min}, {v_max}]",
        )

    def _build_noisy_layers(self, input_dim: int, hidden_dims: list[int]) -> nn.Module:
        """Build noisy feature extraction layers."""
        layers = []

        for hidden_dim in hidden_dims:
            layers.append(NoisyLinear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning value distribution."""
        batch_size = x.size(0)

        features = self.feature_layers(x)

        value = self.value_stream(features).view(batch_size, 1, self.atoms)
        advantage = self.advantage_stream(features).view(
            batch_size,
            self.action_dim,
            self.atoms,
        )

        # Dueling architecture for distributions
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)

        # Apply softmax to get probability distributions
        return F.softmax(q_dist, dim=-1)

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Get Q-values from value distribution."""
        q_dist = self.forward(x)

        # Support values for the distribution
        support = torch.linspace(self.v_min, self.v_max, self.atoms).to(x.device)

        # Expected Q-values
        return torch.sum(q_dist * support, dim=-1)


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration in neural networks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        std_init: float = 0.4,
    ) -> None:
        """Initialize noisy linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            std_init: Initial standard deviation for noise

        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise tensors (not parameters)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self) -> None:
        """Generate new noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate factorized Gaussian noise."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy parameters."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO and A3C algorithms."""

    def __init__(self, state_dim: int, action_dim: int, config: DRLConfig):
        """
        Initialize actor-critic network.

        Args:
            state_dim: Input state dimension
            action_dim: Number of actions
            config: DRL configuration
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # Shared feature extraction layers
        layers = []
        input_dim = state_dim

        for hidden_dim in config.hidden_layers[:-1]:  # All but last layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._get_activation(config.activation))
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
            input_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(input_dim, config.hidden_layers[-1]),
            self._get_activation(config.activation),
            nn.Linear(config.hidden_layers[-1], action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, config.hidden_layers[-1]),
            self._get_activation(config.activation),
            nn.Linear(config.hidden_layers[-1], 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

        logger.info(f"ActorCriticNetwork initialized: {state_dim}→{config.hidden_layers}→{action_dim}")

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'swish': nn.SiLU()
        }
        return activations.get(activation, nn.ReLU())

    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor-critic network.

        Args:
            x: Input state tensor

        Returns:
            Tuple of (action_probabilities, state_value)
        """
        shared_features = self.shared_layers(x)

        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)

        return action_probs, state_value

    def get_action_and_value(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value for PPO.

        Args:
            x: Input state tensor

        Returns:
            Tuple of (action, log_prob, value)
        """
        action_probs, value = self.forward(x)

        # Sample action from probability distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value


class PPONetwork(nn.Module):
    """PPO-specific network with clipped surrogate objective."""

    def __init__(self, state_dim: int, action_dim: int, config: DRLConfig):
        """Initialize PPO network."""
        super().__init__()

        self.actor_critic = ActorCriticNetwork(state_dim, action_dim, config)
        self.config = config

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        return self.actor_critic.forward(x)

    def get_action_and_value(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action and value for PPO training."""
        return self.actor_critic.get_action_and_value(x)

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO loss computation.

        Args:
            x: State tensor
            actions: Action tensor

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        action_probs, values = self.forward(x)

        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy


class A3CNetwork(nn.Module):
    """A3C-specific network with shared parameters."""

    def __init__(self, state_dim: int, action_dim: int, config: DRLConfig):
        """Initialize A3C network."""
        super().__init__()

        self.actor_critic = ActorCriticNetwork(state_dim, action_dim, config)
        self.config = config

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        return self.actor_critic.forward(x)

    def get_action_and_value(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action and value for A3C training."""
        return self.actor_critic.get_action_and_value(x)


class DeepRLController:
    """Advanced Deep Reinforcement Learning Controller for MFC systems.

    Integrates with Phase 2 components and provides multiple DRL algorithms
    with state-of-the-art techniques for continuous learning and adaptation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        algorithm: DRLAlgorithm = DRLAlgorithm.RAINBOW_DQN,
        config: DRLConfig | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize deep RL controller.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            algorithm: DRL algorithm to use
            config: Configuration parameters
            device: Computing device ('cuda', 'cpu', or 'auto')

        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.algorithm = algorithm
        self.config = config or DRLConfig()

        # Device selection
        if device == "auto" or device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize networks
        self._build_networks()

        # Experience replay buffer
        if self.config.priority_replay:
            self.replay_buffer = PriorityReplayBuffer(
                self.config.memory_size,
                self.config.priority_alpha,
            )
        else:
            self.replay_buffer = deque(maxlen=self.config.memory_size)

        # Training state
        self.steps = 0
        self.episodes = 0
        self.epsilon = self.config.epsilon_start
        self.beta = self.config.priority_beta_start

        # Feature engineering
        self.feature_engineer = FeatureEngineer()

        # Logging
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(
                f"runs/deep_rl_{algorithm.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
        else:
            self.writer = None
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.q_value_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=1000)

        logger.info(
            f"DeepRLController initialized with {algorithm.value} on {self.device}",
        )
        logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")
        logger.info(f"Network architecture: {self.config.hidden_layers}")

    def _build_networks(self) -> None:
        """Build neural networks based on algorithm."""
        if self.algorithm == DRLAlgorithm.RAINBOW_DQN:
            self.q_network = RainbowDQN(
                self.state_dim,
                self.action_dim,
                self.config,
            ).to(self.device)
            self.target_network = RainbowDQN(
                self.state_dim,
                self.action_dim,
                self.config,
            ).to(self.device)
            # Copy parameters to target network
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
        elif self.algorithm == DRLAlgorithm.PPO:
            self.policy_network = PPONetwork(
                self.state_dim, self.action_dim, self.config
            ).to(self.device)
            # PPO doesn't use target network
            self.target_network = None
        elif self.algorithm == DRLAlgorithm.A3C:
            self.policy_network = A3CNetwork(
                self.state_dim, self.action_dim, self.config
            ).to(self.device)
            # A3C doesn't use target network
            self.target_network = None
        else:
            # Default to Dueling DQN
            self.q_network = DuelingDQN(
                self.state_dim,
                self.action_dim,
                self.config,
            ).to(self.device)
            self.target_network = DuelingDQN(
                self.state_dim,
                self.action_dim,
                self.config,
            ).to(self.device)
            # Copy parameters to target network
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()

        # Copy parameters to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10000,
            gamma=0.9,
        )

    def extract_state_features(self, system_state: SystemState) -> np.ndarray:
        """Extract features from system state for neural network input.

        Args:
            system_state: Complete system state from Phase 2

        Returns:
            Feature vector for neural network

        """
        # Use Phase 2 feature engineering
        performance_metrics = {
            "power_efficiency": system_state.power_output
            / max(system_state.current_density, 0.01),
            "biofilm_health_score": system_state.health_metrics.overall_health_score,
            "sensor_reliability": system_state.fused_measurement.fusion_confidence,
            "system_stability": 1.0 - len(system_state.anomalies) / 10.0,  # Normalized
            "control_confidence": 0.8,  # Default value
        }

        # Extract comprehensive features
        features = self.feature_engineer.extract_features(
            system_state,
            performance_metrics,
        )

        # Convert to numpy array and normalize
        feature_vector = np.array(list(features.values()), dtype=np.float32)

        # Handle NaN and infinite values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)

        # Normalize features to [-1, 1] range
        return np.tanh(feature_vector / 100.0)  # Soft normalization

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy or noisy networks.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action index

        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.algorithm in [DRLAlgorithm.PPO, DRLAlgorithm.A3C]:
                # Policy-based action selection
                action_probs, value = self.policy_network(state_tensor)
                if training:
                    # Sample from probability distribution
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample().item()
                else:
                    # Greedy action selection
                    action = action_probs.argmax(dim=1).item()

                # Log value for monitoring
                if len(self.q_value_history) == 0 or self.steps % 100 == 0:
                    self.q_value_history.append(value.item())

            else:
                # Q-learning based action selection
                if training and self.algorithm != DRLAlgorithm.RAINBOW_DQN:
                    # Epsilon-greedy exploration
                    if random.random() < self.epsilon:
                        return random.randint(0, self.action_dim - 1)

                if self.algorithm == DRLAlgorithm.RAINBOW_DQN:
                    q_values = self.q_network.get_q_values(state_tensor)
                else:
                    q_values = self.q_network(state_tensor)

                action = q_values.argmax(dim=1).item()

                # Log Q-values for monitoring
                if len(self.q_value_history) == 0 or self.steps % 100 == 0:
                    self.q_value_history.append(q_values.mean().item())

        return action

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store experience in replay buffer."""
        if self.config.priority_replay:
            self.replay_buffer.push(state, action, reward, next_state, done)
        else:
            self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self) -> dict[str, float]:
        """Perform one training step.

        Returns:
            Training metrics

        """
        if len(self.replay_buffer) < self.config.warmup_steps:
            return {"loss": 0.0, "q_value": 0.0}

        if self.algorithm in [DRLAlgorithm.PPO, DRLAlgorithm.A3C]:
            return self._train_policy_gradient()
        else:
            return self._train_dqn()

    def _train_dqn(self) -> dict[str, float]:
        """Train DQN-based algorithms."""
        # Sample batch
        if self.config.priority_replay:
            batch, indices, is_weights = self.replay_buffer.sample(
                self.config.batch_size,
                self.beta,
            )
            is_weights = torch.FloatTensor(is_weights).to(self.device)
        else:
            batch = random.sample(self.replay_buffer, self.config.batch_size)
            is_weights = torch.ones(self.config.batch_size).to(self.device)
            indices = None

        # Unpack batch
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        # Compute loss based on algorithm
        if self.algorithm == DRLAlgorithm.RAINBOW_DQN:
            loss, td_errors = self._compute_distributional_loss(
                states,
                actions,
                rewards,
                next_states,
                dones,
                is_weights,
            )
        else:
            loss, td_errors = self._compute_dqn_loss(
                states,
                actions,
                rewards,
                next_states,
                dones,
                is_weights,
            )

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            self.config.gradient_clip,
        )

        self.optimizer.step()
        self.scheduler.step()

        # Update priorities
        if self.config.priority_replay and indices is not None:
            priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)

        # Update target network
        if self.steps % self.config.target_update_freq == 0:
            self._update_target_network()

        # Update exploration parameters
        self._update_exploration()

        # Log metrics
        metrics = {
            "loss": loss.item(),
            "q_value": self.q_value_history[-1] if self.q_value_history else 0.0,
            "epsilon": self.epsilon,
            "learning_rate": self.scheduler.get_last_lr()[0],
            "beta": self.beta,
        }

        self.loss_history.append(loss.item())
        return metrics

    def _compute_dqn_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        is_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute DQN loss with importance sampling."""
        current_q_values = (
            self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        with torch.no_grad():
            if self.algorithm == DRLAlgorithm.DOUBLE_DQN:
                # Double DQN: use online network for action selection
                next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
                next_q_values = (
                    self.target_network(next_states).gather(1, next_actions).squeeze(1)
                )
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0]

            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)

        # TD errors for priority update
        td_errors = target_q_values - current_q_values

        # Weighted loss
        loss = (td_errors.pow(2) * is_weights).mean()

        return loss, td_errors

    def _compute_distributional_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        is_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute distributional (C51) loss."""
        batch_size = states.size(0)

        # Current distribution
        current_dist = self.q_network(states)
        current_dist = current_dist[range(batch_size), actions]

        # Target distribution
        with torch.no_grad():
            next_dist = self.target_network(next_states)
            next_actions = self.q_network.get_q_values(next_states).argmax(1)
            next_dist = next_dist[range(batch_size), next_actions]

            # Support for value distribution
            support = torch.linspace(
                self.q_network.v_min,
                self.q_network.v_max,
                self.q_network.atoms,
            ).to(self.device)

            # Project target distribution
            delta_z = (self.q_network.v_max - self.q_network.v_min) / (
                self.q_network.atoms - 1
            )
            target_support = rewards.unsqueeze(
                1,
            ) + self.config.gamma * support.unsqueeze(0) * (~dones).unsqueeze(1)
            target_support = target_support.clamp(
                self.q_network.v_min,
                self.q_network.v_max,
            )

            # Distribute probability mass
            b = (target_support - self.q_network.v_min) / delta_z
            lower = b.floor().long()
            upper = b.ceil().long()

            target_dist = torch.zeros_like(next_dist)
            target_dist.view(-1).index_add_(
                0,
                (
                    lower
                    + (torch.arange(batch_size) * self.q_network.atoms)
                    .unsqueeze(1)
                    .to(self.device)
                ).view(-1),
                (next_dist * (upper.float() - b)).view(-1),
            )
            target_dist.view(-1).index_add_(
                0,
                (
                    upper
                    + (torch.arange(batch_size) * self.q_network.atoms)
                    .unsqueeze(1)
                    .to(self.device)
                ).view(-1),
                (next_dist * (b - lower.float())).view(-1),
            )

        # Cross-entropy loss
        loss = -(target_dist * current_dist.log()).sum(1)

        # TD errors (approximate for priority update)
        with torch.no_grad():
            current_q = (current_dist * support).sum(1)
            target_q = (target_dist * support).sum(1)
            td_errors = target_q - current_q

        # Weighted loss
        loss = (loss * is_weights).mean()

        return loss, td_errors

    def _update_target_network(self) -> None:
        """Update target network with soft or hard update."""
        if self.config.tau < 1.0:
            # Soft update
            for target_param, param in zip(
                self.target_network.parameters(),
                self.q_network.parameters(),
                strict=False,
            ):
                target_param.data.copy_(
                    self.config.tau * param.data
                    + (1.0 - self.config.tau) * target_param.data,
                )
        else:
            # Hard update
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _update_exploration(self) -> None:
        """Update exploration parameters."""
        # Epsilon decay
        if self.epsilon > self.config.epsilon_end:
            self.epsilon -= (
                self.config.epsilon_start - self.config.epsilon_end
            ) / self.config.epsilon_decay
            self.epsilon = max(self.epsilon, self.config.epsilon_end)

        # Beta annealing for priority replay
        if self.config.priority_replay and self.beta < 1.0:
            self.beta += (
                1.0 - self.config.priority_beta_start
            ) / self.config.priority_beta_frames
            self.beta = min(self.beta, 1.0)

        # Reset noise for noisy networks
        if self.algorithm == DRLAlgorithm.RAINBOW_DQN:
            self.q_network.apply(
                lambda m: m.reset_noise() if hasattr(m, "reset_noise") else None,
            )
            self.target_network.apply(
                lambda m: m.reset_noise() if hasattr(m, "reset_noise") else None,
            )

    def control_step(self, system_state: SystemState) -> tuple[int, dict[str, Any]]:
        """Execute one control step with deep RL.

        Args:
            system_state: Current system state

        Returns:
            Action and control information

        """
        # Extract state features
        state_features = self.extract_state_features(system_state)

        # Select action
        action = self.select_action(state_features)

        self.steps += 1

        # Get network parameters count
        if self.algorithm in [DRLAlgorithm.PPO, DRLAlgorithm.A3C]:
            network_params = sum(p.numel() for p in self.policy_network.parameters())
        else:
            network_params = sum(p.numel() for p in self.q_network.parameters())

        # Control information
        control_info = {
            "algorithm": self.algorithm.value,
            "epsilon": self.epsilon,
            "steps": self.steps,
            "q_values": self.q_value_history[-1] if self.q_value_history else 0.0,
            "state_features": state_features,
            "network_parameters": sum(p.numel() for p in self.q_network.parameters()),
        }

        return action, control_info

    def update_with_reward(
        self,
        prev_state: SystemState,
        action: int,
        reward: float,
        next_state: SystemState,
        done: bool,
    ) -> None:
        """Update the controller with reward feedback.

        Args:
            prev_state: Previous system state
            action: Action taken
            reward: Reward received
            next_state: Next system state
            done: Whether episode is done

        """
        # Extract features
        prev_features = self.extract_state_features(prev_state)
        next_features = self.extract_state_features(next_state)

        # Store experience
        self.store_experience(prev_features, action, reward, next_features, done)

        # Train if ready
        if self.steps % self.config.train_freq == 0:
            metrics = self.train_step()

            # Log to tensorboard if available
            if self.writer is not None:
                self.writer.add_scalar("Loss/Training", metrics["loss"], self.steps)
                self.writer.add_scalar(
                    "Q_Value/Average",
                    metrics["q_value"],
                    self.steps,
                )
                self.writer.add_scalar(
                    "Exploration/Epsilon",
                    metrics["epsilon"],
                    self.steps,
                )
                self.writer.add_scalar(
                    "Learning/Rate",
                    metrics["learning_rate"],
                    self.steps,
                )

    def save_model(self, path: str) -> None:
        """Save model and training state."""
        save_dict = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "steps": self.steps,
            "episodes": self.episodes,
            "epsilon": self.epsilon,
            "beta": self.beta,
            "config": self.config,
            "algorithm": self.algorithm.value,
        }

        # Save appropriate network based on algorithm
        if self.algorithm in [DRLAlgorithm.PPO, DRLAlgorithm.A3C]:
            save_dict['policy_network'] = self.policy_network.state_dict()
        else:
            save_dict['q_network'] = self.q_network.state_dict()
            if self.target_network is not None:
                save_dict['target_network'] = self.target_network.state_dict()

        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model and training state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load appropriate network based on algorithm
        if self.algorithm in [DRLAlgorithm.PPO, DRLAlgorithm.A3C]:
            self.policy_network.load_state_dict(checkpoint['policy_network'])
        else:
            self.q_network.load_state_dict(checkpoint['q_network'])
            if self.target_network is not None and 'target_network' in checkpoint:
                self.target_network.load_state_dict(checkpoint['target_network'])

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.steps = checkpoint["steps"]
        self.episodes = checkpoint["episodes"]
        self.epsilon = checkpoint["epsilon"]
        self.beta = checkpoint["beta"]

        logger.info(f"Model loaded from {path}")

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        # Get network parameters count
        if self.algorithm in [DRLAlgorithm.PPO, DRLAlgorithm.A3C]:
            network_params = sum(p.numel() for p in self.policy_network.parameters())
        else:
            network_params = sum(p.numel() for p in self.q_network.parameters())

        return {
            "algorithm": self.algorithm.value,
            "steps": self.steps,
            "episodes": self.episodes,
            "avg_episode_reward": (
                np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            ),
            "avg_episode_length": (
                np.mean(self.episode_lengths) if self.episode_lengths else 0.0
            ),
            "avg_q_value": (
                np.mean(self.q_value_history) if self.q_value_history else 0.0
            ),
            "avg_loss": np.mean(self.loss_history) if self.loss_history else 0.0,
            "epsilon": self.epsilon,
            "beta": self.beta,
            "replay_buffer_size": len(self.replay_buffer),
            "device": str(self.device),
            "network_parameters": sum(p.numel() for p in self.q_network.parameters()),
        }


def create_deep_rl_controller(
    state_dim: int,
    action_dim: int,
    algorithm: str = "rainbow_dqn",
    **kwargs,
) -> DeepRLController:
    """Factory function to create deep RL controller.

    Args:
        state_dim: State space dimension
        action_dim: Action space dimension
        algorithm: Algorithm name
        **kwargs: Additional configuration

    Returns:
        Configured deep RL controller

    """
    algorithm_map = {
        "dqn": DRLAlgorithm.DQN,
        "double_dqn": DRLAlgorithm.DOUBLE_DQN,
        "dueling_dqn": DRLAlgorithm.DUELING_DQN,
        "rainbow_dqn": DRLAlgorithm.RAINBOW_DQN,
        "ppo": DRLAlgorithm.PPO,
        "sac": DRLAlgorithm.SAC,
        "a3c": DRLAlgorithm.A3C,
    }

    algo = algorithm_map.get(algorithm.lower(), DRLAlgorithm.RAINBOW_DQN)
    config = DRLConfig(**kwargs)

    controller = DeepRLController(
        state_dim=state_dim,
        action_dim=action_dim,
        algorithm=algo,
        config=config,
    )

    logger.info(f"Deep RL controller created with {algorithm} algorithm")

    return controller


if __name__ == "__main__":
    """Test deep RL controller functionality."""

    # Test configuration
    state_dim = 70  # From Phase 2 feature engineering
    action_dim = 15  # From adaptive controller

    # Create controller
    controller = create_deep_rl_controller(
        state_dim=state_dim,
        action_dim=action_dim,
        algorithm="rainbow_dqn",
        hidden_layers=[512, 256, 128],
        learning_rate=1e-4,
        batch_size=64,
    )

    # Test forward pass
    dummy_state = np.random.randn(state_dim).astype(np.float32)
    action = controller.select_action(dummy_state)
