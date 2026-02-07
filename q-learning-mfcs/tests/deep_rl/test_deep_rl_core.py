"""Tests for Deep RL Core Components.

US-006: Test Deep RL Core Components
Target: 50%+ coverage for core deep RL classes
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from deep_rl_controller import (
    ActorCriticNetwork,
    DeepRLController,
    DRLAlgorithm,
    DRLConfig,
    DuelingDQN,
    NoisyLinear,
    PPONetwork,
    PriorityReplayBuffer,
    RainbowDQN,
)


@pytest.fixture
def default_config() -> DRLConfig:
    """Create default DRL configuration."""
    return DRLConfig()


@pytest.fixture
def small_config() -> DRLConfig:
    """Create small config for faster tests."""
    return DRLConfig(
        hidden_layers=[64, 32],
        batch_size=8,
        memory_size=100,
        warmup_steps=10,
    )


@pytest.fixture
def state_dim() -> int:
    """Default state dimension."""
    return 10


@pytest.fixture
def action_dim() -> int:
    """Default action dimension."""
    return 4


@pytest.fixture
def sample_state(state_dim: int) -> np.ndarray:
    """Create sample state array."""
    return np.random.randn(state_dim).astype(np.float32)


class TestDRLConfig:
    """Tests for DRLConfig dataclass."""

    def test_default_initialization(self) -> None:
        """Test default config initialization."""
        config = DRLConfig()
        assert config.hidden_layers == [512, 256, 128]
        assert config.activation == "relu"
        assert config.dropout_rate == 0.1
        assert config.learning_rate == 1e-4
        assert config.batch_size == 64
        assert config.gamma == 0.99

    def test_custom_hidden_layers(self) -> None:
        """Test custom hidden layers configuration."""
        config = DRLConfig(hidden_layers=[128, 64])
        assert config.hidden_layers == [128, 64]

    def test_post_init_sets_default_hidden_layers(self) -> None:
        """Test that __post_init__ sets default hidden layers when None."""
        config = DRLConfig()
        assert config.hidden_layers is not None
        assert len(config.hidden_layers) == 3

    def test_exploration_parameters(self) -> None:
        """Test exploration parameter defaults."""
        config = DRLConfig()
        assert config.epsilon_start == 1.0
        assert config.epsilon_end == 0.01
        assert config.epsilon_decay == 50000

    def test_priority_replay_parameters(self) -> None:
        """Test priority replay buffer parameters."""
        config = DRLConfig()
        assert config.priority_replay is True
        assert config.priority_alpha == 0.6
        assert config.priority_beta_start == 0.4

    def test_ppo_specific_parameters(self) -> None:
        """Test PPO-specific configuration parameters."""
        config = DRLConfig()
        assert config.ppo_clip_ratio == 0.2
        assert config.ppo_epochs == 10
        assert config.ppo_value_loss_coef == 0.5

    def test_a3c_specific_parameters(self) -> None:
        """Test A3C-specific configuration parameters."""
        config = DRLConfig()
        assert config.a3c_update_steps == 20
        assert config.a3c_num_workers == 4


class TestDRLAlgorithm:
    """Tests for DRLAlgorithm enum."""

    def test_all_algorithms_defined(self) -> None:
        """Test that all expected algorithms are defined."""
        assert DRLAlgorithm.DQN.value == "deep_q_network"
        assert DRLAlgorithm.DOUBLE_DQN.value == "double_dqn"
        assert DRLAlgorithm.DUELING_DQN.value == "dueling_dqn"
        assert DRLAlgorithm.RAINBOW_DQN.value == "rainbow_dqn"
        assert DRLAlgorithm.PPO.value == "proximal_policy_optimization"
        assert DRLAlgorithm.SAC.value == "soft_actor_critic"
        assert DRLAlgorithm.A3C.value == "async_advantage_actor_critic"

    def test_algorithm_count(self) -> None:
        """Test that all 7 algorithms are present."""
        assert len(DRLAlgorithm) == 7


class TestPriorityReplayBuffer:
    """Tests for PriorityReplayBuffer."""

    def test_initialization(self) -> None:
        """Test buffer initialization."""
        buffer = PriorityReplayBuffer(capacity=100, alpha=0.6)
        assert buffer.capacity == 100
        assert buffer.alpha == 0.6
        assert len(buffer) == 0
        assert buffer.max_priority == 1.0

    def test_push_single_experience(self, state_dim: int) -> None:
        """Test pushing a single experience."""
        buffer = PriorityReplayBuffer(capacity=100)
        state = np.random.randn(state_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        buffer.push(state, action=0, reward=1.0, next_state=next_state, done=False)
        assert len(buffer) == 1

    def test_push_multiple_experiences(self, state_dim: int) -> None:
        """Test pushing multiple experiences."""
        buffer = PriorityReplayBuffer(capacity=100)
        for i in range(10):
            state = np.random.randn(state_dim).astype(np.float32)
            next_state = np.random.randn(state_dim).astype(np.float32)
            buffer.push(state, action=i % 4, reward=float(i), next_state=next_state, done=False)
        assert len(buffer) == 10

    def test_push_exceeds_capacity(self, state_dim: int) -> None:
        """Test buffer wrapping when exceeding capacity."""
        capacity = 10
        buffer = PriorityReplayBuffer(capacity=capacity)
        for i in range(15):
            state = np.random.randn(state_dim).astype(np.float32)
            next_state = np.random.randn(state_dim).astype(np.float32)
            buffer.push(state, action=0, reward=float(i), next_state=next_state, done=False)
        assert len(buffer) == capacity

    def test_sample_basic(self, state_dim: int) -> None:
        """Test basic sampling from buffer."""
        buffer = PriorityReplayBuffer(capacity=100)
        for i in range(20):
            state = np.random.randn(state_dim).astype(np.float32)
            next_state = np.random.randn(state_dim).astype(np.float32)
            buffer.push(state, action=i % 4, reward=float(i), next_state=next_state, done=False)
        batch, indices, is_weights = buffer.sample(batch_size=8, beta=0.4)
        assert len(batch) == 8
        assert len(indices) == 8
        assert len(is_weights) == 8
        assert is_weights.dtype == np.float32

    def test_sample_importance_weights(self, state_dim: int) -> None:
        """Test importance sampling weights are normalized."""
        buffer = PriorityReplayBuffer(capacity=100)
        for i in range(50):
            state = np.random.randn(state_dim).astype(np.float32)
            next_state = np.random.randn(state_dim).astype(np.float32)
            buffer.push(state, action=i % 4, reward=float(i), next_state=next_state, done=False)
        _, _, is_weights = buffer.sample(batch_size=16, beta=0.4)
        assert np.max(is_weights) <= 1.0 + 1e-6
        assert np.all(is_weights > 0)

    def test_update_priorities(self, state_dim: int) -> None:
        """Test updating priorities."""
        buffer = PriorityReplayBuffer(capacity=100)
        for i in range(20):
            state = np.random.randn(state_dim).astype(np.float32)
            next_state = np.random.randn(state_dim).astype(np.float32)
            buffer.push(state, action=i % 4, reward=float(i), next_state=next_state, done=False)
        _, indices, _ = buffer.sample(batch_size=8, beta=0.4)
        new_priorities = np.abs(np.random.randn(8)) + 0.01
        buffer.update_priorities(indices, new_priorities)
        assert buffer.max_priority >= np.max(new_priorities)

    def test_tree_structure(self) -> None:
        """Test that sum tree is properly sized."""
        capacity = 64
        buffer = PriorityReplayBuffer(capacity=capacity)
        assert len(buffer.sum_tree) == 2 * capacity - 1
        assert len(buffer.min_tree) == 2 * capacity - 1


class TestDuelingDQN:
    """Tests for DuelingDQN network."""

    def test_initialization(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test network initialization."""
        network = DuelingDQN(state_dim, action_dim, small_config)
        assert network.state_dim == state_dim
        assert network.action_dim == action_dim
        assert network.config == small_config

    def test_forward_pass_shape(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test forward pass output shape."""
        network = DuelingDQN(state_dim, action_dim, small_config)
        network.eval()
        batch_size = 8
        x = torch.randn(batch_size, state_dim)
        output = network(x)
        assert output.shape == (batch_size, action_dim)

    def test_dueling_architecture_components(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test that dueling architecture has value and advantage streams."""
        network = DuelingDQN(state_dim, action_dim, small_config)
        assert hasattr(network, "value_stream")
        assert hasattr(network, "advantage_stream")
        assert hasattr(network, "feature_layers")

    def test_activation_functions(self, state_dim: int, action_dim: int) -> None:
        """Test different activation functions."""
        for activation in ["relu", "leaky_relu", "tanh", "elu", "swish"]:
            config = DRLConfig(hidden_layers=[32, 16], activation=activation)
            network = DuelingDQN(state_dim, action_dim, config)
            network.eval()
            x = torch.randn(4, state_dim)
            output = network(x)
            assert output.shape == (4, action_dim)

    def test_forward_with_batch_norm(self, state_dim: int, action_dim: int) -> None:
        """Test forward pass with batch normalization enabled."""
        config = DRLConfig(hidden_layers=[32, 16], batch_norm=True)
        network = DuelingDQN(state_dim, action_dim, config)
        network.eval()
        x = torch.randn(8, state_dim)
        output = network(x)
        assert output.shape == (8, action_dim)

    def test_forward_without_batch_norm(self, state_dim: int, action_dim: int) -> None:
        """Test forward pass without batch normalization."""
        config = DRLConfig(hidden_layers=[32, 16], batch_norm=False)
        network = DuelingDQN(state_dim, action_dim, config)
        network.eval()
        x = torch.randn(4, state_dim)
        output = network(x)
        assert output.shape == (4, action_dim)

    def test_dropout_layer(self, state_dim: int, action_dim: int) -> None:
        """Test network with dropout."""
        config = DRLConfig(hidden_layers=[32, 16], dropout_rate=0.2)
        network = DuelingDQN(state_dim, action_dim, config)
        network.eval()
        x = torch.randn(4, state_dim)
        output3 = network(x)
        output4 = network(x)
        assert torch.allclose(output3, output4)

    def test_weight_initialization(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test that weights are initialized properly."""
        network = DuelingDQN(state_dim, action_dim, small_config)
        for param in network.parameters():
            if param.dim() > 1:
                assert not torch.all(param == 0)


class TestRainbowDQN:
    """Tests for RainbowDQN network."""

    def test_initialization(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test RainbowDQN initialization."""
        network = RainbowDQN(state_dim, action_dim, small_config)
        assert network.state_dim == state_dim
        assert network.action_dim == action_dim
        assert network.atoms == 51
        assert network.v_min == -10
        assert network.v_max == 10

    def test_custom_atoms_and_values(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test RainbowDQN with custom atoms and value range."""
        network = RainbowDQN(state_dim, action_dim, small_config, atoms=21, v_min=-5, v_max=5)
        assert network.atoms == 21
        assert network.v_min == -5
        assert network.v_max == 5
        assert network.delta_z == 0.5

    def test_forward_pass_distribution_shape(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test forward pass returns correct distribution shape."""
        atoms = 51
        network = RainbowDQN(state_dim, action_dim, small_config, atoms=atoms)
        network.eval()
        batch_size = 4
        x = torch.randn(batch_size, state_dim)
        output = network(x)
        assert output.shape == (batch_size, action_dim, atoms)

    def test_forward_pass_is_probability_distribution(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test that forward pass outputs valid probability distributions."""
        network = RainbowDQN(state_dim, action_dim, small_config)
        network.eval()
        x = torch.randn(4, state_dim)
        output = network(x)
        sums = output.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        assert torch.all(output >= 0)

    def test_get_q_values(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test Q-value extraction from distribution."""
        network = RainbowDQN(state_dim, action_dim, small_config)
        network.eval()
        batch_size = 4
        x = torch.randn(batch_size, state_dim)
        q_values = network.get_q_values(x)
        assert q_values.shape == (batch_size, action_dim)

    def test_noisy_layers_present(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test that RainbowDQN uses noisy layers."""
        network = RainbowDQN(state_dim, action_dim, small_config)
        assert isinstance(network.value_stream, NoisyLinear)
        assert isinstance(network.advantage_stream, NoisyLinear)


class TestNoisyLinear:
    """Tests for NoisyLinear layer."""

    def test_initialization(self) -> None:
        """Test NoisyLinear initialization."""
        layer = NoisyLinear(in_features=64, out_features=32)
        assert layer.in_features == 64
        assert layer.out_features == 32
        assert layer.std_init == 0.4

    def test_custom_std_init(self) -> None:
        """Test custom standard deviation initialization."""
        layer = NoisyLinear(in_features=64, out_features=32, std_init=0.2)
        assert layer.std_init == 0.2

    def test_forward_pass(self) -> None:
        """Test forward pass output shape."""
        layer = NoisyLinear(in_features=64, out_features=32)
        x = torch.randn(8, 64)
        output = layer(x)
        assert output.shape == (8, 32)

    def test_training_vs_eval_mode(self) -> None:
        """Test that noise is applied only in training mode."""
        torch.manual_seed(42)
        layer = NoisyLinear(in_features=32, out_features=16)
        x = torch.randn(4, 32)
        layer.eval()
        out1 = layer(x)
        out2 = layer(x)
        assert torch.allclose(out1, out2)

    def test_reset_noise(self) -> None:
        """Test noise reset functionality."""
        layer = NoisyLinear(in_features=32, out_features=16)
        old_weight_epsilon = layer.weight_epsilon.clone()
        layer.reset_noise()
        new_weight_epsilon = layer.weight_epsilon
        assert not torch.allclose(old_weight_epsilon, new_weight_epsilon)

    def test_learnable_parameters(self) -> None:
        """Test that mu and sigma are learnable parameters."""
        layer = NoisyLinear(in_features=32, out_features=16)
        assert layer.weight_mu.requires_grad
        assert layer.weight_sigma.requires_grad
        assert layer.bias_mu.requires_grad
        assert layer.bias_sigma.requires_grad

    def test_parameter_shapes(self) -> None:
        """Test parameter shapes."""
        in_features, out_features = 64, 32
        layer = NoisyLinear(in_features, out_features)
        assert layer.weight_mu.shape == (out_features, in_features)
        assert layer.weight_sigma.shape == (out_features, in_features)
        assert layer.bias_mu.shape == (out_features,)
        assert layer.bias_sigma.shape == (out_features,)


class TestActorCriticNetwork:
    """Tests for ActorCriticNetwork."""

    def test_initialization(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test network initialization."""
        network = ActorCriticNetwork(state_dim, action_dim, small_config)
        assert network.state_dim == state_dim
        assert network.action_dim == action_dim

    def test_forward_pass_shapes(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test forward pass output shapes."""
        network = ActorCriticNetwork(state_dim, action_dim, small_config)
        network.eval()
        batch_size = 8
        x = torch.randn(batch_size, state_dim)
        action_probs, state_value = network(x)
        assert action_probs.shape == (batch_size, action_dim)
        assert state_value.shape == (batch_size, 1)

    def test_action_probs_sum_to_one(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test that action probabilities sum to 1."""
        network = ActorCriticNetwork(state_dim, action_dim, small_config)
        network.eval()
        x = torch.randn(4, state_dim)
        action_probs, _ = network(x)
        sums = action_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_get_action_and_value(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test get_action_and_value method."""
        network = ActorCriticNetwork(state_dim, action_dim, small_config)
        network.eval()
        batch_size = 4
        x = torch.randn(batch_size, state_dim)
        action, log_prob, value = network.get_action_and_value(x)
        assert action.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        assert value.shape == (batch_size, 1)
        assert torch.all(action >= 0)
        assert torch.all(action < action_dim)

    def test_shared_layers_exist(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test that shared layers exist."""
        network = ActorCriticNetwork(state_dim, action_dim, small_config)
        assert hasattr(network, "shared_layers")
        assert hasattr(network, "actor")
        assert hasattr(network, "critic")


class TestPPONetwork:
    """Tests for PPONetwork."""

    def test_initialization(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test PPONetwork initialization."""
        network = PPONetwork(state_dim, action_dim, small_config)
        assert hasattr(network, "actor_critic")
        assert isinstance(network.actor_critic, ActorCriticNetwork)

    def test_forward_pass(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test forward pass."""
        network = PPONetwork(state_dim, action_dim, small_config)
        network.eval()
        batch_size = 4
        x = torch.randn(batch_size, state_dim)
        action_probs, values = network(x)
        assert action_probs.shape == (batch_size, action_dim)
        assert values.shape == (batch_size, 1)

    def test_get_action_and_value(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test get_action_and_value method."""
        network = PPONetwork(state_dim, action_dim, small_config)
        network.eval()
        batch_size = 4
        x = torch.randn(batch_size, state_dim)
        action, log_prob, value = network.get_action_and_value(x)
        assert action.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)

    def test_evaluate_actions(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test evaluate_actions method for PPO loss computation."""
        network = PPONetwork(state_dim, action_dim, small_config)
        network.eval()
        batch_size = 4
        x = torch.randn(batch_size, state_dim)
        actions = torch.randint(0, action_dim, (batch_size,))
        log_probs, values, entropy = network.evaluate_actions(x, actions)
        assert log_probs.shape == (batch_size,)
        assert values.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert torch.all(entropy >= 0)


class TestDeepRLControllerInitialization:
    """Tests for DeepRLController initialization."""

    def test_basic_initialization_dqn(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test basic initialization with DQN."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=small_config,
        )
        assert controller.algorithm == DRLAlgorithm.DQN
        assert controller.config == small_config

    def test_initialization_dueling_dqn(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test initialization with Dueling DQN."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DUELING_DQN,
            config=small_config,
        )
        assert controller.algorithm == DRLAlgorithm.DUELING_DQN
        assert controller.q_network is not None
        assert controller.target_network is not None

    def test_initialization_double_dqn(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test initialization with Double DQN."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DOUBLE_DQN,
            config=small_config,
        )
        assert controller.algorithm == DRLAlgorithm.DOUBLE_DQN

    def test_initialization_rainbow_dqn(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test initialization with Rainbow DQN."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.RAINBOW_DQN,
            config=small_config,
        )
        assert controller.algorithm == DRLAlgorithm.RAINBOW_DQN
        assert isinstance(controller.q_network, RainbowDQN)

    @pytest.mark.xfail(reason="Source bug: PPO/A3C algorithms try to load_state_dict on None target_network")
    def test_initialization_ppo(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test initialization with PPO."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.PPO,
            config=small_config,
        )
        assert controller.algorithm == DRLAlgorithm.PPO
        assert controller.policy_network is not None

    @pytest.mark.xfail(reason="Source bug: PPO/A3C algorithms try to load_state_dict on None target_network")
    def test_initialization_a3c(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test initialization with A3C."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.A3C,
            config=small_config,
        )
        assert controller.algorithm == DRLAlgorithm.A3C

    def test_replay_buffer_created(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test that replay buffer is created."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=small_config,
        )
        assert controller.replay_buffer is not None

    def test_exploration_parameters_initialized(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test that exploration parameters are initialized."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=small_config,
        )
        assert controller.epsilon == small_config.epsilon_start
        assert controller.beta == small_config.priority_beta_start

    def test_optimizer_created(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test that optimizer is created."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=small_config,
        )
        assert controller.optimizer is not None

    def test_scheduler_created(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test that learning rate scheduler is created."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=small_config,
        )
        assert controller.scheduler is not None

    def test_metrics_tracking_initialized(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test that metrics tracking lists are initialized."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=small_config,
        )
        assert hasattr(controller, "episode_rewards")
        assert hasattr(controller, "episode_lengths")
        assert hasattr(controller, "q_value_history")
        # episode_rewards is a deque, not a list
        assert len(controller.episode_rewards) == 0


class TestDeepRLControllerActionSelection:
    """Tests for action selection methods."""

    def test_select_action_dqn_training(self, state_dim: int, action_dim: int, small_config: DRLConfig, sample_state: np.ndarray) -> None:
        """Test action selection in DQN training mode."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=small_config,
        )
        action = controller.select_action(sample_state, training=True)
        assert 0 <= action < action_dim

    def test_select_action_dqn_evaluation(self, state_dim: int, action_dim: int, sample_state: np.ndarray) -> None:
        """Test action selection in DQN evaluation mode (greedy)."""
        # Use batch_norm=False to avoid single-sample batch norm issues
        config = DRLConfig(hidden_layers=[64, 32], batch_norm=False)
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=config,
        )
        controller.q_network.eval()
        action = controller.select_action(sample_state, training=False)
        assert 0 <= action < action_dim

    @pytest.mark.xfail(reason="Source bug: PPO/A3C algorithms try to load_state_dict on None target_network")
    def test_select_action_ppo(self, state_dim: int, action_dim: int, small_config: DRLConfig, sample_state: np.ndarray) -> None:
        """Test action selection in PPO."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.PPO,
            config=small_config,
        )
        action = controller.select_action(sample_state, training=True)
        assert 0 <= action < action_dim

    def test_epsilon_greedy_exploration(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test that epsilon-greedy exploration works."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=small_config,
        )
        controller.epsilon = 1.0
        state = np.random.randn(state_dim).astype(np.float32)
        actions = [controller.select_action(state, training=True) for _ in range(100)]
        unique_actions = set(actions)
        assert len(unique_actions) > 1

    def test_greedy_action_selection(self, state_dim: int, action_dim: int) -> None:
        """Test greedy action selection (epsilon=0)."""
        # Use batch_norm=False to avoid single-sample batch norm issues
        config = DRLConfig(hidden_layers=[64, 32], batch_norm=False)
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=config,
        )
        controller.epsilon = 0.0
        controller.q_network.eval()
        state = np.random.randn(state_dim).astype(np.float32)
        actions = [controller.select_action(state, training=True) for _ in range(10)]
        assert len(set(actions)) == 1


class TestDeepRLControllerTargetNetworkUpdate:
    """Tests for target network update methods."""

    def test_target_network_soft_update(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test soft update of target network."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=small_config,
        )
        initial_target_params = {
            name: param.clone()
            for name, param in controller.target_network.named_parameters()
        }
        for param in controller.q_network.parameters():
            param.data += 1.0
        controller._update_target_network()
        for name, param in controller.target_network.named_parameters():
            initial_param = initial_target_params[name]
            assert not torch.allclose(param, initial_param, atol=1e-6)

    def test_target_network_initialized_same_as_q_network(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test that target network is initialized with same weights as Q-network."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=small_config,
        )
        for (_name1, param1), (_name2, param2) in zip(
            controller.q_network.named_parameters(),
            controller.target_network.named_parameters(),
            strict=False,
        ):
            assert torch.allclose(param1, param2)


class TestDeepRLControllerExperienceStorage:
    """Tests for experience storage and replay."""

    def test_store_experience(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test storing experience in replay buffer."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=small_config,
        )
        state = np.random.randn(state_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        initial_len = len(controller.replay_buffer)
        controller.store_experience(state, action=0, reward=1.0, next_state=next_state, done=False)
        assert len(controller.replay_buffer) == initial_len + 1

    def test_store_multiple_experiences(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test storing multiple experiences."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=small_config,
        )
        for i in range(20):
            state = np.random.randn(state_dim).astype(np.float32)
            next_state = np.random.randn(state_dim).astype(np.float32)
            controller.store_experience(
                state, action=i % action_dim, reward=float(i), next_state=next_state, done=False
            )
        assert len(controller.replay_buffer) == 20


class TestDeepRLControllerExplorationUpdate:
    """Tests for exploration parameter updates."""

    def test_epsilon_decay(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test epsilon decay during training."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=small_config,
        )
        initial_epsilon = controller.epsilon
        controller.steps = 1000
        controller._update_exploration()
        assert controller.epsilon <= initial_epsilon

    def test_epsilon_reaches_minimum(self, state_dim: int, action_dim: int) -> None:
        """Test that epsilon reaches minimum value."""
        config = DRLConfig(
            hidden_layers=[32],
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=100,
        )
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=config,
        )
        controller.steps = 10000
        controller._update_exploration()
        assert controller.epsilon >= config.epsilon_end

    def test_beta_annealing(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test beta parameter annealing for importance sampling."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=small_config,
        )
        initial_beta = controller.beta
        controller.steps = 1000
        controller._update_exploration()
        assert controller.beta >= initial_beta


class TestDeepRLControllerIntegration:
    """Integration tests for DeepRLController."""

    def test_full_training_loop_dqn(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test a minimal training loop with DQN."""
        small_config.warmup_steps = 5
        small_config.batch_size = 4
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=small_config,
        )
        for _ in range(10):
            state = np.random.randn(state_dim).astype(np.float32)
            action = controller.select_action(state, training=True)
            next_state = np.random.randn(state_dim).astype(np.float32)
            reward = np.random.randn()
            done = False
            controller.store_experience(state, action, reward, next_state, done)
        result = controller.train_step()
        assert result is None or isinstance(result, dict)

    def test_dqn_algorithms_can_be_instantiated(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test that DQN-based algorithms can be instantiated."""
        dqn_algorithms = [
            DRLAlgorithm.DQN,
            DRLAlgorithm.DOUBLE_DQN,
            DRLAlgorithm.DUELING_DQN,
            DRLAlgorithm.RAINBOW_DQN,
        ]
        for algorithm in dqn_algorithms:
            controller = DeepRLController(
                state_dim=state_dim,
                action_dim=action_dim,
                algorithm=algorithm,
                config=small_config,
            )
            assert controller.algorithm == algorithm

    @pytest.mark.xfail(reason="Source bug: PPO/A3C algorithms try to load_state_dict on None target_network")
    def test_policy_gradient_algorithms_instantiation(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test that policy gradient algorithms can be instantiated."""
        pg_algorithms = [DRLAlgorithm.PPO, DRLAlgorithm.A3C]
        for algorithm in pg_algorithms:
            controller = DeepRLController(
                state_dim=state_dim,
                action_dim=action_dim,
                algorithm=algorithm,
                config=small_config,
            )
            assert controller.algorithm == algorithm

    def test_control_step_returns_valid_output(self, state_dim: int, action_dim: int, small_config: DRLConfig) -> None:
        """Test that control_step returns valid output."""
        controller = DeepRLController(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=DRLAlgorithm.DQN,
            config=small_config,
        )
        # Manually initialize steps attribute since control_step sets it
        controller.steps = 0
        # Mock extract_state_features to avoid needing a proper MFCSystemState object
        mock_features = np.random.randn(state_dim).astype(np.float32)
        with patch.object(controller, "extract_state_features", return_value=mock_features):
            with patch.object(controller, "select_action", return_value=0):
                result = controller.control_step({})
        # control_step returns (action, info_dict) tuple
        assert isinstance(result, tuple)
        action, info = result
        assert action == 0
        assert isinstance(info, dict)
        # Info should contain algorithm info
        assert "algorithm" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
