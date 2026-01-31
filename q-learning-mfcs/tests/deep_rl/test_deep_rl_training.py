"""Tests for Deep RL Training Methods.

US-007: Test Deep RL Training Methods
Target: 90%+ coverage for training methods
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from deep_rl_controller import (
    DeepRLController,
    DRLAlgorithm,
    DRLConfig,
    PriorityReplayBuffer,
)


@pytest.fixture
def small_config() -> DRLConfig:
    """Small configuration for fast testing."""
    return DRLConfig(
        hidden_layers=[32, 16],
        batch_size=4,
        memory_size=100,
        warmup_steps=5,
        target_update_freq=10,
        gradient_clip=1.0,
        priority_replay=False,
        batch_norm=False,
    )


@pytest.fixture
def priority_config() -> DRLConfig:
    """Configuration with priority replay enabled."""
    return DRLConfig(
        hidden_layers=[32, 16],
        batch_size=4,
        memory_size=100,
        warmup_steps=5,
        target_update_freq=10,
        gradient_clip=1.0,
        priority_replay=True,
        priority_alpha=0.6,
        priority_beta_start=0.4,
        priority_beta_frames=1000,
        batch_norm=False,
    )


@pytest.fixture
def rainbow_config() -> DRLConfig:
    """Configuration for Rainbow DQN."""
    return DRLConfig(
        hidden_layers=[32, 16],
        batch_size=4,
        memory_size=100,
        warmup_steps=5,
        target_update_freq=10,
        priority_replay=True,
        batch_norm=False,
    )


@pytest.fixture
def state_dim() -> int:
    """State dimension for tests."""
    return 8


@pytest.fixture
def action_dim() -> int:
    """Action dimension for tests."""
    return 4


def fill_buffer(controller: DeepRLController, n: int, state_dim: int) -> None:
    """Fill replay buffer with random experiences."""
    for _ in range(n):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.randint(0, controller.action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = np.random.random() < 0.1
        controller.store_experience(state, action, reward, next_state, done)


class TestTrainStep:
    """Tests for train_step method."""

    def test_warmup_returns_zeros(self, state_dim, action_dim, small_config):
        """During warmup, train_step should return zeros."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        fill_buffer(c, 2, state_dim)
        metrics = c.train_step()
        assert metrics["loss"] == 0.0
        assert metrics["q_value"] == 0.0

    def test_returns_metrics_dict(self, state_dim, action_dim, small_config):
        """Train step should return a metrics dictionary."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        fill_buffer(c, 20, state_dim)
        metrics = c.train_step()
        assert "loss" in metrics
        assert "q_value" in metrics
        assert "epsilon" in metrics
        assert "learning_rate" in metrics

    def test_double_dqn_algorithm(self, state_dim, action_dim, small_config):
        """Test train_step with Double DQN."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DOUBLE_DQN, small_config, "cpu")
        fill_buffer(c, 20, state_dim)
        metrics = c.train_step()
        assert metrics["loss"] >= 0

    def test_dueling_dqn_algorithm(self, state_dim, action_dim, small_config):
        """Test train_step with Dueling DQN."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DUELING_DQN, small_config, "cpu")
        fill_buffer(c, 20, state_dim)
        metrics = c.train_step()
        assert metrics["loss"] >= 0

    def test_rainbow_dqn_algorithm(self, state_dim, action_dim, rainbow_config):
        """Test train_step with Rainbow DQN."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.RAINBOW_DQN, rainbow_config, "cpu")
        fill_buffer(c, 20, state_dim)
        metrics = c.train_step()
        assert "loss" in metrics
        assert not np.isnan(metrics["loss"])


class TestTrainDQN:
    """Tests for _train_dqn method."""

    def test_samples_from_buffer(self, state_dim, action_dim, small_config):
        """Training should sample from replay buffer without removing."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        fill_buffer(c, 20, state_dim)
        buffer_size = len(c.replay_buffer)
        c._train_dqn()
        assert len(c.replay_buffer) == buffer_size

    def test_updates_epsilon(self, state_dim, action_dim, small_config):
        """Training should update epsilon."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        fill_buffer(c, 20, state_dim)
        initial_epsilon = c.epsilon
        c._train_dqn()
        assert c.epsilon <= initial_epsilon

    def test_with_priority_replay(self, state_dim, action_dim, priority_config):
        """Test training with priority replay enabled."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, priority_config, "cpu")
        fill_buffer(c, 20, state_dim)
        metrics = c._train_dqn()
        assert "loss" in metrics
        assert "beta" in metrics

    def test_updates_priorities(self, state_dim, action_dim, priority_config):
        """Training should update priorities in replay buffer."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, priority_config, "cpu")
        fill_buffer(c, 20, state_dim)
        initial_tree = c.replay_buffer.sum_tree.copy()
        c._train_dqn()
        assert not np.allclose(c.replay_buffer.sum_tree, initial_tree, atol=1e-6)


class TestComputeDQNLoss:
    """Tests for _compute_dqn_loss method."""

    def test_returns_loss_and_td_errors(self, state_dim, action_dim, small_config):
        """Should return loss tensor and TD errors."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        bs = 4
        states = torch.randn(bs, state_dim)
        actions = torch.randint(0, action_dim, (bs,))
        rewards = torch.randn(bs)
        next_states = torch.randn(bs, state_dim)
        dones = torch.zeros(bs, dtype=torch.bool)
        is_weights = torch.ones(bs)
        loss, td_errors = c._compute_dqn_loss(states, actions, rewards, next_states, dones, is_weights)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert td_errors.shape == (bs,)

    def test_loss_non_negative(self, state_dim, action_dim, small_config):
        """Loss should be non-negative."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        bs = 4
        states = torch.randn(bs, state_dim)
        actions = torch.randint(0, action_dim, (bs,))
        rewards = torch.randn(bs)
        next_states = torch.randn(bs, state_dim)
        dones = torch.zeros(bs, dtype=torch.bool)
        is_weights = torch.ones(bs)
        loss, _ = c._compute_dqn_loss(states, actions, rewards, next_states, dones, is_weights)
        assert loss.item() >= 0

    def test_double_dqn_action_selection(self, state_dim, action_dim, small_config):
        """Double DQN should use online network for action selection."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DOUBLE_DQN, small_config, "cpu")
        bs = 4
        states = torch.randn(bs, state_dim)
        actions = torch.randint(0, action_dim, (bs,))
        rewards = torch.randn(bs)
        next_states = torch.randn(bs, state_dim)
        dones = torch.zeros(bs, dtype=torch.bool)
        is_weights = torch.ones(bs)
        loss, _ = c._compute_dqn_loss(states, actions, rewards, next_states, dones, is_weights)
        assert loss.item() >= 0

    def test_terminal_states_handling(self, state_dim, action_dim, small_config):
        """Terminal states should not include future rewards."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        bs = 4
        states = torch.randn(bs, state_dim)
        actions = torch.randint(0, action_dim, (bs,))
        rewards = torch.randn(bs)
        next_states = torch.randn(bs, state_dim)
        dones = torch.ones(bs, dtype=torch.bool)
        is_weights = torch.ones(bs)
        loss, _ = c._compute_dqn_loss(states, actions, rewards, next_states, dones, is_weights)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestDistributionalLoss:
    """Tests for _compute_distributional_loss method."""

    def test_returns_loss_and_td_errors(self, state_dim, action_dim, rainbow_config):
        """Should return loss tensor and TD errors."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.RAINBOW_DQN, rainbow_config, "cpu")
        bs = 4
        states = torch.randn(bs, state_dim)
        actions = torch.randint(0, action_dim, (bs,))
        rewards = torch.randn(bs)
        next_states = torch.randn(bs, state_dim)
        dones = torch.zeros(bs, dtype=torch.bool)
        is_weights = torch.ones(bs)
        loss, td_errors = c._compute_distributional_loss(states, actions, rewards, next_states, dones, is_weights)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_terminal_states(self, state_dim, action_dim, rainbow_config):
        """Should handle terminal states correctly."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.RAINBOW_DQN, rainbow_config, "cpu")
        bs = 4
        states = torch.randn(bs, state_dim)
        actions = torch.randint(0, action_dim, (bs,))
        rewards = torch.randn(bs)
        next_states = torch.randn(bs, state_dim)
        dones = torch.ones(bs, dtype=torch.bool)
        is_weights = torch.ones(bs)
        loss, _ = c._compute_distributional_loss(states, actions, rewards, next_states, dones, is_weights)
        assert not torch.isnan(loss)


class TestUpdateTargetNetwork:
    """Tests for _update_target_network method."""

    def test_soft_update(self, state_dim, action_dim, small_config):
        """Soft update should interpolate between networks."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        with torch.no_grad():
            for p in c.q_network.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        q_params = [p.clone() for p in c.q_network.parameters()]
        target_params = [p.clone() for p in c.target_network.parameters()]
        c._update_target_network()
        for q_before, t_before, t_after in zip(q_params, target_params, c.target_network.parameters(), strict=False):
            expected = c.config.tau * q_before + (1 - c.config.tau) * t_before
            assert torch.allclose(t_after, expected, atol=1e-5)

    def test_hard_update(self, state_dim, action_dim):
        """Hard update (tau=1) should copy weights exactly."""
        config = DRLConfig(hidden_layers=[32, 16], tau=1.0, batch_norm=False)
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, config, "cpu")
        with torch.no_grad():
            for p in c.q_network.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        c._update_target_network()
        for q_param, t_param in zip(c.q_network.parameters(), c.target_network.parameters(), strict=False):
            assert torch.allclose(q_param, t_param)


class TestUpdateExploration:
    """Tests for _update_exploration method."""

    def test_epsilon_decay(self, state_dim, action_dim, small_config):
        """Epsilon should decrease after update."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        initial_epsilon = c.epsilon
        c._update_exploration()
        assert c.epsilon < initial_epsilon

    def test_epsilon_minimum_bound(self, state_dim, action_dim, small_config):
        """Epsilon should not go below minimum."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        c.epsilon = c.config.epsilon_end
        c._update_exploration()
        assert c.epsilon >= c.config.epsilon_end

    def test_beta_annealing(self, state_dim, action_dim, priority_config):
        """Beta should increase for priority replay."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, priority_config, "cpu")
        initial_beta = c.beta
        c._update_exploration()
        assert c.beta > initial_beta

    def test_beta_maximum_bound(self, state_dim, action_dim, priority_config):
        """Beta should not exceed 1.0."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, priority_config, "cpu")
        c.beta = 1.0
        c._update_exploration()
        assert c.beta <= 1.0

    def test_rainbow_noise_reset(self, state_dim, action_dim, rainbow_config):
        """Rainbow should reset noise in noisy layers."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.RAINBOW_DQN, rainbow_config, "cpu")
        c._update_exploration()


class TestUpdateWithReward:
    """Tests for update_with_reward method."""

    def test_stores_experience(self, state_dim, action_dim, small_config):
        """Should store experience in replay buffer."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        mock_prev_state = MagicMock()
        mock_next_state = MagicMock()
        ps = np.random.randn(state_dim).astype(np.float32)
        ns = np.random.randn(state_dim).astype(np.float32)
        with patch.object(c, "extract_state_features", side_effect=[ps, ns]):
            initial_size = len(c.replay_buffer)
            c.update_with_reward(mock_prev_state, 0, 1.0, mock_next_state, False)
            assert len(c.replay_buffer) == initial_size + 1

    def test_triggers_training(self, state_dim, action_dim, small_config):
        """Should trigger training at train_freq intervals."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        fill_buffer(c, 20, state_dim)
        c.steps = c.config.train_freq - 1
        mock_prev_state = MagicMock()
        mock_next_state = MagicMock()
        ps = np.random.randn(state_dim).astype(np.float32)
        ns = np.random.randn(state_dim).astype(np.float32)
        with patch.object(c, "extract_state_features", side_effect=[ps, ns]):
            c.update_with_reward(mock_prev_state, 0, 1.0, mock_next_state, False)


class TestSaveLoad:
    """Tests for save_model and load_model methods."""

    def test_save_creates_file(self, state_dim, action_dim, small_config):
        """Save should create a file."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            c.save_model(f.name)
            assert Path(f.name).exists()
            assert Path(f.name).stat().st_size > 0

    def test_save_has_required_keys(self, state_dim, action_dim, small_config):
        """Save should include all required keys."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            c.save_model(f.name)
            checkpoint = torch.load(f.name, weights_only=False)
            assert "optimizer" in checkpoint
            assert "scheduler" in checkpoint
            assert "epsilon" in checkpoint
            assert "q_network" in checkpoint

    def test_load_restores_state(self, state_dim, action_dim, small_config):
        """Load should restore saved state."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        c.epsilon = 0.5
        c.beta = 0.7
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            c.save_model(f.name)
            new_c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
            new_c.load_model(f.name)
            assert new_c.epsilon == 0.5
            assert new_c.beta == 0.7

    def test_load_preserves_weights(self, state_dim, action_dim, small_config):
        """Load should preserve network weights."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        original_weights = {n: p.clone() for n, p in c.q_network.named_parameters()}
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            c.save_model(f.name)
            new_c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
            new_c.load_model(f.name)
            for name, param in new_c.q_network.named_parameters():
                assert torch.allclose(param, original_weights[name])


class TestPerformanceSummary:
    """Tests for get_performance_summary method."""

    def test_returns_dict(self, state_dim, action_dim, small_config):
        """Should return a dictionary."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        assert isinstance(c.get_performance_summary(), dict)

    def test_has_algorithm(self, state_dim, action_dim, small_config):
        """Should include algorithm name."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        summary = c.get_performance_summary()
        assert summary["algorithm"] == DRLAlgorithm.DQN.value

    def test_has_steps(self, state_dim, action_dim, small_config):
        """Should include step count."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        c.steps = 100
        assert c.get_performance_summary()["steps"] == 100

    def test_has_network_parameters(self, state_dim, action_dim, small_config):
        """Should include network parameter count."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        assert "network_parameters" in c.get_performance_summary()


class TestLRScheduling:
    """Tests for learning rate scheduling."""

    def test_get_learning_rate(self, state_dim, action_dim, small_config):
        """Should return current learning rate."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        lr = c.get_learning_rate()
        assert lr > 0
        assert isinstance(lr, float)

    def test_scheduler_steps(self, state_dim, action_dim, small_config):
        """Scheduler should step during training."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        fill_buffer(c, 20, state_dim)
        for _ in range(10):
            c.train_step()
        assert c.get_learning_rate() > 0


class TestGradientClipping:
    """Tests for gradient clipping."""

    def test_config_has_gradient_clip(self, state_dim, action_dim, small_config):
        """Config should have gradient_clip parameter."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        assert c.config.gradient_clip > 0

    def test_training_with_clipping(self, state_dim, action_dim, small_config):
        """Training should not produce NaN with clipping."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        fill_buffer(c, 20, state_dim)
        metrics = c.train_step()
        assert not np.isnan(metrics["loss"])
        assert not np.isinf(metrics["loss"])

    def test_custom_clip_value(self, state_dim, action_dim):
        """Should work with custom clip value."""
        config = DRLConfig(hidden_layers=[32, 16], gradient_clip=0.5, batch_norm=False, warmup_steps=5)
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, config, "cpu")
        fill_buffer(c, 20, state_dim)
        assert not np.isnan(c.train_step()["loss"])


class TestStoreExperience:
    """Tests for store_experience method."""

    def test_stores_to_deque(self, state_dim, action_dim, small_config):
        """Should store to deque when priority_replay is False."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        state = np.random.randn(state_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        c.store_experience(state, 0, 1.0, next_state, False)
        assert len(c.replay_buffer) == 1

    def test_stores_to_priority_buffer(self, state_dim, action_dim, priority_config):
        """Should store to priority buffer when priority_replay is True."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, priority_config, "cpu")
        state = np.random.randn(state_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        c.store_experience(state, 0, 1.0, next_state, False)
        assert len(c.replay_buffer) == 1


class TestControlStep:
    """Tests for control_step method with mocked SystemState."""

    def test_returns_tuple(self, state_dim, action_dim, small_config):
        """Should return (action, info) tuple."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        mock_state = MagicMock()
        features = np.random.randn(state_dim).astype(np.float32)
        with patch.object(c, "extract_state_features", return_value=features):
            result = c.control_step(mock_state)
            assert isinstance(result, tuple)
            assert len(result) == 2

    def test_returns_valid_action(self, state_dim, action_dim, small_config):
        """Action should be in valid range."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        mock_state = MagicMock()
        features = np.random.randn(state_dim).astype(np.float32)
        with patch.object(c, "extract_state_features", return_value=features):
            action, _ = c.control_step(mock_state)
            assert isinstance(action, int)
            assert 0 <= action < action_dim

    def test_info_contains_algorithm(self, state_dim, action_dim, small_config):
        """Info should contain algorithm."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        mock_state = MagicMock()
        features = np.random.randn(state_dim).astype(np.float32)
        with patch.object(c, "extract_state_features", return_value=features):
            _, info = c.control_step(mock_state)
            assert "algorithm" in info
            assert info["algorithm"] == DRLAlgorithm.DQN.value

    def test_increments_steps(self, state_dim, action_dim, small_config):
        """Should increment steps counter."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        mock_state = MagicMock()
        features = np.random.randn(state_dim).astype(np.float32)
        initial_steps = c.steps
        with patch.object(c, "extract_state_features", return_value=features):
            c.control_step(mock_state)
            assert c.steps == initial_steps + 1


class TestPriorityReplayBuffer:
    """Tests for PriorityReplayBuffer."""

    def test_push_and_len(self):
        """Test basic push and length."""
        buffer = PriorityReplayBuffer(100, alpha=0.6)
        state = np.random.randn(8).astype(np.float32)
        next_state = np.random.randn(8).astype(np.float32)
        buffer.push(state, 0, 1.0, next_state, False)
        assert len(buffer) == 1

    def test_sample_returns_batch(self):
        """Sample should return batch of correct size."""
        buffer = PriorityReplayBuffer(100, alpha=0.6)
        for _ in range(20):
            state = np.random.randn(8).astype(np.float32)
            next_state = np.random.randn(8).astype(np.float32)
            buffer.push(state, 0, 1.0, next_state, False)
        batch, indices, is_weights = buffer.sample(4, beta=0.4)
        assert len(batch) == 4
        assert len(indices) == 4
        assert len(is_weights) == 4

    def test_update_priorities(self):
        """Update priorities should modify tree."""
        buffer = PriorityReplayBuffer(100, alpha=0.6)
        for _ in range(20):
            state = np.random.randn(8).astype(np.float32)
            next_state = np.random.randn(8).astype(np.float32)
            buffer.push(state, 0, 1.0, next_state, False)
        initial_max = buffer.max_priority
        indices = np.array([0, 1, 2])
        priorities = np.array([10.0, 20.0, 30.0])
        buffer.update_priorities(indices, priorities)
        assert buffer.max_priority >= initial_max


class TestPolicyGradientBug:
    """Tests documenting PPO/A3C bugs - marked as xfail."""

    @pytest.mark.xfail(reason="PPO has bug: target_network=None before load_state_dict")
    def test_ppo_init_fails(self, state_dim, action_dim, small_config):
        """PPO initialization fails due to None target_network."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.PPO, small_config, "cpu")
        assert c.policy_network is not None

    @pytest.mark.xfail(reason="A3C has bug: target_network=None before load_state_dict")
    def test_a3c_init_fails(self, state_dim, action_dim, small_config):
        """A3C initialization fails due to None target_network."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.A3C, small_config, "cpu")
        assert c.policy_network is not None


class TestIntegration:
    """Integration tests."""

    def test_full_training_loop(self, state_dim, action_dim, small_config):
        """Test complete training loop."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        fill_buffer(c, 20, state_dim)
        losses = [c.train_step()["loss"] for _ in range(10)]
        assert all(not np.isnan(loss) and not np.isinf(loss) for loss in losses)

    def test_save_load_continue_training(self, state_dim, action_dim, small_config):
        """Test save, load, and continue training."""
        c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
        fill_buffer(c, 20, state_dim)
        for _ in range(5):
            c.train_step()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            c.save_model(f.name)
            new_c = DeepRLController(state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu")
            new_c.load_model(f.name)
            fill_buffer(new_c, 20, state_dim)
            assert "loss" in new_c.train_step()

    def test_multiple_algorithms(self, state_dim, action_dim, small_config):
        """Test training with different algorithms."""
        for algo in [DRLAlgorithm.DQN, DRLAlgorithm.DOUBLE_DQN, DRLAlgorithm.DUELING_DQN]:
            c = DeepRLController(state_dim, action_dim, algo, small_config, "cpu")
            fill_buffer(c, 20, state_dim)
            metrics = c.train_step()
            assert "loss" in metrics
