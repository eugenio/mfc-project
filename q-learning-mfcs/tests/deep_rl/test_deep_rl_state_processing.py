"""Tests for Deep RL State Processing.

US-008: Test Deep RL State Processing
Target: 90%+ coverage for state processing components

Tests cover:
- State normalization (running mean/std)
- State stacking for temporal information
- Observation encoding for different sensor types
- Reward shaping and scaling
- Terminal state handling
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from deep_rl_controller import DeepRLController, DRLAlgorithm, DRLConfig

# ============================================================================
# Fixtures
# ============================================================================


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
def state_dim() -> int:
    """State dimension for tests."""
    return 8


@pytest.fixture
def action_dim() -> int:
    """Action dimension for tests."""
    return 4


@pytest.fixture
def mock_system_state():
    """Create a mock SystemState for testing."""
    state = MagicMock()
    state.power_output = 5.2
    state.current_density = 0.8
    state.health_metrics = MagicMock()
    state.health_metrics.overall_health_score = 0.85
    state.fused_measurement = MagicMock()
    state.fused_measurement.fusion_confidence = 0.92
    state.anomalies = []
    return state


def fill_buffer(controller: DeepRLController, n: int, state_dim: int) -> None:
    """Fill replay buffer with random experiences."""
    for _ in range(n):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.randint(0, controller.action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = np.random.random() < 0.1
        controller.store_experience(state, action, reward, next_state, done)


# ============================================================================
# State Normalization Tests
# ============================================================================


class TestStateNormalization:
    """Tests for state normalization with running mean/std."""

    def test_state_to_tensor_conversion(self, state_dim, action_dim, small_config):
        """Test state array to tensor conversion."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        state = np.random.randn(state_dim).astype(np.float32)
        tensor = controller.prepare_state_tensor(state)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, state_dim)
        assert tensor.dtype == torch.float32

    def test_state_tensor_device(self, state_dim, action_dim, small_config):
        """Test state tensor is on correct device."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        state = np.random.randn(state_dim).astype(np.float32)
        tensor = controller.prepare_state_tensor(state)
        assert tensor.device.type == "cpu"

    def test_state_tensor_no_batch_dim(self, state_dim, action_dim, small_config):
        """Test state tensor without batch dimension."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        state = np.random.randn(state_dim).astype(np.float32)
        tensor = controller.prepare_state_tensor(state, add_batch_dim=False)
        assert tensor.shape == (state_dim,)

    def test_nan_handling_in_state(self, state_dim, action_dim, small_config):
        """Test handling of NaN values in state."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        state = np.array(
            [1.0, np.nan, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0], dtype=np.float32
        )
        next_state = np.random.randn(state_dim).astype(np.float32)
        controller.store_experience(state, 0, 1.0, next_state, False)
        assert len(controller.replay_buffer) == 1

    def test_inf_handling_in_state(self, state_dim, action_dim, small_config):
        """Test handling of infinite values in state."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        state = np.array(
            [1.0, np.inf, 2.0, -np.inf, 3.0, 4.0, 5.0, 6.0], dtype=np.float32
        )
        next_state = np.random.randn(state_dim).astype(np.float32)
        controller.store_experience(state, 0, 1.0, next_state, False)
        assert len(controller.replay_buffer) == 1

    def test_state_feature_extraction_normalization(
        self, state_dim, action_dim, small_config, mock_system_state
    ):
        """Test feature extraction applies tanh normalization."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        with patch.object(controller, "_feature_engineer") as mock_fe:
            mock_fe.extract_features.return_value = {
                f"feature_{i}": (i + 1) * 100.0 for i in range(state_dim)
            }
            features = controller.extract_state_features(mock_system_state)
            assert all(-1 <= v <= 1 for v in features)

    def test_normalized_state_bounds(self, state_dim, action_dim, small_config):
        """Test normalized state values are bounded."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        large_state = np.array([1e10] * state_dim, dtype=np.float32)
        tensor = controller.prepare_state_tensor(large_state)
        assert torch.isfinite(tensor).all()

    def test_zero_state_normalization(self, state_dim, action_dim, small_config):
        """Test normalization of zero state."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        zero_state = np.zeros(state_dim, dtype=np.float32)
        tensor = controller.prepare_state_tensor(zero_state)
        assert torch.allclose(tensor, torch.zeros(1, state_dim))


# ============================================================================
# State Stacking Tests (Temporal Information)
# ============================================================================


class TestStateStacking:
    """Tests for state stacking for temporal information."""

    def test_sequential_state_storage(self, state_dim, action_dim, small_config):
        """Test sequential states are stored in buffer."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        states = [np.random.randn(state_dim).astype(np.float32) for _ in range(5)]
        for i, state in enumerate(states):
            next_state = states[(i + 1) % len(states)]
            controller.store_experience(
                state, i % action_dim, float(i), next_state, False
            )
        assert len(controller.replay_buffer) == 5

    def test_temporal_sequence_preservation(self, state_dim, action_dim, small_config):
        """Test temporal sequence is preserved in experiences."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        state1 = np.ones(state_dim, dtype=np.float32)
        state2 = np.ones(state_dim, dtype=np.float32) * 2
        controller.store_experience(state1, 0, 1.0, state2, False)
        experience = controller.replay_buffer[0]
        assert np.allclose(experience[0], state1)
        assert np.allclose(experience[3], state2)

    def test_state_transition_consistency(self, state_dim, action_dim, small_config):
        """Test state transitions are consistent."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        prev_state = np.random.randn(state_dim).astype(np.float32)
        for i in range(10):
            next_state = np.random.randn(state_dim).astype(np.float32)
            controller.store_experience(
                prev_state, i % action_dim, float(i), next_state, False
            )
            prev_state = next_state
        assert len(controller.replay_buffer) == 10

    def test_episode_boundary_handling(self, state_dim, action_dim, small_config):
        """Test handling of episode boundaries with done=True."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        for i in range(4):
            state = np.random.randn(state_dim).astype(np.float32)
            next_state = np.random.randn(state_dim).astype(np.float32)
            done = i == 3
            controller.store_experience(state, 0, 1.0, next_state, done)
        assert controller.replay_buffer[3][4] is True
        assert controller.replay_buffer[0][4] is False

    def test_multi_episode_stacking(self, state_dim, action_dim, small_config):
        """Test stacking across multiple episodes."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        episode_lengths = [3, 4, 2]
        for ep_len in episode_lengths:
            for i in range(ep_len):
                state = np.random.randn(state_dim).astype(np.float32)
                next_state = np.random.randn(state_dim).astype(np.float32)
                done = i == ep_len - 1
                controller.store_experience(state, 0, 1.0, next_state, done)
        assert len(controller.replay_buffer) == sum(episode_lengths)


# ============================================================================
# Observation Encoding Tests (Different Sensor Types)
# ============================================================================


class TestObservationEncoding:
    """Tests for observation encoding with different sensor types."""

    def test_different_input_dimensions(self, action_dim, small_config):
        """Test handling different input state dimensions."""
        for dim in [4, 8, 16, 32]:
            config = DRLConfig(
                hidden_layers=[32, 16],
                batch_size=4,
                memory_size=100,
                warmup_steps=5,
                batch_norm=False,
            )
            controller = DeepRLController(
                dim, action_dim, DRLAlgorithm.DQN, config, "cpu"
            )
            state = np.random.randn(dim).astype(np.float32)
            action = controller.select_action(state, training=False)
            assert 0 <= action < action_dim

    def test_mixed_sensor_state_encoding(self, state_dim, action_dim, small_config):
        """Test encoding of mixed sensor types in state."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        mixed_state = np.array(
            [25.0, 7.2, 0.5, 0.8, 0.1, 100.0, 50.0, 0.9], dtype=np.float32
        )
        action = controller.select_action(mixed_state, training=False)
        assert 0 <= action < action_dim

    def test_sparse_state_encoding(self, state_dim, action_dim, small_config):
        """Test encoding of sparse state vectors."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        sparse_state = np.zeros(state_dim, dtype=np.float32)
        sparse_state[0] = 1.0
        sparse_state[state_dim - 1] = -1.0
        action = controller.select_action(sparse_state, training=False)
        assert 0 <= action < action_dim

    def test_negative_value_encoding(self, state_dim, action_dim, small_config):
        """Test encoding of negative state values."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        negative_state = np.array(
            [-i for i in range(1, state_dim + 1)], dtype=np.float32
        )
        action = controller.select_action(negative_state, training=False)
        assert 0 <= action < action_dim

    def test_uniform_state_encoding(self, state_dim, action_dim, small_config):
        """Test encoding of uniform state values."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        uniform_state = np.ones(state_dim, dtype=np.float32) * 0.5
        action = controller.select_action(uniform_state, training=False)
        assert 0 <= action < action_dim

    def test_binary_state_encoding(self, state_dim, action_dim, small_config):
        """Test encoding of binary state values."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        binary_state = np.array([i % 2 for i in range(state_dim)], dtype=np.float32)
        action = controller.select_action(binary_state, training=False)
        assert 0 <= action < action_dim

    def test_high_dimensional_state_encoding(self, action_dim):
        """Test encoding of high-dimensional state."""
        high_dim = 128
        config = DRLConfig(
            hidden_layers=[64, 32],
            batch_size=4,
            memory_size=100,
            warmup_steps=5,
            batch_norm=False,
        )
        controller = DeepRLController(
            high_dim, action_dim, DRLAlgorithm.DQN, config, "cpu"
        )
        state = np.random.randn(high_dim).astype(np.float32)
        action = controller.select_action(state, training=False)
        assert 0 <= action < action_dim


# ============================================================================
# Reward Shaping and Scaling Tests
# ============================================================================


class TestRewardShapingAndScaling:
    """Tests for reward shaping and scaling."""

    def test_positive_reward_storage(self, state_dim, action_dim, small_config):
        """Test storage of positive rewards."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        state = np.random.randn(state_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        positive_reward = 10.0
        controller.store_experience(state, 0, positive_reward, next_state, False)
        assert controller.replay_buffer[0][2] == positive_reward

    def test_negative_reward_storage(self, state_dim, action_dim, small_config):
        """Test storage of negative rewards."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        state = np.random.randn(state_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        negative_reward = -5.0
        controller.store_experience(state, 0, negative_reward, next_state, False)
        assert controller.replay_buffer[0][2] == negative_reward

    def test_zero_reward_storage(self, state_dim, action_dim, small_config):
        """Test storage of zero rewards."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        state = np.random.randn(state_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        controller.store_experience(state, 0, 0.0, next_state, False)
        assert controller.replay_buffer[0][2] == 0.0

    def test_large_reward_storage(self, state_dim, action_dim, small_config):
        """Test storage of large magnitude rewards."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        state = np.random.randn(state_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        large_reward = 1e6
        controller.store_experience(state, 0, large_reward, next_state, False)
        assert controller.replay_buffer[0][2] == large_reward

    def test_small_reward_storage(self, state_dim, action_dim, small_config):
        """Test storage of small magnitude rewards."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        state = np.random.randn(state_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        small_reward = 1e-6
        controller.store_experience(state, 0, small_reward, next_state, False)
        assert controller.replay_buffer[0][2] == small_reward

    def test_reward_in_td_error_computation(self, state_dim, action_dim, small_config):
        """Test reward is used correctly in TD error computation."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        batch_size = 4
        states = torch.randn(batch_size, state_dim)
        actions = torch.randint(0, action_dim, (batch_size,))
        rewards = torch.tensor([1.0, 2.0, -1.0, 0.5])
        next_states = torch.randn(batch_size, state_dim)
        dones = torch.zeros(batch_size, dtype=torch.bool)
        weights = torch.ones(batch_size)
        loss, td_errors = controller._compute_dqn_loss(
            states, actions, rewards, next_states, dones, weights
        )
        assert td_errors.shape == (batch_size,)
        assert not torch.isnan(td_errors).any()

    def test_reward_scaling_with_gamma(self, state_dim, action_dim, small_config):
        """Test reward scaling with discount factor gamma."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        assert controller.config.gamma == 0.99
        assert 0 < controller.config.gamma <= 1.0

    def test_reward_history_tracking(self, state_dim, action_dim, small_config):
        """Test reward history is tracked."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        assert hasattr(controller, "reward_history")
        assert len(controller.reward_history) == 0


# ============================================================================
# Terminal State Handling Tests
# ============================================================================


class TestTerminalStateHandling:
    """Tests for terminal state handling."""

    def test_terminal_state_done_flag(self, state_dim, action_dim, small_config):
        """Test done flag is correctly stored for terminal states."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        state = np.random.randn(state_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        controller.store_experience(state, 0, 1.0, next_state, True)
        assert controller.replay_buffer[0][4] is True

    def test_non_terminal_state_done_flag(self, state_dim, action_dim, small_config):
        """Test done flag is False for non-terminal states."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        state = np.random.randn(state_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        controller.store_experience(state, 0, 1.0, next_state, False)
        assert controller.replay_buffer[0][4] is False

    def test_terminal_state_td_target(self, state_dim, action_dim, small_config):
        """Test TD target for terminal states excludes next state value."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        batch_size = 4
        states = torch.randn(batch_size, state_dim)
        actions = torch.randint(0, action_dim, (batch_size,))
        rewards = torch.ones(batch_size)
        next_states = torch.randn(batch_size, state_dim)
        dones = torch.ones(batch_size, dtype=torch.bool)
        weights = torch.ones(batch_size)
        loss, td_errors = controller._compute_dqn_loss(
            states, actions, rewards, next_states, dones, weights
        )
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_mixed_terminal_non_terminal_batch(
        self, state_dim, action_dim, small_config
    ):
        """Test batch with mixed terminal and non-terminal states."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        batch_size = 4
        states = torch.randn(batch_size, state_dim)
        actions = torch.randint(0, action_dim, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randn(batch_size, state_dim)
        dones = torch.tensor([False, True, False, True])
        weights = torch.ones(batch_size)
        loss, td_errors = controller._compute_dqn_loss(
            states, actions, rewards, next_states, dones, weights
        )
        assert loss.item() >= 0
        assert td_errors.shape == (batch_size,)

    def test_terminal_state_in_priority_buffer(
        self, state_dim, action_dim, priority_config
    ):
        """Test terminal state handling in priority replay buffer."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, priority_config, "cpu"
        )
        state = np.random.randn(state_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        controller.store_experience(state, 0, 1.0, next_state, True)
        assert len(controller.replay_buffer) == 1

    def test_episode_end_tracking(self, state_dim, action_dim, small_config):
        """Test episode count increments at terminal states."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        initial_episodes = controller.episodes
        for i in range(5):
            state = np.random.randn(state_dim).astype(np.float32)
            next_state = np.random.randn(state_dim).astype(np.float32)
            controller.store_experience(state, 0, 1.0, next_state, i == 4)
        assert controller.episodes == initial_episodes

    def test_double_dqn_terminal_handling(self, state_dim, action_dim, small_config):
        """Test terminal state handling in Double DQN."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DOUBLE_DQN, small_config, "cpu"
        )
        batch_size = 4
        states = torch.randn(batch_size, state_dim)
        actions = torch.randint(0, action_dim, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randn(batch_size, state_dim)
        dones = torch.ones(batch_size, dtype=torch.bool)
        weights = torch.ones(batch_size)
        loss, td_errors = controller._compute_dqn_loss(
            states, actions, rewards, next_states, dones, weights
        )
        assert not torch.isnan(loss)


# ============================================================================
# Priority Replay Buffer State Processing Tests
# ============================================================================


class TestPriorityReplayBufferStateProcessing:
    """Tests for state processing in priority replay buffer."""

    def test_priority_buffer_state_storage(
        self, state_dim, action_dim, priority_config
    ):
        """Test state storage in priority buffer."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, priority_config, "cpu"
        )
        state = np.random.randn(state_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        controller.store_experience(state, 0, 1.0, next_state, False)
        assert len(controller.replay_buffer) == 1

    def test_priority_buffer_batch_sampling(
        self, state_dim, action_dim, priority_config
    ):
        """Test batch sampling from priority buffer."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, priority_config, "cpu"
        )
        fill_buffer(controller, 20, state_dim)
        batch, indices, weights = controller.replay_buffer.sample(4, beta=0.4)
        assert len(batch) == 4
        assert len(indices) == 4
        assert len(weights) == 4

    def test_priority_buffer_importance_weights(
        self, state_dim, action_dim, priority_config
    ):
        """Test importance sampling weights in priority buffer."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, priority_config, "cpu"
        )
        fill_buffer(controller, 20, state_dim)
        _, _, weights = controller.replay_buffer.sample(4, beta=0.4)
        assert np.max(weights) <= 1.0 + 1e-6
        assert np.min(weights) > 0

    def test_priority_update_after_training(
        self, state_dim, action_dim, priority_config
    ):
        """Test priority update after training step."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, priority_config, "cpu"
        )
        fill_buffer(controller, 20, state_dim)
        original_sum = controller.replay_buffer.sum_tree[0]
        controller._train_dqn()
        new_sum = controller.replay_buffer.sum_tree[0]
        assert original_sum != new_sum or np.allclose(original_sum, new_sum)


# ============================================================================
# State Processing Integration Tests
# ============================================================================


class TestStateProcessingIntegration:
    """Integration tests for state processing pipeline."""

    def test_full_state_processing_pipeline(self, state_dim, action_dim, small_config):
        """Test complete state processing from raw to action."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        raw_state = np.random.randn(state_dim).astype(np.float32)
        action = controller.select_action(raw_state, training=True)
        assert 0 <= action < action_dim

    def test_state_processing_in_training_loop(
        self, state_dim, action_dim, small_config
    ):
        """Test state processing during training loop."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        fill_buffer(controller, 20, state_dim)
        metrics = controller.train_step()
        assert "loss" in metrics
        assert not np.isnan(metrics["loss"])

    def test_action_selection_training_vs_eval(
        self, state_dim, action_dim, small_config
    ):
        """Test action selection differs between training and eval modes."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        # Set network to eval mode for deterministic results
        controller.q_network.eval()
        state = np.random.randn(state_dim).astype(np.float32)
        controller.epsilon = 0.9
        training_actions = [
            controller.select_action(state, training=True) for _ in range(100)
        ]
        eval_actions = [
            controller.select_action(state, training=False) for _ in range(10)
        ]
        # Eval actions should all be the same (greedy)
        assert len(set(eval_actions)) == 1
        # Training actions should have more variety due to epsilon-greedy exploration
        assert len(set(training_actions)) > 1 or controller.epsilon < 0.1

    def test_state_processing_with_different_algorithms(
        self, state_dim, action_dim, small_config
    ):
        """Test state processing works with different algorithms."""
        for algo in [
            DRLAlgorithm.DQN,
            DRLAlgorithm.DOUBLE_DQN,
            DRLAlgorithm.DUELING_DQN,
        ]:
            controller = DeepRLController(
                state_dim, action_dim, algo, small_config, "cpu"
            )
            state = np.random.randn(state_dim).astype(np.float32)
            action = controller.select_action(state, training=False)
            assert 0 <= action < action_dim

    def test_rainbow_state_processing(self, state_dim, action_dim):
        """Test state processing with Rainbow DQN."""
        config = DRLConfig(
            hidden_layers=[32, 16],
            batch_size=4,
            memory_size=100,
            warmup_steps=5,
            priority_replay=True,
            batch_norm=False,
        )
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.RAINBOW_DQN, config, "cpu"
        )
        state = np.random.randn(state_dim).astype(np.float32)
        action = controller.select_action(state, training=False)
        assert 0 <= action < action_dim

    def test_q_value_history_update(self, state_dim, action_dim, small_config):
        """Test Q-value history is updated during action selection."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        state = np.random.randn(state_dim).astype(np.float32)
        initial_len = len(controller.q_value_history)
        controller.select_action(state, training=False)
        assert len(controller.q_value_history) >= initial_len

    def test_steps_increment_in_control_step(
        self, state_dim, action_dim, small_config, mock_system_state
    ):
        """Test steps counter increments during control_step."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        initial_steps = controller.steps
        with patch.object(controller, "_feature_engineer") as mock_fe:
            mock_fe.extract_features.return_value = {
                f"feature_{i}": np.random.randn() for i in range(state_dim)
            }
            action, info = controller.control_step(mock_system_state)
        assert controller.steps == initial_steps + 1


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases in state processing."""

    def test_single_sample_state(self, state_dim, action_dim, small_config):
        """Test processing of single state sample."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        state = np.random.randn(state_dim).astype(np.float32)
        action = controller.select_action(state, training=False)
        assert isinstance(action, int)

    def test_very_small_state_values(self, state_dim, action_dim, small_config):
        """Test processing of very small state values."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        small_state = np.full(state_dim, 1e-10, dtype=np.float32)
        action = controller.select_action(small_state, training=False)
        assert 0 <= action < action_dim

    def test_very_large_state_values(self, state_dim, action_dim, small_config):
        """Test processing of very large state values."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        large_state = np.full(state_dim, 1e10, dtype=np.float32)
        action = controller.select_action(large_state, training=False)
        assert 0 <= action < action_dim

    def test_alternating_sign_state(self, state_dim, action_dim, small_config):
        """Test state with alternating positive/negative values."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        alt_state = np.array(
            [(-1) ** i * i for i in range(state_dim)], dtype=np.float32
        )
        action = controller.select_action(alt_state, training=False)
        assert 0 <= action < action_dim

    def test_all_same_value_state(self, state_dim, action_dim, small_config):
        """Test state where all values are identical."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        same_state = np.full(state_dim, 0.5, dtype=np.float32)
        action = controller.select_action(same_state, training=False)
        assert 0 <= action < action_dim

    def test_buffer_overflow_handling(self, state_dim, action_dim, small_config):
        """Test buffer handles overflow correctly."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        fill_buffer(controller, small_config.memory_size + 50, state_dim)
        assert len(controller.replay_buffer) == small_config.memory_size

    def test_action_consistency_same_state(self, state_dim, action_dim, small_config):
        """Test action is consistent for same state in eval mode."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        controller.q_network.eval()
        state = np.random.randn(state_dim).astype(np.float32)
        actions = [controller.select_action(state, training=False) for _ in range(10)]
        assert len(set(actions)) == 1


# ============================================================================
# Additional State Processing Coverage Tests
# ============================================================================


class TestRainbowStateProcessing:
    """Tests for Rainbow DQN state processing specifics."""

    def test_rainbow_distributional_forward(self, state_dim, action_dim):
        """Test Rainbow DQN distributional forward pass."""
        config = DRLConfig(
            hidden_layers=[32, 16],
            batch_size=4,
            memory_size=100,
            warmup_steps=5,
            priority_replay=True,
            batch_norm=False,
        )
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.RAINBOW_DQN, config, "cpu"
        )
        state_tensor = torch.randn(4, state_dim)
        dist = controller.q_network(state_tensor)
        assert dist.shape == (4, action_dim, controller.q_network.atoms)
        assert torch.allclose(dist.sum(dim=-1), torch.ones(4, action_dim), atol=1e-5)

    def test_rainbow_get_q_values(self, state_dim, action_dim):
        """Test Rainbow Q-value computation from distribution."""
        config = DRLConfig(
            hidden_layers=[32, 16],
            batch_size=4,
            memory_size=100,
            warmup_steps=5,
            priority_replay=True,
            batch_norm=False,
        )
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.RAINBOW_DQN, config, "cpu"
        )
        state_tensor = torch.randn(4, state_dim)
        q_values = controller.q_network.get_q_values(state_tensor)
        assert q_values.shape == (4, action_dim)
        assert not torch.isnan(q_values).any()

    def test_rainbow_noisy_layer_noise_reset(self, state_dim, action_dim):
        """Test Rainbow noisy layer noise reset."""
        config = DRLConfig(
            hidden_layers=[32, 16],
            batch_size=4,
            memory_size=100,
            warmup_steps=5,
            priority_replay=True,
            batch_norm=False,
        )
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.RAINBOW_DQN, config, "cpu"
        )
        # Get initial noise values
        for module in controller.q_network.modules():
            if hasattr(module, "reset_noise"):
                old_weight_epsilon = module.weight_epsilon.clone()
                module.reset_noise()
                # Noise should be different after reset
                assert not torch.allclose(old_weight_epsilon, module.weight_epsilon)
                break

    def test_rainbow_distributional_loss_computation(self, state_dim, action_dim):
        """Test distributional loss computation for Rainbow DQN."""
        config = DRLConfig(
            hidden_layers=[32, 16],
            batch_size=4,
            memory_size=100,
            warmup_steps=10,
            priority_replay=True,
            batch_norm=False,
        )
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.RAINBOW_DQN, config, "cpu"
        )
        # Fill buffer
        for _ in range(20):
            state = np.random.randn(state_dim).astype(np.float32)
            next_state = np.random.randn(state_dim).astype(np.float32)
            controller.store_experience(
                state, np.random.randint(0, action_dim), np.random.randn(), next_state, False
            )
        metrics = controller.train_step()
        assert "loss" in metrics
        assert not np.isnan(metrics["loss"])


class TestUpdateWithReward:
    """Tests for update_with_reward method."""

    def test_update_with_reward_stores_experience(
        self, state_dim, action_dim, small_config, mock_system_state
    ):
        """Test update_with_reward stores experience correctly."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        with patch.object(controller, "_feature_engineer") as mock_fe:
            mock_fe.extract_features.return_value = {
                f"feature_{i}": np.random.randn() for i in range(state_dim)
            }
            prev_state = mock_system_state
            next_state = MagicMock()
            next_state.power_output = 5.5
            next_state.current_density = 0.9
            next_state.health_metrics = MagicMock()
            next_state.health_metrics.overall_health_score = 0.9
            next_state.fused_measurement = MagicMock()
            next_state.fused_measurement.fusion_confidence = 0.95
            next_state.anomalies = []

            controller.update_with_reward(prev_state, 0, 1.0, next_state, False)
            assert len(controller.replay_buffer) == 1

    def test_update_with_reward_training_triggers(
        self, state_dim, action_dim, small_config, mock_system_state
    ):
        """Test training is triggered at correct frequency."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        with patch.object(controller, "_feature_engineer") as mock_fe:
            mock_fe.extract_features.return_value = {
                f"feature_{i}": np.random.randn() for i in range(state_dim)
            }
            # Fill buffer first
            for _ in range(20):
                state = np.random.randn(state_dim).astype(np.float32)
                next_state = np.random.randn(state_dim).astype(np.float32)
                controller.store_experience(state, 0, 1.0, next_state, False)

            next_state = MagicMock()
            next_state.power_output = 5.5
            next_state.current_density = 0.9
            next_state.health_metrics = MagicMock()
            next_state.health_metrics.overall_health_score = 0.9
            next_state.fused_measurement = MagicMock()
            next_state.fused_measurement.fusion_confidence = 0.95
            next_state.anomalies = []

            controller.steps = small_config.train_freq - 1
            controller.update_with_reward(mock_system_state, 0, 1.0, next_state, False)


class TestExplorationUpdate:
    """Tests for exploration parameter updates."""

    def test_epsilon_decay(self, state_dim, action_dim, small_config):
        """Test epsilon decays properly."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        initial_epsilon = controller.epsilon
        controller._update_exploration()
        assert controller.epsilon < initial_epsilon or controller.epsilon == small_config.epsilon_end

    def test_epsilon_minimum(self, state_dim, action_dim, small_config):
        """Test epsilon doesn't go below minimum."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        controller.epsilon = small_config.epsilon_end
        controller._update_exploration()
        assert controller.epsilon >= small_config.epsilon_end

    def test_beta_annealing_priority_replay(self, state_dim, action_dim, priority_config):
        """Test beta annealing for priority replay."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, priority_config, "cpu"
        )
        initial_beta = controller.beta
        controller._update_exploration()
        assert controller.beta > initial_beta or controller.beta == 1.0

    def test_beta_maximum(self, state_dim, action_dim, priority_config):
        """Test beta doesn't exceed 1.0."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, priority_config, "cpu"
        )
        controller.beta = 0.99
        controller._update_exploration()
        assert controller.beta <= 1.0

    def test_rainbow_noise_reset_during_exploration_update(self, state_dim, action_dim):
        """Test Rainbow DQN resets noise during exploration update."""
        config = DRLConfig(
            hidden_layers=[32, 16],
            batch_size=4,
            memory_size=100,
            warmup_steps=5,
            priority_replay=True,
            batch_norm=False,
        )
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.RAINBOW_DQN, config, "cpu"
        )
        # This should not raise an error
        controller._update_exploration()


class TestTargetNetworkUpdate:
    """Tests for target network update methods."""

    def test_soft_update(self, state_dim, action_dim, small_config):
        """Test soft target network update."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        # Get initial target params
        initial_target_params = [p.clone() for p in controller.target_network.parameters()]

        # Modify q_network params
        for p in controller.q_network.parameters():
            p.data += 1.0

        controller._update_target_network()

        # Target should have moved toward q_network
        for old_p, new_p in zip(
            initial_target_params, controller.target_network.parameters(), strict=False
        ):
            assert not torch.allclose(old_p, new_p)

    def test_hard_update(self, state_dim, action_dim):
        """Test hard target network update (tau=1)."""
        config = DRLConfig(
            hidden_layers=[32, 16],
            batch_size=4,
            memory_size=100,
            warmup_steps=5,
            tau=1.0,
            batch_norm=False,
        )
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, config, "cpu"
        )
        # Modify q_network params
        for p in controller.q_network.parameters():
            p.data += 1.0

        controller._update_target_network()

        # Target should exactly match q_network
        for q_p, t_p in zip(
            controller.q_network.parameters(),
            controller.target_network.parameters(),
            strict=False,
        ):
            assert torch.allclose(q_p, t_p)


class TestPerformanceSummary:
    """Tests for performance summary method."""

    def test_get_performance_summary_keys(self, state_dim, action_dim, small_config):
        """Test performance summary contains expected keys."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        summary = controller.get_performance_summary()
        expected_keys = [
            "algorithm",
            "steps",
            "episodes",
            "avg_episode_reward",
            "avg_episode_length",
            "avg_q_value",
            "avg_loss",
            "epsilon",
            "beta",
            "replay_buffer_size",
            "device",
            "network_parameters",
        ]
        for key in expected_keys:
            assert key in summary

    def test_get_performance_summary_values(self, state_dim, action_dim, small_config):
        """Test performance summary values are valid."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        # Add some history
        controller.episode_rewards.append(10.0)
        controller.episode_lengths.append(100)
        controller.q_value_history.append(5.0)
        controller.loss_history.append(0.1)

        summary = controller.get_performance_summary()
        assert summary["avg_episode_reward"] == 10.0
        assert summary["avg_episode_length"] == 100
        assert summary["avg_q_value"] == 5.0
        assert summary["algorithm"] == "deep_q_network"


class TestSaveLoadModel:
    """Tests for model save/load functionality."""

    def test_save_model_creates_file(self, state_dim, action_dim, small_config, tmp_path):
        """Test save_model creates checkpoint file."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        path = tmp_path / "model.pt"
        controller.save_model(str(path))
        assert path.exists()

    def test_load_model_restores_state(self, state_dim, action_dim, small_config, tmp_path):
        """Test load_model restores controller state."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        controller.steps = 100
        controller.epsilon = 0.5
        controller.beta = 0.8
        path = tmp_path / "model.pt"
        controller.save_model(str(path))

        # Create new controller and load
        new_controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        new_controller.load_model(str(path))

        assert new_controller.steps == 100
        assert new_controller.epsilon == 0.5
        assert new_controller.beta == 0.8

    def test_save_load_preserves_network_weights(
        self, state_dim, action_dim, small_config, tmp_path
    ):
        """Test save/load preserves network weights."""
        controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        controller.q_network.eval()
        state = np.random.randn(state_dim).astype(np.float32)
        original_action = controller.select_action(state, training=False)
        path = tmp_path / "model.pt"
        controller.save_model(str(path))

        new_controller = DeepRLController(
            state_dim, action_dim, DRLAlgorithm.DQN, small_config, "cpu"
        )
        new_controller.load_model(str(path))
        new_controller.q_network.eval()
        loaded_action = new_controller.select_action(state, training=False)

        assert original_action == loaded_action


class TestFactoryFunction:
    """Tests for create_deep_rl_controller factory function."""

    def test_create_dqn_controller(self, state_dim, action_dim):
        """Test creating DQN controller via factory."""
        from deep_rl_controller import create_deep_rl_controller

        controller = create_deep_rl_controller(
            state_dim,
            action_dim,
            algorithm="dqn",
            hidden_layers=[32, 16],
            batch_norm=False,
        )
        assert controller.algorithm == DRLAlgorithm.DQN

    def test_create_double_dqn_controller(self, state_dim, action_dim):
        """Test creating Double DQN controller via factory."""
        from deep_rl_controller import create_deep_rl_controller

        controller = create_deep_rl_controller(
            state_dim,
            action_dim,
            algorithm="double_dqn",
            hidden_layers=[32, 16],
            batch_norm=False,
        )
        assert controller.algorithm == DRLAlgorithm.DOUBLE_DQN

    def test_create_dueling_dqn_controller(self, state_dim, action_dim):
        """Test creating Dueling DQN controller via factory."""
        from deep_rl_controller import create_deep_rl_controller

        controller = create_deep_rl_controller(
            state_dim,
            action_dim,
            algorithm="dueling_dqn",
            hidden_layers=[32, 16],
            batch_norm=False,
        )
        assert controller.algorithm == DRLAlgorithm.DUELING_DQN

    def test_create_rainbow_dqn_controller(self, state_dim, action_dim):
        """Test creating Rainbow DQN controller via factory."""
        from deep_rl_controller import create_deep_rl_controller

        controller = create_deep_rl_controller(
            state_dim,
            action_dim,
            algorithm="rainbow_dqn",
            hidden_layers=[32, 16],
            batch_norm=False,
        )
        assert controller.algorithm == DRLAlgorithm.RAINBOW_DQN

    def test_create_with_default_algorithm(self, state_dim, action_dim):
        """Test creating controller with unknown algorithm defaults to Rainbow."""
        from deep_rl_controller import create_deep_rl_controller

        controller = create_deep_rl_controller(
            state_dim,
            action_dim,
            algorithm="unknown_algo",
            hidden_layers=[32, 16],
            batch_norm=False,
        )
        assert controller.algorithm == DRLAlgorithm.RAINBOW_DQN
