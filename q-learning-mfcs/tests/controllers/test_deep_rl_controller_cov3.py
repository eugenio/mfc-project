"""Tests for deep_rl_controller module - coverage part 3.

Covers missing lines: store_experience (priority/non-priority),
train_step routing, _train_dqn, _compute_dqn_loss (Double DQN),
_compute_distributional_loss, _update_target_network (soft/hard),
_update_exploration (priority replay), select_action (policy-based),
save_model/load_model (PPO/DQN variants), create_deep_rl_controller.
"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

mock_torch = MagicMock()
mock_nn = MagicMock()
mock_optim = MagicMock()
mock_F = MagicMock()


class _FakeTensor:
    pass


mock_torch.Tensor = _FakeTensor


class MockModule:
    def __init__(self, *a, **kw): pass
    def forward(self, x): return x
    def parameters(self): return []
    def state_dict(self): return {}
    def load_state_dict(self, d): pass
    def train(self, mode=True): return self
    def eval(self): return self
    def to(self, device): return self
    def __call__(self, *a, **kw): return MagicMock()
    def named_parameters(self): return []
    def children(self): return []
    def register_buffer(self, name, tensor): setattr(self, name, tensor)
    def apply(self, fn): return self
    training = True


class _FakeParameter:
    def __init__(self, data=None, requires_grad=True):
        self.data = data if data is not None else MagicMock()
        self.requires_grad = requires_grad
        self.grad = None
        self.shape = getattr(data, "shape", (1,))
    def numel(self): return 0


class _FakeLinear(MockModule):
    def __init__(self, in_f=0, out_f=0, *a, **kw): super().__init__()
    def __call__(self, x): return MagicMock()


mock_nn.Module = MockModule
mock_nn.Linear = _FakeLinear
mock_nn.ReLU = MagicMock(return_value=MagicMock())
mock_nn.LeakyReLU = MagicMock(return_value=MagicMock())
mock_nn.Tanh = MagicMock(return_value=MagicMock())
mock_nn.ELU = MagicMock(return_value=MagicMock())
mock_nn.SiLU = MagicMock(return_value=MagicMock())
mock_nn.Dropout = MagicMock(return_value=MagicMock())
mock_nn.BatchNorm1d = MagicMock(return_value=MagicMock())
mock_nn.Softmax = MagicMock(return_value=MagicMock())
mock_nn.LayerNorm = MagicMock(return_value=MagicMock())
mock_nn.Sequential = MagicMock(return_value=MagicMock())
mock_nn.ModuleList = MagicMock(return_value=[])
mock_nn.ModuleDict = MagicMock(return_value={})
mock_nn.functional = mock_F
mock_nn.Parameter = _FakeParameter
mock_torch.nn = mock_nn
mock_torch.optim = mock_optim
mock_torch.optim.SGD = MagicMock(return_value=MagicMock())
mock_torch.optim.Adam = MagicMock(return_value=MagicMock())
mock_torch.optim.AdamW = MagicMock(return_value=MagicMock(
    zero_grad=MagicMock(), step=MagicMock(),
    state_dict=MagicMock(return_value={}),
    load_state_dict=MagicMock()))
mock_torch.optim.lr_scheduler = MagicMock()
mock_torch.optim.lr_scheduler.StepLR = MagicMock(return_value=MagicMock(
    get_last_lr=MagicMock(return_value=[1e-4]),
    step=MagicMock(), state_dict=MagicMock(return_value={}),
    load_state_dict=MagicMock()))
mock_torch.optim.lr_scheduler.LambdaLR = MagicMock(return_value=MagicMock(
    get_last_lr=MagicMock(return_value=[1e-4]),
    step=MagicMock(), state_dict=MagicMock(return_value={}),
    load_state_dict=MagicMock()))
mock_torch.tensor = MagicMock(return_value=MagicMock())
mock_torch.zeros = MagicMock(return_value=MagicMock())
mock_torch.ones = MagicMock(return_value=MagicMock())
mock_torch.empty = MagicMock(return_value=MagicMock())
mock_torch.randn = MagicMock(return_value=MagicMock(
    sign=MagicMock(return_value=MagicMock(
        mul_=MagicMock(return_value=MagicMock())))))
mock_torch.zeros_like = MagicMock(return_value=MagicMock())
mock_torch.from_numpy = MagicMock(return_value=MagicMock(
    float=MagicMock(return_value=MagicMock())))
mock_torch.arange = MagicMock(return_value=MagicMock(
    unsqueeze=MagicMock(return_value=MagicMock(
        to=MagicMock(return_value=MagicMock()))),
    float=MagicMock(return_value=MagicMock())))
mock_torch.exp = MagicMock(return_value=MagicMock())
mock_torch.linspace = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.device = MagicMock(return_value="cpu")
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available = MagicMock(return_value=False)
mock_torch.no_grad = MagicMock(
    return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
mock_torch.save = MagicMock()
mock_torch.load = MagicMock(return_value={})
mock_torch.FloatTensor = MagicMock(return_value=MagicMock(
    unsqueeze=MagicMock(return_value=MagicMock(
        to=MagicMock(return_value=MagicMock())))))
mock_torch.LongTensor = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.BoolTensor = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.abs = MagicMock(return_value=MagicMock())
mock_torch.cat = MagicMock(return_value=MagicMock(
    shape=(1, 1, 256), device="cpu"))
mock_torch.argmax = MagicMock(return_value=MagicMock(
    item=MagicMock(return_value=0)))
mock_torch.max = MagicMock(return_value=MagicMock(
    item=MagicMock(return_value=0.9)))
mock_torch.sum = MagicMock(return_value=MagicMock())
mock_torch.nn.utils = MagicMock()
mock_torch.nn.utils.clip_grad_norm_ = MagicMock(
    return_value=MagicMock(item=MagicMock(return_value=1.0)))
mock_torch.nn.init = MagicMock()
mock_torch.nn.init.xavier_uniform_ = MagicMock()
mock_torch.nn.init.constant_ = MagicMock()
mock_F.softmax = MagicMock(return_value=MagicMock())
mock_F.linear = MagicMock(return_value=MagicMock())
mock_torch.distributions = MagicMock()

sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_nn
sys.modules["torch.optim"] = mock_optim
sys.modules["torch.nn.functional"] = mock_F
sys.modules["torch.utils"] = MagicMock()
sys.modules["torch.utils.data"] = MagicMock()

import numpy as np
import pytest
import torch_compat

torch_compat.TORCH_AVAILABLE = True
torch_compat.torch = mock_torch
torch_compat.nn = mock_nn
torch_compat.get_device = MagicMock(return_value="cpu")

from deep_rl_controller import (
    DRLAlgorithm,
    DRLConfig,
    DeepRLController,
    PriorityReplayBuffer,
    create_deep_rl_controller,
)


def _make_ctrl(algo=DRLAlgorithm.DQN, priority_replay=False):
    """Create a controller with _build_networks patched."""
    config = DRLConfig(
        hidden_layers=[32, 16],
        priority_replay=priority_replay,
        warmup_steps=2,
        memory_size=100,
        batch_size=4,
        train_freq=1,
        target_update_freq=5,
    )
    with patch.object(DeepRLController, "_build_networks"):
        ctrl = DeepRLController(
            state_dim=10, action_dim=5, algorithm=algo, config=config,
        )
        ctrl.q_network = MagicMock()
        ctrl.target_network = MagicMock()
        ctrl.optimizer = MagicMock()
        ctrl.scheduler = MagicMock()
        ctrl.writer = None
        ctrl.episode_rewards = deque(maxlen=100)
        ctrl.episode_lengths = deque(maxlen=100)
        ctrl.q_value_history = deque(maxlen=1000)
        if priority_replay:
            ctrl.replay_buffer = PriorityReplayBuffer(100)
        else:
            ctrl.replay_buffer = deque(maxlen=100)
        if algo in (DRLAlgorithm.PPO, DRLAlgorithm.A3C):
            mock_action_probs = MagicMock()
            mock_action_probs.argmax.return_value = MagicMock(
                item=MagicMock(return_value=2))
            mock_value = MagicMock()
            mock_value.item.return_value = 0.5
            ctrl.policy_network = MagicMock(
                return_value=(mock_action_probs, mock_value))
            ctrl.target_network = None
    return ctrl


class TestStoreExperience:
    """Cover store_experience lines 850-862."""

    def test_store_non_priority(self):
        ctrl = _make_ctrl(priority_replay=False)
        state = np.zeros(10)
        ctrl.store_experience(state, 0, 1.0, state, False)
        assert len(ctrl.replay_buffer) == 1

    def test_store_priority(self):
        ctrl = _make_ctrl(priority_replay=True)
        state = np.zeros(10)
        ctrl.store_experience(state, 0, 1.0, state, False)
        assert len(ctrl.replay_buffer) == 1


class TestTrainStep:
    """Cover train_step lines 864-877."""

    def test_train_step_warmup(self):
        ctrl = _make_ctrl()
        result = ctrl.train_step()
        assert result == {"loss": 0.0, "q_value": 0.0}

    def test_train_step_routes_to_dqn(self):
        ctrl = _make_ctrl()
        state = np.zeros(10)
        for i in range(5):
            ctrl.replay_buffer.append((state, i % 5, 0.5, state, False))
        ctrl._train_dqn = MagicMock(return_value={"loss": 0.1, "q_value": 0.5})
        result = ctrl.train_step()
        ctrl._train_dqn.assert_called_once()

    def test_train_step_routes_to_policy_gradient(self):
        ctrl = _make_ctrl(algo=DRLAlgorithm.PPO)
        for i in range(5):
            ctrl.replay_buffer.append((np.zeros(10), 0, 0.5, np.zeros(10), False))
        ctrl._train_policy_gradient = MagicMock(
            return_value={"loss": 0.1, "q_value": 0.5})
        result = ctrl.train_step()
        ctrl._train_policy_gradient.assert_called_once()


class TestTrainDQN:
    """Cover _train_dqn lines 879-958."""

    def test_train_dqn_non_priority(self):
        ctrl = _make_ctrl(priority_replay=False)
        state = np.zeros(10)
        for i in range(5):
            ctrl.replay_buffer.append((state, i % 5, 0.5, state, False))
        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.1
        mock_td = MagicMock()
        mock_td.detach.return_value = MagicMock(
            cpu=MagicMock(return_value=MagicMock(
                numpy=MagicMock(return_value=np.array([0.1, 0.2])))))
        ctrl._compute_dqn_loss = MagicMock(return_value=(mock_loss, mock_td))
        ctrl.q_value_history.append(0.5)
        result = ctrl._train_dqn()
        assert "loss" in result
        assert "epsilon" in result

    def test_train_dqn_priority_replay(self):
        ctrl = _make_ctrl(priority_replay=True)
        state = np.zeros(10)
        for i in range(5):
            ctrl.replay_buffer.push(state, i % 5, 0.5, state, False)
        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.2
        mock_td = MagicMock()
        mock_td.detach.return_value = MagicMock(
            cpu=MagicMock(return_value=MagicMock(
                numpy=MagicMock(return_value=np.array([0.1, 0.2])))))
        ctrl._compute_dqn_loss = MagicMock(return_value=(mock_loss, mock_td))
        ctrl.q_value_history.append(0.5)
        result = ctrl._train_dqn()
        assert "loss" in result

    def test_train_dqn_target_update(self):
        ctrl = _make_ctrl(priority_replay=False)
        state = np.zeros(10)
        for i in range(5):
            ctrl.replay_buffer.append((state, 0, 0.5, state, False))
        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.1
        mock_td = MagicMock()
        ctrl._compute_dqn_loss = MagicMock(return_value=(mock_loss, mock_td))
        ctrl.q_value_history.append(0.5)
        ctrl.steps = 5  # Matches target_update_freq
        ctrl._update_target_network = MagicMock()
        ctrl._train_dqn()
        ctrl._update_target_network.assert_called_once()


class TestUpdateTargetNetwork:
    """Cover _update_target_network lines 1077-1092."""

    def test_soft_update(self):
        ctrl = _make_ctrl()
        ctrl.config.tau = 0.005
        mock_target_param = MagicMock()
        mock_param = MagicMock()
        ctrl.target_network.parameters.return_value = [mock_target_param]
        ctrl.q_network.parameters.return_value = [mock_param]
        ctrl._update_target_network()
        mock_target_param.data.copy_.assert_called_once()

    def test_hard_update(self):
        ctrl = _make_ctrl()
        ctrl.config.tau = 1.0
        ctrl._update_target_network()
        ctrl.target_network.load_state_dict.assert_called_once()


class TestUpdateExploration:
    """Cover _update_exploration lines 1094-1117."""

    def test_epsilon_decay(self):
        ctrl = _make_ctrl()
        ctrl.epsilon = 0.5
        ctrl.config.priority_replay = False
        ctrl._update_exploration()
        assert ctrl.epsilon < 0.5

    def test_epsilon_at_minimum(self):
        ctrl = _make_ctrl()
        ctrl.epsilon = 0.01
        ctrl.config.epsilon_end = 0.01
        ctrl._update_exploration()
        assert ctrl.epsilon == 0.01

    def test_beta_annealing(self):
        ctrl = _make_ctrl(priority_replay=True)
        ctrl.epsilon = 0.01
        ctrl.config.epsilon_end = 0.01
        ctrl.beta = 0.5
        ctrl._update_exploration()
        assert ctrl.beta >= 0.5

    def test_rainbow_noise_reset(self):
        ctrl = _make_ctrl(algo=DRLAlgorithm.RAINBOW_DQN)
        ctrl.config.priority_replay = False
        ctrl.epsilon = 0.01
        ctrl.config.epsilon_end = 0.01
        ctrl.target_network = MagicMock()
        ctrl._update_exploration()
        ctrl.q_network.apply.assert_called()
        ctrl.target_network.apply.assert_called()


class TestSelectActionPolicyBased:
    """Cover select_action lines 801-848."""

    def test_select_action_ppo_training(self):
        ctrl = _make_ctrl(algo=DRLAlgorithm.PPO)
        state = np.zeros(10)
        action = ctrl.select_action(state, training=True)

    def test_select_action_ppo_eval(self):
        ctrl = _make_ctrl(algo=DRLAlgorithm.PPO)
        state = np.zeros(10)
        action = ctrl.select_action(state, training=False)

    def test_select_action_dqn_exploration(self):
        ctrl = _make_ctrl(algo=DRLAlgorithm.DQN)
        ctrl.epsilon = 1.0  # Always explore
        state = np.zeros(10)
        action = ctrl.select_action(state, training=True)
        assert 0 <= action < 5

    def test_select_action_rainbow(self):
        ctrl = _make_ctrl(algo=DRLAlgorithm.RAINBOW_DQN)
        ctrl.q_network.get_q_values = MagicMock(return_value=MagicMock(
            argmax=MagicMock(return_value=MagicMock(
                item=MagicMock(return_value=3))),
            mean=MagicMock(return_value=MagicMock(
                item=MagicMock(return_value=0.5))),
        ))
        state = np.zeros(10)
        action = ctrl.select_action(state, training=True)


class TestSaveLoadModel:
    """Cover save_model/load_model lines 1203-1252."""

    def test_save_model_dqn(self):
        ctrl = _make_ctrl(algo=DRLAlgorithm.DQN)
        with patch("deep_rl_controller.torch") as lt:
            ctrl.save_model("/tmp/test_dqn.pt")
            lt.save.assert_called()

    def test_save_model_ppo(self):
        ctrl = _make_ctrl(algo=DRLAlgorithm.PPO)
        with patch("deep_rl_controller.torch") as lt:
            ctrl.save_model("/tmp/test_ppo.pt")
            lt.save.assert_called()

    def test_load_model_dqn(self):
        ctrl = _make_ctrl(algo=DRLAlgorithm.DQN)
        with patch("deep_rl_controller.torch") as lt:
            lt.load.return_value = {
                "steps": 100, "episodes": 10,
                "epsilon": 0.3, "beta": 0.7,
                "q_network": {}, "target_network": {},
                "optimizer": {}, "scheduler": {},
            }
            ctrl.load_model("/tmp/test_dqn.pt")
        assert ctrl.epsilon == 0.3
        assert ctrl.beta == 0.7

    def test_load_model_ppo(self):
        ctrl = _make_ctrl(algo=DRLAlgorithm.PPO)
        with patch("deep_rl_controller.torch") as lt:
            lt.load.return_value = {
                "steps": 50, "episodes": 5,
                "epsilon": 0.2, "beta": 0.8,
                "policy_network": {},
                "optimizer": {}, "scheduler": {},
            }
            ctrl.load_model("/tmp/test_ppo.pt")
        assert ctrl.epsilon == 0.2


class TestCreateDeepRLController:
    """Cover create_deep_rl_controller lines 1287-1327."""

    def test_create_dqn(self):
        with patch.object(DeepRLController, "_build_networks"):
            ctrl = create_deep_rl_controller(10, 5, "dqn")
            assert ctrl.algorithm == DRLAlgorithm.DQN

    def test_create_double_dqn(self):
        with patch.object(DeepRLController, "_build_networks"):
            ctrl = create_deep_rl_controller(10, 5, "double_dqn")
            assert ctrl.algorithm == DRLAlgorithm.DOUBLE_DQN

    def test_create_ppo(self):
        with patch.object(DeepRLController, "_build_networks"):
            ctrl = create_deep_rl_controller(10, 5, "ppo")
            assert ctrl.algorithm == DRLAlgorithm.PPO

    def test_create_a3c(self):
        with patch.object(DeepRLController, "_build_networks"):
            ctrl = create_deep_rl_controller(10, 5, "a3c")
            assert ctrl.algorithm == DRLAlgorithm.A3C

    def test_create_unknown_defaults_rainbow(self):
        with patch.object(DeepRLController, "_build_networks"):
            ctrl = create_deep_rl_controller(10, 5, "unknown_algo")
            assert ctrl.algorithm == DRLAlgorithm.RAINBOW_DQN

    def test_create_with_kwargs(self):
        with patch.object(DeepRLController, "_build_networks"):
            ctrl = create_deep_rl_controller(
                10, 5, "dqn",
                hidden_layers=[64, 32],
                learning_rate=0.001,
            )
            assert ctrl.config.hidden_layers == [64, 32]
            assert ctrl.config.learning_rate == 0.001


class TestPriorityReplayBufferExtended:
    """Cover PriorityReplayBuffer sample/update_priorities."""

    def test_push_and_sample(self):
        buf = PriorityReplayBuffer(capacity=10, alpha=0.6)
        for i in range(5):
            buf.push(np.zeros(4), i, float(i), np.ones(4), False)
        batch, indices, weights = buf.sample(2, beta=0.4)
        assert len(batch) == 2
        assert len(indices) == 2
        assert len(weights) == 2

    def test_update_priorities(self):
        buf = PriorityReplayBuffer(capacity=10, alpha=0.6)
        for i in range(5):
            buf.push(np.zeros(4), i, float(i), np.ones(4), False)
        _, indices, _ = buf.sample(2, beta=0.4)
        buf.update_priorities(indices, np.array([1.0, 2.0]))
        assert buf.max_priority >= 2.0

    def test_buffer_overflow(self):
        buf = PriorityReplayBuffer(capacity=3, alpha=0.6)
        for i in range(5):
            buf.push(np.zeros(4), i, float(i), np.ones(4), False)
        assert len(buf) == 3
