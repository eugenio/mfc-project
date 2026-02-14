"""Tests for deep_rl_controller module - comprehensive coverage part 2.

Covers ActorCriticNetwork, PPONetwork, A3CNetwork, DuelingDQN internals,
control_step, update_with_reward, _train_policy_gradient, and edge cases.
"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import types
import random
from collections import deque, defaultdict

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
mock_nn.Dropout = MagicMock(return_value=MagicMock())
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
mock_torch.optim.lr_scheduler.LambdaLR = MagicMock(return_value=MagicMock(
    get_last_lr=MagicMock(return_value=[1e-4]),
    step=MagicMock(), state_dict=MagicMock(return_value={}),
    load_state_dict=MagicMock()))
mock_torch.tensor = MagicMock(return_value=MagicMock())
mock_torch.zeros = MagicMock(return_value=MagicMock())
mock_torch.ones = MagicMock(return_value=MagicMock())
mock_torch.empty = MagicMock(return_value=MagicMock())
mock_torch.zeros_like = MagicMock(return_value=MagicMock())
mock_torch.randn_like = MagicMock(return_value=MagicMock())
mock_torch.from_numpy = MagicMock(return_value=MagicMock(
    float=MagicMock(return_value=MagicMock())))
mock_torch.arange = MagicMock(return_value=MagicMock(
    unsqueeze=MagicMock(return_value=MagicMock()),
    float=MagicMock(return_value=MagicMock())))
mock_torch.exp = MagicMock(return_value=MagicMock())
mock_torch.sin = MagicMock(return_value=MagicMock())
mock_torch.cos = MagicMock(return_value=MagicMock())
mock_torch.tril = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.matmul = MagicMock(return_value=MagicMock(
    size=MagicMock(return_value=4)))
mock_torch.device = MagicMock(return_value="cpu")
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available = MagicMock(return_value=False)
mock_torch.no_grad = MagicMock(
    return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
mock_torch.save = MagicMock()
mock_torch.load = MagicMock(return_value={})
mock_torch.FloatTensor = MagicMock(return_value=MagicMock(
    unsqueeze=MagicMock(return_value=MagicMock(
        unsqueeze=MagicMock(return_value=MagicMock(
            to=MagicMock(return_value=MagicMock())))))))
mock_torch.LongTensor = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.normal = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.abs = MagicMock(return_value=MagicMock())
mock_torch.topk = MagicMock(return_value=(MagicMock(),
    MagicMock(cpu=MagicMock(return_value=MagicMock(
        numpy=MagicMock(return_value=__import__("numpy").array([0, 1])))))))
mock_torch.stack = MagicMock(return_value=MagicMock(
    median=MagicMock(return_value=(MagicMock(), None))))
mock_torch.median = MagicMock(return_value=(MagicMock(), None))
mock_torch.sort = MagicMock(return_value=(MagicMock(), None))
mock_torch.mean = MagicMock(return_value=MagicMock())
mock_torch.cat = MagicMock(return_value=MagicMock(
    shape=(1, 1, 256), device="cpu"))
mock_torch.argmax = MagicMock(return_value=MagicMock(
    item=MagicMock(return_value=0)))
mock_torch.max = MagicMock(return_value=MagicMock(
    item=MagicMock(return_value=0.9)))
mock_torch.nn.utils = MagicMock()
mock_torch.nn.utils.clip_grad_norm_ = MagicMock(
    return_value=MagicMock(item=MagicMock(return_value=1.0)))
mock_F.softmax = MagicMock(return_value=MagicMock())
mock_torch.distributions = MagicMock()

_orig_torch = sys.modules.get("torch")
_orig_torch_nn = sys.modules.get("torch.nn")
_orig_torch_optim = sys.modules.get("torch.optim")
_orig_torch_nn_functional = sys.modules.get("torch.nn.functional")
_orig_torch_utils = sys.modules.get("torch.utils")
_orig_torch_utils_data = sys.modules.get("torch.utils.data")

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
    DuelingDQN,
    ActorCriticNetwork,
    PPONetwork,
    A3CNetwork,
    NoisyLinear,
    RainbowDQN,
    PriorityReplayBuffer,
    TENSORBOARD_AVAILABLE,
)

# Restore original torch modules to prevent cross-contamination
for _name, _orig in [
    ("torch", _orig_torch),
    ("torch.nn", _orig_torch_nn),
    ("torch.optim", _orig_torch_optim),
    ("torch.nn.functional", _orig_torch_nn_functional),
    ("torch.utils", _orig_torch_utils),
    ("torch.utils.data", _orig_torch_utils_data),
]:
    if _orig is not None:
        sys.modules[_name] = _orig
    else:
        sys.modules.pop(_name, None)


def _make_ctrl_patched(algo=DRLAlgorithm.DQN):
    """Create a controller with _build_networks patched."""
    config = DRLConfig(hidden_layers=[32, 16], priority_replay=False,
                       warmup_steps=5, memory_size=100)
    with patch.object(DeepRLController, "_build_networks"):
        ctrl = DeepRLController(
            state_dim=10, action_dim=5, algorithm=algo, config=config)
        ctrl.q_network = MagicMock()
        ctrl.target_network = MagicMock()
        ctrl.optimizer = MagicMock()
        ctrl.scheduler = MagicMock()
        ctrl.replay_buffer = deque(maxlen=100)
        ctrl.episode_rewards = deque(maxlen=100)
        ctrl.episode_lengths = deque(maxlen=100)
        ctrl.q_value_history = deque(maxlen=1000)
        ctrl.writer = None
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


@pytest.mark.coverage_extra
class TestDuelingDQNClass:
    """Tests for DuelingDQN neural network class."""

    def test_init(self):
        config = DRLConfig(hidden_layers=[32, 16])
        net = DuelingDQN(state_dim=10, action_dim=5, config=config)
        assert net.state_dim == 10
        assert net.action_dim == 5

    def test_get_activation_relu(self):
        config = DRLConfig(hidden_layers=[32])
        net = DuelingDQN(state_dim=10, action_dim=5, config=config)
        act = net._get_activation("relu")

    def test_get_activation_leaky_relu(self):
        config = DRLConfig(hidden_layers=[32])
        net = DuelingDQN(state_dim=10, action_dim=5, config=config)
        act = net._get_activation("leaky_relu")

    def test_get_activation_tanh(self):
        config = DRLConfig(hidden_layers=[32])
        net = DuelingDQN(state_dim=10, action_dim=5, config=config)
        act = net._get_activation("tanh")

    def test_get_activation_elu(self):
        config = DRLConfig(hidden_layers=[32])
        net = DuelingDQN(state_dim=10, action_dim=5, config=config)
        act = net._get_activation("elu")

    def test_get_activation_swish(self):
        config = DRLConfig(hidden_layers=[32])
        net = DuelingDQN(state_dim=10, action_dim=5, config=config)
        act = net._get_activation("swish")

    def test_get_activation_unknown(self):
        config = DRLConfig(hidden_layers=[32])
        net = DuelingDQN(state_dim=10, action_dim=5, config=config)
        act = net._get_activation("unknown")

    def test_init_weights(self):
        config = DRLConfig(hidden_layers=[32])
        net = DuelingDQN(state_dim=10, action_dim=5, config=config)
        # _init_weights checks isinstance(m, nn.Linear) which fails with mock
        # So we just verify the method exists and can be called without error on non-Linear
        net._init_weights(MagicMock())  # Not nn.Linear, so should be no-op

    def test_forward(self):
        config = DRLConfig(hidden_layers=[32, 16])
        net = DuelingDQN(state_dim=10, action_dim=5, config=config)
        mock_input = MagicMock()
        result = net.forward(mock_input)

    def test_no_dropout(self):
        config = DRLConfig(hidden_layers=[32], dropout_rate=0.0)
        net = DuelingDQN(state_dim=10, action_dim=5, config=config)

    def test_no_batch_norm(self):
        config = DRLConfig(hidden_layers=[32], batch_norm=False)
        net = DuelingDQN(state_dim=10, action_dim=5, config=config)


@pytest.mark.coverage_extra
class TestActorCriticNetwork:
    """Tests for ActorCriticNetwork."""

    def test_init(self):
        config = DRLConfig(hidden_layers=[32, 16])
        net = ActorCriticNetwork(state_dim=10, action_dim=5, config=config)
        assert net.state_dim == 10
        assert net.action_dim == 5

    def test_forward(self):
        config = DRLConfig(hidden_layers=[32, 16])
        net = ActorCriticNetwork(state_dim=10, action_dim=5, config=config)
        mock_input = MagicMock()
        result = net.forward(mock_input)

    def test_get_action_and_value(self):
        config = DRLConfig(hidden_layers=[32, 16])
        net = ActorCriticNetwork(state_dim=10, action_dim=5, config=config)
        mock_input = MagicMock()
        result = net.get_action_and_value(mock_input)

    def test_get_activation(self):
        config = DRLConfig(hidden_layers=[32, 16])
        net = ActorCriticNetwork(state_dim=10, action_dim=5, config=config)
        for act_name in ["relu", "leaky_relu", "tanh", "elu", "swish", "xx"]:
            act = net._get_activation(act_name)


@pytest.mark.coverage_extra
class TestPPONetwork:
    """Tests for PPONetwork."""

    def test_init(self):
        config = DRLConfig(hidden_layers=[32, 16])
        net = PPONetwork(state_dim=10, action_dim=5, config=config)
        assert net.config == config

    def test_forward(self):
        config = DRLConfig(hidden_layers=[32, 16])
        net = PPONetwork(state_dim=10, action_dim=5, config=config)
        result = net.forward(MagicMock())

    def test_get_action_and_value(self):
        config = DRLConfig(hidden_layers=[32, 16])
        net = PPONetwork(state_dim=10, action_dim=5, config=config)
        result = net.get_action_and_value(MagicMock())

    def test_evaluate_actions(self):
        config = DRLConfig(hidden_layers=[32, 16])
        net = PPONetwork(state_dim=10, action_dim=5, config=config)
        result = net.evaluate_actions(MagicMock(), MagicMock())


@pytest.mark.coverage_extra
class TestA3CNetwork:
    """Tests for A3CNetwork."""

    def test_init(self):
        config = DRLConfig(hidden_layers=[32, 16])
        net = A3CNetwork(state_dim=10, action_dim=5, config=config)
        assert net.config == config

    def test_forward(self):
        config = DRLConfig(hidden_layers=[32, 16])
        net = A3CNetwork(state_dim=10, action_dim=5, config=config)
        result = net.forward(MagicMock())

    def test_get_action_and_value(self):
        config = DRLConfig(hidden_layers=[32, 16])
        net = A3CNetwork(state_dim=10, action_dim=5, config=config)
        result = net.get_action_and_value(MagicMock())


@pytest.mark.coverage_extra
class TestNoisyLinearClass:
    """Tests for NoisyLinear layer."""

    def test_init(self):
        nl = NoisyLinear(10, 5)
        assert nl.in_features == 10
        assert nl.out_features == 5
        assert nl.std_init == 0.4

    def test_custom_std(self):
        nl = NoisyLinear(10, 5, std_init=0.2)
        assert nl.std_init == 0.2

    def test_reset_parameters(self):
        nl = NoisyLinear(10, 5)
        nl.reset_parameters()

    def test_reset_noise(self):
        nl = NoisyLinear(10, 5)
        nl.reset_noise()

    def test_scale_noise(self):
        nl = NoisyLinear(10, 5)
        result = nl._scale_noise(10)

    def test_forward(self):
        nl = NoisyLinear(10, 5)
        result = nl.forward(MagicMock())


@pytest.mark.coverage_extra
class TestRainbowDQNClass:
    """Tests for RainbowDQN."""

    def test_init(self):
        config = DRLConfig(hidden_layers=[32, 16])
        net = RainbowDQN(state_dim=10, action_dim=5, config=config)
        assert net.state_dim == 10
        assert net.action_dim == 5
        assert net.atoms == 51

    def test_custom_atoms(self):
        config = DRLConfig(hidden_layers=[32])
        net = RainbowDQN(state_dim=10, action_dim=5, config=config,
                         atoms=21, v_min=-5, v_max=5)
        assert net.atoms == 21
        assert net.v_min == -5
        assert net.v_max == 5

    def test_forward(self):
        config = DRLConfig(hidden_layers=[32])
        net = RainbowDQN(state_dim=10, action_dim=5, config=config)
        mock_input = MagicMock()
        mock_input.size.return_value = 4  # batch_size
        result = net.forward(mock_input)

    def test_get_q_values(self):
        config = DRLConfig(hidden_layers=[32])
        net = RainbowDQN(state_dim=10, action_dim=5, config=config)
        result = net.get_q_values(MagicMock())


@pytest.mark.coverage_extra
class TestDeepRLControllerAdvanced:
    """Advanced tests for DeepRLController."""

    def test_control_step(self):
        ctrl = _make_ctrl_patched()
        mock_state = MagicMock()
        mock_state.power_output = 5.0
        mock_state.current_density = 1.0
        mock_state.health_metrics.overall_health_score = 0.8
        mock_state.fused_measurement.fusion_confidence = 0.9
        mock_state.anomalies = []
        ctrl._feature_engineer = MagicMock()
        ctrl._feature_engineer.extract_features.return_value = {
            "f1": 1.0, "f2": 2.0}
        action, info = ctrl.control_step(mock_state)
        assert "algorithm" in info
        assert info["steps"] == 1

    def test_control_step_ppo(self):
        ctrl = _make_ctrl_patched(DRLAlgorithm.PPO)
        mock_state = MagicMock()
        mock_state.power_output = 5.0
        mock_state.current_density = 1.0
        mock_state.health_metrics.overall_health_score = 0.8
        mock_state.fused_measurement.fusion_confidence = 0.9
        mock_state.anomalies = []
        ctrl._feature_engineer = MagicMock()
        ctrl._feature_engineer.extract_features.return_value = {
            "f1": 1.0, "f2": 2.0}
        action, info = ctrl.control_step(mock_state)

    def test_update_with_reward(self):
        ctrl = _make_ctrl_patched()
        ctrl.config.train_freq = 1
        mock_state1 = MagicMock()
        mock_state1.power_output = 5.0
        mock_state1.current_density = 1.0
        mock_state1.health_metrics.overall_health_score = 0.8
        mock_state1.fused_measurement.fusion_confidence = 0.9
        mock_state1.anomalies = []
        mock_state2 = MagicMock()
        mock_state2.power_output = 6.0
        mock_state2.current_density = 1.2
        mock_state2.health_metrics.overall_health_score = 0.85
        mock_state2.fused_measurement.fusion_confidence = 0.92
        mock_state2.anomalies = []
        ctrl._feature_engineer = MagicMock()
        ctrl._feature_engineer.extract_features.return_value = {
            "f1": 1.0, "f2": 2.0}
        ctrl.update_with_reward(mock_state1, 0, 1.0, mock_state2, False)

    def test_load_model_ppo(self):
        ctrl = _make_ctrl_patched(DRLAlgorithm.PPO)
        mock_checkpoint = {
            "steps": 100,
            "episodes": 10,
            "epsilon": 0.5,
            "beta": 0.6,
            "policy_network": {},
            "optimizer": {},
            "scheduler": {},
        }
        with patch("deep_rl_controller.torch") as local_torch:
            local_torch.load.return_value = mock_checkpoint
            local_torch.device.return_value = "cpu"
            ctrl.load_model("/tmp/test.pt")
        assert ctrl.epsilon == 0.5

    def test_update_exploration_rainbow(self):
        ctrl = _make_ctrl_patched(DRLAlgorithm.RAINBOW_DQN)
        ctrl.config.priority_replay = True
        ctrl.beta = 0.5
        ctrl.epsilon = 0.5
        ctrl._update_exploration()

    def test_tensorboard_available(self):
        """Test TENSORBOARD_AVAILABLE constant."""
        assert isinstance(TENSORBOARD_AVAILABLE, bool)

    def test_get_performance_summary_empty(self):
        ctrl = _make_ctrl_patched()
        summary = ctrl.get_performance_summary()
        assert "algorithm" in summary
        assert "avg_episode_reward" in summary
        assert "avg_episode_length" in summary
        assert "replay_buffer_size" in summary
