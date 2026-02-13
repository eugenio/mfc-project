"""Comprehensive tests for deep_rl_controller.py with torch mocked.

Targets 99%+ statement coverage of all classes and functions.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import deque
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
mock_nn.BatchNorm1d = MagicMock(return_value=MagicMock())
mock_nn.SiLU = MagicMock(return_value=MagicMock())
mock_nn.LeakyReLU = MagicMock(return_value=MagicMock())
mock_nn.Tanh = MagicMock(return_value=MagicMock())
mock_nn.ELU = MagicMock(return_value=MagicMock())
mock_nn.Softmax = MagicMock(return_value=MagicMock())
mock_torch.nn = mock_nn
mock_torch.optim = mock_optim
mock_torch.optim.SGD = MagicMock(return_value=MagicMock())
mock_torch.optim.Adam = MagicMock(return_value=MagicMock(
    zero_grad=MagicMock(), step=MagicMock(),
    state_dict=MagicMock(return_value={}),
    load_state_dict=MagicMock()))
mock_torch.optim.AdamW = MagicMock(return_value=MagicMock())
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
mock_torch.zeros_like = MagicMock(return_value=MagicMock())
mock_torch.randn_like = MagicMock(return_value=MagicMock())
mock_torch.from_numpy = MagicMock(return_value=MagicMock(
    float=MagicMock(return_value=MagicMock())))
mock_torch.arange = MagicMock(return_value=MagicMock(
    unsqueeze=MagicMock(return_value=MagicMock()),
    float=MagicMock(return_value=MagicMock())))
mock_torch.exp = MagicMock(return_value=MagicMock())
mock_torch.device = MagicMock(return_value="cpu")
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available = MagicMock(return_value=False)
mock_torch.no_grad = MagicMock(
    return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
mock_torch.save = MagicMock()
mock_torch.load = MagicMock(return_value={})
mock_torch.FloatTensor = MagicMock(return_value=MagicMock(
    unsqueeze=MagicMock(return_value=MagicMock(
        to=MagicMock(return_value=MagicMock()))),
    to=MagicMock(return_value=MagicMock())))
mock_torch.LongTensor = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.BoolTensor = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.nn.utils = MagicMock()
mock_torch.nn.utils.clip_grad_norm_ = MagicMock(
    return_value=MagicMock(item=MagicMock(return_value=1.0)))
mock_torch.nn.init = MagicMock()
mock_torch.linspace = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.randn = MagicMock(return_value=MagicMock(
    sign=MagicMock(return_value=MagicMock(
        mul_=MagicMock(return_value=MagicMock())))))
import numpy as np
import pytest

_orig = {}
import torch_compat
from deep_rl_controller import (
class _FakeTensor:
    pass


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


class TestDRLConfig:
    def test_defaults(self):
        c = DRLConfig()
        assert c.hidden_layers == [512, 256, 128]
        assert c.gamma == 0.99

    def test_custom(self):
        c = DRLConfig(hidden_layers=[64], learning_rate=0.01)
        assert c.hidden_layers == [64]

class TestPriorityReplayBuffer:
    def test_lifecycle(self):
        buf = PriorityReplayBuffer(capacity=10)
        for i in range(12):
            buf.push(np.array([float(i)]), 0, 1.0, np.array([0.0]), False)
        assert len(buf) == 10
        batch, idx, w = buf.sample(3)
        assert len(batch) == 3
        buf.update_priorities(np.array([0, 1]), np.array([2.0, 5.0]))
        assert buf.max_priority == 5.0

class TestNetworkClasses:
    def test_dueling_dqn(self):
        config = DRLConfig(hidden_layers=[32, 16])
        net = DuelingDQN(state_dim=10, action_dim=5, config=config)
        assert net.state_dim == 10
        for act in ["relu", "leaky_relu", "tanh", "elu", "swish", "unknown"]:
            net._get_activation(act)
        net.forward(MagicMock())

    def test_noisy_linear(self):
        nl = NoisyLinear(10, 5)
        nl.reset_parameters()
        nl.reset_noise()
        nl.training = True
        nl.forward(MagicMock())
        nl.training = False
        nl.forward(MagicMock())

    def test_rainbow_dqn(self):
        config = DRLConfig(hidden_layers=[32])
        net = RainbowDQN(state_dim=10, action_dim=5, config=config)
        mock_x = MagicMock()
        mock_x.size.return_value = 4
        mock_x.device = "cpu"
        net.forward(mock_x)
        net.get_q_values(mock_x)

    def test_actor_critic(self):
        config = DRLConfig(hidden_layers=[64, 32])
        net = ActorCriticNetwork(state_dim=10, action_dim=5, config=config)
        net.forward(MagicMock())
        net.get_action_and_value(MagicMock())

    def test_ppo_network(self):
        config = DRLConfig(hidden_layers=[64, 32])
        net = PPONetwork(state_dim=10, action_dim=5, config=config)
        net.forward(MagicMock())
        net.get_action_and_value(MagicMock())
        net.evaluate_actions(MagicMock(), MagicMock())

    def test_a3c_network(self):
        config = DRLConfig(hidden_layers=[64, 32])
        net = A3CNetwork(state_dim=10, action_dim=5, config=config)
        net.forward(MagicMock())
        net.get_action_and_value(MagicMock())

class TestDeepRLController:
    def _make(self, algo=DRLAlgorithm.DQN, priority=False):
        config = DRLConfig(hidden_layers=[32, 16], priority_replay=priority,
                           warmup_steps=5, memory_size=100, batch_size=4)
        if algo in (DRLAlgorithm.PPO, DRLAlgorithm.A3C):
            with patch.object(DeepRLController, "_build_networks"):
                ctrl = DeepRLController(state_dim=10, action_dim=5,
                                        algorithm=algo, config=config)
                ctrl.policy_network = MagicMock(return_value=(
                    MagicMock(), MagicMock(item=MagicMock(return_value=0.5))))
                ctrl.target_network = None
                ctrl.q_network = MagicMock()
                ctrl.optimizer = MagicMock()
                ctrl.scheduler = MagicMock(get_last_lr=MagicMock(return_value=[1e-4]))
                ctrl.replay_buffer = []
                ctrl.episode_rewards = deque(maxlen=100)
                ctrl.episode_lengths = deque(maxlen=100)
                ctrl.q_value_history = deque(maxlen=1000)
                ctrl.writer = None
            return ctrl
        return DeepRLController(state_dim=10, action_dim=5,
                                algorithm=algo, config=config)

    def test_init_variants(self):
        for algo in [DRLAlgorithm.DQN, DRLAlgorithm.DOUBLE_DQN, DRLAlgorithm.DUELING_DQN]:
            ctrl = self._make(algo)
            assert ctrl.algorithm == algo

    def test_select_action_explore(self):
        ctrl = self._make()
        ctrl.epsilon = 1.0
        action = ctrl.select_action(np.zeros(10, dtype=np.float32), True)
        assert 0 <= action < 5

    def test_select_action_exploit(self):
        ctrl = self._make()
        ctrl.epsilon = 0.0
        ctrl.select_action(np.zeros(10, dtype=np.float32), True)

    def test_select_action_ppo(self):
        ctrl = self._make(DRLAlgorithm.PPO)
        mock_dist = MagicMock()
        mock_dist.sample.return_value = MagicMock(item=MagicMock(return_value=1))
        mock_torch.distributions.Categorical.return_value = mock_dist
        ctrl.select_action(np.zeros(10, dtype=np.float32), True)
        ctrl.select_action(np.zeros(10, dtype=np.float32), False)

    def test_store_experience(self):
        ctrl = self._make(priority=True)
        s = np.zeros(10, dtype=np.float32)
        ctrl.store_experience(s, 0, 1.0, s, False)
        assert len(ctrl.replay_buffer) == 1

    def test_train_step_warmup(self):
        ctrl = self._make()
        ctrl.config.warmup_steps = 99999
        assert ctrl.train_step()["loss"] == 0.0

    def test_update_exploration(self):
        ctrl = self._make()
        ctrl.epsilon = 0.5
        ctrl._update_exploration()
        assert ctrl.epsilon < 0.5

    def test_update_target_soft_hard(self):
        ctrl = self._make()
        ctrl.config.tau = 0.005
        ctrl._update_target_network()
        ctrl.config.tau = 1.0
        ctrl._update_target_network()

    def test_save_load_model(self):
        ctrl = self._make()
        ctrl.save_model("/tmp/m.pt")
        mock_torch.load.return_value = {
            "steps": 10, "episodes": 1, "epsilon": 0.1, "beta": 0.9,
            "q_network": {}, "target_network": {}, "optimizer": {}, "scheduler": {},
        }
        ctrl.load_model("/tmp/m.pt")
        assert ctrl.epsilon == 0.1

    def test_get_performance_summary(self):
        ctrl = self._make()
        ctrl.episode_rewards.append(1.0)
        ctrl.episode_lengths.append(10)
        ctrl.q_value_history.append(0.5)
        s = ctrl.get_performance_summary()
        assert "algorithm" in s

    def test_control_step(self):
        ctrl = self._make()
        ctrl._feature_engineer = MagicMock()
        ctrl._feature_engineer.extract_features.return_value = {"f": 1.0}
        ms = MagicMock()
        ms.power_output = 5.0
        ms.current_density = 1.0
        ms.health_metrics.overall_health_score = 0.8
        ms.fused_measurement.fusion_confidence = 0.9
        ms.anomalies = []
        action, info = ctrl.control_step(ms)
        assert "algorithm" in info

class TestFactory:
    def test_all(self):
        for algo in ["dqn", "double_dqn", "ppo", "a3c", "unknown"]:
            with patch.object(DeepRLController, "_build_networks"):
                ctrl = create_deep_rl_controller(10, 5, algorithm=algo)
            assert ctrl is not None