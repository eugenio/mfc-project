"""Tests for base_controller module - comprehensive coverage.

Covers BaseControllerConfig, BaseController, NeuralNetworkController,
and create_base_config factory function.
"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import types

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
    training = True


mock_nn.Module = MockModule
mock_nn.Linear = MagicMock(return_value=MagicMock())
mock_nn.ReLU = MagicMock(return_value=MagicMock())
mock_nn.Sequential = MagicMock(return_value=MagicMock())
mock_nn.functional = mock_F
mock_torch.nn = mock_nn
mock_torch.optim = mock_optim
mock_torch.device = MagicMock(return_value="cpu")
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available = MagicMock(return_value=False)
mock_torch.FloatTensor = MagicMock(return_value=MagicMock(
    unsqueeze=MagicMock(return_value=MagicMock(
        to=MagicMock(return_value=MagicMock())))))
mock_torch.nn.utils = MagicMock()
mock_torch.nn.utils.clip_grad_norm_ = MagicMock(
    return_value=MagicMock(item=MagicMock(return_value=1.0)))

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

from base_controller import (
    BaseController, BaseControllerConfig,
    NeuralNetworkController, create_base_config,
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


class ConcreteController(BaseController):
    def control_step(self, system_state): return 0, {}
    def train_step(self): return {"loss": 0.0}
    def save_model(self, path): pass
    def load_model(self, path): pass
    def get_performance_summary(self): return {}


class TestBaseControllerConfig:
    def test_defaults(self):
        c = BaseControllerConfig()
        assert c.learning_rate == 1e-4 and c.batch_size == 64
        assert c.history_maxlen == 1000 and c.device is None

    def test_custom(self):
        c = BaseControllerConfig(learning_rate=0.01, batch_size=32)
        assert c.learning_rate == 0.01 and c.batch_size == 32

    def test_invalid_lr(self):
        with pytest.raises(ValueError):
            BaseControllerConfig(learning_rate=-1)

    def test_invalid_batch(self):
        with pytest.raises(ValueError):
            BaseControllerConfig(batch_size=0)


class TestBaseController:
    def test_init(self):
        ctrl = ConcreteController(state_dim=10, action_dim=5)
        assert ctrl.state_dim == 10 and ctrl.action_dim == 5
        assert ctrl.steps == 0 and ctrl.episodes == 0

    def test_device(self):
        ctrl = ConcreteController(10, 5, device="cpu")
        assert ctrl.device is not None

    def test_feature_engineer_lazy(self):
        ctrl = ConcreteController(10, 5)
        assert ctrl._feature_engineer is None
        mock_fe = MagicMock()
        ctrl._feature_engineer = mock_fe
        assert ctrl.feature_engineer == mock_fe

    def test_feature_engineer_import(self):
        ctrl = ConcreteController(10, 5)
        mock_mod = types.ModuleType("ml_optimization")
        mock_mod.FeatureEngineer = MagicMock(return_value=MagicMock())
        sys.modules["ml_optimization"] = mock_mod
        try:
            fe = ctrl.feature_engineer
            assert fe is not None
        finally:
            del sys.modules["ml_optimization"]
            ctrl._feature_engineer = None

    def test_get_model_size_no_model(self):
        ctrl = ConcreteController(10, 5)
        result = ctrl.get_model_size()
        assert result["total_parameters"] == 0

    def test_get_model_size_with_model(self):
        ctrl = ConcreteController(10, 5)
        model = MagicMock()
        p = MagicMock(); p.numel.return_value = 100; p.requires_grad = True
        model.parameters.return_value = [p]
        ctrl.model = model
        result = ctrl.get_model_size()
        assert result["total_parameters"] == 100

    def test_get_model_size_explicit(self):
        ctrl = ConcreteController(10, 5)
        model = MagicMock()
        p = MagicMock(); p.numel.return_value = 50; p.requires_grad = True
        model.parameters.return_value = [p]
        result = ctrl.get_model_size(model)
        assert result["total_parameters"] == 50

    def test_get_model_size_q_network(self):
        ctrl = ConcreteController(10, 5)
        model = MagicMock()
        p = MagicMock(); p.numel.return_value = 75; p.requires_grad = True
        model.parameters.return_value = [p]
        ctrl.q_network = model
        result = ctrl.get_model_size()
        assert result["total_parameters"] == 75

    def test_get_model_size_policy_network(self):
        ctrl = ConcreteController(10, 5)
        model = MagicMock()
        p = MagicMock(); p.numel.return_value = 200; p.requires_grad = False
        model.parameters.return_value = [p]
        ctrl.policy_network = model
        result = ctrl.get_model_size()
        assert result["trainable_parameters"] == 0

    def test_clip_gradients(self):
        ctrl = ConcreteController(10, 5)
        model = MagicMock()
        result = ctrl.clip_gradients(model, max_norm=2.0)
        assert result == 1.0

    def test_clip_gradients_no_torch(self):
        ctrl = ConcreteController(10, 5)
        import base_controller as bc_mod
        with patch.object(bc_mod, "TORCH_AVAILABLE", False):
            with pytest.raises(ImportError):
                ctrl.clip_gradients(MagicMock())

    def test_prepare_state_tensor(self):
        ctrl = ConcreteController(10, 5)
        state = np.array([1.0, 2.0, 3.0])
        result = ctrl.prepare_state_tensor(state)

    def test_prepare_state_tensor_no_batch(self):
        ctrl = ConcreteController(10, 5)
        state = np.array([1.0, 2.0])
        result = ctrl.prepare_state_tensor(state, add_batch_dim=False)

    def test_prepare_state_no_torch(self):
        ctrl = ConcreteController(10, 5)
        import base_controller as bc_mod
        with patch.object(bc_mod, "TORCH_AVAILABLE", False):
            with pytest.raises(ImportError):
                ctrl.prepare_state_tensor(np.array([1.0]))

    def test_log_training_step(self):
        ctrl = ConcreteController(10, 5)
        ctrl.log_training_step(0.5)
        assert len(ctrl.loss_history) == 1

    def test_log_training_step_with_metrics(self):
        ctrl = ConcreteController(10, 5)
        ctrl.log_training_step(0.3, {"acc": 0.9})
        assert len(ctrl.training_history) == 1

    def test_log_training_step_multiple(self):
        ctrl = ConcreteController(10, 5)
        for i in range(101):
            ctrl.steps = i
            ctrl.log_training_step(float(i))

    def test_get_history_summary(self):
        ctrl = ConcreteController(10, 5)
        summary = ctrl.get_history_summary()
        assert summary["total_steps"] == 0

    def test_get_history_summary_with_data(self):
        ctrl = ConcreteController(10, 5)
        ctrl.loss_history.extend([0.1, 0.2, 0.3])
        ctrl.reward_history.extend([1.0, 2.0])
        summary = ctrl.get_history_summary()
        assert summary["avg_loss"] == pytest.approx(0.2)

    def test_save_base_state(self):
        ctrl = ConcreteController(10, 5)
        ctrl.steps = 50; ctrl.episodes = 10
        state = ctrl._save_base_state()
        assert state["steps"] == 50 and state["episodes"] == 10

    def test_load_base_state(self):
        ctrl = ConcreteController(10, 5)
        ctrl._load_base_state({
            "steps": 100, "episodes": 20,
            "loss_history": [0.1, 0.2],
            "reward_history": [1.0]})
        assert ctrl.steps == 100 and ctrl.episodes == 20
        assert len(ctrl.loss_history) == 2

    def test_load_base_state_empty(self):
        ctrl = ConcreteController(10, 5)
        ctrl._load_base_state({})
        assert ctrl.steps == 0

    def test_history_maxlen(self):
        ctrl = ConcreteController(10, 5, history_maxlen=5)
        for i in range(10):
            ctrl.loss_history.append(float(i))
        assert len(ctrl.loss_history) == 5


class TestNeuralNetworkController:
    def _make(self):
        class Concrete(NeuralNetworkController):
            def control_step(self, s): return 0, {}
            def train_step(self): return {}
            def save_model(self, p): pass
            def load_model(self, p): pass
            def get_performance_summary(self): return {}
        return Concrete(10, 5)

    def test_init(self):
        ctrl = self._make()
        assert ctrl.learning_rate == 1e-4

    def test_get_learning_rate_no_scheduler(self):
        ctrl = self._make()
        assert ctrl.get_learning_rate() == 1e-4

    def test_get_learning_rate_with_scheduler(self):
        ctrl = self._make()
        ctrl.scheduler = MagicMock()
        ctrl.scheduler.get_last_lr.return_value = [0.001]
        assert ctrl.get_learning_rate() == 0.001

    def test_get_learning_rate_empty_lr(self):
        ctrl = self._make()
        ctrl.scheduler = MagicMock()
        ctrl.scheduler.get_last_lr.return_value = []
        assert ctrl.get_learning_rate() == 1e-4

    def test_step_scheduler(self):
        ctrl = self._make()
        ctrl.scheduler = MagicMock()
        ctrl.step_scheduler()
        ctrl.scheduler.step.assert_called_once()

    def test_step_scheduler_none(self):
        ctrl = self._make()
        ctrl.step_scheduler()

    def test_zero_grad(self):
        ctrl = self._make()
        ctrl.optimizer = MagicMock()
        ctrl.zero_grad()
        ctrl.optimizer.zero_grad.assert_called_once()

    def test_zero_grad_none(self):
        ctrl = self._make()
        ctrl.zero_grad()

    def test_optimizer_step(self):
        ctrl = self._make()
        ctrl.optimizer = MagicMock()
        ctrl.optimizer_step()
        ctrl.optimizer.step.assert_called_once()

    def test_optimizer_step_none(self):
        ctrl = self._make()
        ctrl.optimizer_step()


class TestCreateBaseConfig:
    def test_default(self):
        c = create_base_config()
        assert isinstance(c, BaseControllerConfig)

    def test_custom(self):
        c = create_base_config(learning_rate=0.1)
        assert c.learning_rate == 0.1

    def test_invalid(self):
        with pytest.raises(ValueError):
            create_base_config(learning_rate=-1)
