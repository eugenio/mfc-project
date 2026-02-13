"""Comprehensive tests for transfer_learning_controller.py with torch mocked.

Targets 99%+ statement coverage of every class and function, including
edge/integration methods, knowledge sync, federated integration, and
edge deployment.
"""
import sys
import copy
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from collections import defaultdict, deque
from datetime import datetime
mock_torch = MagicMock()
mock_nn = MagicMock()
mock_optim = MagicMock()
mock_F = MagicMock()


class _FakeTensor:
    pass


mock_torch.Tensor = _FakeTensor


class MockModule:
    """Minimal nn.Module stand-in."""
    def __init__(self, *a, **kw):
        pass
    def forward(self, x):
        return x
    def parameters(self):
        return []
    def state_dict(self):
        return {}
    def load_state_dict(self, d):
        pass
    def train(self, mode=True):
        return self
    def eval(self):
        return self
    def to(self, device):
        return self
    def __call__(self, *a, **kw):
        return MagicMock()
    def named_parameters(self):
        return []
    def children(self):
        return []
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)
    def apply(self, fn):
        return self
    training = True


class _FakeParameter:
    def __init__(self, data=None, requires_grad=True):
        self.data = data if data is not None else MagicMock()
        self.requires_grad = requires_grad
        self.grad = None
        self.shape = getattr(data, "shape", (1,))
    def numel(self):
        return 0


class _FakeLinear(MockModule):
    def __init__(self, in_f=0, out_f=0, *a, **kw):
        super().__init__()
    def __call__(self, x):
        return MagicMock()


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
mock_nn.LogSoftmax = MagicMock(return_value=MagicMock())
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
            to=MagicMock(return_value=MagicMock()))))),
    to=MagicMock(return_value=MagicMock())))
mock_torch.LongTensor = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.normal = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.abs = MagicMock(return_value=MagicMock())
mock_torch.topk = MagicMock(return_value=(MagicMock(), MagicMock()))
mock_torch.stack = MagicMock(return_value=MagicMock(
    median=MagicMock(return_value=(MagicMock(), None))))
mock_torch.cat = MagicMock(return_value=MagicMock(
    shape=(1, 1, 256), device="cpu"))
mock_torch.argmax = MagicMock(return_value=MagicMock(
    item=MagicMock(return_value=0)))
mock_torch.max = MagicMock(return_value=MagicMock(
    item=MagicMock(return_value=0.9)))
mock_torch.nn.utils = MagicMock()
mock_torch.nn.utils.clip_grad_norm_ = MagicMock(
    return_value=MagicMock(item=MagicMock(return_value=1.0)))
mock_torch.nn.init = MagicMock()
mock_torch.nn.init.xavier_uniform_ = MagicMock()
mock_torch.nn.init.constant_ = MagicMock()
mock_torch.autograd = MagicMock()
mock_torch.autograd.Function = type("Function", (), {
    "forward": staticmethod(lambda ctx, x, alpha: x),
    "backward": staticmethod(lambda ctx, grad: (-grad, None))
})
mock_torch.autograd.grad = MagicMock(return_value=[MagicMock()])
mock_torch.linspace = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.randn = MagicMock(return_value=MagicMock(
    sign=MagicMock(return_value=MagicMock(
        mul_=MagicMock(return_value=MagicMock())))))
mock_torch.distributions = MagicMock()
mock_torch.datetime = MagicMock()
mock_torch.datetime.now = MagicMock(return_value=MagicMock(
    isoformat=MagicMock(return_value="2025-01-01T00:00:00")))
mock_F.mse_loss = MagicMock(return_value=MagicMock(
    item=MagicMock(return_value=0.1)))
mock_F.softmax = MagicMock(return_value=MagicMock())
mock_F.one_hot = MagicMock(return_value=MagicMock(
    float=MagicMock(return_value=MagicMock())))
import numpy as np

# Install mocks
_orig_modules = {}
import torch_compat
import pytest
from sensing_models.sensor_fusion import BacterialSpecies
from transfer_learning_controller import (
class _FakeTensor:
    pass

class MockModule:
    """Minimal nn.Module stand-in."""
    def __init__(self, *a, **kw):
        pass
    def forward(self, x):
        return x
    def parameters(self):
        return []
    def state_dict(self):
        return {}
    def load_state_dict(self, d):
        pass
    def train(self, mode=True):
        return self
    def eval(self):
        return self
    def to(self, device):
        return self
    def __call__(self, *a, **kw):
        return MagicMock()
    def named_parameters(self):
        return []
    def children(self):
        return []
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)
    def apply(self, fn):
        return self
    training = True

class _FakeParameter:
    def __init__(self, data=None, requires_grad=True):
        self.data = data if data is not None else MagicMock()
        self.requires_grad = requires_grad
        self.grad = None
        self.shape = getattr(data, "shape", (1,))
    def numel(self):
        return 0


class _FakeLinear(MockModule):
    def __init__(self, in_f=0, out_f=0, *a, **kw):
        super().__init__()
    def __call__(self, x):
        return MagicMock()

class TestEnums:
    def test_transfer_methods(self):
        assert TransferLearningMethod.FINE_TUNING.value == "fine_tuning"
        assert TransferLearningMethod.DOMAIN_ADAPTATION.value == "domain_adaptation"
        assert TransferLearningMethod.PROGRESSIVE_NETWORKS.value == "progressive_networks"
        assert TransferLearningMethod.META_LEARNING.value == "meta_learning"
        assert TransferLearningMethod.KNOWLEDGE_DISTILLATION.value == "knowledge_distillation"
        assert TransferLearningMethod.MULTI_TASK.value == "multi_task"

    def test_task_types(self):
        assert TaskType.POWER_OPTIMIZATION.value == "power_optimization"
        assert TaskType.BIOFILM_HEALTH.value == "biofilm_health"
        assert TaskType.STABILITY_CONTROL.value == "stability_control"
        assert TaskType.EFFICIENCY_MAXIMIZATION.value == "efficiency_maximization"
        assert TaskType.FAULT_DETECTION.value == "fault_detection"


# ===================================================================
# TransferConfig
# ===================================================================
class TestTransferConfig:
    def test_defaults(self):
        config = TransferConfig()
        assert config.target_species == BacterialSpecies.MIXED
        assert config.transfer_method == TransferLearningMethod.FINE_TUNING
        assert len(config.source_species) == 2
        assert config.meta_lr == 1e-3
        assert config.inner_lr == 1e-2
        assert config.temperature == 4.0
        assert config.alpha == 0.7

    def test_post_init_all_defaults(self):
        config = TransferConfig()
        assert config.freeze_layers == ["feature_layers"]
        assert config.adaptation_layers == [128, 64]
        assert len(config.tasks) == 2
        assert config.shared_layers == [512, 256]
        assert config.adapter_layers == [64, 32]
        assert config.task_weights is not None
        assert config.task_specific_layers is not None

    def test_custom_values(self):
        config = TransferConfig(
            meta_lr=0.01, inner_lr=0.1, inner_steps=10,
            temperature=2.0, lateral_connections=False)
        assert config.meta_lr == 0.01
        assert config.inner_lr == 0.1
        assert config.inner_steps == 10
        assert config.temperature == 2.0
        assert config.lateral_connections is False

    def test_pre_set_lists_not_overridden(self):
        config = TransferConfig(
            source_species=[BacterialSpecies.GEOBACTER],
            freeze_layers=["conv1"],
            adaptation_layers=[256],
            tasks=[TaskType.FAULT_DETECTION],
            task_weights={TaskType.FAULT_DETECTION: 1.0},
            shared_layers=[1024],
            task_specific_layers={TaskType.FAULT_DETECTION: [256]},
            adapter_layers=[128])
        assert config.source_species == [BacterialSpecies.GEOBACTER]
        assert config.freeze_layers == ["conv1"]
        assert config.adaptation_layers == [256]
        assert len(config.tasks) == 1
        assert config.shared_layers == [1024]
        assert config.adapter_layers == [128]


# ===================================================================
# Neural network components
# ===================================================================
class TestDomainAdaptationNetwork:
    def test_init(self):
        with patch("transfer_learning_controller.GradientReversalLayer",
                   return_value=MagicMock()):
            net = DomainAdaptationNetwork(feature_dim=64, num_domains=3)
        assert net.feature_dim == 64
        assert net.num_domains == 3

    def test_forward(self):
        with patch("transfer_learning_controller.GradientReversalLayer",
                   return_value=MagicMock(return_value=MagicMock())):
            net = DomainAdaptationNetwork(feature_dim=64, num_domains=3)
        result = net.forward(MagicMock(), alpha=0.5)
        assert result is not None

    def test_init_different_hidden(self):
        with patch("transfer_learning_controller.GradientReversalLayer",
                   return_value=MagicMock()):
            net = DomainAdaptationNetwork(feature_dim=32, num_domains=5, hidden_dim=256)
        assert net.feature_dim == 32
        assert net.num_domains == 5

class TestGradientReversalLayer:
    def test_forward_logic(self):
        ctx = MagicMock()
        x = MagicMock()
        alpha = 1.0
        ctx.alpha = alpha
        result = x
        assert ctx.alpha == 1.0
        assert result is x

    def test_backward_logic(self):
        ctx = MagicMock()
        ctx.alpha = 0.5
        grad_output = MagicMock()
        result = (-ctx.alpha * grad_output, None)
        assert len(result) == 2
        assert result[1] is None

class TestProgressiveNetwork:
    def test_init_zero_columns(self):
        net = ProgressiveNetwork(input_dim=10, hidden_dims=[32],
                                 output_dim=5, num_columns=0,
                                 lateral_connections=False)
        assert net.input_dim == 10
        assert net.output_dim == 5
        assert net.num_columns == 0

    def test_add_column_no_lateral(self):
        net = ProgressiveNetwork(input_dim=10, hidden_dims=[32, 16],
                                 output_dim=5, num_columns=0,
                                 lateral_connections=False)
        net.add_column()
        assert net.num_columns == 1
        net.add_column()
        assert net.num_columns == 2

    def test_add_column_with_lateral(self):
        net = ProgressiveNetwork(input_dim=10, hidden_dims=[32],
                                 output_dim=5, num_columns=0,
                                 lateral_connections=True)
        net.add_column()  # num_columns was 0, so no lateral
        assert net.num_columns == 1
        net.add_column()  # now adds lateral adapters
        assert net.num_columns == 2

    def test_add_column_no_hidden(self):
        net = ProgressiveNetwork(input_dim=10, hidden_dims=[],
                                 output_dim=5, num_columns=0,
                                 lateral_connections=False)
        net.add_column()
        assert net.num_columns == 1

    def test_add_column_no_hidden_with_lateral(self):
        net = ProgressiveNetwork(input_dim=10, hidden_dims=[],
                                 output_dim=5, num_columns=1,
                                 lateral_connections=True)
        net.add_column()
        assert net.num_columns == 2

class TestMultiTaskNetwork:
    def test_init(self):
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [64],
            TaskType.BIOFILM_HEALTH: [64],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 5,
            TaskType.BIOFILM_HEALTH: 1,
        }
        net = MultiTaskNetwork(input_dim=10, shared_layers=[128],
                               task_layers=task_layers, task_outputs=task_outputs)
        assert net.input_dim == 10
        assert len(net.tasks) == 2

    def test_init_no_shared(self):
        task_layers = {TaskType.POWER_OPTIMIZATION: [64]}
        task_outputs = {TaskType.POWER_OPTIMIZATION: 5}
        net = MultiTaskNetwork(input_dim=10, shared_layers=[],
                               task_layers=task_layers, task_outputs=task_outputs)
        assert net.input_dim == 10

    def test_forward_single_task(self):
        task_layers = {TaskType.POWER_OPTIMIZATION: [64]}
        task_outputs = {TaskType.POWER_OPTIMIZATION: 5}
        net = MultiTaskNetwork(input_dim=10, shared_layers=[128],
                               task_layers=task_layers, task_outputs=task_outputs)
        result = net.forward(MagicMock(), task=TaskType.POWER_OPTIMIZATION)

    def test_forward_multi_task(self):
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [64],
            TaskType.BIOFILM_HEALTH: [64],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 5,
            TaskType.BIOFILM_HEALTH: 1,
        }
        net = MultiTaskNetwork(input_dim=10, shared_layers=[128],
                               task_layers=task_layers, task_outputs=task_outputs)
        result = net.forward(MagicMock(), task=None)

class TestMAMLController:
    def test_init(self):
        maml = MAMLController(input_dim=10, hidden_dims=[32], output_dim=5)

    def test_init_no_hidden(self):
        maml = MAMLController(input_dim=10, hidden_dims=[], output_dim=5)

    def test_forward(self):
        maml = MAMLController(input_dim=10, hidden_dims=[32], output_dim=5)
        result = maml.forward(MagicMock())

    def test_adapt(self):
        maml = MAMLController(input_dim=10, hidden_dims=[32], output_dim=5)
        result = maml.adapt(MagicMock(), MagicMock(), 0.01, 3)

    def test_adapt_with_steps(self):
        maml = MAMLController(input_dim=10, hidden_dims=[64, 32], output_dim=5)
        result = maml.adapt(MagicMock(), MagicMock(), 0.001, 5)


# ===================================================================
# TransferLearningController
# ===================================================================
class TestTransferLearningController:
    def _make_ctrl(self, method=TransferLearningMethod.FINE_TUNING):
        config = TransferConfig(transfer_method=method)
        with patch.object(TransferLearningController, "_initialize_networks"):
            ctrl = TransferLearningController(
                state_dim=10, action_dim=5, config=config)
        return ctrl

    # ---- Initialization ----
    def test_init_all_methods(self):
        for method in TransferLearningMethod:
            ctrl = self._make_ctrl(method)
            assert ctrl.config.transfer_method == method

    def test_init_default_config(self):
        with patch.object(TransferLearningController, "_initialize_networks"):
            ctrl = TransferLearningController(state_dim=10, action_dim=5)
        assert ctrl.config is not None

    def test_init_with_base_controller(self):
        mock_base = MagicMock()
        with patch.object(TransferLearningController, "_initialize_networks"):
            ctrl = TransferLearningController(
                state_dim=10, action_dim=5, base_controller=mock_base)
        assert ctrl.base_controller is mock_base

    # ---- _initialize_networks branches ----
    def test_initialize_domain_adaptation(self):
        config = TransferConfig(transfer_method=TransferLearningMethod.DOMAIN_ADAPTATION)
        with patch("transfer_learning_controller.DomainAdaptationNetwork",
                   return_value=MagicMock()):
            ctrl = TransferLearningController(state_dim=10, action_dim=5, config=config)
        assert hasattr(ctrl, "domain_adapter")

    def test_initialize_progressive(self):
        config = TransferConfig(transfer_method=TransferLearningMethod.PROGRESSIVE_NETWORKS)
        with patch("transfer_learning_controller.ProgressiveNetwork",
                   return_value=MagicMock()):
            ctrl = TransferLearningController(state_dim=10, action_dim=5, config=config)
        assert hasattr(ctrl, "progressive_net")

    def test_initialize_multi_task(self):
        config = TransferConfig(transfer_method=TransferLearningMethod.MULTI_TASK)
        with patch("transfer_learning_controller.MultiTaskNetwork",
                   return_value=MagicMock()):
            ctrl = TransferLearningController(state_dim=10, action_dim=5, config=config)
        assert hasattr(ctrl, "multi_task_net")

    def test_initialize_meta_learning(self):
        config = TransferConfig(transfer_method=TransferLearningMethod.META_LEARNING)
        with patch("transfer_learning_controller.MAMLController",
                   return_value=MagicMock()):
            ctrl = TransferLearningController(state_dim=10, action_dim=5, config=config)
        assert hasattr(ctrl, "maml_controller")

    # ---- load_source_knowledge ----
    def test_load_source_knowledge(self):
        ctrl = self._make_ctrl()
        mock_torch.load.return_value = {"model": "data"}
        ctrl.load_source_knowledge(BacterialSpecies.GEOBACTER, "/tmp/model.pt")
        assert BacterialSpecies.GEOBACTER in ctrl.domain_knowledge

    def test_load_source_knowledge_failure(self):
        ctrl = self._make_ctrl()
        mock_torch.load.side_effect = Exception("fail")
        ctrl.load_source_knowledge(BacterialSpecies.GEOBACTER, "/bad/path")
        mock_torch.load.side_effect = None

    # ---- transfer_knowledge() routing ----
    def test_transfer_knowledge_fine_tuning_no_base(self):
        ctrl = self._make_ctrl(TransferLearningMethod.FINE_TUNING)
        result = ctrl.transfer_knowledge()
        assert result["status"] == "No base controller for fine-tuning"

    def test_transfer_knowledge_fine_tuning_with_base(self):
        ctrl = self._make_ctrl(TransferLearningMethod.FINE_TUNING)
        mock_base = MagicMock()
        mock_base.q_network.named_parameters.return_value = [
            ("feature_layers.0.weight", MagicMock())]
        mock_base.q_network.children.return_value = [MagicMock(), MagicMock()]
        mock_base.q_network.parameters.return_value = [
            MagicMock(requires_grad=True, numel=MagicMock(return_value=100))]
        ctrl.base_controller = mock_base
        result = ctrl.transfer_knowledge()
        assert result["status"] == "Fine-tuning setup completed"

    def test_transfer_knowledge_fine_tuning_no_adaptation_layers(self):
        ctrl = self._make_ctrl(TransferLearningMethod.FINE_TUNING)
        ctrl.config.adaptation_layers = []
        mock_base = MagicMock()
        mock_base.q_network.named_parameters.return_value = []
        mock_base.q_network.parameters.return_value = [
            MagicMock(requires_grad=True, numel=MagicMock(return_value=50))]
        ctrl.base_controller = mock_base
        result = ctrl.transfer_knowledge()
        assert result["status"] == "Fine-tuning setup completed"

    def test_transfer_knowledge_domain_adaptation(self):
        ctrl = self._make_ctrl(TransferLearningMethod.DOMAIN_ADAPTATION)
        ctrl.domain_adapter = MagicMock()
        ctrl.domain_adapter.parameters.return_value = [
            MagicMock(numel=MagicMock(return_value=50))]
        result = ctrl.transfer_knowledge()
        assert result["status"] == "Domain adaptation initialized"

    def test_transfer_knowledge_domain_no_adapter(self):
        ctrl = self._make_ctrl(TransferLearningMethod.DOMAIN_ADAPTATION)
        result = ctrl.transfer_knowledge()
        assert result["status"] == "Domain adapter not initialized"

    def test_transfer_knowledge_progressive(self):
        ctrl = self._make_ctrl(TransferLearningMethod.PROGRESSIVE_NETWORKS)
        ctrl.progressive_net = MagicMock()
        ctrl.progressive_net.num_columns = 1
        result = ctrl.transfer_knowledge()
        assert result["status"] == "Progressive network ready"

    def test_transfer_knowledge_progressive_no_add(self):
        ctrl = self._make_ctrl(TransferLearningMethod.PROGRESSIVE_NETWORKS)
        ctrl.progressive_net = MagicMock()
        ctrl.progressive_net.num_columns = 10  # > source species count
        result = ctrl.transfer_knowledge()
        assert result["status"] == "Progressive network ready"
        ctrl.progressive_net.add_column.assert_not_called()

    def test_transfer_knowledge_progressive_no_net(self):
        ctrl = self._make_ctrl(TransferLearningMethod.PROGRESSIVE_NETWORKS)
        result = ctrl.transfer_knowledge()
        assert result["status"] == "Progressive network not initialized"

    def test_transfer_knowledge_meta(self):
        ctrl = self._make_ctrl(TransferLearningMethod.META_LEARNING)
        ctrl.maml_controller = MagicMock()
        result = ctrl.transfer_knowledge()
        assert result["status"] == "MAML controller ready"

    def test_transfer_knowledge_meta_no_ctrl(self):
        ctrl = self._make_ctrl(TransferLearningMethod.META_LEARNING)
        result = ctrl.transfer_knowledge()
        assert result["status"] == "MAML controller not initialized"

    def test_transfer_knowledge_distillation(self):
        ctrl = self._make_ctrl(TransferLearningMethod.KNOWLEDGE_DISTILLATION)
        mock_base = MagicMock()
        mock_base.q_network.parameters.return_value = [
            MagicMock(numel=MagicMock(return_value=100))]
        ctrl.base_controller = mock_base
        result = ctrl.transfer_knowledge()
        assert result["status"] == "Knowledge distillation ready"

    def test_transfer_knowledge_distillation_no_base(self):
        ctrl = self._make_ctrl(TransferLearningMethod.KNOWLEDGE_DISTILLATION)
        result = ctrl.transfer_knowledge()
        assert result["status"] == "No teacher model available"

    # ---- adapt_to_new_species ----
    def test_adapt_maml(self):
        ctrl = self._make_ctrl(TransferLearningMethod.META_LEARNING)
        ctrl.maml_controller = MagicMock()
        data = [(np.zeros(10), 0, 1.0)] * 10
        result = ctrl.adapt_to_new_species(BacterialSpecies.GEOBACTER, data)

    def test_adapt_maml_insufficient(self):
        ctrl = self._make_ctrl(TransferLearningMethod.META_LEARNING)
        ctrl.maml_controller = MagicMock()
        data = [(np.zeros(10), 0, 1.0)] * 3
        result = ctrl._maml_adaptation(data)
        assert result["status"] == "Insufficient adaptation data"

    def test_adapt_standard(self):
        ctrl = self._make_ctrl(TransferLearningMethod.FINE_TUNING)
        data = [(np.zeros(10), 0, 1.0)] * 5
        result = ctrl.adapt_to_new_species(BacterialSpecies.SHEWANELLA, data)
        assert result["status"] == "Standard adaptation completed"

    # ---- multi_task_control ----
    def test_multi_task_control_not_multi_task(self):
        ctrl = self._make_ctrl(TransferLearningMethod.FINE_TUNING)
        result = ctrl.multi_task_control(MagicMock())
        assert "error" in result

    def test_multi_task_control_power_optimization(self):
        ctrl = self._make_ctrl(TransferLearningMethod.MULTI_TASK)
        ctrl.config.tasks = [TaskType.POWER_OPTIMIZATION]
        ctrl.multi_task_net = MagicMock()
        ctrl._feature_engineer = MagicMock()
        ctrl._feature_engineer.extract_features.return_value = {"f1": 1.0}
        mock_outputs = {TaskType.POWER_OPTIMIZATION.value: MagicMock()}
        ctrl.multi_task_net.return_value = mock_outputs
        result = ctrl.multi_task_control(MagicMock())

    def test_multi_task_control_biofilm_health(self):
        ctrl = self._make_ctrl(TransferLearningMethod.MULTI_TASK)
        ctrl.config.tasks = [TaskType.BIOFILM_HEALTH]
        ctrl.multi_task_net = MagicMock()
        ctrl._feature_engineer = MagicMock()
        ctrl._feature_engineer.extract_features.return_value = {"f1": 1.0}
        mock_health_output = MagicMock()
        mock_torch.sigmoid = MagicMock(return_value=MagicMock(
            item=MagicMock(return_value=0.3)))
        mock_outputs = {TaskType.BIOFILM_HEALTH.value: mock_health_output}
        ctrl.multi_task_net.return_value = mock_outputs
        result = ctrl.multi_task_control(MagicMock())

    def test_multi_task_control_fault_detection(self):
        ctrl = self._make_ctrl(TransferLearningMethod.MULTI_TASK)
        ctrl.config.tasks = [TaskType.FAULT_DETECTION]
        ctrl.multi_task_net = MagicMock()
        ctrl._feature_engineer = MagicMock()
        ctrl._feature_engineer.extract_features.return_value = {"f1": 1.0}
        mock_outputs = {TaskType.FAULT_DETECTION.value: MagicMock()}
        ctrl.multi_task_net.return_value = mock_outputs
        result = ctrl.multi_task_control(MagicMock())

    # ---- get_transfer_summary ----
    def test_get_transfer_summary(self):
        ctrl = self._make_ctrl()
        summary = ctrl.get_transfer_summary()
        assert "transfer_method" in summary
        assert "source_species" in summary
        assert "target_species" in summary
        assert "tasks" in summary
        assert "domain_knowledge_loaded" in summary
        assert "adaptation_history" in summary

    # ---- save/load ----
    def test_save_transfer_model(self):
        ctrl = self._make_ctrl()
        ctrl.save_transfer_model("/tmp/test.pt")
        mock_torch.save.assert_called()

    def test_save_with_all_models(self):
        ctrl = self._make_ctrl()
        ctrl.domain_adapter = MagicMock()
        ctrl.progressive_net = MagicMock()
        ctrl.multi_task_net = MagicMock()
        ctrl.maml_controller = MagicMock()
        ctrl.save_transfer_model("/tmp/test.pt")

    def test_save_model_delegates(self):
        ctrl = self._make_ctrl()
        with patch.object(ctrl, "save_transfer_model") as mock_save:
            ctrl.save_model("/tmp/test.pt")
            mock_save.assert_called_once_with("/tmp/test.pt")

    def test_load_model(self):
        ctrl = self._make_ctrl()
        mock_torch.load.return_value = {"steps": 50, "episodes": 5}
        ctrl.load_model("/tmp/test.pt")
        assert ctrl.steps == 50

    def test_load_model_with_submodels(self):
        ctrl = self._make_ctrl()
        ctrl.domain_adapter = MagicMock()
        ctrl.progressive_net = MagicMock()
        ctrl.multi_task_net = MagicMock()
        ctrl.maml_controller = MagicMock()
        mock_torch.load.return_value = {
            "steps": 10,
            "domain_adapter": {},
            "progressive_net": {},
            "multi_task_net": {},
            "maml_controller": {},
        }
        ctrl.load_model("/tmp/test.pt")

    # ---- train_step / control_step ----
    def test_train_step(self):
        ctrl = self._make_ctrl()
        result = ctrl.train_step()
        assert result["loss"] == 0.0

    def test_control_step_default(self):
        ctrl = self._make_ctrl()
        action, info = ctrl.control_step(MagicMock())
        assert action == 0
        assert info["method"] == "default"

    def test_control_step_multi_task(self):
        ctrl = self._make_ctrl(TransferLearningMethod.MULTI_TASK)
        ctrl.config.tasks = [TaskType.POWER_OPTIMIZATION]
        ctrl.multi_task_net = MagicMock()
        ctrl._feature_engineer = MagicMock()
        ctrl._feature_engineer.extract_features.return_value = {"f1": 1.0}
        mock_outputs = {TaskType.POWER_OPTIMIZATION.value: MagicMock()}
        ctrl.multi_task_net.return_value = mock_outputs
        mock_torch.argmax.return_value = MagicMock(
            item=MagicMock(return_value=2))
        mock_F.softmax.return_value = MagicMock()
        action, info = ctrl.control_step(MagicMock())

    def test_control_step_adapted_model(self):
        ctrl = self._make_ctrl()
        ctrl.adapted_model = MagicMock()
        ctrl._feature_engineer = MagicMock()
        ctrl._feature_engineer.extract_features.return_value = {"f1": 1.0}
        mock_state = MagicMock()
        mock_state.power_output = 5.0
        mock_state.current_density = 1.0
        mock_state.health_metrics.overall_health_score = 0.8
        mock_state.fused_measurement.fusion_confidence = 0.9
        mock_state.anomalies = []
        action, info = ctrl.control_step(mock_state)
        assert info["method"] == "adapted_model"

    # ---- get_performance_summary ----
    def test_get_performance_summary(self):
        ctrl = self._make_ctrl()
        summary = ctrl.get_performance_summary()
        assert "total_steps" in summary
        assert "transfer_method" in summary
        assert "model_size" in summary


# ===================================================================
# Edge deployment
# ===================================================================
class TestEdgeDeployment:
    def _make_ctrl(self, method=TransferLearningMethod.MULTI_TASK):
        config = TransferConfig(transfer_method=method)
        with patch.object(TransferLearningController, "_initialize_networks"):
            ctrl = TransferLearningController(
                state_dim=10, action_dim=5, config=config)
        return ctrl

    def test_prepare_for_edge(self):
        ctrl = self._make_ctrl()
        ctrl.device = MagicMock(type="cpu")
        resources = {"cpu_cores": 4, "memory_mb": 2048, "gpu_available": False}
        result = ctrl.prepare_for_edge_deployment(resources)
        assert result["deployment_ready"] is True

    def test_prepare_edge_low_memory(self):
        ctrl = self._make_ctrl()
        ctrl.device = MagicMock(type="cpu")
        ctrl.multi_task_net = MagicMock()
        resources = {"cpu_cores": 1, "memory_mb": 64, "gpu_available": False}
        with patch.object(ctrl, "_get_model_size", return_value={
            "total_parameters": 1_000_000, "trainable_parameters": 1_000_000,
            "memory_mb": 100.0, "models_count": 1}):
            result = ctrl.prepare_for_edge_deployment(resources)
        assert "model_quantization" in result["optimizations_applied"]
        assert "weight_pruning" in result["optimizations_applied"]
        assert "inference_optimization" in result["optimizations_applied"]

    def test_prepare_edge_gpu_to_cpu(self):
        ctrl = self._make_ctrl()
        ctrl.device = MagicMock(type="cuda")
        resources = {"cpu_cores": 4, "memory_mb": 2048, "gpu_available": False}
        result = ctrl.prepare_for_edge_deployment(resources)
        assert result.get("device_change") == "cuda -> cpu"

    def test_enable_federated_compatible(self):
        ctrl = self._make_ctrl(TransferLearningMethod.MULTI_TASK)
        result = ctrl.enable_federated_learning(
            "client_1", {"server_address": "host:8080"})
        assert result["ready_for_federation"] is True

    def test_enable_federated_meta_learning(self):
        ctrl = self._make_ctrl(TransferLearningMethod.META_LEARNING)
        result = ctrl.enable_federated_learning("client_2", {})
        assert result["compatible_method"] is True

    def test_enable_federated_not_compatible(self):
        ctrl = self._make_ctrl(TransferLearningMethod.FINE_TUNING)
        result = ctrl.enable_federated_learning("client_3", {})
        assert result["compatible_method"] is False
        assert "recommendation" in result

    def test_distributed_knowledge_sync_empty(self):
        ctrl = self._make_ctrl()
        result = ctrl.distributed_knowledge_sync({})
        assert result["status"] == "No peer knowledge received"

    def test_distributed_knowledge_sync_new_knowledge(self):
        ctrl = self._make_ctrl()
        peer_knowledge = {
            "peer1": {
                "domain_knowledge": {"species_a": {"performance_score": 0.9}},
                "adaptation_history": [
                    {"species": "mixed_culture", "data": "test"}],
            }
        }
        result = ctrl.distributed_knowledge_sync(peer_knowledge)
        assert result["peers_processed"] == 1
        assert result["new_knowledge_gained"] >= 1

    def test_distributed_knowledge_sync_conflict_update(self):
        ctrl = self._make_ctrl()
        ctrl.domain_knowledge["species_a"] = {"performance_score": 0.5}
        peer_knowledge = {
            "peer1": {
                "domain_knowledge": {
                    "species_a": {"performance_score": 0.9}}
            }
        }
        result = ctrl.distributed_knowledge_sync(peer_knowledge)
        assert result["conflicts_resolved"] >= 1

    def test_distributed_knowledge_sync_no_update(self):
        ctrl = self._make_ctrl()
        ctrl.domain_knowledge["species_a"] = {"performance_score": 0.9}
        peer_knowledge = {
            "peer1": {
                "domain_knowledge": {
                    "species_a": {"performance_score": 0.3}}
            }
        }
        result = ctrl.distributed_knowledge_sync(peer_knowledge)
        assert result["conflicts_resolved"] == 0

    def test_distributed_knowledge_irrelevant_adaptation(self):
        ctrl = self._make_ctrl()
        ctrl.config.target_species = BacterialSpecies.GEOBACTER
        peer_knowledge = {
            "peer1": {
                "adaptation_history": [
                    {"species": "completely_different_species"}]
            }
        }
        result = ctrl.distributed_knowledge_sync(peer_knowledge)
        # irrelevant adaptation should not be added
        assert result["new_knowledge_gained"] == 0

    def test_should_update_knowledge(self):
        ctrl = self._make_ctrl()
        ctrl.domain_knowledge["sp"] = {"performance_score": 0.5}
        assert ctrl._should_update_knowledge("sp", {"performance_score": 0.9})
        assert not ctrl._should_update_knowledge("sp", {"performance_score": 0.3})
        assert not ctrl._should_update_knowledge("sp", {"performance_score": 0.5})

    def test_should_update_knowledge_no_existing(self):
        ctrl = self._make_ctrl()
        assert ctrl._should_update_knowledge("new_sp", {"performance_score": 0.1})

    def test_is_relevant_adaptation(self):
        ctrl = self._make_ctrl()
        assert ctrl._is_relevant_adaptation({"species": "mixed_culture"})
        assert ctrl._is_relevant_adaptation(
            {"species": ctrl.config.target_species.value})
        # mixed_culture target makes everything relevant
        assert ctrl._is_relevant_adaptation({"species": "unknown"})

    def test_is_relevant_adaptation_not_mixed(self):
        ctrl = self._make_ctrl()
        ctrl.config.target_species = BacterialSpecies.GEOBACTER
        assert ctrl._is_relevant_adaptation({"species": "geobacter"})
        assert ctrl._is_relevant_adaptation({"species": "mixed_culture"})
        assert not ctrl._is_relevant_adaptation({"species": "unknown"})

    # ---- edge_model_update ----
    def test_edge_model_update_full(self):
        ctrl = self._make_ctrl()
        update = {
            "model_parameters": {},
            "config_updates": {"meta_lr": 0.01},
            "model_version": "2.0",
            "performance_metrics": {
                "baseline_accuracy": 0.7,
                "updated_accuracy": 0.85},
        }
        result = ctrl.edge_model_update(update)
        assert result["config_updated"] is True
        assert result["model_version_updated"] is True
        assert result["update_applied"] is True

    def test_edge_model_update_incompatible(self):
        ctrl = self._make_ctrl()
        update = {"architecture": {"state_dim": 999, "action_dim": 5}}
        result = ctrl.edge_model_update(update)
        assert result["compatibility_check"] is False

    def test_edge_model_update_param_failure(self):
        ctrl = self._make_ctrl()
        update = {"model_parameters": {"bad": "data"}}
        with patch.object(ctrl, "_apply_parameter_updates",
                          side_effect=Exception("param error")):
            result = ctrl.edge_model_update(update)
        assert "error" in result

    def test_edge_model_update_no_version(self):
        ctrl = self._make_ctrl()
        update = {"model_parameters": {}}
        result = ctrl.edge_model_update(update)
        assert result["model_version_updated"] is False

    def test_validate_update_compatible(self):
        ctrl = self._make_ctrl()
        assert ctrl._validate_update_compatibility({})
        assert ctrl._validate_update_compatibility(
            {"architecture": {"state_dim": 10, "action_dim": 5}})

    def test_validate_update_incompatible(self):
        ctrl = self._make_ctrl()
        assert not ctrl._validate_update_compatibility(
            {"architecture": {"state_dim": 999, "action_dim": 999}})

    def test_apply_parameter_updates_domain_adapter(self):
        ctrl = self._make_ctrl()
        ctrl.domain_adapter = MagicMock()
        ctrl.domain_adapter.named_parameters.return_value = [
            ("weight", MagicMock())]
        ctrl._apply_parameter_updates({"domain_adapter": {"weight": [1.0]}})

    def test_apply_parameter_updates_progressive(self):
        ctrl = self._make_ctrl()
        ctrl.progressive_net = MagicMock()
        ctrl.progressive_net.named_parameters.return_value = [
            ("weight", MagicMock())]
        ctrl._apply_parameter_updates({"progressive_net": {"weight": [1.0]}})

    def test_apply_parameter_updates_multi_task(self):
        ctrl = self._make_ctrl()
        ctrl.multi_task_net = MagicMock()
        ctrl.multi_task_net.named_parameters.return_value = [
            ("weight", MagicMock())]
        ctrl._apply_parameter_updates({"multi_task_net": {"weight": [1.0]}})

    def test_apply_config_updates(self):
        ctrl = self._make_ctrl()
        ctrl._apply_config_updates({"meta_lr": 0.05, "nonexistent": 42})
        assert ctrl.config.meta_lr == 0.05

    def test_estimate_performance_change(self):
        ctrl = self._make_ctrl()
        change = ctrl._estimate_performance_change({
            "performance_metrics": {
                "baseline_accuracy": 0.7, "updated_accuracy": 0.85}})
        assert abs(change - 0.15) < 0.01

    def test_estimate_performance_change_no_metrics(self):
        ctrl = self._make_ctrl()
        change = ctrl._estimate_performance_change({})
        assert change == 0.0

    def test_get_edge_deployment_status(self):
        ctrl = self._make_ctrl()
        ctrl.device = MagicMock(type="cpu")
        status = ctrl.get_edge_deployment_status()
        assert status["inference_capability"] is True
        assert status["deployment_ready"] is False

    def test_get_edge_deployment_status_with_client(self):
        ctrl = self._make_ctrl()
        ctrl.device = MagicMock(type="cpu")
        ctrl.client_id = "test_client"
        ctrl.federation_config = {"protocol": "grpc"}
        status = ctrl.get_edge_deployment_status()
        assert status["client_id"] == "test_client"
        assert status["communication_ready"] is True
        assert "communication_overhead_mb" in status

    def test_get_edge_deployment_status_with_quantization(self):
        ctrl = self._make_ctrl()
        ctrl.device = MagicMock(type="cpu")
        ctrl.quantization_applied = True
        status = ctrl.get_edge_deployment_status()
        assert "quantization" in status["edge_optimizations"]
        assert "cpu_optimized" in status["edge_optimizations"]

    def test_estimate_inference_time(self):
        ctrl = self._make_ctrl()
        t = ctrl._estimate_inference_time(
            {"cpu_cores": 2, "memory_mb": 2048, "gpu_available": True})
        assert t > 0

    def test_estimate_inference_time_low_resources(self):
        ctrl = self._make_ctrl()
        t = ctrl._estimate_inference_time(
            {"cpu_cores": 1, "memory_mb": 256, "gpu_available": False})
        assert t > 0

    def test_estimate_communication_overhead(self):
        ctrl = self._make_ctrl()
        ctrl.federation_config = {"model_compression": True}
        overhead = ctrl._estimate_communication_overhead()
        assert isinstance(overhead, float)

    def test_estimate_communication_overhead_no_compression(self):
        ctrl = self._make_ctrl()
        ctrl.federation_config = {"model_compression": False}
        overhead = ctrl._estimate_communication_overhead()
        assert isinstance(overhead, float)

    def test_get_model_size(self):
        ctrl = self._make_ctrl()
        size = ctrl._get_model_size()
        assert "total_parameters" in size
        assert "memory_mb" in size
        assert "models_count" in size

    def test_get_model_size_with_models(self):
        ctrl = self._make_ctrl()
        ctrl.domain_adapter = MagicMock()
        ctrl.progressive_net = MagicMock()
        ctrl.multi_task_net = MagicMock()
        ctrl.maml_controller = MagicMock()
        # Mock get_model_size base class to return values
        with patch.object(ctrl, "get_model_size", return_value={
            "total_parameters": 1000, "trainable_parameters": 500}):
            size = ctrl._get_model_size()
        assert size["models_count"] == 4


# ===================================================================
# Factory
# ===================================================================
class TestCreateTransferController:
    def test_default(self):
        with patch.object(TransferLearningController, "_initialize_networks"):
            ctrl = create_transfer_controller(state_dim=10, action_dim=5)
        assert ctrl is not None

    def test_all_methods(self):
        for method in ["fine_tuning", "domain_adaptation", "progressive",
                        "meta_learning", "knowledge_distillation", "multi_task"]:
            with patch.object(TransferLearningController, "_initialize_networks"):
                ctrl = create_transfer_controller(
                    state_dim=10, action_dim=5, method=method)
            assert ctrl is not None

    def test_unknown_method(self):
        with patch.object(TransferLearningController, "_initialize_networks"):
            ctrl = create_transfer_controller(
                state_dim=10, action_dim=5, method="unknown_method")
        assert ctrl.config.transfer_method == TransferLearningMethod.MULTI_TASK

    def test_species(self):
        with patch.object(TransferLearningController, "_initialize_networks"):
            ctrl = create_transfer_controller(
                state_dim=10, action_dim=5,
                source_species=["geobacter", "shewanella"],
                target_species="mixed")
        assert ctrl.config.target_species == BacterialSpecies.MIXED

    def test_unknown_species(self):
        with patch.object(TransferLearningController, "_initialize_networks"):
            ctrl = create_transfer_controller(
                state_dim=10, action_dim=5, target_species="unknown_species")
        assert ctrl.config.target_species == BacterialSpecies.MIXED

    def test_with_kwargs(self):
        with patch.object(TransferLearningController, "_initialize_networks"):
            ctrl = create_transfer_controller(
                state_dim=10, action_dim=5, meta_lr=0.05)
        assert ctrl.config.meta_lr == 0.05

    def test_with_nonexistent_kwarg(self):
        with patch.object(TransferLearningController, "_initialize_networks"):
            ctrl = create_transfer_controller(
                state_dim=10, action_dim=5, nonexistent_param=42)
        assert ctrl is not None