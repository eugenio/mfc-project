"""Tests for federated_learning_controller - coverage part 1.

Covers enums, configs, DifferentialPrivacy, SecureAggregation,
ModelCompression, module-level functions, and create_federated_system.
"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import types
import random
import math
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


class _FakeDropout(MockModule):
    def __init__(self, p=0.0, *a, **kw): super().__init__()
    def __call__(self, x): return x


class _FakeLayerNorm(MockModule):
    def __init__(self, *a, **kw): super().__init__()
    def __call__(self, x): return x


class _FakeConv1d(MockModule):
    def __init__(self, *a, **kw): super().__init__()
    def __call__(self, x): return MagicMock()


class _FakeModuleDict(dict, MockModule):
    def __init__(self, *a, **kw):
        dict.__init__(self)
        MockModule.__init__(self)


class _FakeModuleList(list, MockModule):
    def __init__(self, items=None, *a, **kw):
        list.__init__(self, items or [])
        MockModule.__init__(self)


class _FakeSequential(MockModule):
    def __init__(self, *layers, **kw): super().__init__()
    def __call__(self, x): return MagicMock()


mock_nn.Module = MockModule
mock_nn.Linear = _FakeLinear
mock_nn.Dropout = _FakeDropout
mock_nn.LayerNorm = _FakeLayerNorm
mock_nn.Conv1d = _FakeConv1d
mock_nn.ModuleDict = _FakeModuleDict
mock_nn.ModuleList = _FakeModuleList
mock_nn.Sequential = _FakeSequential
mock_nn.ReLU = MagicMock(return_value=MagicMock())
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

from sensing_models.sensor_fusion import BacterialSpecies
from federated_learning_controller import (
    FederatedAlgorithm, AggregationMethod, ClientSelectionStrategy,
    FederatedConfig, ClientInfo, DifferentialPrivacy, SecureAggregation,
    ModelCompression, _select_device, _get_model_size,
    create_federated_system,
)


class TestEnums:
    def test_federated_algorithms(self):
        assert FederatedAlgorithm.FEDAVG.value == "federated_averaging"
        assert FederatedAlgorithm.FEDPROX.value == "federated_proximal"
        assert FederatedAlgorithm.SCAFFOLD.value == "scaffold"
        assert FederatedAlgorithm.FEDNOVA.value == "fed_nova"
        assert FederatedAlgorithm.FEDPER.value == "federated_personalization"
        assert FederatedAlgorithm.FEDBN.value == "federated_batch_normalization"
        assert FederatedAlgorithm.MOON.value == "model_contrastive_learning"
        assert FederatedAlgorithm.FEDOPT.value == "federated_optimization"

    def test_aggregation_methods(self):
        assert AggregationMethod.WEIGHTED_AVERAGE.value == "weighted_average"
        assert AggregationMethod.MEDIAN_AGGREGATION.value == "median"
        assert AggregationMethod.TRIMMED_MEAN.value == "trimmed_mean"
        assert AggregationMethod.BYZANTINE_ROBUST.value == "byzantine_robust"
        assert AggregationMethod.SECURE_AGGREGATION.value == "secure_aggregation"
        assert AggregationMethod.DIFFERENTIAL_PRIVATE.value == "differential_private"

    def test_client_selection(self):
        assert ClientSelectionStrategy.RANDOM.value == "random"
        assert ClientSelectionStrategy.CYCLIC.value == "cyclic"
        assert ClientSelectionStrategy.PERFORMANCE_BASED.value == "performance_based"
        assert ClientSelectionStrategy.RESOURCE_AWARE.value == "resource_aware"
        assert ClientSelectionStrategy.DIVERSITY_BASED.value == "diversity_based"
        assert ClientSelectionStrategy.GRADIENT_BASED.value == "gradient_based"


class TestFederatedConfig:
    def test_defaults(self):
        c = FederatedConfig()
        assert c.num_clients == 10 and c.clients_per_round == 5
        assert c.num_rounds == 100 and c.local_epochs == 5
        assert c.differential_privacy is True and c.secure_aggregation is True
        assert c.global_lr == 1.0 and c.local_lr == 0.01

    def test_custom(self):
        c = FederatedConfig(num_clients=5, local_lr=0.1,
            algorithm=FederatedAlgorithm.FEDPROX,
            aggregation=AggregationMethod.MEDIAN_AGGREGATION,
            noise_multiplier=2.0, compression_ratio=0.5,
            personalization_layers=["l1"], async_mode=True,
            byzantine_clients=2, robust_aggregation=True)
        assert c.num_clients == 5 and c.algorithm == FederatedAlgorithm.FEDPROX
        assert c.noise_multiplier == 2.0 and c.async_mode is True


class TestClientInfo:
    def test_creation(self):
        info = ClientInfo(client_id="c1", site_name="Site A", location="USA",
            mfc_type="dual", bacterial_species=BacterialSpecies.GEOBACTER)
        assert info.client_id == "c1" and info.is_active is True
        assert info.data_samples == 0 and info.local_accuracy == 0.0

    def test_full(self):
        info = ClientInfo(client_id="c2", site_name="Lab B", location="JP",
            mfc_type="single", bacterial_species=BacterialSpecies.SHEWANELLA,
            data_samples=100, computation_power=2.0, is_active=False,
            local_loss=0.15, improvement_rate=0.02)
        assert info.data_samples == 100 and not info.is_active


class TestDifferentialPrivacy:
    def test_init(self):
        dp = DifferentialPrivacy(noise_multiplier=1.0, max_grad_norm=1.0)
        assert dp.noise_multiplier == 1.0

    def test_clip_gradients(self):
        DifferentialPrivacy().clip_gradients(MagicMock())

    def test_add_noise_with_grad(self):
        dp = DifferentialPrivacy()
        model = MagicMock()
        p = MagicMock(); p.grad = MagicMock()
        model.parameters.return_value = [p]
        dp.add_noise(model, "cpu")

    def test_add_noise_no_grad(self):
        dp = DifferentialPrivacy()
        model = MagicMock()
        p = MagicMock(); p.grad = None
        model.parameters.return_value = [p]
        dp.add_noise(model, "cpu")

    def test_compute_privacy_budget(self):
        dp = DifferentialPrivacy(noise_multiplier=1.0)
        assert dp.compute_privacy_budget(steps=100, sampling_rate=0.1) >= 0

    def test_compute_privacy_budget_custom(self):
        dp = DifferentialPrivacy(noise_multiplier=2.0)
        assert dp.compute_privacy_budget(50, 0.2, target_delta=1e-3) >= 0


class TestSecureAggregation:
    def test_init_variants(self):
        assert SecureAggregation(num_clients=10).threshold == 5
        assert SecureAggregation(num_clients=2).threshold == 2
        assert SecureAggregation(num_clients=3).threshold == 2

    def test_generate_masks(self):
        masks = SecureAggregation(3).generate_masks(
            {"w1": MagicMock(), "w2": MagicMock()})
        assert "w1" in masks and "w2" in masks

    def test_mask_model(self):
        sa = SecureAggregation(3)
        assert "w1" in sa.mask_model({"w1": MagicMock()}, {"w1": MagicMock()})

    def test_mask_model_missing(self):
        masked = SecureAggregation(3).mask_model(
            {"w1": MagicMock(), "w2": MagicMock()}, {"w1": MagicMock()})
        assert "w2" in masked

    def test_unmask_insufficient(self):
        with pytest.raises(ValueError, match="Insufficient"):
            SecureAggregation(4).unmask_aggregate([{"w": MagicMock()}], [{}])

    def test_unmask_aggregate(self):
        sa = SecureAggregation(4)
        p = MagicMock()
        p.__add__ = MagicMock(return_value=p)
        p.__sub__ = MagicMock(return_value=p)
        p.__truediv__ = MagicMock(return_value=p)
        mock_torch.zeros_like.return_value = p
        sa.unmask_aggregate([{"w": p}, {"w": p}], [{"w": p}, {"w": p}])


class TestModelCompression:
    def test_init(self):
        mc = ModelCompression()
        assert mc.compression_ratio == 0.1 and mc.quantization_bits == 8

    def test_quantize_tensor(self):
        mc = ModelCompression()
        t = MagicMock()
        t.min.return_value = MagicMock(item=MagicMock(return_value=0.0))
        t.max.return_value = MagicMock(item=MagicMock(return_value=1.0))
        t.__sub__ = MagicMock(return_value=MagicMock(
            __mul__=MagicMock(return_value=MagicMock(
                round=MagicMock(return_value=MagicMock(
                    clamp=MagicMock(return_value=MagicMock(
                        detach=MagicMock(return_value=MagicMock(
                            cpu=MagicMock(return_value=MagicMock(
                                numpy=MagicMock(return_value=np.array(
                                    [0, 1], dtype=np.float32))))))))))))))
        mc._quantize_tensor(t)

    def test_quantize_equal(self):
        mc = ModelCompression()
        t = MagicMock()
        t.min.return_value = MagicMock(item=MagicMock(return_value=0.5))
        t.max.return_value = MagicMock(item=MagicMock(return_value=0.5))
        t.shape = (3,)
        assert isinstance(mc._quantize_tensor(t), np.ndarray)

    def test_dequantize(self):
        ModelCompression()._dequantize_tensor(
            np.array([0, 128, 255], dtype=np.uint8))

    def test_compress_model(self):
        mc = ModelCompression()
        p = MagicMock()
        p.flatten.return_value = MagicMock(__len__=MagicMock(return_value=100))
        p.numel.return_value = 100
        p.shape = (10, 10)
        assert "w" in mc.compress_model({"w": p})

    def test_decompress_model(self):
        ModelCompression().decompress_model({"w": {
            "shape": (10, 10), "indices": np.array([0, 1]),
            "values": np.array([128, 255], dtype=np.uint8),
            "original_size": 100}})


class TestModuleFunctions:
    def test_select_device(self):
        _select_device("auto")
        _select_device(None)
        _select_device("cpu")

    def test_get_model_size(self):
        m = MagicMock()
        p1 = MagicMock(); p1.numel.return_value = 100; p1.requires_grad = True
        p2 = MagicMock(); p2.numel.return_value = 50; p2.requires_grad = False
        m.parameters.return_value = [p1, p2]
        r = _get_model_size(m)
        assert r["total_parameters"] == 150 and r["trainable_parameters"] == 100


class TestCreateFederatedSystem:
    def test_default(self):
        s = create_federated_system(MagicMock(), num_clients=3)
        assert s.config.num_clients == 3

    def test_algorithms(self):
        for a in ["fedavg", "fedprox", "fedper", "scaffold"]:
            create_federated_system(MagicMock(), num_clients=2, algorithm=a)

    def test_unknown(self):
        s = create_federated_system(MagicMock(), num_clients=2, algorithm="x")
        assert s.config.algorithm == FederatedAlgorithm.FEDAVG

    def test_kwargs(self):
        s = create_federated_system(MagicMock(), num_clients=5,
            local_epochs=10, local_lr=0.05)
        assert s.config.local_epochs == 10

    def test_invalid_kwargs(self):
        s = create_federated_system(MagicMock(), num_clients=2, bad_param=42)
        assert not hasattr(s.config, "bad_param")
