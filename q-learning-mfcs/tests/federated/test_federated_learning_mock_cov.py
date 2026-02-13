"""Comprehensive tests for federated_learning_controller.py with torch mocked.

Targets 99%+ statement coverage of all classes and functions.
"""
import copy
import os
import pickle
import sys
import tempfile
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock torch BEFORE importing source modules
# ---------------------------------------------------------------------------
mock_torch = MagicMock()
mock_nn = MagicMock()
mock_optim = MagicMock()
mock_F = MagicMock()


class _FakeTensor:
    pass


mock_torch.Tensor = _FakeTensor


class MockModule:
    """Minimal nn.Module replacement for coverage testing."""

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
mock_nn.BatchNorm1d = MagicMock(return_value=MagicMock())
mock_nn.Conv1d = MagicMock(return_value=MagicMock())
mock_torch.nn = mock_nn
mock_torch.optim = mock_optim
mock_torch.optim.SGD = MagicMock(return_value=MagicMock(
    zero_grad=MagicMock(), step=MagicMock(),
    state_dict=MagicMock(return_value={}),
    load_state_dict=MagicMock()))
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
mock_torch.randn = MagicMock(return_value=MagicMock(
    sign=MagicMock(return_value=MagicMock(
        mul_=MagicMock(return_value=MagicMock())))))
mock_torch.normal = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.topk = MagicMock(return_value=(MagicMock(), MagicMock()))
mock_torch.abs = MagicMock(return_value=MagicMock())
mock_torch.stack = MagicMock(return_value=MagicMock(
    mean=MagicMock(return_value=MagicMock()),
    sort=MagicMock(return_value=(MagicMock(
        __getitem__=MagicMock(return_value=MagicMock())), MagicMock()))))
mock_torch.median = MagicMock(return_value=(MagicMock(), MagicMock()))
mock_torch.sort = MagicMock(return_value=(MagicMock(), MagicMock()))
mock_torch.mean = MagicMock(return_value=MagicMock())
mock_torch.matmul = MagicMock(return_value=MagicMock())
mock_torch.tril = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.argmax = MagicMock(return_value=MagicMock(
    item=MagicMock(return_value=0)))
mock_torch.max = MagicMock(return_value=MagicMock(
    item=MagicMock(return_value=0.9)))
mock_torch.sin = MagicMock(return_value=MagicMock())
mock_torch.cos = MagicMock(return_value=MagicMock())
mock_torch.log = MagicMock(return_value=MagicMock())
mock_F.softmax = MagicMock(return_value=MagicMock())
mock_F.relu = MagicMock(return_value=MagicMock())
mock_F.mse_loss = MagicMock(return_value=MagicMock(
    item=MagicMock(return_value=0.5),
    backward=MagicMock()))
mock_F.one_hot = MagicMock(return_value=MagicMock(
    float=MagicMock(return_value=MagicMock())))

# -- Inject mocks into sys.modules BEFORE importing source modules ----------
_orig = {}
import torch_compat

from federated_learning_controller import (
from sensing_models.sensor_fusion import BacterialSpecies

# -- Restore original sys.modules ------------------------------------------
class _FakeTensor:
    pass

class MockModule:
    """Minimal nn.Module replacement for coverage testing."""

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

class TestFederatedAlgorithm:
    def test_all_values(self):
        assert FederatedAlgorithm.FEDAVG.value == "federated_averaging"
        assert FederatedAlgorithm.FEDPROX.value == "federated_proximal"
        assert FederatedAlgorithm.FEDPER.value == "federated_personalization"
        assert FederatedAlgorithm.FEDBN.value == "federated_batch_normalization"
        assert FederatedAlgorithm.SCAFFOLD.value == "scaffold"
        assert FederatedAlgorithm.FEDNOVA.value == "fed_nova"
        assert FederatedAlgorithm.MOON.value == "model_contrastive_learning"
        assert FederatedAlgorithm.FEDOPT.value == "federated_optimization"


class TestAggregationMethod:
    def test_all_values(self):
        assert AggregationMethod.WEIGHTED_AVERAGE.value == "weighted_average"
        assert AggregationMethod.MEDIAN_AGGREGATION.value == "median"
        assert AggregationMethod.TRIMMED_MEAN.value == "trimmed_mean"
        assert AggregationMethod.BYZANTINE_ROBUST.value == "byzantine_robust"
        assert AggregationMethod.SECURE_AGGREGATION.value == "secure_aggregation"
        assert AggregationMethod.DIFFERENTIAL_PRIVATE.value == "differential_private"

class TestClientSelectionStrategy:
    def test_all_values(self):
        assert ClientSelectionStrategy.RANDOM.value == "random"
        assert ClientSelectionStrategy.CYCLIC.value == "cyclic"
        assert ClientSelectionStrategy.PERFORMANCE_BASED.value == "performance_based"
        assert ClientSelectionStrategy.RESOURCE_AWARE.value == "resource_aware"
        assert ClientSelectionStrategy.DIVERSITY_BASED.value == "diversity_based"
        assert ClientSelectionStrategy.GRADIENT_BASED.value == "gradient_based"


# ===========================================================================
# Tests - FederatedConfig
# ===========================================================================
class TestFederatedConfig:
    def test_defaults(self):
        c = FederatedConfig()
        assert c.num_clients == 10
        assert c.clients_per_round == 5
        assert c.num_rounds == 100
        assert c.local_epochs == 5
        assert c.algorithm == FederatedAlgorithm.FEDAVG
        assert c.aggregation == AggregationMethod.WEIGHTED_AVERAGE
        assert c.client_selection == ClientSelectionStrategy.RANDOM
        assert c.differential_privacy is True
        assert c.noise_multiplier == 1.0
        assert c.max_grad_norm == 1.0
        assert c.secure_aggregation is True
        assert c.compression_ratio == 0.1
        assert c.quantization_bits == 8
        assert c.sparsification_ratio == 0.01
        assert c.personalization_layers == ["output_layer"]
        assert c.local_adaptation_steps == 10
        assert c.byzantine_clients == 0
        assert c.robust_aggregation is False
        assert c.async_mode is False
        assert c.staleness_threshold == 3
        assert c.global_lr == 1.0
        assert c.local_lr == 0.01
        assert c.momentum == 0.9
        assert c.weight_decay == 1e-4

    def test_custom(self):
        c = FederatedConfig(num_clients=3, algorithm=FederatedAlgorithm.FEDPROX)
        assert c.num_clients == 3
        assert c.algorithm == FederatedAlgorithm.FEDPROX


# ===========================================================================
# Tests - ClientInfo
# ===========================================================================
class TestClientInfo:
    def test_defaults(self):
        ci = _make_client_info()
        assert ci.client_id == "c1"
        assert ci.site_name == "Site_c1"
        assert ci.location == "TestLab"
        assert ci.mfc_type == "dual_chamber"
        assert ci.bacterial_species == BacterialSpecies.GEOBACTER
        assert ci.data_samples == 0
        assert ci.computation_power == 1.0
        assert ci.communication_bandwidth == 1.0
        assert ci.reliability_score == 1.0
        assert ci.is_active is True
        assert isinstance(ci.last_update, datetime)
        assert ci.round_participation == []
        assert ci.local_accuracy == 0.0
        assert ci.local_loss == float("inf")
        assert ci.improvement_rate == 0.0

    def test_custom(self):
        ci = ClientInfo(
            client_id="x",
            site_name="Y",
            location="Z",
            mfc_type="single",
            bacterial_species=BacterialSpecies.SHEWANELLA,
            data_samples=100,
            computation_power=2.5,
        )
        assert ci.data_samples == 100
        assert ci.computation_power == 2.5


# ===========================================================================
# Tests - DifferentialPrivacy
# ===========================================================================
class TestDifferentialPrivacy:
    def test_init(self):
        dp = DifferentialPrivacy(noise_multiplier=0.5, max_grad_norm=2.0)
        assert dp.noise_multiplier == 0.5
        assert dp.max_grad_norm == 2.0

    def test_clip_gradients(self):
        dp = DifferentialPrivacy()
        model = _make_model()
        result = dp.clip_gradients(model)
        # clip_grad_norm_ is mocked; returns MagicMock.item() -> 1.0
        assert result is not None

    def test_add_noise(self):
        dp = DifferentialPrivacy()
        model = _make_model()
        # Create params with grad
        param = MagicMock()
        param.grad = MagicMock()
        param.shape = (10,)
        model.parameters.return_value = [param]
        dp.add_noise(model, "cpu")

    def test_add_noise_no_grad(self):
        dp = DifferentialPrivacy()
        model = _make_model()
        param = MagicMock()
        param.grad = None
        param.shape = (10,)
        model.parameters.return_value = [param]
        dp.add_noise(model, "cpu")

    def test_compute_privacy_budget(self):
        dp = DifferentialPrivacy(noise_multiplier=1.0)
        epsilon = dp.compute_privacy_budget(steps=100, sampling_rate=0.1, target_delta=1e-5)
        assert isinstance(epsilon, float)
        assert epsilon >= 0


# ===========================================================================
# Tests - SecureAggregation
# ===========================================================================
class TestSecureAggregation:
    def test_init(self):
        sa = SecureAggregation(num_clients=10)
        assert sa.num_clients == 10
        assert sa.threshold == 5

    def test_init_small(self):
        sa = SecureAggregation(num_clients=3)
        assert sa.threshold == 2

    def test_generate_masks(self):
        sa = SecureAggregation(num_clients=5)
        params = {"w1": MagicMock(), "w2": MagicMock()}
        masks = sa.generate_masks(params)
        assert "w1" in masks
        assert "w2" in masks

    def test_mask_model(self):
        sa = SecureAggregation(num_clients=5)
        params = {"w1": MagicMock(), "w2": MagicMock()}
        masks = {"w1": MagicMock()}
        masked = sa.mask_model(params, masks)
        assert "w1" in masked
        assert "w2" in masked

    def test_unmask_aggregate_insufficient(self):
        sa = SecureAggregation(num_clients=10)
        with pytest.raises(ValueError, match="Insufficient"):
            sa.unmask_aggregate([{"w": MagicMock()}], [{"w": MagicMock()}])

    def test_unmask_aggregate_success(self):
        sa = SecureAggregation(num_clients=4)
        sa.threshold = 2
        p1 = {"w": MagicMock()}
        p2 = {"w": MagicMock()}
        m1 = {"w": MagicMock()}
        m2 = {"w": MagicMock()}
        result = sa.unmask_aggregate([p1, p2], [m1, m2])
        assert "w" in result

    def test_unmask_aggregate_missing_mask_key(self):
        sa = SecureAggregation(num_clients=4)
        sa.threshold = 2
        p1 = {"w": MagicMock()}
        p2 = {"w": MagicMock()}
        m1 = {}  # No mask for "w"
        m2 = {}
        result = sa.unmask_aggregate([p1, p2], [m1, m2])
        assert "w" in result


# ===========================================================================
# Tests - ModelCompression
# ===========================================================================
class TestModelCompression:
    def test_init(self):
        mc = ModelCompression(compression_ratio=0.2, quantization_bits=4, sparsification_ratio=0.05)
        assert mc.compression_ratio == 0.2
        assert mc.quantization_bits == 4
        assert mc.sparsification_ratio == 0.05

    def test_compress_model(self):
        mc = ModelCompression()
        # Create realistic mock params
        param = MagicMock()
        flat = MagicMock()
        flat.__len__ = MagicMock(return_value=100)
        param.flatten.return_value = flat
        param.numel.return_value = 100

        top_vals = MagicMock()
        top_idx = MagicMock()
        top_idx.cpu.return_value = MagicMock(
            numpy=MagicMock(return_value=np.array([0, 1])))
        mock_torch.topk.return_value = (top_vals, top_idx)
        mock_torch.abs.return_value = MagicMock()
        flat.__getitem__ = MagicMock(return_value=MagicMock(
            min=MagicMock(return_value=MagicMock(item=MagicMock(return_value=-1.0))),
            max=MagicMock(return_value=MagicMock(item=MagicMock(return_value=1.0))),
        ))
        param.shape = (10, 10)

        result = mc.compress_model({"w": param})
        assert "w" in result

    def test_quantize_tensor_equal_min_max(self):
        mc = ModelCompression()
        t = MagicMock()
        t.min.return_value = MagicMock(item=MagicMock(return_value=0.5))
        t.max.return_value = MagicMock(item=MagicMock(return_value=0.5))
        t.shape = (3,)
        result = mc._quantize_tensor(t)
        assert result.shape == (3,)

    def test_quantize_tensor_normal(self):
        mc = ModelCompression()
        t = MagicMock()
        t.min.return_value = MagicMock(item=MagicMock(return_value=0.0))
        t.max.return_value = MagicMock(item=MagicMock(return_value=1.0))
        # mock arithmetic chain
        sub_result = MagicMock()
        t.__sub__ = MagicMock(return_value=sub_result)
        mul_result = MagicMock()
        sub_result.__mul__ = MagicMock(return_value=mul_result)
        round_result = MagicMock()
        mul_result.round.return_value = round_result
        clamp_result = MagicMock()
        round_result.clamp.return_value = clamp_result
        det_result = MagicMock()
        clamp_result.detach.return_value = det_result
        cpu_result = MagicMock()
        det_result.cpu.return_value = cpu_result
        cpu_result.numpy.return_value = MagicMock(
            astype=MagicMock(return_value=np.array([0, 128, 255], dtype=np.uint8)))
        result = mc._quantize_tensor(t)
        assert result is not None

    def test_decompress_model(self):
        mc = ModelCompression()
        compressed = {
            "w": {
                "shape": (4, 2),
                "indices": np.array([0, 3]),
                "values": np.array([128, 200], dtype=np.uint8),
                "original_size": 8,
            }
        }
        result = mc.decompress_model(compressed)
        assert "w" in result

    def test_dequantize_tensor(self):
        mc = ModelCompression(quantization_bits=8)
        arr = np.array([0, 128, 255], dtype=np.uint8)
        result = mc._dequantize_tensor(arr)
        assert result is not None


# ===========================================================================
# Tests - Module-level functions
# ===========================================================================
class TestModuleLevelFunctions:
    def test_select_device_auto(self):
        result = _select_device(None)
        assert result is not None

    def test_select_device_auto_explicit(self):
        result = _select_device("auto")
        assert result is not None

    def test_select_device_cpu(self):
        result = _select_device("cpu")
        assert result is not None

    def test_get_model_size(self):
        model = _make_model()
        result = _get_model_size(model)
        assert "total_parameters" in result
        assert "trainable_parameters" in result
        assert "memory_mb" in result


# ===========================================================================
# Tests - FederatedClient
# ===========================================================================
class TestFederatedClient:
    def _make_client(self, dp=True, compress=0.1):
        config = FederatedConfig(
            differential_privacy=dp,
            compression_ratio=compress,
            local_epochs=2,
        )
        model = _make_model()
        info = _make_client_info()
        with patch("federated_learning_controller.copy.deepcopy", return_value=model):
            client = FederatedClient(info, model, config)
        return client

    def test_init_with_dp(self):
        client = self._make_client(dp=True)
        assert client.dp_mechanism is not None
        assert client.client_info.client_id == "c1"

    def test_init_without_dp(self):
        client = self._make_client(dp=False)
        assert client.dp_mechanism is None

    def test_feature_engineer_lazy(self):
        client = self._make_client()
        with patch("federated_learning_controller.FeatureEngineer") as mock_fe:
            mock_fe.return_value = MagicMock()
            fe = client.feature_engineer
            assert fe is not None
            # Second call should use cached
            fe2 = client.feature_engineer
            assert fe2 is fe

    def test_get_model_size(self):
        client = self._make_client()
        size = client.get_model_size()
        assert "total_parameters" in size

    def test_add_local_data(self):
        client = self._make_client()
        client._feature_engineer = MagicMock()
        client._feature_engineer.extract_features.return_value = {"f1": 1.0, "f2": 2.0}

        state = MagicMock()
        state.power_output = 5.0
        state.current_density = 1.0
        state.health_metrics.overall_health_score = 0.8
        state.fused_measurement.fusion_confidence = 0.9
        state.anomalies = []

        client.add_local_data([state], [0], [1.0])
        assert client.client_info.data_samples == 1
        assert len(client.local_data) == 1

    def test_local_train_no_data(self):
        client = self._make_client()
        result = client.local_train({})
        assert "error" in result

    def test_local_train_with_data(self):
        client = self._make_client()
        # Populate local_data
        client.local_data = [
            {"features": MagicMock(), "action": 0, "reward": 1.0, "state": MagicMock()},
            {"features": MagicMock(), "action": 1, "reward": 0.5, "state": MagicMock()},
        ]
        # Mock the model forward pass
        mock_logits = MagicMock()
        mock_logits.size.return_value = 5
        client.local_model.return_value = mock_logits

        result = client.local_train({"w": MagicMock()})
        assert "client_id" in result
        assert "loss" in result
        assert result["compression_used"] is True

    def test_local_train_with_exception(self):
        client = self._make_client()
        client.local_data = [
            {"features": MagicMock(), "action": 0, "reward": 1.0, "state": MagicMock()},
        ]
        # Make model raise an exception during forward pass
        client.local_model.side_effect = RuntimeError("boom")
        result = client.local_train({"w": MagicMock()})
        # Should handle exception and return results
        assert "client_id" in result

    def test_local_train_no_compression(self):
        client = self._make_client(compress=1.0)
        client.local_data = [
            {"features": MagicMock(), "action": 0, "reward": 1.0, "state": MagicMock()},
        ]
        mock_logits = MagicMock()
        mock_logits.size.return_value = 5
        client.local_model.return_value = mock_logits
        result = client.local_train({"w": MagicMock()})
        assert result["compression_used"] is False

    def test_evaluate_model_empty(self):
        client = self._make_client()
        result = client.evaluate_model([])
        assert result["accuracy"] == 0.0
        assert result["loss"] == float("inf")

    def test_evaluate_model_with_data(self):
        client = self._make_client()
        test_data = [
            {"features": MagicMock(), "action": 0},
            {"features": MagicMock(), "action": 1},
        ]
        mock_logits = MagicMock()
        mock_logits.size.return_value = 5
        client.local_model.return_value = mock_logits
        mock_torch.argmax.return_value = MagicMock(item=MagicMock(return_value=0))
        result = client.evaluate_model(test_data)
        assert "accuracy" in result
        assert "loss" in result
        assert "samples_evaluated" in result

    def test_evaluate_model_exception(self):
        client = self._make_client()
        test_data = [{"features": MagicMock(), "action": 0}]
        client.local_model.side_effect = RuntimeError("eval boom")
        result = client.evaluate_model(test_data)
        assert result["accuracy"] == 0.0


# ===========================================================================
# Tests - FederatedServer
# ===========================================================================
class TestFederatedServer:
    def _make_server(self, agg=AggregationMethod.WEIGHTED_AVERAGE,
                     selection=ClientSelectionStrategy.RANDOM,
                     secure=True, num_clients=5, clients_per_round=3):
        config = FederatedConfig(
            aggregation=agg,
            client_selection=selection,
            secure_aggregation=secure,
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            num_rounds=3,
            local_epochs=1,
        )
        model = _make_model()
        with patch("federated_learning_controller.copy.deepcopy", return_value=model):
            server = FederatedServer(model, config)
        return server

    def test_init_with_secure(self):
        server = self._make_server(secure=True)
        assert server.secure_aggregator is not None
        assert server.current_round == 0

    def test_init_without_secure(self):
        server = self._make_server(secure=False)
        assert server.secure_aggregator is None

    def test_register_client(self):
        server = self._make_server()
        info = _make_client_info("c1")
        with patch("federated_learning_controller.copy.deepcopy", return_value=_make_model()):
            success = server.register_client(info)
        assert success is True
        assert "c1" in server.clients

    def test_register_client_failure(self):
        server = self._make_server()
        info = _make_client_info("c_bad")
        with patch("federated_learning_controller.copy.deepcopy", side_effect=RuntimeError("fail")):
            success = server.register_client(info)
        assert success is False

    def test_select_clients_none_available(self):
        server = self._make_server()
        result = server.select_clients(1)
        assert result == []

    def _register_clients_with_data(self, server, n=3):
        """Register n clients with local data."""
        for i in range(n):
            cid = f"c{i}"
            info = _make_client_info(cid)
            info.is_active = True
            info.reliability_score = 1.0 - i * 0.1
            info.local_loss = i * 0.5
            info.computation_power = 1.0 + i
            info.communication_bandwidth = 2.0 + i
            mock_client = MagicMock()
            mock_client.local_data = [1, 2, 3]  # non-empty
            mock_client.client_info = info
            mock_client.compressor = MagicMock()
            mock_client.compressor.decompress_model.return_value = {"w": MagicMock()}
            server.clients[cid] = mock_client
            server.client_info[cid] = info

    def test_select_clients_random(self):
        server = self._make_server(selection=ClientSelectionStrategy.RANDOM)
        self._register_clients_with_data(server, 3)
        selected = server.select_clients(0)
        assert len(selected) == 3

    def test_select_clients_cyclic(self):
        server = self._make_server(selection=ClientSelectionStrategy.CYCLIC)
        self._register_clients_with_data(server, 3)
        selected = server.select_clients(0)
        assert len(selected) == 3

    def test_select_clients_performance_based(self):
        server = self._make_server(selection=ClientSelectionStrategy.PERFORMANCE_BASED)
        self._register_clients_with_data(server, 3)
        selected = server.select_clients(0)
        assert len(selected) == 3

    def test_select_clients_resource_aware(self):
        server = self._make_server(selection=ClientSelectionStrategy.RESOURCE_AWARE)
        self._register_clients_with_data(server, 3)
        selected = server.select_clients(0)
        assert len(selected) == 3

    def test_select_clients_default_fallback(self):
        server = self._make_server(selection=ClientSelectionStrategy.DIVERSITY_BASED)
        self._register_clients_with_data(server, 3)
        selected = server.select_clients(0)
        assert len(selected) == 3

    def test_aggregate_models_empty(self):
        server = self._make_server()
        result = server.aggregate_models([])
        assert isinstance(result, dict)

    def test_aggregate_weighted_average(self):
        server = self._make_server(agg=AggregationMethod.WEIGHTED_AVERAGE)
        self._register_clients_with_data(server)
        updates = [
            {"client_id": "c0", "model_params": {"w": MagicMock()}, "num_samples": 10,
             "compression_used": False},
            {"client_id": "c1", "model_params": {"w": MagicMock()}, "num_samples": 20,
             "compression_used": False},
        ]
        result = server.aggregate_models(updates)
        assert "w" in result

    def test_aggregate_weighted_average_zero_samples(self):
        server = self._make_server(agg=AggregationMethod.WEIGHTED_AVERAGE)
        updates = [
            {"client_id": "c0", "model_params": {"w": MagicMock()}, "num_samples": 0},
        ]
        result = server.aggregate_models(updates)
        assert isinstance(result, dict)

    def test_aggregate_weighted_average_compressed(self):
        server = self._make_server(agg=AggregationMethod.WEIGHTED_AVERAGE)
        self._register_clients_with_data(server)
        updates = [
            {"client_id": "c0", "model_params": {"w": MagicMock()}, "num_samples": 10,
             "compression_used": True},
        ]
        result = server.aggregate_models(updates)
        assert isinstance(result, dict)

    def test_aggregate_median(self):
        server = self._make_server(agg=AggregationMethod.MEDIAN_AGGREGATION)
        self._register_clients_with_data(server)
        updates = [
            {"client_id": "c0", "model_params": {"w": MagicMock()}, "num_samples": 10,
             "compression_used": False},
            {"client_id": "c1", "model_params": {"w": MagicMock()}, "num_samples": 10,
             "compression_used": False},
        ]
        result = server.aggregate_models(updates)
        assert isinstance(result, dict)

    def test_aggregate_trimmed_mean(self):
        server = self._make_server(agg=AggregationMethod.TRIMMED_MEAN)
        self._register_clients_with_data(server)
        updates = [
            {"client_id": "c0", "model_params": {"w": MagicMock()}, "num_samples": 10,
             "compression_used": False},
            {"client_id": "c1", "model_params": {"w": MagicMock()}, "num_samples": 10,
             "compression_used": False},
        ]
        # mock sort return
        sorted_mock = MagicMock()
        sorted_mock.__getitem__ = MagicMock(return_value=MagicMock())
        mock_torch.sort.return_value = (sorted_mock, MagicMock())
        result = server.aggregate_models(updates)
        assert isinstance(result, dict)

    def test_aggregate_byzantine(self):
        server = self._make_server(agg=AggregationMethod.BYZANTINE_ROBUST)
        self._register_clients_with_data(server)
        updates = [
            {"client_id": "c0", "model_params": {"w": MagicMock()}, "num_samples": 10,
             "compression_used": False},
            {"client_id": "c1", "model_params": {"w": MagicMock()}, "num_samples": 10,
             "compression_used": False},
        ]
        result = server.aggregate_models(updates)
        assert isinstance(result, dict)

    def test_aggregate_default_fallback(self):
        """Test aggregation with a method that falls to default."""
        server = self._make_server(agg=AggregationMethod.SECURE_AGGREGATION)
        self._register_clients_with_data(server)
        updates = [
            {"client_id": "c0", "model_params": {"w": MagicMock()}, "num_samples": 10,
             "compression_used": False},
        ]
        result = server.aggregate_models(updates)
        assert isinstance(result, dict)

    def test_run_round_no_clients(self):
        server = self._make_server()
        result = server.run_round(1)
        assert "error" in result

    def test_run_round_success(self):
        server = self._make_server()
        self._register_clients_with_data(server)
        # Mock local_train on each client
        for cid, client in server.clients.items():
            client.local_train.return_value = {
                "client_id": cid,
                "model_params": {"w": MagicMock()},
                "num_samples": 10,
                "loss": 0.5,
                "epochs": 1,
                "batches": 2,
                "compression_used": False,
            }
        result = server.run_round(1)
        assert "round" in result
        assert result["round"] == 1

    def test_run_round_client_error(self):
        server = self._make_server()
        self._register_clients_with_data(server, 1)
        server.clients["c0"].local_train.return_value = {"error": "No data"}
        result = server.run_round(1)
        # Should return error since no successful updates
        assert "error" in result

    def test_run_round_client_exception(self):
        server = self._make_server()
        self._register_clients_with_data(server, 1)
        server.clients["c0"].local_train.side_effect = RuntimeError("crash")
        result = server.run_round(1)
        assert "error" in result

    def test_train_federation(self):
        server = self._make_server()
        self._register_clients_with_data(server)
        for cid, client in server.clients.items():
            client.local_train.return_value = {
                "client_id": cid,
                "model_params": {"w": MagicMock()},
                "num_samples": 10,
                "loss": 0.5,
                "epochs": 1,
                "batches": 2,
                "compression_used": False,
            }
        result = server.train_federation()
        assert "total_rounds" in result
        assert "client_summary" in result

    def test_train_federation_convergence(self):
        server = self._make_server()
        server.config.num_rounds = 10
        self._register_clients_with_data(server)
        call_count = [0]
        for cid, client in server.clients.items():
            def make_result(c=cid):
                call_count[0] += 1
                return {
                    "client_id": c,
                    "model_params": {"w": MagicMock()},
                    "num_samples": 10,
                    "loss": 0.5,  # constant loss -> should converge
                    "epochs": 1,
                    "batches": 2,
                    "compression_used": False,
                }
            client.local_train.side_effect = lambda params, c=cid: make_result(c)
        result = server.train_federation()
        assert "total_rounds" in result

    def test_train_federation_with_errors(self):
        server = self._make_server()
        self._register_clients_with_data(server, 1)
        server.clients["c0"].local_train.return_value = {"error": "No data"}
        result = server.train_federation()
        assert "total_rounds" in result

    def test_get_client_summary(self):
        server = self._make_server()
        self._register_clients_with_data(server, 2)
        summary = server._get_client_summary()
        assert "c0" in summary
        assert "c1" in summary
        assert "site_name" in summary["c0"]

    def test_save_federation(self):
        server = self._make_server()
        self._register_clients_with_data(server, 1)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            server.save_federation(path)
            assert os.path.exists(path)
            with open(path, "rb") as f:
                data = pickle.load(f)
            assert "config" in data
            assert "current_round" in data
        finally:
            os.unlink(path)


# ===========================================================================
# Tests - Factory function
# ===========================================================================
class TestCreateFederatedSystem:
    def test_default(self):
        model = _make_model()
        with patch("federated_learning_controller.copy.deepcopy", return_value=model):
            server = create_federated_system(model)
        assert isinstance(server, FederatedServer)
        assert server.config.algorithm == FederatedAlgorithm.FEDAVG

    def test_fedprox(self):
        model = _make_model()
        with patch("federated_learning_controller.copy.deepcopy", return_value=model):
            server = create_federated_system(model, algorithm="fedprox")
        assert server.config.algorithm == FederatedAlgorithm.FEDPROX

    def test_fedper(self):
        model = _make_model()
        with patch("federated_learning_controller.copy.deepcopy", return_value=model):
            server = create_federated_system(model, algorithm="fedper")
        assert server.config.algorithm == FederatedAlgorithm.FEDPER

    def test_scaffold(self):
        model = _make_model()
        with patch("federated_learning_controller.copy.deepcopy", return_value=model):
            server = create_federated_system(model, algorithm="scaffold")
        assert server.config.algorithm == FederatedAlgorithm.SCAFFOLD

    def test_unknown_algorithm(self):
        model = _make_model()
        with patch("federated_learning_controller.copy.deepcopy", return_value=model):
            server = create_federated_system(model, algorithm="unknown")
        assert server.config.algorithm == FederatedAlgorithm.FEDAVG

    def test_with_kwargs(self):
        model = _make_model()
        with patch("federated_learning_controller.copy.deepcopy", return_value=model):
            server = create_federated_system(
                model, num_clients=3, algorithm="fedavg",
                local_lr=0.001, nonexistent_key=True)
        assert server.config.num_clients == 3
        assert server.config.local_lr == 0.001
def _make_model():
    """Create a minimal mock model for tests."""
    model = MagicMock()
    model.parameters.return_value = []
    model.named_parameters.return_value = []
    model.state_dict.return_value = {}
    model.load_state_dict = MagicMock()
    model.train = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    model.children.return_value = []
    return model

def _make_client_info(cid="c1", species=BacterialSpecies.GEOBACTER):
    return ClientInfo(
        client_id=cid,
        site_name=f"Site_{cid}",
        location="TestLab",
        mfc_type="dual_chamber",
        bacterial_species=species,
    )


# ===========================================================================
# Tests - Enums
# ===========================================================================
