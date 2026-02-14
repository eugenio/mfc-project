"""Tests for federated_learning_controller - coverage part 2.

Covers FederatedClient, FederatedServer (all aggregation methods,
client selection strategies, run_round, train_federation, save).
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

from sensing_models.sensor_fusion import BacterialSpecies
from federated_learning_controller import (
    FederatedAlgorithm, AggregationMethod, ClientSelectionStrategy,
    FederatedConfig, ClientInfo, FederatedClient, FederatedServer,
    create_federated_system,
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


def _make_info(cid="c1", **kw):
    defaults = dict(client_id=cid, site_name="Site", location="US",
        mfc_type="dual", bacterial_species=BacterialSpecies.GEOBACTER)
    defaults.update(kw)
    return ClientInfo(**defaults)


def _make_client(dp=True):
    return FederatedClient(_make_info(), MagicMock(),
        FederatedConfig(differential_privacy=dp))


def _make_server(agg=AggregationMethod.WEIGHTED_AVERAGE,
                 secure=True, n=3, cpr=2):
    model = MagicMock()
    p = MagicMock(); p.numel.return_value = 100
    model.parameters.return_value = [p]
    model.named_parameters.return_value = [("w", p)]
    return FederatedServer(model, FederatedConfig(
        aggregation=agg, num_clients=n, clients_per_round=cpr,
        num_rounds=3, secure_aggregation=secure))


def _register_n(server, n=3, with_data=True, **kw):
    for i in range(n):
        info = _make_info(f"c{i}", **kw)
        server.register_client(info)
        if with_data:
            server.clients[f"c{i}"].local_data = [1, 2, 3]


@pytest.mark.coverage_extra
class TestFederatedClient:
    def test_init_dp(self):
        client = _make_client(dp=True)
        assert client.client_info.client_id == "c1"
        assert client.dp_mechanism is not None

    def test_init_no_dp(self):
        assert _make_client(dp=False).dp_mechanism is None

    def test_get_model_size(self):
        client = _make_client()
        client.local_model = MagicMock()
        p = MagicMock(); p.numel.return_value = 50; p.requires_grad = True
        client.local_model.parameters.return_value = [p]
        assert client.get_model_size()["total_parameters"] == 50

    def test_local_train_no_data(self):
        assert _make_client().local_train({})["error"] == "No local data available"

    def test_local_train_with_data(self):
        client = _make_client()
        client.local_data = [
            {"features": MagicMock(), "action": 0, "reward": 1.0}
            for _ in range(5)]
        # Ensure stack has no side_effect
        mock_torch.stack.side_effect = None
        mock_torch.stack.return_value = MagicMock(
            to=MagicMock(return_value=MagicMock()))
        result = client.local_train({"w": MagicMock()})
        assert "client_id" in result and "loss" in result

    def test_local_train_try_except_path(self):
        """Test that exceptions in the forward pass are caught."""
        client = _make_client()
        client.local_data = [
            {"features": MagicMock(), "action": 0, "reward": 1.0}]
        mock_torch.stack.side_effect = None
        mock_torch.stack.return_value = MagicMock(
            to=MagicMock(return_value=MagicMock()))
        # Make forward pass fail (inside try block)
        client.local_model = MagicMock(side_effect=RuntimeError("forward fail"))
        result = client.local_train({"w": MagicMock()})
        # Should succeed (exception caught) but with 0 batches
        assert result["batches"] == 0

    def test_local_train_compression(self):
        client = _make_client()
        client.config.compression_ratio = 0.5
        client.local_data = [
            {"features": MagicMock(), "action": 0, "reward": 1.0}]
        mock_torch.stack.side_effect = None
        mock_torch.stack.return_value = MagicMock(
            to=MagicMock(return_value=MagicMock()))
        result = client.local_train({"w": MagicMock()})
        assert result["compression_used"] is True

    def test_local_train_no_compression(self):
        client = _make_client()
        client.config.compression_ratio = 1.0
        client.local_data = [
            {"features": MagicMock(), "action": 0, "reward": 1.0}]
        mock_torch.stack.side_effect = None
        mock_torch.stack.return_value = MagicMock(
            to=MagicMock(return_value=MagicMock()))
        result = client.local_train({"w": MagicMock()})
        assert result["compression_used"] is False

    def test_evaluate_model_empty(self):
        r = _make_client().evaluate_model([])
        assert r["accuracy"] == 0.0 and r["loss"] == float("inf")

    def test_evaluate_model_with_data(self):
        client = _make_client()
        feat = MagicMock()
        feat.unsqueeze.return_value = MagicMock(
            to=MagicMock(return_value=MagicMock()))
        result = client.evaluate_model([{"features": feat, "action": 0}])
        assert "accuracy" in result and "samples_evaluated" in result

    def test_evaluate_model_exception(self):
        client = _make_client()
        feat = MagicMock()
        feat.unsqueeze.side_effect = RuntimeError("unsqueeze fail")
        result = client.evaluate_model([{"features": feat, "action": 0}])
        assert result["accuracy"] == 0.0

    def test_feature_engineer_lazy(self):
        client = _make_client()
        assert client._feature_engineer is None
        mock_fe = MagicMock()
        client._feature_engineer = mock_fe
        assert client.feature_engineer == mock_fe

    def test_add_local_data(self):
        client = _make_client()
        client._feature_engineer = MagicMock()
        client._feature_engineer.extract_features.return_value = {"a": 1.0}
        state = MagicMock()
        state.power_output = 1.0
        state.current_density = 0.5
        state.health_metrics.overall_health_score = 0.8
        state.fused_measurement.fusion_confidence = 0.9
        state.anomalies = []
        client.add_local_data([state], [0], [1.0])
        assert len(client.local_data) == 1
        assert client.client_info.data_samples == 1


@pytest.mark.coverage_extra
class TestFederatedServer:
    def test_init(self):
        server = _make_server()
        assert server.current_round == 0
        assert server.secure_aggregator is not None

    def test_init_no_secure(self):
        assert _make_server(secure=False).secure_aggregator is None

    def test_register_client(self):
        server = _make_server()
        assert server.register_client(_make_info("c1")) is True
        assert "c1" in server.clients

    def test_register_client_failure(self):
        server = _make_server()
        with patch.object(FederatedClient, "__init__",
                          side_effect=Exception("fail")):
            assert server.register_client(_make_info("c1")) is False

    def test_select_clients_empty(self):
        assert _make_server().select_clients(1) == []

    def test_select_random(self):
        server = _make_server()
        _register_n(server)
        assert len(server.select_clients(1)) == 2

    def test_select_cyclic(self):
        server = _make_server()
        server.config.client_selection = ClientSelectionStrategy.CYCLIC
        _register_n(server)
        assert len(server.select_clients(0)) == 2

    def test_select_performance(self):
        server = _make_server()
        server.config.client_selection = ClientSelectionStrategy.PERFORMANCE_BASED
        for i in range(3):
            info = _make_info(f"c{i}", reliability_score=float(i + 1),
                local_loss=float(3 - i))
            server.register_client(info)
            server.clients[f"c{i}"].local_data = [1]
        assert len(server.select_clients(0)) == 2

    def test_select_resource(self):
        server = _make_server()
        server.config.client_selection = ClientSelectionStrategy.RESOURCE_AWARE
        for i in range(3):
            info = _make_info(f"c{i}", computation_power=float(i + 1),
                communication_bandwidth=float(i + 1))
            server.register_client(info)
            server.clients[f"c{i}"].local_data = [1]
        assert len(server.select_clients(0)) == 2

    def test_select_default(self):
        server = _make_server()
        server.config.client_selection = ClientSelectionStrategy.DIVERSITY_BASED
        _register_n(server)
        assert len(server.select_clients(0)) == 2

    def test_aggregate_empty(self):
        assert isinstance(_make_server().aggregate_models([]), dict)

    def test_aggregate_weighted_avg(self):
        server = _make_server()
        _register_n(server)
        mock_torch.stack.side_effect = None
        updates = [{"client_id": "c0", "model_params": {"w": MagicMock()},
                     "num_samples": 10, "compression_used": False}]
        server.aggregate_models(updates)

    def test_aggregate_weighted_zero_samples(self):
        server = _make_server()
        updates = [{"client_id": "c0", "model_params": {"w": MagicMock()},
                     "num_samples": 0}]
        server.aggregate_models(updates)

    def test_aggregate_with_compression(self):
        server = _make_server()
        _register_n(server)
        mock_torch.stack.side_effect = None
        updates = [{"client_id": "c0",
            "model_params": {"w": {"shape": (5,),
                "indices": np.array([0]), "values": np.array([1], dtype=np.uint8),
                "original_size": 5}},
            "num_samples": 10, "compression_used": True}]
        server.aggregate_models(updates)

    def test_aggregate_median(self):
        server = _make_server(AggregationMethod.MEDIAN_AGGREGATION)
        mock_torch.stack.side_effect = None
        mock_torch.stack.return_value = MagicMock()
        mock_torch.median.return_value = (MagicMock(), MagicMock())
        updates = [
            {"client_id": f"c{i}", "model_params": {"w": MagicMock()},
             "num_samples": 10, "compression_used": False} for i in range(2)]
        server.aggregate_models(updates)

    def test_aggregate_trimmed_mean(self):
        server = _make_server(AggregationMethod.TRIMMED_MEAN)
        mock_torch.stack.side_effect = None
        mock_torch.stack.return_value = MagicMock()
        sorted_result = MagicMock()
        sorted_result.__getitem__ = MagicMock(return_value=MagicMock())
        mock_torch.sort.return_value = (sorted_result, MagicMock())
        mock_torch.mean.return_value = MagicMock()
        updates = [
            {"client_id": f"c{i}", "model_params": {"w": MagicMock()},
             "num_samples": 10, "compression_used": False} for i in range(3)]
        server.aggregate_models(updates)

    def test_aggregate_byzantine(self):
        server = _make_server(AggregationMethod.BYZANTINE_ROBUST)
        mock_torch.stack.side_effect = None
        mock_torch.stack.return_value = MagicMock()
        sorted_result = MagicMock()
        sorted_result.__getitem__ = MagicMock(return_value=MagicMock())
        mock_torch.sort.return_value = (sorted_result, MagicMock())
        mock_torch.mean.return_value = MagicMock()
        updates = [
            {"client_id": f"c{i}", "model_params": {"w": MagicMock()},
             "num_samples": 10, "compression_used": False} for i in range(3)]
        server.aggregate_models(updates)

    def test_run_round_no_clients(self):
        assert "error" in _make_server().run_round(1)

    def test_run_round_with_clients(self):
        server = _make_server()
        _register_n(server)
        mock_torch.stack.side_effect = None
        mock_torch.stack.return_value = MagicMock(
            to=MagicMock(return_value=MagicMock()))
        result = server.run_round(1)
        assert "round" in result

    def test_get_client_summary(self):
        server = _make_server()
        server.register_client(_make_info("c1"))
        summary = server._get_client_summary()
        assert "c1" in summary
        assert "site_name" in summary["c1"]

    def test_save_federation(self):
        with patch("builtins.open", MagicMock()):
            with patch("pickle.dump"):
                _make_server().save_federation("/tmp/fed.pkl")

    def test_train_federation(self):
        server = _make_server()
        _register_n(server)
        server.config.num_rounds = 2
        mock_torch.stack.side_effect = None
        mock_torch.stack.return_value = MagicMock(
            to=MagicMock(return_value=MagicMock()))
        result = server.train_federation()
        assert "total_rounds" in result and "total_clients" in result

    def test_train_federation_early_stop(self):
        server = _make_server()
        _register_n(server)
        server.config.num_rounds = 10
        # Pre-fill loss history with converged values
        server.global_metrics["loss"].extend([0.1, 0.1, 0.1, 0.1, 0.1])
        mock_torch.stack.side_effect = None
        mock_torch.stack.return_value = MagicMock(
            to=MagicMock(return_value=MagicMock()))
        result = server.train_federation()
        assert "total_rounds" in result

    def test_train_federation_no_clients(self):
        server = _make_server()
        server.config.num_rounds = 2
        result = server.train_federation()
        assert "total_rounds" in result
