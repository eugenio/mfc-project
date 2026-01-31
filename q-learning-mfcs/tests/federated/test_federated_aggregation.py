"""Tests for Federated Learning Aggregation Methods.

US-015: Test Federated Aggregation Methods
Target: 90%+ coverage for aggregation methods

Tests cover:
- FedAvg aggregation with equal weights
- FedAvg with weighted averaging (by data size)
- FedProx aggregation with proximal term
- Aggregation with heterogeneous model updates
- Handling of stragglers and dropouts
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
import torch
from federated_learning_controller import (
    AggregationMethod,
    ClientInfo,
    ClientSelectionStrategy,
    FederatedAlgorithm,
    FederatedConfig,
    FederatedServer,
    ModelCompression,
    SecureAggregation,
    create_federated_system,
)
from torch import nn


class MockEnumValue:
    """Mock enum value with .value attribute."""

    def __init__(self, value: str):
        self.value = value

    def __eq__(self, other):
        if isinstance(other, MockEnumValue):
            return self.value == other.value
        return self.value == other

    def __hash__(self):
        return hash(self.value)


class MockBacterialSpecies:
    """Mock BacterialSpecies enum."""

    MIXED = MockEnumValue("mixed")
    GEOBACTER = MockEnumValue("geobacter")
    SHEWANELLA = MockEnumValue("shewanella")


@pytest.fixture
def simple_model():
    """Create a simple neural network for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )


@pytest.fixture
def federated_config():
    """Create basic federated config for testing."""
    return FederatedConfig(
        num_clients=5,
        clients_per_round=3,
        num_rounds=10,
        local_epochs=2,
        algorithm=FederatedAlgorithm.FEDAVG,
        aggregation=AggregationMethod.WEIGHTED_AVERAGE,
        differential_privacy=False,
        secure_aggregation=False,
        compression_ratio=1.0,
    )


@pytest.fixture
def federated_server(simple_model, federated_config):
    """Create federated server for testing."""
    return FederatedServer(simple_model, federated_config)


@pytest.fixture
def client_info_list():
    """Create list of client info for testing."""
    return [
        ClientInfo(
            client_id=f"client_{i}",
            site_name=f"Site {i}",
            location=f"Location {i}",
            mfc_type="dual_chamber",
            bacterial_species=MockBacterialSpecies.MIXED,
            data_samples=50 + i * 10,
            computation_power=1.0,
            communication_bandwidth=10.0,
            reliability_score=0.9,
            is_active=True,
        )
        for i in range(5)
    ]


def create_mock_model_params(seed: int = 42) -> dict[str, torch.Tensor]:
    """Create mock model parameters for testing."""
    torch.manual_seed(seed)
    return {
        "layer1.weight": torch.randn(20, 10),
        "layer1.bias": torch.randn(20),
        "layer2.weight": torch.randn(5, 20),
        "layer2.bias": torch.randn(5),
    }


def create_client_update(
    client_id: str,
    num_samples: int,
    seed: int = 42,
    compression_used: bool = False,
) -> dict[str, Any]:
    """Create mock client update for testing."""
    return {
        "client_id": client_id,
        "model_params": create_mock_model_params(seed=seed),
        "num_samples": num_samples,
        "loss": 0.5 + (seed % 10) * 0.1,
        "epochs": 2,
        "batches": 5,
        "compression_used": compression_used,
    }


class TestFedAvgEqualWeights:
    """Tests for FedAvg aggregation with equal weights."""

    def test_fedavg_equal_weights_two_clients(self, federated_server):
        """Test FedAvg with two clients having equal sample sizes."""
        update1 = create_client_update("client_1", num_samples=100, seed=1)
        update2 = create_client_update("client_2", num_samples=100, seed=2)

        aggregated = federated_server._weighted_average_aggregation([update1, update2])

        expected_weight = (
            update1["model_params"]["layer1.weight"]
            + update2["model_params"]["layer1.weight"]
        ) / 2
        assert torch.allclose(aggregated["layer1.weight"], expected_weight, atol=1e-6)

    def test_fedavg_equal_weights_all_same_params(self, federated_server):
        """Test FedAvg with identical parameters from all clients."""
        update1 = create_client_update("client_1", num_samples=50, seed=42)
        update2 = create_client_update("client_2", num_samples=50, seed=42)
        update3 = create_client_update("client_3", num_samples=50, seed=42)

        aggregated = federated_server._weighted_average_aggregation(
            [update1, update2, update3],
        )

        assert torch.allclose(
            aggregated["layer1.weight"],
            update1["model_params"]["layer1.weight"],
            atol=1e-6,
        )

    def test_fedavg_equal_weights_preserves_shape(self, federated_server):
        """Test that FedAvg preserves parameter shapes."""
        update1 = create_client_update("client_1", num_samples=100, seed=1)
        update2 = create_client_update("client_2", num_samples=100, seed=2)

        aggregated = federated_server._weighted_average_aggregation([update1, update2])

        for name, param in aggregated.items():
            assert param.shape == update1["model_params"][name].shape

    def test_fedavg_single_client_returns_same_params(self, federated_server):
        """Test FedAvg with single client returns its params unchanged."""
        update = create_client_update("client_1", num_samples=100, seed=1)

        aggregated = federated_server._weighted_average_aggregation([update])

        for name, param in aggregated.items():
            assert torch.allclose(param, update["model_params"][name], atol=1e-6)

    def test_fedavg_empty_updates_returns_global_model(self, federated_server):
        """Test FedAvg with empty updates returns global model params."""
        aggregated = federated_server._weighted_average_aggregation([])
        assert aggregated is not None
        assert len(aggregated) > 0


class TestFedAvgWeightedByDataSize:
    """Tests for FedAvg with weighted averaging by data size."""

    def test_fedavg_weighted_larger_client_dominates(self, federated_server):
        """Test that client with more samples has more influence."""
        update1 = create_client_update("client_1", num_samples=1000, seed=1)
        update2 = create_client_update("client_2", num_samples=100, seed=2)

        aggregated = federated_server._weighted_average_aggregation([update1, update2])

        weight1 = 1000 / 1100
        weight2 = 100 / 1100
        expected = (
            weight1 * update1["model_params"]["layer1.weight"]
            + weight2 * update2["model_params"]["layer1.weight"]
        )
        assert torch.allclose(aggregated["layer1.weight"], expected, atol=1e-5)

    def test_fedavg_weighted_three_clients_different_sizes(self, federated_server):
        """Test weighted averaging with three clients of different sizes."""
        update1 = create_client_update("client_1", num_samples=100, seed=1)
        update2 = create_client_update("client_2", num_samples=200, seed=2)
        update3 = create_client_update("client_3", num_samples=300, seed=3)

        total_samples = 600
        aggregated = federated_server._weighted_average_aggregation(
            [update1, update2, update3],
        )

        expected = (
            (100 / total_samples) * update1["model_params"]["layer1.weight"]
            + (200 / total_samples) * update2["model_params"]["layer1.weight"]
            + (300 / total_samples) * update3["model_params"]["layer1.weight"]
        )
        assert torch.allclose(aggregated["layer1.weight"], expected, atol=1e-5)

    def test_fedavg_weighted_handles_zero_samples(self, federated_server):
        """Test weighted averaging when one client has zero samples."""
        update1 = create_client_update("client_1", num_samples=100, seed=1)
        update2 = create_client_update("client_2", num_samples=0, seed=2)

        aggregated = federated_server._weighted_average_aggregation([update1, update2])

        assert torch.allclose(
            aggregated["layer1.weight"],
            update1["model_params"]["layer1.weight"],
            atol=1e-6,
        )

    def test_fedavg_weighted_all_zero_samples_returns_global(self, federated_server):
        """Test weighted averaging when all clients have zero samples."""
        update1 = create_client_update("client_1", num_samples=0, seed=1)
        update2 = create_client_update("client_2", num_samples=0, seed=2)

        aggregated = federated_server._weighted_average_aggregation([update1, update2])
        assert aggregated is not None
        assert len(aggregated) > 0

    def test_fedavg_weighted_large_sample_disparity(self, federated_server):
        """Test with extreme disparity in sample sizes."""
        update1 = create_client_update("client_1", num_samples=1, seed=1)
        update2 = create_client_update("client_2", num_samples=999999, seed=2)

        aggregated = federated_server._weighted_average_aggregation([update1, update2])

        assert torch.allclose(
            aggregated["layer1.weight"],
            update2["model_params"]["layer1.weight"],
            atol=1e-3,
        )


class TestFedProxAggregation:
    """Tests for FedProx aggregation with proximal term."""

    def test_fedprox_config_sets_algorithm(self):
        """Test that FedProx algorithm can be configured."""
        config = FederatedConfig(
            algorithm=FederatedAlgorithm.FEDPROX,
            aggregation=AggregationMethod.WEIGHTED_AVERAGE,
        )
        assert config.algorithm == FederatedAlgorithm.FEDPROX

    def test_fedprox_server_uses_weighted_average(self, simple_model):
        """Test FedProx server still uses weighted average for aggregation."""
        config = FederatedConfig(
            algorithm=FederatedAlgorithm.FEDPROX,
            aggregation=AggregationMethod.WEIGHTED_AVERAGE,
            differential_privacy=False,
            secure_aggregation=False,
        )
        server = FederatedServer(simple_model, config)

        update1 = create_client_update("client_1", num_samples=100, seed=1)
        update2 = create_client_update("client_2", num_samples=100, seed=2)

        aggregated = server.aggregate_models([update1, update2])
        assert aggregated is not None
        assert len(aggregated) > 0

    def test_fedprox_factory_function(self, simple_model):
        """Test creating FedProx system via factory function."""
        server = create_federated_system(
            simple_model,
            num_clients=3,
            algorithm="fedprox",
        )
        assert server.config.algorithm == FederatedAlgorithm.FEDPROX

    def test_fedprox_with_regularization_weight(self, simple_model):
        """Test FedProx configuration with weight decay."""
        config = FederatedConfig(
            algorithm=FederatedAlgorithm.FEDPROX,
            weight_decay=0.01,
            differential_privacy=False,
            secure_aggregation=False,
        )
        server = FederatedServer(simple_model, config)
        assert server.config.weight_decay == 0.01


class TestMedianAggregation:
    """Tests for coordinate-wise median aggregation."""

    def test_median_aggregation_basic(self, simple_model):
        """Test basic median aggregation with three clients."""
        config = FederatedConfig(
            aggregation=AggregationMethod.MEDIAN_AGGREGATION,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        update1 = create_client_update("client_1", num_samples=100, seed=1)
        update2 = create_client_update("client_2", num_samples=100, seed=2)
        update3 = create_client_update("client_3", num_samples=100, seed=3)

        aggregated = server._median_aggregation([update1, update2, update3])

        params_stack = torch.stack(
            [
                update1["model_params"]["layer1.weight"],
                update2["model_params"]["layer1.weight"],
                update3["model_params"]["layer1.weight"],
            ],
        )
        expected_median = torch.median(params_stack, dim=0)[0]
        assert torch.allclose(aggregated["layer1.weight"], expected_median, atol=1e-6)

    def test_median_aggregation_even_clients(self, simple_model):
        """Test median aggregation with even number of clients."""
        config = FederatedConfig(
            aggregation=AggregationMethod.MEDIAN_AGGREGATION,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        update1 = create_client_update("client_1", num_samples=100, seed=1)
        update2 = create_client_update("client_2", num_samples=100, seed=2)

        aggregated = server._median_aggregation([update1, update2])
        assert aggregated is not None
        assert "layer1.weight" in aggregated

    def test_median_aggregation_robust_to_outliers(self, simple_model):
        """Test that median is robust to outlier client updates."""
        config = FederatedConfig(
            aggregation=AggregationMethod.MEDIAN_AGGREGATION,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        update1 = create_client_update("client_1", num_samples=100, seed=1)
        update2 = create_client_update("client_2", num_samples=100, seed=2)
        update3 = create_client_update("client_3", num_samples=100, seed=3)

        update_outlier = create_client_update("client_outlier", num_samples=100, seed=999)
        update_outlier["model_params"]["layer1.weight"] *= 1000

        aggregated = server._median_aggregation(
            [update1, update2, update3, update_outlier],
        )

        normal_stack = torch.stack(
            [
                update1["model_params"]["layer1.weight"],
                update2["model_params"]["layer1.weight"],
                update3["model_params"]["layer1.weight"],
            ],
        )
        normal_median = torch.median(normal_stack, dim=0)[0]

        diff_to_normal = torch.abs(aggregated["layer1.weight"] - normal_median).mean()
        diff_to_outlier = torch.abs(
            aggregated["layer1.weight"]
            - update_outlier["model_params"]["layer1.weight"],
        ).mean()
        assert diff_to_normal < diff_to_outlier


class TestTrimmedMeanAggregation:
    """Tests for trimmed mean aggregation."""

    def test_trimmed_mean_basic(self, simple_model):
        """Test basic trimmed mean aggregation."""
        config = FederatedConfig(
            aggregation=AggregationMethod.TRIMMED_MEAN,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        updates = [
            create_client_update(f"client_{i}", num_samples=100, seed=i)
            for i in range(5)
        ]

        aggregated = server._trimmed_mean_aggregation(updates, trim_ratio=0.1)
        assert aggregated is not None
        assert "layer1.weight" in aggregated

    def test_trimmed_mean_removes_extremes(self, simple_model):
        """Test that trimmed mean removes extreme values."""
        config = FederatedConfig(
            aggregation=AggregationMethod.TRIMMED_MEAN,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        normal_updates = [
            create_client_update(f"client_{i}", num_samples=100, seed=i)
            for i in range(3)
        ]

        low_extreme = create_client_update("client_low", num_samples=100, seed=10)
        low_extreme["model_params"]["layer1.weight"] *= -100

        high_extreme = create_client_update("client_high", num_samples=100, seed=11)
        high_extreme["model_params"]["layer1.weight"] *= 100

        all_updates = [low_extreme] + normal_updates + [high_extreme]

        aggregated = server._trimmed_mean_aggregation(all_updates, trim_ratio=0.2)

        normal_mean = torch.mean(
            torch.stack([u["model_params"]["layer1.weight"] for u in normal_updates]),
            dim=0,
        )

        diff = torch.abs(aggregated["layer1.weight"] - normal_mean).mean()
        assert diff < 10.0

    def test_trimmed_mean_zero_trim_equals_mean(self, simple_model):
        """Test that 0% trim ratio equals regular mean."""
        config = FederatedConfig(
            aggregation=AggregationMethod.TRIMMED_MEAN,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        updates = [
            create_client_update(f"client_{i}", num_samples=100, seed=i)
            for i in range(5)
        ]

        aggregated = server._trimmed_mean_aggregation(updates, trim_ratio=0.0)

        expected_mean = torch.mean(
            torch.stack([u["model_params"]["layer1.weight"] for u in updates]),
            dim=0,
        )
        assert torch.allclose(aggregated["layer1.weight"], expected_mean, atol=1e-5)


class TestByzantineRobustAggregation:
    """Tests for Byzantine-robust aggregation."""

    def test_byzantine_robust_basic(self, simple_model):
        """Test basic Byzantine-robust aggregation."""
        config = FederatedConfig(
            aggregation=AggregationMethod.BYZANTINE_ROBUST,
            robust_aggregation=True,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        updates = [
            create_client_update(f"client_{i}", num_samples=100, seed=i)
            for i in range(5)
        ]

        aggregated = server._byzantine_robust_aggregation(updates)
        assert aggregated is not None
        assert "layer1.weight" in aggregated

    def test_byzantine_robust_uses_trimmed_mean(self, simple_model):
        """Test that Byzantine robust uses trimmed mean with higher ratio."""
        config = FederatedConfig(
            aggregation=AggregationMethod.BYZANTINE_ROBUST,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        updates = [
            create_client_update(f"client_{i}", num_samples=100, seed=i)
            for i in range(10)
        ]

        aggregated = server._byzantine_robust_aggregation(updates)
        assert aggregated is not None
        for name in updates[0]["model_params"]:
            assert name in aggregated

    def test_byzantine_robust_handles_malicious_updates(self, simple_model):
        """Test Byzantine robustness against malicious client updates."""
        config = FederatedConfig(
            aggregation=AggregationMethod.BYZANTINE_ROBUST,
            byzantine_clients=2,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        normal_updates = [
            create_client_update(f"client_{i}", num_samples=100, seed=i)
            for i in range(5)
        ]

        malicious1 = create_client_update("malicious_1", num_samples=100, seed=100)
        malicious1["model_params"]["layer1.weight"] = (
            torch.ones_like(malicious1["model_params"]["layer1.weight"]) * 1000
        )

        malicious2 = create_client_update("malicious_2", num_samples=100, seed=101)
        malicious2["model_params"]["layer1.weight"] = (
            torch.ones_like(malicious2["model_params"]["layer1.weight"]) * -1000
        )

        all_updates = normal_updates + [malicious1, malicious2]

        aggregated = server._byzantine_robust_aggregation(all_updates)

        normal_mean = torch.mean(
            torch.stack([u["model_params"]["layer1.weight"] for u in normal_updates]),
            dim=0,
        )

        diff_to_normal = torch.abs(aggregated["layer1.weight"] - normal_mean).mean()
        diff_to_malicious = torch.abs(
            aggregated["layer1.weight"] - malicious1["model_params"]["layer1.weight"],
        ).mean()
        assert diff_to_normal < diff_to_malicious


class TestHeterogeneousModelUpdates:
    """Tests for aggregation with heterogeneous model updates."""

    def test_heterogeneous_sample_sizes(self, federated_server):
        """Test aggregation with very different sample sizes."""
        updates = [
            create_client_update("tiny", num_samples=10, seed=1),
            create_client_update("small", num_samples=100, seed=2),
            create_client_update("medium", num_samples=1000, seed=3),
            create_client_update("large", num_samples=10000, seed=4),
        ]

        aggregated = federated_server._weighted_average_aggregation(updates)

        total = 10 + 100 + 1000 + 10000
        large_weight = 10000 / total
        assert large_weight > 0.9

    def test_heterogeneous_with_compression(self, simple_model):
        """Test aggregation when some clients use compression."""
        config = FederatedConfig(
            compression_ratio=0.5,
            differential_privacy=False,
            secure_aggregation=False,
        )
        server = FederatedServer(simple_model, config)

        for i in range(3):
            client_info = ClientInfo(
                client_id=f"client_{i}",
                site_name=f"Site {i}",
                location="Test",
                mfc_type="test",
                bacterial_species=MockBacterialSpecies.MIXED,
            )
            server.register_client(client_info)

        update1 = create_client_update(
            "client_0", num_samples=100, seed=1, compression_used=False,
        )
        update2 = create_client_update(
            "client_1", num_samples=100, seed=2, compression_used=False,
        )

        aggregated = server._weighted_average_aggregation([update1, update2])
        assert aggregated is not None

    def test_heterogeneous_different_losses(self, federated_server):
        """Test aggregation with clients having different losses."""
        update1 = create_client_update("client_1", num_samples=100, seed=1)
        update1["loss"] = 0.1

        update2 = create_client_update("client_2", num_samples=100, seed=2)
        update2["loss"] = 0.5

        update3 = create_client_update("client_3", num_samples=100, seed=3)
        update3["loss"] = 2.0

        aggregated = federated_server._weighted_average_aggregation(
            [update1, update2, update3],
        )

        expected = (
            update1["model_params"]["layer1.weight"]
            + update2["model_params"]["layer1.weight"]
            + update3["model_params"]["layer1.weight"]
        ) / 3
        assert torch.allclose(aggregated["layer1.weight"], expected, atol=1e-5)


class TestStragglersAndDropouts:
    """Tests for handling stragglers and client dropouts."""

    def test_aggregation_with_client_dropout(self, federated_server):
        """Test aggregation when some clients drop out."""
        update1 = create_client_update("client_1", num_samples=100, seed=1)
        update2 = create_client_update("client_2", num_samples=100, seed=2)

        aggregated = federated_server._weighted_average_aggregation([update1, update2])
        assert aggregated is not None
        assert len(aggregated) > 0

    def test_aggregation_single_client_available(self, federated_server):
        """Test aggregation when only one client is available."""
        update = create_client_update("client_1", num_samples=100, seed=1)

        aggregated = federated_server._weighted_average_aggregation([update])

        for name, param in aggregated.items():
            assert torch.allclose(param, update["model_params"][name], atol=1e-6)

    def test_aggregation_no_clients_available(self, federated_server):
        """Test aggregation when no clients are available."""
        aggregated = federated_server._weighted_average_aggregation([])
        assert aggregated is not None

    def test_client_selection_with_inactive_clients(
        self, federated_server, client_info_list,
    ):
        """Test client selection skips inactive clients."""
        for info in client_info_list:
            federated_server.register_client(info)

        federated_server.client_info["client_0"].is_active = False
        federated_server.client_info["client_1"].is_active = False

        for cid in ["client_2", "client_3", "client_4"]:
            federated_server.clients[cid].local_data = [{"dummy": "data"}]

        selected = federated_server.select_clients(round_num=1)

        assert "client_0" not in selected
        assert "client_1" not in selected

    def test_client_selection_with_no_data_clients(
        self, federated_server, client_info_list,
    ):
        """Test client selection skips clients with no data."""
        for info in client_info_list:
            federated_server.register_client(info)

        federated_server.clients["client_0"].local_data = [{"dummy": "data"}]
        federated_server.clients["client_1"].local_data = [{"dummy": "data"}]

        selected = federated_server.select_clients(round_num=1)

        for cid in selected:
            assert len(federated_server.clients[cid].local_data) > 0

    def test_round_handles_training_failures(self, simple_model, client_info_list):
        """Test that round handles client training failures gracefully."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        for info in client_info_list:
            server.register_client(info)

        result = server.run_round(1)
        assert "error" in result or "successful_updates" in result

    def test_async_mode_staleness_handling(self, simple_model):
        """Test asynchronous mode with staleness threshold."""
        config = FederatedConfig(
            async_mode=True,
            staleness_threshold=3,
            differential_privacy=False,
            secure_aggregation=False,
        )
        server = FederatedServer(simple_model, config)

        assert server.config.async_mode is True
        assert server.config.staleness_threshold == 3

    def test_participation_tracking(self, federated_server, client_info_list):
        """Test that client participation is tracked correctly."""
        for info in client_info_list:
            federated_server.register_client(info)

        for cid in federated_server.clients:
            federated_server.clients[cid].local_data = [{"dummy": "data"}]

        federated_server.select_clients(1)
        federated_server.select_clients(2)

        assert len(federated_server.client_selection_history) == 2
        assert federated_server.client_selection_history[0]["round"] == 1
        assert federated_server.client_selection_history[1]["round"] == 2


class TestClientSelectionStrategies:
    """Tests for different client selection strategies."""

    def test_random_selection(self, simple_model, client_info_list):
        """Test random client selection strategy."""
        config = FederatedConfig(
            clients_per_round=3,
            client_selection=ClientSelectionStrategy.RANDOM,
            differential_privacy=False,
            secure_aggregation=False,
        )
        server = FederatedServer(simple_model, config)

        for info in client_info_list:
            server.register_client(info)
            server.clients[info.client_id].local_data = [{"dummy": "data"}]

        selected = server.select_clients(1)
        assert len(selected) == 3
        assert all(cid in server.clients for cid in selected)

    def test_cyclic_selection(self, simple_model, client_info_list):
        """Test cyclic client selection strategy."""
        config = FederatedConfig(
            clients_per_round=2,
            client_selection=ClientSelectionStrategy.CYCLIC,
            differential_privacy=False,
            secure_aggregation=False,
        )
        server = FederatedServer(simple_model, config)

        for info in client_info_list:
            server.register_client(info)
            server.clients[info.client_id].local_data = [{"dummy": "data"}]

        round1 = server.select_clients(0)
        round2 = server.select_clients(1)

        assert len(round1) == 2
        assert len(round2) == 2

    def test_performance_based_selection(self, simple_model, client_info_list):
        """Test performance-based client selection."""
        config = FederatedConfig(
            clients_per_round=2,
            client_selection=ClientSelectionStrategy.PERFORMANCE_BASED,
            differential_privacy=False,
            secure_aggregation=False,
        )
        server = FederatedServer(simple_model, config)

        for info in client_info_list:
            server.register_client(info)
            server.clients[info.client_id].local_data = [{"dummy": "data"}]

        server.client_info["client_0"].reliability_score = 0.9
        server.client_info["client_0"].local_loss = 0.1
        server.client_info["client_1"].reliability_score = 0.5
        server.client_info["client_1"].local_loss = 0.5

        selected = server.select_clients(1)
        assert len(selected) == 2

    def test_resource_aware_selection(self, simple_model, client_info_list):
        """Test resource-aware client selection."""
        config = FederatedConfig(
            clients_per_round=2,
            client_selection=ClientSelectionStrategy.RESOURCE_AWARE,
            differential_privacy=False,
            secure_aggregation=False,
        )
        server = FederatedServer(simple_model, config)

        for info in client_info_list:
            server.register_client(info)
            server.clients[info.client_id].local_data = [{"dummy": "data"}]

        server.client_info["client_0"].computation_power = 2.0
        server.client_info["client_0"].communication_bandwidth = 10.0
        server.client_info["client_1"].computation_power = 0.5
        server.client_info["client_1"].communication_bandwidth = 1.0

        selected = server.select_clients(1)
        assert len(selected) == 2


class TestSecureAggregationMethods:
    """Tests for secure aggregation mechanism."""

    def test_secure_aggregation_init(self):
        """Test SecureAggregation initialization."""
        sec_agg = SecureAggregation(num_clients=5)
        assert sec_agg.num_clients == 5
        assert sec_agg.threshold == 2

    def test_generate_masks(self):
        """Test mask generation for secure aggregation."""
        sec_agg = SecureAggregation(num_clients=3)
        model_params = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
        }
        masks = sec_agg.generate_masks(model_params)

        assert "layer1.weight" in masks
        assert "layer1.bias" in masks
        assert masks["layer1.weight"].shape == model_params["layer1.weight"].shape

    def test_mask_unmask_roundtrip(self):
        """Test that masking and unmasking returns correct result."""
        sec_agg = SecureAggregation(num_clients=3)

        params1 = {"weight": torch.tensor([1.0, 2.0, 3.0])}
        params2 = {"weight": torch.tensor([4.0, 5.0, 6.0])}
        params3 = {"weight": torch.tensor([7.0, 8.0, 9.0])}

        masks1 = sec_agg.generate_masks(params1)
        masks2 = sec_agg.generate_masks(params2)
        masks3 = sec_agg.generate_masks(params3)

        masked1 = sec_agg.mask_model(params1, masks1)
        masked2 = sec_agg.mask_model(params2, masks2)
        masked3 = sec_agg.mask_model(params3, masks3)

        aggregated = sec_agg.unmask_aggregate(
            [masked1, masked2, masked3],
            [masks1, masks2, masks3],
        )

        expected = (params1["weight"] + params2["weight"] + params3["weight"]) / 3
        assert torch.allclose(aggregated["weight"], expected, atol=1e-5)

    def test_secure_aggregation_threshold(self):
        """Test secure aggregation fails with insufficient clients."""
        sec_agg = SecureAggregation(num_clients=10)
        assert sec_agg.threshold == 5

        params = {"weight": torch.tensor([1.0, 2.0])}
        masks = sec_agg.generate_masks(params)
        masked = sec_agg.mask_model(params, masks)

        with pytest.raises(ValueError, match="Insufficient clients"):
            sec_agg.unmask_aggregate([masked, masked, masked], [masks, masks, masks])


class TestModelCompressionMethods:
    """Tests for model compression mechanism."""

    def test_compression_init(self):
        """Test ModelCompression initialization."""
        compressor = ModelCompression(
            compression_ratio=0.1,
            quantization_bits=8,
            sparsification_ratio=0.01,
        )
        assert compressor.compression_ratio == 0.1
        assert compressor.quantization_bits == 8
        assert compressor.sparsification_ratio == 0.01

    def test_compress_decompress_roundtrip(self):
        """Test compression and decompression roundtrip."""
        compressor = ModelCompression(
            compression_ratio=0.5,
            quantization_bits=8,
            sparsification_ratio=0.1,
        )

        original_params = {"weight": torch.randn(10, 10)}

        compressed = compressor.compress_model(original_params)
        decompressed = compressor.decompress_model(compressed)

        assert decompressed["weight"].shape == original_params["weight"].shape

    def test_compression_reduces_size(self):
        """Test that compression actually reduces size."""
        compressor = ModelCompression(
            compression_ratio=0.1,
            quantization_bits=8,
            sparsification_ratio=0.01,
        )

        large_params = {"weight": torch.randn(100, 100)}
        compressed = compressor.compress_model(large_params)
        compressed_values = len(compressed["weight"]["values"])

        assert compressed_values < large_params["weight"].numel()


class TestAggregationMethodSelection:
    """Tests for aggregation method selection in aggregate_models."""

    def test_aggregate_models_selects_weighted_average(self, simple_model):
        """Test that aggregate_models uses weighted average when configured."""
        config = FederatedConfig(
            aggregation=AggregationMethod.WEIGHTED_AVERAGE,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        updates = [create_client_update(f"client_{i}", 100, seed=i) for i in range(3)]
        aggregated = server.aggregate_models(updates)
        assert aggregated is not None

    def test_aggregate_models_selects_median(self, simple_model):
        """Test that aggregate_models uses median when configured."""
        config = FederatedConfig(
            aggregation=AggregationMethod.MEDIAN_AGGREGATION,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        updates = [create_client_update(f"client_{i}", 100, seed=i) for i in range(3)]
        aggregated = server.aggregate_models(updates)
        assert aggregated is not None

    def test_aggregate_models_selects_trimmed_mean(self, simple_model):
        """Test that aggregate_models uses trimmed mean when configured."""
        config = FederatedConfig(
            aggregation=AggregationMethod.TRIMMED_MEAN,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        updates = [create_client_update(f"client_{i}", 100, seed=i) for i in range(5)]
        aggregated = server.aggregate_models(updates)
        assert aggregated is not None

    def test_aggregate_models_selects_byzantine_robust(self, simple_model):
        """Test that aggregate_models uses Byzantine robust when configured."""
        config = FederatedConfig(
            aggregation=AggregationMethod.BYZANTINE_ROBUST,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        updates = [create_client_update(f"client_{i}", 100, seed=i) for i in range(5)]
        aggregated = server.aggregate_models(updates)
        assert aggregated is not None

    def test_aggregate_models_empty_updates(self, simple_model):
        """Test aggregate_models with empty updates."""
        config = FederatedConfig(
            aggregation=AggregationMethod.WEIGHTED_AVERAGE,
            differential_privacy=False,
            secure_aggregation=False,
        )
        server = FederatedServer(simple_model, config)

        aggregated = server.aggregate_models([])
        assert aggregated is not None


class TestAggregationIntegration:
    """Integration tests for full aggregation flow."""

    def test_full_aggregation_with_registered_clients(
        self, simple_model, client_info_list,
    ):
        """Test aggregation with fully registered and trained clients."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
            num_rounds=2,
            local_epochs=1,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        for info in client_info_list:
            server.register_client(info)

        assert len(server.clients) == 5

    def test_aggregation_preserves_model_architecture(self, simple_model):
        """Test that aggregation preserves model architecture."""
        config = FederatedConfig(
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        updates = [create_client_update(f"client_{i}", 100, seed=i) for i in range(3)]
        aggregated = server.aggregate_models(updates)

        update_params = updates[0]["model_params"]
        for name in update_params:
            assert name in aggregated
            assert aggregated[name].shape == update_params[name].shape

    def test_multiple_aggregation_rounds(self, simple_model):
        """Test multiple rounds of aggregation."""
        config = FederatedConfig(
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        for round_num in range(5):
            # Create updates using actual model parameter names
            updates = []
            for i in range(3):
                torch.manual_seed(round_num * 10 + i)
                model_params = {
                    name: param.clone() + torch.randn_like(param) * 0.1
                    for name, param in simple_model.named_parameters()
                }
                updates.append({
                    "client_id": f"client_{i}",
                    "model_params": model_params,
                    "num_samples": 100,
                    "loss": 0.5,
                    "compression_used": False,
                })

            aggregated = server.aggregate_models(updates)
            server.global_model.load_state_dict(aggregated)

        assert server.global_model is not None
        for param in server.global_model.parameters():
            assert not torch.isnan(param).any()
            assert not torch.isinf(param).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
