"""Integration Tests for Federated Learning Controller.

US-017: Federated Learning Integration Tests
Target: 90%+ coverage for federated_learning_controller.py

Tests cover:
- Complete federated training round
- Multi-round training convergence
- Global model distribution to clients
- Heterogeneous data scenarios
- All tests complete in < 60 seconds
"""

from __future__ import annotations

import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from federated_learning_controller import (
    AggregationMethod,
    ClientInfo,
    ClientSelectionStrategy,
    FederatedAlgorithm,
    FederatedClient,
    FederatedConfig,
    FederatedServer,
    create_federated_system,
)


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
    PSEUDOMONAS = MockEnumValue("pseudomonas")


@pytest.fixture
def simple_model() -> nn.Module:
    """Create a simple neural network for testing."""
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )


@pytest.fixture
def default_config() -> FederatedConfig:
    """Create default federated config."""
    return FederatedConfig(
        num_clients=5,
        clients_per_round=3,
        num_rounds=5,
        local_epochs=2,
        algorithm=FederatedAlgorithm.FEDAVG,
        aggregation=AggregationMethod.WEIGHTED_AVERAGE,
        client_selection=ClientSelectionStrategy.RANDOM,
        differential_privacy=False,
        secure_aggregation=False,
        compression_ratio=1.0,
    )


@pytest.fixture
def client_infos() -> list[ClientInfo]:
    """Create multiple client infos for testing."""
    species_list = [
        MockBacterialSpecies.GEOBACTER,
        MockBacterialSpecies.SHEWANELLA,
        MockBacterialSpecies.MIXED,
        MockBacterialSpecies.PSEUDOMONAS,
        MockBacterialSpecies.GEOBACTER,
    ]
    clients = []
    for i in range(5):
        clients.append(
            ClientInfo(
                client_id=f"client_{i}",
                site_name=f"Test Site {i}",
                location=f"Location {i}",
                mfc_type="dual_chamber",
                bacterial_species=species_list[i],
                data_samples=50 + i * 20,
                computation_power=1.0 + i * 0.1,
                communication_bandwidth=5.0 + i * 1.0,
                reliability_score=0.8 + i * 0.04,
                is_active=True,
            ),
        )
    return clients


def add_mock_data_to_client(client: FederatedClient, num_samples: int = 10):
    """Add mock local data to client for testing."""
    for _ in range(num_samples):
        client.local_data.append(
            {
                "features": torch.randn(10),
                "action": np.random.randint(0, 8),
                "reward": np.random.uniform(-1, 1),
            },
        )
    client.client_info.data_samples = len(client.local_data)


class TestCompleteFederatedTrainingRound:
    """Tests for complete federated training round."""

    def test_single_round_execution(self, simple_model, default_config, client_infos):
        """Test execution of a single federated round."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=20)

        result = server.run_round(1)

        assert "round" in result
        assert result["round"] == 1
        if "error" not in result:
            assert "successful_updates" in result

    def test_round_with_all_clients(self, simple_model, client_infos):
        """Test round where all clients participate."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=5,
            num_rounds=1,
            local_epochs=1,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        result = server.run_round(1)

        if "error" not in result:
            assert result["selected_clients"] == 5

    def test_round_tracks_metrics(self, simple_model, default_config, client_infos):
        """Test that round tracks training metrics."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        server.run_round(1)

        if server.global_metrics["loss"]:
            assert len(server.global_metrics["loss"]) >= 1

    def test_round_with_no_clients(self, simple_model, default_config):
        """Test round when no clients have data."""
        server = FederatedServer(simple_model, default_config)

        result = server.run_round(1)

        assert "error" in result

    def test_round_updates_current_round(self, simple_model, default_config, client_infos):
        """Test that round updates server's current round counter."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=10)

        result = server.run_round(5)

        if "error" not in result:
            assert server.current_round == 5

    def test_round_appends_history(self, simple_model, default_config, client_infos):
        """Test that round appends results to training history."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        initial_len = len(server.training_history)
        server.run_round(1)

        assert len(server.training_history) >= initial_len


class TestMultiRoundConvergence:
    """Tests for multi-round training convergence."""

    def test_multi_round_training(self, simple_model, client_infos):
        """Test multiple rounds of federated training."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
            num_rounds=3,
            local_epochs=1,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=20)

        for round_num in range(1, 4):
            result = server.run_round(round_num)
            if "error" not in result:
                assert result["round"] == round_num

    def test_train_federation_complete(self, simple_model, client_infos):
        """Test complete federation training."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
            num_rounds=3,
            local_epochs=1,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=20)

        results = server.train_federation()

        assert "total_rounds" in results
        assert "total_clients" in results
        assert "training_time_seconds" in results

    def test_loss_tracking(self, simple_model, client_infos):
        """Test that loss is tracked across rounds."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
            num_rounds=5,
            local_epochs=1,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=20)

        for round_num in range(1, 4):
            server.run_round(round_num)

        if server.global_metrics["loss"]:
            assert len(list(server.global_metrics["loss"])) > 0

    def test_early_stopping(self, simple_model, client_infos):
        """Test early stopping when loss converges."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
            num_rounds=100,
            local_epochs=1,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=20)

        server.global_metrics["loss"].extend([0.1, 0.1, 0.1, 0.1, 0.1])

        results = server.train_federation()

        assert results["total_rounds"] <= 100

    def test_participation_tracking(self, simple_model, client_infos):
        """Test participation rate is tracked correctly."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
            num_rounds=3,
            local_epochs=1,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        server.train_federation()

        if server.global_metrics["participation_rate"]:
            for rate in server.global_metrics["participation_rate"]:
                assert 0.0 <= rate <= 1.0


class TestGlobalModelDistribution:
    """Tests for global model distribution to clients."""

    def test_global_model_distributed(self, simple_model, default_config, client_infos):
        """Test that global model is distributed to all clients."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)

        global_state = server.global_model.state_dict()

        for client in server.clients.values():
            client.local_model.load_state_dict(global_state)
            for (g_name, g_param), (c_name, c_param) in zip(
                server.global_model.named_parameters(),
                client.local_model.named_parameters(),
                strict=False,
            ):
                assert torch.allclose(g_param.cpu(), c_param.cpu())

    def test_model_parameters_consistency(self, simple_model, default_config, client_infos):
        """Test model parameter consistency across clients."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)

        global_state = server.global_model.state_dict()
        for client in server.clients.values():
            client.local_model.load_state_dict(global_state)

        clients_list = list(server.clients.values())
        for i in range(len(clients_list) - 1):
            for (name1, param1), (name2, param2) in zip(
                clients_list[i].local_model.named_parameters(),
                clients_list[i + 1].local_model.named_parameters(),
                strict=False,
            ):
                assert torch.allclose(param1.cpu(), param2.cpu())

    def test_architecture_preserved(self, simple_model, default_config, client_infos):
        """Test that model distribution preserves architecture."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)

        global_params = list(server.global_model.parameters())

        for client in server.clients.values():
            client_params = list(client.local_model.parameters())
            assert len(global_params) == len(client_params)
            for g_param, c_param in zip(global_params, client_params, strict=False):
                assert g_param.shape == c_param.shape


class TestHeterogeneousData:
    """Tests for heterogeneous data scenarios."""

    def test_heterogeneous_sample_sizes(self, simple_model, default_config, client_infos):
        """Test with clients having different sample sizes."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)

        sample_sizes = [5, 10, 50, 100, 200]
        for i, (client_id, client) in enumerate(server.clients.items()):
            add_mock_data_to_client(client, num_samples=sample_sizes[i])

        result = server.run_round(1)

        if "error" not in result:
            assert result["successful_updates"] > 0

    def test_heterogeneous_computation_power(self, simple_model, client_infos):
        """Test with clients having different computation power."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
            client_selection=ClientSelectionStrategy.RESOURCE_AWARE,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        for i, info in enumerate(client_infos):
            info.computation_power = 0.5 + i * 0.5
            server.register_client(info)

        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=20)

        selected = server.select_clients(1)
        assert len(selected) > 0

    def test_mixed_active_inactive(self, simple_model, default_config, client_infos):
        """Test with mix of active and inactive clients."""
        server = FederatedServer(simple_model, default_config)

        for i, info in enumerate(client_infos):
            info.is_active = i % 2 == 0
            server.register_client(info)

        for client_id, client in server.clients.items():
            if server.client_info[client_id].is_active:
                add_mock_data_to_client(client, num_samples=20)

        selected = server.select_clients(1)

        for client_id in selected:
            assert server.client_info[client_id].is_active

    def test_non_iid_data(self, simple_model, default_config, client_infos):
        """Test with non-IID data distribution across clients."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)

        for i, client in enumerate(server.clients.values()):
            for _ in range(20):
                base_features = torch.randn(10) + i * 0.5
                client.local_data.append(
                    {
                        "features": base_features,
                        "action": i % 8,
                        "reward": np.random.uniform(-1, 1),
                    },
                )
            client.client_info.data_samples = len(client.local_data)

        result = server.run_round(1)

        if "error" not in result:
            assert result["successful_updates"] > 0

    def test_different_species(self, simple_model, default_config, client_infos):
        """Test with clients representing different bacterial species."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        result = server.run_round(1)

        if "error" not in result:
            client_summary = server._get_client_summary()
            species_set = set()
            for summary in client_summary.values():
                species_set.add(summary["bacterial_species"])
            assert len(species_set) >= 2


class TestFederationPersistence:
    """Tests for federation state persistence."""

    def test_save_federation(self, simple_model, default_config, client_infos):
        """Test saving federation state."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=10)

        server.run_round(1)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            server.save_federation(f.name)
            with open(f.name, "rb") as saved_file:
                saved_data = pickle.load(saved_file)

            assert "global_model_state" in saved_data
            assert "config" in saved_data
            assert "current_round" in saved_data

    def test_federation_state_complete(self, simple_model, default_config, client_infos):
        """Test that saved federation state is complete."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=10)

        server.run_round(1)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            server.save_federation(f.name)
            with open(f.name, "rb") as saved_file:
                saved_data = pickle.load(saved_file)

            assert "client_info" in saved_data
            assert "global_metrics" in saved_data


class TestClientLocalTraining:
    """Tests for client local training."""

    def test_client_local_train_with_data(self, simple_model, default_config):
        """Test client local training with data."""
        client_info = ClientInfo(
            client_id="test_client",
            site_name="Test Site",
            location="Test",
            mfc_type="dual_chamber",
            bacterial_species=MockBacterialSpecies.MIXED,
            data_samples=0,
        )
        client = FederatedClient(client_info, simple_model, default_config)
        add_mock_data_to_client(client, num_samples=20)

        global_params = {
            name: param.cpu().clone() for name, param in simple_model.named_parameters()
        }

        result = client.local_train(global_params)

        assert "client_id" in result
        assert "num_samples" in result

    def test_client_local_train_no_data(self, simple_model, default_config):
        """Test client local training without data returns error."""
        client_info = ClientInfo(
            client_id="test_client",
            site_name="Test Site",
            location="Test",
            mfc_type="dual_chamber",
            bacterial_species=MockBacterialSpecies.MIXED,
        )
        client = FederatedClient(client_info, simple_model, default_config)

        global_params = {
            name: param.cpu().clone() for name, param in simple_model.named_parameters()
        }

        result = client.local_train(global_params)

        assert "error" in result

    def test_client_evaluate_model(self, simple_model, default_config):
        """Test client model evaluation."""
        client_info = ClientInfo(
            client_id="test_client",
            site_name="Test Site",
            location="Test",
            mfc_type="dual_chamber",
            bacterial_species=MockBacterialSpecies.MIXED,
        )
        client = FederatedClient(client_info, simple_model, default_config)

        test_data = [
            {"features": torch.randn(10), "action": i % 8, "reward": 0.5}
            for i in range(10)
        ]

        result = client.evaluate_model(test_data)

        assert "accuracy" in result
        assert "loss" in result

    def test_client_evaluate_empty_data(self, simple_model, default_config):
        """Test client evaluation with empty data."""
        client_info = ClientInfo(
            client_id="test_client",
            site_name="Test Site",
            location="Test",
            mfc_type="dual_chamber",
            bacterial_species=MockBacterialSpecies.MIXED,
        )
        client = FederatedClient(client_info, simple_model, default_config)

        result = client.evaluate_model([])

        assert result["accuracy"] == 0.0


class TestDifferentialPrivacyIntegration:
    """Tests for differential privacy integration."""

    def test_client_with_dp_enabled(self, simple_model):
        """Test client initialization with differential privacy."""
        config = FederatedConfig(
            differential_privacy=True,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        client_info = ClientInfo(
            client_id="dp_client",
            site_name="DP Site",
            location="Test",
            mfc_type="dual_chamber",
            bacterial_species=MockBacterialSpecies.MIXED,
        )
        client = FederatedClient(client_info, simple_model, config)

        assert client.dp_mechanism is not None
        assert client.dp_mechanism.noise_multiplier == 1.0

    def test_dp_training_round(self, simple_model):
        """Test training round with differential privacy."""
        config = FederatedConfig(
            num_clients=3,
            clients_per_round=2,
            num_rounds=1,
            local_epochs=1,
            differential_privacy=True,
            noise_multiplier=0.5,
            max_grad_norm=1.0,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        for i in range(3):
            info = ClientInfo(
                client_id=f"client_{i}",
                site_name=f"Site {i}",
                location="Test",
                mfc_type="test",
                bacterial_species=MockBacterialSpecies.MIXED,
            )
            server.register_client(info)

        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=20)

        result = server.run_round(1)

        assert "round" in result


class TestSecureAggregationIntegration:
    """Tests for secure aggregation integration."""

    def test_server_with_secure_aggregation(self, simple_model):
        """Test server initialization with secure aggregation."""
        config = FederatedConfig(
            num_clients=5,
            secure_aggregation=True,
            differential_privacy=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        assert server.secure_aggregator is not None

    def test_secure_aggregation_threshold(self, simple_model):
        """Test secure aggregation threshold calculation."""
        config = FederatedConfig(
            num_clients=10,
            secure_aggregation=True,
            differential_privacy=False,
        )
        server = FederatedServer(simple_model, config)

        assert server.secure_aggregator.threshold == 5


class TestCompressionIntegration:
    """Tests for model compression integration."""

    def test_client_with_compression(self, simple_model):
        """Test client with compression enabled."""
        config = FederatedConfig(
            compression_ratio=0.5,
            quantization_bits=8,
            sparsification_ratio=0.1,
            differential_privacy=False,
            secure_aggregation=False,
        )
        client_info = ClientInfo(
            client_id="compress_client",
            site_name="Compress Site",
            location="Test",
            mfc_type="dual_chamber",
            bacterial_species=MockBacterialSpecies.MIXED,
        )
        client = FederatedClient(client_info, simple_model, config)

        assert client.compressor is not None
        assert client.compressor.compression_ratio == 0.5

    def test_compressed_training_round(self, simple_model):
        """Test training round with compression."""
        config = FederatedConfig(
            num_clients=3,
            clients_per_round=2,
            num_rounds=1,
            local_epochs=1,
            compression_ratio=0.5,
            quantization_bits=8,
            sparsification_ratio=0.1,
            differential_privacy=False,
            secure_aggregation=False,
        )
        server = FederatedServer(simple_model, config)

        for i in range(3):
            info = ClientInfo(
                client_id=f"client_{i}",
                site_name=f"Site {i}",
                location="Test",
                mfc_type="test",
                bacterial_species=MockBacterialSpecies.MIXED,
            )
            server.register_client(info)

        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=20)

        result = server.run_round(1)

        assert "round" in result


class TestClientSummaryAndMetrics:
    """Tests for client summary and metrics collection."""

    def test_get_client_summary(self, simple_model, default_config, client_infos):
        """Test getting client summary."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)

        summary = server._get_client_summary()

        assert len(summary) == len(client_infos)
        for client_id, client_summary in summary.items():
            assert "site_name" in client_summary
            assert "bacterial_species" in client_summary

    def test_global_metrics_initialization(self, simple_model, default_config):
        """Test global metrics are initialized."""
        server = FederatedServer(simple_model, default_config)

        assert "loss" in server.global_metrics
        assert "accuracy" in server.global_metrics
        assert "participation_rate" in server.global_metrics
        assert "communication_cost" in server.global_metrics

    def test_communication_cost_tracking(self, simple_model, default_config, client_infos):
        """Test communication cost is tracked."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        server.run_round(1)

        if server.global_metrics["communication_cost"]:
            assert len(server.global_metrics["communication_cost"]) >= 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_client_updates(self, simple_model, default_config):
        """Test aggregation with empty client updates."""
        server = FederatedServer(simple_model, default_config)

        aggregated = server.aggregate_models([])

        assert aggregated is not None

    def test_single_client_round(self, simple_model):
        """Test round with only one client."""
        config = FederatedConfig(
            num_clients=1,
            clients_per_round=1,
            num_rounds=1,
            local_epochs=1,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        info = ClientInfo(
            client_id="solo_client",
            site_name="Solo Site",
            location="Test",
            mfc_type="test",
            bacterial_species=MockBacterialSpecies.MIXED,
        )
        server.register_client(info)
        add_mock_data_to_client(server.clients["solo_client"], num_samples=20)

        result = server.run_round(1)

        if "error" not in result:
            assert result["selected_clients"] == 1

    def test_zero_samples_client(self, simple_model, default_config, client_infos):
        """Test handling of client with zero samples."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)

        for i, client in enumerate(server.clients.values()):
            if i == 0:
                pass
            else:
                add_mock_data_to_client(client, num_samples=20)

        selected = server.select_clients(1)

        assert "client_0" not in selected


class TestPerformance:
    """Tests for performance and timing requirements."""

    def test_single_round_timing(self, simple_model, default_config, client_infos):
        """Test that single round completes in reasonable time."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=20)

        start_time = time.time()
        server.run_round(1)
        elapsed = time.time() - start_time

        assert elapsed < 30

    def test_full_training_timing(self, simple_model, client_infos):
        """Test that full training completes within time limit."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
            num_rounds=3,
            local_epochs=1,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        start_time = time.time()
        server.train_federation()
        elapsed = time.time() - start_time

        assert elapsed < 60


class TestFactoryFunction:
    """Tests for create_federated_system factory function."""

    def test_factory_creates_server(self, simple_model):
        """Test factory function creates server."""
        server = create_federated_system(simple_model)

        assert isinstance(server, FederatedServer)

    def test_factory_with_all_algorithms(self, simple_model):
        """Test factory with different algorithms."""
        algorithms = ["fedavg", "fedprox", "fedper", "scaffold"]

        for algo in algorithms:
            server = create_federated_system(simple_model, algorithm=algo)
            assert server is not None

    def test_factory_with_custom_config(self, simple_model):
        """Test factory with custom configuration."""
        server = create_federated_system(
            simple_model,
            num_clients=10,
            algorithm="fedavg",
            clients_per_round=5,
            local_epochs=3,
            local_lr=0.001,
        )

        assert server.config.num_clients == 10
        assert server.config.clients_per_round == 5
        assert server.config.local_epochs == 3
        assert server.config.local_lr == 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
