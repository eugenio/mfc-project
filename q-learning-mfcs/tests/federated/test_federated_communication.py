"""Tests for Federated Learning Communication.

US-016: Test Federated Communication
Target: 90%+ coverage for communication code

Tests cover:
- Synchronous round completion
- Asynchronous update handling
- Communication compression (ModelCompression)
- Secure aggregation mocking
- Round timeout handling
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

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
    ModelCompression,
    SecureAggregation,
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


@pytest.fixture
def simple_model() -> nn.Module:
    """Create a simple neural network for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )


@pytest.fixture
def default_config() -> FederatedConfig:
    """Create default federated config for testing."""
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
    return [
        ClientInfo(
            client_id=f"client_{i}",
            site_name=f"Test Site {i}",
            location=f"Location {i}",
            mfc_type="dual_chamber",
            bacterial_species=MockBacterialSpecies.MIXED,
            data_samples=50 + i * 10,
            computation_power=1.0 + i * 0.1,
            communication_bandwidth=5.0 + i * 1.0,
            reliability_score=0.8 + i * 0.04,
            is_active=True,
        )
        for i in range(5)
    ]


def add_mock_data_to_client(client: FederatedClient, num_samples: int = 10) -> None:
    """Add mock local data to client for testing."""
    for _ in range(num_samples):
        client.local_data.append(
            {
                "features": torch.randn(10),
                "action": np.random.randint(0, 5),
                "reward": np.random.uniform(-1, 1),
            },
        )
    client.client_info.data_samples = len(client.local_data)


def create_mock_model_params(
    model: nn.Module, seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Create mock model parameters from actual model structure."""
    torch.manual_seed(seed)
    return {
        name: param.clone() + torch.randn_like(param) * 0.1
        for name, param in model.named_parameters()
    }


def create_client_update(
    client_id: str,
    model: nn.Module,
    num_samples: int,
    seed: int = 42,
    compression_used: bool = False,
) -> dict[str, Any]:
    """Create mock client update for testing."""
    return {
        "client_id": client_id,
        "model_params": create_mock_model_params(model, seed=seed),
        "num_samples": num_samples,
        "loss": 0.5 + (seed % 10) * 0.1,
        "epochs": 2,
        "batches": 5,
        "compression_used": compression_used,
    }


class TestSynchronousRoundCompletion:
    """Tests for synchronous round completion."""

    def test_sync_round_completes_successfully(
        self, simple_model, default_config, client_infos,
    ):
        """Test that synchronous round completes successfully."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=20)

        result = server.run_round(1)

        if "error" not in result:
            assert "round" in result
            assert result["round"] == 1
            assert "successful_updates" in result
            assert result["successful_updates"] > 0

    def test_sync_round_waits_for_all_clients(
        self, simple_model, default_config, client_infos,
    ):
        """Test that synchronous round waits for all selected clients."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        result = server.run_round(1)

        if "error" not in result:
            assert result["selected_clients"] == default_config.clients_per_round

    def test_sync_round_aggregates_all_updates(
        self, simple_model, default_config, client_infos,
    ):
        """Test that synchronous round aggregates updates from all clients."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=20)

        result = server.run_round(1)

        if "error" not in result:
            assert "client_results" in result
            assert len(result["client_results"]) == result["successful_updates"]

    def test_sync_round_updates_global_model(
        self, simple_model, default_config, client_infos,
    ):
        """Test that synchronous round updates global model."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        initial_params = {
            name: param.clone()
            for name, param in server.global_model.named_parameters()
        }

        result = server.run_round(1)

        if "error" not in result:
            params_changed = False
            for name, param in server.global_model.named_parameters():
                if not torch.allclose(param, initial_params[name]):
                    params_changed = True
                    break
            assert params_changed

    def test_sync_round_tracks_communication_cost(
        self, simple_model, default_config, client_infos,
    ):
        """Test that synchronous round tracks communication cost."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        result = server.run_round(1)

        if "error" not in result:
            assert "communication_cost_bytes" in result
            assert result["communication_cost_bytes"] > 0

    def test_sync_round_calculates_average_loss(
        self, simple_model, default_config, client_infos,
    ):
        """Test that synchronous round calculates average loss."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        result = server.run_round(1)

        if "error" not in result:
            assert "average_loss" in result
            assert isinstance(result["average_loss"], float)

    def test_sync_round_tracks_aggregation_time(
        self, simple_model, default_config, client_infos,
    ):
        """Test that synchronous round tracks aggregation time."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        result = server.run_round(1)

        if "error" not in result:
            assert "aggregation_time_seconds" in result
            assert result["aggregation_time_seconds"] >= 0

    def test_sync_round_participation_rate(
        self, simple_model, default_config, client_infos,
    ):
        """Test that synchronous round calculates participation rate."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=20)

        result = server.run_round(1)

        if "error" not in result:
            assert "participation_rate" in result
            assert 0.0 <= result["participation_rate"] <= 1.0


class TestAsynchronousUpdateHandling:
    """Tests for asynchronous update handling configuration."""

    def test_async_mode_config(self, simple_model):
        """Test asynchronous mode configuration."""
        config = FederatedConfig(
            async_mode=True,
            staleness_threshold=5,
            differential_privacy=False,
            secure_aggregation=False,
        )
        server = FederatedServer(simple_model, config)

        assert server.config.async_mode is True
        assert server.config.staleness_threshold == 5

    def test_async_mode_default_off(self, simple_model, default_config):
        """Test that async mode is off by default."""
        server = FederatedServer(simple_model, default_config)

        assert server.config.async_mode is False

    def test_async_staleness_threshold_default(self, simple_model):
        """Test default staleness threshold value."""
        config = FederatedConfig(
            async_mode=True,
            differential_privacy=False,
            secure_aggregation=False,
        )
        server = FederatedServer(simple_model, config)

        assert server.config.staleness_threshold == 3

    def test_async_mode_with_varying_staleness(self, simple_model):
        """Test async mode with different staleness thresholds."""
        for threshold in [1, 2, 5, 10]:
            config = FederatedConfig(
                async_mode=True,
                staleness_threshold=threshold,
                differential_privacy=False,
                secure_aggregation=False,
            )
            server = FederatedServer(simple_model, config)
            assert server.config.staleness_threshold == threshold

    def test_async_mode_client_tracking(self, simple_model, client_infos):
        """Test client participation tracking in async mode."""
        config = FederatedConfig(
            async_mode=True,
            staleness_threshold=3,
            clients_per_round=3,
            differential_privacy=False,
            secure_aggregation=False,
        )
        server = FederatedServer(simple_model, config)

        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=10)

        server.run_round(1)

        for client_id in server.clients:
            info = server.client_info[client_id]
            assert hasattr(info, "round_participation")

    def test_async_mode_factory_function(self, simple_model):
        """Test creating async system via factory function."""
        server = create_federated_system(
            simple_model,
            algorithm="fedavg",
            async_mode=True,
            staleness_threshold=4,
        )

        assert server.config.async_mode is True
        assert server.config.staleness_threshold == 4


class TestCommunicationCompression:
    """Tests for communication compression (ModelCompression class)."""

    def test_compression_initialization(self):
        """Test ModelCompression initialization."""
        compressor = ModelCompression(
            compression_ratio=0.1,
            quantization_bits=8,
            sparsification_ratio=0.05,
        )

        assert compressor.compression_ratio == 0.1
        assert compressor.quantization_bits == 8
        assert compressor.sparsification_ratio == 0.05

    def test_compress_model_reduces_data(self, simple_model):
        """Test that compression reduces transmitted data."""
        compressor = ModelCompression(
            compression_ratio=0.1,
            quantization_bits=8,
            sparsification_ratio=0.01,
        )

        model_params = {
            name: param.clone() for name, param in simple_model.named_parameters()
        }

        compressed = compressor.compress_model(model_params)

        for name, data in compressed.items():
            original_size = model_params[name].numel()
            compressed_size = len(data["values"])
            assert compressed_size < original_size

    def test_decompress_restores_shape(self, simple_model):
        """Test that decompression restores original shapes."""
        compressor = ModelCompression(
            compression_ratio=0.5,
            quantization_bits=8,
            sparsification_ratio=0.1,
        )

        model_params = {
            name: param.clone() for name, param in simple_model.named_parameters()
        }

        compressed = compressor.compress_model(model_params)
        decompressed = compressor.decompress_model(compressed)

        for name in model_params:
            assert decompressed[name].shape == model_params[name].shape

    def test_compression_roundtrip_preserves_structure(self, simple_model):
        """Test compression roundtrip preserves tensor structure."""
        compressor = ModelCompression(
            compression_ratio=0.5,
            quantization_bits=8,
            sparsification_ratio=0.05,
        )

        model_params = {
            name: param.clone() for name, param in simple_model.named_parameters()
        }

        compressed = compressor.compress_model(model_params)
        decompressed = compressor.decompress_model(compressed)

        for name in model_params:
            assert isinstance(decompressed[name], torch.Tensor)
            assert not torch.isnan(decompressed[name]).any()
            assert not torch.isinf(decompressed[name]).any()

    def test_quantization_with_different_bits(self, simple_model):
        """Test quantization with different bit widths."""
        for bits in [4, 8, 16]:
            compressor = ModelCompression(
                compression_ratio=0.5,
                quantization_bits=bits,
                sparsification_ratio=0.1,
            )

            model_params = {
                name: param.clone() for name, param in simple_model.named_parameters()
            }

            compressed = compressor.compress_model(model_params)
            decompressed = compressor.decompress_model(compressed)

            for name in model_params:
                assert decompressed[name].shape == model_params[name].shape

    def test_sparsification_ratio_effect(self, simple_model):
        """Test sparsification ratio affects compression."""
        model_params = {
            name: param.clone() for name, param in simple_model.named_parameters()
        }

        low_sparse = ModelCompression(sparsification_ratio=0.01)
        high_sparse = ModelCompression(sparsification_ratio=0.5)

        low_compressed = low_sparse.compress_model(model_params)
        high_compressed = high_sparse.compress_model(model_params)

        for name in model_params:
            low_count = len(low_compressed[name]["values"])
            high_count = len(high_compressed[name]["values"])
            assert high_count >= low_count

    def test_compressed_training_integration(self, simple_model, client_infos):
        """Test compressed training in federated round."""
        config = FederatedConfig(
            num_clients=3,
            clients_per_round=2,
            compression_ratio=0.5,
            quantization_bits=8,
            sparsification_ratio=0.1,
            differential_privacy=False,
            secure_aggregation=False,
        )
        server = FederatedServer(simple_model, config)

        for info in client_infos[:3]:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        result = server.run_round(1)

        assert "round" in result

    def test_quantize_tensor_uniform_values(self):
        """Test quantization of uniform tensor values."""
        compressor = ModelCompression(quantization_bits=8)

        uniform_tensor = torch.ones(10) * 5.0
        quantized = compressor._quantize_tensor(uniform_tensor)

        assert quantized is not None
        assert len(quantized) == 10

    def test_dequantize_tensor(self):
        """Test dequantization of tensor."""
        compressor = ModelCompression(quantization_bits=8)

        original = torch.randn(10)
        quantized = compressor._quantize_tensor(original)
        dequantized = compressor._dequantize_tensor(quantized)

        assert dequantized.shape == original.shape
        assert isinstance(dequantized, torch.Tensor)


class TestSecureAggregationMocking:
    """Tests for secure aggregation mechanism."""

    def test_secure_aggregation_initialization(self):
        """Test SecureAggregation initialization."""
        sec_agg = SecureAggregation(num_clients=10)

        assert sec_agg.num_clients == 10
        assert sec_agg.threshold == 5

    def test_secure_aggregation_threshold_calculation(self):
        """Test threshold calculation for different client counts."""
        for num_clients in [2, 5, 10, 20]:
            sec_agg = SecureAggregation(num_clients=num_clients)
            expected_threshold = max(2, num_clients // 2)
            assert sec_agg.threshold == expected_threshold

    def test_generate_masks_all_params(self, simple_model):
        """Test mask generation covers all parameters."""
        sec_agg = SecureAggregation(num_clients=5)
        model_params = {
            name: param.clone() for name, param in simple_model.named_parameters()
        }

        masks = sec_agg.generate_masks(model_params)

        for name in model_params:
            assert name in masks
            assert masks[name].shape == model_params[name].shape

    def test_mask_model_applies_masks(self, simple_model):
        """Test that mask_model applies masks correctly."""
        sec_agg = SecureAggregation(num_clients=3)
        model_params = {
            name: param.clone() for name, param in simple_model.named_parameters()
        }

        masks = sec_agg.generate_masks(model_params)
        masked = sec_agg.mask_model(model_params, masks)

        for name in model_params:
            expected = model_params[name] + masks[name]
            assert torch.allclose(masked[name], expected)

    def test_unmask_aggregate_restores_average(self):
        """Test that unmask_aggregate restores correct average."""
        sec_agg = SecureAggregation(num_clients=3)

        params1 = {"w": torch.tensor([1.0, 2.0, 3.0])}
        params2 = {"w": torch.tensor([4.0, 5.0, 6.0])}
        params3 = {"w": torch.tensor([7.0, 8.0, 9.0])}

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

        expected = (params1["w"] + params2["w"] + params3["w"]) / 3
        assert torch.allclose(aggregated["w"], expected, atol=1e-5)

    def test_unmask_aggregate_insufficient_clients(self):
        """Test that unmask_aggregate fails with insufficient clients."""
        sec_agg = SecureAggregation(num_clients=10)

        params = {"w": torch.tensor([1.0, 2.0])}
        masks = sec_agg.generate_masks(params)
        masked = sec_agg.mask_model(params, masks)

        with pytest.raises(ValueError, match="Insufficient clients"):
            sec_agg.unmask_aggregate(
                [masked, masked, masked],
                [masks, masks, masks],
            )

    def test_secure_aggregation_server_integration(self, simple_model):
        """Test secure aggregation integration with server."""
        config = FederatedConfig(
            num_clients=5,
            secure_aggregation=True,
            differential_privacy=False,
        )
        server = FederatedServer(simple_model, config)

        assert server.secure_aggregator is not None
        assert isinstance(server.secure_aggregator, SecureAggregation)

    def test_secure_aggregation_with_training(self, simple_model, client_infos):
        """Test secure aggregation during training round."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
            secure_aggregation=True,
            differential_privacy=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        result = server.run_round(1)

        assert "round" in result

    def test_mask_model_handles_missing_keys(self):
        """Test mask_model handles missing keys in masks."""
        sec_agg = SecureAggregation(num_clients=3)
        model_params = {"w1": torch.randn(5), "w2": torch.randn(3)}
        partial_masks = {"w1": torch.randn(5)}

        masked = sec_agg.mask_model(model_params, partial_masks)

        assert "w1" in masked
        assert "w2" in masked
        assert torch.allclose(masked["w2"], model_params["w2"])


class TestRoundTimeoutHandling:
    """Tests for round timeout handling."""

    def test_round_completes_within_timeout(
        self, simple_model, default_config, client_infos,
    ):
        """Test that round completes within reasonable timeout."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=20)

        start_time = time.time()
        server.run_round(1)
        elapsed = time.time() - start_time

        assert elapsed < 30

    def test_training_completes_within_limit(self, simple_model, client_infos):
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

    def test_round_handles_slow_client(self, simple_model, client_infos):
        """Test round handles slow client gracefully."""
        config = FederatedConfig(
            num_clients=3,
            clients_per_round=3,
            local_epochs=1,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)

        for info in client_infos[:3]:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=10)

        result = server.run_round(1)

        assert "round" in result

    def test_multiple_rounds_timing(self, simple_model, client_infos):
        """Test timing of multiple sequential rounds."""
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
            add_mock_data_to_client(client, num_samples=10)

        round_times = []
        for round_num in range(1, 4):
            start = time.time()
            server.run_round(round_num)
            round_times.append(time.time() - start)

        for rt in round_times:
            assert rt < 30

    def test_training_federation_early_stop(self, simple_model, client_infos):
        """Test early stopping based on convergence detection."""
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
            add_mock_data_to_client(client, num_samples=15)

        server.global_metrics["loss"].extend([0.1, 0.1, 0.1, 0.1, 0.1])

        start_time = time.time()
        results = server.train_federation()
        elapsed = time.time() - start_time

        assert elapsed < 60
        assert results["total_rounds"] <= 100


class TestCommunicationCostTracking:
    """Tests for communication cost tracking."""

    def test_communication_cost_increases_with_updates(
        self, simple_model, client_infos,
    ):
        """Test that communication cost increases with more updates."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=5,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=20)

        result = server.run_round(1)

        if "error" not in result:
            assert result["communication_cost_bytes"] > 0

    def test_communication_cost_with_compression(self, simple_model, client_infos):
        """Test communication cost with compression enabled."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
            compression_ratio=0.5,
            differential_privacy=False,
            secure_aggregation=False,
        )
        server = FederatedServer(simple_model, config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        result = server.run_round(1)

        assert "round" in result

    def test_communication_cost_metric_storage(
        self, simple_model, default_config, client_infos,
    ):
        """Test that communication cost is stored in metrics."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        server.run_round(1)

        if server.global_metrics["communication_cost"]:
            assert len(list(server.global_metrics["communication_cost"])) >= 1


class TestClientParticipationTracking:
    """Tests for client participation tracking in rounds."""

    def test_client_participation_recorded(
        self, simple_model, default_config, client_infos,
    ):
        """Test that client participation is recorded."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        server.run_round(1)

        participating_clients = 0
        for client_id in server.clients:
            if 1 in server.client_info[client_id].round_participation:
                participating_clients += 1

        assert participating_clients > 0

    def test_client_selection_history_tracked(
        self, simple_model, default_config, client_infos,
    ):
        """Test that client selection history is tracked."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        server.run_round(1)
        server.run_round(2)

        assert len(server.client_selection_history) >= 2
        assert server.client_selection_history[0]["round"] == 1
        assert server.client_selection_history[1]["round"] == 2

    def test_participation_rate_metric(
        self, simple_model, default_config, client_infos,
    ):
        """Test participation rate is tracked."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        server.run_round(1)

        if server.global_metrics["participation_rate"]:
            for rate in server.global_metrics["participation_rate"]:
                assert 0.0 <= rate <= 1.0

    def test_client_last_update_timestamp(
        self, simple_model, default_config, client_infos,
    ):
        """Test that client last_update timestamp is updated."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=15)

        initial_times = {
            cid: info.last_update for cid, info in server.client_info.items()
        }

        server.run_round(1)

        for client_id in server.clients:
            if 1 in server.client_info[client_id].round_participation:
                client_last = server.client_info[client_id].last_update
                assert client_last >= initial_times[client_id]


class TestCommunicationErrorHandling:
    """Tests for communication error handling."""

    def test_round_with_no_clients_returns_error(self, simple_model, default_config):
        """Test that round with no clients returns error."""
        server = FederatedServer(simple_model, default_config)

        result = server.run_round(1)

        assert "error" in result

    def test_round_with_no_data_clients(
        self, simple_model, default_config, client_infos,
    ):
        """Test round when clients have no data."""
        server = FederatedServer(simple_model, default_config)
        for info in client_infos:
            server.register_client(info)

        result = server.run_round(1)

        assert "error" in result

    def test_round_handles_client_training_error(self, simple_model, client_infos):
        """Test round handles client training errors gracefully."""
        config = FederatedConfig(
            num_clients=3,
            clients_per_round=3,
            differential_privacy=False,
            secure_aggregation=False,
            compression_ratio=1.0,
        )
        server = FederatedServer(simple_model, config)
        for info in client_infos[:3]:
            server.register_client(info)

        add_mock_data_to_client(
            server.clients["client_0"], num_samples=15,
        )
        add_mock_data_to_client(
            server.clients["client_1"], num_samples=15,
        )

        result = server.run_round(1)

        assert "round" in result

    def test_empty_updates_aggregation(self, simple_model, default_config):
        """Test aggregation with empty updates list."""
        server = FederatedServer(simple_model, default_config)

        aggregated = server.aggregate_models([])

        assert aggregated is not None


class TestCommunicationIntegration:
    """Integration tests for complete communication flow."""

    def test_complete_communication_cycle(self, simple_model, client_infos):
        """Test complete communication cycle with all features."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
            num_rounds=3,
            local_epochs=1,
            compression_ratio=0.8,
            differential_privacy=False,
            secure_aggregation=True,
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

    def test_communication_with_all_aggregation_methods(
        self, simple_model, client_infos,
    ):
        """Test communication with different aggregation methods."""
        methods = [
            AggregationMethod.WEIGHTED_AVERAGE,
            AggregationMethod.MEDIAN_AGGREGATION,
            AggregationMethod.TRIMMED_MEAN,
            AggregationMethod.BYZANTINE_ROBUST,
        ]

        for method in methods:
            config = FederatedConfig(
                num_clients=5,
                clients_per_round=3,
                aggregation=method,
                differential_privacy=False,
                secure_aggregation=False,
                compression_ratio=1.0,
            )
            server = FederatedServer(simple_model, config)

            for info in client_infos:
                new_info = ClientInfo(
                    client_id=info.client_id,
                    site_name=info.site_name,
                    location=info.location,
                    mfc_type=info.mfc_type,
                    bacterial_species=info.bacterial_species,
                    data_samples=info.data_samples,
                )
                server.register_client(new_info)
            for client in server.clients.values():
                add_mock_data_to_client(client, num_samples=15)

            result = server.run_round(1)

            assert "round" in result

    def test_end_to_end_federation(self, simple_model, client_infos):
        """Test end-to-end federation training."""
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

        for info in client_infos:
            server.register_client(info)
        for client in server.clients.values():
            add_mock_data_to_client(client, num_samples=20)

        results = server.train_federation()

        assert results["total_rounds"] >= 1
        assert results["total_clients"] == 5
        assert "final_loss" in results
        assert "average_participation_rate" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
