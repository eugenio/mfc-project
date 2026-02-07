"""Tests for Federated Learning Controller Core.

US-014: Test Federated Learning Core
Target: 50%+ coverage for core classes

Tests cover:
- FederatedLearningController initialization
- Client model creation and registration
- Local training step
- Model weight serialization/deserialization
- Client selection strategies
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

# Add src to path AFTER torch imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from federated_learning_controller import (
    AggregationMethod,
    ClientInfo,
    ClientSelectionStrategy,
    DifferentialPrivacy,
    FederatedAlgorithm,
    FederatedClient,
    FederatedConfig,
    FederatedServer,
    ModelCompression,
    SecureAggregation,
    _get_model_size,
    _select_device,
    create_federated_system,
)

# =============================================================================
# Mock Classes
# =============================================================================


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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_model() -> nn.Module:
    """Create a simple neural network for testing."""
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
    )


@pytest.fixture
def default_config() -> FederatedConfig:
    """Create default federated config."""
    return FederatedConfig(
        num_clients=5,
        clients_per_round=3,
        num_rounds=10,
        local_epochs=2,
        algorithm=FederatedAlgorithm.FEDAVG,
        aggregation=AggregationMethod.WEIGHTED_AVERAGE,
        client_selection=ClientSelectionStrategy.RANDOM,
        differential_privacy=False,
        secure_aggregation=False,
        compression_ratio=1.0,
    )


@pytest.fixture
def dp_config() -> FederatedConfig:
    """Create config with differential privacy."""
    return FederatedConfig(
        num_clients=5,
        clients_per_round=3,
        num_rounds=10,
        local_epochs=2,
        differential_privacy=True,
        noise_multiplier=0.5,
        max_grad_norm=1.0,
        secure_aggregation=False,
        compression_ratio=1.0,
    )


@pytest.fixture
def compression_config() -> FederatedConfig:
    """Create config with compression."""
    return FederatedConfig(
        num_clients=5,
        clients_per_round=3,
        num_rounds=10,
        local_epochs=2,
        differential_privacy=False,
        secure_aggregation=False,
        compression_ratio=0.5,
        quantization_bits=8,
        sparsification_ratio=0.1,
    )


@pytest.fixture
def client_info() -> ClientInfo:
    """Create test client info."""
    return ClientInfo(
        client_id="test_client_1",
        site_name="Test Site",
        location="Test Location",
        mfc_type="dual_chamber",
        bacterial_species=MockBacterialSpecies.GEOBACTER,
        data_samples=100,
        computation_power=1.0,
        communication_bandwidth=10.0,
        reliability_score=0.9,
        is_active=True,
    )


@pytest.fixture
def multiple_client_infos() -> list[ClientInfo]:
    """Create multiple test client infos."""
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
                site_name=f"Site {i}",
                location=f"Location {i}",
                mfc_type="dual_chamber",
                bacterial_species=species_list[i],
                data_samples=np.random.randint(50, 200),
                computation_power=np.random.uniform(0.5, 2.0),
                communication_bandwidth=np.random.uniform(1.0, 10.0),
                reliability_score=np.random.uniform(0.7, 1.0),
                is_active=True,
            ),
        )
    return clients


# =============================================================================
# Test FederatedConfig
# =============================================================================


class TestFederatedConfig:
    """Tests for FederatedConfig dataclass."""

    def test_default_initialization(self):
        """Test default config initialization."""
        config = FederatedConfig()
        assert config.num_clients == 10
        assert config.clients_per_round == 5
        assert config.num_rounds == 100
        assert config.local_epochs == 5
        assert config.algorithm == FederatedAlgorithm.FEDAVG
        assert config.aggregation == AggregationMethod.WEIGHTED_AVERAGE
        assert config.client_selection == ClientSelectionStrategy.RANDOM

    def test_custom_initialization(self):
        """Test custom config initialization."""
        config = FederatedConfig(
            num_clients=20,
            clients_per_round=10,
            num_rounds=50,
            local_epochs=3,
            algorithm=FederatedAlgorithm.FEDPROX,
            aggregation=AggregationMethod.MEDIAN_AGGREGATION,
            client_selection=ClientSelectionStrategy.PERFORMANCE_BASED,
        )
        assert config.num_clients == 20
        assert config.clients_per_round == 10
        assert config.num_rounds == 50
        assert config.local_epochs == 3
        assert config.algorithm == FederatedAlgorithm.FEDPROX
        assert config.aggregation == AggregationMethod.MEDIAN_AGGREGATION
        assert config.client_selection == ClientSelectionStrategy.PERFORMANCE_BASED

    def test_privacy_config(self):
        """Test privacy-related config."""
        config = FederatedConfig(
            differential_privacy=True,
            noise_multiplier=2.0,
            max_grad_norm=0.5,
            secure_aggregation=True,
        )
        assert config.differential_privacy is True
        assert config.noise_multiplier == 2.0
        assert config.max_grad_norm == 0.5
        assert config.secure_aggregation is True

    def test_compression_config(self):
        """Test compression-related config."""
        config = FederatedConfig(
            compression_ratio=0.1,
            quantization_bits=4,
            sparsification_ratio=0.05,
        )
        assert config.compression_ratio == 0.1
        assert config.quantization_bits == 4
        assert config.sparsification_ratio == 0.05

    def test_personalization_layers_default(self):
        """Test default personalization layers."""
        config = FederatedConfig()
        assert config.personalization_layers == ["output_layer"]

    def test_learning_parameters(self):
        """Test learning rate parameters."""
        config = FederatedConfig(
            global_lr=0.5,
            local_lr=0.001,
            momentum=0.95,
            weight_decay=1e-3,
        )
        assert config.global_lr == 0.5
        assert config.local_lr == 0.001
        assert config.momentum == 0.95
        assert config.weight_decay == 1e-3


# =============================================================================
# Test ClientInfo
# =============================================================================


class TestClientInfo:
    """Tests for ClientInfo dataclass."""

    def test_initialization(self, client_info):
        """Test client info initialization."""
        assert client_info.client_id == "test_client_1"
        assert client_info.site_name == "Test Site"
        assert client_info.location == "Test Location"
        assert client_info.mfc_type == "dual_chamber"
        assert client_info.data_samples == 100
        assert client_info.is_active is True

    def test_default_values(self):
        """Test default values."""
        info = ClientInfo(
            client_id="c1",
            site_name="S1",
            location="L1",
            mfc_type="single",
            bacterial_species=MockBacterialSpecies.MIXED,
        )
        assert info.data_samples == 0
        assert info.computation_power == 1.0
        assert info.communication_bandwidth == 1.0
        assert info.reliability_score == 1.0
        assert info.is_active is True
        assert info.local_accuracy == 0.0
        assert info.local_loss == float("inf")

    def test_mutable_defaults(self):
        """Test that mutable defaults are independent."""
        info1 = ClientInfo(
            client_id="c1",
            site_name="S1",
            location="L1",
            mfc_type="single",
            bacterial_species=MockBacterialSpecies.MIXED,
        )
        info2 = ClientInfo(
            client_id="c2",
            site_name="S2",
            location="L2",
            mfc_type="dual",
            bacterial_species=MockBacterialSpecies.GEOBACTER,
        )
        info1.round_participation.append(1)
        assert 1 not in info2.round_participation


# =============================================================================
# Test Enums
# =============================================================================


class TestEnums:
    """Tests for federated learning enums."""

    def test_federated_algorithm_values(self):
        """Test FederatedAlgorithm enum values."""
        assert FederatedAlgorithm.FEDAVG.value == "federated_averaging"
        assert FederatedAlgorithm.FEDPROX.value == "federated_proximal"
        assert FederatedAlgorithm.FEDPER.value == "federated_personalization"
        assert FederatedAlgorithm.SCAFFOLD.value == "scaffold"
        assert FederatedAlgorithm.FEDNOVA.value == "fed_nova"
        assert FederatedAlgorithm.MOON.value == "model_contrastive_learning"

    def test_aggregation_method_values(self):
        """Test AggregationMethod enum values."""
        assert AggregationMethod.WEIGHTED_AVERAGE.value == "weighted_average"
        assert AggregationMethod.MEDIAN_AGGREGATION.value == "median"
        assert AggregationMethod.TRIMMED_MEAN.value == "trimmed_mean"
        assert AggregationMethod.BYZANTINE_ROBUST.value == "byzantine_robust"

    def test_client_selection_strategy_values(self):
        """Test ClientSelectionStrategy enum values."""
        assert ClientSelectionStrategy.RANDOM.value == "random"
        assert ClientSelectionStrategy.CYCLIC.value == "cyclic"
        assert ClientSelectionStrategy.PERFORMANCE_BASED.value == "performance_based"
        assert ClientSelectionStrategy.RESOURCE_AWARE.value == "resource_aware"


# =============================================================================
# Test Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_select_device_cpu(self):
        """Test CPU device selection."""
        device = _select_device("cpu")
        assert device == torch.device("cpu")

    def test_select_device_auto(self):
        """Test auto device selection."""
        device = _select_device("auto")
        assert device in [torch.device("cpu"), torch.device("cuda")]

    def test_select_device_none(self):
        """Test None device selection defaults to auto."""
        device = _select_device(None)
        assert device in [torch.device("cpu"), torch.device("cuda")]

    def test_get_model_size(self, simple_model):
        """Test model size calculation."""
        size_info = _get_model_size(simple_model)
        assert "total_parameters" in size_info
        assert "trainable_parameters" in size_info
        assert "memory_mb" in size_info
        assert size_info["total_parameters"] > 0
        assert size_info["trainable_parameters"] > 0
        assert size_info["memory_mb"] > 0

    def test_get_model_size_accuracy(self, simple_model):
        """Test model size calculation accuracy."""
        size_info = _get_model_size(simple_model)
        # Manual calculation: Linear(10, 32) = 10*32 + 32 = 352
        # Linear(32, 5) = 32*5 + 5 = 165
        # Total = 517
        expected_params = 10 * 32 + 32 + 32 * 5 + 5
        assert size_info["total_parameters"] == expected_params


# =============================================================================
# Test DifferentialPrivacy
# =============================================================================


class TestDifferentialPrivacy:
    """Tests for DifferentialPrivacy class."""

    def test_initialization(self):
        """Test DP initialization."""
        dp = DifferentialPrivacy(noise_multiplier=1.0, max_grad_norm=1.0)
        assert dp.noise_multiplier == 1.0
        assert dp.max_grad_norm == 1.0

    def test_initialization_custom(self):
        """Test DP initialization with custom values."""
        dp = DifferentialPrivacy(noise_multiplier=2.0, max_grad_norm=0.5)
        assert dp.noise_multiplier == 2.0
        assert dp.max_grad_norm == 0.5

    def test_clip_gradients(self, simple_model):
        """Test gradient clipping."""
        dp = DifferentialPrivacy(noise_multiplier=1.0, max_grad_norm=1.0)

        # Create some gradients
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()

        # Clip gradients
        total_norm = dp.clip_gradients(simple_model)
        assert isinstance(total_norm, float)
        assert total_norm >= 0

    def test_add_noise(self, simple_model):
        """Test noise addition to model."""
        dp = DifferentialPrivacy(noise_multiplier=1.0, max_grad_norm=1.0)
        device = torch.device("cpu")

        # Create gradients
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()

        # Store original gradients
        original_grads = {
            name: param.grad.clone() if param.grad is not None else None
            for name, param in simple_model.named_parameters()
        }

        # Add noise
        dp.add_noise(simple_model, device)

        # Check gradients changed
        for name, param in simple_model.named_parameters():
            if param.grad is not None and original_grads[name] is not None:
                # Gradients should be different after adding noise
                assert not torch.allclose(param.grad, original_grads[name], atol=1e-6)

    def test_compute_privacy_budget(self):
        """Test privacy budget computation."""
        dp = DifferentialPrivacy(noise_multiplier=1.0, max_grad_norm=1.0)
        epsilon = dp.compute_privacy_budget(
            steps=100,
            sampling_rate=0.01,
            target_delta=1e-5,
        )
        assert isinstance(epsilon, float)
        assert epsilon >= 0


# =============================================================================
# Test SecureAggregation
# =============================================================================


class TestSecureAggregation:
    """Tests for SecureAggregation class."""

    def test_initialization(self):
        """Test secure aggregation initialization."""
        sa = SecureAggregation(num_clients=10)
        assert sa.num_clients == 10
        assert sa.threshold == 5  # max(2, 10 // 2)

    def test_initialization_small(self):
        """Test secure aggregation with small client count."""
        sa = SecureAggregation(num_clients=3)
        assert sa.num_clients == 3
        assert sa.threshold == 2  # max(2, 3 // 2)

    def test_generate_masks(self, simple_model):
        """Test mask generation."""
        sa = SecureAggregation(num_clients=5)
        model_params = {
            name: param.clone() for name, param in simple_model.named_parameters()
        }
        masks = sa.generate_masks(model_params)

        assert len(masks) == len(model_params)
        for name in model_params:
            assert name in masks
            assert masks[name].shape == model_params[name].shape

    def test_mask_model(self, simple_model):
        """Test model masking."""
        sa = SecureAggregation(num_clients=5)
        model_params = {
            name: param.clone() for name, param in simple_model.named_parameters()
        }
        masks = sa.generate_masks(model_params)
        masked_params = sa.mask_model(model_params, masks)

        for name in model_params:
            expected = model_params[name] + masks[name]
            assert torch.allclose(masked_params[name], expected)

    def test_unmask_aggregate(self, simple_model):
        """Test secure unmasking and aggregation."""
        sa = SecureAggregation(num_clients=5)
        model_params = {
            name: param.clone() for name, param in simple_model.named_parameters()
        }

        # Create multiple masked models
        masked_models = []
        all_masks = []
        for _ in range(3):
            masks = sa.generate_masks(model_params)
            masked = sa.mask_model(model_params, masks)
            masked_models.append(masked)
            all_masks.append(masks)

        # Unmask and aggregate
        aggregated = sa.unmask_aggregate(masked_models, all_masks)

        assert len(aggregated) == len(model_params)
        for name in model_params:
            assert aggregated[name].shape == model_params[name].shape

    def test_unmask_aggregate_insufficient_clients(self, simple_model):
        """Test that aggregation fails with insufficient clients."""
        sa = SecureAggregation(num_clients=10)  # threshold = 5
        model_params = {
            name: param.clone() for name, param in simple_model.named_parameters()
        }

        # Only 2 clients (below threshold of 5)
        masked_models = []
        all_masks = []
        for _ in range(2):
            masks = sa.generate_masks(model_params)
            masked = sa.mask_model(model_params, masks)
            masked_models.append(masked)
            all_masks.append(masks)

        with pytest.raises(ValueError, match="Insufficient clients"):
            sa.unmask_aggregate(masked_models, all_masks)


# =============================================================================
# Test ModelCompression
# =============================================================================


class TestModelCompression:
    """Tests for ModelCompression class."""

    def test_initialization(self):
        """Test compression initialization."""
        mc = ModelCompression(
            compression_ratio=0.1,
            quantization_bits=8,
            sparsification_ratio=0.01,
        )
        assert mc.compression_ratio == 0.1
        assert mc.quantization_bits == 8
        assert mc.sparsification_ratio == 0.01

    def test_compress_model(self, simple_model):
        """Test model compression."""
        mc = ModelCompression(
            compression_ratio=0.1,
            quantization_bits=8,
            sparsification_ratio=0.1,
        )
        model_params = {
            name: param.clone() for name, param in simple_model.named_parameters()
        }
        compressed = mc.compress_model(model_params)

        assert len(compressed) == len(model_params)
        for name in model_params:
            assert "shape" in compressed[name]
            assert "indices" in compressed[name]
            assert "values" in compressed[name]
            assert "original_size" in compressed[name]

    def test_decompress_model(self, simple_model):
        """Test model decompression."""
        mc = ModelCompression(
            compression_ratio=0.1,
            quantization_bits=8,
            sparsification_ratio=0.1,
        )
        model_params = {
            name: param.clone() for name, param in simple_model.named_parameters()
        }
        compressed = mc.compress_model(model_params)
        decompressed = mc.decompress_model(compressed)

        assert len(decompressed) == len(model_params)
        for name in model_params:
            assert decompressed[name].shape == model_params[name].shape

    def test_compress_decompress_preserves_shape(self, simple_model):
        """Test that compression/decompression preserves tensor shapes."""
        mc = ModelCompression(
            compression_ratio=0.5,
            quantization_bits=8,
            sparsification_ratio=0.5,
        )
        model_params = {
            name: param.clone() for name, param in simple_model.named_parameters()
        }
        compressed = mc.compress_model(model_params)
        decompressed = mc.decompress_model(compressed)

        for name in model_params:
            original_shape = model_params[name].shape
            decompressed_shape = decompressed[name].shape
            assert original_shape == decompressed_shape

    def test_quantize_tensor(self):
        """Test tensor quantization."""
        mc = ModelCompression(quantization_bits=8)
        tensor = torch.randn(100)
        quantized = mc._quantize_tensor(tensor)

        assert quantized.dtype == np.uint8
        assert quantized.shape == tensor.shape
        assert quantized.max() <= 255
        assert quantized.min() >= 0

    def test_quantize_constant_tensor(self):
        """Test quantization of constant tensor."""
        mc = ModelCompression(quantization_bits=8)
        tensor = torch.ones(100) * 5.0
        quantized = mc._quantize_tensor(tensor)

        # Constant tensor should quantize to zeros
        assert np.allclose(quantized, 0)

    def test_dequantize_tensor(self):
        """Test tensor dequantization."""
        mc = ModelCompression(quantization_bits=8)
        quantized = np.array([0, 128, 255], dtype=np.uint8)
        dequantized = mc._dequantize_tensor(quantized)

        assert isinstance(dequantized, torch.Tensor)
        assert dequantized.shape == (3,)


# =============================================================================
# Test FederatedClient
# =============================================================================


class TestFederatedClient:
    """Tests for FederatedClient class."""

    def test_initialization(self, simple_model, default_config, client_info):
        """Test client initialization."""
        client = FederatedClient(client_info, simple_model, default_config)

        assert client.client_info == client_info
        assert client.config == default_config
        assert client.local_model is not None
        assert client.optimizer is not None
        assert len(client.local_data) == 0

    def test_initialization_with_dp(self, simple_model, dp_config, client_info):
        """Test client initialization with differential privacy."""
        client = FederatedClient(client_info, simple_model, dp_config)

        assert client.dp_mechanism is not None
        assert client.dp_mechanism.noise_multiplier == dp_config.noise_multiplier
        assert client.dp_mechanism.max_grad_norm == dp_config.max_grad_norm

    def test_initialization_without_dp(self, simple_model, default_config, client_info):
        """Test client initialization without differential privacy."""
        client = FederatedClient(client_info, simple_model, default_config)
        assert client.dp_mechanism is None

    def test_get_model_size(self, simple_model, default_config, client_info):
        """Test getting model size."""
        client = FederatedClient(client_info, simple_model, default_config)
        size_info = client.get_model_size()

        assert "total_parameters" in size_info
        assert "trainable_parameters" in size_info
        assert "memory_mb" in size_info

    def test_model_deep_copy(self, simple_model, default_config, client_info):
        """Test that client model is a deep copy."""
        client = FederatedClient(client_info, simple_model, default_config)

        # Modify original model
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(1.0)

        # Client model should be unchanged
        for orig_param, client_param in zip(
            simple_model.parameters(), client.local_model.parameters(), strict=False,
        ):
            assert not torch.allclose(orig_param, client_param)


# =============================================================================
# Test FederatedServer
# =============================================================================


class TestFederatedServer:
    """Tests for FederatedServer class."""

    def test_initialization(self, simple_model, default_config):
        """Test server initialization."""
        server = FederatedServer(simple_model, default_config)

        assert server.global_model is not None
        assert server.config == default_config
        assert len(server.clients) == 0
        assert len(server.client_info) == 0
        assert server.current_round == 0

    def test_initialization_with_secure_aggregation(self, simple_model):
        """Test server initialization with secure aggregation."""
        config = FederatedConfig(
            num_clients=10,
            secure_aggregation=True,
        )
        server = FederatedServer(simple_model, config)

        assert server.secure_aggregator is not None
        assert server.secure_aggregator.num_clients == 10

    def test_initialization_without_secure_aggregation(
        self, simple_model, default_config,
    ):
        """Test server initialization without secure aggregation."""
        server = FederatedServer(simple_model, default_config)
        assert server.secure_aggregator is None

    def test_register_client(self, simple_model, default_config, client_info):
        """Test client registration."""
        server = FederatedServer(simple_model, default_config)
        success = server.register_client(client_info)

        assert success is True
        assert client_info.client_id in server.clients
        assert client_info.client_id in server.client_info

    def test_register_multiple_clients(
        self, simple_model, default_config, multiple_client_infos,
    ):
        """Test registering multiple clients."""
        server = FederatedServer(simple_model, default_config)

        for info in multiple_client_infos:
            success = server.register_client(info)
            assert success is True

        assert len(server.clients) == len(multiple_client_infos)
        assert len(server.client_info) == len(multiple_client_infos)

    def test_global_model_deep_copy(self, simple_model, default_config):
        """Test that global model is a deep copy."""
        server = FederatedServer(simple_model, default_config)

        # Modify original model
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(1.0)

        # Server model should be unchanged
        for orig_param, server_param in zip(
            simple_model.parameters(), server.global_model.parameters(), strict=False,
        ):
            assert not torch.allclose(orig_param, server_param)


# =============================================================================
# Test Client Selection Strategies
# =============================================================================


class TestClientSelection:
    """Tests for client selection strategies."""

    def _setup_server_with_clients(
        self, simple_model, config, client_infos,
    ) -> FederatedServer:
        """Helper to setup server with registered clients."""
        server = FederatedServer(simple_model, config)
        for info in client_infos:
            server.register_client(info)
        return server

    def _add_data_to_clients(self, server):
        """Helper to add dummy data to clients."""
        for client_id, client in server.clients.items():
            # Add some dummy local data (bypassing the full add_local_data)
            client.local_data = [
                {"features": torch.randn(10), "action": 0, "reward": 1.0},
            ]
            server.client_info[client_id].data_samples = 1

    def test_random_selection(self, simple_model, multiple_client_infos):
        """Test random client selection."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
            client_selection=ClientSelectionStrategy.RANDOM,
        )
        server = self._setup_server_with_clients(
            simple_model, config, multiple_client_infos,
        )
        self._add_data_to_clients(server)

        selected = server.select_clients(1)

        assert len(selected) == 3
        for client_id in selected:
            assert client_id in server.clients

    def test_cyclic_selection(self, simple_model, multiple_client_infos):
        """Test cyclic client selection."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=2,
            client_selection=ClientSelectionStrategy.CYCLIC,
        )
        server = self._setup_server_with_clients(
            simple_model, config, multiple_client_infos,
        )
        self._add_data_to_clients(server)

        selected_round_1 = server.select_clients(0)
        selected_round_2 = server.select_clients(1)

        assert len(selected_round_1) == 2
        assert len(selected_round_2) == 2

    def test_performance_based_selection(self, simple_model, multiple_client_infos):
        """Test performance-based client selection."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=2,
            client_selection=ClientSelectionStrategy.PERFORMANCE_BASED,
        )
        server = self._setup_server_with_clients(
            simple_model, config, multiple_client_infos,
        )
        self._add_data_to_clients(server)

        # Set different losses for clients
        for i, (_client_id, info) in enumerate(server.client_info.items()):
            info.local_loss = float(i)  # Higher loss for later clients

        selected = server.select_clients(1)
        assert len(selected) == 2

    def test_resource_aware_selection(self, simple_model, multiple_client_infos):
        """Test resource-aware client selection."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=2,
            client_selection=ClientSelectionStrategy.RESOURCE_AWARE,
        )
        server = self._setup_server_with_clients(
            simple_model, config, multiple_client_infos,
        )
        self._add_data_to_clients(server)

        selected = server.select_clients(1)
        assert len(selected) == 2

    def test_selection_with_no_active_clients(self, simple_model, multiple_client_infos):
        """Test selection when no clients have data."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
        )
        server = self._setup_server_with_clients(
            simple_model, config, multiple_client_infos,
        )
        # Don't add any data to clients

        selected = server.select_clients(1)
        assert len(selected) == 0

    def test_selection_history(self, simple_model, multiple_client_infos):
        """Test that selection history is recorded."""
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
        )
        server = self._setup_server_with_clients(
            simple_model, config, multiple_client_infos,
        )
        self._add_data_to_clients(server)

        server.select_clients(1)
        server.select_clients(2)

        assert len(server.client_selection_history) == 2
        assert server.client_selection_history[0]["round"] == 1
        assert server.client_selection_history[1]["round"] == 2


# =============================================================================
# Test Aggregation Methods
# =============================================================================


class TestAggregation:
    """Tests for model aggregation methods."""

    def _create_mock_updates(self, simple_model, num_clients=3):
        """Create mock client updates."""
        updates = []
        for i in range(num_clients):
            model_params = {}
            for name, param in simple_model.named_parameters():
                # Slightly perturb parameters
                model_params[name] = param.clone() + torch.randn_like(param) * 0.1
            updates.append(
                {
                    "client_id": f"client_{i}",
                    "model_params": model_params,
                    "num_samples": 100 + i * 10,
                    "loss": 0.5 - i * 0.1,
                    "compression_used": False,
                },
            )
        return updates

    def test_weighted_average_aggregation(self, simple_model, default_config):
        """Test weighted average aggregation."""
        server = FederatedServer(simple_model, default_config)
        updates = self._create_mock_updates(simple_model)

        aggregated = server._weighted_average_aggregation(updates)

        assert len(aggregated) > 0
        for name, param in simple_model.named_parameters():
            assert name in aggregated
            assert aggregated[name].shape == param.shape

    def test_weighted_average_empty_updates(self, simple_model, default_config):
        """Test weighted average with empty updates."""
        server = FederatedServer(simple_model, default_config)

        aggregated = server._weighted_average_aggregation([])

        # Should return current global model params
        for name, param in server.global_model.named_parameters():
            assert name in aggregated
            assert torch.allclose(aggregated[name], param)

    def test_median_aggregation(self, simple_model):
        """Test median aggregation."""
        config = FederatedConfig(
            aggregation=AggregationMethod.MEDIAN_AGGREGATION,
        )
        server = FederatedServer(simple_model, config)
        updates = self._create_mock_updates(simple_model, num_clients=5)

        aggregated = server._median_aggregation(updates)

        for name, param in simple_model.named_parameters():
            assert name in aggregated
            assert aggregated[name].shape == param.shape

    def test_trimmed_mean_aggregation(self, simple_model):
        """Test trimmed mean aggregation."""
        config = FederatedConfig(
            aggregation=AggregationMethod.TRIMMED_MEAN,
        )
        server = FederatedServer(simple_model, config)
        updates = self._create_mock_updates(simple_model, num_clients=5)

        aggregated = server._trimmed_mean_aggregation(updates, trim_ratio=0.2)

        for name, param in simple_model.named_parameters():
            assert name in aggregated
            assert aggregated[name].shape == param.shape

    def test_byzantine_robust_aggregation(self, simple_model):
        """Test Byzantine-robust aggregation."""
        config = FederatedConfig(
            aggregation=AggregationMethod.BYZANTINE_ROBUST,
        )
        server = FederatedServer(simple_model, config)
        updates = self._create_mock_updates(simple_model, num_clients=5)

        aggregated = server._byzantine_robust_aggregation(updates)

        for name, param in simple_model.named_parameters():
            assert name in aggregated
            assert aggregated[name].shape == param.shape

    def test_aggregate_models_routing(self, simple_model):
        """Test that aggregate_models routes to correct method."""
        for method in [
            AggregationMethod.WEIGHTED_AVERAGE,
            AggregationMethod.MEDIAN_AGGREGATION,
            AggregationMethod.TRIMMED_MEAN,
            AggregationMethod.BYZANTINE_ROBUST,
        ]:
            config = FederatedConfig(aggregation=method)
            server = FederatedServer(simple_model, config)
            updates = self._create_mock_updates(simple_model, num_clients=5)

            aggregated = server.aggregate_models(updates)

            assert len(aggregated) > 0


# =============================================================================
# Test Factory Function
# =============================================================================


class TestCreateFederatedSystem:
    """Tests for create_federated_system factory function."""

    def test_default_creation(self, simple_model):
        """Test default system creation."""
        server = create_federated_system(simple_model)

        assert server is not None
        assert isinstance(server, FederatedServer)
        assert server.config.algorithm == FederatedAlgorithm.FEDAVG

    def test_custom_num_clients(self, simple_model):
        """Test creation with custom client count."""
        server = create_federated_system(simple_model, num_clients=10)

        assert server.config.num_clients == 10

    def test_algorithm_selection_fedavg(self, simple_model):
        """Test FedAvg algorithm selection."""
        server = create_federated_system(simple_model, algorithm="fedavg")
        assert server.config.algorithm == FederatedAlgorithm.FEDAVG

    def test_algorithm_selection_fedprox(self, simple_model):
        """Test FedProx algorithm selection."""
        server = create_federated_system(simple_model, algorithm="fedprox")
        assert server.config.algorithm == FederatedAlgorithm.FEDPROX

    def test_algorithm_selection_fedper(self, simple_model):
        """Test FedPer algorithm selection."""
        server = create_federated_system(simple_model, algorithm="fedper")
        assert server.config.algorithm == FederatedAlgorithm.FEDPER

    def test_algorithm_selection_scaffold(self, simple_model):
        """Test SCAFFOLD algorithm selection."""
        server = create_federated_system(simple_model, algorithm="scaffold")
        assert server.config.algorithm == FederatedAlgorithm.SCAFFOLD

    def test_unknown_algorithm_defaults_fedavg(self, simple_model):
        """Test unknown algorithm defaults to FedAvg."""
        server = create_federated_system(simple_model, algorithm="unknown")
        assert server.config.algorithm == FederatedAlgorithm.FEDAVG

    def test_additional_kwargs(self, simple_model):
        """Test additional kwargs are passed to config."""
        server = create_federated_system(
            simple_model,
            num_clients=10,
            algorithm="fedavg",
            clients_per_round=5,
            local_epochs=10,
            local_lr=0.001,
        )

        assert server.config.clients_per_round == 5
        assert server.config.local_epochs == 10
        assert server.config.local_lr == 0.001


# =============================================================================
# Test Weight Serialization
# =============================================================================


class TestWeightSerialization:
    """Tests for model weight serialization/deserialization."""

    def test_state_dict_serialization(self, simple_model):
        """Test state dict can be serialized."""
        state_dict = simple_model.state_dict()

        # Verify structure
        assert len(state_dict) > 0
        for _key, value in state_dict.items():
            assert isinstance(value, torch.Tensor)

    def test_state_dict_load(self, simple_model):
        """Test state dict can be loaded."""
        state_dict = simple_model.state_dict()

        # Create new model and load state
        new_model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )
        new_model.load_state_dict(state_dict)

        # Verify parameters match
        for (_n1, p1), (_n2, p2) in zip(
            simple_model.named_parameters(),
            new_model.named_parameters(),
            strict=False,
        ):
            assert torch.allclose(p1, p2)

    def test_param_extraction_for_aggregation(self, simple_model):
        """Test parameter extraction for aggregation."""
        model_params = {
            name: param.cpu().clone()
            for name, param in simple_model.named_parameters()
        }

        assert len(model_params) > 0
        for _name, param in model_params.items():
            assert isinstance(param, torch.Tensor)
            assert param.device == torch.device("cpu")

    def test_compression_serialization(self, simple_model):
        """Test compressed model serialization."""
        mc = ModelCompression(
            compression_ratio=0.5,
            quantization_bits=8,
            sparsification_ratio=0.1,
        )
        model_params = {
            name: param.clone() for name, param in simple_model.named_parameters()
        }

        compressed = mc.compress_model(model_params)

        # Verify compressed format is serializable
        import pickle

        serialized = pickle.dumps(compressed)
        deserialized = pickle.loads(serialized)

        assert len(deserialized) == len(compressed)


# =============================================================================
# Test Integration
# =============================================================================


class TestIntegration:
    """Integration tests for federated learning components."""

    def test_client_server_workflow(
        self, simple_model, default_config, multiple_client_infos,
    ):
        """Test basic client-server workflow."""
        # Create server
        server = FederatedServer(simple_model, default_config)

        # Register clients
        for info in multiple_client_infos:
            server.register_client(info)

        assert len(server.clients) == len(multiple_client_infos)

        # Get global parameters
        global_params = {
            name: param.cpu().clone()
            for name, param in server.global_model.named_parameters()
        }

        # Each client can receive global parameters
        for client in server.clients.values():
            client.local_model.load_state_dict(global_params)

    def test_aggregation_with_multiple_clients(
        self, simple_model, default_config, multiple_client_infos,
    ):
        """Test aggregation with multiple real clients."""
        server = FederatedServer(simple_model, default_config)

        # Register and create updates
        updates = []
        for _i, info in enumerate(multiple_client_infos):
            server.register_client(info)
            client = server.clients[info.client_id]

            # Get slightly modified parameters
            model_params = {
                name: param.cpu().clone() + torch.randn_like(param) * 0.01
                for name, param in client.local_model.named_parameters()
            }
            updates.append(
                {
                    "client_id": info.client_id,
                    "model_params": model_params,
                    "num_samples": info.data_samples,
                    "loss": 0.5,
                    "compression_used": False,
                },
            )

        # Aggregate
        aggregated = server.aggregate_models(updates)

        # Load aggregated params to global model
        server.global_model.load_state_dict(aggregated)

        # Verify global model is updated
        for name, _param in server.global_model.named_parameters():
            assert name in aggregated
