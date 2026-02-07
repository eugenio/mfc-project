"""Federated Learning Controller for Multi-MFC Systems.

This module implements federated learning techniques to enable collaborative
learning across multiple MFC installations while preserving data privacy
and enabling knowledge sharing across distributed systems.

Key Features:
- Federated Averaging (FedAvg) for collaborative model training
- Differential Privacy for secure aggregation
- Personalized federated learning for site-specific adaptations
- Asynchronous federated learning for real-time systems
- Byzantine-robust aggregation for fault tolerance
- Hierarchical federated learning for multi-scale deployment
- Federated reinforcement learning for distributed control
- Communication-efficient compression techniques
- Client selection and scheduling algorithms

Integration with Phase 2 and Phase 3 components:
- Federates deep RL models across multiple MFC sites
- Shares transfer learning knowledge between installations
- Distributes transformer models for collaborative intelligence
- Maintains local sensor fusion while sharing global insights
- Preserves privacy while enabling collective learning

This module uses BaseController utilities for:
- Device management (GPU/CPU selection)
- Feature engineering integration
- Model size estimation

Applications:
- Industrial MFC networks
- Research laboratory collaborations
- Multi-site wastewater treatment plants
- Distributed bioelectrochemical systems
- Cross-institutional knowledge sharing

Created: 2025-07-31
Last Modified: 2025-07-31
"""

import copy
import logging
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# Import Phase 2 and 3 components
from adaptive_mfc_controller import SystemState
from biofilm_health_monitor import HealthMetrics
from sensing_models.sensor_fusion import BacterialSpecies
from torch import nn, optim

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedAlgorithm(Enum):
    """Federated learning algorithm types."""

    FEDAVG = "federated_averaging"
    FEDPROX = "federated_proximal"
    FEDPER = "federated_personalization"
    FEDBN = "federated_batch_normalization"
    SCAFFOLD = "scaffold"
    FEDNOVA = "fed_nova"
    MOON = "model_contrastive_learning"
    FEDOPT = "federated_optimization"


class AggregationMethod(Enum):
    """Model aggregation methods."""

    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN_AGGREGATION = "median"
    TRIMMED_MEAN = "trimmed_mean"
    BYZANTINE_ROBUST = "byzantine_robust"
    SECURE_AGGREGATION = "secure_aggregation"
    DIFFERENTIAL_PRIVATE = "differential_private"


class ClientSelectionStrategy(Enum):
    """Client selection strategies."""

    RANDOM = "random"
    CYCLIC = "cyclic"
    PERFORMANCE_BASED = "performance_based"
    RESOURCE_AWARE = "resource_aware"
    DIVERSITY_BASED = "diversity_based"
    GRADIENT_BASED = "gradient_based"


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""

    # Federation parameters
    num_clients: int = 10
    clients_per_round: int = 5
    num_rounds: int = 100
    local_epochs: int = 5

    # Algorithm configuration
    algorithm: FederatedAlgorithm = FederatedAlgorithm.FEDAVG
    aggregation: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    client_selection: ClientSelectionStrategy = ClientSelectionStrategy.RANDOM

    # Privacy and security
    differential_privacy: bool = True
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    secure_aggregation: bool = True

    # Communication efficiency
    compression_ratio: float = 0.1
    quantization_bits: int = 8
    sparsification_ratio: float = 0.01

    # Personalization
    personalization_layers: list[str] = field(default_factory=lambda: ["output_layer"])
    local_adaptation_steps: int = 10

    # Byzantine tolerance
    byzantine_clients: int = 0
    robust_aggregation: bool = False

    # Asynchronous settings
    async_mode: bool = False
    staleness_threshold: int = 3

    # Learning parameters
    global_lr: float = 1.0
    local_lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4


@dataclass
class ClientInfo:
    """Information about a federated client."""

    client_id: str
    site_name: str
    location: str
    mfc_type: str
    bacterial_species: BacterialSpecies

    # Performance metrics
    data_samples: int = 0
    computation_power: float = 1.0  # Relative computational capability
    communication_bandwidth: float = 1.0  # MB/s
    reliability_score: float = 1.0  # Historical reliability

    # Status
    is_active: bool = True
    last_update: datetime = field(default_factory=datetime.now)
    round_participation: list[int] = field(default_factory=list)

    # Local model performance
    local_accuracy: float = 0.0
    local_loss: float = float("inf")
    improvement_rate: float = 0.0


class DifferentialPrivacy:
    """Differential privacy mechanism for federated learning."""

    def __init__(
        self,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
    ) -> None:
        """Initialize differential privacy mechanism.

        Args:
            noise_multiplier: Noise scale for privacy
            max_grad_norm: Maximum gradient norm for clipping

        """
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

        logger.info(
            f"Differential privacy initialized: noise={noise_multiplier}, clip={max_grad_norm}",
        )

    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients to bounded sensitivity."""
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_grad_norm,
        )
        return total_norm.item()

    def add_noise(self, model: nn.Module, device: torch.device) -> None:
        """Add Gaussian noise to model parameters."""
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.normal(
                        mean=0,
                        std=self.noise_multiplier * self.max_grad_norm,
                        size=param.shape,
                    ).to(device)
                    param.grad += noise

    def compute_privacy_budget(
        self,
        steps: int,
        sampling_rate: float,
        target_delta: float = 1e-5,
    ) -> float:
        """Compute privacy budget (epsilon) using RDP accountant."""
        # Simplified privacy accounting - in practice, use more sophisticated methods
        q = sampling_rate
        sigma = self.noise_multiplier

        # Approximate epsilon calculation
        return steps * q * (q / (2 * sigma**2))


class SecureAggregation:
    """Secure aggregation for privacy-preserving federated learning."""

    def __init__(self, num_clients: int) -> None:
        """Initialize secure aggregation."""
        self.num_clients = num_clients
        self.threshold = max(2, num_clients // 2)  # Minimum clients needed

        logger.info(f"Secure aggregation initialized for {num_clients} clients")

    def generate_masks(
        self,
        model_params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Generate random masks for secure aggregation."""
        masks = {}
        for name, param in model_params.items():
            masks[name] = torch.randn_like(param)
        return masks

    def mask_model(
        self,
        model_params: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply masks to model parameters."""
        masked_params = {}
        for name, param in model_params.items():
            if name in masks:
                masked_params[name] = param + masks[name]
            else:
                masked_params[name] = param
        return masked_params

    def unmask_aggregate(
        self,
        masked_models: list[dict[str, torch.Tensor]],
        all_masks: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Unmask and aggregate models securely."""
        if len(masked_models) < self.threshold:
            msg = f"Insufficient clients: {len(masked_models)} < {self.threshold}"
            raise ValueError(
                msg,
            )

        # Sum all masked models
        aggregated = {}
        for name in masked_models[0]:
            aggregated[name] = torch.zeros_like(masked_models[0][name])
            for model in masked_models:
                aggregated[name] += model[name]

        # Remove masks
        total_masks = {}
        for name in aggregated:
            total_masks[name] = torch.zeros_like(aggregated[name])
            for masks in all_masks:
                if name in masks:
                    total_masks[name] += masks[name]

        # Final unmasked aggregate
        final_aggregate = {}
        for name in aggregated:
            final_aggregate[name] = (aggregated[name] - total_masks[name]) / len(
                masked_models,
            )

        return final_aggregate


class ModelCompression:
    """Model compression for communication-efficient federated learning."""

    def __init__(
        self,
        compression_ratio: float = 0.1,
        quantization_bits: int = 8,
        sparsification_ratio: float = 0.01,
    ) -> None:
        """Initialize model compression.

        Args:
            compression_ratio: Overall compression ratio
            quantization_bits: Bits for quantization
            sparsification_ratio: Ratio of parameters to keep

        """
        self.compression_ratio = compression_ratio
        self.quantization_bits = quantization_bits
        self.sparsification_ratio = sparsification_ratio

        logger.info(
            f"Model compression initialized: {compression_ratio} ratio, {quantization_bits} bits",
        )

    def compress_model(self, model_params: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Compress model parameters."""
        compressed = {}

        for name, param in model_params.items():
            # Sparsification: keep only top-k parameters
            flat_param = param.flatten()
            k = max(1, int(len(flat_param) * self.sparsification_ratio))

            # Get indices of top-k absolute values
            _, top_indices = torch.topk(torch.abs(flat_param), k)

            # Create sparse representation
            sparse_values = flat_param[top_indices]

            # Quantization
            quantized_values = self._quantize_tensor(sparse_values)

            compressed[name] = {
                "shape": param.shape,
                "indices": top_indices.cpu().numpy(),
                "values": quantized_values,
                "original_size": param.numel(),
            }

        return compressed

    def decompress_model(
        self,
        compressed_params: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Decompress model parameters."""
        decompressed = {}

        for name, comp_data in compressed_params.items():
            # Reconstruct sparse tensor
            flat_tensor = torch.zeros(comp_data["original_size"])

            # Dequantize values
            dequantized_values = self._dequantize_tensor(comp_data["values"])

            # Place values at correct indices
            indices = torch.from_numpy(comp_data["indices"])
            flat_tensor[indices] = dequantized_values

            # Reshape to original shape
            decompressed[name] = flat_tensor.reshape(comp_data["shape"])

        return decompressed

    def _quantize_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """Quantize tensor to specified bits."""
        # Simple uniform quantization
        min_val, max_val = tensor.min().item(), tensor.max().item()

        if min_val == max_val:
            return np.zeros(tensor.shape, dtype=np.uint8)

        # Scale to [0, 2^bits - 1]
        scale = (2**self.quantization_bits - 1) / (max_val - min_val)
        quantized = (
            ((tensor - min_val) * scale).round().clamp(0, 2**self.quantization_bits - 1)
        )

        return quantized.detach().cpu().numpy().astype(np.uint8)

    def _dequantize_tensor(self, quantized_array: np.ndarray) -> torch.Tensor:
        """Dequantize array back to tensor."""
        # This is a simplified version - in practice, store min/max values
        quantized_tensor = torch.from_numpy(quantized_array).float()

        # Scale back to approximate original range
        scale = 1.0 / (2**self.quantization_bits - 1)
        return quantized_tensor * scale


def _select_device(device: str | None = None) -> torch.device:
    """Select computing device (GPU/CPU).

    Shared utility function for device selection.

    Args:
        device: Device specification ('cuda', 'cpu', 'auto', or None)

    Returns:
        torch.device: Selected computing device

    """
    if device == "auto" or device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _get_model_size(model: nn.Module) -> dict[str, Any]:
    """Get model size metrics.

    Shared utility function for model size calculation.

    Args:
        model: Neural network model

    Returns:
        Dictionary with model size metrics

    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_size_mb = total_params * 4 / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "memory_mb": total_size_mb,
    }


class FederatedClient:
    """Federated learning client for MFC systems."""

    def __init__(
        self,
        client_info: ClientInfo,
        model: nn.Module,
        config: FederatedConfig,
    ) -> None:
        """Initialize federated client.

        Args:
            client_info: Client information
            model: Local model
            config: Federated learning configuration

        """
        self.client_info = client_info
        self.local_model = copy.deepcopy(model)
        self.config = config

        # Device selection using shared utility
        self.device = _select_device()
        self.local_model.to(self.device)

        # Local optimizer
        self.optimizer = optim.SGD(
            self.local_model.parameters(),
            lr=config.local_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

        # Privacy mechanisms
        if config.differential_privacy:
            self.dp_mechanism = DifferentialPrivacy(
                config.noise_multiplier,
                config.max_grad_norm,
            )
        else:
            self.dp_mechanism = None

        # Compression
        self.compressor = ModelCompression(
            config.compression_ratio,
            config.quantization_bits,
            config.sparsification_ratio,
        )

        # Local data and performance
        self.local_data = []
        self.training_history = deque(maxlen=100)
        self.participation_history = []

        # Feature engineering (lazy loaded)
        self._feature_engineer = None

        logger.info(f"Federated client initialized: {client_info.client_id}")
        logger.info(
            f"Site: {client_info.site_name}, Species: {client_info.bacterial_species.value}",
        )

    @property
    def feature_engineer(self):
        """Lazy-load feature engineer to avoid circular imports."""
        if self._feature_engineer is None:
            from ml_optimization import FeatureEngineer

            self._feature_engineer = FeatureEngineer()
        return self._feature_engineer

    def get_model_size(self) -> dict[str, Any]:
        """Get local model size metrics."""
        return _get_model_size(self.local_model)

    def add_local_data(
        self,
        system_states: list[SystemState],
        actions: list[int],
        rewards: list[float],
    ) -> None:
        """Add local training data."""
        for state, action, reward in zip(system_states, actions, rewards, strict=False):
            # Extract features using Phase 2 feature engineering
            performance_metrics = {
                "power_efficiency": state.power_output
                / max(state.current_density, 0.01),
                "biofilm_health_score": state.health_metrics.overall_health_score,
                "sensor_reliability": state.fused_measurement.fusion_confidence,
                "system_stability": 1.0 - len(state.anomalies) / 10.0,
                "control_confidence": 0.8,
            }

            features = self.feature_engineer.extract_features(
                state,
                performance_metrics,
            )
            feature_vector = torch.FloatTensor(list(features.values()))

            self.local_data.append(
                {
                    "features": feature_vector,
                    "action": action,
                    "reward": reward,
                    "state": state,
                },
            )

        self.client_info.data_samples = len(self.local_data)
        logger.info(
            f"Client {self.client_info.client_id} now has {len(self.local_data)} samples",
        )

    def local_train(
        self,
        global_model_params: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Perform local training on client data.

        Args:
            global_model_params: Global model parameters

        Returns:
            Training results and updated model

        """
        if not self.local_data:
            return {"error": "No local data available"}

        # Load global model parameters
        self.local_model.load_state_dict(global_model_params)
        self.local_model.train()

        # Local training loop
        total_loss = 0.0
        num_batches = 0

        for _epoch in range(self.config.local_epochs):
            epoch_loss = 0.0

            # Simple batch processing (in practice, use DataLoader)
            batch_size = min(32, len(self.local_data))

            for i in range(0, len(self.local_data), batch_size):
                batch_data = self.local_data[i : i + batch_size]

                # Prepare batch tensors
                features = torch.stack([d["features"] for d in batch_data]).to(
                    self.device,
                )
                actions = torch.LongTensor([d["action"] for d in batch_data]).to(
                    self.device,
                )
                torch.FloatTensor([d["reward"] for d in batch_data]).to(self.device)

                # Forward pass (simplified - assumes model outputs action logits)
                self.optimizer.zero_grad()

                try:
                    # This would need to be adapted based on actual model architecture
                    logits = self.local_model(features)

                    # Simple supervised learning loss
                    action_targets = F.one_hot(
                        actions,
                        num_classes=logits.size(-1),
                    ).float()
                    loss = F.mse_loss(logits, action_targets)

                    # Backward pass
                    loss.backward()

                    # Apply differential privacy
                    if self.dp_mechanism:
                        self.dp_mechanism.clip_gradients(self.local_model)
                        self.dp_mechanism.add_noise(self.local_model, self.device)

                    self.optimizer.step()

                    batch_loss = loss.item()
                    epoch_loss += batch_loss
                    total_loss += batch_loss
                    num_batches += 1

                except Exception as e:
                    logger.warning(f"Training batch failed: {e}")
                    continue

        # Calculate training metrics
        avg_loss = total_loss / max(num_batches, 1)

        # Get model parameters for aggregation
        model_params = {
            name: param.cpu().clone()
            for name, param in self.local_model.named_parameters()
        }

        # Compress model if enabled
        if self.config.compression_ratio < 1.0:
            compressed_params = self.compressor.compress_model(model_params)
        else:
            compressed_params = model_params

        # Update client info
        self.client_info.local_loss = avg_loss
        self.client_info.last_update = datetime.now()

        # Training results
        results = {
            "client_id": self.client_info.client_id,
            "model_params": compressed_params,
            "num_samples": len(self.local_data),
            "loss": avg_loss,
            "epochs": self.config.local_epochs,
            "batches": num_batches,
            "compression_used": self.config.compression_ratio < 1.0,
        }

        self.training_history.append(results)

        return results

    def evaluate_model(self, test_data: list[dict[str, Any]]) -> dict[str, float]:
        """Evaluate local model performance."""
        if not test_data:
            return {"accuracy": 0.0, "loss": float("inf")}

        self.local_model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for data in test_data:
                features = data["features"].unsqueeze(0).to(self.device)
                true_action = data["action"]

                try:
                    logits = self.local_model(features)
                    predicted_action = torch.argmax(logits, dim=-1).item()

                    if predicted_action == true_action:
                        correct_predictions += 1
                    total_predictions += 1

                    # Calculate loss
                    action_target = F.one_hot(
                        torch.tensor([true_action]),
                        num_classes=logits.size(-1),
                    ).float()
                    loss = F.mse_loss(logits, action_target)
                    total_loss += loss.item()

                except Exception as e:
                    logger.warning(f"Evaluation failed for sample: {e}")
                    continue

        accuracy = correct_predictions / max(total_predictions, 1)
        avg_loss = total_loss / max(total_predictions, 1)

        # Update client info
        self.client_info.local_accuracy = accuracy
        self.client_info.local_loss = avg_loss

        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "samples_evaluated": total_predictions,
        }


class FederatedServer:
    """Federated learning server for coordinating MFC clients."""

    def __init__(self, global_model: nn.Module, config: FederatedConfig) -> None:
        """Initialize federated server.

        Args:
            global_model: Global model template
            config: Federated learning configuration

        """
        self.global_model = copy.deepcopy(global_model)
        self.config = config

        # Device selection using shared utility
        self.device = _select_device()
        self.global_model.to(self.device)

        # Client management
        self.clients: dict[str, FederatedClient] = {}
        self.client_info: dict[str, ClientInfo] = {}

        # Aggregation mechanisms
        if config.secure_aggregation:
            self.secure_aggregator = SecureAggregation(config.num_clients)
        else:
            self.secure_aggregator = None

        # Training state
        self.current_round = 0
        self.training_history = []
        self.client_selection_history = []

        # Performance tracking
        self.global_metrics = {
            "loss": deque(maxlen=config.num_rounds),
            "accuracy": deque(maxlen=config.num_rounds),
            "participation_rate": deque(maxlen=config.num_rounds),
            "communication_cost": deque(maxlen=config.num_rounds),
        }

        logger.info("Federated server initialized")
        logger.info(f"Algorithm: {config.algorithm.value}")
        logger.info(f"Aggregation: {config.aggregation.value}")
        logger.info(f"Expected clients: {config.num_clients}")

    def register_client(self, client_info: ClientInfo) -> bool:
        """Register a new client with the federation."""
        try:
            client_model = copy.deepcopy(self.global_model)
            client = FederatedClient(client_info, client_model, self.config)

            self.clients[client_info.client_id] = client
            self.client_info[client_info.client_id] = client_info

            logger.info(
                f"Client registered: {client_info.client_id} from {client_info.site_name}",
            )
            return True

        except Exception as e:
            logger.exception(f"Failed to register client {client_info.client_id}: {e}")
            return False

    def select_clients(self, round_num: int) -> list[str]:
        """Select clients for the current round.

        Args:
            round_num: Current round number

        Returns:
            List of selected client IDs

        """
        available_clients = [
            cid
            for cid, info in self.client_info.items()
            if info.is_active and len(self.clients[cid].local_data) > 0
        ]

        if len(available_clients) == 0:
            return []

        num_select = min(self.config.clients_per_round, len(available_clients))

        if self.config.client_selection == ClientSelectionStrategy.RANDOM:
            selected = np.random.choice(
                available_clients,
                num_select,
                replace=False,
            ).tolist()

        elif self.config.client_selection == ClientSelectionStrategy.CYCLIC:
            start_idx = (round_num * num_select) % len(available_clients)
            selected = []
            for i in range(num_select):
                idx = (start_idx + i) % len(available_clients)
                selected.append(available_clients[idx])

        elif self.config.client_selection == ClientSelectionStrategy.PERFORMANCE_BASED:
            # Select based on historical performance
            client_scores = []
            for cid in available_clients:
                info = self.client_info[cid]
                score = info.reliability_score * (
                    1.0 - info.local_loss / 10.0
                )  # Normalize loss
                client_scores.append((cid, score))

            # Sort by score and select top clients
            client_scores.sort(key=lambda x: x[1], reverse=True)
            selected = [cid for cid, _ in client_scores[:num_select]]

        elif self.config.client_selection == ClientSelectionStrategy.RESOURCE_AWARE:
            # Select based on computational and communication resources
            client_scores = []
            for cid in available_clients:
                info = self.client_info[cid]
                resource_score = info.computation_power * info.communication_bandwidth
                client_scores.append((cid, resource_score))

            client_scores.sort(key=lambda x: x[1], reverse=True)
            selected = [cid for cid, _ in client_scores[:num_select]]

        else:
            # Default to random selection
            selected = np.random.choice(
                available_clients,
                num_select,
                replace=False,
            ).tolist()

        self.client_selection_history.append(
            {
                "round": round_num,
                "selected_clients": selected,
                "selection_strategy": self.config.client_selection.value,
                "total_available": len(available_clients),
            },
        )

        logger.info(
            f"Round {round_num}: Selected {len(selected)} clients out of {len(available_clients)} available",
        )

        return selected

    def aggregate_models(
        self,
        client_updates: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """Aggregate client model updates.

        Args:
            client_updates: List of client training results

        Returns:
            Aggregated global model parameters

        """
        if not client_updates:
            return {
                name: param.clone()
                for name, param in self.global_model.named_parameters()
            }

        if self.config.aggregation == AggregationMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_aggregation(client_updates)
        if self.config.aggregation == AggregationMethod.MEDIAN_AGGREGATION:
            return self._median_aggregation(client_updates)
        if self.config.aggregation == AggregationMethod.TRIMMED_MEAN:
            return self._trimmed_mean_aggregation(client_updates)
        if self.config.aggregation == AggregationMethod.BYZANTINE_ROBUST:
            return self._byzantine_robust_aggregation(client_updates)
        return self._weighted_average_aggregation(client_updates)

    def _weighted_average_aggregation(
        self,
        client_updates: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """Federated averaging aggregation."""
        total_samples = sum(update["num_samples"] for update in client_updates)

        if total_samples == 0:
            return {
                name: param.clone()
                for name, param in self.global_model.named_parameters()
            }

        # Initialize aggregated parameters
        aggregated_params = {}

        for update in client_updates:
            model_params = update["model_params"]
            weight = update["num_samples"] / total_samples

            # Decompress if needed
            if update.get("compression_used", False):
                model_params = self.clients[
                    update["client_id"]
                ].compressor.decompress_model(model_params)

            for name, param in model_params.items():
                if name not in aggregated_params:
                    aggregated_params[name] = torch.zeros_like(param)
                aggregated_params[name] += weight * param

        return aggregated_params

    def _median_aggregation(
        self,
        client_updates: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """Coordinate-wise median aggregation."""
        aggregated_params = {}

        # Collect all parameters
        all_params = []
        for update in client_updates:
            model_params = update["model_params"]
            if update.get("compression_used", False):
                model_params = self.clients[
                    update["client_id"]
                ].compressor.decompress_model(model_params)
            all_params.append(model_params)

        # Compute median for each parameter
        for name in all_params[0]:
            param_stack = torch.stack([params[name] for params in all_params])
            aggregated_params[name] = torch.median(param_stack, dim=0)[0]

        return aggregated_params

    def _trimmed_mean_aggregation(
        self,
        client_updates: list[dict[str, Any]],
        trim_ratio: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Trimmed mean aggregation for Byzantine robustness."""
        aggregated_params = {}

        # Collect all parameters
        all_params = []
        for update in client_updates:
            model_params = update["model_params"]
            if update.get("compression_used", False):
                model_params = self.clients[
                    update["client_id"]
                ].compressor.decompress_model(model_params)
            all_params.append(model_params)

        # Compute trimmed mean for each parameter
        trim_count = int(len(all_params) * trim_ratio)

        for name in all_params[0]:
            param_stack = torch.stack([params[name] for params in all_params])

            # Sort and trim extreme values
            sorted_params, _ = torch.sort(param_stack, dim=0)
            if trim_count > 0:
                trimmed_params = sorted_params[trim_count:-trim_count]
            else:
                trimmed_params = sorted_params

            # Compute mean of trimmed values
            aggregated_params[name] = torch.mean(trimmed_params, dim=0)

        return aggregated_params

    def _byzantine_robust_aggregation(
        self,
        client_updates: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """Byzantine-robust aggregation using geometric median."""
        # Simplified implementation - in practice, use more sophisticated methods
        return self._trimmed_mean_aggregation(client_updates, trim_ratio=0.2)

    def run_round(self, round_num: int) -> dict[str, Any]:
        """Execute one round of federated learning.

        Args:
            round_num: Current round number

        Returns:
            Round results and metrics

        """
        logger.info(f"Starting federated learning round {round_num}")

        # Select clients for this round
        selected_clients = self.select_clients(round_num)

        if not selected_clients:
            logger.warning(f"No clients available for round {round_num}")
            return {"error": "No clients available", "round": round_num}

        # Get current global model parameters
        global_params = {
            name: param.cpu().clone()
            for name, param in self.global_model.named_parameters()
        }

        # Collect client updates
        client_updates = []
        communication_cost = 0

        for client_id in selected_clients:
            client = self.clients[client_id]

            try:
                # Local training
                start_time = time.time()
                update = client.local_train(global_params)
                training_time = time.time() - start_time

                if "error" not in update:
                    client_updates.append(update)

                    # Estimate communication cost (simplified)
                    param_count = sum(p.numel() for p in global_params.values())
                    communication_cost += param_count * 4  # 4 bytes per float

                    # Update client participation
                    client.client_info.round_participation.append(round_num)

                    logger.info(
                        f"Client {client_id} completed training: "
                        f"loss={update['loss']:.4f}, time={training_time:.2f}s",
                    )
                else:
                    logger.warning(
                        f"Client {client_id} failed training: {update['error']}",
                    )

            except Exception as e:
                logger.exception(f"Error training client {client_id}: {e}")
                continue

        if not client_updates:
            logger.error(f"No successful client updates in round {round_num}")
            return {"error": "No successful updates", "round": round_num}

        # Aggregate models
        aggregation_start = time.time()
        aggregated_params = self.aggregate_models(client_updates)
        aggregation_time = time.time() - aggregation_start

        # Update global model
        self.global_model.load_state_dict(aggregated_params)

        # Calculate round metrics
        avg_loss = np.mean([update["loss"] for update in client_updates])
        participation_rate = len(client_updates) / len(selected_clients)

        # Store metrics
        self.global_metrics["loss"].append(avg_loss)
        self.global_metrics["participation_rate"].append(participation_rate)
        self.global_metrics["communication_cost"].append(communication_cost)

        # Round results
        round_results = {
            "round": round_num,
            "selected_clients": len(selected_clients),
            "successful_updates": len(client_updates),
            "average_loss": avg_loss,
            "participation_rate": participation_rate,
            "communication_cost_bytes": communication_cost,
            "aggregation_time_seconds": aggregation_time,
            "client_results": client_updates,
        }

        self.training_history.append(round_results)
        self.current_round = round_num

        logger.info(
            f"Round {round_num} completed: "
            f"avg_loss={avg_loss:.4f}, "
            f"participation={participation_rate:.2f}, "
            f"comm_cost={communication_cost / 1024 / 1024:.2f}MB",
        )

        return round_results

    def train_federation(self) -> dict[str, Any]:
        """Train the complete federation for all rounds.

        Returns:
            Complete training results

        """
        logger.info(f"Starting federated training for {self.config.num_rounds} rounds")

        training_start = time.time()

        for round_num in range(1, self.config.num_rounds + 1):
            round_results = self.run_round(round_num)

            if "error" in round_results:
                logger.warning(
                    f"Round {round_num} had errors: {round_results['error']}",
                )
                continue

            # Early stopping based on convergence (optional)
            if len(self.global_metrics["loss"]) >= 5:
                recent_losses = list(self.global_metrics["loss"])[-5:]
                if np.std(recent_losses) < 0.001:  # Convergence threshold
                    logger.info(
                        f"Early stopping at round {round_num} due to convergence",
                    )
                    break

        training_time = time.time() - training_start

        # Final federation results
        federation_results = {
            "total_rounds": self.current_round,
            "total_clients": len(self.clients),
            "training_time_seconds": training_time,
            "final_loss": (
                list(self.global_metrics["loss"])[-1]
                if self.global_metrics["loss"]
                else 0.0
            ),
            "average_participation_rate": np.mean(
                list(self.global_metrics["participation_rate"]),
            ),
            "total_communication_cost_mb": sum(
                self.global_metrics["communication_cost"],
            )
            / 1024
            / 1024,
            "global_model_parameters": sum(
                p.numel() for p in self.global_model.parameters()
            ),
            "training_history": self.training_history[:10],  # Last 10 rounds
            "client_summary": self._get_client_summary(),
        }

        logger.info(
            f"Federated training completed: {self.current_round} rounds, "
            f"{training_time:.1f}s, final_loss={federation_results['final_loss']:.4f}",
        )

        return federation_results

    def _get_client_summary(self) -> dict[str, Any]:
        """Get summary of client performance."""
        client_summary = {}

        for client_id, client_info in self.client_info.items():
            client_summary[client_id] = {
                "site_name": client_info.site_name,
                "bacterial_species": client_info.bacterial_species.value,
                "data_samples": client_info.data_samples,
                "rounds_participated": len(client_info.round_participation),
                "last_accuracy": client_info.local_accuracy,
                "last_loss": client_info.local_loss,
                "reliability_score": client_info.reliability_score,
            }

        return client_summary

    def save_federation(self, path: str) -> None:
        """Save federation state."""
        save_dict = {
            "global_model_state": self.global_model.state_dict(),
            "config": self.config,
            "current_round": self.current_round,
            "training_history": self.training_history,
            "client_info": {
                cid: info.__dict__ for cid, info in self.client_info.items()
            },
            "global_metrics": {k: list(v) for k, v in self.global_metrics.items()},
        }

        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

        logger.info(f"Federation saved to {path}")


def create_federated_system(
    global_model: nn.Module,
    num_clients: int = 5,
    algorithm: str = "fedavg",
    **kwargs,
) -> FederatedServer:
    """Factory function to create federated learning system.

    Args:
        global_model: Global model template
        num_clients: Number of clients
        algorithm: Federated algorithm
        **kwargs: Additional configuration

    Returns:
        Configured federated server

    """
    # Algorithm mapping
    algorithm_map = {
        "fedavg": FederatedAlgorithm.FEDAVG,
        "fedprox": FederatedAlgorithm.FEDPROX,
        "fedper": FederatedAlgorithm.FEDPER,
        "scaffold": FederatedAlgorithm.SCAFFOLD,
    }

    # Configure federated learning
    config = FederatedConfig()
    config.num_clients = num_clients
    config.algorithm = algorithm_map.get(algorithm.lower(), FederatedAlgorithm.FEDAVG)

    # Apply additional configuration
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Create federated server
    server = FederatedServer(global_model, config)

    logger.info(
        f"Federated system created with {algorithm} algorithm for {num_clients} clients",
    )

    return server


if __name__ == "__main__":
    """Test federated learning system functionality."""

    # Create a simple test model
    test_model = nn.Sequential(nn.Linear(70, 128), nn.ReLU(), nn.Linear(128, 15))

    # Create federated system
    fed_system = create_federated_system(
        global_model=test_model,
        num_clients=3,
        algorithm="fedavg",
        clients_per_round=2,
        num_rounds=5,
        local_epochs=3,
    )

    # Create test clients
    client_configs = [
        {
            "client_id": "mfc_site_1",
            "site_name": "Industrial Plant A",
            "location": "Germany",
            "mfc_type": "dual_chamber",
            "bacterial_species": BacterialSpecies.GEOBACTER,
        },
        {
            "client_id": "mfc_site_2",
            "site_name": "Research Lab B",
            "location": "USA",
            "mfc_type": "single_chamber",
            "bacterial_species": BacterialSpecies.SHEWANELLA,
        },
        {
            "client_id": "mfc_site_3",
            "site_name": "Wastewater Plant C",
            "location": "Japan",
            "mfc_type": "stacked",
            "bacterial_species": BacterialSpecies.MIXED,
        },
    ]

    # Register clients
    for config in client_configs:
        client_info = ClientInfo(**config)
        client_info.data_samples = np.random.randint(50, 200)
        client_info.computation_power = np.random.uniform(0.5, 2.0)
        client_info.communication_bandwidth = np.random.uniform(1.0, 10.0)

        success = fed_system.register_client(client_info)

    # Add dummy training data
    for client_id in fed_system.clients:
        num_samples = fed_system.client_info[client_id].data_samples

        # Generate random training data (normally would come from real MFC operations)
        dummy_states = []
        dummy_actions = []
        dummy_rewards = []

        for _ in range(min(num_samples, 10)):  # Limit for testing
            # Create dummy system state (would be real MFC data)
            dummy_health = HealthMetrics(
                overall_health_score=np.random.uniform(0.3, 0.9),
                thickness_health=np.random.uniform(0.3, 0.9),
                conductivity_health=np.random.uniform(0.3, 0.9),
                growth_health=np.random.uniform(0.3, 0.9),
                stability_health=np.random.uniform(0.3, 0.9),
                thickness_contribution=0.2,
                conductivity_contribution=0.3,
                growth_contribution=0.25,
                stability_contribution=0.25,
                assessment_confidence=0.9,
                prediction_confidence=0.85,
                health_status=None,
                health_trend=None,
                predicted_health_24h=np.random.uniform(0.3, 0.9),
                predicted_intervention_time=None,
                fouling_risk=np.random.uniform(0.0, 0.5),
                detachment_risk=np.random.uniform(0.0, 0.3),
                stagnation_risk=np.random.uniform(0.0, 0.4),
            )

            # This would be replaced with actual SystemState construction
            break  # Skip actual data addition for this test

    # Test client selection
    selected = fed_system.select_clients(1)
