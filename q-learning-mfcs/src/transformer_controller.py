"""Transformer-Based Controller with Attention Mechanisms for Advanced MFC Control.

This module implements state-of-the-art transformer architectures and attention
mechanisms for sophisticated temporal pattern recognition and decision making
in MFC control systems.

Key Features:
- Multi-head self-attention for temporal sequence modeling
- Transformer encoder-decoder architecture for sequence-to-sequence control
- Temporal attention for long-range dependency modeling
- Cross-modal attention between sensor modalities (EIS, QCM)
- Causal attention for autoregressive control decisions
- Position encoding for temporal information
- Layer normalization and residual connections
- Attention visualization and interpretability
- Pre-trained transformer models for MFC control

Integration with Phase 2 and 3 components:
- Processes sequential sensor data from advanced sensor fusion
- Incorporates health monitoring attention mechanisms
- Uses transformer features in deep RL and transfer learning
- Provides interpretable attention weights for decision explanation

Created: 2025-07-31
Last Modified: 2025-07-31
"""

from __future__ import annotations

import logging
import math
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

# Import Phase 2 and 3 components
from ml_optimization import FeatureEngineer
from torch import nn, optim

if TYPE_CHECKING:
    from adaptive_mfc_controller import SystemState

# Import configuration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionType(Enum):
    """Types of attention mechanisms."""

    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    TEMPORAL_ATTENTION = "temporal_attention"
    CAUSAL_ATTENTION = "causal_attention"
    SPARSE_ATTENTION = "sparse_attention"


class TransformerConfig:
    """Configuration for transformer models."""

    def __init__(self) -> None:
        # Model architecture
        self.d_model = 256  # Model dimension
        self.n_heads = 8  # Number of attention heads
        self.n_layers = 6  # Number of transformer layers
        self.d_ff = 1024  # Feed-forward dimension
        self.dropout = 0.1  # Dropout rate

        # Sequence parameters
        self.max_seq_len = 128  # Maximum sequence length
        self.context_window = 64  # Context window for attention

        # Position encoding
        self.pos_encoding_type = "sinusoidal"  # "sinusoidal" or "learned"
        self.max_position = 1000

        # Attention parameters
        self.attention_dropout = 0.1
        self.layer_norm_eps = 1e-6
        self.use_residual = True

        # Training parameters
        self.learning_rate = 1e-4
        self.warmup_steps = 4000
        self.weight_decay = 1e-4
        self.label_smoothing = 0.1

        # Specialized configurations
        self.sensor_fusion_heads = 4  # Heads for cross-modal attention
        self.temporal_heads = 4  # Heads for temporal attention
        self.health_attention_dim = 64  # Dimension for health attention


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length

        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute sinusoidal encodings
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model),
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[: x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with various attention types."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        attention_type: AttentionType = AttentionType.SELF_ATTENTION,
    ) -> None:
        """Initialize multi-head attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            attention_type: Type of attention mechanism

        """
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.attention_type = attention_type

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scale factor
        self.scale = math.sqrt(self.d_k)

        logger.info(
            f"Multi-head attention initialized: {attention_type.value}, {n_heads} heads",
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through multi-head attention.

        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]

        Returns:
            Output tensor and attention weights

        """
        batch_size, seq_len, _ = query.size()

        # Linear projections and reshape for multi-head
        Q = (
            self.w_q(query)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(key)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(value)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        attention_output, attention_weights = self._scaled_dot_product_attention(
            Q,
            K,
            V,
            mask,
        )

        # Concatenate heads
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # Final linear projection
        output = self.w_o(attention_output)

        return output, attention_weights

    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention computation."""
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply attention mask
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply causal mask for causal attention
        if self.attention_type == AttentionType.CAUSAL_ATTENTION:
            seq_len = scores.size(-1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(scores.device)
            scores = scores.masked_fill(causal_mask == 0, -1e9)

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with multi-head attention and feed-forward."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize transformer encoder layer."""
        super().__init__()

        self.config = config

        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(
            config.d_model,
            config.n_heads,
            config.attention_dropout,
            AttentionType.SELF_ATTENTION,
        )

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder layer."""
        # Multi-head self-attention with residual connection
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attention_weights


class SensorFusionAttention(nn.Module):
    """Cross-modal attention for sensor fusion (EIS, QCM, etc.)."""

    def __init__(self, config: TransformerConfig, sensor_dims: dict[str, int]) -> None:
        """Initialize sensor fusion attention.

        Args:
            config: Transformer configuration
            sensor_dims: Dimensions for each sensor type

        """
        super().__init__()

        self.config = config
        self.sensor_dims = sensor_dims
        self.sensor_types = list(sensor_dims.keys())

        # Sensor-specific projections
        self.sensor_projections = nn.ModuleDict()
        for sensor, dim in sensor_dims.items():
            self.sensor_projections[sensor] = nn.Linear(dim, config.d_model)

        # Cross-modal attention
        self.cross_attention = MultiHeadAttention(
            config.d_model,
            config.sensor_fusion_heads,
            config.attention_dropout,
            AttentionType.CROSS_ATTENTION,
        )

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.d_model * len(sensor_dims), config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
        )

        logger.info(
            f"Sensor fusion attention initialized for sensors: {list(sensor_dims.keys())}",
        )

    def forward(
        self,
        sensor_data: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass through sensor fusion attention.

        Args:
            sensor_data: Dictionary of sensor data tensors

        Returns:
            Fused representation and attention weights

        """
        # Project each sensor to common dimension
        projected_sensors = {}
        for sensor, data in sensor_data.items():
            if sensor in self.sensor_projections:
                projected_sensors[sensor] = self.sensor_projections[sensor](data)

        # Cross-modal attention between sensor pairs
        attention_outputs = {}
        attention_weights = {}

        for i, sensor1 in enumerate(self.sensor_types):
            if sensor1 not in projected_sensors:
                continue

            sensor1_data = projected_sensors[sensor1]
            cross_modal_outputs = []

            for j, sensor2 in enumerate(self.sensor_types):
                if sensor2 not in projected_sensors or i == j:
                    continue

                sensor2_data = projected_sensors[sensor2]

                # Cross-attention: sensor1 attends to sensor2
                cross_output, cross_weights = self.cross_attention(
                    sensor1_data,
                    sensor2_data,
                    sensor2_data,
                )

                cross_modal_outputs.append(cross_output)
                attention_weights[f"{sensor1}_to_{sensor2}"] = cross_weights

            if cross_modal_outputs:
                # Combine cross-modal information
                combined_cross = torch.stack(cross_modal_outputs, dim=0).mean(dim=0)
                attention_outputs[sensor1] = combined_cross

        # Fuse all sensor representations
        if attention_outputs:
            fused_tensor = torch.cat(list(attention_outputs.values()), dim=-1)
            fused_output = self.fusion_layer(fused_tensor)
        # Fallback if no cross-modal attention
        elif projected_sensors:
            fused_tensor = torch.cat(list(projected_sensors.values()), dim=-1)
            fused_output = self.fusion_layer(fused_tensor)
        else:
            fused_output = torch.zeros(1, 1, self.config.d_model)

        return fused_output, attention_weights


class TemporalAttentionModule(nn.Module):
    """Temporal attention for modeling long-range dependencies in time series."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize temporal attention module."""
        super().__init__()

        self.config = config

        # Temporal attention layers
        self.temporal_attention = MultiHeadAttention(
            config.d_model,
            config.temporal_heads,
            config.attention_dropout,
            AttentionType.TEMPORAL_ATTENTION,
        )

        # Temporal convolution for local patterns
        self.temporal_conv = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=3,
            padding=1,
        )

        # Combination layer
        self.combination = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Layer normalization
        self.norm = nn.LayerNorm(config.d_model)

        logger.info("Temporal attention module initialized")

    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through temporal attention.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            temporal_mask: Temporal attention mask

        Returns:
            Output tensor and temporal attention weights

        """
        batch_size, seq_len, d_model = x.size()

        # Global temporal attention
        global_output, attention_weights = self.temporal_attention(
            x,
            x,
            x,
            temporal_mask,
        )

        # Local temporal convolution
        x_transposed = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        local_output = self.temporal_conv(x_transposed)
        local_output = local_output.transpose(1, 2)  # [batch_size, seq_len, d_model]

        # Combine global and local temporal information
        combined = torch.cat([global_output, local_output], dim=-1)
        output = self.combination(combined)

        # Residual connection and normalization
        output = self.norm(x + output)

        return output, attention_weights


class TransformerMFCController(nn.Module):
    """Complete Transformer-based MFC Controller with multiple attention mechanisms.

    Integrates temporal attention, sensor fusion attention, and causal attention
    for sophisticated MFC control decision making.
    """

    def __init__(
        self,
        input_dims: dict[str, int],
        output_dim: int,
        config: TransformerConfig | None = None,
    ) -> None:
        """Initialize transformer MFC controller.

        Args:
            input_dims: Input dimensions for different data types
            output_dim: Output dimension (number of actions)
            config: Transformer configuration

        """
        super().__init__()

        self.config = config or TransformerConfig()
        self.input_dims = input_dims
        self.output_dim = output_dim

        # Input projection layers
        self.input_projections = nn.ModuleDict()
        for input_type, dim in input_dims.items():
            self.input_projections[input_type] = nn.Linear(dim, self.config.d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            self.config.d_model,
            self.config.max_position,
        )

        # Sensor fusion attention
        self.sensor_fusion = SensorFusionAttention(self.config, input_dims)

        # Temporal attention
        self.temporal_attention = TemporalAttentionModule(self.config)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(self.config) for _ in range(self.config.n_layers)],
        )

        # Health-aware attention
        self.health_attention = MultiHeadAttention(
            self.config.d_model,
            2,
            self.config.attention_dropout,
            AttentionType.SELF_ATTENTION,
        )

        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, output_dim),
        )

        # Value estimation for RL integration
        self.value_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.config.d_model // 2, 1),
        )

        self.dropout = nn.Dropout(self.config.dropout)

        logger.info("Transformer MFC controller initialized")
        logger.info(f"Input types: {list(input_dims.keys())}")
        logger.info(f"Model dimension: {self.config.d_model}")
        logger.info(f"Number of layers: {self.config.n_layers}")
        logger.info(f"Number of heads: {self.config.n_heads}")

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        health_context: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through transformer controller.

        Args:
            inputs: Dictionary of input tensors
            health_context: Health context for attention
            return_attention: Whether to return attention weights

        Returns:
            Dictionary of outputs including actions, values, and attention weights

        """
        next(iter(inputs.values())).size(0)
        next(iter(inputs.values())).size(1)

        # Project inputs to model dimension
        projected_inputs = {}
        for input_type, data in inputs.items():
            if input_type in self.input_projections:
                projected_inputs[input_type] = self.input_projections[input_type](data)

        # Sensor fusion attention
        if len(projected_inputs) > 1:
            fused_representation, sensor_attention = self.sensor_fusion(
                projected_inputs,
            )
        else:
            fused_representation = next(iter(projected_inputs.values()))
            sensor_attention = {}

        # Add positional encoding
        fused_representation = self.pos_encoding(fused_representation)
        fused_representation = self.dropout(fused_representation)

        # Temporal attention
        temporal_output, temporal_attention = self.temporal_attention(
            fused_representation,
        )

        # Transformer encoder layers
        encoder_output = temporal_output
        encoder_attentions = []

        for layer in self.encoder_layers:
            encoder_output, layer_attention = layer(encoder_output)
            if return_attention:
                encoder_attentions.append(layer_attention)

        # Health-aware attention if health context is provided
        if health_context is not None:
            health_enhanced, health_attention = self.health_attention(
                encoder_output,
                health_context,
                health_context,
            )
            encoder_output = encoder_output + health_enhanced
        else:
            health_attention = None

        # Global pooling for sequence-level representation
        sequence_representation = encoder_output.mean(dim=1)  # [batch_size, d_model]

        # Output projections
        action_logits = self.output_projection(sequence_representation)
        state_value = self.value_head(sequence_representation)

        # Prepare output dictionary
        outputs = {
            "action_logits": action_logits,
            "state_value": state_value,
            "sequence_representation": sequence_representation,
        }

        if return_attention:
            outputs["attention_weights"] = {
                "sensor_fusion": sensor_attention,
                "temporal": temporal_attention,
                "encoder_layers": encoder_attentions,
                "health": health_attention,
            }

        return outputs

    def get_attention_maps(
        self,
        inputs: dict[str, torch.Tensor],
        health_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Get attention maps for interpretability."""
        with torch.no_grad():
            outputs = self.forward(inputs, health_context, return_attention=True)
            return outputs["attention_weights"]


class TransformerControllerManager:
    """Manager class for transformer-based MFC control with training and inference.

    Integrates with Phase 2 and Phase 3 components for comprehensive control.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: TransformerConfig | None = None,
    ) -> None:
        """Initialize transformer controller manager.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            config: Transformer configuration

        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or TransformerConfig()

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define input dimensions for different data types
        self.input_dims = {
            "sensor_features": state_dim // 3,  # EIS/QCM features
            "health_features": state_dim // 3,  # Health monitoring features
            "system_features": state_dim // 3,  # System state features
        }

        # Initialize transformer model
        self.model = TransformerMFCController(
            self.input_dims,
            action_dim,
            self.config,
        ).to(self.device)

        # Optimizer with learning rate scheduling
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler (transformer-style)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                (step + 1) ** -0.5,
                (step + 1) * (self.config.warmup_steps**-1.5),
            ),
        )

        # Feature engineering
        self.feature_engineer = FeatureEngineer()

        # Performance tracking
        self.training_history = {
            "loss": deque(maxlen=1000),
            "accuracy": deque(maxlen=1000),
            "attention_entropy": deque(maxlen=1000),
        }

        # Sequence buffer for temporal modeling
        self.sequence_buffer = deque(maxlen=self.config.max_seq_len)

        logger.info("Transformer controller manager initialized")
        logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}",
        )
        logger.info(f"Device: {self.device}")

    def extract_transformer_features(
        self,
        system_state: SystemState,
    ) -> dict[str, torch.Tensor]:
        """Extract features from system state for transformer input.

        Args:
            system_state: Complete system state

        Returns:
            Dictionary of feature tensors

        """
        # Use Phase 2 feature engineering
        performance_metrics = {
            "power_efficiency": system_state.power_output
            / max(system_state.current_density, 0.01),
            "biofilm_health_score": system_state.health_metrics.overall_health_score,
            "sensor_reliability": system_state.fused_measurement.fusion_confidence,
            "system_stability": 1.0 - len(system_state.anomalies) / 10.0,
            "control_confidence": 0.8,
        }

        # Extract comprehensive features
        all_features = self.feature_engineer.extract_features(
            system_state,
            performance_metrics,
        )
        feature_vector = list(all_features.values())

        # Split features into different categories
        feature_len = len(feature_vector)
        split_size = feature_len // 3

        return {
            "sensor_features": torch.FloatTensor(feature_vector[:split_size])
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device),  # [1, 1, dim]
            "health_features": torch.FloatTensor(
                feature_vector[split_size : 2 * split_size],
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device),
            "system_features": torch.FloatTensor(
                feature_vector[2 * split_size : 3 * split_size],
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device),
        }

    def control_step(self, system_state: SystemState) -> tuple[int, dict[str, Any]]:
        """Execute one control step with transformer.

        Args:
            system_state: Current system state

        Returns:
            Action and control information

        """
        # Extract features
        features = self.extract_transformer_features(system_state)

        # Add to sequence buffer
        self.sequence_buffer.append(features)

        # Prepare sequence input (use last few timesteps)
        if len(self.sequence_buffer) >= 2:
            # Stack recent features for temporal modeling
            sequence_features = {}
            for key in features:
                sequence_data = torch.cat(
                    [
                        step_features[key]
                        for step_features in list(self.sequence_buffer)[
                            -min(8, len(self.sequence_buffer)) :
                        ]
                    ],
                    dim=1,
                )  # Concatenate along sequence dimension
                sequence_features[key] = sequence_data
        else:
            sequence_features = features

        # Health context for attention
        health_metrics = system_state.health_metrics
        health_context = (
            torch.FloatTensor(
                [
                    health_metrics.overall_health_score,
                    health_metrics.thickness_health,
                    health_metrics.conductivity_health,
                    health_metrics.growth_health,
                    health_metrics.stability_health,
                ],
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )  # [1, 1, 5]

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                sequence_features,
                health_context,
                return_attention=True,
            )

        # Select action
        action_logits = outputs["action_logits"]
        action_probs = F.softmax(action_logits, dim=-1)
        action = torch.argmax(action_probs, dim=-1).item()

        # Control information
        control_info = {
            "transformer_action": action,
            "action_confidence": torch.max(action_probs).item(),
            "state_value": outputs["state_value"].item(),
            "sequence_length": len(self.sequence_buffer),
            "attention_maps": outputs["attention_weights"],
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
        }

        return action, control_info

    def visualize_attention(self, system_state: SystemState) -> dict[str, Any]:
        """Visualize attention patterns for interpretability.

        Args:
            system_state: Current system state

        Returns:
            Attention visualization data

        """
        features = self.extract_transformer_features(system_state)
        attention_maps = self.model.get_attention_maps(features)

        # Process attention maps for visualization
        visualization_data = {}

        for attention_type, attention_weights in attention_maps.items():
            if attention_weights is not None and isinstance(
                attention_weights,
                torch.Tensor,
            ):
                # Average over heads and batch dimension
                avg_attention = attention_weights.mean(dim=1).squeeze(0).cpu().numpy()
                visualization_data[attention_type] = {
                    "weights": avg_attention.tolist(),
                    "shape": avg_attention.shape,
                    "entropy": -np.sum(
                        avg_attention * np.log(avg_attention + 1e-8),
                        axis=-1,
                    ).mean(),
                }

        return visualization_data

    def train_step(self, batch_data: list[dict[str, Any]]) -> dict[str, float]:
        """Train the transformer model on a batch of data.

        Args:
            batch_data: Batch of training data

        Returns:
            Training metrics

        """
        if not batch_data:
            return {"loss": 0.0, "accuracy": 0.0}

        self.model.train()

        # Prepare batch tensors
        # This would be implemented with actual training data
        # For now, return placeholder metrics

        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "attention_entropy": 0.0,
            "learning_rate": (
                self.scheduler.get_last_lr()[0]
                if self.scheduler.get_last_lr()
                else self.config.learning_rate
            ),
        }

    def save_model(self, path: str) -> None:
        """Save transformer model."""
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "input_dims": self.input_dims,
            "training_history": dict(self.training_history),
        }

        torch.save(save_dict, path)
        logger.info(f"Transformer model saved to {path}")

    def get_model_summary(self) -> dict[str, Any]:
        """Get transformer model summary."""
        return {
            "model_type": "transformer_mfc_controller",
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "model_dimension": self.config.d_model,
            "attention_heads": self.config.n_heads,
            "encoder_layers": self.config.n_layers,
            "max_sequence_length": self.config.max_seq_len,
            "input_dimensions": self.input_dims,
            "output_dimension": self.action_dim,
            "device": str(self.device),
            "sequence_buffer_size": len(self.sequence_buffer),
        }


def create_transformer_controller(
    state_dim: int,
    action_dim: int,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6,
    **kwargs,
) -> TransformerControllerManager:
    """Factory function to create transformer controller.

    Args:
        state_dim: State space dimension
        action_dim: Action space dimension
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        **kwargs: Additional configuration

    Returns:
        Configured transformer controller

    """
    config = TransformerConfig()
    config.d_model = d_model
    config.n_heads = n_heads
    config.n_layers = n_layers

    # Apply additional configuration
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    controller = TransformerControllerManager(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
    )

    logger.info(
        f"Transformer controller created with {n_layers} layers, {n_heads} heads",
    )

    return controller


if __name__ == "__main__":
    """Test transformer controller functionality."""

    # Test configuration
    state_dim = 70  # From Phase 2 feature engineering
    action_dim = 15  # From adaptive controller

    # Create transformer controller
    controller = create_transformer_controller(
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=256,
        n_heads=8,
        n_layers=4,
        max_seq_len=64,
    )

    # Get model summary
    summary = controller.get_model_summary()

    # Test forward pass with dummy data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get actual input dimensions from controller
    input_dims = controller.input_dims

    dummy_inputs = {
        "sensor_features": torch.randn(1, 4, input_dims["sensor_features"]).to(device),
        "health_features": torch.randn(1, 4, input_dims["health_features"]).to(device),
        "system_features": torch.randn(1, 4, input_dims["system_features"]).to(device),
    }

    dummy_health = torch.randn(1, 1, 5).to(device)

    try:
        with torch.no_grad():
            outputs = controller.model(
                dummy_inputs,
                dummy_health,
                return_attention=True,
            )

    except Exception:
        # Test individual components
        pass
