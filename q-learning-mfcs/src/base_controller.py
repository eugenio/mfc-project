"""Base Controller for MFC Control Systems.

This module provides a base class with common functionality shared across
all MFC controllers (DeepRL, Transfer Learning, Transformer, Federated, Adaptive).

Common Features:
- Device selection (GPU/CPU auto-detection)
- Feature engineering integration
- Performance tracking with deque-based history
- Model size and memory estimation utilities
- Model persistence (save/load) helpers
- Logging configuration

Created: 2025-07-31
Last Modified: 2025-07-31
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from adaptive_mfc_controller import SystemState

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BaseControllerConfig:
    """Base configuration shared across all controller types."""

    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 64

    # History tracking
    history_maxlen: int = 1000

    # GPU settings
    device: str | None = None  # 'cuda', 'cpu', 'auto', or None for auto

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


class BaseController(ABC):
    """Abstract base class for all MFC controllers.

    Provides common functionality for device management, feature engineering,
    performance tracking, and model persistence.

    Subclasses must implement:
        - control_step: Execute one control decision
        - train_step: Perform one training step
        - save_model: Save model to disk
        - load_model: Load model from disk
        - get_performance_summary: Return performance metrics
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str | None = None,
        history_maxlen: int = 1000,
    ) -> None:
        """Initialize base controller.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            device: Computing device ('cuda', 'cpu', 'auto', or None for auto)
            history_maxlen: Maximum length for history tracking deques

        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Device selection (common pattern across all controllers)
        self.device = self._select_device(device)

        # Performance tracking (common pattern across all controllers)
        self._history_maxlen = history_maxlen
        self.loss_history: deque = deque(maxlen=history_maxlen)
        self.reward_history: deque = deque(maxlen=history_maxlen)
        self.training_history: deque = deque(maxlen=history_maxlen)

        # Feature engineering (imported lazily to avoid circular imports)
        self._feature_engineer = None

        # Training state
        self.steps = 0
        self.episodes = 0

        logger.info(f"{self.__class__.__name__} initialized on {self.device}")
        logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")

    def _select_device(self, device: str | None) -> torch.device:
        """Select computing device (GPU/CPU).

        This logic is common across all controllers:
        - DeepRLController, TransferLearningController, TransformerControllerManager,
          FederatedClient, FederatedServer all use the same pattern.

        Args:
            device: Device specification ('cuda', 'cpu', 'auto', or None)

        Returns:
            torch.device: Selected computing device

        """
        if device == "auto" or device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @property
    def feature_engineer(self):
        """Lazy-load feature engineer to avoid circular imports.

        All controllers use FeatureEngineer from ml_optimization.
        """
        if self._feature_engineer is None:
            from ml_optimization import FeatureEngineer

            self._feature_engineer = FeatureEngineer()
        return self._feature_engineer

    def extract_state_features(self, system_state: SystemState) -> np.ndarray:
        """Extract features from system state.

        This pattern is common across DeepRLController, TransferLearningController,
        and TransformerControllerManager.

        Args:
            system_state: Complete system state from Phase 2

        Returns:
            Feature vector for neural network

        """
        # Use Phase 2 feature engineering
        performance_metrics = {
            "power_efficiency": system_state.power_output
            / max(system_state.current_density, 0.01),
            "biofilm_health_score": system_state.health_metrics.overall_health_score,
            "sensor_reliability": system_state.fused_measurement.fusion_confidence,
            "system_stability": 1.0 - len(system_state.anomalies) / 10.0,
            "control_confidence": 0.8,  # Default value
        }

        # Extract comprehensive features
        features = self.feature_engineer.extract_features(
            system_state,
            performance_metrics,
        )

        # Convert to numpy array and normalize
        feature_vector = np.array(list(features.values()), dtype=np.float32)

        # Handle NaN and infinite values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)

        # Normalize features to [-1, 1] range (soft normalization)
        return np.tanh(feature_vector / 100.0)

    def get_model_size(self, model: nn.Module | None = None) -> dict[str, Any]:
        """Get model size metrics.

        This pattern is common across TransferLearningController,
        FederatedClient, and FederatedServer.

        Args:
            model: Optional model to measure. If None, uses self.model if available.

        Returns:
            Dictionary with model size metrics

        """
        if model is None:
            if hasattr(self, "model"):
                model = self.model
            elif hasattr(self, "q_network"):
                model = self.q_network
            elif hasattr(self, "policy_network"):
                model = self.policy_network
            else:
                return {"total_parameters": 0, "memory_mb": 0.0, "trainable": 0}

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Estimate memory usage (4 bytes per float32 parameter)
        total_size_mb = total_params * 4 / (1024 * 1024)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "memory_mb": total_size_mb,
        }

    def clip_gradients(self, model: nn.Module, max_norm: float = 1.0) -> float:
        """Clip gradients to bounded norm.

        This pattern is common across DeepRLController and FederatedClient
        (differential privacy gradient clipping).

        Args:
            model: Model to clip gradients for
            max_norm: Maximum gradient norm

        Returns:
            Total gradient norm before clipping

        """
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()

    def prepare_state_tensor(
        self, state: np.ndarray, add_batch_dim: bool = True
    ) -> torch.Tensor:
        """Convert state array to tensor on correct device.

        This pattern is common across all neural network-based controllers.

        Args:
            state: State as numpy array
            add_batch_dim: Whether to add batch dimension

        Returns:
            State tensor on the correct device

        """
        tensor = torch.FloatTensor(state)
        if add_batch_dim:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def log_training_step(
        self,
        loss: float,
        additional_metrics: dict[str, float] | None = None,
    ) -> None:
        """Log training step metrics.

        Common pattern for tracking training progress across all controllers.

        Args:
            loss: Training loss value
            additional_metrics: Optional additional metrics to log

        """
        self.loss_history.append(loss)

        metrics = {"step": self.steps, "loss": loss, "timestamp": datetime.now()}

        if additional_metrics:
            metrics.update(additional_metrics)

        self.training_history.append(metrics)

        if self.steps % 100 == 0:
            avg_loss = (
                np.mean(list(self.loss_history)[-100:]) if self.loss_history else 0.0
            )
            logger.debug(f"Step {self.steps}: avg_loss={avg_loss:.4f}")

    def get_history_summary(self) -> dict[str, Any]:
        """Get summary of training history.

        Common pattern for performance reporting across all controllers.

        Returns:
            Dictionary with history summary statistics

        """
        return {
            "total_steps": self.steps,
            "total_episodes": self.episodes,
            "avg_loss": np.mean(list(self.loss_history)) if self.loss_history else 0.0,
            "recent_loss": (
                np.mean(list(self.loss_history)[-100:]) if self.loss_history else 0.0
            ),
            "avg_reward": (
                np.mean(list(self.reward_history)) if self.reward_history else 0.0
            ),
            "recent_reward": (
                np.mean(list(self.reward_history)[-100:])
                if self.reward_history
                else 0.0
            ),
            "history_length": len(self.training_history),
        }

    def _save_base_state(self) -> dict[str, Any]:
        """Get base state for saving.

        Returns common state that should be saved by all controllers.

        Returns:
            Dictionary with base state

        """
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "steps": self.steps,
            "episodes": self.episodes,
            "device": str(self.device),
            "loss_history": list(self.loss_history),
            "reward_history": list(self.reward_history),
        }

    def _load_base_state(self, state_dict: dict[str, Any]) -> None:
        """Load base state from checkpoint.

        Args:
            state_dict: State dictionary from checkpoint

        """
        self.steps = state_dict.get("steps", 0)
        self.episodes = state_dict.get("episodes", 0)

        if "loss_history" in state_dict:
            self.loss_history = deque(
                state_dict["loss_history"], maxlen=self._history_maxlen
            )
        if "reward_history" in state_dict:
            self.reward_history = deque(
                state_dict["reward_history"], maxlen=self._history_maxlen
            )

    @abstractmethod
    def control_step(self, system_state: SystemState) -> tuple[int, dict[str, Any]]:
        """Execute one control step.

        Args:
            system_state: Current system state

        Returns:
            Tuple of (action, control_info)

        """
        ...

    @abstractmethod
    def train_step(self) -> dict[str, float]:
        """Perform one training step.

        Returns:
            Training metrics dictionary

        """
        ...

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save model and training state.

        Args:
            path: Path to save the model

        """
        ...

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load model and training state.

        Args:
            path: Path to load the model from

        """
        ...

    @abstractmethod
    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary.

        Returns:
            Dictionary with performance metrics

        """
        ...


class NeuralNetworkController(BaseController):
    """Base class for neural network-based controllers.

    Extends BaseController with common neural network functionality shared by
    DeepRLController, TransformerControllerManager, and TransferLearningController.

    Provides:
        - Optimizer management
        - Learning rate scheduling
        - Inference mode context manager
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        device: str | None = None,
        history_maxlen: int = 1000,
    ) -> None:
        """Initialize neural network controller.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            learning_rate: Learning rate for optimizer
            device: Computing device ('cuda', 'cpu', 'auto', or None)
            history_maxlen: Maximum length for history tracking deques

        """
        super().__init__(state_dim, action_dim, device, history_maxlen)

        self.learning_rate = learning_rate

        # These will be set by subclasses
        self.optimizer = None
        self.scheduler = None

    def get_learning_rate(self) -> float:
        """Get current learning rate from scheduler.

        Returns:
            Current learning rate

        """
        if self.scheduler is not None:
            lr_list = self.scheduler.get_last_lr()
            return lr_list[0] if lr_list else self.learning_rate
        return self.learning_rate

    def step_scheduler(self) -> None:
        """Step the learning rate scheduler."""
        if self.scheduler is not None:
            self.scheduler.step()

    def zero_grad(self) -> None:
        """Zero optimizer gradients."""
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def optimizer_step(self) -> None:
        """Perform optimizer step."""
        if self.optimizer is not None:
            self.optimizer.step()


def create_base_config(**kwargs) -> BaseControllerConfig:
    """Factory function to create base controller configuration.

    Args:
        **kwargs: Configuration parameters

    Returns:
        BaseControllerConfig instance

    """
    return BaseControllerConfig(**kwargs)
