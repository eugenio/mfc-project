"""
Transfer Learning and Multi-Task Learning Controller for MFC Systems

This module implements advanced transfer learning and multi-task learning
techniques to enable knowledge sharing across different MFC configurations,
bacterial species, and operating conditions.

Key Features:
- Domain adaptation for different bacterial species (G. sulfurreducens, S. oneidensis, mixed cultures)
- Multi-task learning for simultaneous power optimization and biofilm health management
- Few-shot learning for new MFC configurations
- Progressive neural networks for continual learning
- Meta-learning (MAML) for rapid adaptation to new conditions
- Knowledge distillation from expert models
- Cross-domain transfer between laboratory and industrial MFCs

The controller builds upon Phase 2 and Phase 3.1 components to provide
intelligent knowledge transfer and multi-objective optimization.

Created: 2025-07-31
Last Modified: 2025-07-31
"""

import copy
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import Phase 2 and 3.1 components
from adaptive_mfc_controller import SystemState
from deep_rl_controller import DeepRLController
from ml_optimization import FeatureEngineer
from sensing_models.sensor_fusion import BacterialSpecies

# Import configuration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransferLearningMethod(Enum):
    """Transfer learning method types."""
    FINE_TUNING = "fine_tuning"
    DOMAIN_ADAPTATION = "domain_adaptation"
    PROGRESSIVE_NETWORKS = "progressive_networks"
    META_LEARNING = "meta_learning"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    MULTI_TASK = "multi_task"


class TaskType(Enum):
    """Different task types for multi-task learning."""
    POWER_OPTIMIZATION = "power_optimization"
    BIOFILM_HEALTH = "biofilm_health"
    STABILITY_CONTROL = "stability_control"
    EFFICIENCY_MAXIMIZATION = "efficiency_maximization"
    FAULT_DETECTION = "fault_detection"


@dataclass
class TransferConfig:
    """Configuration for transfer learning."""

    # Source and target domains
    source_species: list[BacterialSpecies] = None
    target_species: BacterialSpecies = BacterialSpecies.MIXED

    # Transfer learning parameters
    transfer_method: TransferLearningMethod = TransferLearningMethod.FINE_TUNING
    freeze_layers: list[str] = None  # Layers to freeze during transfer
    adaptation_layers: list[int] = None  # New layers for domain adaptation

    # Multi-task learning
    tasks: list[TaskType] = None
    task_weights: dict[TaskType, float] = None
    shared_layers: list[int] = None
    task_specific_layers: dict[TaskType, list[int]] = None

    # Meta-learning (MAML)
    meta_lr: float = 1e-3
    inner_lr: float = 1e-2
    inner_steps: int = 5
    meta_batch_size: int = 16

    # Progressive networks
    lateral_connections: bool = True
    adapter_layers: list[int] = None

    # Knowledge distillation
    temperature: float = 4.0
    alpha: float = 0.7  # Weight for distillation loss

    def __post_init__(self):
        if self.source_species is None:
            self.source_species = [BacterialSpecies.GEOBACTER, BacterialSpecies.SHEWANELLA]
        if self.freeze_layers is None:
            self.freeze_layers = ['feature_layers']
        if self.adaptation_layers is None:
            self.adaptation_layers = [128, 64]
        if self.tasks is None:
            self.tasks = [TaskType.POWER_OPTIMIZATION, TaskType.BIOFILM_HEALTH]
        if self.task_weights is None:
            self.task_weights = {
                TaskType.POWER_OPTIMIZATION: 0.6,
                TaskType.BIOFILM_HEALTH: 0.4
            }
        if self.shared_layers is None:
            self.shared_layers = [512, 256]
        if self.task_specific_layers is None:
            self.task_specific_layers = {
                TaskType.POWER_OPTIMIZATION: [128, 64],
                TaskType.BIOFILM_HEALTH: [128, 64]
            }
        if self.adapter_layers is None:
            self.adapter_layers = [64, 32]


class DomainAdaptationNetwork(nn.Module):
    """Domain adaptation network with adversarial training."""

    def __init__(self, feature_dim: int, num_domains: int, hidden_dim: int = 128):
        """
        Initialize domain adaptation network.

        Args:
            feature_dim: Dimension of input features
            num_domains: Number of source domains
            hidden_dim: Hidden layer dimension
        """
        super(DomainAdaptationNetwork, self).__init__()

        self.feature_dim = feature_dim
        self.num_domains = num_domains

        # Domain classifier for adversarial training
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_domains),
            nn.LogSoftmax(dim=1)
        )

        # Gradient reversal layer
        self.gradient_reversal = GradientReversalLayer()

        logger.info(f"Domain adaptation network initialized for {num_domains} domains")

    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Forward pass with gradient reversal."""
        reversed_features = self.gradient_reversal(features, alpha)
        domain_pred = self.domain_classifier(reversed_features)
        return domain_pred


class GradientReversalLayer(torch.autograd.Function):
    """Gradient reversal layer for adversarial domain adaptation."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class ProgressiveNetwork(nn.Module):
    """Progressive neural network for continual learning."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int,
                 num_columns: int = 1, lateral_connections: bool = True):
        """
        Initialize progressive network.

        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension
            num_columns: Number of progressive columns
            lateral_connections: Whether to use lateral connections
        """
        super(ProgressiveNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_columns = num_columns
        self.lateral_connections = lateral_connections

        # Progressive columns
        self.columns = nn.ModuleList()
        self.lateral_adapters = nn.ModuleList() if lateral_connections else None

        for col in range(num_columns):
            # Main column layers
            column_layers = []
            layer_input_dim = input_dim

            for i, hidden_dim in enumerate(hidden_dims):
                # Add lateral connection dimension if not the first column
                if lateral_connections and col > 0:
                    lateral_dim = sum(prev_hidden for prev_hidden in hidden_dims[:i+1])
                    layer_input_dim += lateral_dim

                column_layers.append(nn.Linear(layer_input_dim, hidden_dim))
                column_layers.append(nn.ReLU())
                layer_input_dim = hidden_dim

            # Output layer
            column_layers.append(nn.Linear(hidden_dims[-1], output_dim))

            self.columns.append(nn.Sequential(*column_layers))

            # Lateral adapters for this column
            if lateral_connections and col > 0:
                adapters = []
                for layer_idx in range(len(hidden_dims)):
                    adapter_input_dim = sum(hidden_dims[:layer_idx+1])
                    adapter_output_dim = hidden_dims[layer_idx]
                    adapters.append(nn.Linear(adapter_input_dim, adapter_output_dim))
                self.lateral_adapters.append(nn.ModuleList(adapters))

        logger.info(f"Progressive network initialized with {num_columns} columns")

    def forward(self, x: torch.Tensor, column_idx: int = -1) -> torch.Tensor:
        """Forward pass through progressive network."""
        if column_idx == -1:
            column_idx = self.num_columns - 1

        # Store intermediate activations for lateral connections
        prev_activations = []

        # Process each column up to the target column
        for col in range(column_idx + 1):
            col_input = x
            col_activations = []

            # Add lateral connections from previous columns
            if self.lateral_connections and col > 0:
                lateral_inputs = []
                for layer_idx in range(len(self.hidden_dims)):
                    if layer_idx < len(prev_activations[0]):
                        # Collect activations from all previous columns at this layer
                        layer_activations = []
                        for prev_col in range(col):
                            if layer_idx < len(prev_activations[prev_col]):
                                layer_activations.append(prev_activations[prev_col][layer_idx])

                        if layer_activations:
                            lateral_input = torch.cat(layer_activations, dim=-1)
                            adapted_lateral = self.lateral_adapters[col-1][layer_idx](lateral_input)
                            lateral_inputs.append(adapted_lateral)

                # Combine with current layer processing
                current_input = col_input
                layer_idx = 0

                for i, layer in enumerate(self.columns[col]):
                    if isinstance(layer, nn.Linear):
                        if layer_idx < len(lateral_inputs):
                            # Concatenate lateral input
                            current_input = torch.cat([current_input, lateral_inputs[layer_idx]], dim=-1)
                        current_input = layer(current_input)
                        col_activations.append(current_input)
                        layer_idx += 1
                    else:
                        current_input = layer(current_input)

                output = current_input
            else:
                # Standard forward pass
                current_input = col_input
                for layer in self.columns[col]:
                    current_input = layer(current_input)
                    if isinstance(layer, nn.Linear):
                        col_activations.append(current_input)
                output = current_input

            prev_activations.append(col_activations)

        return output

    def add_column(self):
        """Add a new column to the progressive network."""
        new_column_layers = []
        layer_input_dim = self.input_dim

        # Add dimensions for lateral connections from all previous columns
        if self.lateral_connections and self.num_columns > 0:
            for i, hidden_dim in enumerate(self.hidden_dims):
                lateral_dim = sum(prev_hidden for prev_hidden in self.hidden_dims[:i+1]) * self.num_columns
                layer_input_dim += lateral_dim

                new_column_layers.append(nn.Linear(layer_input_dim, hidden_dim))
                new_column_layers.append(nn.ReLU())
                layer_input_dim = hidden_dim
        else:
            for hidden_dim in self.hidden_dims:
                new_column_layers.append(nn.Linear(layer_input_dim, hidden_dim))
                new_column_layers.append(nn.ReLU())
                layer_input_dim = hidden_dim

        # Output layer
        new_column_layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))

        # Add the new column
        self.columns.append(nn.Sequential(*new_column_layers))

        # Add lateral adapters for the new column
        if self.lateral_connections and self.num_columns > 0:
            adapters = []
            for layer_idx in range(len(self.hidden_dims)):
                adapter_input_dim = sum(self.hidden_dims[:layer_idx+1]) * self.num_columns
                adapter_output_dim = self.hidden_dims[layer_idx]
                adapters.append(nn.Linear(adapter_input_dim, adapter_output_dim))
            self.lateral_adapters.append(nn.ModuleList(adapters))

        self.num_columns += 1
        logger.info(f"Added new column. Total columns: {self.num_columns}")


class MultiTaskNetwork(nn.Module):
    """Multi-task neural network with shared and task-specific layers."""

    def __init__(self, input_dim: int, shared_layers: list[int],
                 task_layers: dict[TaskType, list[int]], task_outputs: dict[TaskType, int]):
        """
        Initialize multi-task network.

        Args:
            input_dim: Input dimension
            shared_layers: Shared layer dimensions
            task_layers: Task-specific layer dimensions
            task_outputs: Output dimensions for each task
        """
        super(MultiTaskNetwork, self).__init__()

        self.input_dim = input_dim
        self.tasks = list(task_layers.keys())

        # Shared layers
        shared_modules = []
        layer_input_dim = input_dim

        for hidden_dim in shared_layers:
            shared_modules.append(nn.Linear(layer_input_dim, hidden_dim))
            shared_modules.append(nn.ReLU())
            shared_modules.append(nn.Dropout(0.1))
            layer_input_dim = hidden_dim

        self.shared_layers = nn.Sequential(*shared_modules)

        # Task-specific heads
        self.task_heads = nn.ModuleDict()

        for task, task_dims in task_layers.items():
            task_modules = []
            task_input_dim = shared_layers[-1]

            for hidden_dim in task_dims:
                task_modules.append(nn.Linear(task_input_dim, hidden_dim))
                task_modules.append(nn.ReLU())
                task_modules.append(nn.Dropout(0.1))
                task_input_dim = hidden_dim

            # Output layer for this task
            task_modules.append(nn.Linear(task_dims[-1], task_outputs[task]))

            self.task_heads[task.value] = nn.Sequential(*task_modules)

        logger.info(f"Multi-task network initialized for tasks: {[t.value for t in self.tasks]}")

    def forward(self, x: torch.Tensor, task: TaskType | None = None) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass through multi-task network."""
        # Shared feature extraction
        shared_features = self.shared_layers(x)

        if task is not None:
            # Single task forward pass
            return self.task_heads[task.value](shared_features)
        else:
            # Multi-task forward pass
            outputs = {}
            for task_name, head in self.task_heads.items():
                outputs[task_name] = head(shared_features)
            return outputs


class MAMLController(nn.Module):
    """Model-Agnostic Meta-Learning (MAML) controller for few-shot adaptation."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        """
        Initialize MAML controller.

        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension
        """
        super(MAMLController, self).__init__()

        # Build network
        layers = []
        layer_input_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(layer_input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layer_input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.network = nn.Sequential(*layers)

        logger.info(f"MAML controller initialized: {input_dim}→{hidden_dims}→{output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor,
              inner_lr: float, inner_steps: int) -> 'MAMLController':
        """
        Adapt to new task using support set.

        Args:
            support_x: Support input data
            support_y: Support target data
            inner_lr: Inner loop learning rate
            inner_steps: Number of inner loop steps

        Returns:
            Adapted model
        """
        # Create a copy for adaptation
        adapted_model = copy.deepcopy(self)

        # Inner loop optimization
        for step in range(inner_steps):
            # Forward pass
            pred = adapted_model(support_x)
            loss = F.mse_loss(pred, support_y)

            # Compute gradients
            grads = torch.autograd.grad(loss, adapted_model.parameters(), create_graph=True)

            # Update parameters
            for param, grad in zip(adapted_model.parameters(), grads, strict=False):
                param.data = param.data - inner_lr * grad

        return adapted_model


class TransferLearningController:
    """
    Advanced Transfer Learning and Multi-Task Learning Controller.

    Integrates multiple transfer learning techniques for knowledge sharing
    across different MFC configurations and operating conditions.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 config: TransferConfig | None = None,
                 base_controller: DeepRLController | None = None):
        """
        Initialize transfer learning controller.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            config: Transfer learning configuration
            base_controller: Base deep RL controller for transfer
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or TransferConfig()

        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Base controller for transfer
        self.base_controller = base_controller

        # Knowledge base from different domains
        self.domain_knowledge = {}
        self.task_models = {}

        # Initialize networks based on transfer method
        self._initialize_networks()

        # Feature engineering
        self.feature_engineer = FeatureEngineer()

        # Performance tracking
        self.transfer_performance = defaultdict(list)
        self.adaptation_history = []

        logger.info(f"Transfer learning controller initialized with method: {self.config.transfer_method.value}")
        logger.info(f"Source species: {[s.value for s in self.config.source_species]}")
        logger.info(f"Target species: {self.config.target_species.value}")

    def _initialize_networks(self):
        """Initialize networks based on transfer learning method."""
        if self.config.transfer_method == TransferLearningMethod.DOMAIN_ADAPTATION:
            self.domain_adapter = DomainAdaptationNetwork(
                self.state_dim, len(self.config.source_species) + 1
            ).to(self.device)

        elif self.config.transfer_method == TransferLearningMethod.PROGRESSIVE_NETWORKS:
            self.progressive_net = ProgressiveNetwork(
                self.state_dim, self.config.shared_layers, self.action_dim,
                num_columns=len(self.config.source_species) + 1,
                lateral_connections=self.config.lateral_connections
            ).to(self.device)

        elif self.config.transfer_method == TransferLearningMethod.MULTI_TASK:
            task_outputs = {
                TaskType.POWER_OPTIMIZATION: self.action_dim,
                TaskType.BIOFILM_HEALTH: 1,  # Health score
                TaskType.STABILITY_CONTROL: self.action_dim,
                TaskType.EFFICIENCY_MAXIMIZATION: self.action_dim,
                TaskType.FAULT_DETECTION: 2  # Binary classification
            }

            # Filter task outputs to only included tasks
            filtered_outputs = {task: task_outputs[task] for task in self.config.tasks}

            self.multi_task_net = MultiTaskNetwork(
                self.state_dim, self.config.shared_layers,
                self.config.task_specific_layers, filtered_outputs
            ).to(self.device)

        elif self.config.transfer_method == TransferLearningMethod.META_LEARNING:
            self.maml_controller = MAMLController(
                self.state_dim, self.config.shared_layers, self.action_dim
            ).to(self.device)

            self.meta_optimizer = optim.Adam(
                self.maml_controller.parameters(), lr=self.config.meta_lr
            )

    def load_source_knowledge(self, species: BacterialSpecies, model_path: str):
        """
        Load knowledge from source domain.

        Args:
            species: Source bacterial species
            model_path: Path to source model
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.domain_knowledge[species] = checkpoint
            logger.info(f"Loaded source knowledge for {species.value} from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load source knowledge: {e}")

    def transfer_knowledge(self) -> dict[str, Any]:
        """
        Transfer knowledge from source to target domain.

        Returns:
            Transfer learning results
        """
        if self.config.transfer_method == TransferLearningMethod.FINE_TUNING:
            return self._fine_tune_transfer()
        elif self.config.transfer_method == TransferLearningMethod.DOMAIN_ADAPTATION:
            return self._domain_adaptation_transfer()
        elif self.config.transfer_method == TransferLearningMethod.PROGRESSIVE_NETWORKS:
            return self._progressive_transfer()
        elif self.config.transfer_method == TransferLearningMethod.META_LEARNING:
            return self._meta_learning_transfer()
        elif self.config.transfer_method == TransferLearningMethod.KNOWLEDGE_DISTILLATION:
            return self._knowledge_distillation_transfer()
        else:
            return {'status': 'No transfer method specified'}

    def _fine_tune_transfer(self) -> dict[str, Any]:
        """Fine-tuning transfer from source to target domain."""
        if self.base_controller is None:
            return {'status': 'No base controller for fine-tuning'}

        # Freeze specified layers
        for name, param in self.base_controller.q_network.named_parameters():
            for freeze_layer in self.config.freeze_layers:
                if freeze_layer in name:
                    param.requires_grad = False
                    logger.info(f"Frozen layer: {name}")

        # Add adaptation layers if specified
        if self.config.adaptation_layers:
            # Add new layers for domain adaptation
            original_layers = list(self.base_controller.q_network.children())

            # Insert adaptation layers before the final layer
            adaptation_modules = []
            input_dim = self.config.adaptation_layers[0]

            for hidden_dim in self.config.adaptation_layers:
                adaptation_modules.append(nn.Linear(input_dim, hidden_dim))
                adaptation_modules.append(nn.ReLU())
                input_dim = hidden_dim

            # Combine original and adaptation layers
            combined_layers = original_layers[:-1] + adaptation_modules + [original_layers[-1]]
            self.base_controller.q_network = nn.Sequential(*combined_layers)

        return {
            'status': 'Fine-tuning setup completed',
            'frozen_layers': self.config.freeze_layers,
            'adaptation_layers': self.config.adaptation_layers,
            'trainable_params': sum(p.numel() for p in self.base_controller.q_network.parameters() if p.requires_grad)
        }

    def _domain_adaptation_transfer(self) -> dict[str, Any]:
        """Domain adaptation with adversarial training."""
        if not hasattr(self, 'domain_adapter'):
            return {'status': 'Domain adapter not initialized'}

        # Domain adaptation training would be implemented here
        # This is a placeholder for the full implementation

        return {
            'status': 'Domain adaptation initialized',
            'num_domains': len(self.config.source_species) + 1,
            'adapter_params': sum(p.numel() for p in self.domain_adapter.parameters())
        }

    def _progressive_transfer(self) -> dict[str, Any]:
        """Progressive network transfer learning."""
        if not hasattr(self, 'progressive_net'):
            return {'status': 'Progressive network not initialized'}

        # Add new column for target domain
        if self.progressive_net.num_columns <= len(self.config.source_species):
            self.progressive_net.add_column()

        return {
            'status': 'Progressive network ready',
            'num_columns': self.progressive_net.num_columns,
            'lateral_connections': self.config.lateral_connections,
            'current_column': self.progressive_net.num_columns - 1
        }

    def _meta_learning_transfer(self) -> dict[str, Any]:
        """Meta-learning (MAML) transfer."""
        if not hasattr(self, 'maml_controller'):
            return {'status': 'MAML controller not initialized'}

        return {
            'status': 'MAML controller ready',
            'meta_lr': self.config.meta_lr,
            'inner_lr': self.config.inner_lr,
            'inner_steps': self.config.inner_steps
        }

    def _knowledge_distillation_transfer(self) -> dict[str, Any]:
        """Knowledge distillation from teacher to student model."""
        if self.base_controller is None:
            return {'status': 'No teacher model available'}

        # Knowledge distillation setup
        return {
            'status': 'Knowledge distillation ready',
            'temperature': self.config.temperature,
            'alpha': self.config.alpha,
            'teacher_params': sum(p.numel() for p in self.base_controller.q_network.parameters())
        }

    def adapt_to_new_species(self, target_species: BacterialSpecies,
                           adaptation_data: list[tuple[np.ndarray, int, float]]) -> dict[str, Any]:
        """
        Adapt controller to new bacterial species.

        Args:
            target_species: Target bacterial species
            adaptation_data: Adaptation data (state, action, reward) tuples

        Returns:
            Adaptation results
        """
        logger.info(f"Adapting to new species: {target_species.value}")

        if self.config.transfer_method == TransferLearningMethod.META_LEARNING:
            return self._maml_adaptation(adaptation_data)
        else:
            return self._standard_adaptation(target_species, adaptation_data)

    def _maml_adaptation(self, adaptation_data: list[tuple[np.ndarray, int, float]]) -> dict[str, Any]:
        """Few-shot adaptation using MAML."""
        if len(adaptation_data) < 5:
            return {'status': 'Insufficient adaptation data', 'required': 5, 'provided': len(adaptation_data)}

        # Prepare support set
        states = torch.FloatTensor([d[0] for d in adaptation_data]).to(self.device)
        actions = torch.LongTensor([d[1] for d in adaptation_data]).to(self.device)

        # Convert actions to one-hot for regression target
        targets = F.one_hot(actions, self.action_dim).float()

        # Adapt model
        adapted_model = self.maml_controller.adapt(
            states, targets, self.config.inner_lr, self.config.inner_steps
        )

        # Evaluate adaptation performance
        with torch.no_grad():
            pred = adapted_model(states)
            adaptation_loss = F.mse_loss(pred, targets).item()

        # Store adapted model
        self.adapted_model = adapted_model

        return {
            'status': 'MAML adaptation completed',
            'adaptation_samples': len(adaptation_data),
            'adaptation_loss': adaptation_loss,
            'inner_steps': self.config.inner_steps
        }

    def _standard_adaptation(self, target_species: BacterialSpecies,
                           adaptation_data: list[tuple[np.ndarray, int, float]]) -> dict[str, Any]:
        """Standard adaptation for other methods."""
        self.adaptation_history.append({
            'species': target_species.value,
            'samples': len(adaptation_data),
            'timestamp': torch.datetime.now().isoformat()
        })

        return {
            'status': 'Standard adaptation completed',
            'target_species': target_species.value,
            'adaptation_samples': len(adaptation_data)
        }

    def multi_task_control(self, system_state: SystemState) -> dict[TaskType, Any]:
        """
        Multi-task control decision.

        Args:
            system_state: Current system state

        Returns:
            Task-specific control decisions
        """
        if self.config.transfer_method != TransferLearningMethod.MULTI_TASK:
            return {'error': 'Multi-task network not initialized'}

        # Extract features
        state_features = self.feature_engineer.extract_features(
            system_state, {'power_efficiency': 0.8, 'biofilm_health_score': 0.7}
        )
        state_tensor = torch.FloatTensor(list(state_features.values())).unsqueeze(0).to(self.device)

        # Multi-task forward pass
        with torch.no_grad():
            outputs = self.multi_task_net(state_tensor)

        # Process task-specific outputs
        decisions = {}
        for task in self.config.tasks:
            if task == TaskType.POWER_OPTIMIZATION:
                action_probs = F.softmax(outputs[task.value], dim=1)
                action = torch.argmax(action_probs, dim=1).item()
                decisions[task] = {
                    'action': action,
                    'confidence': torch.max(action_probs).item(),
                    'action_distribution': action_probs.cpu().numpy()[0]
                }
            elif task == TaskType.BIOFILM_HEALTH:
                health_pred = torch.sigmoid(outputs[task.value]).item()
                decisions[task] = {
                    'predicted_health': health_pred,
                    'intervention_needed': health_pred < 0.5
                }
            elif task == TaskType.FAULT_DETECTION:
                fault_probs = F.softmax(outputs[task.value], dim=1)
                fault_detected = torch.argmax(fault_probs, dim=1).item()
                decisions[task] = {
                    'fault_detected': bool(fault_detected),
                    'confidence': torch.max(fault_probs).item()
                }

        return decisions

    def get_transfer_summary(self) -> dict[str, Any]:
        """Get transfer learning summary."""
        return {
            'transfer_method': self.config.transfer_method.value,
            'source_species': [s.value for s in self.config.source_species],
            'target_species': self.config.target_species.value,
            'tasks': [t.value for t in self.config.tasks],
            'domain_knowledge_loaded': list(self.domain_knowledge.keys()),
            'adaptation_history': len(self.adaptation_history),
            'device': str(self.device),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }

    def save_transfer_model(self, path: str):
        """Save transfer learning model."""
        save_dict = {
            'config': self.config,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'transfer_performance': dict(self.transfer_performance),
            'adaptation_history': self.adaptation_history
        }

        # Save method-specific models
        if hasattr(self, 'domain_adapter'):
            save_dict['domain_adapter'] = self.domain_adapter.state_dict()
        if hasattr(self, 'progressive_net'):
            save_dict['progressive_net'] = self.progressive_net.state_dict()
        if hasattr(self, 'multi_task_net'):
            save_dict['multi_task_net'] = self.multi_task_net.state_dict()
        if hasattr(self, 'maml_controller'):
            save_dict['maml_controller'] = self.maml_controller.state_dict()

        torch.save(save_dict, path)
        logger.info(f"Transfer learning model saved to {path}")


def create_transfer_controller(state_dim: int, action_dim: int,
                             method: str = "multi_task",
                             source_species: list[str] = None,
                             target_species: str = "mixed",
                             **kwargs) -> TransferLearningController:
    """
    Factory function to create transfer learning controller.

    Args:
        state_dim: State space dimension
        action_dim: Action space dimension
        method: Transfer learning method
        source_species: Source species names
        target_species: Target species name
        **kwargs: Additional configuration

    Returns:
        Configured transfer learning controller
    """
    # Map species names to enums
    species_map = {
        'geobacter': BacterialSpecies.GEOBACTER,
        'shewanella': BacterialSpecies.SHEWANELLA,
        'mixed': BacterialSpecies.MIXED
    }

    method_map = {
        'fine_tuning': TransferLearningMethod.FINE_TUNING,
        'domain_adaptation': TransferLearningMethod.DOMAIN_ADAPTATION,
        'progressive': TransferLearningMethod.PROGRESSIVE_NETWORKS,
        'meta_learning': TransferLearningMethod.META_LEARNING,
        'knowledge_distillation': TransferLearningMethod.KNOWLEDGE_DISTILLATION,
        'multi_task': TransferLearningMethod.MULTI_TASK
    }

    # Configure transfer learning
    config = TransferConfig()
    config.transfer_method = method_map.get(method.lower(), TransferLearningMethod.MULTI_TASK)
    config.target_species = species_map.get(target_species.lower(), BacterialSpecies.MIXED)

    if source_species:
        config.source_species = [species_map.get(s.lower(), BacterialSpecies.MIXED) for s in source_species]

    # Apply additional configuration
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    controller = TransferLearningController(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config
    )

    logger.info(f"Transfer learning controller created with {method} method")

    return controller


if __name__ == "__main__":
    """Test transfer learning controller functionality."""

    # Test configuration
    state_dim = 70  # From Phase 2 feature engineering
    action_dim = 15  # From adaptive controller

    # Test different transfer learning methods
    methods = ['multi_task', 'progressive', 'meta_learning']

    for method in methods:
        print(f"\n🚀 Testing {method.upper()} Transfer Learning")
        print("=" * 50)

        try:
            controller = create_transfer_controller(
                state_dim=state_dim,
                action_dim=action_dim,
                method=method,
                source_species=['geobacter', 'shewanella'],
                target_species='mixed'
            )

            print(f"✅ {method} controller created successfully")

            # Test transfer
            transfer_result = controller.transfer_knowledge()
            print(f"Transfer result: {transfer_result}")

            # Get summary
            summary = controller.get_transfer_summary()
            print(f"Summary: {summary}")

        except Exception as e:
            print(f"❌ {method} test failed: {e}")

    print("\n🏆 Transfer Learning Controller Tests Completed!")
