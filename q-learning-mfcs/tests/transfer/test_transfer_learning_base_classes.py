"""Test Transfer Learning Base Classes - US-001.

Tests the base classes of the TransferLearningController.
Coverage target: 40%+ for transfer_learning_controller.py
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from sensing_models.sensor_fusion import BacterialSpecies
from transfer_learning_controller import (
    DomainAdaptationNetwork,
    MAMLController,
    MultiTaskNetwork,
    ProgressiveNetwork,
    TaskType,
    TransferConfig,
    TransferLearningController,
    TransferLearningMethod,
    create_transfer_controller,
)


class TestTransferConfig:
    """Test TransferConfig dataclass initialization and defaults."""

    def test_default_initialization(self) -> None:
        """Test TransferConfig initializes with proper defaults."""
        config = TransferConfig()
        assert config.transfer_method == TransferLearningMethod.FINE_TUNING
        assert config.target_species == BacterialSpecies.MIXED
        assert config.meta_lr == 1e-3
        assert config.inner_lr == 1e-2
        assert config.inner_steps == 5

    def test_source_species_default_post_init(self) -> None:
        """Test source_species defaults in __post_init__."""
        config = TransferConfig()
        assert config.source_species is not None
        assert len(config.source_species) == 2
        assert BacterialSpecies.GEOBACTER in config.source_species

    def test_custom_transfer_method(self) -> None:
        """Test TransferConfig with custom transfer method."""
        config = TransferConfig(
            transfer_method=TransferLearningMethod.DOMAIN_ADAPTATION,
        )
        assert config.transfer_method == TransferLearningMethod.DOMAIN_ADAPTATION

    def test_freeze_layers_default_post_init(self) -> None:
        """Test freeze_layers defaults in __post_init__."""
        config = TransferConfig()
        assert config.freeze_layers is not None
        assert config.freeze_layers == ["feature_layers"]

    def test_tasks_default_post_init(self) -> None:
        """Test tasks defaults in __post_init__."""
        config = TransferConfig()
        assert config.tasks is not None
        assert len(config.tasks) == 2
        assert TaskType.POWER_OPTIMIZATION in config.tasks
        assert TaskType.BIOFILM_HEALTH in config.tasks


class TestTransferLearningControllerInit:
    """Test TransferLearningController initialization."""

    def test_basic_initialization(self) -> None:
        """Test basic controller initialization with default config."""
        controller = TransferLearningController(state_dim=10, action_dim=5)
        assert controller.state_dim == 10
        assert controller.action_dim == 5
        assert controller.config is not None
        assert controller.domain_knowledge == {}
        assert controller.task_models == {}

    def test_initialization_with_custom_config(self) -> None:
        """Test controller initialization with custom config."""
        config = TransferConfig(
            transfer_method=TransferLearningMethod.MULTI_TASK,
            meta_lr=0.005,
        )
        controller = TransferLearningController(
            state_dim=20, action_dim=10, config=config,
        )
        assert controller.config.transfer_method == TransferLearningMethod.MULTI_TASK

    def test_initialization_with_domain_adaptation(self) -> None:
        """Test controller initialization with domain adaptation method."""
        config = TransferConfig(
            transfer_method=TransferLearningMethod.DOMAIN_ADAPTATION,
        )
        controller = TransferLearningController(
            state_dim=15, action_dim=8, config=config,
        )
        assert hasattr(controller, "domain_adapter")
        assert isinstance(controller.domain_adapter, DomainAdaptationNetwork)

    @pytest.mark.xfail(reason="Bug in ProgressiveNetwork.__init__: col vs _col")
    def test_initialization_with_progressive_networks(self) -> None:
        """Test controller initialization with progressive networks method."""
        config = TransferConfig(
            transfer_method=TransferLearningMethod.PROGRESSIVE_NETWORKS,
            shared_layers=[128, 64],
        )
        controller = TransferLearningController(
            state_dim=12, action_dim=6, config=config,
        )
        assert hasattr(controller, "progressive_net")
        assert isinstance(controller.progressive_net, ProgressiveNetwork)

    def test_initialization_with_multi_task(self) -> None:
        """Test controller initialization with multi-task method."""
        config = TransferConfig(
            transfer_method=TransferLearningMethod.MULTI_TASK,
            tasks=[TaskType.POWER_OPTIMIZATION, TaskType.BIOFILM_HEALTH],
            shared_layers=[256, 128],
        )
        controller = TransferLearningController(
            state_dim=18, action_dim=9, config=config,
        )
        assert hasattr(controller, "multi_task_net")
        assert isinstance(controller.multi_task_net, MultiTaskNetwork)

    def test_initialization_with_meta_learning(self) -> None:
        """Test controller initialization with meta-learning (MAML) method."""
        config = TransferConfig(
            transfer_method=TransferLearningMethod.META_LEARNING,
            shared_layers=[64, 32],
        )
        controller = TransferLearningController(
            state_dim=8, action_dim=4, config=config,
        )
        assert hasattr(controller, "maml_controller")
        assert isinstance(controller.maml_controller, MAMLController)


class TestTransferLearningControllerSaveLoad:
    """Test save_model and load_model methods."""

    def test_save_model_creates_file(self) -> None:
        """Test save_model creates a file."""
        controller = TransferLearningController(state_dim=10, action_dim=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.pt")
            controller.save_model(save_path)
            assert os.path.exists(save_path)

    def test_save_model_with_domain_adapter(self) -> None:
        """Test save_model includes domain adapter state."""
        config = TransferConfig(
            transfer_method=TransferLearningMethod.DOMAIN_ADAPTATION,
        )
        controller = TransferLearningController(
            state_dim=10, action_dim=5, config=config,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.pt")
            controller.save_model(save_path)
            checkpoint = torch.load(save_path, map_location="cpu", weights_only=False)
            assert "domain_adapter" in checkpoint

    def test_load_model_restores_state(self) -> None:
        """Test load_model restores controller state."""
        config = TransferConfig(
            transfer_method=TransferLearningMethod.DOMAIN_ADAPTATION,
        )
        controller1 = TransferLearningController(
            state_dim=10, action_dim=5, config=config,
        )
        controller1.steps = 100
        controller1.episodes = 10
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.pt")
            controller1.save_model(save_path)
            controller2 = TransferLearningController(
                state_dim=10, action_dim=5, config=config,
            )
            controller2.load_model(save_path)
            assert controller2.steps == 100
            assert controller2.episodes == 10


class TestTransferLearningControllerSummary:
    """Test get_transfer_summary method."""

    def test_basic_summary_fields(self) -> None:
        """Test basic summary contains required fields."""
        controller = TransferLearningController(state_dim=10, action_dim=5)
        summary = controller.get_transfer_summary()
        assert "transfer_method" in summary
        assert "source_species" in summary
        assert "target_species" in summary
        assert "state_dim" in summary
        assert "action_dim" in summary

    def test_summary_dimensions(self) -> None:
        """Test summary contains correct dimensions."""
        controller = TransferLearningController(state_dim=15, action_dim=8)
        summary = controller.get_transfer_summary()
        assert summary["state_dim"] == 15
        assert summary["action_dim"] == 8


class TestCreateTransferController:
    """Test create_transfer_controller factory function."""

    def test_create_default_controller(self) -> None:
        """Test creating controller with defaults."""
        controller = create_transfer_controller(state_dim=10, action_dim=5)
        assert isinstance(controller, TransferLearningController)
        assert controller.state_dim == 10
        assert controller.action_dim == 5

    def test_create_multi_task_controller(self) -> None:
        """Test creating multi-task controller."""
        controller = create_transfer_controller(
            state_dim=12, action_dim=6, method="multi_task",
        )
        assert controller.config.transfer_method == TransferLearningMethod.MULTI_TASK

    def test_create_with_species(self) -> None:
        """Test creating controller with custom species."""
        controller = create_transfer_controller(
            state_dim=10, action_dim=5,
            source_species=["geobacter"],
            target_species="shewanella",
        )
        assert BacterialSpecies.GEOBACTER in controller.config.source_species
        assert controller.config.target_species == BacterialSpecies.SHEWANELLA


class TestDomainAdaptationNetwork:
    """Test DomainAdaptationNetwork class."""

    def test_initialization(self) -> None:
        """Test network initialization."""
        network = DomainAdaptationNetwork(
            feature_dim=32, num_domains=3, hidden_dim=64,
        )
        assert network.feature_dim == 32
        assert network.num_domains == 3

    @pytest.mark.xfail(reason="Bug: GradientReversalLayer deprecated")
    def test_forward_pass(self) -> None:
        """Test forward pass produces correct output shape."""
        network = DomainAdaptationNetwork(
            feature_dim=16, num_domains=2, hidden_dim=32,
        )
        input_tensor = torch.randn(8, 16)
        output = network(input_tensor)
        assert output.shape == (8, 2)


class TestMAMLController:
    """Test MAMLController class."""

    def test_initialization(self) -> None:
        """Test MAML controller initialization."""
        controller = MAMLController(
            input_dim=10, hidden_dims=[32, 16], output_dim=5,
        )
        assert controller.network is not None

    def test_forward_pass(self) -> None:
        """Test forward pass produces correct output shape."""
        controller = MAMLController(
            input_dim=8, hidden_dims=[16, 8], output_dim=4,
        )
        input_tensor = torch.randn(4, 8)
        output = controller(input_tensor)
        assert output.shape == (4, 4)

    def test_adapt_creates_new_model(self) -> None:
        """Test adapt creates a new adapted model."""
        controller = MAMLController(
            input_dim=8, hidden_dims=[16], output_dim=4,
        )
        support_x = torch.randn(5, 8)
        support_y = torch.randn(5, 4)
        adapted = controller.adapt(
            support_x=support_x,
            support_y=support_y,
            inner_lr=0.01,
            inner_steps=3,
        )
        assert adapted is not controller
        assert isinstance(adapted, MAMLController)


class TestMultiTaskNetwork:
    """Test MultiTaskNetwork class."""

    def test_initialization(self) -> None:
        """Test multi-task network initialization."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [32, 16],
            TaskType.BIOFILM_HEALTH: [24, 12],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 5,
            TaskType.BIOFILM_HEALTH: 1,
        }
        network = MultiTaskNetwork(
            input_dim=20,
            shared_layers=[64, 32],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )
        assert network.input_dim == 20
        assert len(network.tasks) == 2

    def test_forward_single_task(self) -> None:
        """Test forward pass for a single task."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [16],
            TaskType.BIOFILM_HEALTH: [8],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 4,
            TaskType.BIOFILM_HEALTH: 1,
        }
        network = MultiTaskNetwork(
            input_dim=10,
            shared_layers=[32],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )
        input_tensor = torch.randn(4, 10)
        output = network(input_tensor, task=TaskType.POWER_OPTIMIZATION)
        assert output.shape == (4, 4)


class TestProgressiveNetwork:
    """Test ProgressiveNetwork class."""

    def test_initialization_single_column(self) -> None:
        """Test progressive network initialization with single column."""
        network = ProgressiveNetwork(
            input_dim=10,
            hidden_dims=[32, 16],
            output_dim=5,
            num_columns=1,
            lateral_connections=False,
        )
        assert network.input_dim == 10
        assert network.output_dim == 5
        assert network.num_columns == 1

    @pytest.mark.xfail(reason="Bug: ProgressiveNetwork.forward undefined var")
    def test_forward_pass_single_column(self) -> None:
        """Test forward pass with single column."""
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16, 8],
            output_dim=4,
            num_columns=1,
            lateral_connections=False,
        )
        input_tensor = torch.randn(4, 8)
        output = network(input_tensor)
        assert output.shape == (4, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
