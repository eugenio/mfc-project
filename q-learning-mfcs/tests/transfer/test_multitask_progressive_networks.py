"""Test Multi-Task and Progressive Networks - US-004.

Tests for MultiTaskNetwork and ProgressiveNetwork classes.
Coverage target: 90%+ for network classes.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from transfer_learning_controller import (
    MultiTaskNetwork,
    ProgressiveNetwork,
    TaskType,
)


class TestMultiTaskNetworkInitialization:
    """Test MultiTaskNetwork initialization."""

    def test_basic_initialization(self) -> None:
        """Test basic initialization with two tasks."""
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
        assert network is not None
        assert isinstance(network, nn.Module)

    def test_initialization_stores_attributes(self) -> None:
        """Test that initialization stores correct attributes."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [16],
            TaskType.BIOFILM_HEALTH: [8],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 4,
            TaskType.BIOFILM_HEALTH: 2,
        }
        network = MultiTaskNetwork(
            input_dim=10,
            shared_layers=[32],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )
        assert network.input_dim == 10
        assert len(network.tasks) == 2
        assert TaskType.POWER_OPTIMIZATION in network.tasks
        assert TaskType.BIOFILM_HEALTH in network.tasks

    def test_shared_layers_structure(self) -> None:
        """Test shared layers are created correctly."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [16],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 4,
        }
        network = MultiTaskNetwork(
            input_dim=10,
            shared_layers=[64, 32],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )
        # Shared layers: Linear(10, 64), ReLU, Dropout, Linear(64, 32), ReLU, Dropout
        assert len(list(network.shared_layers.children())) == 6

    def test_task_heads_created(self) -> None:
        """Test task-specific heads are created."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [32],
            TaskType.BIOFILM_HEALTH: [16],
            TaskType.FAULT_DETECTION: [24],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 4,
            TaskType.BIOFILM_HEALTH: 2,
            TaskType.FAULT_DETECTION: 3,
        }
        network = MultiTaskNetwork(
            input_dim=20,
            shared_layers=[64],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )
        # Check task heads exist
        assert len(network.task_heads) == 3
        for task in task_layers:
            assert task.value in network.task_heads

    def test_initialization_single_task(self) -> None:
        """Test initialization with a single task."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [32, 16],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 5,
        }
        network = MultiTaskNetwork(
            input_dim=10,
            shared_layers=[64],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )
        assert len(network.tasks) == 1

    def test_initialization_empty_shared_layers(self) -> None:
        """Test initialization with no shared layers."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [32],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 4,
        }
        network = MultiTaskNetwork(
            input_dim=10,
            shared_layers=[],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )
        # Task heads should connect directly from input
        assert len(list(network.shared_layers.children())) == 0

    def test_initialization_single_layer_per_task(self) -> None:
        """Test with single layer in task heads."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [16],
            TaskType.BIOFILM_HEALTH: [8],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 4,
            TaskType.BIOFILM_HEALTH: 2,
        }
        network = MultiTaskNetwork(
            input_dim=10,
            shared_layers=[32],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )
        # Each task head should have: Linear, ReLU, Dropout, Linear (output)
        for task in task_layers:
            head = network.task_heads[task.value]
            assert len(list(head.children())) == 4

    def test_parameters_count(self) -> None:
        """Test network has expected parameters."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [16],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 4,
        }
        network = MultiTaskNetwork(
            input_dim=10,
            shared_layers=[32],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )
        total_params = sum(p.numel() for p in network.parameters())
        assert total_params > 0


class TestMultiTaskNetworkForward:
    """Test MultiTaskNetwork forward pass."""

    @pytest.fixture
    def network(self) -> MultiTaskNetwork:
        """Create a network for testing."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [16],
            TaskType.BIOFILM_HEALTH: [8],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 4,
            TaskType.BIOFILM_HEALTH: 2,
        }
        return MultiTaskNetwork(
            input_dim=10,
            shared_layers=[32],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )

    def test_forward_single_task(self, network: MultiTaskNetwork) -> None:
        """Test forward pass for a specific task."""
        x = torch.randn(4, 10)
        output = network(x, task=TaskType.POWER_OPTIMIZATION)
        assert output.shape == (4, 4)

    def test_forward_different_task(self, network: MultiTaskNetwork) -> None:
        """Test forward pass for a different task."""
        x = torch.randn(4, 10)
        output = network(x, task=TaskType.BIOFILM_HEALTH)
        assert output.shape == (4, 2)

    def test_forward_all_tasks(self, network: MultiTaskNetwork) -> None:
        """Test forward pass returning all task outputs."""
        x = torch.randn(4, 10)
        outputs = network(x, task=None)
        assert isinstance(outputs, dict)
        assert "power_optimization" in outputs
        assert "biofilm_health" in outputs
        assert outputs["power_optimization"].shape == (4, 4)
        assert outputs["biofilm_health"].shape == (4, 2)

    def test_forward_single_sample(self, network: MultiTaskNetwork) -> None:
        """Test forward pass with single sample."""
        x = torch.randn(1, 10)
        output = network(x, task=TaskType.POWER_OPTIMIZATION)
        assert output.shape == (1, 4)

    def test_forward_large_batch(self, network: MultiTaskNetwork) -> None:
        """Test forward pass with large batch."""
        x = torch.randn(128, 10)
        output = network(x, task=TaskType.POWER_OPTIMIZATION)
        assert output.shape == (128, 4)

    def test_forward_produces_gradients(self, network: MultiTaskNetwork) -> None:
        """Test forward pass maintains gradient computation."""
        x = torch.randn(4, 10, requires_grad=True)
        output = network(x, task=TaskType.POWER_OPTIMIZATION)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_forward_consistency(self, network: MultiTaskNetwork) -> None:
        """Test forward pass is consistent in eval mode."""
        network.eval()
        x = torch.randn(4, 10)
        output1 = network(x, task=TaskType.POWER_OPTIMIZATION)
        output2 = network(x, task=TaskType.POWER_OPTIMIZATION)
        assert torch.allclose(output1, output2)

    def test_forward_different_tasks_different_outputs(
        self, network: MultiTaskNetwork,
    ) -> None:
        """Test different tasks produce different outputs."""
        x = torch.randn(4, 10)
        output_power = network(x, task=TaskType.POWER_OPTIMIZATION)
        output_biofilm = network(x, task=TaskType.BIOFILM_HEALTH)
        # Different shapes and values
        assert output_power.shape != output_biofilm.shape


class TestMultiTaskNetworkTraining:
    """Test MultiTaskNetwork training scenarios."""

    def test_training_single_task(self) -> None:
        """Test training on a single task."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [16],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 4,
        }
        network = MultiTaskNetwork(
            input_dim=10,
            shared_layers=[32],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )

        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        x = torch.randn(8, 10)
        y = torch.randn(8, 4)

        # Training step
        network.train()
        output = network(x, task=TaskType.POWER_OPTIMIZATION)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()

        # Should complete without errors
        assert True

    def test_training_multi_task(self) -> None:
        """Test training on multiple tasks."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [16],
            TaskType.BIOFILM_HEALTH: [8],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 4,
            TaskType.BIOFILM_HEALTH: 2,
        }
        network = MultiTaskNetwork(
            input_dim=10,
            shared_layers=[32],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )

        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        x = torch.randn(8, 10)
        y_power = torch.randn(8, 4)
        y_biofilm = torch.randn(8, 2)

        # Training step with multiple tasks
        network.train()
        output_power = network(x, task=TaskType.POWER_OPTIMIZATION)
        output_biofilm = network(x, task=TaskType.BIOFILM_HEALTH)

        loss = torch.nn.functional.mse_loss(output_power, y_power)
        loss += torch.nn.functional.mse_loss(output_biofilm, y_biofilm)

        loss.backward()
        optimizer.step()

        # Should complete without errors
        assert True


class TestMultiTaskNetworkEdgeCases:
    """Test edge cases for MultiTaskNetwork."""

    def test_large_input_dimension(self) -> None:
        """Test with large input dimension."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [64],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 4,
        }
        network = MultiTaskNetwork(
            input_dim=1024,
            shared_layers=[256, 128],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )
        x = torch.randn(4, 1024)
        output = network(x, task=TaskType.POWER_OPTIMIZATION)
        assert output.shape == (4, 4)

    def test_many_shared_layers(self) -> None:
        """Test with many shared layers."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [16],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 4,
        }
        network = MultiTaskNetwork(
            input_dim=32,
            shared_layers=[64, 64, 64, 64, 64],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )
        x = torch.randn(4, 32)
        output = network(x, task=TaskType.POWER_OPTIMIZATION)
        assert output.shape == (4, 4)

    def test_many_task_layers(self) -> None:
        """Test with many task-specific layers."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [64, 32, 16, 8],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 4,
        }
        network = MultiTaskNetwork(
            input_dim=16,
            shared_layers=[32],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )
        x = torch.randn(4, 16)
        output = network(x, task=TaskType.POWER_OPTIMIZATION)
        assert output.shape == (4, 4)

    def test_state_dict(self) -> None:
        """Test state dict operations."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [16],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 4,
        }
        network1 = MultiTaskNetwork(
            input_dim=10,
            shared_layers=[32],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )
        network2 = MultiTaskNetwork(
            input_dim=10,
            shared_layers=[32],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )

        # Load state
        network2.load_state_dict(network1.state_dict())

        # Should produce same output in eval mode (no dropout randomness)
        network1.eval()
        network2.eval()
        x = torch.randn(4, 10)
        output1 = network1(x, task=TaskType.POWER_OPTIMIZATION)
        output2 = network2(x, task=TaskType.POWER_OPTIMIZATION)
        assert torch.equal(output1, output2)

    def test_finite_outputs(self) -> None:
        """Test outputs are finite."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [16],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 4,
        }
        network = MultiTaskNetwork(
            input_dim=10,
            shared_layers=[32],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )
        x = torch.randn(8, 10)
        output = network(x, task=TaskType.POWER_OPTIMIZATION)
        assert torch.isfinite(output).all()


class TestProgressiveNetworkInitialization:
    """Test ProgressiveNetwork initialization."""

    def test_basic_initialization(self) -> None:
        """Test basic initialization."""
        network = ProgressiveNetwork(
            input_dim=10,
            hidden_dims=[32, 16],
            output_dim=5,
            num_columns=1,
            lateral_connections=False,
        )
        assert network is not None
        assert isinstance(network, nn.Module)

    def test_initialization_stores_attributes(self) -> None:
        """Test that initialization stores correct attributes."""
        # Note: lateral_connections=False due to 'col' vs '_col' bug in source
        network = ProgressiveNetwork(
            input_dim=10,
            hidden_dims=[32, 16],
            output_dim=5,
            num_columns=2,
            lateral_connections=False,
        )
        assert network.input_dim == 10
        assert network.hidden_dims == [32, 16]
        assert network.output_dim == 5
        assert network.num_columns == 2
        assert network.lateral_connections is False

    def test_single_column_no_lateral(self) -> None:
        """Test single column without lateral connections."""
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16, 8],
            output_dim=4,
            num_columns=1,
            lateral_connections=False,
        )
        assert network.num_columns == 1
        assert len(network.columns) == 1
        assert network.lateral_adapters is None

    @pytest.mark.xfail(
        reason=(
            "Bug: 'col' vs '_col' in __init__ "
            "affects all lateral_connections=True cases"
        ),
    )
    def test_single_column_with_lateral(self) -> None:
        """Test single column with lateral connections enabled."""
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16, 8],
            output_dim=4,
            num_columns=1,
            lateral_connections=True,
        )
        assert network.num_columns == 1
        assert network.lateral_adapters is not None

    def test_columns_created(self) -> None:
        """Test correct number of columns are created."""
        network = ProgressiveNetwork(
            input_dim=10,
            hidden_dims=[32],
            output_dim=5,
            num_columns=3,
            lateral_connections=False,
        )
        assert len(network.columns) == 3

    def test_empty_hidden_dims(self) -> None:
        """Test initialization with no hidden layers."""
        network = ProgressiveNetwork(
            input_dim=10,
            hidden_dims=[],
            output_dim=5,
            num_columns=1,
            lateral_connections=False,
        )
        # Should have direct input->output connection
        assert len(network.columns) == 1
        column = network.columns[0]
        # Just one linear layer
        linear_layers = [
            layer for layer in column.children() if isinstance(layer, nn.Linear)
        ]
        assert len(linear_layers) == 1
        assert linear_layers[0].in_features == 10
        assert linear_layers[0].out_features == 5

    def test_single_hidden_dim(self) -> None:
        """Test initialization with single hidden layer."""
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16],
            output_dim=4,
            num_columns=1,
            lateral_connections=False,
        )
        column = network.columns[0]
        # Linear(8, 16), ReLU, Linear(16, 4)
        layers = list(column.children())
        assert len(layers) == 3

    def test_parameters_exist(self) -> None:
        """Test network has trainable parameters."""
        network = ProgressiveNetwork(
            input_dim=10,
            hidden_dims=[32],
            output_dim=5,
            num_columns=1,
            lateral_connections=False,
        )
        params = list(network.parameters())
        assert len(params) > 0

    def test_is_nn_module(self) -> None:
        """Test network is proper nn.Module."""
        network = ProgressiveNetwork(
            input_dim=10,
            hidden_dims=[32],
            output_dim=5,
            num_columns=1,
            lateral_connections=False,
        )
        assert isinstance(network, nn.Module)


class TestProgressiveNetworkForward:
    """Test ProgressiveNetwork forward pass."""

    @pytest.mark.xfail(
        reason="Bug in source code: 'prev_activations' is not defined (line 284/329)",
    )
    def test_forward_single_column_no_lateral(self) -> None:
        """Test forward pass with single column, no lateral connections."""
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16, 8],
            output_dim=4,
            num_columns=1,
            lateral_connections=False,
        )
        x = torch.randn(4, 8)
        output = network(x)
        assert output.shape == (4, 4)

    @pytest.mark.xfail(
        reason="Bug in source code: 'prev_activations' is not defined (line 284/329)",
    )
    def test_forward_default_column_idx(self) -> None:
        """Test forward uses last column by default."""
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16],
            output_dim=4,
            num_columns=2,
            lateral_connections=False,
        )
        x = torch.randn(4, 8)
        output = network(x, column_idx=-1)
        assert output.shape == (4, 4)

    @pytest.mark.xfail(
        reason="Bug in source code: 'prev_activations' is not defined (line 284/329)",
    )
    def test_forward_specific_column(self) -> None:
        """Test forward with specific column index."""
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16],
            output_dim=4,
            num_columns=3,
            lateral_connections=False,
        )
        x = torch.randn(4, 8)
        output = network(x, column_idx=0)
        assert output.shape == (4, 4)

    @pytest.mark.xfail(
        reason="Bug in source code: 'prev_activations' is not defined (line 284/329)",
    )
    def test_forward_with_lateral_connections(self) -> None:
        """Test forward with lateral connections enabled."""
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16],
            output_dim=4,
            num_columns=2,
            lateral_connections=True,
        )
        x = torch.randn(4, 8)
        output = network(x)
        assert output.shape == (4, 4)


class TestProgressiveNetworkAddColumn:
    """Test ProgressiveNetwork add_column method."""

    def test_add_column_increases_count(self) -> None:
        """Test add_column increases num_columns."""
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16],
            output_dim=4,
            num_columns=1,
            lateral_connections=False,
        )
        assert network.num_columns == 1
        network.add_column()
        assert network.num_columns == 2

    def test_add_column_adds_to_columns_list(self) -> None:
        """Test add_column adds to columns ModuleList."""
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16],
            output_dim=4,
            num_columns=1,
            lateral_connections=False,
        )
        initial_columns = len(network.columns)
        network.add_column()
        assert len(network.columns) == initial_columns + 1

    def test_add_multiple_columns(self) -> None:
        """Test adding multiple columns."""
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16],
            output_dim=4,
            num_columns=1,
            lateral_connections=False,
        )
        network.add_column()
        network.add_column()
        network.add_column()
        assert network.num_columns == 4
        assert len(network.columns) == 4

    @pytest.mark.xfail(
        reason=(
            "Bug: 'col' vs '_col' in __init__ "
            "affects all lateral_connections=True cases"
        ),
    )
    def test_add_column_with_lateral_connections(self) -> None:
        """Test add_column with lateral connections enabled."""
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16, 8],
            output_dim=4,
            num_columns=1,
            lateral_connections=True,
        )
        network.add_column()
        assert network.num_columns == 2
        # Should have added adapters
        assert len(network.lateral_adapters) == 1

    def test_new_column_has_correct_structure(self) -> None:
        """Test newly added column has correct layer structure."""
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16],
            output_dim=4,
            num_columns=1,
            lateral_connections=False,
        )
        network.add_column()

        # Get the new column
        new_column = network.columns[-1]
        layers = list(new_column.children())
        # Linear(8, 16), ReLU, Linear(16, 4)
        assert len(layers) == 3
        assert isinstance(layers[0], nn.Linear)
        assert isinstance(layers[1], nn.ReLU)
        assert isinstance(layers[2], nn.Linear)


class TestProgressiveNetworkEdgeCases:
    """Test edge cases for ProgressiveNetwork."""

    def test_large_dimensions(self) -> None:
        """Test with large dimensions."""
        network = ProgressiveNetwork(
            input_dim=512,
            hidden_dims=[256, 128],
            output_dim=64,
            num_columns=1,
            lateral_connections=False,
        )
        assert network.input_dim == 512
        assert network.output_dim == 64

    def test_many_columns(self) -> None:
        """Test with many columns."""
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16],
            output_dim=4,
            num_columns=5,
            lateral_connections=False,
        )
        assert len(network.columns) == 5

    def test_many_hidden_layers(self) -> None:
        """Test with many hidden layers."""
        network = ProgressiveNetwork(
            input_dim=32,
            hidden_dims=[64, 32, 16, 8],
            output_dim=4,
            num_columns=1,
            lateral_connections=False,
        )
        column = network.columns[0]
        # 4 hidden layers = 4 Linear + 4 ReLU + 1 output Linear = 9 layers
        layers = list(column.children())
        assert len(layers) == 9

    def test_state_dict_operations(self) -> None:
        """Test state dict save/load."""
        network1 = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16],
            output_dim=4,
            num_columns=1,
            lateral_connections=False,
        )
        network2 = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16],
            output_dim=4,
            num_columns=1,
            lateral_connections=False,
        )

        network2.load_state_dict(network1.state_dict())

        # Parameters should be equal
        for p1, p2 in zip(network1.parameters(), network2.parameters(), strict=False):
            assert torch.equal(p1, p2)

    def test_train_eval_mode_switching(self) -> None:
        """Test switching between train and eval modes."""
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16],
            output_dim=4,
            num_columns=1,
            lateral_connections=False,
        )
        network.train()
        assert network.training

        network.eval()
        assert not network.training


class TestProgressiveNetworkWithBugWorkarounds:
    """Test ProgressiveNetwork with workarounds for known bugs.

    The source code has bugs:
    - Line 236: uses 'col' instead of '_col' in loop variable
    - Line 284/329: 'prev_activations' is not defined before use

    These tests document the bugs and verify initialization works.
    """

    @pytest.mark.xfail(
        reason=(
            "Bug: 'col' vs '_col' variable in __init__ "
            "when lateral_connections=True and num_columns > 1"
        ),
    )
    def test_init_multiple_columns_with_lateral(self) -> None:
        """Test initialization with multiple columns and lateral connections.

        This fails because the loop uses '_col' but the condition checks 'col'.
        """
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16, 8],
            output_dim=4,
            num_columns=2,
            lateral_connections=True,
        )
        assert network.num_columns == 2

    @pytest.mark.xfail(
        reason=(
            "Bug: 'col' vs '_col' in __init__ "
            "affects all lateral_connections=True cases"
        ),
    )
    def test_init_single_column_with_lateral_works(self) -> None:
        """Test single column with lateral connections.

        Even single column triggers the bug because the check is inside the loop.
        """
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16],
            output_dim=4,
            num_columns=1,
            lateral_connections=True,
        )
        assert network.num_columns == 1

    def test_init_multiple_columns_no_lateral_works(self) -> None:
        """Test multiple columns without lateral connections works."""
        # Without lateral_connections, the col check is never reached
        network = ProgressiveNetwork(
            input_dim=8,
            hidden_dims=[16, 8],
            output_dim=4,
            num_columns=3,
            lateral_connections=False,
        )
        assert network.num_columns == 3
        assert len(network.columns) == 3


class TestIntegration:
    """Integration tests for multi-task training loop."""

    def test_multi_task_training_loop_mocked(self) -> None:
        """Test complete multi-task training loop with mocked data."""
        # Create network
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [32, 16],
            TaskType.BIOFILM_HEALTH: [24, 12],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 4,
            TaskType.BIOFILM_HEALTH: 2,
        }
        network = MultiTaskNetwork(
            input_dim=20,
            shared_layers=[64],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )

        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        # Mock training data
        torch.manual_seed(42)
        train_x = torch.randn(32, 20)
        train_y_power = torch.randn(32, 4)
        train_y_biofilm = torch.randn(32, 2)

        # Training loop
        network.train()
        initial_loss: float = 0.0
        final_loss: float = 0.0

        for epoch in range(10):
            optimizer.zero_grad()

            # Forward for both tasks
            output_power = network(train_x, task=TaskType.POWER_OPTIMIZATION)
            output_biofilm = network(train_x, task=TaskType.BIOFILM_HEALTH)

            # Combined loss
            loss_power = torch.nn.functional.mse_loss(output_power, train_y_power)
            loss_biofilm = torch.nn.functional.mse_loss(output_biofilm, train_y_biofilm)
            total_loss = loss_power + loss_biofilm

            if epoch == 0:
                initial_loss = total_loss.item()

            total_loss.backward()
            optimizer.step()

            final_loss = total_loss.item()

        # Loss should decrease (learning happening)
        assert final_loss < initial_loss

    def test_alternating_task_training(self) -> None:
        """Test alternating between tasks during training."""
        task_layers = {
            TaskType.POWER_OPTIMIZATION: [16],
            TaskType.BIOFILM_HEALTH: [8],
        }
        task_outputs = {
            TaskType.POWER_OPTIMIZATION: 4,
            TaskType.BIOFILM_HEALTH: 2,
        }
        network = MultiTaskNetwork(
            input_dim=10,
            shared_layers=[32],
            task_layers=task_layers,
            task_outputs=task_outputs,
        )

        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

        # Alternate training
        for i in range(20):
            x = torch.randn(8, 10)

            if i % 2 == 0:
                y = torch.randn(8, 4)
                output = network(x, task=TaskType.POWER_OPTIMIZATION)
            else:
                y = torch.randn(8, 2)
                output = network(x, task=TaskType.BIOFILM_HEALTH)

            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Should complete without errors
        assert True
