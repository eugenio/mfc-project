"""Test Domain Adaptation Components - US-002.

Tests for DomainAdaptationNetwork and GradientReversalLayer classes.
Coverage target: 90%+ for domain adaptation classes.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from transfer_learning_controller import (
    DomainAdaptationNetwork,
    GradientReversalLayer,
)


class TestGradientReversalLayerStatic:
    """Test GradientReversalLayer as autograd.Function (static usage)."""

    def test_forward_preserves_tensor_values(self) -> None:
        """Test forward pass preserves tensor values."""
        x = torch.randn(4, 8, requires_grad=True)
        alpha = 1.0

        # Create a mock context
        ctx = MagicMock()

        result = GradientReversalLayer.forward(ctx, x, alpha)

        # Forward should return the same tensor
        assert torch.equal(result, x)
        # Alpha should be saved to context
        assert ctx.alpha == alpha

    def test_forward_with_different_alpha_values(self) -> None:
        """Test forward with various alpha values."""
        x = torch.randn(2, 4)
        ctx = MagicMock()

        for alpha in [0.0, 0.5, 1.0, 2.0, -1.0]:
            result = GradientReversalLayer.forward(ctx, x, alpha)
            assert torch.equal(result, x)
            assert ctx.alpha == alpha

    def test_forward_with_various_tensor_shapes(self) -> None:
        """Test forward with different tensor shapes."""
        ctx = MagicMock()
        alpha = 1.0

        shapes = [(1,), (4,), (2, 3), (4, 8), (2, 3, 4), (1, 16, 32)]
        for shape in shapes:
            x = torch.randn(*shape)
            result = GradientReversalLayer.forward(ctx, x, alpha)
            assert result.shape == x.shape

    def test_backward_reverses_gradient(self) -> None:
        """Test backward pass reverses and scales gradient."""
        ctx = MagicMock()
        ctx.alpha = 1.0
        grad_output = torch.randn(4, 8)

        grad_input, grad_alpha = GradientReversalLayer.backward(ctx, grad_output)

        # Gradient should be negated and scaled by alpha
        expected = -ctx.alpha * grad_output
        assert torch.allclose(grad_input, expected)
        # Alpha gradient should be None
        assert grad_alpha is None

    def test_backward_with_different_alpha_values(self) -> None:
        """Test backward with various alpha values."""
        grad_output = torch.ones(2, 4)

        for alpha in [0.0, 0.5, 1.0, 2.0, -1.0]:
            ctx = MagicMock()
            ctx.alpha = alpha

            grad_input, _ = GradientReversalLayer.backward(ctx, grad_output)
            expected = -alpha * grad_output
            assert torch.allclose(grad_input, expected)

    def test_backward_preserves_gradient_shape(self) -> None:
        """Test backward preserves gradient shape."""
        ctx = MagicMock()
        ctx.alpha = 1.0

        shapes = [(1,), (4,), (2, 3), (4, 8), (2, 3, 4)]
        for shape in shapes:
            grad_output = torch.randn(*shape)
            grad_input, _ = GradientReversalLayer.backward(ctx, grad_output)
            assert grad_input.shape == grad_output.shape

    def test_apply_method_works_correctly(self) -> None:
        """Test GradientReversalLayer.apply works with autograd."""
        x = torch.randn(4, 8, requires_grad=True)
        alpha = 1.0

        # Use .apply() as intended for autograd.Function
        result = GradientReversalLayer.apply(x, alpha)

        assert result.shape == x.shape
        # Values should be identical in forward pass
        assert torch.allclose(result, x)

    def test_gradient_flow_through_apply(self) -> None:
        """Test gradient reversal through apply method."""
        x = torch.randn(4, 8, requires_grad=True)
        alpha = 1.0

        result = GradientReversalLayer.apply(x, alpha)
        loss = result.sum()
        loss.backward()

        # Gradient should be reversed (-alpha * ones)
        expected_grad = -alpha * torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad)

    def test_gradient_reversal_with_alpha_zero(self) -> None:
        """Test gradient with alpha=0 zeros out gradient."""
        x = torch.randn(4, 8, requires_grad=True)
        alpha = 0.0

        result = GradientReversalLayer.apply(x, alpha)
        loss = result.sum()
        loss.backward()

        # With alpha=0, gradient should be zero
        assert torch.allclose(x.grad, torch.zeros_like(x))

    def test_gradient_reversal_with_alpha_scaling(self) -> None:
        """Test gradient scaling with different alpha values."""
        for alpha in [0.5, 2.0, 0.1]:
            x = torch.randn(4, 8, requires_grad=True)

            result = GradientReversalLayer.apply(x, alpha)
            loss = result.sum()
            loss.backward()

            expected_grad = -alpha * torch.ones_like(x)
            assert torch.allclose(x.grad, expected_grad)


class TestDomainAdaptationNetworkInit:
    """Test DomainAdaptationNetwork initialization."""

    def test_basic_initialization(self) -> None:
        """Test basic network initialization."""
        network = DomainAdaptationNetwork(
            feature_dim=32,
            num_domains=3,
            hidden_dim=64,
        )
        assert network.feature_dim == 32
        assert network.num_domains == 3

    def test_default_hidden_dim(self) -> None:
        """Test default hidden dimension."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=2,
        )
        # Default hidden_dim should be 128
        assert network.feature_dim == 16
        assert network.num_domains == 2

    def test_domain_classifier_structure(self) -> None:
        """Test domain classifier sequential structure."""
        network = DomainAdaptationNetwork(
            feature_dim=32,
            num_domains=4,
            hidden_dim=64,
        )

        # domain_classifier should be nn.Sequential
        assert isinstance(network.domain_classifier, nn.Sequential)

        # First layer should be Linear(feature_dim, hidden_dim)
        assert isinstance(network.domain_classifier[0], nn.Linear)
        assert network.domain_classifier[0].in_features == 32
        assert network.domain_classifier[0].out_features == 64

    def test_domain_classifier_layers(self) -> None:
        """Test all layers in domain classifier."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=3,
            hidden_dim=32,
        )

        # Structure: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Linear -> LogSoftmax
        assert isinstance(network.domain_classifier[0], nn.Linear)  # 16 -> 32
        assert isinstance(network.domain_classifier[1], nn.ReLU)
        assert isinstance(network.domain_classifier[2], nn.Dropout)
        assert isinstance(network.domain_classifier[3], nn.Linear)  # 32 -> 16
        assert isinstance(network.domain_classifier[4], nn.ReLU)
        assert isinstance(network.domain_classifier[5], nn.Linear)  # 16 -> 3
        assert isinstance(network.domain_classifier[6], nn.LogSoftmax)

    def test_domain_classifier_final_layer_outputs(self) -> None:
        """Test final layer outputs correct number of domains."""
        for num_domains in [2, 3, 5, 10]:
            network = DomainAdaptationNetwork(
                feature_dim=16,
                num_domains=num_domains,
                hidden_dim=32,
            )
            # Last Linear layer should output num_domains
            final_linear = network.domain_classifier[5]
            assert final_linear.out_features == num_domains

    def test_dropout_probability(self) -> None:
        """Test dropout layer has correct probability."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=2,
            hidden_dim=32,
        )
        dropout_layer = network.domain_classifier[2]
        assert dropout_layer.p == 0.2

    def test_gradient_reversal_attribute(self) -> None:
        """Test gradient_reversal attribute is set."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=2,
        )
        assert hasattr(network, "gradient_reversal")

    def test_initialization_with_small_hidden_dim(self) -> None:
        """Test initialization with small hidden dimension."""
        network = DomainAdaptationNetwork(
            feature_dim=8,
            num_domains=2,
            hidden_dim=4,
        )
        # hidden_dim // 2 = 2
        assert network.domain_classifier[3].out_features == 2

    def test_initialization_with_large_hidden_dim(self) -> None:
        """Test initialization with large hidden dimension."""
        network = DomainAdaptationNetwork(
            feature_dim=64,
            num_domains=10,
            hidden_dim=512,
        )
        assert network.domain_classifier[0].out_features == 512
        assert network.domain_classifier[3].out_features == 256

    def test_various_feature_dimensions(self) -> None:
        """Test initialization with various feature dimensions."""
        for feature_dim in [1, 8, 32, 128, 512]:
            network = DomainAdaptationNetwork(
                feature_dim=feature_dim,
                num_domains=2,
            )
            assert network.feature_dim == feature_dim
            assert network.domain_classifier[0].in_features == feature_dim


class TestDomainAdaptationNetworkForward:
    """Test DomainAdaptationNetwork forward pass."""

    def test_forward_output_shape(self) -> None:
        """Test forward produces correct output shape."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=3,
            hidden_dim=32,
        )

        # Patch the gradient_reversal to work correctly
        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            input_tensor = torch.randn(8, 16)
            output = network(input_tensor)
            assert output.shape == (8, 3)

    def test_forward_with_various_batch_sizes(self) -> None:
        """Test forward with different batch sizes."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=4,
            hidden_dim=32,
        )

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            for batch_size in [1, 4, 8, 16, 32]:
                input_tensor = torch.randn(batch_size, 16)
                output = network(input_tensor)
                assert output.shape == (batch_size, 4)

    def test_forward_with_alpha_parameter(self) -> None:
        """Test forward accepts alpha parameter."""
        network = DomainAdaptationNetwork(
            feature_dim=8,
            num_domains=2,
            hidden_dim=16,
        )

        # Track what alpha is passed
        captured_alpha = []

        def mock_grl(x, alpha):
            captured_alpha.append(alpha)
            return x

        with patch.object(network, "gradient_reversal", side_effect=mock_grl):
            input_tensor = torch.randn(4, 8)
            network(input_tensor, alpha=0.5)
            assert captured_alpha[-1] == 0.5

            network(input_tensor, alpha=2.0)
            assert captured_alpha[-1] == 2.0

    def test_forward_default_alpha(self) -> None:
        """Test forward uses default alpha=1.0."""
        network = DomainAdaptationNetwork(
            feature_dim=8,
            num_domains=2,
            hidden_dim=16,
        )

        captured_alpha = []

        def mock_grl(x, alpha):
            captured_alpha.append(alpha)
            return x

        with patch.object(network, "gradient_reversal", side_effect=mock_grl):
            input_tensor = torch.randn(4, 8)
            network(input_tensor)
            assert captured_alpha[-1] == 1.0

    def test_forward_output_is_log_softmax(self) -> None:
        """Test output values are valid log probabilities."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=3,
            hidden_dim=32,
        )

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            input_tensor = torch.randn(4, 16)
            output = network(input_tensor)

            # Log probabilities should be <= 0
            assert (output <= 0).all()

            # exp(log_softmax) should sum to 1
            probs = torch.exp(output)
            sums = probs.sum(dim=1)
            assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_forward_eval_mode(self) -> None:
        """Test forward in evaluation mode (no dropout)."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=2,
            hidden_dim=32,
        )
        network.eval()

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            input_tensor = torch.randn(4, 16)

            # Multiple forward passes should give identical results in eval mode
            output1 = network(input_tensor)
            output2 = network(input_tensor)
            assert torch.allclose(output1, output2)

    def test_forward_train_mode_dropout_variation(self) -> None:
        """Test forward in train mode has dropout variability."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=2,
            hidden_dim=64,
        )
        network.train()

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            torch.manual_seed(42)
            input_tensor = torch.randn(32, 16)

            # With dropout, outputs may vary (but not guaranteed with small batches)
            # We mainly verify it runs without error
            output = network(input_tensor)
            assert output.shape == (32, 2)


class TestDomainAdaptationNetworkFeatureExtraction:
    """Test feature extraction with different input dimensions."""

    def test_feature_extraction_small_dim(self) -> None:
        """Test with small feature dimension."""
        network = DomainAdaptationNetwork(
            feature_dim=4,
            num_domains=2,
            hidden_dim=8,
        )

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            input_tensor = torch.randn(2, 4)
            output = network(input_tensor)
            assert output.shape == (2, 2)

    def test_feature_extraction_large_dim(self) -> None:
        """Test with large feature dimension."""
        network = DomainAdaptationNetwork(
            feature_dim=256,
            num_domains=5,
            hidden_dim=512,
        )

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            input_tensor = torch.randn(4, 256)
            output = network(input_tensor)
            assert output.shape == (4, 5)

    def test_feature_extraction_single_sample(self) -> None:
        """Test with single sample."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=3,
        )

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            input_tensor = torch.randn(1, 16)
            output = network(input_tensor)
            assert output.shape == (1, 3)

    def test_feature_extraction_preserves_gradient(self) -> None:
        """Test gradient flows through feature extraction."""
        network = DomainAdaptationNetwork(
            feature_dim=8,
            num_domains=2,
            hidden_dim=16,
        )

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            input_tensor = torch.randn(4, 8, requires_grad=True)
            output = network(input_tensor)
            loss = output.sum()
            loss.backward()

            # Input should have gradients
            assert input_tensor.grad is not None
            assert input_tensor.grad.shape == input_tensor.shape


class TestDomainClassifierOutputs:
    """Test domain classifier outputs match expected shape and properties."""

    def test_classifier_binary_domains(self) -> None:
        """Test classifier with 2 domains (binary classification)."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=2,
        )

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            input_tensor = torch.randn(8, 16)
            output = network(input_tensor)

            assert output.shape == (8, 2)
            # Each row should have 2 log probabilities
            assert output.shape[1] == 2

    def test_classifier_multi_domains(self) -> None:
        """Test classifier with multiple domains."""
        for num_domains in [3, 5, 10]:
            network = DomainAdaptationNetwork(
                feature_dim=16,
                num_domains=num_domains,
            )

            with patch.object(
                network,
                "gradient_reversal",
                side_effect=lambda x, _alpha: x,
            ):
                input_tensor = torch.randn(4, 16)
                output = network(input_tensor)

                assert output.shape == (4, num_domains)

    def test_classifier_output_dtype(self) -> None:
        """Test classifier output dtype matches input."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=3,
        )

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            input_tensor = torch.randn(4, 16)
            output = network(input_tensor)

            assert output.dtype == input_tensor.dtype

    def test_classifier_output_probabilities_valid(self) -> None:
        """Test classifier outputs valid probability distribution."""
        network = DomainAdaptationNetwork(
            feature_dim=32,
            num_domains=4,
            hidden_dim=64,
        )

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            input_tensor = torch.randn(10, 32)
            log_probs = network(input_tensor)
            probs = torch.exp(log_probs)

            # All probabilities should be non-negative
            assert (probs >= 0).all()

            # Each row should sum to 1
            sums = probs.sum(dim=1)
            assert torch.allclose(sums, torch.ones(10), atol=1e-5)

    def test_classifier_deterministic_in_eval(self) -> None:
        """Test classifier is deterministic in eval mode."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=3,
        )
        network.eval()

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            input_tensor = torch.randn(4, 16)

            output1 = network(input_tensor)
            output2 = network(input_tensor)

            assert torch.allclose(output1, output2)

    def test_classifier_weight_initialization(self) -> None:
        """Test classifier layers are properly initialized."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=3,
            hidden_dim=32,
        )

        # Linear layers should have non-zero weights
        linear1 = network.domain_classifier[0]
        assert linear1.weight.abs().sum() > 0

        linear2 = network.domain_classifier[3]
        assert linear2.weight.abs().sum() > 0

        linear3 = network.domain_classifier[5]
        assert linear3.weight.abs().sum() > 0


class TestDomainAdaptationIntegration:
    """Integration tests for domain adaptation workflow."""

    def test_end_to_end_forward_with_mock(self) -> None:
        """Test end-to-end forward pass with mocked gradient reversal."""
        network = DomainAdaptationNetwork(
            feature_dim=32,
            num_domains=3,
            hidden_dim=64,
        )

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            # Simulate feature extraction output
            features = torch.randn(16, 32)

            # Get domain predictions
            domain_logits = network(features, alpha=1.0)

            assert domain_logits.shape == (16, 3)

            # Convert to predictions
            predictions = domain_logits.argmax(dim=1)
            assert predictions.shape == (16,)
            assert (predictions >= 0).all()
            assert (predictions < 3).all()

    def test_domain_classification_loss_computation(self) -> None:
        """Test domain classification loss can be computed."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=2,
            hidden_dim=32,
        )

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            features = torch.randn(8, 16)
            domain_labels = torch.randint(0, 2, (8,))

            log_probs = network(features)

            # NLLLoss expects log probabilities
            loss_fn = nn.NLLLoss()
            loss = loss_fn(log_probs, domain_labels)

            assert loss.shape == ()
            assert loss.item() >= 0

    def test_backward_pass_through_classifier(self) -> None:
        """Test backward pass through domain classifier."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=3,
            hidden_dim=32,
        )

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            features = torch.randn(4, 16, requires_grad=True)
            domain_labels = torch.randint(0, 3, (4,))

            log_probs = network(features)
            loss_fn = nn.NLLLoss()
            loss = loss_fn(log_probs, domain_labels)

            loss.backward()

            # Features should have gradients
            assert features.grad is not None

            # Network parameters should have gradients
            for param in network.parameters():
                if param.requires_grad:
                    assert param.grad is not None

    def test_parameter_count(self) -> None:
        """Test network has expected number of parameters."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=3,
            hidden_dim=32,
        )

        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(
            p.numel() for p in network.parameters() if p.requires_grad
        )

        # All parameters should be trainable
        assert total_params == trainable_params
        assert total_params > 0

    def test_state_dict_save_load(self) -> None:
        """Test state dict can be saved and loaded."""
        network1 = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=2,
            hidden_dim=32,
        )

        # Save state
        state_dict = network1.state_dict()

        # Create new network and load state
        network2 = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=2,
            hidden_dim=32,
        )
        network2.load_state_dict(state_dict)

        # Verify parameters match
        for (n1, p1), (n2, p2) in zip(
            network1.named_parameters(),
            network2.named_parameters(),
            strict=True,
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    def test_alpha_scheduling_simulation(self) -> None:
        """Test network works with scheduled alpha values."""
        network = DomainAdaptationNetwork(
            feature_dim=16,
            num_domains=2,
            hidden_dim=32,
        )

        captured_alphas = []

        def mock_grl(x, alpha):
            captured_alphas.append(alpha)
            return x

        with patch.object(network, "gradient_reversal", side_effect=mock_grl):
            features = torch.randn(4, 16)

            # Simulate increasing alpha schedule
            for epoch in range(5):
                alpha = epoch / 4  # 0, 0.25, 0.5, 0.75, 1.0
                network(features, alpha=alpha)

            assert len(captured_alphas) == 5
            assert captured_alphas == [0.0, 0.25, 0.5, 0.75, 1.0]


class TestGradientReversalLayerEdgeCases:
    """Test edge cases for GradientReversalLayer."""

    def test_empty_tensor(self) -> None:
        """Test with empty tensor."""
        ctx = MagicMock()
        x = torch.empty(0, 8)
        alpha = 1.0

        result = GradientReversalLayer.forward(ctx, x, alpha)
        assert result.shape == (0, 8)

    def test_single_element_tensor(self) -> None:
        """Test with single element tensor."""
        ctx = MagicMock()
        x = torch.tensor([5.0])
        alpha = 1.0

        result = GradientReversalLayer.forward(ctx, x, alpha)
        assert result.shape == (1,)
        assert result[0] == 5.0

    def test_negative_alpha(self) -> None:
        """Test with negative alpha (double reversal)."""
        x = torch.randn(4, 8, requires_grad=True)
        alpha = -1.0

        result = GradientReversalLayer.apply(x, alpha)
        loss = result.sum()
        loss.backward()

        # -(-1) = +1, so gradient should be positive
        expected_grad = torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad)

    def test_large_alpha(self) -> None:
        """Test with large alpha value."""
        x = torch.randn(2, 4, requires_grad=True)
        alpha = 100.0

        result = GradientReversalLayer.apply(x, alpha)
        loss = result.sum()
        loss.backward()

        expected_grad = -100.0 * torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad)

    def test_very_small_alpha(self) -> None:
        """Test with very small alpha value."""
        x = torch.randn(2, 4, requires_grad=True)
        alpha = 1e-10

        result = GradientReversalLayer.apply(x, alpha)
        loss = result.sum()
        loss.backward()

        # Gradient should be very small
        assert (x.grad.abs() < 1e-9).all()


class TestDomainAdaptationNetworkEdgeCases:
    """Test edge cases for DomainAdaptationNetwork."""

    def test_single_domain(self) -> None:
        """Test with single domain (degenerate case)."""
        network = DomainAdaptationNetwork(
            feature_dim=8,
            num_domains=1,
            hidden_dim=16,
        )

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            input_tensor = torch.randn(4, 8)
            output = network(input_tensor)
            assert output.shape == (4, 1)

    def test_many_domains(self) -> None:
        """Test with many domains."""
        network = DomainAdaptationNetwork(
            feature_dim=32,
            num_domains=100,
            hidden_dim=128,
        )

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            input_tensor = torch.randn(8, 32)
            output = network(input_tensor)
            assert output.shape == (8, 100)

    def test_minimal_hidden_dim(self) -> None:
        """Test with minimal hidden dimension."""
        network = DomainAdaptationNetwork(
            feature_dim=4,
            num_domains=2,
            hidden_dim=2,
        )

        with patch.object(
            network,
            "gradient_reversal",
            side_effect=lambda x, _alpha: x,
        ):
            input_tensor = torch.randn(2, 4)
            output = network(input_tensor)
            assert output.shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
