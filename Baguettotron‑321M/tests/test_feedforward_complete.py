"""
Complete tests for model/feedforward.py to achieve 95%+ coverage.

This test file adds missing coverage for:
- SwiGLU extra_repr() method
- MLP class with all activation functions
- MLP error handling
- MLP with bias parameter
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baguettotron.model.feedforward import SwiGLU, MLP


class TestSwiGLUComplete:
    """Complete tests for SwiGLU."""

    def test_swiglu_extra_repr(self):
        """Test SwiGLU extra representation for debugging."""
        mlp = SwiGLU(hidden_size=512, intermediate_size=2048)

        repr_str = mlp.extra_repr()

        assert "hidden_size=512" in repr_str
        assert "intermediate_size=2048" in repr_str

    def test_swiglu_string_representation(self):
        """Test that SwiGLU can be converted to string."""
        mlp = SwiGLU(hidden_size=256, intermediate_size=1024)

        str_repr = str(mlp)

        # Should contain class name and extra_repr
        assert "SwiGLU" in str_repr
        assert "256" in str_repr
        assert "1024" in str_repr


class TestMLPActivations:
    """Tests for MLP class with different activation functions."""

    def test_mlp_forward_silu(self):
        """Test standard MLP with SiLU activation."""
        mlp = MLP(hidden_size=512, intermediate_size=2048, activation='silu')

        x = torch.randn(2, 10, 512)
        output = mlp(x)

        assert output.shape == x.shape

        # Verify using SiLU activation
        assert mlp.activation == torch.nn.functional.silu

    def test_mlp_forward_gelu(self):
        """Test MLP with GELU activation."""
        mlp = MLP(hidden_size=256, intermediate_size=1024, activation='gelu')

        x = torch.randn(2, 5, 256)
        output = mlp(x)

        assert output.shape == x.shape

        # Verify using GELU activation
        assert mlp.activation == torch.nn.functional.gelu

    def test_mlp_forward_relu(self):
        """Test MLP with ReLU activation."""
        mlp = MLP(hidden_size=128, intermediate_size=512, activation='relu')

        x = torch.randn(2, 8, 128)
        output = mlp(x)

        assert output.shape == x.shape

        # Verify using ReLU activation
        assert mlp.activation == torch.nn.functional.relu

    def test_mlp_activations_produce_different_outputs(self):
        """Test that different activations produce different outputs."""
        x = torch.randn(1, 5, 64)

        mlp_silu = MLP(hidden_size=64, intermediate_size=128, activation='silu')
        mlp_gelu = MLP(hidden_size=64, intermediate_size=128, activation='gelu')
        mlp_relu = MLP(hidden_size=64, intermediate_size=128, activation='relu')

        # Copy weights to make comparison fair
        with torch.no_grad():
            mlp_gelu.fc1.weight.copy_(mlp_silu.fc1.weight)
            mlp_gelu.fc2.weight.copy_(mlp_silu.fc2.weight)
            mlp_relu.fc1.weight.copy_(mlp_silu.fc1.weight)
            mlp_relu.fc2.weight.copy_(mlp_silu.fc2.weight)

        out_silu = mlp_silu(x)
        out_gelu = mlp_gelu(x)
        out_relu = mlp_relu(x)

        # Outputs should be different due to different activations
        assert not torch.allclose(out_silu, out_gelu, atol=1e-6)
        assert not torch.allclose(out_silu, out_relu, atol=1e-6)


class TestMLPErrorHandling:
    """Tests for MLP error handling."""

    def test_mlp_invalid_activation(self):
        """Test MLP raises error for invalid activation."""
        with pytest.raises(ValueError, match="Unknown activation"):
            MLP(hidden_size=512, intermediate_size=2048, activation='tanh')

    def test_mlp_invalid_activation_types(self):
        """Test various invalid activation names."""
        invalid_activations = ['sigmoid', 'softmax', 'leaky_relu', 'elu', 'invalid']

        for activation in invalid_activations:
            with pytest.raises(ValueError, match="Unknown activation"):
                MLP(hidden_size=256, intermediate_size=1024, activation=activation)


class TestMLPWithBias:
    """Tests for MLP with bias parameter."""

    def test_mlp_with_bias_enabled(self):
        """Test MLP with bias enabled."""
        mlp = MLP(hidden_size=256, intermediate_size=1024, bias=True)

        # Check that bias parameters exist
        assert mlp.fc1.bias is not None
        assert mlp.fc2.bias is not None

        x = torch.randn(2, 10, 256)
        output = mlp(x)

        assert output.shape == x.shape

    def test_mlp_without_bias(self):
        """Test MLP with bias disabled."""
        mlp = MLP(hidden_size=256, intermediate_size=1024, bias=False)

        # Check that bias parameters do NOT exist
        assert mlp.fc1.bias is None
        assert mlp.fc2.bias is None

        x = torch.randn(2, 10, 256)
        output = mlp(x)

        assert output.shape == x.shape

    def test_mlp_bias_affects_output(self):
        """Test that bias actually affects the output."""
        mlp_with_bias = MLP(hidden_size=64, intermediate_size=128, activation='silu', bias=True)
        mlp_no_bias = MLP(hidden_size=64, intermediate_size=128, activation='silu', bias=False)

        # Copy weights
        with torch.no_grad():
            mlp_no_bias.fc1.weight.copy_(mlp_with_bias.fc1.weight)
            mlp_no_bias.fc2.weight.copy_(mlp_with_bias.fc2.weight)

        x = torch.randn(1, 5, 64)

        out_with_bias = mlp_with_bias(x)
        out_no_bias = mlp_no_bias(x)

        # Outputs should be different due to bias
        assert not torch.allclose(out_with_bias, out_no_bias, atol=1e-6)


class TestMLPParameterCounts:
    """Tests for MLP parameter counting."""

    def test_mlp_parameter_count_no_bias(self):
        """Test MLP has correct number of parameters without bias."""
        mlp = MLP(hidden_size=512, intermediate_size=2048, bias=False)

        num_params = sum(p.numel() for p in mlp.parameters())

        # 2 projections without bias
        expected = 512 * 2048 + 2048 * 512
        assert num_params == expected

    def test_mlp_parameter_count_with_bias(self):
        """Test MLP has correct number of parameters with bias."""
        mlp = MLP(hidden_size=512, intermediate_size=2048, bias=True)

        num_params = sum(p.numel() for p in mlp.parameters())

        # 2 projections with bias
        expected = 512 * 2048 + 2048 + 2048 * 512 + 512
        assert num_params == expected

    def test_mlp_vs_swiglu_parameter_difference(self):
        """Test that MLP has fewer parameters than SwiGLU."""
        hidden_size = 512
        intermediate_size = 2048

        mlp = MLP(hidden_size, intermediate_size, bias=False)
        swiglu = SwiGLU(hidden_size, intermediate_size, bias=False)

        mlp_params = sum(p.numel() for p in mlp.parameters())
        swiglu_params = sum(p.numel() for p in swiglu.parameters())

        # SwiGLU has 3 projections, MLP has 2
        assert swiglu_params > mlp_params


class TestMLPGradientFlow:
    """Tests for gradient flow through MLP."""

    def test_mlp_gradient_flow(self):
        """Test that gradients flow through MLP correctly."""
        mlp = MLP(hidden_size=64, intermediate_size=128, activation='silu')

        x = torch.randn(2, 5, 64, requires_grad=True)
        output = mlp(x)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert mlp.fc1.weight.grad is not None
        assert mlp.fc2.weight.grad is not None

    def test_mlp_all_activations_gradient_flow(self):
        """Test gradient flow for all activation functions."""
        activations = ['silu', 'gelu', 'relu']

        for activation in activations:
            mlp = MLP(hidden_size=64, intermediate_size=128, activation=activation)

            x = torch.randn(2, 5, 64, requires_grad=True)
            output = mlp(x)

            loss = output.sum()
            loss.backward()

            # All should have gradients
            assert x.grad is not None
            assert mlp.fc1.weight.grad is not None
            assert mlp.fc2.weight.grad is not None


class TestMLPEdgeCases:
    """Tests for MLP edge cases."""

    def test_mlp_with_tiny_dimensions(self):
        """Test MLP with very small dimensions."""
        mlp = MLP(hidden_size=8, intermediate_size=16, activation='silu')

        x = torch.randn(1, 3, 8)
        output = mlp(x)

        assert output.shape == x.shape

    def test_mlp_with_large_batch(self):
        """Test MLP with large batch size."""
        mlp = MLP(hidden_size=128, intermediate_size=256, activation='gelu')

        x = torch.randn(64, 10, 128)  # Large batch
        output = mlp(x)

        assert output.shape == x.shape

    def test_mlp_with_single_element_batch(self):
        """Test MLP with batch size of 1."""
        mlp = MLP(hidden_size=256, intermediate_size=512, activation='relu')

        x = torch.randn(1, 1, 256)  # Single element
        output = mlp(x)

        assert output.shape == x.shape

    def test_mlp_default_parameters(self):
        """Test MLP with default parameters."""
        mlp = MLP(hidden_size=512, intermediate_size=2048)

        # Default should be silu, no bias
        assert mlp.activation == torch.nn.functional.silu
        assert mlp.fc1.bias is None
        assert mlp.fc2.bias is None

        x = torch.randn(2, 5, 512)
        output = mlp(x)

        assert output.shape == x.shape


class TestSwiGLUVsMLPComparison:
    """Comparison tests between SwiGLU and MLP."""

    def test_swiglu_vs_mlp_output_difference(self):
        """Test that SwiGLU and MLP produce different outputs."""
        hidden_size = 256
        intermediate_size = 512

        swiglu = SwiGLU(hidden_size, intermediate_size, bias=False)
        mlp = MLP(hidden_size, intermediate_size, activation='silu', bias=False)

        x = torch.randn(2, 5, hidden_size)

        out_swiglu = swiglu(x)
        out_mlp = mlp(x)

        # Should have same shape
        assert out_swiglu.shape == out_mlp.shape

        # But different values (different architectures)
        assert not torch.allclose(out_swiglu, out_mlp)

    def test_swiglu_has_more_capacity(self):
        """Test that SwiGLU has more parameters than MLP."""
        hidden_size = 512
        intermediate_size = 2048

        swiglu = SwiGLU(hidden_size, intermediate_size)
        mlp = MLP(hidden_size, intermediate_size)

        swiglu_params = sum(p.numel() for p in swiglu.parameters())
        mlp_params = sum(p.numel() for p in mlp.parameters())

        # SwiGLU should have ~1.5x parameters (3 vs 2 projections)
        ratio = swiglu_params / mlp_params
        assert 1.4 < ratio < 1.6
