"""
Advanced tests for feedforward.py to achieve 95%+ coverage.

Tests MLP class with different activation functions and edge cases.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baguettotron.model.feedforward import MLP, SwiGLU


class TestMLPActivations:
    """Tests for MLP with different activation functions."""

    def test_mlp_silu_activation(self):
        """Test MLP with SiLU activation."""
        mlp = MLP(
            hidden_size=512,
            intermediate_size=2048,
            activation='silu',
            bias=False
        )

        x = torch.randn(2, 10, 512)
        output = mlp(x)

        assert output.shape == x.shape
        assert mlp.activation == torch.nn.functional.silu

    def test_mlp_gelu_activation(self):
        """Test MLP with GELU activation."""
        mlp = MLP(
            hidden_size=512,
            intermediate_size=2048,
            activation='gelu',
            bias=False
        )

        x = torch.randn(2, 10, 512)
        output = mlp(x)

        assert output.shape == x.shape
        assert mlp.activation == torch.nn.functional.gelu

    def test_mlp_relu_activation(self):
        """Test MLP with ReLU activation."""
        mlp = MLP(
            hidden_size=512,
            intermediate_size=2048,
            activation='relu',
            bias=False
        )

        x = torch.randn(2, 10, 512)
        output = mlp(x)

        assert output.shape == x.shape
        assert mlp.activation == torch.nn.functional.relu

    def test_mlp_invalid_activation(self):
        """Test MLP with invalid activation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            MLP(
                hidden_size=512,
                intermediate_size=2048,
                activation='invalid_activation'
            )

    def test_mlp_with_bias(self):
        """Test MLP with bias enabled."""
        mlp = MLP(
            hidden_size=512,
            intermediate_size=2048,
            activation='silu',
            bias=True
        )

        # Check that bias exists
        assert mlp.fc1.bias is not None
        assert mlp.fc2.bias is not None

        x = torch.randn(2, 10, 512)
        output = mlp(x)
        assert output.shape == x.shape

    def test_mlp_without_bias(self):
        """Test MLP without bias."""
        mlp = MLP(
            hidden_size=512,
            intermediate_size=2048,
            activation='silu',
            bias=False
        )

        # Check that bias is None
        assert mlp.fc1.bias is None
        assert mlp.fc2.bias is None

        x = torch.randn(2, 10, 512)
        output = mlp(x)
        assert output.shape == x.shape

    def test_mlp_parameter_count(self):
        """Test MLP parameter count with and without bias."""
        hidden_size = 512
        intermediate_size = 2048

        # Without bias
        mlp_no_bias = MLP(hidden_size, intermediate_size, bias=False)
        params_no_bias = sum(p.numel() for p in mlp_no_bias.parameters())
        expected_no_bias = (hidden_size * intermediate_size) + (intermediate_size * hidden_size)
        assert params_no_bias == expected_no_bias

        # With bias
        mlp_with_bias = MLP(hidden_size, intermediate_size, bias=True)
        params_with_bias = sum(p.numel() for p in mlp_with_bias.parameters())
        expected_with_bias = expected_no_bias + intermediate_size + hidden_size
        assert params_with_bias == expected_with_bias


class TestSwiGLUExtra:
    """Additional tests for SwiGLU coverage."""

    def test_swiglu_extra_repr(self):
        """Test extra_repr method for debugging."""
        mlp = SwiGLU(hidden_size=512, intermediate_size=2048)
        repr_str = mlp.extra_repr()

        assert 'hidden_size=512' in repr_str
        assert 'intermediate_size=2048' in repr_str

    def test_swiglu_zero_input(self):
        """Test SwiGLU with zero input."""
        mlp = SwiGLU(hidden_size=512, intermediate_size=2048)

        x = torch.zeros(2, 10, 512)
        output = mlp(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_swiglu_large_batch(self):
        """Test SwiGLU with large batch size."""
        mlp = SwiGLU(hidden_size=256, intermediate_size=1024)

        x = torch.randn(128, 50, 256)  # Large batch
        output = mlp(x)

        assert output.shape == x.shape

    def test_swiglu_single_element(self):
        """Test SwiGLU with single element."""
        mlp = SwiGLU(hidden_size=64, intermediate_size=256)

        x = torch.randn(1, 1, 64)
        output = mlp(x)

        assert output.shape == x.shape

    def test_swiglu_gradient_flow(self):
        """Test that gradients flow through SwiGLU."""
        mlp = SwiGLU(hidden_size=128, intermediate_size=512)

        x = torch.randn(2, 5, 128, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert mlp.gate_proj.weight.grad is not None
        assert mlp.up_proj.weight.grad is not None
        assert mlp.down_proj.weight.grad is not None


class TestMLPEdgeCases:
    """Edge case tests for MLP."""

    def test_mlp_very_small_dimensions(self):
        """Test MLP with very small dimensions."""
        mlp = MLP(hidden_size=8, intermediate_size=16, activation='silu')

        x = torch.randn(1, 3, 8)
        output = mlp(x)

        assert output.shape == x.shape

    def test_mlp_gradient_flow(self):
        """Test gradient flow through MLP."""
        mlp = MLP(hidden_size=64, intermediate_size=256, activation='gelu')

        x = torch.randn(2, 5, 64, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert x.grad is not None
        assert mlp.fc1.weight.grad is not None
        assert mlp.fc2.weight.grad is not None

    def test_mlp_numerical_stability(self):
        """Test MLP with extreme values."""
        mlp = MLP(hidden_size=64, intermediate_size=256, activation='silu')

        # Very large values
        x_large = torch.randn(2, 5, 64) * 100
        output_large = mlp(x_large)
        assert not torch.isnan(output_large).any()
        assert not torch.isinf(output_large).any()

        # Very small values
        x_small = torch.randn(2, 5, 64) * 0.001
        output_small = mlp(x_small)
        assert not torch.isnan(output_small).any()


class TestMLPComparison:
    """Comparison tests between MLP and SwiGLU."""

    def test_swiglu_vs_mlp_shape(self):
        """Test that SwiGLU and MLP produce same shapes."""
        hidden_size = 256
        intermediate_size = 1024

        mlp = MLP(hidden_size, intermediate_size, activation='silu')
        swiglu = SwiGLU(hidden_size, intermediate_size)

        x = torch.randn(4, 10, hidden_size)

        mlp_out = mlp(x)
        swiglu_out = swiglu(x)

        assert mlp_out.shape == swiglu_out.shape
        assert mlp_out.shape == x.shape

    def test_swiglu_has_more_parameters(self):
        """Test that SwiGLU has more parameters than MLP (3 vs 2 projections)."""
        hidden_size = 256
        intermediate_size = 1024

        mlp = MLP(hidden_size, intermediate_size, bias=False)
        swiglu = SwiGLU(hidden_size, intermediate_size, bias=False)

        mlp_params = sum(p.numel() for p in mlp.parameters())
        swiglu_params = sum(p.numel() for p in swiglu.parameters())

        # SwiGLU has 3 projections (gate, up, down) vs MLP's 2 (fc1, fc2)
        assert swiglu_params > mlp_params

    def test_different_activations_produce_different_outputs(self):
        """Test that different activations produce different outputs."""
        x = torch.randn(2, 5, 128)

        mlp_silu = MLP(128, 512, activation='silu')
        mlp_gelu = MLP(128, 512, activation='gelu')
        mlp_relu = MLP(128, 512, activation='relu')

        # Use same weights for fair comparison
        with torch.no_grad():
            mlp_gelu.fc1.weight.copy_(mlp_silu.fc1.weight)
            mlp_gelu.fc2.weight.copy_(mlp_silu.fc2.weight)
            mlp_relu.fc1.weight.copy_(mlp_silu.fc1.weight)
            mlp_relu.fc2.weight.copy_(mlp_silu.fc2.weight)

        out_silu = mlp_silu(x)
        out_gelu = mlp_gelu(x)
        out_relu = mlp_relu(x)

        # Outputs should be different due to different activations
        assert not torch.allclose(out_silu, out_gelu)
        assert not torch.allclose(out_silu, out_relu)
        assert not torch.allclose(out_gelu, out_relu)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
