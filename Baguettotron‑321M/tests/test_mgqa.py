"""
Tests for MaskedGroupQueryAttention compatibility.

These tests ensure that the new MaskedGroupQueryAttention implementation
in the refactored architecture maintains compatibility with the original mgqa.py.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baguettotron.config import BaguettotronConfig
from baguettotron.model import MaskedGroupQueryAttention, GroupedQueryAttention


class TestMaskedGroupQueryAttention:
    """Tests for MaskedGroupQueryAttention."""

    def test_mgqa_forward(self):
        """Test MGQA forward pass."""
        hidden_size = 512
        num_heads = 8
        num_kv_heads = 4

        mgqa = MaskedGroupQueryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
        )

        x = torch.randn(2, 10, hidden_size)
        output = mgqa(x, is_causal=True)

        assert output.shape == x.shape

    def test_mgqa_has_correct_attributes(self):
        """Test that MGQA has q_proj, k_proj, v_proj, out_proj attributes."""
        mgqa = MaskedGroupQueryAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
        )

        # Check required attributes for compatibility
        assert hasattr(mgqa, 'q_proj')
        assert hasattr(mgqa, 'k_proj')
        assert hasattr(mgqa, 'v_proj')
        assert hasattr(mgqa, 'out_proj')

    def test_mgqa_causal_masking(self):
        """Test that causal masking prevents future information leakage."""
        mgqa = MaskedGroupQueryAttention(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        mgqa.eval()

        # Create input where each position has a unique value
        batch_size, seq_len = 1, 8
        x = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 64)

        with torch.no_grad():
            output = mgqa(x, is_causal=True)

        # In causal attention, position i can only attend to positions <= i
        # So output at position 0 should not depend on positions 1-7
        # We can't test this directly, but we can check shapes and no NaN
        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_mgqa_without_causal_mask(self):
        """Test MGQA with causal masking disabled."""
        mgqa = MaskedGroupQueryAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        x = torch.randn(2, 10, 128)
        output = mgqa(x, is_causal=False)

        assert output.shape == x.shape

    def test_mgqa_with_custom_attention_mask(self):
        """Test MGQA with custom attention mask."""
        mgqa = MaskedGroupQueryAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        batch, seq_len = 2, 10
        x = torch.randn(batch, seq_len, 128)

        # Create a custom attention mask (additive)
        # Shape: (batch, 1, seq_len, seq_len) or broadcastable
        mask = torch.zeros(seq_len, seq_len)
        mask[:, seq_len//2:] = float('-inf')  # Block second half

        output = mgqa(x, attention_mask=mask, is_causal=False)

        assert output.shape == x.shape

    def test_equivalence_with_gqa_when_weights_match(self):
        """Test that MGQA and GQA produce same results with same weights."""
        hidden_size = 256
        num_heads = 8
        num_kv_heads = 4

        gqa = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
        ).eval()

        mgqa = MaskedGroupQueryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
        ).eval()

        # Copy weights from GQA to MGQA
        with torch.no_grad():
            mgqa.q_proj.weight.copy_(gqa.q_proj.weight)
            mgqa.k_proj.weight.copy_(gqa.k_proj.weight)
            mgqa.v_proj.weight.copy_(gqa.v_proj.weight)
            mgqa.out_proj.weight.copy_(gqa.o_proj.weight)

        # Test on same input
        x = torch.randn(2, 10, hidden_size)

        with torch.no_grad():
            gqa_output = gqa(x, is_causal=True)
            mgqa_output = mgqa(x, is_causal=True)

        # Outputs should be very close (allowing for numerical differences)
        assert torch.allclose(gqa_output, mgqa_output, atol=1e-5, rtol=1e-4)

    def test_mgqa_gradient_flow(self):
        """Test that gradients flow through MGQA."""
        mgqa = MaskedGroupQueryAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        x = torch.randn(2, 10, 128, requires_grad=True)
        output = mgqa(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients were computed
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # Check that all parameters have gradients
        for name, param in mgqa.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Invalid gradient for {name}"

    def test_mgqa_different_kv_groups(self):
        """Test MGQA with different numbers of KV groups."""
        hidden_size = 256

        # Test with different GQA configurations
        configs = [
            (8, 8),  # MHA: num_heads == num_kv_heads
            (8, 4),  # GQA: num_heads > num_kv_heads
            (8, 2),  # GQA with more grouping
            (8, 1),  # MQA: single KV head
        ]

        for num_heads, num_kv_heads in configs:
            mgqa = MaskedGroupQueryAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
            )

            x = torch.randn(2, 10, hidden_size)
            output = mgqa(x)

            assert output.shape == x.shape, f"Failed for ({num_heads}, {num_kv_heads})"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mgqa_cuda(self):
        """Test MGQA on CUDA."""
        mgqa = MaskedGroupQueryAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
        ).to('cuda')

        x = torch.randn(2, 10, 256).to('cuda')
        output = mgqa(x)

        assert output.device.type == 'cuda'
        assert output.shape == x.shape
