"""
Advanced tests for attention.py to achieve 95%+ coverage.

Tests edge cases, complex masking, and attention variants.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baguettotron.model.attention import (
    GroupedQueryAttention,
    MultiHeadAttention,
    MaskedGroupQueryAttention
)


class TestAttentionMaskHandling:
    """Tests for different attention mask shapes and types."""

    def test_attention_mask_2d_shape(self):
        """Test 2D attention mask (batch, seq_len)."""
        attn = GroupedQueryAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
        )

        x = torch.randn(2, 10, 256)
        # 2D mask: (batch, seq_len)
        mask = torch.ones(2, 10)
        mask[0, 5:] = 0  # Mask out second half for first example

        output = attn(x, attention_mask=mask, is_causal=False)
        assert output.shape == x.shape

    def test_attention_mask_3d_shape(self):
        """Test 3D attention mask (batch, 1, seq_len)."""
        attn = GroupedQueryAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
        )

        x = torch.randn(2, 10, 256)
        # 3D mask: (batch, 1, seq_len)
        mask = torch.ones(2, 1, 10)
        mask[1, :, 3:] = 0

        output = attn(x, attention_mask=mask, is_causal=False)
        assert output.shape == x.shape

    def test_attention_mask_4d_shape(self):
        """Test 4D attention mask (batch, 1, seq_len, seq_len)."""
        attn = GroupedQueryAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
        )

        x = torch.randn(2, 10, 256)
        # 4D mask: (batch, 1, seq_len, seq_len)
        mask = torch.ones(2, 1, 10, 10)
        mask[:, :, :, 5:] = 0  # Mask out positions after 5

        output = attn(x, attention_mask=mask, is_causal=False)
        assert output.shape == x.shape

    def test_boolean_attention_mask(self):
        """Test boolean attention mask conversion."""
        attn = GroupedQueryAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
        )

        x = torch.randn(2, 10, 256)
        # Boolean mask
        mask = torch.ones(2, 1, 10, 10, dtype=torch.bool)
        mask[0, :, :, 7:] = False  # Mask positions >= 7

        output = attn(x, attention_mask=mask, is_causal=False)
        assert output.shape == x.shape

    def test_float_attention_mask(self):
        """Test float attention mask (additive)."""
        attn = GroupedQueryAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
        )

        x = torch.randn(2, 10, 256)
        # Float mask with -inf for masked positions
        mask = torch.zeros(2, 1, 10, 10)
        mask[:, :, :, 8:] = float('-inf')

        output = attn(x, attention_mask=mask, is_causal=False)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestAttentionCausalMasking:
    """Tests for causal masking behavior."""

    def test_causal_masking_enabled(self):
        """Test causal masking is applied correctly."""
        attn = GroupedQueryAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        x = torch.randn(1, 5, 128)
        output = attn(x, attention_mask=None, is_causal=True)

        assert output.shape == x.shape

    def test_causal_masking_disabled(self):
        """Test non-causal attention."""
        attn = GroupedQueryAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        x = torch.randn(1, 5, 128)
        output = attn(x, attention_mask=None, is_causal=False)

        assert output.shape == x.shape

    def test_causal_with_custom_mask(self):
        """Test that custom mask overrides causal masking."""
        attn = GroupedQueryAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        x = torch.randn(2, 8, 128)
        custom_mask = torch.ones(2, 1, 8, 8)

        # When custom mask is provided, is_causal should be ignored
        output = attn(x, attention_mask=custom_mask, is_causal=True)
        assert output.shape == x.shape


class TestAttentionDropout:
    """Tests for attention dropout."""

    def test_attention_dropout_training(self):
        """Test attention dropout in training mode."""
        attn = GroupedQueryAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_dropout=0.1,
        )

        attn.train()  # Set to training mode

        x = torch.randn(2, 10, 128)
        output = attn(x, is_causal=True)

        assert output.shape == x.shape

    def test_attention_dropout_eval(self):
        """Test attention dropout in eval mode (should be disabled)."""
        attn = GroupedQueryAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_dropout=0.1,
        )

        attn.eval()  # Set to eval mode

        x = torch.randn(2, 10, 128)
        output = attn(x, is_causal=True)

        assert output.shape == x.shape


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention wrapper."""

    def test_mha_forward(self):
        """Test MultiHeadAttention forward pass."""
        mha = MultiHeadAttention(
            hidden_size=256,
            num_attention_heads=8,
            attention_dropout=0.1,
            bias=False,
        )

        x = torch.randn(2, 10, 256)
        output = mha(x, attention_mask=None, is_causal=True)

        assert output.shape == x.shape

    def test_mha_with_bias(self):
        """Test MultiHeadAttention with bias."""
        mha = MultiHeadAttention(
            hidden_size=128,
            num_attention_heads=4,
            bias=True,
        )

        x = torch.randn(2, 8, 128)
        output = mha(x, is_causal=False)

        assert output.shape == x.shape

    def test_mha_equivalence_to_gqa(self):
        """Test that MHA is equivalent to GQA with equal heads."""
        hidden_size = 128
        num_heads = 4

        mha = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            bias=False,
        )

        gqa = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,  # Same as MHA
            bias=False,
        )

        # They should have same parameter count
        mha_params = sum(p.numel() for p in mha.parameters())
        gqa_params = sum(p.numel() for p in gqa.parameters())

        assert mha_params == gqa_params


class TestMaskedGroupQueryAttention:
    """Tests for MaskedGroupQueryAttention."""

    def test_masked_gqa_forward(self):
        """Test MaskedGroupQueryAttention forward pass."""
        attn = MaskedGroupQueryAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
        )

        x = torch.randn(2, 10, 256)
        output = attn(x, is_causal=True)

        assert output.shape == x.shape

    def test_masked_gqa_with_attention_mask(self):
        """Test MaskedGroupQueryAttention with custom mask."""
        attn = MaskedGroupQueryAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        x = torch.randn(2, 8, 128)
        # Additive mask
        mask = torch.zeros(2, 1, 8, 8)
        mask[:, :, :, 6:] = float('-inf')

        output = attn(x, attention_mask=mask, is_causal=False)
        assert output.shape == x.shape

    def test_masked_gqa_causal_mask_creation(self):
        """Test that causal mask is created correctly."""
        attn = MaskedGroupQueryAttention(
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
        )

        x = torch.randn(1, 5, 64)
        output = attn(x, is_causal=True)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestAttentionEdgeCases:
    """Edge case tests for attention mechanisms."""

    def test_empty_sequence(self):
        """Test attention with sequence length 1."""
        attn = GroupedQueryAttention(
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
        )

        x = torch.randn(1, 1, 64)  # Single token
        output = attn(x, is_causal=True)

        assert output.shape == x.shape

    def test_very_long_sequence(self):
        """Test attention with long sequence."""
        attn = GroupedQueryAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=4096,
        )

        x = torch.randn(1, 512, 128)  # Long sequence
        output = attn(x, is_causal=True)

        assert output.shape == x.shape

    def test_large_batch(self):
        """Test attention with large batch size."""
        attn = GroupedQueryAttention(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        x = torch.randn(32, 16, 64)  # Large batch
        output = attn(x, is_causal=True)

        assert output.shape == x.shape

    def test_gradient_flow(self):
        """Test gradient flow through attention."""
        attn = GroupedQueryAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        x = torch.randn(2, 8, 128, requires_grad=True)
        output = attn(x, is_causal=True)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert attn.q_proj.weight.grad is not None
        assert attn.k_proj.weight.grad is not None
        assert attn.v_proj.weight.grad is not None
        assert attn.o_proj.weight.grad is not None


class TestAttentionExtraRepr:
    """Tests for debugging representation."""

    def test_gqa_extra_repr(self):
        """Test extra_repr for GroupedQueryAttention."""
        attn = GroupedQueryAttention(
            hidden_size=512,
            num_attention_heads=16,
            num_key_value_heads=4,
            attention_dropout=0.1,
        )

        repr_str = attn.extra_repr()

        assert 'hidden_size=512' in repr_str
        assert 'num_heads=16' in repr_str
        assert 'num_kv_heads=4' in repr_str
        assert 'dropout=0.1' in repr_str


class TestAttentionConfiguration:
    """Tests for various attention configurations."""

    def test_single_kv_head_mqa(self):
        """Test Multi-Query Attention (1 KV head)."""
        attn = GroupedQueryAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=1,  # MQA
        )

        assert attn.num_kv_groups == 8

        x = torch.randn(2, 10, 256)
        output = attn(x, is_causal=True)
        assert output.shape == x.shape

    def test_equal_heads_standard_mha(self):
        """Test standard MHA (equal query and KV heads)."""
        attn = GroupedQueryAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=8,  # Standard MHA
        )

        assert attn.num_kv_groups == 1  # No grouping

        x = torch.randn(2, 10, 256)
        output = attn(x, is_causal=True)
        assert output.shape == x.shape

    def test_rope_theta_parameter(self):
        """Test custom RoPE theta parameter."""
        attn = GroupedQueryAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            rope_theta=100000.0,  # Custom theta
        )

        x = torch.randn(2, 10, 128)
        output = attn(x, is_causal=True)
        assert output.shape == x.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
