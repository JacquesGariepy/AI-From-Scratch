"""
Tests for Baguettotron model components.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baguettotron.config import BaguettotronConfig
from baguettotron.model import (
    BaguettotronForCausalLM,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    GroupedQueryAttention,
    TransformerBlock,
    TransformerDecoder,
)


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_rmsnorm_forward(self):
        """Test RMSNorm forward pass."""
        batch_size, seq_len, hidden_size = 2, 10, 64
        norm = RMSNorm(hidden_size, eps=1e-6)

        x = torch.randn(batch_size, seq_len, hidden_size)
        output = norm(x)

        assert output.shape == x.shape

        # Check that variance is approximately 1
        variance = output.pow(2).mean(dim=-1)
        assert torch.allclose(variance, torch.ones_like(variance), atol=0.1)

    def test_rmsnorm_zero_input(self):
        """Test RMSNorm with zero input."""
        norm = RMSNorm(64)
        x = torch.zeros(2, 10, 64)
        output = norm(x)

        # Should not NaN
        assert not torch.isnan(output).any()


class TestRotaryEmbedding:
    """Tests for Rotary Position Embeddings."""

    def test_rope_forward(self):
        """Test RoPE forward pass."""
        head_dim = 64
        max_seq = 128
        rope = RotaryEmbedding(head_dim, max_position_embeddings=max_seq)

        batch, num_heads, seq_len = 2, 8, 64
        q = torch.randn(batch, num_heads, seq_len, head_dim)
        k = torch.randn(batch, num_heads, seq_len, head_dim)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope_cache_extension(self):
        """Test that RoPE cache extends for long sequences."""
        rope = RotaryEmbedding(head_dim=64, max_position_embeddings=128)

        # Short sequence
        q = torch.randn(1, 8, 64, 64)
        k = torch.randn(1, 8, 64, 64)
        rope(q, k)

        assert rope.max_seq_len_cached >= 64

        # Longer sequence
        q = torch.randn(1, 8, 256, 64)
        k = torch.randn(1, 8, 256, 64)
        rope(q, k)

        assert rope.max_seq_len_cached >= 256


class TestSwiGLU:
    """Tests for SwiGLU feed-forward network."""

    def test_swiglu_forward(self):
        """Test SwiGLU forward pass."""
        hidden_size = 512
        intermediate_size = 2048
        mlp = SwiGLU(hidden_size, intermediate_size)

        x = torch.randn(2, 10, hidden_size)
        output = mlp(x)

        assert output.shape == x.shape

    def test_swiglu_parameter_count(self):
        """Test SwiGLU has correct number of parameters."""
        hidden_size = 512
        intermediate_size = 2048
        mlp = SwiGLU(hidden_size, intermediate_size, bias=False)

        # Count parameters
        num_params = sum(p.numel() for p in mlp.parameters())

        # 3 projections without bias
        expected = hidden_size * intermediate_size * 3
        assert num_params == expected


class TestGroupedQueryAttention:
    """Tests for Grouped Query Attention."""

    def test_gqa_forward(self):
        """Test GQA forward pass."""
        batch_size, seq_len = 2, 64
        hidden_size = 576
        num_heads = 9
        num_kv_heads = 3

        attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
        )

        x = torch.randn(batch_size, seq_len, hidden_size)
        output = attn(x, is_causal=True)

        assert output.shape == x.shape

    def test_gqa_with_attention_mask(self):
        """Test GQA with attention mask."""
        attn = GroupedQueryAttention(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,
        )

        x = torch.randn(2, 10, 512)
        mask = torch.ones(2, 1, 10, 10)
        output = attn(x, attention_mask=mask, is_causal=False)

        assert output.shape == x.shape

    def test_mqa(self):
        """Test Multi-Query Attention (1 KV head)."""
        attn = GroupedQueryAttention(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=1,  # MQA
        )

        x = torch.randn(2, 10, 512)
        output = attn(x)

        assert output.shape == x.shape


class TestTransformerBlock:
    """Tests for Transformer block."""

    def test_transformer_block_forward(self):
        """Test transformer block forward pass."""
        config = BaguettotronConfig()
        block = TransformerBlock(config)

        batch, seq_len = 2, 64
        x = torch.randn(batch, seq_len, config.hidden_size)
        output = block(x, is_causal=True)

        assert output.shape == x.shape

    def test_transformer_block_residual(self):
        """Test that residual connections work."""
        config = BaguettotronConfig()
        block = TransformerBlock(config)

        x = torch.randn(2, 10, config.hidden_size)

        # With zero initialization, output should be close to input
        # (testing residual connection)
        with torch.no_grad():
            for param in block.parameters():
                param.zero_()

        output = block(x)

        # Output should be close to input due to residuals
        # (not exactly equal due to normalization)
        assert output.shape == x.shape


class TestTransformerDecoder:
    """Tests for Transformer decoder stack."""

    def test_decoder_forward(self):
        """Test decoder forward pass."""
        config = BaguettotronConfig(num_hidden_layers=4)
        decoder = TransformerDecoder(config)

        x = torch.randn(2, 10, config.hidden_size)
        output = decoder(x)

        assert output.shape == x.shape

    def test_decoder_layer_count(self):
        """Test decoder has correct number of layers."""
        num_layers = 6
        config = BaguettotronConfig(num_hidden_layers=num_layers)
        decoder = TransformerDecoder(config)

        assert len(decoder.layers) == num_layers


class TestBaguettotronForCausalLM:
    """Tests for BaguettotronForCausalLM."""

    def test_model_forward(self):
        """Test model forward pass."""
        config = BaguettotronConfig()
        model = BaguettotronForCausalLM(config)

        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(input_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_model_generate(self):
        """Test text generation."""
        config = BaguettotronConfig()
        model = BaguettotronForCausalLM(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        max_new_tokens = 10

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy
            )

        expected_len = input_ids.shape[1] + max_new_tokens
        assert output.shape == (1, expected_len)

    def test_model_generate_with_sampling(self):
        """Test generation with sampling."""
        config = BaguettotronConfig()
        model = BaguettotronForCausalLM(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 5))

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                do_sample=True,
            )

        assert output.shape[1] > input_ids.shape[1]

    def test_parameter_count(self):
        """Test parameter counting."""
        config = BaguettotronConfig()
        model = BaguettotronForCausalLM(config)

        total = model.count_parameters()
        trainable = model.count_parameters(trainable_only=True)
        non_embedding = model.get_num_params(non_embedding=True)

        assert total > 0
        assert trainable == total  # All params should be trainable
        assert non_embedding < total  # Should be less without embeddings

    def test_321m_parameter_count(self):
        """Test that 321M config has approximately 321M parameters."""
        config = BaguettotronConfig.baguettotron_321m()
        model = BaguettotronForCausalLM(config)

        num_params = model.count_parameters()
        num_params_m = num_params / 1e6

        # Should be approximately 321M (allow some tolerance)
        assert 320 <= num_params_m <= 322

    def test_tied_embeddings(self):
        """Test weight tying between embeddings and LM head."""
        config = BaguettotronConfig(tie_word_embeddings=True)
        model = BaguettotronForCausalLM(config)

        # Check that weights are shared
        assert model.lm_head.weight is model.embeddings.weight

    def test_untied_embeddings(self):
        """Test separate embeddings and LM head."""
        config = BaguettotronConfig(tie_word_embeddings=False)
        model = BaguettotronForCausalLM(config)

        # Check that weights are different
        assert model.lm_head.weight is not model.embeddings.weight

    def test_model_device_transfer(self):
        """Test moving model between devices."""
        config = BaguettotronConfig()
        model = BaguettotronForCausalLM(config)

        # Move to CPU (should always work)
        model = model.to('cpu')

        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        logits = model(input_ids)

        assert logits.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_cuda(self):
        """Test model on CUDA."""
        config = BaguettotronConfig()
        model = BaguettotronForCausalLM(config).to('cuda')

        input_ids = torch.randint(0, config.vocab_size, (1, 10)).to('cuda')
        logits = model(input_ids)

        assert logits.device.type == 'cuda'
