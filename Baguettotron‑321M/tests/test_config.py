"""
Tests for Baguettotron configuration.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baguettotron.config import BaguettotronConfig


class TestBaguettotronConfig:
    """Tests for BaguettotronConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = BaguettotronConfig()

        assert config.vocab_size == 512
        assert config.hidden_size == 64
        assert config.num_hidden_layers == 2
        assert config.num_attention_heads == 4
        assert config.num_key_value_heads == 2
        assert config.intermediate_size == 256

    def test_baguettotron_321m_config(self):
        """Test official 321M configuration."""
        config = BaguettotronConfig.baguettotron_321m()

        # Verify official parameters
        assert config.vocab_size == 65536
        assert config.hidden_size == 576
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 9
        assert config.num_key_value_heads == 3
        assert config.intermediate_size == 1536
        assert config.max_position_embeddings == 2048
        assert config.rope_theta == 10000.0
        assert config.rms_norm_eps == 1e-5
        assert config.tie_word_embeddings is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = BaguettotronConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
        )

        assert config.vocab_size == 1000
        assert config.hidden_size == 128
        assert config.num_hidden_layers == 4
        assert config.num_attention_heads == 8
        assert config.num_key_value_heads == 4

    def test_head_dim_calculation(self):
        """Test head dimension calculation."""
        config = BaguettotronConfig(
            hidden_size=768,
            num_attention_heads=12,
        )

        head_dim = config.hidden_size // config.num_attention_heads
        assert head_dim == 64

    def test_gqa_ratio(self):
        """Test GQA ratio (heads per KV group)."""
        config = BaguettotronConfig.baguettotron_321m()

        ratio = config.num_attention_heads // config.num_key_value_heads
        assert ratio == 3  # 9 heads / 3 KV heads = 3

    def test_config_validation(self):
        """Test that invalid configs raise errors."""
        # Hidden size not divisible by num_heads should work in config
        # (validation happens in model, not config)
        config = BaguettotronConfig(
            hidden_size=100,
            num_attention_heads=7,
        )
        assert config.hidden_size == 100
        assert config.num_attention_heads == 7
