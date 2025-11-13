"""
Complete coverage tests for config.py, collator.py, dataset.py, and attention.py.

This test file adds missing coverage to achieve 95%+ for:
- config.py: head_dim, approximate_params, to_dict, from_dict
- collator.py: attention_mask handling, empty lists, non-tensor values
- dataset.py: string arrays, invalid JSONL, preprocessing edge cases
- attention.py: validation errors, mask variants, extra_repr
"""

import pytest
import torch
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baguettotron.config import BaguettotronConfig
from baguettotron.data.collator import (
    DataCollatorForLanguageModeling,
    DataCollatorForCausalLM,
)
from baguettotron.data.dataset import (
    SYNTHDataset,
    preprocess_text_file,
)
from baguettotron.model.attention import (
    GroupedQueryAttention,
    MultiHeadAttention,
    MaskedGroupQueryAttention,
)


# ============================================================================
# CONFIG.PY TESTS
# ============================================================================

class TestBaguettotronConfigComplete:
    """Complete tests for BaguettotronConfig."""

    def test_config_head_dim_property(self):
        """Test head_dim property calculation."""
        config = BaguettotronConfig(
            hidden_size=768,
            num_attention_heads=12
        )

        assert config.head_dim == 64  # 768 / 12

    def test_config_head_dim_property_various_sizes(self):
        """Test head_dim with different configurations."""
        # Baguettotron-321M config
        config = BaguettotronConfig.baguettotron_321m()
        assert config.head_dim == 64  # 576 / 9

        # Custom config
        config2 = BaguettotronConfig(
            hidden_size=1024,
            num_attention_heads=16
        )
        assert config2.head_dim == 64  # 1024 / 16

    def test_config_approximate_params(self):
        """Test parameter count approximation."""
        config = BaguettotronConfig.baguettotron_321m()

        approx_params = config.approximate_params()

        # Should be approximately 321M (allow some tolerance)
        assert 300_000_000 < approx_params < 400_000_000

    def test_config_approximate_params_tiny(self):
        """Test parameter approximation for tiny config."""
        tiny_config = BaguettotronConfig()
        tiny_params = tiny_config.approximate_params()

        assert tiny_params > 0
        assert tiny_params < 200_000  # Should be very small

    def test_config_to_dict(self):
        """Test config serialization to dictionary."""
        config = BaguettotronConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=6
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['vocab_size'] == 1000
        assert config_dict['hidden_size'] == 256
        assert config_dict['num_hidden_layers'] == 6
        assert 'model_type' in config_dict
        assert 'architectures' in config_dict

    def test_config_to_dict_complete(self):
        """Test to_dict includes all fields."""
        config = BaguettotronConfig.baguettotron_321m()
        config_dict = config.to_dict()

        # Check all important fields present
        required_fields = [
            'vocab_size', 'hidden_size', 'num_hidden_layers',
            'num_attention_heads', 'num_key_value_heads',
            'intermediate_size', 'max_position_embeddings',
            'rope_theta', 'rms_norm_eps', 'attention_dropout',
            'tie_word_embeddings', 'hidden_activation', 'use_cache'
        ]

        for field in required_fields:
            assert field in config_dict

    def test_config_from_dict(self):
        """Test config deserialization from dictionary."""
        config_dict = {
            'vocab_size': 32000,
            'hidden_size': 512,
            'num_hidden_layers': 12,
            'num_attention_heads': 8,
            'unknown_param': 'should_be_ignored'
        }

        config = BaguettotronConfig.from_dict(config_dict)

        assert config.vocab_size == 32000
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 8

    def test_config_from_dict_filters_unknown(self):
        """Test from_dict filters unknown parameters."""
        config_dict = {
            'vocab_size': 1000,
            'hidden_size': 256,
            'fake_param': 'should_be_ignored',
            'another_unknown': 123
        }

        # Should not raise error
        config = BaguettotronConfig.from_dict(config_dict)

        assert config.vocab_size == 1000
        assert config.hidden_size == 256

    def test_config_round_trip_serialization(self):
        """Test config can be serialized and deserialized."""
        original = BaguettotronConfig.baguettotron_321m()

        # Serialize to dict
        config_dict = original.to_dict()

        # Deserialize from dict
        restored = BaguettotronConfig.from_dict(config_dict)

        # Verify all fields match
        assert restored.vocab_size == original.vocab_size
        assert restored.hidden_size == original.hidden_size
        assert restored.num_hidden_layers == original.num_hidden_layers
        assert restored.num_attention_heads == original.num_attention_heads
        assert restored.tie_word_embeddings == original.tie_word_embeddings


# ============================================================================
# COLLATOR.PY TESTS
# ============================================================================

class TestDataCollatorComplete:
    """Complete tests for data collators."""

    def test_collator_with_provided_attention_mask(self):
        """Test collator with pre-provided attention masks."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)

        examples = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'attention_mask': torch.tensor([1, 1, 1])
            },
            {
                'input_ids': torch.tensor([4, 5, 6, 7]),
                'attention_mask': torch.tensor([1, 1, 1, 1])
            },
        ]

        batch = collator(examples)

        # Verify attention masks were correctly padded
        assert batch['attention_mask'][0].tolist() == [1, 1, 1, 0]
        assert batch['attention_mask'][1].tolist() == [1, 1, 1, 1]

    def test_data_collator_for_causal_lm_empty_list(self):
        """Test collator with empty example list."""
        collator = DataCollatorForCausalLM()

        batch = collator([])

        assert batch == {}

    def test_data_collator_with_non_tensor_values(self):
        """Test collator with non-tensor values."""
        collator = DataCollatorForCausalLM()

        examples = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'custom_field': 42  # Non-tensor integer
            },
            {
                'input_ids': torch.tensor([4, 5, 6]),
                'custom_field': 7
            },
        ]

        batch = collator(examples)

        assert isinstance(batch['custom_field'], torch.Tensor)
        assert batch['custom_field'].tolist() == [42, 7]


# ============================================================================
# DATASET.PY TESTS
# ============================================================================

class TestSYNTHDatasetComplete:
    """Complete tests for SYNTH dataset."""

    def test_synth_dataset_with_string_array(self, tmp_path):
        """Test SYNTH dataset with direct string array."""
        class MockTokenizer:
            def __call__(self, text, max_length, truncation, padding, return_tensors):
                return {
                    'input_ids': torch.tensor([[1, 2, 3]]),
                    'attention_mask': torch.tensor([[1, 1, 1]])
                }

        data_file = tmp_path / "strings.json"
        with open(data_file, 'w') as f:
            json.dump(["Text 1", "Text 2", "Text 3"], f)

        tokenizer = MockTokenizer()
        dataset = SYNTHDataset(data_file, tokenizer, block_size=10)

        assert len(dataset) == 3

    def test_synth_dataset_with_string_in_jsonl(self, tmp_path):
        """Test SYNTH dataset with strings in JSONL format."""
        class MockTokenizer:
            def __call__(self, text, max_length, truncation, padding, return_tensors):
                return {
                    'input_ids': torch.tensor([[1, 2, 3]]),
                    'attention_mask': torch.tensor([[1, 1, 1]])
                }

        data_file = tmp_path / "strings.jsonl"
        with open(data_file, 'w') as f:
            f.write('"First string"\n')
            f.write('"Second string"\n')

        tokenizer = MockTokenizer()
        dataset = SYNTHDataset(data_file, tokenizer, block_size=10)

        assert len(dataset) == 2

    def test_synth_dataset_with_invalid_jsonl_lines(self, tmp_path, capsys):
        """Test SYNTH dataset handles invalid JSONL lines gracefully."""
        class MockTokenizer:
            def __call__(self, text, max_length, truncation, padding, return_tensors):
                return {
                    'input_ids': torch.tensor([[1, 2]]),
                    'attention_mask': torch.tensor([[1, 1]])
                }

        data_file = tmp_path / "invalid.jsonl"
        with open(data_file, 'w') as f:
            f.write('{"text": "Valid line"}\n')
            f.write('invalid json line\n')  # Invalid
            f.write('{"text": "Another valid"}\n')

        tokenizer = MockTokenizer()
        dataset = SYNTHDataset(data_file, tokenizer, block_size=10)

        # Should have 2 valid examples (invalid line skipped)
        assert len(dataset) == 2

        # Check warning was printed
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "Skipping invalid JSON line" in captured.out


class TestPreprocessTextFileComplete:
    """Complete tests for text preprocessing."""

    def test_preprocess_text_file_final_chunk(self, tmp_path):
        """Test preprocessing includes final chunk when needed."""
        class MockTokenizer:
            def encode(self, text):
                # Return more tokens than block_size but not multiple of stride
                return list(range(135))  # 135 tokens with block_size=50, stride=20

        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.json"

        with open(input_file, 'w') as f:
            f.write("Some text")

        tokenizer = MockTokenizer()
        preprocess_text_file(
            input_path=input_file,
            output_path=output_file,
            tokenizer=tokenizer,
            block_size=50,
            stride=20
        )

        with open(output_file, 'r') as f:
            chunks = json.load(f)

        # Should have included final chunk
        assert len(chunks) > 0
        # Last chunk should be from the end
        assert chunks[-1] == list(range(135-50, 135))


# ============================================================================
# ATTENTION.PY TESTS
# ============================================================================

class TestGroupedQueryAttentionComplete:
    """Complete tests for GroupedQueryAttention."""

    def test_gqa_invalid_hidden_size(self):
        """Test GQA raises error for invalid hidden_size."""
        with pytest.raises(ValueError, match="must be divisible by"):
            GroupedQueryAttention(
                hidden_size=100,  # Not divisible by 12
                num_attention_heads=12,
                num_key_value_heads=4
            )

    def test_gqa_invalid_kv_heads(self):
        """Test GQA raises error for invalid KV heads configuration."""
        with pytest.raises(ValueError, match="must be divisible by"):
            GroupedQueryAttention(
                hidden_size=512,
                num_attention_heads=12,
                num_key_value_heads=5  # 12 not divisible by 5
            )

    def test_gqa_with_3d_attention_mask(self):
        """Test GQA with 3D attention mask."""
        attn = GroupedQueryAttention(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,
        )

        x = torch.randn(2, 10, 512)
        # 3D mask: (batch, 1, seq_len)
        mask = torch.ones(2, 1, 10)

        output = attn(x, attention_mask=mask, is_causal=False)

        assert output.shape == x.shape

    def test_gqa_with_boolean_attention_mask(self):
        """Test GQA with boolean attention mask."""
        attn = GroupedQueryAttention(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,
        )

        x = torch.randn(2, 10, 512)
        # Boolean mask
        mask = torch.ones(2, 1, 10, 10, dtype=torch.bool)
        mask[:, :, :, 5:] = False  # Mask out second half

        output = attn(x, attention_mask=mask, is_causal=False)

        assert output.shape == x.shape

    def test_gqa_extra_repr(self):
        """Test GQA extra representation for debugging."""
        attn = GroupedQueryAttention(
            hidden_size=576,
            num_attention_heads=9,
            num_key_value_heads=3,
            attention_dropout=0.1
        )

        repr_str = attn.extra_repr()

        assert "hidden_size=576" in repr_str
        assert "num_heads=9" in repr_str
        assert "num_kv_heads=3" in repr_str
        assert "dropout=0.1" in repr_str


class TestMultiHeadAttentionComplete:
    """Complete tests for MultiHeadAttention wrapper."""

    def test_multi_head_attention_forward(self):
        """Test standard MultiHeadAttention wrapper."""
        mha = MultiHeadAttention(
            hidden_size=512,
            num_attention_heads=8,
            attention_dropout=0.0
        )

        x = torch.randn(2, 10, 512)
        output = mha(x)

        assert output.shape == x.shape

    def test_multi_head_attention_equivalence(self):
        """Test MHA is equivalent to GQA with equal heads."""
        hidden_size = 512
        num_heads = 8

        mha = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads
        )

        # MHA should use GQA with equal KV heads
        assert mha.gqa.num_kv_heads == num_heads


class TestMaskedGroupQueryAttentionComplete:
    """Complete tests for MaskedGroupQueryAttention."""

    def test_mgqa_invalid_hidden_size(self):
        """Test MGQA raises error for invalid hidden_size."""
        with pytest.raises(ValueError, match="must be divisible by"):
            MaskedGroupQueryAttention(
                hidden_size=100,
                num_attention_heads=12,
                num_key_value_heads=4
            )

    def test_mgqa_invalid_kv_heads(self):
        """Test MGQA raises error for invalid KV heads."""
        with pytest.raises(ValueError, match="must be divisible by"):
            MaskedGroupQueryAttention(
                hidden_size=512,
                num_attention_heads=12,
                num_key_value_heads=5
            )
