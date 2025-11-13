"""
Advanced tests for collator.py to achieve 95%+ coverage.

Tests different padding strategies, edge cases, and error handling.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baguettotron.data.collator import (
    DataCollatorForLanguageModeling,
    DataCollatorForCausalLM,
    default_data_collator,
)


class TestDataCollatorForLanguageModeling:
    """Tests for DataCollatorForLanguageModeling."""

    def test_basic_collation(self):
        """Test basic collation with padding."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)

        examples = [
            {'input_ids': torch.tensor([1, 2, 3])},
            {'input_ids': torch.tensor([4, 5, 6, 7])},
            {'input_ids': torch.tensor([8, 9])},
        ]

        batch = collator(examples)

        # All sequences should be padded to max length (4)
        assert batch['input_ids'].shape == (3, 4)
        assert batch['attention_mask'].shape == (3, 4)
        assert batch['labels'].shape == (3, 4)

    def test_padding_applied_correctly(self):
        """Test that padding is applied to shorter sequences."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)

        examples = [
            {'input_ids': torch.tensor([1, 2, 3, 4, 5])},
            {'input_ids': torch.tensor([6, 7])},
        ]

        batch = collator(examples)

        # Second sequence should be padded with 0
        assert batch['input_ids'][1].tolist() == [6, 7, 0, 0, 0]
        # Attention mask should be 1 for real tokens, 0 for padding
        assert batch['attention_mask'][1].tolist() == [1, 1, 0, 0, 0]

    def test_with_existing_attention_mask(self):
        """Test collation when examples already have attention masks."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)

        examples = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'attention_mask': torch.tensor([1, 1, 1]),
            },
            {
                'input_ids': torch.tensor([4, 5]),
                'attention_mask': torch.tensor([1, 1]),
            },
        ]

        batch = collator(examples)

        # Attention masks should be padded
        assert batch['attention_mask'].shape == (2, 3)
        assert batch['attention_mask'][1].tolist() == [1, 1, 0]

    def test_with_existing_labels(self):
        """Test collation when examples have labels."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)

        examples = [
            {
                'input_ids': torch.tensor([1, 2, 3, 4]),
                'labels': torch.tensor([1, 2, 3, 4]),
            },
            {
                'input_ids': torch.tensor([5, 6]),
                'labels': torch.tensor([5, 6]),
            },
        ]

        batch = collator(examples)

        # Labels should be padded with -100 (ignored by loss)
        assert batch['labels'].shape == (2, 4)
        assert batch['labels'][1].tolist() == [5, 6, -100, -100]

    def test_mlm_raises_error(self):
        """Test that MLM mode raises ValueError."""
        with pytest.raises(ValueError, match="only supports causal language modeling"):
            DataCollatorForLanguageModeling(pad_token_id=0, mlm=True)

    def test_custom_pad_token(self):
        """Test collator with custom pad token ID."""
        collator = DataCollatorForLanguageModeling(pad_token_id=999)

        examples = [
            {'input_ids': torch.tensor([1, 2, 3, 4])},
            {'input_ids': torch.tensor([5])},
        ]

        batch = collator(examples)

        # Should pad with 999
        assert batch['input_ids'][1].tolist() == [5, 999, 999, 999]

    def test_no_padding_needed(self):
        """Test when all sequences have same length (no padding needed)."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)

        examples = [
            {'input_ids': torch.tensor([1, 2, 3])},
            {'input_ids': torch.tensor([4, 5, 6])},
            {'input_ids': torch.tensor([7, 8, 9])},
        ]

        batch = collator(examples)

        # No padding should be added
        assert batch['input_ids'].shape == (3, 3)
        assert not (batch['input_ids'] == 0).any()  # No pad tokens

    def test_single_example(self):
        """Test collation with single example."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)

        examples = [
            {'input_ids': torch.tensor([1, 2, 3, 4, 5])},
        ]

        batch = collator(examples)

        assert batch['input_ids'].shape == (1, 5)
        assert batch['input_ids'][0].tolist() == [1, 2, 3, 4, 5]

    def test_labels_match_input_ids_when_not_provided(self):
        """Test that labels default to input_ids when not provided."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)

        examples = [
            {'input_ids': torch.tensor([1, 2, 3])},
        ]

        batch = collator(examples)

        # Labels should be same as input_ids (for causal LM)
        assert torch.equal(batch['labels'], batch['input_ids'])


class TestDataCollatorForCausalLM:
    """Tests for DataCollatorForCausalLM."""

    def test_basic_stacking(self):
        """Test basic tensor stacking."""
        collator = DataCollatorForCausalLM()

        examples = [
            {
                'input_ids': torch.tensor([1, 2, 3, 4]),
                'labels': torch.tensor([1, 2, 3, 4]),
                'attention_mask': torch.tensor([1, 1, 1, 1]),
            },
            {
                'input_ids': torch.tensor([5, 6, 7, 8]),
                'labels': torch.tensor([5, 6, 7, 8]),
                'attention_mask': torch.tensor([1, 1, 1, 1]),
            },
        ]

        batch = collator(examples)

        assert batch['input_ids'].shape == (2, 4)
        assert batch['labels'].shape == (2, 4)
        assert batch['attention_mask'].shape == (2, 4)

    def test_empty_examples(self):
        """Test collator with empty list."""
        collator = DataCollatorForCausalLM()

        batch = collator([])

        assert batch == {}

    def test_single_example_stacking(self):
        """Test stacking with single example."""
        collator = DataCollatorForCausalLM()

        examples = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([4, 5, 6]),
            }
        ]

        batch = collator(examples)

        assert batch['input_ids'].shape == (1, 3)
        assert batch['labels'].shape == (1, 3)

    def test_return_tensors_parameter(self):
        """Test return_tensors parameter."""
        collator = DataCollatorForCausalLM(return_tensors='pt')

        examples = [
            {
                'input_ids': torch.tensor([1, 2]),
                'labels': torch.tensor([3, 4]),
            }
        ]

        batch = collator(examples)

        assert isinstance(batch['input_ids'], torch.Tensor)
        assert isinstance(batch['labels'], torch.Tensor)

    def test_non_tensor_values_converted(self):
        """Test that non-tensor values are converted to tensors."""
        collator = DataCollatorForCausalLM()

        examples = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'some_scalar': 42,  # Non-tensor value
            }
        ]

        batch = collator(examples)

        assert isinstance(batch['some_scalar'], torch.Tensor)

    def test_all_fields_stacked(self):
        """Test that all fields in examples are stacked."""
        collator = DataCollatorForCausalLM()

        examples = [
            {
                'input_ids': torch.tensor([1, 2]),
                'labels': torch.tensor([3, 4]),
                'attention_mask': torch.tensor([1, 1]),
                'extra_field': torch.tensor([100, 200]),
            },
            {
                'input_ids': torch.tensor([5, 6]),
                'labels': torch.tensor([7, 8]),
                'attention_mask': torch.tensor([1, 0]),
                'extra_field': torch.tensor([300, 400]),
            },
        ]

        batch = collator(examples)

        assert 'input_ids' in batch
        assert 'labels' in batch
        assert 'attention_mask' in batch
        assert 'extra_field' in batch
        assert batch['extra_field'].shape == (2, 2)


class TestDefaultDataCollator:
    """Tests for default_data_collator function."""

    def test_default_collator_delegates(self):
        """Test that default_data_collator delegates to DataCollatorForCausalLM."""
        examples = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([4, 5, 6]),
            },
            {
                'input_ids': torch.tensor([7, 8, 9]),
                'labels': torch.tensor([10, 11, 12]),
            },
        ]

        batch = default_data_collator(examples)

        assert batch['input_ids'].shape == (2, 3)
        assert batch['labels'].shape == (2, 3)

    def test_default_collator_empty_list(self):
        """Test default collator with empty list."""
        batch = default_data_collator([])
        assert batch == {}


class TestCollatorEdgeCases:
    """Edge case tests for collators."""

    def test_very_long_sequences(self):
        """Test collation with very long sequences."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)

        examples = [
            {'input_ids': torch.randint(0, 1000, (2048,))},
            {'input_ids': torch.randint(0, 1000, (1024,))},
        ]

        batch = collator(examples)

        assert batch['input_ids'].shape == (2, 2048)

    def test_single_token_sequences(self):
        """Test collation with single-token sequences."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)

        examples = [
            {'input_ids': torch.tensor([42])},
            {'input_ids': torch.tensor([99])},
        ]

        batch = collator(examples)

        assert batch['input_ids'].shape == (2, 1)

    def test_large_batch(self):
        """Test collation with large batch size."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)

        examples = [
            {'input_ids': torch.randint(0, 100, (10,))}
            for _ in range(128)  # Large batch
        ]

        batch = collator(examples)

        assert batch['input_ids'].shape[0] == 128

    def test_mixed_sequence_lengths(self):
        """Test collation with highly variable sequence lengths."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)

        examples = [
            {'input_ids': torch.tensor([1])},
            {'input_ids': torch.tensor([2, 3, 4, 5, 6])},
            {'input_ids': torch.tensor([7, 8])},
            {'input_ids': torch.tensor([9, 10, 11])},
        ]

        batch = collator(examples)

        # All should be padded to length 5
        assert batch['input_ids'].shape == (4, 5)
        # Check padding
        assert batch['input_ids'][0, 1:].tolist() == [0, 0, 0, 0]
        assert batch['input_ids'][2, 2:].tolist() == [0, 0, 0]


class TestCollatorIntegration:
    """Integration tests for collators with datasets."""

    def test_collator_with_textdataset(self):
        """Test collator integration with TextDataset."""
        from baguettotron.data.dataset import TextDataset

        data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        dataset = TextDataset(data, block_size=4)

        collator = DataCollatorForCausalLM()

        # Simulate DataLoader collation
        examples = [dataset[i] for i in range(len(dataset))]
        batch = collator(examples)

        assert batch['input_ids'].shape == (3, 4)
        assert batch['labels'].shape == (3, 4)
        assert batch['attention_mask'].shape == (3, 4)

    def test_different_collators_produce_same_shapes(self):
        """Test that both collators produce compatible output shapes."""
        examples = [
            {
                'input_ids': torch.tensor([1, 2, 3, 4]),
                'labels': torch.tensor([1, 2, 3, 4]),
                'attention_mask': torch.tensor([1, 1, 1, 1]),
            },
            {
                'input_ids': torch.tensor([5, 6, 7, 8]),
                'labels': torch.tensor([5, 6, 7, 8]),
                'attention_mask': torch.tensor([1, 1, 1, 1]),
            },
        ]

        collator1 = DataCollatorForCausalLM()
        collator2 = default_data_collator

        batch1 = collator1(examples)
        batch2 = collator2(examples)

        assert batch1['input_ids'].shape == batch2['input_ids'].shape
        assert torch.equal(batch1['input_ids'], batch2['input_ids'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
