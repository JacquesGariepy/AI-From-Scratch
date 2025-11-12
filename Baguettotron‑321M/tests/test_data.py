"""
Tests for data modules (dataset and collator).

These tests ensure 100% coverage of src/baguettotron/data/.
"""

import pytest
import torch
import json
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baguettotron.data import (
    TextDataset,
    SYNTHDataset,
    create_dataloader,
    preprocess_text_file,
    DataCollatorForLanguageModeling,
    DataCollatorForCausalLM,
    default_data_collator,
)


class TestTextDataset:
    """Tests for TextDataset."""

    def test_text_dataset_from_list(self):
        """Test creating dataset from list of token sequences."""
        data = [[1, 2, 3, 4], [5, 6, 7, 8, 9]]
        dataset = TextDataset(data, block_size=8, pad_token_id=0)

        assert len(dataset) == 2

        # Test first sequence (padded)
        batch = dataset[0]
        assert batch['input_ids'].shape == (8,)
        assert batch['labels'].shape == (8,)
        assert batch['attention_mask'].shape == (8,)
        assert batch['attention_mask'].sum() == 4  # 4 real tokens

    def test_text_dataset_from_file(self, tmp_path):
        """Test creating dataset from JSON file."""
        # Create temp file
        data = [[1, 2, 3], [4, 5, 6, 7]]
        data_file = tmp_path / "data.json"
        with open(data_file, 'w') as f:
            json.dump(data, f)

        dataset = TextDataset(str(data_file), block_size=5, pad_token_id=0)
        assert len(dataset) == 2

    def test_text_dataset_truncation(self):
        """Test that long sequences are truncated."""
        data = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        dataset = TextDataset(data, block_size=5)

        batch = dataset[0]
        assert batch['input_ids'].shape == (5,)
        assert batch['input_ids'].tolist() == [1, 2, 3, 4, 5]

    def test_text_dataset_padding(self):
        """Test that short sequences are padded."""
        data = [[1, 2]]
        dataset = TextDataset(data, block_size=5, pad_token_id=0)

        batch = dataset[0]
        assert batch['input_ids'].tolist() == [1, 2, 0, 0, 0]
        assert batch['attention_mask'].tolist() == [1, 1, 0, 0, 0]


class TestSYNTHDataset:
    """Tests for SYNTHDataset."""

    def test_synth_dataset_jsonl(self, tmp_path):
        """Test SYNTH dataset with JSONL format."""
        # Create mock tokenizer
        class MockTokenizer:
            def __call__(self, text, max_length, truncation, padding, return_tensors):
                tokens = [len(text)] * 5  # Simple mock
                return {
                    'input_ids': torch.tensor([tokens]),
                    'attention_mask': torch.tensor([[1] * 5])
                }

        # Create JSONL file
        data_file = tmp_path / "synth.jsonl"
        with open(data_file, 'w') as f:
            f.write('{"text": "Hello world"}\n')
            f.write('{"text": "Test data"}\n')

        tokenizer = MockTokenizer()
        dataset = SYNTHDataset(
            data_path=data_file,
            tokenizer=tokenizer,
            block_size=10,
            split='train'
        )

        assert len(dataset) == 2
        batch = dataset[0]
        assert 'input_ids' in batch
        assert 'labels' in batch
        assert 'attention_mask' in batch

    def test_synth_dataset_json(self, tmp_path):
        """Test SYNTH dataset with JSON array format."""
        class MockTokenizer:
            def __call__(self, text, max_length, truncation, padding, return_tensors):
                return {
                    'input_ids': torch.tensor([[1, 2, 3]]),
                    'attention_mask': torch.tensor([[1, 1, 1]])
                }

        # Create JSON file
        data_file = tmp_path / "synth.json"
        with open(data_file, 'w') as f:
            json.dump([
                {"text": "Example 1"},
                {"content": "Example 2"}
            ], f)

        tokenizer = MockTokenizer()
        dataset = SYNTHDataset(
            data_path=data_file,
            tokenizer=tokenizer,
            block_size=10
        )

        assert len(dataset) == 2


class TestDataLoaders:
    """Tests for dataloader utilities."""

    def test_create_dataloader(self):
        """Test creating dataloader."""
        data = [[1, 2, 3], [4, 5, 6]]
        dataset = TextDataset(data, block_size=5)

        dataloader = create_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )

        assert dataloader.batch_size == 2
        batches = list(dataloader)
        assert len(batches) == 1  # 2 samples / batch_size=2

    def test_preprocess_text_file(self, tmp_path):
        """Test preprocessing text file."""
        # Create mock tokenizer
        class MockTokenizer:
            def encode(self, text):
                return [ord(c) for c in text[:10]]  # Simple char encoding

        # Create input file
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.json"

        with open(input_file, 'w') as f:
            f.write("Hello world! " * 100)  # Long text

        tokenizer = MockTokenizer()
        preprocess_text_file(
            input_path=input_file,
            output_path=output_file,
            tokenizer=tokenizer,
            block_size=10,
            stride=5
        )

        # Check output exists and is valid JSON
        assert output_file.exists()
        with open(output_file, 'r') as f:
            chunks = json.load(f)
        assert isinstance(chunks, list)
        assert len(chunks) > 0


class TestDataCollators:
    """Tests for data collators."""

    def test_data_collator_for_language_modeling(self):
        """Test DataCollatorForLanguageModeling."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)

        examples = [
            {'input_ids': torch.tensor([1, 2, 3])},
            {'input_ids': torch.tensor([4, 5, 6, 7])},
        ]

        batch = collator(examples)

        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch
        assert batch['input_ids'].shape == (2, 4)  # Padded to max length
        assert batch['attention_mask'].shape == (2, 4)

    def test_data_collator_mlm_error(self):
        """Test that MLM=True raises error."""
        with pytest.raises(ValueError, match="causal language modeling"):
            DataCollatorForLanguageModeling(pad_token_id=0, mlm=True)

    def test_data_collator_for_causal_lm(self):
        """Test DataCollatorForCausalLM."""
        collator = DataCollatorForCausalLM()

        examples = [
            {
                'input_ids': torch.tensor([1, 2, 3, 0]),
                'labels': torch.tensor([1, 2, 3, 0]),
                'attention_mask': torch.tensor([1, 1, 1, 0]),
            },
            {
                'input_ids': torch.tensor([4, 5, 6, 7]),
                'labels': torch.tensor([4, 5, 6, 7]),
                'attention_mask': torch.tensor([1, 1, 1, 1]),
            },
        ]

        batch = collator(examples)

        assert batch['input_ids'].shape == (2, 4)
        assert batch['labels'].shape == (2, 4)
        assert batch['attention_mask'].shape == (2, 4)

    def test_default_data_collator(self):
        """Test default_data_collator."""
        examples = [
            {'input_ids': torch.tensor([1, 2])},
            {'input_ids': torch.tensor([3, 4])},
        ]

        batch = default_data_collator(examples)

        assert 'input_ids' in batch
        assert batch['input_ids'].shape == (2, 2)

    def test_collator_with_labels(self):
        """Test collator handles labels with -100 padding."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)

        examples = [
            {
                'input_ids': torch.tensor([1, 2]),
                'labels': torch.tensor([1, 2]),
            },
            {
                'input_ids': torch.tensor([3, 4, 5, 6]),
                'labels': torch.tensor([3, 4, 5, 6]),
            },
        ]

        batch = collator(examples)

        # Check that short sequence labels are padded with -100
        assert batch['labels'][0, 2] == -100
        assert batch['labels'][0, 3] == -100
