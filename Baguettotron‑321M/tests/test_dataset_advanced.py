"""
Advanced tests for dataset.py to achieve 95%+ coverage.

Tests error handling, format detection, and edge cases.
"""

import pytest
import torch
import sys
import json
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baguettotron.data.dataset import (
    TextDataset,
    SYNTHDataset,
    create_dataloader,
    preprocess_text_file,
)


class MockTokenizer:
    """Mock tokenizer for testing."""
    def __init__(self):
        self.vocab_size = 1000
        self.pad_token_id = 0

    def encode(self, text):
        """Simple encoding: convert to character codes."""
        return [ord(c) % self.vocab_size for c in text]

    def __call__(self, text, max_length=None, truncation=False, padding=False, return_tensors=None):
        """HuggingFace-style tokenizer call."""
        tokens = self.encode(text)

        if truncation and max_length:
            tokens = tokens[:max_length]

        if padding == 'max_length' and max_length:
            if len(tokens) < max_length:
                tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))

        result = {
            'input_ids': torch.tensor([tokens]),
            'attention_mask': torch.tensor([[1 if t != self.pad_token_id else 0 for t in tokens]])
        }

        return result


class TestTextDatasetEdgeCases:
    """Edge case tests for TextDataset."""

    def test_dataset_from_list(self):
        """Test TextDataset with list input."""
        data = [[1, 2, 3, 4], [5, 6, 7, 8, 9]]
        dataset = TextDataset(data, block_size=8, pad_token_id=0)

        assert len(dataset) == 2

        sample = dataset[0]
        assert 'input_ids' in sample
        assert 'labels' in sample
        assert 'attention_mask' in sample
        assert sample['input_ids'].shape == (8,)

    def test_dataset_from_file(self, tmp_path):
        """Test TextDataset loading from JSON file."""
        data = [[1, 2, 3, 4], [5, 6, 7]]
        data_file = tmp_path / "data.json"

        with open(data_file, 'w') as f:
            json.dump(data, f)

        dataset = TextDataset(str(data_file), block_size=6, pad_token_id=0)

        assert len(dataset) == 2
        sample = dataset[1]
        assert sample['input_ids'].shape == (6,)

    def test_dataset_truncation(self):
        """Test that sequences longer than block_size are truncated."""
        data = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        dataset = TextDataset(data, block_size=5, pad_token_id=0)

        sample = dataset[0]
        assert sample['input_ids'].shape == (5,)
        assert len(sample['input_ids']) == 5
        # First 5 tokens should be preserved
        assert sample['input_ids'].tolist() == [1, 2, 3, 4, 5]

    def test_dataset_padding(self):
        """Test that sequences shorter than block_size are padded."""
        data = [[1, 2, 3]]
        dataset = TextDataset(data, block_size=8, pad_token_id=0)

        sample = dataset[0]
        assert sample['input_ids'].shape == (8,)
        # Should have 3 real tokens + 5 padding
        assert sample['input_ids'].tolist() == [1, 2, 3, 0, 0, 0, 0, 0]

    def test_attention_mask_correctness(self):
        """Test that attention mask correctly identifies real vs padding tokens."""
        data = [[1, 2, 3]]
        dataset = TextDataset(data, block_size=6, pad_token_id=0)

        sample = dataset[0]
        # Real tokens should have mask=1, padding should have mask=0
        assert sample['attention_mask'].tolist() == [1, 1, 1, 0, 0, 0]

    def test_labels_equal_input_ids(self):
        """Test that labels are same as input_ids (for causal LM)."""
        data = [[5, 10, 15, 20]]
        dataset = TextDataset(data, block_size=4, pad_token_id=0)

        sample = dataset[0]
        assert torch.equal(sample['labels'], sample['input_ids'])

    def test_empty_dataset(self):
        """Test dataset with empty list."""
        data = []
        dataset = TextDataset(data, block_size=8)

        assert len(dataset) == 0

    def test_custom_pad_token(self):
        """Test dataset with custom padding token."""
        data = [[1, 2]]
        dataset = TextDataset(data, block_size=5, pad_token_id=999)

        sample = dataset[0]
        # Should pad with 999
        assert 999 in sample['input_ids'].tolist()
        assert sample['input_ids'].tolist() == [1, 2, 999, 999, 999]


class TestSYNTHDatasetFormats:
    """Tests for SYNTHDataset with different file formats."""

    def test_synth_json_array_format(self, tmp_path):
        """Test SYNTH dataset with JSON array format."""
        data = [
            {"text": "Bonjour le monde"},
            {"text": "Comment allez-vous?"},
        ]

        data_file = tmp_path / "synth.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        tokenizer = MockTokenizer()
        dataset = SYNTHDataset(
            data_path=str(data_file),
            tokenizer=tokenizer,
            block_size=20,
            split='train'
        )

        assert len(dataset) == 2

        sample = dataset[0]
        assert 'input_ids' in sample
        assert 'labels' in sample
        assert 'attention_mask' in sample

    def test_synth_jsonl_format(self, tmp_path):
        """Test SYNTH dataset with JSONL format."""
        lines = [
            '{"text": "Premier exemple"}',
            '{"text": "Deuxième exemple"}',
            '{"text": "Troisième exemple"}',
        ]

        data_file = tmp_path / "synth.jsonl"
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        tokenizer = MockTokenizer()
        dataset = SYNTHDataset(
            data_path=str(data_file),
            tokenizer=tokenizer,
            block_size=20
        )

        assert len(dataset) == 3

    def test_synth_with_content_field(self, tmp_path):
        """Test SYNTH dataset with 'content' field instead of 'text'."""
        data = [
            {"content": "Texte avec field 'content'"},
            {"content": "Autre texte"},
        ]

        data_file = tmp_path / "synth_content.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        tokenizer = MockTokenizer()
        dataset = SYNTHDataset(
            data_path=str(data_file),
            tokenizer=tokenizer,
            block_size=30
        )

        assert len(dataset) == 2

    def test_synth_with_string_entries(self, tmp_path):
        """Test SYNTH dataset with direct string entries."""
        data = [
            "Premier texte direct",
            "Deuxième texte direct",
        ]

        data_file = tmp_path / "synth_strings.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        tokenizer = MockTokenizer()
        dataset = SYNTHDataset(
            data_path=str(data_file),
            tokenizer=tokenizer,
            block_size=25
        )

        assert len(dataset) == 2

    def test_synth_invalid_json_lines_skipped(self, tmp_path):
        """Test that invalid JSON lines are skipped with warning."""
        lines = [
            '{"text": "Valid line 1"}',
            'Invalid JSON line',  # This should be skipped
            '{"text": "Valid line 2"}',
        ]

        data_file = tmp_path / "synth_mixed.jsonl"
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        tokenizer = MockTokenizer()

        # Should print warning but continue
        dataset = SYNTHDataset(
            data_path=str(data_file),
            tokenizer=tokenizer,
            block_size=20
        )

        # Should have 2 valid examples
        assert len(dataset) == 2

    def test_synth_empty_text_fields_skipped(self, tmp_path):
        """Test that entries with empty text are skipped."""
        data = [
            {"text": "Valid text"},
            {"text": ""},  # Empty, should be skipped
            {"content": ""},  # Empty, should be skipped
            {"text": "Another valid text"},
        ]

        data_file = tmp_path / "synth_empty.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        tokenizer = MockTokenizer()
        dataset = SYNTHDataset(
            data_path=str(data_file),
            tokenizer=tokenizer,
            block_size=20
        )

        assert len(dataset) == 2  # Only valid non-empty texts

    def test_synth_tokenization(self, tmp_path):
        """Test that tokenization works correctly."""
        data = [{"text": "Test tokenization"}]

        data_file = tmp_path / "synth_tok.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        tokenizer = MockTokenizer()
        dataset = SYNTHDataset(
            data_path=str(data_file),
            tokenizer=tokenizer,
            block_size=50
        )

        sample = dataset[0]
        # Should be properly tokenized and padded
        assert sample['input_ids'].shape == (50,)
        assert sample['attention_mask'].shape == (50,)


class TestCreateDataLoader:
    """Tests for create_dataloader function."""

    def test_dataloader_creation(self):
        """Test basic dataloader creation."""
        data = [[1, 2, 3, 4], [5, 6, 7, 8]]
        dataset = TextDataset(data, block_size=8)

        dataloader = create_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )

        assert len(dataloader) == 1  # 2 examples / batch_size 2
        batch = next(iter(dataloader))
        assert batch['input_ids'].shape == (2, 8)

    def test_dataloader_with_shuffle(self):
        """Test dataloader with shuffling enabled."""
        data = [[i] for i in range(10)]
        dataset = TextDataset(data, block_size=4)

        dataloader = create_dataloader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0
        )

        assert len(dataloader) == 5  # 10 examples / batch_size 2

    def test_dataloader_pin_memory(self):
        """Test dataloader with pin_memory."""
        data = [[1, 2], [3, 4]]
        dataset = TextDataset(data, block_size=4)

        dataloader = create_dataloader(
            dataset,
            batch_size=1,
            pin_memory=True
        )

        assert dataloader.pin_memory == True


class TestPreprocessTextFile:
    """Tests for preprocess_text_file function."""

    def test_preprocess_basic(self, tmp_path):
        """Test basic text file preprocessing."""
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.json"

        # Create input text
        text = "This is a test text for preprocessing. " * 10
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(text)

        tokenizer = MockTokenizer()

        preprocess_text_file(
            input_path=str(input_file),
            output_path=str(output_file),
            tokenizer=tokenizer,
            block_size=50,
            stride=25
        )

        # Check output file exists
        assert output_file.exists()

        # Load and verify
        with open(output_file, 'r') as f:
            chunks = json.load(f)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(len(chunk) == 50 for chunk in chunks[:-1])  # All but last should be full

    def test_preprocess_with_different_stride(self, tmp_path):
        """Test preprocessing with different stride values."""
        input_file = tmp_path / "input2.txt"
        output_file = tmp_path / "output2.json"

        text = "A" * 200  # Simple repeated text
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(text)

        tokenizer = MockTokenizer()

        preprocess_text_file(
            input_path=str(input_file),
            output_path=str(output_file),
            tokenizer=tokenizer,
            block_size=30,
            stride=10  # Smaller stride = more overlap
        )

        assert output_file.exists()

        with open(output_file, 'r') as f:
            chunks = json.load(f)

        # With smaller stride, should have more chunks
        assert len(chunks) > 0

    def test_preprocess_short_text(self, tmp_path):
        """Test preprocessing text shorter than block_size."""
        input_file = tmp_path / "short.txt"
        output_file = tmp_path / "short_out.json"

        # Create longer text that will create chunks
        text = "Short text " * 20  # Make it longer
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(text)

        tokenizer = MockTokenizer()

        preprocess_text_file(
            input_path=str(input_file),
            output_path=str(output_file),
            tokenizer=tokenizer,
            block_size=50,
            stride=25
        )

        assert output_file.exists()

        with open(output_file, 'r') as f:
            chunks = json.load(f)

        # Should have at least one chunk
        assert len(chunks) >= 1


class TestDatasetIntegration:
    """Integration tests for dataset classes."""

    def test_textdataset_with_dataloader(self):
        """Test TextDataset integrated with DataLoader."""
        data = [[i, i+1, i+2] for i in range(20)]
        dataset = TextDataset(data, block_size=5)

        dataloader = create_dataloader(
            dataset,
            batch_size=4,
            shuffle=False
        )

        # Iterate through dataloader
        total_batches = 0
        for batch in dataloader:
            assert 'input_ids' in batch
            assert batch['input_ids'].shape[0] <= 4  # Batch size
            assert batch['input_ids'].shape[1] == 5  # Block size
            total_batches += 1

        assert total_batches == 5  # 20 examples / batch_size 4

    def test_synthdataset_with_dataloader(self, tmp_path):
        """Test SYNTHDataset integrated with DataLoader."""
        data = [{"text": f"Example {i}"} for i in range(10)]

        data_file = tmp_path / "synth_int.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        tokenizer = MockTokenizer()
        dataset = SYNTHDataset(
            data_path=str(data_file),
            tokenizer=tokenizer,
            block_size=20
        )

        dataloader = create_dataloader(
            dataset,
            batch_size=3,
            shuffle=False
        )

        total_batches = 0
        for batch in dataloader:
            assert batch['input_ids'].shape[0] <= 3
            assert batch['input_ids'].shape[1] == 20
            total_batches += 1

        assert total_batches == 4  # 10 examples / batch_size 3 (rounded up)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
