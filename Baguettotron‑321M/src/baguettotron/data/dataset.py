"""
Dataset utilities for Baguettotron training.

This module provides dataset classes for loading and preprocessing text data
for language model training, with support for the PleIAs/SYNTH dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Any
from pathlib import Path
import json


class TextDataset(Dataset):
    """
    Generic text dataset for causal language modeling.

    This dataset handles tokenized text sequences, providing input_ids and labels
    for next-token prediction training.

    Args:
        data: List of token ID sequences or path to preprocessed data file
        block_size: Maximum sequence length (will pad/truncate to this length)
        pad_token_id: Token ID to use for padding (default: 0)

    Attributes:
        data: List of token sequences
        block_size: Maximum sequence length
        pad_token_id: Padding token ID

    Examples:
        >>> # From pre-tokenized data
        >>> token_sequences = [[1, 2, 3, 4], [5, 6, 7, 8, 9]]
        >>> dataset = TextDataset(token_sequences, block_size=8)
        >>> assert len(dataset) == 2
        >>> batch = dataset[0]
        >>> assert batch['input_ids'].shape == (8,)
        >>> assert batch['labels'].shape == (8,)

        >>> # Load from file
        >>> dataset = TextDataset('data/train_tokens.json', block_size=128)
    """

    def __init__(
        self,
        data: List[List[int]] | str | Path,
        block_size: int = 2048,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.block_size = block_size
        self.pad_token_id = pad_token_id

        # Load data if path is provided
        if isinstance(data, (str, Path)):
            with open(data, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = data

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.

        Args:
            idx: Index of the sequence to retrieve

        Returns:
            Dictionary with:
                - input_ids: Token IDs (block_size,)
                - labels: Target token IDs for next-token prediction (block_size,)
                - attention_mask: Mask indicating non-padding tokens (block_size,)
        """
        tokens = self.data[idx]

        # Truncate if too long
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]

        # Pad if too short
        if len(tokens) < self.block_size:
            padding_length = self.block_size - len(tokens)
            tokens = tokens + [self.pad_token_id] * padding_length

        # Convert to tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)

        # For causal LM, labels are the same as input_ids (shifted internally by model)
        labels = input_ids.clone()

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.pad_token_id).long()

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }


class SYNTHDataset(Dataset):
    """
    Dataset for PleIAs/SYNTH synthetic data.

    This dataset is specifically designed for the SYNTH dataset used to train
    Baguettotron, handling the structure and format of synthetic French text.

    Args:
        data_path: Path to SYNTH dataset file (JSON or JSONL)
        tokenizer: Tokenizer to encode text
        block_size: Maximum sequence length
        split: Dataset split ('train', 'validation', 'test')

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")
        >>> dataset = SYNTHDataset(
        ...     data_path='data/synth_train.jsonl',
        ...     tokenizer=tokenizer,
        ...     block_size=2048,
        ...     split='train'
        ... )
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: Any,
        block_size: int = 2048,
        split: str = 'train',
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.split = split

        # Load data
        self.examples = self._load_data()

    def _load_data(self) -> List[str]:
        """Load text examples from file."""
        examples = []

        with open(self.data_path, 'r', encoding='utf-8') as f:
            # Try JSONL format first
            try:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        # Extract text field (adjust based on SYNTH format)
                        text = item.get('text', item.get('content', ''))
                        if text:
                            examples.append(text)
            except json.JSONDecodeError:
                # Fall back to plain JSON
                f.seek(0)
                data = json.load(f)
                if isinstance(data, list):
                    examples = [
                        item.get('text', item.get('content', ''))
                        for item in data
                        if item.get('text') or item.get('content')
                    ]

        return examples

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single tokenized example.

        Args:
            idx: Index of the example

        Returns:
            Dictionary with input_ids, labels, and attention_mask
        """
        text = self.examples[idx]

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.block_size,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for the given dataset.

    Args:
        dataset: PyTorch Dataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data at each epoch
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        DataLoader instance

    Examples:
        >>> dataset = TextDataset(token_sequences, block_size=128)
        >>> dataloader = create_dataloader(
        ...     dataset,
        ...     batch_size=32,
        ...     shuffle=True,
        ...     num_workers=4
        ... )
        >>> for batch in dataloader:
        ...     input_ids = batch['input_ids']  # (32, 128)
        ...     labels = batch['labels']        # (32, 128)
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def preprocess_text_file(
    input_path: str | Path,
    output_path: str | Path,
    tokenizer: Any,
    block_size: int = 2048,
    stride: int = 1024,
) -> None:
    """
    Preprocess a text file into tokenized chunks.

    Reads a text file, tokenizes it, and splits it into overlapping chunks
    of the specified block size. Useful for creating training data from
    large text corpora.

    Args:
        input_path: Path to input text file
        output_path: Path to save tokenized data (JSON)
        tokenizer: Tokenizer to use for encoding
        block_size: Size of each chunk
        stride: Stride between chunks (overlap = block_size - stride)

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")
        >>> preprocess_text_file(
        ...     'data/corpus.txt',
        ...     'data/train_tokens.json',
        ...     tokenizer,
        ...     block_size=2048,
        ...     stride=1024
        ... )
    """
    # Read text
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenize entire text
    tokens = tokenizer.encode(text)

    # Split into chunks
    chunks = []
    for i in range(0, len(tokens) - block_size + 1, stride):
        chunk = tokens[i:i + block_size]
        chunks.append(chunk)

    # Add final chunk if needed
    if len(tokens) > block_size and len(tokens) % stride != 0:
        chunks.append(tokens[-block_size:])

    # Save
    with open(output_path, 'w') as f:
        json.dump(chunks, f)

    print(f"Preprocessed {len(tokens)} tokens into {len(chunks)} chunks")
    print(f"Saved to {output_path}")
