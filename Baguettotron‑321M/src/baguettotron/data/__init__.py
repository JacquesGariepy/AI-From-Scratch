"""
Data utilities for Baguettotron training.

This module provides dataset classes, data collators, and preprocessing
utilities for loading and preparing text data for language model training.

Main Classes:
    TextDataset: Generic text dataset for causal LM
    SYNTHDataset: Dataset for PleIAs/SYNTH synthetic data
    DataCollatorForLanguageModeling: Collator with dynamic padding
    DataCollatorForCausalLM: Simplified collator for pre-padded data

Functions:
    create_dataloader: Create a DataLoader with sensible defaults
    preprocess_text_file: Tokenize and chunk large text files

Example:
    >>> from baguettotron.data import TextDataset, create_dataloader
    >>>
    >>> # Load tokenized data
    >>> dataset = TextDataset('data/train_tokens.json', block_size=2048)
    >>>
    >>> # Create dataloader
    >>> dataloader = create_dataloader(
    ...     dataset,
    ...     batch_size=32,
    ...     shuffle=True,
    ...     num_workers=4
    ... )
    >>>
    >>> # Train
    >>> for batch in dataloader:
    ...     input_ids = batch['input_ids']
    ...     labels = batch['labels']
"""

from .dataset import (
    TextDataset,
    SYNTHDataset,
    create_dataloader,
    preprocess_text_file,
)

from .collator import (
    DataCollatorForLanguageModeling,
    DataCollatorForCausalLM,
    default_data_collator,
)

__all__ = [
    # Datasets
    "TextDataset",
    "SYNTHDataset",
    # Collators
    "DataCollatorForLanguageModeling",
    "DataCollatorForCausalLM",
    "default_data_collator",
    # Utilities
    "create_dataloader",
    "preprocess_text_file",
]
