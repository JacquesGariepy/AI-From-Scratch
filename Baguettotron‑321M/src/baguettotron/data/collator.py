"""
Data collators for batching and padding.

This module provides collator classes that handle batching, padding, and
preparing data for causal language modeling.
"""

import torch
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator for causal language modeling.

    This collator handles dynamic padding and prepares batches for
    next-token prediction training.

    Args:
        pad_token_id: Token ID to use for padding
        mlm: Whether to use masked language modeling (not supported, must be False)

    Examples:
        >>> collator = DataCollatorForLanguageModeling(pad_token_id=0)
        >>> batch = [
        ...     {'input_ids': torch.tensor([1, 2, 3])},
        ...     {'input_ids': torch.tensor([4, 5, 6, 7])},
        ... ]
        >>> collated = collator(batch)
        >>> assert collated['input_ids'].shape == (2, 4)  # Padded to max length
    """

    pad_token_id: int = 0
    mlm: bool = False

    def __post_init__(self):
        if self.mlm:
            raise ValueError(
                "This collator only supports causal language modeling. "
                "Set mlm=False."
            )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of examples into a batch.

        Args:
            examples: List of dictionaries with 'input_ids' and optionally
                     'attention_mask' and 'labels'

        Returns:
            Dictionary with batched tensors:
                - input_ids: (batch_size, max_length)
                - attention_mask: (batch_size, max_length)
                - labels: (batch_size, max_length)
        """
        # Extract input_ids
        input_ids = [example['input_ids'] for example in examples]

        # Find max length in batch
        max_length = max(len(ids) for ids in input_ids)

        # Pad sequences
        padded_input_ids = []
        attention_masks = []
        labels = []

        for example in examples:
            ids = example['input_ids']
            seq_length = len(ids)

            # Pad input_ids
            if seq_length < max_length:
                padding_length = max_length - seq_length
                padded_ids = torch.cat([
                    ids,
                    torch.full((padding_length,), self.pad_token_id, dtype=ids.dtype)
                ])
            else:
                padded_ids = ids

            padded_input_ids.append(padded_ids)

            # Create or pad attention mask
            if 'attention_mask' in example:
                mask = example['attention_mask']
                if seq_length < max_length:
                    padding_length = max_length - seq_length
                    mask = torch.cat([
                        mask,
                        torch.zeros(padding_length, dtype=mask.dtype)
                    ])
            else:
                # Create mask from input_ids
                mask = (padded_ids != self.pad_token_id).long()

            attention_masks.append(mask)

            # Handle labels
            if 'labels' in example:
                lbls = example['labels']
                if seq_length < max_length:
                    padding_length = max_length - seq_length
                    # Use -100 for padding in labels (ignored by loss)
                    lbls = torch.cat([
                        lbls,
                        torch.full((padding_length,), -100, dtype=lbls.dtype)
                    ])
            else:
                # Labels are input_ids shifted by 1 (handled by model)
                lbls = padded_ids.clone()

            labels.append(lbls)

        # Stack into batches
        batch = {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': torch.stack(labels),
        }

        return batch


class DataCollatorForCausalLM:
    """
    Simplified data collator for causal language modeling.

    This collator assumes examples are already padded to the same length
    and simply stacks them into batches.

    Args:
        return_tensors: Type of tensors to return ('pt' for PyTorch)

    Examples:
        >>> collator = DataCollatorForCausalLM()
        >>> batch = [
        ...     {
        ...         'input_ids': torch.tensor([1, 2, 3, 0]),
        ...         'labels': torch.tensor([1, 2, 3, 0]),
        ...         'attention_mask': torch.tensor([1, 1, 1, 0]),
        ...     },
        ...     {
        ...         'input_ids': torch.tensor([4, 5, 6, 7]),
        ...         'labels': torch.tensor([4, 5, 6, 7]),
        ...         'attention_mask': torch.tensor([1, 1, 1, 1]),
        ...     },
        ... ]
        >>> collated = collator(batch)
        >>> assert collated['input_ids'].shape == (2, 4)
    """

    def __init__(self, return_tensors: str = 'pt'):
        self.return_tensors = return_tensors

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate examples into a batch.

        Args:
            examples: List of example dictionaries

        Returns:
            Batched dictionary with stacked tensors
        """
        if not examples:
            return {}

        # Get all keys from first example
        keys = examples[0].keys()

        # Stack each field
        batch = {}
        for key in keys:
            values = [example[key] for example in examples]

            # Stack tensors
            if isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            else:
                batch[key] = torch.tensor(values)

        return batch


def default_data_collator(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Default collator that simply stacks pre-padded examples.

    Args:
        examples: List of example dictionaries with tensor values

    Returns:
        Batched dictionary

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> dataset = TextDataset(data, block_size=128)
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     collate_fn=default_data_collator
        ... )
    """
    return DataCollatorForCausalLM()(examples)
