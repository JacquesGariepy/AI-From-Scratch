"""
Text generation utilities for Baguettotron.

This module provides advanced sampling strategies, stopping criteria,
and utilities for generating high-quality text from Baguettotron models.

Sampling Functions:
    sample_from_logits: Sample tokens with various strategies
    top_k_filtering: Keep only top-k tokens
    top_p_filtering: Nucleus (top-p) sampling
    typical_filtering: Typical decoding
    min_p_filtering: Min-p sampling
    repetition_penalty_apply: Apply repetition penalty

Stopping Criteria:
    StoppingCriteria: Base class for stopping criteria
    MaxLengthCriteria: Stop at maximum length
    EosTokenCriteria: Stop at end-of-sequence token
    StoppingCriteriaList: Combine multiple criteria

Example:
    >>> import torch
    >>> from baguettotron import BaguettotronForCausalLM, BaguettotronConfig
    >>> from baguettotron.generation import sample_from_logits
    >>>
    >>> # Setup model
    >>> config = BaguettotronConfig.baguettotron_321m()
    >>> model = BaguettotronForCausalLM(config)
    >>> model.eval()
    >>>
    >>> # Generate with custom sampling
    >>> input_ids = torch.randint(0, config.vocab_size, (1, 10))
    >>> with torch.no_grad():
    ...     for _ in range(50):
    ...         logits = model(input_ids)
    ...         next_token_logits = logits[:, -1, :]
    ...
    ...         # Sample next token
    ...         next_token = sample_from_logits(
    ...             next_token_logits,
    ...             temperature=0.8,
    ...             top_k=50,
    ...             top_p=0.9,
    ...             do_sample=True
    ...         )
    ...
    ...         # Append to sequence
    ...         input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

Using built-in generate method:
    >>> # Use model's built-in generate method
    >>> output = model.generate(
    ...     input_ids,
    ...     max_new_tokens=50,
    ...     temperature=0.8,
    ...     top_k=50,
    ...     top_p=0.9,
    ...     do_sample=True
    ... )
"""

from .utils import (
    sample_from_logits,
    top_k_filtering,
    top_p_filtering,
    typical_filtering,
    min_p_filtering,
    repetition_penalty_apply,
    StoppingCriteria,
    MaxLengthCriteria,
    EosTokenCriteria,
    StoppingCriteriaList,
)

__all__ = [
    # Sampling functions
    "sample_from_logits",
    "top_k_filtering",
    "top_p_filtering",
    "typical_filtering",
    "min_p_filtering",
    "repetition_penalty_apply",
    # Stopping criteria
    "StoppingCriteria",
    "MaxLengthCriteria",
    "EosTokenCriteria",
    "StoppingCriteriaList",
]
