"""
Text generation utilities for Baguettotron.

This module provides advanced sampling strategies and generation utilities
for high-quality text generation from Baguettotron language models.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable, List
import logging

logger = logging.getLogger(__name__)


def top_k_filtering(
    logits: torch.Tensor,
    top_k: int,
    filter_value: float = float('-inf'),
) -> torch.Tensor:
    """
    Filter logits to keep only top-k tokens.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        top_k: Number of top tokens to keep
        filter_value: Value to use for filtered tokens

    Returns:
        Filtered logits tensor

    Examples:
        >>> logits = torch.randn(1, 1000)
        >>> filtered = top_k_filtering(logits, top_k=50)
        >>> # Only top 50 tokens have non-inf values
    """
    if top_k <= 0:
        return logits

    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value

    return logits


def top_p_filtering(
    logits: torch.Tensor,
    top_p: float,
    filter_value: float = float('-inf'),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """
    Filter logits using nucleus (top-p) sampling.

    Keeps the smallest set of tokens whose cumulative probability exceeds top_p.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        top_p: Cumulative probability threshold (0 < top_p <= 1)
        filter_value: Value to use for filtered tokens
        min_tokens_to_keep: Minimum number of tokens to keep

    Returns:
        Filtered logits tensor

    Examples:
        >>> logits = torch.randn(1, 1000)
        >>> filtered = top_p_filtering(logits, top_p=0.9)
        >>> # Tokens with cumulative prob > 0.9 are filtered
    """
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p

    # Keep at least min_tokens_to_keep
    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

    # Shift right to keep the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = filter_value

    return logits


def typical_filtering(
    logits: torch.Tensor,
    mass: float = 0.9,
    filter_value: float = float('-inf'),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """
    Filter logits using typical decoding.

    Typical decoding selects tokens whose information content is close
    to the expected information content.

    Reference:
        Meister et al. (2022): Typical Decoding for Natural Language Generation
        https://arxiv.org/abs/2202.00666

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        mass: Mass of probability to keep (similar to top_p)
        filter_value: Value to use for filtered tokens
        min_tokens_to_keep: Minimum number of tokens to keep

    Returns:
        Filtered logits tensor

    Examples:
        >>> logits = torch.randn(1, 1000)
        >>> filtered = typical_filtering(logits, mass=0.9)
    """
    if mass >= 1.0:
        return logits

    # Compute probabilities and entropy
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1, keepdim=True)

    # Compute negative log probabilities
    neg_log_probs = -torch.log(probs + 1e-10)

    # Compute deviation from expected information
    deviation = torch.abs(neg_log_probs - entropy)

    # Sort by deviation (ascending)
    sorted_deviation, sorted_indices = torch.sort(deviation, dim=-1)
    sorted_probs = probs.gather(-1, sorted_indices)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens exceeding mass
    sorted_indices_to_remove = cumulative_probs > mass

    # Keep at least min_tokens_to_keep
    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

    # Scatter to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = filter_value

    return logits


def min_p_filtering(
    logits: torch.Tensor,
    min_p: float = 0.05,
    filter_value: float = float('-inf'),
) -> torch.Tensor:
    """
    Filter logits using min-p sampling.

    Keeps tokens with probability >= min_p * max_probability.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        min_p: Minimum probability threshold (relative to max)
        filter_value: Value to use for filtered tokens

    Returns:
        Filtered logits tensor

    Examples:
        >>> logits = torch.randn(1, 1000)
        >>> filtered = min_p_filtering(logits, min_p=0.05)
        >>> # Tokens with prob < 0.05 * max_prob are filtered
    """
    if min_p <= 0:
        return logits

    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True)[0]
    threshold = min_p * max_prob

    indices_to_remove = probs < threshold
    logits[indices_to_remove] = filter_value

    return logits


def repetition_penalty_apply(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    penalty: float = 1.0,
) -> torch.Tensor:
    """
    Apply repetition penalty to logits.

    Reduces the probability of tokens that have already been generated.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        generated_tokens: Previously generated tokens (batch_size, seq_len)
        penalty: Penalty strength (> 1.0 discourages repetition)

    Returns:
        Logits with repetition penalty applied

    Examples:
        >>> logits = torch.randn(1, 1000)
        >>> generated = torch.randint(0, 1000, (1, 50))
        >>> logits = repetition_penalty_apply(logits, generated, penalty=1.2)
    """
    if penalty == 1.0:
        return logits

    batch_size, vocab_size = logits.shape

    # For each token in generated_tokens, apply penalty
    for i in range(batch_size):
        for token in generated_tokens[i].unique():
            # If logit is positive, divide by penalty; if negative, multiply
            if logits[i, token] < 0:
                logits[i, token] *= penalty
            else:
                logits[i, token] /= penalty

    return logits


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    typical_p: Optional[float] = None,
    min_p: Optional[float] = None,
    do_sample: bool = True,
) -> torch.Tensor:
    """
    Sample next tokens from logits with various strategies.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top-k tokens
        top_p: Nucleus sampling threshold
        typical_p: Typical sampling mass
        min_p: Min-p sampling threshold
        do_sample: Whether to sample (True) or use greedy (False)

    Returns:
        Sampled token IDs of shape (batch_size,)

    Examples:
        >>> logits = torch.randn(2, 1000)
        >>>
        >>> # Greedy decoding
        >>> tokens = sample_from_logits(logits, do_sample=False)
        >>>
        >>> # Sampling with temperature
        >>> tokens = sample_from_logits(logits, temperature=0.8, do_sample=True)
        >>>
        >>> # Top-k sampling
        >>> tokens = sample_from_logits(logits, top_k=50, do_sample=True)
        >>>
        >>> # Nucleus (top-p) sampling
        >>> tokens = sample_from_logits(logits, top_p=0.9, do_sample=True)
        >>>
        >>> # Combined: top-k + top-p
        >>> tokens = sample_from_logits(
        ...     logits, temperature=0.8, top_k=50, top_p=0.9, do_sample=True
        ... )
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Apply filtering strategies
    if top_k is not None:
        logits = top_k_filtering(logits, top_k)

    if top_p is not None:
        logits = top_p_filtering(logits, top_p)

    if typical_p is not None:
        logits = typical_filtering(logits, typical_p)

    if min_p is not None:
        logits = min_p_filtering(logits, min_p)

    # Sample or take argmax
    if do_sample:
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    else:
        next_tokens = torch.argmax(logits, dim=-1)

    return next_tokens


class StoppingCriteria:
    """
    Base class for generation stopping criteria.

    Subclasses should implement __call__ to determine when to stop generation.
    """

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        Check if generation should stop.

        Args:
            input_ids: Generated token IDs (batch_size, seq_len)
            scores: Optional logits or scores

        Returns:
            True if generation should stop
        """
        raise NotImplementedError


class MaxLengthCriteria(StoppingCriteria):
    """Stop generation when max length is reached."""

    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids: torch.Tensor, scores: Optional[torch.Tensor] = None) -> bool:
        return input_ids.shape[-1] >= self.max_length


class EosTokenCriteria(StoppingCriteria):
    """Stop generation when EOS token is generated."""

    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.Tensor, scores: Optional[torch.Tensor] = None) -> bool:
        return (input_ids[:, -1] == self.eos_token_id).all().item()


class StoppingCriteriaList(list):
    """List of stopping criteria."""

    def __call__(self, input_ids: torch.Tensor, scores: Optional[torch.Tensor] = None) -> bool:
        """Return True if any criterion is met."""
        return any(criterion(input_ids, scores) for criterion in self)
