"""
Baguettotron Model Components.

This module provides the core neural network components for the Baguettotron-321M
transformer language model, including attention mechanisms, feed-forward networks,
and the complete causal language modeling architecture.

Main Classes:
    BaguettotronForCausalLM: Complete model for autoregressive language generation
    TransformerBlock: Single transformer decoder block with pre-normalization
    TransformerDecoder: Stack of transformer blocks
    GroupedQueryAttention: GQA/MQA/MHA attention with RoPE
    SwiGLU: Gated feed-forward network with SiLU activation
    RMSNorm: Root Mean Square Layer Normalization
    RotaryEmbedding: Rotary Position Embeddings

Example:
    >>> from baguettotron.config import BaguettotronConfig
    >>> from baguettotron.model import BaguettotronForCausalLM
    >>>
    >>> config = BaguettotronConfig.baguettotron_321m()
    >>> model = BaguettotronForCausalLM(config)
    >>>
    >>> import torch
    >>> input_ids = torch.randint(0, config.vocab_size, (1, 128))
    >>> logits = model(input_ids)
"""

from .causal_lm import BaguettotronForCausalLM
from .transformer import TransformerBlock, TransformerDecoder
from .attention import GroupedQueryAttention, MultiHeadAttention, MaskedGroupQueryAttention
from .feedforward import SwiGLU, MLP
from .normalization import RMSNorm
from .rope import RotaryEmbedding, build_rope_cache, apply_rotary_pos_emb

__all__ = [
    # Main model
    "BaguettotronForCausalLM",
    # Transformer components
    "TransformerBlock",
    "TransformerDecoder",
    # Attention mechanisms
    "GroupedQueryAttention",
    "MultiHeadAttention",
    "MaskedGroupQueryAttention",
    # Feed-forward networks
    "SwiGLU",
    "MLP",
    # Normalization
    "RMSNorm",
    # Position embeddings
    "RotaryEmbedding",
    "build_rope_cache",
    "apply_rotary_pos_emb",
]
