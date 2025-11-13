"""
Baguettotron-321M: A 321 Million Parameter Language Model

Baguettotron is a decoder-only transformer language model compatible with
LLaMA architecture, featuring Grouped Query Attention (GQA), SwiGLU feed-forward
networks, RoPE position embeddings, and RMSNorm normalization.

Architecture Highlights:
    - 321 million parameters
    - 24 transformer layers
    - 576 hidden dimensions
    - 9 attention heads with 3 key-value heads (GQA)
    - 1536 intermediate dimensions in SwiGLU
    - 65,536 vocabulary size
    - 2048 maximum sequence length

Quick Start:
    >>> import torch
    >>> from baguettotron import BaguettotronForCausalLM, BaguettotronConfig
    >>>
    >>> # Load official 321M configuration
    >>> config = BaguettotronConfig.baguettotron_321m()
    >>>
    >>> # Create model
    >>> model = BaguettotronForCausalLM(config)
    >>>
    >>> # Generate text
    >>> input_ids = torch.randint(0, config.vocab_size, (1, 10))
    >>> output = model.generate(
    ...     input_ids,
    ...     max_new_tokens=50,
    ...     temperature=0.8,
    ...     do_sample=True
    ... )

Modules:
    config: Configuration dataclasses and utilities
    model: Neural network components and architectures
    data: Dataset loaders and preprocessing utilities
    training: Training loops, optimizers, and utilities
    generation: Text generation and sampling strategies
    tokenization: Tokenizer loading with automatic fallback

References:
    - Official Model: https://huggingface.co/PleIAs/Baguettotron
    - Dataset: https://huggingface.co/datasets/PleIAs/SYNTH
    - Paper: [Baguettotron Technical Report]
"""

__version__ = "0.1.0"
__author__ = "Baguettotron Team"
__license__ = "MIT"

from .config import BaguettotronConfig
from .model import (
    BaguettotronForCausalLM,
    TransformerBlock,
    TransformerDecoder,
    GroupedQueryAttention,
    SwiGLU,
    RMSNorm,
    RotaryEmbedding,
)
from .tokenization import load_tokenizer, TokenizerWrapper

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Configuration
    "BaguettotronConfig",
    # Main model
    "BaguettotronForCausalLM",
    # Tokenization
    "load_tokenizer",
    "TokenizerWrapper",
    # Core components (for advanced usage)
    "TransformerBlock",
    "TransformerDecoder",
    "GroupedQueryAttention",
    "SwiGLU",
    "RMSNorm",
    "RotaryEmbedding",
]
