"""
Transformer block for Baguettotron.

This module implements the transformer decoder block with pre-normalization,
as used in LLaMA and Baguettotron models.
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import GroupedQueryAttention
from .feedforward import SwiGLU
from .normalization import RMSNorm
from ..config import BaguettotronConfig


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block with pre-normalization.

    This block follows the LLaMA architecture:
    1. Input -> RMSNorm -> Attention -> Residual
    2. Hidden -> RMSNorm -> MLP -> Residual

    Pre-normalization (norm before attention/MLP) is used instead of post-normalization
    for better training stability.

    Args:
        config: Model configuration containing all hyperparameters

    Attributes:
        attention_norm: RMSNorm layer before attention
        attention: Grouped query attention layer
        ffn_norm: RMSNorm layer before feed-forward
        feed_forward: SwiGLU feed-forward network

    Shape:
        - Input: (batch_size, seq_len, hidden_size)
        - Output: (batch_size, seq_len, hidden_size)

    Examples:
        >>> from baguettotron.config import BaguettotronConfig
        >>> config = BaguettotronConfig(
        ...     hidden_size=512,
        ...     num_attention_heads=8,
        ...     num_key_value_heads=4,
        ...     intermediate_size=2048
        ... )
        >>> block = TransformerBlock(config)
        >>> x = torch.randn(2, 128, 512)
        >>> output = block(x)
        >>> assert output.shape == x.shape
    """

    def __init__(self, config: BaguettotronConfig):
        super().__init__()

        # Pre-normalization layers
        self.attention_norm = RMSNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.ffn_norm = RMSNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Attention layer
        self.attention = GroupedQueryAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            bias=False,  # LLaMA-style: no bias
        )

        # Feed-forward network (SwiGLU)
        self.feed_forward = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=False,  # LLaMA-style: no bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            is_causal: Whether to use causal attention masking

        Returns:
            Output tensor of same shape as input

        Note:
            Uses pre-normalization with residual connections:
            1. residual = hidden_states
            2. hidden_states = residual + attention(norm(residual))
            3. residual = hidden_states
            4. hidden_states = residual + ffn(norm(residual))
        """
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            is_causal=is_causal,
        )
        hidden_states = residual + hidden_states

        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TransformerDecoder(nn.Module):
    """
    Stack of transformer decoder blocks.

    This module stacks multiple TransformerBlock layers to form the main
    transformer decoder.

    Args:
        config: Model configuration

    Attributes:
        layers: ModuleList of TransformerBlock layers

    Examples:
        >>> config = BaguettotronConfig(num_hidden_layers=12)
        >>> decoder = TransformerDecoder(config)
        >>> x = torch.randn(2, 128, config.hidden_size)
        >>> output = decoder(x)
        >>> assert output.shape == x.shape
    """

    def __init__(self, config: BaguettotronConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through all transformer layers.

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            is_causal: Whether to use causal masking

        Returns:
            Output tensor of same shape as input
        """
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                is_causal=is_causal,
            )

        return hidden_states
