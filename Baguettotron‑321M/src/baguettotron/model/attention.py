"""
Attention mechanisms for Baguettotron.

This module implements Grouped Query Attention (GQA), which is used in
Baguettotron-321M and LLaMA-2 models for efficient multi-head attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with RoPE.

    GQA is a variant of multi-head attention where multiple query heads share
    the same key-value heads. This reduces the memory footprint and computational
    cost while maintaining model quality.

    When num_key_value_heads == num_attention_heads, this is standard MHA.
    When num_key_value_heads == 1, this is Multi-Query Attention (MQA).
    Otherwise, it's GQA with multiple query heads per key-value head.

    Reference:
        Ainslie et al. (2023): GQA: Training Generalized Multi-Query Transformer Models
        https://arxiv.org/abs/2305.13245

    Args:
        hidden_size: Dimension of input embeddings
        num_attention_heads: Number of query attention heads
        num_key_value_heads: Number of key-value heads (for GQA)
        max_position_embeddings: Maximum sequence length
        rope_theta: Base frequency for RoPE
        attention_dropout: Dropout probability for attention weights
        bias: Whether to use bias in projections

    Attributes:
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        num_kv_groups: Number of query heads per key-value head
        head_dim: Dimension of each attention head

    Shape:
        - Input: (batch_size, seq_len, hidden_size)
        - Output: (batch_size, seq_len, hidden_size)

    Examples:
        >>> # Standard GQA with 12 query heads and 4 KV heads
        >>> attn = GroupedQueryAttention(
        ...     hidden_size=768,
        ...     num_attention_heads=12,
        ...     num_key_value_heads=4
        ... )
        >>> x = torch.randn(2, 128, 768)
        >>> output = attn(x)
        >>> assert output.shape == x.shape

        >>> # Multi-Query Attention (1 KV head)
        >>> mqa = GroupedQueryAttention(
        ...     hidden_size=768,
        ...     num_attention_heads=12,
        ...     num_key_value_heads=1
        ... )
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        attention_dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.attention_dropout = attention_dropout

        # Validate configuration
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )

        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({num_key_value_heads})"
            )

        self.head_dim = hidden_size // num_attention_heads
        self.num_kv_groups = num_attention_heads // num_key_value_heads

        # Query projection (full set of heads)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=bias)

        # Key-Value projections (reduced number of heads for GQA)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=bias)

        # Output projection
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=bias)

        # Rotary position embeddings with optional scaling
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            theta=rope_theta,
            rope_scaling=rope_scaling,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Apply grouped query attention.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask (batch, 1, seq_len, seq_len) or (seq_len, seq_len)
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        # q: (batch, seq_len, num_heads * head_dim)
        # k, v: (batch, seq_len, num_kv_heads * head_dim)
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to separate heads
        # q: (batch, num_heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # k, v: (batch, num_kv_heads, seq_len, head_dim)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        query, key = self.rotary_emb(query, key)

        # Repeat K and V for GQA if needed
        if self.num_kv_groups > 1:
            # Repeat each KV head num_kv_groups times
            # (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
            key = key.repeat_interleave(self.num_kv_groups, dim=1)
            value = value.repeat_interleave(self.num_kv_groups, dim=1)

        # ✅ FIX #2: Properly combine causal mask with padding mask
        # Prepare attention mask if provided
        if attention_mask is not None:
            # Handle different mask shapes
            if attention_mask.dim() == 2:
                # Shape: (batch, seq_len) -> (batch, 1, 1, seq_len)
                # This broadcasts to (batch, num_heads, seq_len, seq_len)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                # Shape: (batch, 1, seq_len) -> (batch, 1, 1, seq_len)
                attention_mask = attention_mask.unsqueeze(2)

            # Convert boolean or int mask to additive mask if needed
            if attention_mask.dtype == torch.bool or attention_mask.dtype == torch.long:
                # Create float mask with -inf for masked positions
                float_mask = torch.zeros_like(attention_mask, dtype=query.dtype)
                # For bool: True = attend, False = mask
                # For long: 1 = attend, 0 = mask
                if attention_mask.dtype == torch.bool:
                    float_mask.masked_fill_(~attention_mask, float("-inf"))
                else:  # long
                    float_mask.masked_fill_(attention_mask == 0, float("-inf"))
                attention_mask = float_mask

            # Combine with causal mask if needed
            if is_causal:
                # Create causal mask: upper triangle = True (positions to mask)
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device),
                    diagonal=1
                )
                # Expand to (1, 1, L, L) for broadcasting
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

                # Expand attention_mask to full (B, 1, L, L) shape for causal combination
                # Current shape: (B, 1, 1, L) - broadcast to (B, 1, L, L)
                expanded_mask = attention_mask.expand(-1, -1, seq_len, -1)

                # Apply causal mask to combined mask
                expanded_mask = expanded_mask.clone()
                expanded_mask.masked_fill_(causal_mask, float("-inf"))
                attention_mask = expanded_mask

                # Use manual mask, not automatic causal
                is_causal_flag = False
            else:
                is_causal_flag = False
        else:
            # No padding mask, use automatic causal masking
            is_causal_flag = is_causal

        # Compute attention using scaled_dot_product_attention
        # This automatically handles causal masking and is optimized
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal_flag,  # ✅ FIX #2: Use corrected flag
        )

        # Reshape back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.o_proj(attn_output)

        return output

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"dropout={self.attention_dropout}"
        )


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention (for reference/comparison).

    This is kept for compatibility but is less efficient than GQA.
    Use GroupedQueryAttention with num_kv_heads == num_heads for equivalent behavior.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        # Delegate to GQA with same number of KV heads
        self.gqa = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,  # MHA = GQA with equal heads
            attention_dropout=attention_dropout,
            bias=bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Forward pass delegates to GQA."""
        return self.gqa(hidden_states, attention_mask, is_causal)


class MaskedGroupQueryAttention(nn.Module):
    """
    Masked Group-Query Attention with explicit causal masking.

    This is an alternative implementation of GQA that uses explicit
    attention masking rather than PyTorch's scaled_dot_product_attention.
    It's useful for testing, debugging, and ensuring compatibility with
    custom attention patterns.

    Unlike GroupedQueryAttention which uses F.scaled_dot_product_attention,
    this implementation computes attention scores explicitly with manual masking.

    Args:
        hidden_size: Dimension of input embeddings
        num_attention_heads: Number of query attention heads
        num_key_value_heads: Number of key-value heads (for GQA)
        max_position_embeddings: Maximum sequence length
        rope_theta: Base frequency for RoPE
        bias: Whether to use bias in projections

    Examples:
        >>> # Compatible with original mgqa.py tests
        >>> attn = MaskedGroupQueryAttention(
        ...     hidden_size=512,
        ...     num_attention_heads=8,
        ...     num_key_value_heads=4,
        ... )
        >>> x = torch.randn(2, 10, 512)
        >>> output = attn(x, is_causal=True)
        >>> assert output.shape == x.shape
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads

        # Validate configuration
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )

        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({num_key_value_heads})"
            )

        self.head_dim = hidden_size // num_attention_heads
        self.num_kv_groups = num_attention_heads // num_key_value_heads

        # Query projection (full set of heads)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=bias)

        # Key-Value projections (reduced number of heads for GQA)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=bias)

        # Output projection (named out_proj for compatibility with mgqa.py)
        self.out_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=bias)

        # Rotary position embeddings
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            theta=rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Apply masked grouped query attention.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional additive attention mask (batch, 1, seq_len, seq_len)
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to separate heads
        # q: (batch, num_heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # k, v: (batch, num_kv_heads, seq_len, head_dim)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        query, key = self.rotary_emb(query, key)

        # Expand grouped KV to per-head if needed
        if self.num_kv_groups > 1:
            # Repeat each KV head num_kv_groups times
            key = key.repeat_interleave(self.num_kv_groups, dim=1)
            value = value.repeat_interleave(self.num_kv_groups, dim=1)

        # Compute attention scores
        # (batch, num_heads, seq_len, seq_len)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask if requested
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        # Apply additional attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax over key dimension
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        # (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, value)

        # Reshape back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.out_proj(attn_output)

        return output
