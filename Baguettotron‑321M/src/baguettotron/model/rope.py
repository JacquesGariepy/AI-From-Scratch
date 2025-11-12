"""
Rotary Position Embeddings (RoPE) for Baguettotron.

This module implements RoPE, a method for encoding positional information
in transformer models through rotation matrices applied to query and key vectors.

Reference:
    Su et al. (2021): RoFormer: Enhanced Transformer with Rotary Position Embedding
    https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn
from typing import Tuple


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build RoPE (Rotary Position Embedding) cache for efficient computation.

    Creates cosine and sine embeddings that will be used to rotate query/key vectors
    based on their positions in the sequence.

    Args:
        seq_len: Maximum sequence length to cache
        head_dim: Dimension of each attention head (must be even)
        theta: Base frequency for rotation (default: 10000.0)
        device: Device to create tensors on
        dtype: Data type for the cache tensors

    Returns:
        Tuple of (cos_cache, sin_cache), each of shape (seq_len, head_dim)
        in interleaved format for efficient application

    Note:
        The returned cache uses interleaved format where each dimension pair
        (2i, 2i+1) shares the same rotation frequency. This allows efficient
        application using element-wise operations.

    Examples:
        >>> cos_cache, sin_cache = build_rope_cache(512, 64)
        >>> assert cos_cache.shape == (512, 64)
        >>> assert sin_cache.shape == (512, 64)
    """
    # Compute inverse frequencies for each dimension pair
    # freq_i = 1 / (theta^(2i/head_dim)) for i in [0, head_dim/2)
    half_dim = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim)
    )

    # Create position indices
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)

    # Compute angles: positions × frequencies
    # Shape: (seq_len, head_dim/2)
    angles = torch.outer(positions, inv_freq)

    # Compute cos and sin
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # Interleave: [cos_0, cos_0, cos_1, cos_1, ...]
    # This format allows efficient application to [x_even, x_odd] pairs
    cos_cache = torch.stack([cos_angles, cos_angles], dim=-1).reshape(seq_len, head_dim)
    sin_cache = torch.stack([sin_angles, sin_angles], dim=-1).reshape(seq_len, head_dim)

    if dtype is not None:
        cos_cache = cos_cache.to(dtype)
        sin_cache = sin_cache.to(dtype)

    return cos_cache, sin_cache


def apply_rotary_pos_emb(
    query_or_key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embeddings to query or key tensors.

    This implements the rotation operation:
    [x_even, x_odd] -> [x_even*cos - x_odd*sin, x_even*sin + x_odd*cos]

    Args:
        query_or_key: Query or key tensor of shape (batch, heads, seq_len, head_dim)
        cos: Cosine cache of shape (seq_len, head_dim)
        sin: Sine cache of shape (seq_len, head_dim)

    Returns:
        Rotated tensor of the same shape as input

    Shape:
        - query_or_key: (batch, num_heads, seq_len, head_dim)
        - cos: (seq_len, head_dim) or broadcastable
        - sin: (seq_len, head_dim) or broadcastable
        - output: (batch, num_heads, seq_len, head_dim)

    Examples:
        >>> q = torch.randn(2, 8, 512, 64)  # (batch, heads, seq, head_dim)
        >>> cos, sin = build_rope_cache(512, 64)
        >>> # Broadcast cos/sin to match q shape
        >>> cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
        >>> sin = sin.unsqueeze(0).unsqueeze(0)
        >>> q_rotated = apply_rotary_pos_emb(q, cos, sin)
        >>> assert q_rotated.shape == q.shape
    """
    # Split into even and odd indices
    # x[..., 0::2] extracts elements at positions 0, 2, 4, ...
    # x[..., 1::2] extracts elements at positions 1, 3, 5, ...
    x_even = query_or_key[..., 0::2]
    x_odd = query_or_key[..., 1::2]

    # Extract corresponding cos/sin values
    cos_even = cos[..., 0::2]
    sin_even = sin[..., 0::2]

    # Apply rotation
    # Real part: x_even*cos - x_odd*sin
    # Imaginary part: x_even*sin + x_odd*cos
    rotated_even = x_even * cos_even - x_odd * sin_even
    rotated_odd = x_even * sin_even + x_odd * cos_even

    # Interleave results back
    # Stack along last dimension and reshape to original shape
    rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
    rotated = rotated.flatten(-2)

    return rotated


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding module with caching.

    This module maintains a cache of cos/sin embeddings and automatically
    extends it when sequences longer than the cache are encountered.

    Args:
        head_dim: Dimension of each attention head
        max_position_embeddings: Maximum sequence length to pre-cache (default: 2048)
        theta: Base frequency for rotations (default: 10000.0)

    Attributes:
        head_dim: Dimension of attention heads
        theta: Base rotation frequency
        max_seq_len_cached: Current maximum cached sequence length

    Examples:
        >>> rope = RotaryEmbedding(head_dim=64, max_position_embeddings=2048)
        >>> q = torch.randn(2, 8, 512, 64)
        >>> k = torch.randn(2, 8, 512, 64)
        >>> q_rot, k_rot = rope(q, k)
        >>> assert q_rot.shape == q.shape
        >>> assert k_rot.shape == k.shape
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.max_seq_len_cached = 0

        # Register buffers (not parameters, but part of state_dict)
        self.register_buffer("cos_cache", torch.empty(0), persistent=False)
        self.register_buffer("sin_cache", torch.empty(0), persistent=False)

        # Pre-build cache
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int):
        """Build or extend the RoPE cache."""
        if seq_len > self.max_seq_len_cached:
            # Get device and dtype from existing cache or use defaults
            device = self.cos_cache.device if self.cos_cache.numel() > 0 else None
            dtype = self.cos_cache.dtype if self.cos_cache.numel() > 0 else None

            # Build new cache
            cos_cache, sin_cache = build_rope_cache(
                seq_len,
                self.head_dim,
                self.theta,
                device=device,
                dtype=dtype,
            )

            self.cos_cache = cos_cache
            self.sin_cache = sin_cache
            self.max_seq_len_cached = seq_len

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            query: Query tensor of shape (batch, num_heads, seq_len, head_dim)
            key: Key tensor of shape (batch, num_heads, seq_len, head_dim)

        Returns:
            Tuple of (rotated_query, rotated_key) with same shapes as inputs
        """
        seq_len = query.shape[2]

        # Extend cache if needed
        self._build_cache(seq_len)

        # Get cache for current sequence length and broadcast to query/key shape
        # cos/sin: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)

        # Ensure correct dtype
        cos = cos.to(query.dtype)
        sin = sin.to(key.dtype)

        # Apply rotations
        query_rotated = apply_rotary_pos_emb(query, cos, sin)
        key_rotated = apply_rotary_pos_emb(key, cos, sin)

        return query_rotated, key_rotated
