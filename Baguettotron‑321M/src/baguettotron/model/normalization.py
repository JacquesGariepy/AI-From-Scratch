"""
Normalization layers for Baguettotron.

This module implements RMSNorm (Root Mean Square Layer Normalization),
which is used in LLaMA-style architectures as a more efficient alternative
to LayerNorm.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm simplifies LayerNorm by removing the mean centering and bias,
    focusing only on the root mean square for normalization. This is more
    efficient and works well for transformer models.

    Reference:
        Zhang & Sennrich (2019): Root Mean Square Layer Normalization
        https://arxiv.org/abs/1910.07467

    Args:
        normalized_shape: Dimension to normalize (typically hidden_size)
        eps: Small constant for numerical stability (default: 1e-6)

    Attributes:
        weight: Learnable scale parameter of shape (normalized_shape,)
        eps: Epsilon value for numerical stability

    Shape:
        - Input: (*, normalized_shape)
        - Output: (*, normalized_shape) (same shape as input)

    Examples:
        >>> norm = RMSNorm(512, eps=1e-5)
        >>> x = torch.randn(2, 10, 512)
        >>> output = norm(x)
        >>> assert output.shape == x.shape
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            hidden_states: Input tensor of shape (*, normalized_shape)

        Returns:
            Normalized tensor of the same shape as input

        Note:
            Normalization is computed as:
            output = hidden_states / rms(hidden_states) * weight
            where rms(x) = sqrt(mean(x^2) + eps)
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # Compute RMS
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # Apply learnable scale and convert back to original dtype
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return f"normalized_shape={tuple(self.weight.shape)}, eps={self.eps}"
