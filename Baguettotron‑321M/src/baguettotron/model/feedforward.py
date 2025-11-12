"""
Feed-Forward Networks for Baguettotron.

This module implements the SwiGLU (Swish-Gated Linear Unit) MLP architecture
used in LLaMA and Baguettotron models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    SwiGLU is a gated feed-forward network that uses SiLU (Swish) activation.
    It consists of three linear projections: gate_proj, up_proj, and down_proj.

    The forward pass computes:
        output = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    This architecture is used in LLaMA, PaLM, and other modern LLMs as it
    provides better performance than standard MLP with ReLU/GELU.

    Reference:
        Shazeer (2020): GLU Variants Improve Transformer
        https://arxiv.org/abs/2002.05202

    Args:
        hidden_size: Dimension of input and output
        intermediate_size: Dimension of the intermediate layer
        bias: Whether to use bias in linear layers (default: False for LLaMA-style)

    Attributes:
        gate_proj: Linear projection for the gate
        up_proj: Linear projection for the value
        down_proj: Linear projection back to hidden_size

    Shape:
        - Input: (*, hidden_size)
        - Output: (*, hidden_size)

    Examples:
        >>> mlp = SwiGLU(hidden_size=512, intermediate_size=2048)
        >>> x = torch.randn(2, 10, 512)
        >>> output = mlp(x)
        >>> assert output.shape == x.shape

        >>> # Count parameters
        >>> params = sum(p.numel() for p in mlp.parameters())
        >>> expected = 512 * 2048 * 3  # gate + up + down
        >>> assert params == expected
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Three projections for SwiGLU
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation.

        Args:
            hidden_states: Input tensor of shape (*, hidden_size)

        Returns:
            Output tensor of shape (*, hidden_size)

        Note:
            The computation is:
            1. gate = SiLU(gate_proj(x))
            2. up = up_proj(x)
            3. output = down_proj(gate * up)
        """
        # Compute gate with SiLU activation
        gate = F.silu(self.gate_proj(hidden_states))

        # Compute up projection
        up = self.up_proj(hidden_states)

        # Element-wise multiplication and down projection
        output = self.down_proj(gate * up)

        return output

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )


class MLP(nn.Module):
    """
    Standard MLP with configurable activation.

    This is a simpler alternative to SwiGLU, using a standard two-layer MLP.
    It's kept for compatibility and comparison purposes.

    Args:
        hidden_size: Dimension of input and output
        intermediate_size: Dimension of the intermediate layer
        activation: Activation function name ('silu', 'gelu', 'relu')
        bias: Whether to use bias in linear layers

    Examples:
        >>> mlp = MLP(hidden_size=512, intermediate_size=2048, activation='silu')
        >>> x = torch.randn(2, 10, 512)
        >>> output = mlp(x)
        >>> assert output.shape == x.shape
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)

        # Select activation function
        if activation == "silu":
            self.activation = F.silu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply MLP transformation."""
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
