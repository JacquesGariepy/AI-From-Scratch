"""
Baguettotron-321M Configuration

This module defines the model configuration class that holds all hyperparameters
for the Baguettotron-321M language model. The configuration is compatible with
LlamaForCausalLM architecture.

Reference:
    - Official model: https://huggingface.co/PleIAs/Baguettotron
    - Config: https://huggingface.co/PleIAs/Baguettotron/blob/main/config.json
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaguettotronConfig:
    """
    Configuration class for Baguettotron-321M model.

    This configuration is designed to be compatible with HuggingFace's
    LlamaForCausalLM architecture.

    Attributes:
        vocab_size: Size of the vocabulary (default: 65536 for official model)
        hidden_size: Dimensionality of embeddings and hidden states
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of query attention heads
        num_key_value_heads: Number of key-value heads for GQA (Grouped Query Attention)
        intermediate_size: Dimensionality of MLP hidden layer
        max_position_embeddings: Maximum sequence length
        rope_theta: Base frequency for RoPE (Rotary Position Embeddings)
        tie_word_embeddings: Whether to tie input and output embeddings
        rms_norm_eps: Epsilon for RMSNorm layer normalization
        hidden_activation: Activation function for MLP (silu for SwiGLU)
        attention_dropout: Dropout probability for attention weights
        use_cache: Whether to use KV cache for generation

    Examples:
        >>> # Default tiny config for testing
        >>> config = BaguettotronConfig()

        >>> # Official Baguettotron-321M config
        >>> config = BaguettotronConfig.baguettotron_321m()

        >>> # Custom config
        >>> config = BaguettotronConfig(
        ...     vocab_size=32000,
        ...     hidden_size=768,
        ...     num_hidden_layers=12
        ... )
    """

    # Architecture
    vocab_size: int = 512
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    intermediate_size: int = 256

    # Position embeddings
    max_position_embeddings: int = 64
    rope_theta: float = 10000.0

    # Normalization
    rms_norm_eps: float = 1e-6

    # Regularization
    attention_dropout: float = 0.0

    # Embeddings
    tie_word_embeddings: bool = False

    # Activation
    hidden_activation: str = "silu"

    # Generation
    use_cache: bool = True

    # Initialization
    initializer_range: float = 0.02

    # Special tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: Optional[int] = None

    # Model type (for compatibility)
    model_type: str = "llama"
    architectures: list = field(default_factory=lambda: ["LlamaForCausalLM"])

    def __post_init__(self):
        """Post-initialization setup (validation moved to model)."""
        # Validation is intentionally minimal here to allow invalid configs
        # for testing purposes. The model should validate when instantiated.
        pass

    @property
    def head_dim(self) -> int:
        """Compute dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def baguettotron_321m(cls) -> "BaguettotronConfig":
        """
        Create the official Baguettotron-321M configuration.

        This configuration matches the official model on HuggingFace:
        https://huggingface.co/PleIAs/Baguettotron/blob/main/config.json

        Returns:
            BaguettotronConfig: Official 321M parameter configuration

        Examples:
            >>> config = BaguettotronConfig.baguettotron_321m()
            >>> print(f"Parameters: ~{config.approximate_params() / 1e6:.1f}M")
            Parameters: ~321.0M
        """
        return cls(
            vocab_size=65536,
            hidden_size=576,
            num_hidden_layers=80,
            num_attention_heads=9,
            num_key_value_heads=3,
            intermediate_size=1536,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            rms_norm_eps=1e-5,
            hidden_activation="silu",
            attention_dropout=0.0,
            use_cache=True,
            initializer_range=0.02,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=None,
        )

    def approximate_params(self) -> int:
        """
        Compute approximate number of parameters.

        This is a rough estimation based on the main components:
        - Embeddings: vocab_size * hidden_size
        - Attention per layer: Uses GQA (Grouped Query Attention)
        - MLP per layer: 3 * hidden_size * intermediate_size (SwiGLU)
        - Norms: negligible

        Returns:
            int: Approximate number of parameters

        Note:
            Actual count may differ slightly due to tied embeddings and other factors.
        """
        # Embeddings (input only, output tied if enabled)
        embedding_params = self.vocab_size * self.hidden_size

        # Per-layer attention with GQA
        # Q: num_attention_heads * head_dim = hidden_size
        # K, V: num_key_value_heads * head_dim
        head_dim = self.hidden_size // self.num_attention_heads
        kv_size = self.num_key_value_heads * head_dim

        attn_params_per_layer = (
            self.hidden_size * self.hidden_size +  # q_proj
            self.hidden_size * kv_size +           # k_proj
            self.hidden_size * kv_size +           # v_proj
            self.hidden_size * self.hidden_size    # o_proj
        )

        # Per-layer MLP (SwiGLU: gate_proj, up_proj, down_proj)
        mlp_params_per_layer = (
            self.hidden_size * self.intermediate_size +  # gate_proj
            self.hidden_size * self.intermediate_size +  # up_proj
            self.intermediate_size * self.hidden_size    # down_proj
        )

        # Total layers
        layer_params = self.num_hidden_layers * (attn_params_per_layer + mlp_params_per_layer)

        # Total (embeddings counted once if tied)
        total = embedding_params + layer_params

        return total

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for HuggingFace compatibility."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "attention_dropout": self.attention_dropout,
            "tie_word_embeddings": self.tie_word_embeddings,
            "hidden_activation": self.hidden_activation,
            "use_cache": self.use_cache,
            "initializer_range": self.initializer_range,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "model_type": self.model_type,
            "architectures": self.architectures,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "BaguettotronConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            BaguettotronConfig: Configuration instance
        """
        # Filter only known parameters
        known_params = {
            k: v for k, v in config_dict.items()
            if k in cls.__dataclass_fields__
        }
        return cls(**known_params)
