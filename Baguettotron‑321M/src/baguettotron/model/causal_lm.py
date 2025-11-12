"""
Baguettotron Causal Language Model.

This module implements the complete Baguettotron-321M model for causal
language modeling, compatible with LlamaForCausalLM architecture.
"""

import torch
import torch.nn as nn
from typing import Optional

from .transformer import TransformerDecoder
from .normalization import RMSNorm
from ..config import BaguettotronConfig


class BaguettotronForCausalLM(nn.Module):
    """
    Baguettotron model for causal language modeling.

    This model implements a decoder-only transformer for autoregressive
    language generation. It's compatible with the LLaMA architecture and
    matches the official Baguettotron-321M model on HuggingFace.

    Architecture:
        1. Token embeddings
        2. Stack of transformer decoder blocks
        3. Final RMSNorm
        4. Language modeling head (optionally tied with embeddings)

    Args:
        config: Model configuration

    Attributes:
        config: Model configuration
        embeddings: Token embedding layer
        decoder: Stack of transformer blocks
        norm: Final RMSNorm layer
        lm_head: Language modeling head for next-token prediction

    Shape:
        - Input: (batch_size, seq_len) with token IDs
        - Output: (batch_size, seq_len, vocab_size) with logits

    Examples:
        >>> from baguettotron.config import BaguettotronConfig
        >>> config = BaguettotronConfig.baguettotron_321m()
        >>> model = BaguettotronForCausalLM(config)
        >>> input_ids = torch.randint(0, config.vocab_size, (2, 128))
        >>> logits = model(input_ids)
        >>> assert logits.shape == (2, 128, config.vocab_size)

        >>> # Count parameters
        >>> num_params = sum(p.numel() for p in model.parameters())
        >>> print(f"Parameters: {num_params / 1e6:.1f}M")
    """

    def __init__(self, config: BaguettotronConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )

        # Transformer decoder blocks
        self.decoder = TransformerDecoder(config)

        # Final normalization
        self.norm = RMSNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Language modeling head
        self.lm_head = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
        )

        # Tie embeddings if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights following LLaMA initialization strategy.

        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            # Use normal initialization with std = 1/sqrt(hidden_size)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for causal language modeling.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            is_causal: Whether to use causal masking (default: True)

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)

        Examples:
            >>> config = BaguettotronConfig()
            >>> model = BaguettotronForCausalLM(config)
            >>> input_ids = torch.randint(0, config.vocab_size, (2, 10))
            >>> logits = model(input_ids)
            >>> assert logits.shape == (2, 10, config.vocab_size)

            >>> # Get next token predictions
            >>> next_token_logits = logits[:, -1, :]
            >>> next_tokens = torch.argmax(next_token_logits, dim=-1)
        """
        # Get embeddings
        hidden_states = self.embeddings(input_ids)

        # Pass through transformer decoder
        hidden_states = self.decoder(
            hidden_states,
            attention_mask=attention_mask,
            is_causal=is_causal,
        )

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Starting tokens of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Keep tokens with cumulative probability >= top_p
            do_sample: Whether to sample (True) or use greedy decoding (False)

        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens)

        Examples:
            >>> config = BaguettotronConfig()
            >>> model = BaguettotronForCausalLM(config)
            >>> prompt = torch.randint(0, config.vocab_size, (1, 5))
            >>> # Greedy decoding
            >>> output = model.generate(prompt, max_new_tokens=10)
            >>> assert output.shape == (1, 15)
            >>> # Sampling with temperature
            >>> output = model.generate(
            ...     prompt,
            ...     max_new_tokens=10,
            ...     temperature=0.8,
            ...     do_sample=True
            ... )
        """
        self.eval()
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for last position
                logits = self(generated)
                next_token_logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample or take argmax
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if max length reached
                if generated.shape[1] >= self.config.max_position_embeddings:
                    break

        return generated

    def count_parameters(self, trainable_only: bool = False) -> int:
        """
        Count number of parameters in the model.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Number of parameters

        Examples:
            >>> config = BaguettotronConfig.baguettotron_321m()
            >>> model = BaguettotronForCausalLM(config)
            >>> total = model.count_parameters()
            >>> print(f"Total parameters: {total / 1e6:.1f}M")
            Total parameters: 321.0M
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Return the number of parameters in the model.

        Args:
            non_embedding: If True, subtract embedding parameters

        Returns:
            Number of parameters

        Note:
            For tied embeddings, the lm_head parameters are not double-counted
            as they share the same tensor with embeddings.
        """
        n_params = self.count_parameters()
        if non_embedding:
            n_params -= self.embeddings.weight.numel()
        return n_params
