"""
Tokenization utilities for Baguettotron.

This module provides tokenizer loading with automatic fallback strategies:
1. Official Baguettotron tokenizer (PleIAs/Baguettotron) - vocab_size=65491
2. GPT-2 tokenizer - vocab_size=50257
3. Character-level fallback - configurable vocab_size

The tokenizer handles vocab size mismatches gracefully through padding/truncation
of the embedding layer.
"""

import logging
from typing import Optional, Union, List, Dict, Any
import torch
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TokenizerWrapper:
    """
    Unified tokenizer wrapper with fallback support.

    This class provides a consistent interface for different tokenizers
    and handles vocab size mismatches automatically.
    """

    def __init__(
        self,
        tokenizer_type: str = "auto",
        model_vocab_size: int = 65536,
        chat_template_path: Optional[str] = None,
    ):
        """
        Initialize tokenizer with automatic fallback.

        Args:
            tokenizer_type: Type of tokenizer to use:
                - "auto": Try Baguettotron, fallback to GPT-2, then char-level
                - "baguettotron": Official Baguettotron tokenizer
                - "gpt2": GPT-2 tokenizer
                - "char": Character-level tokenizer
            model_vocab_size: Expected vocabulary size of the model
            chat_template_path: Path to chat_template.json file (optional)
        """
        self.model_vocab_size = model_vocab_size
        self.tokenizer = None
        self.tokenizer_name = None
        self.vocab_size = None
        self.chat_template = None
        self.chat_template_config = None

        # Try to load tokenizer
        if tokenizer_type == "auto":
            self._try_load_auto()
        elif tokenizer_type == "baguettotron":
            self._load_baguettotron()
        elif tokenizer_type == "gpt2":
            self._load_gpt2()
        elif tokenizer_type == "char":
            self._load_char_tokenizer()
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

        # Load chat template if available
        self._load_chat_template(chat_template_path)

        # Log vocab size mismatch if any
        if self.vocab_size != model_vocab_size:
            logger.warning(
                f"Tokenizer vocab size ({self.vocab_size}) != model vocab size ({model_vocab_size}). "
                f"Using vocab size adaptation (embeddings will be padded/truncated)."
            )

    def _try_load_auto(self):
        """Try loading tokenizers in order of preference."""
        # 1. Try official Baguettotron tokenizer
        if self._load_baguettotron(silent=True):
            return

        # 2. Try GPT-2 tokenizer
        if self._load_gpt2(silent=True):
            return

        # 3. Fallback to character-level
        logger.warning(
            "transformers library not available. Using character-level tokenizer fallback. "
            "Install transformers for better results: pip install transformers"
        )
        self._load_char_tokenizer()

    def _load_baguettotron(self, silent: bool = False) -> bool:
        """Load official Baguettotron tokenizer."""
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                "PleIAs/Baguettotron",
                trust_remote_code=False,
            )
            self.tokenizer_name = "baguettotron"
            self.vocab_size = self.tokenizer.vocab_size

            if not silent:
                logger.info(f"Loaded Baguettotron tokenizer (vocab_size={self.vocab_size})")

            return True

        except Exception as e:
            if not silent:
                logger.warning(f"Failed to load Baguettotron tokenizer: {e}")
            return False

    def _load_gpt2(self, silent: bool = False) -> bool:
        """Load GPT-2 tokenizer as fallback."""
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer_name = "gpt2"
            self.vocab_size = self.tokenizer.vocab_size

            if not silent:
                logger.info(f"Loaded GPT-2 tokenizer (vocab_size={self.vocab_size})")

            return True

        except Exception as e:
            if not silent:
                logger.warning(f"Failed to load GPT-2 tokenizer: {e}")
            return False

    def _load_char_tokenizer(self):
        """Load simple character-level tokenizer."""
        self.tokenizer = CharLevelTokenizer(vocab_size=self.model_vocab_size)
        self.tokenizer_name = "char"
        self.vocab_size = self.model_vocab_size

        logger.info(f"Loaded character-level tokenizer (vocab_size={self.vocab_size})")

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], torch.Tensor]:
        """
        Encode text to token IDs.

        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to add special tokens (BOS, EOS)
            return_tensors: If "pt", return PyTorch tensor

        Returns:
            Token IDs as list or tensor
        """
        if isinstance(self.tokenizer, CharLevelTokenizer):
            token_ids = self.tokenizer.encode(text)
            if return_tensors == "pt":
                return torch.tensor([token_ids], dtype=torch.long)
            return token_ids
        else:
            # HuggingFace tokenizer
            result = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
            )

            # Clip tokens to model vocab size if needed
            if self.vocab_size != self.model_vocab_size:
                result = [min(token_id, self.model_vocab_size - 1) for token_id in result]

            if return_tensors == "pt":
                return torch.tensor([result], dtype=torch.long)
            return result

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode (can be list or tensor)
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text
        """
        # Convert tensor to list if needed
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() == 2:
                token_ids = token_ids[0]  # Take first batch item
            token_ids = token_ids.tolist()

        # Clip tokens to vocab size if needed
        # This handles cases where model generates IDs outside tokenizer vocab
        if self.vocab_size != self.model_vocab_size:
            # Map tokens that are outside tokenizer vocab to UNK token (if available)
            unk_token_id = getattr(self.tokenizer, 'unk_token_id', None)
            if unk_token_id is None:
                unk_token_id = 0  # Fallback to 0

            token_ids = [
                token_id if token_id < self.vocab_size else unk_token_id
                for token_id in token_ids
            ]

        if isinstance(self.tokenizer, CharLevelTokenizer):
            return self.tokenizer.decode(token_ids)
        else:
            # HuggingFace tokenizer
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
            )

    def _load_chat_template(self, chat_template_path: Optional[str] = None):
        """
        Load chat template from JSON file.

        Args:
            chat_template_path: Path to chat_template.json (optional)
        """
        # Find chat_template.json
        if chat_template_path is None:
            # Try default locations
            possible_paths = [
                Path(__file__).parent.parent.parent / "assets" / "chat_template.json",
                Path.cwd() / "assets" / "chat_template.json",
                Path.cwd() / "chat_template.json",
            ]

            for path in possible_paths:
                if path.exists():
                    chat_template_path = str(path)
                    break

        if chat_template_path and Path(chat_template_path).exists():
            try:
                with open(chat_template_path, 'r', encoding='utf-8') as f:
                    self.chat_template_config = json.load(f)
                    self.chat_template = self.chat_template_config.get('chat_template')
                    logger.info(f"Loaded chat template from {chat_template_path}")
            except Exception as e:
                logger.warning(f"Failed to load chat template: {e}")
        else:
            logger.debug("No chat template found, using simple text mode only")

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Union[str, List[int], torch.Tensor]:
        """
        Apply chat template to format conversation messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
                Example: [{"role": "user", "content": "Hello"}]
            add_generation_prompt: Whether to add prompt for model generation
            tokenize: Whether to tokenize the result (default: True)
            return_tensors: If "pt", return PyTorch tensor

        Returns:
            Formatted text (if tokenize=False) or token IDs

        Example:
            >>> tokenizer = TokenizerWrapper("auto")
            >>> messages = [
            ...     {"role": "user", "content": "What is AI?"}
            ... ]
            >>> text = tokenizer.apply_chat_template(messages, tokenize=False)
            >>> # Returns: "<|im_start|>user\nWhat is AI?<|im_end|>\n<|im_start|>assistant\n<think>\n"
        """
        if not self.chat_template:
            # Fallback: simple concatenation
            logger.warning("No chat template loaded, using simple message concatenation")
            formatted_text = self._format_messages_simple(messages, add_generation_prompt)
        else:
            # Use Jinja2 template
            formatted_text = self._render_chat_template(messages, add_generation_prompt)

        # Tokenize if requested
        if tokenize:
            return self.encode(formatted_text, return_tensors=return_tensors)

        return formatted_text

    def _format_messages_simple(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Simple fallback for formatting messages without template."""
        formatted = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            formatted += f"{role}: {content}\n"

        if add_generation_prompt:
            formatted += "assistant: "

        return formatted

    def _render_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Render chat template using Jinja2-style formatting.

        This is a simplified implementation of the ChatML format.
        """
        if not self.chat_template_config:
            return self._format_messages_simple(messages, add_generation_prompt)

        # Get special tokens
        bos_token = self.chat_template_config.get('bos_token', '<|im_start|>')
        eos_token = self.chat_template_config.get('eos_token', '<|im_end|>')

        # Format messages according to ChatML template
        formatted = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            formatted += f"{bos_token}{role}\n{content}{eos_token}\n"

        # Add generation prompt if requested
        if add_generation_prompt:
            formatted += f"{bos_token}assistant\n<think>\n"

        return formatted

    def __repr__(self) -> str:
        chat_status = "with chat template" if self.chat_template else "no chat template"
        return (
            f"TokenizerWrapper(type={self.tokenizer_name}, "
            f"vocab_size={self.vocab_size}, "
            f"model_vocab_size={self.model_vocab_size}, "
            f"{chat_status})"
        )


class CharLevelTokenizer:
    """
    Simple character-level tokenizer as fallback.

    This tokenizer uses hash-based encoding to map characters to token IDs
    within the specified vocabulary size.
    """

    def __init__(self, vocab_size: int = 65536):
        """
        Initialize character-level tokenizer.

        Args:
            vocab_size: Size of vocabulary
        """
        self.vocab_size = vocab_size

        # Reserve special token IDs
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    def encode(self, text: str) -> List[int]:
        """
        Encode text using character hashing.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        if not text:
            return [self.eos_token_id]

        # Hash each character to a token ID
        # Reserve first 4 IDs for special tokens
        token_ids = []
        for char in text:
            # Use hash to map character to vocab space
            char_hash = hash(char) % (self.vocab_size - 4)
            token_id = char_hash + 4  # Offset by special tokens
            token_ids.append(token_id)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.

        Note: Character-level decoding with hash function is lossy.
        We represent tokens as [TOKEN_ID] for visibility.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded representation
        """
        # Since hash-based encoding is not reversible, we show token IDs
        # In practice, a real char-level tokenizer would have a bijective mapping
        tokens = []
        for token_id in token_ids:
            if token_id == self.pad_token_id:
                continue
            elif token_id == self.bos_token_id:
                tokens.append("<BOS>")
            elif token_id == self.eos_token_id:
                tokens.append("<EOS>")
            elif token_id == self.unk_token_id:
                tokens.append("<UNK>")
            else:
                # For hash-based tokens, show ID
                tokens.append(f"[{token_id}]")

        return " ".join(tokens)


def load_tokenizer(
    tokenizer_type: str = "auto",
    model_vocab_size: int = 65536,
    chat_template_path: Optional[str] = None,
) -> TokenizerWrapper:
    """
    Load tokenizer with automatic fallback and optional chat template.

    Args:
        tokenizer_type: Type of tokenizer ("auto", "baguettotron", "gpt2", "char")
        model_vocab_size: Expected vocabulary size of the model
        chat_template_path: Path to chat_template.json file (optional)

    Returns:
        TokenizerWrapper instance

    Examples:
        >>> # Auto-detect best available tokenizer
        >>> tokenizer = load_tokenizer("auto", model_vocab_size=65536)
        >>>
        >>> # With chat template
        >>> tokenizer = load_tokenizer("auto", chat_template_path="assets/chat_template.json")
        >>>
        >>> # Format conversation
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> text = tokenizer.apply_chat_template(messages, tokenize=False)
        >>>
        >>> # Encode and decode
        >>> token_ids = tokenizer.encode("Hello, world!", return_tensors="pt")
        >>> text = tokenizer.decode(token_ids)
    """
    return TokenizerWrapper(
        tokenizer_type=tokenizer_type,
        model_vocab_size=model_vocab_size,
        chat_template_path=chat_template_path,
    )
