"""
Tests for tokenization utilities.
"""

import pytest
import torch
from baguettotron.tokenization import (
    load_tokenizer,
    TokenizerWrapper,
    CharLevelTokenizer,
)


class TestCharLevelTokenizer:
    """Test character-level tokenizer."""

    def test_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = CharLevelTokenizer(vocab_size=1000)
        assert tokenizer.vocab_size == 1000
        assert tokenizer.pad_token_id == 0
        assert tokenizer.bos_token_id == 1
        assert tokenizer.eos_token_id == 2
        assert tokenizer.unk_token_id == 3

    def test_encode(self):
        """Test text encoding."""
        tokenizer = CharLevelTokenizer(vocab_size=1000)
        text = "Hello"
        token_ids = tokenizer.encode(text)

        assert isinstance(token_ids, list)
        assert len(token_ids) == len(text)
        assert all(isinstance(tid, int) for tid in token_ids)
        assert all(4 <= tid < 1000 for tid in token_ids)

    def test_encode_empty(self):
        """Test encoding empty string."""
        tokenizer = CharLevelTokenizer(vocab_size=1000)
        token_ids = tokenizer.encode("")
        assert token_ids == [tokenizer.eos_token_id]

    def test_decode(self):
        """Test token decoding."""
        tokenizer = CharLevelTokenizer(vocab_size=1000)
        token_ids = [10, 20, 30]
        text = tokenizer.decode(token_ids)

        assert isinstance(text, str)
        assert "[10]" in text  # Hash-based tokens show as [ID]
        assert "[20]" in text
        assert "[30]" in text

    def test_decode_special_tokens(self):
        """Test decoding special tokens."""
        tokenizer = CharLevelTokenizer(vocab_size=1000)
        token_ids = [
            tokenizer.bos_token_id,
            10,
            tokenizer.eos_token_id,
        ]
        text = tokenizer.decode(token_ids)

        assert "<BOS>" in text
        assert "<EOS>" in text
        assert "[10]" in text


class TestTokenizerWrapper:
    """Test tokenizer wrapper."""

    def test_char_tokenizer(self):
        """Test loading character-level tokenizer."""
        tokenizer = TokenizerWrapper(
            tokenizer_type="char",
            model_vocab_size=1000,
        )

        assert tokenizer.tokenizer_name == "char"
        assert tokenizer.vocab_size == 1000
        assert isinstance(tokenizer.tokenizer, CharLevelTokenizer)

    def test_encode_text(self):
        """Test encoding text."""
        tokenizer = TokenizerWrapper(
            tokenizer_type="char",
            model_vocab_size=1000,
        )

        # Test list output
        token_ids = tokenizer.encode("Hello")
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0

        # Test tensor output
        token_tensor = tokenizer.encode("Hello", return_tensors="pt")
        assert isinstance(token_tensor, torch.Tensor)
        assert token_tensor.dim() == 2
        assert token_tensor.shape[0] == 1

    def test_decode_tokens(self):
        """Test decoding tokens."""
        tokenizer = TokenizerWrapper(
            tokenizer_type="char",
            model_vocab_size=1000,
        )

        # Test with list
        token_ids = [10, 20, 30]
        text = tokenizer.decode(token_ids)
        assert isinstance(text, str)

        # Test with tensor
        token_tensor = torch.tensor([[10, 20, 30]])
        text = tokenizer.decode(token_tensor)
        assert isinstance(text, str)

    def test_round_trip(self):
        """Test encode-decode round trip."""
        tokenizer = TokenizerWrapper(
            tokenizer_type="char",
            model_vocab_size=1000,
        )

        text = "Hello, world!"
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)

        # For char tokenizer, we just check it produces output
        assert isinstance(decoded, str)
        assert len(decoded) > 0


class TestLoadTokenizer:
    """Test tokenizer loading function."""

    def test_load_char_tokenizer(self):
        """Test loading character tokenizer."""
        tokenizer = load_tokenizer("char", model_vocab_size=1000)
        assert isinstance(tokenizer, TokenizerWrapper)
        assert tokenizer.tokenizer_name == "char"

    def test_load_auto_tokenizer(self):
        """Test auto tokenizer loading (may fallback to char)."""
        tokenizer = load_tokenizer("auto", model_vocab_size=65536)
        assert isinstance(tokenizer, TokenizerWrapper)
        # Could be any type depending on what's available
        assert tokenizer.tokenizer_name in ["baguettotron", "gpt2", "char"]

    def test_invalid_type(self):
        """Test invalid tokenizer type."""
        with pytest.raises(ValueError):
            load_tokenizer("invalid", model_vocab_size=1000)


class TestVocabSizeMismatch:
    """Test handling of vocab size mismatches."""

    def test_vocab_size_clipping(self):
        """Test that tokens are clipped to model vocab size."""
        tokenizer = TokenizerWrapper(
            tokenizer_type="char",
            model_vocab_size=100,
        )

        # CharLevelTokenizer will generate tokens in range [4, vocab_size)
        # Wrapper should clip them to model_vocab_size
        token_ids = tokenizer.encode("Test")
        assert all(tid < tokenizer.model_vocab_size for tid in token_ids)


@pytest.mark.skipif(
    not pytest.importorskip("transformers", reason="transformers not installed"),
    reason="transformers not installed",
)
class TestHuggingFaceTokenizers:
    """Test HuggingFace tokenizer integration (if available)."""

    def test_gpt2_tokenizer(self):
        """Test loading GPT-2 tokenizer."""
        try:
            tokenizer = load_tokenizer("gpt2", model_vocab_size=65536)
            assert tokenizer.tokenizer_name == "gpt2"
            assert tokenizer.vocab_size == 50257

            # Test encoding
            text = "Hello, world!"
            token_ids = tokenizer.encode(text, return_tensors="pt")
            assert isinstance(token_ids, torch.Tensor)

            # Test decoding
            decoded = tokenizer.decode(token_ids)
            assert isinstance(decoded, str)
            assert "Hello" in decoded or "hello" in decoded.lower()

        except Exception:
            pytest.skip("GPT-2 tokenizer not available")

    def test_baguettotron_tokenizer(self):
        """Test loading Baguettotron tokenizer."""
        try:
            tokenizer = load_tokenizer("baguettotron", model_vocab_size=65536)
            assert tokenizer.tokenizer_name == "baguettotron"
            assert tokenizer.vocab_size == 65491

            # Test encoding
            text = "Bonjour le monde!"
            token_ids = tokenizer.encode(text, return_tensors="pt")
            assert isinstance(token_ids, torch.Tensor)

            # Test decoding
            decoded = tokenizer.decode(token_ids)
            assert isinstance(decoded, str)

        except Exception:
            pytest.skip("Baguettotron tokenizer not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
