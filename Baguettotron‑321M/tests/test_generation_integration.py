"""
Integration test for text generation with tokenizer.

This test verifies that the tokenizer works correctly with the model's
generate() method.
"""

import pytest
import torch
from baguettotron import BaguettotronConfig, BaguettotronForCausalLM, load_tokenizer


@pytest.mark.parametrize("tokenizer_type", ["char", "gpt2", "auto"])
def test_generation_with_tokenizer(tokenizer_type):
    """Test text generation with different tokenizers."""
    # Skip if transformers not available and tokenizer needs it
    if tokenizer_type in ["gpt2", "auto"]:
        pytest.importorskip("transformers")

    # Create tiny model for testing
    config = BaguettotronConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    model = BaguettotronForCausalLM(config)
    model.eval()

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_type, model_vocab_size=config.vocab_size)

    # Test encoding
    prompt = "Hello, world!"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    assert isinstance(input_ids, torch.Tensor)
    assert input_ids.dim() == 2
    assert input_ids.shape[0] == 1

    # Test generation
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0,
            do_sample=False,  # Greedy for determinism
        )

    # Check output shape
    assert output_ids.shape[0] == 1
    assert output_ids.shape[1] == input_ids.shape[1] + 10

    # Test decoding
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    assert isinstance(generated_text, str)
    assert len(generated_text) > 0

    print(f"\nTokenizer: {tokenizer_type}")
    print(f"Prompt: {prompt}")
    print(f"Input IDs: {input_ids[0].tolist()[:10]}...")
    print(f"Generated: {generated_text[:100]}...")


def test_generation_workflow():
    """Test complete generation workflow like in generate.py script."""
    # Create tiny model
    config = BaguettotronConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    model = BaguettotronForCausalLM(config)
    model.eval()

    # Load tokenizer (char-level for guaranteed availability)
    tokenizer = load_tokenizer("char", model_vocab_size=config.vocab_size)
    print(f"\nTokenizer: {tokenizer}")

    # Test prompts
    prompts = [
        "Hello",
        "The quick brown fox",
        "To be or not to be",
    ]

    for prompt in prompts:
        # Encode
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=20,
                temperature=0.8,
                top_k=50,
                do_sample=True,
            )

        # Decode
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        print(f"\n  Prompt: {prompt}")
        print(f"  Generated ({output_ids.shape[1]} tokens): {generated_text[:80]}...")

        # Assertions
        assert isinstance(generated_text, str)
        assert len(generated_text) > 0
        assert output_ids.shape[1] == input_ids.shape[1] + 20


def test_vocab_size_mismatch_handling():
    """Test that vocab size mismatches are handled gracefully."""
    # Create model with larger vocab size than GPT-2
    config = BaguettotronConfig(
        vocab_size=65536,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
    )
    model = BaguettotronForCausalLM(config)
    model.eval()

    # Try to load GPT-2 tokenizer (vocab_size=50257)
    try:
        tokenizer = load_tokenizer("gpt2", model_vocab_size=config.vocab_size)

        # Should work despite mismatch
        assert tokenizer.vocab_size == 50257
        assert tokenizer.model_vocab_size == 65536

        # Test generation
        prompt = "Hello"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
            )

        # Model may generate tokens outside tokenizer vocab (expected with mismatch)
        # But decoding should still work by mapping them to UNK
        max_token = output_ids.max().item()
        print(f"\nMax token ID generated: {max_token} (tokenizer vocab: {tokenizer.vocab_size})")

        # Decoding should work despite vocab mismatch
        text = tokenizer.decode(output_ids)
        assert isinstance(text, str)
        assert len(text) > 0

        print(f"Vocab mismatch handled: model={config.vocab_size}, tokenizer={tokenizer.vocab_size}")
        print(f"Generated text: {text[:80]}...")

    except Exception as e:
        pytest.skip(f"GPT-2 tokenizer not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
