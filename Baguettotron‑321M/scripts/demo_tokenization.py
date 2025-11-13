#!/usr/bin/env python3
"""
Demo script showing tokenization and text generation.

This script demonstrates:
1. Loading different tokenizer types
2. Encoding and decoding text
3. Handling vocab size mismatches
4. Text generation with real tokenizers
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baguettotron import BaguettotronConfig, BaguettotronForCausalLM, load_tokenizer


def demo_tokenizer_types():
    """Demonstrate different tokenizer types."""
    print("=" * 80)
    print("DEMO 1: Tokenizer Types")
    print("=" * 80)

    test_text = "Hello, world! This is a test."

    for tokenizer_type in ["char", "gpt2", "auto"]:
        print(f"\n{tokenizer_type.upper()} Tokenizer:")
        print("-" * 40)

        try:
            tokenizer = load_tokenizer(tokenizer_type, model_vocab_size=65536)
            print(f"  Type: {tokenizer.tokenizer_name}")
            print(f"  Vocab size: {tokenizer.vocab_size}")
            print(f"  Model vocab size: {tokenizer.model_vocab_size}")

            # Encode
            token_ids = tokenizer.encode(test_text)
            print(f"  Encoded ({len(token_ids)} tokens): {token_ids[:10]}...")

            # Decode
            decoded = tokenizer.decode(token_ids)
            print(f"  Decoded: {decoded[:60]}...")

        except Exception as e:
            print(f"  Error: {e}")


def demo_vocab_mismatch():
    """Demonstrate vocab size mismatch handling."""
    print("\n\n" + "=" * 80)
    print("DEMO 2: Vocab Size Mismatch Handling")
    print("=" * 80)

    # Create model with large vocab
    print("\nModel vocab size: 65536")

    try:
        # Try GPT-2 tokenizer (50257 tokens)
        tokenizer = load_tokenizer("gpt2", model_vocab_size=65536)
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        print(f"Mismatch: {65536 - tokenizer.vocab_size} tokens")

        # Test encoding/decoding
        text = "The tokenizer handles vocab size mismatches gracefully."
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)

        print(f"\nOriginal: {text}")
        print(f"Decoded:  {decoded}")
        print("\nVocab mismatch handled successfully!")

    except Exception as e:
        print(f"Skipped (transformers not available): {e}")


def demo_text_generation():
    """Demonstrate text generation with tokenizer."""
    print("\n\n" + "=" * 80)
    print("DEMO 3: Text Generation")
    print("=" * 80)

    # Create tiny model for demo
    print("\nCreating tiny model for demo...")
    config = BaguettotronConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=256,
    )
    model = BaguettotronForCausalLM(config)
    model.eval()
    print(f"Model created: {model.count_parameters() / 1e6:.2f}M parameters")

    # Load tokenizer
    tokenizer = load_tokenizer("char", model_vocab_size=config.vocab_size)
    print(f"Tokenizer: {tokenizer}")

    # Test prompts
    prompts = [
        "Once upon a time",
        "The meaning of life",
        "Hello, world",
    ]

    print("\nGenerating text:")
    print("-" * 80)

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
        generated = tokenizer.decode(output_ids, skip_special_tokens=True)

        print(f"\nPrompt: {prompt}")
        print(f"Tokens: {input_ids.shape[1]} -> {output_ids.shape[1]}")
        print(f"Output: {generated[:100]}...")


def demo_official_tokenizer():
    """Demonstrate official Baguettotron tokenizer."""
    print("\n\n" + "=" * 80)
    print("DEMO 4: Official Baguettotron Tokenizer")
    print("=" * 80)

    try:
        # Load official tokenizer
        tokenizer = load_tokenizer("baguettotron", model_vocab_size=65536)
        print(f"\nTokenizer: {tokenizer}")

        # Test with French and English
        test_texts = [
            "Bonjour le monde!",
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
        ]

        print("\nTokenization examples:")
        print("-" * 80)

        for text in test_texts:
            token_ids = tokenizer.encode(text)
            decoded = tokenizer.decode(token_ids)

            print(f"\nOriginal:  {text}")
            print(f"Tokens:    {len(token_ids)} tokens")
            print(f"Token IDs: {token_ids[:15]}...")
            print(f"Decoded:   {decoded}")

    except Exception as e:
        print(f"\nSkipped (official tokenizer not available): {e}")
        print("To use official tokenizer: pip install transformers")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Baguettotron Tokenization Demo" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")

    demo_tokenizer_types()
    demo_vocab_mismatch()
    demo_text_generation()
    demo_official_tokenizer()

    print("\n\n" + "=" * 80)
    print("All demos completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Try the generation script: python scripts/generate.py --help")
    print("2. Read the docs: docs/TOKENIZATION_GUIDE.md")
    print("3. Explore tokenizer types: auto, baguettotron, gpt2, char")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
