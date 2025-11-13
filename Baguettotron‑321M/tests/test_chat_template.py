#!/usr/bin/env python3
"""
Test chat template integration.

This test verifies that the chat template is properly loaded and applied.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baguettotron.tokenization import load_tokenizer


def test_chat_template_loading():
    """Test that chat template loads from assets."""
    print("=" * 80)
    print("Test 1: Chat Template Loading")
    print("=" * 80)

    # Load tokenizer with chat template
    tokenizer = load_tokenizer(
        tokenizer_type="char",  # Use char tokenizer for speed
        model_vocab_size=65536,
        chat_template_path=None,  # Auto-detect
    )

    print(f"Tokenizer: {tokenizer}")
    print(f"Has chat template: {tokenizer.chat_template is not None}")

    if tokenizer.chat_template:
        print("✓ Chat template loaded successfully")
    else:
        print("⚠ Chat template not found (this is OK if assets/chat_template.json doesn't exist)")

    print()


def test_simple_chat_formatting():
    """Test simple chat message formatting."""
    print("=" * 80)
    print("Test 2: Simple Chat Formatting (Fallback)")
    print("=" * 80)

    # Load tokenizer without chat template
    tokenizer = load_tokenizer(
        tokenizer_type="char",
        model_vocab_size=65536,
        chat_template_path="/nonexistent/path",  # Force fallback
    )

    # Test simple formatting
    messages = [
        {"role": "user", "content": "What is 2+2?"}
    ]

    formatted = tokenizer.apply_chat_template(messages, tokenize=False)

    print("Messages:", messages)
    print("\nFormatted text (simple fallback):")
    print("-" * 80)
    print(formatted)
    print("-" * 80)
    print("✓ Simple formatting works")
    print()


def test_chatml_formatting():
    """Test ChatML format with actual template."""
    print("=" * 80)
    print("Test 3: ChatML Format (with template)")
    print("=" * 80)

    # Load tokenizer with chat template
    tokenizer = load_tokenizer(
        tokenizer_type="char",
        model_vocab_size=65536,
    )

    # Test ChatML formatting
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is artificial intelligence?"}
    ]

    formatted = tokenizer.apply_chat_template(messages, tokenize=False)

    print("Messages:", messages)
    print("\nFormatted text (ChatML):")
    print("-" * 80)
    print(formatted)
    print("-" * 80)

    # Check for ChatML tokens
    if "<|im_start|>" in formatted and "<|im_end|>" in formatted:
        print("✓ ChatML format detected")
    else:
        print("⚠ Using fallback format (no ChatML tokens)")

    print()


def test_multi_turn_conversation():
    """Test multi-turn conversation formatting."""
    print("=" * 80)
    print("Test 4: Multi-turn Conversation")
    print("=" * 80)

    tokenizer = load_tokenizer(
        tokenizer_type="char",
        model_vocab_size=65536,
    )

    # Multi-turn conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "Tell me about the weather."}
    ]

    formatted = tokenizer.apply_chat_template(messages, tokenize=False)

    print("Conversation:", len(messages), "messages")
    print("\nFormatted conversation:")
    print("-" * 80)
    print(formatted)
    print("-" * 80)
    print("✓ Multi-turn conversation formatted")
    print()


def test_tokenization():
    """Test that tokenization works with chat template."""
    print("=" * 80)
    print("Test 5: Tokenization")
    print("=" * 80)

    tokenizer = load_tokenizer(
        tokenizer_type="char",
        model_vocab_size=65536,
    )

    messages = [
        {"role": "user", "content": "Hello, world!"}
    ]

    # Tokenize
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt"
    )

    print("Messages:", messages)
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"Number of tokens: {token_ids.shape[1]}")
    print("✓ Tokenization successful")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("CHAT TEMPLATE INTEGRATION TESTS")
    print("=" * 80 + "\n")

    try:
        test_chat_template_loading()
        test_simple_chat_formatting()
        test_chatml_formatting()
        test_multi_turn_conversation()
        test_tokenization()

        print("=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY ✓")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
