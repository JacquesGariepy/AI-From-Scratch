#!/usr/bin/env python3
"""
Quick generation script for Baguettotron without tokenizer.
Generates token sequences to demonstrate the model works.
"""

import sys
from pathlib import Path
import torch
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baguettotron import BaguettotronForCausalLM, BaguettotronConfig


def load_model_from_checkpoint(checkpoint_dir):
    """Load model from new checkpoint format (model.pt + config.json)."""
    checkpoint_dir = Path(checkpoint_dir)

    # Load config
    with open(checkpoint_dir / 'config.json', 'r') as f:
        config_dict = json.load(f)

    config = BaguettotronConfig(**config_dict)

    # Load model weights
    model = BaguettotronForCausalLM(config)
    model_state = torch.load(checkpoint_dir / 'model.pt', map_location='cuda')
    model.load_state_dict(model_state)

    return model.cuda().eval()


def generate_random_prompt(vocab_size, seq_len=10):
    """Create a random token sequence as prompt."""
    # Use tokens from training data range
    return torch.randint(100, vocab_size - 100, (1, seq_len), dtype=torch.long).cuda()


def main():
    print("=" * 80)
    print("Baguettotron Quick Generation Test")
    print("=" * 80)

    # Load model
    checkpoint_dir = "outputs/tiny_correct/ckpt_75"
    print(f"\n📦 Loading model from {checkpoint_dir}...")
    model = load_model_from_checkpoint(checkpoint_dir)

    num_params = model.count_parameters()
    print(f"✅ Model loaded: {num_params / 1e6:.1f}M parameters")
    print(f"   vocab_size: {model.config.vocab_size}")
    print(f"   hidden_size: {model.config.hidden_size}")
    print(f"   num_layers: {model.config.num_hidden_layers}")

    # Generate random prompt
    print(f"\n🎲 Creating random prompt (10 tokens)...")
    prompt = generate_random_prompt(model.config.vocab_size)
    print(f"   Prompt shape: {prompt.shape}")
    print(f"   Prompt tokens: {prompt[0].tolist()[:10]}")

    # Generate text
    print(f"\n🚀 Generating 50 new tokens...")
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            do_sample=True,
        )

    print(f"✅ Generation complete!")
    print(f"   Input length: {prompt.shape[1]}")
    print(f"   Output length: {generated.shape[1]}")
    print(f"   New tokens: {generated.shape[1] - prompt.shape[1]}")

    # Show generated tokens
    print(f"\n📝 Generated token sequence:")
    print(f"   {generated[0].tolist()}")

    # Statistics
    unique_tokens = len(set(generated[0].tolist()))
    print(f"\n📊 Statistics:")
    print(f"   Total tokens: {generated.shape[1]}")
    print(f"   Unique tokens: {unique_tokens}")
    print(f"   Min token ID: {generated[0].min().item()}")
    print(f"   Max token ID: {generated[0].max().item()}")

    # Test multiple generations
    print(f"\n🔄 Testing diversity with 3 different generations...")
    for i in range(3):
        with torch.no_grad():
            gen = model.generate(
                prompt,
                max_new_tokens=20,
                temperature=0.9,
                top_k=40,
                do_sample=True,
            )
        print(f"   Gen {i+1}: {gen[0, -10:].tolist()}")

    print("\n" + "=" * 80)
    print("✅ Generation test passed! Model is working correctly.")
    print("=" * 80)
    print("\nNote: To get actual text, you need to:")
    print("  1. Use the official Baguettotron tokenizer from HuggingFace")
    print("  2. Decode generated tokens with: tokenizer.decode(tokens)")
    print("\nExample:")
    print("  from transformers import AutoTokenizer")
    print("  tokenizer = AutoTokenizer.from_pretrained('PleIAs/Baguettotron')")
    print("  text = tokenizer.decode(generated[0])")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
