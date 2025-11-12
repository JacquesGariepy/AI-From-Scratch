#!/usr/bin/env python3
"""
Create demo training data for quick testing without downloading SYNTH.

This script generates synthetic tokenized data for testing the training pipeline.
For real training, use prepare_synth_data.py to download the official SYNTH dataset.

Usage:
    python prepare_demo_data.py
    python prepare_demo_data.py --num-train 5000 --num-eval 500
"""

import argparse
import json
import random
from pathlib import Path


def create_demo_sequences(
    num_sequences: int,
    vocab_size: int = 65536,
    min_length: int = 100,
    max_length: int = 2048,
) -> list:
    """
    Create random token sequences for demo purposes.

    Args:
        num_sequences: Number of sequences to generate
        vocab_size: Size of vocabulary
        min_length: Minimum sequence length
        max_length: Maximum sequence length

    Returns:
        List of token ID sequences
    """
    sequences = []
    for _ in range(num_sequences):
        length = random.randint(min_length, max_length)
        sequence = [random.randint(1, vocab_size - 1) for _ in range(length)]
        sequences.append(sequence)
    return sequences


def main():
    parser = argparse.ArgumentParser(
        description='Create demo training data'
    )
    parser.add_argument(
        '--num-train',
        type=int,
        default=1000,
        help='Number of training sequences'
    )
    parser.add_argument(
        '--num-eval',
        type=int,
        default=100,
        help='Number of evaluation sequences'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=65536,
        help='Vocabulary size (should match model)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating demo training data...")
    print(f"  - Training sequences: {args.num_train}")
    print(f"  - Evaluation sequences: {args.num_eval}")
    print(f"  - Vocabulary size: {args.vocab_size}")
    print(f"  - Max length: {args.max_length}")

    # Generate training data
    print("\nGenerating training data...")
    train_sequences = create_demo_sequences(
        num_sequences=args.num_train,
        vocab_size=args.vocab_size,
        max_length=args.max_length,
    )

    train_file = output_dir / 'train.json'
    with open(train_file, 'w') as f:
        json.dump(train_sequences, f)
    print(f"  ✓ Saved to {train_file}")

    # Generate evaluation data
    print("\nGenerating evaluation data...")
    eval_sequences = create_demo_sequences(
        num_sequences=args.num_eval,
        vocab_size=args.vocab_size,
        max_length=args.max_length,
    )

    eval_file = output_dir / 'eval.json'
    with open(eval_file, 'w') as f:
        json.dump(eval_sequences, f)
    print(f"  ✓ Saved to {eval_file}")

    # Calculate statistics
    train_tokens = sum(len(seq) for seq in train_sequences)
    eval_tokens = sum(len(seq) for seq in eval_sequences)

    print("\n" + "=" * 60)
    print("Demo data created successfully!")
    print("=" * 60)
    print(f"Training:   {args.num_train:,} sequences, {train_tokens:,} tokens")
    print(f"Evaluation: {args.num_eval:,} sequences, {eval_tokens:,} tokens")
    print(f"\nYou can now train with:")
    print(f"  python train.py --train-data {train_file} --eval-data {eval_file}")
    print("\n⚠️  NOTE: This is synthetic random data for testing only!")
    print("   For real training, use prepare_synth_data.py to download SYNTH dataset.")


if __name__ == '__main__':
    main()
