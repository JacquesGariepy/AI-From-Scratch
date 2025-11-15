#!/usr/bin/env python3
"""
Split a dataset into train/eval splits.

Usage:
    python scripts/split_dataset.py \
        --input data/wikipedia_simple_tokens.json \
        --train-output data/wikipedia_train.json \
        --eval-output data/wikipedia_eval.json \
        --eval-ratio 0.05
"""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into train/eval")
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input dataset file (JSON with token sequences)',
    )
    parser.add_argument(
        '--train-output',
        type=str,
        required=True,
        help='Output file for training split',
    )
    parser.add_argument(
        '--eval-output',
        type=str,
        required=True,
        help='Output file for evaluation split',
    )
    parser.add_argument(
        '--eval-ratio',
        type=float,
        default=0.05,
        help='Ratio of data to use for evaluation (default: 0.05 = 5%)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)

    print(f"Loading dataset from {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)

    # Handle both list and dict formats
    if isinstance(data, dict) and 'examples' in data:
        examples = data['examples']
    elif isinstance(data, list):
        examples = data
    else:
        raise ValueError(f"Unknown data format: {type(data)}")

    print(f"Total examples: {len(examples)}")

    # Shuffle
    random.shuffle(examples)

    # Split
    eval_size = int(len(examples) * args.eval_ratio)
    train_size = len(examples) - eval_size

    train_examples = examples[:train_size]
    eval_examples = examples[train_size:]

    print(f"Train examples: {len(train_examples)}")
    print(f"Eval examples: {len(eval_examples)}")

    # Save train split
    print(f"Saving train split to {args.train_output}...")
    with open(args.train_output, 'w') as f:
        json.dump({'examples': train_examples}, f)

    # Save eval split
    print(f"Saving eval split to {args.eval_output}...")
    with open(args.eval_output, 'w') as f:
        json.dump({'examples': eval_examples}, f)

    # Calculate file sizes
    train_path = Path(args.train_output)
    eval_path = Path(args.eval_output)

    print("\n" + "="*60)
    print("Split complete!")
    print("="*60)
    print(f"Train: {train_path.stat().st_size / 1024**2:.1f} MB ({len(train_examples)} examples)")
    print(f"Eval:  {eval_path.stat().st_size / 1024**2:.1f} MB ({len(eval_examples)} examples)")
    print("\nUsage:")
    print(f"  python scripts/train.py \\")
    print(f"    --train-data {args.train_output} \\")
    print(f"    --eval-data {args.eval_output} \\")
    print(f"    --save-total-limit 5")


if __name__ == '__main__':
    main()
