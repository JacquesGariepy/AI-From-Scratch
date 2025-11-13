#!/usr/bin/env python3
"""
Download and prepare the PleIAs/SYNTH dataset from HuggingFace.

This script downloads the official SYNTH dataset used to train Baguettotron
and converts it to the format expected by our training pipeline.

Requirements:
    pip install datasets transformers

Usage:
    # Download and tokenize (recommended)
    python prepare_synth_data.py --tokenize

    # Download raw text only (JSONL format)
    python prepare_synth_data.py --format jsonl

    # Download subset for testing
    python prepare_synth_data.py --subset --max-samples 10000
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm


def download_and_prepare_synth(
    output_dir: str = 'data',
    split: str = 'train',
    tokenize: bool = True,
    tokenizer_name: str = 'PleIAs/Baguettotron',
    max_samples: int = None,
    block_size: int = 2048,
    format: str = 'json',
):
    """
    Download SYNTH dataset from HuggingFace and prepare for training.

    Args:
        output_dir: Directory to save prepared data
        split: Dataset split ('train', 'validation', 'test')
        tokenize: Whether to pre-tokenize the data
        tokenizer_name: HuggingFace tokenizer to use
        max_samples: Maximum number of samples to process (None = all)
        block_size: Maximum sequence length for tokenization
        format: Output format ('json' for tokenized, 'jsonl' for raw text)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: 'datasets' library not installed")
        print("   Install with: pip install datasets")
        return

    print("=" * 70)
    print("Downloading PleIAs/SYNTH dataset from HuggingFace")
    print("=" * 70)
    print(f"Split: {split}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\n📥 Downloading dataset...")
    try:
        # Use streaming for subsets to avoid downloading all shards
        if max_samples:
            print(f"   📊 Subset mode: streaming {max_samples:,} samples...")
            print(f"   ⚡ Only downloading necessary data (not all {split} shards)...")
            # Use streaming to avoid downloading everything
            dataset = load_dataset('PleIAs/SYNTH', split=split, streaming=True)
            is_streaming = True
            print(f"   ✓ Streaming mode activated (will process {max_samples:,} samples)")
        else:
            print("   📦 Full download mode (will download all shards)...")
            dataset = load_dataset('PleIAs/SYNTH', split=split)
            is_streaming = False
            print(f"   ✓ Total samples: {len(dataset):,}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\nℹ️  The SYNTH dataset might require authentication or have access restrictions.")
        print("   Try: huggingface-cli login")
        return

    if tokenize:
        # Option 1: Pre-tokenize (faster training)
        print("\n🔤 Loading tokenizer...")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except ImportError:
            print("❌ Error: 'transformers' library not installed")
            print("   Install with: pip install transformers")
            return
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            return

        print(f"   Tokenizer: {tokenizer_name}")
        print(f"\n⚙️  Tokenizing dataset (block_size={block_size})...")

        token_sequences = []
        count = 0

        # Use tqdm with total if we know the size
        pbar_total = max_samples if is_streaming and max_samples else (len(dataset) if not is_streaming else None)

        for example in tqdm(dataset, desc="Tokenizing", total=pbar_total):
            # Extract text from example
            text = example.get('text', example.get('content', ''))
            if not text:
                continue

            # Tokenize
            tokens = tokenizer.encode(text, max_length=block_size, truncation=True)
            if len(tokens) > 10:  # Skip very short sequences
                token_sequences.append(tokens)
                count += 1

            # Stop early if we're streaming and reached max_samples
            if is_streaming and max_samples and count >= max_samples:
                break

        # Save tokenized data
        output_file = output_dir / f'{split}_tokens.json'
        print(f"\n💾 Saving tokenized data to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(token_sequences, f)

        total_tokens = sum(len(seq) for seq in token_sequences)
        print(f"   ✓ Saved {len(token_sequences):,} sequences ({total_tokens:,} tokens)")

    else:
        # Option 2: Save as JSONL (tokenize during training)
        output_file = output_dir / f'{split}.jsonl'
        print(f"\n💾 Saving raw text to {output_file}...")

        count = 0
        pbar_total = max_samples if is_streaming and max_samples else (len(dataset) if not is_streaming else None)

        with open(output_file, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, desc="Saving", total=pbar_total):
                text = example.get('text', example.get('content', ''))
                if text:
                    json_line = json.dumps({'text': text}, ensure_ascii=False)
                    f.write(json_line + '\n')
                    count += 1

                # Stop early if streaming and reached max_samples
                if is_streaming and max_samples and count >= max_samples:
                    break

        print(f"   ✓ Saved {count:,} examples")

    print("\n" + "=" * 70)
    print("✅ Dataset preparation complete!")
    print("=" * 70)

    if tokenize:
        print(f"\nYou can now train with:")
        print(f"  python train.py --train-data {output_file}")
    else:
        print(f"\nYou can now train with SYNTHDataset:")
        print(f"  from baguettotron.data import SYNTHDataset")
        print(f"  dataset = SYNTHDataset('{output_file}', tokenizer, block_size=2048)")


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare PleIAs/SYNTH dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for prepared data'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'validation', 'test'],
        help='Dataset split to download'
    )
    parser.add_argument(
        '--tokenize',
        action='store_true',
        help='Pre-tokenize the data (recommended for faster training)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='PleIAs/Baguettotron',
        help='HuggingFace tokenizer to use'
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=2048,
        help='Maximum sequence length for tokenization'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='json',
        choices=['json', 'jsonl'],
        help='Output format (json=tokenized, jsonl=raw text)'
    )
    parser.add_argument(
        '--subset',
        action='store_true',
        help='Download only a subset for testing'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=10000,
        help='Maximum samples for subset (used with --subset)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.format == 'json' and not args.tokenize:
        print("⚠️  Warning: Using format='json' without --tokenize")
        print("   Setting --tokenize automatically")
        args.tokenize = True

    # Prepare dataset
    download_and_prepare_synth(
        output_dir=args.output_dir,
        split=args.split,
        tokenize=args.tokenize,
        tokenizer_name=args.tokenizer,
        max_samples=args.max_samples if args.subset else None,
        block_size=args.block_size,
        format=args.format,
    )


if __name__ == '__main__':
    main()
