#!/usr/bin/env python3
"""
Download and prepare Wikipedia dataset for Baguettotron training.

This script downloads a lightweight Wikipedia dataset that is significantly
smaller than SYNTH, making it suitable for environments with limited resources.

Wikipedia dataset sizes:
- 20220301.simple: ~200MB (Simple English Wikipedia)
- 20220301.fr: ~6GB (French Wikipedia subset)
- 20220301.en: ~20GB (Full English Wikipedia)

Requirements:
    pip install datasets transformers

Usage:
    # Quick start with Simple English Wikipedia
    python prepare_wikipedia_data.py --lang simple --max-samples 50000

    # French Wikipedia subset
    python prepare_wikipedia_data.py --lang fr --max-samples 100000

    # Full download (English, large)
    python prepare_wikipedia_data.py --lang en --tokenize
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from typing import Optional


def download_and_prepare_wikipedia(
    output_dir: str = 'data',
    lang: str = 'simple',
    date: str = '20220301',
    tokenize: bool = True,
    tokenizer_name: str = 'PleIAs/Baguettotron',
    max_samples: Optional[int] = None,
    block_size: int = 2048,
    format: str = 'json',
):
    """
    Download Wikipedia dataset and prepare for training.

    Args:
        output_dir: Directory to save prepared data
        lang: Wikipedia language ('simple', 'fr', 'en')
        date: Wikipedia dump date (YYYYMMDD format)
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

    # Map language codes to dataset sizes
    size_info = {
        'simple': '~200MB (Simple English)',
        'fr': '~6GB (French subset)',
        'en': '~20GB (Full English)',
    }

    print("=" * 70)
    print("Downloading Wikipedia dataset from HuggingFace")
    print("=" * 70)
    print(f"Language: {lang} - Estimated size: {size_info.get(lang, 'Unknown')}")
    print(f"Date: {date}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\n📥 Downloading Wikipedia dataset...")
    dataset_name = "wikimedia/wikipedia"
    config_name = f"{date}.{lang}"

    try:
        # Use streaming for subsets to avoid downloading everything
        if max_samples:
            print(f"   📊 Subset mode: streaming {max_samples:,} articles...")
            print(f"   ⚡ Only downloading necessary data...")
            dataset = load_dataset(
                dataset_name,
                config_name,
                split='train',
                streaming=True,
            )
            is_streaming = True
            print(f"   ✓ Streaming mode activated (will process {max_samples:,} articles)")
        else:
            print("   📦 Full download mode...")
            dataset = load_dataset(
                dataset_name,
                config_name,
                split='train',
            )
            is_streaming = False
            print(f"   ✓ Total articles: {len(dataset):,}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\n💡 Available Wikipedia versions:")
        print("   - simple: Simple English Wikipedia (smallest)")
        print("   - fr: French Wikipedia")
        print("   - en: English Wikipedia (largest)")
        print(f"\n💡 Trying with latest date (20231101) instead of {date}...")

        # Try with a more recent date
        try:
            config_name = f"20231101.{lang}"
            if max_samples:
                dataset = load_dataset(
                    dataset_name,
                    config_name,
                    split='train',
                    streaming=True,
                )
                is_streaming = True
            else:
                dataset = load_dataset(
                    dataset_name,
                    config_name,
                    split='train',
                )
                is_streaming = False
            print(f"   ✓ Successfully loaded with date 20231101")
        except Exception as e2:
            print(f"❌ Error with fallback: {e2}")
            print("\nℹ️  Try: python prepare_wikipedia_data.py --lang simple")
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
            # Extract text from Wikipedia article
            text = example.get('text', '')
            if not text or len(text.strip()) < 100:  # Skip very short articles
                continue

            # Tokenize
            tokens = tokenizer.encode(text, max_length=block_size, truncation=True)
            if len(tokens) > 50:  # Skip very short sequences
                token_sequences.append(tokens)
                count += 1

            # Stop early if we're streaming and reached max_samples
            if is_streaming and max_samples and count >= max_samples:
                break

        # Save tokenized data
        output_file = output_dir / f'wikipedia_{lang}_tokens.json'
        print(f"\n💾 Saving tokenized data to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(token_sequences, f)

        total_tokens = sum(len(seq) for seq in token_sequences)
        print(f"   ✓ Saved {len(token_sequences):,} sequences ({total_tokens:,} tokens)")

    else:
        # Option 2: Save as JSONL (tokenize during training)
        output_file = output_dir / f'wikipedia_{lang}.jsonl'
        print(f"\n💾 Saving raw text to {output_file}...")

        count = 0
        pbar_total = max_samples if is_streaming and max_samples else (len(dataset) if not is_streaming else None)

        with open(output_file, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, desc="Saving", total=pbar_total):
                text = example.get('text', '')
                if text and len(text.strip()) >= 100:
                    # Include article title for context
                    title = example.get('title', '')
                    json_line = json.dumps({
                        'text': text,
                        'title': title,
                        'url': example.get('url', '')
                    }, ensure_ascii=False)
                    f.write(json_line + '\n')
                    count += 1

                # Stop early if streaming and reached max_samples
                if is_streaming and max_samples and count >= max_samples:
                    break

        print(f"   ✓ Saved {count:,} articles")

    print("\n" + "=" * 70)
    print("✅ Wikipedia dataset preparation complete!")
    print("=" * 70)

    # Calculate dataset size
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"\n📊 Dataset file size: {size_mb:.1f} MB")

    if tokenize:
        print(f"\nYou can now train with:")
        print(f"  python train.py --train-data {output_file}")
    else:
        print(f"\nYou can now train with TextDataset:")
        print(f"  from baguettotron.data import TextDataset")
        print(f"  dataset = TextDataset('{output_file}', block_size=2048)")

    # Comparison with SYNTH
    print(f"\n📈 Dataset comparison:")
    print(f"  Wikipedia ({lang}): {size_mb:.1f}MB - {count:,} articles")
    print(f"  SYNTH: 500GB+ - billions of tokens")
    print(f"  Demo data: ~5MB - synthetic random data")


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare Wikipedia dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for prepared data'
    )
    parser.add_argument(
        '--lang',
        type=str,
        default='simple',
        choices=['simple', 'fr', 'en'],
        help='Wikipedia language version (simple=smallest, en=largest)'
    )
    parser.add_argument(
        '--date',
        type=str,
        default='20220301',
        help='Wikipedia dump date (YYYYMMDD)'
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
        '--max-samples',
        type=int,
        help='Maximum number of articles to process (default: all)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.format == 'json' and not args.tokenize:
        print("⚠️  Warning: Using format='json' without --tokenize")
        print("   Setting --tokenize automatically")
        args.tokenize = True

    # Prepare dataset
    download_and_prepare_wikipedia(
        output_dir=args.output_dir,
        lang=args.lang,
        date=args.date,
        tokenize=args.tokenize,
        tokenizer_name=args.tokenizer,
        max_samples=args.max_samples,
        block_size=args.block_size,
        format=args.format,
    )


if __name__ == '__main__':
    main()
