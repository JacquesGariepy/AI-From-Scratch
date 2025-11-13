#!/usr/bin/env python3
"""
Interactive dataset setup script for Baguettotron.

This script helps users select and download the appropriate dataset
for their training environment, with options for:
- SYNTH: Full dataset (500GB+) for production training
- Wikipedia: Medium dataset (200MB-20GB) for resource-constrained environments
- Demo: Small synthetic dataset for testing

Usage:
    python scripts/setup_dataset.py
    python scripts/setup_dataset.py --dataset wikipedia --lang simple
    python scripts/setup_dataset.py --dataset demo --num-samples 1000
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional


# Dataset registry with metadata
DATASETS = {
    'synth': {
        'name': 'SYNTH (PleIAs)',
        'description': 'Full production dataset used for official Baguettotron training',
        'size': '500GB+',
        'samples': 'Billions of tokens',
        'quality': 'Production',
        'script': 'prepare_synth_data.py',
        'recommended_for': 'Production training, high-end hardware',
        'requires': ['datasets', 'transformers'],
        'download_time': '4-8 hours (depends on connection)',
    },
    'wikipedia': {
        'name': 'Wikipedia',
        'description': 'Clean, high-quality text from Wikipedia articles',
        'size': '200MB-20GB (depends on language)',
        'samples': '50K-6M articles',
        'quality': 'High',
        'script': 'prepare_wikipedia_data.py',
        'recommended_for': 'Medium-sized training, resource-constrained environments',
        'requires': ['datasets', 'transformers'],
        'download_time': '5-60 minutes (depends on language and size)',
    },
    'demo': {
        'name': 'Demo (Synthetic)',
        'description': 'Random synthetic data for quick testing',
        'size': '5-50MB',
        'samples': '100-10,000 sequences',
        'quality': 'Testing only',
        'script': 'prepare_demo_data.py',
        'recommended_for': 'Quick testing, CI/CD, debugging',
        'requires': [],
        'download_time': 'Instant (generated locally)',
    },
}


def print_dataset_info(dataset_id: str, dataset: Dict[str, Any]):
    """Print formatted information about a dataset."""
    print(f"\n{'=' * 70}")
    print(f"📊 {dataset['name']}")
    print(f"{'=' * 70}")
    print(f"Description:     {dataset['description']}")
    print(f"Size:            {dataset['size']}")
    print(f"Samples:         {dataset['samples']}")
    print(f"Quality:         {dataset['quality']}")
    print(f"Recommended for: {dataset['recommended_for']}")
    print(f"Download time:   {dataset['download_time']}")

    if dataset['requires']:
        print(f"Requirements:    {', '.join(dataset['requires'])}")
    else:
        print(f"Requirements:    None (pure Python)")


def interactive_dataset_selection() -> tuple[str, Dict[str, Any]]:
    """
    Interactive prompt for dataset selection.

    Returns:
        Tuple of (dataset_id, dataset_config)
    """
    print("\n" + "=" * 70)
    print("Baguettotron Dataset Selection")
    print("=" * 70)
    print("\nAvailable datasets:\n")

    # Show dataset options
    for i, (dataset_id, dataset) in enumerate(DATASETS.items(), 1):
        print(f"{i}. {dataset['name']}")
        print(f"   Size: {dataset['size']} | {dataset['description']}")
        print()

    # Get user choice
    while True:
        try:
            choice = input("Select dataset [1-3] (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                print("Setup cancelled.")
                sys.exit(0)

            choice_num = int(choice)
            if 1 <= choice_num <= len(DATASETS):
                dataset_id = list(DATASETS.keys())[choice_num - 1]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(DATASETS)}")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

    dataset = DATASETS[dataset_id]
    print_dataset_info(dataset_id, dataset)

    # Get dataset-specific configuration
    config = {}

    if dataset_id == 'wikipedia':
        print("\n📝 Wikipedia Configuration:")
        print("Languages:")
        print("  1. simple - Simple English (smallest, ~200MB)")
        print("  2. fr     - French (~6GB)")
        print("  3. en     - English (largest, ~20GB)")

        while True:
            lang_choice = input("\nSelect language [1-3]: ").strip()
            lang_map = {'1': 'simple', '2': 'fr', '3': 'en'}
            if lang_choice in lang_map:
                config['lang'] = lang_map[lang_choice]
                break
            print("❌ Invalid choice")

        # Ask for sample limit
        limit = input("\nLimit articles? (press Enter for all, or enter number): ").strip()
        if limit:
            try:
                config['max_samples'] = int(limit)
            except ValueError:
                print("⚠️  Invalid number, downloading all articles")

    elif dataset_id == 'demo':
        print("\n📝 Demo Data Configuration:")
        samples = input("Number of sequences [default: 1000]: ").strip()
        if samples:
            try:
                config['num_samples'] = int(samples)
            except ValueError:
                config['num_samples'] = 1000
        else:
            config['num_samples'] = 1000

    elif dataset_id == 'synth':
        print("\n⚠️  WARNING: SYNTH dataset is very large (500GB+)")
        confirm = input("Download full dataset? (y/N): ").strip().lower()
        if confirm != 'y':
            print("\n💡 Using subset mode instead")
            samples = input("Number of samples [default: 10000]: ").strip()
            if samples:
                try:
                    config['max_samples'] = int(samples)
                except ValueError:
                    config['max_samples'] = 10000
            else:
                config['max_samples'] = 10000
            config['subset'] = True

    # Ask about tokenization
    print("\n📝 Tokenization:")
    print("Pre-tokenizing speeds up training but increases disk usage.")
    tokenize = input("Pre-tokenize data? (Y/n): ").strip().lower()
    config['tokenize'] = tokenize != 'n'

    return dataset_id, config


def run_dataset_preparation(dataset_id: str, config: Dict[str, Any]) -> bool:
    """
    Run the appropriate dataset preparation script.

    Args:
        dataset_id: Dataset identifier
        config: Configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    dataset = DATASETS[dataset_id]
    script_path = Path(__file__).parent.parent / dataset['script']

    if not script_path.exists():
        print(f"❌ Error: Script not found: {script_path}")
        return False

    # Build command
    cmd = [sys.executable, str(script_path)]

    # Add common arguments
    if config.get('tokenize', False):
        cmd.append('--tokenize')

    # Add dataset-specific arguments
    if dataset_id == 'wikipedia':
        if 'lang' in config:
            cmd.extend(['--lang', config['lang']])
        if 'max_samples' in config:
            cmd.extend(['--max-samples', str(config['max_samples'])])

    elif dataset_id == 'demo':
        if 'num_samples' in config:
            cmd.extend(['--num-samples', str(config['num_samples'])])

    elif dataset_id == 'synth':
        if config.get('subset', False):
            cmd.append('--subset')
            if 'max_samples' in config:
                cmd.extend(['--max-samples', str(config['max_samples'])])

    # Run preparation script
    print("\n" + "=" * 70)
    print(f"Running dataset preparation: {' '.join(cmd)}")
    print("=" * 70 + "\n")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Dataset preparation failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        return False


def save_dataset_config(dataset_id: str, config: Dict[str, Any], output_path: str = 'data/dataset_config.json'):
    """Save dataset configuration for future reference."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_data = {
        'dataset': dataset_id,
        'config': config,
        'metadata': DATASETS[dataset_id],
    }

    with open(output_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"\n💾 Configuration saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive dataset setup for Baguettotron',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['synth', 'wikipedia', 'demo'],
        help='Dataset to use (skips interactive selection)'
    )
    parser.add_argument(
        '--lang',
        type=str,
        choices=['simple', 'fr', 'en'],
        help='Wikipedia language (for wikipedia dataset)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to download'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        help='Number of samples for demo dataset'
    )
    parser.add_argument(
        '--tokenize',
        action='store_true',
        help='Pre-tokenize the data'
    )
    parser.add_argument(
        '--no-tokenize',
        action='store_true',
        help='Do not pre-tokenize the data'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets and exit'
    )

    args = parser.parse_args()

    # List datasets and exit
    if args.list:
        print("\n" + "=" * 70)
        print("Available Datasets")
        print("=" * 70)
        for dataset_id, dataset in DATASETS.items():
            print_dataset_info(dataset_id, dataset)
        return

    # Interactive or CLI mode
    if args.dataset:
        # CLI mode with explicit arguments
        dataset_id = args.dataset
        config = {}

        if args.lang:
            config['lang'] = args.lang
        if args.max_samples:
            config['max_samples'] = args.max_samples
        if args.num_samples:
            config['num_samples'] = args.num_samples
        if args.tokenize:
            config['tokenize'] = True
        elif args.no_tokenize:
            config['tokenize'] = False
        else:
            config['tokenize'] = True  # Default

        print_dataset_info(dataset_id, DATASETS[dataset_id])
    else:
        # Interactive mode
        dataset_id, config = interactive_dataset_selection()

    # Confirm and proceed
    print("\n" + "=" * 70)
    print("Ready to download dataset")
    print("=" * 70)
    print(f"Dataset: {DATASETS[dataset_id]['name']}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print()

    proceed = input("Proceed with download? (Y/n): ").strip().lower()
    if proceed == 'n':
        print("Setup cancelled.")
        return

    # Run preparation
    success = run_dataset_preparation(dataset_id, config)

    if success:
        # Save configuration
        save_dataset_config(dataset_id, config)

        print("\n" + "=" * 70)
        print("✅ Dataset setup complete!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Review the prepared data in the 'data' directory")
        print("  2. Train your model:")
        print("     python train.py --train-data data/<dataset_file>")
        print("\n💡 Tip: Use --model-config tiny for quick testing")
    else:
        print("\n❌ Dataset setup failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
