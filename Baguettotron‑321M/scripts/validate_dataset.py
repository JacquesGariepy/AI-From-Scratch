#!/usr/bin/env python3
"""
Dataset validation script for Baguettotron.

This script validates dataset files to ensure they are properly formatted
and ready for training.

Usage:
    python scripts/validate_dataset.py data/train_tokens.json
    python scripts/validate_dataset.py data/wikipedia_simple.jsonl
    python scripts/validate_dataset.py data/*.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List


def validate_tokenized_json(file_path: Path) -> Dict[str, Any]:
    """
    Validate a tokenized JSON dataset file.

    Args:
        file_path: Path to JSON file containing token sequences

    Returns:
        Dictionary with validation results and statistics
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return {
            'valid': False,
            'error': f'Invalid JSON: {e}',
        }
    except Exception as e:
        return {
            'valid': False,
            'error': f'Error reading file: {e}',
        }

    # Validate structure
    if not isinstance(data, list):
        return {
            'valid': False,
            'error': 'Expected a list of token sequences',
        }

    if len(data) == 0:
        return {
            'valid': False,
            'error': 'Dataset is empty',
        }

    # Sample validation
    num_sequences = len(data)
    sequence_lengths = []
    invalid_sequences = []

    for i, seq in enumerate(data[:1000]):  # Check first 1000
        if not isinstance(seq, list):
            invalid_sequences.append(i)
            continue

        if not all(isinstance(token, int) for token in seq):
            invalid_sequences.append(i)
            continue

        sequence_lengths.append(len(seq))

    if invalid_sequences:
        return {
            'valid': False,
            'error': f'Invalid sequences found at indices: {invalid_sequences[:10]}',
        }

    # Calculate statistics
    total_tokens = sum(len(seq) for seq in data)
    avg_length = sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0
    min_length = min(sequence_lengths) if sequence_lengths else 0
    max_length = max(sequence_lengths) if sequence_lengths else 0

    # File size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    return {
        'valid': True,
        'format': 'tokenized_json',
        'num_sequences': num_sequences,
        'total_tokens': total_tokens,
        'avg_length': avg_length,
        'min_length': min_length,
        'max_length': max_length,
        'file_size_mb': file_size_mb,
    }


def validate_jsonl(file_path: Path) -> Dict[str, Any]:
    """
    Validate a JSONL dataset file.

    Args:
        file_path: Path to JSONL file

    Returns:
        Dictionary with validation results and statistics
    """
    num_lines = 0
    invalid_lines = []
    text_lengths = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                num_lines += 1
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        invalid_lines.append(i)
                        continue

                    if 'text' not in data:
                        invalid_lines.append(i)
                        continue

                    text = data['text']
                    if not isinstance(text, str):
                        invalid_lines.append(i)
                        continue

                    text_lengths.append(len(text))

                except json.JSONDecodeError:
                    invalid_lines.append(i)

                # Limit validation to first 1000 lines
                if i >= 1000:
                    break

    except Exception as e:
        return {
            'valid': False,
            'error': f'Error reading file: {e}',
        }

    if num_lines == 0:
        return {
            'valid': False,
            'error': 'Dataset is empty',
        }

    if invalid_lines:
        return {
            'valid': False,
            'error': f'Invalid lines found at: {invalid_lines[:10]}',
        }

    # Calculate statistics
    avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    min_text_length = min(text_lengths) if text_lengths else 0
    max_text_length = max(text_lengths) if text_lengths else 0

    # File size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    return {
        'valid': True,
        'format': 'jsonl',
        'num_examples': num_lines,
        'avg_text_length': avg_text_length,
        'min_text_length': min_text_length,
        'max_text_length': max_text_length,
        'file_size_mb': file_size_mb,
    }


def validate_dataset(file_path: str) -> Dict[str, Any]:
    """
    Validate a dataset file.

    Args:
        file_path: Path to dataset file

    Returns:
        Dictionary with validation results
    """
    path = Path(file_path)

    if not path.exists():
        return {
            'file': str(path),
            'valid': False,
            'error': 'File not found',
        }

    # Determine format based on extension and content
    if path.suffix == '.json':
        result = validate_tokenized_json(path)
    elif path.suffix == '.jsonl':
        result = validate_jsonl(path)
    else:
        # Try to determine format from content
        try:
            with open(path, 'r') as f:
                first_line = f.readline().strip()

            if first_line.startswith('['):
                result = validate_tokenized_json(path)
            elif first_line.startswith('{'):
                result = validate_jsonl(path)
            else:
                result = {
                    'valid': False,
                    'error': 'Unknown format (expected JSON or JSONL)',
                }
        except Exception as e:
            result = {
                'valid': False,
                'error': f'Error determining format: {e}',
            }

    result['file'] = str(path)
    return result


def print_validation_results(results: Dict[str, Any]):
    """Print formatted validation results."""
    print("\n" + "=" * 70)
    print(f"Dataset Validation: {results['file']}")
    print("=" * 70)

    if not results['valid']:
        print(f"❌ INVALID: {results.get('error', 'Unknown error')}")
        return

    print("✅ VALID")
    print(f"\nFormat: {results['format']}")
    print(f"File size: {results['file_size_mb']:.1f} MB")

    if results['format'] == 'tokenized_json':
        print(f"\nStatistics:")
        print(f"  Sequences:    {results['num_sequences']:,}")
        print(f"  Total tokens: {results['total_tokens']:,}")
        print(f"  Avg length:   {results['avg_length']:.1f} tokens")
        print(f"  Min length:   {results['min_length']} tokens")
        print(f"  Max length:   {results['max_length']} tokens")

        # Estimate training info
        print(f"\nTraining estimates:")
        batch_sizes = [8, 16, 32, 64]
        for batch_size in batch_sizes:
            steps_per_epoch = results['num_sequences'] // batch_size
            print(f"  Batch size {batch_size:2d}: ~{steps_per_epoch:,} steps/epoch")

    elif results['format'] == 'jsonl':
        print(f"\nStatistics:")
        print(f"  Examples:         {results['num_examples']:,}")
        print(f"  Avg text length:  {results['avg_text_length']:.1f} chars")
        print(f"  Min text length:  {results['min_text_length']} chars")
        print(f"  Max text length:  {results['max_text_length']} chars")

        print(f"\n⚠️  Note: JSONL format requires tokenization during training")
        print(f"   Consider pre-tokenizing for faster training:")
        print(f"   python prepare_*_data.py --tokenize")


def check_dataset_availability() -> List[Dict[str, Any]]:
    """
    Check for available datasets in the data directory.

    Returns:
        List of dataset information dictionaries
    """
    data_dir = Path('data')
    if not data_dir.exists():
        return []

    datasets = []
    for pattern in ['*.json', '*.jsonl']:
        for file_path in data_dir.glob(pattern):
            # Skip config files
            if 'config' in file_path.name.lower():
                continue

            size_mb = file_path.stat().st_size / (1024 * 1024)
            datasets.append({
                'name': file_path.name,
                'path': str(file_path),
                'size_mb': size_mb,
            })

    return sorted(datasets, key=lambda x: x['size_mb'], reverse=True)


def main():
    parser = argparse.ArgumentParser(
        description='Validate Baguettotron dataset files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        'files',
        nargs='*',
        help='Dataset files to validate (supports wildcards)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets in data directory'
    )

    args = parser.parse_args()

    # List mode
    if args.list or not args.files:
        datasets = check_dataset_availability()

        if not datasets:
            print("\n❌ No datasets found in 'data' directory")
            print("\nTo prepare a dataset, run:")
            print("  python scripts/setup_dataset.py")
            sys.exit(1)

        print("\n" + "=" * 70)
        print("Available Datasets")
        print("=" * 70)
        for i, ds in enumerate(datasets, 1):
            print(f"\n{i}. {ds['name']}")
            print(f"   Path: {ds['path']}")
            print(f"   Size: {ds['size_mb']:.1f} MB")

        print("\nTo validate a dataset, run:")
        print("  python scripts/validate_dataset.py data/<filename>")
        return

    # Validation mode
    all_valid = True
    for file_path in args.files:
        results = validate_dataset(file_path)
        print_validation_results(results)

        if not results['valid']:
            all_valid = False

    # Exit with error code if any validation failed
    if not all_valid:
        print("\n" + "=" * 70)
        print("❌ Some datasets failed validation")
        print("=" * 70)
        sys.exit(1)
    else:
        print("\n" + "=" * 70)
        print("✅ All datasets are valid")
        print("=" * 70)


if __name__ == '__main__':
    main()
