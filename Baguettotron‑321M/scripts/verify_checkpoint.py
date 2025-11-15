#!/usr/bin/env python3
"""
Verify checkpoint integrity - check for NaN/Inf in model weights.

Usage:
    python scripts/verify_checkpoint.py outputs/test_321m/ckpt_290000
"""

import argparse
import sys
from pathlib import Path
import torch


def verify_checkpoint(checkpoint_path: str) -> bool:
    """
    Verify that a checkpoint doesn't contain NaN or Inf values.

    Args:
        checkpoint_path: Path to checkpoint directory or .pt file

    Returns:
        True if checkpoint is healthy, False otherwise
    """
    path = Path(checkpoint_path)

    # Handle both directory and file paths
    if path.is_dir():
        model_path = path / 'model.pt'
    else:
        model_path = path

    if not model_path.exists():
        print(f"❌ Checkpoint not found: {model_path}")
        return False

    print(f"🔍 Verifying checkpoint: {model_path}")
    print()

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Check each parameter
        total_params = len(state_dict)
        nan_params = []
        inf_params = []

        for name, param in state_dict.items():
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)

        # Report results
        print(f"📊 Checkpoint Statistics:")
        print(f"  Total parameters: {total_params}")
        print(f"  NaN parameters: {len(nan_params)}")
        print(f"  Inf parameters: {len(inf_params)}")
        print()

        if nan_params:
            print("❌ NaN detected in parameters:")
            for name in nan_params[:10]:  # Show first 10
                print(f"  - {name}")
            if len(nan_params) > 10:
                print(f"  ... and {len(nan_params) - 10} more")
            print()

        if inf_params:
            print("❌ Inf detected in parameters:")
            for name in inf_params[:10]:  # Show first 10
                print(f"  - {name}")
            if len(inf_params) > 10:
                print(f"  ... and {len(inf_params) - 10} more")
            print()

        # Final verdict
        if nan_params or inf_params:
            print("🚨 CHECKPOINT CORRUPTED - DO NOT USE")
            print("   Use an older checkpoint instead.")
            return False
        else:
            print("✅ CHECKPOINT HEALTHY - Safe to resume training")
            return True

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify checkpoint integrity")
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to checkpoint directory or .pt file',
    )
    args = parser.parse_args()

    is_healthy = verify_checkpoint(args.checkpoint)
    sys.exit(0 if is_healthy else 1)


if __name__ == '__main__':
    main()
