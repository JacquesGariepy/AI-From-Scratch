#!/usr/bin/env python3
"""
Training script for Baguettotron models.

This script provides a command-line interface for training Baguettotron
language models from scratch or fine-tuning existing checkpoints.

Usage:
    python train.py --config configs/train_config.yaml
    python train.py --data data/train.json --epochs 10 --batch-size 32
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from baguettotron import BaguettotronForCausalLM, BaguettotronConfig
from baguettotron.data import TextDataset, create_dataloader, DataCollatorForLanguageModeling
from baguettotron.training import Trainer, create_optimizer, create_scheduler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('train.log'),
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Baguettotron language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML config file (overrides other arguments)',
    )

    # Data arguments
    parser.add_argument(
        '--train-data',
        type=str,
        default='data/train_tokens.json',
        help='Path to training data (JSON file with token sequences)',
    )
    parser.add_argument(
        '--eval-data',
        type=str,
        help='Path to evaluation data (optional)',
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=2048,
        help='Maximum sequence length',
    )

    # Model arguments
    parser.add_argument(
        '--model-config',
        type=str,
        choices=['tiny', '321m', 'custom'],
        default='321m',
        help='Model configuration preset',
    )
    parser.add_argument(
        '--model-config-file',
        type=str,
        help='Path to custom model config YAML (for --model-config custom)',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint to resume training from',
    )

    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size',
    )
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=1,
        help='Number of gradient accumulation steps',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate',
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.1,
        help='Weight decay',
    )
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=1.0,
        help='Maximum gradient norm for clipping',
    )

    # Scheduler arguments
    parser.add_argument(
        '--scheduler',
        type=str,
        default='cosine',
        choices=['linear', 'cosine', 'constant', 'polynomial', 'inverse_sqrt'],
        help='Learning rate scheduler type',
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=1000,
        help='Number of warmup steps',
    )

    # Hardware arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to train on',
    )
    parser.add_argument(
        '--mixed-precision',
        action='store_true',
        help='Use automatic mixed precision (AMP)',
    )
    parser.add_argument(
        '--compile',
        action='store_true',
        help='Use torch.compile for faster training (PyTorch 2.0+)',
    )

    # Logging and checkpointing
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory to save checkpoints and logs',
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='Log every N steps',
    )
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1000,
        help='Evaluate every N steps',
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1000,
        help='Save checkpoint every N steps',
    )

    # Other
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers',
    )

    return parser.parse_args()


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_config(args: argparse.Namespace) -> BaguettotronConfig:
    """Create model configuration from arguments."""
    if args.model_config == '321m':
        return BaguettotronConfig.baguettotron_321m()
    elif args.model_config == 'tiny':
        return BaguettotronConfig()  # Default tiny config
    elif args.model_config == 'custom':
        if not args.model_config_file:
            raise ValueError("--model-config-file required for custom config")
        config_dict = load_config_file(args.model_config_file)
        return BaguettotronConfig(**config_dict)
    else:
        raise ValueError(f"Unknown model config: {args.model_config}")


def main():
    """Main training function."""
    args = parse_args()

    # Load config from file if provided
    if args.config:
        config_dict = load_config_file(args.config)
        # Update args with config
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("=" * 80)
    logger.info("Baguettotron Training")
    logger.info("=" * 80)
    logger.info(f"Training data: {args.train_data}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")

    # Create model
    logger.info("Creating model...")
    model_config = create_model_config(args)
    model = BaguettotronForCausalLM(model_config)

    num_params = model.count_parameters()
    logger.info(f"Model parameters: {num_params / 1e6:.1f}M")

    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    # Create datasets
    logger.info("Loading training data...")
    train_dataset = TextDataset(
        args.train_data,
        block_size=args.block_size,
    )
    logger.info(f"Training examples: {len(train_dataset)}")

    eval_dataset = None
    if args.eval_data:
        logger.info("Loading evaluation data...")
        eval_dataset = TextDataset(
            args.eval_data,
            block_size=args.block_size,
        )
        logger.info(f"Evaluation examples: {len(eval_dataset)}")

    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = create_dataloader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = create_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Create scheduler
    num_training_steps = args.epochs * len(train_dataloader) // args.gradient_accumulation_steps
    logger.info(f"Total training steps: {num_training_steps}")

    scheduler = create_scheduler(
        optimizer,
        scheduler_type=args.scheduler,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        max_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        mixed_precision=args.mixed_precision,
        use_compile=args.compile,
    )

    # Load checkpoint state if resuming
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Train
    logger.info("Starting training...")
    logger.info("=" * 80)

    try:
        train_stats = trainer.train()

        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info(f"Total steps: {train_stats['total_steps']}")
        logger.info(f"Final loss: {train_stats['final_loss']:.4f}")
        logger.info(f"Training time: {train_stats['training_time']:.2f}s")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint('interrupted.pt')
        logger.info("Checkpoint saved to interrupted.pt")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
