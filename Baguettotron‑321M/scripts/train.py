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
import os
import yaml
import torch
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from baguettotron import BaguettotronForCausalLM, BaguettotronConfig
from baguettotron.data import TextDataset, create_dataloader, DataCollatorForLanguageModeling
from baguettotron.training import Trainer, create_optimizer, create_scheduler
from baguettotron.tokenization import load_tokenizer

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
        '--datasets',
        type=str,
        help='Comma-separated list of datasets (synth,wikipedia,demo) or "auto" for auto-detect',
    )
    parser.add_argument(
        '--dataset-weights',
        type=str,
        help='Comma-separated weights for each dataset (e.g., "0.7,0.3")',
    )
    parser.add_argument(
        '--eval-data',
        type=str,
        help='Path to evaluation data (optional)',
    )
    parser.add_argument(
        '--block-size',
        '--max-length',
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
        '--save-steps',
        type=int,
        default=1000,
        help='Save checkpoint every N steps',
    )
    parser.add_argument(
        '--save-total-limit',
        type=int,
        default=None,
        help='Maximum number of checkpoints to keep (default: keep all)',
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


def create_model_config(args: argparse.Namespace, yaml_config: Optional[Dict] = None) -> BaguettotronConfig:
    """Create model configuration from arguments or YAML config."""
    # Priority: YAML 'model' section > --model-config-file > presets
    if yaml_config and 'model' in yaml_config:
        # Use model config from main YAML file
        model_dict = yaml_config['model']

        # Check if using a preset config (e.g., "321m")
        if 'config' in model_dict and isinstance(model_dict['config'], str):
            preset_name = model_dict['config']
            logger.info(f"Using preset model config: {preset_name}")

            # Load the preset
            if preset_name == '321m':
                base_config = BaguettotronConfig.baguettotron_321m()
            elif preset_name == 'tiny':
                base_config = BaguettotronConfig()
            else:
                raise ValueError(f"Unknown preset config: {preset_name}")

            # Apply any overrides from YAML
            overrides = {k: v for k, v in model_dict.items() if k != 'config'}
            if overrides:
                logger.info(f"Applying config overrides: {list(overrides.keys())}")
                # Update base config with overrides
                for key, value in overrides.items():
                    if hasattr(base_config, key):
                        setattr(base_config, key, value)
                    else:
                        logger.warning(f"Unknown config parameter: {key}")

            return base_config
        else:
            # No preset, use all parameters from YAML
            logger.info(f"Using custom model config from YAML: vocab_size={model_dict.get('vocab_size', 'N/A')}")
            return BaguettotronConfig.from_dict(model_dict)
    elif args.model_config == '321m':
        return BaguettotronConfig.baguettotron_321m()
    elif args.model_config == 'tiny':
        return BaguettotronConfig()  # Default tiny config
    elif args.model_config == 'custom':
        if not args.model_config_file:
            raise ValueError("--model-config-file required for custom config")
        config_dict = load_config_file(args.model_config_file)
        return BaguettotronConfig.from_dict(config_dict)
    else:
        raise ValueError(f"Unknown model config: {args.model_config}")


def main():
    """Main training function."""
    args = parse_args()

    # Load config from file if provided
    yaml_config = None
    if args.config:
        yaml_config = load_config_file(args.config)
        config_dict = yaml_config
        # Update args with config - support hierarchical structure
        # Map YAML structure to args attributes
        config_mapping = {
            # Toplevel
            'seed': 'seed',
            'device': 'device',
            # Model config
            'model.config': 'model_config',
            # Train config (support multiple naming conventions)
            'training.epochs': 'epochs',
            'training.max_epochs': 'epochs',
            'train.epochs': 'epochs',
            'training.batch_size': 'batch_size',
            'train.batch_size': 'batch_size',
            'training.gradient_accumulation_steps': 'gradient_accumulation_steps',
            'train.accum_steps': 'gradient_accumulation_steps',
            'training.learning_rate': 'learning_rate',
            'train.lr': 'learning_rate',
            'training.weight_decay': 'weight_decay',
            'train.weight_decay': 'weight_decay',
            'training.warmup_steps': 'warmup_steps',
            'train.warmup_steps': 'warmup_steps',
            'training.max_steps': 'max_steps',
            'train.max_steps': 'max_steps',
            'training.max_grad_norm': 'max_grad_norm',
            'train.grad_clip': 'max_grad_norm',
            'training.scheduler': 'scheduler',
            'train.scheduler': 'scheduler',
            'training.mixed_precision': 'mixed_precision',
            'train.amp': 'mixed_precision',
            'training.save_interval': 'save_interval',
            'train.save_steps': 'save_interval',
            'training.save_total_limit': 'save_total_limit',
            'train.save_total_limit': 'save_total_limit',
            'train.keep_last_checkpoints': 'save_total_limit',  # Alias
            'training.eval_interval': 'eval_interval',
            'train.eval_every': 'eval_interval',
            'training.log_interval': 'log_interval',
            'train.log_every': 'log_interval',
            'training.output_dir': 'output_dir',
            'train.output_dir': 'output_dir',
            'training.block_size': 'block_size',
            'train.block_size': 'block_size',
            'training.use_compile': 'compile',
            'train.use_compile': 'compile',
            # Data config (support multiple naming conventions)
            'data.train_data': 'train_data',
            'dataset.train_data': 'train_data',
            'data.datasets': 'datasets',
            'dataset.datasets': 'datasets',
            'data.dataset_weights': 'dataset_weights',
            'dataset.dataset_weights': 'dataset_weights',
        }

        # Flatten hierarchical config
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_config = flatten_dict(config_dict)

        # Apply config to args
        for config_key, arg_name in config_mapping.items():
            if config_key in flat_config and hasattr(args, arg_name):
                value = flat_config[config_key]
                # Don't override CLI arguments if they differ from defaults
                setattr(args, arg_name, value)

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
    model_config = create_model_config(args, yaml_config)
    model = BaguettotronForCausalLM(model_config)

    num_params = model.count_parameters()
    logger.info(f"Model parameters: {num_params / 1e6:.1f}M")

    # Initialize tokenizer (required for HuggingFace dataset loading)
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(
        tokenizer_type="baguettotron",
        model_vocab_size=model_config.vocab_size
    )
    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Store checkpoint path for later loading by trainer
    checkpoint_dir = None
    if args.checkpoint:
        logger.info(f"Will resume training from checkpoint: {args.checkpoint}")

        # Normalize checkpoint path (handle both directory and file paths)
        if os.path.isdir(args.checkpoint):
            checkpoint_dir = args.checkpoint
            model_path = os.path.join(checkpoint_dir, 'model.pt')
            if not os.path.exists(model_path):
                logger.error(f"❌ Checkpoint directory exists but model.pt not found: {model_path}")
                sys.exit(1)
        else:
            # If file path provided, extract directory
            checkpoint_dir = os.path.dirname(args.checkpoint)

        # Load only model weights here (optimizer/scheduler loaded by trainer later)
        model_path = os.path.join(checkpoint_dir, 'model.pt')
        logger.info(f"Loading model weights from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')

        # Handle both raw state_dict and wrapped checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Handle torch.compile checkpoints (with _orig_mod. prefix)
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            logger.info("  → Detected torch.compile checkpoint, removing _orig_mod. prefix")
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        logger.info(f"✅ Model weights loaded successfully")

    # Create datasets - support both single and multi-dataset modes
    logger.info("Loading training data...")

    # Validate that we have either datasets or train_data
    if not (hasattr(args, 'datasets') and args.datasets) and not (hasattr(args, 'train_data') and args.train_data):
        logger.error("❌ No training data specified!")
        logger.error("Either provide:")
        logger.error("  1. --train-data data/train.json")
        logger.error("  2. --datasets synth,wikipedia")
        logger.error("  3. YAML config with dataset.train_data or dataset.datasets")
        sys.exit(1)

    # Check if using multi-dataset mode
    if hasattr(args, 'datasets') and args.datasets:
        from baguettotron.data.multi_dataset import create_multi_dataset, DatasetConfig

        # Support 3 syntaxes for datasets:
        # 1. String: "wikipedia,synth" (CLI style)
        # 2. List of strings: ["wikipedia", "synth"] (simple YAML)
        # 3. List of dicts: [{name: "wikipedia", path: "...", weight: 1.0}] (full YAML)

        if isinstance(args.datasets, str):
            # Syntax 1: String (CLI)
            dataset_names = [d.strip() for d in args.datasets.split(',') if d.strip()]

            # Parse weights if provided
            if hasattr(args, 'dataset_weights') and args.dataset_weights:
                weights = [float(w.strip()) for w in args.dataset_weights.split(',')]
            else:
                weights = None  # Equal weights

            logger.info(f"Multi-dataset mode: {dataset_names}")
            if weights:
                logger.info(f"Dataset weights: {weights}")

            # Use the create_multi_dataset helper function
            try:
                train_dataset = create_multi_dataset(
                    datasets=args.datasets,
                    weights=weights,
                    data_dir='data',
                    block_size=args.block_size,
                )
            except FileNotFoundError as e:
                logger.error(f"❌ {e}")
                logger.error(f"\nPrepare missing datasets first:")
                for name in dataset_names:
                    logger.error(f"   ./baguettotron dataset prepare --type {name} --tokenize")
                sys.exit(1)

        elif isinstance(args.datasets, list):
            # Check if list of strings or list of dicts
            if len(args.datasets) > 0 and isinstance(args.datasets[0], dict):
                # Syntax 3: List of dicts (full YAML with paths and weights)
                # TextDataset already imported at top of file
                from baguettotron.data.huggingface_loader import (
                    parse_huggingface_config,
                    load_huggingface_dataset
                )

                dataset_configs = []
                for ds_config in args.datasets:
                    name = ds_config.get('name', 'unnamed')
                    source = ds_config.get('source', 'local').lower()
                    weight = ds_config.get('weight', 1.0)

                    # Check if HuggingFace dataset
                    if source in ['huggingface', 'hf', 'hub']:
                        # HuggingFace dataset - requires dataset_name
                        dataset_name = ds_config.get('dataset_name')
                        if not dataset_name:
                            logger.error(f"❌ HuggingFace dataset '{name}' missing 'dataset_name' field")
                            sys.exit(1)

                        dataset_configs.append({
                            'name': name,
                            'source': 'huggingface',
                            'config': ds_config,
                            'weight': weight
                        })
                    else:
                        # Local file dataset - requires path
                        path = ds_config.get('path')
                        if not path:
                            logger.error(f"❌ Dataset '{name}' missing 'path' field")
                            sys.exit(1)

                        dataset_configs.append({
                            'name': name,
                            'source': 'local',
                            'path': path,
                            'weight': weight
                        })

                logger.info(f"Multi-dataset mode: {[d['name'] for d in dataset_configs]}")
                logger.info(f"\n✅ Loading {len(dataset_configs)} datasets:")
                total_weight = sum(d['weight'] for d in dataset_configs)
                for ds in dataset_configs:
                    percentage = (ds['weight'] / total_weight) * 100
                    source_info = f"[{ds['source']}]"
                    logger.info(f"   - {ds['name']} {source_info}: (weight: {percentage:.2f}%)")

                # Load each dataset and create MultiDataset manually
                from baguettotron.data.multi_dataset import MultiDataset

                loaded_datasets = []
                weights = []
                for ds_config in dataset_configs:
                    try:
                        if ds_config['source'] == 'huggingface':
                            # Load from HuggingFace Hub
                            hf_config = parse_huggingface_config(ds_config['config'])
                            if hf_config is None:
                                logger.error(f"❌ Failed to parse HuggingFace config for '{ds_config['name']}'")
                                sys.exit(1)

                            logger.info(f"   📥 Loading {ds_config['name']} from HuggingFace...")
                            dataset = load_huggingface_dataset(
                                hf_config,
                                tokenizer,
                                block_size=args.block_size
                            )
                            loaded_datasets.append(dataset)
                            weights.append(ds_config['weight'])
                            logger.info(f"   ✅ Loaded {ds_config['name']}: {len(dataset)} examples")
                        else:
                            # Load from local file
                            dataset = TextDataset(ds_config['path'], block_size=args.block_size)
                            loaded_datasets.append(dataset)
                            weights.append(ds_config['weight'])
                            logger.info(f"   ✅ Loaded {ds_config['name']}: {len(dataset)} examples")
                    except FileNotFoundError:
                        logger.error(f"❌ Dataset file not found: {ds_config.get('path', 'N/A')}")
                        sys.exit(1)
                    except Exception as e:
                        logger.error(f"❌ Failed to load dataset '{ds_config['name']}': {e}")
                        sys.exit(1)

                logger.info(f"   Total: {sum(len(d) for d in loaded_datasets)} examples")
                train_dataset = MultiDataset(loaded_datasets, weights)

            else:
                # Syntax 2: List of strings (simple YAML)
                dataset_names = args.datasets

                # Parse weights if provided
                if hasattr(args, 'dataset_weights') and args.dataset_weights:
                    if isinstance(args.dataset_weights, str):
                        weights = [float(w.strip()) for w in args.dataset_weights.split(',')]
                    else:
                        weights = args.dataset_weights
                else:
                    weights = None

                logger.info(f"Multi-dataset mode: {dataset_names}")
                if weights:
                    logger.info(f"Dataset weights: {weights}")

                try:
                    train_dataset = create_multi_dataset(
                        datasets=','.join(dataset_names),
                        weights=weights,
                        data_dir='data',
                        block_size=args.block_size,
                    )
                except FileNotFoundError as e:
                    logger.error(f"❌ {e}")
                    logger.error(f"\nPrepare missing datasets first:")
                    for name in dataset_names:
                        logger.error(f"   ./baguettotron dataset prepare --type {name} --tokenize")
                    sys.exit(1)
        else:
            logger.error(f"❌ Invalid datasets format: {type(args.datasets)}")
            sys.exit(1)
    else:
        # Single dataset mode
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

    # Create dataloaders with collator for padding
    from baguettotron.data import DataCollatorForLanguageModeling

    collator = DataCollatorForLanguageModeling(mlm=False)

    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = create_dataloader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
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
        save_total_limit=args.save_total_limit,
        output_dir=args.output_dir,
        mixed_precision=args.mixed_precision,
        use_compile=args.compile,
    )

    # Load trainer state (optimizer, scheduler, step count) if resuming
    if checkpoint_dir:
        trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.pt')
        if os.path.exists(trainer_state_path):
            logger.info(f"Loading trainer state from {trainer_state_path}")
            trainer_state = torch.load(trainer_state_path, map_location='cpu')

            # Restore optimizer state
            if 'optimizer_state_dict' in trainer_state:
                optimizer.load_state_dict(trainer_state['optimizer_state_dict'])
                logger.info("  → Optimizer state restored")

            # Restore scheduler state
            if 'scheduler_state_dict' in trainer_state and scheduler is not None:
                scheduler.load_state_dict(trainer_state['scheduler_state_dict'])
                logger.info("  → Scheduler state restored")

            # Restore trainer state
            if 'global_step' in trainer_state:
                trainer.global_step = trainer_state['global_step']
                logger.info(f"  → Resuming from step {trainer.global_step}")

            if 'epoch' in trainer_state:
                trainer.current_epoch = trainer_state['epoch']
                logger.info(f"  → Resuming from epoch {trainer.current_epoch}")

            logger.info(f"✅ Trainer state restored successfully")
        else:
            logger.warning(f"⚠️  trainer_state.pt not found in {checkpoint_dir}, starting fresh optimizer/scheduler")

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
