"""
Training utilities for Baguettotron.

This module provides a Trainer class for training and evaluating
Baguettotron language models, with support for distributed training,
gradient accumulation, and mixed precision.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import logging
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for Baguettotron language models.

    Handles the training loop, evaluation, checkpointing, and logging
    for causal language modeling.

    Args:
        model: BaguettotronForCausalLM model to train
        train_dataloader: DataLoader for training data
        eval_dataloader: Optional DataLoader for evaluation
        optimizer: Optimizer for training
        scheduler: Optional learning rate scheduler
        device: Device to train on ('cuda', 'cpu', or torch.device)
        max_epochs: Maximum number of training epochs
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping (None to disable)
        log_interval: Log every N training steps
        eval_interval: Evaluate every N training steps
        save_interval: Save checkpoint every N training steps
        output_dir: Directory to save checkpoints and logs
        mixed_precision: Whether to use automatic mixed precision (AMP)
        use_compile: Whether to use torch.compile for faster training (PyTorch 2.0+)

    Attributes:
        model: The model being trained
        optimizer: The optimizer
        scheduler: Learning rate scheduler (if provided)
        device: Training device
        global_step: Current global training step
        epoch: Current epoch

    Examples:
        >>> from baguettotron import BaguettotronForCausalLM, BaguettotronConfig
        >>> from baguettotron.data import TextDataset, create_dataloader
        >>> import torch.optim as optim
        >>>
        >>> # Setup
        >>> config = BaguettotronConfig()
        >>> model = BaguettotronForCausalLM(config)
        >>> dataset = TextDataset('data/train.json', block_size=128)
        >>> dataloader = create_dataloader(dataset, batch_size=32)
        >>> optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        >>>
        >>> # Train
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_dataloader=dataloader,
        ...     optimizer=optimizer,
        ...     device='cuda',
        ...     max_epochs=10,
        ...     output_dir='checkpoints'
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str | torch.device = 'cuda',
        max_epochs: int = 10,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
        log_interval: int = 10,
        eval_interval: int = 1000,
        save_interval: int = 1000,
        save_total_limit: Optional[int] = None,
        output_dir: str | Path = 'outputs',
        mixed_precision: bool = False,
        use_compile: bool = False,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.save_total_limit = save_total_limit
        self.output_dir = Path(output_dir)
        self.mixed_precision = mixed_precision
        self.use_compile = use_compile

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.model.to(self.device)

        # Compile model if requested (PyTorch 2.0+)
        if self.use_compile:
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if mixed_precision else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')

        # Track saved checkpoints for cleanup
        self.saved_checkpoints = []

    def train(self) -> Dict[str, Any]:
        """
        Run the training loop.

        Returns:
            Dictionary with training statistics
        """
        logger.info("Starting training...")
        logger.info(f"  Num epochs: {self.max_epochs}")
        logger.info(f"  Train batch size: {self.train_dataloader.batch_size}")
        logger.info(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps: {self.max_epochs * len(self.train_dataloader) // self.gradient_accumulation_steps}")

        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for epoch in range(self.max_epochs):
            self.epoch = epoch
            epoch_loss = self._train_epoch()

            logger.info(f"Epoch {epoch + 1}/{self.max_epochs} - Loss: {epoch_loss:.4f}")

            # Evaluate if eval_dataloader provided
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                logger.info(f"Eval loss: {eval_metrics['loss']:.4f}")

                # Save best model
                if eval_metrics['loss'] < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics['loss']
                    self.save_checkpoint(filename='best_model')

            # Save epoch checkpoint
            self.save_checkpoint(step=self.global_step)

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        return {
            'total_steps': self.global_step,
            'final_loss': total_loss / self.global_step if self.global_step > 0 else 0,
            'training_time': training_time,
        }

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        epoch_loss = 0.0
        self.model.train()

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}/{self.max_epochs}",
        )

        for step, batch in enumerate(progress_bar):
            loss = self._training_step(batch)
            epoch_loss += loss

            # Monitor GPU memory usage
            if torch.cuda.is_available() and (step + 1) % 50 == 0:
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                mem_percent = (mem_allocated / mem_total) * 100

                if mem_percent > 90:
                    logger.warning(f"GPU memory usage high: {mem_percent:.1f}% ({mem_allocated:.2f}GB / {mem_total:.2f}GB)")
                    # Clear cache to prevent OOM
                    torch.cuda.empty_cache()

            # Update progress bar
            if (step + 1) % self.log_interval == 0:
                avg_loss = epoch_loss / (step + 1)
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'gpu_mem': f'{mem_allocated:.1f}GB'
                    })
                else:
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

            # Evaluation
            if (self.global_step + 1) % self.eval_interval == 0 and self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                logger.info(f"Step {self.global_step + 1} - Eval loss: {eval_metrics['loss']:.4f}")
                self.model.train()

            # Save checkpoint by steps
            if (self.global_step + 1) % self.save_interval == 0:
                self.save_checkpoint(step=self.global_step + 1)

            self.global_step += 1

        return epoch_loss / len(self.train_dataloader)

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step."""
        # Move batch to device (non-blocking for better performance)
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        # Forward pass with mixed precision if enabled
        if self.mixed_precision:
            with torch.amp.autocast('cuda'):
                outputs = self.model(batch['input_ids'], attention_mask=batch.get('attention_mask'))
                loss = self._compute_loss(outputs, batch['labels'])
        else:
            outputs = self.model(batch['input_ids'], attention_mask=batch.get('attention_mask'))
            loss = self._compute_loss(outputs, batch['labels'])

        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps

        # Backward pass
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (with gradient accumulation)
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.max_grad_norm is not None:
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Optimizer step
            if self.mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Zero gradients
            self.optimizer.zero_grad()

        return loss.item() * self.gradient_accumulation_steps

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for language modeling.

        Args:
            logits: Model output logits (batch, seq_len, vocab_size)
            labels: Target token IDs (batch, seq_len)

        Returns:
            Loss tensor
        """
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten for loss computation
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,  # Ignore padding tokens
        )

        return loss

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.

        Returns:
            Dictionary with evaluation metrics
        """
        if self.eval_dataloader is None:
            raise ValueError("No evaluation dataloader provided")

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(batch['input_ids'], attention_mask=batch.get('attention_mask'))
            loss = self._compute_loss(outputs, batch['labels'])

            total_loss += loss.item() * batch['input_ids'].size(0)
            total_tokens += batch['input_ids'].size(0)

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            'loss': avg_loss,
            'perplexity': perplexity,
        }

    def save_checkpoint(self, filename: str = None, step: int = None) -> None:
        """
        Save a checkpoint with separate model.pt and trainer_state.pt files.

        Creates a directory structure like: ckpt_12000/model.pt, ckpt_12000/trainer_state.pt

        Args:
            filename: Deprecated - use step instead
            step: Training step number (uses global_step if None)
        """
        import json
        import shutil

        # Use global_step if no step provided
        if step is None:
            step = self.global_step

        # Create checkpoint directory
        if filename and not filename.endswith('.pt'):
            # Old style: convert to step-based
            checkpoint_dir = self.output_dir / filename
        else:
            # New style: ckpt_XXXXX/
            checkpoint_dir = self.output_dir / f'ckpt_{step}'

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model separately
        model_path = checkpoint_dir / 'model.pt'
        torch.save(self.model.state_dict(), model_path)

        # Save trainer state separately
        trainer_state = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_eval_loss': self.best_eval_loss,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }
        trainer_state_path = checkpoint_dir / 'trainer_state.pt'
        torch.save(trainer_state, trainer_state_path)

        # Save config if model has it
        if hasattr(self.model, 'config'):
            config_path = checkpoint_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(vars(self.model.config), f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

        # Track saved checkpoints for cleanup
        self.saved_checkpoints.append(checkpoint_dir)

        # Cleanup old checkpoints if limit set
        if self.save_total_limit is not None and len(self.saved_checkpoints) > self.save_total_limit:
            # Remove oldest checkpoints
            checkpoints_to_remove = self.saved_checkpoints[:-self.save_total_limit]
            for old_checkpoint in checkpoints_to_remove:
                if old_checkpoint.exists() and old_checkpoint.is_dir():
                    shutil.rmtree(old_checkpoint)
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")
            # Keep only recent checkpoints in list
            self.saved_checkpoints = self.saved_checkpoints[-self.save_total_limit:]

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from step {self.global_step}, epoch {self.epoch}")
