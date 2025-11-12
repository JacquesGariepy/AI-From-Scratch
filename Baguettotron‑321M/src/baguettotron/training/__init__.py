"""
Training utilities for Baguettotron.

This module provides training loops, optimizers, learning rate schedulers,
and utilities for training Baguettotron language models.

Main Classes:
    Trainer: Complete training loop with evaluation and checkpointing

Optimizer Functions:
    create_optimizer: Create optimizer with parameter grouping
    create_optimizer_with_layer_decay: Create optimizer with layer-wise LR decay
    get_parameter_count: Count model parameters
    freeze_parameters: Freeze specific model parameters
    unfreeze_all_parameters: Unfreeze all parameters

Scheduler Functions:
    create_scheduler: Create LR scheduler of specified type
    get_linear_schedule_with_warmup: Linear warmup + linear decay
    get_cosine_schedule_with_warmup: Cosine annealing with warmup
    get_constant_schedule_with_warmup: Constant LR after warmup
    get_polynomial_decay_schedule_with_warmup: Polynomial decay
    get_inverse_sqrt_schedule_with_warmup: Inverse sqrt decay

Example:
    >>> from baguettotron import BaguettotronForCausalLM, BaguettotronConfig
    >>> from baguettotron.data import TextDataset, create_dataloader
    >>> from baguettotron.training import Trainer, create_optimizer, create_scheduler
    >>>
    >>> # Setup
    >>> config = BaguettotronConfig.baguettotron_321m()
    >>> model = BaguettotronForCausalLM(config)
    >>>
    >>> # Data
    >>> train_dataset = TextDataset('data/train.json', block_size=2048)
    >>> train_loader = create_dataloader(train_dataset, batch_size=32)
    >>>
    >>> # Optimizer and scheduler
    >>> optimizer = create_optimizer(model, learning_rate=1e-4, weight_decay=0.1)
    >>> scheduler = create_scheduler(
    ...     optimizer,
    ...     scheduler_type='cosine',
    ...     num_warmup_steps=1000,
    ...     num_training_steps=10000
    ... )
    >>>
    >>> # Train
    >>> trainer = Trainer(
    ...     model=model,
    ...     train_dataloader=train_loader,
    ...     optimizer=optimizer,
    ...     scheduler=scheduler,
    ...     device='cuda',
    ...     max_epochs=10,
    ...     output_dir='checkpoints'
    ... )
    >>> trainer.train()
"""

from .trainer import Trainer
from .optimizer import (
    create_optimizer,
    create_optimizer_with_layer_decay,
    get_parameter_count,
    freeze_parameters,
    unfreeze_all_parameters,
)
from .scheduler import (
    create_scheduler,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_inverse_sqrt_schedule_with_warmup,
)

__all__ = [
    # Trainer
    "Trainer",
    # Optimizer utilities
    "create_optimizer",
    "create_optimizer_with_layer_decay",
    "get_parameter_count",
    "freeze_parameters",
    "unfreeze_all_parameters",
    # Schedulers
    "create_scheduler",
    "get_linear_schedule_with_warmup",
    "get_cosine_schedule_with_warmup",
    "get_constant_schedule_with_warmup",
    "get_polynomial_decay_schedule_with_warmup",
    "get_inverse_sqrt_schedule_with_warmup",
]
