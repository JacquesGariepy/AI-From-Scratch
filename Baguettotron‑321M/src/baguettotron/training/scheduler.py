"""
Learning rate schedulers for Baguettotron training.

This module provides various learning rate scheduling strategies
commonly used for training large language models.
"""

import math
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a learning rate scheduler with linear warmup and linear decay.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: Index of last epoch (for resuming)

    Returns:
        Learning rate scheduler

    Examples:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = get_linear_schedule_with_warmup(
        ...     optimizer,
        ...     num_warmup_steps=1000,
        ...     num_training_steps=10000
        ... )
        >>> for step in range(10000):
        ...     optimizer.step()
        ...     scheduler.step()
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Linear decay
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a learning rate scheduler with cosine annealing and warmup.

    The learning rate decreases following a cosine curve after warmup,
    which is commonly used for transformer training.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles (0.5 is standard)
        last_epoch: Index of last epoch (for resuming)

    Returns:
        Learning rate scheduler

    Examples:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = get_cosine_schedule_with_warmup(
        ...     optimizer,
        ...     num_warmup_steps=1000,
        ...     num_training_steps=10000
        ... )
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a learning rate scheduler with warmup then constant rate.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        last_epoch: Index of last epoch (for resuming)

    Returns:
        Learning rate scheduler

    Examples:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = get_constant_schedule_with_warmup(
        ...     optimizer,
        ...     num_warmup_steps=1000
        ... )
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float = 0.0,
    power: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a learning rate scheduler with polynomial decay after warmup.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        lr_end: Final learning rate (as fraction of initial lr)
        power: Power of the polynomial (1.0 = linear)
        last_epoch: Index of last epoch (for resuming)

    Returns:
        Learning rate scheduler

    Examples:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = get_polynomial_decay_schedule_with_warmup(
        ...     optimizer,
        ...     num_warmup_steps=1000,
        ...     num_training_steps=10000,
        ...     power=2.0  # Quadratic decay
        ... )
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        lr_range = 1.0 - lr_end
        pct_remaining = 1 - (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
        return lr_range * pct_remaining ** power + lr_end

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_inverse_sqrt_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    timescale: Optional[int] = None,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create an inverse square root learning rate scheduler.

    Used in "Attention is All You Need" and other transformer papers.
    Learning rate increases linearly during warmup, then decays as 1/sqrt(step).

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        timescale: Timescale for decay (default: num_warmup_steps)
        last_epoch: Index of last epoch (for resuming)

    Returns:
        Learning rate scheduler

    Examples:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = get_inverse_sqrt_schedule_with_warmup(
        ...     optimizer,
        ...     num_warmup_steps=4000
        ... )
    """
    if timescale is None:
        timescale = num_warmup_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return math.sqrt(float(timescale) / float(max(1, current_step)))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    num_warmup_steps: int = 1000,
    num_training_steps: int = 10000,
    **kwargs
) -> LambdaLR:
    """
    Create a learning rate scheduler of the specified type.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ('linear', 'cosine', 'constant',
                       'polynomial', 'inverse_sqrt')
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        **kwargs: Additional arguments for specific schedulers

    Returns:
        Learning rate scheduler

    Examples:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>>
        >>> # Cosine with warmup (default)
        >>> scheduler = create_scheduler(
        ...     optimizer,
        ...     scheduler_type='cosine',
        ...     num_warmup_steps=1000,
        ...     num_training_steps=10000
        ... )
        >>>
        >>> # Linear with warmup
        >>> scheduler = create_scheduler(
        ...     optimizer,
        ...     scheduler_type='linear',
        ...     num_warmup_steps=1000,
        ...     num_training_steps=10000
        ... )
    """
    scheduler_type = scheduler_type.lower()

    logger.info(f"Creating {scheduler_type} scheduler")
    logger.info(f"  Warmup steps: {num_warmup_steps}")
    logger.info(f"  Training steps: {num_training_steps}")

    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            num_training_steps,
        )
    elif scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            num_training_steps,
            num_cycles=kwargs.get('num_cycles', 0.5),
        )
    elif scheduler_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
        )
    elif scheduler_type == 'polynomial':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            num_training_steps,
            lr_end=kwargs.get('lr_end', 0.0),
            power=kwargs.get('power', 1.0),
        )
    elif scheduler_type == 'inverse_sqrt':
        scheduler = get_inverse_sqrt_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            timescale=kwargs.get('timescale'),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler
