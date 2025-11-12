"""
Optimizer configurations for Baguettotron training.

This module provides utilities for creating and configuring optimizers
with layer-wise learning rate decay and parameter grouping.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
    optimizer_type: str = 'adamw',
    no_decay_params: Optional[List[str]] = None,
) -> torch.optim.Optimizer:
    """
    Create an optimizer with proper parameter grouping.

    Groups parameters into those that should and shouldn't have weight decay,
    typically excluding biases and layer norms from weight decay.

    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters
        eps: Adam epsilon
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd')
        no_decay_params: List of parameter name patterns to exclude from decay
                        (default: ['bias', 'norm', 'embedding'])

    Returns:
        Configured optimizer

    Examples:
        >>> from baguettotron import BaguettotronForCausalLM, BaguettotronConfig
        >>> config = BaguettotronConfig()
        >>> model = BaguettotronForCausalLM(config)
        >>>
        >>> # Standard AdamW with weight decay
        >>> optimizer = create_optimizer(
        ...     model,
        ...     learning_rate=1e-4,
        ...     weight_decay=0.1
        ... )
        >>>
        >>> # SGD optimizer
        >>> optimizer = create_optimizer(
        ...     model,
        ...     learning_rate=0.01,
        ...     optimizer_type='sgd'
        ... )
    """
    if no_decay_params is None:
        no_decay_params = ['bias', 'norm', 'embedding']

    # Separate parameters into groups
    decay_params = []
    no_decay_params_list = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter should have weight decay
        should_decay = True
        for pattern in no_decay_params:
            if pattern in name.lower():
                should_decay = False
                break

        if should_decay:
            decay_params.append(param)
        else:
            no_decay_params_list.append(param)

    # Create parameter groups
    param_groups = [
        {
            'params': decay_params,
            'weight_decay': weight_decay,
            'lr': learning_rate,
        },
        {
            'params': no_decay_params_list,
            'weight_decay': 0.0,
            'lr': learning_rate,
        },
    ]

    logger.info(f"Optimizer: {optimizer_type}")
    logger.info(f"  Params with weight decay: {len(decay_params)}")
    logger.info(f"  Params without weight decay: {len(no_decay_params_list)}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Weight decay: {weight_decay}")

    # Create optimizer
    if optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=learning_rate,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def create_optimizer_with_layer_decay(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    layer_decay: float = 0.65,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with layer-wise learning rate decay.

    Applies exponentially decaying learning rates to deeper layers,
    which can improve training stability and performance for large models.

    Args:
        model: Model to optimize
        learning_rate: Base learning rate for top layers
        weight_decay: Weight decay coefficient
        layer_decay: Decay factor for each layer (lr_layer = lr * layer_decay^depth)
        betas: Adam beta parameters
        eps: Adam epsilon

    Returns:
        AdamW optimizer with layer-wise learning rates

    Examples:
        >>> model = BaguettotronForCausalLM(config)
        >>> optimizer = create_optimizer_with_layer_decay(
        ...     model,
        ...     learning_rate=1e-4,
        ...     layer_decay=0.65
        ... )
        >>> # Layer 0 (embeddings): lr = 1e-4 * 0.65^24
        >>> # Layer 23 (top): lr = 1e-4 * 0.65^1
        >>> # Layer 24 (head): lr = 1e-4
    """
    # Get number of layers
    num_layers = 0
    for name, _ in model.named_parameters():
        if 'decoder.layers' in name:
            # Extract layer number
            layer_num = int(name.split('decoder.layers.')[1].split('.')[0])
            num_layers = max(num_layers, layer_num + 1)

    logger.info(f"Detected {num_layers} transformer layers")

    # Create parameter groups with layer-wise learning rates
    param_groups: Dict[str, Dict[str, Any]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine layer depth (higher depth = closer to input)
        if 'embeddings' in name:
            depth = num_layers + 1  # Embeddings are deepest
        elif 'decoder.layers' in name:
            layer_num = int(name.split('decoder.layers.')[1].split('.')[0])
            depth = num_layers - layer_num
        elif 'norm' in name or 'lm_head' in name:
            depth = 0  # Top layers
        else:
            depth = num_layers // 2  # Default to middle

        # Compute layer-specific learning rate
        layer_lr = learning_rate * (layer_decay ** depth)

        # Determine weight decay
        should_decay = not any(nd in name.lower() for nd in ['bias', 'norm', 'embedding'])
        wd = weight_decay if should_decay else 0.0

        # Create group key
        group_key = f"depth_{depth}_wd_{wd}"

        # Add to parameter group
        if group_key not in param_groups:
            param_groups[group_key] = {
                'params': [],
                'lr': layer_lr,
                'weight_decay': wd,
            }

        param_groups[group_key]['params'].append(param)

    # Convert to list
    param_groups_list = list(param_groups.values())

    logger.info(f"Created {len(param_groups_list)} parameter groups with layer decay {layer_decay}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        param_groups_list,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )

    return optimizer


def get_parameter_count(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in the model.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with parameter counts

    Examples:
        >>> model = BaguettotronForCausalLM(config)
        >>> counts = get_parameter_count(model)
        >>> print(f"Total: {counts['total'] / 1e6:.1f}M")
        >>> print(f"Trainable: {counts['trainable'] / 1e6:.1f}M")
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
    }


def freeze_parameters(model: nn.Module, freeze_patterns: List[str]) -> None:
    """
    Freeze parameters matching given patterns.

    Args:
        model: Model to modify
        freeze_patterns: List of parameter name patterns to freeze

    Examples:
        >>> # Freeze embeddings
        >>> freeze_parameters(model, ['embeddings'])
        >>>
        >>> # Freeze first 12 layers
        >>> freeze_parameters(model, [f'decoder.layers.{i}' for i in range(12)])
    """
    frozen_count = 0

    for name, param in model.named_parameters():
        for pattern in freeze_patterns:
            if pattern in name:
                param.requires_grad = False
                frozen_count += param.numel()
                break

    logger.info(f"Froze {frozen_count:,} parameters")


def unfreeze_all_parameters(model: nn.Module) -> None:
    """
    Unfreeze all parameters in the model.

    Args:
        model: Model to modify

    Examples:
        >>> unfreeze_all_parameters(model)
    """
    for param in model.parameters():
        param.requires_grad = True

    logger.info("Unfroze all parameters")
