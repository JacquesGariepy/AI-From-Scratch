"""
Advanced training metrics for state-of-the-art model monitoring.

This module provides comprehensive metrics tracking including:
- Loss metrics (perplexity, BPC)
- Performance metrics (throughput, GPU utilization)
- Stability metrics (gradient norms, dead neurons)
- Layer-wise analysis
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging
import time
import math

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks SOTA metrics for language model training.

    Metrics tracked:
    - Loss: training loss, validation loss, perplexity, bits-per-character
    - Performance: steps/sec, tokens/sec, GPU memory, GPU utilization
    - Gradients: pre-clip norm, post-clip norm, layer-wise norms
    - Weights: weight norms per layer, effective rank
    - Stability: NaN/Inf detection, loss spikes, gradient variance
    - Activations: mean, std, dead neurons
    """

    def __init__(
        self,
        model: nn.Module,
        log_interval: int = 10,
        track_layer_wise: bool = True,
        track_activations: bool = False,
    ):
        self.model = model
        self.log_interval = log_interval
        self.track_layer_wise = track_layer_wise
        self.track_activations = track_activations

        # Metrics storage
        self.reset_metrics()

        # Timing
        self.start_time = None
        self.step_start_time = None

        # Gradient tracking
        self.grad_norms_history = []
        self.loss_history = []

    def reset_metrics(self):
        """Reset all metrics counters."""
        self.metrics = {
            # Loss metrics
            'train_loss': 0.0,
            'perplexity': 0.0,
            'bits_per_char': 0.0,

            # Performance
            'steps_per_sec': 0.0,
            'tokens_per_sec': 0.0,
            'gpu_mem_allocated': 0.0,
            'gpu_mem_reserved': 0.0,
            'gpu_mem_peak': 0.0,

            # Gradients
            'grad_norm_pre_clip': 0.0,
            'grad_norm_post_clip': 0.0,
            'grad_norm_variance': 0.0,

            # Weights
            'weight_norm': 0.0,

            # Stability
            'nan_count': 0,
            'inf_count': 0,
            'loss_spike': False,
        }

    def start_step(self):
        """Mark the start of a training step."""
        self.step_start_time = time.time()

    def compute_loss_metrics(self, loss: float, num_tokens: Optional[int] = None) -> Dict[str, float]:
        """
        Compute loss-based metrics.

        Args:
            loss: Cross-entropy loss value
            num_tokens: Number of tokens in batch (for throughput)

        Returns:
            Dictionary of loss metrics
        """
        metrics = {}

        # Basic loss
        metrics['loss'] = loss

        # Perplexity (exp of cross-entropy loss)
        try:
            perplexity = math.exp(loss)
            # Cap perplexity at reasonable value to avoid overflow
            metrics['perplexity'] = min(perplexity, 1e6)
        except (OverflowError, ValueError):
            metrics['perplexity'] = float('inf')

        # Bits per character (log2(perplexity))
        if metrics['perplexity'] < float('inf'):
            metrics['bits_per_char'] = math.log2(metrics['perplexity'])
        else:
            metrics['bits_per_char'] = float('inf')

        # Track loss history for spike detection
        self.loss_history.append(loss)
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)

        # Detect loss spikes (sudden increase > 2x recent average)
        if len(self.loss_history) > 10:
            recent_avg = sum(self.loss_history[-10:]) / 10
            if loss > 2 * recent_avg:
                metrics['loss_spike'] = True
                logger.warning(f"Loss spike detected! Current: {loss:.4f}, Recent avg: {recent_avg:.4f}")

        return metrics

    def compute_gradient_metrics(
        self,
        parameters: List[torch.nn.Parameter],
        pre_clip: bool = True
    ) -> Dict[str, float]:
        """
        Compute gradient-based metrics.

        Args:
            parameters: Model parameters to analyze
            pre_clip: Whether this is before or after gradient clipping

        Returns:
            Dictionary of gradient metrics
        """
        metrics = {}

        # Compute global gradient norm
        total_norm = 0.0
        nan_count = 0
        inf_count = 0

        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

                # Check for NaN/Inf
                if torch.isnan(p.grad).any():
                    nan_count += 1
                if torch.isinf(p.grad).any():
                    inf_count += 1

        total_norm = total_norm ** 0.5

        if pre_clip:
            metrics['grad_norm_pre_clip'] = total_norm
            self.grad_norms_history.append(total_norm)
            if len(self.grad_norms_history) > 100:
                self.grad_norms_history.pop(0)

            # Compute gradient variance (stability metric)
            if len(self.grad_norms_history) > 10:
                import statistics
                metrics['grad_norm_variance'] = statistics.variance(self.grad_norms_history[-10:])
        else:
            metrics['grad_norm_post_clip'] = total_norm

        metrics['nan_count'] = nan_count
        metrics['inf_count'] = inf_count

        if nan_count > 0 or inf_count > 0:
            logger.error(f"Detected {nan_count} NaN and {inf_count} Inf gradients!")

        return metrics

    def compute_layer_wise_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute per-layer metrics for detailed analysis.

        Returns:
            Dictionary mapping layer names to their metrics
        """
        layer_metrics = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                layer_metrics[name] = {
                    'grad_norm': param.grad.data.norm(2).item(),
                    'weight_norm': param.data.norm(2).item(),
                    'grad_mean': param.grad.data.mean().item(),
                    'grad_std': param.grad.data.std().item(),
                }

        return layer_metrics

    def compute_gpu_metrics(self) -> Dict[str, float]:
        """
        Compute GPU memory and utilization metrics.

        Returns:
            Dictionary of GPU metrics
        """
        metrics = {}

        if torch.cuda.is_available():
            # Memory in GB
            metrics['gpu_mem_allocated'] = torch.cuda.memory_allocated() / 1024**3
            metrics['gpu_mem_reserved'] = torch.cuda.memory_reserved() / 1024**3
            metrics['gpu_mem_peak'] = torch.cuda.max_memory_allocated() / 1024**3

            # GPU utilization (requires nvidia-ml-py3)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics['gpu_utilization'] = utilization.gpu
                metrics['gpu_memory_utilization'] = utilization.memory
                pynvml.nvmlShutdown()
            except Exception:
                # pynvml not available, skip GPU utilization
                pass

        return metrics

    def compute_throughput_metrics(
        self,
        num_tokens: int,
        elapsed_time: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute throughput metrics.

        Args:
            num_tokens: Number of tokens processed
            elapsed_time: Time elapsed (uses step timer if None)

        Returns:
            Dictionary of throughput metrics
        """
        metrics = {}

        if elapsed_time is None and self.step_start_time is not None:
            elapsed_time = time.time() - self.step_start_time

        if elapsed_time and elapsed_time > 0:
            metrics['steps_per_sec'] = 1.0 / elapsed_time
            metrics['tokens_per_sec'] = num_tokens / elapsed_time

        return metrics

    def compute_weight_metrics(self) -> Dict[str, float]:
        """
        Compute weight-based metrics.

        Returns:
            Dictionary of weight metrics
        """
        metrics = {}

        # Global weight norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.requires_grad:
                param_norm = p.data.norm(2)
                total_norm += param_norm.item() ** 2

        metrics['weight_norm'] = total_norm ** 0.5

        # Embedding effective rank (measure of utilization)
        try:
            if hasattr(self.model, 'embeddings'):
                emb_weight = self.model.embeddings.weight.data
                # Compute singular values
                U, S, V = torch.svd(emb_weight)
                # Effective rank: (sum of singular values)^2 / (sum of squared singular values)
                effective_rank = (S.sum() ** 2) / (S ** 2).sum()
                metrics['embedding_effective_rank'] = effective_rank.item()
        except Exception as e:
            logger.debug(f"Could not compute embedding effective rank: {e}")

        return metrics

    def get_all_metrics(
        self,
        loss: float,
        num_tokens: int,
        parameters: List[torch.nn.Parameter],
        pre_clip: bool = True
    ) -> Dict[str, Any]:
        """
        Compute all metrics at once.

        Args:
            loss: Training loss
            num_tokens: Number of tokens in batch
            parameters: Model parameters
            pre_clip: Whether gradients are pre or post clip

        Returns:
            Dictionary containing all computed metrics
        """
        all_metrics = {}

        # Loss metrics
        all_metrics.update(self.compute_loss_metrics(loss, num_tokens))

        # Gradient metrics
        all_metrics.update(self.compute_gradient_metrics(parameters, pre_clip))

        # GPU metrics
        all_metrics.update(self.compute_gpu_metrics())

        # Throughput metrics
        all_metrics.update(self.compute_throughput_metrics(num_tokens))

        # Weight metrics computed separately (expensive, called less frequently)

        return all_metrics

    def format_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> str:
        """
        Format metrics for logging.

        Args:
            metrics: Dictionary of metrics
            prefix: Prefix to add to metric names

        Returns:
            Formatted string
        """
        lines = []
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                if 'loss' in key or 'norm' in key:
                    lines.append(f"{prefix}{key}: {value:.4f}")
                elif 'per_sec' in key:
                    lines.append(f"{prefix}{key}: {value:.2f}")
                elif 'mem' in key:
                    lines.append(f"{prefix}{key}: {value:.2f}GB")
                else:
                    lines.append(f"{prefix}{key}: {value:.4f}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        return " | ".join(lines)
