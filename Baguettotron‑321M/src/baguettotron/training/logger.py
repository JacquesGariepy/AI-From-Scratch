"""
Advanced logging for training with TensorBoard and WandB support.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch

logger = logging.getLogger(__name__)


class TrainingLogger:
    """
    Unified logger supporting TensorBoard and Weights & Biases.

    Args:
        output_dir: Directory for logs
        use_tensorboard: Enable TensorBoard logging
        use_wandb: Enable Weights & Biases logging
        tensorboard_dir: Custom TensorBoard directory (default: output_dir/tensorboard)
        wandb_project: WandB project name
        wandb_run_name: WandB run name
        wandb_config: WandB configuration dict
    """

    def __init__(
        self,
        output_dir: str | Path,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        tensorboard_dir: Optional[str | Path] = None,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

        # TensorBoard setup
        self.writer = None
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = Path(tensorboard_dir) if tensorboard_dir else self.output_dir / "tensorboard"
                tb_dir.mkdir(parents=True, exist_ok=True)

                self.writer = SummaryWriter(log_dir=str(tb_dir))
                logger.info(f"TensorBoard logging enabled: {tb_dir}")
            except ImportError:
                logger.warning(
                    "TensorBoard not available. Install with: pip install tensorboard"
                )
                self.use_tensorboard = False

        # WandB setup
        self.wandb = None
        if self.use_wandb:
            try:
                import wandb

                self.wandb = wandb
                wandb.init(
                    project=wandb_project or "baguettotron",
                    name=wandb_run_name,
                    config=wandb_config or {},
                    dir=str(self.output_dir),
                )
                logger.info(f"Weights & Biases logging enabled: {wandb_project}")
            except ImportError:
                logger.warning("WandB not available. Install with: pip install wandb")
                self.use_wandb = False

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: int,
        prefix: str = ""
    ):
        """
        Log metrics to all enabled loggers.

        Args:
            metrics: Dictionary of metrics to log
            step: Global step number
            prefix: Prefix for metric names (e.g., "train/", "eval/")
        """
        # TensorBoard logging
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{prefix}{key}", value, step)
            # Flush every time to ensure metrics are written immediately
            self.writer.flush()

        # WandB logging
        if self.wandb is not None and self.wandb.run is not None:
            wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items()
                           if isinstance(v, (int, float, bool))}
            self.wandb.log(wandb_metrics, step=step)

    def log_layer_metrics(
        self,
        layer_metrics: Dict[str, Dict[str, float]],
        step: int,
        prefix: str = "layers/"
    ):
        """
        Log per-layer metrics.

        Args:
            layer_metrics: Dictionary mapping layer names to their metrics
            step: Global step number
            prefix: Prefix for metric names
        """
        if self.writer is not None:
            for layer_name, metrics in layer_metrics.items():
                # Sanitize layer name for TensorBoard
                clean_name = layer_name.replace('.', '/')
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(
                            f"{prefix}{clean_name}/{metric_name}",
                            value,
                            step
                        )
            # Flush to ensure metrics are written
            self.writer.flush()

        if self.wandb is not None and self.wandb.run is not None:
            wandb_metrics = {}
            for layer_name, metrics in layer_metrics.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        wandb_metrics[f"{prefix}{layer_name}/{metric_name}"] = value
            self.wandb.log(wandb_metrics, step=step)

    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: int
    ):
        """
        Log histogram of values (weights, gradients, etc.).

        Args:
            tag: Name for the histogram
            values: Tensor of values
            step: Global step number
        """
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)

        if self.wandb is not None and self.wandb.run is not None:
            self.wandb.log({tag: self.wandb.Histogram(values.cpu().numpy())}, step=step)

    def log_model_graph(
        self,
        model: torch.nn.Module,
        input_shape: tuple
    ):
        """
        Log model architecture graph.

        Args:
            model: PyTorch model
            input_shape: Example input shape (e.g., (batch_size, seq_len))
        """
        if self.writer is not None:
            try:
                dummy_input = torch.zeros(input_shape, dtype=torch.long)
                self.writer.add_graph(model, dummy_input)
                logger.info("Model graph logged to TensorBoard")
            except Exception as e:
                logger.warning(f"Failed to log model graph: {e}")

        if self.wandb is not None and self.wandb.run is not None:
            try:
                self.wandb.watch(model, log="all", log_freq=100)
            except Exception as e:
                logger.warning(f"Failed to watch model in WandB: {e}")

    def log_text(
        self,
        tag: str,
        text: str,
        step: int
    ):
        """
        Log text samples.

        Args:
            tag: Tag for the text
            text: Text content
            step: Global step number
        """
        if self.writer is not None:
            self.writer.add_text(tag, text, step)

        if self.wandb is not None and self.wandb.run is not None:
            self.wandb.log({tag: self.wandb.Html(text)}, step=step)

    def log_learning_rate(
        self,
        lr: float,
        step: int
    ):
        """
        Log current learning rate.

        Args:
            lr: Learning rate value
            step: Global step number
        """
        self.log_metrics({'learning_rate': lr}, step, prefix='optimizer/')

    def close(self):
        """Close all loggers."""
        if self.writer is not None:
            self.writer.close()
            logger.info("TensorBoard writer closed")

        if self.wandb is not None and self.wandb.run is not None:
            self.wandb.finish()
            logger.info("WandB run finished")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
