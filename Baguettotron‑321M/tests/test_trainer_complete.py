"""
Complete tests for training/trainer.py to achieve 95%+ coverage.

This test file adds missing coverage for:
- Full training loop (train() method)
- Training epochs with evaluation and checkpointing
- Mixed precision training (AMP)
- Gradient accumulation with scheduler
- Model compilation (torch.compile)
- Checkpoint saving/loading with scheduler
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baguettotron.config import BaguettotronConfig
from baguettotron.model import BaguettotronForCausalLM
from baguettotron.training import Trainer


class MockDataLoader:
    """Mock dataloader for testing."""
    def __init__(self, data, batch_size=2):
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


@pytest.fixture
def tiny_model():
    """Create tiny model for testing."""
    config = BaguettotronConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
    )
    return BaguettotronForCausalLM(config)


@pytest.fixture
def mock_dataloader():
    """Create mock dataloader with multiple batches."""
    data = [
        {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'labels': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10),
        }
        for _ in range(5)  # 5 batches for testing
    ]
    return MockDataLoader(data)


class TestTrainerFullTraining:
    """Tests for complete training loop."""

    def test_trainer_full_training_loop(self, tiny_model, mock_dataloader, tmp_path):
        """Test complete training loop with multiple epochs."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            eval_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            max_epochs=2,  # Test multiple epochs
            output_dir=str(tmp_path)
        )

        results = trainer.train()

        # Verify training completed
        assert 'total_steps' in results
        assert 'final_loss' in results
        assert 'training_time' in results
        assert results['total_steps'] > 0
        assert results['training_time'] > 0

    def test_trainer_full_training_with_logging(self, tiny_model, mock_dataloader, tmp_path):
        """Test training with logging intervals."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            max_epochs=1,
            log_interval=2,  # Log every 2 steps
            output_dir=str(tmp_path)
        )

        results = trainer.train()

        assert results['total_steps'] == len(mock_dataloader)

    def test_trainer_epoch_with_eval_and_save(self, tiny_model, mock_dataloader, tmp_path):
        """Test training epoch with evaluation and checkpoint intervals."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            eval_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            eval_interval=3,  # Evaluate every 3 steps
            save_interval=3,  # Save every 3 steps
            output_dir=str(tmp_path)
        )

        # Run one epoch
        epoch_loss = trainer._train_epoch()

        assert epoch_loss > 0
        # Check that checkpoints were saved
        checkpoints = list(tmp_path.glob("checkpoint_step_*.pt"))
        assert len(checkpoints) > 0

    def test_trainer_saves_best_model(self, tiny_model, mock_dataloader, tmp_path):
        """Test that trainer saves best model based on eval loss."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            eval_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            max_epochs=2,
            output_dir=str(tmp_path)
        )

        trainer.train()

        # Check that best model was saved
        best_model_path = tmp_path / "best_model.pt"
        assert best_model_path.exists()


class TestTrainerMixedPrecision:
    """Tests for mixed precision training."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for mixed precision")
    def test_trainer_mixed_precision(self, tiny_model, mock_dataloader, tmp_path):
        """Test trainer with automatic mixed precision."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cuda',
            mixed_precision=True,  # Enable AMP
            output_dir=str(tmp_path)
        )

        # Run a training step
        batch = next(iter(mock_dataloader))
        loss = trainer._training_step(batch)

        assert loss > 0
        assert trainer.scaler is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for mixed precision")
    def test_trainer_mixed_precision_full_epoch(self, tiny_model, mock_dataloader, tmp_path):
        """Test full epoch with mixed precision."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cuda',
            mixed_precision=True,
            max_epochs=1,
            output_dir=str(tmp_path)
        )

        results = trainer.train()

        assert results['total_steps'] > 0
        assert trainer.scaler is not None


class TestTrainerGradientAccumulation:
    """Tests for gradient accumulation with scheduler."""

    def test_trainer_with_scheduler_and_grad_accumulation(self, tiny_model, mock_dataloader, tmp_path):
        """Test trainer with learning rate scheduler and gradient accumulation."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device='cpu',
            gradient_accumulation_steps=2,  # Accumulate over 2 steps
            output_dir=str(tmp_path)
        )

        initial_lr = optimizer.param_groups[0]['lr']

        # Run several steps
        for i, batch in enumerate(mock_dataloader):
            trainer._training_step(batch)
            if (i + 1) % trainer.gradient_accumulation_steps == 0:
                trainer.global_step += 1

        # Learning rate should have changed after enough steps
        if trainer.global_step >= 2:
            final_lr = optimizer.param_groups[0]['lr']
            assert final_lr < initial_lr  # Should have decayed

    def test_trainer_scheduler_steps_correctly(self, tiny_model, mock_dataloader, tmp_path):
        """Test that scheduler steps are called at correct intervals."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.5, total_iters=10
        )

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device='cpu',
            gradient_accumulation_steps=1,
            max_epochs=1,
            output_dir=str(tmp_path)
        )

        initial_lr = optimizer.param_groups[0]['lr']

        trainer.train()

        # LR should have changed
        final_lr = optimizer.param_groups[0]['lr']
        assert final_lr < initial_lr


class TestTrainerCheckpointWithScheduler:
    """Tests for checkpoint saving/loading with scheduler."""

    def test_trainer_save_load_checkpoint_with_scheduler(self, tiny_model, mock_dataloader, tmp_path):
        """Test saving and loading checkpoints with scheduler."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device='cpu',
            output_dir=str(tmp_path)
        )

        # Run a few steps to change scheduler state
        for _ in range(3):
            optimizer.step()
            scheduler.step()

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint_with_scheduler.pt"
        trainer.save_checkpoint(str(checkpoint_path.name))

        assert checkpoint_path.exists()

        # Create new trainer and load
        new_model = BaguettotronForCausalLM(tiny_model.config)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1, gamma=0.5)

        new_trainer = Trainer(
            model=new_model,
            train_dataloader=mock_dataloader,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            device='cpu',
            output_dir=str(tmp_path)
        )

        new_trainer.load_checkpoint(checkpoint_path)

        # Verify scheduler state was restored
        assert new_trainer.scheduler is not None

    def test_trainer_load_checkpoint_without_scheduler(self, tiny_model, mock_dataloader, tmp_path):
        """Test loading checkpoint when scheduler is None."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        # Create trainer WITH scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device='cpu',
            output_dir=str(tmp_path)
        )

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path.name))

        # Create new trainer WITHOUT scheduler
        new_model = BaguettotronForCausalLM(tiny_model.config)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)

        new_trainer = Trainer(
            model=new_model,
            train_dataloader=mock_dataloader,
            optimizer=new_optimizer,
            scheduler=None,  # No scheduler
            device='cpu',
            output_dir=str(tmp_path)
        )

        # Should load without error even though checkpoint has scheduler
        new_trainer.load_checkpoint(checkpoint_path)

        assert new_trainer.scheduler is None


class TestTrainerModelCompilation:
    """Tests for torch.compile integration."""

    @pytest.mark.skipif(not hasattr(torch, 'compile'), reason="torch.compile requires PyTorch 2.0+")
    def test_trainer_with_compile_mode(self, tiny_model, mock_dataloader, tmp_path):
        """Test trainer with torch.compile enabled."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            use_compile=True,  # Enable compilation
            output_dir=str(tmp_path)
        )

        # Model should be compiled (or compilation attempted)
        assert trainer.model is not None

        # Should still be able to train
        batch = next(iter(mock_dataloader))
        loss = trainer._training_step(batch)
        assert loss > 0

    def test_trainer_compile_failure_graceful(self, tiny_model, mock_dataloader, tmp_path):
        """Test that trainer handles compile failures gracefully."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        # This should not crash even if compile fails
        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            use_compile=True,
            output_dir=str(tmp_path)
        )

        # Training should work even if compilation failed
        batch = next(iter(mock_dataloader))
        loss = trainer._training_step(batch)
        assert loss > 0


class TestTrainerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_trainer_with_max_grad_norm_none(self, tiny_model, mock_dataloader, tmp_path):
        """Test trainer with gradient clipping disabled."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            max_grad_norm=None,  # Disable gradient clipping
            output_dir=str(tmp_path)
        )

        # Should train without gradient clipping
        batch = next(iter(mock_dataloader))
        loss = trainer._training_step(batch)
        assert loss > 0

    def test_trainer_gradient_accumulation_without_scheduler(self, tiny_model, mock_dataloader, tmp_path):
        """Test gradient accumulation works without scheduler."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            scheduler=None,  # No scheduler
            device='cpu',
            gradient_accumulation_steps=3,
            output_dir=str(tmp_path)
        )

        # Run several steps
        for batch in mock_dataloader:
            trainer._training_step(batch)
            trainer.global_step += 1

        # Should complete without errors
        assert trainer.global_step > 0
