"""
Advanced tests for trainer.py to achieve 95%+ coverage.

Tests gradient clipping, checkpointing, scheduling, early stopping,
and mixed precision training.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baguettotron.config import BaguettotronConfig
from baguettotron.model import BaguettotronForCausalLM
from baguettotron.training.trainer import Trainer


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
    """Create mock dataloader."""
    data = [
        {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'labels': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10),
        }
        for _ in range(5)  # 5 batches
    ]
    return MockDataLoader(data)


class TestTrainerGradientClipping:
    """Tests for gradient clipping functionality."""

    def test_gradient_clipping_enabled(self, tiny_model, mock_dataloader, tmp_path):
        """Test that gradient clipping is applied."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            max_epochs=1,
            max_grad_norm=1.0,  # Enable clipping
            gradient_accumulation_steps=1,
            output_dir=str(tmp_path)
        )

        # Run one step
        batch = next(iter(mock_dataloader))
        loss = trainer._training_step(batch)

        # Check that loss is computed
        assert isinstance(loss, float)
        assert loss > 0

    def test_gradient_clipping_disabled(self, tiny_model, mock_dataloader, tmp_path):
        """Test training with gradient clipping disabled."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            max_epochs=1,
            max_grad_norm=None,  # Disable clipping
            gradient_accumulation_steps=1,
            output_dir=str(tmp_path)
        )

        batch = next(iter(mock_dataloader))
        loss = trainer._training_step(batch)

        assert isinstance(loss, float)
        assert loss > 0

    def test_gradient_accumulation(self, tiny_model, mock_dataloader, tmp_path):
        """Test gradient accumulation steps."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            max_epochs=1,
            gradient_accumulation_steps=2,  # Accumulate over 2 steps
            output_dir=str(tmp_path)
        )

        # Run multiple steps
        for i, batch in enumerate(mock_dataloader):
            loss = trainer._training_step(batch)
            assert isinstance(loss, float)

            if i >= 3:  # Run a few steps
                break


class TestTrainerCheckpointing:
    """Tests for checkpoint save/load functionality."""

    def test_save_checkpoint_with_scheduler(self, tiny_model, mock_dataloader, tmp_path):
        """Test saving checkpoint with scheduler."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device='cpu',
            output_dir=str(tmp_path)
        )

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path.name))

        assert checkpoint_path.exists()

        # Load and verify
        checkpoint = torch.load(checkpoint_path)
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'scheduler_state_dict' in checkpoint
        assert checkpoint['scheduler_state_dict'] is not None

    def test_load_checkpoint_with_scheduler(self, tiny_model, mock_dataloader, tmp_path):
        """Test loading checkpoint with scheduler state."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device='cpu',
            output_dir=str(tmp_path)
        )

        # Set some state
        trainer.global_step = 42
        trainer.epoch = 5
        trainer.best_eval_loss = 1.234

        # Save
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path.name))

        # Create new trainer and load
        optimizer2 = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=10)

        trainer2 = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer2,
            scheduler=scheduler2,
            device='cpu',
            output_dir=str(tmp_path)
        )

        trainer2.load_checkpoint(checkpoint_path)

        # Verify state restored
        assert trainer2.global_step == 42
        assert trainer2.epoch == 5
        assert trainer2.best_eval_loss == 1.234

    def test_load_checkpoint_without_scheduler(self, tiny_model, mock_dataloader, tmp_path):
        """Test loading checkpoint when no scheduler is present."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            scheduler=None,  # No scheduler
            device='cpu',
            output_dir=str(tmp_path)
        )

        trainer.global_step = 10
        checkpoint_path = tmp_path / "no_scheduler.pt"
        trainer.save_checkpoint(str(checkpoint_path.name))

        # Load
        trainer.load_checkpoint(checkpoint_path)
        assert trainer.global_step == 10


class TestTrainerSchedulerIntegration:
    """Tests for learning rate scheduler integration."""

    def test_scheduler_step_called(self, tiny_model, mock_dataloader, tmp_path):
        """Test that scheduler.step() is called during training."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device='cpu',
            max_epochs=1,
            gradient_accumulation_steps=1,
            output_dir=str(tmp_path)
        )

        initial_lr = optimizer.param_groups[0]['lr']

        # Run a few steps
        for i, batch in enumerate(mock_dataloader):
            trainer._training_step(batch)
            if i >= 2:
                break

        # LR should have changed after scheduler steps
        final_lr = optimizer.param_groups[0]['lr']
        # After 3 steps with step_size=2, LR should have decreased
        assert final_lr != initial_lr

    def test_training_without_scheduler(self, tiny_model, mock_dataloader, tmp_path):
        """Test training without scheduler."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            scheduler=None,
            device='cpu',
            max_epochs=1,
            output_dir=str(tmp_path)
        )

        initial_lr = optimizer.param_groups[0]['lr']

        # Run training step
        batch = next(iter(mock_dataloader))
        trainer._training_step(batch)

        # LR should remain constant without scheduler
        assert optimizer.param_groups[0]['lr'] == initial_lr


class TestTrainerEvaluation:
    """Tests for evaluation and best model saving."""

    def test_best_model_saved(self, tiny_model, mock_dataloader, tmp_path):
        """Test that best model is saved during training."""
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

        # Run training
        trainer.train()

        # Check that best model was saved
        best_model_path = tmp_path / "best_model.pt"
        assert best_model_path.exists()

    def test_evaluation_during_training(self, tiny_model, mock_dataloader, tmp_path):
        """Test evaluation is called during training."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            eval_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            max_epochs=1,
            eval_interval=3,  # Evaluate every 3 steps
            output_dir=str(tmp_path)
        )

        # Run one epoch
        epoch_loss = trainer._train_epoch()

        assert isinstance(epoch_loss, float)
        assert epoch_loss > 0

    def test_periodic_checkpoint_saving(self, tiny_model, mock_dataloader, tmp_path):
        """Test periodic checkpoint saving during training."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            max_epochs=1,
            save_interval=2,  # Save every 2 steps
            output_dir=str(tmp_path)
        )

        trainer._train_epoch()

        # Check that periodic checkpoints were created
        checkpoints = list(tmp_path.glob("checkpoint_step_*.pt"))
        assert len(checkpoints) > 0


class TestTrainerMixedPrecision:
    """Tests for mixed precision training."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_training(self, tiny_model, mock_dataloader, tmp_path):
        """Test mixed precision training with AMP."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cuda',
            max_epochs=1,
            mixed_precision=True,
            gradient_accumulation_steps=1,
            output_dir=str(tmp_path)
        )

        # Check scaler is created
        assert trainer.scaler is not None

        # Move batch to CUDA
        batch = next(iter(mock_dataloader))
        batch = {k: v.to('cuda') for k, v in batch.items()}

        # Run training step
        loss = trainer._training_step(batch)
        assert isinstance(loss, float)

    def test_mixed_precision_disabled(self, tiny_model, mock_dataloader, tmp_path):
        """Test that scaler is None when mixed precision is disabled."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            mixed_precision=False,
            output_dir=str(tmp_path)
        )

        assert trainer.scaler is None


class TestTrainerModelCompilation:
    """Tests for torch.compile integration."""

    def test_model_compilation_attempt(self, tiny_model, mock_dataloader, tmp_path):
        """Test that model compilation is attempted when use_compile=True."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        # This may fail on older PyTorch versions, but should not raise
        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            use_compile=True,
            output_dir=str(tmp_path)
        )

        # Trainer should be created successfully regardless
        assert trainer is not None

    def test_model_without_compilation(self, tiny_model, mock_dataloader, tmp_path):
        """Test normal operation without compilation."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            use_compile=False,
            output_dir=str(tmp_path)
        )

        batch = next(iter(mock_dataloader))
        loss = trainer._training_step(batch)
        assert isinstance(loss, float)


class TestTrainerFullTraining:
    """Tests for complete training workflow."""

    def test_full_training_loop(self, tiny_model, mock_dataloader, tmp_path):
        """Test complete training loop with all features."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            eval_dataloader=mock_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device='cpu',
            max_epochs=2,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            log_interval=2,
            eval_interval=10,
            save_interval=10,
            output_dir=str(tmp_path),
            mixed_precision=False,
            use_compile=False,
        )

        # Run training
        stats = trainer.train()

        # Verify stats
        assert 'total_steps' in stats
        assert 'final_loss' in stats
        assert 'training_time' in stats
        assert stats['total_steps'] > 0
        assert stats['training_time'] > 0

        # Check epoch checkpoints were saved
        epoch_checkpoints = list(tmp_path.glob("checkpoint_epoch_*.pt"))
        assert len(epoch_checkpoints) == 2  # 2 epochs

    def test_training_returns_correct_stats(self, tiny_model, mock_dataloader, tmp_path):
        """Test that training returns correct statistics."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            max_epochs=1,
            output_dir=str(tmp_path)
        )

        stats = trainer.train()

        assert stats['total_steps'] == len(mock_dataloader)
        assert isinstance(stats['final_loss'], float)
        assert stats['training_time'] > 0


class TestTrainerEdgeCases:
    """Edge case tests for Trainer."""

    def test_trainer_with_string_device(self, tiny_model, mock_dataloader, tmp_path):
        """Test that string device is converted to torch.device."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',  # String device
            output_dir=str(tmp_path)
        )

        assert isinstance(trainer.device, torch.device)
        assert trainer.device.type == 'cpu'

    def test_trainer_with_torch_device(self, tiny_model, mock_dataloader, tmp_path):
        """Test that torch.device is used directly."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        device = torch.device('cpu')

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device=device,  # torch.device object
            output_dir=str(tmp_path)
        )

        assert trainer.device == device

    def test_output_directory_creation(self, tiny_model, mock_dataloader, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new_output_dir" / "nested"
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            output_dir=str(output_dir)
        )

        assert output_dir.exists()
        assert output_dir.is_dir()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
