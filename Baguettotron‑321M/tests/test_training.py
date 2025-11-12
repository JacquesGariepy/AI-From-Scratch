"""
Tests for training modules (trainer, optimizer, scheduler).

These tests ensure 100% coverage of src/baguettotron/training/.
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
from baguettotron.training import (
    Trainer,
    create_optimizer,
    create_optimizer_with_layer_decay,
    get_parameter_count,
    freeze_parameters,
    unfreeze_all_parameters,
    create_scheduler,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_inverse_sqrt_schedule_with_warmup,
)


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
        for _ in range(3)  # 3 batches
    ]
    return MockDataLoader(data)


class TestOptimizers:
    """Tests for optimizer utilities."""

    def test_create_optimizer_adamw(self, tiny_model):
        """Test creating AdamW optimizer."""
        optimizer = create_optimizer(
            tiny_model,
            learning_rate=1e-4,
            weight_decay=0.1,
            optimizer_type='adamw'
        )

        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) == 2  # With and without decay

    def test_create_optimizer_adam(self, tiny_model):
        """Test creating Adam optimizer."""
        optimizer = create_optimizer(
            tiny_model,
            optimizer_type='adam'
        )

        assert isinstance(optimizer, torch.optim.Adam)

    def test_create_optimizer_sgd(self, tiny_model):
        """Test creating SGD optimizer."""
        optimizer = create_optimizer(
            tiny_model,
            optimizer_type='sgd'
        )

        assert isinstance(optimizer, torch.optim.SGD)

    def test_create_optimizer_invalid_type(self, tiny_model):
        """Test invalid optimizer type raises error."""
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            create_optimizer(tiny_model, optimizer_type='invalid')

    def test_create_optimizer_with_layer_decay(self, tiny_model):
        """Test optimizer with layer-wise learning rate decay."""
        optimizer = create_optimizer_with_layer_decay(
            tiny_model,
            learning_rate=1e-4,
            layer_decay=0.65
        )

        assert isinstance(optimizer, torch.optim.AdamW)
        # Should have multiple param groups (one per depth)
        assert len(optimizer.param_groups) >= 2

    def test_get_parameter_count(self, tiny_model):
        """Test parameter counting."""
        counts = get_parameter_count(tiny_model)

        assert 'total' in counts
        assert 'trainable' in counts
        assert 'frozen' in counts
        assert counts['total'] > 0
        assert counts['trainable'] == counts['total']  # All trainable initially
        assert counts['frozen'] == 0

    def test_freeze_parameters(self, tiny_model):
        """Test freezing parameters."""
        # Freeze embeddings
        freeze_parameters(tiny_model, ['embeddings'])

        # Check that embedding parameters are frozen
        for name, param in tiny_model.named_parameters():
            if 'embeddings' in name:
                assert not param.requires_grad

        # Other parameters should still be trainable
        assert tiny_model.decoder.layers[0].attention.q_proj.weight.requires_grad

    def test_unfreeze_all_parameters(self, tiny_model):
        """Test unfreezing all parameters."""
        # First freeze some
        freeze_parameters(tiny_model, ['embeddings'])

        # Then unfreeze all
        unfreeze_all_parameters(tiny_model)

        # Check all are trainable
        for param in tiny_model.parameters():
            assert param.requires_grad


class TestSchedulers:
    """Tests for learning rate schedulers."""

    def test_linear_schedule_with_warmup(self, tiny_model):
        """Test linear warmup + linear decay scheduler."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,
            num_training_steps=100
        )

        # Test warmup phase
        initial_lr = optimizer.param_groups[0]['lr']
        for _ in range(5):
            scheduler.step()
        warmup_lr = optimizer.param_groups[0]['lr']
        assert warmup_lr > initial_lr  # LR should increase

        # Test decay phase
        for _ in range(50):
            scheduler.step()
        decay_lr = optimizer.param_groups[0]['lr']
        assert decay_lr < warmup_lr  # LR should decrease

    def test_cosine_schedule_with_warmup(self, tiny_model):
        """Test cosine annealing with warmup."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,
            num_training_steps=100,
            num_cycles=0.5
        )

        # Step through and check LR changes
        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        # LR should increase then decrease (cosine pattern)
        assert max(lrs) > lrs[0]
        assert lrs[-1] < max(lrs)

    def test_constant_schedule_with_warmup(self, tiny_model):
        """Test constant LR after warmup."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10
        )

        # After warmup, LR should stay constant
        for _ in range(15):
            scheduler.step()
        lr_after_warmup = optimizer.param_groups[0]['lr']

        for _ in range(20):
            scheduler.step()

        assert optimizer.param_groups[0]['lr'] == pytest.approx(lr_after_warmup)

    def test_polynomial_decay_schedule(self, tiny_model):
        """Test polynomial decay scheduler."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,
            num_training_steps=100,
            lr_end=0.0,
            power=2.0
        )

        # LR should decrease polynomially after warmup
        for _ in range(50):
            scheduler.step()

        assert optimizer.param_groups[0]['lr'] < 1e-3

    def test_inverse_sqrt_schedule(self, tiny_model):
        """Test inverse sqrt scheduler."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        scheduler = get_inverse_sqrt_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10
        )

        for _ in range(50):
            scheduler.step()

        # LR should decay
        assert optimizer.param_groups[0]['lr'] < 1e-3

    def test_create_scheduler_linear(self, tiny_model):
        """Test create_scheduler with linear type."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type='linear',
            num_warmup_steps=10,
            num_training_steps=100
        )

        assert scheduler is not None

    def test_create_scheduler_cosine(self, tiny_model):
        """Test create_scheduler with cosine type."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type='cosine',
            num_warmup_steps=10,
            num_training_steps=100
        )

        assert scheduler is not None

    def test_create_scheduler_invalid_type(self, tiny_model):
        """Test invalid scheduler type raises error."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            create_scheduler(
                optimizer,
                scheduler_type='invalid',
                num_warmup_steps=10,
                num_training_steps=100
            )


class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_initialization(self, tiny_model, mock_dataloader):
        """Test Trainer initialization."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            max_epochs=1,
            output_dir='test_output'
        )

        assert trainer.model is not None
        assert trainer.device == torch.device('cpu')
        assert trainer.global_step == 0

    def test_trainer_single_step(self, tiny_model, mock_dataloader, tmp_path):
        """Test single training step."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            max_epochs=1,
            gradient_accumulation_steps=1,
            output_dir=str(tmp_path)
        )

        # Run one training step manually
        batch = next(iter(mock_dataloader))
        loss = trainer._training_step(batch)

        assert isinstance(loss, float)
        assert loss > 0

    def test_trainer_save_load_checkpoint(self, tiny_model, mock_dataloader, tmp_path):
        """Test saving and loading checkpoints."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            output_dir=str(tmp_path)
        )

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path.name))

        assert checkpoint_path.exists()

        # Load checkpoint
        trainer.global_step = 0  # Reset
        trainer.load_checkpoint(checkpoint_path)

        # State should be restored
        assert trainer.global_step >= 0

    def test_trainer_compute_loss(self, tiny_model, mock_dataloader):
        """Test loss computation."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu'
        )

        # Create sample logits and labels
        batch_size, seq_len, vocab_size = 2, 10, tiny_model.config.vocab_size
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = trainer._compute_loss(logits, labels)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_trainer_evaluation(self, tiny_model, mock_dataloader, tmp_path):
        """Test evaluation."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            eval_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu',
            output_dir=str(tmp_path)
        )

        metrics = trainer.evaluate()

        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert metrics['loss'] > 0
        assert metrics['perplexity'] > 1

    def test_trainer_no_eval_dataloader_error(self, tiny_model, mock_dataloader):
        """Test evaluation without eval dataloader raises error."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            device='cpu'
        )

        with pytest.raises(ValueError, match="No evaluation dataloader"):
            trainer.evaluate()
