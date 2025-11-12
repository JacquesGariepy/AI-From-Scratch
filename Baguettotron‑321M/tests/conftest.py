"""
Pytest configuration and fixtures for Baguettotron tests.
"""

# Ensure project root is on sys.path so `import model` works under pytest importmode=prepend
import sys
import os
import pathlib
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

# Add src to path
sys.path.insert(0, str(ROOT / "src"))

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="argparse")

from baguettotron.config import BaguettotronConfig
from baguettotron.model import BaguettotronForCausalLM


@pytest.fixture
def tiny_config():
    """Fixture for tiny test configuration."""
    return BaguettotronConfig(
        vocab_size=512,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=128,
    )


@pytest.fixture
def small_config():
    """Fixture for small test configuration."""
    return BaguettotronConfig(
        vocab_size=1024,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512,
    )


@pytest.fixture
def official_config():
    """Fixture for official 321M configuration."""
    return BaguettotronConfig.baguettotron_321m()


@pytest.fixture
def tiny_model(tiny_config):
    """Fixture for tiny test model."""
    model = BaguettotronForCausalLM(tiny_config)
    model.eval()
    return model


@pytest.fixture
def small_model(small_config):
    """Fixture for small test model."""
    model = BaguettotronForCausalLM(small_config)
    model.eval()
    return model


@pytest.fixture
def sample_input_ids(tiny_config):
    """Fixture for sample input tensor."""
    batch_size, seq_len = 2, 10
    return torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))


@pytest.fixture
def device():
    """Fixture for device selection."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
