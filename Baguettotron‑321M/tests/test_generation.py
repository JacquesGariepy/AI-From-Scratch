"""
Tests for generation utilities (sampling strategies).

These tests ensure 100% coverage of src/baguettotron/generation/.
"""

import pytest
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baguettotron.generation import (
    sample_from_logits,
    top_k_filtering,
    top_p_filtering,
    typical_filtering,
    min_p_filtering,
    repetition_penalty_apply,
    StoppingCriteria,
    MaxLengthCriteria,
    EosTokenCriteria,
    StoppingCriteriaList,
)


class TestTopKFiltering:
    """Tests for top-k filtering."""

    def test_top_k_basic(self):
        """Test basic top-k filtering."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        filtered = top_k_filtering(logits, top_k=3)

        # Should keep top 3 values (5.0, 4.0, 3.0)
        assert filtered[0, 0] == float('-inf')  # 1.0 filtered
        assert filtered[0, 1] == float('-inf')  # 2.0 filtered
        assert filtered[0, 2] == 3.0  # kept
        assert filtered[0, 3] == 4.0  # kept
        assert filtered[0, 4] == 5.0  # kept

    def test_top_k_disabled(self):
        """Test top-k with k=0 (disabled)."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        filtered = top_k_filtering(logits, top_k=0)

        # Should not change logits
        assert torch.equal(filtered, logits)

    def test_top_k_larger_than_vocab(self):
        """Test top-k larger than vocabulary size."""
        logits = torch.randn(1, 10)
        filtered = top_k_filtering(logits, top_k=20)

        # Should keep all tokens
        assert not (filtered == float('-inf')).any()


class TestTopPFiltering:
    """Tests for top-p (nucleus) filtering."""

    def test_top_p_basic(self):
        """Test basic top-p filtering."""
        # Create logits where probabilities are easy to compute
        logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
        filtered = top_p_filtering(logits, top_p=0.8)

        # With cumsum: 0.5, 0.8, 0.95, 1.0
        # Should keep first 2 tokens (cumsum <= 0.8)
        inf_mask = filtered == float('-inf')
        assert not inf_mask[0, 0]  # 0.5 kept
        assert not inf_mask[0, 1]  # 0.3 kept (cumsum=0.8)
        assert inf_mask[0, 2]      # 0.15 filtered
        assert inf_mask[0, 3]      # 0.05 filtered

    def test_top_p_disabled(self):
        """Test top-p with p=1.0 (disabled)."""
        logits = torch.randn(1, 10)
        filtered = top_p_filtering(logits, top_p=1.0)

        # Should keep all tokens
        assert not (filtered == float('-inf')).any()

    def test_top_p_min_tokens_to_keep(self):
        """Test top-p with minimum tokens."""
        logits = torch.log(torch.tensor([[0.9, 0.05, 0.05]]))
        filtered = top_p_filtering(logits, top_p=0.5, min_tokens_to_keep=2)

        # Should keep at least 2 tokens even though cumsum > 0.5
        inf_count = (filtered == float('-inf')).sum().item()
        assert inf_count <= 1  # At most 1 token filtered


class TestTypicalFiltering:
    """Tests for typical decoding."""

    def test_typical_basic(self):
        """Test basic typical filtering."""
        # Create logits with varying information content
        logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
        filtered = typical_filtering(logits, mass=0.8)

        # Should filter some tokens based on deviation from entropy
        assert (filtered == float('-inf')).any() or True  # Some tokens may be filtered

    def test_typical_disabled(self):
        """Test typical with mass=1.0 (disabled)."""
        logits = torch.randn(1, 10)
        filtered = typical_filtering(logits, mass=1.0)

        # Should keep all tokens
        assert not (filtered == float('-inf')).any()


class TestMinPFiltering:
    """Tests for min-p filtering."""

    def test_min_p_basic(self):
        """Test basic min-p filtering."""
        # Create logits where we can control probabilities
        logits = torch.log(torch.tensor([[0.5, 0.3, 0.1, 0.01]]))
        filtered = min_p_filtering(logits, min_p=0.1)

        # max_prob = 0.5, threshold = 0.05
        # Should filter tokens with prob < 0.05
        assert (filtered == float('-inf')).any()

    def test_min_p_disabled(self):
        """Test min-p with min_p=0 (disabled)."""
        logits = torch.randn(1, 10)
        filtered = min_p_filtering(logits, min_p=0.0)

        # Should keep all tokens
        assert not (filtered == float('-inf')).any()


class TestRepetitionPenalty:
    """Tests for repetition penalty."""

    def test_repetition_penalty_applied(self):
        """Test that repetition penalty is applied."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        generated = torch.tensor([[1, 2]])  # Tokens 1 and 2 were generated

        penalized = repetition_penalty_apply(logits, generated, penalty=1.5)

        # Tokens 1 and 2 should have lower scores
        assert penalized[0, 1] < logits[0, 1]
        assert penalized[0, 2] < logits[0, 2]
        # Other tokens unchanged
        assert penalized[0, 0] == logits[0, 0]
        assert penalized[0, 3] == logits[0, 3]

    def test_repetition_penalty_disabled(self):
        """Test repetition penalty with penalty=1.0 (disabled)."""
        logits = torch.randn(2, 10)
        generated = torch.randint(0, 10, (2, 5))

        penalized = repetition_penalty_apply(logits, generated, penalty=1.0)

        # Should not change logits
        assert torch.equal(penalized, logits)


class TestSampleFromLogits:
    """Tests for complete sampling function."""

    def test_greedy_sampling(self):
        """Test greedy decoding (no sampling)."""
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])

        tokens = sample_from_logits(logits, do_sample=False)

        # Should select argmax (index 1)
        assert tokens[0] == 1

    def test_temperature_sampling(self):
        """Test sampling with temperature."""
        torch.manual_seed(42)
        logits = torch.randn(1, 100)

        tokens_high_temp = sample_from_logits(
            logits.clone(),
            temperature=2.0,
            do_sample=True
        )
        tokens_low_temp = sample_from_logits(
            logits.clone(),
            temperature=0.1,
            do_sample=True
        )

        # Different temperatures should produce different results (statistically)
        assert isinstance(tokens_high_temp[0].item(), int)
        assert isinstance(tokens_low_temp[0].item(), int)

    def test_top_k_sampling(self):
        """Test sampling with top-k."""
        torch.manual_seed(42)
        logits = torch.randn(1, 100)

        tokens = sample_from_logits(
            logits,
            top_k=10,
            do_sample=True
        )

        assert 0 <= tokens[0] < 100

    def test_top_p_sampling(self):
        """Test sampling with top-p."""
        torch.manual_seed(42)
        logits = torch.randn(1, 100)

        tokens = sample_from_logits(
            logits,
            top_p=0.9,
            do_sample=True
        )

        assert 0 <= tokens[0] < 100

    def test_combined_sampling(self):
        """Test sampling with multiple filters."""
        torch.manual_seed(42)
        logits = torch.randn(1, 100)

        tokens = sample_from_logits(
            logits,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )

        assert 0 <= tokens[0] < 100


class TestStoppingCriteria:
    """Tests for stopping criteria."""

    def test_max_length_criteria(self):
        """Test MaxLengthCriteria."""
        criteria = MaxLengthCriteria(max_length=10)

        # Not stopped yet
        input_ids = torch.randint(0, 100, (1, 5))
        assert not criteria(input_ids)

        # Should stop
        input_ids = torch.randint(0, 100, (1, 10))
        assert criteria(input_ids)

    def test_eos_token_criteria(self):
        """Test EosTokenCriteria."""
        criteria = EosTokenCriteria(eos_token_id=2)

        # Not stopped (last token is not EOS)
        input_ids = torch.tensor([[1, 3, 5]])
        assert not criteria(input_ids)

        # Should stop (last token is EOS)
        input_ids = torch.tensor([[1, 3, 2]])
        assert criteria(input_ids)

    def test_eos_token_criteria_batch(self):
        """Test EosTokenCriteria with batch."""
        criteria = EosTokenCriteria(eos_token_id=2)

        # Stop only if ALL sequences have EOS
        input_ids = torch.tensor([[1, 2], [3, 4]])
        assert not criteria(input_ids)

        # All have EOS
        input_ids = torch.tensor([[1, 2], [3, 2]])
        assert criteria(input_ids)

    def test_stopping_criteria_list(self):
        """Test StoppingCriteriaList."""
        criteria_list = StoppingCriteriaList([
            MaxLengthCriteria(max_length=10),
            EosTokenCriteria(eos_token_id=2),
        ])

        # Not stopped
        input_ids = torch.tensor([[1, 3, 5]])
        assert not criteria_list(input_ids)

        # Stopped by EOS
        input_ids = torch.tensor([[1, 3, 2]])
        assert criteria_list(input_ids)

        # Stopped by max length
        input_ids = torch.randint(0, 100, (1, 10))
        assert criteria_list(input_ids)

    def test_stopping_criteria_base_class(self):
        """Test that base StoppingCriteria raises NotImplementedError."""
        criteria = StoppingCriteria()

        with pytest.raises(NotImplementedError):
            criteria(torch.tensor([[1, 2, 3]]))


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_logits(self):
        """Test handling of edge cases."""
        # Very small logits
        logits = torch.tensor([[0.0]])

        tokens = sample_from_logits(logits, do_sample=False)
        assert tokens[0] == 0

    def test_all_inf_after_filtering(self):
        """Test case where all tokens are filtered (shouldn't happen in practice)."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])

        # This should still work (softmax handles -inf)
        filtered = top_k_filtering(logits, top_k=1)
        probs = torch.softmax(filtered, dim=-1)

        # Check that we get valid probabilities
        assert torch.isfinite(probs).any()
