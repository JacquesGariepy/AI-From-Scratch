#!/usr/bin/env python3
"""
Verify 100% alignment between our BaguettotronConfig and official HuggingFace model.

This script compares our implementation against the official PleIAs/Baguettotron config
and identifies any mismatches in parameters, values, or structure.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baguettotron.config import BaguettotronConfig

try:
    from transformers import AutoConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  transformers not available, using cached official config")


def get_official_config() -> Dict[str, Any]:
    """Fetch official Baguettotron config from HuggingFace."""
    if HF_AVAILABLE:
        try:
            config = AutoConfig.from_pretrained('PleIAs/Baguettotron')
            return config.to_dict()
        except Exception as e:
            print(f"⚠️  Failed to fetch from HF: {e}")

    # Fallback to hardcoded official config
    return {
        "vocab_size": 65536,
        "max_position_embeddings": 4096,
        "hidden_size": 576,
        "intermediate_size": 1536,
        "num_hidden_layers": 80,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-05,
        "use_cache": True,
        "rope_theta": 10000.0,
        "attention_dropout": 0.0,
        "tie_word_embeddings": True,
        "bos_token_id": 1,
        "pad_token_id": None,
        "eos_token_id": 2,
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        # Additional HF-specific params (not critical for model architecture)
        "pretraining_tp": 1,
        "rope_scaling": None,
        "attention_bias": False,
        "mlp_bias": False,
        "head_dim": 64,
    }


def get_our_config() -> Dict[str, Any]:
    """Get our Baguettotron-321M config."""
    config = BaguettotronConfig.baguettotron_321m()
    return config.to_dict()


def normalize_value(value: Any) -> Any:
    """Normalize values for comparison."""
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 10)  # Handle floating point precision
    if isinstance(value, list):
        return tuple(sorted(str(v) for v in value))
    return value


def compare_configs(official: Dict[str, Any], ours: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    """
    Compare official and our configs.

    Returns:
        Tuple of (mismatches, missing_in_ours, extra_in_ours)
    """
    # Core architecture parameters that MUST match
    CRITICAL_PARAMS = {
        'vocab_size',
        'hidden_size',
        'num_hidden_layers',
        'num_attention_heads',
        'num_key_value_heads',
        'intermediate_size',
        'max_position_embeddings',
        'rope_theta',
        'rms_norm_eps',
        'tie_word_embeddings',
        'hidden_act',
        'attention_dropout',
        'use_cache',
        'initializer_range',
        'bos_token_id',
        'eos_token_id',
        'pad_token_id',
        'model_type',
        'architectures',
    }

    # Parameters that are HF-specific and not critical for model architecture
    OPTIONAL_PARAMS = {
        'pretraining_tp', 'rope_scaling', 'attention_bias', 'mlp_bias',
        'head_dim', 'return_dict', 'output_hidden_states', 'torchscript',
        'dtype', 'pruned_heads', 'chunk_size_feed_forward', 'is_encoder_decoder',
        'is_decoder', 'cross_attention_hidden_size', 'add_cross_attention',
        'tie_encoder_decoder', 'finetuning_task', 'id2label', 'label2id',
        'task_specific_params', 'problem_type', 'tokenizer_class', 'prefix',
        'sep_token_id', 'decoder_start_token_id', 'max_length', 'min_length',
        'do_sample', 'early_stopping', 'num_beams', 'temperature', 'top_k',
        'top_p', 'typical_p', 'repetition_penalty', 'length_penalty',
        'no_repeat_ngram_size', 'encoder_no_repeat_ngram_size', 'bad_words_ids',
        'num_return_sequences', 'output_scores', 'return_dict_in_generate',
        'forced_bos_token_id', 'forced_eos_token_id', 'remove_invalid_values',
        'exponential_decay_length_penalty', 'suppress_tokens', 'begin_suppress_tokens',
        'num_beam_groups', 'diversity_penalty', '_name_or_path', 'transformers_version',
        'tf_legacy_loss', 'use_bfloat16', 'output_attentions',
    }

    mismatches = []
    missing_in_ours = []
    extra_in_ours = []

    # Map hidden_act -> hidden_activation
    param_mapping = {
        'hidden_act': 'hidden_activation',
    }

    # Check critical parameters
    for param in CRITICAL_PARAMS:
        our_param = param_mapping.get(param, param)

        if param in official and our_param in ours:
            official_val = normalize_value(official[param])
            our_val = normalize_value(ours[our_param])

            if official_val != our_val:
                mismatches.append(
                    f"  ❌ {param}: official={official[param]} vs ours={ours[our_param]}"
                )
        elif param in official and our_param not in ours:
            if param not in OPTIONAL_PARAMS:
                missing_in_ours.append(
                    f"  ⚠️  {param}: missing in our config (value={official[param]})"
                )
        elif param not in official and our_param in ours:
            extra_in_ours.append(
                f"  ℹ️  {our_param}: extra in our config (value={ours[our_param]})"
            )

    return mismatches, missing_in_ours, extra_in_ours


def verify_parameter_count(config: BaguettotronConfig) -> Tuple[int, bool]:
    """Verify parameter count is approximately 321M."""
    params = config.approximate_params()
    params_m = params / 1e6

    # Should be approximately 321M (within 5% tolerance)
    expected = 321e6
    tolerance = 0.05
    is_correct = abs(params - expected) / expected < tolerance

    return params, is_correct


def main():
    """Run verification."""
    print("=" * 80)
    print("Baguettotron Config Alignment Verification")
    print("=" * 80)
    print()

    # Get configs
    print("📥 Fetching official config from HuggingFace...")
    official = get_official_config()

    print("📥 Loading our Baguettotron-321M config...")
    ours = get_our_config()

    # Compare
    print("\n🔍 Comparing configurations...\n")
    mismatches, missing, extra = compare_configs(official, ours)

    # Report results
    all_good = True

    if mismatches:
        all_good = False
        print("🚨 VALUE MISMATCHES (CRITICAL):")
        for mismatch in mismatches:
            print(mismatch)
        print()

    if missing:
        all_good = False
        print("⚠️  MISSING PARAMETERS:")
        for miss in missing:
            print(miss)
        print()

    if extra:
        print("ℹ️  EXTRA PARAMETERS (may be intentional):")
        for ext in extra:
            print(ext)
        print()

    # Verify parameter count
    print("🔢 Parameter Count Verification:")
    config = BaguettotronConfig.baguettotron_321m()
    params, is_correct = verify_parameter_count(config)
    params_m = params / 1e6

    status = "✅" if is_correct else "❌"
    print(f"  {status} Approximate parameters: {params_m:.1f}M")
    print(f"     Expected: ~321M")
    print(f"     Difference: {abs(params_m - 321):.1f}M")

    if not is_correct:
        all_good = False

    print()

    # Final verdict
    print("=" * 80)
    if all_good and not mismatches:
        print("✅ SUCCESS: 100% alignment achieved!")
        print("   All critical parameters match the official HuggingFace model.")
        return 0
    else:
        print("❌ FAILED: Alignment issues detected!")
        print("   Please review mismatches above and update config.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
