#!/usr/bin/env python3
"""
Convert HuggingFace Baguettotron weights to this implementation.

This script maps parameter names from HuggingFace's LlamaForCausalLM format
to the Baguettotron implementation in this repository.

Usage:
    python scripts/convert_hf_weights.py --hf-model PleIAs/Baguettotron --output model.pt

    # Or from local safetensors:
    python scripts/convert_hf_weights.py --hf-weights path/to/model.safetensors --output model.pt
"""

import argparse
import torch
from pathlib import Path
from typing import Dict, Optional
from safetensors.torch import load_file as load_safetensors

# Parameter name mapping: HF → Our implementation
PARAM_NAME_MAPPING = {
    # Embeddings
    "model.embed_tokens.weight": "embeddings.weight",

    # Decoder layers (pattern for all 80 layers)
    # Attention
    "model.layers.{i}.self_attn.q_proj.weight": "decoder.layers.{i}.attention.q_proj.weight",
    "model.layers.{i}.self_attn.k_proj.weight": "decoder.layers.{i}.attention.k_proj.weight",
    "model.layers.{i}.self_attn.v_proj.weight": "decoder.layers.{i}.attention.v_proj.weight",
    "model.layers.{i}.self_attn.o_proj.weight": "decoder.layers.{i}.attention.o_proj.weight",

    # Attention norms
    "model.layers.{i}.input_layernorm.weight": "decoder.layers.{i}.attention_norm.weight",

    # MLP (SwiGLU)
    "model.layers.{i}.mlp.gate_proj.weight": "decoder.layers.{i}.feed_forward.gate_proj.weight",
    "model.layers.{i}.mlp.up_proj.weight": "decoder.layers.{i}.feed_forward.up_proj.weight",
    "model.layers.{i}.mlp.down_proj.weight": "decoder.layers.{i}.feed_forward.down_proj.weight",

    # MLP norms
    "model.layers.{i}.post_attention_layernorm.weight": "decoder.layers.{i}.ffn_norm.weight",

    # Final norm
    "model.norm.weight": "norm.weight",

    # LM head (may be tied with embeddings)
    "lm_head.weight": "lm_head.weight",
}


def create_parameter_mapping(num_layers: int = 80) -> Dict[str, str]:
    """
    Create full parameter name mapping for all layers.

    Args:
        num_layers: Number of transformer layers (80 for Baguettotron-321M)

    Returns:
        Dictionary mapping HF param names to our param names
    """
    mapping = {}

    # Non-layer parameters
    for hf_name, our_name in PARAM_NAME_MAPPING.items():
        if "{i}" not in hf_name:
            mapping[hf_name] = our_name

    # Layer parameters
    for i in range(num_layers):
        for hf_pattern, our_pattern in PARAM_NAME_MAPPING.items():
            if "{i}" in hf_pattern:
                hf_name = hf_pattern.format(i=i)
                our_name = our_pattern.format(i=i)
                mapping[hf_name] = our_name

    return mapping


def convert_state_dict(
    hf_state_dict: Dict[str, torch.Tensor],
    num_layers: int = 80,
    tie_word_embeddings: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Convert HuggingFace state dict to our format.

    Args:
        hf_state_dict: State dict from HuggingFace model
        num_layers: Number of transformer layers
        tie_word_embeddings: Whether embeddings are tied

    Returns:
        State dict compatible with BaguettotronForCausalLM
    """
    mapping = create_parameter_mapping(num_layers)
    converted = {}

    # Track conversion
    converted_params = set()
    missing_params = []

    # Convert parameters
    for hf_name, tensor in hf_state_dict.items():
        if hf_name in mapping:
            our_name = mapping[hf_name]
            converted[our_name] = tensor.clone()
            converted_params.add(hf_name)
        else:
            # Skip RoPE frequencies (computed on the fly)
            if "rotary_emb" not in hf_name:
                missing_params.append(hf_name)

    # Check for tied embeddings
    if tie_word_embeddings:
        if "lm_head.weight" not in converted:
            # Copy from embeddings
            converted["lm_head.weight"] = converted["embeddings.weight"]
            print("ℹ️  Tied lm_head.weight with embeddings.weight")

    # Report
    print(f"\n✅ Converted {len(converted)} parameters")
    print(f"   - HF params: {len(hf_state_dict)}")
    print(f"   - Converted: {len(converted_params)}")
    print(f"   - Our params: {len(converted)}")

    if missing_params:
        print(f"\n⚠️  Skipped {len(missing_params)} HF parameters:")
        for name in missing_params[:5]:
            print(f"   - {name}")
        if len(missing_params) > 5:
            print(f"   - ... and {len(missing_params) - 5} more")

    return converted


def verify_conversion(
    our_state_dict: Dict[str, torch.Tensor],
    expected_num_layers: int = 80,
) -> bool:
    """
    Verify that the converted state dict has all expected parameters.

    Args:
        our_state_dict: Converted state dict
        expected_num_layers: Expected number of layers

    Returns:
        True if verification passes
    """
    # Expected parameter patterns
    expected_patterns = [
        "embeddings.weight",
        "norm.weight",
        "lm_head.weight",
    ]

    # Layer parameters
    for i in range(expected_num_layers):
        expected_patterns.extend([
            f"decoder.layers.{i}.attention.q_proj.weight",
            f"decoder.layers.{i}.attention.k_proj.weight",
            f"decoder.layers.{i}.attention.v_proj.weight",
            f"decoder.layers.{i}.attention.o_proj.weight",
            f"decoder.layers.{i}.attention_norm.weight",
            f"decoder.layers.{i}.feed_forward.gate_proj.weight",
            f"decoder.layers.{i}.feed_forward.up_proj.weight",
            f"decoder.layers.{i}.feed_forward.down_proj.weight",
            f"decoder.layers.{i}.ffn_norm.weight",
        ])

    # Check for missing parameters
    missing = []
    for pattern in expected_patterns:
        if pattern not in our_state_dict:
            missing.append(pattern)

    if missing:
        print(f"\n❌ Verification failed: {len(missing)} missing parameters")
        for name in missing[:5]:
            print(f"   - {name}")
        if len(missing) > 5:
            print(f"   - ... and {len(missing) - 5} more")
        return False

    print(f"\n✅ Verification passed: All {len(expected_patterns)} expected parameters present")
    return True


def load_hf_weights(
    model_name_or_path: str,
    use_safetensors: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Load weights from HuggingFace model or local file.

    Args:
        model_name_or_path: HF model ID or path to weights file
        use_safetensors: Whether to prefer safetensors format

    Returns:
        State dict from HuggingFace model
    """
    from pathlib import Path

    path = Path(model_name_or_path)

    # Local file
    if path.exists():
        if path.suffix == ".safetensors":
            print(f"Loading from safetensors: {path}")
            return load_safetensors(str(path))
        else:
            print(f"Loading from PyTorch: {path}")
            return torch.load(str(path), map_location="cpu")

    # HuggingFace model
    try:
        from transformers import AutoModelForCausalLM

        print(f"Loading HuggingFace model: {model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        return model.state_dict()
    except Exception as e:
        raise ValueError(
            f"Could not load weights from '{model_name_or_path}'. "
            f"Make sure it's either a HuggingFace model ID or a valid file path. "
            f"Error: {e}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Baguettotron weights to this implementation"
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default="PleIAs/Baguettotron",
        help="HuggingFace model ID or path to weights file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="baguettotron_321m.pt",
        help="Output path for converted weights"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=80,
        help="Number of transformer layers (80 for Baguettotron-321M)"
    )
    parser.add_argument(
        "--tie-embeddings",
        action="store_true",
        default=True,
        help="Whether to tie word embeddings"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify converted weights"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("HuggingFace → Baguettotron Weight Converter")
    print("=" * 80)

    # Load HF weights
    print(f"\n📥 Loading weights from: {args.hf_model}")
    hf_state_dict = load_hf_weights(args.hf_model)

    # Convert
    print(f"\n🔄 Converting {len(hf_state_dict)} parameters...")
    our_state_dict = convert_state_dict(
        hf_state_dict,
        num_layers=args.num_layers,
        tie_word_embeddings=args.tie_embeddings,
    )

    # Verify
    if args.verify:
        print("\n🔍 Verifying conversion...")
        if not verify_conversion(our_state_dict, args.num_layers):
            print("\n⚠️  Warning: Verification failed, but continuing anyway")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving converted weights to: {output_path}")
    torch.save(our_state_dict, output_path)

    # Report file size
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"   - File size: {size_mb:.1f} MB")

    print("\n✅ Conversion complete!")
    print(f"\nTo load in your model:")
    print(f"```python")
    print(f"from baguettotron import BaguettotronForCausalLM, BaguettotronConfig")
    print(f"config = BaguettotronConfig.baguettotron_321m()")
    print(f"model = BaguettotronForCausalLM(config)")
    print(f"state_dict = torch.load('{args.output}')")
    print(f"model.load_state_dict(state_dict)")
    print(f"```")


if __name__ == "__main__":
    main()
