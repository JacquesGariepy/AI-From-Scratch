#!/usr/bin/env python3
"""
Text generation script for Baguettotron models.

This script provides a command-line interface for generating text
from trained Baguettotron models.

Usage:
    python generate.py --checkpoint model.pt --prompt "Hello, world"
    python generate.py --checkpoint model.pt --interactive
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from baguettotron import BaguettotronForCausalLM, BaguettotronConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text with Baguettotron",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='321m',
        choices=['tiny', '321m'],
        help='Model configuration (should match checkpoint)',
    )

    # Generation arguments
    parser.add_argument(
        '--prompt',
        type=str,
        help='Prompt text (for non-interactive mode)',
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=100,
        help='Maximum number of tokens to generate',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (higher = more random)',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        help='Top-k sampling (keep only top k tokens)',
    )
    parser.add_argument(
        '--top-p',
        type=float,
        help='Nucleus sampling (keep tokens with cumulative prob >= top_p)',
    )
    parser.add_argument(
        '--do-sample',
        action='store_true',
        default=True,
        help='Use sampling (otherwise greedy decoding)',
    )
    parser.add_argument(
        '--greedy',
        action='store_true',
        help='Use greedy decoding (disables sampling)',
    )

    # Interactive mode
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode',
    )

    # Hardware
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on',
    )

    # Other
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )

    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    config_name: str,
    device: str,
) -> BaguettotronForCausalLM:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config_name: Name of model configuration
        device: Device to load model on

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Create model config
    if config_name == '321m':
        config = BaguettotronConfig.baguettotron_321m()
    else:
        config = BaguettotronConfig()  # Tiny config

    # Create model
    model = BaguettotronForCausalLM(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded with {model.count_parameters() / 1e6:.1f}M parameters")

    return model


def encode_prompt(prompt: str, vocab_size: int) -> torch.Tensor:
    """
    Encode prompt text to token IDs.

    Note: This is a placeholder. In production, use a proper tokenizer.

    Args:
        prompt: Prompt text
        vocab_size: Size of vocabulary

    Returns:
        Token IDs tensor of shape (1, seq_len)
    """
    # For now, use simple character encoding as placeholder
    # In production, use: tokenizer.encode(prompt, return_tensors='pt')

    # Simple placeholder: hash characters to vocab
    token_ids = [hash(c) % vocab_size for c in prompt]

    # Ensure we have at least one token
    if not token_ids:
        token_ids = [0]

    return torch.tensor([token_ids], dtype=torch.long)


def decode_tokens(token_ids: torch.Tensor, vocab_size: int) -> str:
    """
    Decode token IDs to text.

    Note: This is a placeholder. In production, use a proper tokenizer.

    Args:
        token_ids: Token IDs tensor
        vocab_size: Size of vocabulary

    Returns:
        Decoded text
    """
    # For now, use simple character decoding as placeholder
    # In production, use: tokenizer.decode(token_ids)

    # Return placeholder representation
    return f"[Generated {len(token_ids[0])} tokens]"


def generate_text(
    model: BaguettotronForCausalLM,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    do_sample: bool,
    device: str,
) -> str:
    """
    Generate text from prompt.

    Args:
        model: Baguettotron model
        prompt: Prompt text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        do_sample: Whether to sample or use greedy
        device: Device to run on

    Returns:
        Generated text
    """
    # Encode prompt
    input_ids = encode_prompt(prompt, model.config.vocab_size).to(device)

    logger.info(f"Prompt tokens: {input_ids.shape[1]}")
    logger.info("Generating...")

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )

    # Decode
    generated_text = decode_tokens(output_ids, model.config.vocab_size)

    logger.info(f"Generated tokens: {output_ids.shape[1] - input_ids.shape[1]}")

    return generated_text


def interactive_mode(
    model: BaguettotronForCausalLM,
    args: argparse.Namespace,
):
    """
    Run interactive text generation.

    Args:
        model: Baguettotron model
        args: Command-line arguments
    """
    print("\n" + "=" * 80)
    print("Baguettotron Interactive Generation")
    print("=" * 80)
    print("Enter your prompts (press Ctrl+C or type 'quit' to exit)")
    print("=" * 80 + "\n")

    while True:
        try:
            # Get prompt
            prompt = input("\nPrompt: ").strip()

            if not prompt or prompt.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            # Generate
            output = generate_text(
                model=model,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=not args.greedy,
                device=args.device,
            )

            # Print output
            print("\nGenerated:")
            print("-" * 80)
            print(output)
            print("-" * 80)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)


def main():
    """Main generation function."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint,
        config_name=args.config,
        device=args.device,
    )

    # Interactive mode
    if args.interactive:
        interactive_mode(model, args)
        return

    # Single prompt mode
    if not args.prompt:
        logger.error("Either --prompt or --interactive must be specified")
        sys.exit(1)

    logger.info(f"Prompt: {args.prompt}")

    # Generate
    output = generate_text(
        model=model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
        device=args.device,
    )

    # Print output
    print("\n" + "=" * 80)
    print("Generated Text:")
    print("=" * 80)
    print(output)
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
