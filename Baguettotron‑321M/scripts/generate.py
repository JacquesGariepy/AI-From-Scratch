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
from baguettotron.tokenization import load_tokenizer

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

    # Tokenizer
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='auto',
        choices=['auto', 'baguettotron', 'gpt2', 'char'],
        help='Tokenizer to use (auto=try Baguettotron, fallback to GPT-2, then char-level)',
    )

    # Chat template
    parser.add_argument(
        '--chat-mode',
        action='store_true',
        help='Enable chat mode with conversation formatting',
    )
    parser.add_argument(
        '--chat-template',
        type=str,
        help='Path to chat_template.json (default: auto-detect from assets/)',
    )
    parser.add_argument(
        '--system-prompt',
        type=str,
        help='System prompt for chat mode',
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
) -> tuple[BaguettotronForCausalLM, int]:
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

    # Load checkpoint first
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Use config from checkpoint if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            logger.info("Using config from checkpoint")
        else:
            config = None
    else:
        state_dict = checkpoint
        config = None

    # If no config in checkpoint, infer from weights or use default
    if config is None:
        # Check if model was compiled (has _orig_mod. prefix)
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            # Remove _orig_mod. prefix from all keys
            logger.info("Detected torch.compile checkpoint, removing _orig_mod. prefix")
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # Infer dimensions from checkpoint weights
        vocab_size = state_dict['embeddings.weight'].shape[0]
        hidden_size = state_dict['embeddings.weight'].shape[1]

        # Count number of layers
        num_layers = 0
        while f'decoder.layers.{num_layers}.attention_norm.weight' in state_dict:
            num_layers += 1

        # Get intermediate size from feedforward
        intermediate_size = state_dict['decoder.layers.0.feed_forward.gate_proj.weight'].shape[0]

        # Get number of heads from attention projections
        q_proj_dim = state_dict['decoder.layers.0.attention.q_proj.weight'].shape[0]
        k_proj_dim = state_dict['decoder.layers.0.attention.k_proj.weight'].shape[0]

        # Infer head_dim and number of heads
        # For GQA: q_dim = num_attention_heads * head_dim, k_dim = num_key_value_heads * head_dim
        # Common head_dim values: 32, 64, 128
        for head_dim in [32, 64, 128]:
            if q_proj_dim % head_dim == 0 and k_proj_dim % head_dim == 0:
                num_attention_heads = q_proj_dim // head_dim
                num_key_value_heads = k_proj_dim // head_dim
                break
        else:
            # Fallback: assume head_dim = hidden_size
            num_attention_heads = 1
            num_key_value_heads = 1

        logger.info(f"Inferred config from checkpoint: vocab_size={vocab_size}, hidden_size={hidden_size}, "
                   f"num_layers={num_layers}, intermediate_size={intermediate_size}, "
                   f"num_attention_heads={num_attention_heads}, num_key_value_heads={num_key_value_heads}")

        # Create config with inferred dimensions
        config = BaguettotronConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
        )

    # Create model
    model = BaguettotronForCausalLM(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded with {model.count_parameters() / 1e6:.1f}M parameters")

    return model, config.vocab_size




def generate_text(
    model: BaguettotronForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    do_sample: bool,
    device: str,
    chat_mode: bool = False,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Generate text from prompt.

    Args:
        model: Baguettotron model
        tokenizer: Tokenizer instance
        prompt: Prompt text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        do_sample: Whether to sample or use greedy
        device: Device to run on
        chat_mode: Whether to use chat template formatting
        system_prompt: Optional system prompt for chat mode

    Returns:
        Generated text
    """
    # Format prompt based on mode
    if chat_mode and hasattr(tokenizer, 'apply_chat_template'):
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(device)

        logger.info(f"Chat mode enabled")
        logger.info(f"Messages: {messages}")
    else:
        # Regular text mode
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        logger.info(f"Prompt: {prompt}")

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
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    logger.info(f"Generated tokens: {output_ids.shape[1] - input_ids.shape[1]}")

    return generated_text


def interactive_mode(
    model: BaguettotronForCausalLM,
    tokenizer,
    args: argparse.Namespace,
):
    """
    Run interactive text generation.

    Args:
        model: Baguettotron model
        tokenizer: Tokenizer instance
        args: Command-line arguments
    """
    print("\n" + "=" * 80)
    if args.chat_mode:
        print("Baguettotron Chat Mode - Interactive Conversation")
    else:
        print("Baguettotron Interactive Generation")
    print("=" * 80)
    print(f"Tokenizer: {tokenizer}")
    if args.chat_mode:
        print("Chat mode: Enabled (using ChatML format)")
        if args.system_prompt:
            print(f"System prompt: {args.system_prompt}")
    print("Enter your prompts (press Ctrl+C or type 'quit' to exit)")
    print("=" * 80 + "\n")

    # Conversation history for chat mode
    conversation_history = []
    if args.system_prompt and args.chat_mode:
        conversation_history.append({"role": "system", "content": args.system_prompt})

    while True:
        try:
            # Get prompt
            prompt = input("\n> You: ").strip()

            if not prompt or prompt.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            # Generate
            output = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=not args.greedy,
                device=args.device,
                chat_mode=args.chat_mode,
                system_prompt=args.system_prompt if not conversation_history else None,
            )

            # Print output
            print("\n> Assistant:")
            print("-" * 80)
            print(output)
            print("-" * 80)

            # Update conversation history in chat mode
            if args.chat_mode:
                conversation_history.append({"role": "user", "content": prompt})
                conversation_history.append({"role": "assistant", "content": output})

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
    model, model_vocab_size = load_model(
        checkpoint_path=args.checkpoint,
        config_name=args.config,
        device=args.device,
    )

    # Load tokenizer with optional chat template
    logger.info(f"Loading tokenizer (type={args.tokenizer})")
    tokenizer = load_tokenizer(
        tokenizer_type=args.tokenizer,
        model_vocab_size=model_vocab_size,
        chat_template_path=args.chat_template if hasattr(args, 'chat_template') else None,
    )
    logger.info(f"Tokenizer loaded: {tokenizer}")

    # Warn if chat mode requested but no template loaded
    if args.chat_mode and not hasattr(tokenizer, 'chat_template'):
        logger.warning("Chat mode requested but no chat template found. Using simple formatting.")

    # Interactive mode
    if args.interactive:
        interactive_mode(model, tokenizer, args)
        return

    # Single prompt mode
    if not args.prompt:
        logger.error("Either --prompt or --interactive must be specified")
        sys.exit(1)

    # Generate
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
        device=args.device,
        chat_mode=args.chat_mode,
        system_prompt=args.system_prompt,
    )

    # Print output
    print("\n" + "=" * 80)
    print("Generated Text:")
    print("=" * 80)
    print(output)
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
