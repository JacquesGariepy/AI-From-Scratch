# Baguettotron Tokenization Guide

## Overview

The Baguettotron tokenization module provides intelligent tokenizer loading with automatic fallback strategies. It handles vocabulary size mismatches gracefully and works even without external dependencies.

## Features

- **Automatic Tokenizer Detection**: Tries official Baguettotron tokenizer, falls back to GPT-2, then character-level
- **Vocab Size Adaptation**: Handles mismatches between model and tokenizer vocabulary sizes
- **Graceful Degradation**: Works without transformers library using character-level fallback
- **Unified Interface**: Consistent API across all tokenizer types

## Quick Start

### Basic Usage

```python
from baguettotron import load_tokenizer, BaguettotronForCausalLM, BaguettotronConfig
import torch

# Create model
config = BaguettotronConfig.baguettotron_321m()
model = BaguettotronForCausalLM(config)

# Load tokenizer (auto-detects best available)
tokenizer = load_tokenizer("auto", model_vocab_size=config.vocab_size)

# Encode text
prompt = "Hello, world!"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate
output_ids = model.generate(input_ids, max_new_tokens=50)

# Decode
generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print(generated_text)
```

### Command-Line Usage

```bash
# Use auto-detection (default)
python scripts/generate.py --checkpoint model.pt --prompt "Hello" --tokenizer auto

# Force specific tokenizer
python scripts/generate.py --checkpoint model.pt --prompt "Hello" --tokenizer baguettotron
python scripts/generate.py --checkpoint model.pt --prompt "Hello" --tokenizer gpt2
python scripts/generate.py --checkpoint model.pt --prompt "Hello" --tokenizer char

# Interactive mode
python scripts/generate.py --checkpoint model.pt --interactive --tokenizer auto
```

## Tokenizer Types

### 1. Official Baguettotron Tokenizer (Recommended)

- **Vocab Size**: 65,491 tokens
- **Requires**: `transformers` library
- **Best For**: Production use with official model

```python
tokenizer = load_tokenizer("baguettotron", model_vocab_size=65536)
```

### 2. GPT-2 Tokenizer

- **Vocab Size**: 50,257 tokens
- **Requires**: `transformers` library
- **Best For**: Testing, quick prototyping

```python
tokenizer = load_tokenizer("gpt2", model_vocab_size=65536)
```

### 3. Character-Level Tokenizer

- **Vocab Size**: Configurable (matches model)
- **Requires**: Nothing (built-in)
- **Best For**: Fallback when transformers unavailable

```python
tokenizer = load_tokenizer("char", model_vocab_size=65536)
```

### 4. Auto-Detection (Default)

Tries tokenizers in order:
1. Baguettotron → 2. GPT-2 → 3. Character-level

```python
tokenizer = load_tokenizer("auto", model_vocab_size=65536)
```

## Handling Vocab Size Mismatches

The tokenizer automatically handles vocabulary size mismatches between the model and tokenizer:

```python
# Model trained with vocab_size=65536
config = BaguettotronConfig(vocab_size=65536, ...)
model = BaguettotronForCausalLM(config)

# GPT-2 tokenizer has vocab_size=50257
tokenizer = load_tokenizer("gpt2", model_vocab_size=65536)

# Encoding: Tokens clipped to model vocab size
input_ids = tokenizer.encode("Hello")  # IDs will be < 65536

# Decoding: Out-of-vocab tokens mapped to UNK
output_ids = model.generate(input_ids)  # May generate IDs > 50257
text = tokenizer.decode(output_ids)     # Handles gracefully
```

### How It Works

**During Encoding**:
- Input tokens are clipped to `min(token_id, model_vocab_size - 1)`
- Ensures model never sees invalid token IDs

**During Decoding**:
- Tokens outside tokenizer vocab are mapped to UNK token
- Prevents crashes from out-of-vocabulary IDs
- Allows generation to work even with mismatched vocab sizes

## TokenizerWrapper API

### Initialization

```python
from baguettotron.tokenization import TokenizerWrapper

tokenizer = TokenizerWrapper(
    tokenizer_type="auto",      # or "baguettotron", "gpt2", "char"
    model_vocab_size=65536,     # Model's vocabulary size
)
```

### Encoding

```python
# Return list of token IDs
token_ids = tokenizer.encode("Hello, world!")

# Return PyTorch tensor
token_tensor = tokenizer.encode("Hello, world!", return_tensors="pt")

# Control special tokens
token_ids = tokenizer.encode("Hello", add_special_tokens=True)
```

### Decoding

```python
# Decode token IDs (list or tensor)
text = tokenizer.decode([123, 456, 789])
text = tokenizer.decode(torch.tensor([[123, 456, 789]]))

# Control special token handling
text = tokenizer.decode(token_ids, skip_special_tokens=True)
```

### Properties

```python
print(tokenizer.tokenizer_name)    # "baguettotron", "gpt2", or "char"
print(tokenizer.vocab_size)        # Actual tokenizer vocab size
print(tokenizer.model_vocab_size)  # Model's vocab size
```

## Installation Requirements

### Minimal Installation (Character-Level Only)

```bash
pip install torch
```

### Full Installation (All Tokenizers)

```bash
pip install torch transformers
```

Or with Baguettotron:

```bash
pip install -e ".[train]"
```

## Examples

### Example 1: Simple Generation

```python
from baguettotron import BaguettotronConfig, BaguettotronForCausalLM, load_tokenizer

# Setup
config = BaguettotronConfig(vocab_size=65536, hidden_size=576, num_hidden_layers=80)
model = BaguettotronForCausalLM(config)
tokenizer = load_tokenizer("auto", model_vocab_size=config.vocab_size)

# Generate
prompt = "The meaning of life is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output_ids = model.generate(input_ids, max_new_tokens=100, temperature=0.8)
text = tokenizer.decode(output_ids, skip_special_tokens=True)

print(f"Prompt: {prompt}")
print(f"Generated: {text}")
```

### Example 2: Batch Generation

```python
prompts = [
    "Once upon a time",
    "In a galaxy far away",
    "The quick brown fox",
]

for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_new_tokens=50)
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"{prompt} → {text}")
```

### Example 3: Custom Generation Parameters

```python
input_ids = tokenizer.encode("Hello", return_tensors="pt")

output_ids = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.9,        # Higher = more random
    top_k=50,               # Top-k sampling
    top_p=0.95,             # Nucleus sampling
    do_sample=True,         # Enable sampling
)

text = tokenizer.decode(output_ids, skip_special_tokens=True)
```

### Example 4: Vocab Size Detection

```python
import torch

# Load checkpoint
checkpoint = torch.load("model.pt")
state_dict = checkpoint.get("model_state_dict", checkpoint)

# Infer vocab size from embeddings
vocab_size = state_dict["embeddings.weight"].shape[0]
print(f"Model vocab size: {vocab_size}")

# Load matching tokenizer
tokenizer = load_tokenizer("auto", model_vocab_size=vocab_size)
print(f"Tokenizer: {tokenizer}")
```

## Troubleshooting

### Issue: "transformers library not available"

**Solution**: Install transformers:
```bash
pip install transformers
```

Or use character-level tokenizer:
```python
tokenizer = load_tokenizer("char", model_vocab_size=65536)
```

### Issue: Vocab size mismatch warnings

**Solution**: This is expected and handled automatically. The warning is informational.

To silence it:
```python
import logging
logging.getLogger("baguettotron.tokenization").setLevel(logging.ERROR)
```

### Issue: Generated text looks wrong with character tokenizer

**Expected Behavior**: Character-level tokenizer uses hash-based encoding and shows tokens as `[TOKEN_ID]`. This is a fallback and not intended for production.

**Solution**: Install transformers for proper tokenization:
```bash
pip install transformers
```

### Issue: Model generates tokens outside tokenizer vocab

**Solution**: This is handled automatically. Out-of-vocab tokens are mapped to UNK during decoding.

For better results, use a tokenizer with matching vocab size:
- Official Baguettotron: 65,491 tokens
- Model trained with: 65,536 tokens

## Advanced Usage

### Custom Tokenizer Integration

You can wrap any HuggingFace tokenizer:

```python
from transformers import AutoTokenizer
from baguettotron.tokenization import TokenizerWrapper

# Load custom HF tokenizer
hf_tokenizer = AutoTokenizer.from_pretrained("your-tokenizer")

# Wrap it
wrapper = TokenizerWrapper.__new__(TokenizerWrapper)
wrapper.tokenizer = hf_tokenizer
wrapper.tokenizer_name = "custom"
wrapper.vocab_size = hf_tokenizer.vocab_size
wrapper.model_vocab_size = 65536

# Use it
text = wrapper.decode(token_ids)
```

### Extending CharLevelTokenizer

```python
from baguettotron.tokenization import CharLevelTokenizer

class BetterCharTokenizer(CharLevelTokenizer):
    def __init__(self, vocab_size=65536):
        super().__init__(vocab_size)
        # Add character mapping for reversible encoding
        self.char_to_id = {chr(i): i + 4 for i in range(256)}
        self.id_to_char = {i + 4: chr(i) for i in range(256)}

    def encode(self, text):
        return [self.char_to_id.get(c, self.unk_token_id) for c in text]

    def decode(self, token_ids):
        return "".join(self.id_to_char.get(tid, "?") for tid in token_ids)
```

## Performance Considerations

### Tokenizer Loading Time

- **Baguettotron**: ~1-2 seconds (downloads from HF on first use)
- **GPT-2**: ~0.5-1 second (smaller, cached)
- **Character-level**: Instant (no external dependencies)

### Memory Usage

- **Baguettotron**: ~200 MB (tokenizer files)
- **GPT-2**: ~50 MB
- **Character-level**: Negligible

### Generation Speed

Tokenizer type does not affect generation speed significantly. The bottleneck is model inference, not tokenization.

## References

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Official Baguettotron Model](https://huggingface.co/PleIAs/Baguettotron)
- [GPT-2 Tokenizer](https://huggingface.co/gpt2)

## See Also

- [Generation Guide](GENERATION_GUIDE.md) - Advanced generation strategies
- [Dataset Guide](DATASET_GUIDE.md) - Training data preparation
- [Architecture](ARCHITECTURE.md) - Model architecture details
