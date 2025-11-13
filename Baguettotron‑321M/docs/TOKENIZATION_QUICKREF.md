# Tokenization Quick Reference

## CLI Usage

```bash
# Generate with auto-detection (recommended)
python scripts/generate.py --checkpoint model.pt --prompt "Hello" --tokenizer auto

# Force specific tokenizer
python scripts/generate.py --checkpoint model.pt --prompt "Hello" --tokenizer baguettotron
python scripts/generate.py --checkpoint model.pt --prompt "Hello" --tokenizer gpt2
python scripts/generate.py --checkpoint model.pt --prompt "Hello" --tokenizer char

# Interactive mode
python scripts/generate.py --checkpoint model.pt --interactive

# With generation parameters
python scripts/generate.py \
    --checkpoint model.pt \
    --prompt "Once upon a time" \
    --max-new-tokens 100 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.9 \
    --tokenizer auto
```

## Python API

```python
from baguettotron import BaguettotronForCausalLM, BaguettotronConfig, load_tokenizer

# Load model
config = BaguettotronConfig.baguettotron_321m()
model = BaguettotronForCausalLM(config)

# Load tokenizer
tokenizer = load_tokenizer("auto", model_vocab_size=config.vocab_size)

# Generate
prompt = "The meaning of life is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output_ids = model.generate(input_ids, max_new_tokens=100)
text = tokenizer.decode(output_ids, skip_special_tokens=True)
print(text)
```

## Tokenizer Types

| Type | Vocab Size | Requires | Best For |
|------|-----------|----------|----------|
| `baguettotron` | 65,491 | transformers | Production |
| `gpt2` | 50,257 | transformers | Testing |
| `char` | 65,536 | Nothing | Fallback |
| `auto` | Varies | - | Auto-detect best |

## Vocab Size Compatibility

| Model Vocab | Tokenizer | Compatible | Notes |
|-------------|-----------|------------|-------|
| 65,536 | Baguettotron (65,491) | ✅ Yes | Recommended |
| 65,536 | GPT-2 (50,257) | ✅ Yes | Via adaptation |
| 65,536 | Char (65,536) | ✅ Yes | Perfect match |
| Any | Any | ✅ Yes | Auto-adapted |

## Common Patterns

### Pattern 1: Simple Generation

```python
tokenizer = load_tokenizer("auto", model_vocab_size=65536)
input_ids = tokenizer.encode("Hello", return_tensors="pt")
output_ids = model.generate(input_ids, max_new_tokens=50)
text = tokenizer.decode(output_ids)
```

### Pattern 2: Batch Processing

```python
prompts = ["Hello", "Bonjour", "Hola"]
for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_new_tokens=50)
    print(tokenizer.decode(output_ids))
```

### Pattern 3: Custom Parameters

```python
output_ids = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.9,      # Randomness
    top_k=50,             # Top-k sampling
    top_p=0.95,           # Nucleus sampling
    do_sample=True,       # Enable sampling
)
```

### Pattern 4: Vocab Size Detection

```python
checkpoint = torch.load("model.pt")
vocab_size = checkpoint["model_state_dict"]["embeddings.weight"].shape[0]
tokenizer = load_tokenizer("auto", model_vocab_size=vocab_size)
```

## Troubleshooting

### Issue: "transformers library not available"

```bash
pip install transformers
```

Or use char tokenizer:
```python
tokenizer = load_tokenizer("char", model_vocab_size=65536)
```

### Issue: Vocab size mismatch warning

This is normal and handled automatically. To silence:
```python
import logging
logging.getLogger("baguettotron.tokenization").setLevel(logging.ERROR)
```

### Issue: Character tokenizer shows `[TOKEN_ID]`

This is expected behavior for the fallback. Install transformers for proper text:
```bash
pip install transformers
```

## File Locations

```
src/baguettotron/tokenization.py       # Main module
scripts/generate.py                     # CLI script
scripts/demo_tokenization.py            # Demo script
tests/test_tokenization.py              # Unit tests
tests/test_generation_integration.py    # Integration tests
docs/TOKENIZATION_GUIDE.md              # Full guide
docs/TOKENIZATION_FIX_SUMMARY.md        # Fix summary
```

## Testing

```bash
# Unit tests
pytest tests/test_tokenization.py -v

# Integration tests
pytest tests/test_generation_integration.py -v

# All tokenization tests
pytest tests/test_tokenization.py tests/test_generation_integration.py -v

# Demo
python scripts/demo_tokenization.py
```

## Installation

```bash
# Minimal (char tokenizer only)
pip install torch

# Full (all tokenizers)
pip install transformers

# With baguettotron
pip install -e ".[train]"
```

## Key Functions

```python
# Load tokenizer
tokenizer = load_tokenizer(type, model_vocab_size)

# Encode text
token_ids = tokenizer.encode(text, return_tensors="pt")

# Decode tokens
text = tokenizer.decode(token_ids, skip_special_tokens=True)

# Properties
tokenizer.tokenizer_name    # "baguettotron", "gpt2", or "char"
tokenizer.vocab_size        # Actual tokenizer vocab
tokenizer.model_vocab_size  # Model's vocab size
```

## See Also

- **Full Guide**: `docs/TOKENIZATION_GUIDE.md`
- **Fix Summary**: `docs/TOKENIZATION_FIX_SUMMARY.md`
- **Demo**: `python scripts/demo_tokenization.py`
- **Tests**: `pytest tests/test_tokenization.py -v`
