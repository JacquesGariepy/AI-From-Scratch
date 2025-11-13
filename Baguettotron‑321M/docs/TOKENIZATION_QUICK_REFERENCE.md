# Tokenization Quick Reference

## TL;DR

**Problem**: Training uses BPE tokenizer, inference uses character hashing → BROKEN

**Solution**: Use `AutoTokenizer.from_pretrained('PleIAs/Baguettotron')` in inference

**Impact**: 15 lines of code, 30 minutes, fixes all generation

---

## Quick Facts

### Training Tokenization
- **Tokenizer**: `AutoTokenizer.from_pretrained('PleIAs/Baguettotron')`
- **Type**: BPE (Byte-Pair Encoding)
- **Vocab Size**: 65,536 tokens
- **Granularity**: Subword (e.g., "Bonjour" → 1 token, "municipalities" → 1-2 tokens)
- **Example**: `"Hello world"` → `[2, 24748, 1917]`
- **Status**: ✓ Working correctly

### Inference Tokenization (Current)
- **Tokenizer**: Character hash function
- **Type**: Character-level hashing
- **Vocab Size**: 65,536 random indices
- **Granularity**: Character (e.g., "Hello" → 5 tokens)
- **Example**: `"Hello world"` → `[hash('H')%65536, hash('e')%65536, ...]`
- **Status**: ✗ BROKEN - incompatible with training

---

## Token Examples

### Training Data (BPE)
```python
Input:  "Bonjour, comment allez-vous?"
Tokens: [2, 61986, 15, 6108, 3053, 93, 16, 7338, 34]
        ↑   ↑      ↑   ↑     ↑     ↑   ↑   ↑     ↑
       EOS  Bon   ,   comm  ent   -   v  ous   ?
```

### Current Inference (Hash - BROKEN)
```python
Input:  "Bonjour, comment allez-vous?"
Tokens: [hash('B')%65536, hash('o')%65536, hash('n')%65536, ...]
      = [42156, 11792, 33421, ...]  # RANDOM, MEANINGLESS
```

### Fixed Inference (BPE - CORRECT)
```python
Input:  "Bonjour, comment allez-vous?"
Tokens: [2, 61986, 15, 6108, 3053, 93, 16, 7338, 34]
        # SAME AS TRAINING - WORKS!
```

---

## File Locations

### Training Scripts (Working)
- `/scripts/prepare_wikipedia_data.py` - Downloads & tokenizes with BPE
- `/scripts/prepare_synth_data.py` - Downloads & tokenizes with BPE
- `/scripts/train.py` - Loads pre-tokenized data
- `/src/baguettotron/data/dataset.py` - Dataset class

### Inference Scripts (Broken)
- `/scripts/generate.py` - Lines 211-254 (placeholder tokenizer)
- `/legacy/tokenizer.py` - Stub tokenizer (word hashing)

### Data Files
- `/data/wikipedia_simple_tokens.json` - 299MB, 199,925 sequences, BPE tokens
- `/data/quick-test/train_tokens.json` - Test data, BPE tokens

---

## Code Snippets

### Load HuggingFace Tokenizer
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('PleIAs/Baguettotron')
# Vocab size: 65,536
# Type: PreTrainedTokenizerFast (BPE)
```

### Encode Text
```python
# Input: string
# Output: list of token IDs
tokens = tokenizer.encode("Bonjour le monde")
# [2, 61986, 443, 20015]
```

### Decode Tokens
```python
# Input: list of token IDs
# Output: string
text = tokenizer.decode([2, 61986, 443, 20015])
# "<|end_of_text|>Bonjour le monde"

# Skip special tokens
text = tokenizer.decode([2, 61986, 443, 20015], skip_special_tokens=True)
# "Bonjour le monde"
```

### Encode with PyTorch Tensors
```python
# For model input
tokens = tokenizer.encode("Hello", return_tensors='pt')
# torch.Tensor([[2, 24748]])  shape: (1, 2)
```

---

## Special Tokens

```python
tokenizer.eos_token_id  # 2 - <|end_of_text|>
tokenizer.bos_token_id  # None (not set)
tokenizer.pad_token_id  # None (defaults to 0)
```

---

## Vocabulary Statistics

### From Training Data Analysis
```
File: wikipedia_simple_tokens.json
├─ Sequences: 199,925
├─ Avg tokens/seq: 280.4
├─ Token range: 2 to 65,507
└─ Unique tokens (sample): 26,732 / 65,536
```

### Sample Vocabulary Entries
```
Token                    ID
'<|end_of_text|>'    →  2
'Def'                →  5,875
'Ġcomment'           →  6,108
'Hello'              →  24,748
'Ġadministered'      →  17,761
'Bonjour'            →  61,986
```

(Note: `Ġ` indicates space before token)

---

## Quick Fix for generate.py

### Before (Broken)
```python
def encode_prompt(prompt: str, vocab_size: int) -> torch.Tensor:
    token_ids = [hash(c) % vocab_size for c in prompt]  # ✗ WRONG
    return torch.tensor([token_ids], dtype=torch.long)

def decode_tokens(token_ids: torch.Tensor, vocab_size: int) -> str:
    return f"[Generated {len(token_ids[0])} tokens]"  # ✗ WRONG
```

### After (Fixed)
```python
from transformers import AutoTokenizer

# Load once at module level
TOKENIZER = AutoTokenizer.from_pretrained('PleIAs/Baguettotron')

def encode_prompt(prompt: str) -> torch.Tensor:
    return TOKENIZER.encode(prompt, return_tensors='pt')  # ✓ CORRECT

def decode_tokens(token_ids: torch.Tensor) -> str:
    if token_ids.dim() == 2:
        token_ids = token_ids[0]
    return TOKENIZER.decode(token_ids, skip_special_tokens=True)  # ✓ CORRECT
```

---

## Testing

### Verify Tokenizer Works
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('PleIAs/Baguettotron')

# Test encode
tokens = tokenizer.encode("Bonjour")
assert isinstance(tokens, list)
assert all(0 <= t < 65536 for t in tokens)

# Test decode
text = tokenizer.decode(tokens)
assert "bonjour" in text.lower()

# Test roundtrip
original = "Hello, world!"
encoded = tokenizer.encode(original)
decoded = tokenizer.decode(encoded, skip_special_tokens=True)
assert original.lower() in decoded.lower()
```

### Verify Generation Pipeline
```python
# Full pipeline test
model = load_model("checkpoint.pt")
tokenizer = AutoTokenizer.from_pretrained('PleIAs/Baguettotron')

# Encode
prompt = "The weather is"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate
output_ids = model.generate(input_ids, max_new_tokens=10)

# Decode
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Should contain prompt + new text
assert prompt in generated_text
assert len(generated_text) > len(prompt)
```

---

## Common Pitfalls

### 1. Using Wrong Tokenizer
```python
# ✗ WRONG - Training used PleIAs tokenizer, not GPT2
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# ✓ CORRECT
tokenizer = AutoTokenizer.from_pretrained('PleIAs/Baguettotron')
```

### 2. Character-Level Encoding
```python
# ✗ WRONG - Character hashing
tokens = [hash(c) % 65536 for c in text]

# ✓ CORRECT - BPE encoding
tokens = tokenizer.encode(text)
```

### 3. Not Decoding Output
```python
# ✗ WRONG - Just printing token count
output = f"[Generated {len(tokens)} tokens]"

# ✓ CORRECT - Actually decode
output = tokenizer.decode(tokens, skip_special_tokens=True)
```

### 4. Tensor Shape Mismatch
```python
# ✗ WRONG - 1D tensor when model expects 2D
input_ids = torch.tensor([1, 2, 3])  # shape: (3,)

# ✓ CORRECT - Add batch dimension
input_ids = torch.tensor([[1, 2, 3]])  # shape: (1, 3)
# or
input_ids = tokenizer.encode(text, return_tensors='pt')  # automatically batched
```

---

## Dependencies

### Required
```bash
pip install transformers  # For AutoTokenizer
pip install torch         # For model
```

### Optional (for dataset preparation)
```bash
pip install datasets      # For downloading Wikipedia/SYNTH
```

---

## Tokenizer Cache Location

When you load the tokenizer, it downloads to:
```
~/.cache/huggingface/hub/models--PleIAs--Baguettotron/
├── tokenizer.json              # BPE vocab and merges
├── tokenizer_config.json       # Configuration
└── special_tokens_map.json     # Special tokens
```

To force re-download:
```python
tokenizer = AutoTokenizer.from_pretrained(
    'PleIAs/Baguettotron',
    force_download=True
)
```

---

## Offline Usage

### Download and Save Locally
```python
tokenizer = AutoTokenizer.from_pretrained('PleIAs/Baguettotron')
tokenizer.save_pretrained('models/tokenizer')
```

### Load from Local Path
```python
tokenizer = AutoTokenizer.from_pretrained('models/tokenizer')
```

### Bundle with Checkpoint
```python
# Save
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'tokenizer_path': 'models/tokenizer',  # Reference
}, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_path'])
```

---

## Performance Notes

### Encoding Speed
- BPE encoding: ~100,000 tokens/sec
- Fast enough for real-time inference
- Can cache common prompts if needed

### Memory Usage
- Tokenizer: ~50MB in memory
- Vocab size: 65,536 tokens
- Load once, reuse for all requests

---

## Debugging Commands

### Inspect Tokenizer
```python
tokenizer = AutoTokenizer.from_pretrained('PleIAs/Baguettotron')

print(f"Vocab size: {tokenizer.vocab_size}")           # 65491
print(f"Model type: {tokenizer.model_type}")           # (not set)
print(f"Max length: {tokenizer.model_max_length}")     # very large
print(f"EOS token: {tokenizer.eos_token_id}")          # 2
print(f"Vocab entries: {len(tokenizer.get_vocab())}")  # 65536
```

### Inspect Tokens
```python
tokens = tokenizer.encode("Hello world")
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")
print(f"Token range: {min(tokens)} to {max(tokens)}")

# Decode each token
for token_id in tokens:
    token_str = tokenizer.decode([token_id])
    print(f"  {token_id:6d} → {repr(token_str)}")
```

### Compare with Training Data
```python
import json

# Load training data
with open('data/wikipedia_simple_tokens.json') as f:
    training_data = json.load(f)

# Check token range
all_tokens = []
for seq in training_data[:1000]:
    all_tokens.extend(seq)

print(f"Training tokens range: {min(all_tokens)} to {max(all_tokens)}")
print(f"Tokenizer vocab size: {len(tokenizer.get_vocab())}")
print(f"Match: {max(all_tokens) < len(tokenizer.get_vocab())}")
```

---

## Next Steps

1. **Fix generate.py** - Replace placeholder tokenizer (30 min)
2. **Add tests** - Verify tokenization works (1 hour)
3. **Update docs** - Document tokenizer usage (30 min)
4. **Bundle tokenizer** - Save with checkpoints (15 min)
5. **Test generation** - Verify model produces coherent text (15 min)

**Total**: ~2.5 hours to complete fix

---

## References

- **Model**: https://huggingface.co/PleIAs/Baguettotron
- **Tokenizer**: https://huggingface.co/PleIAs/Baguettotron/tree/main
- **Transformers Docs**: https://huggingface.co/docs/transformers/
- **BPE Paper**: https://arxiv.org/abs/1508.07909

---

## Quick Commands

```bash
# Test tokenizer in Python
python3 -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('PleIAs/Baguettotron')
print('Vocab:', t.vocab_size)
print('Encode:', t.encode('Bonjour'))
print('Decode:', t.decode([2, 61986]))
"

# Check training data tokens
python3 -c "
import json
data = json.load(open('data/wikipedia_simple_tokens.json'))
tokens = data[0]
print('Sample:', tokens[:20])
print('Range:', min(tokens), 'to', max(tokens))
"

# Test full pipeline
python3 -c "
from transformers import AutoTokenizer
import torch
t = AutoTokenizer.from_pretrained('PleIAs/Baguettotron')
ids = t.encode('Hello', return_tensors='pt')
print('Encoded:', ids)
text = t.decode(ids[0], skip_special_tokens=True)
print('Decoded:', text)
"
```

---

**Last Updated**: 2024-11-12
**Status**: Analysis complete, fix pending implementation
