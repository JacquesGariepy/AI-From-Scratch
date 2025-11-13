# Tokenization Analysis: Training vs Inference Mismatch

## Executive Summary

**Critical Finding**: There is a complete tokenization mismatch between training and inference in Baguettotron. The model was trained with the official PleIAs/Baguettotron tokenizer (vocab_size=65536), but the inference script uses a placeholder hash-based encoding that is completely incompatible.

## 1. Training Tokenization (What Actually Happened)

### Tokenizer Used During Training

**Source**: `scripts/prepare_wikipedia_data.py` and `scripts/prepare_synth_data.py`

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('PleIAs/Baguettotron')
```

### Tokenizer Specifications

- **Type**: `PreTrainedTokenizerFast` (HuggingFace)
- **Vocab Size**: 65,491 tokens (config) / 65,536 actual (includes special tokens)
- **Architecture**: BPE-based (Byte-Pair Encoding) tokenizer
- **Special Tokens**:
  - BOS: None explicitly set
  - EOS: Token ID 2 (`<|end_of_text|>`)
  - PAD: None explicitly set (defaults to 0)
- **Max Length**: Configured for up to 4096 tokens (model config)

### Training Data Tokenization

```python
# From prepare_wikipedia_data.py line 171
tokens = tokenizer.encode(text, max_length=block_size, truncation=True)
```

**Example encoding**:
```python
Input:  "Bonjour, comment allez-vous?"
Output: [2, 61986, 15, 6108, 3053, 93, 16, 7338, 34]
#       ^  ^      ^   ^     ^     ^   ^   ^     ^
#       |  |      |   |     |     |   |   |     |
#      EOS Bonjour , comment allez - vous ?
```

### Actual Training Data Statistics

**File**: `/data/wikipedia_simple_tokens.json`
- Size: 299.7 MB
- Sequences: 199,925
- Average tokens per sequence: 280.4
- Token range: 2 to 65,507
- Unique tokens in sample: 26,732 (out of 65k vocab)
- Format: `[[token_ids], [token_ids], ...]`

**Sample sequence**:
```json
[2, 25842, 322, 21767, 1912, 351, 265, 8815, 3193, 284, 265, 1232, ...]
```

### Vocabulary Structure

The tokenizer uses BPE with the following characteristics:

```
Token                          ID
'Ġadministered'           ->  17761
'Ġ450'                    ->  18423
'Ġmunicipalities'         ->  42027
'Def'                     ->   5875
'Li'                      ->  13370
```

The `Ġ` prefix indicates a space before the token (standard BPE convention).

## 2. Inference Tokenization (Current Broken State)

### Placeholder Implementation

**Source**: `scripts/generate.py` lines 211-234

```python
def encode_prompt(prompt: str, vocab_size: int) -> torch.Tensor:
    """
    Note: This is a placeholder. In production, use a proper tokenizer.
    """
    # Simple placeholder: hash characters to vocab
    token_ids = [hash(c) % vocab_size for c in prompt]
    return torch.tensor([token_ids], dtype=torch.long)

def decode_tokens(token_ids: torch.Tensor, vocab_size: int) -> str:
    """
    Note: This is a placeholder. In production, use a proper tokenizer.
    """
    return f"[Generated {len(token_ids[0])} tokens]"
```

### Why This Doesn't Work

1. **Different tokenization scheme**:
   - Training: BPE subword tokenization
   - Inference: Character-level hashing

2. **Different token IDs**:
   - Training: "Bonjour" → `[2, 61986, 15, ...]` (meaningful BPE tokens)
   - Inference: "Bonjour" → `[hash('B') % 65536, hash('o') % 65536, ...]` (random)

3. **No semantic relationship**:
   - The hash of character 'B' has no relationship to the BPE token for "Bon"
   - Model learned relationships between BPE tokens, not character hashes

4. **No decoding**:
   - Current decoder just prints "[Generated N tokens]"
   - Cannot convert model output back to text

## 3. Legacy Tokenizer (Stub Implementation)

**Source**: `legacy/tokenizer.py`

```python
class StubTokenizer:
    def __init__(self, vocab_size: int = 1024):
        self.vocab_size = vocab_size
    def encode(self, text: str):
        return [abs(hash(w)) % self.vocab_size for w in text.split()]
    def decode(self, ids):
        return " ".join(f"<id:{i}>" for i in ids)
```

This is another placeholder - word-level hashing instead of character-level, but equally incompatible with the trained model.

## 4. The Training Pipeline

### Data Preparation Flow

1. **Download dataset** (Wikipedia/SYNTH)
   ```bash
   python prepare_wikipedia_data.py --lang simple --max-samples 50000
   ```

2. **Tokenize with HuggingFace tokenizer**
   ```python
   tokenizer = AutoTokenizer.from_pretrained('PleIAs/Baguettotron')
   tokens = tokenizer.encode(text, max_length=2048, truncation=True)
   ```

3. **Save tokenized sequences**
   ```json
   [[2, 25842, 322, ...], [2, 1912, 351, ...], ...]
   ```

4. **Load in training** (`scripts/train.py`)
   ```python
   dataset = TextDataset('data/wikipedia_simple_tokens.json', block_size=2048)
   ```

5. **Collate batches** (`src/baguettotron/data/collator.py`)
   ```python
   collator = DataCollatorForLanguageModeling(pad_token_id=0)
   ```

### What the Model Learned

The model was trained on:
- **Input**: BPE token IDs from the PleIAs/Baguettotron tokenizer
- **Vocab**: 65,536 tokens representing subword units
- **Embeddings**: Learned representations for each of the 65k BPE tokens
- **Patterns**: Statistical relationships between BPE tokens (not characters!)

### Config Alignment

**Model Config** (`src/baguettotron/config.py`):
```python
vocab_size=65536  # Matches tokenizer vocab size
```

**Training Config** (YAML files):
```yaml
model:
  vocab_size: 65536  # Correct size for PleIAs tokenizer
```

## 5. Why Generation is Broken

### The Mismatch Visualized

```
TRAINING:
Text: "Hello world"
  ↓ (PleIAs tokenizer)
Tokens: [2, 9906, 1917]
  ↓ (Model embedding layer)
Embeddings: [[0.1, -0.3, ...], [0.5, 0.2, ...], ...]
  ↓ (Training)
Model learns: "after token 2, token 9906 is likely, then 1917..."

INFERENCE (Current):
Text: "Hello world"
  ↓ (Character hash)
Tokens: [hash('H') % 65536, hash('e') % 65536, ...]
       = [42156, 11792, ...]  ← RANDOM, MEANINGLESS
  ↓ (Model embedding layer)
Embeddings: [[random vector], [random vector], ...]
  ↓ (Generation)
Model generates: Garbage (tokens it never saw during training)
```

### The Fundamental Problem

**The model's embedding layer is a lookup table trained on BPE tokens, not character hashes.**

- Embedding layer shape: `(65536, 576)`
- Each row corresponds to a BPE token learned during training
- Feeding random hash IDs accesses random, untrained embedding vectors
- Model has never seen these patterns and generates nonsense

## 6. Available Tokenizer Resources

### HuggingFace Tokenizer Files

When you call `AutoTokenizer.from_pretrained('PleIAs/Baguettotron')`, it downloads:

```
~/.cache/huggingface/hub/models--PleIAs--Baguettotron/
├── tokenizer.json       # Main tokenizer definition (BPE merges, vocab)
├── tokenizer_config.json # Configuration
└── special_tokens_map.json # Special token definitions
```

These files contain:
- Complete vocabulary (65,536 tokens)
- BPE merge rules
- Special token mappings
- Normalization rules

### Tokenizer Properties

```python
tokenizer = AutoTokenizer.from_pretrained('PleIAs/Baguettotron')

# Properties:
tokenizer.vocab_size          # 65491
len(tokenizer.get_vocab())    # 65536 (includes special tokens)
tokenizer.model_max_length    # Very large (no practical limit)
tokenizer.padding_side        # 'right'
tokenizer.truncation_side     # 'right'
```

## 7. How to Fix Generation

### Solution 1: Use the HuggingFace Tokenizer (Recommended)

**Replace the placeholder functions in `scripts/generate.py`**:

```python
def load_tokenizer(tokenizer_name: str = 'PleIAs/Baguettotron'):
    """Load the actual tokenizer used during training."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_name)

def encode_prompt(prompt: str, tokenizer) -> torch.Tensor:
    """Encode prompt using the real tokenizer."""
    token_ids = tokenizer.encode(prompt, return_tensors='pt')
    return token_ids

def decode_tokens(token_ids: torch.Tensor, tokenizer) -> str:
    """Decode tokens using the real tokenizer."""
    # token_ids shape: (batch_size, seq_len) or (seq_len,)
    if token_ids.dim() == 2:
        token_ids = token_ids[0]  # Take first sequence
    return tokenizer.decode(token_ids, skip_special_tokens=True)
```

### Solution 2: Bundle Tokenizer with Checkpoint

**Save tokenizer alongside model**:

```python
# During training (train.py)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('PleIAs/Baguettotron')
tokenizer.save_pretrained('outputs/tokenizer')

# Save in checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'tokenizer_name': 'PleIAs/Baguettotron',  # Reference
}, 'checkpoint.pt')
```

**Load during generation**:

```python
# In generate.py
tokenizer = AutoTokenizer.from_pretrained('outputs/tokenizer')
# or
tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_name'])
```

### Solution 3: HuggingFace-Compatible Export

**Export model in HF format** (includes tokenizer automatically):

```python
from transformers import LlamaForCausalLM, LlamaConfig

# Convert config
hf_config = LlamaConfig(
    vocab_size=config.vocab_size,
    hidden_size=config.hidden_size,
    # ... other params
)

# Create HF model and load weights
hf_model = LlamaForCausalLM(hf_config)
hf_model.load_state_dict(model.state_dict())

# Save everything (model + tokenizer + config)
hf_model.save_pretrained('hf_model')
tokenizer.save_pretrained('hf_model')

# Now you can use with:
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('hf_model')
tokenizer = AutoTokenizer.from_pretrained('hf_model')
```

## 8. Recommendations

### Immediate Fix (Quick)

1. Add HuggingFace tokenizer to `generate.py`
2. Replace placeholder encode/decode functions
3. Add `transformers` to requirements if not present

### Long-term Fix (Robust)

1. Create proper inference module with tokenizer bundled
2. Add tokenizer to all checkpoint saves
3. Add tokenizer tests to verify encoding/decoding matches training
4. Document tokenizer requirements in README
5. Consider HuggingFace-compatible model export for easier deployment

### Testing Strategy

```python
# Verify tokenization matches training
def test_tokenization_consistency():
    # Load training tokenizer
    tokenizer = AutoTokenizer.from_pretrained('PleIAs/Baguettotron')

    # Encode sample
    text = "Bonjour le monde"
    tokens = tokenizer.encode(text)

    # Decode back
    decoded = tokenizer.decode(tokens)

    # Should match (modulo special tokens)
    assert text.lower() in decoded.lower()

    # Check token range
    assert all(0 <= t < 65536 for t in tokens)
```

## 9. Questions Answered

### Q1: How was the data tokenized during training that resulted in vocab_size=65536?

**Answer**: The data was tokenized using `AutoTokenizer.from_pretrained('PleIAs/Baguettotron')`, which is a BPE tokenizer with 65,536 tokens. This is the official tokenizer from HuggingFace.

### Q2: Is there a consistent tokenization scheme we can reverse engineer?

**Answer**: No need to reverse engineer - the tokenizer is publicly available on HuggingFace. It's a standard BPE tokenizer that can be loaded with the `transformers` library.

### Q3: What does the hash-based character encoding in encode_prompt actually do?

**Answer**: It's a placeholder that hashes each character and takes modulo vocab_size. This produces random token IDs that have no semantic relationship to the BPE tokens the model was trained on. It's completely broken for actual inference.

### Q4: Can we create a compatible decoder for the training encoding?

**Answer**: Yes, but we don't need to - the decoder already exists. It's part of the HuggingFace tokenizer. Just use `tokenizer.decode(token_ids)`.

### Q5: Are there any tokenizer artifacts or configs saved with the checkpoint?

**Answer**: No, the current checkpoints don't include tokenizer information. The training scripts download the tokenizer from HuggingFace each time. This should be fixed to bundle the tokenizer with checkpoints.

## 10. Impact Assessment

### Current State
- **Training**: Working correctly with proper tokenizer
- **Inference**: Completely broken - generates garbage
- **Root Cause**: Tokenization mismatch

### Severity
- **Critical**: Model cannot be used for generation
- **Easy to fix**: Just need to use the proper tokenizer
- **No retraining needed**: Model weights are fine

### Required Changes
1. Update `scripts/generate.py` (10 lines of code)
2. Add tokenizer to checkpoint saving (5 lines)
3. Add `transformers` to requirements (1 line)
4. Add tokenizer tests (20 lines)

**Total effort**: ~2 hours of work

## Conclusion

The tokenization mismatch is the root cause of generation failures. The model was correctly trained with the PleIAs/Baguettotron BPE tokenizer (65k vocab), but inference uses a broken hash-based placeholder. The fix is straightforward: use the same HuggingFace tokenizer during inference that was used during training.

The model architecture and training are sound - we just need to complete the inference pipeline with proper tokenization.
