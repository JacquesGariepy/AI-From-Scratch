# Reasoning Model Analysis - Baguettotron-321M

## Overview

This document analyzes the "Reasoning Model" capabilities of Baguettotron-321M and how they are implemented (or can be implemented) in this reverse-engineered implementation.

**Source**: Based on analysis of the official PleIAs/Baguettotron model card on HuggingFace.

---

## Official Reasoning Model Capabilities

### 1. **Architecture for Reasoning**

**Official Baguettotron:**
- 80 layers (unusually deep for a 321M parameter model)
- Internally termed "baguette" design
- Hypothesis: "Deeper architecture benefits more from dense reasoning data"
- Trained with intensive computation and knowledge synthesis sequences

**Our Implementation:**
```python
# src/baguettotron/config.py
@classmethod
def baguettotron_321m(cls):
    return cls(
        num_hidden_layers=80,  # ✅ Same 80 layers architecture
        hidden_size=576,
        ...
    )
```

**Status**: ✅ **FULLY IMPLEMENTED**
- Our implementation has the exact same 80-layer architecture
- The depth is preserved for reasoning capabilities

---

### 2. **Thinking Tags (`<think>`)**

**Official Baguettotron:**
- Uses `<think>` tags to mark internal reasoning
- Closing with `</think>` can replace thinking traces (but reduces performance)
- Native instruction-following with thinking traces

**Our Implementation:**

**Current State:**
```json
// data/sft.jsonl
{"messages":[
  {"role":"user","content":"Hello"},
  {"role":"assistant","content":"<think>…</think> Hi!"}
]}
```

```json
// configs/chat_template.demo.json
{
  "format": ["<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{think_tag}"],
  "rag_block": "{sources}"
}
```

**Status**: ⚠️ **PARTIALLY IMPLEMENTED**
- Data format supports `<think>` tags
- Template includes `{think_tag}` placeholder
- Model can process these tokens (they're just text tokens)
- **Missing**: Dedicated tokenizer handling for `<think>` as special tokens
- **Missing**: Training on reasoning-dense data

**What's Needed:**
1. Add `<think>` and `</think>` as special tokens in tokenizer
2. Training data with reasoning traces
3. Inference utilities to handle/generate thinking tags

---

### 3. **Qwen-style Instruction Format**

**Official Baguettotron:**
- Uses `<|im_start|>` and `<|im_end|>` delimiters
- Chat format: `<|im_start|>user\nMESSAGE<|im_end|>\n<|im_start|>assistant\n`

**Our Implementation:**

```json
// configs/chat_template.demo.json
"format": ["<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{think_tag}"]
```

```markdown
// README.md - SYNTH Dataset section
Format: Qwen-style chat with `<|im_start|>` / `<|im_end|>`

Example:
<|im_start|>user
What is the capital of France?
<|im_end|>
<|im_start|>assistant
<think>I need to recall European geography</think>
The capital of France is Paris.
```

**Status**: ✅ **DOCUMENTED**
- Format is documented and understood
- Example data shows correct structure
- **Missing**: Tokenizer with these as special tokens

---

### 4. **RAG with Source Tags**

**Official Baguettotron:**
- Uses `<source_N>...</source_N>` for grounding
- Returns answers with `[quote]` references
- Focus on source synthesis over internal knowledge

**Our Implementation:**

```json
// configs/chat_template.demo.json
"rag_block": "{sources}"
```

```markdown
// README.md - Example
<source_1>https://wikipedia.org/France</source_1>
The capital of France is Paris.
```

**Status**: ⚠️ **PARTIALLY DOCUMENTED**
- RAG format is documented
- Template has `{sources}` placeholder
- Example shows `<source_N>` usage
- **Missing**: Implementation of RAG retrieval
- **Missing**: Source parsing/injection utilities

---

### 5. **Multi-turn Reasoning ("Rolling Thinking")**

**Official Baguettotron:**
- Recommendation: "Roll" thinking by appending new traces
- Discard previous thinking traces for each turn
- Maintains context efficiency

**Our Implementation:**

**Status**: ❌ **NOT IMPLEMENTED**
- No specific utilities for multi-turn management
- Would need custom generation logic
- Could be implemented at application level

**What's Needed:**
```python
def generate_with_rolling_thinking(
    model,
    messages,
    keep_thinking=False
):
    """Generate with optional thinking trace management."""
    # 1. Format messages with <think> tags
    # 2. Generate response
    # 3. If not keep_thinking: strip <think>...</think> from history
    # 4. Append to conversation
    pass
```

---

## Summary: What We Have vs. What's Needed

### ✅ Fully Implemented
1. **80-layer architecture** - Exact match with official model
2. **Model structure** - GQA, SwiGLU, RoPE, RMSNorm
3. **Config compatibility** - 100% parameter match

### ⚠️ Partially Implemented (Documentation & Examples)
1. **`<think>` tags** - Format documented, examples provided
2. **Qwen chat format** - Structure documented
3. **RAG source tags** - Format documented, examples provided

### ❌ Not Implemented (Requires Additional Work)
1. **Tokenizer with special tokens**
   - Need `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`
   - Need `<source_N>`, `</source_N>`

2. **Reasoning-dense training data**
   - SYNTH dataset download/preparation (have scripts)
   - Training pipeline (have Trainer class)

3. **Generation utilities**
   - Rolling thinking management
   - RAG source injection
   - Think tag handling

4. **Tokenizer implementation**
   - Need official tokenizer from HuggingFace
   - Or build compatible BPE tokenizer

---

## How to Enable Full Reasoning Capabilities

### Step 1: Get Official Tokenizer

```python
from transformers import AutoTokenizer

# Download official tokenizer
tokenizer = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")

# Special tokens should include:
# <|im_start|>, <|im_end|>, <think>, </think>, <source_N>, </source_N>
```

### Step 2: Prepare Reasoning Data

```python
# Use our existing script
python prepare_synth_data.py --tokenize --split train

# The SYNTH dataset already contains reasoning traces
```

### Step 3: Train (or Fine-tune)

```python
python train.py \
  --train-data data/train_tokens.json \
  --model-config 321m \
  --epochs 10 \
  --batch-size 32
```

### Step 4: Add Generation Utilities

```python
# In src/baguettotron/generation/reasoning.py
def generate_with_thinking(
    model,
    tokenizer,
    prompt,
    include_thinking=True
):
    """Generate with <think> tags."""
    # Format prompt with Qwen style
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    # Tokenize
    input_ids = tokenizer.encode(formatted, return_tensors='pt')

    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.9
    )

    # Decode
    response = tokenizer.decode(output_ids[0])

    # Optionally strip <think>...</think>
    if not include_thinking:
        response = strip_thinking(response)

    return response
```

---

## Recommendations

### For Educational/Research Use:
1. ✅ **Current implementation is sufficient** for understanding architecture
2. ✅ Use documented formats for prompt engineering
3. ⚠️ Load official tokenizer for proper token handling

### For Production Use:
1. ❌ **Use official PleIAs/Baguettotron model** instead
2. ❌ This reverse-engineered implementation lacks trained weights
3. ❌ Reasoning capabilities require trained model + tokenizer

### To Fully Enable Reasoning:
1. Download official tokenizer from HuggingFace
2. Train/fine-tune on SYNTH dataset (reasoning-dense data)
3. Implement generation utilities for thinking tag management
4. Test with reasoning benchmarks

---

## Conclusion

**Our Implementation:**
- ✅ Has the **correct architecture** (80 layers) for reasoning
- ✅ Can **process reasoning tokens** (they're just text)
- ✅ Has **documented formats** for `<think>`, `<source_N>`, etc.
- ⚠️ Lacks **official tokenizer** with special tokens
- ❌ Lacks **trained weights** with reasoning capabilities

**To actually use reasoning:**
- Use official model: `PleIAs/Baguettotron`
- Or train this implementation on SYNTH data with official tokenizer

**Educational Value:**
- This analysis helps understand how reasoning models work
- Architecture analysis shows importance of depth (80 layers)
- Format understanding enables prompt engineering research

---

**Disclaimer**: This analysis is based on publicly available information about the official PleIAs/Baguettotron model. This implementation is an independent reverse engineering project with NO AFFILIATION to PleIAs. For production reasoning capabilities, use the official model.
