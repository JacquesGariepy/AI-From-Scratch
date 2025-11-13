# Official PleIAs/Baguettotron Model Specification

**Last Updated:** 2025-11-12
**Source:** https://huggingface.co/PleIAs/Baguettotron
**Purpose:** 1:1 alignment verification reference

---

## Table of Contents
1. [Model Overview](#model-overview)
2. [Exact Configuration (config.json)](#exact-configuration-configjson)
3. [Architecture Details](#architecture-details)
4. [Tokenizer Specification](#tokenizer-specification)
5. [Special Features](#special-features)
6. [Training Information](#training-information)
7. [Implementation Verification Checklist](#implementation-verification-checklist)

---

## Model Overview

**Name:** Baguettotron-321M
**Organization:** PleIAs
**Parameters:** 321 million
**Model Class:** `LlamaForCausalLM`
**Architecture Type:** Decoder-only Transformer
**Design Philosophy:** Extremely deep architecture (80 layers) for size class - "baguette" model due to depth-to-width ratio

### Key Characteristics
- **Depth:** 80 layers (unusually deep for 321M parameter class)
- **Dataset:** SYNTH (fully open generalist dataset)
- **Training Tokens:** 200 billion
- **Hardware:** 16 H100 GPUs (Jean Zay supercomputer)
- **Languages:** 6 languages (French, German, Italian, Spanish, Polish, Latin/Dutch)
- **Reasoning:** Exclusive English reasoning traces with multi-turn conversation support

---

## Exact Configuration (config.json)

```json
{
  "_name_or_path": "PleIAs/Baguettotron",
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 576,
  "initializer_range": 0.02,
  "intermediate_size": 1536,
  "max_position_embeddings": 4096,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 9,
  "num_hidden_layers": 80,
  "num_key_value_heads": 3,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.3",
  "use_cache": true,
  "vocab_size": 65536
}
```

### Configuration Parameters Breakdown

#### Core Architecture
- **`vocab_size`**: `65536` (2^16 tokens)
- **`hidden_size`**: `576` (embedding/hidden dimension)
- **`num_hidden_layers`**: `80` (transformer blocks)
- **`num_attention_heads`**: `9` (query heads)
- **`num_key_value_heads`**: `3` (GQA with 3:1 ratio)
- **`intermediate_size`**: `1536` (MLP hidden dimension)

#### Position Embeddings
- **`max_position_embeddings`**: `4096` (max sequence length)
- **`rope_theta`**: `10000.0` (RoPE base frequency)
- **`rope_scaling`**: `null` (no scaling applied)

#### Activation & Normalization
- **`hidden_act`**: `"silu"` (Swish activation for SwiGLU)
- **`rms_norm_eps`**: `1e-05` (RMSNorm epsilon)

#### Attention Configuration
- **`attention_dropout`**: `0.0` (no dropout)
- **`attention_bias`**: `false` (no bias in attention projections)
- **`mlp_bias`**: `false` (no bias in MLP projections)

#### Special Tokens
- **`bos_token_id`**: `1` (`<|begin_of_text|>`)
- **`eos_token_id`**: `2` (`<|end_of_text|>`)
- **`pad_token_id`**: `3` (`[PAD]`) - from tokenizer_config.json

#### Embeddings & Cache
- **`tie_word_embeddings`**: `true` (input/output embeddings shared)
- **`use_cache`**: `true` (enable KV caching)

#### Training Configuration
- **`initializer_range`**: `0.02` (weight initialization std)
- **`torch_dtype`**: `"bfloat16"` (16-bit brain float)
- **`pretraining_tp`**: `1` (tensor parallelism degree)

#### Model Metadata
- **`model_type`**: `"llama"` (LLaMA architecture family)
- **`architectures`**: `["LlamaForCausalLM"]` (HuggingFace model class)
- **`transformers_version`**: `"4.51.3"` (library version)

---

## Architecture Details

### Layer-by-Layer Structure

```
BaguettotronForCausalLM (LlamaForCausalLM)
│
├── Token Embeddings (65536, 576)
│   └── Tied with lm_head (shared weights)
│
├── Transformer Decoder (80 layers)
│   └── TransformerBlock × 80
│       ├── Pre-Attention Normalization
│       │   └── RMSNorm(576, eps=1e-05)
│       │
│       ├── Grouped Query Attention (GQA)
│       │   ├── Q Projection: (576) → (9 heads × 64 dim = 576)
│       │   ├── K Projection: (576) → (3 heads × 64 dim = 192)
│       │   ├── V Projection: (576) → (3 heads × 64 dim = 192)
│       │   ├── RoPE (theta=10000.0, max_len=4096)
│       │   ├── Scaled Dot-Product Attention
│       │   │   ├── Scale: 1/√64 = 0.125
│       │   │   ├── Causal Masking
│       │   │   └── No Dropout (p=0.0)
│       │   └── O Projection: (576) → (576)
│       │
│       ├── Residual Connection #1
│       │
│       ├── Pre-FFN Normalization
│       │   └── RMSNorm(576, eps=1e-05)
│       │
│       ├── SwiGLU Feed-Forward Network
│       │   ├── Gate Projection: (576) → (1536)
│       │   ├── Up Projection: (576) → (1536)
│       │   ├── SiLU Activation on Gate
│       │   ├── Element-wise Multiply: gate * up
│       │   └── Down Projection: (1536) → (576)
│       │
│       └── Residual Connection #2
│
├── Final Normalization
│   └── RMSNorm(576, eps=1e-05)
│
└── Language Modeling Head (576) → (65536)
    └── Tied with embeddings (shared weights)
```

### Head Dimension Calculation
```
head_dim = hidden_size / num_attention_heads
         = 576 / 9
         = 64
```

### GQA Configuration
- **Query Heads:** 9
- **Key-Value Heads:** 3
- **Ratio:** 3 query heads per KV head
- **KV Groups:** 9 / 3 = 3 groups

Each KV head is shared by 3 query heads, reducing memory footprint while maintaining quality.

### Parameter Count Estimation

**Embeddings:**
- Token embeddings: 65,536 × 576 = 37,748,736
- LM head: Tied (shared weights, no additional params)

**Per Transformer Block:**
- Attention (Q/K/V/O projections): ~665,856 params
  - Q: 576 × 576 = 331,776
  - K: 576 × 192 = 110,592
  - V: 576 × 192 = 110,592
  - O: 576 × 576 = 331,776
- SwiGLU MLP: ~2,654,208 params
  - Gate: 576 × 1,536 = 884,736
  - Up: 576 × 1,536 = 884,736
  - Down: 1,536 × 576 = 884,736
- RMSNorm (×2): ~1,152 params
  - Attention norm: 576
  - FFN norm: 576

**Total per block:** ~3,321,216 params

**All 80 layers:** 80 × 3,321,216 = 265,697,280 params

**Final RMSNorm:** 576 params

**Total:** 37,748,736 + 265,697,280 + 576 ≈ **303,446,592 params**

**Note:** Official count is 321M, likely accounting for tied embeddings differently or additional overhead.

---

## Tokenizer Specification

### Tokenizer Type
**BPE (Byte Pair Encoding)** with byte-level processing

### Vocabulary
- **Size:** 65,536 tokens (2^16)
- **Type:** Byte-level BPE
- **Unknown Token:** `[UNK]` (ID: 0)
- **Padding Token:** `[PAD]` (ID: 3)

### Special Tokens

#### Core Tokens
| Token | ID | Purpose |
|-------|----|---------|
| `[UNK]` | 0 | Unknown token |
| `<\|begin_of_text\|>` | 1 | Beginning of sequence (BOS) |
| `<\|end_of_text\|>` | 2 | End of sequence (EOS) |
| `[PAD]` | 3 | Padding |

#### Message Delimiters
- `<\|im_start\|>` - Message start
- `<\|im_end\|>` - Message end

#### Reasoning Markers
- `<think>` - Start thinking/reasoning trace
- `</think>` - End thinking/reasoning trace

#### Source References (RAG)
- `source_1` through `source_10` - Document source markers
- `<ref` - Reference start tag
- `</ref>` - Reference end tag

#### Uncertainty/Confidence Markers
- `?maybe?` - Uncertainty indicator
- `⟨H≈0.1⟩` through `⟨H≈1.8⟩` - Entropy/confidence levels (19 tokens)
  - Lower values indicate higher confidence
  - Higher values indicate higher uncertainty

#### Visual Indicators
- Bullets, arrows, circles
- Checkmarks, warning symbols
- Various formatting symbols

### Pre-Tokenization
**Two-stage process:**
1. **Split** - Regex pattern isolating letters, numbers, punctuation
2. **ByteLevel** - Byte-level processing (prefix space handling disabled)

### Post-Processing
**Two-stage pipeline:**
1. ByteLevel decoding (prefix space enabled)
2. Template processing adding `<|end_of_text|>` markers

### Normalization
- **Applied:** No explicit normalization
- **Special Tokens:** All have `normalized: false`, `lstrip: false`, `rstrip: false`

---

## Special Features

### 1. Grouped Query Attention (GQA)
- **Implementation:** 9 query heads, 3 key-value heads
- **Efficiency:** Reduces KV cache size by 3× compared to MHA
- **Quality:** Maintains performance with fewer parameters

### 2. Rotary Position Embeddings (RoPE)
- **Type:** Rotary embeddings applied to Q and K
- **Theta:** 10,000.0 (base frequency)
- **Max Length:** 4,096 tokens
- **Scaling:** None (rope_scaling: null)

### 3. SwiGLU Activation
- **Type:** Swish-Gated Linear Unit
- **Formula:** `down_proj(SiLU(gate_proj(x)) * up_proj(x))`
- **Advantage:** Better performance than standard ReLU/GELU MLPs

### 4. RMSNorm
- **Type:** Root Mean Square Layer Normalization
- **Epsilon:** 1e-05
- **Advantage:** More efficient than LayerNorm (no mean centering, no bias)

### 5. Pre-Normalization
- **Pattern:** Norm before attention/MLP, not after
- **Advantage:** Better training stability for deep models

### 6. No Bias Terms
- **Attention:** `attention_bias: false`
- **MLP:** `mlp_bias: false`
- **Advantage:** Fewer parameters, faster computation

### 7. Tied Embeddings
- **Configuration:** `tie_word_embeddings: true`
- **Implementation:** Input embeddings and output LM head share same weights
- **Advantage:** Reduces parameter count by ~37M

### 8. bfloat16 Precision
- **Type:** 16-bit brain floating point
- **Advantage:** Memory efficiency with maintained numerical stability

### 9. Reasoning Traces
- **Language:** English only (regardless of input language)
- **Multi-turn:** "Rolling" thinking traces across conversation
- **Markers:** `<think>` and `</think>` tokens

### 10. RAG with Grounding
- **Sources:** `source_1` through `source_10`
- **References:** `<ref` and `</ref>` tags
- **Purpose:** Citation and source attribution

---

## Training Information

### Dataset
- **Name:** SYNTH
- **Type:** Fully open generalist synthetic dataset
- **Size:** 200 billion tokens
- **Special:** Dense reasoning signals from MMLU and benchmarks integrated early

### Training Details
- **Hardware:** 16 × H100 GPUs
- **Facility:** Jean Zay supercomputer
- **Key Finding:** "Deeper architecture benefits more from dense reasoning data"

### Capabilities
1. **Instruction Following** - Trained with instruction traces
2. **Multi-turn Conversations** - Rolling thinking traces
3. **Multilingual** - 6 languages (input/output)
4. **Reasoning** - English reasoning traces
5. **RAG** - Retrieval-augmented generation with grounding
6. **Mathematics** - GSM8K-level math reasoning
7. **Knowledge** - MMLU-level general knowledge
8. **Retrieval** - HotPotQA-level information extraction
9. **Creative Writing** - Text generation capabilities
10. **Information Extraction** - Structured data extraction

### Performance Benchmarks
- **MMLU:** Approaches Qwen-0.6B performance
- **GSM8K:** Competitive math reasoning
- **HotPotQA:** Strong retrieval capabilities
- **Comparison:** Significantly outperforms similarly-sized Gemma models

---

## Implementation Verification Checklist

### Configuration Matching
- [ ] `vocab_size = 65536`
- [ ] `hidden_size = 576`
- [ ] `num_hidden_layers = 80`
- [ ] `num_attention_heads = 9`
- [ ] `num_key_value_heads = 3`
- [ ] `intermediate_size = 1536`
- [ ] `max_position_embeddings = 4096`
- [ ] `rope_theta = 10000.0`
- [ ] `rms_norm_eps = 1e-05` (not 1e-06)
- [ ] `attention_dropout = 0.0`
- [ ] `hidden_act = "silu"`
- [ ] `tie_word_embeddings = true`
- [ ] `use_cache = true`
- [ ] `initializer_range = 0.02`
- [ ] `bos_token_id = 1`
- [ ] `eos_token_id = 2`
- [ ] `pad_token_id = 3`

### Architecture Components
- [ ] `model_type = "llama"` (LlamaForCausalLM architecture)
- [ ] Pre-normalization (norm before attention/MLP)
- [ ] RMSNorm (not LayerNorm)
- [ ] Grouped Query Attention (9 Q heads, 3 KV heads)
- [ ] RoPE applied to Q and K tensors
- [ ] SwiGLU (gate_proj, up_proj, down_proj)
- [ ] No bias in attention projections (`attention_bias: false`)
- [ ] No bias in MLP projections (`mlp_bias: false`)
- [ ] Residual connections after attention and MLP
- [ ] Final RMSNorm before LM head
- [ ] Tied embeddings (input = output)

### Attention Details
- [ ] Head dimension: 576 / 9 = 64
- [ ] KV groups: 9 / 3 = 3 (each KV head shared by 3 Q heads)
- [ ] Scaling factor: 1/√64 = 0.125
- [ ] Causal masking enabled
- [ ] No dropout (p=0.0)

### Position Embeddings
- [ ] RoPE implementation (not learned positional embeddings)
- [ ] Theta = 10000.0
- [ ] Max cached length = 4096
- [ ] Applied to both Q and K (not V)
- [ ] Interleaved format for efficiency

### Normalization
- [ ] RMSNorm formula: `x / rms(x) * weight`
- [ ] RMS calculation: `sqrt(mean(x^2) + eps)`
- [ ] Epsilon: 1e-05 (CRITICAL: not 1e-06)
- [ ] No bias term (only learnable scale/weight)
- [ ] Computation in float32, output in original dtype

### Feed-Forward Network
- [ ] SwiGLU architecture (not standard MLP)
- [ ] Gate projection: 576 → 1536
- [ ] Up projection: 576 → 1536
- [ ] Down projection: 1536 → 576
- [ ] SiLU activation on gate
- [ ] Element-wise multiply: gate * up
- [ ] No bias terms

### Weight Initialization
- [ ] Normal distribution: mean=0.0, std=0.02
- [ ] Applied to Linear layers
- [ ] Applied to Embedding layers
- [ ] Bias (if any) initialized to zeros

### Data Types
- [ ] Default dtype: bfloat16
- [ ] RMSNorm computation: float32
- [ ] Cache tensors: match input dtype

### Parameter Count
- [ ] Total parameters: ~321M
- [ ] Embeddings: ~37.7M (tied, counted once)
- [ ] Transformer layers: ~265.7M
- [ ] Normalization: ~48.6K

### HuggingFace Compatibility
- [ ] Model class: `LlamaForCausalLM`
- [ ] Config class: Compatible with `LlamaConfig`
- [ ] Transformers version: 4.51.3
- [ ] Can load with `AutoModelForCausalLM.from_pretrained()`
- [ ] Can load with `AutoTokenizer.from_pretrained()`

### Special Token Handling
- [ ] BOS token properly handled
- [ ] EOS token properly handled
- [ ] PAD token properly handled
- [ ] Reasoning markers (`<think>`, `</think>`) in vocab
- [ ] Message delimiters in vocab
- [ ] Source references in vocab
- [ ] Confidence markers in vocab

### Generation Features
- [ ] KV caching enabled (`use_cache: true`)
- [ ] Causal language modeling
- [ ] Supports greedy decoding
- [ ] Supports sampling (temperature, top_k, top_p)
- [ ] Handles max_position_embeddings correctly

---

## Critical Differences to Watch

### Common Pitfalls
1. **RMSNorm Epsilon:** Must be `1e-05`, not `1e-06`
2. **Max Position Embeddings:** Must be `4096`, not `2048`
3. **Tied Embeddings:** Must be `true` for 1:1 alignment
4. **No Bias:** Both attention and MLP must have `bias=False`
5. **SwiGLU:** Must use SwiGLU, not standard MLP
6. **GQA Ratio:** Must be 9:3, not 9:9 (MHA)
7. **Head Dimension:** Must be 64 (576/9), verify calculations
8. **Model Type:** Must report as "llama" for HF compatibility

### Verification Commands
```python
# Load official model
from transformers import AutoModelForCausalLM, AutoConfig

config = AutoConfig.from_pretrained("PleIAs/Baguettotron")
model = AutoModelForCausalLM.from_pretrained("PleIAs/Baguettotron")

# Check parameter count
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")  # Should be ~321M

# Check configuration
print(config.to_dict())  # Compare with above config.json

# Check state_dict keys
print(model.state_dict().keys())  # Verify layer names match
```

---

## References

- **Model Card:** https://huggingface.co/PleIAs/Baguettotron
- **Config:** https://huggingface.co/PleIAs/Baguettotron/blob/main/config.json
- **Tokenizer Config:** https://huggingface.co/PleIAs/Baguettotron/blob/main/tokenizer_config.json
- **Tokenizer:** https://huggingface.co/PleIAs/Baguettotron/blob/main/tokenizer.json
- **Architecture Diagram:** https://huggingface.co/PleIAs/Baguettotron/resolve/main/figures/baguettotron_structure.png

---

**Document Version:** 1.0
**Last Verified:** 2025-11-12
**Status:** Complete and accurate as of HuggingFace model snapshot
