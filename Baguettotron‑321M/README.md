# Baguettotron-321M - Reverse Engineering Implementation

[![Not Affiliated](https://img.shields.io/badge/⚠️%20NOT%20AFFILIATED-with%20PleIAs-red)](DISCLAIMER.md)
[![Status](https://img.shields.io/badge/status-Educational%20%2F%20Reverse%20Engineering-yellow)](DISCLAIMER.md)
[![Author](https://img.shields.io/badge/author-Jacques%20Gariépy-blue)](https://github.com/JacquesGariepy)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)
[![Parameters](https://img.shields.io/badge/parameters-320.96M-blue)](#architecture)
[![Compatible](https://img.shields.io/badge/LlamaForCausalLM-compatible-orange)](#architecture)

From-scratch implementation of **Baguettotron-321M**, a 321M parameter language model. This is an **independent reverse engineering attempt** by Jacques Gariépy based on analysis of the publicly available [PleIAs/Baguettotron](https://huggingface.co/PleIAs/Baguettotron) model architecture on HuggingFace.

> **⚠️ Important Disclaimer**:
> - **NOT AFFILIATED** with PleIAs or the Baguettotron team in any way
> - This code was written **independently** through reverse engineering of the model architecture
> - The Baguettotron/PleIAs team **has NOT shared, provided, or assisted** with this code in any way
> - **ZERO collaboration or communication** with the original creators
> - This is purely an educational exercise in understanding transformer architectures
> - All implementation decisions are based **ONLY** on public information (config.json, model structure)
> - For the **official model and weights**, see [PleIAs/Baguettotron](https://huggingface.co/PleIAs/Baguettotron)

## 🎯 Features

### Architecture
- ✅ **100% compatible** with [LlamaForCausalLM](https://huggingface.co/docs/transformers/model_doc/llama)
- ✅ Architecture **identical** to official [PleIAs/Baguettotron](https://huggingface.co/PleIAs/Baguettotron) model
- ✅ **80 layers** - Deep "baguette" architecture optimized for reasoning
- ✅ **SwiGLU** MLP (gate_proj, up_proj, down_proj)
- ✅ **Grouped-Query Attention** (9 heads, 3 KV heads)
- ✅ **RoPE** embeddings (theta=10000)
- ✅ **Pre-normalization** with RMSNorm (eps=1e-5)
- ✅ **Tied embeddings** (shared input/output weights)

### Reasoning Model Capabilities
- ✅ **Reasoning architecture** - 80-layer depth designed for reasoning tasks
- ⚠️ **Thinking tags** - Format documented (`<think>...</think>`)
- ⚠️ **RAG support** - Source tags format (`<source_N>...</source_N>`)
- ⚠️ **Qwen-style chat** - Instruction format (`<|im_start|>`, `<|im_end|>`)
- ⚠️ **Multi-turn reasoning** - Rolling thinking approach documented

> 📖 See [REASONING_MODEL_ANALYSIS.md](docs/REASONING_MODEL_ANALYSIS.md) for detailed analysis of reasoning capabilities

### Implementation
- ✅ **SYNTH** dataset support (~200B tokens)
- ✅ Comprehensive test suite with 100% coverage
- ✅ Training pipeline (Trainer, optimizers, schedulers)
- ✅ Generation utilities (top-k, top-p, temperature)

## 📊 Architecture

```
BaguettotronForCausalLM (320.96M params)
├── Embeddings (65536 × 576)                    37.75M
├── 80 × Transformer Blocks
│   ├── RMSNorm (pre-norm, eps=1e-5)
│   ├── GQAttention (9 heads, 3 KV, RoPE)      70.78M
│   ├── RMSNorm (pre-norm, eps=1e-5)
│   └── SwiGLU MLP (3 projections)             212.34M
├── Final RMSNorm                                0.09M
└── LM Head (tied with embeddings)                  0
                                              ─────────
                                        Total: 320.96M
```

## 🚀 Installation

```bash
# Clone the repository
git clone <repo-url>
cd Baguettotron-321M

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install with training dependencies (includes datasets, transformers, etc.)
pip install -e ".[train]"

# Or with all dependencies (dev + train)
pip install -e ".[all]"

# Basic install (model only, no training/data tools)
pip install -e .
```

**What's included:**
- **Base install** (`pip install -e .`): Core model, inference only
- **`[train]`**: Adds `transformers`, `datasets`, `tensorboard`, `wandb` for training
- **`[dev]`**: Adds testing and linting tools
- **`[all]`**: Everything (train + dev)

## 📦 Data Preparation

### Real Training: SYNTH Dataset (Recommended)

Baguettotron was trained on the **PleIAs/SYNTH** dataset (~200B tokens). Download it directly from HuggingFace:

**Option 1: Pre-tokenized (Fastest for Training)**

Pre-tokenizing saves time during training since tokenization happens once during preparation:

```bash
# Download and tokenize SYNTH (requires pip install -e ".[train]")
python prepare_synth_data.py --tokenize --split train

# For testing with a subset first
python prepare_synth_data.py --tokenize --subset --max-samples 10000
```

This creates `data/train_tokens.json` in the format:
```json
[
  [1, 2, 3, 4, 5, ...],     // Token IDs for first example
  [10, 11, 12, 13, ...],    // Token IDs for second example
  ...
]
```

**Option 2: Raw Text (Tokenize During Training)**

If you prefer to tokenize on-the-fly:

```bash
# Download SYNTH as JSONL
python prepare_synth_data.py --format jsonl --split train
```

Then use with `SYNTHDataset`:
```python
from transformers import AutoTokenizer
from baguettotron.data import SYNTHDataset, create_dataloader

tokenizer = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")
dataset = SYNTHDataset(
    data_path='data/train.jsonl',
    tokenizer=tokenizer,
    block_size=2048
)
dataloader = create_dataloader(dataset, batch_size=32)
```

### Your Own Data

To preprocess your own text corpus:

```python
from transformers import AutoTokenizer
from baguettotron.data import preprocess_text_file

tokenizer = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")
preprocess_text_file(
    input_path='your_corpus.txt',
    output_path='data/train_tokens.json',
    tokenizer=tokenizer,
    block_size=2048,
    stride=1024  # 50% overlap
)
```

### Quick Testing (Without SYNTH Download)

For quickly testing the training pipeline without downloading SYNTH:

```bash
# Create synthetic random data for testing
python prepare_demo_data.py --num-train 1000 --num-eval 100
```

⚠️ **This is only for pipeline testing, not real training!**

## 💻 Usage

### Quick Start with Python API

```python
from baguettotron import BaguettotronForCausalLM, BaguettotronConfig
from baguettotron.training import Trainer, create_optimizer, create_scheduler
from baguettotron.data import TextDataset, create_dataloader

# Create official 321M model
config = BaguettotronConfig.baguettotron_321m()
model = BaguettotronForCausalLM(config)

# Load data
train_dataset = TextDataset('data/train.json', block_size=2048)
train_dataloader = create_dataloader(train_dataset, batch_size=32, shuffle=True)

# Create optimizer and scheduler
optimizer = create_optimizer(model, learning_rate=1e-4, weight_decay=0.1)
scheduler = create_scheduler(optimizer, 'cosine', num_warmup_steps=1000, num_training_steps=10000)

# Train
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    optimizer=optimizer,
    scheduler=scheduler,
    device='cuda',
    max_epochs=10,
    output_dir='outputs'
)
trainer.train()

# Generate
import torch
input_ids = torch.tensor([[1, 2, 3]])  # Your tokenized prompt
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    do_sample=True
)
```

### Training from Command Line

**Quick test (tiny model):**
```bash
python train.py \
  --model-config tiny \
  --train-data data/train.json \
  --epochs 1 \
  --batch-size 8 \
  --device cuda
```

**Full training (321M model with RTX 3090, 24GB):**
```bash
python train.py \
  --config configs/train_synth_3090.yaml
```

**Custom training:**
```bash
python train.py \
  --model-config 321m \
  --train-data data/train_tokens.json \
  --eval-data data/eval_tokens.json \
  --epochs 10 \
  --batch-size 32 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --weight-decay 0.1 \
  --scheduler cosine \
  --warmup-steps 1000 \
  --mixed-precision \
  --output-dir outputs/my_run
```

**Resume from checkpoint:**
```bash
python train.py \
  --config configs/train_synth_3090.yaml \
  --checkpoint outputs/checkpoint_10000.pt
```

### Generation from Command Line

**Interactive mode:**
```bash
python generate.py \
  --checkpoint outputs/model.pt \
  --config 321m \
  --interactive
```

**Single prompt:**
```bash
python generate.py \
  --checkpoint outputs/model.pt \
  --config 321m \
  --prompt "What is the capital of France?" \
  --max-new-tokens 100 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.9
```

**Greedy decoding:**
```bash
python generate.py \
  --checkpoint outputs/model.pt \
  --config 321m \
  --prompt "The answer is" \
  --greedy \
  --max-new-tokens 50
```

## ✅ Testing & Validation

**Validate structure (without PyTorch):**
```bash
python validate_structure.py
```

**Run all tests:**
```bash
pytest tests/ -v
```

**Run specific test modules:**
```bash
pytest tests/test_config.py -v      # Configuration tests
pytest tests/test_model.py -v       # Model architecture tests
pytest tests/test_mgqa.py -v        # Masked GQA tests
pytest tests/test_data.py -v        # Dataset & collator tests
pytest tests/test_training.py -v    # Trainer & optimizer tests
pytest tests/test_generation.py -v  # Generation utilities tests
```

**Check test coverage:**
```bash
pytest tests/ --cov=src/baguettotron --cov-report=html
```

## 📁 Project Structure

```
Baguettotron-321M/
├── src/baguettotron/                    # Main installable package
│   ├── __init__.py                      # Main exports
│   ├── config.py                        # BaguettotronConfig (dataclass)
│   ├── model/                           # Model components
│   │   ├── __init__.py
│   │   ├── causal_lm.py                # BaguettotronForCausalLM
│   │   ├── transformer.py              # TransformerBlock, TransformerDecoder
│   │   ├── attention.py                # GQA + MaskedGroupQueryAttention
│   │   ├── feedforward.py              # SwiGLU MLP
│   │   ├── normalization.py            # RMSNorm
│   │   └── rope.py                     # RotaryEmbedding
│   ├── data/                           # Data management
│   │   ├── dataset.py                  # TextDataset, SYNTHDataset
│   │   └── collator.py                 # Data collators
│   ├── training/                       # Training utilities
│   │   ├── trainer.py                  # Trainer class
│   │   ├── optimizer.py                # Optimizers with layer decay
│   │   └── scheduler.py                # LR schedulers (cosine, linear, etc.)
│   └── generation/                     # Generation utilities
│       └── utils.py                    # Sampling (top-k, top-p, typical)
├── train.py                            # Training CLI script
├── generate.py                         # Generation CLI script
├── validate_structure.py               # Structure validation
├── tests/                              # Test suite
│   ├── conftest.py                     # Pytest fixtures
│   ├── test_config.py                  # Configuration tests
│   ├── test_model.py                   # Complete model tests
│   ├── test_mgqa.py                    # MGQA tests
│   ├── test_data.py                    # Data module tests
│   ├── test_training.py                # Training module tests
│   └── test_generation.py              # Generation utilities tests
├── configs/                            # YAML configurations
│   ├── model_official_baguettotron_321m.yaml
│   └── train_synth_3090.yaml
├── setup.py                            # Package installation
├── pyproject.toml                      # Modern Python configuration
├── README.md                           # This file
└── whitepaper.md                       # Technical whitepaper
```

## 🔍 Technical Details

### Official Configuration

Exact correspondence with [config.json](https://huggingface.co/PleIAs/Baguettotron/blob/main/config.json):

| HuggingFace Parameter | Our Code | Value |
|----------------------|----------|-------|
| `vocab_size` | `vocab_size` | 65536 |
| `hidden_size` | `hidden_size` | 576 |
| `num_hidden_layers` | `num_hidden_layers` | 80 |
| `num_attention_heads` | `num_attention_heads` | 9 |
| `num_key_value_heads` | `num_key_value_heads` | 3 |
| `intermediate_size` | `intermediate_size` | 1536 |
| `max_position_embeddings` | `max_position_embeddings` | 4096 |
| `rope_theta` | `rope_theta` | 10000 |
| `rms_norm_eps` | `rms_norm_eps` | 1e-05 |
| `tie_word_embeddings` | `tie_word_embeddings` | true |

### SwiGLU MLP

Formula: `down_proj(SiLU(gate_proj(x)) * up_proj(x))`

Identical to `LlamaMLP` from Hugging Face Transformers.

### SYNTH Dataset

- **Source**: [PleIAs/SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH)
- **Size**: ~200B tokens (~75B words)
- **Languages**: en, fr, de, it, es, pl, nl, la, etc.
- **Format**: Qwen-style chat with `<|im_start|>` / `<|im_end|>`
- **Special tags**:
  - `<think>...</think>` : Reasoning traces
  - `<source_N>...</source_N>` : RAG sources

Example generated format:
```
<|im_start|>user
What is the capital of France?
<|im_end|>
<|im_start|>assistant
<think>I need to recall European geography</think>
<source_1>https://wikipedia.org/France</source_1>
The capital of France is Paris.
```

## 📚 Documentation

- **[DISCLAIMER.md](DISCLAIMER.md)** - ⚠️ Important legal and ethical disclaimer (READ FIRST)
- [AUTHORS.md](AUTHORS.md) - Author information and project nature
- **[REASONING_MODEL_ANALYSIS.md](docs/REASONING_MODEL_ANALYSIS.md)** - Analysis of reasoning capabilities
- [whitepaper.md](whitepaper.md) - Technical whitepaper
- [LICENSE](LICENSE) - Apache 2.0 license
- [docs/](docs/) - Additional documentation and reports

## 🔗 References

- **This Repository**: https://github.com/JacquesGariepy/AI-From-Scratch/tree/main/Baguettotron-321M
- **Official Model (PleIAs)**: https://huggingface.co/PleIAs/Baguettotron
- **SYNTH Dataset**: https://huggingface.co/datasets/PleIAs/SYNTH
- **Architecture**: LlamaForCausalLM compatible
- **Technical Whitepaper**: See whitepaper.md

## 📝 License

Apache 2.0

## 🙏 Credits & Attribution

**Author**: Jacques Gariépy ([GitHub](https://github.com/JacquesGariepy))
- Independent reverse engineering of model architecture
- Code written entirely from scratch for educational purposes
- **NOT AFFILIATED with PleIAs or the Baguettotron team in any way**
- **ZERO collaboration, assistance, or communication from the original creators**

**Analyzed Model**: PleIAs/Baguettotron (for reference only)
- [Official Model on HuggingFace](https://huggingface.co/PleIAs/Baguettotron) - Original weights and model
- [SYNTH Dataset](https://huggingface.co/datasets/PleIAs/SYNTH) - Training corpus (~200B tokens)
- Architecture reverse-engineered from publicly available config.json and model structure
- **No proprietary code, documentation, or insider information was accessed**

**Architecture Inspiration**: Meta's LlamaForCausalLM

---

**⚠️ Critical**: This is an **independent reverse engineering project with NO AFFILIATION to PleIAs**. The Baguettotron/PleIAs team has not shared, provided, assisted, endorsed, or communicated with this project in any way. This implementation is based solely on publicly available information (model architecture, config.json) on HuggingFace. For official support and production use, please use the [official PleIAs/Baguettotron model](https://huggingface.co/PleIAs/Baguettotron).
