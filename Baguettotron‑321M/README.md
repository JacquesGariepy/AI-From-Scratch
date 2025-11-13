# Baguettotron-321M

[![Not Affiliated](https://img.shields.io/badge/⚠️%20NOT%20AFFILIATED-with%20PleIAs-red)](DISCLAIMER.md)
[![Status](https://img.shields.io/badge/status-Educational-yellow)](DISCLAIMER.md)
[![Author](https://img.shields.io/badge/author-Jacques%20Gariépy-blue)](https://github.com/JacquesGariepy)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)
[![Parameters](https://img.shields.io/badge/parameters-320.96M-blue)](#architecture-overview)

**A 321M parameter Qwen/Llama-like transformer model built from scratch for educational purposes.**

Learn how to implement, train, and deploy a modern large language model (LLM) with production-ready code, comprehensive tests, and detailed documentation. This project provides a complete, working implementation of a transformer-based causal language model compatible with the LlamaForCausalLM architecture.

> **⚠️ Important**: This is an **independent reverse engineering project** created by Jacques Gariépy for educational purposes. It is **NOT AFFILIATED** with PleIAs or the Baguettotron team. The implementation is based solely on publicly available architecture information. See [DISCLAIMER.md](DISCLAIMER.md) for details.

---

## 🎯 Overview

This Baguettotron-321M-like is designed to help students, researchers, and ML engineers understand how modern language models work by providing:

- **Clean, readable code** with extensive documentation and type hints
- **Production-ready architecture** using state-of-the-art techniques (GQA, SwiGLU, RoPE)
- **Complete training pipeline** with multi-dataset support and intelligent checkpoint management
- **Comprehensive test suite** with 90%+ coverage
- **Educational resources** including architecture guides and a technical whitepaper

Whether you're learning about transformers for the first time or implementing your own LLM, this project provides a solid foundation with real, working code you can study, modify, and extend.

---

## ✨ Key Features

### 🏗️ Modern Architecture
- **320.96M parameters** - Qwen/Llama-like causal language model
- **80 deep layers** - "Baguette" architecture optimized for reasoning tasks
- **Grouped-Query Attention (GQA)** - 9 query heads, 3 KV heads for efficiency
- **SwiGLU MLP** - State-of-the-art activation function (gate_proj, up_proj, down_proj)
- **RoPE embeddings** - Rotary Position Embeddings for better position encoding
- **RMSNorm** - Faster pre-normalization (eps=1e-5)
- **Tied embeddings** - Shared input/output weights
- **100% LlamaForCausalLM compatible** - Drop-in replacement for Hugging Face models

### 🚀 Training Infrastructure
- **Multi-dataset training** - Train on multiple datasets simultaneously with weighted sampling
- **Intelligent checkpoint system** - Organized structure with automatic rotation
- **YAML configuration** - Easy configuration management
- **Auto-download datasets** - Wikipedia and SYNTH datasets download automatically
- **Mixed precision training** - Automatic Mixed Precision (AMP) for faster training
- **Gradient accumulation** - Simulate larger batch sizes
- **Learning rate scheduling** - Cosine, linear, and constant schedules with warmup
- **TensorBoard & W&B support** - Track training metrics in real-time

### 📊 Dataset Support
- **SYNTH dataset** (~200B tokens) - Official training corpus
- **Wikipedia** (Simple English, French) - For development and testing
- **Demo dataset** - Synthetic data for quick testing
- **Custom datasets** - Easy integration with your own data
- **Multi-dataset mixing** - Train on multiple datasets with weighted sampling

### 🧪 Educational Resources
- **Comprehensive documentation** - Architecture guide, dataset guide, quick start
- **Technical whitepaper** - Deep dive into model design, chat templates, and implementation ([whitepaper.md](whitepaper.md))
- **Chat mode support** - ChatML format for interactive conversations (see [whitepaper.md](whitepaper.md) §3.2)
- **90%+ test coverage** - Learn from working examples and tests
- **CLI and Python API** - Flexible usage for all skill levels
- **Production-ready code** - Modern Python with type hints and best practices

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/JacquesGariepy/AI-From-Scratch
cd AI-From-Scratch/Baguettotron-321M

# Create and activate virtual environment
python3 -m venv ~/.venvs/baguettotron
source ~/.venvs/baguettotron/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install with training dependencies + Wikipedia dataset
pip install -e ".[train,wikipedia]"
```

This will:
- ✅ Install all dependencies
- ✅ Auto-download Wikipedia Simple English (200MB)
- ✅ Set up the CLI tool
- ✅ Make you ready to train!

### Quick Smoke Test (40 seconds)

Test your installation with the fastest smoke test:

```bash
# Run basic smoke test (tests core pipeline)
./baguettotron train --config-file configs/train_321m_smoke_test.yaml
```

**What this tests**: Model loading (321M), data loading (100 samples), training (100 steps), checkpointing.

**Output**: You should see:
```
Model parameters: 321.0M ✅
Training examples: 100
Epoch 1/1: 100%|███████| 50/50 [00:40<00:00, 1.8it/s]
Training completed in 40s ✅
```

### Test HuggingFace Integration (1-2 minutes)

Test HuggingFace dataset loading with a small sample:

```bash
# Test HF integration (50 Wikipedia articles)
./baguettotron train --config-file configs/train_321m_smoke_test_hf.yaml
```

**First run**: Downloads and tokenizes 50 Wikipedia articles (~1-2 minutes)
**Subsequent runs**: Uses cache (30 seconds)

### Your First Training Run (5 minutes)

```bash
# 1. Prepare demo dataset for testing
./baguettotron dataset prepare --type demo

# 2. Train a tiny model
./baguettotron train \
  --data data/train.json \
  --config tiny \
  --epochs 1 \
  --batch-size 8

# 3. Generate text
./baguettotron generate \
  --checkpoint outputs/ckpt_*/model.pt \
  --prompt "Hello, I am" \
  --max-length 50
```

### Realistic Training with Wikipedia (30 minutes)

```bash
# 1. Dataset is already downloaded during install!
# Or manually: ./baguettotron dataset prepare --type wikipedia --tokenize

# 2. Train with Wikipedia
./baguettotron train \
  --datasets wikipedia \
  --config 321m \
  --epochs 5 \
  --batch-size 4 \
  --warmup-steps 500 \
  --save-steps 1000

# 3. Generate text with trained model
./baguettotron generate \
  --checkpoint outputs/ckpt_*/model.pt \
  --prompt "The capital of France is" \
  --temperature 0.8
```

For more detailed guides, see:
- **[QUICKSTART.md](QUICKSTART.md)** - Complete step-by-step guide with 3 scenarios
- **[docs/INSTALLATION_GUIDE.md](docs/INSTALLATION_GUIDE.md)** - Detailed installation instructions
- **[docs/DATASET_GUIDE.md](docs/DATASET_GUIDE.md)** - Complete dataset documentation

### 🎮 RTX 3090 Training (Optimized Configurations)

Train models on NVIDIA RTX 3090 (24GB VRAM) with memory-optimized YAML configurations:

#### **Quick Smoke Test** (Ultra-Fast Validation)
```bash
# Tiny model for instant testing (2 layers, 64 hidden, ~17M params)
python scripts/train.py --config-file configs/tiny_smoke.yaml

# Expected: ~200+ steps/sec, ~6-8GB VRAM, completes in ~1 minute
# Perfect for: Testing code changes, CI/CD, quick validation
```

**Config**: `tiny_smoke.yaml`
- Model: **2 layers, 64 hidden** (~17M parameters)
- Batch: 4 + grad accumulation 8 = **effective 32**
- Sequence length: **256 tokens**
- Memory: **~6-8GB VRAM**

#### **Small Model Training** (Fast Development)
```bash
# Small model for development and prototyping (4 layers, 256 hidden, ~67M params)
python scripts/train.py --config-file configs/small_rtx3090.yaml

# Expected: ~50-100 steps/sec, ~12-16GB VRAM
# Perfect for: Development, prototyping, feature testing
```

**Config**: `small_rtx3090.yaml`
- Model: **4 layers, 256 hidden** (~67M parameters)
- Batch: 2 + grad accumulation 16 = **effective 32**
- Sequence length: **512 tokens**
- Memory: **~12-16GB VRAM**
- Mixed precision: **Enabled**

#### **Full 321M Model** (Production Training)
```bash
# Train with Simple Wikipedia dataset (compatible with current script)
python scripts/train.py --config-file configs/train_321m_rtx3090_simple.yaml

# Expected: ~2.5-3.5 steps/sec, ~19-20GB VRAM
```

**Config**: `train_321m_rtx3090_simple.yaml`
- Model: **321M parameters** (80 layers, 576 hidden)
- Batch: 40 + grad accumulation 4 = **effective 160**
- Sequence length: **1024 tokens**
- Mixed precision: **BF16**
- Gradient checkpointing: **Yes**
- Dataset: **Local file** (data/wikipedia_simple_tokens.json)

**Resume Training from Checkpoint**:
```bash
# Continue training from a saved checkpoint (all 3 methods work)

# Method 1: Using YAML config (recommended - simplest)
python scripts/train.py \
  --config-file configs/train_321m_rtx3090_simple.yaml \
  --checkpoint outputs/rtx3090_simple/ckpt_10000 \
  --epochs 5

# Method 2: Using command-line arguments
python scripts/train.py \
  --train-data data/wikipedia_simple_tokens.json \
  --model-config 321m \
  --checkpoint outputs/rtx3090_simple/ckpt_10000 \
  --epochs 5 \
  --batch-size 40 \
  --gradient-accumulation-steps 4 \
  --block-size 1024 \
  --mixed-precision \
  --compile

# Method 3: Continue in new output directory (keeps old checkpoints separate)
python scripts/train.py \
  --config-file configs/train_321m_rtx3090_simple.yaml \
  --checkpoint outputs/rtx3090_simple/ckpt_10000 \
  --output-dir outputs/rtx3090_simple_continued \
  --epochs 5

# Note: When resuming, the trainer will:
# - Restore model weights from model.pt
# - Restore optimizer state from trainer_state.pt
# - Continue from the saved step count
# - Restore learning rate schedule state

# Real example: Continue from epoch 1 to epoch 5
python scripts/train.py \
  --train-data data/wikipedia_simple_tokens.json \
  --model-config 321m \
  --checkpoint outputs/test_321m/ckpt_99963 \
  --epochs 5 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --block-size 512 \
  --mixed-precision \
  --output-dir outputs/test_321m \
  --compile

# Expected: Loss should improve from ~1.9 to ~1.2-1.5 after 5 total epochs
```

#### **Multi-Dataset Wikipedia** (Advanced)
```bash
# Automatically downloads and mixes 3 Wikipedia datasets
# Note: Requires compatible dataset loader in train.py
python scripts/train.py --config-file configs/train_321m_rtx3090_wikipedia.yaml

# Downloads:
#   - 50K English articles (50% weight)
#   - 20K Simple English articles (30% weight)
#   - 30K French articles (20% weight)
# Total: ~80K articles, ~2-3GB, auto-mixed by weights
```

**Config**: `train_321m_rtx3090_wikipedia.yaml`
- Multi-dataset: **EN + Simple + FR** (auto-download from HuggingFace)
- Batch: 40 + grad accumulation 4 = **effective 160**
- Memory: **~19-20GB VRAM**

#### **Memory Optimization Guide**

If you encounter OOM (Out of Memory):

```bash
# Option 1: Start with tiny model for testing
python scripts/train.py --config-file configs/tiny_smoke.yaml

# Option 2: Use small model for development
python scripts/train.py --config-file configs/small_rtx3090.yaml

# Option 3: Reduce batch size for 321M model
python scripts/train.py \
  --train-data data/wikipedia_simple_tokens.json \
  --model-config 321m \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --block-size 512 \
  --mixed-precision \
  --output-dir outputs/rtx3090
```

**Memory Usage Comparison**:
| Config | Model Size | Seq Len | Memory | Speed | Use Case |
|--------|-----------|---------|--------|-------|----------|
| `tiny_smoke.yaml` | 17M (2 layers) | 256 | ~6-8GB | 200+ steps/sec | **Quick testing** |
| `small_rtx3090.yaml` | 67M (4 layers) | 512 | ~12-16GB | 50-100 steps/sec | **Development** |
| `train_321m_rtx3090_simple.yaml` | 321M (80 layers) | 1024 | ~19-20GB | 2.5-3.5 steps/sec | **Production** |
| `train_321m_rtx3090_wikipedia.yaml` | 321M (80 layers) | 1024 | ~19-20GB | 2.5-3.5 steps/sec | **Multi-dataset** |

📚 **Complete RTX 3090 Documentation**:
- **[configs/RTX3090_GUIDE.md](configs/RTX3090_GUIDE.md)** - Complete RTX 3090 training guide (if exists)
- **[configs/WIKIPEDIA_RTX3090_QUICKSTART.md](configs/WIKIPEDIA_RTX3090_QUICKSTART.md)** - Wikipedia quick start (if exists)

---

## 🏗️ Architecture Overview

### Model Structure

```
BaguettotronForCausalLM (320.96M params)
├── Token Embeddings (65536 × 576)              37.75M params
├── 80 × Transformer Blocks                    283.12M params
│   ├── RMSNorm (pre-norm, eps=1e-5)
│   ├── Grouped-Query Attention
│   │   ├── 9 query heads (Q)
│   │   ├── 3 key-value heads (KV)
│   │   └── RoPE position embeddings
│   ├── RMSNorm (pre-norm, eps=1e-5)
│   └── SwiGLU MLP
│       ├── gate_proj (hidden_size → intermediate_size)
│       ├── up_proj (hidden_size → intermediate_size)
│       └── down_proj (intermediate_size → hidden_size)
├── Final RMSNorm                                0.09M params
└── LM Head (tied with token embeddings)              0 params
                                              ═══════════════
                                        Total: 320.96M params
```

### Technical Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Architecture** | LlamaForCausalLM | HuggingFace compatible |
| **Total Parameters** | 320.96M | Production-ready size |
| **Vocabulary Size** | 65,536 | BPE tokenizer |
| **Hidden Size** | 576 | Model dimension |
| **Number of Layers** | 80 | Deep architecture |
| **Attention Heads** | 9 query, 3 KV | Grouped-Query Attention |
| **Head Dimension** | 64 | Per-head size |
| **Intermediate Size** | 1,536 | MLP hidden dimension |
| **Max Context Length** | 4,096 tokens | Sequence length |
| **Position Encoding** | RoPE (θ=10000) | Rotary embeddings |
| **Activation** | SwiGLU (SiLU) | MLP activation |
| **Normalization** | RMSNorm (eps=1e-5) | Pre-normalization |
| **Precision** | bfloat16 | Mixed precision training |

### Why This Architecture?

1. **Grouped-Query Attention (GQA)**: Reduces KV cache memory by 3x while maintaining quality
2. **SwiGLU Activation**: Better performance than ReLU/GELU on language tasks
3. **80 Layers**: Deep architecture provides better reasoning and compositional abilities
4. **RoPE Embeddings**: Superior position encoding that generalizes to longer sequences
5. **RMSNorm**: Faster and more stable than LayerNorm
6. **Tied Embeddings**: Reduces parameters and improves efficiency

For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 📚 Training Guide

### Training Configurations Overview

Baguettotron includes **multiple YAML configurations** for different use cases. All configurations are in the `configs/` directory:

| Configuration | Time | Features | HuggingFace | Use Case |
|--------------|------|----------|-------------|----------|
| **`train_321m_smoke_test.yaml`** | ~40s | Basic pipeline | ❌ No | **Daily testing** - Quick verification after code changes |
| **`train_321m_smoke_test_hf.yaml`** | ~1-2 min | HF integration | ✅ Yes (50 samples) | **Test HF** - Validate HuggingFace dataset loading |
| **`train_321m_smoke_test_full.yaml`** | ~2-3 min | All features (150+ params) | ✅ Yes (100 samples) | **Pre-production** - Comprehensive validation before long runs |
| **`train_321m_smoke_test_full_fast.yaml`** | ~40s | All features | ❌ No | **CI/CD** - Fast comprehensive test without HF |
| **`train_321m_synth.yaml`** | ~40 hours | Production SYNTH | ✅ Yes (200B tokens) | **Production** - Official SYNTH dataset replication |
| **`train_321m_huggingface_example.yaml`** | Variable | Multi-dataset HF | ✅ Yes (Wikipedia) | **Example** - HuggingFace multi-language training |
| **`train_321m_rtx3090.yaml`** | Variable | Memory-optimized | Optional | **RTX 3090** - Optimized for 24GB VRAM |

#### Configuration Details

**Smoke Tests** (Fast validation):
- **`train_321m_smoke_test.yaml`**: Minimal test with 100 local samples, 100 steps. Perfect for quick "does it work?" checks.
- **`train_321m_smoke_test_hf.yaml`**: Tests HuggingFace integration with 50 Wikipedia articles. First run tokenizes (~1 min), subsequent runs use cache (30s).
- **`train_321m_smoke_test_full.yaml`**: Tests ALL features (150+ YAML parameters, all optimizers, all schedulers, all precision modes) with 100 Wikipedia articles. Complete pre-production validation.
- **`train_321m_smoke_test_full_fast.yaml`**: Same as full but without HuggingFace for instant testing.

**Production Configs**:
- **`train_321m_synth.yaml`**: Official replication config for PleIAs/SYNTH dataset (~200B tokens). Requires 16×H100 or similar hardware. Estimated training time: 40 hours.
- **`train_321m_huggingface_example.yaml`**: Example multi-language training with Wikipedia EN+FR from HuggingFace Hub.
- **`train_321m_rtx3090.yaml`**: Memory-optimized configuration for RTX 3090 (24GB VRAM) with gradient checkpointing and smaller batch sizes.

For detailed smoke test documentation, see **[docs/SMOKE_TESTS_GUIDE.md](docs/SMOKE_TESTS_GUIDE.md)**.

For HuggingFace integration guide, see **[docs/HUGGINGFACE_INTEGRATION.md](docs/HUGGINGFACE_INTEGRATION.md)**.

### Training with Multiple Datasets

Baguettotron supports training on multiple datasets simultaneously with weighted sampling:

```bash
# Train on SYNTH (70%) + Wikipedia (30%)
./baguettotron train \
  --datasets synth,wikipedia \
  --dataset-weights 0.7,0.3 \
  --config 321m \
  --epochs 10 \
  --batch-size 32 \
  --warmup-steps 2000 \
  --save-steps 1000 \
  --save-total-limit 5
```

### Using YAML Configuration (Recommended)

For reproducible training runs, use YAML configuration files:

```bash
./baguettotron train --config-file configs/multi_dataset.yaml
```

Example `configs/multi_dataset.yaml`:

```yaml
# Model configuration
model:
  config: "321m"  # or "tiny" for testing

# Dataset configuration
data:
  datasets:
    - name: "synth"
      weight: 0.7
    - name: "wikipedia"
      weight: 0.3

# Training hyperparameters
training:
  epochs: 10
  batch_size: 32
  learning_rate: 1.0e-4
  warmup_steps: 2000
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0

# Checkpoint management
checkpoint:
  save_steps: 1000
  save_total_limit: 5
  output_dir: "outputs/production"
```

### Advanced Training Options

```bash
# Full 321M model training with all features
./baguettotron train \
  --config-file configs/train_example.yaml \
  --datasets synth,wikipedia \
  --dataset-weights 0.7,0.3 \
  --config 321m \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --warmup-steps 2000 \
  --gradient-accumulation-steps 4 \
  --save-steps 1000 \
  --save-total-limit 5 \
  --output-dir outputs/my_run \
  --use-wandb \
  --wandb-project "baguettotron"
```

### Checkpoint Management

Baguettotron uses an organized checkpoint structure:

```
outputs/
├── ckpt_1000/
│   ├── model.pt              # Model weights
│   ├── trainer_state.pt      # Optimizer, scheduler, step count
│   └── config.json           # Model configuration
├── ckpt_2000/
│   └── ...
└── ckpt_3000/
    └── ...
```

**Features:**
- Separate model and trainer state for clarity
- Automatic checkpoint rotation with `--save-total-limit`
- Easy resumption from any checkpoint
- Full training state preservation

### Resuming Training

```bash
# Resume from a checkpoint
./baguettotron train \
  --config-file configs/multi_dataset.yaml \
  --checkpoint outputs/production/ckpt_5000
```

---

## 🎨 Text Generation

### Command-Line Generation

```bash
# Interactive mode
./baguettotron generate \
  --checkpoint outputs/ckpt_10000/model.pt \
  --interactive

# Single prompt generation
./baguettotron generate \
  --checkpoint outputs/ckpt_10000/model.pt \
  --prompt "The capital of France is" \
  --max-length 100 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.9

# Greedy decoding (deterministic)
./baguettotron generate \
  --checkpoint outputs/ckpt_10000/model.pt \
  --prompt "Once upon a time" \
  --greedy \
  --max-length 200
```

### Python API

```python
from baguettotron import BaguettotronForCausalLM, BaguettotronConfig
from transformers import AutoTokenizer
import torch

# Load model
config = BaguettotronConfig.baguettotron_321m()
model = BaguettotronForCausalLM(config)
model.load_state_dict(torch.load('outputs/ckpt_10000/model.pt'))
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")

# Generate text
prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    do_sample=True
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

---

## 💬 Chat Mode & Conversation

Baguettotron-321M supports **ChatML format** for interactive conversations, as described in the [whitepaper](whitepaper.md#32-conversation-template-chat_templatejson). The chat template uses special tokens (`<|im_start|>`, `<|im_end|>`, `<think>`) to structure multi-turn conversations with proper role separation.

### Quick Start with Chat Mode

```bash
# Interactive conversation mode
./baguettotron generate \
  --checkpoint outputs/ckpt_10000/model.pt \
  --chat-mode \
  --interactive \
  --system-prompt "Tu es un assistant IA expert."

# Single chat message
./baguettotron generate \
  --checkpoint outputs/ckpt_10000/model.pt \
  --chat-mode \
  --prompt "Explique-moi les transformers" \
  --max-length 200
```

### Chat Template Features

- **ChatML Format**: Industry-standard conversation formatting compatible with OpenAI and other models
- **Role Support**: System, user, and assistant roles for clear context
- **Reasoning Tag**: `<think>` token encourages the model to reason before responding
- **Multi-turn Conversations**: Maintains conversation history for context-aware responses
- **Automatic Fallback**: Works even without the template file (simple format)

### Python API for Chat

```python
from baguettotron.tokenization import load_tokenizer

# Load tokenizer with chat template
tokenizer = load_tokenizer("auto", model_vocab_size=65536)

# Format a conversation
messages = [
    {"role": "system", "content": "Tu es un assistant IA."},
    {"role": "user", "content": "Bonjour !"}
]

# Apply chat template
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
)

# Generate response
output = model.generate(input_ids, max_new_tokens=100)
response = tokenizer.decode(output[0], skip_special_tokens=True)
```

**For detailed chat mode documentation**, see:
- **Quick Start**: [docs/CHAT_MODE_QUICKSTART.md](docs/CHAT_MODE_QUICKSTART.md)
- **Complete Guide**: [docs/CHAT_TEMPLATE_GUIDE.md](docs/CHAT_TEMPLATE_GUIDE.md)
- **Technical Details**: See [whitepaper.md](whitepaper.md) Section 3.2 and Appendix D.3

---

## 📁 Project Structure

```
Baguettotron-321M/
├── baguettotron              # Main CLI entry point (executable)
├── Makefile                  # Build automation
├── setup.py                  # Package setup with intelligent dataset management
├── pyproject.toml            # Modern Python packaging
├── requirements.txt          # Core dependencies
│
├── src/baguettotron/         # Source code (installable package)
│   ├── __init__.py           # Package exports
│   ├── config.py             # BaguettotronConfig (dataclass)
│   ├── model/                # Model architecture
│   │   ├── causal_lm.py      # BaguettotronForCausalLM (main model)
│   │   ├── transformer.py    # TransformerBlock, TransformerDecoder
│   │   ├── attention.py      # Grouped-Query Attention + RoPE
│   │   ├── feedforward.py    # SwiGLU MLP
│   │   ├── normalization.py  # RMSNorm
│   │   └── rope.py           # Rotary Position Embeddings
│   ├── data/                 # Data management
│   │   ├── dataset.py        # TextDataset, SYNTHDataset, multi-dataset support
│   │   ├── collator.py       # Data collators
│   │   └── auto_download.py  # Automatic dataset downloading
│   ├── training/             # Training infrastructure
│   │   ├── trainer.py        # Trainer class with AMP, checkpointing
│   │   ├── optimizer.py      # Optimizers with layer decay
│   │   └── scheduler.py      # Learning rate schedulers
│   └── generation/           # Generation utilities
│       └── utils.py          # Sampling strategies (top-k, top-p, typical)
│
├── scripts/                  # Executable scripts
│   ├── setup_dataset.py      # Interactive dataset setup
│   ├── validate_dataset.py   # Dataset validation
│   ├── train.py              # Training script
│   └── generate.py           # Generation script
│
├── tests/                    # Comprehensive test suite (90%+ coverage)
│   ├── conftest.py           # Pytest fixtures
│   ├── test_config.py        # Configuration tests
│   ├── test_attention.py     # Attention mechanism tests
│   ├── test_feedforward.py   # MLP tests
│   ├── test_dataset.py       # Data loading tests
│   ├── test_training.py      # Training loop tests
│   └── test_generation.py    # Generation utilities tests
│
├── configs/                  # YAML configuration files
│   ├── train_321m_smoke_test.yaml        # Quick smoke test (40s)
│   ├── train_321m_smoke_test_hf.yaml     # HF integration test (1-2 min)
│   ├── train_321m_smoke_test_full.yaml   # Full features test (2-3 min)
│   ├── train_321m_smoke_test_full_fast.yaml  # Fast comprehensive test (40s)
│   ├── train_321m_synth.yaml             # Production SYNTH training (~40h)
│   ├── train_321m_huggingface_example.yaml   # HF multi-language example
│   └── train_321m_rtx3090.yaml           # RTX 3090 optimized config
│
├── docs/                     # Documentation
│   ├── ARCHITECTURE.md       # Architecture deep dive
│   ├── DATASET_GUIDE.md      # Complete dataset guide
│   ├── INSTALLATION_GUIDE.md # Installation instructions
│   ├── CLI_GUIDE.md          # CLI usage guide
│   └── INDEX.md              # Documentation index
│
├── data/                     # Data directory (auto-created, gitignored)
├── outputs/                  # Training outputs (gitignored)
└── whitepaper.md             # Technical whitepaper
```

### Clean, Modern Organization

- **Unified CLI**: Single entry point (`baguettotron`) for all operations
- **Modular source code**: Clean separation of concerns
- **Comprehensive tests**: 90%+ coverage with fixtures and utilities
- **Rich documentation**: Guides for every aspect of the project
- **YAML configs**: Reproducible training configurations

---

## ⚙️ Configuration

### Available Model Configurations

```python
from baguettotron import BaguettotronConfig

# Official 321M model
config = BaguettotronConfig.baguettotron_321m()

# Tiny model for testing (fits on CPU)
config = BaguettotronConfig.tiny()

# Custom configuration
config = BaguettotronConfig(
    vocab_size=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,
    intermediate_size=2048,
    max_position_embeddings=2048
)
```

### Dataset Installation Options

```bash
# Training + Wikipedia (recommended for development)
pip install -e ".[train,wikipedia]"

# Training + SYNTH (official dataset)
pip install -e ".[train,synth]"

# Multiple datasets
pip install -e ".[train,synth,wikipedia]"

# Demo dataset (quick testing)
pip install -e ".[train,demo]"

# Full installation with all features
pip install -e ".[all,synth,wikipedia]"
```

### Available Datasets

| Dataset | Size | Installation | Use Case |
|---------|------|--------------|----------|
| **Demo** | 5MB | `[demo]` | Quick testing, CI/CD |
| **Wikipedia Simple** | 200MB | `[wikipedia]` | **Recommended for development** |
| **Wikipedia French** | 6GB | Configure `--lang fr` | Production (medium) |
| **SYNTH** | 100MB subset | `[synth]` | **Official training corpus** |
| **SYNTH Full** | 500GB+ | Manual download | Full production training |

---

## 📖 Educational Resources

### Documentation

**Getting Started:**
- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes with 3 ready-to-use scenarios
- **[docs/INSTALLATION_GUIDE.md](docs/INSTALLATION_GUIDE.md)** - Detailed installation instructions
- **[docs/SMOKE_TESTS_GUIDE.md](docs/SMOKE_TESTS_GUIDE.md)** - Complete smoke test guide (4 configurations) ⭐ NEW

**Core Documentation:**
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Deep dive into model architecture
- **[docs/DATASET_GUIDE.md](docs/DATASET_GUIDE.md)** - Complete dataset documentation
- **[docs/HUGGINGFACE_INTEGRATION.md](docs/HUGGINGFACE_INTEGRATION.md)** - HuggingFace datasets integration ⭐ NEW
- **[docs/CLI_GUIDE.md](docs/CLI_GUIDE.md)** - CLI usage and examples

**Chat Mode:**
- **[docs/CHAT_MODE_QUICKSTART.md](docs/CHAT_MODE_QUICKSTART.md)** - Quick start guide for chat mode
- **[docs/CHAT_TEMPLATE_GUIDE.md](docs/CHAT_TEMPLATE_GUIDE.md)** - Complete chat template documentation

**Reference:**
- **[docs/INDEX.md](docs/INDEX.md)** - Documentation index
- **[whitepaper.md](whitepaper.md)** - Technical whitepaper (includes chat template specifications §3.2, Appendix D.3)

### Learning Resources

1. **Start Here**: [QUICKSTART.md](QUICKSTART.md) - Three scenarios from beginner to advanced
2. **Understand the Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - How it works
3. **Work with Data**: [docs/DATASET_GUIDE.md](docs/DATASET_GUIDE.md) - Prepare and use datasets
4. **Try Chat Mode**: [docs/CHAT_MODE_QUICKSTART.md](docs/CHAT_MODE_QUICKSTART.md) - Interactive conversations
5. **Read the Code**: Start with `src/baguettotron/model/causal_lm.py` - Clean, documented implementation
6. **Study the Tests**: `tests/` directory - Learn from working examples
7. **Deep Dive**: [whitepaper.md](whitepaper.md) - Technical details, chat template design, and implementation decisions

### Code Examples

#### Training with Python API

```python
from baguettotron import BaguettotronForCausalLM, BaguettotronConfig
from baguettotron.training import Trainer, create_optimizer, create_scheduler
from baguettotron.data import TextDataset, create_dataloader
import torch

# 1. Create model
config = BaguettotronConfig.baguettotron_321m()
model = BaguettotronForCausalLM(config)

# 2. Load data
train_dataset = TextDataset('data/train.json', block_size=2048)
train_dataloader = create_dataloader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# 3. Create optimizer and scheduler
optimizer = create_optimizer(
    model,
    learning_rate=1e-4,
    weight_decay=0.1
)
scheduler = create_scheduler(
    optimizer,
    'cosine',
    num_warmup_steps=1000,
    num_training_steps=10000
)

# 4. Train
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    optimizer=optimizer,
    scheduler=scheduler,
    device='cuda',
    max_epochs=10,
    output_dir='outputs',
    save_steps=1000
)

trainer.train()
```

#### Multi-Dataset Training

```python
from baguettotron.data import MultiDatasetLoader, TextDataset

# Create multiple datasets
synth_dataset = TextDataset('data/synth_train.json', block_size=2048)
wiki_dataset = TextDataset('data/wikipedia_tokens.json', block_size=2048)

# Combine with weights
multi_loader = MultiDatasetLoader(
    datasets=[synth_dataset, wiki_dataset],
    weights=[0.7, 0.3],
    batch_size=32
)

# Use in training
for batch in multi_loader:
    # Your training loop
    pass
```

---

## ✅ Testing

Baguettotron includes a comprehensive test suite with 90%+ coverage.

### Run All Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src/baguettotron --cov-report=html

# Run specific test modules
pytest tests/test_config.py -v          # Configuration tests
pytest tests/test_attention.py -v       # Attention mechanism
pytest tests/test_feedforward.py -v     # MLP layers
pytest tests/test_dataset.py -v         # Data loading
pytest tests/test_training.py -v        # Training loop
pytest tests/test_generation.py -v      # Text generation

# View coverage report
open htmlcov/index.html  # Linux/Mac
start htmlcov/index.html  # Windows
```

### Using Makefile

```bash
make test              # Run all tests
make test-coverage     # Run tests with coverage report
make test-fast         # Run fast tests only
```

### What's Tested

- ✅ **Configuration**: All config variations and edge cases
- ✅ **Model Architecture**: Attention, MLP, embeddings, full model
- ✅ **Data Loading**: All dataset types, collators, multi-dataset
- ✅ **Training**: Optimizer, scheduler, trainer, checkpointing
- ✅ **Generation**: Sampling strategies, beam search, batching
- ✅ **Integration**: End-to-end training and generation workflows

---

## 🤝 Contributing

We welcome contributions! Whether you're fixing bugs, adding features, improving documentation, or sharing educational resources.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Add code, tests, and documentation
4. **Run tests**: `make test` - Ensure all tests pass
5. **Format code**: `make format` - Follow code style
6. **Commit changes**: `git commit -m "Add amazing feature"`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**: Describe your changes

### Development Setup

```bash
# Clone and install with dev dependencies
git clone https://github.com/JacquesGariepy/AI-From-Scratch
cd AI-From-Scratch/Baguettotron-321M
pip install -e ".[all,wikipedia]"

# Run tests
make test

# Format code
make format

# Type checking
make typecheck
```

### Code Style

- **Type hints**: Use type hints for all functions
- **Docstrings**: Document all public APIs
- **Tests**: Add tests for new features
- **Format**: Follow PEP 8 (enforced by `black` and `isort`)
- **Comments**: Explain complex logic

---

## 📄 Citation

If you use Baguettotron-321M in your research or educational projects, please cite:

```bibtex
@software{baguettotron321m,
  author = {Gariépy, Jacques},
  title = {Baguettotron-321M: Educational Implementation of a 321M Parameter Language Model},
  year = {2024},
  url = {https://github.com/JacquesGariepy/AI-From-Scratch/tree/main/Baguettotron-321M},
  note = {Independent reverse engineering project for educational purposes}
}
```

**Note**: This is an independent educational project. For the official Baguettotron model, please cite:

```bibtex
@software{pleias_baguettotron,
  author = {PleIAs},
  title = {Baguettotron: Small Reasoning Model},
  year = {2024},
  url = {https://huggingface.co/PleIAs/Baguettotron}
}
```

---

## 📝 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

### What This Means

- ✅ **Use freely**: For personal, educational, or commercial projects
- ✅ **Modify**: Adapt the code to your needs
- ✅ **Distribute**: Share your modifications
- ✅ **Private use**: Use privately without disclosure
- ⚠️ **Notice required**: Include the original copyright notice
- ⚠️ **State changes**: Document modifications you make

---

## 🙏 Acknowledgments

### Author

**Jacques Gariépy** ([GitHub](https://github.com/JacquesGariepy))
- Independent reverse engineering of model architecture
- Code written entirely from scratch for educational purposes
- **NOT AFFILIATED** with PleIAs or the Baguettotron team in any way
- **ZERO collaboration, assistance, or communication** from the original creators

### Analyzed Model

**PleIAs/Baguettotron** (for reference only)
- [Official Model on HuggingFace](https://huggingface.co/PleIAs/Baguettotron) - Original weights and implementation
- [SYNTH Dataset](https://huggingface.co/datasets/PleIAs/SYNTH) - Training corpus (~200B tokens)
- Architecture reverse-engineered from publicly available `config.json` and model structure
- **No proprietary code, documentation, or insider information was accessed**

### Architectural Inspiration

- **Meta's LLaMA** - Foundation for modern transformer architectures
- **Hugging Face Transformers** - Reference implementations and tooling
- **PyTorch** - Deep learning framework

### Educational Impact

This project aims to make transformer architectures accessible to everyone by providing:
- Clear, readable code that students can understand
- Comprehensive documentation for self-paced learning
- Working examples that demonstrate best practices
- A foundation for further research and experimentation

---

## 🔗 References

### Official Resources

- **Official Model**: [PleIAs/Baguettotron](https://huggingface.co/PleIAs/Baguettotron) on Hugging Face
- **SYNTH Dataset**: [PleIAs/SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH) - ~200B tokens
- **Wikipedia Datasets**: [Hugging Face Datasets](https://huggingface.co/datasets/wikipedia)

### This Project

- **Repository**: [AI-From-Scratch/Baguettotron-321M](https://github.com/JacquesGariepy/AI-From-Scratch/tree/main/Baguettotron-321M)
- **Documentation**: [docs/](docs/) directory
- **Whitepaper**: [whitepaper.md](whitepaper.md)
- **Issues**: [GitHub Issues](https://github.com/JacquesGariepy/AI-From-Scratch/issues)

### Learning Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

---

## ⚠️ Important Disclaimer

**This is an independent educational project:**

- ✋ **NOT AFFILIATED** with PleIAs or the Baguettotron team in any way
- ✋ **ZERO collaboration or communication** with the original creators
- ✋ **No proprietary information** was accessed or used
- ✋ Based **solely on publicly available** architecture information (`config.json`)
- ✅ Created **entirely from scratch** by Jacques Gariépy for educational purposes
- ✅ All implementation decisions made **independently** through reverse engineering

**For official support and production use**, please visit:
- [Official PleIAs/Baguettotron Model](https://huggingface.co/PleIAs/Baguettotron)

See [DISCLAIMER.md](DISCLAIMER.md) for full legal details.

---

## 💬 Support

### Questions and Discussions

- **Documentation**: Check [docs/INDEX.md](docs/INDEX.md) for all guides
- **Quick Start**: See [QUICKSTART.md](QUICKSTART.md) for step-by-step instructions
- **Issues**: Report bugs on [GitHub Issues](https://github.com/JacquesGariepy/AI-From-Scratch/issues)
- **Educational Questions**: Open a discussion on GitHub

### Common Issues

**"CUDA out of memory"**
```bash
# Reduce batch size
./baguettotron train --batch-size 1 --config tiny

# Or use CPU
./baguettotron train --device cpu --config tiny
```

**"Dataset not found"**
```bash
# List available datasets
./baguettotron dataset list

# Prepare missing dataset
./baguettotron dataset prepare --type wikipedia --tokenize
```

**"Checkpoint directory already exists"**
```bash
# Use a new output directory
./baguettotron train --output-dir outputs/run-$(date +%Y%m%d-%H%M%S)
```

---

## 🎯 Roadmap

### Current Version (v1.0)
- ✅ Complete 321M parameter implementation
- ✅ Multi-dataset training support
- ✅ Intelligent checkpoint management
- ✅ YAML configuration system
- ✅ Comprehensive test suite (90%+ coverage)
- ✅ Full documentation

### Planned Features (v1.1+)
- [ ] Multi-GPU training (Distributed Data Parallel)
- [ ] Flash Attention 2 integration
- [ ] 4-bit and 8-bit quantization
- [ ] LoRA fine-tuning support
- [ ] Model parallelism for larger models
- [ ] Streaming dataset support
- [ ] Web UI for generation
- [ ] Pre-trained checkpoints
- [ ] More example notebooks

### Long-term Vision
- Educational video series
- Interactive tutorials
- Community model zoo
- Integration with popular frameworks
- Performance optimizations

---

**Built with ❤️ for the AI community. Happy learning!**
