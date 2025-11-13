# Baguettotron CLI Guide

Complete guide to the unified Baguettotron CLI.

## 🎯 Overview

Baguettotron provides a modern and unified CLI for all operations:
- ✅ Training (single & multi-dataset)
- ✅ Generation
- ✅ Dataset management

## 📝 Main Commands

### Training

#### Single Dataset (Traditional)
```bash
# Training with a single dataset
./baguettotron train \
  --data data/train.json \
  --config tiny \
  --epochs 10 \
  --batch-size 32
```

#### Multi-Dataset with Auto-Detection
```bash
# Detects and uses all available datasets
./baguettotron train \
  --datasets auto \
  --config tiny \
  --epochs 10
```

#### Multi-Dataset with Manual Selection
```bash
# Specific datasets with custom weights
./baguettotron train \
  --datasets synth,wikipedia \
  --dataset-weights 0.7,0.3 \
  --config 321m \
  --epochs 100 \
  --batch-size 64 \
  --output-dir outputs/multi-dataset
```

### Generation

```bash
# Text generation
./baguettotron generate \
  --checkpoint outputs/model.pt \
  --prompt "Hello" \
  --max-length 100 \
  --temperature 0.8
```

### Dataset Management

#### Prepare a Dataset
```bash
# Wikipedia
./baguettotron dataset prepare \
  --type wikipedia \
  --lang simple \
  --tokenize

# SYNTH
./baguettotron dataset prepare \
  --type synth \
  --max-samples 10000 \
  --tokenize

# Demo
./baguettotron dataset prepare \
  --type demo
```

#### List Datasets
```bash
./baguettotron dataset list
```

#### Validate a Dataset
```bash
./baguettotron dataset validate --file data/train.json
```

## 🔧 Detailed Arguments

### Train Command

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--data` | str | Data file (single mode) | - |
| `--datasets` | str | Datasets (comma-separated or "auto") | - |
| `--dataset-weights` | str | Dataset weights (e.g., "0.7,0.3") | Equal |
| `--config` | str | Model configuration (tiny/321m/custom) | 321m |
| `--epochs` | int | Number of epochs | 10 |
| `--batch-size` | int | Batch size | 32 |
| `--output-dir` | str | Output directory | outputs |

**Note:** You must specify either `--data` or `--datasets`, not both.

---

**Version:** 1.0.0
**Last Update:** 2024-11-12
**Multi-dataset Support:** ✅ Complete
