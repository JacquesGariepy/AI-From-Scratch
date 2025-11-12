# Data Directory

This directory contains training and evaluation data for Baguettotron.

## Current Files

- **train.json** - Training data (tokenized sequences)
- **eval.json** - Evaluation data (tokenized sequences)
- **sft.jsonl** - Example JSONL format data

## Data Format

### Pre-tokenized JSON (Recommended)

Fast training with pre-tokenized data:

```json
[
  [1, 2, 3, 4, 5, ...],     // Token IDs for first sequence
  [10, 11, 12, 13, ...],    // Token IDs for second sequence
  ...
]
```

Used with `TextDataset`:
```python
from baguettotron.data import TextDataset
dataset = TextDataset('data/train.json', block_size=2048)
```

### Raw Text JSONL (Alternative)

Tokenize on-the-fly during training:

```jsonl
{"text": "First example text..."}
{"text": "Second example text..."}
```

Used with `SYNTHDataset`:
```python
from transformers import AutoTokenizer
from baguettotron.data import SYNTHDataset

tokenizer = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")
dataset = SYNTHDataset('data/synth.jsonl', tokenizer, block_size=2048)
```

## Preparing Data

### Quick Testing

Create demo data for pipeline testing:
```bash
python prepare_demo_data.py --num-train 1000 --num-eval 100
```

### Real Training: SYNTH Dataset

Download and prepare the official SYNTH dataset:

```bash
# Pre-tokenized (fastest)
python prepare_synth_data.py --tokenize --split train

# Test with subset first
python prepare_synth_data.py --tokenize --subset --max-samples 10000

# Raw text (JSONL)
python prepare_synth_data.py --format jsonl --split train
```

### Your Own Data

Preprocess your text corpus:
```python
from transformers import AutoTokenizer
from baguettotron.data import preprocess_text_file

tokenizer = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")
preprocess_text_file(
    input_path='corpus.txt',
    output_path='data/custom_train.json',
    tokenizer=tokenizer,
    block_size=2048,
    stride=1024
)
```

## Data Statistics

The official Baguettotron model was trained on:
- **Dataset**: PleIAs/SYNTH
- **Size**: ~200B tokens (~75B words)
- **Languages**: en, fr, de, it, es, pl, nl, la, etc.
- **Format**: Qwen-style chat with reasoning tags

## See Also

- [Main README](../README.md) - Full documentation
- [prepare_demo_data.py](../prepare_demo_data.py) - Create test data
- [prepare_synth_data.py](../prepare_synth_data.py) - Download SYNTH
