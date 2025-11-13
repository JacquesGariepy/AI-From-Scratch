# Chat Mode - Quick Start

Quick guide for using chat mode with Baguettotron-321M.

## 🚀 Basic Usage

### Simple Chat Mode

```bash
./baguettotron generate \
  --checkpoint outputs/tiny_correct/ckpt_50/model.pt \
  --chat-mode \
  --prompt "Hello!" \
  --max-length 100 \
  --config tiny
```

**Result:** The prompt is automatically formatted in ChatML before generation.

### Interactive Mode (Recommended)

```bash
./baguettotron generate \
  --checkpoint outputs/tiny_correct/ckpt_50/model.pt \
  --chat-mode \
  --interactive \
  --config tiny \
  --system-prompt "You are an expert AI assistant."
```

**Interactive Interface:**
```
> You: Hello!
> Assistant: [generated response]

> You: How are you?
> Assistant: [generated response]

> You: quit
Goodbye!
```

## 🎛️ Available Options

| Option | Description | Example |
|--------|-------------|---------|
| `--chat-mode` | Enable conversation mode | Required for chat |
| `--interactive` | Interactive mode | Real-time conversations |
| `--system-prompt` | Configure behavior | "You are an expert..." |
| `--chat-template` | Path to custom template | `assets/chat_template.json` |
| `--tokenizer` | Tokenizer type | `auto`, `baguettotron`, `gpt2` |
| `--temperature` | Control creativity | `0.8` (default: 1.0) |
| `--top-k` | Top-k sampling | `50` |
| `--top-p` | Nucleus sampling | `0.9` |
| `--max-length` | Max tokens to generate | `100` |

## 📝 Practical Examples

### French Assistant

```bash
./baguettotron generate \
  --checkpoint outputs/model.pt \
  --chat-mode \
  --interactive \
  --system-prompt "You answer only in French, clearly and concisely."
```

### Creative Mode (High Temperature)

```bash
./baguettotron generate \
  --checkpoint outputs/model.pt \
  --chat-mode \
  --interactive \
  --temperature 1.2 \
  --top-p 0.95 \
  --system-prompt "You are a creative writer."
```

### Precise Mode (Low Temperature)

```bash
./baguettotron generate \
  --checkpoint outputs/model.pt \
  --chat-mode \
  --interactive \
  --temperature 0.5 \
  --system-prompt "You are a precise technical assistant."
```

### Non-Interactive (Single Question)

```bash
./baguettotron generate \
  --checkpoint outputs/model.pt \
  --chat-mode \
  --prompt "Explain transformers to me" \
  --max-length 200 \
  --system-prompt "You are an AI expert."
```

## 🔍 Chat Template Verification

To verify that the chat template is loaded:

```bash
# The tokenizer should display "with chat template"
./baguettotron generate --checkpoint outputs/model.pt --chat-mode --interactive

# You should see:
# Tokenizer: TokenizerWrapper(..., with chat template)
```

## 🐍 Python Usage

```python
from baguettotron import BaguettotronForCausalLM, BaguettotronConfig
from baguettotron.tokenization import load_tokenizer
import torch

# Load model and tokenizer
config = BaguettotronConfig.tiny()
model = BaguettotronForCausalLM(config)
model.load_state_dict(torch.load('outputs/model.pt'))
model.eval()

tokenizer = load_tokenizer("auto", model_vocab_size=config.vocab_size)

# Conversation
messages = [
    {"role": "system", "content": "You are an AI assistant."},
    {"role": "user", "content": "Hello!"}
]

# Generate
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
)

output = model.generate(input_ids, max_new_tokens=100, temperature=0.8)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
```

## ❓ Troubleshooting

### "No chat template found"

**Solution:** The file `assets/chat_template.json` is missing. Check that it exists.

```bash
ls -la assets/chat_template.json
```

### "Using simple formatting"

**Solution:** Fallback mode is active. Chat will work but without full ChatML format.

### Chaotic Generation

**Possible Causes:**
1. Model too small (< 100M parameters)
2. Not enough training
3. Temperature too high

**Solutions:**
- Use a larger model
- Reduce temperature: `--temperature 0.7`
- Use `--greedy` for deterministic generation

### Interactive Mode Not Responding

**Solution:** Make sure the model is loaded correctly. Check the checkpoint path.

## 📊 Mode Comparison

| Mode | Command | Usage |
|------|----------|-------|
| **Simple** | `--chat-mode --prompt "..."` | Single question |
| **Interactive** | `--chat-mode --interactive` | Multiple conversations |
| **Non-chat** | Without `--chat-mode` | Standard text generation |

## 🎯 Recommended Commands

### For Development/Testing

```bash
# Tiny model, fast generation
./baguettotron generate \
  --checkpoint outputs/tiny_correct/ckpt_50/model.pt \
  --chat-mode \
  --interactive \
  --config tiny \
  --max-length 50
```

### For Production

```bash
# 321M model, maximum quality
./baguettotron generate \
  --checkpoint outputs/ckpt_10000/model.pt \
  --chat-mode \
  --interactive \
  --config 321m \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.9 \
  --system-prompt "You are an expert and helpful AI assistant."
```

## 📚 Resources

- **Complete Guide**: `docs/CHAT_TEMPLATE_GUIDE.md`
- **Tests**: `tests/test_chat_template.py`
- **Integration**: `docs/CHAT_TEMPLATE_INTEGRATION.md`
- **Template**: `assets/chat_template.json`

## 🎉 Ready to Chat!

Your Baguettotron is now configured for intelligent conversations in ChatML mode!

```bash
# Start your first conversation
./baguettotron generate \
  --checkpoint outputs/model.pt \
  --chat-mode \
  --interactive
```

---

**Note**: Examples use relative paths. Adjust according to your project structure.
