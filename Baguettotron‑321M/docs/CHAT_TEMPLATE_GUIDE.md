# Chat Template Guide

This guide explains how to use the chat template (`chat_template.json`) with Baguettotron-321M for conversations formatted in ChatML style.

## Overview

The chat template allows automatic formatting of conversations between the user and assistant using the **ChatML** (Chat Markup Language) format with special tokens:

- `<|im_start|>` : Start of a message
- `<|im_end|>` : End of a message
- `<think>` : Special tag for reasoning mode

## chat_template.json File

The file is located in `assets/chat_template.json`:

```json
{
  "chat_template": "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}",
  "eos_token": "<|im_end|>",
  "bos_token": "<|im_start|>",
  "stop": ["<|im_end|>"],
  "roles": {
    "user": "user",
    "assistant": "assistant",
    "system": "system"
  }
}
```

## Command-Line Usage

### Simple Chat Mode

```bash
./baguettotron generate \
  --checkpoint outputs/ckpt_10000/model.pt \
  --chat-mode \
  --prompt "What is artificial intelligence?" \
  --max-new-tokens 200
```

### Chat Mode with System Prompt

```bash
./baguettotron generate \
  --checkpoint outputs/ckpt_10000/model.pt \
  --chat-mode \
  --system-prompt "You are an AI assistant who responds in French." \
  --prompt "Explain transformers to me." \
  --max-new-tokens 300
```

### Interactive Chat Mode

```bash
./baguettotron generate \
  --checkpoint outputs/ckpt_10000/model.pt \
  --chat-mode \
  --interactive \
  --system-prompt "You are a helpful and friendly assistant."
```

**Example Interactive Session:**

```
================================================================================
Baguettotron Chat Mode - Interactive Conversation
================================================================================
Tokenizer: TokenizerWrapper(type=baguettotron, vocab_size=65491, model_vocab_size=65536, with chat template)
Chat mode: Enabled (using ChatML format)
System prompt: You are a helpful and friendly assistant.
Enter your prompts (press Ctrl+C or type 'quit' to exit)
================================================================================

> You: Hello!

> Assistant:
--------------------------------------------------------------------------------
Hello! How can I help you today?
--------------------------------------------------------------------------------

> You: Tell me about transformers in AI

> Assistant:
--------------------------------------------------------------------------------
Transformers are a neural network architecture...
--------------------------------------------------------------------------------
```

### Using a Custom Template Path

```bash
./baguettotron generate \
  --checkpoint outputs/model.pt \
  --chat-mode \
  --chat-template /path/to/my_chat_template.json \
  --interactive
```

## Python Usage

### Simple Example

```python
from baguettotron import BaguettotronForCausalLM, BaguettotronConfig
from baguettotron.tokenization import load_tokenizer
import torch

# Load model
config = BaguettotronConfig.baguettotron_321m()
model = BaguettotronForCausalLM(config)
model.load_state_dict(torch.load('outputs/ckpt_10000/model.pt'))
model.eval()

# Load tokenizer with chat template
tokenizer = load_tokenizer(
    tokenizer_type="auto",
    model_vocab_size=config.vocab_size,
    chat_template_path="assets/chat_template.json"
)

# Create a conversation
messages = [
    {"role": "system", "content": "You are an AI assistant."},
    {"role": "user", "content": "What is AI?"}
]

# Apply chat template and tokenize
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
)

# Generate response
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        do_sample=True
    )

# Decode
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
```

### Multi-Turn Conversation

```python
from baguettotron.tokenization import load_tokenizer

tokenizer = load_tokenizer("auto", chat_template_path="assets/chat_template.json")

# Conversation with history
conversation = [
    {"role": "system", "content": "You are a Python expert."},
    {"role": "user", "content": "How do I create a list?"},
    {"role": "assistant", "content": "In Python, use: my_list = [1, 2, 3]"},
    {"role": "user", "content": "And how do I add an element?"}
]

# Format complete conversation
formatted_text = tokenizer.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=False
)

print(formatted_text)
```

**Output:**

```
<|im_start|>system
You are a Python expert.<|im_end|>
<|im_start|>user
How do I create a list?<|im_end|>
<|im_start|>assistant
In Python, use: my_list = [1, 2, 3]<|im_end|>
<|im_start|>user
And how do I add an element?<|im_end|>
<|im_start|>assistant
<think>
```

## ChatML Format

ChatML structures conversations this way:

```
<|im_start|>system
[Optional system prompt]<|im_end|>
<|im_start|>user
[User message]<|im_end|>
<|im_start|>assistant
<think>
[Generated response by model]
```

### Supported Roles

- **system**: System instructions to configure model behavior
- **user**: User messages
- **assistant**: Model responses

### Special Tokens

- `<|im_start|>`: Marks the beginning of a message with a role
- `<|im_end|>`: Marks the end of a message
- `<think>`: Indicates the model should reason before responding

## Fallback Mode

If the `chat_template.json` file is not found, the tokenizer automatically uses a simple format:

```
user: [message]
assistant: [response]
```

This allows full compatibility even without the template.

## Tests

To test chat template integration:

```bash
# Unit tests
python tests/test_chat_template.py

# Manual test with generate.py script
python scripts/generate.py \
  --checkpoint outputs/model.pt \
  --chat-mode \
  --prompt "Test message" \
  --max-new-tokens 50
```

## Advantages of Chat Template

1. **Standardized Format**: Compatible with other chat models (GPT, Claude, etc.)
2. **Clear Context**: Roles are explicit for the model
3. **Reasoning**: The `<think>` tag encourages the model to reflect
4. **Multi-Turn Conversations**: Maintains history correctly formatted
5. **Flexibility**: Supports system prompts to customize behavior

## Troubleshooting

### "No chat template found"

The tokenizer searches for `chat_template.json` in these locations (in order):

1. Path specified via `--chat-template`
2. `Baguettotron-321M/assets/chat_template.json`
3. `./assets/chat_template.json`
4. `./chat_template.json`

Make sure the file exists in one of these locations.

### "Using simple formatting"

If you see this message, fallback mode is active. Chat will still work, but without full ChatML format.

### Unknown Tokens

If the model generates `<|im_start|>` or `<|im_end|>` in its output, these tokens may not be in the vocabulary. This is normal - they are used only for input formatting.

## Complete Example

Here is a complete example using the chat template:

```python
#!/usr/bin/env python3
"""Example chat template usage."""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from baguettotron import BaguettotronForCausalLM, BaguettotronConfig
from baguettotron.tokenization import load_tokenizer

# Configuration
checkpoint_path = "outputs/ckpt_10000/model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
print("Loading model...")
config = BaguettotronConfig.baguettotron_321m()
model = BaguettotronForCausalLM(config)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

# Load tokenizer with chat template
print("Loading tokenizer...")
tokenizer = load_tokenizer(
    tokenizer_type="auto",
    model_vocab_size=config.vocab_size,
)

# Create a conversation
conversation = [
    {"role": "system", "content": "You are an AI expert in programming."},
    {"role": "user", "content": "What is a transformer in deep learning?"}
]

print("\n" + "="*80)
print("CONVERSATION")
print("="*80)
for msg in conversation:
    print(f"{msg['role'].upper()}: {msg['content']}")

# Apply template and generate
input_ids = tokenizer.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
).to(device)

print("\n" + "="*80)
print("GENERATION")
print("="*80)
print(f"Input tokens: {input_ids.shape[1]}")

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=200,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        do_sample=True
    )

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\n" + "="*80)
print("RESPONSE")
print("="*80)
print(response)
```

## References

- ChatML Format: Used by OpenAI, Anthropic, and others
- Hugging Face Chat Templates: https://huggingface.co/docs/transformers/chat_templating
- Official Baguettotron: https://huggingface.co/PleIAs/Baguettotron

---

**Note**: This guide is part of the educational implementation of Baguettotron-321M. For the official model, consult PleIAs documentation.
