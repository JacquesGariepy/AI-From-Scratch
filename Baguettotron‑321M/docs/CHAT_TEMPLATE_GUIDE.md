# Chat Template Guide

Ce guide explique comment utiliser le chat template (`chat_template.json`) avec Baguettotron-321M pour des conversations formatées en style ChatML.

## Vue d'ensemble

Le chat template permet de formater automatiquement les conversations entre l'utilisateur et l'assistant en utilisant le format **ChatML** (Chat Markup Language) avec les tokens spéciaux :

- `<|im_start|>` : Début d'un message
- `<|im_end|>` : Fin d'un message
- `<think>` : Tag spécial pour le mode raisonnement

## Fichier chat_template.json

Le fichier se trouve dans `assets/chat_template.json` :

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

## Utilisation en ligne de commande

### Mode chat simple

```bash
./baguettotron generate \
  --checkpoint outputs/ckpt_10000/model.pt \
  --chat-mode \
  --prompt "Qu'est-ce que l'intelligence artificielle ?" \
  --max-new-tokens 200
```

### Mode chat avec prompt système

```bash
./baguettotron generate \
  --checkpoint outputs/ckpt_10000/model.pt \
  --chat-mode \
  --system-prompt "Tu es un assistant IA qui répond en français." \
  --prompt "Explique-moi les transformers." \
  --max-new-tokens 300
```

### Mode interactif avec chat

```bash
./baguettotron generate \
  --checkpoint outputs/ckpt_10000/model.pt \
  --chat-mode \
  --interactive \
  --system-prompt "Tu es un assistant serviable et amical."
```

**Exemple de session interactive :**

```
================================================================================
Baguettotron Chat Mode - Interactive Conversation
================================================================================
Tokenizer: TokenizerWrapper(type=baguettotron, vocab_size=65491, model_vocab_size=65536, with chat template)
Chat mode: Enabled (using ChatML format)
System prompt: Tu es un assistant serviable et amical.
Enter your prompts (press Ctrl+C or type 'quit' to exit)
================================================================================

> You: Bonjour !

> Assistant:
--------------------------------------------------------------------------------
Bonjour ! Comment puis-je vous aider aujourd'hui ?
--------------------------------------------------------------------------------

> You: Parle-moi des transformers en IA

> Assistant:
--------------------------------------------------------------------------------
Les transformers sont une architecture de réseau de neurones...
--------------------------------------------------------------------------------
```

### Utiliser un chemin personnalisé pour le template

```bash
./baguettotron generate \
  --checkpoint outputs/model.pt \
  --chat-mode \
  --chat-template /chemin/vers/mon_chat_template.json \
  --interactive
```

## Utilisation en Python

### Exemple simple

```python
from baguettotron import BaguettotronForCausalLM, BaguettotronConfig
from baguettotron.tokenization import load_tokenizer
import torch

# Charger le modèle
config = BaguettotronConfig.baguettotron_321m()
model = BaguettotronForCausalLM(config)
model.load_state_dict(torch.load('outputs/ckpt_10000/model.pt'))
model.eval()

# Charger le tokenizer avec chat template
tokenizer = load_tokenizer(
    tokenizer_type="auto",
    model_vocab_size=config.vocab_size,
    chat_template_path="assets/chat_template.json"
)

# Créer une conversation
messages = [
    {"role": "system", "content": "Tu es un assistant IA."},
    {"role": "user", "content": "Qu'est-ce que l'IA ?"}
]

# Appliquer le chat template et tokenizer
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
)

# Générer la réponse
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        do_sample=True
    )

# Décoder
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
```

### Conversation multi-tours

```python
from baguettotron.tokenization import load_tokenizer

tokenizer = load_tokenizer("auto", chat_template_path="assets/chat_template.json")

# Conversation avec historique
conversation = [
    {"role": "system", "content": "Tu es un expert en Python."},
    {"role": "user", "content": "Comment créer une liste ?"},
    {"role": "assistant", "content": "En Python, utilisez : ma_liste = [1, 2, 3]"},
    {"role": "user", "content": "Et comment ajouter un élément ?"}
]

# Formater la conversation complète
formatted_text = tokenizer.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=False
)

print(formatted_text)
```

**Sortie :**

```
<|im_start|>system
Tu es un expert en Python.<|im_end|>
<|im_start|>user
Comment créer une liste ?<|im_end|>
<|im_start|>assistant
En Python, utilisez : ma_liste = [1, 2, 3]<|im_end|>
<|im_start|>user
Et comment ajouter un élément ?<|im_end|>
<|im_start|>assistant
<think>
```

## Format ChatML

Le format ChatML structure les conversations de cette manière :

```
<|im_start|>system
[Prompt système optionnel]<|im_end|>
<|im_start|>user
[Message de l'utilisateur]<|im_end|>
<|im_start|>assistant
<think>
[Réponse générée par le modèle]
```

### Rôles supportés

- **system** : Instructions système pour configurer le comportement du modèle
- **user** : Messages de l'utilisateur
- **assistant** : Réponses du modèle

### Token spéciaux

- `<|im_start|>` : Marque le début d'un message avec un rôle
- `<|im_end|>` : Marque la fin d'un message
- `<think>` : Indique que le modèle doit raisonner avant de répondre

## Mode fallback

Si le fichier `chat_template.json` n'est pas trouvé, le tokenizer utilise automatiquement un format simple :

```
user: [message]
assistant: [réponse]
```

Cela permet une compatibilité totale même sans le template.

## Tests

Pour tester l'intégration du chat template :

```bash
# Tests unitaires
python tests/test_chat_template.py

# Test manuel avec script generate.py
python scripts/generate.py \
  --checkpoint outputs/model.pt \
  --chat-mode \
  --prompt "Test message" \
  --max-new-tokens 50
```

## Avantages du Chat Template

1. **Format standardisé** : Compatible avec d'autres modèles de chat (GPT, Claude, etc.)
2. **Contexte clair** : Les rôles sont explicites pour le modèle
3. **Raisonnement** : Le tag `<think>` encourage le modèle à réfléchir
4. **Conversations multi-tours** : Maintient l'historique correctement formaté
5. **Flexibilité** : Supporte les prompts système pour personnaliser le comportement

## Dépannage

### "No chat template found"

Le tokenizer cherche `chat_template.json` dans ces emplacements (dans l'ordre) :

1. Chemin spécifié via `--chat-template`
2. `Baguettotron-321M/assets/chat_template.json`
3. `./assets/chat_template.json`
4. `./chat_template.json`

Assurez-vous que le fichier existe dans l'un de ces emplacements.

### "Using simple formatting"

Si vous voyez ce message, le mode fallback est activé. Le chat fonctionnera toujours, mais sans le format ChatML complet.

### Tokens inconnus

Si le modèle génère `<|im_start|>` ou `<|im_end|>` dans sa sortie, ces tokens peuvent ne pas être dans le vocabulaire. C'est normal - ils sont utilisés uniquement pour le formatage de l'entrée.

## Exemple complet

Voici un exemple complet d'utilisation du chat template :

```python
#!/usr/bin/env python3
"""Exemple d'utilisation du chat template."""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from baguettotron import BaguettotronForCausalLM, BaguettotronConfig
from baguettotron.tokenization import load_tokenizer

# Configuration
checkpoint_path = "outputs/ckpt_10000/model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Charger le modèle
print("Chargement du modèle...")
config = BaguettotronConfig.baguettotron_321m()
model = BaguettotronForCausalLM(config)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

# Charger le tokenizer avec chat template
print("Chargement du tokenizer...")
tokenizer = load_tokenizer(
    tokenizer_type="auto",
    model_vocab_size=config.vocab_size,
)

# Créer une conversation
conversation = [
    {"role": "system", "content": "Tu es un assistant IA expert en programmation."},
    {"role": "user", "content": "Qu'est-ce qu'un transformer en deep learning ?"}
]

print("\n" + "="*80)
print("CONVERSATION")
print("="*80)
for msg in conversation:
    print(f"{msg['role'].upper()}: {msg['content']}")

# Appliquer le template et générer
input_ids = tokenizer.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
).to(device)

print("\n" + "="*80)
print("GÉNÉRATION")
print("="*80)
print(f"Tokens d'entrée: {input_ids.shape[1]}")

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
print("RÉPONSE")
print("="*80)
print(response)
```

## Références

- Format ChatML : Utilisé par OpenAI, Anthropic, et d'autres
- Hugging Face Chat Templates : https://huggingface.co/docs/transformers/chat_templating
- Baguettotron officiel : https://huggingface.co/PleIAs/Baguettotron

---

**Note** : Ce guide fait partie de l'implémentation éducative de Baguettotron-321M. Pour le modèle officiel, consultez la documentation de PleIAs.
