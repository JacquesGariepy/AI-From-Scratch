# Chat Mode - Quick Start

Guide rapide pour utiliser le mode chat avec Baguettotron-321M.

## 🚀 Utilisation basique

### Mode chat simple

```bash
./baguettotron generate \
  --checkpoint outputs/tiny_correct/ckpt_50/model.pt \
  --chat-mode \
  --prompt "Bonjour !" \
  --max-length 100 \
  --config tiny
```

**Résultat :** Le prompt est automatiquement formaté en ChatML avant génération.

### Mode interactif (Recommandé)

```bash
./baguettotron generate \
  --checkpoint outputs/tiny_correct/ckpt_50/model.pt \
  --chat-mode \
  --interactive \
  --config tiny \
  --system-prompt "Tu es un assistant IA expert."
```

**Interface interactive :**
```
> You: Bonjour !
> Assistant: [réponse générée]

> You: Comment vas-tu ?
> Assistant: [réponse générée]

> You: quit
Goodbye!
```

## 🎛️ Options disponibles

| Option | Description | Exemple |
|--------|-------------|---------|
| `--chat-mode` | Active le mode conversation | Obligatoire pour chat |
| `--interactive` | Mode interactif | Conversations en temps réel |
| `--system-prompt` | Configure le comportement | "Tu es un expert..." |
| `--chat-template` | Chemin vers template personnalisé | `assets/chat_template.json` |
| `--tokenizer` | Type de tokenizer | `auto`, `baguettotron`, `gpt2` |
| `--temperature` | Contrôle la créativité | `0.8` (défaut: 1.0) |
| `--top-k` | Top-k sampling | `50` |
| `--top-p` | Nucleus sampling | `0.9` |
| `--max-length` | Tokens max à générer | `100` |

## 📝 Exemples pratiques

### Assistant en français

```bash
./baguettotron generate \
  --checkpoint outputs/model.pt \
  --chat-mode \
  --interactive \
  --system-prompt "Tu réponds uniquement en français, de manière claire et concise."
```

### Mode créatif (température élevée)

```bash
./baguettotron generate \
  --checkpoint outputs/model.pt \
  --chat-mode \
  --interactive \
  --temperature 1.2 \
  --top-p 0.95 \
  --system-prompt "Tu es un écrivain créatif."
```

### Mode précis (température basse)

```bash
./baguettotron generate \
  --checkpoint outputs/model.pt \
  --chat-mode \
  --interactive \
  --temperature 0.5 \
  --system-prompt "Tu es un assistant technique précis."
```

### Sans mode interactif (une seule question)

```bash
./baguettotron generate \
  --checkpoint outputs/model.pt \
  --chat-mode \
  --prompt "Explique-moi les transformers" \
  --max-length 200 \
  --system-prompt "Tu es un expert en IA."
```

## 🔍 Vérification du chat template

Pour vérifier que le chat template est chargé :

```bash
# Le tokenizer doit afficher "with chat template"
./baguettotron generate --checkpoint outputs/model.pt --chat-mode --interactive

# Vous devriez voir :
# Tokenizer: TokenizerWrapper(..., with chat template)
```

## 🐍 Utilisation Python

```python
from baguettotron import BaguettotronForCausalLM, BaguettotronConfig
from baguettotron.tokenization import load_tokenizer
import torch

# Charger modèle et tokenizer
config = BaguettotronConfig.tiny()
model = BaguettotronForCausalLM(config)
model.load_state_dict(torch.load('outputs/model.pt'))
model.eval()

tokenizer = load_tokenizer("auto", model_vocab_size=config.vocab_size)

# Conversation
messages = [
    {"role": "system", "content": "Tu es un assistant IA."},
    {"role": "user", "content": "Bonjour !"}
]

# Générer
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

## ❓ Dépannage

### "No chat template found"

**Solution :** Le fichier `assets/chat_template.json` est manquant. Vérifiez qu'il existe.

```bash
ls -la assets/chat_template.json
```

### "Using simple formatting"

**Solution :** Le mode fallback est activé. Le chat fonctionnera mais sans format ChatML complet.

### Génération chaotique

**Causes possibles :**
1. Modèle trop petit (< 100M paramètres)
2. Pas assez d'entraînement
3. Température trop élevée

**Solutions :**
- Utilisez un modèle plus grand
- Réduisez la température : `--temperature 0.7`
- Utilisez `--greedy` pour génération déterministe

### Mode interactif ne répond pas

**Solution :** Assurez-vous que le modèle est chargé correctement. Vérifiez le chemin du checkpoint.

## 📊 Comparaison des modes

| Mode | Commande | Usage |
|------|----------|-------|
| **Simple** | `--chat-mode --prompt "..."` | Une seule question |
| **Interactif** | `--chat-mode --interactive` | Conversations multiples |
| **Non-chat** | Sans `--chat-mode` | Génération de texte standard |

## 🎯 Commandes recommandées

### Pour développement/test

```bash
# Modèle tiny, génération rapide
./baguettotron generate \
  --checkpoint outputs/tiny_correct/ckpt_50/model.pt \
  --chat-mode \
  --interactive \
  --config tiny \
  --max-length 50
```

### Pour production

```bash
# Modèle 321M, qualité maximale
./baguettotron generate \
  --checkpoint outputs/ckpt_10000/model.pt \
  --chat-mode \
  --interactive \
  --config 321m \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.9 \
  --system-prompt "Tu es un assistant IA expert et bienveillant."
```

## 📚 Ressources

- **Guide complet** : `docs/CHAT_TEMPLATE_GUIDE.md`
- **Tests** : `tests/test_chat_template.py`
- **Intégration** : `docs/CHAT_TEMPLATE_INTEGRATION.md`
- **Template** : `assets/chat_template.json`

## 🎉 Prêt à discuter !

Votre Baguettotron est maintenant configuré pour des conversations intelligentes en mode ChatML !

```bash
# Lancez votre première conversation
./baguettotron generate \
  --checkpoint outputs/model.pt \
  --chat-mode \
  --interactive
```

---

**Note** : Les exemples utilisent des chemins relatifs. Ajustez selon votre structure de projet.
