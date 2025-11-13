# 🚀 Baguettotron Quick Start Guide

Guide complet pour débutants : préparer les données, entraîner et générer du texte en 3 étapes simples.

## 📋 Prérequis

```bash
# Python 3.8+ avec CUDA (recommandé)
python --version
nvidia-smi  # Vérifier CUDA

# Clone le projet
git clone <repo-url>
cd Baguettotron‑321M

# Installation
pip install -e ".[train]"
```

---

## 🎯 Scénario 1: Test Rapide (5 minutes)

**Pour tester rapidement sans téléchargement:**

### Étape 1: Préparer les données de démo

```bash
./baguettotron dataset prepare --type demo
```

Résultat: `data/train.json` (76KB, données synthétiques)

### Étape 2: Entraîner le modèle

```bash
./baguettotron train \
  --data data/train.json \
  --config 321m \
  --epochs 1 \
  --batch-size 2 \
  --save-steps 10 \
  --save-total-limit 2 \
  --output-dir outputs/demo
```

Résultat: Checkpoints dans `outputs/demo/ckpt_*/`

### Étape 3: Générer du texte

```bash
./baguettotron generate \
  --checkpoint outputs/demo/ckpt_*/model.pt \
  --prompt "Bonjour, je suis" \
  --max-length 50
```

---

## 📚 Scénario 2: Entraînement Réaliste (30 minutes)

**Avec Wikipedia (300MB, 241K articles):**

### Étape 1: Préparer Wikipedia

```bash
# Télécharge et tokenise Wikipedia Simple English
./baguettotron dataset prepare --type wikipedia --tokenize
```

Résultat: `data/wikipedia_simple_tokens.json` (300MB)

Durée: ~2-3 minutes

### Étape 2: Entraîner avec Wikipedia

```bash
./baguettotron train \
  --datasets wikipedia \
  --config 321m \
  --epochs 5 \
  --batch-size 4 \
  --warmup-steps 500 \
  --save-steps 1000 \
  --save-total-limit 3 \
  --output-dir outputs/wikipedia
```

Durée: ~20-30 minutes (selon GPU)

### Étape 3: Générer du texte

```bash
./baguettotron generate \
  --checkpoint outputs/wikipedia/ckpt_*/model.pt \
  --prompt "The capital of France is" \
  --max-length 100 \
  --temperature 0.8
```

---

## 🎓 Scénario 3: Multi-Dataset Training (Production)

**Combinaison Wikipedia + SYNTH:**

### Étape 1: Préparer les datasets

```bash
# Wikipedia (déjà fait dans scénario 2)
./baguettotron dataset prepare --type wikipedia --tokenize

# SYNTH (subset pour test)
./baguettotron dataset prepare --type synth --tokenize --max-samples 10000
```

### Étape 2: Entraîner avec multi-dataset

**Option A: Ligne de commande**
```bash
./baguettotron train \
  --datasets synth,wikipedia \
  --dataset-weights 0.7,0.3 \
  --config 321m \
  --epochs 10 \
  --batch-size 32 \
  --warmup-steps 2000 \
  --save-steps 1000 \
  --save-total-limit 5 \
  --output-dir outputs/production
```

**Option B: Fichier YAML (recommandé)**
```bash
./baguettotron train --config-file configs/multi_dataset.yaml
```

### Étape 3: Générer avec modèle entraîné

```bash
./baguettotron generate \
  --checkpoint outputs/production/ckpt_*/model.pt \
  --prompt "Explain quantum computing" \
  --max-length 200 \
  --temperature 0.7 \
  --top-p 0.9
```

---

## 🛠️ Commandes Utiles

### Lister les datasets disponibles

```bash
./baguettotron dataset list
```

### Valider un dataset

```bash
./baguettotron dataset validate --file data/wikipedia_simple_tokens.json
```

### Vérifier les checkpoints

```bash
ls -lh outputs/*/ckpt_*/
```

### Reprendre l'entraînement

```bash
./baguettotron train \
  --config-file configs/multi_dataset.yaml \
  --checkpoint outputs/production/ckpt_5000
```

---

## 📊 Structure des Checkpoints

Chaque checkpoint contient:

```
outputs/production/
├── ckpt_1000/
│   ├── model.pt           # Poids du modèle
│   ├── trainer_state.pt   # État de l'optimiseur
│   └── config.json        # Configuration du modèle
├── ckpt_2000/
│   └── ...
└── ckpt_3000/
    └── ...
```

**Avantages:**
- Séparation claire modèle/état
- Rotation automatique (garde seulement N derniers)
- Reprise d'entraînement facile

---

## 💡 Conseils

### Pour GPU limitée (8-16GB)

```bash
./baguettotron train \
  --data data/train.json \
  --config tiny \
  --batch-size 2 \
  --epochs 10
```

### Pour GPU puissante (24GB+)

```bash
./baguettotron train \
  --datasets synth,wikipedia \
  --dataset-weights 0.7,0.3 \
  --config 321m \
  --batch-size 32 \
  --epochs 100
```

### Monitoring avec TensorBoard

```bash
tensorboard --logdir outputs/production/
```

---

## ❓ Dépannage

### "Dataset not found"

```bash
# Vérifier les datasets disponibles
ls -lh data/*.json

# Préparer le dataset manquant
./baguettotron dataset prepare --type wikipedia --tokenize
```

### "CUDA out of memory"

```bash
# Réduire batch size
./baguettotron train --batch-size 1 ...

# Ou utiliser config tiny
./baguettotron train --config tiny ...
```

### "Checkpoint directory already exists"

```bash
# Utiliser un nouveau dossier
./baguettotron train --output-dir outputs/run-$(date +%Y%m%d-%H%M%S)
```

---

## 📚 Prochaines Étapes

1. **Lire la documentation complète:** `README.md`
2. **Explorer les configs YAML:** `configs/*.yaml`
3. **Tester les datasets:** `docs/DATASET_GUIDE.md`
4. **Optimiser l'entraînement:** `docs/TRAINING_GUIDE.md`

---

## 🎯 Exemple Complet End-to-End

```bash
# 1. Préparation
./baguettotron dataset prepare --type wikipedia --tokenize

# 2. Entraînement
./baguettotron train \
  --datasets wikipedia \
  --config 321m \
  --epochs 5 \
  --batch-size 4 \
  --warmup-steps 500 \
  --save-steps 1000 \
  --save-total-limit 3 \
  --output-dir outputs/my-model

# 3. Génération
./baguettotron generate \
  --checkpoint outputs/my-model/ckpt_*/model.pt \
  --prompt "Once upon a time" \
  --max-length 100

# 4. Vérification des checkpoints
tree outputs/my-model/
```

**Durée totale:** ~30-45 minutes
**Résultat:** Modèle entraîné prêt à générer du texte

---

## ✅ Checklist de Succès

- [ ] Données préparées (`data/*.json` existe)
- [ ] Entraînement complété sans erreur
- [ ] Checkpoints créés (`outputs/*/ckpt_*/`)
- [ ] Génération fonctionne
- [ ] Résultats satisfaisants

**Félicitations ! Vous maîtrisez maintenant Baguettotron ! 🎉**
