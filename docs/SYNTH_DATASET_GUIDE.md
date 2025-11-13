# 📚 Guide d'Utilisation du Dataset SYNTH

## Vue d'Ensemble

Le dataset **SYNTH** de PleIAs est un dataset multilingue de 200B tokens (~75B mots) utilisé pour entraîner Baguettotron-321M. Ce guide explique comment télécharger et utiliser différentes tailles du dataset pour vos besoins.

---

## 🚀 Quick Start

### Télécharger un Petit Subset (Recommandé pour Tests)

```bash
# 10 samples pour smoke test (très rapide)
python prepare_synth_data.py --tokenize --subset --max-samples 10

# 100 samples pour demo
python prepare_synth_data.py --tokenize --subset --max-samples 100

# 10,000 samples pour développement (défaut)
python prepare_synth_data.py --tokenize --subset

# 1M samples pour validation
python prepare_synth_data.py --tokenize --subset --max-samples 1000000
```

### Télécharger le Dataset Complet

```bash
# Dataset complet (plusieurs heures de téléchargement)
python prepare_synth_data.py --tokenize --split train
```

---

## 📖 Options Disponibles

### Options de Base

```bash
python prepare_synth_data.py [OPTIONS]
```

| Option | Description | Défaut |
|--------|-------------|--------|
| `--subset` | Active le mode subset | False |
| `--max-samples N` | Nombre max d'échantillons | 10000 (avec --subset) |
| `--tokenize` | Pré-tokeniser les données | False |
| `--split SPLIT` | Split à télécharger (train/validation/test) | train |
| `--output-dir DIR` | Dossier de sortie | data |
| `--tokenizer NAME` | Tokenizer HuggingFace | PleIAs/Baguettotron |
| `--block-size N` | Longueur max de séquence | 2048 |
| `--format FORMAT` | Format de sortie (json/jsonl) | json |

### Exemples d'Utilisation

#### 1. Smoke Test (Ultra Rapide)
```bash
python prepare_synth_data.py --tokenize --subset --max-samples 10 --output-dir data/smoke
```
- ⏱️ Temps: ~10 secondes
- 💾 Taille: ~50KB
- 🎯 Usage: Tests unitaires, CI/CD

#### 2. Demo (Rapide)
```bash
python prepare_synth_data.py --tokenize --subset --max-samples 100 --output-dir data/demo
```
- ⏱️ Temps: ~30 secondes
- 💾 Taille: ~500KB
- 🎯 Usage: Démos, prototypes, debugging

#### 3. Développement (Moyen)
```bash
python prepare_synth_data.py --tokenize --subset --max-samples 10000 --output-dir data/dev
```
- ⏱️ Temps: ~5 minutes
- 💾 Taille: ~50MB
- 🎯 Usage: Développement de features, tests d'overfitting

#### 4. Validation (Long)
```bash
python prepare_synth_data.py --tokenize --subset --max-samples 1000000 --output-dir data/val
```
- ⏱️ Temps: ~1 heure
- 💾 Taille: ~5GB
- 🎯 Usage: Validation d'architecture, tuning hyperparamètres

#### 5. Production (Très Long)
```bash
python prepare_synth_data.py --tokenize --split train --output-dir data/full
```
- ⏱️ Temps: Plusieurs heures
- 💾 Taille: ~100GB+
- 🎯 Usage: Entraînement complet, modèles de production

---

## 🛠️ Scripts de Convenance

### Linux/Mac
```bash
# Utiliser le script rapide
chmod +x scripts/download_subset.sh

# Télécharger 100 samples
./scripts/download_subset.sh 100

# Télécharger 1000 samples dans un dossier spécifique
./scripts/download_subset.sh 1000 data/custom
```

### Windows
```bat
REM Télécharger 100 samples
scripts\download_subset.bat 100

REM Télécharger 1000 samples dans un dossier spécifique
scripts\download_subset.bat 1000 data\custom
```

---

## 📊 Structure du Dataset

### Format Pré-Tokenisé (--tokenize)

**Fichier de sortie**: `data/train_tokens.json`

```json
[
  [1, 234, 5678, 90, ...],  // Séquence 1 (liste de token IDs)
  [1, 456, 7890, 12, ...],  // Séquence 2
  ...
]
```

**Avantages**:
- ✅ Entraînement plus rapide (pas de tokenisation pendant training)
- ✅ Utilisation avec `TextDataset`
- ✅ Moins d'overhead CPU

**Usage en code**:
```python
from baguettotron.data import TextDataset

dataset = TextDataset(
    data_path='data/train_tokens.json',
    block_size=2048
)
```

### Format JSONL Brut (--format jsonl)

**Fichier de sortie**: `data/train.jsonl`

```jsonl
{"text": "<|im_start|>user\nQuestion...\n<|im_end|>\n..."}
{"text": "<|im_start|>user\nAutre question...\n<|im_end|>\n..."}
```

**Avantages**:
- ✅ Flexibilité (changer tokenizer sans re-télécharger)
- ✅ Inspection facile du contenu
- ✅ Moins d'espace disque

**Usage en code**:
```python
from baguettotron.data import SYNTHDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('PleIAs/Baguettotron')
dataset = SYNTHDataset(
    data_path='data/train.jsonl',
    tokenizer=tokenizer,
    block_size=2048
)
```

---

## 🔑 Authentification HuggingFace

Le dataset SYNTH peut nécessiter une authentification:

```bash
# Se connecter à HuggingFace
huggingface-cli login

# Ou définir le token
export HF_TOKEN="your_token_here"
```

---

## 💡 Cas d'Usage Recommandés

### 1. CI/CD Pipeline
```yaml
# .github/workflows/test.yml
- name: Download test dataset
  run: |
    python prepare_synth_data.py --tokenize --subset --max-samples 10 --output-dir data/ci

- name: Run training smoke test
  run: |
    python train.py --train-data data/ci/train_tokens.json --max-steps 10
```

### 2. Notebooks de Demo
```python
# demo.ipynb
!python prepare_synth_data.py --tokenize --subset --max-samples 100 --output-dir data/demo

from baguettotron.data import TextDataset
dataset = TextDataset('data/demo/train_tokens.json', block_size=512)

# Visualiser quelques exemples
for i in range(5):
    print(f"Exemple {i}: {len(dataset[i])} tokens")
```

### 3. Développement Local
```bash
# Télécharger une fois
python prepare_synth_data.py --tokenize --subset --max-samples 10000 --output-dir data/dev

# Réutiliser pour tests multiples
python train.py --train-data data/dev/train_tokens.json --config configs/tiny.json --max-epochs 5
python train.py --train-data data/dev/train_tokens.json --config configs/small.json --max-epochs 3
```

### 4. Benchmark de Vitesse
```bash
# Tester vitesse de chargement
time python prepare_synth_data.py --tokenize --subset --max-samples 10000

# Comparer formats
time python prepare_synth_data.py --tokenize --subset --max-samples 10000
time python prepare_synth_data.py --format jsonl --subset --max-samples 10000
```

---

## 📈 Estimation de Taille et Temps

| Samples | Taille Approx | Temps Download | Temps Tokenize | Total |
|---------|---------------|----------------|----------------|-------|
| 10 | 50 KB | 5s | 2s | **~10s** |
| 100 | 500 KB | 10s | 5s | **~15s** |
| 1,000 | 5 MB | 30s | 20s | **~50s** |
| 10,000 | 50 MB | 2min | 3min | **~5min** |
| 100,000 | 500 MB | 10min | 15min | **~25min** |
| 1,000,000 | 5 GB | 30min | 45min | **~75min** |
| Full (~200B tokens) | 100+ GB | Heures | Heures | **>24h** |

*Note: Les temps varient selon votre connexion et CPU*

---

## 🐛 Troubleshooting

### Erreur: "Dataset requires authentication"
```bash
# Solution
huggingface-cli login
# Ou
export HF_TOKEN="your_token_here"
```

### Erreur: "datasets library not installed"
```bash
# Solution
pip install datasets transformers
```

### Erreur: Out of Memory
```bash
# Solution: Réduire max-samples ou block-size
python prepare_synth_data.py --tokenize --subset --max-samples 1000 --block-size 1024
```

### Le téléchargement est très lent
```bash
# Solution: Utiliser un subset plus petit
python prepare_synth_data.py --tokenize --subset --max-samples 100

# Ou télécharger sans tokeniser (plus rapide)
python prepare_synth_data.py --format jsonl --subset --max-samples 10000
```

---

## 🔄 Workflow Complet Exemple

```bash
# 1. Développement avec subset
python prepare_synth_data.py --tokenize --subset --max-samples 1000 --output-dir data/dev
python train.py --train-data data/dev/train_tokens.json --config configs/debug.json

# 2. Validation avec subset plus grand
python prepare_synth_data.py --tokenize --subset --max-samples 100000 --output-dir data/val
python train.py --train-data data/val/train_tokens.json --config configs/tiny.json

# 3. Production avec dataset complet
python prepare_synth_data.py --tokenize --split train --output-dir data/full
python train.py --train-data data/full/train_tokens.json --config configs/baguettotron_321m.json
```

---

## 📚 Ressources

- **Dataset HuggingFace**: [PleIAs/SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH)
- **Tokenizer**: [PleIAs/Baguettotron](https://huggingface.co/PleIAs/Baguettotron)
- **Documentation Complète**: `docs/data/README.md`
- **Code Source**: `src/baguettotron/data/dataset.py`

---

## 🎯 Recommandations

1. **Pour Débuter**: Utilisez `--max-samples 100` pour tester rapidement
2. **Pour Développer**: Utilisez `--max-samples 10000` pour itérer sur votre code
3. **Pour Valider**: Utilisez `--max-samples 1000000` avant production
4. **Pour Production**: Téléchargez le dataset complet avec `--split train`

5. **Performance**: Préférez `--tokenize` pour des entraînements rapides
6. **Flexibilité**: Utilisez `--format jsonl` si vous voulez changer de tokenizer

---

**Date de dernière mise à jour**: 2025-11-12
**Version**: 1.0.0
