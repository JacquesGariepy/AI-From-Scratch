# Baguettotron Dataset Guide

Guide complet pour sélectionner et préparer les datasets pour l'entraînement de Baguettotron.

## 📊 Datasets Disponibles

### 1. **SYNTH** (Production - Très Grand)
- **Taille**: 500GB+
- **Samples**: Milliards de tokens
- **Qualité**: Production
- **Temps de téléchargement**: 4-8 heures
- **Recommandé pour**: Entraînement production, hardware haute performance
- **Requirements**: `datasets`, `transformers`, connexion rapide

**Utilisation:**
```bash
# Téléchargement complet (500GB+)
python prepare_synth_data.py --tokenize

# Subset pour testing (recommandé)
python prepare_synth_data.py --subset --max-samples 10000 --tokenize
```

### 2. **Wikipedia** (Moyen - Recommandé)
- **Taille**: 200MB - 20GB (selon la langue)
- **Samples**: 50K - 6M articles
- **Qualité**: Haute
- **Temps de téléchargement**: 5-60 minutes
- **Recommandé pour**: Environnements avec ressources limitées
- **Requirements**: `datasets`, `transformers`

**Options de langue:**
- `simple`: Simple English Wikipedia (~200MB, 200K articles)
- `fr`: Wikipedia Français (~6GB, 2M articles)
- `en`: Wikipedia Anglais complet (~20GB, 6M articles)

**Utilisation:**
```bash
# Simple English (le plus petit, recommandé pour commencer)
python prepare_wikipedia_data.py --lang simple --tokenize

# French Wikipedia
python prepare_wikipedia_data.py --lang fr --max-samples 100000 --tokenize

# English Wikipedia (full)
python prepare_wikipedia_data.py --lang en --tokenize
```

### 3. **Demo** (Petit - Testing)
- **Taille**: 5-50MB
- **Samples**: 100-10,000 séquences
- **Qualité**: Testing seulement (données synthétiques aléatoires)
- **Temps de téléchargement**: Instantané (généré localement)
- **Recommandé pour**: Tests rapides, CI/CD, debugging
- **Requirements**: Aucun

**Utilisation:**
```bash
# Quick test
python prepare_demo_data.py --num-samples 1000

# Larger test set
python prepare_demo_data.py --num-samples 10000
```

## 🚀 Configuration Interactive

La façon la plus simple de configurer un dataset est d'utiliser le script interactif:

```bash
python scripts/setup_dataset.py
```

Ce script vous guidera à travers:
1. Sélection du dataset (SYNTH, Wikipedia, ou Demo)
2. Configuration spécifique (langue, taille, etc.)
3. Options de tokenization
4. Téléchargement et préparation automatique

### Mode Non-Interactif

Pour l'automatisation (CI/CD, scripts):

```bash
# Wikipedia Simple English
python scripts/setup_dataset.py \
  --dataset wikipedia \
  --lang simple \
  --tokenize

# Demo dataset
python scripts/setup_dataset.py \
  --dataset demo \
  --num-samples 5000 \
  --tokenize

# SYNTH subset
python scripts/setup_dataset.py \
  --dataset synth \
  --max-samples 10000 \
  --tokenize
```

## 📏 Comparaison des Datasets

| Dataset | Taille | Articles/Samples | Temps Download | Qualité | Cas d'usage |
|---------|--------|------------------|----------------|---------|-------------|
| **SYNTH** | 500GB+ | Milliards | 4-8h | Production | Entraînement final |
| **Wikipedia (en)** | ~20GB | 6M articles | 1-2h | Haute | Production moyenne |
| **Wikipedia (fr)** | ~6GB | 2M articles | 20-40min | Haute | Production moyenne |
| **Wikipedia (simple)** | ~200MB | 200K articles | 5-10min | Haute | **Recommandé pour commencer** |
| **Demo** | 5-50MB | 100-10K | Instantané | Testing | Tests rapides |

## 🎯 Recommandations par Scénario

### Développement et Testing
```bash
# Recommandation: Wikipedia Simple English
python scripts/setup_dataset.py --dataset wikipedia --lang simple --tokenize
```
- ✅ Taille gérable (200MB)
- ✅ Données réelles de haute qualité
- ✅ Téléchargement rapide
- ✅ Suffisant pour valider le pipeline

### Entraînement Moyen (Resources Limitées)
```bash
# Recommandation: Wikipedia French ou subset
python scripts/setup_dataset.py --dataset wikipedia --lang fr --max-samples 100000 --tokenize
```
- ✅ Balance taille/qualité
- ✅ Données de production
- ✅ Adapté aux environnements contraints

### Production (Hardware Puissant)
```bash
# Recommandation: SYNTH complet
python scripts/setup_dataset.py --dataset synth --tokenize
```
- ✅ Dataset officiel
- ✅ Meilleure qualité
- ⚠️ Requiert beaucoup d'espace disque
- ⚠️ Téléchargement long

## 📝 Format des Données

### Format Tokenisé (JSON)
Recommandé pour un entraînement plus rapide:

```json
[
  [101, 2023, 2003, 1037, 6434, 102],
  [101, 2178, 6434, 2007, 3618, 102],
  ...
]
```

**Avantages:**
- ✅ Entraînement plus rapide
- ✅ Moins de CPU usage pendant l'entraînement
- ❌ Plus d'espace disque

### Format JSONL (Raw Text)
Pour tokenization dynamique:

```jsonl
{"text": "This is an example sentence."}
{"text": "Another example with more content."}
```

**Avantages:**
- ✅ Moins d'espace disque
- ✅ Plus flexible
- ❌ Tokenization pendant l'entraînement (plus lent)

## 🔧 Entraînement avec les Datasets

### Avec données tokenisées

```bash
python train.py \
  --train-data data/wikipedia_simple_tokens.json \
  --model-config tiny \
  --epochs 10 \
  --batch-size 32
```

### Avec données JSONL

```bash
python train.py \
  --train-data data/wikipedia_simple.jsonl \
  --model-config tiny \
  --epochs 10 \
  --batch-size 32
```

### Configuration Complète

```bash
python train.py \
  --train-data data/wikipedia_fr_tokens.json \
  --model-config 321m \
  --epochs 100 \
  --batch-size 64 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --mixed-precision \
  --compile
```

## 📊 Validation du Dataset

Vérifier les informations sur votre dataset:

```bash
# Via configuration sauvegardée
cat data/dataset_config.json

# Taille du fichier
ls -lh data/*.json

# Nombre de séquences (pour JSON tokenisé)
python -c "import json; data=json.load(open('data/wikipedia_simple_tokens.json')); print(f'Sequences: {len(data):,}')"
```

## 🔄 Changer de Dataset

Pour changer de dataset:

1. **Exécutez à nouveau le setup:**
   ```bash
   python scripts/setup_dataset.py
   ```

2. **Ou téléchargez directement:**
   ```bash
   python prepare_wikipedia_data.py --lang fr --tokenize
   ```

3. **Utilisez le nouveau dataset:**
   ```bash
   python train.py --train-data data/wikipedia_fr_tokens.json
   ```

## ⚡ Astuces de Performance

### Réduire l'Utilisation Mémoire
```bash
# Utilisez des séquences plus courtes
python prepare_wikipedia_data.py --lang simple --block-size 512 --tokenize

# Limitez le nombre d'articles
python prepare_wikipedia_data.py --lang en --max-samples 50000 --tokenize
```

### Accélérer le Téléchargement
```bash
# Mode streaming (télécharge uniquement ce qui est nécessaire)
python prepare_wikipedia_data.py --lang simple --max-samples 10000 --tokenize
```

### Optimiser l'Espace Disque
```bash
# Utilisez JSONL au lieu de JSON tokenisé
python prepare_wikipedia_data.py --lang simple --format jsonl

# Ou utilisez des séquences plus courtes
python prepare_wikipedia_data.py --lang simple --block-size 1024 --tokenize
```

## 🐛 Dépannage

### Erreur: Dataset not found
```bash
# Vérifiez que datasets est installé
pip install datasets transformers

# Vérifiez votre connexion internet
ping huggingface.co
```

### Téléchargement trop lent
```bash
# Utilisez un subset plus petit
python prepare_wikipedia_data.py --lang simple --max-samples 10000 --tokenize

# Ou passez au dataset demo
python prepare_demo_data.py --num-samples 5000
```

### Manque d'espace disque
```bash
# Utilisez Wikipedia Simple au lieu de en/fr
python prepare_wikipedia_data.py --lang simple --tokenize

# Ou réduisez le block size
python prepare_wikipedia_data.py --lang simple --block-size 512 --tokenize
```

### Authentication error (SYNTH)
```bash
# Certains datasets nécessitent une authentification HuggingFace
huggingface-cli login
```

## 📚 Ressources

- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
- [SYNTH Dataset](https://huggingface.co/datasets/PleIAs/SYNTH)
- [Wikipedia Dataset](https://huggingface.co/datasets/wikipedia)
- [Baguettotron Model](https://huggingface.co/PleIAs/Baguettotron)

## 🎓 Best Practices

1. **Commencez petit**: Utilisez Wikipedia Simple ou Demo pour valider votre pipeline
2. **Pre-tokenize**: Activez `--tokenize` pour un entraînement plus rapide
3. **Sauvegardez la config**: Le script sauvegarde automatiquement dans `data/dataset_config.json`
4. **Validez avant l'entraînement**: Vérifiez la taille et le format des données
5. **Montez en charge progressivement**: Demo → Wikipedia Simple → Wikipedia FR → SYNTH

## ❓ FAQ

**Q: Quel dataset choisir pour commencer?**
A: Wikipedia Simple (`--lang simple`) - 200MB, haute qualité, téléchargement rapide.

**Q: SYNTH est trop gros pour mon système, quelles alternatives?**
A: Wikipedia French (6GB) ou Wikipedia Simple (200MB) offrent une bonne qualité avec une taille gérable.

**Q: Puis-je utiliser mon propre dataset?**
A: Oui! Créez un fichier JSON avec une liste de séquences de tokens, ou JSONL avec `{"text": "..."}` par ligne.

**Q: Faut-il pre-tokenizer?**
A: Oui, recommandé. La pre-tokenization accélère l'entraînement en évitant de tokenizer à chaque epoch.

**Q: Combien d'espace disque nécessaire?**
A:
- Demo: ~50MB
- Wikipedia Simple: ~500MB (tokenisé)
- Wikipedia FR: ~15GB (tokenisé)
- SYNTH: 1TB+ (tokenisé)

---

Pour plus d'aide, consultez les scripts avec `--help`:
```bash
python scripts/setup_dataset.py --help
python prepare_wikipedia_data.py --help
python prepare_synth_data.py --help
python prepare_demo_data.py --help
```
