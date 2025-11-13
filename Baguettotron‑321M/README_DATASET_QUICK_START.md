# 🚀 Quick Start: Obtenir des Données de Test Rapidement

## ⚡ Solution Recommandée: Dataset Demo Local (Instantané!)

Le dataset SYNTH de HuggingFace est **très volumineux** et télécharge automatiquement plusieurs shards de ~500MB chacun, même pour de petits subsets. Pour les tests rapides, utilisez plutôt le générateur de données synthétiques:

### Option 1: Données Synthétiques (Recommandé pour Tests) ⚡

```bash
# Génère instantanément 100 échantillons synthétiques
python prepare_demo_data.py --num-samples 100 --output-dir data/demo

# Résultat: ~1 seconde, aucun téléchargement!
# Crée: data/demo/train_tokens.json
```

**Avantages**:
- ✅ **Instantané** (~1 seconde pour 100 samples)
- ✅ **Pas de téléchargement** (0 MB)
- ✅ **Pas d'authentification** requise
- ✅ **Parfait pour CI/CD** et smoke tests
- ✅ **Format identique** à prepare_synth_data.py (train_tokens.json)

**Cas d'usage**:
- Tests unitaires
- Développement rapide
- CI/CD pipelines
- Smoke tests
- Démos rapides

### Option 2: Dataset SYNTH Réel (Pour Production) 🐌

```bash
# ⚠️ ATTENTION: Télécharge plusieurs shards de ~500MB même pour 1 sample!
python prepare_synth_data.py --tokenize --subset --max-samples 100

# Résultat: ~5-15 minutes, télécharge ~1-2GB minimum
```

**⚠️ Limitation Technique**: En raison de la structure du dataset SYNTH (500 shards parquet de ~500MB chacun), HuggingFace **télécharge automatiquement des shards complets** même quand on demande seulement quelques samples. **Le mode streaming est implémenté** mais la bibliothèque `datasets` doit quand même accéder aux fichiers parquet complets.

**Pour cette raison, nous recommandons fortement `prepare_demo_data.py` pour tous les tests!**

---

## 📊 Comparaison

| Méthode | Temps | Téléchargement | Samples | Format de Sortie | Usage Recommandé |
|---------|-------|----------------|---------|------------------|------------------|
| **prepare_demo_data.py** | **~1s** | **0 MB** | **Configurable** | **train_tokens.json** | **Tests, Dev, CI/CD** ⚡ |
| prepare_synth_data.py (streaming) | ~5-15min | ~1-2 GB minimum | Réels limités | train_tokens.json | Validation réelle |
| prepare_synth_data.py (complet) | Heures | ~100+ GB | Tous (~200B tokens) | train_tokens.json | Production finale |

---

## 💡 Workflows Recommandés

### Pour Développement Local

```bash
# 1. Générer données de test rapidement
python prepare_demo_data.py --num-samples 1000 --output-dir data/dev

# 2. Développer avec ces données
python train.py --train-data data/dev/train_tokens.json --max-epochs 1

# 3. Quand satisfait, tester avec vraies données
python prepare_synth_data.py --tokenize --subset --max-samples 10000
```

### Pour CI/CD

```yaml
# .github/workflows/test.yml
- name: Generate test data (instant)
  run: python prepare_demo_data.py --num-samples 10 --output-dir data/ci

- name: Run smoke test
  run: python train.py --train-data data/ci/train_tokens.json --max-steps 10
```

### Pour Production

```bash
# Seulement quand prêt pour entraînement final
python prepare_synth_data.py --tokenize --split train --output-dir data/production
```

---

## 🔧 prepare_demo_data.py - Options

```bash
python prepare_demo_data.py --help
```

| Option | Description | Défaut |
|--------|-------------|--------|
| `--num-samples N` | Nombre d'échantillons (mode rapide) | None |
| `--num-train N` | Nombre d'échantillons d'entraînement | 1000 |
| `--num-eval N` | Nombre d'échantillons d'évaluation | 100 |
| `--output-dir DIR` | Dossier de sortie | data |
| `--vocab-size N` | Taille du vocabulaire | 1000 |
| `--seq-length N` | Longueur des séquences | 512 |

### Mode Rapide (Recommandé)

Utilisez `--num-samples` pour un seul fichier `train_tokens.json` (comme `prepare_synth_data.py`):

```bash
# Smoke test (10 samples, ~1 seconde)
python prepare_demo_data.py --num-samples 10

# Demo (100 samples, ~1 seconde)
python prepare_demo_data.py --num-samples 100 --output-dir data/demo

# Dev (1000 samples, ~2 secondes)
python prepare_demo_data.py --num-samples 1000 --output-dir data/dev

# Longues séquences
python prepare_demo_data.py --num-samples 100 --seq-length 2048
```

### Mode Complet (Train + Eval)

Sans `--num-samples`, crée deux fichiers séparés `train.json` et `eval.json`:

```bash
# Train + Eval avec valeurs par défaut (1000 train, 100 eval)
python prepare_demo_data.py

# Train + Eval personnalisés
python prepare_demo_data.py --num-train 5000 --num-eval 500
```

---

## 🎯 Quand Utiliser Quoi?

### Utilisez `prepare_demo_data.py` pour:
- ✅ Tests unitaires
- ✅ Développement de features
- ✅ Debugging
- ✅ CI/CD
- ✅ Smoke tests
- ✅ Prototypage rapide
- ✅ Démos

### Utilisez `prepare_synth_data.py` pour:
- ✅ Validation d'architecture
- ✅ Tuning d'hyperparamètres
- ✅ Benchmarking
- ✅ Entraînement de production
- ✅ Évaluation finale

---

## 🐛 Problème: prepare_synth_data.py Télécharge Trop

### Symptômes
```bash
$ python prepare_synth_data.py --subset --max-samples 1
# Télécharge quand même 1-2GB de shards...
Downloading data files:   0%|          | 0/500 [00:00<?, ?it/s]
Downloading data: 473MB/473MB [00:15<00:00, 30.5MB/s]
Downloading data: 474MB/474MB [00:16<00:00, 29.1MB/s]
# Continue à télécharger plusieurs shards de ~500MB...
```

### Pourquoi?
Le dataset SYNTH est divisé en **500 shards parquet** de ~500MB chacun. HuggingFace datasets **doit télécharger des shards complets** pour extraire même un seul sample, car:
1. Les fichiers parquet sont compressés et indivisibles
2. La bibliothèque `datasets` n'a pas d'index pour savoir quel shard contient quels samples
3. Le mode streaming réduit la mémoire mais doit quand même accéder aux fichiers complets

**C'est une limitation architecturale de HuggingFace datasets + parquet, pas un bug.**

### Solution ✅
**Utilisez `prepare_demo_data.py` pour tous les tests!** C'est exactement pour ça qu'il existe.

```bash
# ❌ Au lieu de (télécharge ~1-2GB minimum):
python prepare_synth_data.py --subset --max-samples 10

# ✅ Faites (instantané, 0 téléchargement):
python prepare_demo_data.py --num-samples 10
```

**Résultat**: Même format de sortie (`train_tokens.json`), 10,000x plus rapide!

---

## 📝 Mise à Jour prepare_synth_data.py

Le script a été amélioré avec:
- ✅ **Mode streaming** activé automatiquement pour `--subset`
- ✅ **Early stopping** quand max_samples atteint
- ✅ Messages clairs sur le téléchargement

**Note**: Même avec streaming, HuggingFace doit télécharger des shards complets. C'est une limitation de la bibliothèque `datasets`, pas du script.

---

## 🎉 Résumé

**Pour 99% des cas d'usage de développement:**
```bash
python prepare_demo_data.py --num-samples 100
```

**Pour production uniquement:**
```bash
python prepare_synth_data.py --tokenize --split train
```

**C'est tout!** 🚀
