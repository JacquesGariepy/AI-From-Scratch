# Installation Guide - Baguettotron Multi-Dataset System

Guide complet d'installation avec support multi-dataset intelligent.

## 🚀 Installation Rapide

### Option 1: Installation avec Wikipedia (Recommandé)
```bash
# Clone le repo
git clone <repo-url>
cd Baguettotron‑321M

# Installation avec training + Wikipedia Simple (200MB)
pip install -e ".[train,wikipedia]"
```

### Option 2: Installation Complète Multi-Dataset
```bash
# Training + plusieurs datasets
pip install -e ".[train,synth,wikipedia]"
```

### Option 3: Téléchargement Datasets Seulement
```bash
# Juste télécharger les datasets (pas de training)
pip install -e ".[dataset,wikipedia]"
```

## 📦 Extras Disponibles

### Extras de Base
- `train` - Dépendances d'entraînement (transformers, datasets, tensorboard, wandb)
- `dev` - Outils de développement (pytest, black, mypy, etc.)
- `dataset` - Téléchargement datasets uniquement (sans training)
- `all` - Tout (train + dev + all-datasets)

### Extras par Dataset
- `synth` - Dataset SYNTH (PleIAs/SYNTH, 500GB+ complet, subset par défaut)
- `wikipedia` - Dataset Wikipedia (wikimedia/wikipedia, 200MB-20GB selon langue)
- `demo` - Dataset synthétique pour testing (5MB, généré localement)

### Extras Combinés
- `all-datasets` - Tous les datasets
- `all` - Installation complète

## 🎯 Exemples d'Installation

### 1. Développement avec Wikipedia
```bash
# Training + Wikipedia + outils dev
pip install -e ".[train,dev,wikipedia]"
```

### 2. Production avec SYNTH
```bash
# Training + SYNTH dataset
pip install -e ".[train,synth]"
```

### 3. Multi-Dataset (SYNTH + Wikipedia)
```bash
# Training + plusieurs datasets
pip install -e ".[train,synth,wikipedia]"
```

### 4. Testing avec Demo
```bash
# Juste le dataset demo pour tests
pip install -e ".[demo]"
```

### 5. Installation Complète
```bash
# Tout installer
pip install -e ".[all,synth,wikipedia,demo]"
```

## 🔄 Fonctionnement Intelligent

### Auto-Détection des Datasets
Le système détecte automatiquement:
1. Quels extras dataset ont été installés
2. Si les datasets sont déjà téléchargés
3. Télécharge seulement ce qui manque

### Exemple de Workflow
```bash
# 1. Installation avec Wikipedia
pip install -e ".[train,wikipedia]"
# ✓ Installe les dépendances
# ✓ Télécharge Wikipedia Simple (200MB)

# 2. Ajout de SYNTH plus tard
pip install -e ".[synth]"
# ✓ Télécharge SYNTH subset
# ✓ Wikipedia déjà présent, pas re-téléchargé

# 3. Training multi-dataset
./baguettotron train --datasets auto
# ✓ Détecte Wikipedia + SYNTH
# ✓ Entraîne avec les deux datasets
```

## 📊 Gestion Multi-Dataset

### Auto-Détection
```bash
# Détecte et utilise tous les datasets disponibles
./baguettotron train --datasets auto

# Équivalent Python
python scripts/train.py --datasets auto
```

### Datasets Spécifiques
```bash
# Utiliser des datasets spécifiques
./baguettotron train --datasets synth,wikipedia

# Avec poids personnalisés (70% SYNTH, 30% Wikipedia)
./baguettotron train --datasets synth,wikipedia --dataset-weights 0.7,0.3
```

### Vérifier les Datasets Disponibles
```bash
# Liste les datasets téléchargés
python -m baguettotron.data.auto_download --list

# Informations détaillées
./baguettotron dataset list
```

## 🔧 Configuration Avancée

### Variables d'Environnement
```bash
# Spécifier les datasets à télécharger
export BAGUETTOTRON_DATASETS=wikipedia,demo
pip install -e ".[train]"
```

### Téléchargement Manuel
```bash
# Si l'auto-download échoue, téléchargement manuel
python scripts/setup_dataset.py --dataset wikipedia --lang simple --tokenize
```

### Forcer le Re-téléchargement
```bash
# Re-télécharger même si déjà présent
python -m baguettotron.data.auto_download --datasets wikipedia --force
```

## 📁 Structure des Datasets

Après installation, vos datasets seront dans:
```
data/
├── train_tokens.json              # Demo ou SYNTH
├── wikipedia_simple_tokens.json   # Wikipedia Simple
├── wikipedia_fr_tokens.json       # Wikipedia FR
├── wikipedia_en_tokens.json       # Wikipedia EN
└── dataset_config.json            # Configuration
```

## 🎓 Cas d'Usage Recommandés

### Débutant - Testing Rapide
```bash
pip install -e ".[demo]"
./baguettotron train --train-data data/train.json --config tiny --epochs 5
```
**Avantages:**
- ✅ Instantané (pas de téléchargement)
- ✅ Petit dataset pour valider le pipeline
- ✅ Entraînement rapide

### Développeur - Développement
```bash
pip install -e ".[train,dev,wikipedia]"
./baguettotron train --datasets wikipedia --config tiny --epochs 10
```
**Avantages:**
- ✅ Vraies données (Wikipedia 200MB)
- ✅ Outils de développement inclus
- ✅ Entraînement raisonnable

### Chercheur - Multi-Dataset
```bash
pip install -e ".[train,synth,wikipedia]"
./baguettotron train --datasets auto --dataset-weights 0.6,0.4 --config 321m
```
**Avantages:**
- ✅ Diversité des données
- ✅ Meilleure généralisation
- ✅ Contrôle des proportions

### Production - SYNTH Complet
```bash
pip install -e ".[train,synth]"
./baguettotron train --datasets synth --config 321m --epochs 100
```
**Avantages:**
- ✅ Dataset officiel complet
- ✅ Meilleure qualité
- ✅ Compatible avec official model

## 🔍 Dépannage

### "Dataset auto-download failed"
```bash
# Solution 1: Téléchargement manuel
python scripts/setup_dataset.py

# Solution 2: Vérifier les dépendances
pip install datasets transformers huggingface-hub
```

### "No datasets found"
```bash
# Vérifier quels datasets sont disponibles
python -m baguettotron.data.auto_download --list

# Télécharger un dataset manuellement
./baguettotron dataset prepare --type wikipedia --lang simple
```

### "Authentication required for SYNTH"
```bash
# Certains datasets nécessitent HuggingFace login
huggingface-cli login
pip install -e ".[synth]"
```

### Espace Disque Insuffisant
```bash
# Option 1: Wikipedia Simple au lieu de SYNTH
pip install -e ".[train,wikipedia]"

# Option 2: Demo dataset
pip install -e ".[demo]"

# Option 3: Subset SYNTH (défaut)
# Le système télécharge automatiquement un subset de 10K samples
```

## 📚 Références

### Commandes Principales
```bash
# Installation
pip install -e ".[train,DATASET]"

# Training
./baguettotron train --datasets auto

# Gestion datasets
./baguettotron dataset list
./baguettotron dataset validate --file data/train.json

# Auto-download manuel
python -m baguettotron.data.auto_download --datasets wikipedia
```

### Makefile Shortcuts
```bash
make install              # Installation de base
make dataset-wiki         # Télécharge Wikipedia
make dataset-demo         # Génère demo data
make train-tiny           # Entraînement rapide
```

## ⚡ Performance Tips

### Réduire Temps de Téléchargement
```bash
# Utilisez Wikipedia Simple (200MB)
pip install -e ".[wikipedia]"
# Temps: ~5-10 minutes

# Vs Wikipedia EN (20GB)
# Temps: ~1-2 heures
```

### Optimiser l'Espace Disque
```bash
# Demo: 5MB
pip install -e ".[demo]"

# Wikipedia Simple: 200MB
pip install -e ".[wikipedia]"

# SYNTH subset: 100MB (vs 500GB+ full)
pip install -e ".[synth]"
```

### Training Multi-GPU
```bash
# Installation complète
pip install -e ".[train,all-datasets]"

# Training avec plusieurs datasets
./baguettotron train \
  --datasets auto \
  --batch-size 64 \
  --mixed-precision \
  --compile
```

## 🎯 Workflow Recommandé

1. **Démarrer Petit**
   ```bash
   pip install -e ".[demo]"
   make train-tiny
   ```

2. **Passer à Wikipedia**
   ```bash
   pip install -e ".[train,wikipedia]"
   ./baguettotron train --datasets wikipedia --config tiny
   ```

3. **Multi-Dataset**
   ```bash
   pip install -e ".[synth,wikipedia]"
   ./baguettotron train --datasets auto --dataset-weights 0.6,0.4
   ```

4. **Production**
   ```bash
   pip install -e ".[train,synth]"
   ./baguettotron train --datasets synth --config 321m
   ```

---

**Dernière mise à jour:** 2024-11-12
**Version:** 1.0.0
**Support:** Multi-dataset intelligent avec auto-download
