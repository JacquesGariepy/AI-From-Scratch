# Multi-Dataset Training Guide

Guide complet pour l'entraînement avec plusieurs datasets simultanément.

## 🎯 Concept

Baguettotron supporte l'entraînement avec plusieurs datasets en même temps:
- Combine SYNTH + Wikipedia + Demo
- Sampling pondéré pour contrôler les proportions
- Détection automatique des datasets disponibles
- Meilleure généralisation du modèle

## 🚀 Quick Start

### 1. Installation Multi-Dataset
```bash
# Installer avec plusieurs datasets
pip install -e ".[train,synth,wikipedia]"
```

### 2. Training avec Auto-Détection
```bash
# Détecte et utilise tous les datasets disponibles
./baguettotron train --datasets auto --config tiny
```

### 3. Training avec Datasets Spécifiques
```bash
# Utiliser SYNTH + Wikipedia avec poids personnalisés
./baguettotron train \
  --datasets synth,wikipedia \
  --dataset-weights 0.7,0.3 \
  --config 321m \
  --epochs 100
```

## 📊 Options de Configuration

### Auto-Détection (Recommandé)
```bash
# Utilise tous les datasets trouvés dans data/
./baguettotron train --datasets auto

# Le système détecte automatiquement:
# ✓ data/train_tokens.json (SYNTH ou demo)
# ✓ data/wikipedia_simple_tokens.json
# ✓ data/wikipedia_fr_tokens.json
# Et les combine avec poids égaux
```

### Datasets Spécifiques
```bash
# Sélectionner quels datasets utiliser
./baguettotron train --datasets synth,wikipedia

# Avec poids égaux (défaut: 0.5, 0.5)
```

### Poids Personnalisés
```bash
# 70% SYNTH, 30% Wikipedia
./baguettotron train \
  --datasets synth,wikipedia \
  --dataset-weights 0.7,0.3

# 50% SYNTH, 30% Wikipedia FR, 20% Demo
./baguettotron train \
  --datasets synth,wikipedia_fr,demo \
  --dataset-weights 0.5,0.3,0.2
```

## 🔧 Fonctionnement Technique

### Weighted Random Sampling
Le système utilise `WeightedRandomSampler` de PyTorch:

```python
# Exemple: 70% SYNTH (10K samples), 30% Wikipedia (50K samples)
synth_weight = 0.7
wiki_weight = 0.3

# Probabilité par sample:
synth_sample_prob = 0.7 / 10000  # Plus élevée car moins de samples
wiki_sample_prob = 0.3 / 50000   # Plus faible car plus de samples

# Résultat: ~70% des batches viennent de SYNTH
```

### Mixing Strategy
```python
from baguettotron.data.multi_dataset import create_multi_dataset

# Création du multi-dataset
dataset = create_multi_dataset(
    datasets=['synth', 'wikipedia'],
    weights=[0.7, 0.3],
    block_size=2048
)

# Le dataset combine intelligemment:
# - Charge tous les datasets
# - Crée un sampler pondéré
# - Mélange pendant l'entraînement
```

## 📈 Cas d'Usage Recommandés

### Cas 1: Développement Rapide
**Objectif:** Valider le pipeline rapidement

```bash
# Installation
pip install -e ".[train,demo,wikipedia]"

# Training
./baguettotron train \
  --datasets auto \
  --config tiny \
  --epochs 5 \
  --batch-size 16
```

**Résultat:**
- Entraînement rapide (quelques minutes)
- Validation du multi-dataset
- Données de qualité (Wikipedia) + volume (demo)

### Cas 2: Recherche et Expérimentation
**Objectif:** Tester différentes combinaisons

```bash
# Installation
pip install -e ".[train,synth,wikipedia]"

# Expérience 1: Wikipedia seulement
./baguettotron train --datasets wikipedia --config 321m

# Expérience 2: SYNTH seulement
./baguettotron train --datasets synth --config 321m

# Expérience 3: Mix 50/50
./baguettotron train \
  --datasets synth,wikipedia \
  --dataset-weights 0.5,0.5 \
  --config 321m

# Expérience 4: Favor SYNTH (70/30)
./baguettotron train \
  --datasets synth,wikipedia \
  --dataset-weights 0.7,0.3 \
  --config 321m
```

**Analyse:**
- Comparer les perplexités finales
- Évaluer la généralisation
- Trouver le meilleur mix

### Cas 3: Production Multi-Langue
**Objectif:** Modèle multilingue

```bash
# Installation
pip install -e ".[train,wikipedia]"

# Télécharger plusieurs langues
python scripts/prepare_wikipedia_data.py --lang simple --tokenize
python scripts/prepare_wikipedia_data.py --lang fr --tokenize
python scripts/prepare_wikipedia_data.py --lang en --max-samples 100000 --tokenize

# Training multilingue
./baguettotron train \
  --datasets wikipedia_simple,wikipedia_fr,wikipedia_en \
  --dataset-weights 0.2,0.4,0.4 \
  --config 321m \
  --epochs 100
```

**Résultat:**
- Modèle comprenant plusieurs langues
- Bon équilibre entre langues
- Généralisation inter-linguale

### Cas 4: Production Officielle
**Objectif:** Reproduction du modèle officiel

```bash
# Installation
pip install -e ".[train,synth]"

# Training avec SYNTH complet (comme l'officiel)
./baguettotron train \
  --datasets synth \
  --config 321m \
  --epochs 100 \
  --batch-size 64 \
  --mixed-precision \
  --compile
```

**Note:** Pour le modèle officiel exact, utilisez uniquement SYNTH.

## 🎨 Stratégies de Mixing

### 1. Equal Weighting (Par Défaut)
```bash
# Poids égaux pour chaque dataset
./baguettotron train --datasets synth,wikipedia
# Équivalent à: --dataset-weights 0.5,0.5
```

**Quand utiliser:**
- Vous voulez un mix équilibré
- Pas de préférence particulière
- Exploration initiale

### 2. Quality-Based Weighting
```bash
# Favoriser le dataset de meilleure qualité
./baguettotron train \
  --datasets synth,demo \
  --dataset-weights 0.9,0.1
```

**Quand utiliser:**
- Un dataset est clairement meilleur
- Vous voulez juste un peu de diversité
- Production avec fallback

### 3. Size-Based Weighting
```bash
# Compenser les différences de taille
# SYNTH: 10K samples, Wikipedia: 200K samples
# Donner plus de poids à SYNTH
./baguettotron train \
  --datasets synth,wikipedia \
  --dataset-weights 0.7,0.3
```

**Quand utiliser:**
- Les datasets ont des tailles très différentes
- Vous voulez un sampling plus uniforme
- Éviter le sur-apprentissage du gros dataset

### 4. Task-Based Weighting
```bash
# Optimiser pour une tâche spécifique
# Ex: Raisonnement (SYNTH) + Culture générale (Wikipedia)
./baguettotron train \
  --datasets synth,wikipedia \
  --dataset-weights 0.6,0.4
```

**Quand utiliser:**
- Objectif de tâche spécifique
- Certains datasets sont plus pertinents
- Fine-tuning pour un domaine

## 🔍 Monitoring Multi-Dataset

### Vérifier le Mixing Pendant Training
```python
# Les logs affichent la source de chaque batch
# Exemple de sortie:
# Epoch 1/10
# Batch 1: synth (70%), wikipedia (30%)
# Batch 2: synth (65%), wikipedia (35%)
# ...
# Average: synth (68%), wikipedia (32%)
```

### Metrics par Dataset
```python
# Training script peut log des métriques par dataset
# - Loss par dataset
# - Perplexity par dataset
# - Tokens/sec par dataset
```

## 💡 Best Practices

### 1. Commencer Simple
```bash
# D'abord un seul dataset
./baguettotron train --datasets wikipedia

# Puis ajouter progressivement
./baguettotron train --datasets wikipedia,demo
./baguettotron train --datasets wikipedia,synth
```

### 2. Valider Chaque Dataset
```bash
# Vérifier chaque dataset avant de combiner
./baguettotron dataset validate --file data/wikipedia_simple_tokens.json
./baguettotron dataset validate --file data/train_tokens.json
```

### 3. Expérimenter avec les Poids
```bash
# Tester différents ratios
for ratio in "0.5,0.5" "0.7,0.3" "0.3,0.7"; do
  ./baguettotron train \
    --datasets synth,wikipedia \
    --dataset-weights $ratio \
    --output-dir outputs/ratio_${ratio}
done
```

### 4. Monitor la Distribution
```python
# Vérifier que le sampling respecte les poids
# Si vous demandez 70/30, vous devriez obtenir ~70/30
```

## 🚀 Commandes Utiles

### Lister les Datasets Disponibles
```bash
# Via auto-download module
python -m baguettotron.data.auto_download --list

# Via CLI
./baguettotron dataset list
```

### Télécharger des Datasets Additionnels
```bash
# Télécharger Wikipedia FR
python scripts/prepare_wikipedia_data.py --lang fr --tokenize

# Télécharger SYNTH subset
python scripts/prepare_synth_data.py --subset --max-samples 10000 --tokenize
```

### Training avec Config YAML
```yaml
# configs/multi_dataset.yaml
datasets:
  - name: synth
    weight: 0.7
  - name: wikipedia
    weight: 0.3

model:
  config: 321m

training:
  epochs: 100
  batch_size: 64
```

```bash
./baguettotron train --config configs/multi_dataset.yaml
```

## 📊 Résultats Attendus

### Avantages du Multi-Dataset
✅ **Meilleure généralisation:** Le modèle apprend de sources variées
✅ **Robustesse:** Moins de sur-apprentissage sur un dataset
✅ **Flexibilité:** Contrôle du mix via poids
✅ **Scalabilité:** Facile d'ajouter de nouveaux datasets

### Compromis
⚠️ **Complexité:** Plus de paramètres à tuner
⚠️ **Temps:** Training peut être plus long
⚠️ **Espace disque:** Plusieurs datasets requis

## 🎯 Conclusion

Le système multi-dataset de Baguettotron offre:
1. **Flexibilité:** Combinez autant de datasets que voulu
2. **Contrôle:** Poids personnalisables par dataset
3. **Simplicité:** Auto-détection et configuration facile
4. **Performance:** Sampling pondéré efficace

**Recommandation:**
Commencez avec Wikipedia Simple, puis ajoutez progressivement d'autres datasets selon vos besoins.

---

**Dernière mise à jour:** 2024-11-12
**Version:** 1.0.0
**Feature:** Multi-dataset intelligent training
