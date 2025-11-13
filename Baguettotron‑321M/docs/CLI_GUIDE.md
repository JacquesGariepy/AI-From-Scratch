# Baguettotron CLI Guide

Guide complet du CLI unifié de Baguettotron.

## 🎯 Vue d'Ensemble

Baguettotron fournit un CLI moderne et unifié pour toutes les opérations:
- ✅ Training (single & multi-dataset)
- ✅ Generation
- ✅ Dataset management

## 📝 Commandes Principales

### Training

#### Single Dataset (Traditionnel)
```bash
# Training avec un seul dataset
./baguettotron train \
  --data data/train.json \
  --config tiny \
  --epochs 10 \
  --batch-size 32
```

#### Multi-Dataset avec Auto-Détection
```bash
# Détecte et utilise tous les datasets disponibles
./baguettotron train \
  --datasets auto \
  --config tiny \
  --epochs 10
```

#### Multi-Dataset avec Sélection Manuelle
```bash
# Datasets spécifiques avec poids personnalisés
./baguettotron train \
  --datasets synth,wikipedia \
  --dataset-weights 0.7,0.3 \
  --config 321m \
  --epochs 100 \
  --batch-size 64 \
  --output-dir outputs/multi-dataset
```

### Generation

```bash
# Génération de texte
./baguettotron generate \
  --checkpoint outputs/model.pt \
  --prompt "Bonjour" \
  --max-length 100 \
  --temperature 0.8
```

### Dataset Management

#### Préparer un Dataset
```bash
# Wikipedia
./baguettotron dataset prepare \
  --type wikipedia \
  --lang simple \
  --tokenize

# SYNTH
./baguettotron dataset prepare \
  --type synth \
  --max-samples 10000 \
  --tokenize

# Demo
./baguettotron dataset prepare \
  --type demo
```

#### Lister les Datasets
```bash
./baguettotron dataset list
```

#### Valider un Dataset
```bash
./baguettotron dataset validate --file data/train.json
```

## 🔧 Arguments Détaillés

### Train Command

| Argument | Type | Description | Défaut |
|----------|------|-------------|--------|
| `--data` | str | Fichier de données (mode single) | - |
| `--datasets` | str | Datasets (comma-separated ou "auto") | - |
| `--dataset-weights` | str | Poids des datasets (e.g., "0.7,0.3") | Equal |
| `--config` | str | Configuration modèle (tiny/321m/custom) | 321m |
| `--epochs` | int | Nombre d'epochs | 10 |
| `--batch-size` | int | Taille du batch | 32 |
| `--output-dir` | str | Répertoire de sortie | outputs |

**Note:** Vous devez spécifier soit `--data` soit `--datasets`, pas les deux.

### Generate Command

| Argument | Type | Description | Défaut |
|----------|------|-------------|--------|
| `--checkpoint` | str | Chemin du checkpoint (requis) | - |
| `--prompt` | str | Prompt de génération | "" |
| `--max-length` | int | Longueur maximale | 100 |
| `--temperature` | float | Température de sampling | 1.0 |

### Dataset Prepare Command

| Argument | Type | Description | Défaut |
|----------|------|-------------|--------|
| `--type` | str | Type de dataset (wikipedia/synth/demo) (requis) | - |
| `--lang` | str | Langue (simple/fr/en) pour Wikipedia | - |
| `--max-samples` | int | Nombre max d'échantillons | All |
| `--tokenize` | flag | Pre-tokenizer les données | False |

### Dataset Validate Command

| Argument | Type | Description |
|----------|------|-------------|
| `--file` | str | Fichier dataset à valider (requis) |

## 📚 Exemples d'Utilisation

### Scénario 1: Quick Start
```bash
# 1. Préparer dataset
./baguettotron dataset prepare --type demo

# 2. Training rapide
./baguettotron train --data data/train.json --config tiny --epochs 5

# 3. Génération
./baguettotron generate --checkpoint outputs/checkpoint.pt --prompt "Test"
```

### Scénario 2: Multi-Dataset Development
```bash
# 1. Préparer plusieurs datasets
./baguettotron dataset prepare --type wikipedia --lang simple --tokenize
./baguettotron dataset prepare --type demo

# 2. Lister les datasets disponibles
./baguettotron dataset list

# 3. Training avec auto-détection
./baguettotron train --datasets auto --config tiny --epochs 10
```

### Scénario 3: Production avec SYNTH + Wikipedia
```bash
# 1. Préparer datasets
./baguettotron dataset prepare --type synth --max-samples 50000 --tokenize
./baguettotron dataset prepare --type wikipedia --lang fr --tokenize

# 2. Valider
./baguettotron dataset validate --file data/synth_tokens.json
./baguettotron dataset validate --file data/wikipedia_fr_tokens.json

# 3. Training avec poids optimisés
./baguettotron train \
  --datasets synth,wikipedia \
  --dataset-weights 0.7,0.3 \
  --config 321m \
  --epochs 100 \
  --batch-size 64 \
  --output-dir outputs/production

# 4. Génération
./baguettotron generate \
  --checkpoint outputs/production/final_model.pt \
  --prompt "Bonjour, comment allez-vous?" \
  --max-length 200 \
  --temperature 0.9
```

## 🎨 Modes de Training

### Mode 1: Single Dataset (Traditionnel)
```bash
./baguettotron train --data data/train.json --config tiny
```
**Utilise:** Un seul fichier dataset
**Avantages:** Simple, direct
**Cas d'usage:** Testing, développement rapide

### Mode 2: Multi-Dataset Auto
```bash
./baguettotron train --datasets auto --config tiny
```
**Utilise:** Tous les datasets trouvés dans data/
**Avantages:** Pas besoin de spécifier
**Cas d'usage:** Expérimentation, utiliser tout ce qui est disponible

### Mode 3: Multi-Dataset Manuel
```bash
./baguettotron train --datasets synth,wikipedia --dataset-weights 0.6,0.4
```
**Utilise:** Datasets spécifiés avec poids
**Avantages:** Contrôle total
**Cas d'usage:** Production, optimisation fine

## 🔍 Validation et Debugging

### Vérifier les Arguments
```bash
# Voir l'aide complète
./baguettotron --help
./baguettotron train --help
./baguettotron dataset --help
```

### Dry-Run (sans exécution)
```bash
# Le CLI affiche ce qui serait exécuté
# (ajoutez echo devant la commande pour voir)
```

### Logs et Output
```bash
# Training avec logs
./baguettotron train --datasets auto --config tiny 2>&1 | tee training.log

# Check output directory
ls -lh outputs/
```

## ⚡ Tips & Tricks

### 1. Alias Utiles
```bash
# Ajoutez à votre .bashrc ou .zshrc
alias btrain='./baguettotron train'
alias bgen='./baguettotron generate'
alias bdata='./baguettotron dataset'

# Utilisation
btrain --datasets auto --config tiny
bgen --checkpoint model.pt --prompt "Hello"
bdata list
```

### 2. Scripts Batch
```bash
#!/bin/bash
# train_all.sh - Entraîne avec différentes configs

for config in tiny 321m; do
  ./baguettotron train \
    --datasets auto \
    --config $config \
    --epochs 10 \
    --output-dir outputs/$config
done
```

### 3. Makefile Integration
```makefile
train-multi:
	./baguettotron train \
		--datasets synth,wikipedia \
		--dataset-weights 0.7,0.3 \
		--config 321m

generate:
	./baguettotron generate \
		--checkpoint outputs/model.pt \
		--prompt "Test"
```

## 🐛 Troubleshooting

### Erreur: "Either --data or --datasets must be specified"
**Cause:** Aucun argument de données fourni
**Solution:** Ajoutez soit `--data FILE` soit `--datasets auto`

```bash
# ❌ Erreur
./baguettotron train --config tiny

# ✅ Correct
./baguettotron train --datasets auto --config tiny
```

### Erreur: "No datasets found"
**Cause:** Aucun dataset dans data/
**Solution:** Préparez un dataset d'abord

```bash
./baguettotron dataset prepare --type demo
./baguettotron train --datasets auto --config tiny
```

### Erreur: "Dataset X not found"
**Cause:** Dataset spécifié non disponible
**Solution:** Listez les datasets disponibles

```bash
./baguettotron dataset list
./baguettotron train --datasets <available-dataset>
```

## 📊 Comparaison avec Scripts Directs

### Ancien Workflow (Scripts)
```bash
python scripts/prepare_wikipedia_data.py --lang simple --tokenize
python scripts/train.py --train-data data/wikipedia_simple_tokens.json --config tiny
python scripts/generate.py --checkpoint outputs/model.pt --prompt "Test"
```

### Nouveau Workflow (CLI)
```bash
./baguettotron dataset prepare --type wikipedia --lang simple --tokenize
./baguettotron train --data data/wikipedia_simple_tokens.json --config tiny
./baguettotron generate --checkpoint outputs/model.pt --prompt "Test"
```

**Avantages du CLI:**
- ✅ Syntaxe cohérente
- ✅ Validation des arguments
- ✅ Messages d'erreur clairs
- ✅ Multi-dataset natif
- ✅ Auto-détection

## 🎯 Best Practices

1. **Utilisez --datasets auto pour l'expérimentation**
   ```bash
   ./baguettotron train --datasets auto --config tiny
   ```

2. **Validez toujours vos datasets**
   ```bash
   ./baguettotron dataset validate --file data/train.json
   ```

3. **Spécifiez des poids pour la production**
   ```bash
   ./baguettotron train --datasets synth,wiki --dataset-weights 0.7,0.3
   ```

4. **Utilisez des output-dir descriptifs**
   ```bash
   ./baguettotron train --datasets auto --output-dir outputs/experiment-1
   ```

5. **Sauvegardez vos commandes**
   ```bash
   # Créez un script pour reproduire
   echo './baguettotron train --datasets auto --config tiny' > train.sh
   ```

---

**Version:** 1.0.0
**Dernière mise à jour:** 2024-11-12
**Support multi-dataset:** ✅ Complet
