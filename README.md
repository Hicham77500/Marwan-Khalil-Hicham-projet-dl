# ⚡ Pokémon Identifier — Gen 1

Classifieur d'images Deep Learning : identifier un Pokémon de la **première génération** (151 espèces) à partir d'une photo.

**Groupe** : Marwan · Khalil · Hicham (lead)  
**Architecture** : Transfer Learning MobileNetV2  
**Dataset** : [7,000 Labeled Pokemon (Kaggle)](https://www.kaggle.com/datasets/lantian773030/pokemonclassification)

---

## Installation

```bash
# Cloner le repo
git clone git@github.com:Hicham77500/Marwan-Khalil-Hicham-projet-dl.git
cd Marwan-Khalil-Hicham-projet-dl

# Environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate

# Dépendances (inférence / WebApp)
pip install -r requirements.txt

# Entraînement & évaluation (optionnel)
pip install -r requirements-train.txt
```

### Dataset

**Dataset utilisé** : [7,000 Labeled Pokemon](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) (Lance Zhang, Kaggle)

> **Pourquoi c'est difficile à trouver ?** Le cours parle de « 7000 Labelled Pokemon » (orthographe britannique), mais sur Kaggle le dataset s'appelle **« 7,000 Labeled Pokemon »** (orthographe américaine) et son slug est `lantian773030/pokemonclassification` — pas « labelled », pas « 7000 » dans l'URL. Un miroir HuggingFace existe sous `fcakyon/pokemon-classification`.

**Ne pas utiliser** : `vishalsubbiah/pokemon-images-and-types` — seulement 151 images (1 par classe), insuffisant pour l'entraînement.

| Dataset | URL | Images | Classes | Structure |
|---------|-----|--------|---------|-----------|
| **7,000 Labeled Pokemon** (choisi) | [Kaggle](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) · [HF](https://huggingface.co/datasets/fcakyon/pokemon-classification) | ~6 900 | ~150 Gen 1 | `data/raw/<Pokemon>/*.jpg` |
| Pokemon Generation One | [Kaggle](https://www.kaggle.com/datasets/thedagger/pokemon-generation-one) | ~10 000 | 151 Gen 1 | dossier par classe |
| Ultimate Pokémon Images | [Kaggle](https://www.kaggle.com/datasets/shivanshcoding/1282-pokemon-139542-images-updated-pokedex-dataset) | ~130 000 | 1 000+ (Gen I–IX) | sous-dossier classification |

**Téléchargement** (choisir une option) :

```bash
# Option 1 — Kaggle (nécessite kaggle.json dans ~/.kaggle/)
pip install kaggle
kaggle datasets download -d lantian773030/pokemonclassification -p data/raw --unzip

# Option 2 — HuggingFace (sans compte Kaggle, recommandé)
pip install huggingface_hub
hf download fcakyon/pokemon-classification --repo-type dataset --local-dir data/hf_tmp
mkdir -p data/raw
cd data/hf_tmp/data && for z in train.zip valid.zip test.zip; do unzip -q -o "$z" -d ../../raw; done
cd ../../.. && rm -rf data/hf_tmp
```

Vérifier l'installation :

```bash
python3 -c "from src.data_loader import inspect_dataset; inspect_dataset()"
# Attendu : ~6 900 images, ~148 classes, 35–66 img/classe
```

Structure attendue après extraction :
```
data/raw/
├── Bulbasaur/
│   ├── img001.jpg
│   └── ...
├── Pikachu/
│   └── ...
├── pokemon.csv          # optionnel (filtre Gen 1)
└── ... (~150 dossiers)
```

### Modèle entraîné

Le modèle est versionné dans le dépôt :

- `models/pokemon_classifier.keras` (~12 MB)
- `models/class_names.txt`

---

## Déploiement Streamlit Cloud

1. Pousser sur `main` (le modèle et `requirements.txt` doivent être sur GitHub).
2. Sur [share.streamlit.io](https://share.streamlit.io), connecter le repo et choisir `app.py`.
3. **Advanced settings → Python version : 3.11** (obligatoire pour TensorFlow).
4. Redéployer si l'installation des dépendances échoue encore.

---

## Usage

### 1. Inspecter le dataset

```bash
python -c "from src.data_loader import inspect_dataset; inspect_dataset()"
```

### 2. Entraîner le modèle

```bash
python -m src.train                  # Entraînement complet (2 phases)
python -m src.train --epochs 5       # Test rapide
python -m src.train --no-finetune    # Phase 1 uniquement
```

### 3. Évaluer

```bash
python -m src.evaluate
```

### 4. Lancer la WebApp

```bash
streamlit run app.py
```

Ouvrir http://localhost:8501 dans le navigateur.

---

## Architecture

```
MobileNetV2 (ImageNet) → GlobalAveragePooling2D → Dropout(0.3) → Dense(151, softmax)
```

| Paramètre | Valeur |
|-----------|--------|
| Input | 224×224×3 |
| Classes | 148 (Gen 1, 3 noms manquants dans le dataset source) |
| Optimizer | Adam (lr=1e-4) |
| Loss | sparse_categorical_crossentropy |
| Baseline aléatoire | 0.66% (1/151) |

**Justification** : dataset de ~7000 images pour 151 classes (~46 img/classe) → transfer learning obligatoire. MobileNetV2 léger, entraînement < 20 min sur Colab T4.

---

## Résultats

| Configuration | Val Accuracy | Test Accuracy | Temps |
|---------------|-------------|---------------|-------|
| Phase 1 (base gelée, 10 epochs) | 65.35% | — | ~7 min |
| Phase 2 (+ fine-tuning, 5 epochs) | 78.76% | **75.54%** | ~3 min |

**Dataset** : 6 905 images · 148 classes · 35–66 img/classe · split 5178/1036/691

### Jalon qualité

- [x] Le modèle bat le baseline aléatoire (0.66%) → **75.54%** (~114×)
- [x] La loss de validation descend sur plusieurs epochs
- [x] Meilleure config : Phase 2 (fine-tuning) — +13% val accuracy vs phase 1

---

## Tests adversariaux

| Input | Comportement observé | Explication |
|-------|---------------------|-------------|
| Image PNG normale d'un Pokémon (happy path) | Prédiction correcte, confidence > 80% | Dataset bien représenté |
| Image 5×5 pixels | Prédiction aberrante, confidence élevée | Le resize introduit des artefacts, le modèle hallucine |
| Aucune image (champ vide) | Message "Aucune image sélectionnée" | Gestion explicite dans `app.py` |
| Photo de chat (hors classes) | Prédit un Pokémon aléatoire, confidence variable | Softmax normalise toujours à 1.0 — hors-distribution non détecté nativement. Avertissement si confidence < 30% |
| Image .tiff ou .bmp | Erreur "type non supporté" | Seuls JPG et PNG sont gérés par le file_uploader |
| Bulbasaur vs Ivysaur (classes proches) | Confusion possible, confidence modérée | Classification fine-grained : évolutions visuellement similaires |
| Image retournée / très sombre | Prédiction incorrecte possible | Data augmentation limitée (flip, brightness) — pas de rotation 180° |

---

## Structure du projet

```
├── app.py                  # WebApp Streamlit (Khalil)
├── CONCEPTION.md           # Document de conception
├── requirements.txt
├── src/
│   ├── config.py           # Configuration centralisée (Hicham)
│   ├── data_loader.py      # Chargement dataset (Hicham)
│   ├── model.py            # Architecture MobileNetV2 (Marwan)
│   ├── train.py            # Script d'entraînement (Marwan)
│   ├── predict.py          # Inférence (Khalil)
│   └── evaluate.py         # Métriques + confusion matrix (Khalil)
├── data/raw/               # Images Pokémon (non versionné)
├── models/                 # Modèle entraîné (versionné pour le déploiement)
└── logs/                   # TensorBoard logs
```

---

## Limites connues

1. **Hors-distribution** : le modèle prédit toujours une classe parmi les 151, même pour une image qui n'est pas un Pokémon (softmax force une distribution)
2. **Évolutions similaires** : Bulbasaur/Ivysaur/Venusaur, Charmander/Charmeleon/Charizard sont souvent confondues
3. **Résolution** : images < 32×32 pixels donnent des prédictions peu fiables
4. **Artefacts** : images de fan-art, pixel art ou sprites peuvent être mal classifiées (dataset majoritairement des illustrations officielles)

---

## Équipe

| Membre | Rôle | Branche Git |
|--------|------|-------------|
| Hicham | Lead, infrastructure, data pipeline | `main` |
| Marwan | Modèle + entraînement | `marwan` |
| Khalil | WebApp + évaluation | `khalil` |
