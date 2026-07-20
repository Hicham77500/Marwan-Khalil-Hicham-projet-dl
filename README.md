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

- `models/pokemon_classifier.keras` (~26 MB) — modèle Keras complet
- `models/pokemon_classifier.onnx` (~9.7 MB) — format servi par la WebApp
- `models/class_names.txt` — 150 noms, alignés sur les sorties du modèle
- `models/training_results.json` — métriques du run

---

## Déploiement Streamlit Cloud

1. Pousser sur `main` (le modèle, `runtime.txt` et `requirements.txt` doivent être sur GitHub).
2. Sur [share.streamlit.io](https://share.streamlit.io), connecter le repo et choisir `app.py`.
3. **Advanced settings → Python version : 3.11** (obligatoire pour TensorFlow).
4. Rebooter/redéployer l'app après la mise à jour des dépendances.

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
MobileNetV2 (ImageNet) → GlobalAveragePooling2D → Dropout(0.3) → Dense(150, softmax)
```

| Paramètre | Valeur |
|-----------|--------|
| Input | 224×224×3 |
| Classes | 150 (Gen 1, 1 nom manquant dans le dataset source) |
| Optimizer | Adam (lr=1e-4 phase 1, 1e-5 phase 2) |
| Loss | sparse_categorical_crossentropy |
| Baseline aléatoire | 0.67% (1/150) |

**Justification** : dataset de 6 820 images pour 150 classes (~45 img/classe) → transfer learning obligatoire, un CNN from scratch mémoriserait le train set sans généraliser. MobileNetV2 léger (2,45 M paramètres dont seulement 192 K entraînables en phase 1), entraînement en 7 min sur Colab T4 pour une contrainte de 20 min.

---

## Résultats

| Configuration | Val Accuracy | Val Loss | Test Accuracy | Temps |
|---------------|-------------|----------|---------------|-------|
| Phase 1 (base gelée, 10 epochs, lr 1e-4) | 62.17% | 2.15 | — | ~4 min |
| Phase 2 (+ fine-tuning 54 couches, 5 epochs, lr 1e-5) | **74.68%** | **1.39** | **74.93%** | ~3 min |

**Dataset** : 6 820 images · 150 classes · split 5115/1023/682 · entraînement total **7.1 min** sur Colab T4

**Apport du fine-tuning** : +12.5 points de val accuracy (62.17% → 74.68%). Dégeler les
54 couches hautes de MobileNetV2 avec un learning rate divisé par 10 permet au réseau de
spécialiser ses features sur les Pokémon, là où la phase 1 ne pouvait qu'apprendre une
combinaison linéaire de features ImageNet figées.

### Jalon qualité

- [x] **Le modèle bat le baseline aléatoire** (0.67%) → **74.93%**, soit **112× le hasard**
- [x] **La loss de validation descend sur plusieurs epochs** : 4.69 → 1.39, décroissance
      monotone sur les 15 epochs. `val_accuracy` progresse à chaque epoch sans exception,
      l'EarlyStopping ne s'est jamais déclenché → **pas d'overfitting observé**
- [x] **Meilleure configuration : Phase 2 (fine-tuning)** — +12.5 points vs phase 1 seule

> **Note** : les deux phases se sont terminées sur leur meilleur epoch, le modèle
> progressait donc encore. Avec un budget d'epochs plus large (le run ne prend que 7 min
> pour une contrainte de 20), on dépasserait vraisemblablement les 80%.

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
