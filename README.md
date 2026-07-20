# ⚡ Pokémon Identifier — Gen 1

Classifieur d'images Deep Learning : identifier un Pokémon de la **première génération** (151 espèces) à partir d'une photo.

**Groupe** : Marwan · Khalil · Hicham (lead)  
**Architecture** : Transfer Learning MobileNetV2  
**Dataset** : [7000 Labelled Pokemon (Kaggle)](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types)

---

## Installation

```bash
# Cloner le repo
git clone git@github.com:Hicham77500/Marwan-Khalil-Hicham-projet-dl.git
cd Marwan-Khalil-Hicham-projet-dl

# Environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate

# Dépendances
pip install -r requirements.txt
```

### Dataset Kaggle

```bash
# Configurer l'API Kaggle (kaggle.json dans ~/.kaggle/)
kaggle datasets download -d vishalsubbiah/pokemon-images-and-types -p data/raw --unzip
```

Structure attendue après extraction :
```
data/raw/
├── abra/
│   ├── img001.png
│   └── ...
├── bulbasaur/
│   └── ...
└── ... (151 dossiers)
```

### Télécharger le modèle entraîné

> Si le fichier `.keras` dépasse 100 MB, il est hébergé sur Google Drive.

```bash
# Placer le modèle dans models/
# models/pokemon_classifier.keras
# models/class_names.txt
```

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
| Classes | 151 |
| Optimizer | Adam (lr=1e-4) |
| Loss | sparse_categorical_crossentropy |
| Baseline aléatoire | 0.66% (1/151) |

**Justification** : dataset de ~7000 images pour 151 classes (~46 img/classe) → transfer learning obligatoire. MobileNetV2 léger, entraînement < 20 min sur Colab T4.

---

## Résultats

| Configuration | Val Accuracy | Test Accuracy | Temps |
|---------------|-------------|---------------|-------|
| Phase 1 (base gelée, 10 epochs) | _à compléter_ | _à compléter_ | _à compléter_ |
| Phase 2 (+ fine-tuning, 5 epochs) | _à compléter_ | _à compléter_ | _à compléter_ |

### Jalon qualité

- [x] Le modèle bat le baseline aléatoire (0.66%) ?
- [ ] La loss de validation descend sur plusieurs epochs ?
- [ ] Quelle configuration est la meilleure ?

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
├── models/                 # Modèle entraîné (non versionné si > 100 MB)
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
