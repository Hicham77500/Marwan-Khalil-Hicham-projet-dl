# Document de Conception — Classifieur Pokémon Gen 1

> **Groupe** : Marwan, Khalil & Hicham (lead)  
> **Date** : Jour 5 — Projet Deep Learning  
> **Repo** : [Marwan-Khalil-Hicham-projet-dl](https://github.com/Hicham77500/Marwan-Khalil-Hicham-projet-dl)

---

## 1. Le problème

Identifier automatiquement un Pokémon de la **première génération** (151 espèces) à partir d'une photo. Classification fine-grained : Bulbasaur vs Ivysaur vs Venusaur, c'est subtil.

**Cas d'usage** : app mobile pour collectionneurs, outil éducatif, démo accessible à tout public.

---

## 2. Le dataset

| Propriété | Valeur |
|-----------|--------|
| **Source** | [7,000 Labeled Pokemon (Kaggle)](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) |
| **Miroir** | [fcakyon/pokemon-classification (HuggingFace)](https://huggingface.co/datasets/fcakyon/pokemon-classification) |
| **Taille** | 6 905 images (après filtre Gen 1) |
| **Classes** | 148 (Gen 1, 3 noms absents : MrMime, Nidoran♀/♂) |
| **Format** | Images JPG/PNG, organisées par dossier/classe |
| **Split** | 75% train / 15% val / 10% test (stratifié) |

**Distribution** : ~46 images par classe en moyenne. Certaines classes sont sous-représentées (évolutions rares).

---

## 3. Architecture

```
Input (224×224×3)
    │
    ▼
MobileNetV2 (pré-entraîné ImageNet, base gelée puis fine-tunée)
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
Dropout (0.3)
    │
    ▼
Dense (151, softmax)
    │
    ▼
Prédiction : nom du Pokémon + confidence
```

### Justification du choix

| Choix | Pourquoi |
|-------|----------|
| **CNN + Transfer Learning** | Données = images → architecture Vision |
| **MobileNetV2** | Dataset petit (~7000 img), entraînement < 20 min sur Colab T4 |
| **Pas from scratch** | 46 img/classe insuffisant pour apprendre des features visuelles from scratch |
| **2 phases** | Phase 1 : tête gelée. Phase 2 : fine-tuning couches hautes (LR ÷ 10) |
| **Data augmentation** | Flip, brightness, contrast pour lutter contre l'overfitting |

### Baseline

- **Aléatoire** : 1/151 ≈ **0.66%** accuracy
- **Objectif** : > 70% accuracy (battre largement le singe qui clique au hasard)

---

## 4. Plan d'entraînement

| Paramètre | Valeur |
|-----------|--------|
| Plateforme | Google Colab T4 / Kaggle Notebooks |
| Optimizer | Adam |
| Learning rate | 1e-4 (phase 1), 1e-5 (fine-tuning) |
| Loss | sparse_categorical_crossentropy |
| Batch size | 32 |
| Epochs | 10 (phase 1) + 5 (fine-tuning) |
| Callbacks | EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint |
| Métrique cible | val_accuracy > 70% |

---

## 5. Répartition des rôles

| Membre | Rôle | Fichiers | Checkpoint |
|--------|------|----------|------------|
| **Hicham** (lead) | Infrastructure, config, data pipeline, README, git | `src/config.py`, `src/data_loader.py`, `README.md` | Conception + premier commit |
| **Marwan** | Modèle + entraînement | `src/model.py`, `src/train.py` | Premier `model.fit()` + modèle convergé |
| **Khalil** | WebApp + évaluation + tests adversariaux | `app.py`, `src/predict.py`, `src/evaluate.py` | WebApp live + README tests |

---

## 6. Livrable WebApp

- **Framework** : Streamlit
- **Fonctionnalités** :
  - Upload image (JPG/PNG)
  - Prédiction avec nom du Pokémon (pas juste un index)
  - Top 5 prédictions avec barres de confiance
  - Avertissement si confidence < 30% (hors-distribution)
  - Message clair si aucune image uploadée

---

## 7. Questions ouvertes

- [ ] Filtrer strictement les 151 premiers Pokémon ou garder toutes les classes du dataset ?
- [ ] MobileNetV2 vs EfficientNetB0 : comparer les deux si le temps le permet
- [ ] Modèle > 100 MB → héberger sur Google Drive (lien dans README)

---

## 8. Jalon qualité (avant déploiement)

- [ ] Le modèle bat-il le baseline aléatoire (0.66%) ? → **OUI si accuracy > 1%**
- [ ] La loss de validation est-elle en baisse ? → Vérifier courbes TensorBoard
- [ ] Quelle configuration a produit le meilleur résultat ? → Documenter dans README
