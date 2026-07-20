# 📋 Partie de Marwan — Modèle & Entraînement

Salut Marwan ! Voici ta partie du projet. Tu t'occupes de **l'architecture du modèle** et du **script d'entraînement**.

---

## Tes fichiers

| Fichier | Description |
|---------|-------------|
| `src/model.py` | Architecture MobileNetV2 + transfer learning |
| `src/train.py` | Pipeline d'entraînement en 2 phases |

## Ce que Hicham a déjà fait pour toi

- `src/config.py` — tous les hyperparamètres (batch size, epochs, LR, etc.)
- `src/data_loader.py` — chargement dataset, split train/val/test, tf.data.Dataset avec augmentation
- Structure du projet + README

Tu n'as **pas besoin** de toucher au data loading. Concentre-toi sur le modèle et l'entraînement.

---

## Ta mission (checkpoints)

### Checkpoint 1 — Premier `model.fit()` (avant le déjeuner)

```bash
# 1. Télécharger le dataset (si pas déjà fait)
kaggle datasets download -d vishalsubbiah/pokemon-images-and-types -p data/raw --unzip

# 2. Vérifier que les données chargent
python -c "from src.data_loader import inspect_dataset; inspect_dataset()"

# 3. Lancer un entraînement rapide (1-2 epochs pour tester le pipeline)
python -m src.train --epochs 2 --no-finetune
```

**Commit attendu** : `feat: pipeline de base, premier epoch`

### Checkpoint 2 — Modèle convergé (après-midi)

```bash
# Entraînement complet (2 phases)
python -m src.train

# Vérifier les résultats
cat models/training_results.json
```

**Objectifs** :
- Val accuracy > 70%
- Battre le baseline aléatoire (0.66%)
- Modèle sauvegardé dans `models/pokemon_classifier.keras`

**Commit attendu** : `feat: modèle convergé, accuracy X%`

---

## Architecture à implémenter (déjà codée dans `model.py`)

```
Phase 1 — Base gelée (10 epochs) :
    MobileNetV2 (frozen) → GlobalAveragePooling2D → Dropout(0.3) → Dense(151)

Phase 2 — Fine-tuning (5 epochs) :
    Dégeler à partir de la couche 100 → LR divisé par 10
```

### Fonctions clés dans `model.py`

- `build_model()` — construit le modèle avec base gelée
- `compile_model()` — Adam + sparse categorical crossentropy
- `unfreeze_base_model()` — dégèle les couches hautes pour phase 2
- `get_callbacks()` — EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

---

## Conseils

1. **Commence sur Colab** si ta machine est lente : upload le repo, installe les deps, lance `train.py`
2. **Regarde TensorBoard** : `tensorboard --logdir logs/` pour diagnostiquer overfitting/underfitting
3. **Si overfitting** (val_loss monte, train_loss descend) : augmente le dropout, ajoute plus d'augmentation
4. **Si underfitting** (les deux stagnent) : augmente les epochs, baisse le LR
5. **Compare** : lance avec et sans fine-tuning, documente la différence dans le README

---

## Questions à anticiper en soutenance

- "Pourquoi MobileNetV2 et pas EfficientNet ?" → Léger, rapide sur Colab T4, suffisant pour 7000 images
- "Pourquoi transfer learning ?" → 46 images/classe, trop peu pour CNN from scratch
- "Comment tu as détecté l'overfitting ?" → Courbes TensorBoard : val_loss qui remonte
- "Quelle classe est la plus difficile ?" → Les évolutions (Bulbasaur/Ivysaur/Venusaur)

---

## Branche Git

```bash
git checkout marwan
# Travaille sur tes fichiers
git add src/model.py src/train.py
git commit -m "feat: modèle convergé, accuracy X%"
```

Bonne chance ! 🚀
