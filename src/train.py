"""Script d'entraînement du classifieur Pokémon.

Responsable : Marwan

Usage :
    python -m src.train                  # Entraînement complet (2 phases)
    python -m src.train --epochs 5       # Test rapide
    python -m src.train --no-finetune    # Phase 1 uniquement
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    LOGS_DIR,
    MODEL_PATH,
    NUM_CLASSES,
    RANDOM_BASELINE,
)
from src.data_loader import (
    create_class_mapping,
    get_data_paths,
    make_tf_dataset,
    save_class_names,
    split_dataset,
)
from src.model import build_model, compile_model, get_callbacks, unfreeze_base_model


def train(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    fine_tune: bool = True,
    fine_tune_epochs: int = 5,
) -> dict:
    """Pipeline d'entraînement en 2 phases."""
    print("=" * 60)
    print("ENTRAÎNEMENT — Classifieur Pokémon Gen 1")
    print(f"Baseline aléatoire : {RANDOM_BASELINE:.2%} ({NUM_CLASSES} classes)")
    print("=" * 60)

    # 1. Charger les données
    image_paths, labels = get_data_paths()
    class_to_idx, idx_to_class = create_class_mapping(labels)
    save_class_names(idx_to_class)

    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_dataset(
        image_paths, labels
    )

    train_ds = make_tf_dataset(train_paths, train_labels, class_to_idx, batch_size, shuffle=True, augment=True)
    val_ds = make_tf_dataset(val_paths, val_labels, class_to_idx, batch_size, shuffle=False)
    test_ds = make_tf_dataset(test_paths, test_labels, class_to_idx, batch_size, shuffle=False)

    # 2. Phase 1 — Base gelée
    print("\n--- Phase 1 : Entraînement tête de classification (base gelée) ---")
    model = build_model(num_classes=len(idx_to_class), trainable_base=False)
    model = compile_model(model, learning_rate=LEARNING_RATE)
    model.summary()

    phase1_epochs = max(epochs - fine_tune_epochs, 3) if fine_tune else epochs
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=get_callbacks(str(LOGS_DIR / "phase1")),
        verbose=1,
    )

    # 3. Phase 2 — Fine-tuning
    history2 = None
    if fine_tune:
        print("\n--- Phase 2 : Fine-tuning (couches hautes dégelées) ---")
        model = unfreeze_base_model(model)
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            callbacks=get_callbacks(str(LOGS_DIR / "phase2")),
            verbose=1,
        )

    # 4. Évaluation sur le test set
    print("\n--- Évaluation finale sur test set ---")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy : {test_accuracy:.2%}")
    print(f"Baseline aléatoire : {RANDOM_BASELINE:.2%}")
    print(f"Amélioration vs baseline : {test_accuracy / RANDOM_BASELINE:.1f}x")

    # 5. Sauvegarde
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    print(f"\nModèle sauvegardé : {MODEL_PATH}")

    # 6. Métriques
    results = {
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "baseline": RANDOM_BASELINE,
        "num_classes": len(idx_to_class),
        "epochs_phase1": phase1_epochs,
        "epochs_phase2": fine_tune_epochs if fine_tune else 0,
        "timestamp": datetime.now().isoformat(),
        "phase1_val_accuracy": float(max(history1.history["val_accuracy"])),
    }
    if history2:
        results["phase2_val_accuracy"] = float(max(history2.history["val_accuracy"]))

    results_path = MODEL_PATH.parent / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Résultats sauvegardés : {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Entraîner le classifieur Pokémon")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Nombre total d'epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-finetune", action="store_true", help="Désactiver le fine-tuning")
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        fine_tune=not args.no_finetune,
        fine_tune_epochs=args.fine_tune_epochs,
    )


if __name__ == "__main__":
    main()
