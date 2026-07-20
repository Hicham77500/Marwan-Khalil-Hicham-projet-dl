"""Évaluation du modèle — métriques, matrice de confusion, analyse d'erreurs.

Responsable : Khalil
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.config import MODEL_PATH, MODELS_DIR, NUM_CLASSES, RANDOM_BASELINE
from src.data_loader import create_class_mapping, get_data_paths, make_tf_dataset, split_dataset


def evaluate_model(model_path: Path | None = None, save_plots: bool = True) -> dict:
    """Évalue le modèle sur le test set et génère la matrice de confusion."""
    model_path = model_path or MODEL_PATH
    model = tf.keras.models.load_model(str(model_path))

    image_paths, labels = get_data_paths()
    class_to_idx, idx_to_class = create_class_mapping(labels)
    _, _, test_paths, _, _, test_labels = split_dataset(image_paths, labels)
    test_ds = make_tf_dataset(test_paths, test_labels, class_to_idx, shuffle=False)

    # Prédictions
    y_true = []
    y_pred = []
    all_preds = []
    for images, labels_batch in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels_batch.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        all_preds.append(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    all_preds = np.concatenate(all_preds)

    accuracy = np.mean(y_true == y_pred)
    # Top-5 : la bonne classe figure dans les 5 prédictions les plus probables.
    # Pertinent en fine-grained : les évolutions proches (Bulbasaur/Ivysaur)
    # se retrouvent souvent dans le top-5 même quand le top-1 se trompe.
    top5_indices = np.argsort(all_preds, axis=1)[:, -5:]
    top5_accuracy = float(np.mean([t in row for t, row in zip(y_true, top5_indices)]))
    report = classification_report(y_true, y_pred, target_names=idx_to_class, output_dict=True)

    # Paires de classes les plus confondues
    cm = confusion_matrix(y_true, y_pred)
    confused_pairs = _find_confused_pairs(cm, idx_to_class, top_n=10)

    results = {
        "test_accuracy": float(accuracy),
        "test_top5_accuracy": top5_accuracy,
        "baseline": RANDOM_BASELINE,
        "beats_baseline": accuracy > RANDOM_BASELINE,
        "most_confused_pairs": confused_pairs,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }

    print(f"\nTest Accuracy : {accuracy:.2%}")
    print(f"Top-5 Accuracy : {top5_accuracy:.2%}")
    print(f"Baseline aléatoire : {RANDOM_BASELINE:.2%}")
    print(f"Macro F1 : {report['macro avg']['f1-score']:.3f}")
    print("\nTop 5 paires confondues :")
    for pair in confused_pairs[:5]:
        print(f"  {pair['true_class']} ↔ {pair['predicted_class']} : {pair['count']} erreurs")

    if save_plots:
        _plot_confusion_matrix(cm, idx_to_class, MODELS_DIR / "confusion_matrix.png")
        results["confusion_matrix_path"] = str(MODELS_DIR / "confusion_matrix.png")

    results_path = MODELS_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def _find_confused_pairs(cm: np.ndarray, class_names: list[str], top_n: int = 10) -> list[dict]:
    """Trouve les paires de classes les plus souvent confondues."""
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true_class": class_names[i],
                    "predicted_class": class_names[j],
                    "count": int(cm[i, j]),
                })
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_n]


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path):
    """Génère et sauvegarde la matrice de confusion (top 20 classes pour lisibilité)."""
    n = min(20, len(class_names))
    cm_subset = cm[:n, :n]
    names_subset = class_names[:n]

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_subset, annot=True, fmt="d", cmap="Blues", xticklabels=names_subset, yticklabels=names_subset)
    plt.title("Matrice de confusion — Top 20 classes")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Matrice de confusion sauvegardée : {output_path}")


if __name__ == "__main__":
    evaluate_model()
