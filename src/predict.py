"""Utilitaires d'inférence pour le classifieur Pokémon.

Responsable : Khalil
"""

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import CLASS_NAMES_PATH, CONFIDENCE_THRESHOLD, IMG_SIZE, MODEL_PATH


def load_class_names(path=None) -> list[str]:
    """Charge la liste des noms de classes depuis le fichier texte."""
    path = path or CLASS_NAMES_PATH
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Prétraite une image PIL pour l'inférence MobileNetV2."""
    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


def predict(
    model: tf.keras.Model,
    image: Image.Image,
    class_names: list[str],
    top_k: int = 5,
) -> dict:
    """Prédit le Pokémon à partir d'une image.

    Returns:
        dict avec predicted_class, confidence, top_k predictions, is_low_confidence
    """
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_predictions = [
        {"class": class_names[i], "confidence": float(predictions[i])}
        for i in top_indices
    ]

    best = top_predictions[0]
    return {
        "predicted_class": best["class"],
        "confidence": best["confidence"],
        "top_k": top_predictions,
        "is_low_confidence": best["confidence"] < CONFIDENCE_THRESHOLD,
    }
