"""Utilitaires d'inférence ONNX pour le classifieur Pokémon."""

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.config import CLASS_NAMES_PATH, CONFIDENCE_THRESHOLD, IMG_SIZE

# Au-dessus de ce seuil d'entropie normalisée, la distribution softmax est
# trop plate pour être fiable, même si la confiance top-1 dépasse le seuil.
HIGH_ENTROPY_THRESHOLD = 0.5


def load_class_names(path=None) -> list[str]:
    """Charge la liste des noms de classes depuis le fichier texte."""
    path = path or CLASS_NAMES_PATH
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Prétraite une image PIL pour l'inférence MobileNetV2 exportée en ONNX."""
    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(img_array, axis=0)


def prediction_entropy(probabilities: np.ndarray) -> float:
    """Entropie normalisée de la distribution softmax (0 = certain, 1 = uniforme).

    Complète le seuil de confiance pour détecter le hors-distribution :
    une image qui n'est pas un Pokémon produit souvent une distribution
    plate sur les 151 classes, donc une entropie élevée.
    """
    probs = np.clip(probabilities, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(len(probs)))


def predict(
    session: ort.InferenceSession,
    image: Image.Image,
    class_names: list[str],
    top_k: int = 5,
) -> dict:
    """Prédit le Pokémon à partir d'une image.

    Returns:
        dict avec predicted_class, confidence, top_k predictions, entropy,
        is_low_confidence, is_uncertain
    """
    img_array = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    predictions = session.run([output_name], {input_name: img_array})[0][0]

    top_k = min(top_k, len(class_names))
    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_predictions = [
        {"class": class_names[i], "confidence": float(predictions[i])}
        for i in top_indices
    ]

    entropy = prediction_entropy(predictions)
    best = top_predictions[0]
    is_low_confidence = best["confidence"] < CONFIDENCE_THRESHOLD
    return {
        "predicted_class": best["class"],
        "confidence": best["confidence"],
        "top_k": top_predictions,
        "entropy": entropy,
        "is_low_confidence": is_low_confidence,
        "is_uncertain": is_low_confidence or entropy > HIGH_ENTROPY_THRESHOLD,
    }
