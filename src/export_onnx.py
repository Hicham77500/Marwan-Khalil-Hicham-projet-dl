"""Export du modèle entraîné (.keras) vers ONNX pour la WebApp.

Responsable : Marwan

La WebApp (`app.py`) charge `models/pokemon_classifier.onnx` via onnxruntime
plutôt que TensorFlow : ça évite d'embarquer TF (~600 MB) sur Streamlit Cloud.
Ce script régénère l'ONNX à partir du `.keras` produit par `src/train.py`.

À relancer après chaque entraînement, sinon la démo continue de servir
l'ancien modèle.

Usage :
    python -m src.export_onnx
    python -m src.export_onnx --opset 13
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src.config import CLASS_NAMES_PATH, MODEL_PATH, TRAINED_MODEL_PATH


def export_to_onnx(
    keras_path: Path = TRAINED_MODEL_PATH,
    onnx_path: Path = MODEL_PATH,
    opset: int = 13,
) -> Path:
    """Convertit le modèle Keras en ONNX via un SavedModel intermédiaire.

    On passe par un SavedModel plutôt que de convertir l'objet Keras
    directement : tf2onnx travaille alors sur le graphe TensorFlow, ce qui le
    rend indépendant de la version de Keras (Keras 3 a changé l'API de
    sérialisation).
    """
    import tensorflow as tf

    if not keras_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {keras_path}\n"
            "Lancez d'abord l'entraînement : python -m src.train"
        )

    model = tf.keras.models.load_model(str(keras_path))
    num_classes = model.output_shape[-1]
    print(f"Modèle chargé : {keras_path.name} ({num_classes} classes en sortie)")

    tmp_dir = Path(tempfile.mkdtemp(prefix="pokemon_savedmodel_"))
    try:
        saved_model_dir = tmp_dir / "saved_model"
        model.export(str(saved_model_dir))
        print(f"SavedModel intermédiaire : {saved_model_dir}")

        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            str(saved_model_dir),
            "--output",
            str(onnx_path),
            "--opset",
            str(opset),
        ]
        print(f"Conversion ONNX (opset {opset})...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "Echec de la conversion tf2onnx.\n"
                f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Modèle ONNX écrit : {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    _verify(onnx_path, num_classes)
    return onnx_path


def _verify(onnx_path: Path, expected_classes: int) -> None:
    """Recharge l'ONNX et vérifie que la sortie correspond aux classes connues.

    Garde-fou contre le piège principal : un ONNX dont le nombre de sorties ne
    correspond plus à `class_names.txt`. La WebApp indexerait alors la liste
    des noms avec un indice décalé et afficherait le mauvais Pokémon.
    """
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    output = session.run(None, {input_meta.name: dummy})[0]

    if output.shape[-1] != expected_classes:
        raise ValueError(
            f"Incohérence : l'ONNX sort {output.shape[-1]} classes, "
            f"le modèle Keras en annonce {expected_classes}."
        )

    if CLASS_NAMES_PATH.exists():
        n_names = len([l for l in CLASS_NAMES_PATH.read_text().splitlines() if l.strip()])
        if n_names != expected_classes:
            raise ValueError(
                f"Incohérence : {CLASS_NAMES_PATH.name} contient {n_names} noms "
                f"mais le modèle sort {expected_classes} classes. "
                "Relancez l'entraînement pour régénérer les deux ensemble."
            )
        print(f"Vérification OK : {expected_classes} classes, noms alignés.")
    else:
        print(f"Vérification partielle : {expected_classes} classes en sortie "
              f"({CLASS_NAMES_PATH.name} absent).")


def main():
    parser = argparse.ArgumentParser(description="Exporter le modèle Keras en ONNX")
    parser.add_argument("--keras-path", type=Path, default=TRAINED_MODEL_PATH)
    parser.add_argument("--onnx-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    export_to_onnx(args.keras_path, args.onnx_path, args.opset)


if __name__ == "__main__":
    main()
