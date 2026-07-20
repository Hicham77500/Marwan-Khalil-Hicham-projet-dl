"""WebApp Streamlit — Classifieur Pokémon Gen 1."""

import json
import time

import onnxruntime as ort
import streamlit as st
from PIL import Image

from src.config import CLASS_NAMES_PATH, CONFIDENCE_THRESHOLD, MODEL_PATH, MODELS_DIR
from src.predict import load_class_names, predict

TRAINING_RESULTS_PATH = MODELS_DIR / "training_results.json"

st.set_page_config(
    page_title="Pokémon Identifier",
    page_icon="⚡",
    layout="centered",
)

@st.cache_resource
def load_model():
    """Charge la session ONNX une seule fois au démarrage."""
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def render_sidebar(num_classes: int):
    """Affiche les informations du modèle déployé dans la barre latérale."""
    with st.sidebar:
        st.header("À propos du modèle")
        st.markdown(
            f"- **Architecture** : MobileNetV2 (transfer learning)\n"
            f"- **Classes** : {num_classes} Pokémon Gen 1\n"
            f"- **Entrée** : 224×224 RGB\n"
            f"- **Seuil de confiance** : {CONFIDENCE_THRESHOLD:.0%}"
        )
        if TRAINING_RESULTS_PATH.exists():
            with open(TRAINING_RESULTS_PATH) as f:
                results = json.load(f)
            accuracy = results.get("test_accuracy")
            baseline = results.get("baseline")
            if accuracy is not None:
                st.metric("Test accuracy (modèle déployé)", f"{accuracy:.1%}")
            if accuracy is not None and baseline:
                st.caption(f"Baseline aléatoire : {baseline:.2%} (×{accuracy / baseline:.0f})")


def main():
    st.title("⚡ Pokémon Identifier — Gen 1")
    st.write(
        "Uploadez une image de Pokémon et découvrez lequel c'est parmi les **151** de la première génération !"
    )

    # Vérifier que le modèle existe
    if not MODEL_PATH.exists():
        st.error(
            "Modèle non trouvé. Entraînez d'abord le modèle :\n\n"
            "```bash\n# Générer models/pokemon_classifier.onnx puis relancer l'app\n```"
        )
        st.stop()

    model = load_model()
    class_names = load_class_names()
    render_sidebar(len(class_names))

    uploaded_file = st.file_uploader(
        "Choisir une image",
        type=["jpg", "jpeg", "png"],
        help="Formats acceptés : JPG, PNG",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception:
            st.error("Impossible de lire l'image. Vérifiez le format (JPG ou PNG).")
            st.stop()

        # Vérifier résolution minimale
        if image.size[0] < 10 or image.size[1] < 10:
            st.warning("Image très basse résolution — la prédiction peut être peu fiable.")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Image uploadée", use_container_width=True)

        with col2:
            start = time.perf_counter()
            result = predict(model, image, class_names)
            inference_ms = (time.perf_counter() - start) * 1000

            # Affichage principal
            pokemon_name = result["predicted_class"].replace("-", " ").title()
            confidence = result["confidence"]

            if result["is_uncertain"]:
                reason = (
                    f"confiance faible (< {CONFIDENCE_THRESHOLD:.0%})"
                    if result["is_low_confidence"]
                    else f"distribution trop plate (entropie {result['entropy']:.2f})"
                )
                st.warning(
                    f"**{pokemon_name}** ({confidence:.1%} de confiance)\n\n"
                    f"⚠️ Prédiction incertaine : {reason} — "
                    "l'image est peut-être hors distribution (pas un Pokémon Gen 1)."
                )
            else:
                st.success(f"**{pokemon_name}** — {confidence:.1%} de confiance")

            st.caption(f"Inférence : {inference_ms:.0f} ms (CPU, ONNX Runtime)")

            # Top 5 prédictions
            st.subheader("Top 5 prédictions")
            for i, pred in enumerate(result["top_k"]):
                name = pred["class"].replace("-", " ").title()
                bar_width = pred["confidence"]
                st.progress(bar_width, text=f"{i+1}. {name} ({pred['confidence']:.1%})")

    else:
        st.info("Aucune image sélectionnée. Uploadez une image pour commencer.")

    # Footer
    st.divider()
    st.caption(
        "Projet Deep Learning — Marwan, Khalil & Hicham | "
        "Transfer Learning MobileNetV2 | 151 classes"
    )


if __name__ == "__main__":
    main()
