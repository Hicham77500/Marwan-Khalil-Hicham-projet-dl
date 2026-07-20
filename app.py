"""WebApp Streamlit — Classifieur Pokémon Gen 1.

Responsable : Khalil

Usage :
    streamlit run app.py
"""

import streamlit as st
import tensorflow as tf
from PIL import Image

from src.config import CLASS_NAMES_PATH, CONFIDENCE_THRESHOLD, MODEL_PATH
from src.predict import load_class_names, predict

st.set_page_config(
    page_title="Pokémon Identifier",
    page_icon="⚡",
    layout="centered",
)

@st.cache_resource
def load_model():
    """Charge le modèle une seule fois au démarrage."""
    return tf.keras.models.load_model(str(MODEL_PATH))


def main():
    st.title("⚡ Pokémon Identifier — Gen 1")
    st.write(
        "Uploadez une image de Pokémon et découvrez lequel c'est parmi les **151** de la première génération !"
    )

    # Vérifier que le modèle existe
    if not MODEL_PATH.exists():
        st.error(
            "Modèle non trouvé. Entraînez d'abord le modèle :\n\n"
            "```bash\npython -m src.train\n```"
        )
        st.stop()

    model = load_model()
    class_names = load_class_names()

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
            result = predict(model, image, class_names)

            # Affichage principal
            pokemon_name = result["predicted_class"].replace("-", " ").title()
            confidence = result["confidence"]

            if result["is_low_confidence"]:
                st.warning(
                    f"**{pokemon_name}** ({confidence:.1%} de confiance)\n\n"
                    f"⚠️ Confiance faible (< {CONFIDENCE_THRESHOLD:.0%}) — "
                    "l'image est peut-être hors distribution (pas un Pokémon Gen 1)."
                )
            else:
                st.success(f"**{pokemon_name}** — {confidence:.1%} de confiance")

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
