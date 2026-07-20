"""Configuration centralisée du projet Pokémon Gen 1."""

from pathlib import Path

# --- Chemins ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Dataset : 7,000 Labeled Pokemon (lantian773030/pokemonclassification)
# Miroir HuggingFace : fcakyon/pokemon-classification
# Structure après extraction : data/raw/<NomPokemon>/*.jpg
KAGGLE_DATASET = "lantian773030/pokemonclassification"
HUGGINGFACE_DATASET = "fcakyon/pokemon-classification"

# --- Classes ---
NUM_CLASSES = 151  # Pokémon 1ère génération (#001 Bulbasaur → #151 Mew)

# --- Images ---
IMG_SIZE = (224, 224)
IMG_CHANNELS = 3

# --- Entraînement (optimisé Colab T4, < 20 min) ---
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.10
RANDOM_SEED = 42

# --- Transfer learning ---
BASE_MODEL = "MobileNetV2"  # Alternative : EfficientNetB0
FINE_TUNE_AT = 100  # Couche à partir de laquelle on dégèle pour fine-tuning
DROPOUT_RATE = 0.3

# --- Modèle sauvegardé ---
MODEL_PATH = MODELS_DIR / "pokemon_classifier.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.txt"

# --- Baseline ---
# Accuracy aléatoire pour 151 classes équilibrées
RANDOM_BASELINE = 1 / NUM_CLASSES  # ≈ 0.66%

# --- WebApp ---
CONFIDENCE_THRESHOLD = 0.30  # En dessous : avertissement hors-distribution
