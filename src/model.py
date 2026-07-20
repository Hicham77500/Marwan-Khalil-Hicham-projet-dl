"""Architecture du modèle — Transfer Learning MobileNetV2.

Responsable : Marwan
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from src.config import (
    BASE_MODEL,
    DROPOUT_RATE,
    FINE_TUNE_AT,
    IMG_SIZE,
    LEARNING_RATE,
    NUM_CLASSES,
)


def build_model(num_classes: int = NUM_CLASSES, trainable_base: bool = False) -> tf.keras.Model:
    """Construit le classifieur Pokémon avec MobileNetV2 pré-entraîné ImageNet.

    Architecture :
        MobileNetV2 (base gelée) → GlobalAveragePooling2D → Dropout → Dense(151, softmax)

    Justification :
        - Dataset ~7000 images pour 151 classes ≈ 46 images/classe → trop petit pour CNN from scratch
        - MobileNetV2 léger, entraînement < 20 min sur Colab T4
        - Les couches basses détectent déjà bords/textures/formes génériques (ImageNet)
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = trainable_base

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="pokemon_classifier_mobilenetv2")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """Compile le modèle avec Adam et sparse categorical crossentropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_base_model(model: tf.keras.Model, fine_tune_at: int = FINE_TUNE_AT) -> tf.keras.Model:
    """Dégèle les couches hautes de MobileNetV2 pour le fine-tuning (phase 2).

    Stratégie en 2 phases :
        Phase 1 : base gelée, entraîner uniquement la tête de classification
        Phase 2 : dégeler à partir de la couche `fine_tune_at`, LR réduit (1e-5)
    """
    base_model = model.layers[1]  # MobileNetV2
    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    frozen = sum(1 for l in base_model.layers if not l.trainable)
    total = len(base_model.layers)
    print(f"Fine-tuning : {frozen}/{total} couches gelées, {total - frozen} dégelées")
    return model


def get_callbacks(logs_dir: str = "logs") -> list:
    """Callbacks : early stopping, reduce LR, TensorBoard, checkpoint."""
    import os

    os.makedirs(logs_dir, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=logs_dir),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/checkpoint.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]
