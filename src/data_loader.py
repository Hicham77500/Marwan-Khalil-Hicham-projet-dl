"""Chargement et préparation du dataset Pokémon.

Responsable : Hicham (lead)
"""

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    IMG_SIZE,
    NUM_CLASSES,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TEST_SPLIT,
    VALIDATION_SPLIT,
)


def get_data_paths(data_dir: Path | None = None) -> tuple[list[str], list[str]]:
    """Parcourt le dossier d'images et retourne (chemins, labels).

    Structure attendue :
        data/raw/bulbasaur/img001.png
        data/raw/ivysaur/img002.jpg
        ...
    """
    data_dir = data_dir or RAW_DATA_DIR
    image_paths: list[str] = []
    labels: list[str] = []

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {data_dir}\n"
            "Téléchargez le dataset Kaggle et extrayez-le dans data/raw/\n"
            "Voir README.md section 'Installation du dataset'."
        )

    valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name.lower().strip()
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in valid_extensions:
                image_paths.append(str(img_file))
                labels.append(class_name)

    if len(image_paths) == 0:
        raise ValueError(f"Aucune image trouvée dans {data_dir}")

    print(f"Dataset chargé : {len(image_paths)} images, {len(set(labels))} classes")
    return image_paths, labels


def create_class_mapping(labels: list[str]) -> tuple[dict[str, int], list[str]]:
    """Crée le mapping nom → index et la liste ordonnée des classes."""
    unique_classes = sorted(set(labels))
    if len(unique_classes) > NUM_CLASSES:
        print(f"Attention : {len(unique_classes)} classes trouvées, on garde les {NUM_CLASSES} premières")
        unique_classes = unique_classes[:NUM_CLASSES]

    class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}
    idx_to_class = unique_classes
    return class_to_idx, idx_to_class


def load_and_preprocess_image(path: str, label: str, class_to_idx: dict[str, int]) -> tuple[tf.Tensor, tf.Tensor]:
    """Charge et prétraite une image pour MobileNetV2."""
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    # Préprocessing MobileNetV2 : scale [-1, 1]
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    label_idx = class_to_idx.get(label.lower(), 0)
    return img, label_idx


def split_dataset(
    image_paths: list[str],
    labels: list[str],
    val_split: float = VALIDATION_SPLIT,
    test_split: float = TEST_SPLIT,
) -> tuple[list, list, list, list, list, list]:
    """Split train / val / test stratifié."""
    # D'abord séparer le test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_split, stratify=labels, random_state=RANDOM_SEED
    )
    # Puis séparer train et val
    relative_val = val_split / (1 - test_split)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=relative_val, stratify=train_val_labels, random_state=RANDOM_SEED
    )

    print(f"Split → Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


def make_tf_dataset(
    image_paths: list[str],
    labels: list[str],
    class_to_idx: dict[str, int],
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    augment: bool = False,
) -> tf.data.Dataset:
    """Crée un tf.data.Dataset à partir des chemins d'images."""
    label_indices = [class_to_idx[l.lower()] for l in labels]

    def _load(path, label_idx):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, label_idx

    ds = tf.data.Dataset.from_tensor_slices((image_paths, label_indices))
    ds = ds.map(lambda p, l: _load(p, l), num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(
            lambda img, lbl: (_apply_augmentation(img), lbl),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(image_paths), 1000), seed=RANDOM_SEED)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _apply_augmentation(image: tf.Tensor) -> tf.Tensor:
    """Data augmentation légère pour la Vision."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image


def inspect_dataset(data_dir: Path | None = None) -> dict:
    """Inspecte le dataset : shapes, distribution des classes, exemples."""
    paths, labels = get_data_paths(data_dir)
    class_to_idx, idx_to_class = create_class_mapping(labels)

    from collections import Counter

    distribution = Counter(labels)

    info = {
        "total_images": len(paths),
        "num_classes": len(idx_to_class),
        "min_per_class": min(distribution.values()),
        "max_per_class": max(distribution.values()),
        "avg_per_class": len(paths) / len(idx_to_class),
        "classes": idx_to_class,
        "distribution": dict(distribution.most_common(10)),
    }

    print("=" * 50)
    print("INSPECTION DATASET")
    print("=" * 50)
    for key, val in info.items():
        if key != "classes":
            print(f"  {key}: {val}")
    print(f"  classes (5 premières): {idx_to_class[:5]}...")
    print("=" * 50)

    return info


def save_class_names(class_names: list[str], output_path: Path | None = None) -> None:
    """Sauvegarde la liste des noms de classes pour la WebApp."""
    from src.config import CLASS_NAMES_PATH

    output_path = output_path or CLASS_NAMES_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(class_names))
    print(f"Noms de classes sauvegardés : {output_path}")
