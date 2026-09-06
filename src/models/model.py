"""Model definition module for the MRI image classification project.

The model architecture is intentionally modular so it can be updated after the
actual dataset is inspected. This initial version prepares a CNN-based image-classification
baseline using Keras.
"""

from __future__ import annotations

from typing import Optional

from tensorflow import keras


def build_model(input_shape: tuple[int, int, int] = (224, 224, 3), num_classes: Optional[int] = None) -> keras.Model:
    """Build a simple CNN classifier for MRI image classification.

    Args:
        input_shape: Input image shape as (height, width, channels).
        num_classes: Number of output classes. If None, a placeholder value is used.

    Returns:
        A compiled Keras model.
    """
    if num_classes is None:
        num_classes = 2

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="batten_cnn",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
