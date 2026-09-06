"""Data augmentation utilities for training data only."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
from tensorflow import keras


def build_augmentation_pipeline() -> keras.Sequential:
    """Create a basic data augmentation model for training images.

    The augmentation should only be applied to the training set, never to validation
    or test data. This module is intentionally simple and can be extended later.
    """
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomZoom(0.05),
        ],
        name="augmentation_layer",
    )


def augment_batch(images: np.ndarray, augmentation_model: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply augmentation to a batch of images."""
    return augmentation_model(images)
