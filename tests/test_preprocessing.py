"""Tests for preprocessing utilities."""

from pathlib import Path

import numpy as np

from src.data.preprocessing import normalize_image, resize_image


def test_resize_image_changes_shape():
    """Resizing should return the expected shape."""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    resized = resize_image(image, target_size=(32, 32))
    assert resized.shape == (32, 32, 3)


def test_normalize_image_scales_values():
    """Normalization should scale values into the [0, 1] range."""
    image = np.array([[[255, 0], [128, 64]]], dtype=np.uint8)
    normalized = normalize_image(image)
    assert normalized.max() <= 1.0
    assert normalized.min() >= 0.0
