"""Reusable preprocessing utilities for MRI images."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from src.config import IMAGE_SIZE
from src.data.loader import load_image


def resize_image(image: np.ndarray, target_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Resize an image to a target size using OpenCV."""
    if image.ndim != 3:
        raise ValueError("Expected a 3-channel image array.")
    height, width = target_size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image values to the [0, 1] range."""
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def preprocess_image(image_path: str | Path, target_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Apply the standard preprocessing pipeline to a single MRI image."""
    image = load_image(image_path, target_size=target_size)
    image = resize_image(image, target_size)
    image = normalize_image(image)
    return image


def preprocess_dataset(image_paths: List[str | Path], target_size: tuple[int, int] = IMAGE_SIZE) -> List[np.ndarray]:
    """Preprocess a collection of images and return processed arrays."""
    processed: List[np.ndarray] = []
    for image_path in image_paths:
        processed.append(preprocess_image(image_path, target_size))
    return processed


def validate_image_shape(image: np.ndarray, expected_shape: Tuple[int, int, int]) -> bool:
    """Check whether an image has the expected shape."""
    return image.shape == expected_shape
