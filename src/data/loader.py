"""Utilities for loading MRI images and dataset metadata."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import EXPECTED_IMAGE_EXTENSIONS, RAW_DATA_DIR


def list_image_files(directory: str | Path | None = None) -> List[Path]:
    """Return all image files under a directory with supported extensions."""
    base_dir = Path(directory) if directory is not None else RAW_DATA_DIR
    if not base_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {base_dir}")

    image_files: List[Path] = []
    for candidate in sorted(base_dir.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in EXPECTED_IMAGE_EXTENSIONS:
            image_files.append(candidate)
    return image_files


def load_image(image_path: str | Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """Load a single image as a NumPy array.

    Args:
        image_path: Path to an image file.
        target_size: Optional resize target in the form (height, width).

    Returns:
        A NumPy array representing the image in RGB format.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        image = Image.open(path)
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Image is unreadable or corrupted: {path}") from exc

    array = np.asarray(image)
    if target_size is not None:
        array = cv2.resize(array, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    return array


def check_dataset_structure(data_dir: str | Path | None = None) -> List[str]:
    """Check and return class directories present in the dataset.

    This function is intentionally generic and can be reused when the dataset
    structure is finalized by the user.
    """
    base_dir = Path(data_dir) if data_dir is not None else RAW_DATA_DIR
    if not base_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {base_dir}")

    class_dirs = [
        item.name
        for item in sorted(base_dir.iterdir())
        if item.is_dir() and not item.name.startswith(".")
    ]
    return class_dirs


def check_image_dimensions(image_paths: List[str | Path], expected_size: tuple[int, int]) -> List[Tuple[str, tuple[int, int]]]:
    """Return mismatched image sizes for a list of images."""
    mismatches: List[Tuple[str, tuple[int, int]]] = []
    for image_path in image_paths:
        array = load_image(image_path)
        shape = array.shape[:2]
        if shape != expected_size:
            mismatches.append((str(image_path), shape))
    return mismatches


def check_invalid_images(image_paths: List[str | Path]) -> List[str]:
    """Return a list of corrupted or unreadable image paths."""
    invalid: List[str] = []
    for image_path in image_paths:
        try:
            load_image(image_path)
        except (FileNotFoundError, ValueError):
            invalid.append(str(image_path))
    return invalid
