"""Train/validation/test splitting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split


def split_dataset(file_paths: List[str | Path], test_size: float = 0.2, validation_size: float = 0.2, random_state: int = 42) -> Tuple[List[str | Path], List[str | Path], List[str | Path]]:
    """Split a list of file paths into train, validation, and test sets.

    The validation split is created from the remaining data after taking the test set.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    train_files, test_files = train_test_split(file_paths, test_size=test_size, random_state=random_state)
    train_files, val_files = train_test_split(train_files, test_size=validation_size, random_state=random_state)
    return train_files, val_files, test_files
