"""Training pipeline for the MRI image classification model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, METRICS_DIR, MODEL_PATH, TRAINING_HISTORY_PATH
from src.data.augmentation import build_augmentation_pipeline
from src.models.model import build_model


def train_model(model: Any, train_dataset: Any, validation_dataset: Any, epochs: int = 20, callbacks: list | None = None) -> Any:
    """Train a Keras model on the training dataset and return its history."""
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks or [],
    )
    return history


def save_training_history(history: Any, output_path: str | Path = TRAINING_HISTORY_PATH) -> None:
    """Save training history to CSV."""
    history_frame = pd.DataFrame(history.history)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history_frame.to_csv(path, index=False)


def save_model(model: Any, output_path: str | Path = MODEL_PATH) -> None:
    """Save the trained Keras model to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def plot_training_history(history: Any, output_dir: str | Path = FIGURES_DIR) -> None:
    """Generate accuracy and loss graphs from training history."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(history.history.get("accuracy", []), label="Training Accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "training_accuracy.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history.get("loss", []), label="Training Loss")
    plt.plot(history.history.get("val_loss", []), label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "training_loss.png")
    plt.close()


def build_training_pipeline(input_shape: tuple[int, int, int], num_classes: int) -> Any:
    """Create a model and augmentation pipeline for the training stage."""
    model = build_model(input_shape=input_shape, num_classes=num_classes)
    augmentation_model = build_augmentation_pipeline()
    return model, augmentation_model
