"""Prediction utilities for a trained MRI image classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config import IMAGE_SIZE, MODEL_PATH
from src.data.preprocessing import preprocess_image


def load_trained_model(model_path: str | Path = MODEL_PATH) -> Any:
    """Load a saved Keras model from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Trained model file not found: {path}")
    from tensorflow import keras

    return keras.models.load_model(path)


def predict_single_image(model: Any, image_path: str | Path, target_size: tuple[int, int] = IMAGE_SIZE) -> tuple[str, float | None]:
    """Make a prediction for a single MRI image.

    Returns a label name and optional confidence value if the selected model exposes it.
    """
    image = preprocess_image(image_path, target_size=target_size)
    image = np.expand_dims(image, axis=0)

    probabilities = model.predict(image, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = f"class_{predicted_index}"
    confidence = float(np.max(probabilities)) if probabilities.size > 0 else None
    return predicted_class, confidence


def prediction_message(predicted_class: str, confidence: float | None = None) -> str:
    """Construct a user-friendly message about the prediction outcome.

    This output is deliberately framed as an AI research prototype and is not a medical diagnosis.
    """
    base = "AI research prototype prediction: "
    if confidence is not None:
        return f"{base}{predicted_class} with confidence {confidence:.4f}. This is not a medical diagnosis."
    return f"{base}{predicted_class}. This is not a medical diagnosis."
