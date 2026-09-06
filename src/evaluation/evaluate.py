"""Evaluation utilities for MRI classification results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from src.config import FIGURES_DIR, METRICS_DIR


def evaluate_model(model: Any, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    """Evaluate a model on a test set and return metrics.

    This function is a reusable template for later dataset-specific evaluation.
    """
    y_pred = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(y_pred, axis=1)

    accuracy = metrics.accuracy_score(y_test, predicted_labels)
    precision = metrics.precision_score(y_test, predicted_labels, average="weighted", zero_division=0)
    recall = metrics.recall_score(y_test, predicted_labels, average="weighted", zero_division=0)
    f1 = metrics.f1_score(y_test, predicted_labels, average="weighted", zero_division=0)
    confusion = metrics.confusion_matrix(y_test, predicted_labels)
    report = metrics.classification_report(y_test, predicted_labels, output_dict=True, zero_division=0)

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": confusion,
        "classification_report": report,
    }
    return results


def save_metrics(results: dict[str, Any], output_dir: str | Path = METRICS_DIR) -> None:
    """Save evaluation metrics to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1_score": results["f1_score"],
    }
    np.save(output_path / "metrics.npy", summary)

    with open(output_path / "classification_report.txt", "w", encoding="utf-8") as file:
        for key, value in results["classification_report"].items():
            if isinstance(value, dict):
                file.write(f"Class: {key}\n")
                for metric_name, metric_value in value.items():
                    file.write(f"  {metric_name}: {metric_value}\n")
                file.write("\n")


def save_confusion_matrix(cm: np.ndarray, output_dir: str | Path = FIGURES_DIR) -> None:
    """Save a confusion matrix figure."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png")
    plt.close()
