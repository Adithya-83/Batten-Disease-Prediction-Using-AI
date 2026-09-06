"""Tests for the basic model architecture."""

from src.models.model import build_model


def test_model_builds_successfully():
    """The basic CNN architecture should build without errors."""
    model = build_model(input_shape=(32, 32, 3), num_classes=2)
    assert model is not None
    assert model.input_shape == (None, 32, 32, 3)
