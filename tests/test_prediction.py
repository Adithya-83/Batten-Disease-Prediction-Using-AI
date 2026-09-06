"""Tests for prediction utilities."""

import numpy as np

from src.prediction.predict import prediction_message


def test_prediction_message_mentions_research_prototype():
    """Prediction output should clearly state it is not a diagnosis."""
    message = prediction_message("class_1", confidence=0.91)
    assert "AI research prototype prediction" in message
    assert "not a medical diagnosis" in message.lower()
    assert "0.9100" in message
