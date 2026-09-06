"""Central configuration for the MRI classification project.

These values are intentionally kept in one place so the project can be adapted to
any dataset structure after the real MRI data is provided.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
REPORTS_DIR = RESULTS_DIR / "reports"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

MODEL_PATH = MODELS_DIR / "batten_model.keras"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.csv"

# Dataset config placeholders. Update after receiving the actual dataset.
DATASET_CLASSES = None
EXPECTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"]


def ensure_directories() -> None:
    """Create required project directories if they do not already exist."""
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SPLITS_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        REPORTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
