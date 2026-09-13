import os
import numpy as np
import tensorflow as tf

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BATTEN_DIR = os.path.join(BASE_DIR, "data", "mri_training", "batten")
NORMAL_DIR = os.path.join(BASE_DIR, "data", "mri_training", "non_batten")
MODEL_PATH = os.path.join(BASE_DIR, "models", "batten_mri_efficientnet.keras")

IMAGE_SIZE = 224
SEED = 42

# ------------------------------------------------------------
# COLLECT SAME DATASET
# ------------------------------------------------------------

batten_files = [
    os.path.join(BATTEN_DIR, f)
    for f in os.listdir(BATTEN_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

normal_files = [
    os.path.join(NORMAL_DIR, f)
    for f in os.listdir(NORMAL_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

all_files = batten_files + normal_files
all_labels = [0] * len(batten_files) + [1] * len(normal_files)

# ------------------------------------------------------------
# RECREATE EXACT SAME SPLIT AS TINYVIT
# ------------------------------------------------------------

train_files, temp_files, train_labels, temp_labels = train_test_split(
    all_files,
    all_labels,
    test_size=0.30,
    stratify=all_labels,
    random_state=SEED
)

val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files,
    temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=SEED
)

print("Test images:", len(test_files))

# ------------------------------------------------------------
# LOAD EFFICIENTNET
# ------------------------------------------------------------

print("Loading EfficientNet...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

# ------------------------------------------------------------
# EVALUATE
#
# Current EfficientNet mapping:
# sigmoid output = P(non_batten)
# Batten probability = 1 - sigmoid output
# ------------------------------------------------------------

predictions = []
targets = []

for path, label in zip(test_files, test_labels):

    image = Image.open(path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    raw_output = float(
        model.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    batten_probability = 1.0 - raw_output

    predicted_class = (
        0 if batten_probability >= 0.50
        else 1
    )

    predictions.append(predicted_class)
    targets.append(label)

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

accuracy = accuracy_score(
    targets,
    predictions
)

precision = precision_score(
    targets,
    predictions,
    pos_label=0,
    zero_division=0
)

recall = recall_score(
    targets,
    predictions,
    pos_label=0,
    zero_division=0
)

f1 = f1_score(
    targets,
    predictions,
    pos_label=0,
    zero_division=0
)

cm = confusion_matrix(
    targets,
    predictions
)

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("\n======================================")
print("EFFICIENTNET FINAL TEST RESULTS")
print("======================================")

print(f"Accuracy        : {accuracy:.4f}")
print(f"Batten Precision: {precision:.4f}")
print(f"Batten Recall   : {recall:.4f}")
print(f"Batten F1       : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)