import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "batten_mri_efficientnet.keras",
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "data",
    "mri_training",
)

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
SEED = 42

# ============================================================
# COLLECT ALL IMAGE PATHS
# ============================================================

class_names = [
    "batten",
    "non_batten",
]

class_to_label = {
    "batten": 0,
    "non_batten": 1,
}

image_paths = []
labels = []

for class_name in class_names:

    class_dir = os.path.join(
        DATASET_DIR,
        class_name,
    )

    if not os.path.isdir(class_dir):
        raise FileNotFoundError(
            f"Dataset folder not found: {class_dir}"
        )

    for filename in os.listdir(class_dir):

        file_path = os.path.join(
            class_dir,
            filename,
        )

        if os.path.isfile(file_path):
            image_paths.append(file_path)
            labels.append(class_to_label[class_name])

image_paths = np.array(image_paths)
labels = np.array(labels)

print("\n" + "=" * 60)
print("MRI DATASET")
print("=" * 60)

print(f"\nTotal images: {len(image_paths)}")

print(
    f"Batten images: {np.sum(labels == 0)}"
)

print(
    f"Non-Batten images: {np.sum(labels == 1)}"
)

# ============================================================
# STRATIFIED VALIDATION SPLIT
# ============================================================

(
    train_paths,
    validation_paths,
    train_labels,
    validation_labels,
) = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=SEED,
    stratify=labels,
)

print("\nValidation set:")
print(
    f"Batten: {np.sum(validation_labels == 0)}"
)

print(
    f"Non-Batten: {np.sum(validation_labels == 1)}"
)

# ============================================================
# LOAD VALIDATION IMAGES
# ============================================================

def load_image(path):

    image = tf.keras.utils.load_img(
        path,
        target_size=IMAGE_SIZE,
        color_mode="rgb",
    )

    image = tf.keras.utils.img_to_array(image)

    return image


validation_images = np.array(
    [
        load_image(path)
        for path in validation_paths
    ],
    dtype=np.float32,
)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained MRI model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

# ============================================================
# MODEL PREDICTION
# ============================================================

print("\nRunning predictions...")

raw_predictions = model.predict(
    validation_images,
    verbose=0,
).reshape(-1)

# ============================================================
# MODEL LABEL MAPPING
# ============================================================
#
# Training folders:
#
# batten     = 0
# non_batten = 1
#
# Sigmoid output = P(non_batten)
#
# Therefore:
#
# P(batten) = 1 - P(non_batten)
#
# ============================================================

non_batten_probability = raw_predictions

batten_probability = (
    1.0 - non_batten_probability
)

# ============================================================
# CONVERT TO BINARY LABELS
# ============================================================

true_batten = (
    validation_labels == 0
).astype(int)

predicted_batten = (
    batten_probability >= 0.50
).astype(int)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    true_batten,
    predicted_batten,
)

precision = precision_score(
    true_batten,
    predicted_batten,
    zero_division=0,
)

recall = recall_score(
    true_batten,
    predicted_batten,
    zero_division=0,
)

f1 = f1_score(
    true_batten,
    predicted_batten,
    zero_division=0,
)

cm = confusion_matrix(
    true_batten,
    predicted_batten,
    labels=[0, 1],
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL MRI MODEL EVALUATION")
print("=" * 60)

print(
    f"\nValidation images: {len(validation_labels)}"
)

print("\nMetrics:")

print(
    f"Accuracy :  {accuracy:.4f}  ({accuracy:.2%})"
)

print(
    f"Precision:  {precision:.4f}  ({precision:.2%})"
)

print(
    f"Recall   :  {recall:.4f}  ({recall:.2%})"
)

print(
    f"F1-score :  {f1:.4f}  ({f1:.2%})"
)

print("\nConfusion Matrix")
print("-----------------")

print(cm)

print("\nMatrix interpretation:")
print(
    "Rows = Actual [Batten, Non-Batten]"
)

print(
    "Columns = Predicted [Batten, Non-Batten]"
)

print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        true_batten,
        predicted_batten,
        labels=[0, 1],
        target_names=[
            "Batten",
            "Non-Batten",
        ],
        zero_division=0,
    )
)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)