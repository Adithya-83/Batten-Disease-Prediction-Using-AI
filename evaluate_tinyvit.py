import os
import numpy as np
import torch
import timm

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
MODEL_PATH = os.path.join(BASE_DIR, "models", "batten_mri_tinyvit.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 8
SEED = 42

DEVICE = torch.device("cpu")

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
# RECREATE EXACT SAME SPLIT
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
# TRANSFORM
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

class MRIDataset(Dataset):

    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        image = Image.open(
            self.files[index]
        ).convert("RGB")

        image = transform(image)

        return image, torch.tensor(
            self.labels[index],
            dtype=torch.long
        )


test_dataset = MRIDataset(
    test_files,
    test_labels
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model = timm.create_model(
    "tiny_vit_5m_224",
    pretrained=False,
    num_classes=2
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

# ------------------------------------------------------------
# EVALUATE
# ------------------------------------------------------------

predictions = []
targets = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        preds = torch.argmax(
            outputs,
            dim=1
        )

        predictions.extend(
            preds.cpu().numpy()
        )

        targets.extend(
            labels.numpy()
        )

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

print("\n======================================")
print("TINYVIT FINAL TEST RESULTS")
print("======================================")

print(f"Accuracy        : {accuracy:.4f}")
print(f"Batten Precision: {precision:.4f}")
print(f"Batten Recall   : {recall:.4f}")
print(f"Batten F1       : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)