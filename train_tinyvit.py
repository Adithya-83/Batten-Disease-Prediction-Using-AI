import os
import random
import numpy as np
import torch
import timm

from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BATTEN_DIR = os.path.join(BASE_DIR, "data", "mri_training", "batten")
NORMAL_DIR = os.path.join(BASE_DIR, "data", "mri_training", "non_batten")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "batten_mri_tinyvit.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-4
SEED = 42

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# COLLECT IMAGE PATHS
# ============================================================

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

print("Batten images:", len(batten_files))
print("Normal images:", len(normal_files))


# ============================================================
# LABELS
# 0 = Batten
# 1 = Normal
# ============================================================

all_files = batten_files + normal_files
all_labels = [0] * len(batten_files) + [1] * len(normal_files)


# ============================================================
# FIXED STRATIFIED SPLIT
# 70% TRAIN / 15% VALIDATION / 15% TEST
# ============================================================

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

print("\nDataset split:")
print("Training:", len(train_files))
print("Validation:", len(val_files))
print("Testing:", len(test_files))


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

class MRIDataset(Dataset):

    def __init__(self, files, labels, transform):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        path = self.files[index]
        label = self.labels[index]

        image = Image.open(path).convert("RGB")

        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = MRIDataset(
    train_files,
    train_labels,
    train_transform
)

val_dataset = MRIDataset(
    val_files,
    val_labels,
    eval_transform
)

test_dataset = MRIDataset(
    test_files,
    test_labels,
    eval_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading pretrained TinyViT...")

model = timm.create_model(
    "tiny_vit_5m_224",
    pretrained=True,
    num_classes=2
)

model = model.to(DEVICE)


# ============================================================
# FREEZE MOST BACKBONE LAYERS
# ============================================================

for param in model.parameters():
    param.requires_grad = False

# Train classifier head
for param in model.head.parameters():
    param.requires_grad = True


# ============================================================
# CLASS-WEIGHTED LOSS
# ============================================================

train_labels_np = np.array(train_labels)

class_counts = np.bincount(train_labels_np)

class_weights = len(train_labels_np) / (
    len(class_counts) * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# TRAINING
# ============================================================

best_val_f1 = -1.0

print("\nStarting TinyViT training...\n")

for epoch in range(EPOCHS):

    model.train()

    train_predictions = []
    train_targets = []

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_predictions.extend(
            predictions.cpu().numpy()
        )

        train_targets.extend(
            labels.cpu().numpy()
        )

    train_accuracy = accuracy_score(
        train_targets,
        train_predictions
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_predictions = []
    val_targets = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_targets.extend(
                labels.numpy()
            )

    val_accuracy = accuracy_score(
        val_targets,
        val_predictions
    )

    val_recall = recall_score(
        val_targets,
        val_predictions,
        pos_label=0,
        zero_division=0
    )

    val_f1 = f1_score(
        val_targets,
        val_predictions,
        pos_label=0,
        zero_division=0
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"Batten Recall: {val_recall:.4f} | "
        f"Batten F1: {val_f1:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": "tiny_vit_5m_224",
                "image_size": IMAGE_SIZE,
                "class_mapping": {
                    0: "batten",
                    1: "non_batten"
                }
            },
            MODEL_PATH
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# FINAL TEST
# ============================================================

test_predictions = []
test_targets = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        test_predictions.extend(
            predictions.cpu().numpy()
        )

        test_targets.extend(
            labels.numpy()
        )


# ============================================================
# FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    test_targets,
    test_predictions
)

precision = precision_score(
    test_targets,
    test_predictions,
    pos_label=0,
    zero_division=0
)

recall = recall_score(
    test_targets,
    test_predictions,
    pos_label=0,
    zero_division=0
)

f1 = f1_score(
    test_targets,
    test_predictions,
    pos_label=0,
    zero_division=0
)

cm = confusion_matrix(
    test_targets,
    test_predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\n======================================")
print("TinyViT FINAL TEST RESULTS")
print("======================================")

print(f"Accuracy       : {accuracy:.4f}")
print(f"Batten Precision: {precision:.4f}")
print(f"Batten Recall   : {recall:.4f}")
print(f"Batten F1       : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nBest model saved to:")
print(MODEL_PATH)