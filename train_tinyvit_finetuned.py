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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BATTEN_DIR = os.path.join(
    BASE_DIR, "data", "mri_training", "batten"
)

NORMAL_DIR = os.path.join(
    BASE_DIR, "data", "mri_training", "non_batten"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "batten_mri_tinyvit_finetuned.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 8

HEAD_EPOCHS = 10
FINETUNE_EPOCHS = 25

HEAD_LR = 1e-4
FINETUNE_LR = 1e-5

SEED = 42
PATIENCE = 7

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# COLLECT IMAGE FILES
# ============================================================

batten_files = [
    os.path.join(BATTEN_DIR, f)
    for f in os.listdir(BATTEN_DIR)
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg")
    )
]

normal_files = [
    os.path.join(NORMAL_DIR, f)
    for f in os.listdir(NORMAL_DIR)
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg")
    )
]

batten_files = sorted(batten_files)
normal_files = sorted(normal_files)

print("Batten images:", len(batten_files))
print("Normal images:", len(normal_files))


# ============================================================
# LABELS
#
# 0 = BATTEN
# 1 = NON-BATTEN
# ============================================================

all_files = batten_files + normal_files

all_labels = (
    [0] * len(batten_files)
    + [1] * len(normal_files)
)


# ============================================================
# FIXED STRATIFIED SPLIT
#
# 70% TRAIN
# 15% VALIDATION
# 15% TEST
# ============================================================

train_files, temp_files, train_labels, temp_labels = (
    train_test_split(
        all_files,
        all_labels,
        test_size=0.30,
        stratify=all_labels,
        random_state=SEED
    )
)

val_files, test_files, val_labels, test_labels = (
    train_test_split(
        temp_files,
        temp_labels,
        test_size=0.50,
        stratify=temp_labels,
        random_state=SEED
    )
)


print("\nDataset split:")
print("Training:", len(train_files))
print("Validation:", len(val_files))
print("Testing:", len(test_files))


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        8
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

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

    def __init__(
        self,
        files,
        labels,
        transform
    ):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        image_path = self.files[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return image, label


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
# CREATE PRETRAINED TINYVIT
# ============================================================

print("\nLoading pretrained TinyViT...")

model = timm.create_model(
    "tiny_vit_5m_224",
    pretrained=True,
    num_classes=2
)

model = model.to(DEVICE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_labels_np = np.array(
    train_labels
)

class_counts = np.bincount(
    train_labels_np,
    minlength=2
)

class_weights = (
    len(train_labels_np)
    / (2.0 * class_counts)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate_validation():

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for images, labels in val_loader:

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

    return accuracy, precision, recall, f1


# ============================================================
# PHASE 1
# CLASSIFIER HEAD TRAINING
# ============================================================

print("\n======================================")
print("PHASE 1 — CLASSIFIER HEAD")
print("======================================")


for param in model.parameters():

    param.requires_grad = False


for param in model.head.parameters():

    param.requires_grad = True


optimizer = optim.AdamW(
    model.head.parameters(),
    lr=HEAD_LR,
    weight_decay=1e-4
)


best_f1 = -1.0

best_state = None


for epoch in range(HEAD_EPOCHS):

    model.train()

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()


    accuracy, precision, recall, f1 = (
        evaluate_validation()
    )


    print(
        f"Head Epoch {epoch + 1:02d}/"
        f"{HEAD_EPOCHS} | "
        f"Val Acc: {accuracy:.4f} | "
        f"Batten Precision: {precision:.4f} | "
        f"Batten Recall: {recall:.4f} | "
        f"Batten F1: {f1:.4f}"
    )


    if f1 > best_f1:

        best_f1 = f1

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }


# ============================================================
# RESTORE BEST HEAD
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ============================================================
# SAVE PHASE 1 FALLBACK
#
# This guarantees that a valid model exists even if
# Phase 2 does not improve validation F1.
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "model_name":
            "tiny_vit_5m_224",

        "image_size":
            IMAGE_SIZE,

        "class_mapping": {
            0: "batten",
            1: "non_batten"
        }
    },
    MODEL_PATH
)

print("\nBest Phase 1 model saved as fallback.")


# ============================================================
# PHASE 2
# FINE-TUNE FINAL BACKBONE STAGES
# ============================================================

print("\n======================================")
print("PHASE 2 — TINYVIT FINE-TUNING")
print("======================================")


# Freeze everything first

for param in model.parameters():

    param.requires_grad = False


# ------------------------------------------------------------
# TinyViT backbone fine-tuning
# ------------------------------------------------------------

if hasattr(model, "layers"):

    layers_to_train = model.layers[-2:]

    for layer in layers_to_train:

        for param in layer.parameters():

            param.requires_grad = True


# ------------------------------------------------------------
# Always train classification head
# ------------------------------------------------------------

for param in model.head.parameters():

    param.requires_grad = True


trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]


print(
    "Trainable parameters:",
    sum(
        parameter.numel()
        for parameter in trainable_parameters
    )
)


optimizer = optim.AdamW(
    trainable_parameters,
    lr=FINETUNE_LR,
    weight_decay=1e-4
)


epochs_without_improvement = 0


# ============================================================
# FINE-TUNING LOOP
# ============================================================

for epoch in range(FINETUNE_EPOCHS):

    model.train()

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()


    accuracy, precision, recall, f1 = (
        evaluate_validation()
    )


    print(
        f"FineTune Epoch {epoch + 1:02d}/"
        f"{FINETUNE_EPOCHS} | "
        f"Val Acc: {accuracy:.4f} | "
        f"Batten Precision: {precision:.4f} | "
        f"Batten Recall: {recall:.4f} | "
        f"Batten F1: {f1:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if f1 > best_f1:

        best_f1 = f1

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "model_name":
                    "tiny_vit_5m_224",

                "image_size":
                    IMAGE_SIZE,

                "class_mapping": {
                    0: "batten",
                    1: "non_batten"
                }
            },
            MODEL_PATH
        )

        epochs_without_improvement = 0

        print(
            "  ✓ Best fine-tuned model saved"
        )

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


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

        preds = torch.argmax(
            outputs,
            dim=1
        )

        test_predictions.extend(
            preds.cpu().numpy()
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
# FINAL RESULTS
# ============================================================

print("\n======================================")
print("TINYVIT FINE-TUNED FINAL RESULTS")
print("======================================")

print(
    f"Accuracy        : {accuracy:.4f}"
)

print(
    f"Batten Precision: {precision:.4f}"
)

print(
    f"Batten Recall   : {recall:.4f}"
)

print(
    f"Batten F1       : {f1:.4f}"
)

print("\nConfusion Matrix:")

print(cm)

print("\nBest model saved to:")

print(MODEL_PATH)