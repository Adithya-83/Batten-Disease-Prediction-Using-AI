# Architecture Overview

The project follows a standard MRI image classification workflow:

MRI Image
→ Preprocessing
→ AI / Deep Learning Model
→ Prediction
→ Evaluation

## Pipeline description

1. MRI images are loaded from the raw dataset folder.
2. The images are checked for format, readability, and dimensions.
3. Preprocessing includes resizing, normalization, and optional augmentation on the training set.
4. The model is selected and instantiated in a modular manner.
5. Training and validation run on the processed data.
6. Prediction generates a class label and confidence if the model supports it.
7. Evaluation produces quantitative metrics and performance plots.

This project is intentionally structured to support later changes in the model architecture once the real dataset is inspected.
