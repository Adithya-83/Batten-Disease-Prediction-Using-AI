# Methodology

The planned development workflow is:

Data Collection
→ Data Exploration
→ Preprocessing
→ Data Augmentation
→ Dataset Splitting
→ Model Development
→ Training & Validation
→ Prediction
→ Evaluation

## Details

### Data Collection
The actual MRI dataset must be provided separately and placed in `data/raw/`.

### Data Exploration
Inspect image formats, sizes, class counts, and dataset health.

### Preprocessing
Resize images, normalize pixel values, and prepare them for model input.

### Data Augmentation
Apply augmentation only to the training set to increase variability without introducing data leakage.

### Dataset Splitting
Separate the dataset into train, validation, and test subsets.

### Model Development
Create a baseline CNN image classifier that can later be reconfigured depending on the final dataset.

### Training & Validation
Train the model, monitor training and validation metrics, and save the trained artifacts.

### Prediction
Use a trained model to predict a class for a new MRI image. The output is clearly framed as a research prototype, not a medical diagnosis.

### Evaluation
Calculate metrics such as accuracy, precision, recall, F1-score, and a confusion matrix.
