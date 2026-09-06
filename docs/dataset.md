# Dataset Information

This project expects the actual MRI dataset to be supplied separately by the user.

## Required information

Before training, the final dataset details should be recorded here, including:

- number of classes
- class names
- image dimensions
- image file format(s)
- train/validation/test split strategy
- any metadata associated with each image

## Expected directory structure

```text
data/raw/
├── class_1/
├── class_2/
└── ...
```

The code is written to allow easy reconfiguration after the real dataset is available.
