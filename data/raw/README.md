# Raw Dataset Folder

Place the original MRI dataset in this directory before running preprocessing or training.

Expected structure:

```text
data/raw/
├── class_1/
│   ├── image_001.png
│   └── ...
├── class_2/
│   ├── image_001.png
│   └── ...
└── ...
```

Notes:
- The exact class names and number of classes will depend on the real dataset.
- Do not add any patient, clinical, or sensitive information to the project repository.
- Keep the actual dataset out of Git by using the repository's `.gitignore` rules.
- Update the dataset configuration in `src/config.py` once the real dataset structure is available.
