from pathlib import Path
import shutil

# CHANGE THIS to the actual location of your Normal folder
SOURCE_NORMAL = Path(r"C:\Users\Adithya\Downloads\archive (3)\Normal")

# Project destination
PROJECT_ROOT = Path(
    r"C:\Users\Adithya\OneDrive\Desktop\Batten_Disease_Prediction_Using_AI"
)

DESTINATION = PROJECT_ROOT / "data" / "mri_dataset" / "non_batten"

DESTINATION.mkdir(parents=True, exist_ok=True)

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

count = 0

for image_path in SOURCE_NORMAL.rglob("*"):

    if image_path.is_file() and image_path.suffix.lower() in image_extensions:

        destination_file = (
            DESTINATION / f"normal_{count:05d}{image_path.suffix.lower()}"
        )

        shutil.copy2(image_path, destination_file)

        count += 1

print(f"Copied {count} Normal MRI images.")
print(f"Destination: {DESTINATION}")