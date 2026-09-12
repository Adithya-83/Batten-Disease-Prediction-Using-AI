from pathlib import Path
import cv2
import shutil

# ============================================================
# PATHS
# ============================================================

BATTTEN_DIR = Path(
    r"C:\Users\Adithya\OneDrive\Desktop\Batten_Disease_Prediction_Using_AI\data\mri_dataset\batten"
)

CLEANED_DIR = BATTTEN_DIR.parent / "batten_cleaned"

CLEANED_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

images = sorted([
    p for p in BATTTEN_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
])

print("=" * 60)
print("BATTEN MRI IMAGE CLEANING TOOL")
print("=" * 60)
print(f"Images found: {len(images)}")
print()
print("Instructions:")
print("1. Select ONLY the MRI region with your mouse.")
print("2. Press ENTER or SPACE to confirm the crop.")
print("3. Press C to cancel the current crop.")
print("4. Press ESC to skip an image.")
print()
print("For multiple MRI panels, crop the most useful MRI panel.")
print("=" * 60)

cleaned_count = 0
skipped_count = 0

for index, image_path in enumerate(images, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path.name}")
        skipped_count += 1
        continue

    display_image = image.copy()

    # Resize large screenshots for easier viewing
    max_width = 1200
    max_height = 800

    height, width = display_image.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
        1.0
    )

    if scale < 1:
        display_image = cv2.resize(
            display_image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    window_name = (
        f"Image {index}/{len(images)} - "
        f"{image_path.name}"
    )

    print(f"[{index}/{len(images)}] Processing {image_path.name}")

    # Select MRI region
    roi = cv2.selectROI(
        window_name,
        display_image,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyAllWindows()

    x, y, w, h = roi

    # ESC / no selection
    if w == 0 or h == 0:
        print("  -> Skipped")
        skipped_count += 1
        continue

    # Convert ROI coordinates back to original image size
    if scale != 1.0:
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

    cropped = image[y:y+h, x:x+w]

    output_name = f"batten_{cleaned_count + 1:02d}.png"
    output_path = CLEANED_DIR / output_name

    cv2.imwrite(str(output_path), cropped)

    cleaned_count += 1

    print(f"  -> Saved: {output_name}")

print()
print("=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)
print(f"Cleaned images: {cleaned_count}")
print(f"Skipped images: {skipped_count}")
print(f"Cleaned folder: {CLEANED_DIR}")
print()
print("IMPORTANT:")
print("Review the cleaned images before replacing the originals.")
print("=" * 60)