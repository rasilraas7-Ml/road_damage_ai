from pathlib import Path
import random

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "potholes"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "dataset_samples"
)

TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_LABELS = DATASET_DIR / "train" / "labels"

VALID_IMAGES = DATASET_DIR / "valid" / "images"
VALID_LABELS = DATASET_DIR / "valid" / "labels"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# Number of random images to visualize
NUM_RANDOM_SAMPLES = 10

# Reproducible random selection
RANDOM_SEED = 42


# ============================================================
# FIND IMAGES
# ============================================================

def get_image_files(folder):

    if not folder.exists():
        return []

    return sorted(
        [
            file
            for file in folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


# ============================================================
# READ YOLO LABELS
# ============================================================

def read_yolo_labels(label_path):

    annotations = []

    if not label_path.exists():
        return annotations

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            try:

                class_id = int(float(parts[0]))

                center_x = float(parts[1])
                center_y = float(parts[2])

                width = float(parts[3])
                height = float(parts[4])

                annotations.append(
                    {
                        "class_id": class_id,
                        "center_x": center_x,
                        "center_y": center_y,
                        "width": width,
                        "height": height,
                    }
                )

            except ValueError:

                continue

    return annotations


# ============================================================
# YOLO → PIXEL COORDINATES
# ============================================================

def yolo_to_pixel_coordinates(
    annotation,
    image_width,
    image_height,
):

    center_x = annotation["center_x"]
    center_y = annotation["center_y"]

    box_width = annotation["width"]
    box_height = annotation["height"]

    # YOLO normalized coordinates
    # Convert center coordinates to pixels

    center_x_pixel = center_x * image_width
    center_y_pixel = center_y * image_height

    width_pixel = box_width * image_width
    height_pixel = box_height * image_height

    # Convert center coordinates to corner coordinates

    x1 = int(
        center_x_pixel - (width_pixel / 2)
    )

    y1 = int(
        center_y_pixel - (height_pixel / 2)
    )

    x2 = int(
        center_x_pixel + (width_pixel / 2)
    )

    y2 = int(
        center_y_pixel + (height_pixel / 2)
    )

    # Make sure coordinates stay inside image

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))

    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    return x1, y1, x2, y2


# ============================================================
# DRAW ANNOTATIONS
# ============================================================

def draw_annotations(
    image,
    annotations,
):

    image_height, image_width = image.shape[:2]

    for index, annotation in enumerate(
        annotations,
        start=1,
    ):

        x1, y1, x2, y2 = (
            yolo_to_pixel_coordinates(
                annotation,
                image_width,
                image_height,
            )
        )

        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"POTHOLE #{index}"
        )

        # Text size
        (
            text_width,
            text_height,
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )

        # Make sure label stays inside image
        label_y = max(
            y1,
            text_height + baseline + 5
        )

        # Background rectangle
        cv2.rectangle(
            image,
            (
                x1,
                label_y
                - text_height
                - baseline
                - 5,
            ),
            (
                x1 + text_width + 8,
                label_y,
            ),
            (0, 255, 0),
            -1,
        )

        # Label text
        cv2.putText(
            image,
            label,
            (
                x1 + 4,
                label_y - 4,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return image


# ============================================================
# ADD IMAGE INFORMATION
# ============================================================

def add_image_information(
    image,
    filename,
    number_of_potholes,
):

    text_lines = [
        f"Image: {filename}",
        f"Potholes: {number_of_potholes}",
    ]

    y_position = 30

    for text in text_lines:

        cv2.putText(
            image,
            text,
            (
                15,
                y_position,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y_position += 30

    return image


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(
    image_path,
    image_folder,
    label_folder,
    output_folder,
):

    relative_path = image_path.relative_to(
        image_folder
    )

    label_path = (
        label_folder
        / relative_path.with_suffix(".txt")
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"[WARNING] Could not read: "
            f"{image_path}"
        )

        return False

    annotations = read_yolo_labels(
        label_path
    )

    image = draw_annotations(
        image,
        annotations,
    )

    image = add_image_information(
        image,
        image_path.name,
        len(annotations),
    )

    # Preserve relative folder structure

    output_path = (
        output_folder
        / relative_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if success:

        print(
            f"[SAVED] {output_path}"
        )

        return True

    print(
        f"[ERROR] Could not save: "
        f"{output_path}"
    )

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ROAD DAMAGE AI - ANNOTATION VISUALIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Output directory:")
    print(
        f"  {OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Random generator
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Get training images
    # --------------------------------------------------------

    train_images = get_image_files(
        TRAIN_IMAGES
    )

    valid_images = get_image_files(
        VALID_IMAGES
    )

    print()
    print(
        f"Training images found : "
        f"{len(train_images)}"
    )

    print(
        f"Validation images found : "
        f"{len(valid_images)}"
    )

    # --------------------------------------------------------
    # Select random samples
    # --------------------------------------------------------

    train_sample_count = min(
        NUM_RANDOM_SAMPLES // 2,
        len(train_images),
    )

    valid_sample_count = min(
        NUM_RANDOM_SAMPLES - train_sample_count,
        len(valid_images),
    )

    train_samples = random.sample(
        train_images,
        train_sample_count,
    )

    valid_samples = random.sample(
        valid_images,
        valid_sample_count,
    )

    samples = [
        (
            image,
            TRAIN_IMAGES,
            TRAIN_LABELS,
            "train",
        )
        for image in train_samples
    ]

    samples += [
        (
            image,
            VALID_IMAGES,
            VALID_LABELS,
            "valid",
        )
        for image in valid_samples
    ]

    # --------------------------------------------------------
    # Process samples
    # --------------------------------------------------------

    successful = 0

    print()
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    for (
        image_path,
        image_folder,
        label_folder,
        split_name,
    ) in samples:

        output_folder = (
            OUTPUT_DIR
            / split_name
        )

        success = process_image(
            image_path,
            image_folder,
            label_folder,
            output_folder,
        )

        if success:
            successful += 1

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print(
        f"Images processed : "
        f"{successful}/{len(samples)}"
    )

    print()
    print("Open this folder in VS Code/File Explorer:")
    print(
        f"  {OUTPUT_DIR}"
    )

    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()