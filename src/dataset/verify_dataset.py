from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "potholes"

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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_files(folder):
    """
    Find all supported image files recursively.
    """

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


def get_label_files(folder):
    """
    Find all YOLO .txt annotation files recursively.
    """

    if not folder.exists():
        return []

    return sorted(
        [
            file
            for file in folder.rglob("*.txt")
            if file.is_file()
        ]
    )


def get_expected_label_path(image_path, image_folder, label_folder):
    """
    Convert an image path into its corresponding YOLO label path.
    """

    relative_path = image_path.relative_to(image_folder)

    return label_folder / relative_path.with_suffix(".txt")


def validate_yolo_label(label_path, num_classes=1):
    """
    Validate a YOLO annotation file.

    Expected format:

    class_id center_x center_y width height
    """

    errors = []
    annotations = []

    try:
        with open(label_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

    except Exception as error:
        return [], [f"Could not read file: {error}"]

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        # Ignore empty lines
        if not line:
            continue

        parts = line.split()

        # YOLO format requires exactly 5 values
        if len(parts) != 5:
            errors.append(
                f"Line {line_number}: expected 5 values, "
                f"found {len(parts)}"
            )
            continue

        try:
            class_id = int(float(parts[0]))

            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:
            errors.append(
                f"Line {line_number}: contains non-numeric values"
            )
            continue

        # Validate class ID
        if class_id < 0 or class_id >= num_classes:
            errors.append(
                f"Line {line_number}: invalid class ID {class_id}"
            )

        # Validate normalized coordinates
        values = [
            ("center_x", center_x),
            ("center_y", center_y),
            ("width", width),
            ("height", height),
        ]

        for name, value in values:

            if not np.isfinite(value):
                errors.append(
                    f"Line {line_number}: {name} is not finite"
                )

            elif value < 0 or value > 1:
                errors.append(
                    f"Line {line_number}: {name}={value} "
                    f"is outside [0, 1]"
                )

        annotations.append(
            {
                "class_id": class_id,
                "center_x": center_x,
                "center_y": center_y,
                "width": width,
                "height": height,
            }
        )

    return annotations, errors


def analyze_split(
    split_name,
    image_folder,
    label_folder,
    num_classes=1,
):
    """
    Analyze one dataset split.
    """

    print()
    print("=" * 70)
    print(f"{split_name.upper()} DATASET")
    print("=" * 70)

    images = get_image_files(image_folder)
    labels = get_label_files(label_folder)

    print(f"Images found : {len(images)}")
    print(f"Labels found : {len(labels)}")

    image_stems = {
        image.relative_to(image_folder).with_suffix("")
        for image in images
    }

    label_stems = {
        label.relative_to(label_folder).with_suffix("")
        for label in labels
    }

    matched = image_stems & label_stems
    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    print(f"Matched pairs: {len(matched)}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Orphan labels : {len(orphan_labels)}")

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    invalid_annotations = []

    class_counts = {}

    total_boxes = 0

    image_records = []

    for image_path in images:

        relative_image = image_path.relative_to(image_folder)

        label_path = label_folder / relative_image.with_suffix(".txt")

        # Image information
        image = cv2.imread(str(image_path))

        if image is None:

            image_records.append(
                {
                    "split": split_name,
                    "image": str(relative_image),
                    "width": None,
                    "height": None,
                    "channels": None,
                    "boxes": 0,
                }
            )

            continue

        height, width = image.shape[:2]

        channels = 1 if image.ndim == 2 else image.shape[2]

        annotations = []
        errors = []

        if label_path.exists():

            annotations, errors = validate_yolo_label(
                label_path,
                num_classes=num_classes,
            )

            if errors:

                for error in errors:

                    invalid_annotations.append(
                        {
                            "image": str(relative_image),
                            "label": str(
                                label_path.relative_to(label_folder)
                            ),
                            "error": error,
                        }
                    )

        boxes = len(annotations)

        total_boxes += boxes

        for annotation in annotations:

            class_id = annotation["class_id"]

            class_counts[class_id] = (
                class_counts.get(class_id, 0) + 1
            )

        image_records.append(
            {
                "split": split_name,
                "image": str(relative_image),
                "width": width,
                "height": height,
                "channels": channels,
                "boxes": boxes,
            }
        )

    # --------------------------------------------------------
    # Print class distribution
    # --------------------------------------------------------

    print()
    print("Class distribution:")

    if class_counts:

        for class_id, count in sorted(class_counts.items()):

            print(
                f"  Class {class_id}: "
                f"{count} bounding boxes"
            )

    else:

        print("  No valid annotations found.")

    print()
    print(f"Total bounding boxes: {total_boxes}")
    print(f"Invalid annotations : {len(invalid_annotations)}")

    # --------------------------------------------------------
    # Missing files
    # --------------------------------------------------------

    if missing_labels:

        print()
        print("First missing labels:")

        for item in sorted(missing_labels)[:10]:

            print(f"  {item}")

    if orphan_labels:

        print()
        print("First orphan labels:")

        for item in sorted(orphan_labels)[:10]:

            print(f"  {item}")

    return {
        "images": images,
        "labels": labels,
        "matched": matched,
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "invalid_annotations": invalid_annotations,
        "class_counts": class_counts,
        "total_boxes": total_boxes,
        "image_records": image_records,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ROAD DAMAGE AI - DATASET VERIFICATION")
    print("=" * 70)

    print()
    print(f"Dataset path:")
    print(f"  {DATASET_DIR}")

    print()
    print("Checking dataset directories...")

    directories = [
        DATASET_DIR,
        TRAIN_IMAGES,
        TRAIN_LABELS,
        VALID_IMAGES,
        VALID_LABELS,
    ]

    missing_directories = []

    for directory in directories:

        if directory.exists():

            print(f"  [OK] {directory}")

        else:

            print(f"  [MISSING] {directory}")
            missing_directories.append(directory)

    if missing_directories:

        print()
        print("ERROR: Required dataset directories are missing.")

        return

    # --------------------------------------------------------
    # Analyze train
    # --------------------------------------------------------

    train_result = analyze_split(
        "train",
        TRAIN_IMAGES,
        TRAIN_LABELS,
        num_classes=1,
    )

    # --------------------------------------------------------
    # Analyze validation
    # --------------------------------------------------------

    valid_result = analyze_split(
        "valid",
        VALID_IMAGES,
        VALID_LABELS,
        num_classes=1,
    )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    all_records = (
        train_result["image_records"]
        + valid_result["image_records"]
    )

    df = pd.DataFrame(all_records)

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    total_images = len(df)

    total_boxes = (
        train_result["total_boxes"]
        + valid_result["total_boxes"]
    )

    total_missing_labels = (
        len(train_result["missing_labels"])
        + len(valid_result["missing_labels"])
    )

    total_orphan_labels = (
        len(train_result["orphan_labels"])
        + len(valid_result["orphan_labels"])
    )

    total_invalid_annotations = (
        len(train_result["invalid_annotations"])
        + len(valid_result["invalid_annotations"])
    )

    print()
    print("=" * 70)
    print("FINAL DATASET REPORT")
    print("=" * 70)

    print()
    print(f"Training images       : {len(train_result['images'])}")
    print(f"Training labels       : {len(train_result['labels'])}")

    print()

    print(f"Validation images     : {len(valid_result['images'])}")
    print(f"Validation labels     : {len(valid_result['labels'])}")

    print()

    print(f"Total images          : {total_images}")
    print(f"Total bounding boxes  : {total_boxes}")

    print()

    print(f"Missing labels        : {total_missing_labels}")
    print(f"Orphan labels         : {total_orphan_labels}")
    print(f"Invalid annotations   : {total_invalid_annotations}")

    # ========================================================
    # IMAGE SIZE ANALYSIS
    # ========================================================

    if not df.empty:

        valid_sizes = df.dropna(
            subset=["width", "height"]
        )

        if not valid_sizes.empty:

            print()
            print("=" * 70)
            print("IMAGE SIZE ANALYSIS")
            print("=" * 70)

            print(
                f"Minimum width  : "
                f"{int(valid_sizes['width'].min())}"
            )

            print(
                f"Maximum width  : "
                f"{int(valid_sizes['width'].max())}"
            )

            print(
                f"Minimum height : "
                f"{int(valid_sizes['height'].min())}"
            )

            print(
                f"Maximum height : "
                f"{int(valid_sizes['height'].max())}"
            )

            print(
                f"Average width  : "
                f"{valid_sizes['width'].mean():.2f}"
            )

            print(
                f"Average height : "
                f"{valid_sizes['height'].mean():.2f}"
            )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    reports_folder = PROJECT_ROOT / "outputs" / "reports"

    reports_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = reports_folder / "dataset_images.csv"

    df.to_csv(
        csv_path,
        index=False,
    )

    print()
    print(f"Image analysis saved to:")
    print(f"  {csv_path}")

    # ========================================================
    # DATASET STATUS
    # ========================================================

    print()
    print("=" * 70)

    if (
        total_images > 0
        and total_boxes > 0
        and total_missing_labels == 0
        and total_orphan_labels == 0
        and total_invalid_annotations == 0
    ):

        print("DATASET STATUS: READY")

    else:

        print("DATASET STATUS: NEEDS ATTENTION")

    print("=" * 70)
    print()


if __name__ == "__main__":
    main()