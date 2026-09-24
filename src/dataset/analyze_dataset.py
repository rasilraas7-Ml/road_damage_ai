from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "potholes"

SPLITS = {
    "train": {
        "images": DATASET_DIR / "train" / "images",
        "labels": DATASET_DIR / "train" / "labels",
    },
    "valid": {
        "images": DATASET_DIR / "valid" / "images",
        "labels": DATASET_DIR / "valid" / "labels",
    },
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# FIND IMAGES
# ============================================================

def get_images(folder):
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

def read_labels(label_path):

    annotations = []

    if not label_path.exists():
        return annotations

    with open(label_path, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

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

    return annotations


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ROAD DAMAGE AI - DATASET ANALYSIS")
    print("=" * 70)

    image_records = []
    box_records = []

    # --------------------------------------------------------
    # PROCESS TRAIN + VALID
    # --------------------------------------------------------

    for split_name, paths in SPLITS.items():

        images_folder = paths["images"]
        labels_folder = paths["labels"]

        images = get_images(images_folder)

        print()
        print(f"Processing {split_name}: {len(images)} images")

        for image_path in images:

            relative_path = image_path.relative_to(
                images_folder
            )

            label_path = labels_folder / relative_path.with_suffix(
                ".txt"
            )

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            image_height, image_width = image.shape[:2]

            annotations = read_labels(label_path)

            number_of_boxes = len(annotations)

            # ------------------------------------------------
            # IMAGE RECORD
            # ------------------------------------------------

            image_records.append(
                {
                    "split": split_name,
                    "image": str(relative_path),
                    "width": image_width,
                    "height": image_height,
                    "aspect_ratio": image_width / image_height,
                    "pothole_count": number_of_boxes,
                }
            )

            # ------------------------------------------------
            # BOUNDING BOX RECORDS
            # ------------------------------------------------

            for annotation in annotations:

                box_width_normalized = annotation["width"]
                box_height_normalized = annotation["height"]

                box_width_pixels = (
                    box_width_normalized * image_width
                )

                box_height_pixels = (
                    box_height_normalized * image_height
                )

                box_area_pixels = (
                    box_width_pixels * box_height_pixels
                )

                image_area_pixels = (
                    image_width * image_height
                )

                area_percentage = (
                    box_area_pixels
                    / image_area_pixels
                ) * 100

                box_records.append(
                    {
                        "split": split_name,
                        "image": str(relative_path),
                        "class_id": annotation["class_id"],
                        "center_x": annotation["center_x"],
                        "center_y": annotation["center_y"],
                        "width_normalized": box_width_normalized,
                        "height_normalized": box_height_normalized,
                        "width_pixels": box_width_pixels,
                        "height_pixels": box_height_pixels,
                        "area_pixels": box_area_pixels,
                        "area_percentage": area_percentage,
                    }
                )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    images_df = pd.DataFrame(image_records)

    boxes_df = pd.DataFrame(box_records)

    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print()
    print(f"Total images       : {len(images_df)}")
    print(f"Total potholes     : {len(boxes_df)}")

    # ========================================================
    # SPLIT ANALYSIS
    # ========================================================

    print()
    print("=" * 70)
    print("TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    split_counts = images_df["split"].value_counts()

    print(split_counts)

    # ========================================================
    # POTHOLES PER IMAGE
    # ========================================================

    print()
    print("=" * 70)
    print("POTHOLES PER IMAGE")
    print("=" * 70)

    pothole_statistics = images_df[
        "pothole_count"
    ].describe()

    print(pothole_statistics)

    print()

    print("Distribution:")

    pothole_distribution = (
        images_df["pothole_count"]
        .value_counts()
        .sort_index()
    )

    for count, number_of_images in pothole_distribution.items():

        print(
            f"  {count} pothole(s) : "
            f"{number_of_images} images"
        )

    # ========================================================
    # IMAGE RESOLUTIONS
    # ========================================================

    print()
    print("=" * 70)
    print("MOST COMMON IMAGE RESOLUTIONS")
    print("=" * 70)

    resolution_counts = (
        images_df
        .groupby(["width", "height"])
        .size()
        .sort_values(ascending=False)
    )

    print(
        resolution_counts.head(15)
    )

    # ========================================================
    # ASPECT RATIO
    # ========================================================

    print()
    print("=" * 70)
    print("ASPECT RATIO")
    print("=" * 70)

    print(
        images_df["aspect_ratio"].describe()
    )

    # ========================================================
    # BOUNDING BOX SIZE
    # ========================================================

    print()
    print("=" * 70)
    print("BOUNDING BOX WIDTH")
    print("=" * 70)

    print(
        boxes_df["width_pixels"].describe()
    )

    print()
    print("=" * 70)
    print("BOUNDING BOX HEIGHT")
    print("=" * 70)

    print(
        boxes_df["height_pixels"].describe()
    )

    # ========================================================
    # BOUNDING BOX AREA
    # ========================================================

    print()
    print("=" * 70)
    print("BOUNDING BOX AREA")
    print("=" * 70)

    print(
        boxes_df["area_pixels"].describe()
    )

    # ========================================================
    # IMAGE AREA PERCENTAGE
    # ========================================================

    print()
    print("=" * 70)
    print("POTHOLE AREA AS % OF IMAGE")
    print("=" * 70)

    print(
        boxes_df["area_percentage"].describe()
    )

    # ========================================================
    # SMALL POTHOLES
    # ========================================================

    small_potholes = boxes_df[
        boxes_df["area_percentage"] < 1
    ]

    medium_potholes = boxes_df[
        (boxes_df["area_percentage"] >= 1)
        & (boxes_df["area_percentage"] < 10)
    ]

    large_potholes = boxes_df[
        boxes_df["area_percentage"] >= 10
    ]

    print()
    print("=" * 70)
    print("POTHOLE SIZE CATEGORIES")
    print("=" * 70)

    print(
        f"Small   (<1% image)    : "
        f"{len(small_potholes)}"
    )

    print(
        f"Medium  (1-10%)        : "
        f"{len(medium_potholes)}"
    )

    print(
        f"Large   (>=10%)        : "
        f"{len(large_potholes)}"
    )

    # ========================================================
    # NUMPY STATISTICS
    # ========================================================

    areas = boxes_df["area_percentage"].to_numpy()

    print()
    print("=" * 70)
    print("NUMPY ANALYSIS")
    print("=" * 70)

    print(
        f"Mean pothole area %   : "
        f"{np.mean(areas):.4f}%"
    )

    print(
        f"Median pothole area % : "
        f"{np.median(areas):.4f}%"
    )

    print(
        f"Std deviation         : "
        f"{np.std(areas):.4f}"
    )

    # ========================================================
    # SAVE CSV FILES
    # ========================================================

    reports_folder = (
        PROJECT_ROOT
        / "outputs"
        / "reports"
    )

    reports_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    images_csv = (
        reports_folder
        / "image_analysis.csv"
    )

    boxes_csv = (
        reports_folder
        / "bounding_box_analysis.csv"
    )

    images_df.to_csv(
        images_csv,
        index=False,
    )

    boxes_df.to_csv(
        boxes_csv,
        index=False,
    )

    print()
    print("=" * 70)
    print("REPORTS SAVED")
    print("=" * 70)

    print(
        f"Image analysis:"
        f"\n  {images_csv}"
    )

    print(
        f"Bounding box analysis:"
        f"\n  {boxes_csv}"
    )

    print()
    print("=" * 70)
    print("DATASET ANALYSIS COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()