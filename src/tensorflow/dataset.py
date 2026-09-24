import os
import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "potholes"
)

TRAIN_IMAGES_DIR = os.path.join(
    DATASET_ROOT,
    "train",
    "images"
)

TRAIN_LABELS_DIR = os.path.join(
    DATASET_ROOT,
    "train",
    "labels"
)

VALID_IMAGES_DIR = os.path.join(
    DATASET_ROOT,
    "valid",
    "images"
)

VALID_LABELS_DIR = os.path.join(
    DATASET_ROOT,
    "valid",
    "labels"
)

IMAGE_SIZE = 640


# ============================================================
# YOLO LABEL READER
# ============================================================

def read_yolo_labels(
    label_path,
    image_width,
    image_height
):

    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as file:

        lines = file.readlines()

    for line in lines:

        values = line.strip().split()

        if len(values) != 5:
            continue

        class_id = int(values[0])

        x_center = float(values[1])
        y_center = float(values[2])

        width = float(values[3])
        height = float(values[4])

        # YOLO normalized coordinates
        x_center *= image_width
        y_center *= image_height

        width *= image_width
        height *= image_height

        # Convert center format
        # to x1, y1, x2, y2

        x1 = x_center - width / 2
        y1 = y_center - height / 2

        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # Clip coordinates
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))

        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))

        boxes.append(
            [
                x1,
                y1,
                x2,
                y2
            ]
        )

    return boxes


# ============================================================
# LOAD IMAGE + LABEL
# ============================================================

def load_image_and_boxes(
    image_path,
    label_path
):

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise ValueError(
            f"Could not read image:\n"
            f"{image_path}"
        )

    original_height, original_width = (
        image.shape[:2]
    )

    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Read YOLO boxes
    boxes = read_yolo_labels(
        label_path,
        original_width,
        original_height
    )

    # Resize image
    image = cv2.resize(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    # Scale bounding boxes
    scale_x = (
        IMAGE_SIZE /
        original_width
    )

    scale_y = (
        IMAGE_SIZE /
        original_height
    )

    resized_boxes = []

    for box in boxes:

        x1, y1, x2, y2 = box

        resized_boxes.append(
            [
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y
            ]
        )

    # Normalize image
    image = image.astype(
        np.float32
    ) / 255.0

    return (
        image,
        np.array(
            resized_boxes,
            dtype=np.float32
        )
    )


# ============================================================
# DATASET GENERATOR
# ============================================================

def get_image_files(
    image_directory
):

    extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )

    files = []

    for filename in os.listdir(
        image_directory
    ):

        if filename.lower().endswith(
            extensions
        ):

            files.append(
                os.path.join(
                    image_directory,
                    filename
                )
            )

    files.sort()

    return files


# ============================================================
# GENERATOR
# ============================================================

def dataset_generator(
    images_directory,
    labels_directory
):

    image_files = get_image_files(
        images_directory
    )

    for image_path in image_files:

        filename = os.path.basename(
            image_path
        )

        name = os.path.splitext(
            filename
        )[0]

        label_path = os.path.join(
            labels_directory,
            name + ".txt"
        )

        image, boxes = (
            load_image_and_boxes(
                image_path,
                label_path
            )
        )

        yield image, boxes


# ============================================================
# DATASET INFORMATION
# ============================================================

def get_dataset_counts():

    train_images = get_image_files(
        TRAIN_IMAGES_DIR
    )

    valid_images = get_image_files(
        VALID_IMAGES_DIR
    )

    print(
        f"Training images: "
        f"{len(train_images)}"
    )

    print(
        f"Validation images: "
        f"{len(valid_images)}"
    )

    return (
        len(train_images),
        len(valid_images)
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("TENSORFLOW DATASET PIPELINE TEST")
    print("=" * 60)

    print(
        f"\nTensorFlow version: "
        f"{tf.__version__}"
    )

    train_count, valid_count = (
        get_dataset_counts()
    )

    print(
        "\nLoading first training image..."
    )

    image_files = get_image_files(
        TRAIN_IMAGES_DIR
    )

    if len(image_files) == 0:

        raise RuntimeError(
            "No training images found."
        )

    first_image = image_files[0]

    filename = os.path.basename(
        first_image
    )

    name = os.path.splitext(
        filename
    )[0]

    label_path = os.path.join(
        TRAIN_LABELS_DIR,
        name + ".txt"
    )

    image, boxes = (
        load_image_and_boxes(
            first_image,
            label_path
        )
    )

    print(
        f"\nImage: {filename}"
    )

    print(
        f"Image shape: "
        f"{image.shape}"
    )

    print(
        f"Image dtype: "
        f"{image.dtype}"
    )

    print(
        f"Image min: "
        f"{image.min():.4f}"
    )

    print(
        f"Image max: "
        f"{image.max():.4f}"
    )

    print(
        f"Bounding boxes: "
        f"{len(boxes)}"
    )

    if len(boxes) > 0:

        print(
            "\nFirst bounding box:"
        )

        print(
            boxes[0]
        )

    print("\nDataset pipeline test complete.")