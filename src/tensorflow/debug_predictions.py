import os
import sys
import numpy as np
import tensorflow as tf
import cv2

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT DATASET
# ============================================================

from src.tensorflow.dataset import (
    VALID_IMAGES_DIR,
    VALID_LABELS_DIR,
    load_image_and_boxes,
    get_image_files,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "tensorflow",
    "best_pothole_detector.keras"
)

CONFIDENCE_THRESHOLD = 0.40

IMAGE_SIZE = 640

GRID_SIZE = 20

NUM_IMAGES = 10


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0.0,
        x2 - x1
    )

    intersection_height = max(
        0.0,
        y2 - y1
    )

    intersection_area = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0.0, box1[2] - box1[0]) *
        max(0.0, box1[3] - box1[1])
    )

    area2 = (
        max(0.0, box2[2] - box2[0]) *
        max(0.0, box2[3] - box2[1])
    )

    union = (
        area1 +
        area2 -
        intersection_area
    )

    if union <= 0:
        return 0.0

    return intersection_area / union


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("TensorFlow Prediction Debugger")
print("=" * 70)

print("\nLoading model:")
print(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("✅ Model loaded")


# ============================================================
# GET VALIDATION IMAGES
# ============================================================

image_files = get_image_files(
    VALID_IMAGES_DIR
)

print(
    f"\nValidation images available: "
    f"{len(image_files)}"
)

print(
    f"Inspecting first "
    f"{NUM_IMAGES} images..."
)


# ============================================================
# DEBUG LOOP
# ============================================================

for image_index, image_path in enumerate(
    image_files[:NUM_IMAGES]
):

    filename = os.path.basename(
        image_path
    )

    label_path = os.path.join(
        VALID_LABELS_DIR,
        os.path.splitext(filename)[0] + ".txt"
    )

    print("\n")
    print("=" * 70)
    print(
        f"IMAGE {image_index + 1}/{NUM_IMAGES}"
    )
    print(
        f"File: {filename}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image, ground_truth_boxes = (
        load_image_and_boxes(
            image_path,
            label_path
        )
    )

    ground_truth_boxes = (
        ground_truth_boxes.tolist()
    )

    print(
        f"\nGround truth boxes: "
        f"{len(ground_truth_boxes)}"
    )

    for index, box in enumerate(
        ground_truth_boxes
    ):

        print(
            f"  GT {index + 1}: "
            f"{np.round(box, 4)}"
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    input_tensor = np.expand_dims(
        image,
        axis=0
    )

    predictions = model(
        input_tensor,
        training=False
    )

    classification_logits = (
        predictions["classification"]
    )

    bbox_predictions = (
        predictions["bounding_boxes"]
    )

    # --------------------------------------------------------
    # SOFTMAX
    # --------------------------------------------------------

    probabilities = tf.nn.softmax(
        classification_logits,
        axis=-1
    ).numpy()[0]

    bbox_predictions = (
        bbox_predictions.numpy()[0]
    )

    # --------------------------------------------------------
    # FIND HIGHEST CONFIDENCE CELLS
    # --------------------------------------------------------

    candidates = []

    for row in range(GRID_SIZE):

        for col in range(GRID_SIZE):

            confidence = float(
                probabilities[
                    row,
                    col,
                    1
                ]
            )

            candidates.append(
                (
                    confidence,
                    row,
                    col
                )
            )

    candidates.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    print(
        "\nTop 10 classification predictions:"
    )

    for rank, (
        confidence,
        row,
        col
    ) in enumerate(
        candidates[:10]
    ):

        raw_box = bbox_predictions[
            row,
            col
        ]

        clipped_box = np.clip(
            raw_box,
            0.0,
            1.0
        )

        x1, y1, x2, y2 = clipped_box

        # Fix ordering
        x1, x2 = sorted(
            [x1, x2]
        )

        y1, y2 = sorted(
            [y1, y2]
        )

        predicted_box = [
            x1,
            y1,
            x2,
            y2
        ]

        best_iou = 0.0

        for gt_box in ground_truth_boxes:

            iou = calculate_iou(
                predicted_box,
                gt_box
            )

            best_iou = max(
                best_iou,
                iou
            )

        print(
            f"\n  #{rank + 1}"
        )

        print(
            f"  Grid cell: "
            f"({row}, {col})"
        )

        print(
            f"  Confidence: "
            f"{confidence:.6f}"
        )

        print(
            f"  RAW bbox: "
            f"{np.round(raw_box, 6)}"
        )

        print(
            f"  Clipped bbox: "
            f"{np.round(predicted_box, 6)}"
        )

        print(
            f"  Best GT IoU: "
            f"{best_iou:.6f}"
        )

    # --------------------------------------------------------
    # HIGH CONFIDENCE PREDICTIONS
    # --------------------------------------------------------

    high_confidence = [
        item
        for item in candidates
        if item[0] >= CONFIDENCE_THRESHOLD
    ]

    print(
        "\nPredictions above confidence "
        f"{CONFIDENCE_THRESHOLD}: "
        f"{len(high_confidence)}"
    )

    for index, (
        confidence,
        row,
        col
    ) in enumerate(
        high_confidence[:10]
    ):

        raw_box = bbox_predictions[
            row,
            col
        ]

        print(
            f"\nPrediction {index + 1}:"
        )

        print(
            f"  Cell: ({row}, {col})"
        )

        print(
            f"  Confidence: "
            f"{confidence:.6f}"
        )

        print(
            f"  Raw bbox: "
            f"{np.round(raw_box, 6)}"
        )

    # --------------------------------------------------------
    # RAW BBOX STATISTICS
    # --------------------------------------------------------

    print(
        "\nBounding-box output statistics:"
    )

    print(
        f"  Min: "
        f"{bbox_predictions.min():.6f}"
    )

    print(
        f"  Max: "
        f"{bbox_predictions.max():.6f}"
    )

    print(
        f"  Mean: "
        f"{bbox_predictions.mean():.6f}"
    )

    print(
        f"  Values < 0: "
        f"{np.sum(bbox_predictions < 0)}"
    )

    print(
        f"  Values > 1: "
        f"{np.sum(bbox_predictions > 1)}"
    )


print("\n")
print("=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)