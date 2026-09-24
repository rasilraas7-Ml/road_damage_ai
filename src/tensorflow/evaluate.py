import os
import sys
import csv
import cv2
import numpy as np
import tensorflow as tf


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
    load_image_and_boxes,
    get_image_files,
    VALID_IMAGES_DIR,
    VALID_LABELS_DIR,
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

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

REPORT_DIR = os.path.join(
    OUTPUT_DIR,
    "reports"
)

PREDICTION_DIR = os.path.join(
    OUTPUT_DIR,
    "predictions",
    "tensorflow_evaluation"
)

IMAGE_SIZE = 640

CONFIDENCE_THRESHOLD = 0.40

IOU_THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70
]

MAX_VISUALIZATIONS = 20

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

os.makedirs(
    PREDICTION_DIR,
    exist_ok=True
)


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):
    """
    Box format:
        [x1, y1, x2, y2]

    Both boxes must use the same coordinate system.
    """

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

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
        max(
            0.0,
            box1[2] - box1[0]
        )
        *
        max(
            0.0,
            box1[3] - box1[1]
        )
    )

    area2 = (
        max(
            0.0,
            box2[2] - box2[0]
        )
        *
        max(
            0.0,
            box2[3] - box2[1]
        )
    )

    union_area = (
        area1 +
        area2 -
        intersection_area
    )

    if union_area <= 0:
        return 0.0

    return (
        intersection_area /
        union_area
    )


# ============================================================
# NMS
# ============================================================

def non_max_suppression(
    boxes,
    scores,
    iou_threshold=0.50
):

    if len(boxes) == 0:
        return []

    boxes = np.asarray(
        boxes,
        dtype=np.float32
    )

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    order = scores.argsort()[::-1]

    keep = []

    while len(order) > 0:

        current = order[0]

        keep.append(current)

        if len(order) == 1:
            break

        remaining = order[1:]

        current_box = boxes[current]

        remaining_boxes = boxes[
            remaining
        ]

        xx1 = np.maximum(
            current_box[0],
            remaining_boxes[:, 0]
        )

        yy1 = np.maximum(
            current_box[1],
            remaining_boxes[:, 1]
        )

        xx2 = np.minimum(
            current_box[2],
            remaining_boxes[:, 2]
        )

        yy2 = np.minimum(
            current_box[3],
            remaining_boxes[:, 3]
        )

        widths = np.maximum(
            0.0,
            xx2 - xx1
        )

        heights = np.maximum(
            0.0,
            yy2 - yy1
        )

        intersection = (
            widths *
            heights
        )

        current_area = (
            max(
                0.0,
                current_box[2] -
                current_box[0]
            )
            *
            max(
                0.0,
                current_box[3] -
                current_box[1]
            )
        )

        remaining_areas = (
            np.maximum(
                0.0,
                remaining_boxes[:, 2] -
                remaining_boxes[:, 0]
            )
            *
            np.maximum(
                0.0,
                remaining_boxes[:, 3] -
                remaining_boxes[:, 1]
            )
        )

        union = (
            current_area +
            remaining_areas -
            intersection
        )

        ious = np.divide(
            intersection,
            union,
            out=np.zeros_like(
                intersection
            ),
            where=union > 0
        )

        order = remaining[
            ious <= iou_threshold
        ]

    return keep


# ============================================================
# DECODE PREDICTIONS
# ============================================================

def decode_predictions(
    classification_logits,
    bbox_predictions,
    confidence_threshold
):
    """
    Decode TensorFlow model output.

    Classification:
        [batch, 20, 20, 2]

    Bounding boxes:
        [batch, 20, 20, 4]

    Bounding box format:
        normalized [x1, y1, x2, y2]
    """

    probabilities = tf.nn.softmax(
        classification_logits,
        axis=-1
    ).numpy()

    bbox_predictions = (
        bbox_predictions.numpy()
    )

    batch_size = (
        probabilities.shape[0]
    )

    results = []

    for batch_index in range(
        batch_size
    ):

        boxes = []
        scores = []

        grid_height = (
            probabilities.shape[1]
        )

        grid_width = (
            probabilities.shape[2]
        )

        for row in range(
            grid_height
        ):

            for col in range(
                grid_width
            ):

                confidence = float(
                    probabilities[
                        batch_index,
                        row,
                        col,
                        1
                    ]
                )

                if (
                    confidence <
                    confidence_threshold
                ):
                    continue

                box = bbox_predictions[
                    batch_index,
                    row,
                    col
                ].copy()

                # Convert to valid normalized coordinates
                box = np.clip(
                    box,
                    0.0,
                    1.0
                )

                x1, y1, x2, y2 = box

                # Correct coordinate ordering
                x1, x2 = sorted(
                    [x1, x2]
                )

                y1, y2 = sorted(
                    [y1, y2]
                )

                if x2 <= x1:
                    continue

                if y2 <= y1:
                    continue

                boxes.append(
                    [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    ]
                )

                scores.append(
                    confidence
                )

        # ----------------------------------------------------
        # NMS
        # ----------------------------------------------------

        if len(boxes) > 0:

            keep = non_max_suppression(
                boxes,
                scores,
                iou_threshold=0.50
            )

            boxes = [
                boxes[index]
                for index in keep
            ]

            scores = [
                scores[index]
                for index in keep
            ]

        results.append(
            {
                "boxes": boxes,
                "scores": scores
            }
        )

    return results


# ============================================================
# CONVERT GROUND TRUTH TO NORMALIZED
# ============================================================

def normalize_ground_truth_boxes(
    boxes
):
    """
    dataset.py returns boxes in 640x640
    pixel coordinates.

    Training converts them to normalized
    coordinates before creating targets.

    Evaluation therefore does the same.
    """

    normalized_boxes = []

    for box in boxes:

        x1, y1, x2, y2 = box

        x1 = (
            float(x1) /
            IMAGE_SIZE
        )

        y1 = (
            float(y1) /
            IMAGE_SIZE
        )

        x2 = (
            float(x2) /
            IMAGE_SIZE
        )

        y2 = (
            float(y2) /
            IMAGE_SIZE
        )

        normalized_boxes.append(
            [
                np.clip(x1, 0.0, 1.0),
                np.clip(y1, 0.0, 1.0),
                np.clip(x2, 0.0, 1.0),
                np.clip(y2, 0.0, 1.0)
            ]
        )

    return normalized_boxes


# ============================================================
# MATCH PREDICTIONS
# ============================================================

def match_predictions(
    predicted_boxes,
    predicted_scores,
    ground_truth_boxes,
    iou_threshold
):

    if len(predicted_boxes) == 0:

        return {
            "tp": 0,
            "fp": 0,
            "fn": len(
                ground_truth_boxes
            )
        }

    if len(ground_truth_boxes) == 0:

        return {
            "tp": 0,
            "fp": len(
                predicted_boxes
            ),
            "fn": 0
        }

    # Highest confidence first
    order = np.argsort(
        predicted_scores
    )[::-1]

    matched_ground_truth = set()

    tp = 0
    fp = 0

    for prediction_index in order:

        prediction_box = (
            predicted_boxes[
                prediction_index
            ]
        )

        best_iou = 0.0
        best_gt_index = -1

        for gt_index, gt_box in enumerate(
            ground_truth_boxes
        ):

            if gt_index in (
                matched_ground_truth
            ):
                continue

            iou = calculate_iou(
                prediction_box,
                gt_box
            )

            if iou > best_iou:

                best_iou = iou
                best_gt_index = (
                    gt_index
                )

        if (
            best_iou >= iou_threshold
            and best_gt_index >= 0
        ):

            tp += 1

            matched_ground_truth.add(
                best_gt_index
            )

        else:

            fp += 1

    fn = (
        len(ground_truth_boxes)
        -
        len(matched_ground_truth)
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    tp,
    fp,
    fn
):

    precision = (
        tp /
        (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp /
        (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 *
        precision *
        recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return (
        precision,
        recall,
        f1
    )


# ============================================================
# DRAW RESULTS
# ============================================================

def draw_results(
    image_path,
    ground_truth_boxes,
    predicted_boxes,
    predicted_scores,
    output_path
):

    image = cv2.imread(
        image_path
    )

    if image is None:
        return

    height, width = (
        image.shape[:2]
    )

    # --------------------------------------------------------
    # Ground truth - GREEN
    # --------------------------------------------------------

    for box in ground_truth_boxes:

        x1 = int(
            box[0] * width
        )

        y1 = int(
            box[1] * height
        )

        x2 = int(
            box[2] * width
        )

        y2 = int(
            box[3] * height
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            "GT",
            (
                x1,
                max(
                    20,
                    y1 - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------------
    # Predictions - RED
    # --------------------------------------------------------

    for box, score in zip(
        predicted_boxes,
        predicted_scores
    ):

        x1 = int(
            box[0] * width
        )

        y1 = int(
            box[1] * height
        )

        x2 = int(
            box[2] * width
        )

        y2 = int(
            box[3] * height
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        label = (
            f"Pothole "
            f"{score:.2f}"
        )

        cv2.putText(
            image,
            label,
            (
                x1,
                max(
                    20,
                    y1 - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

    cv2.imwrite(
        output_path,
        image
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print(
        "TensorFlow Pothole Detector - Evaluation"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading model...")
    print(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "✅ Model loaded successfully"
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "Loading validation dataset"
    )

    print(
        "=" * 70
    )

    image_files = get_image_files(
        VALID_IMAGES_DIR
    )

    print(
        f"\nValidation images: "
        f"{len(image_files)}"
    )

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD}"
    )

    print(
        f"IoU thresholds: "
        f"{IOU_THRESHOLDS}"
    )

    # --------------------------------------------------------
    # TOTALS
    # --------------------------------------------------------

    totals = {}

    for threshold in IOU_THRESHOLDS:

        totals[threshold] = {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }

    csv_rows = []

    visualization_count = 0

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    for image_index, image_path in enumerate(
        image_files
    ):

        filename = os.path.basename(
            image_path
        )

        label_path = os.path.join(
            VALID_LABELS_DIR,
            os.path.splitext(
                filename
            )[0] + ".txt"
        )

        if not os.path.exists(
            label_path
        ):

            print(
                f"⚠ Missing label: "
                f"{filename}"
            )

            continue

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image, ground_truth_boxes = (
            load_image_and_boxes(
                image_path,
                label_path
            )
        )

        ground_truth_boxes = (
            ground_truth_boxes.tolist()
        )

        # IMPORTANT:
        # Convert 640x640 pixel coordinates
        # to normalized 0-1 coordinates.

        ground_truth_boxes = (
            normalize_ground_truth_boxes(
                ground_truth_boxes
            )
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        input_tensor = np.expand_dims(
            image,
            axis=0
        )

        predictions = model(
            input_tensor,
            training=False
        )

        classification_logits = (
            predictions[
                "classification"
            ]
        )

        bbox_predictions = (
            predictions[
                "bounding_boxes"
            ]
        )

        decoded = decode_predictions(
            classification_logits,
            bbox_predictions,
            CONFIDENCE_THRESHOLD
        )[0]

        predicted_boxes = (
            decoded["boxes"]
        )

        predicted_scores = (
            decoded["scores"]
        )

        # ----------------------------------------------------
        # MATCH
        # ----------------------------------------------------

        for iou_threshold in (
            IOU_THRESHOLDS
        ):

            result = match_predictions(
                predicted_boxes,
                predicted_scores,
                ground_truth_boxes,
                iou_threshold
            )

            totals[
                iou_threshold
            ]["tp"] += result["tp"]

            totals[
                iou_threshold
            ]["fp"] += result["fp"]

            totals[
                iou_threshold
            ]["fn"] += result["fn"]

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        csv_rows.append(
            {
                "image": filename,

                "ground_truth_count":
                    len(
                        ground_truth_boxes
                    ),

                "prediction_count":
                    len(
                        predicted_boxes
                    ),

                "max_confidence":
                    (
                        max(
                            predicted_scores
                        )
                        if predicted_scores
                        else 0.0
                    )
            }
        )

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        if (
            visualization_count
            <
            MAX_VISUALIZATIONS
        ):

            output_path = os.path.join(
                PREDICTION_DIR,
                (
                    f"{visualization_count + 1:02d}_"
                    f"{filename}"
                )
            )

            draw_results(
                image_path,
                ground_truth_boxes,
                predicted_boxes,
                predicted_scores,
                output_path
            )

            visualization_count += 1

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            (image_index + 1) % 25 == 0
            or
            image_index ==
            len(image_files) - 1
        ):

            print(
                f"Processed "
                f"{image_index + 1}/"
                f"{len(image_files)}"
            )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL EVALUATION RESULTS"
    )

    print(
        "=" * 70
    )

    results_rows = []

    for iou_threshold in (
        IOU_THRESHOLDS
    ):

        tp = totals[
            iou_threshold
        ]["tp"]

        fp = totals[
            iou_threshold
        ]["fp"]

        fn = totals[
            iou_threshold
        ]["fn"]

        precision, recall, f1 = (
            calculate_metrics(
                tp,
                fp,
                fn
            )
        )

        print(
            f"\nIoU Threshold: "
            f"{iou_threshold:.2f}"
        )

        print(
            f"TP: {tp}"
        )

        print(
            f"FP: {fp}"
        )

        print(
            f"FN: {fn}"
        )

        print(
            f"Precision: "
            f"{precision:.4f}"
        )

        print(
            f"Recall:    "
            f"{recall:.4f}"
        )

        print(
            f"F1 Score:  "
            f"{f1:.4f}"
        )

        results_rows.append(
            {
                "confidence_threshold":
                    CONFIDENCE_THRESHOLD,

                "iou_threshold":
                    iou_threshold,

                "TP":
                    tp,

                "FP":
                    fp,

                "FN":
                    fn,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1
            }
        )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_path = os.path.join(
        REPORT_DIR,
        "tensorflow_evaluation_metrics.csv"
    )

    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "confidence_threshold",
                "iou_threshold",
                "TP",
                "FP",
                "FN",
                "precision",
                "recall",
                "f1"
            ]
        )

        writer.writeheader()

        writer.writerows(
            results_rows
        )

    # ========================================================
    # SAVE PER IMAGE
    # ========================================================

    per_image_path = os.path.join(
        REPORT_DIR,
        "tensorflow_evaluation_per_image.csv"
    )

    with open(
        per_image_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "ground_truth_count",
                "prediction_count",
                "max_confidence"
            ]
        )

        writer.writeheader()

        writer.writerows(
            csv_rows
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "Evaluation complete"
    )

    print(
        "=" * 70
    )

    print(
        "\nMetrics saved to:"
    )

    print(
        metrics_path
    )

    print(
        "\nPer-image results saved to:"
    )

    print(
        per_image_path
    )

    print(
        "\nVisualizations saved to:"
    )

    print(
        PREDICTION_DIR
    )

    print(
        "\n" + "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()