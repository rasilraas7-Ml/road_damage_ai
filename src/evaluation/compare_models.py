"""
Stage 10 - PyTorch vs TensorFlow Model Comparison

Models:
1. PyTorch Faster R-CNN ResNet-50 FPN
2. TensorFlow MobileNetV2 Grid Detector

Evaluation:
- Precision
- Recall
- F1 Score
- TP / FP / FN
- IoU thresholds
- Inference speed
- FPS
- Parameter count
- Model file size

IMPORTANT:
These are custom IoU-based detection metrics.
They are NOT COCO mAP.
"""

import sys
import time
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import tensorflow as tf


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT PYTORCH MODEL
# ============================================================

from src.pytorch.model import create_model as create_pytorch_model


# ============================================================
# IMAGE SETTINGS
# ============================================================

PYTORCH_IMAGE_SIZE = 640
TENSORFLOW_IMAGE_SIZE = 640


# ============================================================
# DATASET PATHS
# ============================================================

VAL_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "potholes"
    / "valid"
    / "images"
)

VAL_LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "potholes"
    / "valid"
    / "labels"
)


# ============================================================
# MODEL PATHS
# ============================================================

PYTORCH_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "pytorch"
    / "best_pothole_detector.pth"
)

TENSORFLOW_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tensorflow"
    / "best_pothole_detector.keras"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

METRICS_CSV = (
    OUTPUT_DIR
    / "model_comparison_metrics.csv"
)

MODEL_INFO_CSV = (
    OUTPUT_DIR
    / "model_comparison_model_info.csv"
)

PER_IMAGE_CSV = (
    OUTPUT_DIR
    / "model_comparison_per_image.csv"
)

JSON_REPORT = (
    OUTPUT_DIR
    / "model_comparison.json"
)


# ============================================================
# EVALUATION SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.40

IOU_THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]

NMS_IOU_THRESHOLD = 0.50


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# IOU
# ============================================================

def calculate_iou(box_a, box_b):
    """
    Calculate IoU between two boxes.

    Box format:
    [x1, y1, x2, y2]

    Coordinates must use the same scale.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(
        ax1,
        bx1
    )

    intersection_y1 = max(
        ay1,
        by1
    )

    intersection_x2 = min(
        ax2,
        bx2
    )

    intersection_y2 = min(
        ay2,
        by2
    )

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union_area = (
        area_a
        + area_b
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return (
        intersection_area
        / union_area
    )


# ============================================================
# NMS
# ============================================================

def non_max_suppression(
    boxes,
    scores,
    iou_threshold=0.50
):
    """
    Simple Non-Maximum Suppression.
    """

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

        filtered = []

        for index in remaining:

            iou = calculate_iou(
                boxes[current],
                boxes[index]
            )

            if iou < iou_threshold:
                filtered.append(index)

        order = np.array(
            filtered,
            dtype=np.int64
        )

    return keep


# ============================================================
# MATCH PREDICTIONS
# ============================================================

def match_predictions(
    predicted_boxes,
    predicted_scores,
    ground_truth_boxes,
    iou_threshold
):
    """
    Greedy one-to-one matching.

    Predictions are processed from
    highest confidence to lowest confidence.
    """

    if len(predicted_boxes) == 0:

        return {
            "tp": 0,
            "fp": 0,
            "fn": len(ground_truth_boxes),
        }

    if len(ground_truth_boxes) == 0:

        return {
            "tp": 0,
            "fp": len(predicted_boxes),
            "fn": 0,
        }

    prediction_order = np.argsort(
        predicted_scores
    )[::-1]

    matched_ground_truth = set()

    tp = 0
    fp = 0

    for prediction_index in prediction_order:

        prediction_box = (
            predicted_boxes[
                prediction_index
            ]
        )

        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt_box in enumerate(
            ground_truth_boxes
        ):

            if gt_index in matched_ground_truth:
                continue

            iou = calculate_iou(
                prediction_box,
                gt_box
            )

            if iou > best_iou:

                best_iou = iou
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and best_iou >= iou_threshold
        ):

            tp += 1

            matched_ground_truth.add(
                best_gt_index
            )

        else:

            fp += 1

    fn = (
        len(ground_truth_boxes)
        - len(matched_ground_truth)
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    tp,
    fp,
    fn
):
    """
    Calculate precision, recall and F1.
    """

    if tp + fp > 0:

        precision = (
            tp
            / (tp + fp)
        )

    else:

        precision = 0.0

    if tp + fn > 0:

        recall = (
            tp
            / (tp + fn)
        )

    else:

        recall = 0.0

    if precision + recall > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# VALIDATION IMAGES
# ============================================================

def get_validation_images():

    if not VAL_IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Validation image directory not found:\n"
            f"{VAL_IMAGE_DIR}"
        )

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    images = sorted(
        [
            path
            for path in VAL_IMAGE_DIR.iterdir()
            if path.suffix.lower()
            in extensions
        ]
    )

    return images


# ============================================================
# GROUND TRUTH
# ============================================================

def get_ground_truth(
    image_path
):
    """
    Read YOLO annotation.

    Returns normalized boxes:
    [x1, y1, x2, y2]

    All values are in [0,1].
    """

    label_path = (
        VAL_LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    if not label_path.exists():
        return []

    boxes = []

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            class_id = int(
                float(parts[0])
            )

            # Dataset has one class:
            # 0 = pothole
            if class_id != 0:
                continue

            x_center = float(
                parts[1]
            )

            y_center = float(
                parts[2]
            )

            box_width = float(
                parts[3]
            )

            box_height = float(
                parts[4]
            )

            x1 = (
                x_center
                - box_width / 2.0
            )

            y1 = (
                y_center
                - box_height / 2.0
            )

            x2 = (
                x_center
                + box_width / 2.0
            )

            y2 = (
                y_center
                + box_height / 2.0
            )

            boxes.append(
                [
                    np.clip(
                        x1,
                        0.0,
                        1.0
                    ),
                    np.clip(
                        y1,
                        0.0,
                        1.0
                    ),
                    np.clip(
                        x2,
                        0.0,
                        1.0
                    ),
                    np.clip(
                        y2,
                        0.0,
                        1.0
                    ),
                ]
            )

    return boxes


# ============================================================
# PYTORCH MODEL
# ============================================================

def load_pytorch_model():

    print()
    print("=" * 70)
    print("Loading PyTorch model")
    print("=" * 70)

    print()
    print(
        f"Model path:\n"
        f"{PYTORCH_MODEL_PATH}"
    )

    if not PYTORCH_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nPyTorch model not found:\n"
            f"{PYTORCH_MODEL_PATH}"
        )

    model = create_pytorch_model()

    checkpoint = torch.load(
        PYTORCH_MODEL_PATH,
        map_location=DEVICE
    )

    print(
        f"Checkpoint type: "
        f"{type(checkpoint).__name__}"
    )

    # --------------------------------------------------------
    # Detect checkpoint format
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

            print(
                "Checkpoint format: "
                "model_state_dict"
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

            print(
                "Checkpoint format: "
                "state_dict"
            )

        else:

            state_dict = checkpoint

            print(
                "Checkpoint format: "
                "direct state_dict"
            )

    else:

        state_dict = checkpoint

        print(
            "Checkpoint format: "
            "direct state_dict"
        )

    model.load_state_dict(
        state_dict
    )

    model.to(
        DEVICE
    )

    model.eval()

    print()
    print(
        "✅ PyTorch model loaded successfully"
    )

    return model


# ============================================================
# TENSORFLOW MODEL
# ============================================================

def load_tensorflow_model():

    print()
    print("=" * 70)
    print("Loading TensorFlow model")
    print("=" * 70)

    print()
    print(
        f"Model path:\n"
        f"{TENSORFLOW_MODEL_PATH}"
    )

    if not TENSORFLOW_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nTensorFlow model not found:\n"
            f"{TENSORFLOW_MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        TENSORFLOW_MODEL_PATH,
        compile=False
    )

    print()
    print(
        "✅ TensorFlow model loaded successfully"
    )

    return model


# ============================================================
# PYTORCH PREDICTION
# ============================================================

def predict_pytorch(
    model,
    image_path
):
    """
    Run Faster R-CNN prediction.

    Returns normalized boxes and scores.
    """

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        return [], []

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    resized = cv2.resize(
        image_rgb,
        (
            PYTORCH_IMAGE_SIZE,
            PYTORCH_IMAGE_SIZE
        )
    )

    tensor = (
        torch.from_numpy(
            resized
        )
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    tensor = tensor.to(
        DEVICE
    )

    with torch.no_grad():

        outputs = model(
            [tensor]
        )

    output = outputs[0]

    boxes = (
        output["boxes"]
        .detach()
        .cpu()
        .numpy()
    )

    scores = (
        output["scores"]
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Confidence filter
    # --------------------------------------------------------

    mask = (
        scores
        >= CONFIDENCE_THRESHOLD
    )

    boxes = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:

        return [], []

    # --------------------------------------------------------
    # NMS
    # --------------------------------------------------------

    keep = non_max_suppression(
        boxes,
        scores,
        NMS_IOU_THRESHOLD
    )

    boxes = boxes[keep]
    scores = scores[keep]

    # --------------------------------------------------------
    # Normalize coordinates
    # --------------------------------------------------------

    normalized_boxes = []

    for box in boxes:

        x1, y1, x2, y2 = box

        x1 = (
            x1
            / PYTORCH_IMAGE_SIZE
        )

        y1 = (
            y1
            / PYTORCH_IMAGE_SIZE
        )

        x2 = (
            x2
            / PYTORCH_IMAGE_SIZE
        )

        y2 = (
            y2
            / PYTORCH_IMAGE_SIZE
        )

        x1 = np.clip(
            x1,
            0.0,
            1.0
        )

        y1 = np.clip(
            y1,
            0.0,
            1.0
        )

        x2 = np.clip(
            x2,
            0.0,
            1.0
        )

        y2 = np.clip(
            y2,
            0.0,
            1.0
        )

        if x2 <= x1:
            continue

        if y2 <= y1:
            continue

        normalized_boxes.append(
            [
                x1,
                y1,
                x2,
                y2,
            ]
        )

    return (
        normalized_boxes,
        scores.tolist()
    )


# ============================================================
# TENSORFLOW DECODER
# ============================================================

def decode_tensorflow_predictions(
    predictions
):
    """
    Convert TensorFlow model outputs
    into normalized bounding boxes.
    """

    classification = np.asarray(
        predictions[
            "classification"
        ]
    )

    bounding_boxes = np.asarray(
        predictions[
            "bounding_boxes"
        ]
    )

    # Remove batch dimension
    class_map = classification[0]

    bbox_map = bounding_boxes[0]

    # Class 1 = pothole
    pothole_scores = (
        class_map[:, :, 1]
    )

    height, width = (
        pothole_scores.shape
    )

    boxes = []
    scores = []

    for row in range(height):

        for col in range(width):

            score = float(
                pothole_scores[
                    row,
                    col
                ]
            )

            if score < CONFIDENCE_THRESHOLD:
                continue

            box = bbox_map[
                row,
                col
            ]

            if len(box) != 4:
                continue

            x1 = float(
                box[0]
            )

            y1 = float(
                box[1]
            )

            x2 = float(
                box[2]
            )

            y2 = float(
                box[3]
            )

            x1 = np.clip(
                x1,
                0.0,
                1.0
            )

            y1 = np.clip(
                y1,
                0.0,
                1.0
            )

            x2 = np.clip(
                x2,
                0.0,
                1.0
            )

            y2 = np.clip(
                y2,
                0.0,
                1.0
            )

            if x2 <= x1:
                continue

            if y2 <= y1:
                continue

            boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ]
            )

            scores.append(
                score
            )

    if len(boxes) == 0:

        return [], []

    # --------------------------------------------------------
    # NMS
    # --------------------------------------------------------

    keep = non_max_suppression(
        boxes,
        scores,
        NMS_IOU_THRESHOLD
    )

    boxes = [
        boxes[index]
        for index in keep
    ]

    scores = [
        scores[index]
        for index in keep
    ]

    return boxes, scores


# ============================================================
# TENSORFLOW PREDICTION
# ============================================================

def predict_tensorflow(
    model,
    image_path
):
    """
    Run TensorFlow prediction.
    """

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        return [], []

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    resized = cv2.resize(
        image_rgb,
        (
            TENSORFLOW_IMAGE_SIZE,
            TENSORFLOW_IMAGE_SIZE
        )
    )

    image_input = (
        resized.astype(
            np.float32
        )
        / 255.0
    )

    image_input = np.expand_dims(
        image_input,
        axis=0
    )

    predictions = model(
        image_input,
        training=False
    )

    return decode_tensorflow_predictions(
        predictions
    )


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_pytorch_parameters(
    model
):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def count_tensorflow_parameters(
    model
):

    return model.count_params()


# ============================================================
# FILE SIZE
# ============================================================

def get_file_size_mb(
    path
):

    if not path.exists():

        return 0.0

    size_bytes = (
        path.stat().st_size
    )

    return (
        size_bytes
        / (1024 * 1024)
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_name,
    model,
    prediction_function,
    image_paths
):

    print()
    print("=" * 70)
    print(
        f"EVALUATING {model_name.upper()}"
    )
    print("=" * 70)

    results = []

    total_inference_time = 0.0
    total_predictions = 0
    total_ground_truth = 0

    # --------------------------------------------------------
    # Process validation images
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        ground_truth_boxes = (
            get_ground_truth(
                image_path
            )
        )

        start_time = (
            time.perf_counter()
        )

        predicted_boxes, predicted_scores = (
            prediction_function(
                model,
                image_path
            )
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        total_inference_time += elapsed

        total_predictions += (
            len(predicted_boxes)
        )

        total_ground_truth += (
            len(ground_truth_boxes)
        )

        image_result = {
            "model": model_name,
            "image": image_path.name,
            "ground_truth": len(
                ground_truth_boxes
            ),
            "predictions": len(
                predicted_boxes
            ),
            "inference_time": elapsed,
        }

        # ----------------------------------------------------
        # IoU evaluation
        # ----------------------------------------------------

        for iou_threshold in (
            IOU_THRESHOLDS
        ):

            match = match_predictions(
                predicted_boxes,
                predicted_scores,
                ground_truth_boxes,
                iou_threshold
            )

            metrics = calculate_metrics(
                match["tp"],
                match["fp"],
                match["fn"]
            )

            key = (
                f"{iou_threshold:.2f}"
            )

            image_result[
                f"tp_{key}"
            ] = match["tp"]

            image_result[
                f"fp_{key}"
            ] = match["fp"]

            image_result[
                f"fn_{key}"
            ] = match["fn"]

            image_result[
                f"precision_{key}"
            ] = metrics["precision"]

            image_result[
                f"recall_{key}"
            ] = metrics["recall"]

            image_result[
                f"f1_{key}"
            ] = metrics["f1"]

        results.append(
            image_result
        )

        if (
            index % 25 == 0
            or index == len(image_paths)
        ):

            print(
                f"Processed "
                f"{index}/{len(image_paths)}"
            )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    summary = []

    for iou_threshold in (
        IOU_THRESHOLDS
    ):

        key = (
            f"{iou_threshold:.2f}"
        )

        tp = sum(
            result[
                f"tp_{key}"
            ]
            for result in results
        )

        fp = sum(
            result[
                f"fp_{key}"
            ]
            for result in results
        )

        fn = sum(
            result[
                f"fn_{key}"
            ]
            for result in results
        )

        metrics = calculate_metrics(
            tp,
            fp,
            fn
        )

        summary.append(
            {
                "model": model_name,
                "iou_threshold":
                    iou_threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision":
                    metrics["precision"],
                "recall":
                    metrics["recall"],
                "f1":
                    metrics["f1"],
            }
        )

    average_time = (
        total_inference_time
        / len(image_paths)
    )

    fps = (
        1.0 / average_time
        if average_time > 0
        else 0.0
    )

    print()
    print(
        f"{model_name} inference:"
    )

    print(
        f"Average inference time: "
        f"{average_time:.4f} sec/image"
    )

    print(
        f"Approximate FPS: "
        f"{fps:.2f}"
    )

    print(
        f"Average predictions/image: "
        f"{total_predictions / len(image_paths):.2f}"
    )

    print(
        f"Average ground-truth boxes/image: "
        f"{total_ground_truth / len(image_paths):.2f}"
    )

    return {
        "summary": summary,
        "per_image": results,
        "average_inference_time":
            average_time,
        "fps":
            fps,
        "total_predictions":
            total_predictions,
        "total_ground_truth":
            total_ground_truth,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "STAGE 10 - PYTORCH VS TENSORFLOW"
    )
    print("MODEL COMPARISON")
    print("=" * 70)

    print()
    print(
        f"Project root:\n"
        f"{PROJECT_ROOT}"
    )

    print()
    print(
        f"Validation directory:\n"
        f"{VAL_IMAGE_DIR}"
    )

    print()
    print(
        f"PyTorch model:\n"
        f"{PYTORCH_MODEL_PATH}"
    )

    print()
    print(
        f"TensorFlow model:\n"
        f"{TENSORFLOW_MODEL_PATH}"
    )

    print()
    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD}"
    )

    print(
        f"IoU thresholds: "
        f"{IOU_THRESHOLDS}"
    )

    print()
    print(
        "Metric type:"
    )

    print(
        "Custom IoU-based "
        "Precision / Recall / F1"
    )

    print(
        "NOT COCO mAP"
    )

    # ========================================================
    # VALIDATION DATA
    # ========================================================

    image_paths = (
        get_validation_images()
    )

    print()
    print(
        f"Validation images found: "
        f"{len(image_paths)}"
    )

    if len(image_paths) == 0:

        raise RuntimeError(
            "No validation images found."
        )

    # ========================================================
    # LOAD PYTORCH
    # ========================================================

    pytorch_model = (
        load_pytorch_model()
    )

    # ========================================================
    # LOAD TENSORFLOW
    # ========================================================

    tensorflow_model = (
        load_tensorflow_model()
    )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    pytorch_parameters = (
        count_pytorch_parameters(
            pytorch_model
        )
    )

    tensorflow_parameters = (
        count_tensorflow_parameters(
            tensorflow_model
        )
    )

    pytorch_size = (
        get_file_size_mb(
            PYTORCH_MODEL_PATH
        )
    )

    tensorflow_size = (
        get_file_size_mb(
            TENSORFLOW_MODEL_PATH
        )
    )

    print()
    print("=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)

    print()
    print("PyTorch")
    print(
        "Architecture: "
        "Faster R-CNN ResNet-50 FPN"
    )

    print(
        f"Parameters: "
        f"{pytorch_parameters:,}"
    )

    print(
        f"Model size: "
        f"{pytorch_size:.2f} MB"
    )

    print()
    print("TensorFlow")
    print(
        "Architecture: "
        "MobileNetV2 Grid Detector"
    )

    print(
        f"Parameters: "
        f"{tensorflow_parameters:,}"
    )

    print(
        f"Model size: "
        f"{tensorflow_size:.2f} MB"
    )

    # ========================================================
    # PYTORCH EVALUATION
    # ========================================================

    pytorch_results = evaluate_model(
        model_name="PyTorch",
        model=pytorch_model,
        prediction_function=predict_pytorch,
        image_paths=image_paths,
    )

    # ========================================================
    # TENSORFLOW EVALUATION
    # ========================================================

    tensorflow_results = evaluate_model(
        model_name="TensorFlow",
        model=tensorflow_model,
        prediction_function=predict_tensorflow,
        image_paths=image_paths,
    )

    # ========================================================
    # COMBINE METRICS
    # ========================================================

    summary_df = pd.DataFrame(
        pytorch_results["summary"]
        + tensorflow_results["summary"]
    )

    # ========================================================
    # MODEL INFORMATION TABLE
    # ========================================================

    model_info_df = pd.DataFrame(
        [
            {
                "model":
                    "PyTorch",
                "framework":
                    "PyTorch",
                "architecture":
                    "Faster R-CNN ResNet-50 FPN",
                "parameters":
                    pytorch_parameters,
                "model_size_mb":
                    pytorch_size,
                "average_inference_seconds":
                    pytorch_results[
                        "average_inference_time"
                    ],
                "fps":
                    pytorch_results[
                        "fps"
                    ],
                "total_predictions":
                    pytorch_results[
                        "total_predictions"
                    ],
                "total_ground_truth":
                    pytorch_results[
                        "total_ground_truth"
                    ],
            },

            {
                "model":
                    "TensorFlow",
                "framework":
                    "TensorFlow",
                "architecture":
                    "MobileNetV2 Grid Detector",
                "parameters":
                    tensorflow_parameters,
                "model_size_mb":
                    tensorflow_size,
                "average_inference_seconds":
                    tensorflow_results[
                        "average_inference_time"
                    ],
                "fps":
                    tensorflow_results[
                        "fps"
                    ],
                "total_predictions":
                    tensorflow_results[
                        "total_predictions"
                    ],
                "total_ground_truth":
                    tensorflow_results[
                        "total_ground_truth"
                    ],
            },
        ]
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    summary_df.to_csv(
        METRICS_CSV,
        index=False
    )

    model_info_df.to_csv(
        MODEL_INFO_CSV,
        index=False
    )

    # ========================================================
    # PER IMAGE RESULTS
    # ========================================================

    per_image_df = pd.concat(
        [
            pd.DataFrame(
                pytorch_results[
                    "per_image"
                ]
            ),

            pd.DataFrame(
                tensorflow_results[
                    "per_image"
                ]
            ),
        ],
        ignore_index=True
    )

    per_image_df.to_csv(
        PER_IMAGE_CSV,
        index=False
    )

    # ========================================================
    # JSON REPORT
    # ========================================================

    report = {
        "evaluation": {
            "validation_images":
                len(image_paths),

            "confidence_threshold":
                CONFIDENCE_THRESHOLD,

            "iou_thresholds":
                IOU_THRESHOLDS,

            "metric_type":
                "Custom IoU-based "
                "Precision/Recall/F1",

            "coco_map":
                False,
        },

        "models": {
            "PyTorch": {
                "architecture":
                    "Faster R-CNN ResNet-50 FPN",

                "parameters":
                    int(
                        pytorch_parameters
                    ),

                "model_size_mb":
                    pytorch_size,

                "average_inference_seconds":
                    pytorch_results[
                        "average_inference_time"
                    ],

                "fps":
                    pytorch_results[
                        "fps"
                    ],

                "metrics":
                    pytorch_results[
                        "summary"
                    ],
            },

            "TensorFlow": {
                "architecture":
                    "MobileNetV2 Grid Detector",

                "parameters":
                    int(
                        tensorflow_parameters
                    ),

                "model_size_mb":
                    tensorflow_size,

                "average_inference_seconds":
                    tensorflow_results[
                        "average_inference_time"
                    ],

                "fps":
                    tensorflow_results[
                        "fps"
                    ],

                "metrics":
                    tensorflow_results[
                        "summary"
                    ],
            },
        },
    }

    with open(
        JSON_REPORT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    for iou_threshold in (
        IOU_THRESHOLDS
    ):

        print()
        print(
            f"IoU Threshold: "
            f"{iou_threshold:.2f}"
        )

        for model_name in (
            "PyTorch",
            "TensorFlow",
        ):

            rows = summary_df[
                (
                    summary_df[
                        "model"
                    ]
                    == model_name
                )
                &
                (
                    summary_df[
                        "iou_threshold"
                    ]
                    == iou_threshold
                )
            ]

            if rows.empty:
                continue

            row = rows.iloc[0]

            print(
                f"{model_name:<12} "
                f"TP: {int(row['tp']):<5} "
                f"FP: {int(row['fp']):<5} "
                f"FN: {int(row['fn']):<5} "
                f"P: {row['precision']:.4f} "
                f"R: {row['recall']:.4f} "
                f"F1: {row['f1']:.4f}"
            )

    # ========================================================
    # SPEED
    # ========================================================

    print()
    print("=" * 70)
    print("INFERENCE SPEED")
    print("=" * 70)

    for _, row in model_info_df.iterrows():

        print(
            f"{row['model']:<12} "
            f"{row['average_inference_seconds']:.4f} "
            f"sec/image | "
            f"{row['fps']:.2f} FPS"
        )

    # ========================================================
    # MODEL SIZE
    # ========================================================

    print()
    print("=" * 70)
    print("MODEL SIZE")
    print("=" * 70)

    for _, row in model_info_df.iterrows():

        print(
            f"{row['model']:<12} "
            f"{int(row['parameters']):,} parameters | "
            f"{row['model_size_mb']:.2f} MB"
        )

    # ========================================================
    # SAVED REPORTS
    # ========================================================

    print()
    print("=" * 70)
    print("REPORTS SAVED")
    print("=" * 70)

    print()
    print(
        f"Metrics:\n"
        f"{METRICS_CSV}"
    )

    print()
    print(
        f"Model information:\n"
        f"{MODEL_INFO_CSV}"
    )

    print()
    print(
        f"Per-image results:\n"
        f"{PER_IMAGE_CSV}"
    )

    print()
    print(
        f"JSON report:\n"
        f"{JSON_REPORT}"
    )

    print()
    print("=" * 70)
    print("STAGE 10 MODEL COMPARISON COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()