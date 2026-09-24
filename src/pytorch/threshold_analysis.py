import os
import csv
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import PotholeDataset
from dataloader import detection_collate_fn
from model import create_model


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


# ============================================================
# VALIDATION DATASET
# ============================================================

VAL_IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "potholes",
    "valid",
    "images"
)

VAL_LABEL_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "potholes",
    "valid",
    "labels"
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "pytorch",
    "best_pothole_detector.pth"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

REPORT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "reports"
)

GRAPH_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "graphs"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

os.makedirs(
    GRAPH_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 640

BATCH_SIZE = 2

NUM_WORKERS = 0


# Confidence thresholds
CONFIDENCE_THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


# IoU threshold used for determining
# whether a prediction is a correct detection.
IOU_THRESHOLD = 0.50


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

def print_configuration():

    print()
    print("=" * 70)
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 70)

    print(
        f"Device          : {DEVICE}"
    )

    print(
        f"Image size      : {IMAGE_SIZE}"
    )

    print(
        f"Batch size      : {BATCH_SIZE}"
    )

    print(
        f"IoU threshold   : {IOU_THRESHOLD}"
    )

    print()

    print(
        "Confidence thresholds:"
    )

    for threshold in CONFIDENCE_THRESHOLDS:

        print(
            f"  {threshold:.2f}"
        )

    print()

    print(
        "Model:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        "Validation images:"
    )

    print(
        VAL_IMAGE_DIR
    )

    print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print("=" * 70)
    print("LOADING TRAINED MODEL")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATH}"
        )

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Best model checkpoint
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print()
        print(
            "Checkpoint information:"
        )

        print(
            f"Epoch      : "
            f"{checkpoint.get('epoch', 'unknown')}"
        )

        print(
            f"Train loss : "
            f"{checkpoint.get('train_loss', 'unknown')}"
        )

        print(
            f"Val loss   : "
            f"{checkpoint.get('val_loss', 'unknown')}"
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(DEVICE)

    model.eval()

    print()
    print(
        "Model loaded successfully."
    )

    return model


# ============================================================
# CREATE VALIDATION DATASET
# ============================================================

def create_dataset():

    print()
    print("=" * 70)
    print("CREATING VALIDATION DATASET")
    print("=" * 70)

    dataset = PotholeDataset(
        image_dir=VAL_IMAGE_DIR,
        label_dir=VAL_LABEL_DIR,
        image_size=IMAGE_SIZE
    )

    print()

    print(
        f"Validation images: "
        f"{len(dataset)}"
    )

    return dataset


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    box_a,
    box_b
):

    # Box format:
    #
    # [x1, y1, x2, y2]

    x1 = max(
        box_a[0],
        box_b[0]
    )

    y1 = max(
        box_a[1],
        box_b[1]
    )

    x2 = min(
        box_a[2],
        box_b[2]
    )

    y2 = min(
        box_a[3],
        box_b[3]
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
        intersection_width
        * intersection_height
    )

    area_a = (
        max(
            0.0,
            box_a[2] - box_a[0]
        )
        *
        max(
            0.0,
            box_a[3] - box_a[1]
        )
    )

    area_b = (
        max(
            0.0,
            box_b[2] - box_b[0]
        )
        *
        max(
            0.0,
            box_b[3] - box_b[1]
        )
    )

    union_area = (
        area_a
        +
        area_b
        -
        intersection_area
    )

    if union_area <= 0:

        return 0.0

    return (
        intersection_area
        /
        union_area
    )


# ============================================================
# MATCH PREDICTIONS
# ============================================================

def match_predictions(
    prediction_boxes,
    prediction_scores,
    ground_truth_boxes,
    iou_threshold
):

    # --------------------------------------------------------
    # Sort predictions by confidence
    # --------------------------------------------------------

    if len(prediction_boxes) == 0:

        return (
            0,
            0,
            len(ground_truth_boxes)
        )

    sorted_indices = np.argsort(
        prediction_scores
    )[::-1]

    prediction_boxes = (
        prediction_boxes[
            sorted_indices
        ]
    )

    prediction_scores = (
        prediction_scores[
            sorted_indices
        ]
    )

    # --------------------------------------------------------
    # Track matched GT boxes
    # --------------------------------------------------------

    matched_ground_truth = set()

    true_positives = 0

    false_positives = 0

    # --------------------------------------------------------
    # Match predictions
    # --------------------------------------------------------

    for prediction_box in prediction_boxes:

        best_iou = 0.0

        best_gt_index = -1

        for gt_index, gt_box in enumerate(
            ground_truth_boxes
        ):

            # Already matched
            if gt_index in matched_ground_truth:

                continue

            iou = calculate_iou(
                prediction_box,
                gt_box
            )

            if iou > best_iou:

                best_iou = iou

                best_gt_index = gt_index

        # ----------------------------------------------------
        # Correct detection
        # ----------------------------------------------------

        if (
            best_iou >= iou_threshold
            and best_gt_index >= 0
        ):

            true_positives += 1

            matched_ground_truth.add(
                best_gt_index
            )

        # ----------------------------------------------------
        # Incorrect detection
        # ----------------------------------------------------

        else:

            false_positives += 1

    # --------------------------------------------------------
    # Missed detections
    # --------------------------------------------------------

    false_negatives = (
        len(ground_truth_boxes)
        -
        len(matched_ground_truth)
    )

    return (
        true_positives,
        false_positives,
        false_negatives
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    true_positives,
    false_positives,
    false_negatives
):

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision_denominator = (
        true_positives
        +
        false_positives
    )

    if precision_denominator > 0:

        precision = (
            true_positives
            /
            precision_denominator
        )

    else:

        precision = 0.0

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall_denominator = (
        true_positives
        +
        false_negatives
    )

    if recall_denominator > 0:

        recall = (
            true_positives
            /
            recall_denominator
        )

    else:

        recall = 0.0

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    if (
        precision
        +
        recall
    ) > 0:

        f1 = (
            2
            *
            precision
            *
            recall
            /
            (
                precision
                +
                recall
            )
        )

    else:

        f1 = 0.0

    return (
        precision,
        recall,
        f1
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_thresholds():

    print_configuration()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = create_dataset()

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=detection_collate_fn
    )

    # --------------------------------------------------------
    # Prepare statistics
    # --------------------------------------------------------

    statistics = {}

    for threshold in CONFIDENCE_THRESHOLDS:

        statistics[threshold] = {

            "true_positives": 0,

            "false_positives": 0,

            "false_negatives": 0
        }

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING THRESHOLD ANALYSIS")
    print("=" * 70)

    total_images = len(dataset)

    processed_images = 0

    with torch.no_grad():

        for batch_index, (
            images,
            targets
        ) in enumerate(data_loader):

            # ------------------------------------------------
            # Move images to device
            # ------------------------------------------------

            images_device = [
                image.to(DEVICE)
                for image in images
            ]

            # ------------------------------------------------
            # Model predictions
            # ------------------------------------------------

            predictions = model(
                images_device
            )

            # ------------------------------------------------
            # Process images
            # ------------------------------------------------

            for image_index in range(
                len(images)
            ):

                prediction = predictions[
                    image_index
                ]

                target = targets[
                    image_index
                ]

                # Ground truth
                ground_truth_boxes = (
                    target["boxes"]
                    .cpu()
                    .numpy()
                )

                # Predictions
                prediction_boxes = (
                    prediction["boxes"]
                    .cpu()
                    .numpy()
                )

                prediction_scores = (
                    prediction["scores"]
                    .cpu()
                    .numpy()
                )

                # ------------------------------------------------
                # Test every confidence threshold
                # ------------------------------------------------

                for confidence_threshold in (
                    CONFIDENCE_THRESHOLDS
                ):

                    confidence_mask = (
                        prediction_scores
                        >= confidence_threshold
                    )

                    filtered_boxes = (
                        prediction_boxes[
                            confidence_mask
                        ]
                    )

                    filtered_scores = (
                        prediction_scores[
                            confidence_mask
                        ]
                    )

                    (
                        tp,
                        fp,
                        fn
                    ) = match_predictions(
                        filtered_boxes,
                        filtered_scores,
                        ground_truth_boxes,
                        IOU_THRESHOLD
                    )

                    statistics[
                        confidence_threshold
                    ]["true_positives"] += tp

                    statistics[
                        confidence_threshold
                    ]["false_positives"] += fp

                    statistics[
                        confidence_threshold
                    ]["false_negatives"] += fn

                processed_images += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                batch_index == 0
                or (batch_index + 1) % 25 == 0
                or processed_images == total_images
            ):

                print(
                    f"Processed "
                    f"{processed_images}/"
                    f"{total_images} images"
                )

    # ========================================================
    # CALCULATE FINAL RESULTS
    # ========================================================

    results = []

    print()
    print()
    print("=" * 70)
    print("THRESHOLD ANALYSIS RESULTS")
    print("=" * 70)

    for confidence_threshold in (
        CONFIDENCE_THRESHOLDS
    ):

        tp = statistics[
            confidence_threshold
        ]["true_positives"]

        fp = statistics[
            confidence_threshold
        ]["false_positives"]

        fn = statistics[
            confidence_threshold
        ]["false_negatives"]

        precision, recall, f1 = (
            calculate_metrics(
                tp,
                fp,
                fn
            )
        )

        result = {

            "confidence_threshold":
                confidence_threshold,

            "iou_threshold":
                IOU_THRESHOLD,

            "true_positives":
                tp,

            "false_positives":
                fp,

            "false_negatives":
                fn,

            "precision":
                precision,

            "recall":
                recall,

            "f1_score":
                f1
        }

        results.append(
            result
        )

        print()
        print(
            f"Confidence: "
            f"{confidence_threshold:.2f}"
        )

        print("-" * 50)

        print(
            f"TP        : {tp}"
        )

        print(
            f"FP        : {fp}"
        )

        print(
            f"FN        : {fn}"
        )

        print(
            f"Precision : "
            f"{precision:.4f}"
        )

        print(
            f"Recall    : "
            f"{recall:.4f}"
        )

        print(
            f"F1 Score  : "
            f"{f1:.4f}"
        )

    # ========================================================
    # FIND BEST F1
    # ========================================================

    best_result = max(
        results,
        key=lambda item: item[
            "f1_score"
        ]
    )

    best_threshold = (
        best_result[
            "confidence_threshold"
        ]
    )

    best_precision = (
        best_result[
            "precision"
        ]
    )

    best_recall = (
        best_result[
            "recall"
        ]
    )

    best_f1 = (
        best_result[
            "f1_score"
        ]
    )

    print()
    print("=" * 70)
    print("BEST F1 THRESHOLD")
    print("=" * 70)

    print(
        f"Confidence threshold : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Precision            : "
        f"{best_precision:.4f}"
    )

    print(
        f"Recall               : "
        f"{best_recall:.4f}"
    )

    print(
        f"F1 Score             : "
        f"{best_f1:.4f}"
    )

    print("=" * 70)

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = os.path.join(
        REPORT_DIR,
        "threshold_analysis.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [

            "confidence_threshold",

            "iou_threshold",

            "true_positives",

            "false_positives",

            "false_negatives",

            "precision",

            "recall",

            "f1_score"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print()
    print(
        "CSV report saved:"
    )

    print(
        csv_path
    )

    # ========================================================
    # PREPARE GRAPH DATA
    # ========================================================

    thresholds = [
        result[
            "confidence_threshold"
        ]
        for result in results
    ]

    precisions = [
        result[
            "precision"
        ]
        for result in results
    ]

    recalls = [
        result[
            "recall"
        ]
        for result in results
    ]

    f1_scores = [
        result[
            "f1_score"
        ]
        for result in results
    ]

    # ========================================================
    # PRECISION / RECALL / F1 GRAPH
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        thresholds,
        precisions,
        marker="o",
        label="Precision"
    )

    plt.plot(
        thresholds,
        recalls,
        marker="o",
        label="Recall"
    )

    plt.plot(
        thresholds,
        f1_scores,
        marker="o",
        label="F1 Score"
    )

    plt.xlabel(
        "Confidence Threshold"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Confidence Threshold Analysis"
    )

    plt.xticks(
        thresholds
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    graph_path = os.path.join(
        GRAPH_DIR,
        "confidence_threshold_analysis.png"
    )

    plt.savefig(
        graph_path,
        dpi=150
    )

    plt.close()

    print()
    print(
        "Threshold graph saved:"
    )

    print(
        graph_path
    )

    # ========================================================
    # TP / FP / FN GRAPH
    # ========================================================

    true_positives = [
        result[
            "true_positives"
        ]
        for result in results
    ]

    false_positives = [
        result[
            "false_positives"
        ]
        for result in results
    ]

    false_negatives = [
        result[
            "false_negatives"
        ]
        for result in results
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        thresholds,
        true_positives,
        marker="o",
        label="True Positives"
    )

    plt.plot(
        thresholds,
        false_positives,
        marker="o",
        label="False Positives"
    )

    plt.plot(
        thresholds,
        false_negatives,
        marker="o",
        label="False Negatives"
    )

    plt.xlabel(
        "Confidence Threshold"
    )

    plt.ylabel(
        "Number of Detections"
    )

    plt.title(
        "Detection Counts vs Confidence Threshold"
    )

    plt.xticks(
        thresholds
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    count_graph_path = os.path.join(
        GRAPH_DIR,
        "threshold_detection_counts.png"
    )

    plt.savefig(
        count_graph_path,
        dpi=150
    )

    plt.close()

    print()
    print(
        "Detection-count graph saved:"
    )

    print(
        count_graph_path
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("THRESHOLD ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Images analyzed : "
        f"{processed_images}"
    )

    print(
        f"IoU threshold   : "
        f"{IOU_THRESHOLD}"
    )

    print(
        f"Best confidence : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Best F1 score   : "
        f"{best_f1:.4f}"
    )

    print()
    print(
        "Reports:"
    )

    print(
        csv_path
    )

    print()
    print(
        "Graphs:"
    )

    print(
        graph_path
    )

    print(
        count_graph_path
    )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    analyze_thresholds()