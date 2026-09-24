import os
import csv
import cv2
import torch
import numpy as np

from torch.utils.data import DataLoader

from dataset import PotholeDataset
from dataloader import detection_collate_fn
from model import create_model


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

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "pytorch",
    "best_pothole_detector.pth"
)

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

REPORT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "reports"
)

PREDICTION_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "predictions",
    "evaluation"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

os.makedirs(
    PREDICTION_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 640

BATCH_SIZE = 2

NUM_WORKERS = 0

# Minimum confidence for a prediction
CONFIDENCE_THRESHOLD = 0.50

# IoU thresholds
IOU_THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70
]

# Number of images to save with visual predictions
NUM_VISUALIZATIONS = 20


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
    print("POTHOLE DETECTOR EVALUATION")
    print("=" * 70)

    print(f"Device               : {DEVICE}")
    print(f"Image size           : {IMAGE_SIZE}")
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Confidence threshold : {CONFIDENCE_THRESHOLD}")

    print()
    print("IoU thresholds:")

    for threshold in IOU_THRESHOLDS:
        print(f"  {threshold:.2f}")

    print()
    print(f"Model:")
    print(MODEL_PATH)

    print()
    print(f"Validation images:")
    print(VAL_IMAGE_DIR)

    print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print("=" * 70)
    print("LOADING MODEL")
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

    # best_pothole_detector.pth contains
    # model_state_dict
    if isinstance(checkpoint, dict) and \
       "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        saved_epoch = checkpoint.get(
            "epoch",
            "unknown"
        )

        saved_train_loss = checkpoint.get(
            "train_loss",
            "unknown"
        )

        saved_val_loss = checkpoint.get(
            "val_loss",
            "unknown"
        )

        print()
        print("Checkpoint information:")

        print(
            f"Epoch       : {saved_epoch}"
        )

        print(
            f"Train loss  : {saved_train_loss}"
        )

        print(
            f"Val loss    : {saved_val_loss}"
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(DEVICE)

    # Evaluation mode
    model.eval()

    print()
    print("Model loaded successfully.")

    return model


# ============================================================
# CREATE DATASET
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
        f"Validation samples: {len(dataset)}"
    )

    return dataset


# ============================================================
# IOU CALCULATION
# ============================================================

def calculate_iou(
    box_a,
    box_b
):

    # box format:
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

    area_a = max(
        0.0,
        box_a[2] - box_a[0]
    ) * max(
        0.0,
        box_a[3] - box_a[1]
    )

    area_b = max(
        0.0,
        box_b[2] - box_b[0]
    ) * max(
        0.0,
        box_b[3] - box_b[1]
    )

    union_area = (
        area_a
        + area_b
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


# ============================================================
# MATCH PREDICTIONS TO GROUND TRUTH
# ============================================================

def match_predictions(
    prediction_boxes,
    prediction_scores,
    ground_truth_boxes,
    iou_threshold
):

    # Sort predictions by confidence
    sorted_indices = np.argsort(
        prediction_scores
    )[::-1]

    prediction_boxes = (
        prediction_boxes[sorted_indices]
    )

    prediction_scores = (
        prediction_scores[sorted_indices]
    )

    matched_ground_truth = set()

    true_positives = 0

    false_positives = 0

    matched_ious = []

    # --------------------------------------------------------
    # Match each prediction
    # --------------------------------------------------------

    for prediction_box in prediction_boxes:

        best_iou = 0.0

        best_ground_truth_index = -1

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

                best_ground_truth_index = (
                    gt_index
                )

        # ----------------------------------------------------
        # True positive
        # ----------------------------------------------------

        if (
            best_iou >= iou_threshold
            and best_ground_truth_index >= 0
        ):

            true_positives += 1

            matched_ground_truth.add(
                best_ground_truth_index
            )

            matched_ious.append(
                best_iou
            )

        # ----------------------------------------------------
        # False positive
        # ----------------------------------------------------

        else:

            false_positives += 1

    # --------------------------------------------------------
    # False negatives
    # --------------------------------------------------------

    false_negatives = (
        len(ground_truth_boxes)
        - len(matched_ground_truth)
    )

    return (
        true_positives,
        false_positives,
        false_negatives,
        matched_ious
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    true_positives,
    false_positives,
    false_negatives
):

    precision_denominator = (
        true_positives
        + false_positives
    )

    recall_denominator = (
        true_positives
        + false_negatives
    )

    if precision_denominator > 0:

        precision = (
            true_positives
            / precision_denominator
        )

    else:

        precision = 0.0

    if recall_denominator > 0:

        recall = (
            true_positives
            / recall_denominator
        )

    else:

        recall = 0.0

    if (
        precision + recall
    ) > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    return (
        precision,
        recall,
        f1
    )


# ============================================================
# DRAW PREDICTIONS
# ============================================================

def draw_predictions(
    image_tensor,
    prediction_boxes,
    prediction_scores,
    ground_truth_boxes,
    filename
):

    # --------------------------------------------------------
    # Tensor -> NumPy
    # --------------------------------------------------------

    image = (
        image_tensor
        .detach()
        .cpu()
        .numpy()
    )

    image = (
        image.transpose(1, 2, 0)
        * 255.0
    )

    image = np.clip(
        image,
        0,
        255
    ).astype(
        np.uint8
    )

    # RGB -> BGR
    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # Draw ground truth boxes
    # --------------------------------------------------------

    for box in ground_truth_boxes:

        x1, y1, x2, y2 = [
            int(value)
            for value in box
        ]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            image,
            "GT",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # --------------------------------------------------------
    # Draw predictions
    # --------------------------------------------------------

    for box, score in zip(
        prediction_boxes,
        prediction_scores
    ):

        x1, y1, x2, y2 = [
            int(value)
            for value in box
        ]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = (
            f"Pothole "
            f"{score * 100:.1f}%"
        )

        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = os.path.join(
        PREDICTION_DIR,
        filename
    )

    cv2.imwrite(
        output_path,
        image
    )

    return output_path


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

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

    print()
    print("=" * 70)
    print("STARTING EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    statistics = {}

    for threshold in IOU_THRESHOLDS:

        statistics[threshold] = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "ious": []
        }

    visualization_count = 0

    total_images = len(dataset)

    processed_images = 0

    # --------------------------------------------------------
    # Disable gradients
    # --------------------------------------------------------

    with torch.no_grad():

        for batch_index, (
            images,
            targets
        ) in enumerate(data_loader):

            # Move images to device
            images_device = [
                image.to(DEVICE)
                for image in images
            ]

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            predictions = model(
                images_device
            )

            # ------------------------------------------------
            # Process each image
            # ------------------------------------------------

            for image_index in range(
                len(images)
            ):

                image = images[
                    image_index
                ]

                target = targets[
                    image_index
                ]

                prediction = predictions[
                    image_index
                ]

                # --------------------------------------------
                # Ground truth
                # --------------------------------------------

                ground_truth_boxes = (
                    target["boxes"]
                    .cpu()
                    .numpy()
                )

                # --------------------------------------------
                # Predictions
                # --------------------------------------------

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

                # --------------------------------------------
                # Confidence filtering
                # --------------------------------------------

                confidence_mask = (
                    prediction_scores
                    >= CONFIDENCE_THRESHOLD
                )

                prediction_boxes = (
                    prediction_boxes[
                        confidence_mask
                    ]
                )

                prediction_scores = (
                    prediction_scores[
                        confidence_mask
                    ]
                )

                # --------------------------------------------
                # Calculate metrics for each IoU threshold
                # --------------------------------------------

                for threshold in IOU_THRESHOLDS:

                    (
                        tp,
                        fp,
                        fn,
                        matched_ious
                    ) = match_predictions(
                        prediction_boxes,
                        prediction_scores,
                        ground_truth_boxes,
                        threshold
                    )

                    statistics[
                        threshold
                    ]["tp"] += tp

                    statistics[
                        threshold
                    ]["fp"] += fp

                    statistics[
                        threshold
                    ]["fn"] += fn

                    statistics[
                        threshold
                    ]["ious"].extend(
                        matched_ious
                    )

                # --------------------------------------------
                # Save visualizations
                # --------------------------------------------

                if (
                    visualization_count
                    < NUM_VISUALIZATIONS
                ):

                    original_filename = (
                        dataset.image_files[
                            processed_images
                        ]
                    )

                    output_filename = (
                        f"{visualization_count + 1:03d}_"
                        f"{original_filename}"
                    )

                    output_path = (
                        draw_predictions(
                            image,
                            prediction_boxes,
                            prediction_scores,
                            ground_truth_boxes,
                            output_filename
                        )
                    )

                    print()
                    print(
                        "Saved visualization:"
                    )

                    print(
                        output_path
                    )

                    visualization_count += 1

                processed_images += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            current = min(
                processed_images,
                total_images
            )

            if (
                batch_index == 0
                or (batch_index + 1) % 25 == 0
                or current == total_images
            ):

                print(
                    f"\nProcessed "
                    f"{current}/"
                    f"{total_images} images"
                )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    results = []

    for threshold in IOU_THRESHOLDS:

        tp = statistics[
            threshold
        ]["tp"]

        fp = statistics[
            threshold
        ]["fp"]

        fn = statistics[
            threshold
        ]["fn"]

        ious = statistics[
            threshold
        ]["ious"]

        precision, recall, f1 = (
            calculate_metrics(
                tp,
                fp,
                fn
            )
        )

        if len(ious) > 0:

            average_iou = float(
                np.mean(ious)
            )

        else:

            average_iou = 0.0

        print()
        print(
            f"IoU Threshold: "
            f"{threshold:.2f}"
        )

        print("-" * 50)

        print(
            f"True Positives  : {tp}"
        )

        print(
            f"False Positives : {fp}"
        )

        print(
            f"False Negatives : {fn}"
        )

        print(
            f"Precision       : "
            f"{precision:.4f}"
        )

        print(
            f"Recall          : "
            f"{recall:.4f}"
        )

        print(
            f"F1 Score        : "
            f"{f1:.4f}"
        )

        print(
            f"Average IoU     : "
            f"{average_iou:.4f}"
        )

        results.append(
            {
                "iou_threshold": threshold,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "average_iou": average_iou
            }
        )

    # ========================================================
    # SAVE CSV REPORT
    # ========================================================

    report_path = os.path.join(
        REPORT_DIR,
        "evaluation_results.csv"
    )

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "iou_threshold",
            "true_positives",
            "false_positives",
            "false_negatives",
            "precision",
            "recall",
            "f1_score",
            "average_iou"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print()
    print("=" * 70)

    print(
        "EVALUATION COMPLETE"
    )

    print("=" * 70)

    print()
    print(
        "Evaluation report:"
    )

    print(
        report_path
    )

    print()
    print(
        "Prediction visualizations:"
    )

    print(
        PREDICTION_DIR
    )

    print()
    print(
        f"Images evaluated: "
        f"{processed_images}"
    )

    print(
        f"Visualizations saved: "
        f"{visualization_count}"
    )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluate()