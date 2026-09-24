import os
import sys
import argparse

import cv2
import torch


# ============================================================
# PROJECT ROOT
# ============================================================

# predict.py
#   ↓
# src/
#   ↓
# project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.pytorch.model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "pytorch",
    "best_pothole_detector.pth"
)

DEFAULT_CONFIDENCE_THRESHOLD = 0.40

IMAGE_SIZE = 640

DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "predictions",
    "image"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading pothole detection model...")

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(DEVICE)

    model.eval()

    print("Model loaded successfully.")
    print(f"Device: {DEVICE}")

    return model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    original_height, original_width = image.shape[:2]

    # Resize exactly like training
    resized_image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # BGR -> RGB
    rgb_image = cv2.cvtColor(
        resized_image,
        cv2.COLOR_BGR2RGB
    )

    # NumPy -> Tensor
    image_tensor = torch.from_numpy(
        rgb_image
    ).float() / 255.0

    # HWC -> CHW
    image_tensor = image_tensor.permute(
        2,
        0,
        1
    )

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    return (
        image_tensor,
        original_width,
        original_height
    )


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    image,
    boxes,
    scores,
    original_width,
    original_height
):

    output_image = image.copy()

    scale_x = original_width / IMAGE_SIZE
    scale_y = original_height / IMAGE_SIZE

    detection_count = len(boxes)

    for box, score in zip(boxes, scores):

        x1, y1, x2, y2 = box

        # Convert back to original image coordinates
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)

        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # Keep coordinates inside image
        x1 = max(
            0,
            min(x1, original_width - 1)
        )

        y1 = max(
            0,
            min(y1, original_height - 1)
        )

        x2 = max(
            0,
            min(x2, original_width - 1)
        )

        y2 = max(
            0,
            min(y2, original_height - 1)
        )

        confidence = float(score) * 100

        # Bounding box
        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = f"Pothole {confidence:.1f}%"

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        label_top = max(
            0,
            y1 - text_height - baseline - 5
        )

        label_bottom = label_top + text_height + baseline + 5

        # Label background
        cv2.rectangle(
            output_image,
            (
                x1,
                label_top
            ),
            (
                x1 + text_width + 5,
                label_bottom
            ),
            (0, 255, 0),
            -1
        )

        # Label text
        cv2.putText(
            output_image,
            label,
            (
                x1 + 2,
                label_bottom - baseline - 3
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    # Detection count
    count_text = (
        f"Potholes detected: "
        f"{detection_count}"
    )

    cv2.rectangle(
        output_image,
        (10, 10),
        (350, 50),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output_image,
        count_text,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return output_image


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    image_path,
    confidence_threshold
):

    print("\n" + "=" * 60)
    print("POTHOLE DETECTION")
    print("=" * 60)

    print(f"Image: {image_path}")
    print(
        f"Confidence threshold: "
        f"{confidence_threshold}"
    )

    # Read image
    image = cv2.imread(
        image_path
    )

    if image is None:

        raise FileNotFoundError(
            f"\nCould not read image:\n"
            f"{image_path}"
        )

    original_height, original_width = image.shape[:2]

    print(
        f"Original image size: "
        f"{original_width} x "
        f"{original_height}"
    )

    # Preprocess
    image_tensor, _, _ = preprocess_image(
        image
    )

    # Inference
    print("\nRunning model inference...")

    with torch.no_grad():

        predictions = model(
            image_tensor
        )

    prediction = predictions[0]

    boxes = (
        prediction["boxes"]
        .detach()
        .cpu()
    )

    scores = (
        prediction["scores"]
        .detach()
        .cpu()
    )

    labels = (
        prediction["labels"]
        .detach()
        .cpu()
    )

    # Confidence filtering
    keep = scores >= confidence_threshold

    boxes = boxes[keep]

    scores = scores[keep]

    labels = labels[keep]

    print(
        f"\nDetections above threshold: "
        f"{len(boxes)}"
    )

    # Print detections
    if len(boxes) > 0:

        print("\nDetected potholes:")

        scale_x = (
            original_width /
            IMAGE_SIZE
        )

        scale_y = (
            original_height /
            IMAGE_SIZE
        )

        for i, (box, score) in enumerate(
            zip(boxes, scores),
            start=1
        ):

            x1, y1, x2, y2 = (
                box.tolist()
            )

            x1 *= scale_x
            x2 *= scale_x

            y1 *= scale_y
            y2 *= scale_y

            print(
                f"  {i}. "
                f"Confidence: "
                f"{score.item() * 100:.2f}% | "
                f"Box: "
                f"({int(x1)}, {int(y1)}) -> "
                f"({int(x2)}, {int(y2)})"
            )

    else:

        print(
            "\nNo potholes detected."
        )

    # Draw
    output_image = draw_detections(
        image,
        boxes,
        scores,
        original_width,
        original_height
    )

    return output_image


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_output(
    output_image,
    image_path,
    output_path=None
):

    if output_path is None:

        os.makedirs(
            DEFAULT_OUTPUT_DIR,
            exist_ok=True
        )

        filename = os.path.basename(
            image_path
        )

        name, extension = (
            os.path.splitext(filename)
        )

        output_path = os.path.join(
            DEFAULT_OUTPUT_DIR,
            f"{name}_detected{extension}"
        )

    else:

        output_directory = os.path.dirname(
            output_path
        )

        if output_directory:

            os.makedirs(
                output_directory,
                exist_ok=True
            )

    success = cv2.imwrite(
        output_path,
        output_image
    )

    if not success:

        raise RuntimeError(
            f"\nCould not save output:\n"
            f"{output_path}"
        )

    print("\n" + "=" * 60)
    print("RESULT SAVED")
    print("=" * 60)

    print(output_path)

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Pothole detection using "
            "Faster R-CNN"
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=(
            "Confidence threshold "
            "(default: 0.40)"
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image path"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display result image"
    )

    args = parser.parse_args()

    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:

        raise ValueError(
            "Threshold must be between "
            "0.0 and 1.0"
        )

    # Check model
    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"\nModel not found:\n"
            f"{MODEL_PATH}"
        )

    # Check image
    if not os.path.exists(
        args.image
    ):

        raise FileNotFoundError(
            f"\nInput image not found:\n"
            f"{args.image}"
        )

    # Load model
    model = load_model()

    # Predict
    output_image = predict(
        model,
        args.image,
        args.threshold
    )

    # Save
    output_path = save_output(
        output_image,
        args.image,
        args.output
    )

    # Show
    if args.show:

        cv2.imshow(
            "Pothole Detection",
            output_image
        )

        print(
            "\nPress any key inside "
            "the image window to close it."
        )

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    print(
        "\nPrediction completed successfully."
    )

    print(
        f"Output: {output_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()