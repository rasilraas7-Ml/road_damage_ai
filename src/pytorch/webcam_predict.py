import os
import sys
import argparse
import time

import cv2
import torch


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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\n" + "=" * 60)
    print("LOADING POTHOLE DETECTION MODEL")
    print("=" * 60)

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
# PREPROCESS FRAME
# ============================================================

def preprocess_frame(frame):

    original_height, original_width = (
        frame.shape[:2]
    )

    resized_frame = cv2.resize(
        frame,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    rgb_frame = cv2.cvtColor(
        resized_frame,
        cv2.COLOR_BGR2RGB
    )

    frame_tensor = torch.from_numpy(
        rgb_frame
    ).float() / 255.0

    frame_tensor = frame_tensor.permute(
        2,
        0,
        1
    )

    frame_tensor = frame_tensor.unsqueeze(0)

    frame_tensor = frame_tensor.to(DEVICE)

    return (
        frame_tensor,
        original_width,
        original_height
    )


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    frame,
    boxes,
    scores,
    original_width,
    original_height
):

    output_frame = frame.copy()

    scale_x = (
        original_width /
        IMAGE_SIZE
    )

    scale_y = (
        original_height /
        IMAGE_SIZE
    )

    detection_count = len(boxes)

    # --------------------------------------------------------
    # Draw bounding boxes
    # --------------------------------------------------------

    for box, score in zip(
        boxes,
        scores
    ):

        x1, y1, x2, y2 = box.tolist()

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)

        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # Keep coordinates inside frame

        x1 = max(
            0,
            min(
                x1,
                original_width - 1
            )
        )

        y1 = max(
            0,
            min(
                y1,
                original_height - 1
            )
        )

        x2 = max(
            0,
            min(
                x2,
                original_width - 1
            )
        )

        y2 = max(
            0,
            min(
                y2,
                original_height - 1
            )
        )

        confidence = (
            float(score) * 100
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            output_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"Pothole "
            f"{confidence:.1f}%"
        )

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

        label_bottom = (
            label_top +
            text_height +
            baseline +
            5
        )

        cv2.rectangle(
            output_frame,
            (
                x1,
                label_top
            ),
            (
                x1 + text_width + 6,
                label_bottom
            ),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            output_frame,
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

    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    panel_height = 85

    cv2.rectangle(
        output_frame,
        (0, 0),
        (
            original_width,
            panel_height
        ),
        (0, 0, 0),
        -1
    )

    # Detection count

    cv2.putText(
        output_frame,
        f"Potholes: {detection_count}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # FPS

    return output_frame


# ============================================================
# DETECT FRAME
# ============================================================

def detect_frame(
    model,
    frame,
    confidence_threshold
):

    (
        frame_tensor,
        original_width,
        original_height
    ) = preprocess_frame(
        frame
    )

    with torch.no_grad():

        predictions = model(
            frame_tensor
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

    # --------------------------------------------------------
    # Confidence filtering
    # --------------------------------------------------------

    keep = (
        scores >= confidence_threshold
    )

    boxes = boxes[keep]

    scores = scores[keep]

    labels = labels[keep]

    # --------------------------------------------------------
    # Draw
    # --------------------------------------------------------

    output_frame = draw_detections(
        frame,
        boxes,
        scores,
        original_width,
        original_height
    )

    return (
        output_frame,
        len(boxes)
    )


# ============================================================
# MAIN WEBCAM LOOP
# ============================================================

def run_webcam(
    model,
    camera_index,
    confidence_threshold
):

    print("\n" + "=" * 60)
    print("LIVE WEBCAM POTHOLE DETECTION")
    print("=" * 60)

    print(
        f"Camera index: {camera_index}"
    )

    print(
        f"Confidence threshold: "
        f"{confidence_threshold}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print("\nOpening camera...")

    # --------------------------------------------------------
    # Open webcam
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        camera_index
    )

    if not cap.isOpened():

        raise RuntimeError(
            "\nCould not open webcam.\n\n"
            "Try another camera index:\n"
            "  --camera 1\n"
            "  --camera 2\n"
        )

    # Try to use a reasonable webcam resolution

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    print(
        "\nWebcam opened successfully."
    )

    print(
        "Press Q to quit."
    )

    # ========================================================
    # FPS VARIABLES
    # ========================================================

    frame_count = 0

    start_time = time.time()

    display_fps = 0.0

    # ========================================================
    # WEBCAM LOOP
    # ========================================================

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print(
                    "\nCould not read frame."
                )

                break

            frame_count += 1

            # ------------------------------------------------
            # Model inference
            # ------------------------------------------------

            inference_start = time.time()

            output_frame, detection_count = (
                detect_frame(
                    model,
                    frame,
                    confidence_threshold
                )
            )

            inference_time = (
                time.time() -
                inference_start
            )

            inference_fps = (
                1.0 / inference_time
                if inference_time > 0
                else 0
            )

            # ------------------------------------------------
            # Calculate average FPS
            # ------------------------------------------------

            elapsed = (
                time.time() -
                start_time
            )

            display_fps = (
                frame_count / elapsed
                if elapsed > 0
                else 0
            )

            # ------------------------------------------------
            # FPS information
            # ------------------------------------------------

            cv2.putText(
                output_frame,
                f"Inference: {inference_fps:.2f} FPS",
                (15, 57),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                output_frame,
                "Press Q to quit",
                (
                    15,
                    output_frame.shape[0] - 15
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            cv2.imshow(
                "Pothole Detection - Webcam",
                output_frame
            )

            # ------------------------------------------------
            # Quit
            # ------------------------------------------------

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                print(
                    "\nStopping webcam..."
                )

                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

    # ========================================================
    # SUMMARY
    # ========================================================

    total_time = (
        time.time() -
        start_time
    )

    average_fps = (
        frame_count / total_time
        if total_time > 0
        else 0
    )

    print("\n" + "=" * 60)
    print("WEBCAM SESSION COMPLETE")
    print("=" * 60)

    print(
        f"Frames processed: "
        f"{frame_count}"
    )

    print(
        f"Session time: "
        f"{total_time:.2f} seconds"
    )

    print(
        f"Average FPS: "
        f"{average_fps:.2f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Live pothole detection "
            "using Faster R-CNN"
        )
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help=(
            "Camera index "
            "(default: 0)"
        )
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

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate threshold
    # --------------------------------------------------------

    if not 0.0 <= args.threshold <= 1.0:

        raise ValueError(
            "Threshold must be between "
            "0.0 and 1.0"
        )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"\nModel not found:\n"
            f"{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Start webcam
    # --------------------------------------------------------

    run_webcam(
        model=model,
        camera_index=args.camera,
        confidence_threshold=args.threshold
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()