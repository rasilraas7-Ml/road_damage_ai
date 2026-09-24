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

DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "predictions",
    "video"
)

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

    original_height, original_width = frame.shape[:2]

    # Resize exactly like training
    resized_frame = cv2.resize(
        frame,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # OpenCV BGR -> RGB
    rgb_frame = cv2.cvtColor(
        resized_frame,
        cv2.COLOR_BGR2RGB
    )

    # NumPy -> Tensor
    frame_tensor = torch.from_numpy(
        rgb_frame
    ).float() / 255.0

    # HWC -> CHW
    frame_tensor = frame_tensor.permute(
        2,
        0,
        1
    )

    # Add batch dimension
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
    original_height,
    frame_number
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
    # Draw each detection
    # --------------------------------------------------------

    for box, score in zip(
        boxes,
        scores
    ):

        x1, y1, x2, y2 = box.tolist()

        # Convert 640x640 coordinates
        # back to original frame coordinates
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

        # Label background
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

        # Label text
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
    # INFO PANEL
    # ========================================================

    info_height = 65

    cv2.rectangle(
        output_frame,
        (0, 0),
        (
            original_width,
            info_height
        ),
        (0, 0, 0),
        -1
    )

    # Detection count
    count_text = (
        f"Potholes: {detection_count}"
    )

    cv2.putText(
        output_frame,
        count_text,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Frame number
    frame_text = (
        f"Frame: {frame_number}"
    )

    cv2.putText(
        output_frame,
        frame_text,
        (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return output_frame


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def process_frame(
    model,
    frame,
    confidence_threshold,
    frame_number
):

    original_height, original_width = (
        frame.shape[:2]
    )

    # Preprocess
    frame_tensor, _, _ = preprocess_frame(
        frame
    )

    # Inference
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
        original_height,
        frame_number
    )

    return output_frame, len(boxes)


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(
    model,
    video_path,
    output_path,
    confidence_threshold,
    show
):

    print("\n" + "=" * 60)
    print("VIDEO POTHOLE DETECTION")
    print("=" * 60)

    print(f"Input video:")
    print(video_path)

    print(
        f"\nConfidence threshold: "
        f"{confidence_threshold}"
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"\nCould not open video:\n"
            f"{video_path}"
        )

    # --------------------------------------------------------
    # Video properties
    # --------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    print("\nVideo information:")
    print(
        f"  Resolution: "
        f"{width} x {height}"
    )

    print(
        f"  FPS: {fps:.2f}"
    )

    print(
        f"  Frames: {total_frames}"
    )

    print(
        f"  Duration: "
        f"{duration:.2f} seconds"
    )

    # --------------------------------------------------------
    # Fix invalid FPS
    # --------------------------------------------------------

    if fps <= 0:

        fps = 30.0

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "\nCould not create output video.\n"
            "Try using a different output path."
        )

    # ========================================================
    # PROCESS FRAMES
    # ========================================================

    frame_number = 0

    total_detections = 0

    start_time = time.time()

    print("\nProcessing video...")
    print(
        "Press Q in the video window "
        "to stop early."
    )

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # ------------------------------------------------
            # Process frame
            # ------------------------------------------------

            processed_frame, detection_count = (
                process_frame(
                    model,
                    frame,
                    confidence_threshold,
                    frame_number
                )
            )

            total_detections += (
                detection_count
            )

            # ------------------------------------------------
            # Write frame
            # ------------------------------------------------

            writer.write(
                processed_frame
            )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            if show:

                cv2.imshow(
                    "Pothole Detection - Video",
                    processed_frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):

                    print(
                        "\nStopped by user."
                    )

                    break

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                frame_number % 10 == 0
                or frame_number == 1
            ):

                elapsed = (
                    time.time() -
                    start_time
                )

                processing_fps = (
                    frame_number /
                    elapsed
                    if elapsed > 0
                    else 0
                )

                if total_frames > 0:

                    progress = (
                        frame_number /
                        total_frames
                    ) * 100

                    print(
                        f"\rProgress: "
                        f"{progress:6.2f}% | "
                        f"Frame: "
                        f"{frame_number}/"
                        f"{total_frames} | "
                        f"Detections: "
                        f"{detection_count} | "
                        f"Speed: "
                        f"{processing_fps:.2f} FPS",
                        end=""
                    )

                else:

                    print(
                        f"\rFrame: "
                        f"{frame_number} | "
                        f"Detections: "
                        f"{detection_count} | "
                        f"Speed: "
                        f"{processing_fps:.2f} FPS",
                        end=""
                    )

    finally:

        cap.release()

        writer.release()

        if show:

            cv2.destroyAllWindows()

    # ========================================================
    # SUMMARY
    # ========================================================

    elapsed = (
        time.time() -
        start_time
    )

    average_fps = (
        frame_number / elapsed
        if elapsed > 0
        else 0
    )

    print("\n")

    print("=" * 60)
    print("VIDEO PROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"Frames processed: "
        f"{frame_number}"
    )

    print(
        f"Total detections: "
        f"{total_detections}"
    )

    print(
        f"Processing time: "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"Average processing speed: "
        f"{average_fps:.2f} FPS"
    )

    print(
        f"\nOutput video:\n"
        f"{output_path}"
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Pothole detection on video "
            "using Faster R-CNN"
        )
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video"
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
        help=(
            "Optional output video path"
        )
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Show processed video while "
            "processing"
        )
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDATION
    # ========================================================

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
    # Check video
    # --------------------------------------------------------

    if not os.path.exists(
        args.video
    ):

        raise FileNotFoundError(
            f"\nInput video not found:\n"
            f"{args.video}"
        )

    # --------------------------------------------------------
    # Create default output path
    # --------------------------------------------------------

    if args.output is None:

        os.makedirs(
            DEFAULT_OUTPUT_DIR,
            exist_ok=True
        )

        filename = os.path.basename(
            args.video
        )

        name, _ = os.path.splitext(
            filename
        )

        output_path = os.path.join(
            DEFAULT_OUTPUT_DIR,
            f"{name}_detected.mp4"
        )

    else:

        output_path = args.output

        output_directory = os.path.dirname(
            output_path
        )

        if output_directory:

            os.makedirs(
                output_directory,
                exist_ok=True
            )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Process video
    # --------------------------------------------------------

    process_video(
        model=model,
        video_path=args.video,
        output_path=output_path,
        confidence_threshold=args.threshold,
        show=args.show
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()