# ============================================================
# ROADGUARD AI
# Flask Web Application
# Road Pothole Detection
#
# Features:
#   1. Image Upload Detection
#   2. Browser Camera Detection
#   3. OpenCV Image Processing
#   4. PyTorch Faster R-CNN
#   5. Live Detection API
# ============================================================

import os
import sys
import uuid
import base64

import cv2
import torch
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
)

from werkzeug.utils import secure_filename


# ============================================================
# 1. PROJECT PATHS
# ============================================================

APP_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(APP_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# 2. IMPORT MODEL
# ============================================================

from src.pytorch.model import create_model


# ============================================================
# 3. FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(
        APP_DIR,
        "templates"
    ),
    static_folder=os.path.join(
        APP_DIR,
        "static"
    )
)


# ============================================================
# 4. CONFIGURATION
# ============================================================

UPLOAD_FOLDER = os.path.join(
    APP_DIR,
    "static",
    "uploads"
)

RESULT_FOLDER = os.path.join(
    APP_DIR,
    "static",
    "results"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "pytorch",
    "best_pothole_detector.pth"
)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
    "bmp"
}

MAX_FILE_SIZE = 25 * 1024 * 1024

CONFIDENCE_THRESHOLD = 0.40

IMAGE_SIZE = 640


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# ============================================================
# 5. CREATE DIRECTORIES
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)


# ============================================================
# 6. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("=" * 60)
print("ROADGUARD AI")
print("=" * 60)
print(
    f"Device: {DEVICE}"
)
print(
    f"Model: {MODEL_PATH}"
)
print(
    f"Confidence threshold: {CONFIDENCE_THRESHOLD}"
)
print("=" * 60)
print()


# ============================================================
# 7. LOAD MODEL
# ============================================================

print("Loading PyTorch model...")


# IMPORTANT:
# Existing create_model() does NOT accept num_classes.
model = create_model()


# ============================================================
# 8. LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

else:

    model.load_state_dict(
        checkpoint
    )


model.to(DEVICE)

model.eval()


print("Model loaded successfully.")
print()


# ============================================================
# 9. HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):
    """
    Check whether a file has an allowed image extension.
    """

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = (
        filename
        .rsplit(".", 1)[1]
        .lower()
    )

    return extension in ALLOWED_EXTENSIONS


# ------------------------------------------------------------


def generate_unique_filename(
    original_filename
):
    """
    Generate a unique safe filename.
    """

    safe_name = secure_filename(
        original_filename
    )

    extension = ""

    if "." in safe_name:

        extension = (
            "."
            + safe_name
            .rsplit(".", 1)[1]
            .lower()
        )

    unique_id = uuid.uuid4().hex[:12]

    return (
        f"road_image_{unique_id}"
        f"{extension}"
    )


# ============================================================
# 10. CORE AI DETECTION FUNCTION
# ============================================================

def run_detection(
    original_image
):
    """
    Run Faster R-CNN detection on an OpenCV image.

    Input:
        original_image = BGR OpenCV image

    Returns:
        annotated_image
        detections
        average_confidence
        highest_confidence
    """

    if original_image is None:

        raise ValueError(
            "Invalid image."
        )


    # --------------------------------------------------------
    # Original image dimensions
    # --------------------------------------------------------

    original_height, original_width = (
        original_image.shape[:2]
    )


    # --------------------------------------------------------
    # OpenCV BGR -> RGB
    # --------------------------------------------------------

    rgb_image = cv2.cvtColor(
        original_image,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Resize for model
    # --------------------------------------------------------

    resized_image = cv2.resize(
        rgb_image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        interpolation=cv2.INTER_LINEAR
    )


    # --------------------------------------------------------
    # Convert image -> PyTorch tensor
    # --------------------------------------------------------

    image_tensor = (
        torch.from_numpy(
            resized_image
        )
        .permute(2, 0, 1)
        .float()
        / 255.0
    )


    image_tensor = image_tensor.to(
        DEVICE
    )


    # --------------------------------------------------------
    # AI INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        predictions = model(
            [image_tensor]
        )


    prediction = predictions[0]


    # --------------------------------------------------------
    # Get predictions
    # --------------------------------------------------------

    boxes = (
        prediction["boxes"]
        .detach()
        .cpu()
        .numpy()
    )

    scores = (
        prediction["scores"]
        .detach()
        .cpu()
        .numpy()
    )

    labels = (
        prediction["labels"]
        .detach()
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # Confidence filtering
    # --------------------------------------------------------

    keep = (
        scores >=
        CONFIDENCE_THRESHOLD
    )


    boxes = boxes[keep]

    scores = scores[keep]

    labels = labels[keep]


    # --------------------------------------------------------
    # Scale boxes back to original image
    # --------------------------------------------------------

    scale_x = (
        original_width /
        IMAGE_SIZE
    )

    scale_y = (
        original_height /
        IMAGE_SIZE
    )


    # --------------------------------------------------------
    # Create output image
    # --------------------------------------------------------

    annotated_image = (
        original_image.copy()
    )


    detections = []


    # ========================================================
    # DRAW DETECTIONS
    # ========================================================

    for box, score, label in zip(
        boxes,
        scores,
        labels
    ):

        x1, y1, x2, y2 = box


        # Scale coordinates
        x1 = int(
            x1 * scale_x
        )

        y1 = int(
            y1 * scale_y
        )

        x2 = int(
            x2 * scale_x
        )

        y2 = int(
            y2 * scale_y
        )


        # Keep inside image
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


        # ----------------------------------------------------
        # Save detection
        # ----------------------------------------------------

        detections.append({

            "confidence":
                float(score),

            "label":
                "Pothole",

            "box": [
                x1,
                y1,
                x2,
                y2
            ]
        })


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            (99, 230, 163),
            3
        )


        # ----------------------------------------------------
        # Detection label
        # ----------------------------------------------------

        label_text = (
            f"Pothole "
            f"{score * 100:.1f}%"
        )


        font = (
            cv2.FONT_HERSHEY_SIMPLEX
        )

        font_scale = 0.65

        thickness = 2


        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label_text,
            font,
            font_scale,
            thickness
        )


        label_y = max(
            y1,
            text_height + 10
        )


        # Label background
        cv2.rectangle(
            annotated_image,
            (
                x1,
                label_y - text_height - 10
            ),
            (
                x1 + text_width + 10,
                label_y
            ),
            (99, 230, 163),
            -1
        )


        # Label text
        cv2.putText(
            annotated_image,
            label_text,
            (
                x1 + 5,
                label_y - 5
            ),
            font,
            font_scale,
            (5, 15, 10),
            thickness,
            cv2.LINE_AA
        )


    # ========================================================
    # ADD STATUS INFORMATION
    # ========================================================

    detection_count = len(
        detections
    )


    if detection_count > 0:

        confidence_values = [
            item["confidence"]
            for item in detections
        ]

        average_confidence = float(
            np.mean(
                confidence_values
            )
        )

        highest_confidence = float(
            np.max(
                confidence_values
            )
        )

    else:

        average_confidence = 0.0

        highest_confidence = 0.0


    # --------------------------------------------------------
    # Status panel
    # --------------------------------------------------------

    status_text = (
        f"POTHOLES: {detection_count}"
    )


    cv2.rectangle(
        annotated_image,
        (15, 15),
        (235, 60),
        (10, 15, 20),
        -1
    )


    cv2.rectangle(
        annotated_image,
        (15, 15),
        (235, 60),
        (99, 230, 163),
        1
    )


    cv2.putText(
        annotated_image,
        status_text,
        (28, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (99, 230, 163),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "image":
            annotated_image,

        "detections":
            detections,

        "detection_count":
            detection_count,

        "average_confidence":
            average_confidence,

        "highest_confidence":
            highest_confidence,

        "image_width":
            original_width,

        "image_height":
            original_height
    }


# ============================================================
# 11. IMAGE DETECTION FUNCTION
# ============================================================

def detect_image(
    image_path,
    result_path
):
    """
    Image upload detection.
    """

    original_image = cv2.imread(
        image_path
    )


    if original_image is None:

        raise ValueError(
            "Unable to read uploaded image."
        )


    result = run_detection(
        original_image
    )


    success = cv2.imwrite(
        result_path,
        result["image"]
    )


    if not success:

        raise RuntimeError(
            "Failed to save result image."
        )


    return result


# ============================================================
# 12. HOME ROUTE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# 13. IMAGE ANALYSIS ROUTE
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    if "image" not in request.files:

        return redirect(
            url_for("index")
        )


    file = request.files["image"]


    if file.filename == "":

        return redirect(
            url_for("index")
        )


    if not allowed_file(
        file.filename
    ):

        return (
            "Invalid image format.",
            400
        )


    unique_filename = (
        generate_unique_filename(
            file.filename
        )
    )


    upload_path = os.path.join(
        UPLOAD_FOLDER,
        unique_filename
    )


    result_filename = (
        "detected_"
        + unique_filename
    )


    result_path = os.path.join(
        RESULT_FOLDER,
        result_filename
    )


    file.save(
        upload_path
    )


    try:

        result = detect_image(
            upload_path,
            result_path
        )


        result_image_url = url_for(
            "static",
            filename=(
                "results/"
                + result_filename
            )
        )


        return render_template(

            "result.html",

            result_image=
                result_image_url,

            detections=
                result["detections"],

            detection_count=
                result["detection_count"],

            average_confidence=
                result["average_confidence"],

            highest_confidence=
                result["highest_confidence"],

            image_width=
                result["image_width"],

            image_height=
                result["image_height"]
        )


    except Exception as error:

        print()
        print("=" * 60)
        print("IMAGE DETECTION ERROR")
        print("=" * 60)
        print(error)
        print("=" * 60)
        print()


        return (
            f"""
            <h1>Detection Error</h1>
            <pre>{error}</pre>
            <a href="/">Back</a>
            """,
            500
        )


# ============================================================
# 14. CAMERA DETECTION API
# ============================================================

@app.route(
    "/camera/detect",
    methods=["POST"]
)
def camera_detect():

    try:

        # ----------------------------------------------------
        # Browser sends:
        # multipart/form-data
        # frame = JPEG image
        # ----------------------------------------------------

        if "frame" not in request.files:

            return jsonify({
                "success": False,
                "error":
                    "No camera frame received."
            }), 400


        frame_file = request.files[
            "frame"
        ]


        frame_bytes = (
            frame_file.read()
        )


        if not frame_bytes:

            return jsonify({
                "success": False,
                "error":
                    "Empty camera frame."
            }), 400


        # ----------------------------------------------------
        # Bytes -> NumPy
        # ----------------------------------------------------

        np_array = np.frombuffer(
            frame_bytes,
            np.uint8
        )


        # ----------------------------------------------------
        # NumPy -> OpenCV image
        # ----------------------------------------------------

        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )


        if frame is None:

            return jsonify({
                "success": False,
                "error":
                    "Could not decode camera frame."
            }), 400


        # ----------------------------------------------------
        # OpenCV + Faster R-CNN
        # ----------------------------------------------------

        result = run_detection(
            frame
        )


        # ----------------------------------------------------
        # Encode processed image
        # ----------------------------------------------------

        success, encoded_image = (
            cv2.imencode(
                ".jpg",
                result["image"],
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    80
                ]
            )
        )


        if not success:

            raise RuntimeError(
                "Could not encode detection frame."
            )


        # ----------------------------------------------------
        # JPEG -> Base64
        # ----------------------------------------------------

        image_base64 = base64.b64encode(
            encoded_image.tobytes()
        ).decode(
            "utf-8"
        )


        # ----------------------------------------------------
        # Return JSON
        # ----------------------------------------------------

        return jsonify({

            "success":
                True,

            "image":
                "data:image/jpeg;base64,"
                + image_base64,

            "detection_count":
                result[
                    "detection_count"
                ],

            "average_confidence":
                result[
                    "average_confidence"
                ],

            "highest_confidence":
                result[
                    "highest_confidence"
                ],

            "detections":
                result[
                    "detections"
                ]
        })


    except Exception as error:

        print()
        print(
            "CAMERA DETECTION ERROR:"
        )
        print(error)
        print()


        return jsonify({

            "success":
                False,

            "error":
                str(error)

        }), 500


# ============================================================
# 15. HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "healthy",

        "model":
            "Faster R-CNN ResNet-50 FPN",

        "device":
            str(DEVICE),

        "confidence_threshold":
            CONFIDENCE_THRESHOLD
    })


# ============================================================
# 16. FILE SIZE ERROR
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return (
        """
        <h1>File Too Large</h1>

        <p>
            Maximum allowed image size is 25 MB.
        </p>

        <a href="/">
            Go back
        </a>
        """,
        413
    )


# ============================================================
# 17. RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("ROADGUARD AI SERVER")
    print("=" * 60)
    print(
        "Open: http://127.0.0.1:5000"
    )
    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )