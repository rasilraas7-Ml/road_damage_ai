import os
import sys
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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
# IMPORT PROJECT FILES
# ============================================================

from src.tensorflow.dataset import (
    TRAIN_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    VALID_IMAGES_DIR,
    VALID_LABELS_DIR,
    IMAGE_SIZE,
    get_image_files,
    load_image_and_boxes,
)

from src.tensorflow.model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.0001

GRID_SIZE = 20

# Model classification output:
# 0 = background
# 1 = pothole
NUM_CLASSES = 2

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "tensorflow"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

GRAPH_DIR = os.path.join(
    OUTPUT_DIR,
    "graphs"
)

REPORT_DIR = os.path.join(
    OUTPUT_DIR,
    "reports"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    GRAPH_DIR,
    exist_ok=True
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# ============================================================
# INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("TENSORFLOW POTHOLE DETECTOR TRAINING")
print("=" * 70)

print(
    f"\nTensorFlow version: "
    f"{tf.__version__}"
)

print(
    f"Image size: "
    f"{IMAGE_SIZE}x{IMAGE_SIZE}"
)

print(
    f"Grid size: "
    f"{GRID_SIZE}x{GRID_SIZE}"
)

print(
    f"Batch size: "
    f"{BATCH_SIZE}"
)

print(
    f"Epochs: "
    f"{EPOCHS}"
)

print(
    f"Learning rate: "
    f"{LEARNING_RATE}"
)


# ============================================================
# DATASET FILES
# ============================================================

train_image_files = get_image_files(
    TRAIN_IMAGES_DIR
)

valid_image_files = get_image_files(
    VALID_IMAGES_DIR
)


print("\nDataset:")

print(
    f"Training images: "
    f"{len(train_image_files)}"
)

print(
    f"Validation images: "
    f"{len(valid_image_files)}"
)


if len(train_image_files) == 0:

    raise RuntimeError(
        "No training images found."
    )


if len(valid_image_files) == 0:

    raise RuntimeError(
        "No validation images found."
    )


# ============================================================
# CREATE GRID TARGETS
# ============================================================

def create_targets(boxes):

    # --------------------------------------------------------
    # CLASS TARGET
    #
    # 0 = background
    # 1 = pothole
    # --------------------------------------------------------

    class_target = np.zeros(
        (
            GRID_SIZE,
            GRID_SIZE
        ),
        dtype=np.int32
    )

    # --------------------------------------------------------
    # BOUNDING BOX TARGET
    #
    # [x1, y1, x2, y2]
    #
    # normalized 0-1
    # --------------------------------------------------------

    bbox_target = np.zeros(
        (
            GRID_SIZE,
            GRID_SIZE,
            4
        ),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # PROCESS EVERY BOX
    # --------------------------------------------------------

    for box in boxes:

        if len(box) != 4:
            continue

        x1, y1, x2, y2 = box

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        x1 = float(x1) / IMAGE_SIZE
        y1 = float(y1) / IMAGE_SIZE

        x2 = float(x2) / IMAGE_SIZE
        y2 = float(y2) / IMAGE_SIZE

        # ----------------------------------------------------
        # Clip
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Ignore invalid boxes
        # ----------------------------------------------------

        if x2 <= x1:
            continue

        if y2 <= y1:
            continue

        # ----------------------------------------------------
        # Find center
        # ----------------------------------------------------

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0

        # ----------------------------------------------------
        # Convert center to grid
        # ----------------------------------------------------

        grid_x = int(
            center_x * GRID_SIZE
        )

        grid_y = int(
            center_y * GRID_SIZE
        )

        grid_x = min(
            max(grid_x, 0),
            GRID_SIZE - 1
        )

        grid_y = min(
            max(grid_y, 0),
            GRID_SIZE - 1
        )

        # ----------------------------------------------------
        # Handle multiple boxes in same cell
        # ----------------------------------------------------

        if class_target[
            grid_y,
            grid_x
        ] == 1:

            old_box = (
                bbox_target[
                    grid_y,
                    grid_x
                ]
            )

            old_area = (
                old_box[2] -
                old_box[0]
            ) * (
                old_box[3] -
                old_box[1]
            )

            new_area = (
                x2 - x1
            ) * (
                y2 - y1
            )

            if new_area <= old_area:
                continue

        # ----------------------------------------------------
        # Store pothole
        # ----------------------------------------------------

        class_target[
            grid_y,
            grid_x
        ] = 1

        bbox_target[
            grid_y,
            grid_x
        ] = np.array(
            [
                x1,
                y1,
                x2,
                y2
            ],
            dtype=np.float32
        )

    return (
        class_target,
        bbox_target
    )


# ============================================================
# DATA GENERATOR
# ============================================================

def generator(
    image_files,
    labels_directory
):

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

        try:

            image, boxes = (
                load_image_and_boxes(
                    image_path,
                    label_path
                )
            )

            class_target, bbox_target = (
                create_targets(boxes)
            )

            yield (
                image,
                (
                    class_target,
                    bbox_target
                )
            )

        except Exception as error:

            print(
                f"\nSkipping image: "
                f"{image_path}"
            )

            print(
                f"Reason: {error}"
            )


# ============================================================
# TF DATASET
# ============================================================

OUTPUT_SIGNATURE = (

    tf.TensorSpec(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ),
        dtype=tf.float32
    ),

    (

        tf.TensorSpec(
            shape=(
                GRID_SIZE,
                GRID_SIZE
            ),
            dtype=tf.int32
        ),

        tf.TensorSpec(
            shape=(
                GRID_SIZE,
                GRID_SIZE,
                4
            ),
            dtype=tf.float32
        )

    )
)


train_dataset = tf.data.Dataset.from_generator(
    lambda: generator(
        train_image_files,
        TRAIN_LABELS_DIR
    ),
    output_signature=OUTPUT_SIGNATURE
)


valid_dataset = tf.data.Dataset.from_generator(
    lambda: generator(
        valid_image_files,
        VALID_LABELS_DIR
    ),
    output_signature=OUTPUT_SIGNATURE
)


# ============================================================
# SHUFFLE
# ============================================================

train_dataset = train_dataset.shuffle(
    buffer_size=len(train_image_files),
    reshuffle_each_iteration=True
)


# ============================================================
# BATCH
# ============================================================

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

valid_dataset = valid_dataset.batch(
    BATCH_SIZE
)


# ============================================================
# PREFETCH
# ============================================================

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

valid_dataset = valid_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating TensorFlow model...")

model = create_model()

print(
    "\nModel created successfully."
)

model.summary()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)


# ============================================================
# CLASSIFICATION LOSS
# ============================================================

classification_loss_fn = (
    tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )
)


# ============================================================
# BOUNDING BOX LOSS
# ============================================================

def calculate_bbox_loss(
    bbox_targets,
    bbox_predictions,
    class_targets
):

    # --------------------------------------------------------
    # Huber loss per coordinate
    #
    # Result:
    #
    # [batch, 20, 20, 4]
    # --------------------------------------------------------

    bbox_loss = tf.keras.losses.huber(
        bbox_targets,
        bbox_predictions
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Keras Huber already reduces the final coordinate
    # dimension.
    #
    # Therefore:
    #
    # bbox_loss shape =
    #
    # [batch, 20, 20]
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Object mask
    #
    # [batch, 20, 20]
    # --------------------------------------------------------

    object_mask = tf.cast(
        class_targets == 1,
        tf.float32
    )

    # --------------------------------------------------------
    # Mask bbox loss
    # --------------------------------------------------------

    masked_loss = (
        bbox_loss *
        object_mask
    )

    # --------------------------------------------------------
    # Count objects
    # --------------------------------------------------------

    object_count = tf.reduce_sum(
        object_mask
    )

    # --------------------------------------------------------
    # Average only over cells containing potholes
    # --------------------------------------------------------

    final_bbox_loss = (
        tf.reduce_sum(
            masked_loss
        )
        /
        tf.maximum(
            object_count,
            1.0
        )
    )

    return final_bbox_loss


# ============================================================
# TRAINING STEP
# ============================================================

@tf.function
def train_step(
    images,
    class_targets,
    bbox_targets
):

    with tf.GradientTape() as tape:

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        predictions = model(
            images,
            training=True
        )

        # ----------------------------------------------------
        # Named outputs
        # ----------------------------------------------------

        class_predictions = (
            predictions[
                "classification"
            ]
        )

        bbox_predictions = (
            predictions[
                "bounding_boxes"
            ]
        )

        # ----------------------------------------------------
        # Classification loss
        # ----------------------------------------------------

        class_loss = (
            classification_loss_fn(
                class_targets,
                class_predictions
            )
        )

        # ----------------------------------------------------
        # Bounding-box loss
        # ----------------------------------------------------

        bbox_loss = calculate_bbox_loss(
            bbox_targets,
            bbox_predictions,
            class_targets
        )

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        total_loss = (
            class_loss +
            bbox_loss
        )

    # --------------------------------------------------------
    # Gradients
    # --------------------------------------------------------

    gradients = tape.gradient(
        total_loss,
        model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(
            gradients,
            model.trainable_variables
        )
    )

    return (
        total_loss,
        class_loss,
        bbox_loss
    )


# ============================================================
# VALIDATION STEP
# ============================================================

@tf.function
def validation_step(
    images,
    class_targets,
    bbox_targets
):

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    predictions = model(
        images,
        training=False
    )

    # --------------------------------------------------------
    # Named outputs
    # --------------------------------------------------------

    class_predictions = (
        predictions[
            "classification"
        ]
    )

    bbox_predictions = (
        predictions[
            "bounding_boxes"
        ]
    )

    # --------------------------------------------------------
    # Classification loss
    # --------------------------------------------------------

    class_loss = (
        classification_loss_fn(
            class_targets,
            class_predictions
        )
    )

    # --------------------------------------------------------
    # Bounding box loss
    # --------------------------------------------------------

    bbox_loss = calculate_bbox_loss(
        bbox_targets,
        bbox_predictions,
        class_targets
    )

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    total_loss = (
        class_loss +
        bbox_loss
    )

    return (
        total_loss,
        class_loss,
        bbox_loss
    )


# ============================================================
# HISTORY
# ============================================================

history = {
    "train_loss": [],
    "train_class_loss": [],
    "train_bbox_loss": [],
    "val_loss": [],
    "val_class_loss": [],
    "val_bbox_loss": []
}


best_val_loss = float(
    "inf"
)


# ============================================================
# START TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(
    EPOCHS
):

    epoch_start = time.time()

    print(
        f"\nEpoch "
        f"{epoch + 1}/{EPOCHS}"
    )

    print(
        "-" * 70
    )

    # ========================================================
    # TRAIN METRICS
    # ========================================================

    train_loss_metric = (
        tf.keras.metrics.Mean()
    )

    train_class_metric = (
        tf.keras.metrics.Mean()
    )

    train_bbox_metric = (
        tf.keras.metrics.Mean()
    )

    train_batches = 0

    # ========================================================
    # TRAIN LOOP
    # ========================================================

    for images, targets in train_dataset:

        class_targets, bbox_targets = (
            targets
        )

        (
            total_loss,
            class_loss,
            bbox_loss
        ) = train_step(
            images,
            class_targets,
            bbox_targets
        )

        train_loss_metric.update_state(
            total_loss
        )

        train_class_metric.update_state(
            class_loss
        )

        train_bbox_metric.update_state(
            bbox_loss
        )

        train_batches += 1

        if train_batches % 50 == 0:

            print(
                f"  Train batch "
                f"{train_batches} | "
                f"Loss: "
                f"{train_loss_metric.result():.4f}"
            )

    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    val_loss_metric = (
        tf.keras.metrics.Mean()
    )

    val_class_metric = (
        tf.keras.metrics.Mean()
    )

    val_bbox_metric = (
        tf.keras.metrics.Mean()
    )

    val_batches = 0

    # ========================================================
    # VALIDATION LOOP
    # ========================================================

    for images, targets in valid_dataset:

        class_targets, bbox_targets = (
            targets
        )

        (
            total_loss,
            class_loss,
            bbox_loss
        ) = validation_step(
            images,
            class_targets,
            bbox_targets
        )

        val_loss_metric.update_state(
            total_loss
        )

        val_class_metric.update_state(
            class_loss
        )

        val_bbox_metric.update_state(
            bbox_loss
        )

        val_batches += 1

    # ========================================================
    # RESULTS
    # ========================================================

    train_loss = float(
        train_loss_metric.result()
    )

    train_class_loss = float(
        train_class_metric.result()
    )

    train_bbox_loss = float(
        train_bbox_metric.result()
    )

    val_loss = float(
        val_loss_metric.result()
    )

    val_class_loss = float(
        val_class_metric.result()
    )

    val_bbox_loss = float(
        val_bbox_metric.result()
    )

    epoch_time = (
        time.time() -
        epoch_start
    )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history[
        "train_loss"
    ].append(
        train_loss
    )

    history[
        "train_class_loss"
    ].append(
        train_class_loss
    )

    history[
        "train_bbox_loss"
    ].append(
        train_bbox_loss
    )

    history[
        "val_loss"
    ].append(
        val_loss
    )

    history[
        "val_class_loss"
    ].append(
        val_class_loss
    )

    history[
        "val_bbox_loss"
    ].append(
        val_bbox_loss
    )

    # ========================================================
    # PRINT
    # ========================================================

    print("\nResults:")

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Class Loss: "
        f"{train_class_loss:.4f}"
    )

    print(
        f"Train BBox Loss: "
        f"{train_bbox_loss:.4f}"
    )

    print(
        f"Val Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Class Loss: "
        f"{val_class_loss:.4f}"
    )

    print(
        f"Val BBox Loss: "
        f"{val_bbox_loss:.4f}"
    )

    print(
        f"Epoch Time: "
        f"{epoch_time:.2f} seconds"
    )

    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    checkpoint_path = os.path.join(
        MODEL_DIR,
        f"checkpoint_epoch_{epoch + 1}.keras"
    )

    model.save(
        checkpoint_path
    )

    print(
        "\nCheckpoint saved:"
    )

    print(
        checkpoint_path
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_path = os.path.join(
            MODEL_DIR,
            "best_pothole_detector.keras"
        )

        model.save(
            best_model_path
        )

        print(
            "\n*** NEW BEST MODEL ***"
        )

        print(
            f"Best validation loss: "
            f"{best_val_loss:.4f}"
        )

        print(
            f"Saved to:"
        )

        print(
            best_model_path
        )


# ============================================================
# FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "final_pothole_detector.keras"
)

model.save(
    final_model_path
)

print(
    "\nFinal model saved:"
)

print(
    final_model_path
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_csv_path = os.path.join(
    REPORT_DIR,
    "tensorflow_training_history.csv"
)


with open(
    history_csv_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "epoch,"
        "train_loss,"
        "train_class_loss,"
        "train_bbox_loss,"
        "val_loss,"
        "val_class_loss,"
        "val_bbox_loss\n"
    )

    for index in range(
        EPOCHS
    ):

        file.write(
            f"{index + 1},"
            f"{history['train_loss'][index]:.6f},"
            f"{history['train_class_loss'][index]:.6f},"
            f"{history['train_bbox_loss'][index]:.6f},"
            f"{history['val_loss'][index]:.6f},"
            f"{history['val_class_loss'][index]:.6f},"
            f"{history['val_bbox_loss'][index]:.6f}\n"
        )


print(
    "\nTraining history saved:"
)

print(
    history_csv_path
)


# ============================================================
# TRAINING GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

epochs_range = range(
    1,
    EPOCHS + 1
)

plt.plot(
    epochs_range,
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    epochs_range,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "TensorFlow Pothole Detector Training Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


loss_graph_path = os.path.join(
    GRAPH_DIR,
    "tensorflow_training_loss.png"
)

plt.savefig(
    loss_graph_path,
    dpi=150
)

plt.close()


print(
    "\nTraining graph saved:"
)

print(
    loss_graph_path
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TRAINING COMPLETE"
)

print(
    "=" * 70
)

print(
    f"\nBest validation loss: "
    f"{best_val_loss:.6f}"
)

print(
    "\nBest model:"
)

print(
    os.path.join(
        MODEL_DIR,
        "best_pothole_detector.keras"
    )
)

print(
    "\nFinal model:"
)

print(
    final_model_path
)

print(
    "\nTraining history:"
)

print(
    history_csv_path
)

print(
    "\nTraining graph:"
)

print(
    loss_graph_path
)

print(
    "\nTensorFlow training finished successfully."
)