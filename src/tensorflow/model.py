import os
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 640

NUM_CLASSES = 1

BACKBONE = "MobileNetV2"


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    print("\n" + "=" * 60)
    print("CREATING TENSORFLOW POTHOLE DETECTION MODEL")
    print("=" * 60)

    print(
        f"Image size: "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Classes: {NUM_CLASSES}"
    )

    print(
        f"Backbone: {BACKBONE}"
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ),
        name="image"
    )

    # --------------------------------------------------------
    # Backbone
    # --------------------------------------------------------

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ),
        include_top=False,
        weights="imagenet"
    )

    # Freeze backbone initially
    backbone.trainable = False

    features = backbone(
        inputs,
        training=False
    )

    # --------------------------------------------------------
    # Feature processing
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        256,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(features)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    classification = tf.keras.layers.Conv2D(
        NUM_CLASSES + 1,
        kernel_size=1,
        padding="same",
        name="classification"
    )(x)

    # --------------------------------------------------------
    # Bounding box regression head
    # --------------------------------------------------------

    bounding_boxes = tf.keras.layers.Conv2D(
        4,
        kernel_size=1,
        padding="same",
        name="bounding_boxes"
    )(x)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "classification": classification,
            "bounding_boxes": bounding_boxes
        },
        name="tensorflow_pothole_detector"
    )

    print(
        "\nModel created successfully."
    )

    print(
        f"Total parameters: "
        f"{model.count_params():,}"
    )

    return model


# ============================================================
# MODEL SUMMARY
# ============================================================

def print_model_summary():

    model = create_model()

    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    model.summary()


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    print_model_summary()

    print("\n" + "=" * 60)
    print("TESTING FORWARD PASS")
    print("=" * 60)

    # Create dummy image
    dummy_image = tf.random.uniform(
        shape=(
            1,
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ),
        minval=0,
        maxval=1,
        dtype=tf.float32
    )

    model = create_model()

    outputs = model(
        dummy_image,
        training=False
    )

    print(
        "\nInput shape:"
    )

    print(
        dummy_image.shape
    )

    print(
        "\nClassification output shape:"
    )

    print(
        outputs["classification"].shape
    )

    print(
        "\nBounding box output shape:"
    )

    print(
        outputs["bounding_boxes"].shape
    )

    print(
        "\nTensorFlow model forward pass "
        "successful."
    )