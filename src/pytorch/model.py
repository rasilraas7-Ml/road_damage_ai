import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 2

# Class 0 = background
# Class 1 = pothole

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    print()
    print("=" * 70)
    print("CREATING FASTER R-CNN MODEL")
    print("=" * 70)

    print()
    print(f"Number of classes: {NUM_CLASSES}")
    print("Class 0: background")
    print("Class 1: pothole")

    print()
    print(f"Device: {DEVICE}")

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=NUM_CLASSES,
    )

    model = model.to(DEVICE)

    print()
    print("Model created successfully.")

    return model


# ============================================================
# MODEL INFORMATION
# ============================================================

def print_model_information(model):

    print()
    print("=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        f"Total parameters     : {total_parameters:,}"
    )

    print(
        f"Trainable parameters : {trainable_parameters:,}"
    )

    print(
        f"Device               : {DEVICE}"
    )


# ============================================================
# TEST MODEL
# ============================================================

def test_model():

    print()
    print("=" * 70)
    print("TESTING MODEL")
    print("=" * 70)

    model = create_model()

    print_model_information(model)

    model.eval()

    # Test image
    test_image = torch.rand(
        3,
        640,
        640,
        dtype=torch.float32,
        device=DEVICE,
    )

    print()
    print(
        f"Test image shape: {test_image.shape}"
    )

    print()
    print("Running forward pass...")

    with torch.no_grad():

        predictions = model(
            [test_image]
        )

    prediction = predictions[0]

    print()
    print("Forward pass successful.")

    print()
    print("=" * 70)
    print("MODEL OUTPUT")
    print("=" * 70)

    print()
    print(
        f"Boxes       : {prediction['boxes'].shape}"
    )

    print(
        f"Labels      : {prediction['labels'].shape}"
    )

    print(
        f"Scores      : {prediction['scores'].shape}"
    )

    print()
    print(
        "The model can accept an image "
        "and produce detection outputs."
    )

    print()
    print("=" * 70)
    print("MODEL TEST COMPLETE")
    print("=" * 70)
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    test_model()