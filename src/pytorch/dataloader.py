from pathlib import Path

import torch

from torch.utils.data import DataLoader

from dataset import PotholeDataset


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 4

IMAGE_SIZE = 640

NUM_WORKERS = 0


# ============================================================
# CUSTOM COLLATE FUNCTION
# ============================================================

def detection_collate_fn(batch):
    """
    Custom collate function for object detection.

    Each image can contain a different number
    of bounding boxes.

    Example:

        Image 1 → 2 boxes
        Image 2 → 5 boxes
        Image 3 → 1 box

    Therefore, targets cannot be stacked
    into one tensor.

    We keep images as a list and targets
    as a list.
    """

    images = []

    targets = []

    for image, target in batch:

        images.append(image)

        targets.append(target)

    return images, targets


# ============================================================
# CREATE DATASET
# ============================================================

def create_train_dataset():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    images_dir = (
        project_root
        / "data"
        / "raw"
        / "potholes"
        / "train"
        / "images"
    )

    labels_dir = (
        project_root
        / "data"
        / "raw"
        / "potholes"
        / "train"
        / "labels"
    )

    dataset = PotholeDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        image_size=IMAGE_SIZE,
    )

    return dataset


# ============================================================
# CREATE DATALOADER
# ============================================================

def create_train_dataloader():

    dataset = create_train_dataset()

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=detection_collate_fn,
    )

    return dataloader


# ============================================================
# TEST DATALOADER
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PYTORCH DATALOADER TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Create DataLoader
    # --------------------------------------------------------

    dataloader = create_train_dataloader()

    print()
    print(
        f"Number of batches: "
        f"{len(dataloader)}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    # --------------------------------------------------------
    # Get first batch
    # --------------------------------------------------------

    images, targets = next(
        iter(dataloader)
    )

    print()
    print("=" * 70)
    print("FIRST BATCH")
    print("=" * 70)

    print()
    print(
        f"Number of images: "
        f"{len(images)}"
    )

    print(
        f"Number of targets: "
        f"{len(targets)}"
    )

    # --------------------------------------------------------
    # Inspect each image
    # --------------------------------------------------------

    for index, (
        image,
        target,
    ) in enumerate(
        zip(images, targets)
    ):

        print()
        print(
            f"Image {index + 1}"
        )

        print(
            f"  Tensor shape : "
            f"{image.shape}"
        )

        print(
            f"  Tensor dtype : "
            f"{image.dtype}"
        )

        print(
            f"  Boxes        : "
            f"{target['boxes'].shape}"
        )

        print(
            f"  Labels       : "
            f"{target['labels'].shape}"
        )

        print(
            f"  Image ID     : "
            f"{target['image_id'].item()}"
        )

    # --------------------------------------------------------
    # Verify values
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BATCH VALIDATION")
    print("=" * 70)

    all_valid = True

    for image in images:

        if image.shape != (
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):

            all_valid = False

        if image.dtype != torch.float32:

            all_valid = False

        if image.min() < 0:

            all_valid = False

        if image.max() > 1:

            all_valid = False

    if all_valid:

        print(
            "All image tensors are valid."
        )

    else:

        print(
            "WARNING: Tensor validation failed."
        )

    print()
    print("=" * 70)
    print("DATALOADER TEST COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":

    main()