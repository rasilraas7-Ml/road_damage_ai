import os
import csv
import time
import torch
import matplotlib.pyplot as plt

from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
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

# Dataset paths
TRAIN_IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "potholes",
    "train",
    "images"
)

TRAIN_LABEL_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "potholes",
    "train",
    "labels"
)

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


# Model output directory
MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "pytorch"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "reports"
)

GRAPH_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "graphs"
)


# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


# ============================================================
# TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 640

BATCH_SIZE = 2

NUM_EPOCHS = 10

LEARNING_RATE = 0.005

MOMENTUM = 0.9

WEIGHT_DECAY = 0.0005

STEP_SIZE = 3

GAMMA = 0.1

NUM_WORKERS = 0


# Device
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
    print("FASTER R-CNN TRAINING")
    print("=" * 70)

    print(f"Device          : {DEVICE}")
    print(f"Image size      : {IMAGE_SIZE}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Epochs          : {NUM_EPOCHS}")
    print(f"Learning rate   : {LEARNING_RATE}")
    print(f"Momentum        : {MOMENTUM}")
    print(f"Weight decay    : {WEIGHT_DECAY}")
    print(f"Step size       : {STEP_SIZE}")
    print(f"Gamma           : {GAMMA}")
    print(f"Workers         : {NUM_WORKERS}")

    print()
    print("Training images :", TRAIN_IMAGE_DIR)
    print("Validation images:", VAL_IMAGE_DIR)

    print("=" * 70)


# ============================================================
# CREATE DATASETS
# ============================================================

def create_datasets():

    print()
    print("=" * 70)
    print("CREATING DATASETS")
    print("=" * 70)

    train_dataset = PotholeDataset(
        image_dir=TRAIN_IMAGE_DIR,
        label_dir=TRAIN_LABEL_DIR,
        image_size=IMAGE_SIZE
    )

    val_dataset = PotholeDataset(
        image_dir=VAL_IMAGE_DIR,
        label_dir=VAL_LABEL_DIR,
        image_size=IMAGE_SIZE
    )

    print()
    print(f"Training samples   : {len(train_dataset)}")
    print(f"Validation samples : {len(val_dataset)}")

    return train_dataset, val_dataset


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    train_dataset,
    val_dataset
):

    print()
    print("=" * 70)
    print("CREATING DATALOADERS")
    print("=" * 70)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=detection_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=detection_collate_fn
    )

    print(
        f"Training batches   : {len(train_loader)}"
    )

    print(
        f"Validation batches : {len(val_loader)}"
    )

    return train_loader, val_loader


# ============================================================
# CREATE MODEL
# ============================================================

def create_training_model():

    print()
    print("=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    model = create_model()

    print("Faster R-CNN model created.")

    return model


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch
):

    model.train()

    total_loss = 0.0

    start_time = time.time()

    total_batches = len(data_loader)

    print()
    print(
        f"Epoch {epoch} - Training"
    )

    print("-" * 70)

    for batch_index, (
        images,
        targets
    ) in enumerate(data_loader):

        # Move images to device
        images = [
            image.to(device)
            for image in images
        ]

        # Move targets to device
        targets = [

            {
                key: value.to(device)
                for key, value in target.items()
            }

            for target in targets
        ]

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        loss_dict = model(
            images,
            targets
        )

        # ----------------------------------------------------
        # Calculate total loss
        # ----------------------------------------------------

        losses = sum(
            loss
            for loss in loss_dict.values()
        )

        loss_value = losses.item()

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        total_loss += loss_value

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index == 0
            or (batch_index + 1) % 25 == 0
            or (batch_index + 1) == total_batches
        ):

            print(
                f"Batch "
                f"{batch_index + 1:4d}/"
                f"{total_batches} "
                f"| Loss: "
                f"{loss_value:.4f}"
            )

    average_loss = (
        total_loss / total_batches
    )

    elapsed_time = time.time() - start_time

    print("-" * 70)

    print(
        f"Training Loss : {average_loss:.4f}"
    )

    print(
        f"Time          : "
        f"{elapsed_time / 60:.2f} minutes"
    )

    return average_loss


# ============================================================
# VALIDATE ONE EPOCH
# ============================================================

def validate_one_epoch(
    model,
    data_loader,
    device,
    epoch
):

    # Faster R-CNN behaves differently during
    # training and evaluation.
    #
    # For this first training stage we calculate
    # validation loss by temporarily using train mode
    # while disabling gradients.

    model.train()

    total_loss = 0.0

    start_time = time.time()

    total_batches = len(data_loader)

    print()
    print(
        f"Epoch {epoch} - Validation"
    )

    print("-" * 70)

    with torch.no_grad():

        for batch_index, (
            images,
            targets
        ) in enumerate(data_loader):

            images = [
                image.to(device)
                for image in images
            ]

            targets = [

                {
                    key: value.to(device)
                    for key, value in target.items()
                }

                for target in targets
            ]

            # Calculate validation loss
            loss_dict = model(
                images,
                targets
            )

            losses = sum(
                loss
                for loss in loss_dict.values()
            )

            loss_value = losses.item()

            total_loss += loss_value

            if (
                batch_index == 0
                or (batch_index + 1) % 25 == 0
                or (batch_index + 1) == total_batches
            ):

                print(
                    f"Batch "
                    f"{batch_index + 1:4d}/"
                    f"{total_batches} "
                    f"| Loss: "
                    f"{loss_value:.4f}"
                )

    average_loss = (
        total_loss / total_batches
    )

    elapsed_time = time.time() - start_time

    print("-" * 70)

    print(
        f"Validation Loss : "
        f"{average_loss:.4f}"
    )

    print(
        f"Time             : "
        f"{elapsed_time / 60:.2f} minutes"
    )

    return average_loss


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    train_loss,
    val_loss,
    filename
):

    checkpoint_path = os.path.join(
        MODEL_DIR,
        filename
    )

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "train_loss":
                train_loss,

            "val_loss":
                val_loss
        },
        checkpoint_path
    )

    print(
        f"Checkpoint saved: "
        f"{checkpoint_path}"
    )


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_best_model(
    model,
    epoch,
    train_loss,
    val_loss
):

    best_model_path = os.path.join(
        MODEL_DIR,
        "best_pothole_detector.pth"
    )

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "train_loss":
                train_loss,

            "val_loss":
                val_loss
        },
        best_model_path
    )

    print()
    print(
        "⭐ Best model saved:"
    )

    print(
        best_model_path
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_history(history):

    csv_path = os.path.join(
        OUTPUT_DIR,
        "training_history.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss"
            ]
        )

        for row in history:

            writer.writerow(row)

    print()
    print(
        f"Training history saved:"
    )

    print(csv_path)


# ============================================================
# PLOT LOSS
# ============================================================

def plot_loss(history):

    epochs = [
        row[0]
        for row in history
    ]

    train_losses = [
        row[1]
        for row in history
    ]

    val_losses = [
        row[2]
        for row in history
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        train_losses,
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_losses,
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Faster R-CNN Training and Validation Loss"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    graph_path = os.path.join(
        GRAPH_DIR,
        "training_loss.png"
    )

    plt.savefig(
        graph_path,
        dpi=150
    )

    plt.close()

    print()
    print(
        f"Loss graph saved:"
    )

    print(graph_path)


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train():

    print_configuration()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset, val_dataset = (
        create_datasets()
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader, val_loader = (
        create_dataloaders(
            train_dataset,
            val_dataset
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_training_model()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    scheduler = StepLR(
        optimizer,
        step_size=STEP_SIZE,
        gamma=GAMMA
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = []

    best_val_loss = float("inf")

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    print(
        "⚠️ CPU training can take a long time."
    )

    print(
        "Do not close the terminal while training."
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        epoch_start = time.time()

        print()
        print()
        print("#" * 70)

        print(
            f"EPOCH {epoch}/{NUM_EPOCHS}"
        )

        print("#" * 70)

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            DEVICE,
            epoch
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss = validate_one_epoch(
            model,
            val_loader,
            DEVICE,
            epoch
        )

        # ----------------------------------------------------
        # Learning-rate update
        # ----------------------------------------------------

        scheduler.step()

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history.append(
            [
                epoch,
                train_loss,
                val_loss
            ]
        )

        save_history(history)

        # ----------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            train_loss,
            val_loss,
            f"checkpoint_epoch_{epoch}.pth"
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_best_model(
                model,
                epoch,
                train_loss,
                val_loss
            )

        # ----------------------------------------------------
        # Epoch summary
        # ----------------------------------------------------

        epoch_time = (
            time.time()
            - epoch_start
        )

        print()
        print("=" * 70)
        print(
            f"EPOCH {epoch} COMPLETE"
        )
        print("=" * 70)

        print(
            f"Training loss   : "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation loss : "
            f"{val_loss:.4f}"
        )

        print(
            f"Learning rate   : "
            f"{current_lr:.6f}"
        )

        print(
            f"Epoch time      : "
            f"{epoch_time / 60:.2f} minutes"
        )

        print("=" * 70)

    # --------------------------------------------------------
    # Final model
    # --------------------------------------------------------

    final_model_path = os.path.join(
        MODEL_DIR,
        "final_pothole_detector.pth"
    )

    torch.save(
        model.state_dict(),
        final_model_path
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Final model:"
    )

    print(
        final_model_path
    )

    print(
        f"Best validation loss:"
        f" {best_val_loss:.4f}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_loss(history)

    print()
    print("=" * 70)
    print("ALL TRAINING OUTPUTS SAVED")
    print("=" * 70)

    print(
        f"Models  : {MODEL_DIR}"
    )

    print(
        f"Reports : {OUTPUT_DIR}"
    )

    print(
        f"Graphs  : {GRAPH_DIR}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    train()