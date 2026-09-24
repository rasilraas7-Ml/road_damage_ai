import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class PotholeDataset(Dataset):

    def __init__(self, image_dir, label_dir, image_size=640):

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size

        # Supported image extensions
        self.image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

        # Find all images
        self.image_files = sorted([
            file for file in os.listdir(self.image_dir)
            if file.lower().endswith(self.image_extensions)
        ])

        print("=" * 70)
        print("INITIALIZING POTHOLE DATASET")
        print("=" * 70)

        print(f"Image directory : {self.image_dir}")
        print(f"Label directory : {self.label_dir}")
        print(f"Image size      : {self.image_size}")
        print(f"Number of images: {len(self.image_files)}")

        print("\nClass mapping:")
        print("YOLO class 0 -> Faster R-CNN class 1")
        print("Faster R-CNN class 0 -> background")
        print("Faster R-CNN class 1 -> pothole")

        print("=" * 70)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        # --------------------------------------------------
        # 1. Get image filename
        # --------------------------------------------------

        image_filename = self.image_files[index]

        image_path = os.path.join(
            self.image_dir,
            image_filename
        )

        # --------------------------------------------------
        # 2. Load image
        # --------------------------------------------------

        image = cv2.imread(image_path)

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        # Original dimensions
        original_height, original_width = image.shape[:2]

        # --------------------------------------------------
        # 3. Convert BGR -> RGB
        # --------------------------------------------------

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # --------------------------------------------------
        # 4. Load YOLO annotation
        # --------------------------------------------------

        label_filename = os.path.splitext(
            image_filename
        )[0] + ".txt"

        label_path = os.path.join(
            self.label_dir,
            label_filename
        )

        boxes = []
        labels = []

        if os.path.exists(label_path):

            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as file:

                lines = file.readlines()

            for line in lines:

                line = line.strip()

                if not line:
                    continue

                values = line.split()

                if len(values) != 5:
                    continue

                # YOLO format:
                # class_id center_x center_y width height

                class_id = int(values[0])

                center_x = float(values[1])
                center_y = float(values[2])
                box_width = float(values[3])
                box_height = float(values[4])

                # --------------------------------------------------
                # Convert normalized YOLO coordinates to pixels
                # --------------------------------------------------

                center_x *= original_width
                center_y *= original_height

                box_width *= original_width
                box_height *= original_height

                x1 = center_x - box_width / 2
                y1 = center_y - box_height / 2

                x2 = center_x + box_width / 2
                y2 = center_y + box_height / 2

                # --------------------------------------------------
                # Keep coordinates inside image
                # --------------------------------------------------

                x1 = max(0, min(x1, original_width))
                y1 = max(0, min(y1, original_height))

                x2 = max(0, min(x2, original_width))
                y2 = max(0, min(y2, original_height))

                # Ignore invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([
                    x1,
                    y1,
                    x2,
                    y2
                ])

                # --------------------------------------------------
                # IMPORTANT:
                #
                # YOLO:
                # 0 = pothole
                #
                # Faster R-CNN:
                # 0 = background
                # 1 = pothole
                #
                # Therefore:
                # YOLO class 0 -> Faster R-CNN class 1
                # --------------------------------------------------

                labels.append(class_id + 1)

        # --------------------------------------------------
        # 5. Convert boxes to NumPy
        # --------------------------------------------------

        if len(boxes) == 0:

            boxes = np.zeros(
                (0, 4),
                dtype=np.float32
            )

        else:

            boxes = np.array(
                boxes,
                dtype=np.float32
            )

        # --------------------------------------------------
        # 6. Resize image to 640x640
        # --------------------------------------------------

        new_width = self.image_size
        new_height = self.image_size

        resized_image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR
        )

        # --------------------------------------------------
        # 7. Scale bounding boxes
        # --------------------------------------------------

        scale_x = new_width / original_width
        scale_y = new_height / original_height

        if len(boxes) > 0:

            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x

            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y

        # --------------------------------------------------
        # 8. Convert image to PyTorch tensor
        # --------------------------------------------------

        # HWC -> CHW
        image_tensor = torch.from_numpy(
            resized_image
        ).permute(2, 0, 1)

        # uint8 -> float32
        image_tensor = image_tensor.float()

        # 0-255 -> 0-1
        image_tensor = image_tensor / 255.0

        # Make sure memory is contiguous
        image_tensor = image_tensor.contiguous()

        # --------------------------------------------------
        # 9. Convert boxes to tensor
        # --------------------------------------------------

        boxes_tensor = torch.as_tensor(
            boxes,
            dtype=torch.float32
        )

        # --------------------------------------------------
        # 10. Convert labels to tensor
        # --------------------------------------------------

        labels_tensor = torch.as_tensor(
            labels,
            dtype=torch.int64
        )

        # --------------------------------------------------
        # 11. Image ID
        # --------------------------------------------------

        image_id = torch.tensor(
            [index],
            dtype=torch.int64
        )

        # --------------------------------------------------
        # 12. Create Faster R-CNN target
        # --------------------------------------------------

        target = {

            "boxes": boxes_tensor,

            "labels": labels_tensor,

            "image_id": image_id
        }

        # --------------------------------------------------
        # 13. Return
        # --------------------------------------------------

        return image_tensor, target


# ============================================================
# TEST DATASET
# ============================================================

def test_dataset():

    print("\n")
    print("=" * 70)
    print("TESTING POTHOLE DATASET")
    print("=" * 70)

    # --------------------------------------------------
    # Dataset paths
    # --------------------------------------------------

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )

    train_image_dir = os.path.join(
        project_root,
        "data",
        "raw",
        "potholes",
        "train",
        "images"
    )

    train_label_dir = os.path.join(
        project_root,
        "data",
        "raw",
        "potholes",
        "train",
        "labels"
    )

    # --------------------------------------------------
    # Create dataset
    # --------------------------------------------------

    dataset = PotholeDataset(
        image_dir=train_image_dir,
        label_dir=train_label_dir,
        image_size=640
    )

    # --------------------------------------------------
    # Dataset length
    # --------------------------------------------------

    print("\nDataset length:")
    print(len(dataset))

    # --------------------------------------------------
    # Get first image
    # --------------------------------------------------

    image, target = dataset[0]

    print("\n")
    print("=" * 70)
    print("FIRST SAMPLE")
    print("=" * 70)

    print(
        f"Image tensor shape : {image.shape}"
    )

    print(
        f"Image dtype        : {image.dtype}"
    )

    print(
        f"Image min value    : {image.min().item():.4f}"
    )

    print(
        f"Image max value    : {image.max().item():.4f}"
    )

    print(
        f"Boxes shape        : {target['boxes'].shape}"
    )

    print(
        f"Labels shape       : {target['labels'].shape}"
    )

    print(
        f"Labels             : {target['labels']}"
    )

    print(
        f"Image ID           : {target['image_id']}"
    )

    # --------------------------------------------------
    # Verify labels
    # --------------------------------------------------

    if len(target["labels"]) > 0:

        unique_labels = torch.unique(
            target["labels"]
        )

        print(
            f"\nUnique labels     : {unique_labels.tolist()}"
        )

        if torch.all(
            (unique_labels >= 1)
            & (unique_labels <= 1)
        ):

            print(
                "✅ Label mapping is correct."
            )

        else:

            print(
                "❌ WARNING: Unexpected label values."
            )

    # --------------------------------------------------
    # Final checks
    # --------------------------------------------------

    assert image.shape == (
        3,
        640,
        640
    ), "Image shape is incorrect."

    assert image.dtype == torch.float32, (
        "Image dtype is incorrect."
    )

    assert image.min() >= 0.0, (
        "Image contains values below 0."
    )

    assert image.max() <= 1.0, (
        "Image contains values above 1."
    )

    assert target["boxes"].dtype == torch.float32, (
        "Boxes dtype is incorrect."
    )

    assert target["labels"].dtype == torch.int64, (
        "Labels dtype is incorrect."
    )

    if len(target["labels"]) > 0:

        assert torch.all(
            target["labels"] == 1
        ), (
            "Pothole labels must be class 1 "
            "for Faster R-CNN."
        )

    print("\n")
    print("=" * 70)
    print("DATASET TEST COMPLETE")
    print("=" * 70)

    print("\n✅ Dataset is ready for Faster R-CNN training.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    test_dataset()