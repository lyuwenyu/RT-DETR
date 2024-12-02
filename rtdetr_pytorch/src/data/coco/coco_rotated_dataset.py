from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import src
import torch.utils.data
import torchvision
from PIL import Image


@dataclass
class BoundingBox:
    category: str
    x1: float
    x2: float
    y1: float
    y2: float
    rotation: float = 0

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def x_center(self):
        return self.x1 + self.width / 2

    @property
    def y_center(self):
        return self.y1 + self.height / 2

    @property
    def area(self):
        return self.width * self.height


class RotatedCocoDataset(torch.utils.data.Dataset):
    def __init__(self, image_files: List[Path], labels: List[List[BoundingBox]]):
        self.image_files = image_files
        self.labels = labels

        if len(self.image_files) != len(self.labels):
            raise ValueError(
                "Mismatch in the number of images and labels. Please check the dataset directories to ensure each image has a corresponding label file."
            )
        self.filter_unlabeled_images()
        self.create_classname_index()
        self.input_size = [640, 640]  # Required by rtdetr
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.input_size),
                src.data.transforms.ToImage(),
                src.data.transforms.ToDtype(),
            ]
        )

    @property
    def num_classes(self) -> int:
        return len(self.label_indices)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        labels = self.labels[idx]
        image = Image.open(self.image_files[idx]).convert("RGB")
        image_tensor = self.transforms(image)

        targets_dict: Dict = {
            "boxes": [],
            "labels": [],
            "image_id": [idx],
            "area": [],
            "iscrowd": [],
            "orig_size": image.size,
            "size": self.input_size,
            "angles": [],
        }

        for label in labels:
            targets_dict["boxes"].append(
                [label.x_center, label.y_center, label.width, label.height]
            )
            targets_dict["labels"].append(self.label_indices[label.category])
            targets_dict["area"].append(label.area)
            targets_dict["iscrowd"].append(0)
            targets_dict["angles"].append(normalize_angle(label.rotation))

        targets_dict = {k: torch.tensor(np.array(v)) for k, v in targets_dict.items()}
        targets_dict["boxes"] = targets_dict["boxes"].float()
        targets_dict["area"] = targets_dict["area"].float()
        targets_dict["angles"] = targets_dict["angles"].float()
        targets_dict["labels"] = targets_dict["labels"].long()

        # ensure every tensor of boxes have the correct shape [n,4], even empty ones [0,4]
        if targets_dict["boxes"].shape.__len__() != 2:
            targets_dict["boxes"] = targets_dict["boxes"].reshape([-1, 4])

        # box coordinates are fractions of the image width and height
        targets_dict["boxes"][:, 0] /= image.size[0]
        targets_dict["boxes"][:, 2] /= image.size[0]
        targets_dict["boxes"][:, 1] /= image.size[1]
        targets_dict["boxes"][:, 3] /= image.size[1]
        targets_dict["area"] /= image.size[0] * image.size[1]

        return image_tensor, targets_dict

    def filter_unlabeled_images(self):
        labeled_image_files = []
        nonempty_label_files = []

        for image, image_annotations in zip(self.image_files, self.labels):
            if image_annotations:
                labeled_image_files.append(image)
                nonempty_label_files.append(image_annotations)

        return labeled_image_files, nonempty_label_files

    def create_classname_index(self):
        self.label_indices: Dict[str, int] = {}
        for label in self.labels:
            for object in label:
                if object.category not in self.label_indices:
                    self.label_indices[object.category] = len(self.label_indices)


def normalize_angle(angle):
    """
    Normalize an angle to be within -1 (-90 degrees) and 1 (90 degrees).
    e.g. [0, 350, 20, 80, 180, 160] -> [  0 -10  20  80   0 -20]

    Parameters:
    angle (float): The angle in degrees [0 to 360).

    Returns:
    float: The normalized angle between -1 and 1 degrees.
    """
    # Convert angle to the range -90 to 90 degrees
    angle = ((angle + 90) % 180) - 90

    angle /= 90

    return angle
