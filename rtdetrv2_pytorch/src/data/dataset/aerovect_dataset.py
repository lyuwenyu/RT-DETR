import torch
import torch.utils.data
import torchvision
import torchvision.transforms.v2 as T

import os
import glob
import json
from PIL import Image
from typing import Optional, Callable

from functools import reduce

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register

__all__ = ["AerovectDetection"]

# Mappings to convert Aerovect class ids to COCO class ids since we're using a RT-DETR model that was trained on COCO but then evaluation on Aerovect dataset.

# Aircraft
labels_map_aircraft = {
    1: 4,  # "Airplane",
}

# Car
labels_map_car = {
    25: 2,  # "Passenger Car",
}

# Car
labels_map_car_extended = {
    25: 2,  # "Passenger Car",
    31: 2,  # "SUV",
    38: 2,  # "Van"
}

# Pedestrian
labels_map_pedestrian = {
    27: 0,  # "Pedestrian",
}

# Bus but there's no bus in the aerovect GT
labels_map_bus = {
    7: 5,  # "Bus",
    29: 5,  # "Push-back Tractor",
}

# Truck - miniset (Exclude tractors, dollys, utility vehicles, carts, stairs, van, suv)
labels_map_truck_miniset = {
    2: 7,  # "Ambulance",
    6: 7,  # "Bobtail Truck",
    9: 7,  # "Catering Truck",
    12: 7,  # "Deicing Truck",
    15: 7,  # "Dump Truck",
    18: 7,  # "Fire Truck",
    20: 7,  # "Fuel Truck Large",
    21: 7,  # "Fuel Truck Small",
    23: 7,  # "Lavatory Truck",
    28: 7,  # "Pickup Truck",
}

# Truck easyset (removed some classes from superset that is hard/confusing the model, e.g. Empty Dolly, Dolly with Pallet)
labels_map_truck_easyset = {
    2: 7,  # "Ambulance",
    3: 7,  # "Baggage Cart",
    6: 7,  # "Bobtail Truck",
    9: 7,  # "Catering Truck",
    11: 7,  # "Conventional Baggage Tractor",
    12: 7,  # "Deicing Truck",
    14: 7,  # "Dolly with ULD",
    15: 7,  # "Dump Truck",
    16: 7,  # "Electric Baggage Tractor",
    18: 7,  # "Fire Truck",
    20: 7,  # "Fuel Truck Large",
    21: 7,  # "Fuel Truck Small",
    23: 7,  # "Lavatory Truck",
    28: 7,  # "Pickup Truck",
    29: 7,  # "Push-back Tractor",
    31: 7,  # "SUV",
    36: 7,  # "ULD",
    37: 7,  # "Utility Vehicle",
    38: 7,  # "Van"
}

# Truck - superset (include any vehicle that's not a car, bus, or aircraft)
labels_map_truck_superset = {
    2: 7,  # "Ambulance",
    3: 7,  # "Baggage Cart",
    6: 7,  # "Bobtail Truck",
    8: 7,  # "Cargo Loader",
    9: 7,  # "Catering Truck",
    11: 7,  # "Conventional Baggage Tractor",
    12: 7,  # "Deicing Truck",
    13: 7,  # "Dolly with Pallet",
    14: 7,  # "Dolly with ULD",
    15: 7,  # "Dump Truck",
    16: 7,  # "Electric Baggage Tractor",
    17: 7,  # "Empty Dolly",
    18: 7,  # "Fire Truck",
    19: 7,  # "Forklift",
    20: 7,  # "Fuel Truck Large",
    21: 7,  # "Fuel Truck Small",
    23: 7,  # "Lavatory Truck",
    28: 7,  # "Pickup Truck",
    30: 5,  # "Push-back tractor-towbarless",
    31: 7,  # "SUV",
    32: 7,  # "Stairs - motorized",
    35: 7,  # "Transporter",
    36: 7,  # "ULD",
    37: 7,  # "Utility Vehicle",
    38: 7,  # "Van"
}

# TOGGLE to enable by category set
labels_maps_list = [
    labels_map_aircraft,
    labels_map_car,
    # labels_map_car_extended,
    labels_map_pedestrian,
    # labels_map_bus,
    # labels_map_truck_miniset,
    labels_map_truck_easyset,
    # labels_map_truck_superset,
]

labels_map_unused = {
    0: -1,  # "Air Conditioning Unit",
    4: -1,  # "Barrier",
    5: -1,  # "Belt Loader",
    10: -1,  # "Cone",
    22: -1,  # "Ground Power Unit",
    24: -1,  # "Omitted",
    26: -1,  # "Passenger Ramp",
    33: -1,  # "Stairs - non motorized",
    34: -1,  # "Suitcase",
}

# Combine all label maps
labels_map = reduce(lambda a, b: {**a, **b}, labels_maps_list)


def dump_json(data, output_file) -> None:
    """Dump a dir object to a JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as out_f:
        json.dump(data, out_f, indent=4)
        out_f.write("\n")


dump_json(labels_map, "output/aerovect/labels_map.json")


@register()
class AerovectDetection(DetDataset):
    __inject__ = [
        "transforms",
    ]

    def __init__(self, image_paths, label_paths, transforms: Optional[Callable] = None):
        self.images = sorted(glob.glob(image_paths))

        self.ids = []
        self.label_files = []
        for idx, image_file in enumerate(self.images):
            label_file = image_file.replace("images", "labels").replace(".jpg", ".txt")
            assert os.path.exists(label_file), f"Label file {label_file} does not exist"

            self.ids.append(idx)
            self.label_files.append(label_file)

        self._transforms = transforms

        print(f"Aerovect dataset initialized with {len(self.images)} images")

    def _parse_label(self, label_file):
        with open(label_file, "r") as f:
            labels_text = f.readlines()

        labels = []
        for line in labels_text:
            line = line.split()
            class_id, x1, y1, x2, y2 = (
                int(line[0]),
                float(line[1]),
                float(line[2]),
                float(line[3]),
                float(line[4]),
            )

            labels.append(
                {
                    "class_id": class_id,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return labels

    def _build_target(self, idx, image, label_file):
        labels = self._parse_label(label_file)

        output = {}
        output["image_id"] = torch.tensor([idx])

        # initialize empty lists
        for k in ["area", "boxes", "labels", "iscrowd"]:
            output[k] = []

        # Iterate over labels and build target for each label
        for label in labels:
            coco_label = labels_map.get(label["class_id"])  # map to COCO classes
            if coco_label is None:
                continue  # skip if class is disabled for evaluation (commented out from labels_map)

            box = label["bbox"]
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            box_height = y2 - y1

            output["boxes"].append(box)
            output["labels"].append(coco_label)
            output["area"].append((box_width) * (box_height))
            output["iscrowd"].append(0)

        w, h = image.size
        boxes = (
            torch.tensor(output["boxes"])
            if len(output["boxes"]) > 0
            else torch.zeros(0, 4)
        )
        output["boxes"] = convert_to_tv_tensor(
            boxes, "boxes", box_format="xyxy", spatial_size=[h, w]
        )
        output["labels"] = torch.tensor(output["labels"])
        output["area"] = torch.tensor(output["area"])
        output["iscrowd"] = torch.tensor(output["iscrowd"])
        output["orig_size"] = torch.tensor([w, h])

        return output

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        target = self._build_target(idx, image, self.label_files[idx])

        return image, target
