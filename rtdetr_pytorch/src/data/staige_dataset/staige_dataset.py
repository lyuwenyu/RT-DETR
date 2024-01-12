import csv
import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import datapoints
from torchvision.io import read_image

from src.core import register
from ..coco import mscoco_category2name

__all__ = ['StaigeDataset']


mscoco_name2category = {v: k for k, v in mscoco_category2name.items()}
staige_labels2coco_name = {
    "ball": "sports ball",
    "goalkeeper": "person",
    "player": "person",
    "referee": "person",
    "horse": "horse",
    "mounted_horse": "horse",
    "vehicle": "car",
    # TODO: hurdle
}
staige_labels2coco_label = {k: mscoco_name2category[v] for k, v in staige_labels2coco_name.items()}


@register
class StaigeDataset(Dataset):
    __inject__ = ['transforms']

    def __init__(self, annotations_file, img_dir, transforms, classes_file=None):
        self.annotations = pd.read_csv(annotations_file, header=None)
        if classes_file is not None:
            with open(classes_file, mode='r') as f:
                reader = csv.reader(f)
                self.classes = {rows[0]:rows[1] for rows in reader}
        else:
            self.classes = staige_labels2coco_label
        self.img_paths = self.annotations.iloc[:, 0].unique()
        self.img_dir = img_dir
        self._transforms = transforms

    def __getitem__(self, idx):
        relative_img_path = self.img_paths[idx]
        img_path = os.path.join(self.img_dir, relative_img_path)
        image = read_image(img_path)

        image_annotations = self.annotations[self.annotations.iloc[:, 0] == relative_img_path]
        bboxes = []
        labels = []
        areas = []
        iscrowds = []
        for row in image_annotations.iterrows():
            _, x1, y1, x2, y2, class_name = row[1]
            bboxes.append([float(x1), float(y1), float(x2), float(y2)])
            labels.append(int(self.classes[class_name]))
            areas.append((x2-x1)*(y2-y1))
            iscrowds.append(0)  # TODO: Add occluded property?
        bboxes_tensor = datapoints.BoundingBox(
            bboxes,
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=image.shape[1:],  # h w
        )
        labels_tensor = torch.tensor(labels)
        areas_tensor = torch.tensor(areas)
        iscrowds_tensor = torch.tensor(iscrowds)
        image_id = torch.tensor([idx])
        orig_size = torch.tensor([image.shape[2], image.shape[1]])
        size = torch.tensor([image.shape[2], image.shape[1]])
        target = {
            "boxes": bboxes_tensor,
            "labels": labels_tensor,
            "image_id": image_id,
            "area": areas_tensor,
            "iscrowd": iscrowds_tensor,
            "orig_size": orig_size,
            "size": size,
        }

        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.img_paths)
