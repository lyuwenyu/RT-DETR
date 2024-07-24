"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from sympy import im
import torch
import torchvision
import torchvision.transforms.functional as TVF 

import os
from PIL import Image
from typing import Optional, Callable

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register

@register()
class VOCDetection(torchvision.datasets.VOCDetection, DetDataset):
    __inject__ = ['transforms', ]

    def __init__(self, root: str, ann_file: str = "trainval.txt", label_file: str = "label_list.txt", transforms: Optional[Callable] = None):

        with open(os.path.join(root, ann_file), 'r') as f:
            lines = [x.strip() for x in f.readlines()]
            lines = [x.split(' ') for x in lines]

        self.images = [os.path.join(root, lin[0]) for lin in lines]
        self.targets = [os.path.join(root, lin[1]) for lin in lines]
        assert len(self.images) == len(self.targets)

        with open(os.path.join(root + label_file), 'r') as f:
            labels = f.readlines()
            labels = [lab.strip() for lab in labels]

        self.transforms = transforms
        self.labels_map = {lab: i for i, lab in enumerate(labels)}
        
    def __getitem__(self, index: int):
        image, target = self.load_item(index)
        if self.transforms is not None:
            image, target, _ = self.transforms(image, target, self)        
        # target["orig_size"] = torch.tensor(TVF.get_image_size(image))
        return image, target

    def load_item(self, index: int):
        image = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
        
        output = {}
        output["image_id"] = torch.tensor([index])
        for k in ['area', 'boxes', 'labels', 'iscrowd']:
            output[k] = []
            
        for blob in target['annotation']['object']:
            box = [float(v) for v in blob['bndbox'].values()]
            output["boxes"].append(box)
            output["labels"].append(blob['name'])
            output["area"].append((box[2] - box[0]) * (box[3] - box[1]))
            output["iscrowd"].append(0)

        w, h = image.size
        boxes = torch.tensor(output["boxes"]) if len(output["boxes"]) > 0 else torch.zeros(0, 4)
        output['boxes'] = convert_to_tv_tensor(boxes, 'boxes', box_format='xyxy', spatial_size=[h, w])
        output['labels'] = torch.tensor([self.labels_map[lab] for lab in output["labels"]])
        output['area'] = torch.tensor(output['area'])
        output["iscrowd"] = torch.tensor(output["iscrowd"])
        output["orig_size"] = torch.tensor([w, h])
        
        return image, output
    
