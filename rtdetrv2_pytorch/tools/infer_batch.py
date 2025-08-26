"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import os
import sys
import json

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
from PIL import Image, ImageDraw

from src.core import YAMLConfig


def convert_to_coco_format(results):
    images = []
    annotations = []
    categories = []

    # Assuming categories are from 1 to 91 like in COCO
    for i in range(1, 92):
        categories.append({'id': i, 'name': str(i), 'supercategory': 'object'})

    ann_id = 1
    for result in results:
        images.append({
            "id": result['image_id'],
            "width": result['width'],
            "height": result['height'],
            "file_name": os.path.basename(result['file_name']),
        })

        labels = result['labels'].cpu().detach().numpy().flatten()
        boxes = result['boxes'].cpu().detach().numpy().reshape(-1, 4)
        scores = result['scores'].cpu().detach().numpy().flatten()

        for i in range(len(labels)):
            x1, y1, x2, y2 = boxes[i]
            width = x2 - x1
            height = y2 - y1

            annotations.append({
                "id": ann_id,
                "image_id": result['image_id'],
                "category_id": int(labels[i]),
                "bbox": [float(x1), float(y1), float(width), float(height)],
                "area": float(width * height),
                "score": float(scores[i]),
                "iscrowd": 0,
            })
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = [os.path.join(args.im_dir, f) for f in os.listdir(args.im_dir) if os.path.splitext(f)[1].lower() in image_extensions]

    results = []
    for i, im_file in enumerate(image_files):
        im_pil = Image.open(im_file).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None].to(args.device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        results.append({
            'file_name': im_file,
            'image_id': i,
            'height': h,
            'width': w,
            'labels': labels,
            'boxes': boxes,
            'scores': scores
        })

    # draw([im_pil], labels, boxes, scores)

    coco_results = convert_to_coco_format(results)
    with open(args.output_file, 'w') as f:
        json.dump(coco_results, f, indent=4)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('--im-dir', type=str, help='path to image directory')
    parser.add_argument('--output-file', type=str, default='coco_annotations.json', help='path to output COCO annotations file')
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
