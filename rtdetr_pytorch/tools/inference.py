"""by lyuwenyu
"""

import os 
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import numpy as np 

from src.core import YAMLConfig

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
from pathlib import Path


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
        raise AttributeError('only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            print(self.postprocessor.deploy_mode)
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)
    

    model = Model()

    image_paths = json.loads(args.image_paths)
    for image_path in image_paths:
        # Load the original image without resizing
        original_im = Image.open(image_path).convert('RGB')
        original_size = original_im.size
        print(original_size)

        scale_x = 640 / original_size[0]
        scale_y = 640 / original_size[1]

        # Resize the image for model input
        im = original_im.resize((640, 640))
        im_data = ToTensor()(im)[None]
        print(im_data.shape)

        #no grad
        with torch.no_grad():
            outputs = model(im_data, torch.tensor([[640, 640]]))
            image_path_stem = Path(image_path).stem
            image_path_parent = Path(image_path).parent
            image_path_output = image_path_parent / f"{image_path_stem}.json"
            labels, boxes, scores = outputs

            boxes[0][:, 0] /= scale_x
            boxes[0][:, 1] /= scale_y
            boxes[0][:, 2] /= scale_x
            boxes[0][:, 3] /= scale_y

            # Save the result to a json file
            with open(image_path_output, 'w') as f:
                json.dump(
                    {
                        "scores": scores[0].cpu().numpy().tolist(),
                        "labels": labels[0].cpu().numpy().tolist(),
                        "bboxes": boxes[0].cpu().numpy().tolist(),
                    }, f)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--image_paths', '-i', type=str, )

    args = parser.parse_args()

    main(args)
