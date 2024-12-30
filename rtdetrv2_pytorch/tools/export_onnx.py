"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn 
import datetime
from src.core import YAMLConfig
import json
import onnx
import onnxsim

def add_meta(onnx_model, key, value):
    # Add meta to model
    meta = onnx_model.metadata_props.add()
    meta.key = key
    meta.value = value

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

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()

    data = torch.rand(1, 3, 1280, 1280)
    size = torch.tensor([[1280, 1280]])
    _ = model(data, size)

    dynamic_axes = {
        'images': {0: 'N', },
        'orig_target_sizes': {0: 'N'}
    }

    torch.onnx.export(
        model, 
        (data, size), 
        args.output_file,
        input_names=['images', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=dynamic_axes,
        opset_version=16, 
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        onnx_model = onnx.load(args.output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')
    
    add_meta(onnx_model, 
             key="data", 
             value=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    add_meta(onnx_model, 
             key="classes", 
             value=json.dumps(args.class_names))
    add_meta(onnx_model, 
             key="model", 
             value="RT-DETR")
    onnx.save(onnx_model, args.output_file)

    if args.simplify:
        onnx_model_simplify, check = onnxsim.simplify(args.output_file)
        onnx.save(onnx_model_simplify, args.output_file)
        print(f'Successfully simplified onnx model: {check}...')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--output_file', '-o', type=str, default='model.onnx')
    parser.add_argument('--class_names', nargs='+', default=['marine_mammal', 'marker', 'unknown', 'vessel'], help='class list in the same order as class enum')
    parser.add_argument('--check',  action='store_true', default=False,)
    parser.add_argument('--simplify',  action='store_true', default=False,)
    args = parser.parse_args()

    main(args)
