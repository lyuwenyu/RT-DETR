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
import numpy as np
from onnx import TensorProto, helper

def add_meta(onnx_model, key, value):
    # Add meta to model
    meta = onnx_model.metadata_props.add()
    meta.key = key
    meta.value = value

def main(args, ):
    """main
    """
    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    update_dict.update({k: v for k, v in args.__dict__.items() \
                        if k not in ['update', ] and v is not None})
    cfg = YAMLConfig(args.config, **update_dict)

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

    resize_h, resize_w = (args.input_size, args.input_size)
    data = torch.rand(1, args.image_channels, resize_h, resize_w)
    size = torch.tensor([[resize_h, resize_w]])
    _ = model(data, size)

    # Enable dynamic batch size
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
    classes = list(cfg.train_dataloader.dataset.category2name.values())

    onnx_model = onnx.load(args.output_file)
    add_meta(onnx_model, 
             key="date", 
             value=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    add_meta(onnx_model, 
             key="classes", 
             value=json.dumps(classes))
    add_meta(onnx_model, 
             key="model", 
             value="RT-DETR")
    add_meta(onnx_model, 
             key="input_shape", 
             value=json.dumps((resize_h, resize_w)))    
    onnx.save(onnx_model, args.output_file)

    if args.check:
        onnx_model = onnx.load(args.output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        onnx_model_simplify, check = onnxsim.simplify(args.output_file)
        onnx.save(onnx_model_simplify, args.output_file)
        print(f'Successfully simplified onnx model: {check}...')

    if args.fix_dimensions:
        constant_value = np.array([[ resize_h,resize_w ]], dtype=np.int64)
        const_input(args.output_file, "orig_target_sizes", constant_value)


def const_input(model_path, input_name, constant_value):
    model = onnx.load(model_path)

    initializer = helper.make_tensor(
        name=input_name,
        data_type=TensorProto.INT64,
        dims=constant_value.shape,
        vals=constant_value.flatten(),
    )

    model.graph.initializer.append(initializer)

    inputs_to_keep = [inp for inp in model.graph.input if inp.name != input_name]

    del model.graph.input[:]
    model.graph.input.extend(inputs_to_keep)

    model_simplified, check = onnxsim.simplify(model)

    if check:
        onnx.save(model_simplified, model_path)
        print(f"Simplified model saved to {model_path}")
    else:
        print("Simplification failed, saving original modification")
        onnx.save(model, model_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--resume', '-r', type=str)
    parser.add_argument('--output_file', '-o', type=str, default='model.onnx')
    parser.add_argument('--input_size', '-s', type=int, default=1280, help="-s 640 for IR, -s 1280 for RGB")
    parser.add_argument('--image_channels', '-i', type=int, default=3, help="Number of image channels. IR has 1 channel")
    parser.add_argument('--check',  action='store_true', default=False,)
    parser.add_argument('--simplify',  action='store_true', default=False,)
    parser.add_argument("--fix_dimensions", action='store_true', default=False)
    args = parser.parse_args()

    main(args)