import os
import argparse
from rtdetr_pytorch.tools.export_onnx import main


def convert_onnx(checkpoint_path: str, config_path: str, output_onnx_path: str = None):
    if output_onnx_path is None:
        output_onnx_path = checkpoint_path.replace('.pth', '.onnx')
    if os.path.exists(output_onnx_path):
        return output_onnx_path
    args = get_args()
    args.config = str(config_path)
    args.resume = str(checkpoint_path)
    args.file_name = str(output_onnx_path)
    args.check = True
    main(args)
    return output_onnx_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--file-name', '-f', type=str, default='model.onnx')
    parser.add_argument('--check',  action='store_true', default=False,)
    parser.add_argument('--simplify',  action='store_true', default=False,)
    args = parser.parse_args([])
    return args