'''by lyuwenyu
'''

import torch 
import torchvision

import numpy as np 
import onnxruntime as ort 

from utils import yolo_insert_nms

class YOLOv8(torch.nn.Module):
    def __init__(self, name) -> None:
        super().__init__()
        from ultralytics import YOLO
        # Load a model
        # build a new model from scratch
        # model = YOLO(f'{name}.yaml')  

        # load a pretrained model (recommended for training)
        model = YOLO(f'{name}.pt')  
        self.model = model.model

    def forward(self, x):
        '''https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L216
        '''
        pred: torch.Tensor = self.model(x)[0] # n 84 8400,
        pred = pred.permute(0, 2, 1)
        boxes, scores = pred.split([4, 80], dim=-1)
        boxes = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')

        return boxes, scores



def export_onnx(name='yolov8n'):
    '''export onnx
    '''
    m = YOLOv8(name)

    x = torch.rand(1, 3, 640, 640)
    dynamic_axes = {
        'image': {0: '-1'}
    }
    torch.onnx.export(m, x, f'{name}.onnx', 
                      input_names=['image'], 
                      output_names=['boxes', 'scores'], 
                      opset_version=13, 
                      dynamic_axes=dynamic_axes)

    data = np.random.rand(1, 3, 640, 640).astype(np.float32)
    sess = ort.InferenceSession(f'{name}.onnx')
    _ = sess.run(output_names=None, input_feed={'image': data})


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='yolov8l')
    parser.add_argument('--score_threshold', type=float, default=0.001)
    parser.add_argument('--iou_threshold', type=float, default=0.7)
    parser.add_argument('--max_output_boxes', type=int, default=300)
    args = parser.parse_args()

    export_onnx(name=args.name)
    
    yolo_insert_nms(path=f'{args.name}.onnx', 
                    score_threshold=args.score_threshold, 
                    iou_threshold=args.iou_threshold, 
                    max_output_boxes=args.max_output_boxes, )

