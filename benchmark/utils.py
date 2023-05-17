'''by lyuwenyu
'''

import time 
import contextlib
import numpy as np
from PIL import Image
from collections import OrderedDict

import onnx
import torch 
import onnx_graphsurgeon


def to_binary_data(path, size=(640, 640), output_name='input_tensor.bin'):
    '''--loadInputs='image:input_tensor.bin'
    '''
    im = Image.open(path).resize(size)
    data = np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None] / 255.
    data.tofile(output_name)


def yolo_insert_nms(path, score_threshold=0.01, iou_threshold=0.7, max_output_boxes=300, simplify=False):
    '''
    http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/api/onnxops/onnx__EfficientNMS_TRT.html
    https://huggingface.co/spaces/muttalib1326/Punjabi_Character_Detection/blob/3dd1e17054c64e5f6b2254278f96cfa2bf418cd4/utils/add_nms.py
    '''
    onnx_model = onnx.load(path)

    if simplify:
        from onnxsim import simplify
        onnx_model, _ = simplify(onnx_model,  overwrite_input_shapes={'image': [1, 3, 640, 640]})

    graph = onnx_graphsurgeon.import_onnx(onnx_model)
    graph.toposort()
    graph.fold_constants()
    graph.cleanup()

    topk = max_output_boxes
    attrs = OrderedDict(plugin_version='1',
                        background_class=-1,
                        max_output_boxes=topk,
                        score_threshold=score_threshold,
                        iou_threshold=iou_threshold,
                        score_activation=False,
                        box_coding=0, )

    outputs = [onnx_graphsurgeon.Variable('num_dets', np.int32, [-1, 1]),
               onnx_graphsurgeon.Variable('det_boxes', np.float32, [-1, topk, 4]),
               onnx_graphsurgeon.Variable('det_scores', np.float32, [-1, topk]),
               onnx_graphsurgeon.Variable('det_classes', np.int32, [-1, topk])]

    graph.layer(op='EfficientNMS_TRT', 
                name="batched_nms", 
                inputs=[graph.outputs[0], 
                        graph.outputs[1]], 
                outputs=outputs, 
                attrs=attrs, )

    graph.outputs = outputs
    graph.cleanup().toposort()

    onnx.save(onnx_graphsurgeon.export_onnx(graph), f'yolo_w_nms.onnx')


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self, ):
        self.total = 0
        
    def __enter__(self, ):
        self.start = self.time()
        return self 
    
    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start
    
    def reset(self, ):
        self.total = 0
    
    def time(self, ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()
