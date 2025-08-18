"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision


__all__ = ['DetDETRPostProcessor']

from .box_revert import box_revert
from .box_revert import BoxProcessFormat

def mod(a, b):
    out = a - a // b * b
    return out

class DetDETRPostProcessor(nn.Module):
    def __init__(
        self, 
        num_classes=80, 
        use_focal_loss=True, 
        num_top_queries=300, 
        box_process_format=BoxProcessFormat.RESIZE,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.box_process_format = box_process_format
        self.deploy_mode = False 

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    def forward(self, outputs, **kwargs):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = index % self.num_classes
            # labels = mod(index, self.num_classes) # for tensorrt
            index = index // self.num_classes
            boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
            
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        if kwargs is not None:
            boxes = box_revert(
                boxes, 
                in_fmt='cxcywh',
                out_fmt='xyxy',
                process_fmt=self.box_process_format,
                normalized=True,
                **kwargs,
            )

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        
        return results
        
    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 
