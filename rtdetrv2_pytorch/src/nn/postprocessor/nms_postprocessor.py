"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn.functional as F 
import torch.distributed
import torchvision
from torch import Tensor 
from typing import List, Dict

from ...core import register

from typing import Dict 


__all__ = ['DetNMSPostProcessor', ]


@register()
class DetNMSPostProcessor(torch.nn.Module):
    def __init__(self, \
                iou_threshold=0.01, 
                score_threshold=0.1, 
                keep_topk=300, 
                box_fmt='cxcywh',
                logit_fmt='sigmoid',
                image_dimensions=(640,640)) -> None:
        super().__init__()
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.keep_topk = keep_topk
        self.box_fmt = box_fmt.lower()
        self.logit_fmt = logit_fmt.lower()
        self.logit_func = getattr(F, self.logit_fmt, None)
        self.deploy_mode = False 
        self.image_dimensions = image_dimensions
    
    def forward(self, outputs: Dict[str, Tensor], **kwargs) -> List[Dict[str, Tensor]]:
        #iou_threshold = kwargs.get('iou_threshold', self.iou_threshold)
        #score_threshold = kwargs.get('score_threshold', self.score_threshold)
        #keep_topk = kwargs.get('keep_topk', self.keep_topk)
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        patch_size = torch.tensor(self.image_dimensions, dtype=torch.float32).to(boxes.device)
        patch_size = patch_size.repeat(boxes.size(0), 1)  # Repeat for batch size

        pred_boxes = torchvision.ops.box_convert(boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        pred_boxes *= patch_size.repeat(1, 2).unsqueeze(1)

        values, pred_labels = torch.max(logits, dim=-1)
        
        if self.logit_func:
            pred_scores = self.logit_func(values)
        else:
            pred_scores = values

        # TODO for onnx export
        blobs = {
            'labels': pred_labels, 
            'boxes': pred_boxes,
            'scores': pred_scores
        }
        return blobs

        # results = []
        # for i in range(logits.shape[0]):
        #     score_keep = pred_scores[i] > score_threshold
        #     pred_box = pred_boxes[i][score_keep]
        #     pred_label = pred_labels[i][score_keep]
        #     pred_score = pred_scores[i][score_keep]

        #     keep = torchvision.ops.batched_nms(pred_box, pred_score, pred_label, iou_threshold)            
        #     keep = keep[:keep_topk]

        #     blob = {
        #         'labels': pred_label[keep],
        #         'boxes': pred_box[keep],
        #         'scores': pred_score[keep],
        #     }

        #     results.append(blob)
        
        # # Add debug logs
        # print("results")
        # print(type(results))
        # for idx, result in enumerate(results):
        #     print(f"Type of results[{idx}]:", type(result))
        #     print(f"Keys in results[{idx}]:", result.keys())
        #     for key in result:
        #         print(f"Type of results[{idx}]['{key}']:", type(result[key]))

        # return blobs

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 
