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
    def __init__(self, 
                box_fmt='cxcywh',
                logit_fmt='sigmoid',
                image_dimensions=(640, 640)) -> None:
        super().__init__()
        self.box_fmt = box_fmt.lower()
        self.logit_fmt = logit_fmt.lower()
        self.logit_func = getattr(F, self.logit_fmt, None)
        self.deploy_mode = False 
        self.image_dimensions = image_dimensions
    
    def forward(self, 
                outputs: Dict[str, Tensor], 
                iou_threshold=0.01, 
                score_threshold=0.1,
                apply_score_filtering_and_nms=True,
                keep_topk=300) -> List[Dict[str, Tensor]]:
        # Use the provided values directly
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.apply_score_filtering_and_nms = apply_score_filtering_and_nms
        self.keep_topk = keep_topk

        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        orig_target_sizes = torch.tensor(self.image_dimensions, dtype=torch.float32).to(boxes.device)
        orig_target_sizes = orig_target_sizes.repeat(boxes.size(0), 1)  # Repeat for batch size

        pred_boxes = torchvision.ops.box_convert(boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        pred_boxes *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        values, pred_labels = torch.max(logits, dim=-1)
        
        if self.logit_func:
            pred_scores = self.logit_func(values)
        else:
            pred_scores = values

        # Apply score filtering and NMS if enabled
        results = []
        if apply_score_filtering_and_nms:
            for i in range(logits.shape[0]):
                # Apply score threshold
                keep_indices = pred_scores[i] > score_threshold
                pred_box = pred_boxes[i][keep_indices]
                pred_label = pred_labels[i][keep_indices]
                pred_score = pred_scores[i][keep_indices]

                # Perform NMS
                keep = torchvision.ops.batched_nms(pred_box, pred_score, pred_label, iou_threshold=iou_threshold)
                keep = keep[:keep_topk]
                blob = {
                    'labels': pred_label[keep],
                    'boxes': pred_box[keep],
                    'scores': pred_score[keep],
                }
                results.append(blob)
        else:
            for i in range(pred_boxes.size(0)):
                out = {
                    "boxes": pred_boxes[i],
                    "labels": pred_labels[i],
                    "scores": pred_scores[i],
                }
                results.append(out)

        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 
