"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision

from ...core import register


__all__ = ['RTDETRPostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class RTDETRPostProcessor(nn.Module):
    __share__ = [
        'num_classes', 
        'use_focal_loss', 
        'num_top_queries', 
        'remap_mscoco_category'
    ]
    
    def __init__(
        self, 
        num_classes=80, 
        use_focal_loss=True, 
        num_top_queries=300, 
        remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False 

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        # Classes are 1-indexed (hence the num_classes + 1). The 0th index always has a 
        # near-zero score for a trained model
        if not self.training:

            # Ignore 0-th index (see above comment)
            scores = F.sigmoid(logits[:,:,1:])
            
            # This gives duplicate indices after integer division
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            index = index // (self.num_classes - 1)

            # Get the unique indexes and find max score for each index
            batch_size = logits.shape[0]
            max_scores = torch.zeros(scores.shape, device=scores.device, dtype=scores.dtype)
            padded_unique_indices = torch.zeros(scores.shape, device=scores.device, dtype=torch.long)
            for b in range(batch_size):
                unique_box_indices, inverse_indices = torch.unique(index[b], dim=0, return_inverse=True)
                unique_box_indices = F.pad(unique_box_indices, (0, self.num_top_queries - unique_box_indices.size(0)), value=0.0)

                reduced_scores = torch.scatter_reduce(torch.zeros(self.num_top_queries, device=scores.device, dtype=scores.dtype), 
                                    dim=0, index=inverse_indices, src=scores[b], reduce='amax')
                max_scores[b] = reduced_scores
                padded_unique_indices[b] = unique_box_indices

            scores = max_scores
            index = padded_unique_indices

            # Probability of each class
            soft_labels = F.softmax(logits[:,:,1:], dim=-1)
            labels = soft_labels.gather(dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        else:
            if self.use_focal_loss:
                scores = F.sigmoid(logits)
                scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
                # TODO for older tensorrt
                # labels = index % self.num_classes
                labels = mod(index, self.num_classes) # this will never index the 0-th score since the 0-th score is always near zero for a trained model
                index = index // self.num_classes
                boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
                
            else:
                scores = F.softmax(logits)[:, :, :-1]
                scores, labels = scores.max(dim=-1)
                if scores.shape[1] > self.num_top_queries:
                    scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                    labels = torch.gather(labels, dim=1, index=index)
                    boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))
    
        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        
        return results
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 
