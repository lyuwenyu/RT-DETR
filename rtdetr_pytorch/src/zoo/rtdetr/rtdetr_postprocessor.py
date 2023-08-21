"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision

from src.core import register


__all__ = ['RTDETRPostProcessor']


@register
class RTDETRPostProcessor(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category']
    
    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False 

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes):

        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
            labels = index % self.num_classes
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
            labels = torch.tensor([self.mscoco_label_category_map[int(x.item()) + 1] for x in labels.flatten()])\
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

    @property
    def iou_types(self, ):
        return ('bbox', )


    @property
    def mscoco_label_category_map(self, ):
        return {
                1: 1,
                2: 2,
                3: 3,
                4: 4,
                5: 5,
                6: 6,
                7: 7,
                8: 8,
                9: 9,
                10: 10,
                11: 11,
                12: 13,
                13: 14,
                14: 15,
                15: 16,
                16: 17,
                17: 18,
                18: 19,
                19: 20,
                20: 21,
                21: 22,
                22: 23,
                23: 24,
                24: 25,
                25: 27,
                26: 28,
                27: 31,
                28: 32,
                29: 33,
                30: 34,
                31: 35,
                32: 36,
                33: 37,
                34: 38,
                35: 39,
                36: 40,
                37: 41,
                38: 42,
                39: 43,
                40: 44,
                41: 46,
                42: 47,
                43: 48,
                44: 49,
                45: 50,
                46: 51,
                47: 52,
                48: 53,
                49: 54,
                50: 55,
                51: 56,
                52: 57,
                53: 58,
                54: 59,
                55: 60,
                56: 61,
                57: 62,
                58: 63,
                59: 64,
                60: 65,
                61: 67,
                62: 70,
                63: 72,
                64: 73,
                65: 74,
                66: 75,
                67: 76,
                68: 77,
                69: 78,
                70: 79,
                71: 80,
                72: 81,
                73: 82,
                74: 84,
                75: 85,
                76: 86,
                77: 87,
                78: 88,
                79: 89,
                80: 90
            }

