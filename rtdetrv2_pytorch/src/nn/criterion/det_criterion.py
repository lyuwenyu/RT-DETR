"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn.functional as F 
import torch.distributed
import torchvision

from ...misc import box_ops
from ...misc import dist_utils
from ...core import register


@register()
class DetCriterion(torch.nn.Module):
    """Default Detection Criterion
    """
    __share__ = ['num_classes']
    __inject__ = ['matcher']

    def __init__(self, 
                losses, 
                weight_dict, 
                num_classes=80, 
                alpha=0.75, 
                gamma=2.0, 
                box_fmt='cxcywh',
                matcher=None):
        """
        Args:
            losses (list[str]): requested losses, support ['boxes', 'vfl', 'focal']
            weight_dict (dict[str, float)]: corresponding losses weight, including
                ['loss_bbox', 'loss_giou', 'loss_vfl', 'loss_focal']
            box_fmt (str): in box format, 'cxcywh' or 'xyxy'
            matcher (Matcher): matcher used to match source to target
        """
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.box_fmt = box_fmt
        assert matcher is not None, ''
        self.matcher = matcher

    def forward(self, outputs, targets, **kwargs):
        """
        Args:
            outputs: Dict[Tensor], 'pred_boxes', 'pred_logits', 'meta'.
            targets, List[Dict[str, Tensor]], len(targets) == batch_size.
            kwargs, store other information such as current epoch id.
        Return:
            losses, Dict[str, Tensor]
        """
        matched = self.matcher(outputs, targets)
        values = matched['values']
        indices = matched['indices']
        num_boxes = self._get_positive_nums(indices)
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)
        return losses 

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])        
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_positive_nums(self, indices):
        # number of positive samples
        num_pos = sum(len(i) for (i, _) in indices)
        num_pos = torch.as_tensor([num_pos], dtype=torch.float32, device=indices[0][0].device)
        if dist_utils.is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_pos)
        num_pos = torch.clamp(num_pos / dist_utils.get_world_size(), min=1).item()
        return num_pos

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].to(src_logits.dtype)
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.sum() / num_boxes
        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        src_boxes = torchvision.ops.box_convert(src_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        target_boxes = torchvision.ops.box_convert(target_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        iou, _ = box_ops.elementwise_box_iou(src_boxes.detach(), target_boxes)
        
        src_logits: torch.Tensor = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = iou.to(src_logits.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        src_score = F.sigmoid(src_logits.detach())
        weight = self.alpha * src_score.pow(self.gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')        
        loss = loss.sum() / num_boxes
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)        
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        src_boxes = torchvision.ops.box_convert(src_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        target_boxes = torchvision.ops.box_convert(target_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        loss_giou = 1 - box_ops.elementwise_generalized_box_iou(src_boxes, target_boxes)
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_boxes_giou(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)        
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        src_boxes = torchvision.ops.box_convert(src_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        target_boxes = torchvision.ops.box_convert(target_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        loss_giou = 1 - box_ops.elementwise_generalized_box_iou(src_boxes, target_boxes)
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'giou': self.loss_boxes_giou,
            'vfl': self.loss_labels_vfl,
            'focal': self.loss_labels_focal,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
