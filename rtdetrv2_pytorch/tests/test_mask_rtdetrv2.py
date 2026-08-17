import unittest

import torch

from src.data.dataset.coco_dataset import convert_coco_poly_to_mask
from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
from src.zoo.rtdetr.rtdetrv2_criterion import RTDETRCriterionv2
from src.zoo.rtdetr.matcher import HungarianMatcher


class MaskRTDETRv2Test(unittest.TestCase):
    def test_polygon_conversion(self):
        masks = convert_coco_poly_to_mask(
            [[[1, 1, 5, 1, 5, 5, 1, 5]]], height=8, width=8
        )
        self.assertEqual(tuple(masks.shape), (1, 8, 8))
        self.assertEqual(int(masks.sum()), 16)

    def test_mask_criterion_with_empty_targets(self):
        criterion = RTDETRCriterionv2(
            matcher=HungarianMatcher(
                weight_dict={
                    'cost_class': 4,
                    'cost_bbox': 5,
                    'cost_giou': 2,
                    'cost_mask': 5,
                    'cost_dice': 5,
                },
                num_sample_points=64,
            ),
            weight_dict={
                'loss_vfl': 1,
                'loss_bbox': 5,
                'loss_giou': 2,
                'loss_mask': 5,
                'loss_dice': 5,
            },
            losses=['vfl', 'boxes', 'masks'],
            num_classes=3,
            num_sample_points=64,
        )
        outputs = {
            'pred_logits': torch.randn(1, 8, 3),
            'pred_boxes': torch.rand(1, 8, 4),
            'pred_masks': torch.randn(1, 8, 16, 16),
        }
        targets = [{
            'labels': torch.empty(0, dtype=torch.int64),
            'boxes': torch.empty((0, 4)),
            'masks': torch.empty((0, 16, 16)),
        }]
        losses = criterion(outputs, targets)
        self.assertTrue(torch.isfinite(torch.stack(list(losses.values()))).all())
        self.assertEqual(float(losses['loss_mask']), 0.0)
        self.assertEqual(float(losses['loss_dice']), 0.0)

    def test_postprocessor_returns_original_size_masks(self):
        postprocessor = RTDETRPostProcessor(
            num_classes=3, use_focal_loss=True, num_top_queries=2
        )
        outputs = {
            'pred_logits': torch.randn(1, 4, 3),
            'pred_boxes': torch.rand(1, 4, 4),
            'pred_masks': torch.randn(1, 4, 16, 16),
        }
        result = postprocessor(outputs, torch.tensor([[32, 24]]))[0]
        self.assertEqual(tuple(result['masks'].shape), (2, 1, 24, 32))
        self.assertEqual(result['masks'].dtype, torch.bool)


if __name__ == '__main__':
    unittest.main()
