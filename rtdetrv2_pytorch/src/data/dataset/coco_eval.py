"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib

# MiXaiLL76 replacing pycocotools with faster-coco-eval for better performance and support.
"""

from ...core import register
from faster_coco_eval.utils.pytorch import FasterCocoEvaluator

@register()
class CocoEvaluator(FasterCocoEvaluator):
    pass
