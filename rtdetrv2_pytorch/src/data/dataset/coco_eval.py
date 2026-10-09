"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib

# MiXaiLL76 replacing pycocotools with faster-coco-eval for better performance and support.
"""

import copy
import io
from contextlib import redirect_stdout

import numpy as np
import torch.distributed as dist
from faster_coco_eval import COCOeval_faster
from faster_coco_eval.utils.pytorch import FasterCocoEvaluator

from ...core import register


def _ultrafast_tools():
    try:
        from ultrafast_pycocotools import COCO, COCOeval
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'Install the optional evaluator with pip install "ultrafast-pycocotools>=0.1.11,<0.2".'
        ) from exc
    return COCO, COCOeval


class _UltrafastCOCOeval(COCOeval_faster):
    """Native matching/accumulation with the existing faster-coco-eval summaries."""

    def __init__(self, coco_gt, iouType, **kwargs):
        super().__init__(None, iouType=iouType, **kwargs)
        self.cocoGt = coco_gt
        self.params.imgIds = sorted(coco_gt.getImgIds())
        self.params.catIds = sorted(coco_gt.catToImgs if iouType == "keypoints" else coco_gt.getCatIds())

    def evaluate(self):
        self.params.imgIds = list(np.unique(self.params.imgIds))
        if self.params.useCats:
            self.params.catIds = list(np.unique(self.params.catIds))
        self.params.maxDets = sorted(self.params.maxDets)
        _, evaluator = _ultrafast_tools()
        self._native = evaluator(
            self.cocoGt,
            self.cocoDt,
            self.params.iouType,
            lvis_style=self.lvis_style,
            lvis_protocol="coco",
            print_function=self.print_function,
        )
        # Do not assign Faster Params wholesale: its deprecated useSegm property
        # would reinterpret keypoints as bbox in the reference-compatible API.
        for name in (
            "imgIds",
            "catIds",
            "iouThrs",
            "recThrs",
            "maxDets",
            "areaRng",
            "areaRngLbl",
            "useCats",
            "kpt_oks_sigmas",
        ):
            if hasattr(self.params, name):
                setattr(self._native.params, name, copy.deepcopy(getattr(self.params, name)))
        self._native.evaluate()
        self._paramsEval = copy.deepcopy(self.params)
        if self.lvis_style:
            self.freq_groups = self._prepare_freq_group()

    def accumulate(self):
        self._native.accumulate()
        self.eval = dict(self._native.eval)
        self.eval["params"] = self.params


@register()
class CocoEvaluator(FasterCocoEvaluator):
    def __init__(self, coco_gt, iou_types, lvis_style=False, ranges=None, backend="faster_coco_eval"):
        if backend not in ("faster_coco_eval", "ultrafast"):
            raise ValueError(f"Unknown COCO backend {backend!r}")
        self.backend = backend
        self._range_kwargs = {} if ranges is None else {"ranges": ranges}
        if backend == "ultrafast":
            coco_class, _ = _ultrafast_tools()
        super().__init__(coco_gt, iou_types, lvis_style=lvis_style, **self._range_kwargs)
        if backend == "ultrafast":
            # Reuse the already-copied dataset, keeping the caller's object intact.
            self.coco_gt = coco_class(self.coco_gt.dataset, verbose=False)
            self.cleanup()

    def cleanup(self):
        if self.backend == "faster_coco_eval":
            return super().cleanup()
        self.coco_eval = {
            iou_type: _UltrafastCOCOeval(
                self.coco_gt,
                iouType=iou_type,
                lvis_style=self.lvis_style,
                print_function=print,
                separate_eval=True,
                **self._range_kwargs,
            )
            for iou_type in self.iou_types
        }
        self.img_ids = []
        self.eval_imgs = {key: [] for key in self.iou_types}
        self.stats_as_dict = {key: [] for key in self.iou_types}
        self._predictions = {key: {} for key in self.iou_types}

    def prepare_for_coco_segmentation(self, predictions):
        if self.backend == "faster_coco_eval":
            return super().prepare_for_coco_segmentation(predictions)
        from ultrafast_pycocotools import mask as mask_util

        results = []
        for image_id, prediction in predictions.items():
            if not prediction or len(prediction["masks"]) == 0:
                continue
            masks = prediction["masks"][:, 0].detach().cpu().numpy() > 0.5
            rles = mask_util.encode(np.asfortranarray(masks.transpose(1, 2, 0), dtype=np.uint8))
            for rle, label, score in zip(rles, prediction["labels"].tolist(), prediction["scores"].tolist()):
                rle["counts"] = rle["counts"].decode("utf-8")
                results.append({"image_id": image_id, "category_id": int(label), "score": score, "segmentation": rle})
        return results

    def update(self, predictions):
        if self.backend == "faster_coco_eval":
            return super().update(predictions)
        image_ids = sorted(predictions)
        self.img_ids.extend(image_ids)
        for iou_type in self.iou_types:
            per_image = {image_id: [] for image_id in image_ids}
            for result in self.prepare(predictions, iou_type):
                per_image[result["image_id"]].append(result)
            for image_id in image_ids:
                self._predictions[iou_type].setdefault(image_id, per_image[image_id])

    def synchronize_between_processes(self):
        if self.backend == "faster_coco_eval":
            return super().synchronize_between_processes()
        actual_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        world_size = actual_size if self.world_size is None else self.world_size
        if world_size not in (1, actual_size):
            raise ValueError("world_size must be 1 or match the initialized process group")
        for iou_type in self.iou_types:
            gathered = [self._predictions[iou_type]]
            if world_size > 1:
                gathered = [None] * world_size
                dist.all_gather_object(gathered, self._predictions[iou_type])
            merged = {}
            for per_rank in gathered:
                for image_id, predictions in per_rank.items():
                    merged.setdefault(image_id, predictions)
            # Match the existing merge: sorted IDs and first occurrence in rank/batch order.
            image_ids = sorted(merged)
            results = [item for image_id in image_ids for item in merged[image_id]]
            evaluator = self.coco_eval[iou_type]
            with redirect_stdout(io.StringIO()):
                evaluator.cocoDt = self.coco_gt.loadRes(results)
                evaluator.params.imgIds = image_ids
                evaluator.evaluate()
            self._predictions[iou_type].clear()
