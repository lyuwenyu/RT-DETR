"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
import math
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler
import torchvision

import numpy as np

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs,
):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def output_to_file(results, valid_cat_ids=None):
    """Use this function to output predictions to CMD line for further visualization (e.g., copy paste in jupyter notebook)"""
    output = []
    for result in results:
        for score, label_id, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            score, label = score.item(), label_id.item()

            # Skip predictions that are not in the valid categories
            if valid_cat_ids is not None and label not in valid_cat_ids:
                continue

            line = f"{label} {score:.2f} {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f}"
            output.append(line)

        output.append("---\n")
    return output


def nms(results, config={"nms_threshold": 0.7}):
    """This function is a simple wrapper around torchvision.ops.nms to perform NMS on the results."""
    for i, result in enumerate(results):
        boxes = result["boxes"]  # xyxy
        scores = result["scores"]
        labels = result["labels"]

        keep_indices = torchvision.ops.nms(boxes, scores, config["nms_threshold"])

        results[i]["boxes"] = boxes[keep_indices]
        results[i]["scores"] = scores[keep_indices]
        results[i]["labels"] = labels[keep_indices]

    return results


def score_filter(results, config={"score_threshold": 0.0}):
    """This function performs score thresholding on the results."""
    for i, result in enumerate(results):
        boxes = result["boxes"]  # xyxy
        scores = result["scores"]
        labels = result["labels"]

        keep_indices = scores > config["score_threshold"]

        results[i]["boxes"] = boxes[keep_indices]
        results[i]["scores"] = scores[keep_indices]
        results[i]["labels"] = labels[keep_indices]

    return results


def class_remapping(results, config={"remap_dict": {}}):
    """This function remaps the class labels in the results."""
    for i, result in enumerate(results):
        labels = result["labels"]

        for old_label, new_label in config["remap_dict"].items():
            labels[labels == old_label] = new_label

        results[i]["labels"] = labels

    return results


def additional_postprocess(results, methods={}):
    """This function is implement any additional post-processing (e.g. NMS) that RT-DETR doesn't implement."""
    for method_name, method_config in methods.items():
        if method_name == "nms":
            results = nms(results, method_config)
        elif method_name == "score_filter":
            results = score_filter(results, method_config)
        elif method_name == "class_remapping":
            results = class_remapping(results, method_config)
        else:
            raise NotImplementedError(
                f"Post-processing method {method_name} is not implemented."
            )

    return results


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    # CUSTOM EVAL PARAMETERS - TOGGLE ME
    output_predictions=True,
    additional_postprocess_methods={
        # Uncomment to enable (disabled by default)
        # "nms": {"nms_threshold": 0.2},
        # "score_filter": {"score_threshold": 0.3},
        # "class_remapping": {"remap_dict": {2: 7, 5: 7}},
    },
):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    output_lines = []

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # orig_target_sizes = torch.tensor([[samples.shape[-1], samples.shape[-2]]], device=samples.device)

        results = postprocessor(outputs, orig_target_sizes)
        results = additional_postprocess(
            results, methods=additional_postprocess_methods
        )

        if output_predictions:
            output_lines += output_to_file(results, valid_cat_ids={0, 2, 4, 5, 7})

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    if output_predictions:
        with open("predictions.txt", "w") as f:
            f.write("\n".join(output_lines))

        with open("image_ids.txt", "w") as f:
            f.write("\n".join(np.array(coco_evaluator.img_ids).astype(str)))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator
