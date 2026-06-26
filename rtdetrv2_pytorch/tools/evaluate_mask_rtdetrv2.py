"""Evaluate Mask RT-DETR v2 checkpoints on COCO-style instance segmentation data.

This tool mirrors ``tools/train.py --test-only`` but also writes a compact
metrics JSON file and can export COCO-format bbox/segm predictions.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.misc.logger import MetricLogger
from src.solver import TASKS


COCO_METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


def _named_metrics(values):
    return {name: float(value) for name, value in zip(COCO_METRIC_NAMES, values)}


@torch.no_grad()
def evaluate_and_collect(model, postprocessor, data_loader, coco_evaluator, device, save_predictions=False):
    model.eval()
    coco_evaluator.cleanup()

    prediction_results = defaultdict(list)
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        outputs = model(samples)
        orig_target_sizes = torch.stack([target["orig_size"] for target in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        res = {
            target["image_id"].item(): _to_cpu(output)
            for target, output in zip(targets, results)
        }

        if save_predictions:
            for iou_type in coco_evaluator.iou_types:
                prediction_results[iou_type].extend(coco_evaluator.prepare(res, iou_type))

        coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = {}
    if "bbox" in coco_evaluator.iou_types:
        stats["bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
    if "segm" in coco_evaluator.iou_types:
        stats["segm"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    if save_predictions:
        gathered_results = dist_utils.all_gather(dict(prediction_results))
        merged_results = defaultdict(list)
        for rank_results in gathered_results:
            for iou_type, rows in rank_results.items():
                merged_results[iou_type].extend(rows)
        prediction_results = dict(merged_results)

    return stats, prediction_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Mask RT-DETR v2 checkpoint")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to a mask RT-DETR YAML config")
    parser.add_argument(
        "-r",
        "--resume",
        "--checkpoint",
        dest="resume",
        type=str,
        help="Checkpoint to evaluate, for example output/mask_rtdetrv2/best.pth",
    )
    parser.add_argument("-d", "--device", type=str, help="Device, for example cuda, mps, or cpu")
    parser.add_argument("--seed", type=int, help="Evaluation seed")
    parser.add_argument("--output-dir", type=str, help="Directory for metrics and optional predictions")
    parser.add_argument("--metrics-file", type=str, help="Metrics JSON path. Defaults to <output-dir>/metrics.json")
    parser.add_argument("--save-predictions", action="store_true", help="Write COCO-format prediction JSON files")
    parser.add_argument(
        "--prediction-prefix",
        type=str,
        default="predictions",
        help="Prediction file prefix used with --save-predictions",
    )
    parser.add_argument("-u", "--update", nargs="+", help="YAML overrides, e.g. num_classes=2")
    parser.add_argument("--print-method", type=str, default="builtin", help="Print method: builtin or rich")
    parser.add_argument("--print-rank", type=int, default=0, help="Distributed rank allowed to print")
    parser.add_argument("--local-rank", type=int, help="Local rank for distributed launchers")
    return parser.parse_args()


def main(args):
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    update_dict = yaml_utils.parse_cli(args.update)
    for key in ["resume", "device", "seed"]:
        value = getattr(args, key)
        if value is not None:
            update_dict[key] = value
    if args.output_dir is not None:
        update_dict["output_dir"] = args.output_dir

    cfg = YAMLConfig(args.config, **update_dict)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.output_dir) / "eval_mask"
    metrics_file = Path(args.metrics_file) if args.metrics_file else output_dir / "metrics.json"

    stats, prediction_results = evaluate_and_collect(
        solver.ema.module if solver.ema else solver.model,
        solver.postprocessor,
        solver.val_dataloader,
        solver.evaluator,
        solver.device,
        save_predictions=args.save_predictions,
    )

    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.resume).resolve()) if args.resume else None,
        "device": str(solver.device),
        "iou_types": list(solver.evaluator.iou_types),
        "metrics": stats,
        "metrics_named": {iou_type: _named_metrics(values) for iou_type, values in stats.items()},
    }

    if dist_utils.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote metrics to {metrics_file}")

        if args.save_predictions:
            for iou_type, rows in prediction_results.items():
                prediction_file = output_dir / f"{args.prediction_prefix}_{iou_type}.json"
                prediction_file.write_text(json.dumps(rows), encoding="utf-8")
                print(f"Wrote {iou_type} predictions to {prediction_file}")

    dist_utils.cleanup()


if __name__ == "__main__":
    main(parse_args())
