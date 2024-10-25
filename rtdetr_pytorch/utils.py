import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR, OneCycleLR

import supervisely as sly


def pred_2_annotation(pred, map_label2name, img_size_hw, topk=None):
    # pred.keys = ['boxes', 'labels', 'scores']
    ann = sly.Annotation(img_size_hw)
    for box, label, score in zip(
        pred["boxes"][:topk], pred["labels"][:topk], pred["scores"][:topk]
    ):
        obj_class = sly.ObjClass(name=map_label2name[int(label)], geometry_type=sly.Rectangle)
        tags = sly.Tag(sly.TagMeta("score", value_type="any_number"), float(score))
        box = list(map(int, box))
        label = sly.Label(
            sly.Rectangle(top=box[1], left=box[0], bottom=box[3], right=box[2]), obj_class, [tags]
        )
        ann = ann.add_label(label)
    return ann


def prepare_result(sample, result, orig_target_size, base_ds):
    h, w = sample.shape[-2:]
    boxes = result["boxes"].cpu()
    boxes = torch.clamp(boxes / orig_target_size.repeat(2).cpu(), 0.0, 1.0) * torch.tensor(
        [w, h, w, h]
    )
    map_label2name = {
        base_ds.cats[i]["id"]: base_ds.cats[i]["name"] for i in range(len(base_ds.cats))
    }
    labels = [map_label2name[i] for i in result["labels"].tolist()]
    prediction = {"boxes": boxes, "labels": labels, "scores": result["scores"].cpu().numpy()}
    img = sample.permute(1, 2, 0).cpu().contiguous().numpy()
    img = (img * 255).astype("uint8")
    return img, prediction


def collect_per_class_metrics(coco_evaluator, base_ds):
    # the shape of eval is: [iouThrs, recThrs, catIds, areaRng, maxDets]
    eval = coco_evaluator.coco_eval["bbox"].eval
    class_ap = {}
    class_ar = {}
    for i, catId in enumerate(coco_evaluator.coco_eval["bbox"].params.catIds):
        # AP
        s = eval["precision"][:, :, catId, 0, 2]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        class_ap[base_ds.cats[catId]["name"]] = mean_s

        # AR
        s = eval["recall"][:, catId, 0, 2]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        class_ar[base_ds.cats[catId]["name"]] = mean_s
    return class_ap, class_ar


def is_by_epoch(lr_scheduler):
    if isinstance(lr_scheduler, (OneCycleLR, CosineAnnealingLR, LinearLR)):
        return False
    elif isinstance(lr_scheduler, (MultiStepLR)):
        return True
    else:
        raise NotImplementedError(f"Unsupported lr_scheduler: {lr_scheduler}")


def name2cls(name):
    if name == "OneCycleLR":
        return OneCycleLR
    elif name == "CosineAnnealingLR":
        return CosineAnnealingLR
    elif name == "MultiStepLR":
        return MultiStepLR
    elif name == "LinearLR":
        return LinearLR
    else:
        raise NotImplementedError(f"Unsupported lr_scheduler: {name}")
