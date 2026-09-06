"""Detection record and conversion from the RT-DETRv2 postprocessor output.

The postprocessor in deploy mode returns a fixed 300 queries per frame with no
score filtering and no NMS (see src/zoo/rtdetr/rtdetr_postprocessor.py), with
boxes already in absolute xyxy pixel coordinates of the original frame. Labels
are contiguous 0..79 COCO indices, because deploy mode returns before the
category remapping block. This module is the one place that knows all of that.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Detection:
    name: str
    label: int
    score: float
    box: Tuple[float, float, float, float]   # x1, y1, x2, y2 in frame pixels

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def centre(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def from_model_output(labels, boxes, scores, class_names, score_threshold):
    """Turn one frame of (labels, boxes, scores) into filtered Detections.

    Accepts torch tensors or numpy arrays, batched (1, N, ...) or unbatched.
    """
    labels, boxes, scores = (_to_numpy(a) for a in (labels, boxes, scores))

    if labels.ndim == 2:            # strip the batch dimension
        labels, boxes, scores = labels[0], boxes[0], scores[0]

    keep = scores > score_threshold
    labels, boxes, scores = labels[keep], boxes[keep], scores[keep]

    out = []
    for lab, box, scr in zip(labels, boxes, scores):
        idx = int(lab)
        name = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
        out.append(Detection(name=name, label=idx, score=float(scr),
                             box=tuple(float(v) for v in box)))
    return out


def _to_numpy(a):
    if hasattr(a, 'detach'):
        a = a.detach().cpu().numpy()
    return np.asarray(a)
