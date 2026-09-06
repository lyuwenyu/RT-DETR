"""RT-DETRv2 video inference — processes frame-by-frame and writes annotated output."""

import sys
from pathlib import Path
# Allow 'from src.core import ...' when run from references/deploy/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
from PIL import Image

from src.core import YAMLConfig

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

_COLORS = [
    (255,  56,  56), (255, 157, 151), (255, 112,  31), (255, 178,  29),
    (207, 210,  49), ( 72, 249,  10), (146, 204,  23), ( 61, 219, 134),
    ( 26, 147,  52), (  0, 212, 187), ( 44, 153, 168), (  0, 194, 255),
    ( 52,  69, 147), (100, 115, 255), (  0,  24, 236), (132,  56, 255),
    ( 82,   0, 133), (203,  56, 255), (255, 149, 200), (255,  55, 199),
]


def _annotate(frame_bgr, labels, boxes, scores, thrh):
    scr = scores[0]
    mask = scr > thrh
    lab  = labels[0][mask]
    box  = boxes[0][mask]
    scrs = scores[0][mask]

    for j, b in enumerate(box):
        cls_id   = lab[j].item()
        cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
        color    = _COLORS[cls_id % len(_COLORS)]
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {scrs[j].item():.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame_bgr, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame_bgr


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)

    if not args.resume:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    checkpoint = torch.load(args.resume, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().to(args.device)
    model.eval()

    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    cap = cv2.VideoCapture(args.video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_file}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Input : {args.video_file}")
    print(f"Size  : {width}x{height}  FPS: {fps:.1f}  Frames: {total}")
    print(f"Output: {args.output}")
    print(f"Threshold: {args.thrh}")
    print()

    frame_idx = 0
    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            pil_img   = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            orig_size = torch.tensor([width, height])[None].to(args.device)
            im_data   = transforms(pil_img)[None].to(args.device)

            labels, boxes, scores = model(im_data, orig_size)
            annotated = _annotate(frame_bgr.copy(), labels, boxes, scores, args.thrh)
            writer.write(annotated)

            frame_idx += 1
            if frame_idx % 50 == 0 or frame_idx == total:
                pct = frame_idx / max(total, 1) * 100
                print(f"  Frame {frame_idx:>5}/{total}  ({pct:5.1f}%)", flush=True)

    cap.release()
    writer.release()
    print(f"\nFinished. Output saved to: {args.output}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',     type=str, required=True)
    parser.add_argument('-r', '--resume',     type=str, required=True)
    parser.add_argument('-f', '--video-file', type=str, required=True)
    parser.add_argument('-d', '--device',     type=str, default='cpu')
    parser.add_argument('--thrh',             type=float, default=0.5)
    parser.add_argument('-o', '--output',     type=str, default='video_result.mp4')
    args = parser.parse_args()
    main(args)