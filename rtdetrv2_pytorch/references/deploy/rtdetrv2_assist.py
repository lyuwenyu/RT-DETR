"""RT-DETRv2 blind-assistance mode -- spoken guidance and a walkable path.

Runs the detector and a monocular depth net over a webcam or video, and turns
the result into audio a blind user can walk behind: terse speech for what is
around them, a stereo beep whose rate encodes proximity, and a steering cue
toward the clearest way forward.

    python references/deploy/rtdetrv2_assist.py \\
        -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \\
        -r weights/rtdetrv2_r50vd_6x_coco_ema.pth \\
        --source 0 --device cuda:0 --overlay

NOT A SAFETY DEVICE. This is experimental software built on a camera that can
be wrong. It does not replace a white cane or a guide dog.
"""

import gc
import sys
import time
from pathlib import Path

# Allow 'from src...' when run from references/deploy/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from src.core import YAMLConfig
from src.assist.config import AssistConfig
from src.assist.core import GuidanceCore
from src.assist.depth import build_depth
from src.assist.detections import from_model_output
from src.assist.geometry import Intrinsics
from src.assist.io.camera import Camera
from src.assist.io.overlay import draw as draw_overlay
from src.assist.io.sonifier import Sonifier
from src.assist.io.speech import Speaker, NullSpeaker

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

DISCLAIMER = ('Assist mode ready. This is not a safety device. '
              'Keep using your cane.')


def _build_detector(args):
    """Load RT-DETRv2 in deploy mode -- same pattern as rtdetrv2_video.py."""
    cfg = YAMLConfig(args.config, resume=args.resume)

    checkpoint = torch.load(args.resume, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    # Release the checkpoint before the depth model allocates. An r50 file is
    # 165 MB and r101 is 293 MB, and holding both it and the loaded weights
    # doubles the peak while ONNX Runtime is also starting up.
    del checkpoint, state
    gc.collect()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    return Model().to(args.device).eval()


def main(args):
    detector = _build_detector(args)
    depth_net = build_depth(args.depth_model, input_size=args.depth_size,
                            device=args.device)

    cam = Camera(args.source, args.width, args.height).start()
    intr = Intrinsics.from_hfov(cam.width, cam.height, args.hfov)

    acfg = AssistConfig()
    acfg.hfov_deg = args.hfov
    acfg.score_threshold = args.thrh
    core = GuidanceCore(acfg, intr)

    audio = not args.no_audio
    speaker = (Speaker(rate=args.speech_rate, voice=args.voice) if audio
               else NullSpeaker()).start()
    sonifier = Sonifier(volume=args.volume).start() if audio else None

    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    orig_size = torch.tensor([cam.width, cam.height])[None].to(args.device)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, min(cam.fps, 30.0),
                                 (cam.width, cam.height))

    print('Source     : {}  ({}x{} @ {:.0f} fps)'.format(
        args.source, cam.width, cam.height, cam.fps))
    print('Detector   : {}  on {}'.format(Path(args.resume).name, args.device))
    print('Depth      : {}  [{}] at {}px'.format(
        Path(args.depth_model).name, depth_net.provider, depth_net.input_size))
    print('Camera FOV : {:.0f} deg  ->  fx {:.0f} px'.format(args.hfov, intr.fx))
    print('Audio      : {}'.format('speech + beeps' if audio else 'OFF'))
    print()
    print('*** NOT A SAFETY DEVICE -- does not replace a cane or guide dog ***')
    print()

    speaker.say(DISCLAIMER, priority=9)

    frame_idx = 0
    fps_ema = None
    last_speech = None
    timings = {'det': 0.0, 'depth': 0.0, 'guide': 0.0}

    try:
        while True:
            frame, stamp = cam.read()
            if frame is None:
                break
            loop_start = time.perf_counter()

            t0 = time.perf_counter()
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            im_data = transforms(pil)[None].to(args.device)
            with torch.no_grad():
                labels, boxes, scores = detector(im_data, orig_size)
            timings['det'] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            depth_rel = depth_net.infer(frame)
            timings['depth'] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            detections = from_model_output(labels, boxes, scores,
                                           COCO_CLASSES, acfg.score_threshold)
            out = core.update(detections, depth_rel, now=stamp)
            timings['guide'] = (time.perf_counter() - t0) * 1000

            if out.speech:
                last_speech = out.speech
                speaker.say(out.speech, priority=out.announcement.priority)
                print('  [{:6.1f}s] {}'.format(stamp % 1e4, out.speech), flush=True)
            if sonifier is not None:
                sonifier.apply(out.beep)

            dt = time.perf_counter() - loop_start
            inst = 1.0 / max(dt, 1e-6)
            fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst

            if args.overlay or writer is not None:
                vis = draw_overlay(frame, out, intr, acfg, fps=fps_ema,
                                   timings=timings, last_speech=last_speech)
                if writer is not None:
                    writer.write(vis)
                if args.overlay:
                    cv2.imshow('RT-DETRv2 assist', vis)
                    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                        break

            frame_idx += 1
            if not args.overlay and frame_idx % 30 == 0:
                print('  frame {:>5}  {:.1f} fps  det {det:.0f} / depth {depth:.0f}'
                      ' / guide {guide:.0f} ms'.format(frame_idx, fps_ema, **timings),
                      flush=True)

    except KeyboardInterrupt:
        print('\nInterrupted.')
    finally:
        cam.release()
        if writer is not None:
            writer.release()
        if sonifier is not None:
            sonifier.stop()
        speaker.stop()
        if args.overlay:
            cv2.destroyAllWindows()

    print('\nProcessed {} frames.'.format(frame_idx))
    if args.output:
        print('Overlay written to: {}'.format(args.output))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',       type=str, required=True)
    parser.add_argument('-r', '--resume',       type=str, required=True)
    parser.add_argument('-s', '--source',       type=str, default='0',
                        help='webcam index or video file path')
    parser.add_argument('-d', '--device',       type=str, default='cuda:0')
    parser.add_argument('-o', '--output',       type=str, default=None,
                        help='write the debug overlay to this mp4')
    parser.add_argument('--thrh',               type=float, default=0.45)
    parser.add_argument('--hfov',               type=float, default=65.0,
                        help='camera horizontal field of view in degrees')
    parser.add_argument('--depth-model',        type=str,
                        default='weights/depth_anything_v2_small/onnx/model.onnx')
    parser.add_argument('--depth-size',         type=int, default=392)
    parser.add_argument('--width',              type=int, default=None)
    parser.add_argument('--height',             type=int, default=None)
    parser.add_argument('--overlay',            action='store_true',
                        help='show the debug window')
    parser.add_argument('--no-audio',           action='store_true',
                        help='run silently; still prints what it would say')
    parser.add_argument('--volume',             type=float, default=0.35)
    parser.add_argument('--speech-rate',        type=int, default=190)
    parser.add_argument('--voice',              type=str, default=None)
    args = parser.parse_args()
    main(args)
