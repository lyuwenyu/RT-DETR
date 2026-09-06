# Deployment

Reference inference scripts for RT-DETRv2.

| Script | What it does |
|---|---|
| `rtdetrv2_torch.py` | Single image, PyTorch |
| `rtdetrv2_video.py` | Video file, PyTorch, writes an annotated mp4 |
| `rtdetrv2_onnxruntime.py` | Single image via ONNX Runtime |
| `rtdetrv2_tensorrt.py` | Single image via a TensorRT engine |
| `rtdetrv2_assist.py` | **Blind-assistance mode** -- spoken guidance and a walkable path |

---

## Blind-assistance mode

> **NOT A SAFETY DEVICE.** Experimental software built on a camera that can be
> wrong. It does not replace a white cane or a guide dog.

Turns detections into audio a blind user can walk behind: terse speech for what
is around them, a stereo beep whose rate encodes proximity, and a steering cue
toward the clearest way forward.

### What it does

* **Metric distances from one ordinary camera.** A monocular depth network only
  produces *relative* depth on an unknown scale. The detector supplies the
  missing scale: a person is about 1.7 m tall, so pinhole ranging on their box
  gives a true distance, and enough such pairs fix the depth map in metres. No
  depth sensor, no stereo rig, no user calibration step.
* **A path, not just a list of objects.** The metric depth is lifted to 3D, the
  floor is RANSAC-fitted each frame, and everything is classified by height
  above it. Rays are then cast across the forward arc through a top-down
  occupancy grid dilated by half a shoulder width, so "is this walkable" means
  "does a person fit".
* **Drop-offs.** Ground that sits *below* the floor plane -- a kerb, a step
  down, an unfenced edge. COCO has no class for any of that, and no detector
  will ever find one; it is visible only as geometry, and it is the
  highest-stakes hazard for someone who cannot see it coming.
* **Two audio channels.** Speech is too slow to be a collision warning, so
  proximity rides a beep (rate = range, stereo pan = bearing, a distinctly low
  tone for drop-offs) that is never blocked by speech. Speech itself is
  rationed to roughly one short phrase every two seconds. **Silence is the
  default:** when the path is clear, it says nothing.

### Setup

```bash
pip install opencv-python pyttsx3 sounddevice huggingface-hub
pip install onnxruntime-gpu==1.16.3          # see the note in requirements.txt

python -c "from huggingface_hub import hf_hub_download as d;   d('onnx-community/depth-anything-v2-small', 'onnx/model.onnx',     local_dir='weights/depth_anything_v2_small')"
```

### Run

Live, with audio and the debug window:

```bash
python references/deploy/rtdetrv2_assist.py   -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml   -r weights/rtdetrv2_r50vd_6x_coco_ema.pth   --source 0 --device cuda:0 --overlay
```

Over a recorded walk, silent, writing an annotated mp4:

```bash
python references/deploy/rtdetrv2_assist.py   -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml   -r weights/rtdetrv2_r50vd_6x_coco_ema.pth   --source walk.mp4 --device cuda:0 --no-audio -o assist_debug.mp4
```

`--no-audio` still prints every phrase it would have spoken, which is the
fastest way to judge whether the system is too chatty.

### Options that matter

| Flag | Default | Notes |
|---|---|---|
| `--hfov` | 65 | Camera horizontal FOV in degrees. **Set this.** Every distance scales with it; a telephoto lens left at 65 will read far too close. |
| `--depth-size` | 392 | Depth input side, rounded to a multiple of 14. 518 is sharper, 322 is faster. |
| `--thrh` | 0.45 | Detection score threshold. |
| `--overlay` | off | Debug window: boxes, corridor rays drawn on the floor, top-down grid. |
| `--no-audio` | off | Silent; phrases are printed instead. |

### Reading the overlay

Corridor rays are drawn where they actually fall on the floor, and again in the
top-down inset. Green is open, red is blocked by an obstacle, magenta is a
drop-off; the thick ray with the white marker is the chosen heading. Box colour
is the hazard zone -- red immediate, amber near, green context, grey untracked.
The status block reports the fitted floor height and tilt, whether the depth
scale is calibrated yet, and the per-stage latency.

If it says `floor: NOT FOUND`, the ground plane fit failed and the whole path
half of the system is inactive -- usually a bad `--hfov`, or a scene with no
visible ground.

### Architecture

`src/assist/core.py` holds `GuidanceCore.update()`, which is pure: detections
and a depth map in, an announcement out. No camera, no audio device, no files.
Everything platform-specific lives in `src/assist/io/`. That boundary is what
makes the behaviour testable without hardware (`pytest tests/assist/`), and it
is the module an Android port would transcribe.

### Known limits

* Distances depend on `--hfov` being roughly right.
* Scale calibration needs something of known height in view. In a bare corridor
  it goes stale, and the system then omits spoken distances rather than stating
  a number it does not trust.
* The planner reasons about geometry only. Flat grass beside a footpath is
  "walkable" to it, so it may steer off a path that a sighted person would
  stay on. Distinguishing surfaces would need terrain segmentation.
* COCO has no class for door, kerb, stair or pole. Only the drop-off layer
  covers any of that, and only where the ground plane fit succeeds.
