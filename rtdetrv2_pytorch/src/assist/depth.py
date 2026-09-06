"""Monocular depth estimation behind a swappable interface.

The default backend is Depth-Anything-V2-Small run through ONNX Runtime. ONNX
rather than PyTorch for two reasons: it is independent of the torch version
this repo is pinned to, and it is the same runtime the eventual Android build
uses, so what gets tuned here is what ships there.

Output is *relative inverse depth* -- larger means nearer, on an arbitrary
scale. Converting that to metres is not this module's job; see calibration.py,
which recovers the scale from the detector's own known-height objects.
"""

import os

import cv2
import numpy as np

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _enable_cuda_dlls():
    """Let ONNX Runtime find the CUDA libraries that ship with torch.

    onnxruntime-gpu built for CUDA 11.8 needs cublas/cudnn on the DLL search
    path. Torch's cu118 wheel already bundles exactly those, so pointing at its
    lib directory avoids a second multi-gigabyte CUDA install.
    """
    try:
        import torch
        lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.isdir(lib) and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(lib)
    except Exception:
        pass


class DepthEstimator:
    """Interface: infer(frame_bgr) -> relative inverse depth at frame size."""

    def infer(self, frame_bgr):
        raise NotImplementedError


class OnnxDepth(DepthEstimator):
    def __init__(self, model_path, input_size=518, device='cuda',
                 gpu_mem_limit_gb=3.0):
        _enable_cuda_dlls()
        import onnxruntime as ort

        # The ViT patch size is 14, so the input side must be a multiple of it.
        self.input_size = int(round(input_size / 14)) * 14

        so = ort.SessionOptions()
        so.log_severity_level = 3

        providers = ['CPUExecutionProvider']
        if device.startswith('cuda'):
            providers = [('CUDAExecutionProvider', {
                'device_id': 0,
                # The default arena grabs a next-power-of-two block, which on an
                # 8 GB laptop GPU already holding the detector fails outright and
                # reports itself as an out-of-memory error.
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': int(gpu_mem_limit_gb * 1024 ** 3),
                'cudnn_conv_algo_search': 'HEURISTIC',
            }), 'CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.provider = self.session.get_providers()[0]

    def infer(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        n = self.input_size

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (n, n), interpolation=cv2.INTER_AREA)
        x = x.astype(np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])

        depth = self.session.run(None, {self.input_name: x})[0]
        depth = np.squeeze(depth)

        # Back to frame resolution so the guidance core can share one
        # coordinate system with the detector's boxes.
        return cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)


class ConstantDepth(DepthEstimator):
    """A flat map, for exercising the pipeline without a depth model."""

    def __init__(self, value=1.0):
        self.value = value
        self.provider = 'constant'

    def infer(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        return np.full((h, w), self.value, dtype=np.float32)


def build_depth(model_path, input_size=518, device='cuda'):
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            'Depth model not found at {!r}. Fetch it with:\n'
            "  python -c \"from huggingface_hub import hf_hub_download as d; "
            "d('onnx-community/depth-anything-v2-small', 'onnx/model.onnx', "
            "local_dir='weights/depth_anything_v2_small')\"".format(model_path))
    return OnnxDepth(model_path, input_size=input_size, device=device)
