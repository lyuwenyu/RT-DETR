# Copyright 2023 lyuwenyu. All Rights Reserved.
# Copyright (c) 2025 Hitbee-dev. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
# NOTICE: This file has been heavily modified by [Hitbee-dev] from the original source.
# Modifications include restructuring for broader GPU architecture compatibility
# (including NVIDIA Blackwell), improved modularity, and enhanced testability.
# ==============================================================================

import time
import numpy as np
import torch
import tensorrt as trt
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont

class TRTInference(object):
    """
    A high-level wrapper for TensorRT inference, designed for ease of use and flexibility.
    This class handles engine loading, context creation, and dynamic buffer allocation.
    """
    def __init__(self, engine_path, device='cuda:0', verbose=False):
        """
        Initializes the TRTInference instance.

        Args:
            engine_path (str): Path to the serialized TensorRT engine file.
            device (str): The device to run inference on (e.g., 'cuda:0').
            verbose (bool): If True, enables verbose logging from the TensorRT logger.
        """
        self.engine_path = engine_path
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
        
        trt.init_libnvinfer_plugins(self.logger, '')
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        self.input_names, self.output_names = self._get_io_names()

        self.buffers_allocated = False
        self.gpu_buffers = OrderedDict()
        self.binding_addrs = OrderedDict()

        print(f"[TRTInference] Initialized successfully. Engine: '{engine_path}'.")

    def _load_engine(self, path):
        """Loads a TensorRT engine from a file."""
        with open(path, 'rb') as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from '{path}'.")
        return engine

    def _get_io_names(self):
        """Parses input and output tensor names from the engine."""
        input_names, output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)
        return input_names, output_names

    def _allocate_buffers(self, blob: dict):
        """
        Allocates GPU buffers for inputs and outputs based on the first inference request.
        This "lazy allocation" strategy handles dynamic input shapes gracefully.
        """
        print("[TRTInference] First inference call detected. Allocating GPU buffers...")
        for name in self.input_names:
            tensor = blob[name]
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
            self.context.set_input_shape(name, shape)
            self.gpu_buffers[name] = torch.empty(shape, dtype=dtype, device=self.device)
            self.binding_addrs[name] = self.gpu_buffers[name].data_ptr()
            print(f"  - Input '{name}': allocated buffer with shape {shape}.")

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = torch.from_numpy(np.array(0, dtype=dtype)).dtype
            self.gpu_buffers[name] = torch.empty(shape, dtype=torch_dtype, device=self.device)
            self.binding_addrs[name] = self.gpu_buffers[name].data_ptr()
            print(f"  - Output '{name}': allocated buffer with shape {shape}.")

        self.buffers_allocated = True
        print("[TRTInference] GPU buffers allocated successfully.")

    def __call__(self, blob: dict):
        """
        Executes inference on the loaded TensorRT engine.

        Args:
            blob (dict): A dictionary mapping input tensor names to their corresponding
                         torch.Tensor data on the GPU.

        Returns:
            dict: A dictionary mapping output tensor names to their corresponding
                  torch.Tensor results on the GPU.
        """
        if not self.buffers_allocated:
            self._allocate_buffers(blob)
            
        for name in self.input_names:
            self.gpu_buffers[name].copy_(blob[name])

        self.context.execute_v2(bindings=list(self.binding_addrs.values()))
        
        return {name: self.gpu_buffers[name] for name in self.output_names}

# --- Visualization Utility Function ---
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def visualize_detections(image_pil, boxes, scores, labels, class_names=COCO_CLASSES, threshold=0.5):
    """
    Draws bounding boxes on a PIL image. This function is a general-purpose utility.

    Args:
        image_pil (PIL.Image.Image): The image to draw on.
        boxes (torch.Tensor): A tensor of bounding boxes (shape: [N, 4]).
        scores (torch.Tensor): A tensor of confidence scores (shape: [N]).
        labels (torch.Tensor): A tensor of class labels (shape: [N]).
        class_names (list): A list of strings corresponding to class labels.
        threshold (float): The confidence threshold for displaying detections.

    Returns:
        PIL.Image.Image: The image with detections drawn on it.
    """
    img_draw = image_pil.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Ensure tensors are on CPU and converted to NumPy for processing
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    
    count = 0
    for i in range(len(scores)):
        score = scores[i]
        if score < threshold:
            continue
        
        count += 1
        box = boxes[i]
        label_idx = int(labels[i])
        
        xmin, ymin, xmax, ymax = box
        class_name = class_names[label_idx] if label_idx < len(class_names) else f'CLS-{label_idx}'
        color = 'red' # Keep it simple or use a color map
        
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=3)
        
        text = f"{class_name}: {score:.2f}"
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((xmin, ymin), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((xmin, ymin), text, fill="white", font=font)
        
    print(f"   - Found {count} objects above threshold {threshold}.")
    return img_draw

if __name__ == '__main__':
    import argparse
    import torchvision.transforms as T
    import os

    parser = argparse.ArgumentParser(description="Test script for the TRTInference wrapper.")
    parser.add_argument('--engine', type=str, required=True, help="Path to the TensorRT engine file.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image file.")
    parser.add_argument('--output', type=str, default='output.jpg', help="Path to save the output image with detections.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to run inference on.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Confidence threshold for displaying detections.")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This script requires a GPU.")
    
    print("--- TRTInference Wrapper Test ---")
    
    print("\n1. Initializing TRTInference...")
    trt_model = TRTInference(args.engine, device=args.device)
    
    print("\n2. Preprocessing input image...")
    image_pil = Image.open(args.image).convert('RGB')
    w, h = image_pil.size
    
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    image_tensor = transforms(image_pil).unsqueeze(0).to(args.device)
    orig_size_tensor = torch.tensor([[w, h]], dtype=torch.int64, device=args.device)

    blob = {
        'images': image_tensor,
        'orig_target_sizes': orig_size_tensor
    }
    print(f"   - Original image size: {w}x{h}")
    print(f"   - Input tensor shape: {image_tensor.shape}")

    print("\n3. Running inference...")
    start_time = time.time()
    output_gpu = trt_model(blob)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"\n4. Inference complete in { (end_time - start_time) * 1000:.2f} ms.")
    
    print("\n5. Post-processing and saving output image...")
    output_labels = output_gpu['labels'][0]
    output_boxes = output_gpu['boxes'][0]
    output_scores = output_gpu['scores'][0]
    
    # Use the new, separate visualization function
    result_image = visualize_detections(
        image_pil, 
        output_boxes, 
        output_scores, 
        output_labels, 
        threshold=args.threshold
    )
    
    result_image.save(args.output)
    print(f"   - Output image with detections saved to: {os.path.abspath(args.output)}")

    print("\n--- Test finished successfully ---")