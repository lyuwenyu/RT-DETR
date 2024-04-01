import re
import subprocess
from itertools import cycle
import os
from pathlib import Path
from time import time
import torch
import argparse
import onnxruntime
from PIL import Image
from torchvision.transforms import ToTensor
# from cachetools import cached, LRUCache
from tqdm import tqdm


argparser = argparse.ArgumentParser()
argparser.add_argument("--checkpoint", type=str, default="models/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth")
argparser.add_argument("--n", type=int, default=None)
argparser.add_argument("--cpu", action="store_true")
args = argparser.parse_args()
print(args)

if not args.cpu:
    assert torch.cuda.is_available(), "CUDA is not available"
    providers = ["CUDAExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]


# @cached(cache=LRUCache(maxsize=128))
def prepare_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((640, 640))
    img_tensor = ToTensor()(img)[None].numpy()
    size = torch.tensor([[640, 640]]).numpy()
    return img_tensor, size


def get_model_name(checkpoint):
    s = os.path.basename(checkpoint).split("_")
    model_name = "_".join(s[:2])
    if s[2] == "m":
        model_name += "_m"
    model_name += "_6x_coco.yml"
    return model_name


def convert_onnx(checkpoint):
    model_name = get_model_name(checkpoint)
    config = f"rtdetr_pytorch/configs/rtdetr/{model_name}"
    output_model = os.path.abspath(f"models/{model_name.replace('.yml', '.onnx')}")
    if os.path.exists(output_model):
        print(f"ONNX model {output_model} already exists.")
        return output_model
    cmd = f"python rtdetr_pytorch/tools/export_onnx.py -c {os.path.abspath(config)} -r {os.path.abspath(checkpoint)} -f {output_model} --check"
    subprocess.run(cmd.split())
    return output_model


def get_gpu_memory_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')    
    memory_usages = re.findall(r'\d+', output)
    memory_usages = [int(x) for x in memory_usages]
    assert len(memory_usages) == 1, "Multiple GPUs are not supported"
    return memory_usages[0]


gpu_memory_before = get_gpu_memory_usage()

onnx_model_path = convert_onnx(args.checkpoint)
session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

images = list(Path("data/COCO2017/val2017/img").glob("*.jpg"))

# warmup 3 times
for img_path in images[:3]:
    img, size = prepare_image(img_path)
    session.run(output_names=None, input_feed={'images': img, "orig_target_sizes": size})


speed_tests = []
for img_path in tqdm(images[:args.n]):
    img, size = prepare_image(img_path)
    t0 = time()
    session.run(output_names=None, input_feed={'images': img, "orig_target_sizes": size})
    dt_ms = (time() - t0) * 1000
    speed_tests.append({"inference_time": dt_ms})


gpu_memory_after = get_gpu_memory_usage()
gpu_memory = gpu_memory_after - gpu_memory_before

# average tests
import pandas as pd
df = pd.DataFrame(speed_tests)
avg_df = df.mean().to_frame().T
std_df = df.std().to_frame().T
std_df.columns = [f"{col}_std" for col in std_df.columns]
df = pd.concat([avg_df, std_df], axis=1)
df["gpu_memory_usage"] = gpu_memory
df.to_csv(f"speed_test_v2_{get_model_name(args.checkpoint)}.csv")

print(df)