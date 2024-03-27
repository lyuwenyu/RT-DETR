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
    if s[3] == "m":
        model_name += "_m"
    model_name += "_6x_coco.yml"
    return model_name


def convert_onnx(checkpoint):
    model_name = get_model_name(checkpoint)
    config = f"rtdetr_pytorch/configs/rtdetr/{model_name}"
    cmd = f"python rtdetr_pytorch/tools/export_onnx.py -c {os.path.abspath(config)} -r {os.path.abspath(checkpoint)} --check"
    subprocess.run(cmd.split())


convert_onnx(args.checkpoint)


images = list(Path("data/COCO2017/val2017/img").glob("*.jpg"))

session = onnxruntime.InferenceSession("model.onnx", providers=providers)


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


# average tests
import pandas as pd
df = pd.DataFrame(speed_tests)
avg_df = df.mean().to_frame().T
std_df = df.std().to_frame().T
std_df.columns = [f"{col}_std" for col in std_df.columns]
df = pd.concat([avg_df, std_df], axis=1)
df.to_csv(f"speed_test_{get_model_name(args.checkpoint)}.csv")

print(f"Average speed test (N={args.n}):")
print(avg_df)
print("Std speed test:")
print(std_df)
