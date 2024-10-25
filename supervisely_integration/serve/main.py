import os
import numpy as np
from dotenv import load_dotenv

import supervisely as sly
from rtdetr_pytorch.serve import RTDETR

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

source_path = __file__
settings_path = os.path.join(os.path.dirname(source_path), "inference_settings.yaml")

model = RTDETR(
    use_gui=True,
    custom_inference_settings=settings_path,
)

if True:
    model.serve()
else:
    # Test
    images_np = [np.array(Image.open("rtdetr_pytorch/image_02.jpg"))]
    model.load_model(0, None, "cuda", "ONNXRuntime")
    model.predict_benchmark(images_np, settings={})
