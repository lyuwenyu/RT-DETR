from typing import Dict, List
import numpy as np
import supervisely as sly
from supervisely.nn.prediction_dto import PredictionBBox
import torch
import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
from src.solver import DetSolver
from src.core import YAMLConfig
from src.data.transforms import Resize, ToImageTensor, ConvertDtype, Compose
from src.data.coco.coco_dataset import mscoco_category2name
from model_list import get_models
from PIL import Image, ImageOps


if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))
root_dir = Path(__file__).parent.parent


class RTDETR(sly.nn.inference.ObjectDetection):
    def load_on_device(self, model_dir: str, device: str = "cpu"):
        gui = self.gui
        if gui.get_model_source() == "Pretrained models":
            idx = self.gui._models_table.get_selected_row_index()
            model_dict = get_models()[idx]
            model = model_dict["name"]
            checkpoint_url = model_dict["meta"]["url"]
            arch = model.split('_coco')[0]
            config_name = f"{arch}_6x_coco.yml"
            config_path = f'{root_dir}/rtdetr_pytorch/configs/rtdetr/{config_name}'
            size = [640, 640]
            dataset_name = model_dict["dataset"]
            class_names = list(mscoco_category2name.values())

            _ = torch.hub.load_state_dict_from_url(checkpoint_url, model_dir)
            name = os.path.basename(checkpoint_url)
            checkpoint_path = f"{model_dir}/{name}"
        else:
            custom_weights_link = self.gui.get_custom_link()
            checkpoint_path, config_path = self.download_custom_files(custom_weights_link, model_dir)
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            size = config["val_dataloader"]["dataset"]["transforms"]["ops"][0]["size"]
            meta = config["sly_metadata"]
            model = meta["model"]
            dataset_name = meta["project_name"]
            class_names = meta["classes"]
        
        self.model_name = model
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.device = device

        obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in class_names]
        self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
        
        cfg = YAMLConfig(
            config_path,
            resume=checkpoint_path,
        )

        solver = DetSolver(cfg)
        solver.setup()
        solver.resume(solver.cfg.resume)
        self.solver = solver
        self.model = solver.ema.module if solver.ema else solver.model
        self.model.eval()

        self.transform = Compose(ops=[
            Resize(size),
            ToImageTensor(),
            ConvertDtype()
        ])
        
    def get_info(self):
        info = super().get_info()
        info["model_name"] = self.model_name
        info["pretrained_dataset"] = self.dataset_name
        info["device"] = self.device
        return info

    def get_classes(self) -> List[str]:
        return self.class_names
    
    def get_models(self):
        # list of dicts
        models = get_models()
        for m in models:
            m.pop("meta")
        return models

    def predict(self, image_path: str, settings: dict) -> List[PredictionBBox]:
        conf_tresh = settings.get("confidence_thresh", 0.45)

        img = Image.open(image_path).convert("RGB")
        try:
            img = ImageOps.exif_transpose(img)
        except:
            pass
        w, h = img.size
        orig_target_sizes = torch.as_tensor([int(w), int(h)]).unsqueeze(0)
        postprocessors = self.solver.postprocessor
        with torch.no_grad():
            samples = self.transform(img)[None]
            samples = samples.to(self.device)
            orig_target_sizes = orig_target_sizes.to(self.device)
            outputs = self.model(samples)
            results = postprocessors(outputs, orig_target_sizes)
        predictions = []
        results = results[0]
        if not postprocessors.remap_mscoco_category:
            classes = [self.class_names[i] for i in results["labels"].cpu().numpy()]
        else:
            classes = [mscoco_category2name[i] for i in results["labels"].cpu().numpy()]
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        for class_name, bbox_xyxy, score in zip(classes, boxes, scores):
            if score < conf_tresh:
                continue
            bbox_yxyx = np.array([bbox_xyxy[1], bbox_xyxy[0], bbox_xyxy[3], bbox_xyxy[2]])
            bbox_yxyx = np.clip(bbox_yxyx, 0, None)
            bbox_yxyx = np.round(bbox_yxyx).astype(int).tolist()
            predictions.append(PredictionBBox(class_name, bbox_yxyx, float(score)))
        return predictions
    
    def download_custom_files(self, custom_link: str, model_dir: str):
        # download weights (.pth)
        weight_filename = os.path.basename(custom_link)
        weights_dst_path = os.path.join(model_dir, weight_filename)
        self.download(
            src_path=custom_link,
            dst_path=weights_dst_path,
        )

        # download config.py
        custom_dir = os.path.dirname(custom_link)
        config_path = self.download(
            src_path=os.path.join(custom_dir, "config.yml"),
            dst_path=os.path.join(model_dir, "config.yml"),
        )

        return weights_dst_path, config_path


model = RTDETR(
    use_gui=True,
    custom_inference_settings={
        "confidence_thresh": 0.45
    },
)

if True:
    model.serve()
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    for i in range(len(model.get_models())):
        if i <= 4:
            continue
        model.gui._models_table.select_row(i)
        model.load_on_device("models", device)
        image_path = "rtdetr_pytorch/image_02.jpg"
        results = model.predict(image_path, settings={})
        vis_path = f"image_02_prediction_{i}.jpg"
        model.visualize(results, image_path, vis_path, thickness=5)
        print(f"predictions and visualization have been saved: {vis_path}")