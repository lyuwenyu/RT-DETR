import os
import time
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List
from PIL import Image, ImageOps
import torch
from torchvision.transforms import ToTensor
import onnxruntime

from dotenv import load_dotenv
import supervisely as sly
from supervisely.nn.prediction_dto import PredictionBBox
from supervisely.app.widgets import (
    Widget,
    PretrainedModelsSelector,
    CustomModelsSelector,
    RadioTabs,
    Container,
    SelectString,
    Field
)

from src.solver import DetSolver
from src.core import YAMLConfig
from src.data.transforms import Resize, ToImageTensor, ConvertDtype, Compose
from src.data.coco.coco_dataset import mscoco_category2name
from model_list import get_models


DEFAULT_CONF = 0.4

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))
root_dir = Path(__file__).parent.parent


class RTDETR(sly.nn.inference.ObjectDetection):
    def initialize_custom_gui(self) -> Widget:
        """Create custom GUI layout for model selection. This method is called once when the application is started."""
        models = get_models()
        self.pretrained_models_table = PretrainedModelsSelector(models)
        team_id = sly.env.team_id()
        self.custom_models_table = CustomModelsSelector(
            team_id,
            checkpoints=[],
            show_custom_checkpoint_path=True
        )
        self.model_source_tabs = RadioTabs(
            titles=["Pretrained models", "Custom models"],
            descriptions=["Publicly available models", "Models trained by you in Supervisely"],
            contents=[self.pretrained_models_table, self.custom_models_table],
        )
        self.runtime_select = SelectString(["PyTorch", "ONNXRuntime", "TensorRT"])
        runtime_field = Field(self.runtime_select, "Runtime", "Select a runtime for inference.")
        layout = Container([self.model_source_tabs, runtime_field])
        return layout
    
    def get_params_from_gui(self) -> dict:
        model_source = self.model_source_tabs.get_active_tab()
        device = self.gui.get_device()
        runtime = self.runtime_select.get_value()
        if model_source == "Pretrained models":
            model_idx = self.pretrained_models_table.get_selected_row_index()
            checkpoint_path = None
        elif model_source == "Custom models":
            model_idx = None
            if self.custom_models_table.use_custom_checkpoint_path():
                checkpoint_path = self.custom_models_table.get_custom_checkpoint_path()
            else:
                raise NotImplementedError()
        load_model_args = {
            "pretrained_model_idx": model_idx,
            "custom_checkpoint_path": checkpoint_path,
            "device": device,
            "runtime": runtime,
        }
        return load_model_args

    def load_model(
        self,
        pretrained_model_idx: int,
        custom_checkpoint_path: str,
        device: str,
        runtime: str,
    ):
        self.device = device
        self.runtime = runtime

        # 1. download
        if pretrained_model_idx is not None:
            checkpoint_path, config_path = self._download_pretrained_model(pretrained_model_idx)
            self._load_meta_pretained_model(pretrained_model_idx)
        elif custom_checkpoint_path is not None:
            checkpoint_path, config_path = self._download_custom_model(custom_checkpoint_path)
            self._load_meta_custom_model(config_path)
        else:
            raise ValueError("Both pretrained_model_idx and custom_checkpoint_path are None.")
        
        # 2. load model
        if runtime == "PyTorch":
            self._load_pytorch(checkpoint_path, config_path, device)
        elif runtime == "ONNXRuntime":
            # runtime = ONNX and weights is .pth
            from convert_onnx import convert_onnx
            onnx_model_path = convert_onnx(checkpoint_path, config_path)
            self._load_onnx(onnx_model_path, device)
        else:
            raise NotImplementedError()
        
        # 3. load meta
        if pretrained_model_idx is not None:
            self._load_meta_pretained_model(pretrained_model_idx)
        elif custom_checkpoint_path is not None:
            self._load_meta_custom_model(config_path)

    def _download_pretrained_model(self, model_idx: int):
        model_dict = get_models()[model_idx]
        model = model_dict["name"]
        checkpoint_url = model_dict["meta"]["url"]
        arch = model.split('_coco')[0]
        config_name = f"{arch}_6x_coco.yml"
        config_path = f'{root_dir}/rtdetr_pytorch/configs/rtdetr/{config_name}'
        _ = torch.hub.load_state_dict_from_url(checkpoint_url, self.model_dir)
        name = os.path.basename(checkpoint_url)
        checkpoint_path = f"{self.model_dir}/{name}"
        return checkpoint_path, config_path

    def _download_custom_model(self, custom_checkpoint_path):
        # download weights (.pth)
        weight_filename = os.path.basename(custom_checkpoint_path)
        weights_dst_path = os.path.join(self.model_dir, weight_filename)
        if not sly.is_debug_with_sly_net() or (sly.is_debug_with_sly_net() and not os.path.exists(weights_dst_path)):
            self.download(
                src_path=custom_checkpoint_path,
                dst_path=weights_dst_path,
            )
        # download config.py
        custom_dir = os.path.dirname(custom_checkpoint_path)
        config_path = self.download(
            src_path=os.path.join(custom_dir, "config.yml"),
            dst_path=os.path.join(self.model_dir, "config.yml"),
        )
        # del "__include__" and rewrite the config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if "__include__" in config:
            config.pop("__include__")
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
        return weights_dst_path, config_path

    def _load_pytorch(self, checkpoint_path, config_path, device):
        cfg = YAMLConfig(
            config_path,
            resume=checkpoint_path,
            tuning='',
        )
        # TODO: Why is this not set while training?
        cfg.yaml_cfg["HybridEncoder"]["eval_spatial_size"] = self.img_size
        cfg.yaml_cfg["RTDETRTransformer"]["eval_spatial_size"] = self.img_size
        solver = DetSolver(cfg)
        solver.setup()
        solver.resume(solver.cfg.resume)
        self.solver = solver
        self.model = solver.ema.module if solver.is_ema_loaded else solver.model
        self.model.eval()
        self.transform = Compose(ops=[
            Resize(self.img_size),
            ToImageTensor(),
            ConvertDtype()
        ])

    def _load_onnx(self, onnx_model_path, device):
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            assert torch.cuda.is_available(), "CUDA is not available"
            providers = ["CUDAExecutionProvider"]
        self.onnx_session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

    def _load_meta_pretained_model(self, model_idx: int):
        model_dict = get_models()[model_idx]
        self.model_name = model_dict["name"]
        self.dataset_name = model_dict["dataset"]
        self.class_names = list(mscoco_category2name.values())
        self.img_size = [640, 640]
        self._load_obj_classes(self.class_names)
    
    def _load_meta_custom_model(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        meta = config["sly_metadata"]
        self.model_name = meta["model"]
        self.dataset_name = meta["project_name"]
        self.class_names = meta["classes"]
        self.img_size = config["val_dataloader"]["dataset"]["transforms"]["ops"][0]["size"]
        self._load_obj_classes(self.class_names)
    
    def _load_obj_classes(self, class_names: List[str]):
        obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in class_names]
        self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
        self._get_confidence_tag_meta()

    def predict(self, image_path: str, settings: dict) -> List[PredictionBBox]:
        conf_tresh = settings.get("confidence_thresh", DEFAULT_CONF)
        if self.runtime == "PyTorch":
            classes, boxes, scores = self._predict_pytorch(image_path, settings)
        elif self.runtime == "ONNXRuntime":
            classes, boxes, scores = self._predict_onnx(image_path, settings)
        else:
            raise NotImplementedError()
        predictions = self._format_predictions(classes, boxes, scores, conf_tresh)
        return predictions

    def _predict_pytorch(self, image_path: str, settings: dict = None):
        img = self._read_image(image_path)
        w, h = img.size
        orig_target_sizes = torch.as_tensor([int(w), int(h)]).unsqueeze(0)
        postprocessors = self.solver.postprocessor
        with torch.no_grad():
            samples = self.transform(img)[None]
            samples = samples.to(self.device)
            orig_target_sizes = orig_target_sizes.to(self.device)
            outputs = self.model(samples)
            results = postprocessors(outputs, orig_target_sizes)
        results = results[0]
        if not postprocessors.remap_mscoco_category:
            classes = [self.class_names[i] for i in results["labels"].cpu().numpy()]
        else:
            classes = [mscoco_category2name[i] for i in results["labels"].cpu().numpy()]
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        return classes, boxes, scores
    
    def _predict_onnx(self, image_path: str, settings: dict = None):
        img = self._read_image(image_path)
        w, h = img.size
        def prepare_image(img, img_size):
            img = img.resize(tuple(img_size))
            img_tensor = ToTensor()(img)[None].numpy()
            size = np.array([list(img_size)], dtype=int)
            return img_tensor, size
        img_input, size_input = prepare_image(img, self.img_size)
        ## Profile start
        labels, boxes, scores = self.onnx_session.run(output_names=None, input_feed={'images': img_input, "orig_target_sizes": size_input})
        ## Profile end
        labels, boxes, scores = labels[0], boxes[0], scores[0]
        boxes_orig = boxes / np.array(self.img_size*2) * np.array([w, h, w, h])
        classes = [self.class_names[label] for label in labels]
        return classes, boxes_orig, scores
    
    def _format_predictions(self, class_names, boxes: np.ndarray, scores, conf_tresh):
        predictions = []
        for class_name, bbox_xyxy, score in zip(class_names, boxes, scores):
            if score < conf_tresh:
                continue
            bbox_xyxy = np.round(bbox_xyxy).astype(int)
            bbox_xyxy = np.clip(bbox_xyxy, 0, None)
            bbox_yxyx = [bbox_xyxy[1], bbox_xyxy[0], bbox_xyxy[3], bbox_xyxy[2]]
            bbox_yxyx = list(map(int, bbox_yxyx))
            predictions.append(PredictionBBox(class_name, bbox_yxyx, float(score)))
        return predictions

    def get_info(self):
        info = super().get_info()
        info["model_name"] = self.model_name
        info["pretrained_dataset"] = self.dataset_name
        info["device"] = self.device
        return info

    def get_classes(self) -> List[str]:
        return self.class_names

    def _read_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        try:
            img = ImageOps.exif_transpose(img)
        except:
            pass
        return img


model = RTDETR(
    use_gui=True,
    custom_inference_settings={
        "confidence_thresh": 0.4
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