import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
sys.path.append(current_dir)

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime
import torch
import yaml
from dotenv import load_dotenv
from model_list import get_models
from PIL import Image
from src.core import YAMLConfig
from src.data.coco.coco_dataset import mscoco_category2name
from src.data.transforms import Compose, ConvertDtype, Resize, ToImageTensor
from src.solver import DetSolver
from torchvision.transforms import ToTensor

import supervisely as sly
import supervisely_integration.serve.workflow as w
from supervisely.app.widgets import (
    Container,
    CustomModelsSelector,
    Field,
    PretrainedModelsSelector,
    RadioTabs,
    SelectString,
    Widget,
)
from supervisely.io.fs import get_file_name
from supervisely.nn.artifacts.rtdetr import RTDETR as RTDETRArtifacts
from supervisely.nn.inference import CheckpointInfo, Timer
from supervisely.nn.prediction_dto import PredictionBBox

DEFAULT_CONF = 0.4

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))
root_dir = Path(__file__).parent.parent


class PyTorchInference:
    def __init__(self, checkpoint_path, config_path, device, img_size, class_names):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        self.img_size = img_size
        self.class_names = class_names

    def load_model(self):
        cfg = YAMLConfig(
            self.config_path,
            resume=self.checkpoint_path,
            tuning="",
        )
        # TODO: Why is eval_spatial_size not set while training?
        cfg.yaml_cfg["HybridEncoder"]["eval_spatial_size"] = self.img_size
        cfg.yaml_cfg["RTDETRTransformer"]["eval_spatial_size"] = self.img_size
        solver = DetSolver(cfg)
        solver.setup()
        solver.resume(solver.cfg.resume)
        self.solver = solver
        self.postprocessor = solver.postprocessor
        self.model = solver.ema.module if solver.is_ema_loaded else solver.model
        self.model.eval()
        self.model.to(self.device)
        self.transform = Compose(ops=[Resize(self.img_size), ToImageTensor(), ConvertDtype()])

    def predict_benchmark(
        self, images_np: List[np.ndarray], settings: dict
    ) -> Tuple[List[List[PredictionBBox]], dict]:
        # 1. Preprocess
        with Timer() as preprocess_timer:
            imgs_pil = [Image.fromarray(img) for img in images_np]
            orig_target_sizes = torch.as_tensor([img.size for img in imgs_pil]).to(self.device)
            samples = torch.stack(self.transform(imgs_pil)).to(self.device)
        # 2. Inference
        with Timer() as inference_timer:
            with torch.no_grad():
                outputs = self.model(samples)
        # 3. Postprocess
        with Timer() as postprocess_timer:
            results = self.postprocessor(outputs, orig_target_sizes)
            predictions = []
            for res in results:
                if not self.postprocessor.remap_mscoco_category:
                    classes = [self.class_names[i] for i in res["labels"].cpu().numpy()]
                else:
                    classes = [mscoco_category2name[i] for i in res["labels"].cpu().numpy()]
                boxes = res["boxes"].cpu().numpy()
                scores = res["scores"].cpu().numpy()
                conf_tresh = settings.get("confidence_threshold", DEFAULT_CONF)
                predictions.append(format_prediction(classes, boxes, scores, conf_tresh))
        benchmark = {
            "preprocess": preprocess_timer.get_time(),
            "inference": inference_timer.get_time(),
            "postprocess": postprocess_timer.get_time(),
        }
        return predictions, benchmark

    def shutdown(self):
        del self.model
        del self.solver


class ONNXInference:
    def __init__(self, onnx_model_path, device, img_size, class_names):
        self.onnx_model_path = onnx_model_path
        self.device = device
        self.img_size = list(img_size)
        self.class_names = class_names

    def load_model(self):
        if self.device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            assert torch.cuda.is_available(), "CUDA is not available"
            providers = ["CUDAExecutionProvider"]
        self.onnx_session = onnxruntime.InferenceSession(self.onnx_model_path, providers=providers)

    def predict_benchmark(
        self, images_np: List[np.ndarray], settings: dict
    ) -> Tuple[List[List[PredictionBBox]], dict]:
        # 1. Preprocess
        with Timer() as preprocess_timer:
            imgs = []
            orig_sizes = []
            for img_np in images_np:
                img = Image.fromarray(img_np)
                orig_sizes.append(list(img.size))
                img = img.resize(tuple(self.img_size))
                img = ToTensor()(img)[None].numpy()
                imgs.append(img)
            img_input = np.concatenate(imgs, axis=0)
            size_input = np.array(self.img_size * len(images_np), dtype=int).reshape(-1, 2)
        # 2. Inference
        with Timer() as inference_timer:
            labels, boxes, scores = self.onnx_session.run(
                output_names=None,
                input_feed={"images": img_input, "orig_target_sizes": size_input},
            )
        # 3. Postprocess
        with Timer() as postprocess_timer:
            predictions = []
            for i, (labels, boxes, scores) in enumerate(zip(labels, boxes, scores)):
                w, h = orig_sizes[i]
                boxes_orig = boxes / np.array(self.img_size * 2) * np.array([w, h, w, h])
                classes = [self.class_names[label] for label in labels]
                conf_tresh = settings.get("confidence_threshold", DEFAULT_CONF)
                predictions.append(format_prediction(classes, boxes_orig, scores, conf_tresh))
        benchmark = {
            "preprocess": preprocess_timer.get_time(),
            "inference": inference_timer.get_time(),
            "postprocess": postprocess_timer.get_time(),
        }
        return predictions, benchmark

    def shutdown(self):
        # at the moment, onnxruntime does not have a method to release resources
        # see https://github.com/microsoft/onnxruntime/issues/17142
        del self.onnx_session

    def _prepare_image(self, img: Image.Image, img_size: list):
        img = img.resize(tuple(img_size))
        img_tensor = ToTensor()(img)[None].numpy()
        size = np.array([list(img_size)], dtype=int)
        return img_tensor, size


class RTDETR(sly.nn.inference.ObjectDetection):
    team_id = sly.env.team_id()
    in_train = False

    def initialize_custom_gui(self) -> Widget:
        """Create custom GUI layout for model selection. This method is called once when the application is started."""
        models = get_models()
        self.pretrained_models_table = PretrainedModelsSelector(models)
        team_id = self.team_id

        custom_models = RTDETRArtifacts(team_id).get_list()
        self.custom_models_table = CustomModelsSelector(
            team_id, train_infos=custom_models, show_custom_checkpoint_path=True
        )
        self.model_source_tabs = RadioTabs(
            titles=["Pretrained models", "Custom models"],
            descriptions=[
                "Publicly available models",
                "Models trained by you in Supervisely",
            ],
            contents=[self.pretrained_models_table, self.custom_models_table],
        )
        self.runtime_select = SelectString(["PyTorch", "ONNXRuntime"])  # @TODO: "TensorRT"
        runtime_field = Field(self.runtime_select, "Runtime", "Select a runtime for inference.")
        layout = Container([self.model_source_tabs, runtime_field])
        return layout

    def get_params_from_gui(self) -> dict:
        model_source = self.model_source_tabs.get_active_tab()
        device = self.gui.get_device()
        runtime = self.runtime_select.get_value()
        if model_source == "Pretrained models":
            model_params = self.pretrained_models_table.get_selected_model_params()
        elif model_source == "Custom models":
            model_params = self.custom_models_table.get_selected_model_params()
        else:
            raise NotImplementedError()
        load_model_args = {
            "device": device,
            "runtime": runtime,
            **model_params,
        }

        # -------------------------------------- Add Workflow Input -------------------------------------- #
        if not self.in_train:
            w.workflow_input(self.api, model_params)
        # ----------------------------------------------- - ---------------------------------------------- #

        return load_model_args

    def load_model(
        self,
        device: str,
        model_source: str,
        task_type: str,
        checkpoint_name: str,
        checkpoint_url: str,
        runtime: str,
        config_url: Optional[str] = None,
    ):
        """
        Load model method is used to deploy model.

        :param device: The device on which the model will be deployed.
        :type device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        :param model_source: Specifies whether the model is pretrained or custom.
        :type model_source: Literal["Pretrained models", "Custom models"]
        :param task_type: The type of the computer vision task the model is designed for.
        :type task_type: Literal["object detection"]
        :param checkpoint_name: The name of the checkpoint from which the model is loaded.
        :type checkpoint_name: str
        :param checkpoint_url: The URL where the model checkpoint can be downloaded.
        :type checkpoint_url: str
        :param runtime: The runtime used for inference. Supported runtimes are PyTorch, ONNXRuntime, and TensorRT.
        :type runtime: str
        :param config_url: The URL where the model config can be downloaded.
        :type config_url: Optional[str]
        """

        self.device = device
        self.task_type = task_type
        self.runtime = runtime
        self.model_source = model_source

        # 1. download
        if model_source == "Pretrained models":
            checkpoint_path, config_path = self._download_pretrained_model(
                checkpoint_name, checkpoint_url
            )
            self._load_meta_pretained_model(checkpoint_name)
        elif model_source == "Custom models":
            checkpoint_path, config_path = self._download_custom_model(
                checkpoint_name, checkpoint_url, config_url
            )
            self._load_meta_custom_model(config_path)
        else:
            raise ValueError("Both pretrained_model_idx and custom_checkpoint_path are None.")

        # 2. load model
        if self.runtime == "PyTorch":
            self.pytorch_inference = PyTorchInference(
                checkpoint_path, config_path, device, self.img_size, self.class_names
            )
            self.pytorch_inference.load_model()
        elif self.runtime == "ONNXRuntime":
            # when runtime is ONNX and weights is .pth
            from convert_onnx import convert_onnx

            onnx_model_path = convert_onnx(checkpoint_path, config_path)
            self.onnx_inference = ONNXInference(
                onnx_model_path, device, self.img_size, self.class_names
            )
            self.onnx_inference.load_model()
        else:
            raise NotImplementedError()

        # 3. load meta
        if self.model_source == "Pretrained models":
            self._load_meta_pretained_model(checkpoint_name)
        elif self.model_source == "Custom models":
            self._load_meta_custom_model(config_path)

        self.checkpoint_info = CheckpointInfo(
            self.model_name,
            "RT-DETR",
            self.model_source,
        )

    def _download_pretrained_model(self, checkpoint_name: str, checkpoint_url: str):
        model = checkpoint_name
        checkpoint_url = checkpoint_url
        arch = model.split("_coco")[0]
        config_name = f"{arch}_6x_coco.yml"
        config_path = f"{root_dir}/rtdetr_pytorch/configs/rtdetr/{config_name}"
        _ = torch.hub.load_state_dict_from_url(checkpoint_url, self.model_dir)
        name = os.path.basename(checkpoint_url)
        checkpoint_path = f"{self.model_dir}/{name}"
        return checkpoint_path, config_path

    def _download_custom_model(self, checkpoint_name: str, checkpoint_url: str, config_url: str):
        # download weights (.pth)
        weight_filename = checkpoint_name
        weights_dst_path = os.path.join(self.model_dir, weight_filename)
        if not sly.is_debug_with_sly_net() or (
            sly.is_debug_with_sly_net() and not os.path.exists(weights_dst_path)
        ):
            self.download(
                src_path=checkpoint_url,
                dst_path=weights_dst_path,
            )
        # download config.yml
        local_config_path = os.path.join(os.path.dirname(self.model_dir), "config.yml")
        config_path = self.download(
            src_path=config_url,
            dst_path=local_config_path,
        )
        # del "__include__" and rewrite the config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "__include__" in config:
            config.pop("__include__")
            with open(config_path, "w") as f:
                yaml.dump(config, f)
        return weights_dst_path, config_path

    def _load_meta_pretained_model(self, checkpoint_name: str):
        checkpoint_name = get_file_name(checkpoint_name)
        models = get_models()
        model_dict = None
        for model in models:
            if model["Model"] == checkpoint_name:
                model_dict = model
                break
        if model_dict is None:
            raise ValueError(f"Model {checkpoint_name} not found in the list of models.")

        self.model_name = model_dict["Model"]
        self.dataset_name = model_dict["dataset"]
        self.class_names = list(mscoco_category2name.values())
        self.img_size = [640, 640]
        self._load_obj_classes(self.class_names)

    def _load_meta_custom_model(self, config_path):
        with open(config_path, "r") as f:
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

    def predict_benchmark(
        self, images_np: List[np.ndarray], settings: dict
    ) -> Tuple[List[List[PredictionBBox]], dict]:
        if self.runtime == "PyTorch":
            predictions, benchmark = self.pytorch_inference.predict_benchmark(images_np, settings)
        elif self.runtime == "ONNXRuntime":
            predictions, benchmark = self.onnx_inference.predict_benchmark(images_np, settings)
        else:
            raise NotImplementedError()
        return predictions, benchmark

    def get_info(self):
        info = super().get_info()
        info["model_name"] = self.model_name
        info["pretrained_dataset"] = self.dataset_name
        info["device"] = self.device
        return info

    def get_classes(self) -> List[str]:
        return self.class_names

    def shutdown_model(self):
        if self.runtime == "PyTorch":
            self.pytorch_inference.shutdown()
            del self.pytorch_inference
        elif self.runtime == "ONNXRuntime":
            self.onnx_inference.shutdown()
            del self.onnx_inference
        super().shutdown_model()


def format_prediction(
    classes: list, boxes: np.ndarray, scores: list, conf_tresh: float
) -> List[PredictionBBox]:
    predictions = []
    for class_name, bbox_xyxy, score in zip(classes, boxes, scores):
        if score < conf_tresh:
            continue
        bbox_xyxy = np.round(bbox_xyxy).astype(int)
        bbox_xyxy = np.clip(bbox_xyxy, 0, None)
        bbox_yxyx = [bbox_xyxy[1], bbox_xyxy[0], bbox_xyxy[3], bbox_xyxy[2]]
        bbox_yxyx = list(map(int, bbox_yxyx))
        predictions.append(PredictionBBox(class_name, bbox_yxyx, float(score)))
    return predictions
