import os
import shutil
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import yaml
from pycocotools.coco import COCO

from supervisely.nn.artifacts.artifacts import TrainInfo
from supervisely.io.json import dump_json_file
from dataclasses import asdict

import rtdetr_pytorch.train as train_cli
import supervisely as sly
import supervisely_integration.train.globals as g
import supervisely_integration.train.ui.classes as classes_ui
import supervisely_integration.train.ui.input as input_ui
import supervisely_integration.train.ui.parameters as parameters_ui
import supervisely_integration.train.ui.splits as splits_ui
import supervisely_integration.train.workflow as w
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    DoneLabel,
    Empty,
    Field,
    FolderThumbnail,
    LineChart,
    Progress,
    ReportThumbnail,
    SlyTqdm,
)
from supervisely.io.fs import get_file_name, get_file_name_with_ext
from supervisely_integration.train.ui.model_benchmark import run_model_benchmark
from supervisely_integration.train.ui.project_cached import download_project

# TODO: Fix import, now it's causing error
# from rtdetr_pytorch.src.misc.sly_logger import Logs

start_train_btn = Button("Train")
stop_train_btn = Button("Stop", button_type="danger")
stop_train_btn.hide()
stop_train_btn.disable()

btn_container = Container(
    [start_train_btn, stop_train_btn, Empty()],
    "horizontal",
    overflow="wrap",
    fractions=[1, 1, 10],
    gap=1,
)

loss = LineChart("Loss", series=[{"name": "Loss", "data": []}])
learning_rate = LineChart(
    "Learning Rate",
    series=[
        {"name": "lr0", "data": []},
        {"name": "lr1", "data": []},
        {"name": "lr2", "data": []},
        {"name": "lr3", "data": []},
    ],
)
cuda_memory = LineChart("CUDA Memory", series=[{"name": "Memory", "data": []}])
validation_metrics = LineChart(
    "Validation Metrics",
    series=[
        {"name": "AP@IoU=0.50:0.95|maxDets=100", "data": []},
        {"name": "AP@IoU=0.50|maxDets=100", "data": []},
        {"name": "AP@IoU=0.75|maxDets=100", "data": []},
        {"name": "AR@IoU=0.50:0.95|maxDets=1", "data": []},
        {"name": "AR@IoU=0.50:0.95|maxDets=10", "data": []},
        {"name": "AR@IoU=0.50:0.95|maxDets=100", "data": []},
    ],
)


# charts_grid = Grid([loss, learning_rate, cuda_memory, validation_metrics], columns=2, gap=5)

charts_grid = Container(
    [
        Container([loss, learning_rate], direction="horizontal", widgets_style="display: null"),
        Container(
            [cuda_memory, validation_metrics], direction="horizontal", widgets_style="display: null"
        ),
    ]
)

charts_grid_f = Field(charts_grid, "Training and validation metrics")
charts_grid_f.hide()

progress_bar_download_project = Progress()
progress_bar_prepare_project = Progress()
progress_bar_download_model = Progress()
progress_bar_epochs = Progress(hide_on_finish=False)
progress_bar_iters = Progress(hide_on_finish=False)
progress_bar_upload_artifacts = Progress()

# Model benchmark
model_benchmark_pbar = SlyTqdm()
model_benchmark_pbar_secondary = Progress(hide_on_finish=False)
creating_report_f = Field(Empty(), "", "Creating report on model...")
creating_report_f.hide()
model_benchmark_report = ReportThumbnail()
model_benchmark_report.hide()

output_folder = FolderThumbnail()
output_folder.hide()

success_msg = DoneLabel("Training completed. Training artifacts were uploaded to Team Files.")
success_msg.hide()

card = Card(
    title="Training Progress",
    description="Task progress, detailed logs, metrics charts, and other visualizations",
    content=Container(
        [
            success_msg,
            output_folder,
            model_benchmark_report,
            creating_report_f,
            progress_bar_download_project,
            progress_bar_prepare_project,
            progress_bar_download_model,
            progress_bar_epochs,
            progress_bar_iters,
            progress_bar_upload_artifacts,
            model_benchmark_pbar,
            model_benchmark_pbar_secondary,
            btn_container,
            charts_grid_f,
        ]
    ),
    lock_message="Select parameterts to unlock",
)
card.lock()


def create_experiment(
    model_name, remote_dir, report_id=None, eval_metrics=None, primary_metric_name=None
):
    train_info = TrainInfo(**g.sly_rtdetr_generated_metadata)
    experiment_info = g.rtdetr_artifacts.convert_train_to_experiment_info(train_info)
    experiment_info.experiment_name = f"{g.TASK_ID} {g.project_info.name} {model_name}"
    experiment_info.model_name = model_name
    experiment_info.framework_name = f"{g.rtdetr_artifacts.framework_name}"
    experiment_info.train_size = g.train_size
    experiment_info.val_size = g.val_size
    experiment_info.evaluation_report_id = report_id
    experiment_info.experiment_report_id = None
    if report_id is not None:
        experiment_info.evaluation_report_link = f"/model-benchmark?id={str(report_id)}"
    experiment_info.evaluation_metrics = eval_metrics

    experiment_info_json = asdict(experiment_info)
    experiment_info_json["project_preview"] = g.project_info.image_preview_url
    experiment_info_json["primary_metric"] = primary_metric_name

    g.api.task.set_output_experiment(g.task_id, experiment_info_json)
    experiment_info_json.pop("project_preview")
    experiment_info_json.pop("primary_metric")

    experiment_info_path = os.path.join(g.artifacts_dir, "experiment_info.json")
    remote_experiment_info_path = os.path.join(remote_dir, "experiment_info.json")
    dump_json_file(experiment_info_json, experiment_info_path)
    g.api.file.upload(g.team_id, experiment_info_path, remote_experiment_info_path)

def iter_callback(logs):
    iter_idx = logs.iter_idx
    loss.add_to_series("Loss", (iter_idx, logs.loss))
    add_lrs(iter_idx, logs.lrs)
    cuda_memory.add_to_series("Memory", (iter_idx, logs.cuda_memory))


def eval_callback(logs):
    add_metrics(logs.epoch, logs.evaluation_metrics)


def add_lrs(iter_idx: int, lrs: Dict[str, float]):
    for series_name, lr in lrs.items():
        learning_rate.add_to_series(series_name, (iter_idx, lr))


def add_metrics(epoch: int, metrics: Dict[str, float]):
    for series_name, metric in metrics.items():
        if series_name.startswith("per_class"):
            continue
        validation_metrics.add_to_series(series_name, (epoch, metric))


train_cli.setup_callbacks(iter_callback=iter_callback, eval_callback=eval_callback)


@start_train_btn.click
def run_training():
    project_dir = os.path.join(g.data_dir, "sly_project")
    g.project_dir = project_dir
    # iter_progress = Progress("Iterations", hide_on_finish=False)

    g.USE_CACHE = input_ui.use_cache_checkbox.is_checked()
    download_project(
        api=g.api,
        project_id=g.PROJECT_ID,
        project_dir=project_dir,
        use_cache=g.USE_CACHE,
        progress=progress_bar_download_project,
    )
    g.project = sly.read_project(project_dir)
    # prepare split files
    try:
        splits_ui.dump_train_val_splits(project_dir)
    except Exception:
        if not g.USE_CACHE:
            raise
        sly.logger.warn(
            "Failed to dump train/val splits. Trying to re-download project.", exc_info=True
        )
        download_project(
            api=g.api,
            project_id=g.PROJECT_ID,
            project_dir=project_dir,
            use_cache=False,
            progress=progress_bar_download_project,
        )
        splits_ui.dump_train_val_splits(project_dir)
        g.project = sly.read_project(project_dir)

    # add prepare model progress
    with progress_bar_prepare_project(
        message="Preparing project...", total=1
    ) as prepare_project_pbar:
        g.splits = splits_ui.trainval_splits.get_splits()
        sly.logger.debug("Read splits from the widget...")
        create_trainval()
        custom_config = parameters_ui.read_parameters(len(g.splits[0]))
        prepare_config(custom_config)
        prepare_project_pbar.update(1)

    stop_train_btn.enable()
    charts_grid_f.show()

    cfg = train()
    save_config(cfg)

    progress_bar_epochs.hide()
    progress_bar_iters.hide()

    # Upload artifacts
    remote_artifacts_dir, file_info = upload_model(cfg.output_dir)

    # Model Benchmark
    dataset_infos = g.api.dataset.get_list(g.PROJECT_ID)
    ds_name_to_id = {ds.name: ds.id for ds in dataset_infos}
    selected_classes = classes_ui.train_classes_selector.get_selected_classes()
    train_set, val_set = g.splits

    # paths
    remote_weights_path = g.rtdetr_artifacts.get_weights_path(remote_artifacts_dir)
    remote_config_path = g.rtdetr_artifacts.get_config_path(remote_artifacts_dir)
    local_artifacts_dir = os.path.join(cfg.output_dir, "upload")
    local_checkpoints_dir = os.path.join(local_artifacts_dir, "weights")


    benchmark_report_template, report_id, eval_metrics, primary_metric_name = (
        None,
        None,
        None,
        None,
    )

    # Implement model evaluation benchmark here
    model_benchmark_done = False
    if parameters_ui.run_model_benchmark_checkbox.is_checked():
        model_benchmark_done, benchmark_report_template, report_id, eval_metrics, primary_metric_name = run_model_benchmark(
            api=g.api,
            root_source_path=g.CURRENT_DIR,
            local_artifacts_dir=local_artifacts_dir,
            remote_weights_dir=remote_weights_path,
            remote_config_path=remote_config_path,
            project_info=g.project_info,
            dataset_infos=dataset_infos,
            ds_name_to_id=ds_name_to_id,
            train_val_split=splits_ui.trainval_splits,
            train_set=train_set,
            val_set=val_set,
            selected_classes=selected_classes,
            use_speedtest=parameters_ui.run_speedtest_checkbox.is_checked(),
            model_benchmark_report=model_benchmark_report,
            creating_report_f=creating_report_f,
            model_benchmark_pbar=model_benchmark_pbar,
            model_benchmark_pbar_secondary=model_benchmark_pbar_secondary,
        )
    # ------------------------------------------

    if not model_benchmark_done:
        benchmark_report_template = None
    w.workflow_output(
        g.api,
        "RT-DETR",
        remote_artifacts_dir,
        get_file_name(g.best_checkpoint_path),
        benchmark_report_template,
    )

    try:
        create_experiment(
            model_name="RT-DETR",
            remote_dir=remote_artifacts_dir,
            report_id=report_id,
            eval_metrics=eval_metrics,
            primary_metric_name=primary_metric_name,
        )
    except Exception as e:
        sly.logger.warning(
            f"Couldn't create experiment, this training session will not appear in experiments table. Error: {e}"
        )



    # hide buttons
    start_train_btn.hide()
    stop_train_btn.hide()

    # add file tb
    output_folder.set(file_info)
    # add success text
    success_msg.show()
    output_folder.show()
    start_train_btn.disable()
    stop_train_btn.disable()

    # stop app

    g.app.stop()


@stop_train_btn.click
def stop_training():
    # TODO: Implement the stop process
    g.STOP_TRAINING = True
    stop_train_btn.disable()


def prepare_config(custom_config: Dict[str, Any]):
    if g.model_mode == g.MODEL_MODES[0]:
        model_name = g.train_mode.pretrained[0]
        arch = model_name.split("_coco")[0]
        config_name = f"{arch}_6x_coco"
        sly.logger.info(f"Model name: {model_name}, arch: {arch}, config_name: {config_name}")
    else:
        model_name = get_file_name_with_ext(g.train_mode.custom)
        config_name = "custom"
        sly.logger.info(f"Model name: {model_name}, config_name: {config_name}")

    if g.model_mode == g.MODEL_MODES[0]:
        custom_config["__include__"] = [f"{config_name}.yml"]
    else:
        custom_config["__include__"] = [
            "../dataset/coco_detection.yml",
            "../runtime.yml",
            "./include/dataloader.yml",
            "./include/optimizer.yml",
            "./include/rtdetr_r50vd.yml",
        ]
    custom_config["remap_mscoco_category"] = False
    custom_config["num_classes"] = len(g.selected_classes)
    custom_config["train_dataloader"]["dataset"]["img_folder"] = f"{g.train_dataset_path}/img"
    custom_config["train_dataloader"]["dataset"][
        "ann_file"
    ] = f"{g.train_dataset_path}/coco_anno.json"
    custom_config["val_dataloader"]["dataset"]["img_folder"] = f"{g.val_dataset_path}/img"
    custom_config["val_dataloader"]["dataset"]["ann_file"] = f"{g.val_dataset_path}/coco_anno.json"
    selected_classes = g.selected_classes
    custom_config["sly_metadata"] = {
        "classes": selected_classes,
        "project_id": g.PROJECT_ID,
        "project_name": g.project_info.name,
        "model": model_name,
    }

    g.custom_config_path = os.path.join(g.CONFIG_PATHS_DIR, "custom.yml")
    with open(g.custom_config_path, "w") as f:
        yaml.dump(custom_config, f)


def train():
    file_info = None
    if g.model_mode == g.MODEL_MODES[0]:
        model = g.train_mode.pretrained[0]
        finetune = g.train_mode.finetune
    else:
        model = g.train_mode.custom
        file_info = g.api.file.get_info_by_path(g.TEAM_ID, model)
        finetune = True

    # ---------------------------------- Init And Set Workflow Input --------------------------------- #
    w.workflow_input(g.api, g.project_info, file_info)
    # ----------------------------------------------- - ---------------------------------------------- #

    cfg = train_cli.train(
        model,
        finetune,
        g.custom_config_path,
        progress_bar_download_model,
        progress_bar_epochs,
        progress_bar_iters,
        stop_train_btn,
        charts_grid_f,
    )
    return cfg


def save_config(cfg):
    if "__include__" in cfg.yaml_cfg:
        cfg.yaml_cfg.pop("__include__")

    output_path = os.path.join(cfg.output_dir, "config.yml")

    with open(output_path, "w") as f:
        yaml.dump(cfg.yaml_cfg, f)


def upload_model(output_dir):
    remote_artifacts_dir = f"/RT-DETR/{g.project_info.name}/{g.TASK_ID}"
    remote_weights_path = g.rtdetr_artifacts.get_weights_path(remote_artifacts_dir)
    remote_config_path = g.rtdetr_artifacts.get_config_path(remote_artifacts_dir)
    local_artifacts_dir = os.path.join(output_dir, "upload")
    local_checkpoints_dir = os.path.join(local_artifacts_dir, "weights")
    sly.fs.mkdir(local_artifacts_dir)
    sly.fs.mkdir(local_checkpoints_dir)
    sly.logger.info(f"Local artifacts dir: {local_artifacts_dir}")

    # Move last checkpoint to checkpoints folder
    shutil.move(g.latest_checkpoint_path, f"{local_checkpoints_dir}/{g.latest_checkpoint_name}")

    # Move best checkpoint to checkpoints folder
    best_checkpoint_file_name = get_file_name(g.best_checkpoint_path)
    shutil.move(g.best_checkpoint_path, f"{local_checkpoints_dir}/{best_checkpoint_file_name}.pth")

    # Move log and config files to artifacts folder
    shutil.move(f"{output_dir}/log.txt", f"{local_artifacts_dir}/log.txt")
    shutil.move(f"{output_dir}/config.yml", f"{local_artifacts_dir}/config.yml")

    # Save link to app ui
    app_url = f"/apps/sessions/{g.TASK_ID}"
    app_link_path = os.path.join(local_artifacts_dir, "open_app.lnk")
    with open(app_link_path, "w") as text_file:
        print(app_url, file=text_file)

    # Upload artifacts
    local_files = sly.fs.list_files_recursively(local_artifacts_dir)
    total_size = sum([sly.fs.get_file_size(file_path) for file_path in local_files])
    with progress_bar_upload_artifacts(
        message="Uploading train artifacts to Team Files...",
        total=total_size,
        unit="bytes",
        unit_scale=True,
    ) as artifacts_pbar:
        out_path = g.api.file.upload_directory(
            sly.env.team_id(),
            local_artifacts_dir,
            remote_artifacts_dir,
            progress_size_cb=artifacts_pbar,
        )

    # Upload train metadata
    g.sly_rtdetr_generated_metadata = g.rtdetr_artifacts.generate_metadata(
        app_name=g.rtdetr_artifacts.app_name,
        task_id=g.TASK_ID,
        artifacts_folder=remote_artifacts_dir,
        weights_folder=remote_weights_path,
        weights_ext=g.rtdetr_artifacts.weights_ext,
        project_name=g.project_info.name,
        task_type="object detection",
        config_path=remote_config_path,
    )

    file_info = g.api.file.get_info_by_path(
        g.TEAM_ID, os.path.join(remote_artifacts_dir, "open_app.lnk")
    )
    sly.logger.info("Training artifacts uploaded successfully")
    sly.output.set_directory(remote_artifacts_dir)
    return out_path, file_info


def create_trainval():
    # g.splits = splits.trainval_splits.get_splits()
    train_items, val_items = g.splits
    g.train_size, g.val_size = len(train_items), len(val_items)
    sly.logger.debug(f"Creating trainval datasets from splits: {g.splits}...")
    train_items: List[sly.project.project.ItemInfo]
    val_items: List[sly.project.project.ItemInfo]

    converted_project_dir = os.path.join(g.CONVERTED_DIR, g.project_info.name)
    sly.logger.debug(f"Converted project will be saved to {converted_project_dir}.")
    sly.fs.mkdir(converted_project_dir)
    train_dataset_path = os.path.join(converted_project_dir, "train")
    val_dataset_path = os.path.join(converted_project_dir, "val")
    sly.logger.debug(
        f"Train dataset path: {train_dataset_path}, val dataset path: {val_dataset_path}."
    )

    g.train_dataset_path = train_dataset_path
    g.val_dataset_path = val_dataset_path

    project_meta_path = os.path.join(converted_project_dir, "meta.json")
    sly.json.dump_json_file(g.project.meta.to_json(), project_meta_path)

    for items, dataset_path in zip(
        [train_items, val_items], [train_dataset_path, val_dataset_path]
    ):
        prepare_dataset(dataset_path, items)

    g.converted_project = sly.Project(converted_project_dir, sly.OpenMode.READ)
    sly.logger.info(f"Project created in {converted_project_dir}")

    for dataset_fs in g.converted_project.datasets:
        dataset_fs: sly.Dataset
        selected_classes = g.selected_classes

        coco_anno = get_coco_annotations(dataset_fs, g.converted_project.meta, selected_classes)
        coco_anno_path = os.path.join(dataset_fs.directory, "coco_anno.json")
        sly.json.dump_json_file(coco_anno, coco_anno_path)

    sly.logger.info("COCO annotations created")


def prepare_dataset(dataset_path: str, items: List[sly.project.project.ItemInfo]):
    sly.logger.debug(f"Preparing dataset in {dataset_path}...")
    img_dir = os.path.join(dataset_path, "img")
    ann_dir = os.path.join(dataset_path, "ann")
    sly.fs.mkdir(img_dir)
    sly.fs.mkdir(ann_dir)
    for item in items:
        src_img_path = os.path.join(g.project_dir, fix_widget_path(item.img_path))
        src_ann_path = os.path.join(g.project_dir, fix_widget_path(item.ann_path))
        dst_img_path = os.path.join(img_dir, item.name)
        dst_ann_path = os.path.join(ann_dir, f"{item.name}.json")
        sly.fs.copy_file(src_img_path, dst_img_path)
        sly.fs.copy_file(src_ann_path, dst_ann_path)

    sly.logger.info(f"Dataset prepared in {dataset_path}")


def fix_widget_path(bugged_path: str) -> str:
    """Fixes the broken ItemInfo paths from TrainValSplits widget.
    Removes the first two folders from the path.

    Bugged path: app_data/1IkWRgJG62f1ZuZ/ds0/ann/pexels_2329440.jpeg.json
    Corrected path: ds0/ann/pexels_2329440.jpeg.json

    :param bugged_path: Path to fix
    :type bugged_path: str
    :return: Fixed path
    :rtype: str
    """
    path = bugged_path.split("/")

    if sly.is_development():
        updated_path = path[8:]
    else:
        updated_path = path[7:]
    correct_path = "/".join(updated_path)
    return correct_path


def get_coco_annotations(dataset: sly.Dataset, meta: sly.ProjectMeta, selected_classes: List[str]):
    coco_anno = {"images": [], "categories": [], "annotations": []}
    cat2id = {name: i for i, name in enumerate(selected_classes)}
    img_id = 1
    ann_id = 1
    for name in dataset.get_items_names():
        ann = dataset.get_ann(name, meta)
        img_dict = {
            "id": img_id,
            "height": ann.img_size[0],
            "width": ann.img_size[1],
            "file_name": name,
        }
        coco_anno["images"].append(img_dict)

        for label in ann.labels:
            if isinstance(label.geometry, (sly.Bitmap, sly.Polygon)):
                rect = label.geometry.to_bbox()
            elif isinstance(label.geometry, sly.Rectangle):
                rect = label.geometry
            else:
                continue
            class_name = label.obj_class.name
            if class_name not in selected_classes:
                continue
            x, y, x2, y2 = rect.left, rect.top, rect.right, rect.bottom
            ann_dict = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat2id[class_name],
                "bbox": [x, y, x2 - x, y2 - y],
                "area": (x2 - x) * (y2 - y),
                "iscrowd": 0,
            }
            coco_anno["annotations"].append(ann_dict)
            ann_id += 1

        img_id += 1

    coco_anno["categories"] = [{"id": i, "name": name} for name, i in cat2id.items()]
    # Test:
    coco_api = COCO()
    coco_api.dataset = coco_anno
    coco_api.createIndex()
    return coco_anno


# parameters handlers
