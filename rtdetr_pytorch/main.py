import os
import shutil
from dotenv import load_dotenv
import supervisely as sly
import supervisely.app.widgets as widgets
import yaml
from convert_to_coco import get_coco_annotations
from checkpoints import checkpoints
import sly_imgaug


# Globals
if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()
project_id = sly.env.project_id()
project_name = api.project.get_info_by_id(project_id).name
project_dir = "sly_dataset"
custom_config_path = f"rtdetr_pytorch/configs/rtdetr/custom.yml"
with open(f"rtdetr_pytorch/configs/rtdetr/placeholder.yml", 'r') as f:
    placeholder_config = f.read()


class UI:
    def __init__(self) -> None:
        project_view = widgets.ProjectThumbnail(api.project.get_info_by_id(project_id))
        self.models = widgets.SelectString(values=checkpoints.keys())
        self.finetune = widgets.Checkbox("Finetune", True)
        self.train_dataset = widgets.SelectDataset(project_id=project_id, compact=True)
        self.val_dataset = widgets.SelectDataset(project_id=project_id, compact=True)
        ds_id = api.dataset.get_list(project_id)[0].id
        self.train_dataset.set_dataset_id(ds_id)
        self.val_dataset.set_dataset_id(ds_id)
        self.selected_classes = widgets.ClassesTable(project_id=project_id)
        self.selected_classes.select_all()
        self.custom_config = widgets.Editor(placeholder_config, language_mode="yaml", height_lines=25)
        self.run_button = widgets.Button("Train")
        self.success_msg = widgets.DoneLabel("Training completed. Checkpoints were uploaded to Team Files.")
        self.folder_thumb = widgets.FolderThumbnail()
        self.success_msg.hide()
        self.folder_thumb.hide()
        
        self.container = widgets.Container([
            project_view,
            self.models,
            self.finetune,
            self.train_dataset,
            self.val_dataset,
            self.selected_classes,
            self.custom_config,
            self.run_button,
            self.success_msg,
            self.folder_thumb,
        ])

        @self.run_button.click
        def run():
            prepare_data()
            prepare_config()
            cfg = train()
            save_config(cfg)
            out_path = upload_model(cfg.output_dir)
            success(out_path)


def prepare_data():
    train_dataset_id = ui.train_dataset.get_selected_id()
    train_dataset_name = api.dataset.get_info_by_id(train_dataset_id).name

    val_dataset_id = ui.val_dataset.get_selected_id()
    val_dataset_name = api.dataset.get_info_by_id(val_dataset_id).name

    selected_classes = ui.selected_classes.get_selected_classes()

    # download
    if not os.path.exists(project_dir):
        sly.download(api, project_id, project_dir, dataset_ids=[train_dataset_id, val_dataset_id])
    project = sly.read_project(project_dir)
    meta = project.meta

    train_dataset : sly.Dataset = project.datasets.get(train_dataset_name)
    coco_anno = get_coco_annotations(train_dataset, meta, selected_classes)
    sly.json.dump_json_file(coco_anno, f"{train_dataset.directory}/coco_anno.json", indent=None)

    val_dataset : sly.Dataset = project.datasets.get(val_dataset_name)
    coco_anno = get_coco_annotations(val_dataset, meta, selected_classes)
    sly.json.dump_json_file(coco_anno, f"{val_dataset.directory}/coco_anno.json", indent=None)


def prepare_config():
    train_dataset_id = ui.train_dataset.get_selected_id()
    train_dataset_name = api.dataset.get_info_by_id(train_dataset_id).name

    val_dataset_id = ui.val_dataset.get_selected_id()
    val_dataset_name = api.dataset.get_info_by_id(val_dataset_id).name

    custom_config_text = ui.custom_config.get_value()
    model = ui.models.get_value()
    arch = model.split('_coco')[0]
    config_name = f"{arch}_6x_coco"

    custom_config = yaml.safe_load(custom_config_text) or {}
    custom_config["__include__"] = [f"{config_name}.yml"]
    custom_config["remap_mscoco_category"] = False
    custom_config["num_classes"] = len(ui.selected_classes.get_selected_classes())
    if "train_dataloader" not in custom_config:
        custom_config["train_dataloader"] = {
            "dataset": {
                "img_folder": f"{project_dir}/{train_dataset_name}/img",
                "ann_file": f"{project_dir}/{train_dataset_name}/coco_anno.json"
            }
        }
    else:
        custom_config["train_dataloader"]["dataset"]["img_folder"] = f"{project_dir}/{train_dataset_name}/img"
        custom_config["train_dataloader"]["dataset"]["ann_file"] = f"{project_dir}/{train_dataset_name}/coco_anno.json"
    if "val_dataloader" not in custom_config:
        custom_config["val_dataloader"] = {
            "dataset": {
                "img_folder": f"{project_dir}/{val_dataset_name}/img",
                "ann_file": f"{project_dir}/{val_dataset_name}/coco_anno.json"
            }
        }
    else:
        custom_config["val_dataloader"]["dataset"]["img_folder"] = f"{project_dir}/{val_dataset_name}/img"
        custom_config["val_dataloader"]["dataset"]["ann_file"] = f"{project_dir}/{val_dataset_name}/coco_anno.json"
    selected_classes = ui.selected_classes.get_selected_classes()
    custom_config["sly_metadata"] = {
        "classes": selected_classes,
        "project_id": project_id,
        "project_name": project_name,
        "model": model,
    }

    # save custom config
    with open(custom_config_path, 'w') as f:
        yaml.dump(custom_config, f)


def train():
    import train as train_cli
    model = ui.models.get_value()
    finetune = ui.finetune.is_checked()
    cfg = train_cli.train(model, finetune, custom_config_path)
    return cfg


def save_config(cfg):
    if "__include__" in cfg.yaml_cfg:
        cfg.yaml_cfg.pop("__include__")
    os.makedirs("output", exist_ok=True)
    with open(f"output/config.yml", 'w') as f:
        yaml.dump(cfg.yaml_cfg, f)


def upload_model(output_dir):
    model = ui.models.get_value()
    task_id = api.task_id or ""
    team_files_dir = f"/RT-DETR/{project_name}_{project_id}/{task_id}_{model}"
    local_dir = f"{output_dir}/upload"
    os.makedirs(local_dir, exist_ok=True)

    checkpoints = [f for f in os.listdir(output_dir) if f.endswith('.pth')]
    latest_checkpoint = sorted(checkpoints)[-1]
    shutil.move(f"{output_dir}/{latest_checkpoint}", f"{local_dir}/{latest_checkpoint}")
    shutil.move(f"{output_dir}/log.txt", f"{local_dir}/log.txt")
    shutil.move("output/config.yml", f"{local_dir}/config.yml")

    out_path = api.file.upload_directory(
        sly.env.team_id(),
        local_dir,
        team_files_dir,
    )
    return out_path


def success(out_path):
    file_info = api.file.get_info_by_path(sly.env.team_id(), out_path + "/log.txt")
    ui.folder_thumb.set(info=file_info)
    ui.folder_thumb.show()
    ui.success_msg.show()
    if sly.is_production():
        api.task.set_output_directory(api.task_id, file_info.id, out_path)
        app.stop()


ui = UI()
app = sly.Application(ui.container)


# def _run():
#     prepare_data()
#     prepare_config()
#     cfg = train()
#     save_config(cfg)
#     out_path = upload_model(cfg.output_dir)
#     success(out_path)

# import train as train_cli
# train_cli.setup_callbacks()
# _run()
