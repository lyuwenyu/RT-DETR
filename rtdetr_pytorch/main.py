import os
from dotenv import load_dotenv
import supervisely as sly
import supervisely.app.widgets as widgets
import yaml
from convert_to_coco import get_coco_annotations
from checkpoints import checkpoints


# Globals
if sly.is_development():
    load_dotenv("local.env")
api = sly.Api()
project_id = sly.env.project_id()
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
        self.selected_classes = widgets.ClassesTable(project_id=project_id)
        self.selected_classes.select_all()
        self.custom_config = widgets.Editor(placeholder_config, language_mode="yaml", height_lines=25)
        self.run_button = widgets.Button("Train")
        
        self.container = widgets.Container([
            project_view,
            self.models,
            self.finetune,
            self.train_dataset,
            self.val_dataset,
            self.selected_classes,
            self.custom_config,
            self.run_button
        ])

        @self.run_button.click
        def run():
            prepare_data()
            prepare_config()
            train()


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
    sly.json.dump_json_file(coco_anno, f"{train_dataset.directory}/coco_anno.json")

    val_dataset : sly.Dataset = project.datasets.get(val_dataset_name)
    coco_anno = get_coco_annotations(val_dataset, meta, selected_classes)
    sly.json.dump_json_file(coco_anno, f"{val_dataset.directory}/coco_anno.json")


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
    custom_config["train_dataloader"] = {
        "dataset": {
            "img_folder": f"{project_dir}/{train_dataset_name}/img",
            "ann_file": f"{project_dir}/{train_dataset_name}/coco_anno.json"
        }
    }
    custom_config["val_dataloader"] = {
        "dataset": {
            "img_folder": f"{project_dir}/{val_dataset_name}/img",
            "ann_file": f"{project_dir}/{val_dataset_name}/coco_anno.json"
        }
    }

    # save custom config
    with open(custom_config_path, 'w') as f:
        yaml.dump(custom_config, f)


def train():
    import train as train_cli
    model = ui.models.get_value()
    finetune = ui.finetune.is_checked()
    train_cli.train(model, finetune, custom_config_path)


ui = UI()
app = sly.Application(ui.container)
