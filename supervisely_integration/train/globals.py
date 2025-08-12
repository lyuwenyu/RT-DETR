import os
import sys
from typing import Optional

from dotenv import load_dotenv

import supervisely as sly
from rtdetr_pytorch.model_list import _models
from supervisely.nn.artifacts.rtdetr import RTDETR

if sly.is_development:
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

# region constants
cwd = os.getcwd()
sly.logger.debug(f"Current working directory: {cwd}")
rtdetr_pytorch_path = os.path.join(cwd, "rtdetr_pytorch")
sys.path.insert(0, rtdetr_pytorch_path)
sly.logger.debug("Added rtdetr_pytorch to the system path")
CONFIG_PATHS_DIR = os.path.join(rtdetr_pytorch_path, "configs", "rtdetr")
default_config_path = os.path.join(CONFIG_PATHS_DIR, "placeholder.yml")
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sly.logger.debug(f"Current directory: {CURRENT_DIR}")
TEMP_DIR = os.path.join(CURRENT_DIR, "temp")
DOWNLOAD_DIR = os.path.join(TEMP_DIR, "download")
CONVERTED_DIR = os.path.join(TEMP_DIR, "converted")
sly.fs.mkdir(DOWNLOAD_DIR, remove_content_if_exists=True)
sly.fs.mkdir(CONVERTED_DIR, remove_content_if_exists=True)
sly.logger.debug(f"Download dir: {DOWNLOAD_DIR}, converted dir: {CONVERTED_DIR}")
OUTPUT_DIR = os.path.join(TEMP_DIR, "output")
sly.fs.mkdir(OUTPUT_DIR, remove_content_if_exists=True)
sly.logger.debug(f"Output dir: {OUTPUT_DIR}")
AUG_TEMPLATES_DIR = os.path.join(CURRENT_DIR, "aug_templates")

data_dir = os.path.join(CURRENT_DIR, "data")
sly.fs.mkdir(data_dir, remove_content_if_exists=True)

MODEL_MODES = ["Pretrained models", "Custom weights"]
TABLE_COLUMNS = [
    "Name",
    "Dataset",
    "AP_Val",
    "Params(M)",
    "FRPS(T4)",
]
PRETRAINED_MODELS = [
    [value for key, value in model_info.items() if key != "meta"] for model_info in _models
]
OPTIMIZERS = ["Adam", "AdamW", "SGD"]
SCHEDULERS = [
    "Without scheduler",
    "CosineAnnealingLR",
    "LinearLR",
    "MultiStepLR",
    "OneCycleLR",
]

# endregion

# region envvars
TASK_ID = sly.env.task_id()
TEAM_ID = sly.env.team_id()
WORKSPACE_ID = sly.env.workspace_id()
PROJECT_ID = sly.env.project_id()

USE_CACHE = True
STOP_TRAINING = False

rtdetr_artifacts = RTDETR(TEAM_ID)
sly_rtdetr_generated_metadata = None
train_size = None
val_size = None

# endregion
api = sly.Api.from_env()
augs = []
# region state
team = api.team.get_info_by_id(TEAM_ID)
stepper = None
project_info = None
project_meta = None
project_dir = None
project = None
converted_project = None
train_dataset_path = None
val_dataset_path = None
custom_config_path = None
train_mode = None
selected_classes = None
splits = None
widgets = None
# endregion

model_mode = None
best_checkpoint_path = None
latest_checkpoint_name = "last.pth"
latest_checkpoint_path = None

app = None

def update_step(back: Optional[bool] = False, step: Optional[int] = None) -> None:
    if step is None:
        current_step = stepper.get_active_step()
        sly.logger.debug(f"Current step: {current_step}")
        step = current_step - 1 if back else current_step + 1
    sly.logger.debug(f"Next step: {step}")
    stepper.set_active_step(step)


def read_augs():
    aug_files = sly.fs.list_files(
        AUG_TEMPLATES_DIR, valid_extensions=[".json"], ignore_valid_extensions_case=True
    )
    sly.logger.debug(f"Found {len(aug_files)} augmentation templates")

    for aug_path in aug_files:
        aug_name = sly.utils.camel_to_snake(sly.fs.get_file_name(aug_path))
        template = {
            "label": aug_name,
            "value": aug_path,
        }
        augs.append(template)

    sly.logger.debug(f"Prepared {len(augs)} augmentation templates")


read_augs()
