import os
from pathlib import Path

from supervisely.app.widgets import AugmentationsWithTabs, Button, Card, Container, Switch

import supervisely_integration.train.globals as g
import supervisely_integration.train.ui.parameters as parameters
import supervisely_integration.train.ui.utils as utils


def name_from_path(aug_path):
    name = os.path.basename(aug_path).split(".json")[0].capitalize()
    name = " + ".join(name.split("_"))
    return name


template_dir = "supervisely_integration/train/aug_templates"
template_paths = list(map(str, Path(template_dir).glob("*.json")))
template_paths = sorted(template_paths, key=lambda x: x.replace(".", "_"))[::-1]

templates = [{"label": name_from_path(path), "value": path} for path in template_paths]


switcher = Switch(True)
augments = AugmentationsWithTabs(g, task_type="detection", templates=templates)

select_augs_button = Button("Select")
container = Container([switcher, augments, select_augs_button])

card = Card(
    title="Training augmentations",
    description="Choose one of the prepared templates or provide custom pipeline",
    content=container,
    lock_message="Select splits to unlock",
)
card.lock("Confirm splits.")


def reset_widgets():
    if switcher.is_switched():
        augments.show()
    else:
        augments.hide()


def get_selected_aug():
    # path to aug pipline (.json file)
    if switcher.is_switched():
        return augments._current_augs._template_path
    else:
        return None


@switcher.value_changed
def on_switch(is_switched: bool):
    reset_widgets()


reset_widgets()


@select_augs_button.click
def splits_selected():
    if select_augs_button.text == "Select":
        # widgets to disable
        utils.disable_enable([switcher, augments, container, card], True)

        utils.update_custom_button_params(select_augs_button, utils.reselect_params)
        g.update_step(step=6)

        # unlock
        parameters.card.unlock()
    else:
        # lock
        parameters.card.lock()
        utils.update_custom_button_params(select_augs_button, utils.select_params)

        # widgets to enable
        utils.disable_enable([switcher, augments, container, card], False)
        g.update_step(back=True)
