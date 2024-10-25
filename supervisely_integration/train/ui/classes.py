from typing import Optional

import supervisely as sly
from supervisely.app.widgets import Button, Card, ClassesTable, Container, Text

import supervisely_integration.train.globals as g
import supervisely_integration.train.ui.augmentations as augmentations
import supervisely_integration.train.ui.parameters as parameters
import supervisely_integration.train.ui.splits as splits
import supervisely_integration.train.ui.utils as utils

empty_notification = Text("Please, select at least one class.", status="warning")
train_classes_selector = ClassesTable(project_id=g.PROJECT_ID)
train_classes_selector.hide()

select_classes_button = Button("Select")

card = Card(
    title="Training classes",
    description="Select classes to train the model on",
    content=Container([train_classes_selector, select_classes_button]),
    lock_message="Select model to unlock",
)
card.lock()


def fill_classes_selector(clear: Optional[bool] = False):
    if not clear:
        train_classes_selector.set_project_meta(g.project_meta)
        train_classes_selector.select_all()
        train_classes_selector.show()
    else:
        train_classes_selector.set_project_meta(sly.ProjectMeta())
        train_classes_selector.hide()


@select_classes_button.click
def classes_selected():
    if select_classes_button.text == "Select":
        # widgets to disable
        utils.disable_enable([empty_notification, train_classes_selector, card], True)

        selected_classes = train_classes_selector.get_selected_classes()
        if not selected_classes:
            return
        g.selected_classes = selected_classes
        sly.logger.info(f"Selected classes: {selected_classes}")

        utils.update_custom_button_params(select_classes_button, utils.reselect_params)
        g.update_step()

        # unlock
        splits.card.unlock()
    else:
        g.selected_classes = None

        # lock
        splits.card.lock()
        utils.update_custom_button_params(splits.select_splits_button, utils.select_params)

        augmentations.card.lock()
        utils.update_custom_button_params(augmentations.select_augs_button, utils.select_params)

        parameters.card.lock()
        # utils.update_custom_button_params(parameters.run_training, utils.select_params)

        utils.update_custom_button_params(select_classes_button, utils.select_params)

        # widgets to enable
        utils.disable_enable([empty_notification, train_classes_selector, card], False)
        g.update_step(back=True)
