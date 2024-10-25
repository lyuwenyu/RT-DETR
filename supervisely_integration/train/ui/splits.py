import supervisely as sly
from supervisely.app.widgets import Button, Card, Container, TrainValSplits

import supervisely_integration.train.globals as g
import supervisely_integration.train.ui.augmentations as augmentations
import supervisely_integration.train.ui.parameters as parameters
import supervisely_integration.train.ui.utils as utils

trainval_splits = TrainValSplits(g.PROJECT_ID)
select_splits_button = Button("Select")

card = Card(
    title="Train / Validation splits",
    description="Select splits for training and validation",
    content=Container([trainval_splits, select_splits_button]),
    lock_message="Select classes to unlock",
)
card.lock()


@select_splits_button.click
def splits_selected():
    if select_splits_button.text == "Select":
        # widgets to disable
        utils.disable_enable([trainval_splits, card], True)

        # g.splits = trainval_splits.get_splits()

        utils.update_custom_button_params(select_splits_button, utils.reselect_params)
        g.update_step()

        # unlock
        augmentations.card.unlock()
    else:
        # lock
        augmentations.card.lock()
        utils.update_custom_button_params(augmentations.select_augs_button, utils.select_params)

        parameters.card.lock()
        # utils.update_custom_button_params(parameters.run_training, utils.select_params)

        utils.update_custom_button_params(select_splits_button, utils.select_params)

        # widgets to enable
        utils.disable_enable([trainval_splits, card], False)
        g.update_step(back=True)


def dump_train_val_splits(project_dir):
    # splits._project_id = None
    trainval_splits._project_fs = sly.Project(project_dir, sly.OpenMode.READ)
    g.splits = trainval_splits.get_splits()
    train_split, val_split = g.splits
    app_dir = g.data_dir  # @TODO: change to sly.app.get_synced_data_dir()?
    sly.json.dump_json_file(train_split, f"{app_dir}/train_split.json")
    sly.json.dump_json_file(val_split, f"{app_dir}/val_split.json")
