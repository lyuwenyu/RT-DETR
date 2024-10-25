from collections import namedtuple
from typing import Dict, List, Optional

import supervisely as sly
from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    Container,
    Field,
    RadioTable,
    RadioTabs,
    TeamFilesSelector,
)

import supervisely_integration.train.globals as g
import supervisely_integration.train.ui.augmentations as augmentations
import supervisely_integration.train.ui.classes as classes
import supervisely_integration.train.ui.parameters as parameters
import supervisely_integration.train.ui.splits as splits
import supervisely_integration.train.ui.utils as utils

TrainMode = namedtuple("TrainMode", ["pretrained", "custom", "finetune"])


select_custom_weights = TeamFilesSelector(g.TEAM_ID, selection_file_type="file")
pretrained_models_table = RadioTable(columns=g.TABLE_COLUMNS, rows=g.PRETRAINED_MODELS)

finetune_checkbox = Checkbox("Fine-tune", True)
finetune_field = Field(
    finetune_checkbox,
    title="Enable fine-tuning",
    description="Fine-tuning allows you to continue training a model from a checkpoint. If not selected, model will be trained from scratch and will be configured as in selected checkpoint.",
)

select_model_button = Button("Select")

model_mode = RadioTabs(
    g.MODEL_MODES.copy(),
    contents=[
        Container([pretrained_models_table, finetune_field]),
        select_custom_weights,
    ],
)

card = Card(
    title="Select a model",
    description="Select a model to train",
    content=Container([model_mode, select_model_button]),
    lock_message="Click on the Change model button to select another model",
)


@select_model_button.click
def model_selected():
    if select_model_button.text == "Select":
        # widgets to disable
        utils.disable_enable(
            [
                select_custom_weights,
                pretrained_models_table,
                finetune_checkbox,
                finetune_field,
                model_mode,
                card,
            ],
            True,
        )

        mode = model_mode.get_active_tab()
        g.model_mode = mode
        if mode == g.MODEL_MODES[0]:
            pretrained: List[str] = pretrained_models_table.get_selected_row()
            custom = None
            sly.logger.debug(f"Selected mode: {mode}, selected pretrained model: {pretrained}")
        else:
            pretrained = None
            custom: List[str] = select_custom_weights.get_selected_paths()
            # TODO: Add single-item mode to the widget and remove indexing
            custom = custom[0] if custom else None
            sly.logger.debug(f"Selected mode: {mode}, path to custom weights: {custom}")
        finetune = finetune_checkbox.is_checked()
        g.train_mode = TrainMode(pretrained, custom, finetune)
        classes.fill_classes_selector()

        utils.update_custom_button_params(select_model_button, utils.reselect_params)
        g.update_step()

        # unlock
        classes.card.unlock()
    else:
        g.train_mode = None
        g.model_mode = None
        classes.fill_classes_selector(clear=True)

        # lock
        classes.card.lock()
        utils.update_custom_button_params(classes.select_classes_button, utils.select_params)

        splits.card.lock()
        utils.update_custom_button_params(splits.select_splits_button, utils.select_params)

        augmentations.card.lock()
        utils.update_custom_button_params(augmentations.select_augs_button, utils.select_params)

        parameters.card.lock()
        # utils.update_custom_button_params(parameters.run_training, utils.select_params)

        utils.update_custom_button_params(select_model_button, utils.select_params)

        # widgets to enable
        utils.disable_enable(
            [
                select_custom_weights,
                pretrained_models_table,
                finetune_checkbox,
                finetune_field,
                model_mode,
                card,
            ],
            False,
        )
        g.update_step(back=True)
