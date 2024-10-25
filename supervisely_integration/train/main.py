import supervisely as sly
from supervisely.app.widgets import Stepper

import supervisely_integration.train.globals as g
import supervisely_integration.train.ui.augmentations as augmentations
import supervisely_integration.train.ui.classes as classes
import supervisely_integration.train.ui.input as input
import supervisely_integration.train.ui.model as model
import supervisely_integration.train.ui.output as output
import supervisely_integration.train.ui.parameters as parameters
import supervisely_integration.train.ui.splits as splits

g.stepper = Stepper(
    widgets=[
        input.card,
        model.card,
        classes.card,
        splits.card,
        augmentations.card,
        parameters.card,
        output.card,
    ],
    active_step=2,
    widget_id="main_stepper",
)

app = sly.Application(layout=g.stepper)
g.app = app
