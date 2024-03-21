import supervisely as sly
from supervisely.app.widgets import Container

import supervisely_integration.train.ui.input as input

layout = Container([input.card])

app = sly.Application(layout=layout)
