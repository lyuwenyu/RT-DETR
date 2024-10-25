import supervisely as sly
from supervisely.app.widgets import Card, Checkbox, Container, ProjectThumbnail, Text
from supervisely.project.download import is_cached

import supervisely_integration.train.globals as g

g.project_info = g.api.project.get_info_by_id(g.PROJECT_ID)
g.project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(g.PROJECT_ID))
sly.logger.info("Project meta saved into globals.")
sly.logger.info(f"Selected project: {g.project_info.name} with ID: {g.project_info.id}")
project_thumbnail = ProjectThumbnail(g.project_info)

if is_cached(g.PROJECT_ID):
    _text = "Use cached data stored on the agent to optimize project download"
else:
    _text = "Cache data on the agent to optimize project download for future trainings"
use_cache_text = Text(_text)
use_cache_checkbox = Checkbox(use_cache_text, checked=g.USE_CACHE)

card = Card(
    title="Selected project",
    description="The project that will be used for training.",
    content=Container(widgets=[project_thumbnail, use_cache_checkbox]),
)
