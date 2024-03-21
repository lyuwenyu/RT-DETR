import os

import supervisely as sly
from dotenv import load_dotenv

if sly.is_development:
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

# region envvars
team_id = sly.env.team_id()
wotkspace_id = sly.env.workspace_id()
project_id = sly.env.project_id()

# endregion
api = sly.Api.from_env()

# region state
selected_project_info = None
# endregion
