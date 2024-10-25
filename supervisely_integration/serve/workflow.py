# This module contains functions that are used to configure the input and output of the workflow for the current app.

import supervisely as sly


def workflow_input(api: sly.Api, model_params: dict):
    try:
        checkpoint_url = model_params.get("checkpoint_url")
        checkpoint_name = model_params.get("checkpoint_name")
        model_name = "RT-DETR"

        meta = sly.WorkflowMeta(node_settings=sly.WorkflowSettings(title=f"Serve {model_name}"))

        sly.logger.debug(
            f"Workflow Input: Checkpoint URL - {checkpoint_url}, Checkpoint Name - {checkpoint_name}"
        )
        if checkpoint_url and api.file.exists(sly.env.team_id(), checkpoint_url):
            api.app.workflow.add_input_file(checkpoint_url, model_weight=True, meta=meta)
        else:
            sly.logger.debug(
                f"Checkpoint {checkpoint_url} not found in Team Files. Cannot set workflow input"
            )
    except Exception as e:
        sly.logger.debug(f"Failed to add input to the workflow: {repr(e)}")


def workflow_output(self):
    raise NotImplementedError("add_output is not implemented in this workflow")
