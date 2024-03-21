from supervisely.app.widgets import Button, Card, Container, ProjectThumbnail, SelectProject

import supervisely_integration.train.globals as g

select_project = SelectProject(g.project_id, g.wotkspace_id, compact=True)
select_project_button = Button("Select project")
change_project_button = Button("Change project")
change_project_button.hide()

project_thumbnail = ProjectThumbnail()
project_thumbnail.hide()

card = Card(
    title="1️⃣ Select a project",
    description="Select a project to train a model on",
    collapsable=True,
    content=Container([select_project, select_project_button, project_thumbnail]),
    content_top_right=change_project_button,
    lock_message="Click on the Change project button to select another project",
)


@select_project_button.click
def project_selected():
    g.selected_project_info = g.api.project.get_info_by_id(select_project.get_selected_id())
    project_thumbnail.set(g.selected_project_info)
    project_thumbnail.show()
    change_project_button.show()
    card.lock()


@change_project_button.click
def change_project():
    g.selected_project_info = None
    project_thumbnail.set(None)
    project_thumbnail.hide()
    change_project_button.hide()
    card.unlock()
