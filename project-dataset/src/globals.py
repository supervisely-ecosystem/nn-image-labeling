import os
import supervisely_lib as sly

my_app: sly.AppService = sly.AppService(ignore_task_id=True)

team_id = int(os.environ["context.teamId"])
project_id = int(os.environ["modal.state.slyProjectId"])


task_id = my_app.task_id
api: sly.Api = my_app.public_api
project_info = api.project.get_info_by_id(project_id)
