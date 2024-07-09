import importlib
import os

import supervisely as sly
from dotenv import load_dotenv

w = importlib.import_module("project-dataset.src.workflow")

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ABSOLUTE_PATH)
STATIC_DIR = os.path.join(PARENT_DIR, "temp")
sly.fs.mkdir(STATIC_DIR)

if sly.is_development():
    load_dotenv(os.path.join(PARENT_DIR, "local.env"))
    load_dotenv(os.path.expanduser("~/supervisely.env"))

team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)

sly.logger.info(
    f"TEAM_ID: {team_id}, WORKSPACE_ID: {workspace_id}, PROJECT_ID: {project_id}, DATASET_ID: {dataset_id}"
)

api = sly.Api.from_env()

workflow = w.Workflow(api)

if dataset_id:
    dataset_info = api.dataset.get_info_by_id(dataset_id)
    project_id = dataset_info.project_id

# region ui-settings
selected_project = None
selected_datasets = None
model_session_id = None
model_meta = None
inference_settings = None
# endregion

# region ui-consants
deployed_nn_tags = ["deployed_nn"]
inference_modes = ["full image", "sliding window"]
add_predictions_modes = ["merge with existing labels", "replace existing labels"]
# endregion

# region caches
input_images = None
project_info = api.project.get_info_by_id(project_id)
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
# endregion

# region sliding window parameters
det_model_meta = None
model_info = None
inference_request_uuid = None
# endregion
