import os
from collections import defaultdict

import supervisely as sly
from dotenv import load_dotenv

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ABSOLUTE_PATH)

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

# region ui-settings
selected_project = None
selected_datasets = None
model_session_id = None
model_meta = None

# endregion

# region ui-consants
deployed_nn_tags = ["deployed_nn"]
inference_modes = ["full image", "sliding window"]
add_predictions_modes = ["merge with existing labels", "replace existing labels"]

ann_cache = defaultdict(list)  # only one (current) image in cache
project_info = None
input_images = None
project_meta: sly.ProjectMeta = None

image_grid_options = {
    "opacity": 0.5,
    "fillRectangle": True,  # False,
    "enableZoom": False,
    "syncViews": True,
    "showPreview": True,
    "selectable": False,
    "showOpacityInHeader": True,
}

empty_gallery = {
    "content": {"projectMeta": {}, "annotations": {}, "layout": []},
    "options": image_grid_options,
}

# sliding window part

det_model_meta = None
model_info = None
inference_request_uuid = None
