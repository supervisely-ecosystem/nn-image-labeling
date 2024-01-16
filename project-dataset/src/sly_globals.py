import os
import pathlib
import sys
from collections import defaultdict

import supervisely as sly
from supervisely.app.v1.app_service import AppService
from supervisely.io.fs import file_exists

root_source_path = str(pathlib.Path(sys.argv[0]).parents[2])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)
parent_path = str(pathlib.Path(sys.argv[0]).parents[1])
sly.logger.info(f"Parent directory: {parent_path}")

# only for debug
from dotenv import load_dotenv

if sly.is_development():
    local_debug_env_path = os.path.join(parent_path, "debug.env")
    secret_debug_env_path = os.path.join(parent_path, "secret_debug.env")
    supervisely_env_path = os.path.expanduser("~/supervisely.env")

    if not file_exists(secret_debug_env_path) and not file_exists(supervisely_env_path):
        raise Exception(
            f"Credentials not found. Please create secret env file with credentials. "
            "Read more: https://developer.supervisely.com/getting-started/basics-of-authentication"
        )

    load_dotenv(local_debug_env_path)
    load_dotenv(supervisely_env_path)
    load_dotenv(secret_debug_env_path, override=True)


task_id = int(os.environ["TASK_ID"])

owner_id = int(os.environ["context.userId"])
team_id = int(os.environ["context.teamId"])
workspace_id = int(os.environ["context.workspaceId"])


project_id = None
dataset_id = os.environ.get("modal.state.slyDatasetId")
input_datasets = []
if dataset_id is not None:
    dataset_id = int(dataset_id)
else:
    project_id = int(os.environ["modal.state.slyProjectId"])

my_app: AppService = AppService(ignore_task_id=True)
api = my_app.public_api

model_meta: sly.ProjectMeta = None

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
