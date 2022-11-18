import os
import pathlib
import sys
from collections import defaultdict

import supervisely as sly
from supervisely.app.v1.app_service import AppService

root_source_path = str(pathlib.Path(sys.argv[0]).parents[2])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

# only for debug
# from dotenv import load_dotenv
# load_dotenv(os.path.expanduser("~/supervisely.env"))
# load_dotenv("project-dataset/debug.env")

task_id = int(os.environ['TASK_ID'])

owner_id = int(os.environ['context.userId'])
team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])


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
    "fillRectangle": True, #False,
    "enableZoom": False,
    "syncViews": True,
    "showPreview": True,
    "selectable": False,
    "showOpacityInHeader": True,
}

empty_gallery = {
    "content": {
        "projectMeta": {},
        "annotations": {},
        "layout": []
    },
    "options": image_grid_options,
}

# sliding window part

det_model_meta = None

