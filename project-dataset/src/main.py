import importlib

import supervisely as sly
from supervisely.app.widgets import Container

input_data = importlib.import_module("project-dataset.src.ui.input_data")
connect_nn = importlib.import_module("project-dataset.src.ui.connect_nn")
nn_info = importlib.import_module("project-dataset.src.ui.nn_info")
inference_settings = importlib.import_module("project-dataset.src.ui.inference_settings")
inference_preview = importlib.import_module("project-dataset.src.ui.inference_preview")
output_data = importlib.import_module("project-dataset.src.ui.output_data")
g = importlib.import_module("project-dataset.src.globals")

layout = Container(
    widgets=[
        input_data.card,
        connect_nn.card,
        nn_info.card,
        inference_settings.card,
        inference_preview.card,
        output_data.card,
    ],
)

app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)
