import supervisely as sly
from supervisely.app.widgets import Container

import project_dataset.src.ui.connect_nn as connect_nn
import project_dataset.src.ui.inference_preview as inference_preview
import project_dataset.src.ui.inference_settings as inference_settings
import project_dataset.src.ui.input_data as input_data
import project_dataset.src.ui.nn_info as nn_info
import project_dataset.src.ui.output_data as output_data

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

app = sly.Application(layout=layout)
