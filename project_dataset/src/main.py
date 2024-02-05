import supervisely as sly
from supervisely.app.widgets import Container

import project_dataset.src.ui.connect_nn as connect_nn
import project_dataset.src.ui.input_data as input_data
import project_dataset.src.ui.nn_info as nn_info

layout = Container(widgets=[input_data.card, connect_nn.card, nn_info.card])

app = sly.Application(layout=layout)
