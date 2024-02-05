import supervisely as sly
from supervisely.app.widgets import Button, Card, Container, ModelInfo, SelectAppSession, Text

import project_dataset.src.globals as g
import project_dataset.src.ui.nn_info as nn_info

select_session = SelectAppSession(g.team_id, g.deployed_nn_tags)
connect_button = Button("Connect", icon="zmdi zmdi-check")
disconnect_button = Button("Disconnect", icon="zmdi zmdi-close")
disconnect_button.hide()
error_text = Text(status="warning")
error_text.hide()

model_info = ModelInfo()
model_info.hide()

card = Card(
    "2️⃣ Choose model",
    "Select a deployed neural network to start and click on Select button.",
    content=Container([select_session, connect_button, model_info, disconnect_button, error_text]),
    collapsable=True,
    lock_message="Select a project or a dataset on step 1️⃣.",
)
card.collapse()
card.lock()


@connect_button.click
def model_selected():
    g.model_session_id = select_session.get_selected_id()
    sly.logger.info(f"Select button was clicked. Model: {g.model_info}")
    if g.model_session_id is None:
        error_text.text = "No model was selected, please select a model and try again."
        error_text.show()
        return

    connect_status = connect_to_model()
    if not connect_status:
        error_text.text = (
            "Couldn't connect to the model. Make sure that model is deployed and try again."
        )
        error_text.show()
        return

    g.model_meta = get_model_meta()

    error_text.hide()
    model_info.set_session_id(g.model_session_id)
    model_info.show()
    disconnect_button.show()

    connect_button.hide()
    select_session.hide()

    nn_info.load_classes()
    nn_info.load_tags()

    nn_info.card.unlock()
    nn_info.card.uncollapse()


@disconnect_button.click
def model_changed():
    connect_button.show()
    select_session.show()

    disconnect_button.hide()
    g.model_session_id = None
    g.model_meta = None
    sly.logger.info(f"Change button was clicked. Model session: {g.model_session_id}")
    model_info.hide()
    nn_info.card.lock()
    nn_info.card.collapse()


def connect_to_model():
    try:
        session_info = g.api.task.send_request(g.model_session_id, "get_session_info", data={})
        sly.logger.info(f"Connected to model session: {session_info}")
        return True
    except Exception as e:
        sly.logger.warning(
            f"Couldn't get model info. Make sure that model is deployed and try again. Reason: {e}"
        )
        return False


def get_model_meta():
    meta_json = g.api.task.send_request(g.model_session_id, "get_output_classes_and_tags", data={})
    return sly.ProjectMeta.from_json(meta_json)
