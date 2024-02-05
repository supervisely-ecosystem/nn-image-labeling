import supervisely as sly
from supervisely.app.widgets import Button, Card, Container, ModelInfo, SelectAppSession, Text

import project_dataset.src.globals as g

select_session = SelectAppSession(g.team_id, g.deployed_nn_tags)
connect_button = Button("Connect", icon="zmdi zmdi-check")
disconnect_button = Button("Disconnect", icon="zmdi zmdi-close")
disconnect_button.hide()
no_sessions_text = Text(
    "App session was not selected, please select it and press the button again.", status="warning"
)
no_sessions_text.hide()

model_info = ModelInfo()
model_info.hide()

card = Card(
    "2️⃣ Choose model",
    "Select a deployed neural network to start and click on Select button.",
    content=Container(
        [select_session, connect_button, model_info, disconnect_button, no_sessions_text]
    ),
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
        no_sessions_text.show()
        return
    no_sessions_text.hide()
    model_info.set_session_id(g.model_session_id)
    model_info.show()
    disconnect_button.show()

    connect_button.hide()
    select_session.hide()
    # TODO: Unlock next card in UI


@disconnect_button.click
def model_changed():
    connect_button.show()
    select_session.show()

    disconnect_button.hide()
    g.model_session_id = None
    g.model_meta = None
    sly.logger.info(f"Change button was clicked. Model: {g.model_info}")
    model_info.hide()
    # TODO: Lock next card in UI
