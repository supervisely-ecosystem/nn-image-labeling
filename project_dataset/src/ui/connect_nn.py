import supervisely as sly
from supervisely.app.widgets import Button, Card, Container, ModelInfo, SelectAppSession, Text

import project_dataset.src.globals as g

select_session = SelectAppSession(g.team_id, g.deployed_nn_tags)
select_button = Button("Select")
change_button = Button("Change")
change_button.hide()
no_sessions_text = Text(
    "App session was not selected, please select it and press the button again.", status="warning"
)
no_sessions_text.hide()

model_info = ModelInfo()
model_info.hide()

card = Card(
    "2️⃣ Choose model",
    "Select a deployed neural network to start and click on Select button.",
    content=Container([select_session, select_button, model_info, no_sessions_text]),
    content_top_right=change_button,
    collapsable=True,
    lock_message="Click on Change button to select another app session, input data on step 1️⃣ must be selected.",
)
card.collapse()
card.lock()


@select_button.click
def model_selected():
    g.model_session_id = select_session.get_selected_id()
    sly.logger.info(f"Select button was clicked. Model: {g.model_info}")
    if g.model_session_id is None:
        no_sessions_text.show()
        return
    no_sessions_text.hide()
    model_info.set_session_id(g.model_session_id)
    model_info.show()
    change_button.show()

    select_button.hide()
    select_session.hide()
    # TODO: Unlock next card in UI


@change_button.click
def model_changed():
    select_button.show()
    select_session.show()

    change_button.hide()
    g.model_session_id = None
    g.model_meta = None
    sly.logger.info(f"Change button was clicked. Model: {g.model_info}")
    model_info.hide()
    # TODO: Lock next card in UI
