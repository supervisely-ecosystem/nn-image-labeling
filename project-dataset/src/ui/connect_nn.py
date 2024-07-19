import importlib

import supervisely as sly
import yaml
from supervisely.app.widgets import Button, Card, Container, ModelInfo, SelectAppSession, Text

g = importlib.import_module("project-dataset.src.globals")
inference_preview = importlib.import_module("project-dataset.src.ui.inference_preview")
inference_settings = importlib.import_module("project-dataset.src.ui.inference_settings")
nn_info = importlib.import_module("project-dataset.src.ui.nn_info")
output_data = importlib.import_module("project-dataset.src.ui.output_data")

select_session = SelectAppSession(g.team_id, g.deployed_nn_tags)
connect_button = Button("Connect", icon="zmdi zmdi-check")
disconnect_button = Button("Disconnect", icon="zmdi zmdi-close")
disconnect_button.hide()
error_text = Text(status="warning")
error_button = Button(button_size="small")
error_container = Container([error_text, error_button])
error_button.hide
error_container.hide()

model_info = ModelInfo()
model_info.hide()

card = Card(
    "2️⃣ Choose model",
    "Select a deployed neural network to start and click on Select button.",
    content=Container(
        [select_session, connect_button, model_info, disconnect_button, error_container]
    ),
    collapsable=True,
    lock_message="Select a project or a dataset on step 1️⃣.",
)
card.collapse()
card.lock()


@connect_button.click
def model_selected() -> None:
    """Connects to the selected model session and changes the UI state."""
    g.model_session_id = select_session.get_selected_id()
    if g.model_session_id is None:
        error_text.text = "No model was selected, please select a model and try again."
        error_container.show()
        return

    connect_status = connect_to_model()
    if not connect_status:
        error_text.text = (
            "Couldn't connect to the model. Make sure that model is deployed and try again."
        )
        app_url = f"/apps/sessions/{g.model_session_id}"

        error_button.text = "OPEN APP SESSION"
        error_button.link = app_url
        error_button.show()
        error_container.show()

        return

    g.model_meta = get_model_meta()
    g.inference_settings = get_inference_settings()
    inference_settings.additional_settings.set_text(g.inference_settings["settings"])

    error_button.hide()
    error_container.hide()
    model_info.set_session_id(g.model_session_id)
    g.model_info = g.api.task.send_request(g.model_session_id, "get_session_info", data={})
    sly.logger.debug(f"Model info was saved to globals: {g.model_info}")
    model_info.show()
    disconnect_button.show()

    connect_button.hide()
    select_session.hide()

    nn_info.load_classes()
    nn_info.load_tags()

    nn_info.card.unlock()
    nn_info.card.uncollapse()

    inference_settings.card.unlock()
    inference_settings.card.uncollapse()

    inference_preview.card.unlock()
    inference_preview.card.uncollapse()

    inference_preview.random_image_checkbox.enable()
    inference_preview.preview_button.enable()

    output_data.card.unlock()
    output_data.card.uncollapse()


@disconnect_button.click
def model_changed() -> None:
    """Changes the UI state when the model is changed."""
    connect_button.show()
    select_session.show()

    disconnect_button.hide()
    g.model_session_id = None
    g.model_meta = None
    sly.logger.info(f"Change button was clicked. Model session: {g.model_session_id}")
    model_info.hide()

    nn_info.card.lock()
    nn_info.card.collapse()

    inference_settings.card.lock()
    inference_settings.card.collapse()

    inference_preview.card.lock()
    inference_preview.card.collapse()

    inference_preview.random_image_checkbox.disable()
    inference_preview.preview_button.disable()

    output_data.card.lock()
    output_data.card.collapse()


def connect_to_model() -> bool:
    """Connects to the selected model session.

    :return: True if the connection was successful, False otherwise.
    :rtype: bool
    """
    try:
        session_info = g.api.task.send_request(g.model_session_id, "get_session_info", data={})
        sly.logger.info(f"Connected to model session: {session_info}")
        return True
    except Exception as e:
        sly.logger.warning(
            f"Couldn't get model info. Make sure that model is deployed and try again. Reason: {e}"
        )
        return False


def get_model_meta() -> sly.ProjectMeta:
    """Returns model meta in Supervisely format.

    :return: Model meta in Supervisely format.
    :rtype: sly.ProjectMeta
    """
    meta_json = g.api.task.send_request(g.model_session_id, "get_output_classes_and_tags", data={})
    return sly.ProjectMeta.from_json(meta_json)


def get_inference_settings() -> str:
    """Returns custom inference settings for the model.
    The settings are returned as a string in YAML format.

    :return: Custom inference settings for the model.
    :rtype: str
    """
    inference_settings = g.api.task.send_request(
        g.model_session_id, "get_custom_inference_settings", data={}
    )
    if inference_settings["settings"] is None or len(inference_settings["settings"]) == 0:
        inference_settings["settings"] = ""
        sly.logger.info("Model doesn't support custom inference settings.")
    elif isinstance(inference_settings["settings"], dict):
        inference_settings["settings"] = yaml.dump(
            inference_settings["settings"], allow_unicode=True
        )
    return inference_settings
