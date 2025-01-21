import importlib
import os

import yaml
from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    Container,
    Editor,
    Field,
    Flexbox,
    Input,
    Select,
    Text,
)
from supervisely.io.fs import silent_remove

g = importlib.import_module("project-dataset.src.globals")
inference_preview = importlib.import_module("project-dataset.src.ui.inference_preview")

inference_mode = Select(items=[Select.Item(mode) for mode in g.inference_modes])
inference_mode.set_value(g.inference_modes[0])
mode_field = Field(
    content=inference_mode,
    title="Inference mode",
    description="Select how to process images: full images or using sliding window.",
)

class_tag_suffix = Input(value="model", minlength=1)
class_tag_suffix_field = Field(
    content=class_tag_suffix,
    title="Class and tag suffix",
    description="Suffix that will be added to class and tag names.",
)
always_add_suffix = Checkbox("Always add suffix to model predictions")

add_predictions_mode = Select(items=[Select.Item(mode) for mode in g.add_predictions_modes])
add_predictions_mode.set_value(g.add_predictions_modes[0])
add_predictions_mode_field = Field(
    content=add_predictions_mode,
    title="Add predictions mode",
    description="Select how to add predictions to the project: by merging with existing labels or by replacing them.",
)

additional_setting_load_checkbox = Checkbox("Load additional settings from file", checked=False)
additional_settings = Editor(language_mode="yaml", height_lines=20)
additional_settings_input = Input("")

additional_settings_btn = Button(text="Save settings", icon="zmdi zmdi-floppy")
additional_settings_reset_btn = Button(text="Reset settings", plain=True, icon="zmdi zmdi-refresh")

additional_settings_text = Text("Enter path to save inference settings")
additional_settings_text_result = Text(
    "<b>Settings have been saved successfully</b>", status="success"
)
additional_settings_text_result.hide()

additional_settings_container = Container(
    [
        additional_setting_load_checkbox,
        additional_settings,
        additional_settings_text,
        additional_settings_input,
        Flexbox([additional_settings_btn, additional_settings_reset_btn], gap=5),
        additional_settings_text_result,
    ]
)
additional_settings_field = Field(
    content=additional_settings_container,
    title="Additional settings",
    description="Model specific inference settings in YAML format.",
)

card = Card(
    "4️⃣ Inference settings",
    "Choose additional settings for model inference.",
    content=Container(
        [
            mode_field,
            class_tag_suffix_field,
            always_add_suffix,
            add_predictions_mode_field,
            additional_settings_field,
        ]
    ),
    collapsable=True,
    lock_message="Connect to the deployed neural network on step 2️⃣.",
)
card.lock()
card.collapse()


@inference_mode.value_changed
def inference_mode_changed(mode: str) -> None:
    """Changes the UI state based on the selected inference mode."""
    if mode == "sliding window":
        inference_preview.sliding_window_container.show()
        inference_preview.preview_gallery.hide()
    else:
        inference_preview.sliding_window_container.hide()
        inference_preview.preview_gallery.show()


@additional_setting_load_checkbox.value_changed
def show_load_inference_settings(checked: bool) -> None:
    """Shows or hides the load settings button based on the checkbox value."""
    if checked:
        additional_settings_input.set_value("/nn-image-labeling/")
        additional_settings_btn.text = "Load Settings"
        additional_settings_btn.icon = "zmdi zmdi-cloud-download"
        additional_settings_text.set(
            "<b>Enter path to '.yaml' file to load inference settings</b>", status="text"
        )
        additional_settings_text_result.hide()
        additional_settings_text_result.set(
            "Settings have been loaded successfully", status="success"
        )
    else:
        additional_settings_input.set_value(g.additional_settings_save_path)
        additional_settings_btn.text = "Save Settings"
        additional_settings_btn.icon = "zmdi zmdi-floppy"
        additional_settings_text.set(
            "<b>Enter path to '.yaml' file to save inference settings</b>", status="text"
        )
        additional_settings_text_result.hide()
        additional_settings_text_result.set(
            "Settings have been saved successfully", status="success"
        )


@additional_settings_btn.click
def save_load_settings():
    # Load settings
    if additional_setting_load_checkbox.is_checked():
        remote_settings_path = additional_settings_input.get_value()
        file_info = g.api.file.get_info_by_path(g.team_id, remote_settings_path)
        if file_info is None:
            additional_settings_text_result.set(
                "Settings file not found, please check the path", status="warning"
            )
            additional_settings_text_result.show()
            return
        local_settings_path = os.path.join(g.STATIC_DIR, "inference_settings.yaml")
        if os.path.exists(local_settings_path):
            silent_remove(local_settings_path)
        g.api.file.download(g.team_id, remote_settings_path, local_settings_path)
        with open(local_settings_path, "r") as file:
            inference_settings = yaml.safe_load(file)
        additional_settings.set_text(yaml.dump(inference_settings))
        additional_settings_text_result.set(
            "Settings have been loaded successfully", status="success"
        )
        additional_settings_text_result.show()
    # Save settings
    else:
        inference_settings = additional_settings.get_text()
        local_settings_path = os.path.join(g.STATIC_DIR, "inference_settings.yaml")
        if os.path.exists(local_settings_path):
            silent_remove(local_settings_path)
        with open(local_settings_path, "w") as file:
            file.write(inference_settings)
        remote_settings_path = additional_settings_input.get_value()
        file_info = g.api.file.get_info_by_path(g.team_id, remote_settings_path)
        if file_info:
            additional_settings_text_result.set(
                "File already exists, please choose another file name", status="warning"
            )
            additional_settings_text_result.show()
            return

        g.api.file.upload(g.team_id, local_settings_path, remote_settings_path)
        additional_settings_text_result.set(
            "Settings have been saved successfully", status="success"
        )
        additional_settings_text_result.show()


@additional_settings_reset_btn.click
def reset_inference_settings():
    additional_settings.set_text(g.inference_settings["settings"])
    additional_settings_text_result.set("Settings have been reset successfully", status="success")
    additional_settings_text_result.show()
