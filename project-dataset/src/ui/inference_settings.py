import importlib
import json
import os
from pathlib import Path

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
from supervisely.io.fs import get_file_ext, silent_remove


def load_templates():
    file_infos = g.api.file.list(g.team_id, g.remote_template_dir, False, "fileinfo")
    if len(file_infos) == 0:
        additional_settings_text_result.set("No templates found", status="info")
        additional_settings_text_result.show()
        return [], {}

    items = []
    items_map = {}
    for file_info in file_infos:
        local_path = os.path.join(g.STATIC_DIR, file_info.name)
        if get_file_ext(local_path) != ".yaml":
            continue

        g.api.file.download(g.team_id, file_info.path, local_path)
        with open(local_path, "r") as file:
            inference_settings = yaml.safe_load(file)
        items.append(Select.Item(file_info.name, file_info.name))
        items_map[file_info.name] = inference_settings
        silent_remove(local_path)
    return items, items_map


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

# ignore prediction if IoU with existing label ≥ threshold",
ignore_labels_if_iou = Checkbox("enable")
iou_threshold_field = Field(
    content=Container([ignore_labels_if_iou]),
    title="Skip predictions if IoU with existing label ≥ threshold",
    description="If enabled, the `nms_iou_thresh_with_gt` setting will be used to filter predictions based on IoU with existing labels.",
)
add_predictions_mode = Select(items=[Select.Item(mode) for mode in g.add_predictions_modes])
add_predictions_mode.set_value(g.add_predictions_modes[0])
add_predictions_mode_field = Field(
    content=add_predictions_mode,
    title="Add predictions mode",
    description="Select how to add predictions to the project: by merging with existing labels or by replacing them.",
)

additional_setting_load_checkbox = Checkbox("Load additional settings from file", checked=False)
additional_settings = Editor(language_mode="yaml", height_lines=20)
additional_settings_input = Input("settings_template.yaml")

additional_settings_btn = Button(text="Save settings", icon="zmdi zmdi-floppy")
additional_settings_reset_btn = Button(text="Reset settings", plain=True, icon="zmdi zmdi-refresh")

additional_settings_text = Text(
    "<b>Enter name of the template with '.yaml' extension to save inference settings</b>"
)
additional_settings_text_result = Text(
    "<b>Settings have been saved successfully</b>", status="success"
)
additional_settings_text_result.hide()

items, g.templates_map = load_templates()
additional_settings_selector = Select(items, placeholder="Select template")
additional_settings_selector.hide()
additional_settings_selector_refresh_btn = Button("", "text", icon="zmdi zmdi-refresh")
additional_settings_selector_refresh_btn.hide()

additional_settings_container = Container(
    [
        additional_setting_load_checkbox,
        additional_settings,
        additional_settings_text,
        additional_settings_input,
        Flexbox([additional_settings_selector, additional_settings_selector_refresh_btn], gap=3),
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
            iou_threshold_field,
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
        additional_settings_text.set("<b>Select inference settings template</b>", status="text")
        additional_settings_input.hide()
        additional_settings_selector.show()
        additional_settings_selector_refresh_btn.show()
        additional_settings_btn.text = "Load Settings"
        additional_settings_btn.icon = "zmdi zmdi-cloud-download"
        if len(items) == 0:
            additional_settings_btn.disable()
            additional_settings_text_result.set("No templates found", status="info")
            additional_settings_text_result.show()

        additional_settings_text_result.hide()
    else:
        additional_settings_selector.hide()
        additional_settings_selector_refresh_btn.hide()
        additional_settings_input.show()
        additional_settings_btn.text = "Save Settings"
        additional_settings_btn.icon = "zmdi zmdi-floppy"
        additional_settings_btn.enable()
        additional_settings_text.set(
            "<b>Enter name of the template with '.yaml' extension to save inference settings</b>",
            status="text",
        )
        additional_settings_text_result.hide()


@additional_settings_btn.click
def save_load_settings():
    # Load settings
    if additional_setting_load_checkbox.is_checked():
        template_name = additional_settings_selector.get_value()
        inference_settings = g.templates_map[template_name]
        additional_settings.set_text(yaml.dump(inference_settings))
        additional_settings_text_result.set(
            "Settings have been loaded successfully", status="success"
        )
        additional_settings_text_result.show()
    # Save settings
    else:
        remote_settings_path = g.remote_template_dir + additional_settings_input.get_value()
        if not remote_settings_path.endswith(".yaml"):
            additional_settings_text_result.set(
                "File name should have '.yaml' extension", status="warning"
            )
            additional_settings_text_result.show()
            return

        inference_settings = additional_settings.get_text()
        local_settings_path = os.path.join(g.STATIC_DIR, "inference_settings.yaml")
        if os.path.exists(local_settings_path):
            silent_remove(local_settings_path)
        with open(local_settings_path, "w") as file:
            file.write(inference_settings)
        file_info = g.api.file.get_info_by_path(g.team_id, remote_settings_path)
        if file_info:
            additional_settings_text_result.set(
                "File already exists, please choose another file name", status="warning"
            )
            additional_settings_text_result.show()
            return

        g.api.file.upload(g.team_id, local_settings_path, remote_settings_path)
        additional_settings_text_result.set(
            f"Settings have been saved successfully to: '{remote_settings_path}'", status="success"
        )
        additional_settings_text_result.show()


@additional_settings_reset_btn.click
def reset_inference_settings():
    additional_settings.set_text(g.inference_settings["settings"])
    additional_settings_text_result.set("Settings have been reset successfully", status="success")
    additional_settings_text_result.show()


@additional_settings_selector_refresh_btn.click
def refresh_tempaltes():
    items, g.templates_map = load_templates()
    additional_settings_selector.set(items)
    if len(items) == 0:
        additional_settings_btn.disable()
    else:
        additional_settings_btn.enable()
