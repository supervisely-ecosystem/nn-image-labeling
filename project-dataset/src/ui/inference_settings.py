import importlib

from supervisely.app.widgets import Card, Checkbox, Container, Editor, Field, Input, Select

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

additional_settings = Editor(language_mode="yaml", height_lines=20)
additional_settings_field = Field(
    content=additional_settings,
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
