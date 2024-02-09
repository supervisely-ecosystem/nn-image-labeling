import supervisely as sly
from supervisely.app.widgets import Button, Card, Container, SelectDataset, Text

import project_dataset.src.globals as g
import project_dataset.src.ui.connect_nn as connect_nn
import project_dataset.src.ui.inference_preview as inference_preview

select_dataset = SelectDataset(multiselect=True, project_id=g.project_id, default_id=g.dataset_id)
select_button = Button("Select")
change_button = Button("Change")
change_button.hide()
no_datasets_text = Text(
    "No datasets were selected, please select at least one dataset.", status="warning"
)
no_datasets_text.hide()

card = Card(
    "1️⃣ Choose input",
    "Select a project or a dataset to start",
    content=Container([select_dataset, select_button, no_datasets_text]),
    content_top_right=change_button,
    collapsable=True,
    lock_message="Click on Chhange button to select another project or dataset.",
)


@select_button.click
def datasets_selected():
    g.selected_project = select_dataset.get_selected_project_id()
    g.selected_datasets = select_dataset.get_selected_ids()
    sly.logger.info(
        f"Select button was clicked. Project: {g.selected_project}, Datasets: {g.selected_datasets}"
    )
    if not g.selected_datasets:
        no_datasets_text.show()
        return
    no_datasets_text.hide()

    g.input_images = None
    cache_input_images()
    inference_preview.create_image_selector()

    connect_nn.card.unlock()
    connect_nn.card.uncollapse()
    card.lock()
    change_button.show()


@change_button.click
def datasets_changed():
    connect_nn.card.lock()
    connect_nn.card.collapse()
    card.unlock()
    change_button.hide()
    g.selected_project = None
    g.selected_datasets = None
    sly.logger.info(
        f"Change button was clicked. Project: {g.selected_project}, Datasets: {g.selected_datasets}"
    )


def cache_input_images() -> None:
    """Cache input images for the model inference."""
    g.input_images = []
    for dataset_id in g.selected_datasets:
        g.input_images.extend(g.api.image.get_list(dataset_id))
    sly.logger.debug(f"Input images were cached: {len(g.input_images)} images.")
