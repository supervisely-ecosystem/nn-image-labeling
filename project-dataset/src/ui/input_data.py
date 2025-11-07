import importlib
from collections import defaultdict
from typing import List

import supervisely as sly
from supervisely.app.widgets import Button, Card, Container, Progress, SelectDatasetTree, Text

caching_progress = Progress(hide_on_finish=True)

g = importlib.import_module("project-dataset.src.globals")
connect_nn = importlib.import_module("project-dataset.src.ui.connect_nn")
inference_preview = importlib.import_module("project-dataset.src.ui.inference_preview")

select_dataset = SelectDatasetTree(
    multiselect=True,
    project_id=g.project_id,
    default_id=g.dataset_id,
    select_all_datasets=g.dataset_id is None,
    width=250,
)
if g.dataset_id:
    select_dataset.set_dataset_ids([g.dataset_id])
select_button = Button("Select")
change_button = Button("Change")
change_button.hide()
no_datasets_text = Text(
    "No datasets were selected, please select at least one dataset.", status="warning"
)
caching_progress.hide()
no_datasets_text.hide()

card = Card(
    "1️⃣ Choose input",
    "Select a project or a dataset to start",
    content=Container([select_dataset, select_button, caching_progress, no_datasets_text]),
    content_top_right=change_button,
    collapsable=True,
    lock_message="Click on the Change button to select another project or dataset.",
)


@select_button.click
def datasets_selected() -> None:
    """Changes the UI state based on the selected datasets,
    caches input images and creates the image selector."""
    selected_project = select_dataset.get_selected_project_id()
    if selected_project != g.selected_project:
        g.project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(selected_project))
    g.init_cache_globals(selected_project)
    g.selected_project = selected_project
    g.selected_datasets = select_dataset.get_selected_ids()
    sly.logger.info(
        f"Select button was clicked. Project: {g.selected_project}, Datasets: {g.selected_datasets}"
    )

    if not g.selected_datasets:
        no_datasets_text.show()
        return
    no_datasets_text.hide()
    g.selected_datasets_aggregated = get_selected_datasets()
    g.input_images = None
    cache_input_images()
    inference_preview.create_image_selector()

    connect_nn.card.unlock()
    connect_nn.card.uncollapse()
    card.lock()
    change_button.show()


@change_button.click
def datasets_changed():
    """Changes the UI state based on the selected datasets and clears the cached input images."""
    connect_nn.card.lock()
    connect_nn.card.collapse()
    card.unlock()
    change_button.hide()
    g.selected_project = None
    g.selected_datasets = None
    g.selected_datasets_aggregated = None
    sly.logger.info(
        f"Change button was clicked. Project: {g.selected_project}, Datasets: {g.selected_datasets}"
    )


def get_selected_datasets() -> List[sly.DatasetInfo]:
    selected_ids = set(g.selected_datasets)
    all_datasets = []

    if not g.selected_datasets:
        return g.project_datasets

    parent_to_children = defaultdict(list)
    for ds in g.project_datasets:
        current = ds
        while parent_id := current.parent_id:
            parent_to_children[parent_id].append(ds)
            current = g.id_to_info[parent_id]

    for dataset in g.project_datasets:
        if dataset.id in selected_ids:
            datasets = [dataset] + parent_to_children.get(dataset.id, [])
            if len(datasets) > 1:
                sly.logger.debug(
                    f"Adding dataset {len(datasets) - 1} children dataset(s) to the selected dataset '{dataset.name}'."
                )
            all_datasets.extend(datasets)

    return all_datasets


@sly.timeit
def cache_input_images(
    api: sly.Api = g.api,
    selected_datasets: List[sly.DatasetInfo] = g.selected_datasets_aggregated,
) -> None:
    """
    Cache input images for the model inference.
    """
    # Calculate total images count
    total_images = sum(ds.images_count for ds in selected_datasets)

    async def _cache_images():
        g.input_images = []
        for dataset in selected_datasets:
            async for imgs in api.image.get_list_generator_async(dataset.id):
                g.input_images.extend(imgs)
                caching_progress.update(len(g.input_images))

    caching_progress.show()
    with caching_progress(message="Caching input images...", total=total_images):
        sly.run_coroutine(_cache_images())

    sly.logger.debug(f"Input images were cached: {len(g.input_images)} images.")
    sly.logger.debug(f"Input images were cached: {len(g.input_images)} images.")
