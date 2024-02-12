import importlib
import os
from datetime import datetime
from random import choice
from time import sleep
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import supervisely as sly
import yaml
from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    Container,
    Empty,
    Field,
    GridGallery,
    InputNumber,
    ReloadableArea,
    Select,
    VideoPlayer,
)

g = importlib.import_module("project-dataset.src.globals")
settings = importlib.import_module("project-dataset.src.ui.inference_settings")
nn_info = importlib.import_module("project-dataset.src.ui.nn_info")

preview_gallery = GridGallery(
    columns_number=2,
    sync_views=True,
    enable_zoom=True,
    resize_on_zoom=True,
    empty_message="Click Preview button to see the result.",
)
preview_gallery.hide()
preview_video = VideoPlayer()
preview_video.hide()

random_image_checkbox = Checkbox("Random image", checked=True)
random_image_checkbox.disable()
preview_button = Button("Preview", icon="zmdi zmdi-eye")
preview_button.disable()

window_height = InputNumber(value=256, min=1, max=10000)
window_height_field = Field(
    content=window_height,
    title="Sliding window height",
    description="Set the height of the sliding window in pixels.",
)

window_width = InputNumber(value=256, min=1, max=10000)
window_width_field = Field(
    content=window_width,
    title="Sliding window width",
    description="Set the width of the sliding window in pixels.",
)

vertical_overlap = InputNumber(value=32, min=0, max=10000)
vertical_overlap_field = Field(
    content=vertical_overlap,
    title="Vertical overlap",
    description="Set the vertical overlap in pixels.",
)

horizontal_overlap = InputNumber(value=32, min=0, max=10000)
horizontal_overlap_field = Field(
    content=horizontal_overlap,
    title="Horizontal overlap",
    description="Set the horizontal overlap in pixels.",
)

vizulazation_fps = InputNumber(value=4, min=1, max=20)
vizulazation_fps_field = Field(
    content=vizulazation_fps,
    title="Visualization FPS",
    description="Set the visualization frames per second.",
)

sliding_window_settings = Container(
    [
        window_height_field,
        window_width_field,
        vertical_overlap_field,
        horizontal_overlap_field,
        vizulazation_fps_field,
    ]
)
sliding_window_container = Container(
    [sliding_window_settings, preview_video], direction="horizontal"
)
sliding_window_container.hide()

image_selector_container = Container()
image_selector_ra = ReloadableArea(content=image_selector_container)
image_selector_ra.hide()

card = Card(
    "5️⃣ Inference preview",
    "Preview the model inference on a random image.",
    content=Container([preview_gallery, sliding_window_container]),
    content_top_right=Container(
        [image_selector_ra, Container([Empty(), random_image_checkbox]), preview_button],
        direction="horizontal",
        style="padding: 5px",
    ),
    collapsable=True,
    lock_message="Connect to the deployed neural network on step 2️⃣.",
)
card.lock()
card.collapse()


@preview_button.click
def create_preview() -> None:
    """Select the function to create a preview based on the inference mode."""
    if settings.inference_mode.get_value() == "sliding window":
        window_preview()
    else:
        full_preview()


def full_preview() -> None:
    """Create a full image preview of the model inference and show it in the gallery."""
    preview_video.hide()
    try:
        inference_settings = yaml.safe_load(settings.additional_settings.get_value())
    except Exception as e:
        inference_settings = {}
        sly.logger.warning(
            f"Model Inference launched without additional settings. \n" f"Reason: {e}",
            exc_info=True,
        )

    if random_image_checkbox.is_checked():
        image_info: sly.ImageInfo = choice(g.input_images)
    else:
        image_info = get_selected_image()

    sly.logger.info(
        f"Image with id {image_info.id} and name {image_info.name} was selected for preview"
    )

    input_ann, result_ann, result_project_meta = apply_model_to_image(
        image_info, inference_settings
    )

    preview_gallery.clean_up()

    # Add image with original annotations to the gallery.
    preview_gallery.append(
        image_info.preview_url,
        input_ann,
        f"{image_info.name} (original)",
    )

    # Add image with result annotations to the gallery.
    preview_gallery.append(
        image_info.preview_url,
        result_ann,
        f"{image_info.name} (result)",
    )

    preview_gallery.show()


def window_preview() -> None:
    """Create a sliding window preview of the model inference and show it in the gallery."""
    preview_gallery.hide()
    if random_image_checkbox.is_checked():
        image_info: sly.ImageInfo = choice(g.input_images)
    else:
        image_info = get_selected_image()

    sly.logger.info(
        f"Image with id {image_info.id} and name {image_info.name} was selected for preview"
    )

    check_sliding_sizes_by_image(image_info)
    inference_setting = get_sliding_window_params()

    ann_pred_res = g.api.task.send_request(
        g.model_session_id,
        "inference_image_id",
        data={"image_id": image_info.id, "settings": inference_setting},
        timeout=200,
    )

    try:
        predictions = ann_pred_res["data"]["slides"]
    except Exception as ex:
        raise ValueError("Cannot parse slides predictions, reason: {}".format(repr(ex)))

    image_np = g.api.image.download_np(image_info.id)
    video_name = write_video(image_np, predictions)

    preview_video.set_video(f"/static/{video_name}")
    preview_video.show()


def create_image_selector() -> None:
    """Create a selector with the input images and show it in the reloadable area."""
    items = []
    for image_info in g.input_images:
        items.append(
            Select.Item(
                label=image_info.name,
                value=image_info.id,
            )
        )

    image_selector = Select(items, filterable=True, placeholder="Select image")
    image_selector.set_value("")
    image_selector_ra._content._widgets = [image_selector]
    image_selector_ra.reload()


@random_image_checkbox.value_changed
def toggle_image_select_mode(is_random: bool) -> None:
    """Toggle the image selection mode between random and manual."""
    if is_random:
        image_selector_ra.hide()
    else:
        image_selector_ra.show()


def get_selected_image() -> sly.ImageInfo:
    """Returns the selected image from the image selector.

    :return: Selected image.
    :rtype: sly.ImageInfo
    """
    image_selector = image_selector_ra._content._widgets[0]
    selected_value = image_selector.get_value()
    return g.api.image.get_info_by_id(int(selected_value))


# region legacy
# This code comes from the legacy version of the app mostly as is.
# Functions were modified to use the global variables with less arguments
# and UI state was removed (as it is not used in the new version of the app).
# Consider refactoring logic of functions later.
def check_sliding_sizes_by_image(image_info: sly.ImageInfo) -> None:
    """Checks sliding window sizes by the image and updates them if necessary.

    :param image_info: Image to check sliding window sizes by.
    :type image_info: sly.ImageInfo
    """
    if window_height.get_value() > image_info.height:
        window_height.value = image_info.height

    if window_width.get_value() > image_info.width:
        window_width.value = image_info.width


def write_video(
    image_np: np.ndarray,
    predictions: Dict[str, Any],
    last_two_frames_copies: Optional[int] = 8,
    max_video_size: Optional[int] = 1080,
) -> str:
    """Writes a video with the model inference preview.

    :param image_np: Image to write the video from.
    :type image_np: np.ndarray
    :param predictions: Model inference predictions.
    :type predictions: Dict[str, Any]
    :param last_two_frames_copies: Number of copies of the last two frames to add to the video.
    :type last_two_frames_copies: Optional[int]
    :param max_video_size: Maximum size of the video.
    :type max_video_size: Optional[int]
    :return: Video name.
    :rtype: str
    """
    scale_ratio = None
    if image_np.shape[1] > max_video_size:
        scale_ratio = max_video_size / image_np.shape[1]
        image_np = cv2.resize(
            image_np, (int(image_np.shape[1] * scale_ratio), int(image_np.shape[0] * scale_ratio))
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_name = f"preview_{timestamp}.mp4"

    video_path = os.path.join(g.STATIC_DIR, video_name)
    video = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"VP90"),
        vizulazation_fps.get_value(),
        (image_np.shape[1], image_np.shape[0]),
    )

    for i, pred in enumerate(predictions):
        rect = pred["rectangle"]
        rect = sly.Rectangle.from_json(rect)
        if scale_ratio is not None:
            rect = rect.scale(scale_ratio)
        labels = pred["labels"]
        for label_ind, label in enumerate(labels):
            labels[label_ind] = sly.Label.from_json(label, g.model_meta)

        frame = image_np.copy()
        rect.draw_contour(frame, [255, 0, 0], thickness=5)
        for label in labels:
            if scale_ratio is not None:
                label = sly.Label(label.geometry.scale(scale_ratio), label.obj_class)
            label.draw_contour(frame, thickness=3)
        sly.image.write(os.path.join(g.STATIC_DIR, f"{i:05d}.jpg"), frame)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if i >= len(predictions) - 2:
            for n in range(last_two_frames_copies):
                video.write(frame_bgr)
        else:
            video.write(frame_bgr)

    video.release()
    return video_name


def apply_model_to_image(
    image_info: sly.ImageInfo, inference_setting: Dict[str, Union[float, bool]]
) -> Tuple[sly.Annotation, sly.Annotation, sly.ProjectMeta]:
    """Adapter function to apply model to a single image.

    :param image_info: Image to apply model to.
    :type image_info: sly.ImageInfo
    :param inference_setting: Inference settings.
    :type inference_setting: Dict[str, Union[float, bool]]
    :return: Original annotation, result annotation, result project meta.
    :rtype: Tuple[sly.Annotation, sly.Annotation, sly.ProjectMeta]
    """
    original_anns, result_anns, res_project_meta = apply_model_to_images(
        image_info.dataset_id, [image_info], inference_setting
    )
    if original_anns is None:
        original_ann = sly.Annotation.from_json(
            g.api.annotation.download(image_info.id).annotation, g.project_meta
        )
    else:
        original_ann = original_anns[0]
    return original_ann, result_anns[0], res_project_meta


def apply_model_to_images(
    dataset_id: int,
    image_infos: List[sly.ImageInfo],
    inference_settings: Dict[str, Union[float, bool]],
) -> Tuple[List[sly.Annotation], List[sly.Annotation], sly.ProjectMeta]:
    """Applies model to the images.

    :param dataset_id: Dataset ID.
    :type dataset_id: int
    :param image_infos: Images to apply model to.
    :type image_infos: List[sly.ImageInfo]
    :param inference_settings: Inference settings.
    :type inference_settings: Dict[str, Union[float, bool]]
    :return: Original annotations, result annotations, result project meta.
    :rtype: Tuple[List[sly.Annotation], List[sly.Annotation], sly.ProjectMeta]
    """
    # Reading parameters from widgets: inference_mode.
    inference_mode = settings.inference_mode.get_value()
    image_ids = [image_info.id for image_info in image_infos]

    if inference_mode == "sliding_window":
        inference_settings.update(get_sliding_window_params())
        sly.logger.info("Sliding window is enabled. Inference settings updated.")

    try:
        if inference_mode == "sliding_window":
            sly.logger.info("Running inference in sliding window mode...")
            if g.model_info.get("async_image_inference_support") is True:
                sly.logger.info("Running inference in async mode...")

                def get_inference_progress(inference_request_uuid):
                    sly.logger.debug("Requesting inference progress...")
                    result = g.api.task.send_request(
                        g.model_session_id,
                        "get_inference_progress",
                        data={"inference_request_uuid": inference_request_uuid},
                    )
                    return result

                current = 0
                ann_pred_json = []
                for image_id in image_ids:
                    pred_json = g.api.task.send_request(
                        g.model_session_id,
                        "inference_image_id_async",
                        data={
                            "image_id": image_id,
                            "settings": inference_settings,
                        },
                    )
                    g.inference_request_uuid = pred_json["inference_request_uuid"]

                    is_inferring = True
                    if current == 0:
                        sly.logger.info(
                            f"Inferring image id{image_id}: {current} / {len(image_ids)}"
                        )
                    while is_inferring:
                        progress = get_inference_progress(g.inference_request_uuid)
                        is_inferring = progress["is_inferring"]
                        sly.logger.debug(
                            f"Inferring image id{image_id}: {current} / {len(image_ids)}"
                        )
                        sleep(1)
                    current += 1
                    sly.logger.info(f"Inferring image id{image_id}: {current} / {len(image_ids)}")
                    result = progress["result"]
                    ann_pred_json.append(result)
        else:
            sly.logger.info("Running inference in full image mode...")
            ann_pred_json = g.api.task.send_request(
                g.model_session_id,
                "inference_batch_ids",
                data={
                    "dataset_id": dataset_id,
                    "batch_ids": image_ids,
                    "settings": inference_settings,
                },
            )
        if not isinstance(ann_pred_json, list):
            raise ValueError(
                f"Sequence with annotation predictions must be a 'list'. Predictions: '{ann_pred_json}'"
            )
        if len(ann_pred_json) != len(image_ids):
            raise RuntimeError(
                "Can not match number of images ids and number of predictions, len(img_ids) != len(ann_pred_json)"
            )
    except Exception as e:
        sly.logger.warn(
            f"Couldn't process predictions by batch. Attempting to process predictions one by one. Error: {e}"
        )
        sly.logger.info(
            "INFERENCE DEBUG INFO (BATCH)",
            extra={
                "nn_session_id": g.model_session_id,
                "dataset_id": dataset_id,
                "batch_ids": str(image_ids),
                "settings": str(inference_settings),
            },
        )

        sly.logger.info("Running inference in sync mode...")
        ann_pred_json = []
        for image_id in image_ids:
            try:
                pred_json = g.api.task.send_request(
                    g.model_session_id,
                    "inference_batch_ids",
                    data={
                        "dataset_id": dataset_id,
                        "batch_ids": [image_id],
                        "settings": inference_settings,
                    },
                )[0]
                validate_ann(pred_json)
                ann_pred_json.append(pred_json)
            except Exception as e:
                sly.logger.info(
                    "INFERENCE DEBUG INFO (PER IMG)",
                    extra={
                        "nn_session_id": g.model_session_id,
                        "dataset_id": dataset_id,
                        "image_id": image_id,
                        "settings": str(inference_settings),
                    },
                )
                image_info = g.api.image.get_info_by_id(id=image_id)
                sly.logger.warn(
                    f"Couldn't process annotation prediction for image: {image_info.name} (ID: {image_id}). Image remain unchanged. Error: {e}"
                )
                pred_json = sly.Annotation(img_size=(image_info.height, image_info.width)).to_json()
                ann_pred_json.append(pred_json)

    ann_preds = []
    for img_id, pred_json in zip(image_ids, ann_pred_json):
        try:
            if isinstance(pred_json, dict) and "annotation" in pred_json.keys():
                pred_json = pred_json["annotation"]
            ann_pred = sly.Annotation.from_json(pred_json, g.model_meta)
            ann_preds.append(ann_pred)
        except Exception as e:
            sly.logger.warn(
                "Can not process predictions from serving",
                extra={"image_id": img_id, "details": repr(e)},
            )
            sly.logger.debug("Response from serving app", extra={"serving_response": pred_json})
            img_info = g.api.image.get_info_by_id(img_id)
            ann_pred = sly.Annotation(img_size=(img_info.height, img_info.width))
            ann_preds.append(ann_pred)

    res_project_meta = g.project_meta.clone()
    res_anns = []
    for ann_pred in ann_preds:
        res_ann, res_project_meta = postprocess(
            ann_pred,
            res_project_meta,
        )
        res_anns.append(res_ann)

    add_mode = settings.add_predictions_mode.get_value()

    original_anns = None
    if add_mode == "merge":
        original_anns = g.api.annotation.download_batch(dataset_id, image_ids)
        original_anns = [
            sly.Annotation.from_json(ann_info.annotation, g.project_meta)
            for ann_info in original_anns
        ]

        merged_anns = []
        for ann, pred in zip(original_anns, res_anns):
            merged_anns.append(ann.merge(pred))
    else:
        merged_anns = res_anns
    return original_anns, merged_anns, res_project_meta


def get_sliding_window_params() -> Dict[str, Union[int, str, bool]]:
    """Reads sliding window parameters from the widgets and returns them as a dictionary.
    The dictionary contains the following fields:
        - inference_mode: str
        - sliding_window_params with the following fields:
            - windowHeight: int
            - windowWidth: int
            - overlapY: int
            - overlapX: int
            - borderStrategy: str
            - naive: bool

    :return: Sliding window parameters.
    :rtype: Dict[str, Union[int, str, bool]]
    """
    return {
        "inference_mode": "sliding_window",
        "sliding_window_params": {
            "windowHeight": window_height.get_value(),
            "windowWidth": window_width.get_value(),
            "overlapY": vertical_overlap.get_value(),
            "overlapX": horizontal_overlap.get_value(),
            "borderStrategy": "shift_window",
            "naive": False,
        },
    }


def validate_ann(ann_json: Dict[Any, Any]) -> None:
    """Validates annotation prediction in Supervisely JSON format.

    :param ann_json: Annotation prediction in Supervisely JSON format.
    :type ann_json: Dict[Any, Any]
    :raises ValueError: If some of the received annotation prediction values are invalid.
    :raises ValueError: If image 'height' is not 'int'.
    :raises ValueError: If image 'width' is not 'int'.
    """
    if (
        not isinstance(ann_json["description"], str)
        and not isinstance(ann_json["size"], dict)
        and not isinstance(ann_json["tags"], list)
        and not isinstance(ann_json["objects"], list)
        and not isinstance(ann_json["customBigData"], dict)
    ):
        raise ValueError(
            "Some of the received annotation prediction values are invalid:"
            f"description: {ann_json.get('description', None)}"
            f"size: {ann_json.get('size', None)}"
            f"tags: {ann_json.get('tags', None)}"
            f"objects: {ann_json.get('objects', None)}"
            f"customBigData: {ann_json.get('customBigData', None)}"
        )

    if not isinstance(ann_json["size"]["height"], int):
        raise ValueError(f"Image 'height' must be 'int', not {type(ann_json['size']['height'])}")

    if not isinstance(ann_json["size"]["width"], int):
        raise ValueError(f"Image 'width' must be 'int', not {type(ann_json['size']['width'])}")


def postprocess(
    ann: sly.Annotation,
    project_meta: sly.ProjectMeta,
) -> Tuple[sly.Annotation, sly.ProjectMeta]:
    """Postprocesses annotation after model inference and returns the result annotation and project meta.
    Removes classes and tags that are not selected in the UI.

    :param ann: Annotation to postprocess.
    :type ann: sly.Annotation
    :param project_meta: Project meta.
    :type project_meta: sly.ProjectMeta
    :return: Result annotation, result project meta.
    :rtype: Tuple[sly.Annotation, sly.ProjectMeta]
    """
    # Reading parameters from widgets: selected classes, selected tags.
    keep_classes = [obj_class.name for obj_class in nn_info.select_classes.get_selected_classes()]
    keep_tags = [tag_meta.name for tag_meta in nn_info.select_tags.get_selected_tags()]

    res_project_meta, class_mapping, tag_meta_mapping = merge_metas(project_meta)

    image_tags = []
    for tag in ann.img_tags:
        if tag.meta.name not in keep_tags:
            continue
        image_tags.append(tag.clone(meta=tag_meta_mapping[tag.meta.name]))

    new_labels = []
    for label in ann.labels:
        if label.obj_class.name not in keep_classes:
            continue
        label_tags = []
        for tag in label.tags:
            if tag.meta.name not in keep_tags:
                continue
            label_tags.append(tag.clone(meta=tag_meta_mapping[tag.meta.name]))
        new_label = label.clone(
            obj_class=class_mapping[label.obj_class.name.strip()],
            tags=sly.TagCollection(label_tags),
        )
        new_labels.append(new_label)

    res_ann = ann.clone(labels=new_labels, img_tags=sly.TagCollection(image_tags))
    return res_ann, res_project_meta


def merge_metas(
    project_meta: sly.ProjectMeta,
) -> Tuple[sly.ProjectMeta, Dict[str, sly.ObjClass], Dict[str, sly.TagMeta]]:
    result_meta = project_meta.clone()

    def _merge(
        result_meta: sly.ProjectMeta,
        data_type=Literal["class", "tag"],
    ):
        """Merges classes or tags from the model meta to the project meta.

        :param result_meta: Project meta to merge to.
        :type result_meta: sly.ProjectMeta
        :param data_type: Data type to merge (class or tag).
        :type data_type: Literal["class", "tag"]
        :return: Result project meta, mapping of the model classes/tags to the project classes/tags.
        :rtype: Tuple[sly.ProjectMeta, Dict[str, sly.ObjClass]]
        """
        if data_type == "class":
            project_collection = project_meta.obj_classes
            keep_names = [
                obj_class.name for obj_class in nn_info.select_classes.get_selected_classes()
            ]
            model_collection = g.model_meta.obj_classes
        else:
            project_collection = project_meta.tag_metas
            keep_names = [tag_meta.name for tag_meta in nn_info.select_tags.get_selected_tags()]
            model_collection = g.model_meta.tag_metas
        mapping = {}
        for name in keep_names:
            model_item = model_collection.get(name)
            res_item, res_name = find_item(
                project_collection,
                model_item,
            )
            if res_item is None:
                res_item = model_item.clone(name=res_name)
                result_meta = (
                    result_meta.add_obj_class(res_item)
                    if data_type == "class"
                    else result_meta.add_tag_meta(res_item)
                )
            mapping[model_item.name.strip()] = res_item
        return result_meta, mapping

    result_meta, class_mapping = _merge(
        result_meta,
        data_type="class",
    )
    result_meta, tag_mapping = _merge(
        result_meta,
        data_type="tag",
    )
    return result_meta, class_mapping, tag_mapping


def find_item(
    collection: Union[sly.ObjClassCollection, sly.TagMetaCollection],
    item: Union[sly.ObjClass, sly.TagMeta],
) -> Tuple[Union[sly.ObjClass, sly.TagMeta], str]:
    """Finds an item in the collection and returns it or generates a new name for the item.

    :param collection: Collection to search in.
    :type collection: Union[sly.ObjClassCollection, sly.TagMetaCollection]
    :param item: Item to find.
    :type item: Union[sly.ObjClass, sly.TagMeta]
    :return: Found item, new name for the item.
    :rtype: Tuple[Union[sly.ObjClass, sly.TagMeta], str]
    """
    suffix = settings.class_tag_suffix.get_value()
    use_suffix = settings.always_add_suffix.is_checked()
    index = 0
    res_name = item.name.strip()
    while True:
        existing_item = collection.get(res_name.strip())
        if existing_item is None:
            if use_suffix is True:
                res_name = generate_res_name(item, suffix, index)
                existing_item = collection.get(res_name)
                if existing_item is not None:
                    return existing_item, None
            return None, res_name
        else:
            if existing_item == item.clone(name=res_name):
                if use_suffix is True:
                    res_name = generate_res_name(item, suffix, index)
                    existing_item = collection.get(res_name)
                    if existing_item is None:
                        return None, res_name
                    elif existing_item == item.clone(name=res_name):
                        res_name = generate_res_name(item, suffix, index)
                        existing_item = collection.get(res_name)
                        if existing_item is None:
                            return None, res_name
                        return existing_item, None
                    else:
                        index += 1
                        res_name = generate_res_name(item, suffix, index)
                        existing_item = collection.get(res_name)
                        if existing_item is None:
                            return None, res_name
                return existing_item, None
            else:
                res_name = generate_res_name(item, suffix, index)
                index += 1


def generate_res_name(item: Union[sly.ObjClass, sly.TagMeta], suffix: str, index: int) -> str:
    """Generates a new name for the item.

    :param item: Item to generate a new name for.
    :type item: Union[sly.ObjClass, sly.TagMeta]
    :param suffix: Suffix to add to the name.
    :type suffix: str
    :param index: Index to add to the name.
    :type index: int
    :return: New name for the item.
    :rtype: str
    """
    return f"{item.name}-{suffix}" if index == 0 else f"{item.name}-{suffix}-{index}"


# endregion
