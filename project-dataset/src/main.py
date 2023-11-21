import yaml
import time
import random
import supervisely as sly


from shared_utils.connect import get_model_info, log_settings
from shared_utils.inference import postprocess
from shared_utils.ui2 import clean_error
import init_ui as ui

import sliding_window

import sly_globals as g


@g.my_app.callback("connect")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def connect(api: sly.Api, task_id, context, state, app_logger):
    clean_error(api, task_id)
    g.model_meta, g.model_info = get_model_info(api, task_id, context, state, app_logger)
    if g.model_meta is not None:
        actual_ui_state = api.task.get_field(task_id, "state")
        preview(api, task_id, context, actual_ui_state, app_logger)


@g.my_app.callback("disconnect")
@sly.timeit
def disconnect(api: sly.Api, task_id, context, state, app_logger):
    g.model_meta = None

    new_data = {}
    new_state = {}
    ui.init(new_data, new_state)
    fields = [
        {"field": "data", "payload": new_data, "append": True},
        {"field": "state", "payload": new_state, "append": True},
    ]
    api.task.set_fields(task_id, fields)


@g.my_app.callback("select_all_classes")
@sly.timeit
def select_all_classes(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.classes", [True] * len(g.model_meta.obj_classes))


@g.my_app.callback("deselect_all_classes")
@sly.timeit
def deselect_all_classes(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.classes", [False] * len(g.model_meta.obj_classes))


@g.my_app.callback("select_all_tags")
@sly.timeit
def select_all_tags(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.tags", [True] * len(g.model_meta.tag_metas))


@g.my_app.callback("deselect_all_tags")
@sly.timeit
def deselect_all_tags(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.tags", [False] * len(g.model_meta.tag_metas))


@g.my_app.callback("preview")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def preview(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.processing", True)
    try:
        inf_setting = yaml.safe_load(state["settings"])
    except Exception as e:
        inf_setting = {}
        app_logger.warn(
            f"Model Inference launched without additional settings. \n" f"Reason: {e}",
            exc_info=True,
        )

    if state["randomImagePreview"] is True:
        image_info = random.choice(g.input_images)
    else:
        image_info = [
            image_info
            for image_info in g.input_images
            if image_info.id == state["previewOnImageId"]
        ][0]

    input_ann, res_ann, res_project_meta = apply_model_to_image(
        api, state, image_info.dataset_id, image_info.id, inf_setting
    )

    preview_gallery = {
        "content": {
            "projectMeta": res_project_meta.to_json(),
            "annotations": {
                "input": {
                    "url": image_info.path_original,
                    "figures": [label.to_json() for label in input_ann.labels],
                    "info": {
                        "title": "input",
                    },
                },
                "output": {
                    "url": image_info.path_original,
                    "figures": [label.to_json() for label in res_ann.labels],
                    "info": {
                        "title": "output",
                    },
                },
            },
            "layout": [["input"], ["output"]],
        },
        "options": g.image_grid_options,
    }

    fields = [
        {"field": "state.processing", "payload": False},
        {"field": "data.gallery", "payload": preview_gallery},
    ]
    api.task.set_fields(task_id, fields)


def apply_model_to_image(api, state, dataset_id, image_id, inf_setting):
    orig_anns, res_anns, res_project_meta = apply_model_to_images(
        api, state, dataset_id, [image_id], inf_setting
    )
    if orig_anns is None:
        orig_ann = sly.Annotation.from_json(
            api.annotation.download(image_id).annotation, g.project_meta
        )
    else:
        orig_ann = orig_anns[0]
    return orig_ann, res_anns[0], res_project_meta


def validate_ann_pred_json(ann_pred_json: dict):
    if (
        not isinstance(ann_pred_json["description"], str)
        and not isinstance(ann_pred_json["size"], dict)
        and not isinstance(ann_pred_json["tags"], list)
        and not isinstance(ann_pred_json["objects"], list)
        and not isinstance(ann_pred_json["customBigData"], dict)
    ):
        raise ValueError(
            "Some of the received annotation prediction values are invalid:"
            f"description: {ann_pred_json.get('description', None)}"
            f"size: {ann_pred_json.get('size', None)}"
            f"tags: {ann_pred_json.get('tags', None)}"
            f"objects: {ann_pred_json.get('objects', None)}"
            f"customBigData: {ann_pred_json.get('customBigData', None)}"
        )

    if not isinstance(ann_pred_json["size"]["height"], int):
        raise ValueError(
            f"Image 'height' must be 'int', not {type(ann_pred_json['size']['height'])}"
        )

    if not isinstance(ann_pred_json["size"]["width"], int):
        raise ValueError(f"Image 'width' must be 'int', not {type(ann_pred_json['size']['width'])}")


def apply_model_to_images(api: sly.Api, state, dataset_id, ids, inf_setting):
    nn_session_id = state["sessionId"]
    add_mode = state["addMode"]

    if state["infMode"] == "sliding_window":
        inf_setting.update(sliding_window.get_sliding_window_params_from_state(state))

    try:
        sly.logger.info("Starting inference...")
        if state["infMode"] == "sliding_window":
            # Running async inference
            if g.model_info.get("async_image_inference_support") is True:

                def get_inference_progress(inference_request_uuid):
                    sly.logger.debug("Requesting inference progress...")
                    result = g.api.task.send_request(
                        state["sessionId"],
                        "get_inference_progress",
                        data={"inference_request_uuid": inference_request_uuid},
                    )
                    return result

                current = 0
                ann_pred_json = []
                for img_id in ids:
                    pred_json = g.api.task.send_request(
                        state["sessionId"],
                        "inference_image_id_async",
                        data={
                            "image_id": img_id,
                            "settings": inf_setting,
                        },
                    )
                    g.inference_request_uuid = pred_json["inference_request_uuid"]

                    is_inferring = True
                    if current == 0:
                        sly.logger.info(f"Inferring image id{img_id}: {current} / {len(ids)}")
                    while is_inferring:
                        progress = get_inference_progress(g.inference_request_uuid)
                        is_inferring = progress["is_inferring"]
                        sly.logger.debug(f"Inferring image id{img_id}: {current} / {len(ids)}")
                        time.sleep(1)
                    current += 1
                    sly.logger.info(f"Inferring image id{img_id}: {current} / {len(ids)}")
                    result = progress["result"]
                    ann_pred_json.append(result)
        else:
            # Running inference for full image
            ann_pred_json = api.task.send_request(
                nn_session_id,
                "inference_batch_ids",
                data={
                    "dataset_id": dataset_id,
                    "batch_ids": ids,
                    "settings": inf_setting,
                },
            )
        if type(ann_pred_json) != list:
            raise ValueError(
                f"Sequence with annotation predictions must be a 'list'. Predictions: '{ann_pred_json}'"
            )
        if len(ann_pred_json) != len(ids):
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
                "nn_session_id": nn_session_id,
                "dataset_id": dataset_id,
                "batch_ids": ids,
                "settings": str(inf_setting),
            },
        )

        # Fallback to sync inference version
        ann_pred_json = []
        for img_id in ids:
            try:
                pred_json = api.task.send_request(
                    nn_session_id,
                    "inference_batch_ids",
                    data={
                        "dataset_id": dataset_id,
                        "batch_ids": [img_id],
                        "settings": inf_setting,
                    },
                )[0]
                validate_ann_pred_json(pred_json)
                ann_pred_json.append(pred_json)
            except Exception as e:
                sly.logger.info(
                    "INFERENCE DEBUG INFO (PER IMG)",
                    extra={
                        "nn_session_id": nn_session_id,
                        "dataset_id": dataset_id,
                        "image_id": img_id,
                        "settings": str(inf_setting),
                    },
                )
                image_info = api.image.get_info_by_id(id=img_id)
                sly.logger.warn(
                    f"Couldn't process annotation prediction for image: {image_info.name} (ID: {img_id}). Image remain unchanged. Error: {e}"
                )
                pred_json = sly.Annotation(img_size=(image_info.height, image_info.width)).to_json()
                ann_pred_json.append(pred_json)

    ann_preds = []
    for img_id, pred_json in zip(ids, ann_pred_json):
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
            img_info = api.image.get_info_by_id(img_id)
            ann_pred = sly.Annotation(img_size=(img_info.height, img_info.width))
            ann_preds.append(ann_pred)

    res_project_meta = g.project_meta.clone()
    res_anns = []
    for ann_pred in ann_preds:
        res_ann, res_project_meta = postprocess(
            api, g.project_id, ann_pred, res_project_meta, g.model_meta, state
        )
        res_anns.append(res_ann)

    original_anns = None
    if add_mode == "merge":
        original_anns = api.annotation.download_batch(dataset_id, ids)
        original_anns = [
            sly.Annotation.from_json(ann_info.annotation, g.project_meta)
            for ann_info in original_anns
        ]

        merged_anns = []
        for ann, pred in zip(original_anns, res_anns):
            merged_anns.append(ann.merge(pred))
    else:
        merged_anns = res_anns
        pass  # replace (data prepared, nothing to do)
    return original_anns, merged_anns, res_project_meta


def get_images_for_preview_list(max_size=100):
    images_for_preview_list = []
    for index, image_info in enumerate(g.input_images):
        images_for_preview_list.append(
            {
                "label": image_info.name,
                "value": image_info.id,
            }
        )

        if len(images_for_preview_list) >= max_size:
            break

    return images_for_preview_list


@g.my_app.callback("apply_model")
@sly.timeit
def apply_model(api: sly.Api, task_id, context, state, app_logger):
    def _update_progress(progress):
        fields = [
            {
                "field": "data.progress",
                "payload": int(progress.current * 100 / progress.total),
            },
            {"field": "data.progressCurrent", "payload": progress.current},
            {"field": "data.progressTotal", "payload": progress.total},
        ]
        api.task.set_fields(task_id, fields)

    try:
        inf_setting = yaml.safe_load(state["settings"])
        log_settings(settings=inf_setting, msg="FINAL INFERENCE SETTINGS")
    except Exception as e:
        inf_setting = {}
        app_logger.warn(
            f"Model Inference launched without additional settings. \n" f"Reason: {e}",
            exc_info=True,
        )

    res_project_meta = g.project_meta.clone()
    res_project = api.project.create(
        g.workspace_id, state["resProjectName"], change_name_if_conflict=True
    )
    api.project.update_meta(res_project.id, res_project_meta.to_json())

    progress = sly.Progress("Inference", len(g.input_images), need_info_log=True)

    for dataset in g.input_datasets:
        res_dataset = api.dataset.create(res_project.id, dataset.name, dataset.description)
        images = api.image.get_list(dataset.id)

        for batch in sly.batched(images, batch_size=10):
            try:
                image_ids, res_names, res_metas = [], [], []
                for image_info in batch:
                    image_ids.append(image_info.id)
                    res_names.append(image_info.name)
                    res_metas.append(image_info.meta)
                _, res_anns, final_project_meta = apply_model_to_images(
                    api, state, dataset.id, image_ids, inf_setting
                )
            except Exception as e:
                sly.logger.warn(
                    msg=f"Couldn't process images by batch, images will be processed one by one, error: {e}."
                )

                image_ids, res_names, res_anns, res_metas = [], [], [], []
                for image_info in batch:
                    try:
                        _, res_ann, final_project_meta = apply_model_to_image(
                            api, state, dataset.id, image_info.id, inf_setting
                        )

                        image_ids.append(image_info.id)
                        res_names.append(image_info.name)
                        res_anns.append(res_ann)
                        res_metas.append(image_info.meta)
                    except Exception as e:
                        sly.logger.warn(
                            msg=f"Image: {image_info.name} (ID: {image_info.id}) couldn't be processed, image will be skipped, error: {e}.",
                            extra={
                                "image_name": image_info.name,
                                "image_id": image_info.id,
                                "image_meta": image_info.meta,
                            },
                        )
                        continue

            if res_project_meta != final_project_meta:
                res_project_meta = final_project_meta
                api.project.update_meta(res_project.id, res_project_meta.to_json())

            res_images_infos = api.image.upload_ids(
                res_dataset.id, res_names, image_ids, metas=res_metas
            )
            res_ids = [image_info.id for image_info in res_images_infos]
            try:
                api.annotation.upload_anns(res_ids, res_anns)
            except:
                for res_img_info, ann in zip(res_images_infos, res_anns):
                    try:
                        api.annotation.upload_ann(res_img_info.id, ann)
                    except Exception as e:
                        sly.logger.warn(
                            msg=f"Image: {res_img_info.name} (Image ID: {res_img_info.id}) couldn't be uploaded, image will be skipped, error: {e}.",
                            extra={
                                "image_name": res_img_info.name,
                                "image_id": res_img_info.id,
                                "image_meta": res_img_info.meta,
                                "image_ann": ann,
                            },
                        )
                        continue

            progress.iters_done_report(len(res_ids))
            if progress.need_report():
                _update_progress(progress)

    res_project = api.project.get_info_by_id(res_project.id)  # to refresh reference_image_url
    fields = [
        {"field": "data.resProjectId", "payload": res_project.id},
        {"field": "data.resProjectName", "payload": res_project.name},
        {
            "field": "data.resProjectPreviewUrl",
            "payload": api.image.preview_url(res_project.reference_image_url, 100, 100),
        },
    ]
    api.task.set_fields(task_id, fields)
    api.task.set_output_project(task_id, res_project.id, res_project.name)
    g.my_app.stop()


def main():
    data = {}
    state = {}
    data["ownerId"] = g.owner_id
    data["teamId"] = g.team_id

    dataset_info = None
    if g.project_id is None:
        dataset_info = g.my_app.public_api.dataset.get_info_by_id(g.dataset_id)
        g.input_datasets.append(dataset_info)
        g.project_id = dataset_info.project_id
    else:
        g.input_datasets = g.my_app.public_api.dataset.get_list(g.project_id)
    project_info = g.my_app.public_api.project.get_info_by_id(g.project_id)

    g.input_images = []
    for ds_info in g.input_datasets:
        g.input_images.extend(g.my_app.public_api.image.get_list(ds_info.id))

    data["imagesForPreview"] = get_images_for_preview_list()
    state["previewOnImageId"] = g.input_images[0].id if len(g.input_images) > 0 else None

    g.project_meta = sly.ProjectMeta.from_json(g.my_app.public_api.project.get_meta(g.project_id))

    ui.init(data, state)
    data["emptyGallery"] = g.empty_gallery
    ui.init_input_project(
        g.my_app.public_api, data, project_info, len(g.input_images), dataset_info
    )
    state["resProjectName"] = project_info.name + " (inf)"
    ui.init_output_project(data)

    sliding_window.init(data, state)

    g.my_app.run(data=data, state=state)


# @TODO: progress bar пропал после обновления страницы и снова появилась кнопка
if __name__ == "__main__":
    sly.main_wrapper("main", main)
