import os
import yaml
import pathlib
import sys
from collections import defaultdict
import supervisely as sly
from supervisely.imaging.color import random_rgb, generate_rgb

root_source_path = str(pathlib.Path(sys.argv[0]).parents[2])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

from init_ui import init_ui
from shared_utils.connect import get_model_info
from shared_utils.inference import postprocess
from dotenv import load_dotenv
import ruamel.yaml
import io

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("annotation-tool/debug.env")

owner_id = int(os.environ["context.userId"])
team_id = int(os.environ["context.teamId"])

my_app: sly.AppService = sly.AppService(ignore_task_id=True)
model_meta: sly.ProjectMeta = None
session_info: dict = None
# list for storing colors of bounding boxes (used in prompt-based object detection)
box_colors = []
# list for storing colors of masks (used in promptable segmentation)
mask_colors = [[255, 0, 0]]

ann_cache = defaultdict(list)  # only one (current) image in cache


@my_app.callback("connect")
@sly.timeit
def connect(api: sly.Api, task_id, context, state, app_logger):
    global model_meta, session_info
    model_meta, session_info = get_model_info(api, task_id, context, state, app_logger)
    if session_info.get("task type") == "promptable segmentation":
        # add positive and negative point classes to project meta
        project_id = context.get("projectId")
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
        for obj_class_name in ["positive", "negative"]:
            if not project_meta.get_obj_class(obj_class_name):
                if obj_class_name == "positive":
                    new_class = sly.ObjClass(obj_class_name, sly.Point, [51, 255, 51])
                elif obj_class_name == "negative":
                    new_class = sly.ObjClass(obj_class_name, sly.Point, [255, 0, 0])
                project_meta = project_meta.add_obj_class(new_class)
                api.project.update_meta(project_id, project_meta)
    elif session_info.get("task type") == "pose estimation" and not session_info.get(
        "detector_included"
    ):
        fields = [{"field": "state.allow_full_image_inference", "payload": False}]
        api.task.set_fields(task_id, fields)


@my_app.callback("disconnect")
@sly.timeit
def disconnect(api: sly.Api, task_id, context, state, app_logger):
    global model_meta
    model_meta = None

    new_data = {}
    new_state = {}
    init_ui(new_data, new_state)
    fields = [
        {"field": "data", "payload": new_data, "append": True},
        {"field": "state", "payload": new_state, "append": True},
    ]
    api.task.set_fields(task_id, fields)


@my_app.callback("select_all_classes")
@sly.timeit
def select_all_classes(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.classes", [True] * len(model_meta.obj_classes))


@my_app.callback("deselect_all_classes")
@sly.timeit
def deselect_all_classes(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.classes", [False] * len(model_meta.obj_classes))


@my_app.callback("select_all_tags")
@sly.timeit
def select_all_tags(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.tags", [True] * len(model_meta.tag_metas))


@my_app.callback("deselect_all_tags")
@sly.timeit
def deselect_all_tags(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.tags", [False] * len(model_meta.tag_metas))


@my_app.callback("inference")
@sly.timeit
def inference(api: sly.Api, task_id, context, state, app_logger):
    global model_meta
    project_id = context.get("projectId")
    image_id = context.get("imageId")
    figure_id = context.get("figureId")

    try:
        inference_setting = yaml.safe_load(state["settings"])
    except Exception as e:
        inference_setting = {}
        app_logger.warn(
            f"Model Inference launched without additional settings. \n" f"Reason: {e}",
            exc_info=True,
        )

    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

    if image_id not in ann_cache:
        # keep only current image for simplicity
        ann_cache.clear()

    ann_json = api.annotation.download(image_id).annotation
    ann = sly.Annotation.from_json(ann_json, project_meta)
    ann_cache[image_id].append(ann)

    data = {"image_id": image_id, "settings": inference_setting}

    label_roi = None
    if figure_id is not None:
        label_roi = ann.get_label_by_id(figure_id)

    if label_roi is not None:
        object_roi: sly.Rectangle = label_roi.geometry.to_bbox()
        data["rectangle"] = object_roi.to_json()
        if session_info.get("task type") == "prompt-based object detection":
            # load settings string
            settings_str = state["settings"]
            ryaml = ruamel.yaml.YAML()
            settings = ryaml.load(settings_str)
            # set necessary parameters
            settings["mode"] = "reference_image"
            settings["reference_image_id"] = image_id
            settings["reference_class_name"] = label_roi.obj_class.name
            settings["reference_bbox"] = [
                object_roi.top,
                object_roi.left,
                object_roi.bottom,
                object_roi.right,
            ]
            # transform dict back to string
            stream = io.BytesIO()
            ryaml.dump(settings, stream)
            # decode string
            settings = stream.getvalue()
            settings = settings.decode("utf-8")
            app_logger.info("Switching model to reference image mode")
            app_logger.info(f"Image with id {image_id} was selected as reference image")
            # update necessary fields
            fields = [
                {"field": "state.settings", "payload": settings},
                {"field": "state.processing", "payload": False},
            ]
            api.task.set_fields(task_id, fields)
            return
        elif session_info.get("task type") == "promptable segmentation":
            # load settings string
            settings_str = state["settings"]
            ryaml = ruamel.yaml.YAML()
            settings = ryaml.load(settings_str)
            # get annotation geometry types
            current_ann = api.annotation.download(image_id).annotation
            current_ann = sly.Annotation.from_json(current_ann, project_meta)
            # geometries = [object["geometryType"] for object in current_ann]
            geometries = [label.geometry.geometry_name() for label in current_ann.labels]
            # set necessary parameters
            points_outside_box = False
            if "point" in geometries:
                # check if points are located inside object roi
                points = []
                for label in current_ann.labels:
                    if label.geometry.geometry_name() == "point":
                        if object_roi.contains(label.geometry.to_bbox()):
                            points.append(label)
                if len(points) > 0:
                    settings["mode"] = "combined"
                    app_logger.info("Switching model to combined mode")
                    settings["input_image_id"] = image_id
                    settings["point_coordinates"] = [
                        [point.geometry.col, point.geometry.row] for point in points
                    ]
                    point_labels = []
                    for point in points:
                        if point.obj_class.name == "positive":
                            point_labels.append(1)
                        elif point.obj_class.name == "negative":
                            point_labels.append(0)
                    settings["point_labels"] = point_labels
                    settings["bbox_coordinates"] = [
                        object_roi.top,
                        object_roi.left,
                        object_roi.bottom,
                        object_roi.right,
                    ]
                    settings["bbox_class_name"] = label_roi.obj_class.name
                else:
                    points_outside_box = True
            if "point" not in geometries or points_outside_box:
                settings["mode"] = "bbox"
                app_logger.info("Switching model to bbox mode")
                settings["bbox_coordinates"] = [
                    object_roi.top,
                    object_roi.left,
                    object_roi.bottom,
                    object_roi.right,
                ]
                settings["bbox_class_name"] = label_roi.obj_class.name
                settings["input_image_id"] = image_id
            # transform dict back to string
            stream = io.BytesIO()
            ryaml.dump(settings, stream)
            # decode string
            settings = stream.getvalue()
            settings = settings.decode("utf-8")
            # update necessary fields
            fields = [
                {"field": "state.settings", "payload": settings},
                {"field": "state.processing", "payload": False},
            ]
            api.task.set_fields(task_id, fields)
            data["settings"] = yaml.safe_load(settings)
    else:
        if session_info.get("task type") == "promptable segmentation":
            # get annotation geometry types
            current_ann = api.annotation.download(image_id).annotation
            current_ann = sly.Annotation.from_json(current_ann, project_meta)
            geometries = [label.geometry.geometry_name() for label in current_ann.labels]
            if "point" in geometries:
                # load settings string
                settings_str = state["settings"]
                ryaml = ruamel.yaml.YAML()
                settings = ryaml.load(settings_str)
                # set necessary parameters
                settings["mode"] = "points"
                settings["input_image_id"] = image_id
                points = [
                    label
                    for label in current_ann.labels
                    if label.geometry.geometry_name() == "point"
                ]
                settings["point_coordinates"] = [
                    [point.geometry.col, point.geometry.row] for point in points
                ]
                point_labels = []
                for point in points:
                    if point.obj_class.name == "positive":
                        point_labels.append(1)
                    elif point.obj_class.name == "negative":
                        point_labels.append(0)
                settings["point_labels"] = point_labels
                # transform dict back to string
                stream = io.BytesIO()
                ryaml.dump(settings, stream)
                # decode string
                settings = stream.getvalue()
                settings = settings.decode("utf-8")
                app_logger.info("Switching model to points mode")
                # update necessary fields
                fields = [
                    {"field": "state.settings", "payload": settings},
                    {"field": "state.processing", "payload": False},
                ]
                api.task.set_fields(task_id, fields)
                data["settings"] = yaml.safe_load(settings)
            else:
                # load settings string
                settings_str = state["settings"]
                ryaml = ruamel.yaml.YAML()
                settings = ryaml.load(settings_str)
                # set necessary parameters
                settings["mode"] = "raw"
                # transform dict back to string
                stream = io.BytesIO()
                ryaml.dump(settings, stream)
                # decode string
                settings = stream.getvalue()
                settings = settings.decode("utf-8")
                app_logger.info("Switching model to raw mode")
                # update necessary fields
                fields = [
                    {"field": "state.settings", "payload": settings},
                    {"field": "state.processing", "payload": False},
                ]
                api.task.set_fields(task_id, fields)
                data["settings"] = yaml.safe_load(settings)

    if session_info.get("task type") == "promptable segmentation":
        if data["settings"]["replace_masks"]:
            # delete old masks if necessary to avoid overlapping
            for label in ann.labels:
                if label.geometry.geometry_name() == "bitmap":
                    if data["settings"]["mode"] in ("raw", "points"):
                        ann = ann.delete_label(label)
                    elif data["settings"]["mode"] in ("bbox", "combined"):
                        bbox = sly.Rectangle(*data["settings"]["bbox_coordinates"])
                        mask_bbox = label.geometry.to_bbox()
                        if bbox.contains(mask_bbox):
                            ann = ann.delete_label(label)

    ann_pred_json = api.task.send_request(state["sessionId"], "inference_image_id", data=data)
    if session_info.get("task type") == "prompt-based object detection":
        # add tag to model meta if necessary
        global model_meta
        if not model_meta.get_tag_meta("confidence"):
            model_meta = model_meta.add_tag_meta(sly.TagMeta("confidence", value_type="any_number"))
        # add obj class to model meta if necessary
        for object in ann_pred_json["annotation"]["objects"]:
            class_name = object["classTitle"]
            obj_class = model_meta.get_obj_class(class_name)
            if obj_class is None:
                global box_colors
                if len(box_colors) > 0:
                    color = generate_rgb(box_colors)
                else:
                    color = random_rgb()
                box_colors.append(color)
                obj_class = sly.ObjClass(class_name, sly.Rectangle, color)
                model_meta = model_meta.add_obj_class(obj_class)
    elif session_info.get("task type") == "promptable segmentation":
        # add obj class to model meta if necessary
        for object in ann_pred_json["annotation"]["objects"]:
            class_name = object["classTitle"]
            obj_class = model_meta.get_obj_class(class_name)
            if obj_class is None:
                global mask_colors
                if data["settings"]["mode"] == "raw":
                    color = generate_rgb(mask_colors)
                    mask_colors.append(color)
                else:
                    color = mask_colors[0]
                obj_class = sly.ObjClass(class_name, sly.Bitmap, color)
                model_meta = model_meta.add_obj_class(obj_class)

    if isinstance(ann_pred_json, dict) and "annotation" in ann_pred_json.keys():
        ann_pred_json = ann_pred_json["annotation"]
    try:
        ann_pred = sly.Annotation.from_json(ann_pred_json, model_meta)
    except Exception as e:
        sly.logger.warn(
            "Can not process predictions from serving",
            extra={"image_id": image_id, "details": repr(e)},
        )
        sly.logger.debug("Response from serving app", extra={"serving_response": ann_pred_json})
        ann_pred = sly.Annotation(img_size=ann.img_size)

    if session_info.get("task type") == "salient object segmentation" and label_roi is not None:
        target_class_name = label_roi.obj_class.name + "_mask"
        target_class = project_meta.get_obj_class(target_class_name)
        if target_class is None:
            target_class = sly.ObjClass(target_class_name, sly.Bitmap, [255, 0, 0])
            project_meta = project_meta.add_obj_class(target_class)
            api.project.update_meta(project_id, project_meta)
        final_labels = []
        for label in ann_pred.labels:
            final_labels.append(label.clone(obj_class=target_class))  # only one object
        ann_pred = ann_pred.clone(labels=final_labels)

    if session_info.get("task type") in (
        "prompt-based object detection",
        "promptable segmentation",
    ):
        # add tag to project meta if necessary
        if not project_meta.get_tag_meta("confidence"):
            project_meta = project_meta.add_tag_meta(
                sly.TagMeta("confidence", value_type="any_number")
            )
            api.project.update_meta(project_id, project_meta)
        # add obj class to project meta if necessary
        for label in ann_pred.labels:
            if not project_meta.get_obj_class(label.obj_class.name):
                project_meta = project_meta.add_obj_class(label.obj_class)
                api.project.update_meta(project_id, project_meta)

    if not (
        session_info.get("task type") == "salient object segmentation" and label_roi is not None
    ) and session_info.get("task type") not in (
        "prompt-based object detection",
        "promptable segmentation",
    ):
        res_ann, res_project_meta = postprocess(
            api, project_id, ann_pred, project_meta, model_meta, state
        )
    else:
        res_ann = ann_pred

    if state["addMode"] == "merge":
        res_ann = ann.merge(res_ann)
    else:
        pass  # replace (data prepared, nothing to do)

    if "task type" in session_info.keys() and (
        not (
            session_info.get("task type") == "salient object segmentation" and label_roi is not None
        )
        and session_info.get("task type")
        not in ("prompt-based object detection", "promptable segmentation")
        and res_project_meta != project_meta
    ):
        api.project.update_meta(project_id, res_project_meta.to_json())

    api.annotation.upload_ann(image_id, res_ann)
    fields = [
        {"field": "data.rollbackIds", "payload": list(ann_cache.keys())},
        {"field": "state.processing", "payload": False},
    ]
    api.task.set_fields(task_id, fields)


@my_app.callback("undo")
@sly.timeit
def undo(api: sly.Api, task_id, context, state, app_logger):
    image_id = context["imageId"]
    if image_id in ann_cache:
        ann = ann_cache[image_id].pop()
        if len(ann_cache[image_id]) == 0:
            del ann_cache[image_id]
        api.annotation.upload_ann(image_id, ann)

    fields = [
        {"field": "data.rollbackIds", "payload": list(ann_cache.keys())},
        {"field": "state.processing", "payload": False},
    ]
    api.task.set_fields(task_id, fields)


def main():
    data = {}
    state = {}
    data["ownerId"] = owner_id
    data["teamId"] = team_id
    init_ui(data, state)

    my_app.run(data=data, state=state)


if __name__ == "__main__":
    sly.main_wrapper("main", main)
