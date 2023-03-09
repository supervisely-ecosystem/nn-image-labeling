import os
import yaml
import pathlib
import sys
from collections import defaultdict
import supervisely as sly

root_source_path = str(pathlib.Path(sys.argv[0]).parents[2])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

from init_ui import init_ui
from shared_utils.connect import get_model_info
from shared_utils.inference import postprocess


owner_id = int(os.environ["context.userId"])
team_id = int(os.environ["context.teamId"])

my_app: sly.AppService = sly.AppService(ignore_task_id=True)
model_meta: sly.ProjectMeta = None
session_info: dict = None

ann_cache = defaultdict(list)  # only one (current) image in cache


@my_app.callback("connect")
@sly.timeit
def connect(api: sly.Api, task_id, context, state, app_logger):
    global model_meta, session_info
    model_meta, session_info = get_model_info(api, task_id, context, state, app_logger)


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
        object_roi: sly.Rectangle = label_roi.geometry.to_bbox()
        data["rectangle"] = object_roi.to_json()

    ann_pred_json = api.task.send_request(state["sessionId"], "inference_image_id", data=data)

    # # update model meta if needed
    # if session_info.get("task type") == "salient object segmentation" and figure_id is not None:
    #     # add new obj class to model meta if needed
    #     rectangle_data = data["rectangle"]
    #     objclass_info = api.object_class.get_info_by_id(id=rectangle_data["classId"])
    #     class_name = objclass_info.name + "_mask"
    #     global model_meta
    #     if not model_meta.get_obj_class(class_name):  # if obj class is not in model meta
    #         model_meta = model_meta.add_obj_class(sly.ObjClass(class_name, sly.Bitmap, [255, 0, 0]))

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
        
    if session_info.get("task type") == "salient object segmentation" and figure_id is not None:   
        target_class_name = label_roi.obj_class.name + "_mask"  
        target_class = project_meta.get_obj_class(target_class_name) 
        if target_class is None:
            target_class = sly.ObjClass(target_class_name, sly.Bitmap, [255, 0, 0])
            project_meta = project_meta.add_obj_class(target_class)
            api.project.update_meta(project_meta)
            # delete - api.project.pull_meta_ids(project_id, project_meta)
        
        final_labels = []
        for label in ann_pred.labels:
            final_labels.append(label.clone(obj_class=target_class)) # only one object
        final_ann_pred = ann_pred.clone(labels=final_labels)
                        
    res_ann, res_project_meta = postprocess(
        api, project_id, final_ann_pred, project_meta, model_meta, state
    )

    if state["addMode"] == "merge":
        res_ann = ann.merge(res_ann)
    else:
        pass  # replace (data prepared, nothing to do)

    if res_project_meta != project_meta:
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
