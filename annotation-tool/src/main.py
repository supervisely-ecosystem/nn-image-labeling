import os
import yaml
import supervisely_lib as sly

import cache
from init_ui import unit_ui
import shared_utils.ui2 as ui
from shared_utils.connect import get_model_info
from shared_utils.merge_metas import merge_metas

owner_id = int(os.environ['context.userId'])
team_id = int(os.environ['context.teamId'])

my_app: sly.AppService = sly.AppService(ignore_task_id=True)
model_meta: sly.ProjectMeta = None


@my_app.callback("connect")
@sly.timeit
def connect(api: sly.Api, task_id, context, state, app_logger):
    global model_meta
    model_meta = get_model_info(api, task_id, context, state, app_logger)


@my_app.callback("disconnect")
@sly.timeit
def disconnect(api: sly.Api, task_id, context, state, app_logger):
    global model_meta
    model_meta = None

    new_data = {}
    new_state = {}
    unit_ui(new_data, new_state)
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


def _postprocess(api: sly.Api, project_id, ann: sly.Annotation, project_meta: sly.ProjectMeta, state):
    keep_classes = ui.get_keep_classes(state) #@TODO: for debug ['dog'] #
    keep_tags = ui.get_keep_tags(state)
    suffix = state["suffix"]

    res_project_meta, class_mapping, tag_meta_mapping = merge_metas(project_meta, model_meta,
                                                                    keep_classes, keep_tags,
                                                                    suffix)

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

        new_label = label.clone(obj_class=class_mapping[label.obj_class.name], tags=sly.TagCollection(label_tags))
        new_labels.append(new_label)

    if len(res_project_meta.obj_classes) != len(project_meta.obj_classes) or \
       len(res_project_meta.tag_metas) != len(project_meta.tag_metas):
        cache.update_project_meta(api, project_id, res_project_meta)

    res_ann = ann.clone(labels=new_labels, img_tags=sly.TagCollection(image_tags))
    return res_ann


@my_app.callback("inference")
@sly.timeit
def inference(api: sly.Api, task_id, context, state, app_logger):
    global metas_lock, backup_ann_lock
    project_id = context["projectId"]
    image_id = context["imageId"]

    project_meta = cache.get_project_meta(api, project_id)
    cache.backup_ann(api, image_id, project_meta)

    last_annotation_json = api.annotation.download(image_id).annotation
    last_annotation = sly.Annotation.from_json(last_annotation_json, project_meta)

    inference_setting = {}
    try:
        inference_setting = yaml.safe_load(state["settings"])
    except Exception as e:
        app_logger.warn(repr(e))

    ann_json = api.task.send_request(state["sessionId"],
                                     "inference_image_id",
                                     data={
                                         "image_id": image_id,
                                         "settings": inference_setting
                                     })
    ann_pred = sly.Annotation.from_json(ann_json, model_meta)
    new_ann: sly.Annotation = _postprocess(api, project_id, ann_pred, project_meta, state)
    if state["addMode"] == "merge":
        new_ann = last_annotation.add_labels(new_ann.labels)
        new_ann = new_ann.add_tags(new_ann.img_tags)
    else:
        # replace (data prepared, nothing to do)
        pass

    api.annotation.upload_ann(image_id, new_ann)
    fields = [
        {"field": "data.rollbackIds", "payload": list(cache.anns.keys())},
    ]
    api.task.set_fields(task_id, fields)


@my_app.callback("rollback")
@sly.timeit
def rollback(api: sly.Api, task_id, context, state, app_logger):
    try:
        image_id = context["imageId"]
        ann = cache.restore_ann(image_id)
        api.annotation.upload_ann(image_id, ann)
        cache.remove_ann(image_id)
        fields = [
            {"field": "data.rollbackIds", "payload": list(cache.anns.keys())},
        ]
        api.task.set_fields(task_id, fields)
    except Exception as e:
        app_logger.warn(repr(e))
    return


def main():
    data = {}
    state = {}
    data["ownerId"] = owner_id
    data["teamId"] = team_id
    unit_ui(data, state)

    #state["sessionId"] = 2614 #@TODO: for debug
    my_app.run(data=data, state=state)


if __name__ == "__main__":
    sly.main_wrapper("main", main)
