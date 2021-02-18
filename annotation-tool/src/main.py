import os
from typing import List
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
    keep_classes = ['dog'] # ui.get_keep_classes(state) #@TODO: fr debug
    keep_tags = ui.get_keep_tags(state)
    suffix = state["suffix"]

    res_project_meta, class_mapping, tag_meta_mapping = merge_metas(project_meta, model_meta,
                                                                    keep_classes, keep_tags,
                                                                    suffix)

    x = 10
    x += 1
    return


    def _find_free_name(collection, name):
        free_name = name
        item = collection.get(free_name)
        if item is not None:
            free_name = f"{name}-{suffix}"
            item = collection.get(free_name)
        iter = 1
        while item is not None:
            free_name = f"{name}-{suffix}-{iter}"
            item = collection.get(free_name)
            iter += 1
        return free_name

    res_meta = project_meta.clone()
    tag_mapping = {}  # old name to new meta
    class_mapping = {}  # old name to new meta

    def _compare_tag(res_meta: sly.ProjectMeta, tag: sly.Tag, new_tags: List):
        if tag.meta.name in tag_mapping:
            new_tags.append(tag.clone(meta=tag_mapping[tag.meta.name]))
            return
        original_tag_meta = res_meta.tag_metas.get(tag.meta.name)
        if original_tag_meta is None:
            res_meta = res_meta.add_tag_meta(tag.meta)
            new_tags.append(tag)
        elif original_tag_meta != tag.meta:  # conflict
            new_tag_name = _find_free_name(res_meta.tag_metas, tag.meta.name)
            new_tag_meta = tag.meta.clone(name=new_tag_name)
            tag_mapping[new_tag_name] = new_tag_meta
            res_meta = res_meta.add_tag_meta(new_tag_meta)
            new_tags.append(tag.clone(meta=new_tag_meta))
        else:
            new_tags.append(tag)
        return res_meta

    image_tags = []
    for tag in ann.img_tags:
        if tag.meta.name not in keep_tags:
            continue
        res_meta = _compare_tag(res_meta, tag, image_tags)

    new_labels = []
    for label in ann.labels:
        if label.obj_class.name not in keep_classes:
            continue

        label_tags = []
        for tag in label.tags:
            if tag.meta.name not in keep_tags:
                continue
            res_meta = _compare_tag(res_meta, tag, label_tags)

        if label.obj_class.name in class_mapping:
            new_labels.append(label.clone(obj_class=class_mapping[label.obj_class.name],
                                          tags=sly.TagCollection(label_tags)))
            continue

        original_class = res_meta.obj_classes.get(label.obj_class.name)
        if original_class is None:
            res_meta = res_meta.add_obj_class(label.obj_class)
            new_labels.append(label.clone(tags=sly.TagCollection(label_tags)))
        elif original_class != label.obj_class:  # conflict
            new_class_name = _find_free_name(res_meta.obj_classes, label.obj_class.name)
            new_class = label.obj_class.clone(name=new_class_name)
            res_meta = res_meta.add_obj_class(new_class)
            new_labels.append(label.clone(obj_class=new_class, tags=sly.TagCollection(label_tags)))
        else:
            new_labels.append(label.clone(tags=sly.TagCollection(label_tags)))

    if len(res_meta.obj_classes) != len(project_meta.obj_classes) or \
       len(res_meta.tag_metas) != len(project_meta.tag_metas):
        cache.update_project_meta(api, project_id, res_meta)

    res_ann = ann.clone(labels=new_labels, img_tags=sly.TagCollection(image_tags))
    return res_ann


@my_app.callback("inference")
@sly.timeit
def inference(api: sly.Api, task_id, context, state, app_logger):
    #state["sessionId"] = 2392  # @TODO: FOR DEBUG

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
        # replace
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

    state["sessionId"] = 2611 #@TODO: for debug
    my_app.run(data=data, state=state)


#@TODO: bug in merge meta
if __name__ == "__main__":
    sly.main_wrapper("main", main)
