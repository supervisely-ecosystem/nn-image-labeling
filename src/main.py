import os
from typing import List
import supervisely_lib as sly

import cache

owner_id = int(os.environ['context.userId'])
team_id = int(os.environ['context.teamId'])

my_app: sly.AppService = sly.AppService(ignore_task_id=True)
model_meta: sly.ProjectMeta = None


@my_app.callback("get_model_info")
@sly.timeit
def get_model_info(api: sly.Api, task_id, context, state, app_logger):
    global model_meta

    try:
        info = api.task.send_request(state["sessionId"], "get_session_info", data={})
        info["session"] = state["sessionId"]
        app_logger.debug("Session Info", extra={"info": info})

        meta_json = api.task.send_request(state["sessionId"], "get_output_classes_and_tags", data={})
        model_meta = sly.ProjectMeta.from_json(meta_json)

        inf_settings = api.task.send_request(state["sessionId"], "get_custom_inference_settings", data={})

        fields = [
            {"field": "data.info", "payload": info},
            {"field": "data.classes", "payload": model_meta.obj_classes.to_json()},
            {"field": "state.classes", "payload": [True] * len(model_meta.obj_classes)},
            {"field": "data.tags", "payload": model_meta.tag_metas.to_json()},
            {"field": "state.tags", "payload": [True] * len(model_meta.tag_metas)},
            {"field": "data.connected", "payload": True},
            {"field": "data.connectionError", "payload": ""},
            {"field": "state.settings", "payload": inf_settings["settings"]}
        ]
        api.task.set_fields(task_id, fields)
    except Exception as e:
        fields = [
            {"field": "data.connected", "payload": False},
            {"field": "data.connectionError", "payload": repr(e)},
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


def _postprocess(api: sly.Api, project_id, ann: sly.Annotation, project_meta: sly.ProjectMeta, suffix):
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
        res_meta = _compare_tag(res_meta, tag, image_tags)

    new_labels = []
    for label in ann.labels:
        label_tags = []
        for tag in label.tags:
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

    res_ann = ann.clone(labels=new_labels, img_tags=new_tags)
    return res_ann


@my_app.callback("inference")
@sly.timeit
def inference(api: sly.Api, task_id, context, state, app_logger):
    global metas_lock, backup_ann_lock
    project_id = context["projectId"]
    image_id = context["imageId"]

    project_meta = cache.get_project_meta(api, project_id)
    cache.backup_ann(api, image_id, project_meta)
    ann_json = api.task.send_request(state["sessionId"],
                                     "inference_image_id",
                                     data={
                                         "image_id": image_id,
                                         "debug_visualization": True if app_logger.level <= 10 else False  # 10 - debug
                                     })
    ann = sly.Annotation.from_json(ann_json, model_meta)
    new_ann = _postprocess(api, project_id, ann, project_meta, state["suffix"])
    api.annotation.upload_ann(image_id, new_ann)


def main():
    data = {}
    data["ownerId"] = owner_id
    data["info"] = {}
    data["classes"] = []
    data["tags"] = []
    data["connected"] = False
    data["connectionError"] = ""

    state = {}
    state["sessionId"] = "2361" #@TODO: for debug
    state["classes"] = []
    state["tags"] = []
    state["tabName"] = "info"
    state["suffix"] = "model"
    state["settings"] = "# empty"
    my_app.run(data=data, state=state)


#@TODO: filter predicted classes and tags (image + objects)
#@TODO: merge annotations / replace annotations / undo prediction
if __name__ == "__main__":
    sly.main_wrapper("main", main)
