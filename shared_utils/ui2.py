from typing import List
import supervisely_lib as sly


def set_model_info(api: sly.Api, task_id, model_meta: sly.ProjectMeta, model_info: dict, inf_settings: dict):
    fields = [
        {"field": "data.info", "payload": model_info},
        {"field": "state.classesInfo", "payload": model_meta.obj_classes.to_json()},
        {"field": "state.classes", "payload": [True] * len(model_meta.obj_classes)},
        {"field": "state.tagsInfo", "payload": model_meta.tag_metas.to_json()},
        {"field": "state.tags", "payload": [True] * len(model_meta.tag_metas)},
        {"field": "data.connected", "payload": True},
        {"field": "data.connectionError", "payload": ""},
        {"field": "state.settings", "payload": inf_settings["settings"]}
    ]
    api.task.set_fields(task_id, fields)


def set_error(api: sly.Api, task_id, e: Exception):
    sly.logger.error(repr(e))
    fields = [
        {"field": "data.connected", "payload": False},
        {"field": "data.connectionError", "payload": repr(e)},
    ]
    api.task.set_fields(task_id, fields)


def _get_keep_names(infos: List[dict], flags: List[bool]):
    keep_names = []
    for info, flag in zip(infos, flags):
        if flag is True:
            name = info.get("name", info.get("title", None))
            keep_names.append(name)
    return keep_names


def get_keep_classes(state):
    keep_names = _get_keep_names(state["classesInfo"], state["classes"])
    return keep_names


def get_keep_tags(state):
    keep_names = _get_keep_names(state["tagsInfo"], state["tags"])
    return keep_names