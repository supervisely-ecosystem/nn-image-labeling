

def set_model_info(api, task_id, model_meta, model_info, inf_settings):
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
    fields = [
        {"field": "data.connected", "payload": False},
        {"field": "data.connectionError", "payload": repr(e)},
    ]
    api.task.set_fields(task_id, fields)


def _get_keep_names(infos: List[dict], flags: List[bool]):
    keep_names = []
    for info, flag in zip(infos, flags):
        if flag is True:
            name = info.get("name", info["title"])
            keep_names.append(name)
    return keep_names