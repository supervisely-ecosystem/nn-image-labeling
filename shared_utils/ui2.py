from typing import Any, Dict, List, Optional, Union, Literal
import supervisely as sly


def set_model_info(
    api: sly.Api, task_id, model_meta: sly.ProjectMeta, model_info: dict, inf_settings: dict
):
    disabledSW = True
    if (
        "sliding_window_support" in model_info.keys()
        and model_info["sliding_window_support"] is not None
    ):
        if isinstance(model_info["sliding_window_support"], bool):
            if model_info["sliding_window_support"]:
                disabledSW = False
        elif isinstance(model_info["sliding_window_support"], str):
            if model_info["sliding_window_support"] != "none":
                disabledSW = False
    fields = [
        {"field": "data.info", "payload": format_info(model_info, "video")},
        {"field": "state.classesInfo", "payload": model_meta.obj_classes.to_json()},
        {"field": "state.classes", "payload": [True] * len(model_meta.obj_classes)},
        {"field": "state.tagsInfo", "payload": model_meta.tag_metas.to_json()},
        {"field": "state.tags", "payload": [True] * len(model_meta.tag_metas)},
        {"field": "data.connected", "payload": True},
        {"field": "data.connectionError", "payload": ""},
        {"field": "data.inferenceError", "payload": ""},
        {"field": "data.servingLink", "payload": ""},
        {"field": "state.settings", "payload": inf_settings["settings"]},
        {"field": "state.disabledSW", "payload": disabledSW},
    ]
    api.task.set_fields(task_id, fields)


def set_error(
    api: sly.Api,
    task_id,
    e: Union[Exception, str],
    error_type: Literal["connection", "inference"] = "connection",
    session_id: str = None,
    log_error: bool = True,
):
    err = e if isinstance(e, str) else repr(e)
    if log_error:
        sly.logger.error(err)
    else:
        sly.logger.warn(err)
    fields = [
        {"field": f"data.{error_type}Error", "payload": err},
    ]
    if error_type == "connection":
        fields.append({"field": "data.connected", "payload": False})
    elif error_type == "inference" and session_id is not None:
        serving_app_link = f"{api.server_address}/apps/sessions/{session_id}"
        fields.append({"field": "data.servingLink", "payload": serving_app_link})

    api.task.set_fields(task_id, fields)


def clean_error(api: sly.Api, task_id):
    fields = [
        {"field": "data.connectionError", "payload": ""},
        {"field": "data.inferenceError", "payload": ""},
        {"field": "data.servingLink", "payload": ""},
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


def format_info(info: Dict[str, Any], trigger_to_del: Optional[str] = None) -> Dict[str, Any]:
    formated_info = {}

    for name, data in info.items():
        if trigger_to_del is not None:
            if trigger_to_del.lower() in name.lower():
                sly.logger.debug(
                    f"Field {name} excluded from session info: found `{trigger_to_del}` in name"
                )
                continue

        new_name = name.replace("_", " ").capitalize()
        formated_info[new_name] = data

    return formated_info
