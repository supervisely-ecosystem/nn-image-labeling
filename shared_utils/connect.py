import supervisely_lib as sly
import shared_utils.ui2 as ui


def get_model_info(api: sly.Api, task_id, context, state, app_logger) -> sly.ProjectMeta:
    model_meta = None
    try:
        info = api.task.send_request(state["sessionId"], "get_session_info", data={})
        info["session"] = state["sessionId"]
        app_logger.debug("Session Info", extra={"info": info})

        meta_json = api.task.send_request(state["sessionId"], "get_output_classes_and_tags", data={})
        model_meta = sly.ProjectMeta.from_json(meta_json)

        inf_settings = api.task.send_request(state["sessionId"], "get_custom_inference_settings", data={})
        inf_settings['settings'] = "" if inf_settings['settings'] == {} or inf_settings['settings'] is None else inf_settings['settings']

        ui.set_model_info(api, task_id, model_meta, info, inf_settings)
    except Exception as e:
        ui.set_error(api, task_id, e)

    return model_meta


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