import supervisely as sly
import shared_utils.ui2 as ui
import yaml


def get_model_info(
    api: sly.Api, task_id, context, state, app_logger
) -> sly.ProjectMeta:
    model_meta = None
    try:
        info = api.task.send_request(state["sessionId"], "get_session_info", data={})
        log_settings(settings=info, msg="⚙️MODEL SETTINGS⚙️")
        info["session"] = state["sessionId"]
        app_logger.debug("Session Info", extra={"info": info})

        meta_json = api.task.send_request(
            state["sessionId"], "get_output_classes_and_tags", data={}
        )
        model_meta = sly.ProjectMeta.from_json(meta_json)

        try:
            inf_settings = api.task.send_request(
                state["sessionId"], "get_custom_inference_settings", data={}
            )

            if inf_settings["settings"] is None or len(inf_settings["settings"]) == 0:
                inf_settings["settings"] = ""
                sly.logger.info("Model doesn't support custom inference settings.")
            elif isinstance(inf_settings["settings"], dict):
                inf_settings["settings"] = yaml.dump(inf_settings["settings"], allow_unicode=True)
        except Exception as ex:
            inf_settings = {"settings": ""}
            sly.logger.info(
                "Model doesn't support custom inference settings.\n"
                f"Reason: {repr(ex)}"
            )

        log_settings(settings=inf_settings, msg="⚙️INFERENCE SETTINGS⚙️")
        ui.set_model_info(api, task_id, model_meta, info, inf_settings)
    except Exception as e:
        ui.set_error(api, task_id, e)

    return model_meta


def set_model_info(api, task_id, model_meta, model_info, inf_settings):
    disabledSW = True
    if "sliding_window_support" in model_info.keys() and model_info["sliding_window_support"] is not None:
        if isinstance(model_info["sliding_window_support"], bool):
            if model_info["sliding_window_support"]:
                disabledSW = False
        elif isinstance(model_info["sliding_window_support"], str):
            disabledSW = False

    fields = [
        {"field": "data.info", "payload": model_info},
        {"field": "state.classesInfo", "payload": model_meta.obj_classes.to_json()},
        {"field": "state.classes", "payload": [True] * len(model_meta.obj_classes)},
        {"field": "state.tagsInfo", "payload": model_meta.tag_metas.to_json()},
        {"field": "state.tags", "payload": [True] * len(model_meta.tag_metas)},
        {"field": "data.connected", "payload": True},
        {"field": "data.connectionError", "payload": ""},
        {"field": "state.settings", "payload": inf_settings["settings"]},
        {"field": "state.disabledSW", "payload": disabledSW},
    ]
    api.task.set_fields(task_id, fields)


def set_error(api: sly.Api, task_id, e: Exception):
    fields = [
        {"field": "data.connected", "payload": False},
        {"field": "data.connectionError", "payload": repr(e)},
    ]
    api.task.set_fields(task_id, fields)


def log_settings(settings, msg):
    sly.logger.info(msg=msg)
    for key, value in settings.items():
        sly.logger.info(f"{key}: {value}")
