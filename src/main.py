import os
import supervisely_lib as sly

owner_id = int(os.environ['context.userId'])
team_id = int(os.environ['context.teamId'])

my_app: sly.AppService = sly.AppService(ignore_task_id=True)
meta: sly.ProjectMeta = None


@my_app.callback("manual_selected_image_changed")
def event_next_image(api: sly.Api, task_id, context, state, app_logger):
    pass


@my_app.callback("get_model_info")
@sly.timeit
def get_model_info(api: sly.Api, task_id, context, state, app_logger):
    global meta
    info = api.task.send_request(state["sessionId"], "get_session_info", data={})
    info["session"] = state["sessionId"]
    app_logger.debug("Session Info", extra={"info": info})

    meta_json = api.task.send_request(state["sessionId"], "get_output_classes_and_tags", data={})
    meta = sly.ProjectMeta.from_json(meta_json)

    fields = [
        {"field": "data.info", "payload": info},
        {"field": "data.classes", "payload": meta.obj_classes.to_json()},
        {"field": "state.classes", "payload": [True] * len(meta.obj_classes)},
        {"field": "data.tags", "payload": meta.tag_metas.to_json()},
        {"field": "state.tags", "payload": [True] * len(meta.tag_metas)},
    ]
    api.task.set_fields(task_id, fields)


def main():
    data = {}
    data["ownerId"] = owner_id
    data["info"] = {}
    data["classes"] = []
    data["tags"] = []

    state = {}
    state["sessionId"] = "2163" #@TODO: for debug
    state["classes"] = []
    state["tags"] = []
    my_app.run(data=data, state=state)


#@TODO: check UI only for one user
if __name__ == "__main__":
    sly.main_wrapper("main", main)
