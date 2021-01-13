import os
import supervisely_lib as sly

owner_id = int(os.environ['context.userId'])
team_id = int(os.environ['context.teamId'])

my_app: sly.AppService = sly.AppService(ignore_task_id=True)
model_meta: sly.ProjectMeta = None

# project_meta = {}

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

        fields = [
            {"field": "data.info", "payload": info},
            {"field": "data.classes", "payload": model_meta.obj_classes.to_json()},
            {"field": "state.classes", "payload": [True] * len(model_meta.obj_classes)},
            {"field": "data.tags", "payload": model_meta.tag_metas.to_json()},
            {"field": "state.tags", "payload": [True] * len(model_meta.tag_metas)},
            {"field": "data.connected", "payload": True},
            {"field": "data.connectionError", "payload": ""},
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


# def _update_meta(prediction: sly.Annotation):
#     for tag in prediction.img_tags:
#         tag: sly.Tag
#         original_tag_meta = project_meta.tag_metas.get(tag.meta.name)
#         if original_tag_meta is None:
#             project_meta

@my_app.callback("inference")
@sly.timeit
def inference(api: sly.Api, task_id, context, state, app_logger):
    image_id = context["imageId"]
    ann_json = api.task.send_request(state["sessionId"],
                                     "inference_image_id",
                                     data={
                                         "image_id": image_id,
                                         "debug_visualization": True
                                     })
    ann = sly.Annotation.from_json(ann_json, model_meta)

    project_meta_json = api.project.get_meta(context["projectId"])
    project_meta = sly.ProjectMeta.from_json(project_meta_json)



def main():
    data = {}
    data["ownerId"] = owner_id
    data["info"] = {}
    data["classes"] = []
    data["tags"] = []
    data["connected"] = False
    data["connectionError"] = ""

    state = {}
    state["sessionId"] = "2163" #@TODO: for debug
    state["classes"] = []
    state["tags"] = []
    state["tabName"] = "info"
    my_app.run(data=data, state=state)


#@TODO: check UI only for one user
#@TODO: merge annotations / replace annotations / undo prediction
#@TODO: merge metas
if __name__ == "__main__":
    sly.main_wrapper("main", main)
