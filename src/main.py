import supervisely_lib as sly

owner_id = int(os.environ['context.userId'])
team_id = int(os.environ['context.teamId'])

app: sly.AppService = sly.AppService(ignore_task_id=True)


@ag.app.callback("manual_selected_image_changed")
def event_next_image(api: sly.Api, task_id, context, state, app_logger):
    pass

@ag.app.callback("delete_reference")
@sly.timeit
def delete_reference(api: sly.Api, task_id, context, state, app_logger):
    pass


def main():
    data = {}
    data["user"] = {}

    data["catalog"] = {"columns": [], "data": []}
    data["ownerId"] = ag.owner_id
    data["targetProject"] = {"id": ag.project.id, "name": ag.project.name}
    data["currentMeta"] = {}
    data["fieldName"] = ag.field_name

    state = {}
    state["selectedTab"] = "product"
    state["targetClass"] = ag.target_class_name
    state["multiselectClass"] = ag.multiselect_class_name
    state["user"] = {}

    sly.logger.info("Initialize catalog ...")
    catalog.init()
    data["catalog"] = json.loads(catalog.df.to_json(orient="split"))
    data["emptyGallery"] = references.empty_gallery

    sly.logger.info("Initialize existing references ...")
    references.index_existing()

    ag.app.run(data=data, state=state)

#@TODO: save references to file at the end
#@TODO: redme - how to hide object properties on object select event
#@TODO: readme - create classes before start
if __name__ == "__main__":
    sly.main_wrapper("main", main)
