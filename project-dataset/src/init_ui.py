def init_input_project(api, data, project_info):
    data["projectId"] = project_info.id
    data["projectName"] = project_info.name
    data["projectPreviewUrl"] = api.image.preview_url(project_info.reference_image_url, 100, 100)
    data["projectItemsCount"] = project_info.items_count


def init_output_project(data):
    data["resProjectId"] = None
    data["resProjectName"] = None
    data["resProjectPreviewUrl"] = None


def init(data, state):
    data["info"] = {}
    data["classes"] = []
    data["tags"] = []
    data["connected"] = False
    data["connectionError"] = ""
    data["ssOptions"] = {
        "sessionTags": ["deployed_nn"],
        "showLabel": False,
        "size": "small"
    }

    state["sessionId"] = ""
    state["classes"] = []
    state["tags"] = []
    state["tabName"] = "info"
    state["suffix"] = "model"
    state["settings"] = "# empty"
    state["addMode"] = "merge"
    state["processing"] = False
