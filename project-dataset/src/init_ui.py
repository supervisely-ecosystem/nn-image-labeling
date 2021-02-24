from globals import project_info, api


def init_input_project(data):
    data["projectId"] = project_info.id
    data["projectName"] = project_info.name
    data["projectPreviewUrl"] = api.image.preview_url(project_info.reference_image_url, 100, 100)
    data["projectItemsCount"] = project_info.items_count


def init_ui(data, state):
    data["projectId"] = None
    data["projectName"] = None
    data["projectPreviewUrl"] = None

    data["resProjectId"] = None
    data["resProjectName"] = None
    data["resProjectPreviewUrl"] = None


# def unit_ui(data, state):
#     data["info"] = {}
#     data["classes"] = []
#     data["tags"] = []
#     data["connected"] = False
#     data["connectionError"] = ""
#     data["rollbackIds"] = []
#     data["ssOptions"] = {
#         "sessionTags": ["deployed_nn"],
#         "showLabel": False,
#         "size": "mini"
#     }
#
#     state["sessionId"] = ""
#     state["classes"] = []
#     state["tags"] = []
#     state["tabName"] = "info"
#     state["suffix"] = "model"
#     state["settings"] = "# empty"
#     state["addMode"] = "merge"
#     state["processing"] = False