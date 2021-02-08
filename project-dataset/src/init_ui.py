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