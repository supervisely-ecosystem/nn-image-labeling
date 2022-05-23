import os


def init_input_project(api, data, project_info, count, dataset_info):
    data["projectId"] = project_info.id
    data["projectName"] = project_info.name
    if dataset_info is not None:
        data["projectName"] += f" / {dataset_info.name}"
    data["projectPreviewUrl"] = api.image.preview_url(
        project_info.reference_image_url, 100, 100
    )
    data["projectItemsCount"] = count


def init_sliding_window_settings(state):
    state["windowHeight"] = 256
    state["windowWidth"] = 256
    state["overlapY"] = 32
    state["overlapX"] = 32
    state["borderStrategy"] = "shift_window"  # "add_padding"
    state["fps"] = 4
    state["drawLabels"] = True


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
        "size": "small",
    }
    data["gallery"] = None
    data["started"] = False
    data["progress"] = 0
    data["progressCurrent"] = 0
    data["progressTotal"] = 0

    state["sessionId"] = int(os.environ.get("state.sessionId", ""))
    state["classes"] = []
    state["tags"] = []
    state["tabName"] = "info"
    state["infMode"] = "fi"  # roi, sw
    state["suffix"] = "model"
    state["settings"] = "# empty"
    state["addMode"] = "merge"
    state["processing"] = False

    init_sliding_window_settings(state)
