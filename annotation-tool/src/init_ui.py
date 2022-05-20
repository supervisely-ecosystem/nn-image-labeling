
def init_ui(data, state):
    data["info"] = {}
    data["classes"] = []
    data["tags"] = []
    data["connected"] = False
    data["connectionError"] = ""
    data["rollbackIds"] = []
    data["ssOptions"] = {
        "sessionTags": ["deployed_nn"],
        "showLabel": False,
        "size": "mini"
    }

    state["sessionId"] = ""
    state["classes"] = []
    state["tags"] = []
    state["tabName"] = "info"
    state["suffix"] = "model"
    state["modelSettings"] = "# empty"
    state["addMode"] = "merge"
    state["processing"] = False

    state["appSettings"] = "imageOrRoi"  # "slidingWindow"
    state["windowHeight"] = 256
    state["windowWidth"] = 256
    state["overlapY"] = 32
    state["overlapX"] = 32
    state["borderStrategy"] = "shift_window"

    state["debugSlidingWindow"] = False
