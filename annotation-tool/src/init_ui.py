def init_ui(data, state):
    data["info"] = {}
    data["classes"] = []
    data["tags"] = []
    data["connected"] = False
    data["connectionError"] = ""
    data["rollbackIds"] = []
    data["ssOptions"] = {
        "sessionTags": ["deployed_nn", "deployed_nn_keypoints", "deployed_nn_cls"],
        "showLabel": False,
        "size": "mini",
        "sessionTagsCombination": False,
    }

    state["sessionId"] = ""
    state["classes"] = []
    state["tags"] = []
    state["tabName"] = "info"
    state["suffix"] = "model"
    state["settings"] = "# empty"
    state["addMode"] = "merge"
    state["processing"] = False