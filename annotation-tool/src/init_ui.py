def init_ui(data, state):
    data["info"] = {}
    data["classes"] = []
    data["tags"] = []
    data["connected"] = False
    data["connectionError"] = ""
    data["rollbackIds"] = []
    data["ssOptions"] = {
        "sessionTags": [
            "deployed_nn",
            "deployed_nn_keypoints",
            "deployed_nn_object_segmentation",
            "deployed_owl_vit_object_detection",
            "sly_smart_annotation",
        ],
        "showLabel": False,
        "size": "mini",
        "sessionTagsOperation": "or",
    }

    state["sessionId"] = ""
    state["classes"] = []
    state["tags"] = []
    state["tabName"] = "info"
    state["suffix"] = "model"
    state["useModelSuffix"] = False
    state["settings"] = "# empty"
    state["addMode"] = "merge"
    state["processing"] = False
