
def unit_ui(data, state):
    data["info"] = {}
    data["classes"] = []
    data["tags"] = []
    data["connected"] = False
    data["connectionError"] = ""

    state["sessionId"] = "2361"  # @TODO: for debug
    state["classes"] = []
    state["tags"] = []
    state["tabName"] = "info"
    state["suffix"] = "model"
    state["settings"] = "# empty"