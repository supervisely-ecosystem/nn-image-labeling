import supervisely as sly


def check_compatibility(func):
    def wrapper(self, *args, **kwargs):
        if self.is_compatible is None:
            self.is_compatible = self.check_instance_ver_compatibility()
        if not self.is_compatible:
            return
        return func(self, *args, **kwargs)

    return wrapper


class Workflow:
    def __init__(self):
        self.is_compatible = None
        self._min_instance_version = "6.9.22"

    @check_compatibility
    def add_input(self, api: sly.Api, project_id: int, state: dict):
        if api.project.get_info_by_id(project_id).version:  # to prevent cycled workflow
            api.app.add_input_project(project_id)
        api.app.add_input_task(int(state["sessionId"]))

    @check_compatibility
    def add_output(self, api: sly.Api, project_id: int):
        api.app.add_output_project(project_id)

    def check_instance_ver_compatibility(self, api: sly.Api):
        if api.instance_version < self._min_instance_version:
            sly.logger.info(
                f"Supervisely instance version does not support workflow and versioning features. To use them, please update your instance minimum to version {self._min_instance_version}."
            )
            return False
        return True
