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
    def __init__(self, api: sly.Api, min_instance_version: str = None):
        self.is_compatible = None
        self.api = api
        self._min_instance_version = (
            "6.9.31" if min_instance_version is None else min_instance_version
        )

    def check_instance_ver_compatibility(self):
        if self.api.instance_version < self._min_instance_version:
            sly.logger.info(
                f"Supervisely instance version does not support workflow features. To use them, please update your instance minimum to version {self._min_instance_version}."
            )
            return False
        return True

    @check_compatibility
    def add_input(self, project_id: int = None, session_id: int = None, dataset_id: int = None):
        if session_id is not None:
            self.api.app.workflow.add_input_task(int(session_id))
        if dataset_id is not None:
            self.api.app.workflow.add_input_dataset(dataset_id)
        if (
            project_id and self.api.project.get_info_by_id(project_id).version
        ):  # to prevent cycled workflow
            self.api.app.workflow.add_input_project(project_id)

    @check_compatibility
    def add_output(self, project_id: int = None, dataset_id: int = None):
        if project_id is not None:
            self.api.app.workflow.add_output_project(project_id)
        if dataset_id is not None:
            self.api.app.workflow.add_output_dataset(dataset_id)
