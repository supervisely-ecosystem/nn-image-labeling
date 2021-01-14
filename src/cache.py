import threading
import supervisely_lib as sly

metas = {}  # project id -> project meta
metas_lock = threading.Lock()

# for Undo operation
anns = {}  # image_id -> labels
anns_lock = threading.Lock()


def get_project_meta(api: sly.Api, project_id):
    project_meta = metas.get(project_id, None)
    if project_meta is None:
        project_meta_json = api.project.get_meta(project_id)
        project_meta = sly.ProjectMeta.from_json(project_meta_json)
        metas_lock.acquire()
        metas[project_id] = project_meta
        metas_lock.release()
    return project_meta


def update_project_meta(api: sly.Api, project_id, project_meta: sly.ProjectMeta):
    api.project.update_meta(project_id, project_meta.to_json())
    metas_lock.acquire()
    metas[project_id] = project_meta
    metas_lock.release()


def backup_ann(api: sly.Api, image_id, project_meta: sly.ProjectMeta):
    ann_json = api.annotation.download(image_id).annotation
    ann = sly.Annotation.from_json(ann_json, project_meta)
    anns_lock.acquire()
    anns[image_id] = ann
    anns_lock.release()


def restore_ann(image_id):
    if image_id not in anns:
        raise KeyError(f"Implementation error: image_id = {image_id}")
    return anns[image_id]