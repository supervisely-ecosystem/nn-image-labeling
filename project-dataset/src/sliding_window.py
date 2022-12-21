from matplotlib import scale
import supervisely as sly
from supervisely.geometry.sliding_windows_fuzzy import SlidingWindowsFuzzy, SlidingWindowBorderStrategy

import os
import random
import cv2
import math
import imgaug.augmenters as iaa

import sly_globals as g


def init(data, state):
    state["inference_mode"] = "full"  # ["full", "sliding_window"]
    state["windowHeight"] = 256
    state["windowWidth"] = 256
    state["overlapY"] = 32
    state["overlapX"] = 32
    state["borderStrategy"] = "shift_window"  # ["shift_window", "add_padding", "change_size"]
    state["fps"] = 4
    state["drawLabels"] = True

    data["done4"] = False
    state["collapsed4"] = True
    state["disabled4"] = True
    data["videoUrl"] = None

    data["progress"] = 0
    data["progressCurrent"] = 0
    data["progressTotal"] = 0

    state["previewLoading"] = False
    data["progressPreview"] = 0
    data["progressPreviewMessage"] = "Rendering frames"
    data["progressPreviewCurrent"] = 0
    data["progressPreviewTotal"] = 0


def restart(data, state):
    data['done4'] = False


def check_sliding_sizes_by_image(img_info, state):
    if state["windowHeight"] > img_info.height:
        state["windowHeight"] = img_info.height

    if state["windowWidth"] > img_info.width:
        state["windowWidth"] = img_info.width


def refresh_progress_preview(progress: sly.Progress):
    fields = [
        {"field": "data.progressPreview", "payload": int(progress.current * 100 / progress.total)},
        {"field": "data.progressPreviewMessage", "payload": progress.message},
        {"field": "data.progressPreviewCurrent", "payload": progress.current},
        {"field": "data.progressPreviewTotal", "payload": progress.total},
    ]

    g.api.task.set_fields(g.task_id, fields)


def write_video(state, img, predictions, last_two_frames_copies=8, max_video_size=1080):
    scale_ratio = None
    if img.shape[1] > max_video_size:
        scale_ratio = max_video_size / img.shape[1]
        img = cv2.resize(img, (int(img.shape[1] * scale_ratio), int(img.shape[0] * scale_ratio)))

    video_path = os.path.join(g.my_app.data_dir, "preview.mp4")
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'VP90'), state["fps"], (img.shape[1], img.shape[0]))
    report_every = max(5, math.ceil(len(predictions) / 100))
    progress = sly.Progress("Rendering frames", len(predictions))
    refresh_progress_preview(progress)

    for i, pred in enumerate(predictions):
        rect = pred["rectangle"]
        rect = sly.Rectangle.from_json(rect)
        if scale_ratio is not None:
            rect = rect.scale(scale_ratio)
        labels = pred["labels"]
        for label_ind, label in enumerate(labels):
            labels[label_ind] = sly.Label.from_json(label, g.model_meta)

        frame = img.copy()
        rect.draw_contour(frame, [255, 0, 0], thickness=5)
        for label in labels:
            if scale_ratio is not None:
                label = sly.Label(label.geometry.scale(scale_ratio), label.obj_class)
            label.draw_contour(frame, thickness=3)
        sly.image.write(os.path.join(g.my_app.data_dir, f"{i:05d}.jpg"), frame)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if i >= len(predictions) - 2:
            for n in range(last_two_frames_copies):
                video.write(frame_bgr)
        else:
            video.write(frame_bgr)

        progress.iter_done_report()
        if i % report_every == 0:
            refresh_progress_preview(progress)

    progress = sly.Progress("Saving video file", 1)
    progress.iter_done_report()
    refresh_progress_preview(progress)
    video.release()

    progress = sly.Progress("Uploading video", 1)
    progress.iter_done_report()
    refresh_progress_preview(progress)
    remote_video_path = os.path.join(f"/video_preview", "preview.mp4")
    if g.api.file.exists(g.team_id, remote_video_path):
        g.api.file.remove(g.team_id, remote_video_path)
    file_info = g.api.file.upload(g.team_id, video_path, remote_video_path)
    return file_info


def get_sliding_window_params_from_state(state):
    return {
        "inference_mode": state["infMode"],
        "sliding_window_params": {
            "windowHeight": state["windowHeight"],
            "windowWidth": state["windowWidth"],
            "overlapY": state["overlapY"],
            "overlapX": state["overlapX"],
            "borderStrategy": state["borderStrategy"],
            "naive": False
        }
    }


@g.my_app.callback("sliding-window-preview")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def preview(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "data.videoUrl", "payload": None},
        {"field": "state.previewLoading", "payload": True},
    ]
    api.task.set_fields(task_id, fields)

    if state['randomImagePreview'] is True:
        image_info = random.choice(g.input_images)
    else:
        image_info = [image_info for image_info in g.input_images if image_info.id == state['previewOnImageId']][0]

    check_sliding_sizes_by_image(image_info, state)
    inf_setting = get_sliding_window_params_from_state(state)

    ann_pred_res = api.task.send_request(state['sessionId'], "inference_image_id",
                                         data={
                                             "image_id": image_info.id,
                                             "settings": inf_setting
                                         }, timeout=200)

    try:
        predictions = ann_pred_res["data"]["slides"]
    except Exception as ex:
        raise ValueError('Cannot parse slides predictions, reason: {}'.format(repr(ex)))

    img = api.image.download_np(image_info.id)
    file_info = write_video(state, img, predictions)

    fields = [
        {"field": "state.previewLoading", "payload": False},
        {"field": "data.videoUrl", "payload": file_info.storage_path},
    ]
    api.task.set_fields(task_id, fields)
