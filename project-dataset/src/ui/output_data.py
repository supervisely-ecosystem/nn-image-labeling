import importlib

import supervisely as sly
import yaml
from supervisely.app.widgets import Button, Card, Container, Input, Progress, ProjectThumbnail

g = importlib.import_module("project-dataset.src.globals")
settings = importlib.import_module("project-dataset.src.ui.inference_settings")
inference_preview = importlib.import_module("project-dataset.src.ui.inference_preview")
nn_info = importlib.import_module("project-dataset.src.ui.nn_info")

output_project_name = Input(f"{g.project_info.name}_inference", minlength=1)
apply_button = Button("Apply model to input data", icon="zmdi zmdi-check")

inference_progress = Progress()

output_project_thumbnail = ProjectThumbnail()
output_project_thumbnail.hide()

api: sly.Api = g.api

card = Card(
    "6️⃣ Output data",
    "New project with predictions will be created. Original project will not be modified.",
    content=Container(
        [output_project_name, apply_button, inference_progress, output_project_thumbnail]
    ),
    collapsable=True,
    lock_message="Connect to the deployed neural network on step 2️⃣.",
)
card.lock()
card.collapse()


def apply_model_ds(
    src_project, dst_project, inference_settings, res_project_meta, save_imag_tags=False
):
    def process_ds(src_ds_info, parent_id):
        t = time.time()
        dst_dataset_info = api.dataset.create(
            dst_project.id, src_ds_info.name, src_ds_info.description, parent_id=parent_id
        )
        dst_dataset_infos[src_ds_info] = dst_dataset_info

        src_images = api.image.get_list(src_ds_info.id)
        src_ds_image_infos_dict[src_ds_info.id] = {
            image_info.id: image_info for image_info in src_images
        }
        src_image_ids = [image.id for image in src_images]
        if len(src_image_ids) > 0:
            dst_img_infos = api.image.copy_batch(dst_dataset_info.id, src_image_ids)
        else:
            dst_img_infos = []
        for image_info in dst_img_infos:
            dst_image_infos_dict.setdefault(dst_dataset_info.id, {})[image_info.name] = image_info
        timer.setdefault(src_ds_info.id, {})["copy"] = time.time() - t

        pbar.update(1)
        return dst_dataset_info.id

    def process_ds_tree(ds_tree, parent_id=None):
        for ds_info, children in ds_tree.items():
            current_ds_id = process_ds(ds_info, parent_id)
            if children:
                process_ds_tree(children, current_ds_id)

    def count_selected_ds(data):
        ds_count = 0
        images_count = 0

        for ds_info, children in data.items():
            ds_count += 1
            images_count += ds_info.images_count

            if children:
                nested_ds_count, nested_images_count = count_selected_ds(children)
                ds_count += nested_ds_count
                images_count += nested_images_count

        return ds_count, images_count

    def ds_tree_to_list(data):
        dataset_list = []

        for ds_info, children in data.items():
            dataset_list.append(ds_info)
            if children:
                dataset_list.extend(ds_tree_to_list(children))

        return dataset_list

    import time

    timer = {}
    dst_dataset_infos = {}
    try:
        # 1. Create destination datasets
        src_ds_tree = api.dataset.get_tree(src_project)
        src_ds_tree = {k: v for k, v in src_ds_tree.items() if k.id in g.selected_datasets}
        src_ds_list = ds_tree_to_list(src_ds_tree)
        selected_ds_count, selected_images_count = count_selected_ds(src_ds_tree)

        dst_dataset_infos = {}
        dst_image_infos_dict = {}  # dataset_id -> name -> image_info
        src_ds_image_infos_dict = {}  # dataset_id -> image_id -> [image_infos]

        with inference_progress(message="Creating datasets...", total=selected_ds_count) as pbar:
            process_ds_tree(src_ds_tree)
        # 2. Apply model to the datasets
        with inference_progress(
            message="Processing images...", total=selected_images_count
        ) as pbar:
            for src_dataset_info in src_ds_list:
                # iterating over batches of predictions
                if src_dataset_info.images_count == 0:
                    continue
                t = time.time()
                for (
                    _,
                    merged_ann_infos_batch,
                    final_project_meta,
                ) in inference_preview.apply_model_to_datasets(
                    src_project,
                    [src_dataset_info.id],
                    inference_settings,
                    classes=[
                        obj_class.name
                        for obj_class in nn_info.select_classes.get_selected_classes()
                    ],
                    batch_size=50,
                    image_infos=list(src_ds_image_infos_dict[src_dataset_info.id].values()),
                ):
                    timer.setdefault(src_dataset_info.id, {}).setdefault("items", 0)
                    timer[src_dataset_info.id]["items"] += len(merged_ann_infos_batch)
                    timer.setdefault(src_dataset_info.id, {}).setdefault("apply_model", 0)
                    timer[src_dataset_info.id]["apply_model"] += time.time() - t
                    t = time.time()
                    dst_dataset_info = dst_dataset_infos[src_dataset_info]
                    # Update project meta if needed
                    if res_project_meta != final_project_meta:
                        res_project_meta = final_project_meta
                        api.project.update_meta(dst_project.id, res_project_meta.to_json())
                        timer.setdefault(src_dataset_info.id, {}).setdefault("update_meta", 0)
                        timer[src_dataset_info.id]["update_meta"] += time.time() - t
                        t = time.time()

                    dst_anns = []
                    dst_image_infos = []
                    for ann_info in merged_ann_infos_batch:
                        src_image_id = ann_info.image_id
                        src_image_info = src_ds_image_infos_dict[src_dataset_info.id][src_image_id]

                        dst_image_infos.append(
                            dst_image_infos_dict[dst_dataset_info.id][src_image_info.name]
                        )
                        dst_anns.append(
                            sly.Annotation.from_json(ann_info.annotation, res_project_meta)
                        )
                    timer.setdefault(src_dataset_info.id, {}).setdefault("prepare_anns", 0)
                    timer[src_dataset_info.id]["prepare_anns"] += time.time() - t
                    t = time.time()
                    # upload_annotations
                    try:
                        api.annotation.upload_anns(
                            [image_info.id for image_info in dst_image_infos], dst_anns
                        )
                        pbar.update(len(dst_anns))
                    except:
                        for img_info, ann in zip(dst_image_infos, dst_anns):
                            try:
                                api.annotation.upload_ann(img_info.id, ann)
                            except Exception as e:
                                sly.logger.warn(
                                    msg=f"Image: {img_info.name} (Image ID: {img_info.id}) couldn't be uploaded, image will be skipped, error: {e}.",
                                    extra={
                                        "image_name": img_info.name,
                                        "image_id": img_info.id,
                                        "image_meta": img_info.meta,
                                        "image_ann": ann,
                                    },
                                )
                                continue
                            finally:
                                pbar.update(1)
                    finally:
                        timer.setdefault(src_dataset_info.id, {}).setdefault("upload_anns", 0)
                        timer[src_dataset_info.id]["upload_anns"] += time.time() - t
                        t = time.time()
    except Exception:
        api.dataset.remove_batch([ds.id for ds in dst_dataset_infos.values()])
        raise
    finally:
        sly.logger.debug("Timer:", extra={"timer": timer})


@apply_button.click
def apply_model():
    """Applies the model to the input data and creates a new project with the predictions.
    After the process is finished, the new project will be shown and the app will be stopped."""
    try:
        inference_settings = yaml.safe_load(settings.additional_settings.get_value())
        if inference_settings is None:
            inference_settings = {}
        if g.model_info["app_name"].startswith("Serve Segment Anything"):
            inference_settings["mode"] = "raw"
        sly.logger.info(f"Final Inference Settings: {inference_settings}")

    except Exception as e:
        inference_settings = {}
        sly.logger.warning(
            f"Model Inference launched without additional settings. \n" f"Reason: {e}",
            exc_info=True,
        )

    res_project_meta = g.project_meta.clone()
    res_project = api.project.create(
        g.workspace_id, output_project_name.get_value(), change_name_if_conflict=True
    )
    api.project.update_meta(res_project.id, res_project_meta.to_json())

    # -------------------------------------- Add Workflow Input -------------------------------------- #
    g.workflow.add_input(project_id=g.selected_project, session_id=g.model_session_id)
    # ----------------------------------------------- - ---------------------------------------------- #

    try:
        apply_model_ds(g.selected_project, res_project, inference_settings, res_project_meta)
    except Exception as e:
        sly.logger.warn(
            msg=f"Couldn't apply model to the input data, error: {e}.",
            exc_info=True,
        )

        with inference_progress(message="Processing images...", total=len(g.input_images)) as pbar:
            for dataset_id in g.selected_datasets:
                dataset_info = api.dataset.get_info_by_id(dataset_id)
                if dataset_info is None:
                    if not api.project.exists(g.workspace_id, g.project_info.name):
                        raise RuntimeError("Input project no longer exists")
                    sly.logger.error(
                        f"Input dataset (id: {dataset_id}) is not found, images could not be processed."
                    )
                    continue
                res_dataset = api.dataset.create(
                    res_project.id,
                    dataset_info.name,
                    dataset_info.description,
                    dataset_info.parent_id,
                )

                final_project_meta = None
                image_infos = api.image.get_list(dataset_info.id)
                for batched_image_infos in sly.batched(image_infos, batch_size=10):
                    try:
                        image_ids, res_names, res_metas = [], [], []
                        for image_info in batched_image_infos:
                            image_ids.append(image_info.id)
                            res_names.append(image_info.name)
                            res_metas.append(image_info.meta)
                        _, res_anns, final_project_meta = inference_preview.apply_model_to_images(
                            dataset_info.id, batched_image_infos, inference_settings
                        )
                    except Exception as e:
                        sly.logger.warn(
                            msg=f"Couldn't process images by batch, images will be processed one by one, error: {e}."
                        )
                        image_ids, res_names, res_anns, res_metas = [], [], [], []
                        for image_info in batched_image_infos:
                            try:
                                _, res_ann, final_project_meta = (
                                    inference_preview.apply_model_to_image(
                                        image_info, inference_settings
                                    )
                                )

                                image_ids.append(image_info.id)
                                res_names.append(image_info.name)
                                res_anns.append(res_ann)
                                res_metas.append(image_info.meta)
                            except Exception as e:
                                sly.logger.warn(
                                    msg=f"Image: {image_info.name} (ID: {image_info.id}) couldn't be processed, image will be skipped, error: {e}.",
                                    extra={
                                        "image_name": image_info.name,
                                        "image_id": image_info.id,
                                        "image_meta": image_info.meta,
                                    },
                                )
                                continue

                    if res_project_meta != final_project_meta:
                        res_project_meta = final_project_meta
                        api.project.update_meta(res_project.id, res_project_meta.to_json())

                    res_images_infos = api.image.upload_ids(
                        res_dataset.id, res_names, image_ids, metas=res_metas
                    )
                    res_ids = [image_info.id for image_info in res_images_infos]
                    try:
                        api.annotation.upload_anns(res_ids, res_anns)
                    except:
                        for res_img_info, ann in zip(res_images_infos, res_anns):
                            try:
                                api.annotation.upload_ann(res_img_info.id, ann)
                            except Exception as e:
                                sly.logger.warn(
                                    msg=f"Image: {res_img_info.name} (Image ID: {res_img_info.id}) couldn't be uploaded, image will be skipped, error: {e}.",
                                    extra={
                                        "image_name": res_img_info.name,
                                        "image_id": res_img_info.id,
                                        "image_meta": res_img_info.meta,
                                        "image_ann": ann,
                                    },
                                )
                                continue
                    pbar.update(len(batched_image_infos))
    output_project_thumbnail.set(api.project.get_info_by_id(res_project.id))
    output_project_thumbnail.show()
    # -------------------------------------- Add Workflow Output ------------------------------------- #
    g.workflow.add_output(project_id=res_project.id)
    # ----------------------------------------------- - ---------------------------------------------- #
    main = importlib.import_module("project-dataset.src.main")

    main.app.stop()
