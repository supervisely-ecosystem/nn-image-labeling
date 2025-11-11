import importlib
from typing import List

import supervisely as sly
import yaml
from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    Container,
    Empty,
    Field,
    Input,
    Progress,
    ProjectThumbnail,
)

g = importlib.import_module("project-dataset.src.globals")
settings = importlib.import_module("project-dataset.src.ui.inference_settings")
inference_preview = importlib.import_module("project-dataset.src.ui.inference_preview")
nn_info = importlib.import_module("project-dataset.src.ui.nn_info")

continue_field = Field(
    content=Empty(),
    title="Continue",
    description="If selected, predictions will be uploaded to the source project. Already annotated images will be skipped",
)
continue_predict_checkbox = Checkbox(content=continue_field)
output_project_name = Input(f"{g.project_info.name}_inference", minlength=1)


@continue_predict_checkbox.value_changed
def _continue_predict_checkbox_value_changed(is_checked):
    if is_checked:
        output_project_name.hide()
    else:
        output_project_name.show()


apply_button = Button("Apply model to input data", icon="zmdi zmdi-check")

progress_main = Progress()
progress_secondary = Progress(hide_on_finish=True)

output_project_thumbnail = ProjectThumbnail()
output_project_thumbnail.hide()

api: sly.Api = g.api

card = Card(
    "6️⃣ Output data",
    "New project with predictions will be created. Original project will not be modified.",
    content=Container(
        [
            continue_predict_checkbox,
            output_project_name,
            apply_button,
            progress_main,
            progress_secondary,
            output_project_thumbnail,
        ]
    ),
    collapsable=True,
    lock_message="Connect to the deployed neural network on step 2️⃣.",
)
card.lock()
card.collapse()


def apply_model_ds(
    src_project,
    dst_project,
    inference_settings,
    res_project_meta,
    save_imag_tags=False,
    continue_predict=False,
):
    def process_ds(src_ds_info: sly.DatasetInfo, parent_id):
        t = time.time()
        sly.logger.info(
            f"Starting to process dataset: {src_ds_info.name} "
            f"(ID: {src_ds_info.id}, Images: {src_ds_info.images_count})"
        )
        dst_dataset_info = api.dataset.create(
            dst_project.id, src_ds_info.name, src_ds_info.description, parent_id=parent_id
        )
        dst_dataset_infos[src_ds_info] = dst_dataset_info
        l_time = time.time()
        sly.logger.info("Starting to collect source images info")
        src_images = api.image.get_list(src_ds_info.id)
        l_time = time.time() - l_time
        sly.logger.info(f"Collected {len(src_images)} image infos from dataset: {src_ds_info.name} in {l_time:.2f} seconds")
        src_ds_image_infos_dict[src_ds_info.id] = {
            image_info.id: image_info for image_info in src_images
        }
        # src_image_ids = [image.id for image in src_images]
        if len(src_images) > 0:
            with progress_secondary(
                message=f"Copying images from dataset: {src_ds_info.name}", total=len(src_images)
            ) as pbar2:
                dst_img_infos = api.image.copy_batch_optimized(
                    src_dataset_id=src_ds_info.id,
                    src_image_infos=src_images,
                    dst_dataset_id=dst_dataset_info.id,
                    progress_cb=pbar2.update,
                )
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

    # def ds_tree_to_list(data):
    #     dataset_list = []

    #     for ds_info, children in data.items():
    #         dataset_list.append(ds_info)
    #         if children:
    #             dataset_list.extend(ds_tree_to_list(children))

    #     return dataset_list

    def filter_tree(ds_tree, ids):
        filtered = {}
        for ds_info, children in ds_tree.items():
            filtered_children = filter_tree(children, ids) if children else {}
            if ds_info.id in ids:
                filtered[ds_info] = filtered_children
            else:
                filtered.update(filtered_children)
        return filtered

    import time

    timer = {}
    dst_dataset_infos = {}
    try:
        # 1. Create destination datasets
        src_ds_list = g.selected_datasets_aggregated
        selected_ids = [ds.id for ds in src_ds_list]
        src_ds_tree = filter_tree(g.src_ds_tree, selected_ids)
        selected_ds_count = len(selected_ids)
        selected_images_count = sum(ds.images_count for ds in src_ds_list)

        dst_dataset_infos = {}
        dst_image_infos_dict = {}  # dataset_id -> name -> image_info
        src_ds_image_infos_dict = {}  # dataset_id -> image_id -> [image_infos]

        with progress_main(message="Creating datasets...", total=selected_ds_count) as pbar:
            process_ds_tree(src_ds_tree)
        # 2. Apply model to the datasets
        with progress_main(message="Processing images...", total=selected_images_count) as pbar:
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
    finally:
        sly.logger.debug("Timer:", extra={"timer": timer})


def apply_continue_predict(project_info, project_meta, inference_settings):
    with progress_main(message="Processing images...", total=len(g.input_images)) as pbar:
        image_infos = [image_info for image_info in g.input_images if image_info.labels_count == 0]
        pbar.update(len(g.input_images) - len(image_infos))
        pbar.refresh()
        for dataset_info in g.selected_datasets_aggregated:
            ds_image_infos = [
                image_info for image_info in image_infos if image_info.dataset_id == dataset_info.id
            ]
            if len(ds_image_infos) == 0:
                continue
            for ds_image_infos_batch in sly.batched(ds_image_infos):
                _, res_anns, final_project_meta = inference_preview.apply_model_to_images(
                    dataset_info.id, ds_image_infos_batch, inference_settings
                )
                if project_meta != final_project_meta:
                    project_meta = final_project_meta
                    api.project.update_meta(project_info.id, project_meta.to_json())

                image_ids = [image_info.id for image_info in ds_image_infos_batch]
                try:
                    api.annotation.upload_anns(image_ids, res_anns)
                except:
                    for res_img_info, ann in zip(image_ids, res_anns):
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
                pbar.update(len(ds_image_infos_batch))


def get_ds_chain(target: sly.DatasetInfo, all_datasets: List[sly.DatasetInfo], chain:List = None):
    """return nested datasets chain from top to bottom"""
    if target.parent_id is None:
        return [target]
    if chain is None:
        chain = []
    target_parent = next(ds for ds in all_datasets if ds.id == target.parent_id)
    return [*get_ds_chain(target_parent, all_datasets, chain), target]

def get_chain_in_dst(chain: List[sly.DatasetInfo], dst_datasets: List[sly.DatasetInfo]):
    dst_chain = []
    parent = None
    for ds in chain:
        found = False
        for dst_ds in dst_datasets:
            if dst_ds.parent_id == parent and dst_ds.name == ds.name:
                parent = dst_ds
                dst_chain.append(dst_ds)
                found = True
                break
        if not found:
            break
    return dst_chain

def create_missing_datasets(src_chain: List[sly.DatasetInfo], dst_project_id: int, dst_chain: List[sly.DatasetInfo]) -> List[sly.DatasetInfo]:
    if src_chain < dst_chain:
        raise RuntimeError("Unexpected error! The destination dataset depth is more than source")
    while len(src_chain) > len(dst_chain):
        if len(dst_chain) == 0:
            parent_id = None
        else:
            parent_id = dst_chain[-1].id
        src_dataset = src_chain[len(dst_chain)]
        dst_dataset = api.dataset.create(project_id=dst_project_id, name=src_dataset.name, description=src_dataset.description, parent_id=parent_id)
        dst_chain.append(dst_dataset)
    return dst_chain

def get_or_create_dataset_in_dst(src_dataset: sly.DatasetInfo, dst_project_id: int, src_all_datasets: List[sly.DatasetInfo] = None, dst_all_datasets: List[sly.DatasetInfo] = None):
    if src_all_datasets is None:
        src_all_datasets = api.dataset.get_list(src_dataset.project_id, recursive=True)
    if dst_all_datasets is None:
        dst_all_datasets = api.dataset.get_list(dst_project_id, recursive=True)
    src_chain = get_ds_chain(src_dataset, src_all_datasets)
    dst_chain = get_chain_in_dst(src_chain, dst_all_datasets)
    dst_chain = create_missing_datasets(src_chain, dst_project_id, dst_chain)
    return dst_chain[-1]
    

def apply_model_safe(res_project: sly.ProjectInfo, res_project_meta: sly.ProjectMeta, inference_settings, batch_size=10):
    dst_all_datasets = api.dataset.get_list(res_project.id, recursive=True)
    with progress_main(message="Processing images...", total=len(g.input_images)) as pbar:
        for dataset_info in g.selected_datasets_aggregated:
            
            res_dataset = get_or_create_dataset_in_dst(
                src_dataset=dataset_info,
                dst_project_id=res_project.id,
                src_all_datasets=g.selected_dataset_aggregated,
                dst_all_datasets=dst_all_datasets
            )

            final_project_meta = None
            image_infos = api.image.get_list(dataset_info.id)
            for batched_image_infos in sly.batched(image_infos, batch_size=batch_size):
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
                            _, res_ann, final_project_meta = inference_preview.apply_model_to_image(
                                image_info, inference_settings
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

                existing = api.image.get_list(res_dataset.id)
                existing_names = [image_info.name for image_info in existing]
                missing_names, missing_ids, missing_metas = [], [], []
                for name, src_image_id, meta in zip(res_names, image_ids, res_metas):
                    if name not in existing_names:
                        missing_names.append(name)
                        missing_ids.append(src_image_id)
                        missing_metas.append(meta)
                if missing_names:
                    uploaded = api.image.upload_ids(res_dataset.id, missing_names, missing_ids, metas=missing_metas)
                res_images_infos = []
                name_to_info = {image_info.name: image_info for image_info in existing+uploaded}
                for name in res_names:
                    res_images_infos.append(name_to_info[name])
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

    continue_predict = continue_predict_checkbox.is_checked()
    if continue_predict:
        apply_continue_predict(g.project_info, g.project_meta.clone(), inference_settings)
    else:
        res_project_meta = g.project_meta.clone()
        res_project = api.project.create(
            g.workspace_id, output_project_name.get_value(), change_name_if_conflict=True
        )
        api.project.update_meta(res_project.id, res_project_meta.to_json())

        # -------------------------------------- Add Workflow Input -------------------------------------- #
        g.workflow.add_input(project_id=g.selected_project, session_id=g.model_session_id)
        # ----------------------------------------------- - ---------------------------------------------- #

        if g.model_info["task type"] == "prompt-based object detection":
            apply_model_safe(res_project, res_project_meta, inference_settings, batch_size=2)
        else:
            try:
                apply_model_ds(
                    g.selected_project, res_project, inference_settings, res_project_meta
                )
            except Exception as e:
                sly.logger.warn(
                    msg=f"Couldn't apply model to the input data, error: {e}.",
                    exc_info=True,
                )

                apply_model_safe(res_project, res_project_meta, inference_settings)

        output_project_thumbnail.set(api.project.get_info_by_id(res_project.id))
        output_project_thumbnail.show()
        # -------------------------------------- Add Workflow Output ------------------------------------- #
        g.workflow.add_output(project_id=res_project.id)
        # ----------------------------------------------- - ---------------------------------------------- #
    main = importlib.import_module("project-dataset.src.main")

    main.app.stop()
