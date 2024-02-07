import supervisely as sly
import yaml
from supervisely.app.widgets import Button, Card, Container, Input, Progress, ProjectThumbnail

import project_dataset.src.globals as g
import project_dataset.src.ui.inference_settings as settings
from project_dataset.src.ui.inference_preview import apply_model_to_image, apply_model_to_images

output_project_name = Input(f"{g.project_info.name}_inference", minlength=1)
apply_button = Button("Apply model to input data", icon="zmdi zmdi-check")

inference_progress = Progress()

output_project_thumbnail = ProjectThumbnail()
output_project_thumbnail.hide()

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


@apply_button.click
def apply_model():
    try:
        inference_settings = yaml.safe_load(settings.additional_settings.get_value())
        sly.logger.info("Final Inference Settings:")
        for key, value in settings.items():
            sly.logger.info(f"{key}: {value}")
    except Exception as e:
        inference_settings = {}
        sly.logger.warning(
            f"Model Inference launched without additional settings. \n" f"Reason: {e}",
            exc_info=True,
        )

    res_project_meta = g.project_meta.clone()
    res_project = g.api.project.create(
        g.workspace_id, output_project_name.get_value(), change_name_if_conflict=True
    )
    g.api.project.update_meta(res_project.id, res_project_meta.to_json())

    with inference_progress(message="Processing images...", total=len(g.input_images)) as pbar:
        for dataset_id in g.selected_datasets:
            dataset_info = g.api.dataset.get_info_by_id(dataset_id)
            res_dataset = g.api.dataset.create(
                res_project.id, dataset_info.name, dataset_info.description
            )
            image_infos = g.api.image.get_list(dataset_info.id)

            for batched_image_infos in sly.batched(image_infos, batch_size=10):
                try:
                    image_ids, res_names, res_metas = [], [], []
                    for image_info in batched_image_infos:
                        image_ids.append(image_info.id)
                        res_names.append(image_info.name)
                        res_metas.append(image_info.meta)
                    _, res_anns, final_project_meta = apply_model_to_images(
                        dataset_info.id, batched_image_infos, inference_settings
                    )
                except Exception as e:
                    sly.logger.warn(
                        msg=f"Couldn't process images by batch, images will be processed one by one, error: {e}."
                    )

                    image_ids, res_names, res_anns, res_metas = [], [], [], []
                    for image_info in batched_image_infos:
                        try:
                            _, res_ann, final_project_meta = apply_model_to_image(
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
                    g.api.project.update_meta(res_project.id, res_project_meta.to_json())

                res_images_infos = g.api.image.upload_ids(
                    res_dataset.id, res_names, image_ids, metas=res_metas
                )
                res_ids = [image_info.id for image_info in res_images_infos]
                try:
                    g.api.annotation.upload_anns(res_ids, res_anns)
                except:
                    for res_img_info, ann in zip(res_images_infos, res_anns):
                        try:
                            g.api.annotation.upload_ann(res_img_info.id, ann)
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

    output_project_thumbnail.set(res_project)
    output_project_thumbnail.show()

    from project_dataset.src.main import app

    app.stop()
