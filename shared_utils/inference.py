import supervisely as sly
import shared_utils.ui2 as ui
from shared_utils.merge_metas import merge_metas


OUTPUT_GEOMETRIES = {
    "model": None,
    "polygon": sly.Polygon,
}


def _can_convert_geometry(source_geometry, target_geometry):
    if target_geometry is None or source_geometry == target_geometry:
        return True
    return target_geometry in source_geometry.allowed_transforms()


def _get_output_model_meta(model_meta: sly.ProjectMeta, output_geometry: str):
    target_geometry = OUTPUT_GEOMETRIES.get(output_geometry)
    if target_geometry is None:
        return model_meta

    output_classes = []
    for obj_class in model_meta.obj_classes:
        if _can_convert_geometry(obj_class.geometry_type, target_geometry):
            obj_class = obj_class.clone(geometry_type=target_geometry, geometry_config={})
        output_classes.append(obj_class)
    return model_meta.clone(obj_classes=output_classes)


def postprocess(
    api: sly.Api,
    project_id,
    ann: sly.Annotation,
    project_meta: sly.ProjectMeta,
    model_meta: sly.ProjectMeta,
    state,
):
    keep_classes = ui.get_keep_classes(state)  # @TODO: for debug ['dog'] #
    keep_tags = ui.get_keep_tags(state)
    output_model_meta = _get_output_model_meta(
        model_meta, state.get("outputGeometry", "model")
    )
    res_project_meta, class_mapping, tag_meta_mapping = merge_metas(
        project_meta,
        output_model_meta,
        keep_classes,
        keep_tags,
        state["suffix"],
        state["useModelSuffix"],
    )

    image_tags = []
    for tag in ann.img_tags:
        if tag.meta.name not in keep_tags:
            continue
        image_tags.append(tag.clone(meta=tag_meta_mapping[tag.meta.name]))

    new_labels = []
    for label in ann.labels:
        if label.obj_class.name not in keep_classes:
            continue
        label_tags = []
        for tag in label.tags:
            if tag.meta.name not in keep_tags:
                continue
            label_tags.append(tag.clone(meta=tag_meta_mapping[tag.meta.name]))
        output_class = class_mapping[label.obj_class.name.strip()]
        label = label.clone(tags=sly.TagCollection(label_tags))
        if label.geometry.geometry_name() == output_class.geometry_type.geometry_name():
            new_labels.append(label.clone(obj_class=output_class))
        else:
            new_labels.extend(label.convert(output_class))

    res_ann = ann.clone(labels=new_labels, img_tags=sly.TagCollection(image_tags))
    return res_ann, res_project_meta
