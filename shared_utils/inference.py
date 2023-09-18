import supervisely as sly
import shared_utils.ui2 as ui
from shared_utils.merge_metas import merge_metas


def postprocess(
    api: sly.Api,
    project_id,
    ann: sly.Annotation,
    project_meta: sly.ProjectMeta,
    model_meta: sly.ProjectMeta,
    state,
):
    try:
        keep_classes = ui.get_keep_classes(state)  # @TODO: for debug ['dog'] #
        keep_tags = ui.get_keep_tags(state)
    except KeyError as e:
        Exception(
            "Can not get classesInfo/tagsInfo from state. "
            "You are probably trying to use a model trained outside Supervisely. "
            "You can use one of pretrained models or train a model in Supervisely via corresponding app. "
            f"{repr(e)}"
        )
    res_project_meta, class_mapping, tag_meta_mapping = merge_metas(
        project_meta, model_meta, keep_classes, keep_tags, state["suffix"], state["useModelSuffix"]
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
        new_label = label.clone(
            obj_class=class_mapping[label.obj_class.name], tags=sly.TagCollection(label_tags)
        )
        new_labels.append(new_label)

    res_ann = ann.clone(labels=new_labels, img_tags=sly.TagCollection(image_tags))
    return res_ann, res_project_meta
