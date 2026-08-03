import supervisely as sly
from supervisely.collection.key_indexed_collection import KeyIndexedCollection


def generate_res_name(item, suffix, index):
    return f"{item.name}-{suffix}" if index == 0 else f"{item.name}-{suffix}-{index}"


def items_match(existing_item, model_item, candidate_name):
    candidate = model_item.clone(name=candidate_name)
    if isinstance(model_item, sly.ObjClass):
        return (
            existing_item.name == candidate.name
            and existing_item.geometry_type == candidate.geometry_type
            and existing_item.geometry_config == candidate.geometry_config
        )
    return existing_item == candidate


def find_item(
    collection: KeyIndexedCollection,
    item,
    suffix,
    use_suffix: bool = False,
):
    base_name = item.name.strip()

    if use_suffix is False:
        existing_item = collection.get(base_name)
        if existing_item is None:
            return None, base_name
        if items_match(existing_item, item, base_name):
            return existing_item, None

    index = 0
    while True:
        res_name = generate_res_name(item, suffix, index)
        existing_item = collection.get(res_name)
        if existing_item is None:
            return None, res_name
        if items_match(existing_item, item, res_name):
            return existing_item, None
        index += 1


def merge_metas(
    project_meta: sly.ProjectMeta,
    model_meta: sly.ProjectMeta,
    keep_model_classes,
    keep_model_tags,
    suffix,
    use_suffix: bool = False,
):
    res_meta = project_meta.clone()

    def _merge(keep_names, res_meta, project_collection, model_collection, is_class=False):
        mapping = {}  # old name to new meta
        for name in keep_names:
            model_item = model_collection.get(name)
            res_item, res_name = find_item(project_collection, model_item, suffix, use_suffix)
            if res_item is None:
                res_item = model_item.clone(name=res_name)
                res_meta = (
                    res_meta.add_obj_class(res_item)
                    if is_class
                    else res_meta.add_tag_meta(res_item)
                )
            mapping[model_item.name.strip()] = res_item
        return res_meta, mapping

    res_meta, class_mapping = _merge(
        keep_model_classes, res_meta, res_meta.obj_classes, model_meta.obj_classes, is_class=True
    )
    res_meta, tag_mapping = _merge(
        keep_model_tags, res_meta, res_meta.tag_metas, model_meta.tag_metas, is_class=False
    )
    return res_meta, class_mapping, tag_mapping
