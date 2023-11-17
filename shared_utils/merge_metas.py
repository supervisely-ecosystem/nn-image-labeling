import supervisely as sly
from supervisely.collection.key_indexed_collection import KeyIndexedCollection


def generate_res_name(item, suffix, index):
    return f"{item.name}-{suffix}" if index == 0 else f"{item.name}-{suffix}-{index}"


def find_item(
    collection: KeyIndexedCollection,
    item,
    suffix,
    use_suffix: bool = False,
):
    index = 0
    res_name = item.name.strip()
    if res_name == "cat-model":
        print("horse")
    while True:
        existing_item = collection.get(res_name.strip())
        if existing_item is None:
            if use_suffix is True:
                res_name = generate_res_name(item, suffix, index)
                existing_item = collection.get(res_name)
                if existing_item is not None:
                    return existing_item, None
            return None, res_name
        else:
            if existing_item == item.clone(name=res_name):
                if use_suffix is True:
                    res_name = generate_res_name(item, suffix, index)
                    existing_item = collection.get(res_name)
                    if existing_item is None:
                        return None, res_name
                    elif existing_item == item.clone(name=res_name):
                        res_name = generate_res_name(item, suffix, index)
                        existing_item = collection.get(res_name)
                        if existing_item is None:
                            return None, res_name
                        return existing_item, None
                    else:
                        index += 1
                        res_name = generate_res_name(item, suffix, index)
                        existing_item = collection.get(res_name)
                        if existing_item is None:
                            return None, res_name
                return existing_item, None
            else:
                res_name = generate_res_name(item, suffix, index)
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
