import supervisely_lib as sly
from supervisely_lib.collection.key_indexed_collection import KeyIndexedCollection

# def _find_item(collection, name):
#     free_name = name
#     item = collection.get(free_name)
#     if item is not None:
#         free_name = f"{name}-{suffix}"
#         item = collection.get(free_name)
#     iter = 1
#     while item is not None:
#         free_name = f"{name}-{suffix}-{iter}"
#         item = collection.get(free_name)
#         iter += 1
#     return free_name


# def _compare_tag(res_meta: sly.ProjectMeta, tag: sly.Tag, new_tags: List):
#     if tag.meta.name in tag_mapping:
#         new_tags.append(tag.clone(meta=tag_mapping[tag.meta.name]))
#         return
#     original_tag_meta = res_meta.tag_metas.get(tag.meta.name)
#     if original_tag_meta is None:
#         res_meta = res_meta.add_tag_meta(tag.meta)
#         new_tags.append(tag)
#     elif original_tag_meta != tag.meta:  # conflict
#         new_tag_name = _find_free_name(res_meta.tag_metas, tag.meta.name)
#         new_tag_meta = tag.meta.clone(name=new_tag_name)
#         tag_mapping[new_tag_name] = new_tag_meta
#         res_meta = res_meta.add_tag_meta(new_tag_meta)
#         new_tags.append(tag.clone(meta=new_tag_meta))
#     else:
#         new_tags.append(tag)
#     return res_meta


def find_item(collection: KeyIndexedCollection, item, suffix):
    index = 0
    res_name = item.name
    while True:
        existing_item = collection.get(res_name)
        if existing_item is None:
            return None, res_name
        else:
            if existing_item == item.clone(name=res_name):
                return existing_item, None
            else:
                res_name = f"{item.name}-{suffix}" if index == 0 else f"{item.name}-{suffix}-{index}"
                index += 1


def merge_metas(project_meta: sly.ProjectMeta, model_meta: sly.ProjectMeta,
                keep_model_classes, keep_model_tags, suffix):
    res_meta = project_meta.clone()

    def _merge(keep_names, res_meta, project_collection, model_collection, add_method):
        mapping = {}  # old name to new meta
        for name in keep_names:
            model_item = model_collection.get(name)
            res_item, res_name = find_item(project_collection, model_item, suffix)
            if res_item is None:
                res_item = res_item.clone(name=res_name)
                mapping[model_item.name] = res_item
                res_meta = add_method(res_item)
        return res_meta, mapping

    res_meta, class_mapping = _merge(keep_model_classes, res_meta, res_meta.obj_classes, model_meta.obj_classes, res_meta.add_obj_class)
    res_meta, tag_mapping = _merge(keep_model_tags, res_meta, res_meta.tag_metas, model_meta.tag_metas, res_meta.add_tag_meta)
    return res_meta, class_mapping, tag_mapping
