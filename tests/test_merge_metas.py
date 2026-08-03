import unittest

import supervisely as sly

from shared_utils.merge_metas import merge_metas


class MergeMetasTestCase(unittest.TestCase):
    def test_reuses_same_name_and_geometry(self):
        project_class = sly.ObjClass("Cat", sly.Bitmap, [1, 2, 3])
        model_class = sly.ObjClass("Cat", sly.Bitmap, [200, 201, 202])

        result_meta, class_mapping, _ = merge_metas(
            sly.ProjectMeta(obj_classes=[project_class]),
            sly.ProjectMeta(obj_classes=[model_class]),
            ["Cat"],
            [],
            "model",
        )

        self.assertEqual([obj_class.name for obj_class in result_meta.obj_classes], ["Cat"])
        self.assertIs(class_mapping["Cat"], project_class)

    def test_does_not_treat_any_geometry_as_exact_match(self):
        project_class = sly.ObjClass("Cat", sly.AnyGeometry)
        model_class = sly.ObjClass("Cat", sly.Bitmap)

        result_meta, class_mapping, _ = merge_metas(
            sly.ProjectMeta(obj_classes=[project_class]),
            sly.ProjectMeta(obj_classes=[model_class]),
            ["Cat"],
            [],
            "model",
        )

        self.assertEqual(
            [obj_class.name for obj_class in result_meta.obj_classes],
            ["Cat", "Cat-model"],
        )
        self.assertEqual(class_mapping["Cat"].geometry_type, sly.Bitmap)

    def test_creates_suffixed_class_for_geometry_conflict(self):
        project_class = sly.ObjClass("Cat", sly.Polygon)
        model_class = sly.ObjClass("Cat", sly.Bitmap)

        result_meta, class_mapping, _ = merge_metas(
            sly.ProjectMeta(obj_classes=[project_class]),
            sly.ProjectMeta(obj_classes=[model_class]),
            ["Cat"],
            [],
            "model",
        )

        self.assertEqual(
            [obj_class.name for obj_class in result_meta.obj_classes],
            ["Cat", "Cat-model"],
        )
        self.assertEqual(class_mapping["Cat"].geometry_type, sly.Bitmap)

    def test_skips_incompatible_existing_forced_suffix(self):
        incompatible_class = sly.ObjClass("Cat-model", sly.Polygon)
        model_class = sly.ObjClass("Cat", sly.Bitmap)

        result_meta, class_mapping, _ = merge_metas(
            sly.ProjectMeta(obj_classes=[incompatible_class]),
            sly.ProjectMeta(obj_classes=[model_class]),
            ["Cat"],
            [],
            "model",
            use_suffix=True,
        )

        self.assertEqual(
            [obj_class.name for obj_class in result_meta.obj_classes],
            ["Cat-model", "Cat-model-1"],
        )
        self.assertEqual(class_mapping["Cat"].name, "Cat-model-1")
        self.assertEqual(class_mapping["Cat"].geometry_type, sly.Bitmap)


if __name__ == "__main__":
    unittest.main()
