import unittest

import numpy as np
import supervisely as sly

from shared_utils.inference import postprocess


def make_state(model_meta, output_geometry):
    return {
        "classesInfo": model_meta.obj_classes.to_json(),
        "classes": [True] * len(model_meta.obj_classes),
        "tagsInfo": model_meta.tag_metas.to_json(),
        "tags": [True] * len(model_meta.tag_metas),
        "suffix": "model",
        "useModelSuffix": False,
        "outputGeometry": output_geometry,
    }


class OutputGeometryTestCase(unittest.TestCase):
    def test_converts_bitmap_components_and_reuses_polygon_class(self):
        model_class = sly.ObjClass("Cat", sly.Bitmap, [200, 201, 202])
        project_class = sly.ObjClass("Cat", sly.Polygon, [1, 2, 3])
        model_meta = sly.ProjectMeta(obj_classes=[model_class])
        project_meta = sly.ProjectMeta(obj_classes=[project_class])
        mask = np.zeros((12, 16), dtype=bool)
        mask[1:9, 1:9] = True
        mask[3:6, 3:6] = False
        mask[2:5, 12:15] = True
        annotation = sly.Annotation(
            img_size=mask.shape,
            labels=[sly.Label(sly.Bitmap(mask), model_class)],
        )

        result, result_meta = postprocess(
            None,
            None,
            annotation,
            project_meta,
            model_meta,
            make_state(model_meta, "polygon"),
        )

        self.assertEqual(len(result.labels), 2)
        self.assertTrue(all(isinstance(label.geometry, sly.Polygon) for label in result.labels))
        self.assertTrue(all(label.obj_class is project_class for label in result.labels))
        self.assertEqual([obj_class.name for obj_class in result_meta.obj_classes], ["Cat"])
        self.assertEqual(sum(len(label.geometry.interior) for label in result.labels), 1)

    def test_converts_rectangle_to_polygon(self):
        model_class = sly.ObjClass("Cat", sly.Rectangle)
        model_meta = sly.ProjectMeta(obj_classes=[model_class])
        annotation = sly.Annotation(
            img_size=(20, 20),
            labels=[sly.Label(sly.Rectangle(1, 2, 10, 12), model_class)],
        )

        result, result_meta = postprocess(
            None,
            None,
            annotation,
            sly.ProjectMeta(),
            model_meta,
            make_state(model_meta, "polygon"),
        )

        self.assertEqual(len(result.labels), 1)
        self.assertIsInstance(result.labels[0].geometry, sly.Polygon)
        self.assertEqual(result.labels[0].obj_class.geometry_type, sly.Polygon)
        self.assertEqual(result_meta.get_obj_class("Cat").geometry_type, sly.Polygon)

    def test_keeps_geometry_that_cannot_convert_to_polygon(self):
        model_class = sly.ObjClass("Landmark", sly.Point)
        model_meta = sly.ProjectMeta(obj_classes=[model_class])
        annotation = sly.Annotation(
            img_size=(20, 20),
            labels=[sly.Label(sly.Point(4, 5), model_class)],
        )

        result, result_meta = postprocess(
            None,
            None,
            annotation,
            sly.ProjectMeta(),
            model_meta,
            make_state(model_meta, "polygon"),
        )

        self.assertEqual(len(result.labels), 1)
        self.assertIsInstance(result.labels[0].geometry, sly.Point)
        self.assertEqual(result_meta.get_obj_class("Landmark").geometry_type, sly.Point)

    def test_model_output_is_unchanged_by_default(self):
        model_class = sly.ObjClass("Cat", sly.Bitmap)
        model_meta = sly.ProjectMeta(obj_classes=[model_class])
        mask = np.ones((5, 5), dtype=bool)
        annotation = sly.Annotation(
            img_size=mask.shape,
            labels=[sly.Label(sly.Bitmap(mask), model_class)],
        )
        state = make_state(model_meta, "model")
        state.pop("outputGeometry")

        result, result_meta = postprocess(
            None,
            None,
            annotation,
            sly.ProjectMeta(),
            model_meta,
            state,
        )

        self.assertEqual(len(result.labels), 1)
        self.assertIsInstance(result.labels[0].geometry, sly.Bitmap)
        self.assertEqual(result_meta.get_obj_class("Cat").geometry_type, sly.Bitmap)


if __name__ == "__main__":
    unittest.main()
