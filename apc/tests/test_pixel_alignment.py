import copy
import unittest

from apc.perception.pixel_alignment import PixelAlignment


def evidence():
    return {"schema_version": "apc_pixel_alignment_v1", "image_sha256": "fixture",
            "image_size": [600, 450], "viewport": [1000, 800],
            "measurement_method": "manual_visible_landmarks",
            "anchors": [{"description": f"visible corner {i}", "dom_xy": [x, y],
                         "pixel_xy": [x * .5 + 20, y * .5 + 10]}
                        for i, (x, y) in enumerate(((20, 20), (980, 20), (20, 780), (980, 780)))]}


class PixelAlignmentTests(unittest.TestCase):
    def construct(self, value=None):
        return PixelAlignment(value if value is not None else evidence(), image_sha256="fixture",
                              image_size=[600, 450], viewport=[1000, 800])

    def test_scale_translation_and_normalized_box(self):
        actual = self.construct().box(100, 200, 60, 80)
        for a, b in zip(actual, [70/600, 110/450, 100/600, 150/450]):
            self.assertAlmostEqual(a, b)

    def test_binding_measurement_and_landmark_failures(self):
        cases = []
        for key, value in (("image_sha256", "tampered"), ("image_size", [601, 450]),
                           ("viewport", [999, 800]), ("measurement_method", "neural_predictions"),
                           ("anchors", [])):
            item = evidence()
            item[key] = value
            cases.append(item)
        item = evidence()
        item["anchors"][0]["pixel_xy"][0] += 30
        cases.append(item)
        item = evidence()
        item["anchors"][0]["dom_xy"][0] = True
        cases.append(item)
        item = evidence()
        item["anchors"][0]["pixel_xy"][1] = float("nan")
        cases.append(item)
        item = evidence()
        for anchor in item["anchors"]:
            anchor["dom_xy"][1] = 20
        cases.append(item)
        item = evidence()
        for anchor in item["anchors"]:
            anchor["pixel_xy"][1] = anchor["dom_xy"][1] * .4 + 10
        cases.append(item)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.construct(case)

    def test_outside_coverage_and_image_rejected(self):
        alignment = self.construct()
        for box in ((0, 200, 60, 80), (100, 200, -1, 80), (100, 750, 60, 80)):
            with self.assertRaises(ValueError):
                alignment.box(*box)
        # A clipped raster can have a valid local transform; missing cards must
        # still be rejected instead of squeezed into the image by width scaling.
        clipped = copy.deepcopy(evidence())
        clipped["anchors"] = [{"description": str(i), "dom_xy": [x, y], "pixel_xy": [x, y]}
                              for i, (x, y) in enumerate(((20, 20), (590, 20), (20, 440), (590, 440)))]
        cropped = self.construct(clipped)
        with self.assertRaises(ValueError):
            cropped.box(605, 200, 62, 78)
