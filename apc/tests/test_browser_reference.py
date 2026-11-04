import unittest
from apc.perception.evaluate_browser_reference import provisional_targets


class BrowserReferenceTests(unittest.TestCase):
    def test_normalization_uses_dom_not_screenshot_size(self):
        labels = {"viewport": {"width": 1280, "height": 720}, "hero": [],
                  "board": [{"text": "10\n♠", "box": {"x":640,"y":360,"width":64,"height":72}}]}
        target, = provisional_targets(labels)
        self.assertEqual(target["identity"], "Ts")
        self.assertEqual(target["box_xyxy"], [.5,.5,.55,.6])
        labels["board"][0]["box"]["x"] = 1281
        with self.assertRaises(ValueError):
            provisional_targets(labels)
