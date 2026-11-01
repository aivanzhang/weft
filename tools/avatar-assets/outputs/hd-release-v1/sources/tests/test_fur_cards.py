"""Analytic skinning, atlas and budget checks; not visual acceptance of a character."""

from copy import deepcopy
import unittest

import numpy as np

from avatar_assets.scene import validate_scene
from scripts.blender_fur_cards import ATLAS_SIZE, card_geometry, atlas_pixels
from tests.test_scene_quality import quality_scene


class FurCardTests(unittest.TestCase):
    def test_curved_cards_retain_interpolated_skin_and_in_bounds_uv(self):
        settings = {"count": 60, "length": .02, "width": .025, "seed": 4, "mode": "cards", "direction": [1, 0, 0]}
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        weights = [{"root": 1}, {"tip": 1}, {"root": .5, "tip": .5}]
        result = card_geometry(vertices, [(0, 1, 2)], weights, settings)
        self.assertEqual(result, card_geometry(vertices, [(0, 1, 2)], weights, settings))
        self.assertEqual((len(result["vertices"]), len(result["faces"])), (360, 240))
        for index in range(0, 360, 6):
            root = np.mean(result["vertices"][index:index+2], axis=0)
            middle = np.mean(result["vertices"][index+2:index+4], axis=0)
            tip = np.mean(result["vertices"][index+4:index+6], axis=0)
            self.assertAlmostEqual(root[2], -.0005)
            self.assertGreater(tip[2], middle[2])
            self.assertGreater(tip[0]-middle[0], middle[0]-root[0])
            inherited = result["weights"][index]
            self.assertAlmostEqual(inherited["tip"], root[0]+root[1]*.5)
            self.assertAlmostEqual(sum(inherited.values()), 1)
            self.assertEqual(result["weights"][index:index+6], [inherited]*6)
        self.assertTrue(all(0 < value < 1 for uv in result["uv"] for value in uv))
        self.assertTrue(all(abs(sum(v*v for v in normal)-1) < 1e-7 for normal in result["normals"]))
        self.assertNotEqual(result["vertices"], card_geometry(vertices, [(0, 1, 2)], weights, {**settings, "seed": 5})["vertices"])

    def test_cutout_atlas_is_fine_transparent_and_seeded(self):
        pixels = atlas_pixels([.8, .6, .4, 1])
        np.testing.assert_array_equal(pixels, atlas_pixels([.8, .6, .4, 1]))
        self.assertEqual(pixels.shape, (ATLAS_SIZE, ATLAS_SIZE, 4))
        self.assertEqual(pixels.dtype, np.float32)
        self.assertTrue(np.isfinite(pixels).all())
        self.assertTrue((pixels >= 0).all() and (pixels <= 1).all())
        coverage = np.mean(pixels[:, :, 3] > .45)
        self.assertGreater(coverage, .05)
        self.assertLess(coverage, .3)
        self.assertTrue((pixels[::256, :, 3] == 0).all())
        self.assertFalse(np.array_equal(pixels[:256, :256], pixels[:256, 256:512]))

    def test_card_vertex_and_texture_aggregate_limits(self):
        scene = quality_scene()
        part = scene["parts"][0]
        part["fur"].update(mode="cards", width=.025, count=6000)
        scene["parts"].append({**deepcopy(part), "name": "other-card-surface"})
        validate_scene(scene)
        scene["parts"][-1]["fur"]["count"] += 1
        with self.assertRaisesRegex(ValueError, "fur vertex"):
            validate_scene(scene)
        scene = quality_scene()
        scene["materials"]["body"]["surface"]["resolution"] = 2048
        for i in range(4):
            scene["materials"][f"hd-{i}"] = deepcopy(scene["materials"]["body"])
        with self.assertRaisesRegex(ValueError, "texture pixel"):
            validate_scene(scene)

    def test_native_groom_budgets_cannot_be_bypassed_by_many_emitters(self):
        scene = quality_scene()
        groom = {"parents": 7500, "children": 15, "length": .02, "radius": .0001, "roughness": .001, "seed": 7}
        scene["parts"][0]["native_groom"] = deepcopy(groom)
        scene["parts"][1]["native_groom"] = deepcopy(groom)
        validate_scene(scene)
        scene["parts"][1]["native_groom"]["parents"] += 1
        with self.assertRaisesRegex(ValueError, "native groom limit"):
            validate_scene(scene)
        for key, value in (("parents", True), ("children", 31), ("length", .1), ("radius", 0), ("roughness", float("nan")), ("seed", -1)):
            bad = quality_scene()
            bad["parts"][0]["native_groom"] = {**groom, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_scene(bad)

    def test_new_options_do_not_pass_through_v1_or_accept_unbounded_values(self):
        for field, value in (("mode", "python"), ("direction", [0, 0, 0]), ("width", .041), ("count", 10001)):
            scene = quality_scene()
            scene["parts"][0]["fur"].update({"mode": "cards", "width": .025, field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_scene(scene)


if __name__ == "__main__":
    unittest.main()
