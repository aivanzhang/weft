"""Data-only comb integration checks; native deformation proof is separate."""

import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from avatar_assets.build3d import ROOT, _build_3d
from avatar_assets.scene import validate_scene
from tests.test_scene_quality import quality_scene


BASE = {"parents": 200, "children": 10, "length": .02,
        "radius": .0003, "roughness": .001, "seed": 9}


class PreparedWithoutLaunching(Exception):
    pass


class NativeCombSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the actual pure validator without requiring Blender's Python runtime.
        path = ROOT / "scripts/blender_native_groom.py"
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("_require", "validate_groom_settings")
                 or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "MAX_FIBERS"
                                                         for target in node.targets)]
        namespace = {"math": math}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        cls.helper_validate = staticmethod(namespace["validate_groom_settings"])

    def validate_both(self, groom):
        self.assertEqual(self.helper_validate(groom), groom)
        scene = quality_scene()
        scene["parts"][0]["native_groom"] = deepcopy(groom)
        self.assertEqual(validate_scene(scene)["parts"][0]["native_groom"], groom)

    def reject_both(self, groom):
        with self.assertRaises(ValueError):
            self.helper_validate(groom)
        scene = quality_scene()
        scene["parts"][0]["native_groom"] = deepcopy(groom)
        with self.assertRaises(ValueError):
            validate_scene(scene)

    def test_legacy_radial_settings_remain_exact_without_implicit_comb(self):
        original = deepcopy(BASE)
        self.validate_both(BASE)
        self.assertEqual(BASE, original)
        self.assertNotIn("direction", self.helper_validate(BASE))
        self.assertNotIn("comb", self.helper_validate(BASE))

    def test_paired_direction_and_comb_accept_inclusive_bounds_without_rewriting_input(self):
        for direction, amount in (([0, 0, -1], .65), ([-1, 1, 1], .95), ([1, 0, 0], 0), ([0, .000001, 0], .5)):
            with self.subTest(direction=direction, amount=amount):
                self.validate_both(dict(BASE, direction=direction, comb=amount))

    def test_partial_pairs_and_unknown_fields_are_rejected(self):
        for extra in ({"direction": [0, 0, -1]}, {"comb": .65}, {"unknown": 1},
                      {"direction": [0, 0, -1], "comb": .65, "script": "not executable input"}):
            with self.subTest(extra=extra):
                self.reject_both(dict(BASE, **extra))

    def test_invalid_directions_and_comb_values_are_rejected_by_both_layers(self):
        for direction in ([0, 0, 0], [1e-10, 0, 0], [1.01, 0, 0], [0, -1.01, 0],
                          [True, 0, 0], [float("nan"), 0, 0], [0, float("inf"), 0],
                          [1, 0], [0, 0, 1, 0], (0, 0, -1), "down", None):
            with self.subTest(direction=direction):
                self.reject_both(dict(BASE, direction=direction, comb=.65))
        for amount in (-.01, .950001, float("nan"), float("inf"), True, "0.65", None):
            with self.subTest(amount=amount):
                self.reject_both(dict(BASE, direction=[0, 0, -1], comb=amount))

    def test_combing_does_not_expand_per_part_or_aggregate_fiber_budgets(self):
        groom = dict(BASE, parents=7500, children=15, direction=[0, 0, -1], comb=.65)
        scene = quality_scene()
        scene["parts"][0]["native_groom"] = deepcopy(groom)
        scene["parts"][1]["native_groom"] = deepcopy(groom)
        validate_scene(scene)
        scene["parts"][1]["native_groom"]["parents"] += 1
        with self.assertRaisesRegex(ValueError, "Aggregate native groom limit"):
            validate_scene(scene)
        self.reject_both(dict(groom, parents=10001, children=0))
        self.reject_both(dict(groom, parents=10000, children=24))

    def test_build_preparation_snapshots_exact_comb_helper_without_native_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = quality_scene()
            scene["parts"][0]["native_groom"] = dict(BASE, direction=[0, 0, -1], comb=.65)
            config, executable = root / "scene.json", root / "never-executed-blender"
            config.write_text(json.dumps(scene))
            executable.write_text("analytic fixture; never executed")
            with patch("avatar_assets.build3d.run_bounded", side_effect=PreparedWithoutLaunching) as run:
                with self.assertRaises(PreparedWithoutLaunching):
                    _build_3d(config, root / "new-build", executable, 384, 8, 600)
            run.assert_called_once()
            self.assertEqual(run.call_args.kwargs["timeout"], 600)
            self.assertEqual(run.call_args.kwargs["memory_mb"], 2048)
            copied = root / "new-build/sources/blender_comb_groom.py"
            self.assertEqual(copied.read_bytes(), (ROOT / "scripts/blender_comb_groom.py").read_bytes())
            preserved = json.loads((root / "new-build/sources/scene.json").read_text())
            self.assertEqual(preserved["parts"][0]["native_groom"], scene["parts"][0]["native_groom"])

    def test_comb_wall_cap_is_opt_in_and_checked_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene, config, executable = quality_scene(), root / "scene.json", root / "never-executed-blender"
            executable.write_text("analytic fixture; never executed")
            scene["parts"][0]["native_groom"] = dict(BASE, direction=[0, 0, -1], comb=.65)
            config.write_text(json.dumps(scene))
            with self.assertRaisesRegex(ValueError, "at most 600 seconds"):
                _build_3d(config, root / "rejected", executable, 384, 8, 601)
            self.assertFalse((root / "rejected").exists())
            scene["parts"][0]["native_groom"] = dict(BASE)
            config.write_text(json.dumps(scene))
            with patch("avatar_assets.build3d.run_bounded", side_effect=PreparedWithoutLaunching):
                with self.assertRaises(PreparedWithoutLaunching):
                    _build_3d(config, root / "legacy", executable, 384, 8, 601)


if __name__ == "__main__":
    unittest.main()
