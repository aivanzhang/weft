"""Scene-contract tests use a tiny analytic rig, not a completed avatar."""

import copy
import json
import math
import unittest
from pathlib import Path

from avatar_assets.scene import reskin, validate_scene


def analytic_scene():
    return {
        "schema_version": 1,
        "id": "analytic-rig",
        "subject": "Two-part analytic rig for contract validation",
        "style": "Analytic fixture, not generated artwork",
        "prompt": "Exercise the scene contract without rendering an avatar.",
        "authored_assumptions": ["All geometry and joints are test-only assumptions."],
        "materials": {"body": {"color": [0.2, 0.4, 0.6, 1], "roughness": 0.5, "metallic": 0}},
        "bones": [
            {"name": "root", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]},
            {"name": "head", "parent": "root", "head": [0, 0, 1], "tail": [0, 0, 2]},
        ],
        "parts": [
            {"name": "body", "shape": "sphere", "location": [0, 0, 0.5],
             "scale": [0.3, 0.3, 0.5], "material": "body", "bone": "root"},
            {"name": "head", "shape": "sphere", "location": [0, 0, 1.4],
             "scale": [0.3, 0.3, 0.3], "material": "body", "bone": "head"},
        ],
        "states": [{
            "id": "look", "meaning": "Head turns and returns", "loop": True, "duration": 2,
            "keyframes": [
                {"frame": 0, "bones": {"head": {"rotation": [0, 0, 0]}}},
                {"frame": 24, "bones": {"head": {"rotation": [0, 0, 0.3]}}},
                {"frame": 48, "bones": {"head": {"rotation": [0, 0, 0]}}},
            ],
        }],
        "attachments": [],
        "camera": {"target": [0, 0, 1], "position": [3, -6, 3], "orthographic_scale": 3},
        "background": [0.95, 0.95, 0.95, 1],
    }


class SceneContractTests(unittest.TestCase):
    def reject(self, mutate):
        scene = analytic_scene()
        mutate(scene)
        with self.assertRaises(ValueError):
            validate_scene(scene)

    def test_valid_scene_is_copied_without_mutating_editable_source(self):
        source = analytic_scene()
        original = copy.deepcopy(source)
        validated = validate_scene(source)
        self.assertEqual(source, original)
        validated["parts"][0]["scale"][0] = 9
        self.assertEqual(source, original)

    def test_authored_examples_pass_actual_scene_contract(self):
        examples = Path(__file__).resolve().parents[1] / "examples/scenes"
        for name in ("fox-craftsperson", "articulated-lamp", "panda-shoulder-dragon"):
            with self.subTest(name=name):
                source = json.loads((examples / (name + ".json")).read_text())
                validated = validate_scene(source)
                self.assertEqual(source, validated)

    def test_subject_style_prompt_and_assumptions_are_required(self):
        for key in ("subject", "style", "prompt", "authored_assumptions"):
            with self.subTest(key=key):
                self.reject(lambda s, key=key: s.update({key: [] if key == "authored_assumptions" else ""}))

    def test_unsafe_scene_id_rejected(self):
        self.reject(lambda s: s.update(id="../escaped"))

    def test_duplicate_bones_and_parts_rejected(self):
        self.reject(lambda s: s["bones"].append(copy.deepcopy(s["bones"][1])))
        self.reject(lambda s: s["parts"].append(copy.deepcopy(s["parts"][1])))

    def test_cycles_forward_parents_and_multiple_roots_rejected(self):
        self.reject(lambda s: s["bones"][0].update(parent="head"))
        self.reject(lambda s: s["bones"][1].update(parent="head"))
        self.reject(lambda s: s["bones"][1].update(parent=None))

    def test_zero_length_and_nonfinite_bones_rejected(self):
        self.reject(lambda s: s["bones"][0].update(tail=[0, 0, 0]))
        self.reject(lambda s: s["bones"][0].update(head=[math.nan, 0, 0]))

    def test_geometry_count_and_skeleton_count_limits(self):
        self.reject(lambda s: s.update(parts=[{**s["parts"][0], "name": f"part-{i}"} for i in range(65)]))
        self.reject(lambda s: s.update(bones=[{**s["bones"][0], "name": f"bone-{i}"} for i in range(33)]))

    def test_material_count_limit(self):
        self.reject(lambda s: s.update(materials={f"mat-{i}": {"color": [1, 1, 1, 1]} for i in range(17)}))

    def test_unknown_geometry_material_and_bone_rejected(self):
        self.reject(lambda s: s["parts"][0].update(shape="python-script"))
        self.reject(lambda s: s["parts"][0].update(material="missing"))
        self.reject(lambda s: s["parts"][0].update(bone="missing"))

    def test_negative_tiny_and_out_of_bounds_geometry_rejected(self):
        for value in ([-1, 1, 1], [0.001, 1, 1], [21, 1, 1]):
            with self.subTest(scale=value):
                self.reject(lambda s, v=value: s["parts"][0].update(scale=v))
        self.reject(lambda s: s["parts"][0].update(location=[0, 0, math.inf]))

    def test_material_channels_must_be_valid_opaque_numbers(self):
        for value in ([1, 0, 0, 0.5], [-0.1, 0, 0, 1], [True, 0, 0, 1]):
            with self.subTest(color=value):
                self.reject(lambda s, v=value: s["materials"]["body"].update(color=v))
        self.reject(lambda s: s["materials"]["body"].update(metallic=2))

    def test_state_ids_meaning_loop_and_count_are_enforced(self):
        self.reject(lambda s: s["states"].append(copy.deepcopy(s["states"][0])))
        self.reject(lambda s: s["states"][0].update(meaning=""))
        self.reject(lambda s: s["states"][0].update(loop="yes"))
        self.reject(lambda s: s.update(states=[{**s["states"][0], "id": f"state-{i}"} for i in range(13)]))

    def test_durations_are_bounded_and_frame_aligned(self):
        for duration in (0, -1, 21, 0.51, True, "four", math.nan):
            with self.subTest(duration=duration):
                self.reject(lambda s, v=duration: s["states"][0].update(duration=v))

    def test_keyframes_must_span_clip_monotonically(self):
        self.reject(lambda s: s["states"][0]["keyframes"][0].update(frame=1))
        self.reject(lambda s: s["states"][0]["keyframes"][1].update(frame=0))
        self.reject(lambda s: s["states"][0]["keyframes"][-1].update(frame=47))
        self.reject(lambda s: s["states"][0]["keyframes"][-1].update(frame=49))

    def test_keyframe_count_is_bounded(self):
        self.reject(lambda s: s["states"][0].update(keyframes=[{"frame": i, "bones": {}} for i in range(65)]))

    def test_unknown_keyed_bones_and_transforms_rejected(self):
        self.reject(lambda s: s["states"][0]["keyframes"][1].update(bones={"unknown": {"rotation": [0, 0, 0]}}))
        self.reject(lambda s: s["states"][0]["keyframes"][1].update(bones={"head": {"execute": "script.py"}}))
        self.reject(lambda s: s["states"][0]["keyframes"][1].update(bones={"head": {"location": [0, 0, 6]}}))

    def test_camera_cannot_coincide_with_target(self):
        self.reject(lambda s: s["camera"].update(position=s["camera"]["target"]))

    def test_camera_scale_is_a_finite_number_not_bool(self):
        self.reject(lambda s: s["camera"].update(orthographic_scale=True))
        self.reject(lambda s: s["camera"].update(orthographic_scale=math.inf))

    def test_background_channels_cannot_be_negative(self):
        self.reject(lambda s: s.update(background=[-0.2, 0.9, 0.9, 1]))

    def test_attachment_requires_real_parent_child_relationship(self):
        self.reject(lambda s: s.update(attachments=[{
            "name": "bad-attachment", "parent_bone": "head", "child_bone": "missing",
            "placement": "above head", "authored_assumption": "Test only",
        }]))
        self.reject(lambda s: s.update(attachments=[{
            "name": "bad-parent", "parent_bone": "head", "child_bone": "root",
            "placement": "above head", "authored_assumption": "Test only",
        }]))


class SkinInvariantTests(unittest.TestCase):
    def test_reskin_changes_only_named_materials(self):
        source = analytic_scene()
        original = copy.deepcopy(source)
        replacement = {"color": [0.6, 0.1, 0.3, 1], "roughness": 0.8, "metallic": 0.2}
        result = reskin(source, {"body": replacement})
        self.assertEqual(source, original)
        self.assertEqual(replacement, result["materials"]["body"])
        for key in source.keys() - {"materials"}:
            self.assertEqual(source[key], result[key], key)

    def test_reskin_cannot_add_materials_or_invalid_properties(self):
        with self.assertRaises(ValueError):
            reskin(analytic_scene(), {"unknown": {"color": [1, 0, 0, 1]}})
        with self.assertRaises(ValueError):
            reskin(analytic_scene(), {"body": {"color": [1, 0, 0, 0.2]}})


if __name__ == "__main__":
    unittest.main()
