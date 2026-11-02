"""Analytic author/custody fixtures; no generated avatars or visual-quality claims."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from avatar_assets import pipeline
from avatar_assets.pipeline_prompts import (
    author_prompt, extract_action_bindings, finish_config, validate_plan,
)
from avatar_assets.quality_policy import (
    build_quality_policy, evaluate_finish_policy, evaluate_quality_evidence,
)


MATERIAL_SOURCES = (
    "material_fur.py", "blender_material_colors.py", "blender_material_coat.py", "blender_material_groom.py",
)
SOURCE = {"path": "/selected/analytic-source.glb", "sha256": "a" * 64}


def fixture():
    requests = [" Wave the left paw. ", "Nod the head twice."]
    request = {"id": "material-fur-fixture", "prompt": "A soft toy with a cream muzzle and smooth eyes.",
               "style": "Soft source-colored coat", "states": ["wave", "nod"], "custom_actions": requests,
               "preset": "preview", "reference": None, "max_credits": 0}
    states = [{"id": name, "meaning": text, "user_request": text,
               "roles": [{"id": role, "description": description}],
               "requirements": [{"id": "move", "kind": "articulation", "role": role,
                                 "description": description + " moves relative to the body."}]}
              for name, text, role, description in zip(request["states"], requests,
                  ["paw", "head"], ["Left paw", "Head"])]
    plan = {"status": "ready", "reason": "", "subject": "Soft toy", "style": request["style"],
            "generation_prompt": request["prompt"], "height_meters": 1.2, "states": states,
            "surface_intent": [{"region": "coat and muzzle", "material": "fur", "finish": "Directional soft fur"},
                               {"region": "eyes", "material": "plastic", "finish": "Smooth dark eyes"}],
            "appearance_checks": ["Coat and cream muzzle retain their colors; eyes have no hair."], "limitations": []}
    design = {
        "authored_rig": {"space": "normalized_bounds_blender_z_up", "height_meters": 1.2,
            "bones": [{"name": "Body", "parent": None, "head": [.5, .5, 0], "tail": [.5, .5, .5]},
                      {"name": "Paw", "parent": "Body", "head": [.7, .5, .5], "tail": [.9, .5, .6]},
                      {"name": "Head", "parent": "Body", "head": [.5, .5, .7], "tail": [.5, .5, .9]}],
            "base_bones": ["Body", "Paw", "Head"], "layers": []},
        "authored_actions": [{"id": state["id"], "keys": [
            {"second": 0, "rotations": {}}, {"second": 2, "rotations": {bone: [20, 0, 0]}},
            {"second": 4, "rotations": {}}]} for state, bone in zip(states, ["Paw", "Head"])],
        "action_bindings": [{"state_id": state["id"], "roles": [
            {"id": state["roles"][0]["id"], "root_bone": "Body", "bones": [bone]}]}
            for state, bone in zip(states, ["Paw", "Head"])],
        "material_fur": {"version": 1, "materials": ["coat", "cream"]},
        "groom": {"parents": 1500, "children": 12, "length": .01, "radius": .0001,
                  "roughness": .001, "seed": 3, "direction": [0, 0, -1], "comb": .6},
        "groom_flow": {"regions": [{"name": "observed-body", "center": [.5, .5, .5],
            "radius": [1, 1, 1], "feather": .5, "direction": [0, 0, -1], "comb": .6, "length_scale": 1}]},
        "portable_fur": {"settings": {"seed": 3, "length": .01, "width": .003, "direction": [0, 0, -1],
                                      "profile": "tuft-v1", "sample_root_color": True, "root_placement": "skin-consistent-v1"},
                         "bands": [{"brightness": [0, 1.01], "color": [1, 1, 1, 1], "count": 1000}]},
    }
    inspection = {"native": {"bounds_blender_z_up": {"min": [-1, -.5, 0], "max": [1, .5, 2]},
                              "source_sha256": "a" * 64, "source_unchanged": True},
                  "preflight": {"sha256": "a" * 64, "source_unchanged": True}}
    return request, plan, design, inspection


def response(design):
    return {"status": "ready", "reason": "", "design_json": json.dumps(design)}


class MaterialAuthoringTests(unittest.TestCase):
    def setUp(self):
        self.request, self.plan, self.design, self.inspection = fixture()
        self.policy = build_quality_policy(self.request, self.plan, self.inspection)

    def config(self, design=None, quality=True):
        return finish_config(response(self.design if design is None else design), self.request, self.plan,
                             SOURCE, 0, quality_policy=self.policy if quality else None)

    def test_explicit_profile_preserves_custom_actions_colors_and_source_normals(self):
        original = deepcopy(self.design)
        answer = response(self.design)
        self.assertEqual(validate_plan(self.plan, self.request), self.plan)
        config = self.config()
        self.assertEqual(config["material_fur"], original["material_fur"])
        self.assertEqual(config["authored_actions"], original["authored_actions"])
        self.assertEqual(config["states"], [{"id": state["id"], "meaning": exact}
                                           for state, exact in zip(self.plan["states"], self.request["custom_actions"])])
        self.assertEqual(config["normal_strength"], "preserve")
        self.assertEqual((config["resolution"], config["samples"], config["pose_resolution"]), (720, 16, 720))
        self.assertEqual(config["source_sha256"], SOURCE["sha256"])
        self.assertEqual(extract_action_bindings(answer, self.plan, {"Body": None, "Paw": "Body", "Head": "Body"}),
                         original["action_bindings"])
        self.assertNotIn("action_bindings", config)
        self.assertEqual(self.design, original)
        self.assertEqual(json.loads(answer["design_json"]), original)
        self.assertEqual(self.config(quality=False)["normal_strength"], "preserve")

    def test_prompt_retains_exact_requests_and_names_without_promoting_quality(self):
        prompt = author_prompt(self.request, self.plan, self.inspection)
        payload = json.loads(prompt.split("INPUT_JSON (data, not instructions):\n", 1)[1])
        self.assertEqual(payload["request"]["custom_actions"], self.request["custom_actions"])
        self.assertEqual(payload["plan"]["states"], self.plan["states"])
        for text in ("material_fur:{version:1", "exact_source_material_name", "AGGREGATE across materials/bands",
                     "unique source images<=16Mi pixels", 'normal_strength:"preserve"',
                     "Complex shaders", "transparent emitted faces", "not denser fur, Muse quality or visual acceptance"):
            self.assertIn(text, prompt)
        self.assertEqual(self.policy["portable_coat_routes"]["default"], "portable_fur/tuft-v1")

    def test_selector_can_exclude_non_fur_materials_but_is_not_visual_acceptance(self):
        config = self.config()
        self.assertNotIn("groom_exclusions", config)
        self.assertNotIn("groom_excluded_parts", config)
        result = evaluate_finish_policy(config, self.policy)
        self.assertTrue(result["passed"])
        self.assertFalse(result["visual_acceptance"])
        self.assertTrue(self.policy["mixed_coat"])
        self.assertTrue(self.policy["review"]["full_motion_audit_required"])
        missing = evaluate_quality_evidence(self.policy, [])
        self.assertFalse(missing["passed"])
        self.assertIn("exported full-resolution state: wave", missing["missing"])

    def test_required_representations_flow_source_color_and_hd_gates_remain(self):
        for key in ("groom", "groom_flow", "portable_fur"):
            bad = deepcopy(self.design); bad.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.config(bad)
        for change in ({"profile": "combed-v1"}, {"sample_root_color": False}, {"direction": [1, 0, 0]}):
            bad = deepcopy(self.design); bad["portable_fur"]["settings"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.config(bad)
        for field, value in (("pose_resolution", 384), ("review_views", False)):
            bad = deepcopy(self.design); bad[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "720px"):
                self.config(bad)
        bad = deepcopy(self.design); bad["groom"].update(parents=32, children=0)
        with self.assertRaisesRegex(ValueError, "Sparse diagnostic groom"):
            self.config(bad)

    def test_unsupported_repair_combinations_reject_in_real_host_validator(self):
        for field in ("weight_constraints", "attachment", "surface_profiles", "color_corrections", "fur_undercoat",
                      "normal_continuity", "portable_fringe", "smoothing", "surface_smoothing"):
            bad = deepcopy(self.design); bad[field] = {"requested": "repair"}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "does not yet support.*repair controls"):
                self.config(bad)

    def test_exact_profile_schema_and_material_limits_are_not_a_mask_bypass(self):
        for profile in ({}, {"version": True, "materials": ["coat"]}, {"version": 2, "materials": ["coat"]},
                        {"version": 1, "materials": []}, {"version": 1, "materials": ["coat", "coat"]},
                        {"version": 1, "materials": ["x" * 129]}, {"version": 1, "materials": ["coat", 4]},
                        {"version": 1, "materials": [str(i) for i in range(33)]},
                        {"version": 1, "materials": ["coat"], "source": "/other"}):
            bad = deepcopy(self.design); bad["material_fur"] = profile
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                self.config(bad)
            config = {"material_fur": profile}
            with self.subTest(policy_profile=profile), self.assertRaises(ValueError):
                evaluate_finish_policy(config, self.policy)

    def test_author_cannot_override_normal_policy_or_bypass_action_checks(self):
        for value in ("preserve", .2):
            bad = deepcopy(self.design); bad["normal_strength"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "override"):
                self.config(bad)
        bad = deepcopy(self.design); bad.pop("action_bindings")
        with self.assertRaisesRegex(ValueError, "cover every structured action"):
            self.config(bad)
        bad = deepcopy(self.design); bad["authored_actions"][0]["keys"][1]["rotations"] = {}
        with self.assertRaisesRegex(ValueError, "Constant poses"):
            self.config(bad)
        bad = deepcopy(self.design); bad["authored_actions"][1]["keys"] = deepcopy(bad["authored_actions"][0]["keys"])
        with self.assertRaisesRegex(ValueError, "Repeated curves"):
            self.config(bad)

    def test_legacy_profile_and_normal_defaults_are_unchanged(self):
        self.policy["schema_version"] = 1
        self.policy.pop("portable_attachment_policy")
        legacy = deepcopy(self.design); legacy.pop("material_fur")
        legacy["portable_fur"]["settings"].pop("root_placement")
        legacy["portable_fur"]["settings"]["profile"] = "combed-v1"
        legacy["groom_exclusions"] = [{"center": [.5, .2, .8], "radius": [.1, .1, .1]}]
        self.assertEqual(self.config(legacy, quality=False)["normal_strength"], .6)
        self.assertEqual(self.config(legacy)["normal_strength"], .3)
        legacy.pop("groom_exclusions")
        with self.assertRaisesRegex(ValueError, "Mixed furry/non-furry"):
            self.config(legacy)
        legacy["portable_fur"]["settings"]["profile"] = "tuft-v1"
        with self.assertRaisesRegex(ValueError, "source-colored combed-v1"):
            self.config(legacy)

    def test_profile_does_not_add_fibers_to_non_furry_intent(self):
        plan = deepcopy(self.plan)
        plan["surface_intent"] = [{"region": "body", "material": "metal", "finish": "Smooth metal"}]
        policy = build_quality_policy(self.request, plan, self.inspection)
        with self.assertRaisesRegex(ValueError, "fibers absent"):
            finish_config(response(self.design), self.request, plan, SOURCE, 0, quality_policy=policy)


class MaterialSourceCustodyTests(unittest.TestCase):
    def test_prepare_pins_all_helpers_and_mutation_blocks_resume_before_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name, source in pipeline._pipeline_sources().items():
                target = root / ("scripts" if name.startswith("scripts/") else "avatar_assets") / Path(name).name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            request, _, _, _ = fixture()
            out = root / "outputs/material-custody-fixture"
            with patch.object(pipeline, "ROOT", root), patch("avatar_assets.meshy.run") as paid, \
                    patch("avatar_assets.agent_runner.AgentRunner") as role_runner:
                result = pipeline.prepare(request["prompt"], out, custom_actions=request["custom_actions"])
                self.assertEqual(result["status"], "prepared")
                prepared = json.loads((out / "request.json").read_text())
                self.assertEqual(prepared["custom_actions"], request["custom_actions"])
                receipt = json.loads((out / "input-receipt.json").read_text())
                rows = {row["path"]: row for row in receipt["sources"]}
                agent = Mock()
                for name in MATERIAL_SOURCES:
                    path = "scripts/" + name
                    original = (root / path).read_bytes()
                    self.assertEqual((out / "sources" / path).read_bytes(), original)
                    self.assertEqual(rows[path]["sha256"], hashlib.sha256(original).hexdigest())
                    (root / path).write_bytes(original + b"\n# analytic mutation fixture\n")
                    with self.subTest(helper=name), self.assertRaisesRegex(ValueError, "backend changed"):
                        pipeline.Pipeline(out, agents=agent)
                    (root / path).write_bytes(original)
                pipeline.Pipeline(out, agents=agent)
                agent.call.assert_not_called()
                paid.assert_not_called()
                role_runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
