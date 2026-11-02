"""Policy/coverage fixtures do not certify appearance or use generated media."""

from copy import deepcopy
import unittest

from avatar_assets.quality_policy import (
    build_quality_policy, evaluate_finish_policy, evaluate_quality_evidence, validate_surface_intent,
)


def inspection():
    return {"native": {"bounds_blender_z_up": {"min": [-1, -.5, 0], "max": [1, .5, 2]},
                       "source_sha256": "a"*64, "source_unchanged": True},
            "preflight": {"sha256": "a"*64, "source_unchanged": True}}


def policy(furry=True, preset="preview"):
    surfaces = ([{"region": "body coat", "material": "fur", "finish": "Soft source-colored directional fibers"},
                 {"region": "eyes and nose", "material": "skin", "finish": "Smooth non-furry features"}]
                if furry else [{"region": "robot housing", "material": "metal", "finish": "Brushed copper panels"}])
    return build_quality_policy({"preset": preset}, {"status": "ready", "surface_intent": surfaces,
                                                      "states": [{"id": "dance"}]}, inspection())


def coat_config():
    return {"source_sha256": "a"*64, "pose_resolution": 720, "review_views": True,
            "groom": {"parents": 1500, "children": 12},
            "groom_flow": {"regions": [{"name": "observed-coat-region"}]},
            "portable_fur": {"settings": {"sample_root_color": True, "profile": "combed-v1"}},
            "groom_exclusions": [{"center": [.5, .2, .8], "radius": [.1, .1, .1]}]}


def evidence():
    rows = []
    for representation in ("native", "exported"):
        for view in ("front", "side", "back", "three_quarter"):
            rows.append({"id": representation+"-"+view, "kind": "view", "representation": representation,
                         "view": view, "width": 720, "height": 720})
        rows.append({"id": representation+"-dance", "kind": "state", "representation": representation,
                     "state_ids": ["dance"], "width": 5760, "height": 720, "pose_width": 720, "pose_height": 720})
    rows.append({"id": "portrait", "kind": "portrait", "representation": "native", "width": 1600, "height": 1600})
    return rows


class QualityPolicyTests(unittest.TestCase):
    def test_varied_materials_get_different_prerequisites_not_coordinates(self):
        fur, metal = policy(), policy(False)
        self.assertIn("regional_groom_flow", fur["required_features"])
        self.assertIn("measured_non_fur_masks", fur["required_features"])
        self.assertEqual(metal["required_features"], [])
        self.assertNotIn("groom", fur["candidate_settings"])
        self.assertEqual(fur["inspection"]["bounds_blender_z_up"], inspection()["native"]["bounds_blender_z_up"])
        self.assertEqual(fur["candidate_settings"]["pose_resolution"], 720)
        self.assertTrue(fur["candidate_settings"]["review_views"])
        config = {"source_sha256": "a"*64, "pose_resolution": 720, "review_views": True}
        self.assertTrue(evaluate_finish_policy(config, metal)["passed"])
        self.assertFalse(evaluate_finish_policy(coat_config(), metal)["passed"])

    def test_paint_only_or_unmasked_fur_cannot_pass_structural_policy(self):
        valid = coat_config()
        self.assertTrue(evaluate_finish_policy(valid, policy())["passed"])
        for field in ("groom", "portable_fur", "groom_flow", "groom_exclusions"):
            bad = deepcopy(valid); bad.pop(field)
            with self.subTest(field=field):
                self.assertFalse(evaluate_finish_policy(bad, policy())["passed"])
        for change in ({"pose_resolution": 384}, {"review_views": False}, {"source_sha256": "b"*64},
                       {"groom": {"parents": 16, "children": 0}}):
            self.assertFalse(evaluate_finish_policy({**valid, **change}, policy())["passed"])
        self.assertFalse(evaluate_finish_policy(valid, policy())["visual_acceptance"])

    def test_inspection_cannot_be_invented_or_rebound(self):
        for mutate in (lambda d: d["preflight"].update(sha256="b"*64),
                       lambda d: d["native"].update(source_unchanged=False),
                       lambda d: d["native"]["bounds_blender_z_up"].update(max=[1, float("inf"), 2]),
                       lambda d: d["native"]["bounds_blender_z_up"].update(max=[-1, -.5, 0])):
            bad = inspection(); mutate(bad)
            with self.assertRaises(ValueError):
                build_quality_policy({}, {"status": "ready", "states": []}, bad)
        missing = build_quality_policy({}, {"status": "ready", "states": []})
        self.assertFalse(evaluate_finish_policy(coat_config(), missing)["passed"])

    def test_full_resolution_requires_both_representations_and_all_views(self):
        rows = evidence()
        self.assertTrue(evaluate_quality_evidence(policy(), rows)["passed"])
        for keep in (lambda r: r["id"] != "exported-back", lambda r: r["id"] != "native-dance",
                     lambda r: r["representation"] != "exported", lambda r: r["kind"] != "portrait"):
            self.assertFalse(evaluate_quality_evidence(policy(), [row for row in rows if keep(row)])["passed"])
        self.assertFalse(evaluate_quality_evidence(policy(), rows)["visual_acceptance"])

    def test_large_sheet_of_tiny_poses_is_not_hd_state_evidence(self):
        rows = evidence()
        for row in rows:
            if row["kind"] == "state":
                row.update(pose_width=128, pose_height=128)
        result = evaluate_quality_evidence(policy(), rows)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["missing"]), 2)
        rows = evidence(); rows[-1].update(width=720, height=720)
        self.assertTrue(evaluate_quality_evidence(policy(), rows)["passed"])
        self.assertFalse(evaluate_quality_evidence(policy(preset="muse"), rows)["passed"])

    def test_inputs_reject_unsafe_extra_fields_nonfinite_and_oversized_arrays(self):
        surface = {"region": "coat", "material": "fur", "finish": "Soft"}
        for bad in ([{**surface, "source": "/unrelated"}], [{**surface, "material": "execute"}], [surface]*9,
                    [{**surface, "finish": "x"*601}], [surface, surface]):
            with self.assertRaises(ValueError):
                validate_surface_intent(bad)
        for mutate in (lambda r: r[0].update(width=True), lambda r: r[0].update(height=20000),
                       lambda r: r[0].update(width=float("nan")), lambda r: r[0].update(approved=True),
                       lambda r: r.append(deepcopy(r[0]))):
            rows = evidence(); mutate(rows)
            with self.assertRaises(ValueError):
                evaluate_quality_evidence(policy(), rows)


if __name__ == "__main__":
    unittest.main()
