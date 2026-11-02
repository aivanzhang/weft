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
            "portable_fur": {"settings": {"sample_root_color": True, "profile": "tuft-v1",
                                            "root_placement": "skin-consistent-v1"}},
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
    def test_new_fur_requires_safe_placement_without_material_owned_selector(self):
        current = policy(); config = coat_config(); original = deepcopy(config)
        self.assertEqual(current['schema_version'], 2)
        self.assertEqual(current['portable_attachment_policy'], 'skin-consistent-portable-fur-v1')
        self.assertEqual(current['portable_coat_routes']['default'], 'portable_fur/tuft-v1')
        self.assertNotIn('material_fur', config)
        self.assertTrue(evaluate_finish_policy(config, current)['passed'])
        self.assertEqual(config, original)
        for changes in ({'profile': 'combed-v1'}, {'root_placement': None}, {'sample_root_color': False}):
            bad = deepcopy(config); bad['portable_fur']['settings'].update(changes)
            with self.subTest(changes=changes):
                self.assertFalse(evaluate_finish_policy(bad, current)['passed'])
        self.assertIsNone(policy(False)['portable_coat_routes']['root_placement'])

    def test_recorded_v1_combed_and_material_tuft_policies_are_not_reinterpreted(self):
        saved = policy(); saved['schema_version'] = 1; saved.pop('portable_attachment_policy')
        saved['portable_coat_routes'] = {'default': 'portable_fur/combed-v1', 'optional': [],
                                        'automatic_fringe_selection': False, 'visual_acceptance': False}
        old = coat_config(); old['portable_fur']['settings'] = {'profile': 'combed-v1', 'sample_root_color': True}
        original = deepcopy((saved, old))
        self.assertTrue(evaluate_finish_policy(old, saved)['passed'])
        self.assertEqual((saved, old), original)
        old['material_fur'] = {'version': 1, 'materials': ['observed-coat']}
        old['portable_fur']['settings']['profile'] = 'tuft-v1'
        self.assertTrue(evaluate_finish_policy(old, saved)['passed'])
        self.assertFalse(evaluate_finish_policy(old, policy())['passed'])

    def test_policy_version_and_route_cannot_be_silently_downgraded(self):
        for version, marker in ((3, 'skin-consistent-portable-fur-v1'), (True, None),
                                (2, None), (1, 'skin-consistent-portable-fur-v1')):
            changed = policy(); changed.update(schema_version=version, portable_attachment_policy=marker)
            with self.subTest(version=version, marker=marker), self.assertRaises(ValueError):
                evaluate_finish_policy(coat_config(), changed)

    def test_new_author_guide_has_one_skin_consistent_route_and_no_combed_default(self):
        from avatar_assets.pipeline_prompts import author_prompt
        plan = {'status': 'ready', 'surface_intent': policy()['surface_intent'], 'states': [{'id': 'dance'}]}
        for request in ({'preset': 'preview'}, {'preset': 'preview', 'agent_backend': {'provider': 'openai'}}):
            prompt = author_prompt(request, plan, inspection())
            self.assertIn('require profile:"tuft-v1", sample_root_color:true and root_placement:"skin-consistent-v1"', prompt)
            self.assertIn('This does not require material_fur', prompt)
            self.assertNotIn('Combed cards remain', prompt)
            self.assertNotIn('Keep source-colored combed', prompt)

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
