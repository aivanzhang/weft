"""Analytic weight correction tests; no generated-asset or visual acceptance claims."""

from copy import deepcopy
import unittest

import numpy as np

from avatar_assets.stable_role_projection import project_stable_roles


def fixture():
    joints = np.tile([0, 1, 2, 0], (24, 1))
    weights = np.array([[.8, .2, 0, 0]]*8 + [[0, 1, 0, 0]]*8 + [[0, 0, 1, 0]]*8)
    names = ["Body", "Hand", "Foot"]
    parents = {"Body": None, "Hand": "Body", "Foot": "Body"}
    roles = [{"state_id": "wave", "role_id": "torso", "bones": ["Body"], "stable": True},
             {"state_id": "wave", "role_id": "hand", "bones": ["Hand"], "stable": False},
             {"state_id": "wave", "role_id": "feet", "bones": ["Foot"], "stable": True}]
    return joints, weights, names, parents, roles


def project(values):
    joints, weights, names, parents, roles = values
    return project_stable_roles(joints, weights, names, roles, bone_parents=parents,
                                surface_kind="rigid_hard_surface")


class StableRoleProjectionTests(unittest.TestCase):
    def test_removes_body_leakage_without_relabeling_or_weakening_hand_motion(self):
        values = fixture()
        joints, original, names, _, _ = values
        joints_before, original_before = joints.copy(), original.copy()
        result, report = project(values)
        translation = np.array([[0, 0, 0], [.5, 0, 0], [0, 0, 0]])
        before_motion = np.sum(translation[joints]*original[:, :, None], axis=1)
        after_motion = np.sum(translation[joints]*result[:, :, None], axis=1)
        np.testing.assert_allclose(before_motion[:8, 0], .1)
        np.testing.assert_array_equal(after_motion[:8], 0)
        np.testing.assert_array_equal(before_motion[8:], after_motion[8:])
        np.testing.assert_array_equal(result[8:], original[8:])
        np.testing.assert_array_equal(original, original_before)
        np.testing.assert_array_equal(joints, joints_before)
        self.assertFalse(report["quality_approved"])
        self.assertEqual(report["changed_vertices"], 8)
        self.assertEqual(report["removed_influences"], 8)
        hand = report["roles"][1]
        self.assertEqual(hand["core_vertices_before"], 8)
        self.assertEqual(hand["core_indices_before_sha256"], hand["core_indices_after_sha256"])
        self.assertEqual(hand["fringe_vertices_before"], 16)
        self.assertEqual(hand["fringe_vertices_after"], 8)
        self.assertEqual(hand["lost_fringe_indices"], list(range(8)))
        self.assertEqual(hand["gained_fringe_count"], 0)

    def test_multi_bone_ratios_retained_without_co_rigidity_claim(self):
        joints = np.tile([0, 1, 2, 0], (8, 1))
        weights = np.tile([.6, .1, .3, 0], (8, 1))
        names = ["Base", "Panel", "Arm"]
        roles = [{"state_id": "wave", "role_id": "base", "bones": ["Base", "Panel"], "stable": True}]
        result, report = project((joints, weights, names, {name: None for name in names}, roles))
        np.testing.assert_allclose(result, np.tile([6/7, 1/7, 0, 0], (8, 1)))
        self.assertEqual(report["roles"][0]["mode"], "foreign_influence_removal_not_verified_rigid")
        self.assertFalse(report["roles"][0]["co_rigid_verified"])
        # Different target transforms still move this panel after foreign removal.
        self.assertGreater(float(result[0, 1])*.2, .02)

    def test_core_tie_and_conflicting_stable_ownership_reject_before_mutation(self):
        values = list(fixture())
        values[1][:8] = [.5, .5, 0, 0]
        before = values[1].copy()
        with self.assertRaisesRegex(ValueError, "tied role ownership"):
            project(values)
        np.testing.assert_array_equal(values[1], before)
        values[4][1]["stable"] = True
        with self.assertRaisesRegex(ValueError, "tied role ownership"):
            project(values)

    def test_new_core_membership_from_overlapping_targets_rejects(self):
        names = ["A", "B", "C", "D"]
        joints = np.tile([0, 1, 2, 3], (16, 1))
        weights = np.array([[.6, .1, 0, .3]]*8 + [[0, .4, .2, .4]]*8)
        roles = [{"state_id": "move", "role_id": "first", "bones": ["A", "B"], "stable": True},
                 {"state_id": "move", "role_id": "second", "bones": ["B", "C"], "stable": True}]
        with self.assertRaisesRegex(ValueError, "core membership"):
            project((joints, weights, names, {name: None for name in names}, roles))

    def test_new_fringe_from_rescaling_shared_bone_rejects(self):
        names = ["A", "B", "C"]
        joints = np.tile([0, 1, 2, 0], (8, 1))
        weights = np.tile([.56, .04, .4, 0], (8, 1))
        roles = [{"state_id": "move", "role_id": "base", "bones": ["A", "B"], "stable": True},
                 {"state_id": "move", "role_id": "limb", "bones": ["B"], "stable": False}]
        with self.assertRaisesRegex(ValueError, "gained fringe"):
            project((joints, weights, names, {name: None for name in names}, roles))

    def test_identical_stable_targets_across_states_share_frozen_ownership(self):
        values = list(fixture())
        duplicate = deepcopy(values[4][0])
        duplicate["state_id"] = "greet"
        values[4].append(duplicate)
        result, report = project(values)
        self.assertEqual(report["protected_vertices"], 16)
        self.assertEqual(report["changed_vertices"], 8)
        self.assertEqual(report["roles"][0]["core_indices_before_sha256"],
                         report["roles"][-1]["core_indices_before_sha256"])
        np.testing.assert_allclose(result.sum(axis=1), 1)
        values[4].append(deepcopy(duplicate))
        with self.assertRaisesRegex(ValueError, "Duplicate role"):
            project(values)

    def test_seam_copies_keep_separate_indices_and_equal_weights(self):
        values = fixture()
        result, report = project(values)
        np.testing.assert_array_equal(result[0], result[7])
        self.assertEqual(report["roles"][1]["lost_fringe_indices"], list(range(8)))
        self.assertEqual(report["roles"][0]["core_vertices_after"], 8)
        self.assertIn("separate exported", report["index_space"])

    def test_unselected_encoding_error_remains_exact_and_selected_normalizes(self):
        values = list(fixture())
        values[1] = values[1].astype(np.float32)
        values[1][8:16] *= .9995
        result, _ = project(values)
        np.testing.assert_array_equal(result[8:16].astype(np.float32), values[1][8:16])
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all(np.count_nonzero(result, axis=1) <= 4))

    def test_inherited_root_travel_is_not_repaired_or_approved(self):
        joints, weights, names, parents, roles = fixture()
        result, report = project((joints, weights, names, parents, roles))
        inherited = np.tile([.1, 0, 0], (len(names), 1))
        displaced = np.sum(inherited[joints]*result[:, :, None], axis=1)
        np.testing.assert_allclose(displaced[:, 0], .1)
        self.assertFalse(report["quality_approved"])

    def test_weight_projection_can_create_a_boundary_stretch(self):
        values = fixture()
        joints, weights, _, _, _ = values
        result, _ = project(values)
        rest = np.array([[0, 0, 0], [.001, 0, 0]])
        edge = [0, 8]
        translations = np.array([[0, 0, 0], [.5, 0, 0], [0, 0, 0]])
        before = rest + np.sum(translations[joints[edge]]*weights[edge, :, None], axis=1)
        after = rest + np.sum(translations[joints[edge]]*result[edge, :, None], axis=1)
        self.assertGreater(np.linalg.norm(after[1]-after[0]), np.linalg.norm(before[1]-before[0]))

    def test_explicit_surface_opt_in_and_complete_roles_are_required(self):
        joints, weights, names, parents, roles = fixture()
        for surface in ("organic", "mixed", "metal", None, True):
            with self.subTest(surface=surface), self.assertRaisesRegex(ValueError, "opt-in"):
                project_stable_roles(joints, weights, names, roles, bone_parents=parents, surface_kind=surface)
        with self.assertRaises(TypeError):
            project_stable_roles(joints, weights, names, roles, bone_parents=parents)
        roles[0]["script"] = "run"
        with self.assertRaisesRegex(ValueError, "exact"):
            project_stable_roles(joints, weights, names, roles, bone_parents=parents,
                                  surface_kind="rigid_hard_surface")

    def test_nonfinite_boolean_malformed_and_oversized_arrays_reject(self):
        for field, replacement in ((0, np.full((24, 4), True)), (0, np.full((24, 4), 99)),
                                   (0, np.zeros((24, 4), dtype=float)), (1, np.full((24, 4), np.nan)),
                                   (1, np.full((24, 4), np.inf)), (1, np.full((24, 4), -.25)),
                                   (1, np.zeros((24, 4))), (1, np.ones((100001, 4))),
                                   (1, np.ones((24, 3))), (1, [[True, 0, 0, 0]]*24),
                                   (0, [[10**100, 0, 0, 0]]*24), (1, [[10**100, 0, 0, 0]]*24)):
            with self.subTest(field=field, shape=np.shape(replacement)), self.assertRaises(ValueError):
                values = list(fixture())
                values[field] = replacement
                project(values)

    def test_invalid_bones_roles_hierarchy_and_insufficient_coverage_reject(self):
        for kind in ("unknown", "duplicate_bone", "cycle", "missing_parent", "role_limit", "zero_coverage", "no_stable"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                values = list(fixture())
                if kind == "unknown":
                    values[4][0]["bones"] = ["missing"]
                elif kind == "duplicate_bone":
                    values[2][1] = values[2][0]
                elif kind == "cycle":
                    values[3]["Body"] = "Hand"
                elif kind == "missing_parent":
                    del values[3]["Foot"]
                elif kind == "role_limit":
                    values[4] *= 11
                elif kind == "zero_coverage":
                    values[1][:8] = [0, 1, 0, 0]
                else:
                    for role in values[4]:
                        role["stable"] = False
                project(values)


if __name__ == "__main__":
    unittest.main()
