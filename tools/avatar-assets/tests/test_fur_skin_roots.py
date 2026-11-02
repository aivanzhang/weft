"""Analytic skin/root contracts, not generated-avatar deformation acceptance."""

from copy import deepcopy
import hashlib
import json
import math
import time
import unittest
from unittest.mock import patch

import numpy as np

from scripts import fur_skin_roots as skin
from scripts.fur_vertex_roots import select_vertex_roots


def domains(uniform_area=.5, varying_area=.5):
    first, second = math.sqrt(2*uniform_area), math.sqrt(2*varying_area)
    points = np.array([[0., 0, 0], [first, 0, 0], [0, first, 0],
                       [10, 0, 0], [10+second, 0, 0], [10, second, 0]])
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    weights = [{"A": 1.}, {"A": 1.}, {"A": 1.}, {"A": 1.}, {"B": 1.}, {"A": 1.}]
    return points, faces, weights


def lbs(point, weights, transforms):
    total = math.fsum(weights.values())
    homogeneous = np.append(point, 1.)
    return sum((transforms[name]@homogeneous)[:3]*(weight/total) for name, weight in weights.items())


class SkinRootSelectorTests(unittest.TestCase):
    def test_uniform_multibone_interior_skinning_commutes(self):
        points, faces, _ = domains()
        points, faces = points[:3], faces[:1]
        weights = [{"A": .25, "B": .5, "C": .25} for _ in range(3)]
        transforms = {name: np.eye(4) for name in ("A", "B", "C")}
        transforms["B"][:2, :2] = [[0, -1], [1, 0]]
        transforms["C"][:3, 3] = [.4, -.2, .8]
        deformed = np.array([lbs(point, row, transforms) for point, row in zip(points, weights)])
        selected = skin.select_skin_roots(points, faces, weights, 64, 31)
        for row in selected["roots"]:
            self.assertEqual(row["binding_kind"], skin.UNIFORM)
            self.assertTrue(all(value > 0 for value in row["barycentric"]))
            self.assertNotIn("vertex_index", row)
            bary = np.array(row["barycentric"])
            surface = bary@deformed
            inherited = {name: sum(bary[index]*weights[index][name] for index in range(3)) for name in transforms}
            attached = lbs(bary@points, inherited, transforms)
            np.testing.assert_allclose(attached, surface, rtol=0, atol=1e-14)
        self.assertEqual(selected["summary"]["maximum_positive_influences"], 3)

    def test_mixed_weight_barycentric_noncommutation_requires_exact_vertex(self):
        points, faces, _ = domains()
        points, faces = points[:3], faces[:1]
        weights = [{"A": 1.}, {"B": 1.}, {"A": 1.}]
        transforms = {"A": np.eye(4), "B": np.eye(4)}
        transforms["B"][:2, :2] = [[0, -1], [1, 0]]
        deformed = np.array([lbs(point, row, transforms) for point, row in zip(points, weights)])
        bary = np.array([.25, .5, .25])
        interpolated = lbs(bary@points, {"A": .5, "B": .5}, transforms)
        np.testing.assert_allclose(interpolated-bary@deformed, [.125, -.375, 0], rtol=0, atol=1e-15)
        selected = skin.select_skin_roots(points, faces, weights, 3, 7)
        for row in selected["roots"]:
            self.assertEqual(row["binding_kind"], skin.VARYING)
            source = row["vertex_index"]
            np.testing.assert_array_equal(lbs(points[source], weights[source], transforms), deformed[source])
            self.assertEqual(row["barycentric"][row["corner"]], 1.)
        self.assertEqual(len({row["vertex_index"] for row in selected["roots"]}), 3)

    def test_normalization_is_named_exact_and_keeps_tiny_positive_influences(self):
        points, faces, _ = domains(); points, faces = points[:3], faces[:1]
        weights = [{"A": .25, "B": .75, "Unused": 0.}, {"B": 3., "A": 1.}, {"A": .5, "B": 1.5}]
        result = skin.select_skin_roots(points, faces, weights, 5, 8)
        self.assertEqual(result["summary"]["canonical_weight_classes"], 1)
        self.assertEqual(result["summary"]["ignored_zero_weight_entries"], 1)
        self.assertTrue(all(row["binding_kind"] == skin.UNIFORM for row in result["roots"]))
        for tiny in (1e-8, 2**-40, math.nextafter(0., 1.)):
            changed = [{"A": 1.}, {"A": 1., "B": tiny}, {"A": 1.}]
            with self.subTest(tiny=tiny):
                result = skin.select_skin_roots(points, faces, changed, 3, 8)
                self.assertEqual(result["summary"]["canonical_weight_classes"], 2)
                self.assertTrue(all(row["binding_kind"] == skin.VARYING for row in result["roots"]))
        names = [{"A": 1.}, {"Renamed": 1.}, {"A": 1.}]
        self.assertEqual(skin.select_skin_roots(points, faces, names, 1, 8)["roots"][0]["binding_kind"], skin.VARYING)

    def test_all_uniform_coarse_geometry_keeps_full_requested_count(self):
        points, faces, weights = domains(); points, faces, weights = points[:3], faces[:1], weights[:3]
        selected = skin.select_skin_roots(points, faces, weights, 5000, 9)
        self.assertEqual(len(selected["roots"]), 5000)
        self.assertTrue(all(row["binding_kind"] == skin.UNIFORM for row in selected["roots"]))
        self.assertEqual(selected["summary"]["domains"][0]["resolved_count"], 5000)
        self.assertFalse(selected["summary"]["count_reduction"])
        self.assertFalse(selected["summary"]["capacity_reallocation"])

    def test_hamilton_is_global_with_stable_ties_and_recorded_minimum_transfer(self):
        points, faces, weights = domains()
        for total, expected in ((0, [0, 0]), (1, [1, 0]), (2, [1, 1]), (3, [2, 1])):
            with self.subTest(total=total):
                result = skin.select_skin_roots(points, faces, weights, total, 4)
                self.assertEqual([row["resolved_count"] for row in result["summary"]["domains"]], expected)
                self.assertIsNone(result["summary"]["minimum_domain_adjustment"])
        points, faces, weights = domains(4.5, .5)
        result = skin.select_skin_roots(points, faces, weights, 10, 4)
        self.assertEqual([row["expected_count"] for row in result["summary"]["domains"]], [9., 1.])
        self.assertEqual([row["resolved_count"] for row in result["summary"]["domains"]], [9, 1])
        result = skin.select_skin_roots(points, faces, weights, 2, 4)
        self.assertEqual([row["hamilton_count"] for row in result["summary"]["domains"]], [2, 0])
        self.assertEqual([row["resolved_count"] for row in result["summary"]["domains"]], [1, 1])
        self.assertEqual(result["summary"]["minimum_domain_adjustment"], {"from": skin.UNIFORM, "to": skin.VARYING, "count": 1})

    def test_varying_capacity_failure_never_shifts_roots_to_uniform_region(self):
        points, faces, weights = domains(.5, 4.5)
        with patch.object(skin._vertex_roots, "select_vertex_roots") as sampler:
            with self.assertRaisesRegex(ValueError, "requires 9 roots but has 3.*refusing count reduction or reallocation"):
                skin.select_skin_roots(points, faces, weights, 10, 2)
            sampler.assert_not_called()

    def test_original_triangle_remap_stable_order_and_frozen_vertex_sampling(self):
        points, faces, weights = domains()
        faces = np.array([[0, 0, 1], faces[1], faces[0]], dtype=np.int32)
        result = skin.select_skin_roots(points, faces, weights, 3, 18)
        self.assertEqual([row["triangle_index"] for row in result["roots"]], [1, 2, 2])
        expected = select_vertex_roots(points, faces[[1]], 1, 18)["roots"][0]
        self.assertEqual(result["roots"][0], {**expected, "triangle_index": 1, "binding_kind": skin.VARYING})
        self.assertEqual(result["summary"]["excluded_degenerate_triangles"], 1)

    def test_uniform_area_strata_cover_large_and_small_faces_without_reduction(self):
        points, faces, _ = domains(.5, 4.5)
        weights = [{"A": .2, "B": .8} for _ in points]
        selected = skin.select_skin_roots(points, faces, weights, 100, 27)
        self.assertEqual(sum(row["triangle_index"] == 0 for row in selected["roots"]), 10)
        self.assertEqual(sum(row["triangle_index"] == 1 for row in selected["roots"]), 90)
        for row in selected["roots"]:
            self.assertTrue(all(0 < value < 1 for value in row["barycentric"]))
            self.assertAlmostEqual(sum(row["barycentric"]), 1, places=15)

    def test_deterministic_bytes_and_source_inputs_unchanged(self):
        points, faces, weights = domains(); points = points.astype(np.float32)
        original = (points.tobytes(), faces.tobytes(), deepcopy(weights))
        points.flags.writeable = False; faces.flags.writeable = False
        result = skin.select_skin_roots(points, faces, weights, 4, 13)
        encoded = json.dumps(result, sort_keys=True, allow_nan=False)
        self.assertEqual(encoded, json.dumps(skin.select_skin_roots(points.tolist(), tuple(map(tuple, faces)), weights, 4, 13), sort_keys=True, allow_nan=False))
        self.assertNotEqual(encoded, json.dumps(skin.select_skin_roots(points, faces, weights, 4, 14), sort_keys=True, allow_nan=False))
        self.assertEqual((points.tobytes(), faces.tobytes(), weights), original)
        self.assertLess(len(json.dumps(result["summary"])), 4000)

    def test_invalid_weight_rows_do_not_hide_cutoffs_or_nonfinite_normalization(self):
        points, faces, weights = domains()
        invalid = ({}, {"A": 0}, {"A": -.1}, {"A": True}, {"A": np.nan}, {"A": np.inf}, {"A": "1"},
                   {"A": 10**1000}, {"": 1}, {"A\0B": 1}, {"A"*129: 1}, {str(i): .2 for i in range(5)},
                   {str(i): (1 if i == 0 else 0) for i in range(65)},
                   {"A": 2., "B": math.nextafter(0., 1.)}, {"A": 1e308, "B": 1e308})
        for row in invalid:
            changed = deepcopy(weights); changed[0] = row
            with self.subTest(row=list(row)[:5]), self.assertRaises(ValueError):
                skin.select_skin_roots(points, faces, changed, 1, 0)
        with self.assertRaisesRegex(ValueError, "Weights must match"):
            skin.select_skin_roots(points, faces, weights[:-1], 1, 0)
        points = np.concatenate((points, [[9., 9, 9]])); weights.append(None)
        self.assertEqual(skin.select_skin_roots(points, faces, weights, 2, 0)["summary"]["referenced_vertices"], 6)

    def test_invalid_geometry_and_budget_guards_precede_weight_work(self):
        points, faces, weights = domains()
        for count, seed in ((-1, 0), (5001, 0), (True, 0), (1., 0), (1, -1), (1, 2**31), (1, True)):
            with self.subTest(count=count, seed=seed), self.assertRaises(ValueError):
                skin.select_skin_roots(points, faces, weights, count, seed)
        for bad_faces in (faces.astype(float), [[True, 1, 2]], [[0, 1, 6]], [[0, -1, 2]], [[0, 1]]):
            with self.subTest(faces=type(bad_faces).__name__), self.assertRaises(ValueError):
                skin.select_skin_roots(points, bad_faces, weights, 1, 0)
        with self.assertRaisesRegex(ValueError, "finite"):
            skin.select_skin_roots(np.full_like(points, np.nan), faces, weights, 1, 0)
        cases = [(np.broadcast_to(points[:1], (100001, 3)), faces, []),
                 (points.tolist(), np.broadcast_to(faces[:1], (100001, 3)), weights)]
        with patch.object(skin._vertex_roots, "_triples", side_effect=AssertionError("oversize input converted")):
            for bad_points, bad_faces, bad_weights in cases:
                with self.assertRaises(ValueError):
                    skin.select_skin_roots(bad_points, bad_faces, bad_weights, 1, 0)
        empty = skin.select_skin_roots(points, [], [None]*6, 0, 0)
        self.assertEqual(empty["roots"], [])
        with self.assertRaisesRegex(ValueError, "eligible nondegenerate"):
            skin.select_skin_roots(points, [], [None]*6, 1, 0)

    def test_maximum_mixed_source_and_root_bounds_are_practical(self):
        size = 100000; indices = np.arange(size, dtype=np.int32)
        points = np.column_stack((indices//3, indices % 3 == 1, indices % 3 == 2)).astype(np.float32)
        faces = np.tile(np.arange(99999, dtype=np.int32).reshape(-1, 3), (4, 1))[:size]
        first, second = {"A": 1.}, {"B": 1.}
        weights = [second if index//3 % 2 and index % 3 == 1 else first for index in range(size)]
        before = hashlib.sha256(points.tobytes()+faces.tobytes()).hexdigest()
        started = time.monotonic(); result = skin.select_skin_roots(points, faces, weights, 5000, 54321)
        elapsed = time.monotonic()-started
        self.assertLess(elapsed, 10)
        self.assertEqual(len(result["roots"]), 5000)
        self.assertEqual(sum(row["resolved_count"] for row in result["summary"]["domains"]), 5000)
        self.assertTrue(all(row["resolved_count"] > 0 for row in result["summary"]["domains"]))
        varying = [row for row in result["roots"] if row["binding_kind"] == skin.VARYING]
        self.assertEqual(len(varying), len({row["vertex_index"] for row in varying}))
        self.assertEqual([row["triangle_index"] for row in result["roots"]], sorted(row["triangle_index"] for row in result["roots"]))
        self.assertEqual(hashlib.sha256(points.tobytes()+faces.tobytes()).hexdigest(), before)
        self.assertEqual((first, second), ({"A": 1.}, {"B": 1.}))
        print("Maximum analytic mixed selector: %.6fs;100000vertices/100000triangles/5000roots" % elapsed)


if __name__ == "__main__":
    unittest.main()
