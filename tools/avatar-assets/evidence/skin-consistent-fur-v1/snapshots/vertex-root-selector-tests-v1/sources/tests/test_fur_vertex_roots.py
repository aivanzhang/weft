"""Exact-root analytic fixtures, not an avatar motion or appearance acceptance."""

from collections import Counter
import hashlib
import json
import time
import unittest
from unittest.mock import patch

import numpy as np

from scripts import fur_vertex_roots as roots


class VertexRootSelectorTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
        self.triangles = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)

    def select(self, count=3, seed=0):
        return roots.select_vertex_roots(self.vertices, self.triangles, count, seed)

    def test_seeded_area_ownership_is_exact_count_and_stably_ordered(self):
        result = self.select()
        self.assertEqual(result["roots"], [
            {"triangle_index": 0, "corner": 1, "vertex_index": 1, "barycentric": [0., 1., 0.]},
            {"triangle_index": 0, "corner": 2, "vertex_index": 2, "barycentric": [0., 0., 1.]},
            {"triangle_index": 1, "corner": 2, "vertex_index": 3, "barycentric": [0., 0., 1.]},
        ])
        self.assertEqual(result, self.select())
        self.assertEqual([row["vertex_index"] for row in self.select(seed=1)["roots"]], [0, 2, 3])
        self.assertEqual(result["summary"]["policy"], roots.POLICY)
        self.assertEqual(result["summary"]["total_eligible_area"], 1.)
        self.assertEqual(result["summary"]["eligible_source_vertices"], 4)
        self.assertEqual(result["summary"]["selected_roots"], 3)
        self.assertFalse(result["summary"]["source_attributes_modified"])

    def test_one_hot_roots_retain_source_position_uv_and_weight_ownership(self):
        uv = np.array([[.01, .2], [.4, .8], [.9, .1], [.7, .7]])
        weights = np.array([[1., 0, 0, 0], [.2, .8, 0, 0], [0, 0, 1, 0], [.3, .1, .2, .4]])
        for row in self.select(count=4)["roots"]:
            ids = self.triangles[row["triangle_index"]]
            self.assertEqual(int(ids[row["corner"]]), row["vertex_index"])
            self.assertEqual(row["barycentric"].count(1.), 1)
            self.assertEqual(row["barycentric"].count(0.), 2)
            for source in (self.vertices, uv, weights):
                value = np.sum(source[ids]*np.array(row["barycentric"])[:, None], axis=0)
                np.testing.assert_array_equal(value, source[row["vertex_index"]])
        self.assertEqual(len({row["vertex_index"] for row in self.select(count=4)["roots"]}), 4)

    def test_triangle_area_not_triangle_count_controls_sampling(self):
        vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0],
                             [5, 0, 0], [8, 0, 0], [5, 3, 0]])
        triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        counts = Counter(roots.select_vertex_roots(vertices, triangles, 1, seed)["roots"][0]["vertex_index"]
                         for seed in range(256))
        large_triangle_hits = sum(counts[index] for index in (3, 4, 5))
        self.assertGreater(large_triangle_hits, 210)
        self.assertLess(large_triangle_hits, 250)
        self.assertTrue(all(counts[index] > 0 for index in range(6)))
        # Shared vertices receive both incident contributions, not one face's weight.
        shared = Counter(self.select(count=1, seed=seed)["roots"][0]["vertex_index"] for seed in range(256))
        self.assertGreater(shared[1]+shared[2], 145)
        self.assertLess(shared[1]+shared[2], 195)

    def test_equal_keys_tie_by_source_id_before_return_order(self):
        faces = np.array([[2, 0, 1]], dtype=np.int32)
        for draw in (.5, 0.):
            with self.subTest(draw=draw), patch.object(roots.random, "Random") as random_class:
                random_class.return_value.random.return_value = draw
                result = roots.select_vertex_roots(self.vertices[:3], faces, 2, 23)
            self.assertEqual([row["vertex_index"] for row in result["roots"]], [0, 1])
            self.assertEqual([row["corner"] for row in result["roots"]], [1, 2])

    def test_degenerate_threshold_and_representative_ignore_ineligible_faces(self):
        points = np.array([[0., 0, 0], [1, 0, 0], [0, 2e-14, 0], [0, 2.0002e-14, 0], [2, 2, 2]])
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 0, 1], [0, 1, 1]], dtype=np.int32)
        result = roots.select_vertex_roots(points, faces, 3, 14)
        self.assertEqual([row["vertex_index"] for row in result["roots"]], [0, 1, 3])
        self.assertEqual([row["triangle_index"] for row in result["roots"]], [1, 1, 1])
        self.assertEqual(result["summary"]["eligible_triangles"], 1)
        self.assertEqual(result["summary"]["excluded_degenerate_triangles"], 3)
        self.assertEqual(result["summary"]["eligible_source_vertices"], 3)
        with self.assertRaisesRegex(ValueError, "unique eligible.*3"):
            roots.select_vertex_roots(points, faces, 4, 14)

    def test_distinct_coincident_ids_are_not_welded_or_silently_repeated(self):
        points = np.concatenate((self.vertices[:3], self.vertices[:3]))
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        result = roots.select_vertex_roots(points, faces, 6, 1)
        self.assertEqual([row["vertex_index"] for row in result["roots"]], list(range(6)))
        self.assertEqual(result["summary"]["coincident_position_groups"], 3)
        self.assertEqual(result["summary"]["coincident_eligible_vertices"], 6)
        self.assertEqual(result["summary"]["coincident_extra_source_ids"], 3)
        with self.assertRaisesRegex(ValueError, "unique eligible.*6"):
            roots.select_vertex_roots(points, faces, 7, 1)

    def test_coarse_mesh_rejects_requested_count_without_reduction(self):
        with self.assertRaisesRegex(ValueError, "unique eligible.*4"):
            self.select(count=5)
        empty = roots.select_vertex_roots(self.vertices, [], 0, 2**31-1)
        self.assertEqual(empty["roots"], [])
        self.assertEqual(empty["summary"]["total_eligible_area"], 0)
        with self.assertRaisesRegex(ValueError, "unique eligible.*0"):
            roots.select_vertex_roots(self.vertices, [], 1, 0)

    def test_list_tuple_array_inputs_remain_byte_identical(self):
        points = self.vertices.astype(np.float32); faces = self.triangles.astype(np.uint32)
        originals = (points.tobytes(), faces.tobytes())
        points.flags.writeable = False; faces.flags.writeable = False
        expected = roots.select_vertex_roots(points, faces, 3, 0)
        self.assertEqual(expected, roots.select_vertex_roots(points.tolist(), faces.tolist(), 3, 0))
        self.assertEqual(expected, roots.select_vertex_roots(tuple(map(tuple, points)), tuple(map(tuple, faces)), 3, 0))
        self.assertEqual((points.tobytes(), faces.tobytes()), originals)
        self.assertEqual(json.dumps(expected, sort_keys=True, allow_nan=False), json.dumps(self.select(), sort_keys=True, allow_nan=False))

    def test_noninteger_indices_nonfinite_points_and_malformed_rows_reject(self):
        bad_points = ([], [0, 1, 2], [[0, 1]], [[True, 0, 0]], [["0", 0, 0]],
                      np.zeros((1, 3), dtype=bool), np.zeros((1, 3), dtype=complex),
                      np.full((1, 3), np.inf), np.full((1, 3), np.nan), np.array(0),
                      np.array([[0, 1, 2]], dtype=object), [[10**1000, 0, 0]])
        for points in bad_points:
            with self.subTest(points=type(points).__name__), self.assertRaises(ValueError):
                roots.select_vertex_roots(points, [[0, 0, 0]], 0, 0)
        bad_faces = ([[0., 1., 2.]], [[False, 1, 2]], [[0, -1, 2]], [[0, 1, 4]], [[0, 1]],
                     [0, 1, 2], np.zeros((1, 3), dtype=bool), np.array([[0, 1, 2]], dtype=object),
                     np.array([[0, 1, 2**64-1]], dtype=np.uint64), np.full((1, 3), np.nan))
        for faces in bad_faces:
            with self.subTest(faces=type(faces).__name__), self.assertRaises(ValueError):
                roots.select_vertex_roots(self.vertices, faces, 0, 0)
        huge = np.array([[0., 0, 0], [1e308, 0, 0], [0, 1e308, 0]])
        with self.assertRaisesRegex(ValueError, "areas exceed finite"):
            roots.select_vertex_roots(huge, [[0, 1, 2]], 1, 0)

    def test_count_seed_and_oversized_shapes_reject_before_geometry_allocation(self):
        for count in (-1, 5001, True, 1., np.int64(1)):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.select(count=count)
        for seed in (-1, 2**31, True, 1., np.int64(1)):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.select(seed=seed)
        cases = [(np.broadcast_to(np.zeros((1, 3)), (100001, 3)), self.triangles),
                 (self.vertices.tolist(), np.broadcast_to(np.zeros((1, 3), dtype=int), (100001, 3))),
                 ([[0., 0, 0]]*100001, self.triangles), (self.vertices, [[0, 1, 2]]*100001)]
        with patch.object(roots.np, "asarray", side_effect=AssertionError("oversize input was converted")), \
                patch.object(roots.np, "cross", side_effect=AssertionError("oversize geometry was allocated")):
            for points, faces in cases:
                with self.subTest(vertices=len(points), triangles=len(faces)), self.assertRaises(ValueError):
                    roots.select_vertex_roots(points, faces, 1, 0)

    def test_maximum_source_bounds_have_exact_unique_count_and_bounded_summary(self):
        count = 100000
        indices = np.arange(count, dtype=np.int32)
        points = np.column_stack((indices//3, indices % 3 == 1, indices % 3 == 2)).astype(np.float32)
        faces = np.tile(np.arange(99999, dtype=np.int32).reshape(-1, 3), (4, 1))[:count]
        before = hashlib.sha256(points.tobytes()+faces.tobytes()).hexdigest()
        started = time.monotonic()
        result = roots.select_vertex_roots(points, faces, 5000, 12345)
        elapsed = time.monotonic()-started
        self.assertLess(elapsed, 10)
        self.assertEqual(result["summary"]["source_vertices"], 100000)
        self.assertEqual(result["summary"]["source_triangles"], 100000)
        self.assertEqual(result["summary"]["eligible_source_vertices"], 99999)
        self.assertEqual(len(result["roots"]), 5000)
        self.assertEqual(len({row["vertex_index"] for row in result["roots"]}), 5000)
        order = [(row["triangle_index"], row["corner"]) for row in result["roots"]]
        self.assertEqual(order, sorted(order))
        self.assertLess(len(json.dumps(result["summary"])), 2000)
        self.assertEqual(hashlib.sha256(points.tobytes()+faces.tobytes()).hexdigest(), before)
        print("Maximum analytic selector: %.6fs;100000vertices/100000triangles/5000unique roots" % elapsed)


if __name__ == "__main__":
    unittest.main()
