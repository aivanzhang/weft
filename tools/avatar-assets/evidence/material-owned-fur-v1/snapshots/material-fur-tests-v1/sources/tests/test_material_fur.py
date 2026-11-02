"""Analytic material ownership and color fixtures, not groom or avatar quality proof."""

import time
import tracemalloc
import unittest

import numpy as np

from scripts.fur_root_sampling import RootColorSampler
from scripts.material_fur import (OWNERSHIP_POLICY, ConstantRootSampler, allocate_counts,
    constant_selection_rgb, selection_rgb, unique_vertex_sources)


class AllocationTests(unittest.TestCase):
    def test_exact_largest_remainders_minima_and_stable_ties(self):
        self.assertEqual(allocate_counts([1, 1, 1], 2), [1, 1, 0])
        self.assertEqual(allocate_counts([1, 3], 9, minimum=2), [3, 6])
        self.assertEqual(allocate_counts([7, 1, 2], 11), [8, 1, 2])
        self.assertEqual(allocate_counts([1, 9], 32, minimum=16), [16, 16])
        self.assertEqual(allocate_counts([1, 2], 0), [0, 0])
        self.assertEqual(allocate_counts([1e308, 1e308], 3), [2, 1])
        self.assertEqual(allocate_counts([1e-300, 1e300], 3), [0, 3])

    def test_varied_bounded_groups_preserve_global_budget_and_inputs(self):
        rng = np.random.default_rng(512)
        for count in (1, 2, 8, 32):
            weights = rng.uniform(.001, 20, count); original = weights.copy()
            for total in (count*16, 2400, 10000):
                values = allocate_counts(weights, total, minimum=16)
                self.assertEqual(sum(values), total)
                self.assertTrue(all(type(value) is int and value >= 16 for value in values))
                self.assertEqual(values, allocate_counts(weights, total, minimum=16))
            np.testing.assert_array_equal(weights, original)

    def test_invalid_weights_counts_and_insufficient_minimum_reject(self):
        for weights in ([], [0], [-1], [np.nan], [np.inf], [True, 2], np.array(True),
                        [[1, 2]], ['1'], [1]*33, np.ones(3, dtype=bool)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                allocate_counts(weights, 5)
        for args in ({'total': True}, {'total': -1}, {'total': 10001}, {'total': 1.5},
                     {'total': 1, 'minimum': 1}, {'total': 2, 'minimum': True},
                     {'total': 0, 'minimum': 1}, {'total': 4, 'max_total': 3},
                     {'total': 1, 'max_total': 10001}, {'total': 1, 'max_total': False}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                allocate_counts([1, 1], **args)


class VertexOwnershipTests(unittest.TestCase):
    def fixture(self):
        vertices = np.array([2, 0, 1, 0, 2, 3, 3], dtype=np.int32)
        slots = np.array([1, 0, 0, 0, 1, 2, 2], dtype=np.int16)
        uvs = np.array([[.1, .2], [.2, .3], [.4, .5], [.2, .3], [.1, .2], [0, 0], [1, 1]], dtype=np.float32)
        return vertices, slots, uvs

    def test_first_loop_is_admitted_only_after_material_and_textured_seam_proof(self):
        vertices, slots, uv = self.fixture(); originals = [a.copy() for a in (vertices, slots, uv)]
        first, report = unique_vertex_sources(4, vertices, slots, uv, {0, 1})
        np.testing.assert_array_equal(first, [1, 2, 0, 5])
        self.assertEqual(report['policy'], OWNERSHIP_POLICY)
        self.assertEqual(report['material_vertex_counts'], [
            {'slot': 0, 'vertices': 2, 'textured': True}, {'slot': 1, 'vertices': 1, 'textured': True},
            {'slot': 2, 'vertices': 1, 'textured': False}])
        for current, original in zip((vertices, slots, uv), originals):
            np.testing.assert_array_equal(current, original)

    def test_cross_material_vertex_rejects_even_when_both_materials_are_constant(self):
        vertices, slots, uv = self.fixture(); slots[3] = 2
        with self.assertRaisesRegex(ValueError, 'Vertex 0.*material'):
            unique_vertex_sources(4, vertices, slots, uv, [])

    def test_textured_uvs_need_exact_bytes_not_wrapped_or_epsilon_equivalence(self):
        for value in (.20000002, 1.2):
            vertices, slots, uv = self.fixture(); uv[3, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'exact loop UV'):
                unique_vertex_sources(4, vertices, slots, uv, [0, 1])
        vertices, slots, uv = self.fixture(); uv[1, 0] = 0.; uv[3, 0] = -0.
        with self.assertRaisesRegex(ValueError, 'exact loop UV'):
            unique_vertex_sources(4, vertices, slots, uv, [0])
        first, _ = unique_vertex_sources(4, vertices, slots, uv, [])
        self.assertEqual(first[0], 1)

    def test_malformed_unreferenced_or_oversized_ownership_rejects(self):
        vertices, slots, uv = self.fixture()
        cases = [(True, vertices, slots, uv, []), (100001, vertices, slots, uv, []),
            (5, vertices, slots, uv, []), (4, vertices.astype(float), slots, uv, []),
            (4, vertices, slots.astype(bool), uv, []), (4, vertices, slots, uv.astype(np.float16), []),
            (4, vertices, slots, np.full_like(uv, np.inf), []),
            (4, vertices, slots, uv, [0, 0]), (4, vertices, slots, uv, [True]),
            (4, vertices, slots, uv, [32]), (4, vertices, slots, uv, np.array(0)),
            (1, np.zeros(300001, dtype=np.int32), np.zeros(300001, dtype=np.int32), np.zeros((300001, 2)), [])]
        for args in cases:
            with self.subTest(shape=np.shape(args[1])), self.assertRaises(ValueError):
                unique_vertex_sources(*args)
        bad_slots = slots.copy(); bad_slots[0] = 32
        with self.assertRaises(ValueError): unique_vertex_sources(4, vertices, bad_slots, uv, [])

    def test_maximum_vertex_loop_bound_has_deterministic_linear_work(self):
        count = 100000; vertices = np.repeat(np.arange(count, dtype=np.int32), 3)
        slots = (vertices % 32).astype(np.int16)
        uv = np.repeat(np.column_stack((np.arange(count)/count, np.zeros(count))).astype(np.float32), 3, axis=0)
        first, report = unique_vertex_sources(count, vertices, slots, uv, list(range(32)))
        np.testing.assert_array_equal(first, np.arange(count, dtype=np.int32)*3)
        self.assertEqual(report['loop_count'], 300000)


class SelectionColorTests(unittest.TestCase):
    def test_nearest_repeat_wrap_and_row_orientation_preserve_legacy_samples(self):
        pixels = np.array([[[1., 0, 0, .2], [0, 1, 0, .3]], [[0, 0, 1, .4], [1, 1, 1, .5]]], dtype=np.float32)
        uvs = np.array([[0, 0], [.49, .49], [.5, 0], [0, .5], [1, 1], [-.1, .1], [2.4, -.2]], dtype=np.float32)
        expected = pixels[[0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0], :3]
        result = selection_rgb(pixels, uvs)
        self.assertEqual(result.tobytes(), expected.tobytes())
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(selection_rgb(pixels[::-1], np.array([[0, .9]], dtype=float)), [[1, 0, 0]])
        color = selection_rgb(pixels, np.array([[.5, .25]], dtype=float))[0]
        self.assertEqual(tuple(color), (0, 1, 0))
        self.assertEqual(RootColorSampler(pixels, np.array([[[.5, .25]]*3]))(0, (1, 0, 0)), (.5, .5, 0))

    def test_factor_is_linear_not_an_encoded_rgb_multiply(self):
        pixels = np.array([[[.5, .5, .5], [1, 1, 1]]], dtype=np.float64)
        uv = np.array([[.1, .5], [.9, .5]])
        result = selection_rgb(pixels, uv, factor=(.25, .5, 1))
        np.testing.assert_allclose(result[0], [.25648321840585103, .3607802138332792, .5], atol=1e-12)
        np.testing.assert_allclose(result[1], [.5370987304831942, .7353569830524495, 1.], atol=1e-12)
        self.assertGreater(result[0, 1], .25)
        np.testing.assert_array_equal(pixels, np.array([[[.5, .5, .5], [1, 1, 1]]]))

    def test_all_rgb_rows_are_validated_even_when_queries_do_not_sample_them(self):
        uv = np.zeros((1, 2), dtype=np.float32)
        for value in (np.nan, -1e-6, 1.0001):
            pixels = np.zeros((65, 2, 4), dtype=np.float32); pixels[-1, -1, 1] = value
            with self.subTest(value=value), self.assertRaises(ValueError): selection_rgb(pixels, uv)
        for pixels in (np.zeros((0, 2, 3)), np.zeros((1, 1, 3), dtype=np.uint8),
                       np.broadcast_to(np.zeros((1, 1, 3)), (4096, 4097, 3))):
            with self.assertRaises(ValueError): selection_rgb(pixels, uv)
        pixels = np.zeros((1, 1, 3), dtype=np.float32)
        for bad in (np.full((1, 2), np.nan), np.zeros((100001, 2)), np.zeros((1, 2), dtype=int)):
            with self.assertRaises(ValueError): selection_rgb(pixels, bad)
        for factor in ((True, 1, 1), (-.1, 1, 1), (1.1, 1, 1), (float('inf'), 1, 1)):
            with self.assertRaises(ValueError): selection_rgb(pixels, uv, factor=factor)

    def test_16mi_pixel_validation_borrows_image_and_uses_only_chunk_scratch(self):
        pixels = np.broadcast_to(np.array([[[.5, .25, 1, 1]]], dtype=np.float32), (4096, 4096, 4))
        tracemalloc.start()
        try:
            started = time.monotonic(); result = selection_rgb(pixels, np.array([[.25, .25]], dtype=np.float32))
            _, peak = tracemalloc.get_traced_memory()
            self.assertLess(peak, 8*1024**2)
            self.assertLess(time.monotonic()-started, 10)
        finally:
            tracemalloc.stop()
        np.testing.assert_array_equal(result, [[.5, .25, 1]])
        self.assertFalse(pixels.flags.writeable)
        self.assertEqual(selection_rgb(pixels[:1, :1], np.empty((0, 2), dtype=np.float32)).shape, (0, 3))


class ConstantColorTests(unittest.TestCase):
    def test_constant_selection_is_encoded_but_root_color_stays_linear(self):
        linear = [.25, .5, 1.]
        selected = constant_selection_rgb(linear, 3)
        np.testing.assert_allclose(selected, [[.53709873, .73535698, 1.]]*3, atol=5e-8)
        self.assertEqual(selected.dtype, np.float32)
        sampler = ConstantRootSampler(linear, 2)
        self.assertEqual(sampler(0, (.2, .3, .5)), (.25, .5, 1.))
        self.assertEqual(sampler(1, (1, 0, 0)), (.25, .5, 1.))
        linear[0] = 1
        self.assertEqual(sampler(0, (0, 1, 0)), (.25, .5, 1.))
        self.assertEqual(constant_selection_rgb([0, 0, 0], 0).shape, (0, 3))

    def test_constant_root_addresses_follow_existing_triangle_barycentric_contract(self):
        sampler = ConstantRootSampler((0., .25, .5), 2)
        for triangle, bary in ((True, (1, 0, 0)), (-1, (1, 0, 0)), (2, (1, 0, 0)),
                (np.int32(0), (1, 0, 0)), (0, (1, 1, 0)), (0, (-1, 1, 1)),
                (0, (np.nan, 0, 1)), (0, (1, 0)), (0, (np.inf, 0, 0))):
            with self.subTest(triangle=triangle, bary=bary), self.assertRaises(ValueError): sampler(triangle, bary)
        for count in (True, -1, 0, 100001):
            with self.assertRaises(ValueError): ConstantRootSampler((0, 0, 0), count)
        for rgb in ([True, 0, 0], [np.nan, 0, 0], [-.1, 0, 0], [1.01, 0, 0], [0, 0]):
            with self.assertRaises(ValueError): constant_selection_rgb(rgb, 1)
            with self.assertRaises(ValueError): ConstantRootSampler(rgb, 1)


if __name__ == '__main__':
    unittest.main()
