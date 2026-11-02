"""Analytic texture seams and exact roots; not generated-fur quality claims."""

import unittest
import numpy as np

from scripts.fur_root_sampling import RootColorSampler


class RootSamplerTests(unittest.TestCase):
    def test_shared_vertex_uses_its_triangle_uv_not_an_arbitrary_first_loop(self):
        pixels = np.array([[[1., 0, 0], [0, 0, 1.]]], dtype=np.float32)
        uvs = np.array([[[.25, .5]]*3, [[.75, .5]]*3], dtype=np.float32)
        sampler = RootColorSampler(pixels, uvs)
        self.assertEqual(sampler(0, (1, 0, 0)), (1, 0, 0))
        self.assertEqual(sampler(1, (1, 0, 0)), (0, 0, 1))

    def test_color_is_sampled_at_root_not_blended_from_vertex_colors(self):
        pixels = np.array([[[1., 0, 0], [0, 1., 0], [0, 1., 0], [1., 0, 0]]], dtype=np.float32)
        uvs = np.array([[[.125, .5], [.875, .5], [.125, .5]]], dtype=np.float32)
        sampler = RootColorSampler(pixels, uvs)
        self.assertEqual(sampler(0, (.5, .5, 0)), (0, 1, 0))

    def test_repeat_and_srgb_filtering_use_linear_light(self):
        pixels = np.array([[[0., 0, 0], [1., 1, 1]]], dtype=np.float32)
        uvs = np.array([[[1.5, .5]]*3, [[-.5, .5]]*3], dtype=np.float32)
        sampler = RootColorSampler(pixels, uvs)
        self.assertEqual(sampler(0, (1, 0, 0)), (.5, .5, .5))
        self.assertEqual(sampler(1, (1, 0, 0)), (.5, .5, .5))
        gray = np.full((1, 1, 3), .5, dtype=np.float32)
        self.assertAlmostEqual(RootColorSampler(gray, uvs)(0, (1, 0, 0))[0], .21404114)
        self.assertEqual(RootColorSampler(gray, uvs, pixels_are_linear=True)(0, (1, 0, 0)), (.5, .5, .5))

    def test_invalid_roots_and_arrays_fail_before_sampling(self):
        pixels = np.zeros((1, 1, 3), dtype=np.float32)
        uvs = np.zeros((1, 3, 2), dtype=np.float32)
        sampler = RootColorSampler(pixels, uvs)
        for triangle, bary in ((True, (1, 0, 0)), (1, (1, 0, 0)), (0, (1, 1, 0)),
                               (0, (-1, 1, 1)), (0, (float('nan'), 0, 1))):
            with self.assertRaises(ValueError):
                sampler(triangle, bary)
        for bad in (np.full_like(pixels, np.inf), np.full_like(pixels, 2), np.zeros((1, 1)), pixels.astype(np.uint8)):
            with self.assertRaises(ValueError):
                RootColorSampler(bad, uvs)


if __name__ == '__main__':
    unittest.main()
