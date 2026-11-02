"""Analytic texture seams and exact roots; not generated-fur quality claims."""

import unittest
import numpy as np

from scripts.fur_root_sampling import ROOT_COLOR_POLICY, RootColorSampler, validate_root_sampling_contract


class RootSamplerTests(unittest.TestCase):
    def test_two_by_two_centers_and_periodic_boundaries_have_independent_expected_colors(self):
        pixels = np.array([[[1., 0, 0], [0, 1., 0]], [[0, 0, 1.], [1., 1., 1.]]])
        cases = [((.25, .25), (1, 0, 0)), ((.75, .25), (0, 1, 0)),
                 ((.25, .75), (0, 0, 1)), ((.75, .75), (1, 1, 1)),
                 ((0, 0), (.5, .5, .5)), ((1, 1), (.5, .5, .5)),
                 ((.5, .25), (.5, .5, 0)), ((-.25, .25), (0, 1, 0)),
                 ((1.25, .25), (1, 0, 0))]
        uvs = np.array([[uv]*3 for uv, _ in cases])
        sampler = RootColorSampler(pixels, uvs)
        for i, (uv, expected) in enumerate(cases):
            with self.subTest(uv=uv):
                self.assertEqual(sampler(i, (1, 0, 0)), expected)

    def test_paired_row_and_uv_orientation_preserves_the_same_physical_texels(self):
        top_down = np.array([[[1., 0, 0], [0, 1., 0]], [[0, 0, 1.], [1., 1., 1.]]])
        gltf_uv = np.array([[[.25, .25]]*3, [[.75, .75]]*3])
        blender_uv = np.array([[[.25, .75]]*3, [[.75, .25]]*3])
        gltf = RootColorSampler(top_down, gltf_uv)
        native = RootColorSampler(top_down[::-1], blender_uv)
        wrong = RootColorSampler(top_down, blender_uv)
        for index, expected in enumerate(((1., 0, 0), (1., 1., 1.))):
            self.assertEqual(gltf(index, (1, 0, 0)), expected)
            self.assertEqual(native(index, (1, 0, 0)), expected)
            self.assertNotEqual(wrong(index, (1, 0, 0)), expected)

    def test_mid_gray_is_decoded_before_filtering_and_linear_input_is_not_decoded(self):
        pixels = np.array([[[.5, .5, .5], [1., 1., 1.]], [[0., 0, 0], [0., 0, 0]]])
        uvs = np.array([[[.5, .25]]*3])
        encoded = RootColorSampler(pixels, uvs)(0, (1, 0, 0))
        self.assertAlmostEqual(encoded[0], .6070205702411163, places=12)
        self.assertGreater(abs(encoded[0]-.5225215539683921), .08)
        self.assertEqual(RootColorSampler(pixels, uvs, pixels_are_linear=True)(0, (1, 0, 0)), (.75, .75, .75))

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


class RootSamplingContractTests(unittest.TestCase):
    def source(self, slot=0, **changes):
        return {'slot': slot, 'interpolation': 'Linear', 'extension': 'REPEAT',
                'projection': 'FLAT', 'vector_linked': False, 'image_matches': True, **changes}

    def test_versioned_contract_records_every_validated_material_without_mutating_inputs(self):
        sources = [self.source(2), self.source(0)]
        before = [dict(row) for row in sources]
        report = validate_root_sampling_contract(sources, 'BodyUV', 'BodyUV')
        self.assertEqual(report['policy'], ROOT_COLOR_POLICY)
        self.assertEqual(report['policy'], 'repeat-texel-centers-linear-lod0-srgb-v1')
        self.assertEqual(report['material_slots'], [0, 2])
        self.assertEqual(sources, before)

    def test_unsupported_filter_in_either_material_is_rejected(self):
        for mode in ('Closest', 'Cubic', 'Smart'):
            for index in (0, 1):
                sources = [self.source(0), self.source(1)]
                sources[index]['interpolation'] = mode
                with self.subTest(mode=mode, index=index), self.assertRaisesRegex(ValueError, 'Linear/REPEAT'):
                    validate_root_sampling_contract(sources, 'UVMap', 'UVMap')

    def test_unsupported_wrap_projection_image_or_uv_route_is_rejected(self):
        for changes in ({'extension': 'EXTEND'}, {'extension': 'CLIP'}, {'extension': 'MIRROR'},
                        {'projection': 'BOX'}, {'vector_linked': True}, {'image_matches': False},
                        {'vector_linked': 0}, {'image_matches': 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_root_sampling_contract([self.source(**changes)], 'UVMap', 'UVMap')
        for sampled, rendered in (('UV0', 'UV1'), ('UV0', None), (None, None), ('', ''), ('x'*129, 'x'*129)):
            with self.subTest(sampled=sampled, rendered=rendered), self.assertRaisesRegex(ValueError, 'UV'):
                validate_root_sampling_contract([self.source()], sampled, rendered)

    def test_invalid_or_oversized_material_descriptors_reject(self):
        for sources in ([], [self.source(i) for i in range(33)], [self.source(), self.source()],
                        [self.source(True)], [self.source(32)], [self.source(unexpected=1)], [None]):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                validate_root_sampling_contract(sources, 'UVMap', 'UVMap')


if __name__ == '__main__':
    unittest.main()
