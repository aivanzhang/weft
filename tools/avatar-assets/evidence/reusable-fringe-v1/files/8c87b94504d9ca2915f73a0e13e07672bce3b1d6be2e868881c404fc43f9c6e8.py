"""Analytic sampling/packing evidence only; no generated-avatar quality claim."""

import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from scripts.fringe_color_atlas import (SOURCE_POLICY, build_color_atlas, encode_srgb,
                                      sample_alpha, sample_source_rgb)


def fixture(count=1, *, spans=None):
    spans = np.full(count, .013) if spans is None else np.asarray(spans)
    positions = np.zeros((count, 4, 3), dtype=np.float32)
    positions[:, [1, 3], 0] = spans[:, None]
    positions[:, [2, 3], 2] = .008
    uv = np.empty((count, 4, 2), dtype=np.float64)
    uv[:, [0, 2], 0] = (.5-spans/(2*.0208))[:, None]
    uv[:, [1, 3], 0] = (.5+spans/(2*.0208))[:, None]
    uv[:, [0, 1], 1] = .985
    uv[:, [2, 3], 1] = .015
    ids = np.arange(count)
    uv = (uv+np.column_stack((ids % 4, ids//4 % 4))[:, None])/4
    weights = np.tile([.3, .2, .1, .4], (count*4, 1)).astype(np.float32)
    fins = {'report': {'profile': SOURCE_POLICY}, 'positions': positions.reshape(-1, 3),
            'uv': uv.reshape(-1, 2).astype(np.float32), 'root_source_uv': np.tile([[.03, .5], [.97, .5]], (count, 1, 1)),
            'weights': weights, 'joints': np.tile([0, 1, 2, 3], (count*4, 1)).astype(np.uint16),
            'normals': np.tile([0., 0., 1.], (count*4, 1))}
    # Independent physical two-tip alpha: two 1.6mm-wide bands in the 20.8mm tile domain.
    x = ((np.arange(256)+.5)/256-.5)*.0208
    alpha_row = (np.minimum(abs(x-.0025), abs(x+.0025)) <= .0008).astype(np.uint8)*255
    silhouette = np.tile(np.broadcast_to(alpha_row, (256, 256)), (4, 4))
    return fins, np.zeros(count, dtype=np.int32), {0: {'pixels': np.full((1, 1, 3), 120, np.uint8), 'factor': np.ones(3)}}, silhouette


def fingerprints(fins, materials, textures, silhouette):
    arrays = [v for v in fins.values() if isinstance(v, np.ndarray)]+[materials, silhouette]
    arrays += [v for texture in textures.values() for v in texture.values()]
    return [hashlib.sha256(v.tobytes()).hexdigest() for v in arrays]


def cell(output, source):
    slot = int(output['permutation']['destination_for_source_fin'][source])
    columns = output['report']['cell_grid'][0]
    return output['pixels'][(slot//columns)*32:(slot//columns+1)*32, (slot % columns)*64:(slot % columns+1)*64]


def linear(pixels):
    values = pixels.astype(np.float64)/255
    return np.where(values <= .04045, values/12.92, ((values+.055)/1.055)**2.4)


def box_level(pixels, levels):
    """Modeled linear-light box levels; not a measured GPU mip generator."""
    for _ in range(levels):
        height, width = pixels.shape[:2]
        rgba = np.empty_like(pixels, dtype=np.float64)
        rgba[..., :3] = linear(pixels[..., :3])
        rgba[..., 3] = pixels[..., 3]/255
        averaged = rgba.reshape(height//2, 2, width//2, 2, 4).mean(axis=(1, 3))
        pixels = np.empty(averaged.shape, dtype=np.uint8)
        pixels[..., :3] = encode_srgb(averaged[..., :3])
        pixels[..., 3] = np.rint(255*averaged[..., 3]).astype(np.uint8)
    return pixels


class SourceSamplerTests(unittest.TestCase):
    def test_two_by_two_centers_seams_negative_wrap_and_top_down_orientation(self):
        pixels = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], np.uint8)
        cases = [((.25, .25), (1, 0, 0)), ((.75, .25), (0, 1, 0)),
                 ((.25, .75), (0, 0, 1)), ((.75, .75), (1, 1, 1)),
                 ((0, 0), (.5, .5, .5)), ((1, 1), (.5, .5, .5)),
                 ((.5, .25), (.5, .5, 0)), ((-.25, .25), (0, 1, 0)), ((1.25, .25), (1, 0, 0))]
        actual = sample_source_rgb(pixels, np.array([uv for uv, _ in cases]), np.ones(3))
        np.testing.assert_array_equal(actual, [rgb for _, rgb in cases])
        uv = np.array([[.25, .25], [.75, .75]])
        paired = uv.copy(); paired[:, 1] = 1-paired[:, 1]
        np.testing.assert_array_equal(sample_source_rgb(pixels, uv, np.ones(3)),
                                      sample_source_rgb(pixels[::-1], paired, np.ones(3)))

    def test_decode_each_texel_before_filter_then_apply_linear_factor_alpha_is_not_color(self):
        pixels = np.array([[[128, 128, 128, 0], [255, 255, 255, 0]]], np.uint8)
        result = sample_source_rgb(pixels, np.array([[.5, .5]]), np.array([.5, .25, 1.]))
        expected = .6079302500569496
        np.testing.assert_allclose(result, [[expected*.5, expected*.25, expected]], atol=1e-14)
        self.assertGreater(abs(result[0, 2]-(((128/255+1)/2+.055)/1.055)**2.4), .08)

    def test_alpha_clamps_instead_of_repeating(self):
        alpha = np.array([[0, 255]], np.uint8)
        np.testing.assert_array_equal(sample_alpha(alpha, np.array([[-1., .5], [0., .5], [1., .5], [2., .5]])),
                                      [0, 0, 1, 1])

    def test_sampling_rejects_bad_arrays_before_allocating_result(self):
        good = np.zeros((1, 1, 3), np.uint8)
        for pixels, uv, factor in ((good.astype(float), np.array([[0., 0.]]), np.ones(3)),
                                   (good, np.zeros((262145, 2)), np.ones(3)),
                                   (good, np.array([[np.nan, 0.]]), np.ones(3)),
                                   (good, np.array([[1e7, 0.]]), np.ones(3)),
                                   (good, np.array([[0, 0]]), np.ones(3)),
                                   (good, np.array([[0., 0.]]), np.array([1., -.1, 1.])),
                                   (good, np.array([[0., 0.]]), np.array([1., np.inf, 1.]))):
            with self.subTest(shape=uv.shape), self.assertRaises(ValueError):
                sample_source_rgb(pixels, uv, factor)


class ColorAtlasTests(unittest.TestCase):
    def test_stripes_retain_interior_markings_not_endpoint_interpolation(self):
        args = fixture(9, spans=np.linspace(.0085, .02, 9))
        fins, ids, textures, _ = args
        stripes = np.tile(np.array([0, 0, 255, 255], dtype=np.uint8), 32)
        textures[0]['pixels'] = np.broadcast_to(stripes[None, :, None], (8, 128, 3))
        output = build_color_atlas(*args)
        t = (np.arange(60)+.5)/60
        query = fins['root_source_uv'][0, 0]+t[:, None]*(fins['root_source_uv'][0, 1]-fins['root_source_uv'][0, 0])
        truth = sample_source_rgb(textures[0]['pixels'], query, np.ones(3))
        actual = linear(cell(output, 0)[16, 2:62, :3])
        self.assertLess(np.max(abs(actual-truth)), .0046)
        self.assertGreater(actual.max()-actual.min(), .99)
        self.assertFalse(output['report']['quality_approved'])
        self.assertEqual(output['report']['nominal_tip_audit']['tip_count'], 18)

    def test_colored_gradient_and_second_material_preserve_factor_and_rows(self):
        fins, ids, textures, silhouette = fixture(7)
        x = np.linspace(0, 1, 64)[None, :]
        y = np.linspace(0, 1, 32)[:, None]
        gradient = np.empty((32, 64, 3)); gradient[..., 0] = x; gradient[..., 1] = y; gradient[..., 2] = 1-x
        textures[0]['pixels'] = encode_srgb(gradient)
        textures[3] = {'pixels': np.full((2, 2, 4), [30, 180, 240, 0], np.uint8), 'factor': np.array([1., .5, .25])}
        ids[1::2] = 3
        fins['root_source_uv'][:, :, 1] = np.linspace(.1, .9, 7)[:, None]
        before = fingerprints(fins, ids, textures, silhouette)
        output = build_color_atlas(fins, ids, textures, silhouette)
        self.assertEqual(before, fingerprints(fins, ids, textures, silhouette))
        self.assertTrue(output['report']['technical_gate_passed'])
        self.assertGreater(output['nominal_tips']['source_rgb'][6, 0, 1], output['nominal_tips']['source_rgb'][0, 0, 1])
        expected = linear(np.array([30, 180, 240], np.uint8))*np.array([1, .5, .25])
        np.testing.assert_allclose(output['nominal_tips']['source_rgb'][1], [expected, expected], atol=1e-14)
        self.assertEqual(output['report']['source_material_ids'], [0, 3])

    def test_source_edges_cross_repeat_seams_without_modulo_of_endpoints(self):
        fins, ids, textures, silhouette = fixture(2)
        textures[0]['pixels'] = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]], np.uint8)
        fins['root_source_uv'][0] = [[-.125, .5], [.375, .5]]
        fins['root_source_uv'][1] = [[.875, .5], [1.375, .5]]
        fins['uv'].reshape(2, 4, 2)[1] = fins['uv'].reshape(2, 4, 2)[0]
        original = fins['root_source_uv'].copy()
        output = build_color_atlas(fins, ids, textures, silhouette)
        np.testing.assert_array_equal(cell(output, 0), cell(output, 1))
        # At the edge midpoint raw interpolation reaches red, not the blue between wrapped endpoints.
        midpoint = linear(cell(output, 0)[16, [31, 32], :3]).mean(axis=0)
        self.assertGreater(midpoint[0], .98)
        self.assertLess(midpoint[2], .02)
        wrong_uv = np.mod(original[0], 1).mean(axis=0)[None]
        np.testing.assert_array_equal(sample_source_rgb(textures[0]['pixels'], wrong_uv, np.ones(3)), [[0, 0, 1]])
        np.testing.assert_array_equal(fins['root_source_uv'], original)
        self.assertTrue(np.all((output['uv'] >= 0)&(output['uv'] <= 1)))

    def test_packing_keeps_exact_cells_geometry_order_and_lod0_samples(self):
        args = fixture(71)
        fins, _, textures, _ = args
        x = np.linspace(0, 255, 256).astype(np.uint8)
        textures[0]['pixels'] = np.broadcast_to(np.stack([x, x[::-1], x], axis=1)[None], (4, 256, 3))
        centers = np.linspace(.01, .99, 71)[np.random.default_rng(9).permutation(71)]
        fins['root_source_uv'][:, :, 0] = centers[:, None]
        before = fingerprints(*args)
        unpacked = build_color_atlas(*args, pack_by_color=False)
        packed = build_color_atlas(*args, pack_by_color=True)
        replay = build_color_atlas(*args, pack_by_color=True, batch_fins=7)
        permutation = packed['permutation']
        np.testing.assert_array_equal(permutation['destination_for_source_fin'][permutation['source_fin_at_destination']], np.arange(71))
        self.assertFalse(np.array_equal(permutation['source_fin_at_destination'], np.arange(71)))
        for source in range(71):
            np.testing.assert_array_equal(cell(unpacked, source), cell(packed, source))
        np.testing.assert_array_equal(packed['pixels'], replay['pixels'])
        np.testing.assert_array_equal(packed['uv'], replay['uv'])
        np.testing.assert_allclose(packed['nominal_tips']['atlas_rgb'], unpacked['nominal_tips']['atlas_rgb'], atol=2e-6, rtol=0)
        np.testing.assert_allclose(packed['nominal_tips']['alpha'], unpacked['nominal_tips']['alpha'], atol=2e-6, rtol=0)
        self.assertEqual(before, fingerprints(*args))
        self.assertEqual(packed['uv'].dtype, np.float32)
        self.assertEqual(packed['report']['unused_cells'], 57)
        self.assertFalse(packed['pixels'][32:, 7*64:].any())

    def test_physical_crop_keeps_two_tips_and_width_instead_of_stretching_each_fin(self):
        args = fixture(3, spans=[.0085, .013, .020])
        output = build_color_atlas(*args)
        t = (np.arange(4096)+.5)/4096
        for index, span in enumerate((.0085, .013, .020)):
            crop = args[0]['uv'].reshape(3, 4, 2)[index].astype(float)
            assigned = output['uv'].reshape(3, 4, 2)[index].astype(float)
            widths = []
            for pixels, uv in ((args[3], crop), (output['pixels'][..., 3], assigned)):
                query = uv[0]+t[:, None]*(uv[1]-uv[0]); query[:, 1] = (uv[0, 1]+uv[2, 1])*.5
                mask = sample_alpha(pixels, query) >= .18
                diff = np.diff(np.r_[False, mask, False].astype(int))
                starts, stops = np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)
                self.assertEqual(len(starts), 2)
                widths.append((stops-starts)*span/4096)
            self.assertLessEqual(np.max(abs(widths[0]-widths[1])), 2*span/60)

    def test_failed_color_and_alpha_are_retained_not_dropped_or_whitened(self):
        fins, ids, textures, silhouette = fixture()
        # A narrow source mark between the 60 resampling centers defeats this finite atlas.
        textures[0]['pixels'] = np.zeros((1, 4096, 3), np.uint8)
        fraction = .5-.0025/.013
        fins['root_source_uv'][0] = [[0, .5], [1, .5]]
        x = int(np.floor(fraction*4096))
        textures[0]['pixels'][0, x-1:x+2] = 255
        output = build_color_atlas(fins, ids, textures, silhouette)
        audit = output['report']['nominal_tip_audit']
        self.assertFalse(audit['color_gate_passed'])
        self.assertTrue(audit['alpha_gate_passed'])
        self.assertGreater(audit['maximum_linear_component_error'], .9)
        self.assertEqual(audit['tip_count'], 2)
        self.assertEqual(audit['omitted_or_hidden_samples'], 0)
        self.assertEqual(output['nominal_tips']['error'].shape, (1, 2))
        blank = build_color_atlas(fins, ids, textures, np.zeros_like(silhouette))
        self.assertFalse(blank['report']['technical_gate_passed'])
        self.assertEqual(blank['report']['nominal_tip_audit']['tips_below_mask_cutoff'], 2)

    def test_arbitrary_counts_padding_and_shared_pixel_budget(self):
        for count in (1, 63, 64, 65):
            with self.subTest(count=count):
                fins, ids, textures, silhouette = fixture(count)
                textures[1] = {'pixels': textures[0]['pixels'], 'factor': np.array([.8, .8, .8])}
                ids[::2] = 1
                output = build_color_atlas(fins, ids, textures, silhouette, padding=4)
                self.assertEqual(output['report']['source_pixels'], 1)
                self.assertEqual(output['report']['nominal_tip_audit']['tip_count'], 2*count)
                for source in range(count):
                    rgba = cell(output, source)
                    self.assertFalse(rgba[:4, :, 3].any() or rgba[-4:, :, 3].any()
                                     or rgba[:, :4, 3].any() or rgba[:, -4:, 3].any())

    def test_three_material_image_pixel_sum_is_allowed_below_shared_cap(self):
        fins, ids, _, silhouette = fixture(3)
        textures = {slot: {'pixels': np.broadcast_to(np.array([[[50+slot*80, 40, 90]]], np.uint8), (1536, 2048, 3)),
                           'factor': np.ones(3)} for slot in range(3)}
        ids[:] = np.arange(3)
        output = build_color_atlas(fins, ids, textures, silhouette)
        self.assertEqual(output['report']['source_pixels'], 9_437_184)
        self.assertTrue(output['report']['technical_gate_passed'])

    def test_modeled_coarse_filter_neighbor_contamination_is_reduced_not_guaranteed_eliminated(self):
        # The model is BOX→bilinear, not an actual GPU/anisotropic visibility test.
        oracle = np.array([[[0, 0, 0, 0], [255, 255, 255, 255]], [[0, 0, 0, 0], [255, 255, 255, 255]]], np.uint8)
        np.testing.assert_array_equal(box_level(oracle, 1)[0, 0], [188, 188, 188, 128])
        fins, ids, textures, silhouette = fixture(64)
        textures[0]['pixels'] = np.array([[[0, 0, 0], [255, 255, 255]]], np.uint8)
        fins['root_source_uv'][:, :, 0] = np.where(np.arange(64) % 2, .75, .25)[:, None]
        fins['root_source_uv'][:, :, 1] = .5
        results = []
        for packed in (False, True):
            output = build_color_atlas(fins, ids, textures, silhouette, pack_by_color=packed)
            mip = box_level(output['pixels'], 5)
            uv = output['uv'].reshape(64, 4, 2).astype(float)
            # At the physical root-edge endpoint, LOD5 can read a neighboring cell.
            query = uv[:, 0].copy(); query[:, 1] = (uv[:, 0, 1]+uv[:, 2, 1])*.5
            self.assertEqual(mip.shape[0], 1)
            # np.interp clamps at either end, matching the atlas rather than source REPEAT.
            luminance = np.interp(query[:, 0]*mip.shape[1]-.5, np.arange(mip.shape[1]), linear(mip[0, :, 0]))
            dark = np.arange(64) % 2 == 0
            results.append(np.count_nonzero(luminance[dark] > .05))
        self.assertGreater(results[0], 20)
        self.assertLess(results[1], results[0]/4)

    def test_invalid_settings_and_topology_contract_reject_before_atlas_allocation(self):
        for changes in ({'cell_width': 32}, {'cell_height': 16}, {'padding': 1}, {'padding': True},
                        {'batch_fins': 129}, {'batch_fins': False}, {'pack_by_color': 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_color_atlas(*fixture(), **changes)
        mutations = [lambda a: a[0]['report'].update(profile='unknown'),
                     lambda a: a[0]['weights'].__setitem__((0, 0), np.nan),
                     lambda a: a[0]['weights'].__setitem__((0, 0), -.1),
                     lambda a: a[0]['joints'].__setitem__((0, 0), 64),
                     lambda a: a[0]['positions'].__setitem__((1, 0), .1),
                     lambda a: a[0]['uv'].__setitem__((0, 0), .2),
                     lambda a: a[0]['root_source_uv'].__setitem__((0, 0, 0), -1e6-1),
                     lambda a: a[1].__setitem__(0, 10),
                     lambda a: a[2][0].update(factor=np.array([1., 2., 1.])),
                     lambda a: a[2].__setitem__(True, {'pixels': np.zeros((1, 1, 3), np.uint8), 'factor': np.ones(3)})]
        for mutate in mutations:
            args = fixture(); mutate(args)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                build_color_atlas(*args)
        with self.assertRaises(ValueError):
            build_color_atlas(*fixture(4801))
        args = fixture(); args[2][0]['pixels'] = np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (4096, 4096, 3))
        args[2][1] = {'pixels': np.zeros((1, 1, 3), np.uint8), 'factor': np.ones(3)}
        with self.assertRaisesRegex(ValueError, 'budget'):
            build_color_atlas(*args)

    def test_internal_deadline_stops_before_large_output(self):
        with patch('scripts.fringe_color_atlas.time.monotonic', side_effect=[0., 21.]), self.assertRaisesRegex(ValueError, 'deadline'):
            build_color_atlas(*fixture())

    def test_maximum_size_remains_bounded_and_reports_every_tip(self):
        args = fixture(4800)
        args[2][0]['pixels'] = np.full((4096, 4096, 4), 120, np.uint8)
        output = build_color_atlas(*args)
        self.assertEqual(output['pixels'].shape, (2400, 4096, 4))
        self.assertEqual(output['pixels'].nbytes, 39_321_600)
        self.assertEqual(output['report']['nominal_tip_audit']['tip_count'], 9600)
        self.assertTrue(output['report']['technical_gate_passed'])
        self.assertEqual(output['report']['source_pixels'], 16_777_216)
        self.assertLess(output['report']['elapsed_seconds'], 20)
        self.assertEqual(output['report']['unused_cells'], 0)


if __name__ == '__main__':
    unittest.main()
