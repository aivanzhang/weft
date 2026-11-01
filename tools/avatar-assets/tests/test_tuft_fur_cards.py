"""Analytic tuft/UV/skin fixtures; these do not establish rendered fur quality."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from scripts.blender_fur_cards import (ATLAS_PROFILES, atlas_pixels, card_geometry, tuft_export_budget)
from scripts.groom_flow import evaluate_prepared_flow, prepare_flow


class TuftFurCardTests(unittest.TestCase):
    def setUp(self):
        self.vertices = [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (1., 1., 0.)]
        self.triangles = [(0, 1, 2), (1, 3, 2)]
        self.weights = [{'a': .3, 'b': .3, 'c': .2, 'd': .2}, {'e': .4, 'a': .6},
                        {'f': .6, 'b': .4}, {'c': .2, 'g': .8}]
        self.settings = {'count': 120, 'length': .015, 'width': .01, 'seed': 51,
                         'direction': [1., 0., 0.], 'comb': .65}

    def cards(self, **kwargs):
        return card_geometry(self.vertices, self.triangles, self.weights, self.settings,
                             profile='tuft-v1', root_color_sampler=kwargs.pop('root_color_sampler', lambda *_: (.1, .3, .6)),
                             **kwargs)

    def field(self):
        return prepare_flow([0, 0, -.5], [1, 1, 1], [1, 0, 0], .65, {'regions': [
            {'name': 'one-side', 'center': [.15, .5, .5], 'radius': [.32, .8, .8],
             'feather': .1, 'direction': [0, 1, 0], 'comb': .8, 'length_scale': 1.5},
            {'name': 'other-side', 'center': [.85, .5, .5], 'radius': [.32, .8, .8],
             'feather': .1, 'direction': [0, -1, 0], 'comb': .45, 'length_scale': .6},
        ]})

    def test_legacy_combed_and_distance_atlas_remain_exact(self):
        vertices = [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
        weights = [{'root': 1.}, {'arm': 1.}, {'root': .25, 'tip': .75}, {'back': 1.}]
        settings = {'count': 3, 'length': .02, 'width': .005, 'seed': 17, 'direction': [1., .5, -1.]}
        value = card_geometry(vertices, [(0, 1, 2), (0, 3, 1)], weights, settings, profile='combed-v1')
        self.assertEqual(hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                         '74c0fa8ccb57529f0e90724937cf65d13011e15b95a6c65c719dd1b830352398')
        self.assertEqual(hashlib.sha256(atlas_pixels([.8, .6, .4, 1], profile='distance-v1').tobytes()).hexdigest(),
                         '5a6d38d1664402db381bbaf61fe76771adb012260af3b80a49b82cbc13635ed8')

    def test_seeded_layers_preserve_six_positions_four_faces_skin_and_root(self):
        result = self.cards()
        self.assertEqual(result, self.cards())
        self.assertEqual((len(result['vertices']), len(result['faces'])), (720, 480))
        self.assertEqual(Counter(r['layer'] for r in result['root_samples']), {'pile': 80, 'tuft': 40})
        points = np.asarray(result['vertices'])
        for root in result['root_samples']:
            index = root['root_index']*6
            rows = points[index:index+6].reshape(3, 2, 3)
            centers = rows.mean(axis=1)
            np.testing.assert_allclose(centers[0], np.asarray(root['position'])-np.asarray(root['normal'])*root['root_inset_m'], atol=2e-16)
            self.assertAlmostEqual(sum(np.linalg.norm(centers[i]-centers[i-1]) for i in (1, 2)), root['length_m'], places=13)
            inherited = result['weights'][index]
            self.assertLessEqual(len(inherited), 4)
            self.assertAlmostEqual(sum(inherited.values()), 1)
            self.assertEqual(result['weights'][index:index+6], [inherited]*6)
            self.assertEqual(result['colors'][index:index+6], [(.1, .3, .6)]*6)
            self.assertTrue(all(0 < value < 1 for pair in result['uv'][index:index+6] for value in pair))
            self.assertTrue(.001 <= root['length_m'] <= .04 and .0003 <= root['width_m'] <= .02)
        for a, b, c in result['faces']:
            cross = np.cross(points[b]-points[a], points[c]-points[a])
            self.assertGreater(np.linalg.norm(cross), 0)
            self.assertGreaterEqual(cross@np.asarray(result['normals'][a]), -1e-18)

    def test_exact_uv_callback_keeps_shared_vertex_uv_islands_distinct(self):
        # The triangles share vertices1/2 but use unrelated red and blue UV islands.
        uvs = np.asarray([[(0, 0), (.4, 0), (0, 1)], [(.6, 0), (1, 1), (.6, 1)]])
        calls = []
        def sample(ordinal, bary):
            uv = np.asarray(bary)@uvs[ordinal]
            calls.append((ordinal, tuple(bary), uv))
            return (.7, .05, .01) if uv[0] < .5 else (.01, .05, .7)
        result = self.cards(root_color_sampler=sample)
        self.assertEqual(len(calls), self.settings['count'])
        self.assertEqual(Counter(c[0] for c in calls), {0: 60, 1: 60})
        for root, (ordinal, bary, uv) in zip(result['root_samples'], calls):
            self.assertEqual(root['triangle_index'], ordinal)
            np.testing.assert_allclose(root['barycentric'], bary)
            expected = (.7, .05, .01) if ordinal == 0 else (.01, .05, .7)
            self.assertEqual(result['colors'][root['root_index']*6], expected)
        self.triangles.insert(0, (0, 0, 0))
        ordinals = []
        self.cards(root_color_sampler=lambda ordinal, bary: ordinals.append(ordinal) or (.1, .2, .3))
        self.assertEqual(set(ordinals), {1, 2})

    def test_regional_direction_comb_and_length_use_existing_flow_math(self):
        field = self.field()
        result = self.cards(prepared_flow=field)
        neutral = self.cards()
        self.assertEqual(result['weights'], neutral['weights'])
        self.assertEqual(result['uv'], neutral['uv'])
        for root, base in zip(result['root_samples'], neutral['root_samples']):
            direction, comb, scale = evaluate_prepared_flow(root['position'], field)
            np.testing.assert_allclose(root['direction'], direction)
            self.assertEqual((root['comb'], root['length_scale']), (comb, scale))
            self.assertEqual(root['position'], base['position'])
            self.assertAlmostEqual(root['length_m']/base['length_m'], scale, places=12)
            centers = np.asarray(result['vertices'][root['root_index']*6:root['root_index']*6+6]).reshape(3, 2, 3).mean(axis=1)
            normal, tangent = np.asarray(root['normal']), np.asarray(direction)
            tangent -= normal*(tangent@normal)
            tangent /= np.linalg.norm(tangent)
            for segment, t in enumerate((.5, 1.)):
                lean = comb*(.30+.70*t)
                expected = normal*(1-lean)+tangent*lean
                expected /= np.linalg.norm(expected)
                actual = centers[segment+1]-centers[segment]
                np.testing.assert_allclose(actual/np.linalg.norm(actual), expected, atol=1e-13)
        self.assertTrue(result['root_summary']['regional_flow'])
        self.assertLess(result['root_summary']['length_scale_range'][0], .7)
        self.assertGreater(result['root_summary']['length_scale_range'][1], 1.4)

    def test_dimensions_clamp_and_parallel_flow_stays_finite_outward(self):
        self.settings.update(length=.04, width=.02, direction=[0, 0, 1], comb=0)
        field = prepare_flow([0, 0, -.5], [1, 1, 1], [0, 0, 1], 0, {'regions': [
            {'name': 'whole', 'center': [.5, .5, .5], 'radius': [2, 2, 2], 'feather': .1,
             'direction': [0, 0, 1], 'comb': 0, 'length_scale': 2}]})
        result = self.cards(prepared_flow=field, normals=[(0, 0, -1)]*4)
        self.assertTrue(np.isfinite(result['vertices']).all())
        self.assertGreater(result['root_summary']['length_clamped'], 0)
        self.assertGreater(result['root_summary']['width_clamped'], 0)
        self.assertLessEqual(result['root_summary']['length_range_m'][1], .04)
        self.assertLessEqual(result['root_summary']['width_range_m'][1], .02)
        points = np.asarray(result['vertices'])
        for a, b, c in result['faces']:
            self.assertGreaterEqual(np.cross(points[b]-points[a], points[c]-points[a])@np.asarray(result['normals'][a]), -1e-18)

    def test_neutral_atlas_has_three_separated_fibers_at_analytic_eight_pixel_width(self):
        pixels = atlas_pixels([1, 1, 1, 1], profile='tuft-v1')
        np.testing.assert_array_equal(pixels, atlas_pixels([1, 1, 1, 1], profile='tuft-v1'))
        self.assertEqual(pixels.shape, (1024, 1024, 4))
        self.assertTrue(np.isfinite(pixels).all())
        self.assertTrue((pixels[:, :, :3] == 1).all())
        self.assertTrue((pixels[::256, :, 3] == 0).all() and (pixels[:, ::256, 3] == 0).all())
        cutoff = ATLAS_PROFILES['tuft-v1']['cutoff']
        self.assertGreater((pixels[:, :, 3] > cutoff).mean(), .35)
        self.assertLess((pixels[:, :, 3] > cutoff).mean(), .55)
        for y in range(0, 1024, 256):
            for x in range(0, 1024, 256):
                tile = pixels[y:y+256, x:x+256, 3]
                row = np.asarray(Image.fromarray(tile).resize((8, 8), Image.Resampling.BOX))[2] > cutoff
                self.assertEqual(np.sum(np.diff(np.r_[False, row, False].astype(int)) == 1), 3)
        with self.assertRaisesRegex(ValueError, 'neutral white'):
            atlas_pixels([.8, .8, .8, 1], profile='tuft-v1')

    def test_combined_export_budget_uses_worst_case_corners_and_never_silently_trims(self):
        report = tuft_export_budget(51137, 3600)
        self.assertEqual(report['projected_max_vertices'], 98433)
        self.assertEqual(report['maximum_cards'], 3730)
        tuft_export_budget(51137, 3730)
        with self.assertRaisesRegex(ValueError, 'aggregate'):
            tuft_export_budget(51137, 3731)
        for values in ((True, 2), (0, 5001), (-1, 1), (100001, 0), (99999, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                tuft_export_budget(*values)

    def test_maximum_card_count_keeps_root_diagnostics_bounded(self):
        self.settings['count'] = 5000
        sampler = Mock(return_value=(.1, .2, .3))
        result = self.cards(root_color_sampler=sampler)
        self.assertEqual(sampler.call_count, 5000)
        self.assertEqual((len(result['vertices']), len(result['faces'])), (30000, 20000))
        self.assertEqual(len(result['root_samples']), 128)
        self.assertEqual([result['root_samples'][i]['root_index'] for i in (0, -1)], [0, 4999])
        self.assertEqual(result['root_summary']['roots'], 5000)

    def test_bad_settings_emitter_or_sampler_fail_without_partial_scene_mutation(self):
        original = deepcopy(self.settings)
        sampler = Mock(return_value=(.1, .2, .3))
        for field, value in (('count', True), ('count', 5001), ('count', -1), ('length', .0009),
                             ('length', .041), ('width', .0201), ('direction', [0, 0, 0]),
                             ('comb', .96), ('seed', -1), ('length', float('nan')), ('width', True)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.settings = {**original, field: value}
                self.cards(root_color_sampler=sampler)
        self.settings = original
        self.assertEqual(sampler.call_count, 0)
        for color in ((float('nan'), 0, 0), (1.1, 0, 0), (True, 0, 0), (0, 0), 'rgb'):
            with self.subTest(color=color), self.assertRaises(ValueError):
                self.cards(root_color_sampler=lambda *_: color)
        with self.assertRaisesRegex(ValueError, 'exact root UV sampler'):
            self.cards(colors=[(0, 0, 0)]*4)
        with self.assertRaisesRegex(ValueError, 'prepare_flow'):
            self.cards(prepared_flow={'regions': []})
        self.triangles = [(0, 1, 4)]
        with self.assertRaisesRegex(ValueError, 'triangle'):
            self.cards()
        self.triangles = [(0, 1, 2)]
        self.vertices[0] = (10**1000, 0, 0)
        with self.assertRaisesRegex(ValueError, 'finite tuft'):
            self.cards()

    def test_new_inputs_are_not_silently_applied_to_old_profiles(self):
        for profile in ('legacy', 'combed-v1'):
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, 'require tuft-v1'):
                card_geometry(self.vertices, self.triangles, self.weights, self.settings, profile=profile,
                              root_color_sampler=lambda *_: (0, 0, 0))

    def test_oversized_emitters_and_invalid_skin_are_rejected_before_sampling(self):
        class Oversized:
            def __len__(self): return 100001
            def __iter__(self): raise AssertionError('Oversized emitter must not be traversed')
        sampler = Mock(return_value=(.1, .2, .3))
        original = self.vertices
        self.vertices = Oversized()
        with self.assertRaisesRegex(ValueError, 'limit exceeded'):
            self.cards(root_color_sampler=sampler)
        self.vertices = original
        self.weights[0] = {'a': float('nan')}
        with self.assertRaisesRegex(ValueError, 'skin weight'):
            self.cards(root_color_sampler=sampler)
        self.assertEqual(sampler.call_count, 0)


if __name__ == '__main__':
    unittest.main()
