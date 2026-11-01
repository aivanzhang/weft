"""Analytic ribbon tests; no generated-character or native-render acceptance."""

import hashlib
import json
import math
import unittest

import numpy as np

from scripts.blender_fur_cards import card_geometry


class CombedFurCardTests(unittest.TestCase):
    def setUp(self):
        self.vertices = [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
        self.triangles = [(0, 1, 2), (0, 3, 1)]
        self.weights = [{'root': 1.}, {'arm': 1.}, {'root': .25, 'tip': .75}, {'back': 1.}]
        self.settings = {'count': 3, 'length': .02, 'width': .005, 'seed': 17, 'direction': [1., .5, -1.]}

    def cards(self, **kwargs):
        return card_geometry(self.vertices, self.triangles, self.weights, self.settings, **kwargs)

    def test_exact_legacy_default_fixtures_before_new_profiles(self):
        cases = [(None, '8ca1e218ff1e9ff4875c076dc733f389d46323ac87d3e2dec9daa6722755765a'),
                 ([(0., 0., 1.), (.1, 0., .99498743710662), (0., .1, .99498743710662), (0., 1., 0.)],
                  '10ba9ae79186d1962182f5dfef1d686a35e895a29beb6b503955b1198e04b37b')]
        for normals, expected in cases:
            with self.subTest(normals=normals):
                default = self.cards(normals=normals)
                self.assertEqual(default, self.cards(normals=normals, profile='legacy'))
                encoded = json.dumps(default, sort_keys=True, separators=(',', ':')).encode()
                self.assertEqual(hashlib.sha256(encoded).hexdigest(), expected)
                self.assertNotIn('colors', default)

    def test_same_seed_preserves_root_size_uv_and_skin_samples(self):
        self.settings['count'] = 120
        before, after = self.cards(), self.cards(profile='combed-v1')
        self.assertEqual(after, self.cards(profile='combed-v1'))
        for key in ('normals', 'weights', 'uv'):
            self.assertEqual(before[key], after[key])
        self.assertEqual(len(after['vertices']), 720)
        self.assertEqual(len(after['faces']), 480)
        for index in range(0, 720, 6):
            a = np.asarray(before['vertices'][index:index+6]).reshape(3, 2, 3)
            b = np.asarray(after['vertices'][index:index+6]).reshape(3, 2, 3)
            centers_a, centers_b = a.mean(axis=1), b.mean(axis=1)
            np.testing.assert_allclose(centers_a[0], centers_b[0], atol=2e-16, rtol=0)
            n = np.asarray(after['normals'][index])
            length_a = (centers_a[-1]-centers_a[0])@n/.82
            length_b = (centers_b[-1]-centers_b[0])@n/.35
            self.assertAlmostEqual(length_a, length_b, places=13)
            self.assertAlmostEqual(np.linalg.norm(a[0, 1]-a[0, 0]), np.linalg.norm(b[0, 1]-b[0, 0]), places=13)
        self.assertNotEqual(before['vertices'], after['vertices'])

    def test_combed_cards_lie_along_flow_with_bounded_roll_and_monotonic_taper(self):
        self.settings['count'] = 90
        result = self.cards(profile='combed-v1')
        direction = np.asarray(self.settings['direction'])
        for index in range(0, len(result['vertices']), 6):
            points = np.asarray(result['vertices'][index:index+6]).reshape(3, 2, 3)
            center = points.mean(axis=1)
            normal = np.asarray(result['normals'][index])
            projected = direction-normal*(direction@normal)
            comb = projected/np.linalg.norm(projected)
            across = np.cross(normal, comb)
            side = points[0, 1]-points[0, 0]
            side /= np.linalg.norm(side)
            self.assertGreaterEqual(side@across, math.cos(.15)-1e-12)
            self.assertAlmostEqual(side@comb, 0, places=12)
            self.assertLessEqual(abs(side@normal), math.sin(.15)+1e-12)
            length = (center[2]-center[0])@normal/.35
            self.assertGreaterEqual(length, self.settings['length']*.72-1e-12)
            self.assertLessEqual(length, self.settings['length']*1.17+1e-12)
            for row, t in ((1, .5), (2, 1.)):
                delta = center[row]-center[0]
                self.assertAlmostEqual(delta@normal, length*.35*t, places=12)
                self.assertAlmostEqual(delta@comb, length*.8*t*t, places=12)
            widths = np.linalg.norm(points[:, 1]-points[:, 0], axis=1)
            np.testing.assert_allclose(widths/widths[0], [1, .8/.92, .15/.92], atol=1e-12, rtol=0)
            self.assertGreater(widths[0], widths[1])
            self.assertGreater(widths[1], widths[2])
            self.assertGreater((center[2]-center[0])@comb, (center[2]-center[0])@normal)

    def test_faces_wind_outward_even_with_opposite_normals_or_parallel_comb(self):
        self.vertices, self.triangles = self.vertices[:3], self.triangles[:1]
        self.weights = self.weights[:3]
        self.settings['count'] = 80
        for normal, direction in (((0, 0, 1), (1, 0, 0)), ((0, 0, -1), (1, .5, 0)),
                                  ((0, 0, 1), (0, 0, 1))):
            with self.subTest(normal=normal, direction=direction):
                self.settings['direction'] = direction
                result = self.cards(profile='combed-v1', normals=[normal]*3)
                vertices = np.asarray(result['vertices'])
                self.assertTrue(np.isfinite(vertices).all())
                for a, b, c in result['faces']:
                    cross = np.cross(vertices[b]-vertices[a], vertices[c]-vertices[a])
                    self.assertGreater(float(cross@np.asarray(normal)), 0)
                for index in range(0, len(vertices), 6):
                    self.assertAlmostEqual(np.mean(vertices[index:index+2], axis=0)@np.asarray(normal), -.0005)

    def test_four_normalized_skin_influences_and_uv_layout_preserved(self):
        self.weights = [{'a': .3, 'b': .3, 'c': .2, 'd': .2}, {'e': .2, 'f': .3, 'g': .5},
                        {'a': .2, 'e': .4, 'h': .4}, {'i': 1.}]
        self.settings['count'] = 30
        result = self.cards(profile='combed-v1')
        for index in range(0, len(result['vertices']), 6):
            inherited = result['weights'][index]
            self.assertLessEqual(len(inherited), 4)
            self.assertTrue(all(weight > 0 for weight in inherited.values()))
            self.assertAlmostEqual(sum(inherited.values()), 1)
            self.assertEqual(result['weights'][index:index+6], [inherited]*6)
            self.assertTrue(all(index <= v < index+6 for face in result['faces'][index//6*4:index//6*4+4] for v in face))
            uv = result['uv'][index:index+6]
            self.assertTrue(all(0 < v < 1 for pair in uv for v in pair))
            self.assertEqual(uv[0][0], uv[2][0])
            self.assertEqual(uv[0][0], uv[4][0])
            self.assertEqual(uv[1][0], uv[3][0])
            self.assertLess(uv[0][1], uv[2][1])
            self.assertLess(uv[2][1], uv[4][1])

    def test_root_colors_are_barycentric_constant_per_card_without_new_rng_draws(self):
        self.vertices, self.triangles = self.vertices[:3], self.triangles[:1]
        self.weights = self.weights[:3]
        self.settings['count'] = 30
        colors = np.asarray([(1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.float32)
        original = colors.copy()
        for profile in ('legacy', 'combed-v1'):
            with self.subTest(profile=profile):
                result = self.cards(profile=profile, colors=colors)
                expected = self.cards(profile=profile)
                self.assertEqual({k:v for k, v in result.items() if k != 'colors'}, expected)
                self.assertEqual(len(result['colors']), len(result['vertices']))
                for index in range(0, len(result['vertices']), 6):
                    root = np.mean(result['vertices'][index:index+2], axis=0)
                    color = result['colors'][index]
                    np.testing.assert_allclose(color, [1-root[0]-root[1], root[0], root[1]], atol=2e-16, rtol=0)
                    self.assertEqual(result['colors'][index:index+6], [color]*6)
                    self.assertTrue(all(0 <= channel <= 1 for channel in color))
                    self.assertEqual(color, tuple(color))
        np.testing.assert_array_equal(colors, original)

    def test_zero_count_does_not_add_unrequested_color_output(self):
        self.settings['count'] = 0
        self.assertNotIn('colors', self.cards(profile='combed-v1'))
        colored = self.cards(profile='combed-v1', colors=[(.2, .4, .6)]*len(self.vertices))
        self.assertEqual(colored['colors'], [])
        self.assertTrue(all(not value for value in colored.values()))

    def test_unknown_profiles_and_malformed_rgb_are_rejected(self):
        for profile in ('upright', '', None, [], 1):
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, 'geometry profile'):
                self.cards(profile=profile)
        valid = [(.2, .4, .6)]*len(self.vertices)
        bad = [valid[:-1], valid+valid, [0]*4, [(0, 0)]*4, [(0, 0, 0, 1)]*4,
               [(0, 0, float('nan'))]*4, [(0, float('inf'), 0)]*4, [(-.01, 0, 0)]*4,
               [(0, 1.01, 0)]*4, [(0, True, 0)]*4, [(0, '1', 0)]*4,
               {'red': 0, 'green': 0, 'blue': 0, 'alpha': 1}, '0000', 1, ['000']*4]
        for colors in bad:
            with self.subTest(colors=colors), self.assertRaisesRegex(ValueError, 'root colors'):
                self.cards(profile='combed-v1', colors=colors)


if __name__ == '__main__':
    unittest.main()
