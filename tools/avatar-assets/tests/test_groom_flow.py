"""Analytic grooming-field checks; no character, render or physical claims."""

from copy import deepcopy
import math
import unittest
from unittest.mock import patch

from scripts.groom_flow import evaluate_flow, evaluate_prepared_flow, prepare_flow, validate_flow


def region(name='field-a', **changes):
    return {'name': name, 'center': [.5, .5, .5], 'radius': [.2, .2, .2], 'feather': .5,
            'direction': [1, 0, 0], 'comb': .8, 'length_scale': 1.6, **changes}


def evaluate(root, regions, base_direction=(0, 0, -1), base_comb=.2):
    return evaluate_flow(root, [0]*3, [1]*3, base_direction, base_comb, {'regions': regions})


class GroomFlowTests(unittest.TestCase):
    def test_outside_preserves_exact_normalized_baseline(self):
        result = evaluate([0, 0, 0], [region()])
        self.assertEqual(result, ((0., 0., -1.), .2, 1.))
        result = evaluate([0, 0, 0], [region()], base_direction=(.3, .4, 0))
        self.assertEqual(result, ((.6, .8, 0.), .2, 1.))

    def test_full_interior_and_smooth_feather_boundaries(self):
        field = region()
        self.assertEqual(evaluate([.5, .5, .5], [field]), ((1., 0., 0.), .8, 1.6))
        inside = evaluate([.7, .5, .5], [field])
        outside = evaluate([.8, .5, .5], [field])
        self.assertEqual(inside, ((1., 0., 0.), .8, 1.6))
        self.assertEqual(outside, ((0., 0., -1.), .2, 1.))
        midpoint = evaluate([.75, .5, .5], [field])
        self.assertAlmostEqual(midpoint[1], .5)
        self.assertAlmostEqual(midpoint[2], 1.3)
        self.assertAlmostEqual(midpoint[0][0], math.sqrt(.5))
        self.assertAlmostEqual(midpoint[0][2], -math.sqrt(.5))
        near_inside = evaluate([.700001, .5, .5], [field])
        near_outside = evaluate([.799999, .5, .5], [field])
        self.assertLess(math.dist(near_inside[0], inside[0]), 1e-8)
        self.assertLess(math.dist(near_outside[0], outside[0]), 1e-8)

    def test_opposing_cancellation_keeps_previous_direction(self):
        direction, comb, scale = evaluate([.75, .5, .5], [region(direction=[-1, 0, 0])], base_direction=(1, 0, 0))
        self.assertEqual(direction, (1., 0., 0.))
        self.assertAlmostEqual(comb, .5)
        self.assertAlmostEqual(scale, 1.3)
        self.assertEqual(evaluate([.5, .5, .5], [region(direction=[-1, 0, 0])],
                                  base_direction=(1, 0, 0))[0], (-1., 0., 0.))

    def test_distinct_regions_are_subject_independent_and_ordered(self):
        left = region('left-zone', center=[.2, .5, .5], radius=[.1]*3, direction=[0, 1, 0],
                      comb=.35, length_scale=.25)
        right = region('right-zone', center=[.8, .5, .5], radius=[.1]*3, direction=[0, -1, 0],
                       comb=.95, length_scale=2)
        self.assertEqual(evaluate([.2, .5, .5], [left, right]), ((0., 1., 0.), .35, .25))
        self.assertEqual(evaluate([.8, .5, .5], [left, right]), ((0., -1., 0.), .95, 2.))
        overlapping = region('later', direction=[0, 1, 0], comb=.1, length_scale=.5)
        self.assertEqual(evaluate([.5]*3, [region(), overlapping]), ((0., 1., 0.), .1, .5))
        self.assertEqual(evaluate([.5]*3, [overlapping, region()]), ((1., 0., 0.), .8, 1.6))

    def test_ordered_interpolation_stays_finite_normalized_and_bounded(self):
        fields = [region(f'zone-{index}', center=[.1+index*.06, .5, .5], radius=[.12]*3,
                         direction=[(-1)**index, .2, .3], comb=.95 if index % 2 else 0,
                         length_scale=2 if index % 2 else .25) for index in range(12)]
        prepared = prepare_flow([0]*3, [1]*3, [0, 0, -1], .5, {'regions': fields})
        for step in range(101):
            direction, comb, scale = evaluate_prepared_flow([step/100, .5, .5], prepared)
            self.assertAlmostEqual(math.hypot(*direction), 1)
            self.assertTrue(all(math.isfinite(value) for value in (*direction, comb, scale)))
            self.assertTrue(0 <= comb <= .95)
            self.assertTrue(.25 <= scale <= 2)

    def test_preparation_detaches_config_and_avoids_per_root_validation(self):
        settings = {'regions': [region()]}
        original = deepcopy(settings)
        validated = validate_flow(settings)
        validated['regions'][0]['direction'][0] = 0
        self.assertEqual(settings, original)
        prepared = prepare_flow([2, 3, 4], [4, 2, 6], [0, 0, -1], .2, settings)
        settings['regions'][0]['direction'][:] = [0, 1, 0]
        with patch('scripts.groom_flow.validate_flow', side_effect=AssertionError('Repeated configuration validation')):
            self.assertEqual(evaluate_prepared_flow([4, 4, 7], prepared), ((1., 0., 0.), .8, 1.6))

    def test_invalid_injected_nonfinite_and_oversized_configs_are_rejected(self):
        bad_configs = [{}, {'regions': []}, {'regions': [region()]*13}, {'regions': [region(), region()]},
                       {'regions': [region()], 'script': 'execute'},
                       {'regions': [region(name='../bad')]}, {'regions': [region(script='execute')]},
                       {'regions': [region(radius=[.001, 1, 1])]},
                       {'regions': [region(center=[-1.1, 0, 0])]}, {'regions': [region(feather=.049)]},
                       {'regions': [region(direction=[0, 0, 0])]}, {'regions': [region(direction=[2, 0, 0])]},
                       {'regions': [region(length_scale=.24)]}, {'regions': [region(length_scale=2.01)]},
                       {'regions': [region(length_scale=10**1000)]},
                       {'regions': [region(comb=.951)]}]
        for field in ('feather', 'comb', 'length_scale'):
            for value in (True, float('nan'), float('inf')):
                bad_configs.append({'regions': [region(**{field: value})]})
        for field in ('center', 'radius', 'direction'):
            for value in (False, float('nan'), float('inf')):
                bad_configs.append({'regions': [region(**{field: [value, .5, .5]})]})
        for settings in bad_configs:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_flow(settings)

    def test_invalid_base_bounds_and_roots_fail_explicitly(self):
        fields = {'regions': [region()]}
        for low, span, direction, comb in (([0]*3, [1, 0, 1], [0, 0, -1], .2),
                                            ([0]*3, [1]*3, [0, 0, 0], .2),
                                            ([0]*3, [1]*3, [0, 0, -1], True),
                                            ([float('nan'), 0, 0], [1]*3, [0, 0, -1], .2)):
            with self.subTest(span=span, direction=direction), self.assertRaises(ValueError):
                prepare_flow(low, span, direction, comb, fields)
        prepared = prepare_flow([0]*3, [1]*3, [0, 0, -1], .2, fields)
        for root in ([0, 0], [True, 0, 0], [float('nan'), 0, 0], [float('inf'), 0, 0], 'bad'):
            with self.subTest(root=root), self.assertRaises(ValueError):
                evaluate_prepared_flow(root, prepared)


if __name__ == '__main__':
    unittest.main()
