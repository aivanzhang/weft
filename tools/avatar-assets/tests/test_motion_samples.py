"""Analytic motion sampling checks, not generated animation acceptance."""

from copy import deepcopy
import math
import unittest

from scripts.motion_samples import sample_action


def action(seconds, angles=None):
    angles = angles if angles is not None else [0]*len(seconds)
    return {'id': 'analytic-wave', 'keys': [
        {'second': second, 'rotations': {'arm': [0, angle, 0]}}
        for second, angle in zip(seconds, angles)]}


class MotionSampleTests(unittest.TestCase):
    def test_seven_authored_keys_include_short_wave_extrema(self):
        seconds = [0, .5, 1, 1.5, 2.5, 3, 4]
        design = action(seconds, [0, 10, 0, 35, -35, 10, 0])
        original = deepcopy(design)
        result = sample_action(design, (0, 96))
        self.assertEqual(result['seconds'], seconds)
        self.assertEqual(result['frames'], [0, 12, 24, 36, 60, 72, 96])
        self.assertEqual(result['sampling_method'], 'authored_keyframes')
        self.assertEqual(design, original)

    def test_bounded_keys_keep_endpoints_and_brief_informative_poses(self):
        seconds = [index/4 for index in range(17)]
        angles = [45 if second == 1.25 else -40 if second == 2.75 else 0 for second in seconds]
        result = sample_action(action(seconds, angles), (0, 96))
        self.assertEqual(len(result['seconds']), 8)
        self.assertEqual(result['seconds'], sorted(set(result['seconds'])))
        self.assertEqual((result['seconds'][0], result['seconds'][-1]), (0, 4))
        self.assertTrue(set(result['seconds']) <= set(seconds))
        self.assertIn(1.25, result['seconds'])
        self.assertIn(2.75, result['seconds'])
        self.assertEqual(result['frames'], [second*24 for second in result['seconds']])
        self.assertEqual(result['sampling_method'], 'authored_keyframes_bounded_pose_error')

    def test_equal_poses_remain_bounded_without_duplicate_samples(self):
        result = sample_action(action([index/8 for index in range(32)]), (0, 93))
        self.assertEqual(len(result['seconds']), 8)
        self.assertEqual(result['seconds'], sorted(set(result['seconds'])))
        self.assertEqual((result['seconds'][0], result['seconds'][-1]), (0, 31/8))

    def test_authored_subframes_are_not_rounded(self):
        result = sample_action(action([0, .17, 4]), (0, 96))
        self.assertEqual(result['frames'][1], .17*24)
        self.assertNotEqual(result['frames'][1], round(result['frames'][1]))

    def test_legacy_provider_sampling_keeps_five_fractions(self):
        result = sample_action(None, (12, 132), fps=30)
        self.assertEqual(result['frames'], [12, 42, 72, 102, 132])
        self.assertEqual(result['seconds'], [.4, 1.4, 2.4, 3.4, 4.4])
        self.assertEqual(result['sampling_method'], 'legacy_uniform_five')

    def test_rotation_only_outputs_match_prior_samples_exactly(self):
        seconds = [index/4 for index in range(17)]
        angles = [45 if second == 1.25 else -40 if second == 2.75 else 0 for second in seconds]
        result = sample_action(action(seconds, angles), (0, 96))
        self.assertEqual(result, {
            'frames': [0, 24, 30, 36, 60, 66, 72, 96],
            'seconds': [0, 1, 1.25, 1.5, 2.5, 2.75, 3, 4],
            'sampling_method': 'authored_keyframes_bounded_pose_error',
        })
        self.assertEqual(sample_action(action([index/8 for index in range(32)]), (0, 93)), {
            'frames': [0, 21, 33, 45, 57, 69, 81, 93],
            'seconds': [0, .875, 1.375, 1.875, 2.375, 2.875, 3.375, 3.875],
            'sampling_method': 'authored_keyframes_bounded_pose_error',
        })

    def test_pure_translation_keeps_brief_extrema_beyond_sample_cap(self):
        seconds = [index/4 for index in range(17)]
        design = action(seconds)
        for key in design['keys']:
            key['rotations'] = {}
            key['translations_m'] = {'CompanionRoot': (
                [0, 0, 1.5] if key['second'] == 1.25 else
                [-1.25, 0, 0] if key['second'] == 2.75 else [0, 0, 0])}
        original = deepcopy(design)
        result = sample_action(design, (0, 96))
        self.assertEqual(len(result['seconds']), 8)
        self.assertEqual(result['seconds'], sorted(set(result['seconds'])))
        self.assertEqual((result['seconds'][0], result['seconds'][-1]), (0, 4))
        self.assertIn(1.25, result['seconds'])
        self.assertIn(2.75, result['seconds'])
        self.assertEqual(design, original)

    def test_mixed_root_companion_and_rotation_keep_independent_gestures(self):
        seconds = [index/4 for index in range(17)]
        design = action(seconds, [30 if second == .75 else 0 for second in seconds])
        for key in design['keys']:
            if key['second'] == 1.75:
                key['translations_m'] = {'Root': [0, .5, 0]}
            elif key['second'] == 2.75:
                key['translations_m'] = {'CompanionRoot': [0, 0, .8]}
        sparse = sample_action(design, (0, 96))
        self.assertEqual(len(sparse['seconds']), 8)
        self.assertIn(.75, sparse['seconds'])
        self.assertIn(1.75, sparse['seconds'])
        self.assertIn(2.75, sparse['seconds'])
        for key in design['keys']:
            translations = key.setdefault('translations_m', {})
            for bone in ['Root', 'CompanionRoot']:
                translations.setdefault(bone, [0, 0, 0])
        self.assertEqual(sample_action(design, (0, 96)), sparse)

    def test_translation_normalization_matches_equivalent_rotation_error(self):
        seconds = [index/4 for index in range(17)]
        meters = [1 if second == 1.25 else -.5 if second == 2.75 else 0 for second in seconds]
        rotations = action(seconds, [value*60 for value in meters])
        translations = action(seconds)
        for key, value in zip(translations['keys'], meters):
            key['rotations'] = {}
            key['translations_m'] = {'CompanionRoot': [0, 0, value]}
        self.assertEqual(sample_action(translations, (0, 96)), sample_action(rotations, (0, 96)))

    def test_translation_bounds_and_channels_do_not_overwrite_rotations(self):
        design = action([0, 2, 4])
        for key in design['keys']:
            key['rotations'] = {'Root': [1, 2, 3], 'CompanionRoot': [3, 2, 1]}
            key['translations_m'] = {'Root': (-2, 0, 2), 'CompanionRoot': (2, -2, 0)}
        self.assertEqual(sample_action(design, (0, 96))['frames'], [0, 48, 96])
        design = action([index/4 for index in range(17)])
        for key in design['keys']:
            key['rotations'] = {'CompanionRoot': [0, 60 if key['second'] == 1.25 else 0, 0]}
            key['translations_m'] = {'CompanionRoot': [0, 0, 1 if key['second'] == 2.75 else 0]}
        samples = sample_action(design, (0, 96))['seconds']
        self.assertIn(1.25, samples)
        self.assertIn(2.75, samples)

    def test_invalid_translation_values_are_rejected(self):
        invalid = [None, True, '0,0,0', {}, [], [0, 0], [0, 0, 0, 0],
                   [0, False, 0], [0, '0', 0], [0, math.nan, 0],
                   [math.inf, 0, 0], [0, 0, -math.inf],
                   [2.00001, 0, 0], [0, -2.00001, 0], [10**1000, 0, 0]]
        for value in invalid:
            design = action([0, 4])
            design['keys'][0]['translations_m'] = {'CompanionRoot': value}
            with self.subTest(value=value), self.assertRaises(ValueError):
                sample_action(design, (0, 96))

    def test_translation_maps_require_named_bones_and_at_most_64_entries(self):
        for value in [None, True, [], 'Root', {None: [0, 0, 0]}, {False: [0, 0, 0]},
                      {7: [0, 0, 0]}, {('Root',): [0, 0, 0]},
                      {str(i): [0, 0, 0] for i in range(65)}]:
            design = action([0, 4])
            design['keys'][0]['translations_m'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                sample_action(design, (0, 96))
        design = action([index/8 for index in range(32)])
        baseline = sample_action(design, (0, 93))
        for key in design['keys']:
            key['translations_m'] = {}
        self.assertEqual(sample_action(design, (0, 93)), baseline)
        for key in design['keys']:
            key['translations_m'] = {str(i): [0, 0, 0] for i in range(64)}
        self.assertEqual(sample_action(design, (0, 93)), baseline)

    def test_translation_does_not_relax_required_fields_or_limits(self):
        malformed = [
            {'second': 0, 'translations_m': {'Root': [0, 0, 0]}},
            {'second': 0, 'rotations': {}, 'translation': [0, 0, 0]},
            {'second': 0, 'rotations': {}, 'root_translation_m': [0, 0, 0]},
            {'second': 0, 'rotations': {}, 'translations_m': {}, 'extra': 1},
            {'second': 0, 'rotations': {str(i): [0, 0, 0] for i in range(65)},
             'translations_m': {}},
        ]
        for key in malformed:
            design = action([0, 4])
            design['keys'][0] = key
            with self.subTest(key=key), self.assertRaises(ValueError):
                sample_action(design, (0, 96))
        design = action([index/8 for index in range(33)])
        for key in design['keys']:
            key['translations_m'] = {'Root': [0, 0, 0]}
        with self.assertRaisesRegex(ValueError, '2–32'):
            sample_action(design, (0, 96))

    def test_invalid_and_oversized_key_lists_are_rejected(self):
        for seconds in ([0], [0, 0, 4], [0, math.nan, 4], [0, math.inf],
                        [index/10 for index in range(33)]):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                sample_action(action(seconds), (0, 96))
        with self.assertRaisesRegex(ValueError, 'actual action range'):
            sample_action(action([0, 4]), (1, 96))
        with self.assertRaises(ValueError):
            sample_action(None, (0, 0))


if __name__ == '__main__':
    unittest.main()
