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
