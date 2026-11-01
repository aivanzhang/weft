"""Bounded motion controls; analytic contracts, not generated-asset quality tests."""

from copy import deepcopy
import unittest

from avatar_assets.finish_generated import validate
from tests.test_finish_quality_contracts import finish_config


def config():
    result = finish_config()
    result['authored_rig'] = {'space': 'normalized_bounds_blender_z_up', 'height_meters': 1,
        'bones': [{'name': 'Root', 'parent': None, 'head': [.5,.5,0], 'tail': [.5,.5,1]}],
        'base_bones': ['Root'], 'layers': []}
    result['authored_actions'] = [{'id': 'idle', 'interpolation': 'LINEAR', 'keys': [
        {'second': 0, 'rotations': {}},
        {'second': 1, 'rotations': {}, 'translations_m': {'Root': [.2,0,.1]}}]}]
    result['camera_fit'] = 'motion'
    return result


class MotionControlsTests(unittest.TestCase):
    def test_translation_and_motion_framing_preserve_config(self):
        value = config()
        before = deepcopy(value)
        self.assertEqual(validate(value), before)

    def test_translations_require_linear_and_actual_bones(self):
        for change in ('bezier', 'missing-bone', 'legacy-root-field'):
            value = config()
            action = value['authored_actions'][0]
            if change == 'bezier':
                action['interpolation'] = 'BEZIER'
            elif change == 'missing-bone':
                action['keys'][1]['translations_m'] = {'Absent': [0,0,.1]}
            else:
                action['keys'][1]['root_translation_m'] = [0,0,.1]
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(value)

    def test_invalid_and_oversized_translations_fail_before_native_work(self):
        for vector in ([True,0,0], [float('nan'),0,0], [float('inf'),0,0], [2.01,0,0],
                       [0,0], 'xyz', None):
            value = config()
            value['authored_actions'][0]['keys'][1]['translations_m'] = {'Root': vector}
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                validate(value)
        value = config()
        value['authored_actions'][0]['keys'][1]['translations_m'] = {f'Bone{i}': [0,0,0] for i in range(65)}
        with self.assertRaisesRegex(ValueError, 'translations'):
            validate(value)

    def test_motion_fit_requires_actions_and_rejects_unknown_modes(self):
        for mode in ('motion', 'automatic'):
            value = finish_config()
            value['camera_fit'] = mode
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate(value)

    def test_old_rotation_only_actions_keep_default_interpolation(self):
        value = config()
        action = value['authored_actions'][0]
        del action['interpolation']
        del action['keys'][1]['translations_m']
        action['keys'][1]['rotations'] = {'Root': [0,0,15]}
        self.assertNotIn('interpolation', validate(value)['authored_actions'][0])


if __name__ == '__main__':
    unittest.main()
