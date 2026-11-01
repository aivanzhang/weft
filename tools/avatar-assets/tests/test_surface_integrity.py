"""Analytic surface regression metrics; not self-intersection or visual validation."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from scripts.surface_integrity import SurfaceChangeError, enforce_surface_change, inspect_surface_change


def mesh(count=2):
    points = np.tile([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], (count, 1))
    points[:, 0] += np.repeat(np.arange(count)*2, 3)
    return points, np.arange(count*3).reshape(count, 3)


class SurfaceIntegrityTests(unittest.TestCase):
    def test_unchanged_and_rigid_translation_pass_without_area_changes(self):
        original, triangles = mesh()
        saved = original.copy()
        for offset in (np.zeros(3), np.array([2, -3, .5])):
            report = enforce_surface_change(original, original+offset, triangles)
            self.assertEqual(report['regression_gate']['status'], 'passed')
            self.assertEqual(report['new_degenerate_count'], 0)
            self.assertEqual(report['min_area_ratio'], 1)
            self.assertEqual(report['median_area_ratio'], 1)
            self.assertEqual(report['face_normals_changed_over_90_degrees_count'], 0)
            self.assertAlmostEqual(report['vertex_max_shift_m'], float(np.linalg.norm(offset)))
            self.assertAlmostEqual(report['vertex_rms_shift_m'], float(np.linalg.norm(offset)))
            json.dumps(report, allow_nan=False)
        np.testing.assert_array_equal(original, saved)

    def test_one_small_nondegenerate_face_is_tolerated_and_counted(self):
        original, triangles = mesh()
        refined = original.copy()
        refined[2, 1] = .005
        report = enforce_surface_change(original, refined, triangles)
        self.assertEqual(report['regression_gate']['status'], 'passed')
        self.assertEqual(report['area_ratio_below_0_01_count'], 1)
        self.assertEqual(report['area_ratio_below_0_001_count'], 0)
        self.assertEqual(report['new_degenerate_count'], 0)
        self.assertAlmostEqual(report['min_area_ratio'], .005)
        self.assertAlmostEqual(report['median_area_ratio'], .5025)
        self.assertFalse(report['regression_gate']['quality_proof'])

    def test_widespread_collapse_rejects_and_retains_diagnostics(self):
        original, triangles = mesh()
        refined = original.copy()
        refined[[2, 5], 1] = .0005
        with self.assertRaisesRegex(SurfaceChangeError, 'severely collapsed') as failure:
            enforce_surface_change(original, refined, triangles)
        report = failure.exception.report
        self.assertEqual(report['new_degenerate_count'], 0)
        self.assertEqual(report['area_ratio_below_0_01_count'], 2)
        self.assertEqual(report['area_ratio_below_0_001_count'], 2)
        self.assertEqual(report['regression_gate']['allowed_area_ratio_below_0_01_count'], 1)
        self.assertEqual(report['regression_gate']['status'], 'failed')

    def test_collapse_allowance_scales_and_is_capped_at_sixteen(self):
        for count, allowed in ((2000, 2), (17000, 16)):
            original, triangles = mesh(count)
            refined = original.copy()
            refined[np.arange(allowed)*3+2, 1] = .005
            report = enforce_surface_change(original, refined, triangles)
            self.assertEqual(report['regression_gate']['allowed_area_ratio_below_0_01_count'], allowed)
            refined[allowed*3+2, 1] = .005
            with self.subTest(count=count), self.assertRaises(SurfaceChangeError):
                enforce_surface_change(original, refined, triangles)

    def test_new_degeneracy_is_rejected_even_inside_the_small_face_allowance(self):
        original, triangles = mesh()
        refined = original.copy()
        refined[2, 1] = 0
        with self.assertRaisesRegex(SurfaceChangeError, 'newly degenerate') as failure:
            enforce_surface_change(original, refined, triangles)
        self.assertEqual(failure.exception.report['new_degenerate_count'], 1)
        self.assertEqual(failure.exception.report['area_ratio_below_0_01_count'], 1)

    def test_source_degeneracy_remains_explicit_without_a_repair_claim(self):
        original, triangles = mesh()
        original[2, 1] = 0
        report = enforce_surface_change(original, original.copy(), triangles)
        self.assertEqual(report['regression_gate']['status'], 'passed_with_source_degeneracy')
        self.assertEqual(report['source_degenerate_count'], 1)
        self.assertEqual(report['source_degenerate_remaining_count'], 1)
        self.assertEqual(report['nonzero_source_count'], 1)
        refined = original.copy()
        refined[2, 1] = 1
        report = enforce_surface_change(original, refined, triangles)
        self.assertEqual(report['source_degenerate_now_nondegenerate_count'], 1)
        self.assertFalse(report['source_degeneracy_repair_claimed'])
        all_degenerate = inspect_surface_change(original[:3], original[:3], [[0, 1, 2]])
        self.assertIsNone(all_degenerate['min_area_ratio'])
        self.assertIsNone(all_degenerate['median_area_ratio'])
        json.dumps(all_degenerate, allow_nan=False)

    def test_reversed_normal_is_counted_without_inventing_an_extra_gate(self):
        original, triangles = mesh(1)
        refined = original.copy()
        refined[2, 1] = -1
        report = enforce_surface_change(original, refined, triangles)
        self.assertEqual(report['face_normals_changed_over_90_degrees_count'], 1)
        self.assertEqual(report['min_area_ratio'], 1)
        self.assertFalse(report['self_intersections_checked'])
        self.assertFalse(report['identity_checked'])

    def test_invalid_oversized_and_nonfinite_inputs_fail_before_geometry(self):
        original, triangles = mesh(1)
        invalid = [(np.zeros((100001, 3)), original, triangles),
                   (original, original, np.zeros((100001, 3), dtype=int)),
                   (original, original[:2], triangles), (original*np.nan, original, triangles),
                   (original, np.full((3, 3), np.inf), triangles),
                   (original, original, [[0, 1, 3]]), (original, original, [[0, 1, -1]]),
                   (original, original, [[False, True, False]]),
                   (original, original, [[0., 1., 2.]]), (original.astype(bool), original, triangles),
                   (np.array(1), original, triangles), (original, original, []), ([], [], triangles)]
        with patch('scripts.surface_integrity._measure_triangles', side_effect=AssertionError('Premature geometry')):
            for arguments in invalid:
                with self.subTest(shape=np.shape(arguments[0])), self.assertRaises(ValueError):
                    inspect_surface_change(*arguments)


if __name__ == '__main__':
    unittest.main()
