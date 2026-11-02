"""Analytic edge diagnostics do not establish visual or physical validity."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from scripts.deformation_diagnostics import (MAX_CANDIDATE_ROWS, MAX_ROWS, SEVERE_HEIGHT_FRACTION,
                                            SEVERE_RATIO, diagnose_stretched_edges, retain_stretched_edges)


def crowded_edges():
    rest, posed = [[0., 0., 1.]], [[0., 0., 1.]]
    for index in range(5):
        start = [index*2., 0., 0.]
        rest.extend([start, [start[0]+(.1 if index < 4 else .001), 0., 0.]])
        posed.extend([start, [start[0]+(.7 if index < 4 else .051), 0., 0.]])
    return np.asarray(rest), np.asarray(posed), np.arange(1, 11).reshape(5, 2)


class DeformationDiagnosticTests(unittest.TestCase):
    def test_thin_stretched_strip_identifies_actual_rest_vertices(self):
        rest = np.array([[0, 0, 0], [1, 1, 1], [.4, .3, .5], [.401, .3, .5]])
        posed = rest.copy()
        posed[3, 0] += .1
        rows = diagnose_stretched_edges(rest, posed, [[2, 3]])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['endpoint_indices'], [2, 3])
        self.assertAlmostEqual(rows[0]['rest_length'], .001)
        self.assertAlmostEqual(rows[0]['posed_length'], .101)
        self.assertAlmostEqual(rows[0]['stretch_ratio'], 101)
        self.assertAlmostEqual(rows[0]['excess_length'], .1)
        np.testing.assert_allclose(rows[0]['rest_normalized_bounds'], rest[[2, 3]])
        np.testing.assert_array_equal(rest[3], [.401, .3, .5])

    def test_rigid_motion_does_not_report_stretch(self):
        rest = np.array([[0, 0, 0], [1, 0, 1], [0, 1, .5]], dtype=float)
        posed = rest @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) + [2, 3, 4]
        self.assertEqual(diagnose_stretched_edges(rest, posed, [[0, 1], [1, 2], [2, 0]]), [])

    def test_zero_tiny_and_subthreshold_edges_are_ignored(self):
        rest = np.array([[0, 0, 0], [1, 0, 1], [.1, 0, .1], [.1, 0, .1],
                         [.2, 0, .2], [.2+1e-9, 0, .2], [.3, 0, .3], [.301, 0, .3],
                         [.4, 0, .4], [.5, 0, .4]])
        posed = rest.copy()
        posed[[3, 5, 7, 9], 0] += [1, .1, .001, .04]
        self.assertEqual(diagnose_stretched_edges(rest, posed, [[2, 3], [4, 5], [6, 7], [8, 9]]), [])

    def test_only_four_distinct_worst_extensions_are_returned(self):
        rest = [[0, 0, 0], [1, 0, 1]]
        for i in range(6):
            rest.extend([[i/10, 0, .5], [i/10+.01, 0, .5]])
        rest = np.array(rest)
        posed = rest.copy()
        posed[[3, 5, 7, 9, 11, 13], 0] += [.06, .01, .08, .03, .09, .02]
        edges = [[10, 11], [11, 10]] + [[i, i+1] for i in range(2, 14, 2)]
        rows = diagnose_stretched_edges(rest, posed, edges)
        self.assertEqual(len(rows), 4)
        np.testing.assert_allclose([row['excess_length'] for row in rows], [.09, .08, .06, .03])
        self.assertEqual(len({tuple(sorted(row['endpoint_indices'])) for row in rows}), 4)

    def test_flat_bounds_are_normalized_without_division_by_zero(self):
        rest = np.array([[0, 0, 0], [1, 0, 0], [.5, 0, 0], [.51, 0, 0]])
        posed = rest.copy()
        posed[3, 0] = .6
        rows = diagnose_stretched_edges(rest, posed, [[2, 3]])
        np.testing.assert_allclose(rows[0]['rest_normalized_bounds'], [[.5, 0, 0], [.51, 0, 0]])
        self.assertEqual(diagnose_stretched_edges(rest, posed, []), [])
        self.assertEqual(diagnose_stretched_edges(np.zeros((2, 3)), np.ones((2, 3)), [[0, 1]]), [])

    def test_severe_fifth_edge_is_retained_before_larger_moderate_extensions(self):
        rest, posed, edges = crowded_edges()
        rows = diagnose_stretched_edges(rest, posed, edges)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]['endpoint_indices'], [9, 10])
        self.assertGreater(rows[0]['stretch_ratio'], 8)
        self.assertGreater(rows[0]['excess_length'], .03)
        self.assertLess(rows[0]['excess_length'], rows[1]['excess_length'])

    def test_severe_witness_survives_cross_pose_aggregation_and_deduplication(self):
        rest, first, edges = crowded_edges()
        second = first.copy()
        second[10] = rest[10]
        rows = []
        for time, posed in ((.6, first), (1.1, second)):
            for row in diagnose_stretched_edges(rest, posed, edges):
                rows.append({**row, 'second': time, 'endpoint_influences': [{'Body': 1.}, {'Arm': 1.}]})
        duplicate = deepcopy(rows[0])
        duplicate['endpoint_indices'].reverse()
        duplicate['second'] = 2.2
        rows.append(duplicate)
        saved = deepcopy(rows)
        retained = retain_stretched_edges(rows, 1.)
        self.assertEqual(len(retained), 4)
        self.assertEqual(retained[0]['endpoint_indices'], [9, 10])
        self.assertEqual(retained[0]['second'], .6)
        self.assertEqual(len({tuple(sorted(row['endpoint_indices'])) for row in retained}), 4)
        self.assertEqual(rows, saved)

    def test_priority_uses_unchanged_strict_gate_thresholds_and_row_caps(self):
        from avatar_assets import deformation_gate
        self.assertEqual((SEVERE_RATIO, SEVERE_HEIGHT_FRACTION, MAX_ROWS),
                         (deformation_gate.SEVERE_RATIO, deformation_gate.SEVERE_HEIGHT_FRACTION, deformation_gate.MAX_ROWS))
        self.assertEqual((SEVERE_RATIO, SEVERE_HEIGHT_FRACTION, MAX_ROWS, MAX_CANDIDATE_ROWS), (8., .03, 4, 32))
        def row(index, rest, excess):
            return {'endpoint_indices': [index*2, index*2+1], 'rest_length': rest, 'posed_length': rest+excess,
                    'excess_length': excess, 'stretch_ratio': (rest+excess)/rest}
        rows = [row(0, 1., 7.), row(1, .001, .03), row(2, .001, .031), row(3, 1., 6.), row(4, 1., 5.)]
        retained = retain_stretched_edges(rows, 1.)
        self.assertEqual(retained[0]['endpoint_indices'], [4, 5])
        self.assertEqual(retained[1]['endpoint_indices'], [0, 1])
        self.assertEqual(retained[1]['stretch_ratio'], 8.)

    def test_retention_rejects_oversized_nonfinite_and_inconsistent_measurements(self):
        rest, posed, edges = crowded_edges()
        row = diagnose_stretched_edges(rest, posed, edges)[0]
        invalid = [([row]*33, 1.), ([row], 0.), ([row], True), ([row], float('nan')),
                   ([{**row, 'stretch_ratio': float('inf')}], 1.),
                   ([{**row, 'stretch_ratio': True}], 1.),
                   ([{**row, 'excess_length': row['excess_length']+1}], 1.),
                   ([{**row, 'endpoint_indices': [0, 100000]}], 1.),
                   ([{**row, 'second': float('inf')}], 1.)]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                retain_stretched_edges(*arguments)
        self.assertEqual(retain_stretched_edges([], 1.), [])

    def test_invalid_and_oversized_inputs_fail_before_edge_geometry(self):
        points = np.array([[0., 0., 0.], [0., 0., 1.]])
        invalid = [
            (np.zeros((100001, 3)), points, [[0, 1]]),
            (points, points, np.zeros((300001, 2), dtype=int)),
            (points, points[:1], [[0, 1]]),
            (points, [[0, 0, float('nan')], [0, 0, 1]], [[0, 1]]),
            (points, [[0, 0, float('inf')], [0, 0, 1]], [[0, 1]]),
            (points, points, [[0, 2]]),
            (points, points, [[0, -1]]),
            (points, points, [[0., 1.]]),
            (points, points, [[False, True]]),
            (points, points, [[0, 1, 1]]),
            ([[[0, 1]]*3]*2, points, [[0, 1]]),
            (points.astype(complex), points, [[0, 1]]),
            ([], [], []),
        ]
        with patch('scripts.deformation_diagnostics._lengths', side_effect=AssertionError('Premature geometry work')):
            for index, arguments in enumerate(invalid):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    diagnose_stretched_edges(*arguments)


if __name__ == '__main__':
    unittest.main()
