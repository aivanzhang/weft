"""Analytic smoothing behavior; not recovered detail or volume/visual acceptance."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from avatar_assets.surface_smoothing_config import validate_surface_smoothing
from scripts.surface_smoothing import refine_positions


def config(**changes):
    return {'iterations': 12, 'strength': .3, 'max_displacement_ratio': .03,
            'feather': .25, 'exclusions': [], **changes}


def mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.4, .4, 1.2], [.4, .4, 1.2]])
    edges = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3],
                      [0, 4], [1, 4], [2, 5], [3, 5]])
    return vertices, edges


def spiky_sphere():
    rows, columns = 12, 24
    vertices, edges = [], []
    for row in range(rows):
        theta = np.pi*(row+1)/(rows+1)
        for column in range(columns):
            phi = 2*np.pi*column/columns
            radius = 1+(.1 if (row+column) % 2 else -.1)
            vertices.append(radius*np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]))
            index = row*columns+column
            edges.append([index, row*columns+(column+1) % columns])
            if row+1 < rows:
                edges.append([index, index+columns])
    return np.asarray(vertices), edges


class SurfaceSmoothingTests(unittest.TestCase):
    def test_omitted_method_is_identical_to_explicit_taubin(self):
        vertices, edges = mesh()
        default, report = refine_positions(vertices, edges, config())
        explicit, _ = refine_positions(vertices, edges, config(method='taubin'))
        np.testing.assert_array_equal(default, explicit)
        self.assertEqual(report['method_id'], 'taubin')
        self.assertEqual(report['steps_per_iteration'], 2)
        self.assertFalse(report['shrinkage_intended'])

    def test_seams_share_delta_and_protected_points_remain_exact(self):
        vertices, edges = mesh()
        original, original_edges = vertices.copy(), edges.copy()
        settings = config(exclusions=[{'center': [0, 0, 0], 'radius': [.15]*3}])
        saved_settings = deepcopy(settings)
        result, report = refine_positions(vertices, edges, settings)
        np.testing.assert_array_equal(result[4], result[5])
        np.testing.assert_array_equal(result[0], vertices[0])
        np.testing.assert_array_equal(vertices, original)
        np.testing.assert_array_equal(edges, original_edges)
        self.assertEqual(settings, saved_settings)
        self.assertGreater(np.linalg.norm(result[4]-original[4]), 0)
        self.assertEqual(report['coincident_vertices_merged'], 1)
        self.assertEqual(report['excluded_vertices'], 1)
        self.assertTrue(report['authored_not_recovered'])
        self.assertFalse(report['volume_preservation_verified'])

    def test_global_and_per_step_shifts_are_bounded_and_reported(self):
        vertices, edges = mesh()
        result, report = refine_positions(vertices, edges, config(iterations=20, strength=.5, max_displacement_ratio=.001))
        shifts = np.linalg.norm(result-vertices, axis=1)
        self.assertTrue(np.isfinite(result).all())
        self.assertLessEqual(float(shifts.max()), report['displacement_limit_m']+1e-12)
        self.assertLessEqual(report['max_step_displacement_m'], report['displacement_limit_m']+1e-12)
        self.assertAlmostEqual(report['max_displacement_m'], float(shifts.max()))
        self.assertAlmostEqual(report['rms_displacement_m'], float(np.sqrt(np.mean(shifts*shifts))))
        self.assertGreater(report['clipped_vertex_step_events'], 0)

    def test_high_frequency_spiky_sphere_variation_is_reduced(self):
        vertices, edges = spiky_sphere()
        result, report = refine_positions(vertices, edges, config(iterations=16, max_displacement_ratio=.05))
        before, after = np.linalg.norm(vertices, axis=1), np.linalg.norm(result, axis=1)
        self.assertLess(float(after.std()), float(before.std())*.5)
        self.assertLess(abs(float(after.mean()-before.mean())), .02)
        self.assertEqual(report['iterations_executed'], 16)

    def test_laplacian_attenuates_spikes_with_explicit_bounded_shrinkage(self):
        vertices, edges = spiky_sphere()
        result, report = refine_positions(vertices, edges, config(method='laplacian', iterations=20,
                                                                  strength=.35, max_displacement_ratio=.04))
        before, after = np.linalg.norm(vertices, axis=1), np.linalg.norm(result, axis=1)
        self.assertLess(float(after.std()), float(before.std())*.6)
        self.assertLess(float(after.mean()), float(before.mean()))
        self.assertLessEqual(float(np.linalg.norm(result-vertices, axis=1).max()), report['displacement_limit_m']+1e-12)
        self.assertTrue(report['shrinkage_intended'])
        self.assertFalse(report['volume_preservation_intended'])
        self.assertFalse(report['volume_preservation_verified'])
        self.assertEqual(report['method_id'], 'laplacian')
        self.assertEqual(report['steps_per_iteration'], 1)
        self.assertIsNone(report['negative_step_strength'])

    def test_laplacian_preserves_protection_seams_and_per_step_cap(self):
        vertices, edges = mesh()
        original = vertices.copy()
        result, report = refine_positions(vertices, edges, config(method='laplacian', iterations=20,
            max_displacement_ratio=.001, exclusions=[{'center': [0]*3, 'radius': [.15]*3}]))
        np.testing.assert_array_equal(result[0], original[0])
        np.testing.assert_array_equal(result[4], result[5])
        np.testing.assert_array_equal(vertices, original)
        self.assertTrue(np.isfinite(result).all())
        self.assertLessEqual(report['max_step_displacement_m'], report['displacement_limit_m']+1e-12)
        self.assertLessEqual(report['max_displacement_m'], report['displacement_limit_m']+1e-12)

    def test_full_exclusion_and_empty_edge_graph_are_explicit_noops(self):
        vertices, edges = mesh()
        result, report = refine_positions(vertices, edges, config(exclusions=[{'center': [.5]*3, 'radius': [3]*3}]))
        np.testing.assert_array_equal(result, vertices)
        self.assertEqual(report['iterations_executed'], 0)
        result, report = refine_positions(vertices, [], config())
        np.testing.assert_array_equal(result, vertices)
        self.assertEqual(report['iterations_executed'], 0)
        self.assertEqual(report['isolated_welded_vertices'], 5)

    def test_strict_configuration_rejects_invalid_and_unbounded_inputs(self):
        settings = config(exclusions=[{'center': [.5]*3, 'radius': [.1]*3}])
        validate_surface_smoothing(settings)['exclusions'][0]['center'][0] = 0
        self.assertEqual(settings['exclusions'][0]['center'][0], .5)
        for changes in ({'iterations': True}, {'iterations': 21}, {'strength': float('nan')},
                        {'method': 'unbounded'}, {'method': None}, {'method': []},
                        {'strength': .51}, {'max_displacement_ratio': .051}, {'feather': .01},
                        {'exclusions': settings['exclusions']*17}, {'color_mask': [0, 1]},
                        {'exclusions': [{'center': [0, 0, 0], 'radius': [0, 1, 1]}]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_surface_smoothing(config(**changes))

    def test_invalid_geometry_is_rejected_before_smoothing(self):
        vertices, edges = mesh()
        invalid = [(vertices, [[0, 6]]), (vertices, [[0, -1]]), (vertices, [[False, True]]),
                   (vertices, [[0., 1.]]), (vertices*np.nan, edges),
                   (np.zeros((100001, 3)), edges), (vertices, np.zeros((300001, 2), dtype=int)),
                   (np.zeros((3, 3), dtype=bool), edges), (np.array(1), edges), ([], [])]
        with patch('scripts.surface_smoothing._smooth', side_effect=AssertionError('Premature smoothing')):
            for points, graph in invalid:
                with self.subTest(points=np.shape(points)), self.assertRaises(ValueError):
                    refine_positions(points, graph, config())
        with self.assertRaisesRegex(ValueError, 'nondegenerate'):
            refine_positions([[0, 0, 0], [1, 1, 0]], [[0, 1]], config())


if __name__ == '__main__':
    unittest.main()
