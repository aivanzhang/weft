"""Analytic audit contracts; these fixtures do not certify any generated avatar."""

import unittest

import numpy as np

from scripts.blender_verify_compound_motion import (
    camera_projection, edge_peak, frame_schedule, owned_parts, root_local_points,
)


class CompoundMotionAuditTests(unittest.TestCase):
    def test_full_frame_schedule_preserves_endpoints_and_enforces_aggregate_limit(self):
        self.assertEqual(frame_schedule({'takeoff': [0, 96]})['takeoff'], list(range(97)))
        self.assertEqual(frame_schedule({'fractional': [.25, 2.5]})['fractional'], [.25, 1.25, 2.25, 2.5])
        self.assertEqual(len(frame_schedule({'maximum': [0, 240]})['maximum']), 241)
        with self.assertRaisesRegex(ValueError, 'Aggregate'):
            frame_schedule({'one': [0, 120], 'two': [0, 120]})
        for value in ([0, float('nan')], [True, 12], [0, 241], [2, 2]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                frame_schedule({'bad': value})

    def test_final_weight_membership_ignores_material_rename_but_rejects_mixed_ownership(self):
        names = ['Body', 'PartRoot', 'Fin']
        parts = [{'name': 'device', 'materials': ['original-renamed-later'], 'min_vertices': 2, 'max_vertices': 4}]
        subsets = {'device': ['PartRoot', 'Fin']}
        weights = np.array([[1., 0, 0], [0, 1, 0], [0, .25, .75]])
        labels, selected = owned_parts(weights, names, parts, subsets, {'device': 2})
        np.testing.assert_array_equal(labels, [0, 1, 1])
        np.testing.assert_array_equal(selected[0], [1, 2])
        mixed = weights.copy(); mixed[2] = [.1, .2, .7]
        with self.assertRaisesRegex(ValueError, 'outside'):
            owned_parts(mixed, names, parts, subsets, {'device': 2})
        with self.assertRaisesRegex(ValueError, 'finishing proof'):
            owned_parts(weights, names, parts, subsets, {'device': 3})
        with self.assertRaisesRegex(ValueError, 'overlap'):
            owned_parts(weights, names, parts+[dict(parts[0], name='other')],
                        {**subsets, 'other': subsets['device']}, {'device': 2, 'other': 2})

    def test_fur_envelope_detects_clipping_that_base_vertices_do_not(self):
        points = np.array([[-.95, -.1, -2.], [.95, .1, -2.]])
        view = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]], float)
        result = camera_projection(points, np.eye(4), view, True, .1, (.01, 100))
        self.assertTrue(result['base_vertices']['inside_image'])
        self.assertFalse(result['fur_margin_world_aabb']['inside_image'])
        self.assertAlmostEqual(result['fur_margin_world_aabb']['minimum_image_margin'], -.025)
        perspective = camera_projection(points, np.eye(4), view, False, 0., (.01, 100))
        self.assertAlmostEqual(perspective['base_vertices']['uv_max'][0], .7375)
        behind = camera_projection(-points, np.eye(4), view, True, 0., (.01, 100))
        self.assertFalse(behind['base_vertices']['inside_depth'])
        self.assertEqual(behind['base_vertices']['points_at_or_behind_camera'], 2)
        with self.assertRaises(ValueError):
            camera_projection(points, np.eye(4), view, True, True, (.01, 100))

    def test_part_root_rigid_motion_is_not_misreported_as_articulation(self):
        points = np.array([[.1, .2, .3], [.2, .4, .5]])
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
        matrix = np.eye(4); matrix[:3, :3] = rotation*1.7; matrix[:3, 3] = [.5, -.2, .8]
        moved = points@rotation.T+matrix[:3, 3]
        np.testing.assert_allclose(root_local_points(moved, matrix), points, atol=1e-15)
        articulated = moved.copy(); articulated[1, 2] += .025
        displacement = np.linalg.norm(root_local_points(articulated, matrix)-points, axis=1)
        np.testing.assert_allclose(displacement, [0, .025], atol=1e-15)
        bad = matrix.copy(); bad[0, 0] = .7
        with self.assertRaisesRegex(ValueError, 'Sheared'):
            root_local_points(moved, bad)

    def test_edge_peak_retains_actual_edge_and_frame_identity(self):
        edges = np.array([[0, 1], [1, 2], [2, 3]])
        result = edge_peak(edges, np.array([0, 2]), np.array([1., .5, 0.]), np.array([1.2, .8, .1]), 37)
        self.assertEqual(result['edge_index'], 0)
        self.assertEqual(result['vertex_indices'], [0, 1])
        self.assertEqual(result['frame'], 37)
        self.assertAlmostEqual(result['extension_m'], .2)
        self.assertEqual(result['ignored_tiny_edges'], 1)

    def test_invalid_or_oversized_arrays_fail_before_projection_or_membership(self):
        frame = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]], float)
        for points in (np.zeros((100001, 3)), np.array([[float('nan'), 0, -1]]), np.zeros((1, 2))):
            with self.subTest(shape=points.shape), self.assertRaises(ValueError):
                camera_projection(points, np.eye(4), frame, True, 0., (.01, 100))
        with self.assertRaises(ValueError):
            owned_parts(np.ones((100001, 1)), ['Body'], [], {}, {})


if __name__ == '__main__':
    unittest.main()
