"""Analytic topology checks; these do not establish generated-character likeness."""

import math
import unittest

import numpy as np

from avatar_assets.partition_generated import (
    _components, partition_topology, select_vertices, validate_config,
    triangulate_boundary_3d, triangles_intersect_3d,
    split_pinch_loops,
    contact_color_candidate,
    contact_override_matches,
)


def config():
    return {'schema_version': 1, 'source_sha256': 'a'*64, 'part_name': 'Companion', 'height_meters': 1.2,
            'selector': {'regions': [{'center': [.5, .5, .5], 'radius': [1, 1, 1]}],
                         'dominant_channel': 1, 'dominant_ratio': 1.05, 'expand_meters': 0,
                         'min_faces': 1, 'max_faces': 100000},
            'island_face_limit': 40, 'fill_face_limit': 200,
            'closure_colors_srgb': [[.8, .8, .8], [.1, .5, .1]], 'preview_offset_m': [0, 0, .3],
            'rights': 'Analytic fixture; closure surfaces are authored assumptions.'}


class PartitionTests(unittest.TestCase):
    def test_config_is_detached_and_rejects_unbounded_or_unknown_fields(self):
        value = config()
        result = validate_config(value)
        result['selector']['regions'][0]['center'][0] = 0
        self.assertEqual(value['selector']['regions'][0]['center'][0], .5)
        for key, invalid in [('source_sha256', 'oops'), ('part_name', '../bad'), ('height_meters', True),
                             ('height_meters', math.nan), ('island_face_limit', 101),
                             ('fill_face_limit', 501), ('preview_offset_m', [math.inf, 0, 0])]:
            value = config()
            value[key] = invalid
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(value)
        value = config()
        value['code'] = 'never execute'
        with self.assertRaises(ValueError):
            validate_config(value)

    def test_selection_requires_color_and_spatial_region(self):
        points = np.array([[0, 0, 0], [1, 1, 1], [.2, .2, .2], [.21, .2, .2]])
        colors = np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=float)
        selector = config()['selector']
        selector['regions'] = [{'center': [.2, .2, .2], 'radius': [.1, .1, .1]}]
        self.assertEqual(select_vertices(points, colors, selector).tolist(), [False, False, False, True])
        selector['expand_meters'] = .02
        self.assertEqual(select_vertices(points, colors, selector).tolist(), [False, False, True, True])
        colors[0, 0] = math.nan
        with self.assertRaises(ValueError):
            select_vertices(points, colors, selector)

    def test_closed_tetrahedron_cut_duplicates_boundary_without_dropping_faces(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
        original = points.copy(), faces.copy()
        selected = np.array([True, True, True, False])
        result = partition_topology(points, faces, selected)
        self.assertEqual(result['report']['source_open_edges'], 0)
        self.assertEqual(result['report']['predicted_closure_triangles'], 2)
        self.assertEqual(result['report']['part_faces'], 1)
        self.assertEqual(result['report']['body_faces'], 3)
        self.assertEqual(sorted(map(len, result['loops'])), [3, 3])
        np.testing.assert_array_equal(result['points'][result['faces']], points[faces])
        for face in result['faces']:
            self.assertEqual(len(set(result['vertex_labels'][face])), 1)
        body = set(result['faces'][result['face_labels'] == 0].ravel())
        part = set(result['faces'][result['face_labels'] == 1].ravel())
        self.assertFalse(body & part)
        np.testing.assert_array_equal(points, original[0])
        np.testing.assert_array_equal(faces, original[1])

    def test_local_winding_repair_keeps_source_corner_mapping(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        faces = np.array([[0, 2, 1], [0, 1, 3], [1, 3, 2], [2, 0, 3]])
        result = partition_topology(points, faces, np.array([True, True, True, False]))
        self.assertEqual(result['report']['reoriented_source_face_ids'], [2])
        np.testing.assert_array_equal(result['points'][result['faces']],
                                      points[result['source_vertex_per_corner']])
        for before, after in zip(faces, result['source_vertex_per_corner']):
            self.assertEqual(set(before), set(after))

    def test_invalid_arrays_and_degenerate_welds_fail_before_native_work(self):
        points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
        selected = np.array([True, True, True, False])
        for broken in [faces.astype(float), np.array([[0, 1, 99]]), np.array([[0, 0, 1]])]:
            with self.subTest(faces=broken), self.assertRaises(ValueError):
                partition_topology(points, broken, selected)
        for mask in [selected.astype(int), selected[:3], np.ones(4, dtype=bool)]:
            with self.assertRaises(ValueError):
                partition_topology(points, faces, mask)
        with self.assertRaises(ValueError):
            partition_topology(points, faces, selected, island_limit=1000)

    def test_fragmented_components_are_complete_and_deterministic(self):
        count = 10000
        components = _components(np.ones(count, dtype=bool), [()] * count)
        self.assertEqual(components, [[i] for i in range(count)])

    def test_3d_triangulation_preserves_nonplanar_boundary_and_rejects_pinches(self):
        points = np.array([[0, 0, 0], [1, 0, .2], [1, 1, 0], [0, 1, -.3]], dtype=float)
        triangles = triangulate_boundary_3d(points)
        self.assertEqual(triangles.shape, (2, 3))
        self.assertTrue(all(len(set(t)) == 3 for t in triangles))
        with self.assertRaisesRegex(ValueError, 'pinched'):
            triangulate_boundary_3d(np.vstack([points, points[0]]))
        with self.assertRaises(ValueError):
            triangulate_boundary_3d(np.zeros((257, 3)))

    def test_triangle_intersections_include_coplanar_overlap_not_contacts(self):
        a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        self.assertTrue(triangles_intersect_3d(a, a.copy()))
        self.assertFalse(triangles_intersect_3d(a, a+[0, 0, 1]))
        self.assertFalse(triangles_intersect_3d(a, np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]])))
        self.assertTrue(triangles_intersect_3d(a, np.array([[.2, .2, -1], [.2, .2, 1], [.7, .2, 0]])))

    def test_pinched_cycles_split_without_zero_area_bridge_triangles(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],
                           [0, 0, 0], [-1, 0, 0], [-1, -1, 0]], dtype=float)
        loops, aliases, omitted = split_pinch_loops(points, [[0, 1, 2, 3, 4, 5]])
        self.assertEqual(sorted(map(len, loops)), [3, 3])
        self.assertEqual(aliases, {3: 0})
        self.assertFalse(omitted)
        for loop in loops:
            self.assertEqual(triangulate_boundary_3d(points[loop]).shape, (1, 3))
        loops, aliases, omitted = split_pinch_loops(points, [[0, 1, 3, 4, 5]])
        self.assertEqual(list(map(len, loops)), [3])
        self.assertEqual(omitted, [[0, 1]])

    def test_contact_palette_repair_protects_front_features_and_light_claws(self):
        def candidate(*args):
            return contact_color_candidate(*args, channel=1, ratio=1.05)
        self.assertTrue(candidate(1, [.05, .06, .06], .2, .5, 0))
        self.assertFalse(candidate(1, [.05, .06, .06], -.2, .5, 0))
        self.assertFalse(candidate(1, [.05, .06, .06], .2, -.5, 0))
        self.assertFalse(candidate(1, [.8, .8, .7], .2, .5, 0))
        self.assertFalse(candidate(1, [.02, .1, .01], .2, .5, 0))
        self.assertTrue(candidate(0, [.05, .2, .05], 0, 0, 0))
        self.assertFalse(candidate(0, [.8, .8, .8], 0, 0, 0))
        self.assertTrue(contact_color_candidate(0, [.2, .05, .05], 0, 0, 0, channel=0, ratio=1.2))
        self.assertFalse(contact_color_candidate(0, [.05, .2, .05], 0, 0, 0, channel=0, ratio=1.2))

    def test_contact_override_is_spatial_color_and_part_limited(self):
        rule = {'name': 'rear_light_contact', 'part': 1, 'min': [0, 0, .63], 'max': [.45, 1, 1],
                'min_mean': .45, 'max_chroma': .18, 'min_normal_y': -.1,
                'palette': 'part', 'min_faces': 0, 'max_faces': 100}
        self.assertTrue(contact_override_matches(rule, 1, [.9, .9, .85], [.2, .5, .7], .2))
        for part, color, position, normal in [(0, [.9]*3, [.2, .5, .7], .2),
                (1, [.9]*3, [.2, .5, .4], .2), (1, [.1, .6, .1], [.2, .5, .7], .2),
                (1, [.9]*3, [.2, .5, .7], -.5)]:
            self.assertFalse(contact_override_matches(rule, part, color, position, normal))
        value = config()
        value.update(repair_contact_uvs=True, contact_repair_assumptions='Authored rear-color repair',
                     contact_color_overrides=[rule])
        self.assertEqual(validate_config(value)['contact_color_overrides'][0], rule)
        value['contact_color_overrides'][0]['max_faces'] = 1001
        with self.assertRaises(ValueError):
            validate_config(value)


if __name__ == '__main__':
    unittest.main()
