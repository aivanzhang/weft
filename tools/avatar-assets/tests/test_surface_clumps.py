"""Analytic surface-patch invariants; not an avatar appearance or motion proof."""

import copy
import hashlib
import unittest

import numpy as np

from scripts.groom_flow import prepare_flow
from scripts.surface_clumps import ADAPTIVE_POLICY, POLICY, SurfaceClumpError, build_surface_clumps


def fixture():
    points = np.array([[-.01, 0, 0], [.01, 0, 0], [0, .02, 0], [0, -.02, 0],
                       [.06, 0, 0], [.08, 0, 0], [.07, .02, 0], [.07, -.02, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 0, 3], [4, 5, 6], [5, 4, 7]], dtype=np.int32)
    uv = np.array([[0, .5], [1, .5], [.5, 0], [.5, 1]]*2, dtype=np.float32)
    joints = np.zeros((8, 4), dtype=np.uint16); joints[:, 1] = 1
    weights = np.zeros((8, 4), dtype=np.float32); weights[:, :2] = (.25, .75)
    weights[2, :2] = (.75, .25)
    source = {'positions': points, 'triangles': faces, 'corner_normals': np.tile([0., 0, 1], (4, 3, 1)),
              'uv0': uv[faces], 'material_ids': np.zeros(4, dtype=np.int16), 'part_ids': np.zeros(4, dtype=np.int16),
              'fur_faces': np.ones(4, dtype=bool), 'protected_faces': np.zeros(4, dtype=bool),
              'joints': joints, 'weights': weights}
    roots = np.array([[0, 1], [4, 5]], dtype=np.int32)
    positions = np.concatenate((points[roots], points[roots]+[0., 0, .008]), axis=1).reshape(-1, 3).astype(np.float32)
    atlas_uv = []
    for cell in (0, 1):
        a, b = (cell*64+2)/128, (cell*64+62)/128
        atlas_uv.extend([[a, 30/32], [b, 30/32], [a, 2/32], [b, 2/32]])
    fringe = {'positions': positions, 'normals': np.tile([0., 0, 1], (8, 1)),
              'joints': np.tile(joints[roots], (1, 2, 1)).reshape(-1, 4),
              'weights': np.tile(weights[roots], (1, 2, 1)).reshape(-1, 4),
              'uv_gltf': np.asarray(atlas_uv, dtype=np.float32), 'root_vertex_ids': roots,
              'root_face_ids': np.array([0, 2]), 'root_corner_ids': np.array([[0, 1], [0, 1]]),
              'destination_for_source_fin': np.array([0, 1])}
    atlas = np.full((32, 128, 4), 127, dtype=np.uint8)
    flow = prepare_flow([-.02, -.03, -.01], [.12, .06, .02], [0, 1, 0], .6,
        {'regions': [{'name': 'analytic-flow', 'center': [.5, .5, .5], 'radius': [3, 3, 3],
                      'direction': [0, 1, 0], 'comb': .6, 'length_scale': 1., 'feather': .5}]})
    settings = {'profile': POLICY, 'seed': 7123, 'lift_m': .0012, 'surface_fin_ids': np.array([0], dtype=np.int32)}
    return source, fringe, atlas, flow, settings


def gradient(slot, uv):
    assert slot == 0
    return np.column_stack((uv[:, 0], uv[:, 1], np.full(len(uv), .25)))


def run(data, sampler=gradient, **budget):
    source, fringe, atlas, flow, settings = data
    return build_surface_clumps(source, fringe, atlas, flow, sampler, settings,
                               source_export_vertices=budget.get('vertices', len(source['positions'])),
                               source_export_triangles=budget.get('triangles', len(source['triangles'])))


class SurfaceClumpTests(unittest.TestCase):
    def test_actual_source_faces_roots_normals_and_weights_are_preserved_without_pruning(self):
        data = fixture(); before = copy.deepcopy(data[:3]); result = run(data)
        source, fringe, atlas = data[:3]
        self.assertEqual(result['positions'].shape, (18, 3))
        self.assertEqual(result['triangles'].shape, (6, 3))
        np.testing.assert_array_equal(result['source_vertex_ids'][:6], [0, 1, 2, 1, 0, 3])
        np.testing.assert_array_equal(result['positions'][[0, 1, 3, 4]], source['positions'][[0, 1, 1, 0]])
        np.testing.assert_allclose(result['positions'][[2, 5]], source['positions'][[2, 3]]+[0, 0, .0012], atol=1e-9)
        np.testing.assert_array_equal(result['weights'][:6], source['weights'][result['source_vertex_ids'][:6]])
        np.testing.assert_array_equal(result['joints'][:6], source['joints'][result['source_vertex_ids'][:6]])
        np.testing.assert_array_equal(result['normals'], np.tile([0., 0, 1], (18, 1)))
        for current, original in zip(data[:3], before):
            if isinstance(current, dict):
                for key in current:
                    np.testing.assert_array_equal(current[key], original[key])
            else:
                np.testing.assert_array_equal(current, original)
        self.assertTrue(result['report']['intersections']['passed'])
        self.assertFalse(result['report']['quality_approved'])

    def test_retained_contour_has_exact_oriented_attributes_and_cell_bytes(self):
        data = fixture(); result = run(data); fringe = data[1]
        expected = 4+np.array([0, 1, 3, 0, 3, 2, 3, 1, 0, 2, 3, 0])
        for key, original in (('positions', 'positions'), ('normals', 'normals'), ('uv', 'uv_gltf'),
                              ('weights', 'weights'), ('joints', 'joints')):
            np.testing.assert_array_equal(result[key][6:], fringe[original][expected])
        np.testing.assert_array_equal(result['pixels'][:, 64:], data[2][:, 64:])
        np.testing.assert_array_equal(result['fin_vertex_ranges'], [[0, 6], [6, 12]])

    def test_source_color_varies_over_triangle_not_only_along_root_and_is_deterministic(self):
        data = fixture(); result = run(data); again = run(data)
        np.testing.assert_array_equal(result['pixels'], again['pixels'])
        np.testing.assert_array_equal(result['positions'], again['positions'])
        image = result['pixels']
        self.assertLess(int(image[6, 32, 1]), int(image[25, 32, 1]))
        self.assertTrue(np.all(image[2:30, 2:62, 2] == 137))
        self.assertGreater(np.count_nonzero(image[:32, :64, 3] >= 46), 100)
        self.assertLess(np.count_nonzero(image[:32, :64, 3] >= 46), 800)
        self.assertEqual(image[2, 2, 3], 0)

    def test_protected_material_part_and_nonfur_crossings_fail_before_color_sampling(self):
        for key, value in (('protected_faces', True), ('fur_faces', False), ('material_ids', 1), ('part_ids', 1)):
            data = fixture(); data[0][key][1] = value
            with self.subTest(key=key), self.assertRaisesRegex(SurfaceClumpError, 'ownership'):
                run(data, sampler=lambda *args: self.fail('Sampler ran after ownership failure'))

    def test_inward_source_normals_and_large_geometric_turns_retain_failure(self):
        data = fixture(); data[0]['corner_normals'][0, 2] = [0, 0, -1]
        with self.assertRaisesRegex(SurfaceClumpError, 'Source corner normals'):
            run(data)
        data = fixture(); data[0]['positions'][2, 1] = .0001
        with self.assertRaises(SurfaceClumpError) as caught:
            run(data)
        self.assertIn('15degrees', caught.exception.report['failure'])
        self.assertGreater(caught.exception.report['maximum_geometric_turn_degrees'], 15)

    def test_new_crossing_retains_exact_triangle_witness_and_never_samples_color(self):
        data = fixture(); source = data[0]
        source['positions'] = np.concatenate((source['positions'], np.array([[-.002, .009, .0006],
            [.002, .009, .0006], [0, .017, .0006]], dtype=np.float32)))
        source['triangles'] = np.concatenate((source['triangles'], [[8, 9, 10]]))
        source['corner_normals'] = np.concatenate((source['corner_normals'], np.tile([0., 0, 1], (1, 3, 1))))
        source['uv0'] = np.concatenate((source['uv0'], np.zeros((1, 3, 2))))
        source['joints'] = np.concatenate((source['joints'], np.zeros((3, 4), dtype=np.uint16)))
        source['weights'] = np.concatenate((source['weights'], np.tile([1., 0, 0, 0], (3, 1))))
        for key, value in (('material_ids', 0), ('part_ids', 0), ('fur_faces', False), ('protected_faces', True)):
            source[key] = np.append(source[key], value)
        with self.assertRaises(SurfaceClumpError) as caught:
            run(data, sampler=lambda *args: self.fail('Sampler ran after intersection'))
        witness = caught.exception.report['intersections']['witnesses'][0]
        self.assertEqual(witness['other_triangle'], 4)
        self.assertEqual(witness['other_kind'], 'source')
        self.assertEqual(np.asarray(witness['patch_points_m']).shape, (3, 3))

    def test_malformed_counts_nonfinite_skin_and_budget_are_rejected(self):
        for change in ('nan', 'bool_lift', 'too_high', 'root', 'weight', 'duplicate_influence', 'budget'):
            data = fixture()
            if change == 'nan': data[0]['positions'][0, 0] = np.nan
            elif change == 'bool_lift': data[4]['lift_m'] = True
            elif change == 'too_high': data[4]['lift_m'] = .00121
            elif change == 'root': data[4]['surface_fin_ids'] = np.array([0, 0])
            elif change == 'weight': data[0]['weights'][0, 0] = .4
            elif change == 'duplicate_influence': data[0]['joints'][0, 1] = 0
            with self.subTest(change=change), self.assertRaises(SurfaceClumpError):
                run(data, **({'vertices': 99999} if change == 'budget' else {}))

    def test_whole_object_budget_counts_both_surface_and_opposed_contour_corners(self):
        result = run(fixture(), vertices=90000, triangles=99990)
        self.assertEqual(result['report']['export_budget'], {'vertices_with_reserve': 94114,
                                                           'triangles': 99996, 'reserve_vertices': 4096})
        with self.assertRaisesRegex(SurfaceClumpError, '100k'):
            run(fixture(), triangles=99995)


class AdaptiveSurfaceClumpTests(unittest.TestCase):
    def test_thin_face_uses_shared_14degree_ceiling_without_moving_roots(self):
        data = fixture(); source = data[0]
        source['positions'][2, 1] = .0001
        with self.assertRaisesRegex(SurfaceClumpError, '15degrees'):
            run(data)
        data[4]['profile'] = ADAPTIVE_POLICY
        result = run(data)
        expected = float(source['positions'][2, 1])*np.tan(np.deg2rad(14))
        self.assertAlmostEqual(result['surface_card_lifts_m'][0], expected, places=12)
        np.testing.assert_allclose(result['surface_card_actual_tip_displacements_m'][0], [expected, expected], atol=1e-11)
        np.testing.assert_array_equal(result['positions'][[0, 1, 3, 4]], source['positions'][[0, 1, 1, 0]])
        np.testing.assert_array_equal(result['weights'][:6], source['weights'][result['source_vertex_ids'][:6]])
        self.assertAlmostEqual(result['report']['maximum_geometric_turn_degrees'], 14, places=4)
        self.assertTrue(result['report']['intersections']['passed'])
        self.assertEqual(result['report']['surface_cards'], 1)

    def test_tilted_normals_use_stricter_of_both_face_ceilings(self):
        data = fixture(); source = data[0]
        source['positions'][[2, 3], 1] = [.001, -.001]
        normal = np.array([.3, .4, np.sqrt(.75)])
        source['corner_normals'][:2] = normal
        data[4]['profile'] = ADAPTIVE_POLICY
        result = run(data)
        expected = float(source['positions'][2, 1])*np.tan(np.deg2rad(14))/(normal[2]+.4*np.tan(np.deg2rad(14)))
        self.assertAlmostEqual(result['surface_card_lifts_m'][0], expected, places=12)
        delta = result['positions'][[2, 5]].astype(float)-source['positions'][[2, 3]]
        np.testing.assert_allclose(delta, np.tile(normal*expected, (2, 1)), atol=1e-10)
        self.assertLessEqual(result['report']['maximum_geometric_turn_degrees'], 15)
        self.assertGreater(result['report']['maximum_geometric_turn_degrees'], 13.999)

    def test_cap_and_retained_contour_are_unchanged_for_healthy_faces(self):
        data = fixture(); fixed = run(data)
        data[4]['profile'] = ADAPTIVE_POLICY
        adaptive = run(data)
        self.assertEqual(adaptive['surface_card_lifts_m'].tolist(), [.0012])
        for key, value in fixed.items():
            if key != 'report':
                np.testing.assert_array_equal(adaptive[key], value)
        self.assertEqual(adaptive['report']['adaptive_lift']['reduced_cards'], 0)
        self.assertEqual(adaptive['report']['policy'], ADAPTIVE_POLICY)
        data[4]['lift_m'] = .0005
        smaller = run(data)
        self.assertEqual(smaller['surface_card_lifts_m'].tolist(), [.0005])

    def test_v1_fixed_profile_has_frozen_array_and_atlas_bytes(self):
        result = run(fixture()); digest = hashlib.sha256()
        for key in sorted(k for k in result if k != 'report'):
            digest.update(key.encode()); digest.update(result[key].tobytes())
        self.assertEqual(digest.hexdigest(), 'abd3620eae437eb91e3ae97cbd9b5d3cda4d28ae745b7411334fbb5daf4410d7')
        self.assertNotIn('adaptive_lift', result['report'])
        self.assertNotIn('surface_card_lifts_m', result)


if __name__ == '__main__':
    unittest.main()
