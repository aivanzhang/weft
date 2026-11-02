"""Analytic authoring tests, not generated-avatar or rendered-fur acceptance."""

import copy
import hashlib
import json
import time
import unittest
from unittest.mock import patch

import numpy as np

from scripts.edge_fringe import ATLAS_DOMAIN_M, FRAME_POLICY, build_edge_fringe, export_budget, two_tip_atlas, validate_settings


def settings(count=16, minimum=1, **changes):
    result = {'seed': 831, 'min_fins': minimum, 'max_fins': count, 'length_m': .008,
              'min_spacing_m': .001, 'min_edge_m': .0085, 'max_edge_m': .020, 'comb': .2}
    result.update(changes)
    return result


def data(points, faces, normals, uv):
    points, faces = np.asarray(points, dtype=np.float64), np.asarray(faces, dtype=np.int32)
    nv, nf = len(points), len(faces)
    weights = np.zeros((nv, 4), dtype=np.float64)
    weights[:, 0] = np.linspace(.2, .8, nv)
    weights[:, 1] = 1-weights[:, 0]
    return {'points': points, 'faces': faces, 'corner_normals': np.asarray(normals, dtype=np.float64),
            'uv0': np.asarray(uv, dtype=np.float64), 'materials': np.zeros(nf, dtype=np.int32),
            'parts': np.zeros(nf, dtype=np.int32), 'fur_faces': np.ones(nf, dtype=bool),
            'non_fur_faces': np.zeros(nf, dtype=bool), 'closure_faces': np.zeros(nf, dtype=bool),
            'joints': np.tile([0, 1, 0, 0], (nv, 1)).astype(np.uint16), 'weights': weights,
            'flow_directions': np.tile([1., .1, .2], (nv, 1)),
            'design_views': np.asarray([[1., 0, .2], [-1., 0, .2], [0, 1., .2], [0, 0, 1.]]),
            'source_export_vertices': nv, 'source_export_triangles': nf}


def quads(count=4, angular=False, sides=None):
    points, faces, normals, uv = [], [], [], []
    for i in range(count):
        side = sides[i] if sides is not None else .007
        angle = np.radians((i % 3)*30 if angular else 0)
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        local = np.array([[0., 0, 0], [side, 0, 0], [side, side, 0], [0, side, 0]])
        points.extend(local@rotation.T+[i*.04, 0, 0])
        faces.extend(np.array([[0, 1, 2], [0, 2, 3]])+i*4)
        normals.extend(np.tile(rotation[:, 2], (2, 3, 1)))
        uv.extend([[[0., 0], [1, 0], [1, 1]], [[0., 0], [1, 1], [0, 1]]])
    result = data(points, faces, normals, uv)
    result['materials'] = np.repeat(np.arange(count) % 32, 2).astype(np.int32)
    result['parts'] = np.repeat(np.arange(count) % 256, 2).astype(np.int32)
    return result


def grid(nx=10, ny=9, curved=True):
    xx, yy = np.meshgrid(np.arange(nx)*.01, np.arange(ny)*.01)
    z = .1*(xx**2+yy**2) if curved else np.zeros_like(xx)
    points = np.column_stack((xx.ravel(), yy.ravel(), z.ravel()))
    ids = np.arange(nx*ny).reshape(ny, nx)
    a, b, c, d = (item.ravel() for item in (ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]))
    faces = np.stack((np.column_stack((a, b, c)), np.column_stack((a, c, d))), axis=1).reshape(-1, 3)
    normals = np.column_stack((-.2*xx.ravel(), -.2*yy.ravel(), np.ones(nx*ny))) if curved else np.tile([0., 0, 1.], (nx*ny, 1))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    uv = np.column_stack((xx.ravel()/xx.max(), yy.ravel()/yy.max()))
    return data(points, faces, normals[faces], uv[faces])


def build(fixture, config=None):
    return build_edge_fringe(**fixture, settings=config or settings())


class EdgeFringeTests(unittest.TestCase):
    def test_curved_and_angular_surfaces_preserve_source_correspondence(self):
        for fixture in (grid(), quads(8, angular=True)):
            saved = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in fixture.items()}
            result = build(fixture)
            ids, face, corner = (result[key] for key in ('root_vertex_ids', 'root_face_ids', 'root_corner_ids'))
            count = len(ids)
            np.testing.assert_array_equal(fixture['faces'][face[:, None], corner], ids)
            np.testing.assert_array_equal(result['root_source_uv'], fixture['uv0'][face[:, None], corner])
            np.testing.assert_array_equal(result['material_ids'], fixture['materials'][face])
            np.testing.assert_array_equal(result['positions'].reshape(count, 4, 3)[:, :2], fixture['points'][ids].astype(np.float32))
            expected = fixture['corner_normals'][face[:, None], corner]
            expected /= np.linalg.norm(expected, axis=2, keepdims=True)
            np.testing.assert_array_equal(result['normals'].reshape(count, 4, 3), np.tile(expected, (1, 2, 1)).astype(np.float32))
            self.assertTrue(np.all(np.linalg.norm(result['normals'], axis=1) > .999999))
            self.assertGreater(result['report']['min_outward_dot'], .9)
            self.assertNotIn('colors', result)
            self.assertEqual(result['report']['profile'], FRAME_POLICY)
            self.assertFalse(result['report']['quality_approved'])
            for key, original in saved.items():
                np.testing.assert_array_equal(fixture[key], original)

    def test_opposed_windings_keep_outward_source_shading_on_both_sides(self):
        result = build(quads(), settings(4))
        count = result['report']['selected_fins']
        first, second = np.split(result['triangles'], 2)
        np.testing.assert_array_equal(second, first[:, ::-1])
        self.assertEqual(len(first), 2*count)
        positions = result['positions'][first].astype(np.float64)
        cross = np.cross(positions[:, 1]-positions[:, 0], positions[:, 2]-positions[:, 0])
        self.assertTrue(np.all(np.linalg.norm(cross, axis=1) > 1e-8))
        plane = cross/np.linalg.norm(cross, axis=1, keepdims=True)
        self.assertTrue(np.all(np.abs(np.einsum('fc,fc->f', plane, result['normals'][first[:, 0]])) < .3))
        self.assertEqual(result['report']['material_requirement'], {'alphaMode': 'MASK', 'alphaCutoff': .18, 'doubleSided': False})

    def test_exact_skin_inheritance_preserves_roots_under_independent_joint_motion(self):
        fixture = quads(6, angular=True)
        result = build(fixture)
        ids = result['root_vertex_ids']
        source = fixture['weights'][ids]
        expected = np.tile(source/source.sum(axis=2, keepdims=True), (1, 2, 1)).astype(np.float32)
        np.testing.assert_array_equal(result['weights'].reshape(-1, 4, 4), expected)
        np.testing.assert_array_equal(result['joints'].reshape(-1, 4, 4), np.tile(fixture['joints'][ids], (1, 2, 1)))
        angle = .6
        matrices = np.tile(np.eye(4), (2, 1, 1))
        matrices[1, :3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        matrices[0, :3, 3], matrices[1, :3, 3] = [.1, -.2, .3], [-.2, .1, .05]
        root = fixture['points'][ids].reshape(-1, 3).astype(np.float32)
        source_joints = fixture['joints'][ids].reshape(-1, 4)
        source_weights = expected[:, :2].reshape(-1, 4)
        def posed(points, joints, weights):
            homogeneous = np.column_stack((points, np.ones(len(points))))
            return np.sum(np.einsum('nkij,nj->nki', matrices[joints], homogeneous)[..., :3]*weights[..., None], axis=1)
        actual = posed(result['positions'], result['joints'], result['weights']).reshape(-1, 4, 3)[:, :2].reshape(-1, 3)
        np.testing.assert_array_equal(actual, posed(root, source_joints, source_weights))

    def test_exact_material_part_uv_normal_and_exclusion_boundaries_are_not_crossed(self):
        cases = [('material_seam', lambda f: f['materials'].__setitem__(1, 31)),
                 ('moving_part_seam', lambda f: f['parts'].__setitem__(1, 255)),
                 ('UV_seam', lambda f: f['uv0'].__setitem__((1, 0, 0), .001)),
                 ('normal_seam', lambda f: f['corner_normals'].__setitem__((1, 0), [.1, 0, np.sqrt(.99)])),
                 ('authored_closure', lambda f: f['closure_faces'].__setitem__(1, True)),
                 ('non_fur', lambda f: f['non_fur_faces'].__setitem__(1, True)),
                 ('non_fur', lambda f: f['fur_faces'].__setitem__(1, False)),
                 ('inconsistent_winding', lambda f: f['faces'].__setitem__(1, f['faces'][1, ::-1]))]
        for reason, change in cases:
            with self.subTest(reason=reason):
                fixture = quads()
                change(fixture)
                result = build(fixture, settings(4))
                self.assertEqual(result['report']['selected_fins'], 3)
                self.assertEqual(result['report']['exclusions'][reason], 1)
                self.assertTrue(np.all(result['root_face_ids'] >= 2))
                self.assertFalse(np.any(result['root_vertex_ids'] < 4))

    def test_exact_coincident_alias_is_excluded_even_on_nonfur_separate_indices(self):
        fixture = quads()
        old_nv, old_nf = len(fixture['points']), len(fixture['faces'])
        clone_ids = fixture['faces'][0]
        fixture['points'] = np.concatenate((fixture['points'], fixture['points'][clone_ids]))
        fixture['faces'] = np.concatenate((fixture['faces'], np.arange(old_nv, old_nv+3)[None]))
        for name in ('joints', 'weights', 'flow_directions'):
            fixture[name] = np.concatenate((fixture[name], fixture[name][clone_ids]))
        for name in ('corner_normals', 'uv0', 'materials', 'parts', 'fur_faces', 'non_fur_faces', 'closure_faces'):
            fixture[name] = np.concatenate((fixture[name], fixture[name][:1]))
        fixture['fur_faces'][-1], fixture['non_fur_faces'][-1] = False, True
        fixture['source_export_vertices'], fixture['source_export_triangles'] = old_nv+3, old_nf+1
        result = build(fixture, settings(4))
        self.assertEqual(result['report']['coincident_shared_edge_aliases'], 1)
        self.assertEqual(result['report']['exclusions']['coincident_edge_alias'], 1)
        self.assertTrue(np.all(result['root_vertex_ids'] >= 4))
        self.assertEqual(len(fixture['points']), old_nv+3)

    def test_nonmanifold_and_opposed_source_normal_roots_are_excluded(self):
        fixture = quads()
        fixture['faces'] = np.concatenate((fixture['faces'], fixture['faces'][:1]))
        for name in ('corner_normals', 'uv0', 'materials', 'parts', 'fur_faces', 'non_fur_faces', 'closure_faces'):
            fixture[name] = np.concatenate((fixture[name], fixture[name][:1]))
        fixture['source_export_triangles'] += 1
        result = build(fixture)
        self.assertEqual(result['report']['nonmanifold_edges'], 1)
        self.assertTrue(np.all(result['root_vertex_ids'] >= 4))
        fixture = quads()
        fixture['corner_normals'][:2] *= -1
        result = build(fixture)
        self.assertEqual(result['report']['exclusions']['incoherent_normals'], 1)
        self.assertTrue(np.all(result['root_vertex_ids'] >= 4))

    def test_seeded_bytes_and_report_are_deterministic_except_elapsed_time(self):
        fixture = grid(14, 13)
        a, b = build(fixture), build(fixture)
        for key in a:
            if key != 'report':
                self.assertEqual(a[key].dtype, b[key].dtype)
                self.assertEqual(a[key].tobytes(), b[key].tobytes(), key)
        a['report'].pop('elapsed_seconds'); b['report'].pop('elapsed_seconds')
        self.assertEqual(a['report'], b['report'])
        different = build(fixture, settings(seed=832))
        self.assertNotEqual(a['root_vertex_ids'].tobytes(), different['root_vertex_ids'].tobytes())
        json.dumps(a['report'], allow_nan=False)

    def test_negative_repeat_uvs_preserve_raw_endpoints_and_seam_identity(self):
        fixture = quads(2)
        fixture['uv0'] = fixture['uv0']*1.16+[-.153, -.151]
        result = build(fixture, settings(2, minimum=2))
        face, corner = result['root_face_ids'], result['root_corner_ids']
        np.testing.assert_array_equal(result['root_source_uv'], fixture['uv0'][face[:, None], corner])
        self.assertLess(result['root_source_uv'].min(), 0)
        self.assertGreater(result['root_source_uv'].max(), 1)
        # Equivalent wrapped values still mark a raw chart seam, never a weld.
        fixture['uv0'][1, 0, 0] += 1
        result = build(fixture, settings(2))
        self.assertEqual(result['report']['exclusions']['UV_seam'], 1)
        self.assertEqual(result['report']['selected_fins'], 1)

    def test_allocation_mass_tracks_area_not_equal_edge_counts(self):
        fixture = quads(200, sides=[.007]*100+[.014]*100)
        result = build(fixture, settings(80))
        small = np.count_nonzero(result['root_face_ids'] < 200)
        self.assertGreater(80-small, small)
        area = 100*(.007**2+.014**2)
        self.assertAlmostEqual(result['report']['allocation']['edge_area_mass_sum_m2'], area, places=12)
        self.assertAlmostEqual(result['report']['allocation']['represented_face_area_m2'], area, places=12)

    def test_physical_crop_preserves_supported_domain_and_atlas_is_bounded(self):
        result = build(quads(2, sides=[.007, .014]), settings(2))
        uv = result['uv'].reshape(-1, 4, 2)
        span = np.linalg.norm(np.diff(result['positions'].reshape(-1, 4, 3)[:, :2], axis=1), axis=2).ravel()
        np.testing.assert_allclose((uv[:, 1, 0]-uv[:, 0, 0])*4*ATLAS_DOMAIN_M, span, atol=1e-8, rtol=0)
        atlas = two_tip_atlas(831)
        self.assertEqual(atlas.shape, (1024, 1024, 4))
        self.assertEqual(atlas.dtype, np.uint8)
        self.assertTrue(np.all(atlas[..., :3] == 255))
        self.assertEqual(hashlib.sha256(atlas.tobytes()).digest(), hashlib.sha256(two_tip_atlas(831).tobytes()).digest())
        alpha = atlas[:256, :256, 3]
        occupied = np.flatnonzero((alpha > 46).any(axis=0))
        self.assertTrue(np.all(np.abs((occupied/255-.5)*ATLAS_DOMAIN_M) < .0034))
        self.assertGreater(len(np.flatnonzero(np.diff(occupied) > 1)), 0)
        self.assertTrue(np.all(alpha[:4] == 0) and np.all(alpha[-4:] == 0))
        for args in ((True,), (-1,), (1, True), (1, 2048), (1.,)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                two_tip_atlas(*args)

    def test_settings_and_conservative_export_budgets_never_shrink_requests(self):
        cfg = settings()
        self.assertEqual(validate_settings(cfg), cfg)
        self.assertIsNot(validate_settings(cfg), cfg)
        budget = export_budget(51367, 62581, 3711)
        self.assertEqual(budget['projected_vertices'], 99995)
        self.assertEqual(budget['projected_triangles'], 77425)
        for values in ((51367, 62581, 3712), (51367, 62581, 4800), (3, 99997, 1),
                       (True, 2, 1), (4, True, 1), (4, 2, True), (4, 2, 4801), (4, 2, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                export_budget(*values)
        bad = [{'x': 1}, {**cfg, 'other': 2}, {k:v for k, v in cfg.items() if k != 'seed'}]
        for key in cfg:
            for value in (True, None, '1', float('nan'), float('inf')):
                bad.append({**cfg, key: value})
        bad += [{**cfg, 'min_edge_m': .0084}, {**cfg, 'max_edge_m': .0201}, {**cfg, 'min_fins': 17},
                {**cfg, 'length_m': .0009}, {**cfg, 'min_spacing_m': 0}, {**cfg, 'comb': .36}]
        for value in bad:
            with self.subTest(settings=value), self.assertRaises(ValueError):
                validate_settings(value)

    def test_invalid_oversized_nonfinite_inputs_fail_before_edge_work(self):
        fixture = quads()
        invalid = [('points', np.zeros((100001, 3))), ('faces', np.zeros((100001, 3), dtype=int)),
                   ('points', fixture['points'].astype(bool)), ('faces', fixture['faces'].astype(float)),
                   ('points', np.full_like(fixture['points'], np.nan)), ('corner_normals', np.full_like(fixture['corner_normals'], np.inf)),
                   ('uv0', fixture['uv0']+1e6+1), ('joints', fixture['joints']+64), ('weights', fixture['weights']*0),
                   ('flow_directions', fixture['flow_directions']*0), ('design_views', np.zeros((17, 3))),
                   ('materials', fixture['materials']+32), ('parts', fixture['parts']+256),
                   ('fur_faces', fixture['fur_faces'].astype(int)), ('source_export_vertices', 3),
                   ('source_export_triangles', 1)]
        bad_index = fixture['faces'].copy(); bad_index[0, 0] = -1
        invalid.append(('faces', bad_index))
        with patch('scripts.edge_fringe._edge_table', side_effect=AssertionError('Premature edge work')):
            for key, value in invalid:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    build({**fixture, key: value})
        degenerate = copy.deepcopy(fixture); degenerate['points'][1] = degenerate['points'][0]
        with self.assertRaisesRegex(ValueError, 'Degenerate'):
            build(degenerate)

    def test_insufficient_spacing_minimum_and_deadline_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            build(quads(2), settings(3, minimum=3))
        fixture = quads(2)
        fixture['points'][4:, 0] -= .03
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            build(fixture, settings(2, minimum=2, min_spacing_m=.05))
        with patch('scripts.edge_fringe.time.monotonic', side_effect=[0., 11.]), self.assertRaisesRegex(ValueError, 'deadline'):
            build(quads())

    def test_maximum_candidate_scan_is_bounded_with_actual_spacing(self):
        fixture = grid(192, 192, curved=False)
        started = time.monotonic()
        result = build(fixture, settings(4800, minimum=500, min_spacing_m=.05))
        report = result['report']
        self.assertEqual(report['examined_candidates'], 32000)
        self.assertLess(report['selected_fins'], 4800)
        self.assertGreaterEqual(report['selected_fins'], 500)
        self.assertGreater(report['spacing_rejected'], 20000)
        self.assertLess(report['elapsed_seconds'], 10)
        print(json.dumps({'analytic_fixture': 'maximum_candidate_scan', 'elapsed_seconds': time.monotonic()-started,
                          'source_vertices': len(fixture['points']), 'source_faces': len(fixture['faces']),
                          'selected_fins': report['selected_fins'], 'examined_candidates': report['examined_candidates'],
                          'projected_vertices': report['export_budget']['projected_vertices'],
                          'quality_approved': False}))

    def test_maximum_output_triangle_budget_uses_real_large_input(self):
        fixture = grid(225, 224, curved=False)
        result = build(fixture, settings(24, minimum=24))
        self.assertEqual(len(fixture['faces']), 99904)
        self.assertEqual(result['report']['export_budget']['projected_triangles'], 100000)
        self.assertEqual(result['report']['selected_fins'], 24)
        self.assertLess(result['report']['elapsed_seconds'], 10)
        print(json.dumps({'analytic_fixture': 'maximum_projected_triangles',
                          'elapsed_seconds': result['report']['elapsed_seconds'],
                          'source_vertices': len(fixture['points']), 'source_faces': len(fixture['faces']),
                          'selected_fins': 24, 'projected_triangles': 100000, 'quality_approved': False}))


if __name__ == '__main__':
    unittest.main()
