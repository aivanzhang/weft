"""Analytic normal-transfer and mutation-boundary tests; not native render evidence."""

from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from avatar_assets.normal_continuity import (
    FACE_ATTRIBUTE, POINT_ATTRIBUTE, NormalContinuityError, plan_normal_continuity, validate_normal_continuity,
)
from scripts.blender_normal_continuity import apply_normal_continuity


def config(**changes):
    return {'name': 'contact', 'materials': ['cap'], 'reference_materials': ['surface'],
            'max_distance_m': .01, 'min_vertices': 3, 'max_vertices': 3, **changes}


def geometry():
    reference = np.array([[0., 0, 0], [1., 0, 0], [0, 1., 0]])
    points = np.concatenate((reference, reference+[0, 0, .005], reference+[2, 0, 0]))
    faces = np.arange(9).reshape(3, 3)
    normals = np.array([[0., 0, 1]]*3+[[0., 0, -1]]*3+[[1., 0, 0]]*3)
    return points, faces, [0, 1, 2], ['surface', 'cap', 'other'], normals


class Values:
    def __init__(self, count, width=1, values=None):
        self.values = np.zeros((count, width)) if values is None else np.array(values, dtype=float)

    def __len__(self):
        return len(self.values)

    def foreach_get(self, field, output):
        output[:] = self.values.reshape(-1)

    def foreach_set(self, field, values):
        self.values[:] = np.asarray(values).reshape(self.values.shape)


class Attributes(dict):
    def __init__(self, vertices, faces):
        super().__init__()
        self.vertices, self.faces = vertices, faces

    def new(self, name, type, domain):
        value = SimpleNamespace(name=name, data_type=type, domain=domain,
            data=Values(self.vertices if domain == 'POINT' else self.faces, 3 if type == 'FLOAT_VECTOR' else 1))
        self[name] = value
        return value


class Mesh:
    def __init__(self, points, faces, slots, names, normals):
        self.vertices = [SimpleNamespace(co=tuple(map(float, point)), groups=[SimpleNamespace(group=0, weight=1.)]) for point in points]
        self.polygons = [SimpleNamespace(vertices=tuple(map(int, face)), material_index=slot, use_smooth=False)
                         for face, slot in zip(faces, slots)]
        self.loops = [SimpleNamespace(vertex_index=int(i)) for face in faces for i in face]
        self.corner_normals = [SimpleNamespace(vector=tuple(map(float, normal))) for normal in normals]
        self.materials = [SimpleNamespace(name=name) for name in names]
        self.uv_layers = [SimpleNamespace(name='UVMap', data=Values(len(self.loops), 2,
            np.arange(len(self.loops)*2).reshape(-1, 2)/100))]
        self.attributes = Attributes(len(points), len(faces))
        self.normal_writes, self.updates = 0, 0

    def normals_split_custom_set(self, normals):
        self.corner_normals = [SimpleNamespace(vector=tuple(normal)) for normal in normals]
        self.normal_writes += 1

    def update(self):
        self.updates += 1


def obj():
    return SimpleNamespace(type='MESH', data=Mesh(*geometry()), matrix_world=np.eye(4),
                           vertex_groups=[SimpleNamespace(index=0, name='Root')])


class NormalContinuityTests(unittest.TestCase):
    def test_config_is_detached_exact_bounded_and_nonoverlapping(self):
        original = [config()]
        result = validate_normal_continuity(original)
        result[0]['materials'][0] = 'different'
        self.assertEqual(original, [config()])
        self.assertEqual(validate_normal_continuity([]), [])
        bad = [[config()]*5, [config(name='../bad')], [config(extra=True)], [config(materials=[])],
               [config(materials=['cap', 'cap'])], [config(materials=[' cap'])],
               [config(materials=['cap\n'])], [config(materials=['x'*129])],
               [config(reference_materials=['cap'])], [config(), config(name='second')],
               [config(), config(name='second', materials=['other'], reference_materials=['cap'])],
               [config(max_distance_m=True)], [config(max_distance_m=float('nan'))],
               [config(max_distance_m=float('inf'))], [config(max_distance_m=-.001)],
               [config(max_distance_m=.03001)], [config(min_vertices=0)],
               [config(max_vertices=10001)], [config(max_vertices=True)], [config(min_vertices=4)]]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_normal_continuity(value)

    def test_exact_and_nearest_transfer_only_target_corners(self):
        values = geometry()
        before = deepcopy(values)
        planned = plan_normal_continuity(*values, [config()])
        np.testing.assert_array_equal(planned['corner_normals'][:3], values[-1][:3])
        np.testing.assert_array_equal(planned['corner_normals'][6:], values[-1][6:])
        np.testing.assert_array_equal(planned['corner_normals'][3:6], [[0, 0, 1]]*3)
        np.testing.assert_array_equal(planned['repaired_faces'], [False, True, False])
        np.testing.assert_array_equal(np.flatnonzero(planned['repaired_vertices']), [3, 4, 5])
        self.assertEqual(planned['report']['corrections'][0]['angle_degrees']['max'], 180.)
        self.assertAlmostEqual(planned['report']['corrections'][0]['maximum_distance_m'], .005)
        json.dumps(planned['report'], allow_nan=False)
        for a, b in zip(values, before):
            np.testing.assert_array_equal(a, b)
        exact = list(geometry())
        exact[0][3:6] = exact[0][:3]
        exact_result = plan_normal_continuity(*exact, [config(max_distance_m=0)])
        self.assertEqual(exact_result['report']['candidate_pairs_tested'], 0)
        self.assertEqual(exact_result['report']['corrections'][0]['maximum_distance_m'], 0)
        with self.assertRaisesRegex(ValueError, 'matched 0 vertices'):
            plan_normal_continuity(*values, [config(max_distance_m=0)])

    def test_original_reference_split_corners_average_at_exact_positions(self):
        points, faces, slots, names, normals = geometry()
        points[6:9] = points[:3]
        slots[2] = 0
        normals[6:9] = [0, 1, 0]
        result = plan_normal_continuity(points, faces, slots, names, normals, [config()])
        np.testing.assert_allclose(result['vertex_normals'][3:6], [[0, 2**-.5, 2**-.5]]*3)
        self.assertEqual(result['report']['corrections'][0]['reference_unique_positions'], 3)
        self.assertEqual(result['report']['corrections'][0]['reference_corners'], 6)
        normals[6:9] = [0, 0, -1]
        with self.assertRaisesRegex(ValueError, 'cancelling or zero reference'):
            plan_normal_continuity(points, faces, slots, names, normals, [config()])

    def test_shared_vertex_does_not_rewrite_unrelated_corner_and_cancelled_point_has_original_fallback(self):
        points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]]
        faces = [[0, 1, 2], [0, 3, 4]]
        normals = np.array([[0., 0, 1]]*3+[[0., 0, -1]]*3)
        result = plan_normal_continuity(points, faces, [0, 1], ['surface', 'cap'], normals,
                                        [config(max_distance_m=0)])
        np.testing.assert_array_equal(result['corner_normals'][:3], normals[:3])
        np.testing.assert_array_equal(result['corner_normals'][3:], [[0, 0, 1]]*3)
        self.assertEqual(result['report']['point_normal_cancellation_fallbacks'], 1)

    def test_each_transferred_corner_must_face_outward_even_if_mean_is_positive(self):
        for bad_normal in ([0, 0, -1], [1, 0, 0]):
            values = list(geometry())
            values[-1][0] = bad_normal
            original = deepcopy(values)
            with self.subTest(normal=bad_normal), self.assertRaises(NormalContinuityError) as error:
                plan_normal_continuity(*values, [config()])
            report = error.exception.report
            self.assertEqual(report['outside_hemisphere_corners'], 1)
            self.assertEqual(report['degenerate_faces'], 0)
            self.assertEqual(report['witnesses'][0]['corner'], 3)
            self.assertEqual(report['witnesses'][0]['face'], 1)
            json.dumps(report, allow_nan=False)
            for actual, before in zip(values, original):
                np.testing.assert_array_equal(actual, before)

    def test_polygon_area_handles_collinear_first_three_corners_and_scale(self):
        polygon = np.array([[0., 0, 0], [.5, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        for scale in (1e-6, 1, 1e5):
            points = np.concatenate((polygon, polygon))*scale
            result = plan_normal_continuity(points, [list(range(5)), list(range(5, 10))],
                [0, 1], ['surface', 'cap'], [[0, 0, 1]]*10,
                [config(max_distance_m=0, min_vertices=5, max_vertices=5)])
            self.assertEqual(result['report']['corrections'][0]['hemisphere_check']['minimum_face_dot'], 1)

    def test_degenerate_or_cancelling_changed_polygons_fail_without_guessing(self):
        for polygon in ([[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                        [[0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0]]):
            count = len(polygon)
            with self.subTest(polygon=polygon), self.assertRaises(NormalContinuityError) as error:
                plan_normal_continuity(polygon+polygon,
                    [list(range(count)), list(range(count, count*2))], [0, 1], ['surface', 'cap'],
                    [[0, 0, 1]]*(count*2),
                    [config(max_distance_m=0, min_vertices=count, max_vertices=count)])
            report = error.exception.report
            self.assertEqual(report['degenerate_faces'], 1)
            self.assertIsNone(report['minimum_face_dot'])
            json.dumps(report, allow_nan=False)

    def test_rejection_witnesses_are_bounded_and_disabled_transfer_is_unchanged(self):
        points, faces, slots, names, normals = geometry()
        normals[:3] = [0, 0, -1]
        copies = 12
        points = np.tile(points, (copies, 1))
        faces = np.concatenate([faces+i*9 for i in range(copies)])
        slots, normals = slots*copies, np.tile(normals, (copies, 1))
        with self.assertRaises(NormalContinuityError) as error:
            plan_normal_continuity(points, faces, slots, names, normals,
                [config(min_vertices=copies*3, max_vertices=copies*3)])
        self.assertEqual(error.exception.report['outside_hemisphere_corners'], 36)
        self.assertEqual(len(error.exception.report['witnesses']), 32)
        disabled = plan_normal_continuity(points, faces, slots, names, normals, [])
        np.testing.assert_array_equal(disabled['corner_normals'], normals)

    def test_hemisphere_rejection_precedes_all_native_mutations(self):
        value = obj()
        value.data.corner_normals[0].vector = (0, 0, -1)
        original = deepcopy(value.data)
        with self.assertRaises(NormalContinuityError):
            apply_normal_continuity(value, [config()])
        self.assertEqual((value.data.normal_writes, value.data.updates), (0, 0))
        self.assertFalse(value.data.attributes)
        self.assertEqual(value.data.vertices, original.vertices)
        self.assertEqual(value.data.corner_normals, original.corner_normals)
        self.assertFalse(any(face.use_smooth for face in value.data.polygons))
        np.testing.assert_array_equal(value.data.uv_layers[0].data.values, original.uv_layers[0].data.values)

    def test_invalid_geometry_bounds_and_ambiguous_materials_fail(self):
        values = geometry()
        for index, bad in [(0, np.zeros((100001, 3))), (0, [[True, 0, 0]]*9),
                           (0, np.full((9, 3), float('nan'))), (0, np.full((9, 3), 1e308)),
                           (1, np.zeros((100001, 3), dtype=int)), (1, np.zeros((75001, 4), dtype=int)),
                           (1, [[0., 1., 2.]]), (1, [[0, 1, 99]]),
                           (2, [True, 1, 2]), (2, [0, 1, 3]), (2, [0, 1]),
                           (3, ['surface', 'cap', 'cap']), (4, np.zeros((9, 3))),
                           (4, np.ones((9, 3))), (4, np.zeros((300001, 3)))]:
            changed = list(values)
            changed[index] = bad
            with self.subTest(index=index), self.assertRaises(ValueError):
                plan_normal_continuity(*changed, [config()])
        with self.assertRaisesRegex(ValueError, 'absent exact material'):
            plan_normal_continuity(*values, [config(materials=['Cap'])])
        with self.assertRaisesRegex(ValueError, 'matched 3 vertices'):
            plan_normal_continuity(*values, [config(min_vertices=2, max_vertices=2)])

    def test_nearest_search_budget_and_target_count_bound_precede_mutation(self):
        points, faces, slots, names, normals = geometry()
        points[:3] = [[0, 0, 0], [.001, 0, 0], [.002, 0, 0]]
        points[3:6] = points[:3]+[0, .001, 0]
        with patch('avatar_assets.normal_continuity.MAX_PAIRS', 5), self.assertRaisesRegex(ValueError, 'candidate pairs'):
            plan_normal_continuity(points, faces, slots, names, normals, [config()])
        with patch('avatar_assets.normal_continuity.MAX_TARGET_VERTICES', 2), self.assertRaisesRegex(ValueError, '10000 target'):
            plan_normal_continuity(*geometry(), [config(min_vertices=2, max_vertices=2)])

    def test_native_adapter_writes_attributes_and_preserves_positions_uvs_weights(self):
        value = obj()
        original = deepcopy(value.data)
        report = apply_normal_continuity(value, [config()])
        self.assertEqual(report['status'], 'applied')
        self.assertEqual(report['protected_before'], report['protected_after'])
        self.assertTrue(report['geometry_uv_skin_unchanged'])
        self.assertEqual([face.use_smooth for face in value.data.polygons], [False, True, False])
        self.assertEqual((value.data.normal_writes, value.data.updates), (1, 1))
        normal, repaired = value.data.attributes[POINT_ATTRIBUTE], value.data.attributes[FACE_ATTRIBUTE]
        self.assertEqual((normal.data_type, normal.domain), ('FLOAT_VECTOR', 'POINT'))
        self.assertEqual((repaired.data_type, repaired.domain), ('BOOLEAN', 'FACE'))
        np.testing.assert_allclose(np.linalg.norm(normal.data.values, axis=1), 1)
        np.testing.assert_array_equal(repaired.data.values[:, 0], [0, 1, 0])
        self.assertEqual(value.data.vertices, original.vertices)
        np.testing.assert_array_equal(value.data.uv_layers[0].data.values, original.uv_layers[0].data.values)
        json.dumps(report, allow_nan=False)

    def test_invalid_native_inputs_do_not_mutate_mesh_or_overwrite_attributes(self):
        value = obj()
        self.assertEqual(apply_normal_continuity(value, [])['status'], 'not_requested')
        self.assertFalse(value.data.attributes)
        for mutate in [lambda o: setattr(o, 'matrix_world', np.diag([2, 1, 1, 1])),
                       lambda o: setattr(o.data.loops[0], 'vertex_index', 1),
                       lambda o: setattr(o.data.vertices[0].groups[0], 'weight', float('nan')),
                       lambda o: o.data.attributes.update({POINT_ATTRIBUTE: 'preserved'}),
                       lambda o: setattr(o.data.uv_layers[0].data, 'values', np.full((9, 2), float('inf')))]:
            value = obj()
            mutate(value)
            previous = dict(value.data.attributes)
            with self.assertRaises(ValueError):
                apply_normal_continuity(value, [config()])
            self.assertEqual(value.data.normal_writes, 0)
            self.assertEqual(dict(value.data.attributes), previous)
            self.assertFalse(any(face.use_smooth for face in value.data.polygons))
        value = obj()
        with self.assertRaisesRegex(ValueError, 'outside its bounds'):
            apply_normal_continuity(value, [config(min_vertices=4, max_vertices=4)])
        self.assertFalse(value.data.attributes)
        self.assertEqual(value.data.normal_writes, 0)


if __name__ == '__main__':
    unittest.main()
