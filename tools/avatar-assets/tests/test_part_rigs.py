"""Analytic part-skin ownership and math; not native loading or flight acceptance."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from avatar_assets.part_rigs import part_rig_config, part_skin_weights, validate_part_rigs
from scripts.blender_part_rigs import apply_part_rigs


def rig_config():
    return {'space': 'normalized_bounds_blender_z_up', 'height_meters': 1.2,
            'bones': [{'name': 'Body', 'parent': None, 'head': [.3, .5, 0], 'tail': [.3, .5, 1]},
                      {'name': 'PartRoot', 'parent': 'Body', 'head': [.7, .5, .35], 'tail': [.7, .5, .65]},
                      {'name': 'Wing', 'parent': 'PartRoot', 'head': [.7, .5, .6], 'tail': [.98, .5, .6]}],
            'base_bones': ['Body'], 'layers': []}


def part_config():
    return {'name': 'analytic-part', 'materials': ['Part material'], 'root_bone': 'PartRoot',
            'layers': [{'bones': ['Wing'], 'min': [.8, .1, .3], 'max': [1, .9, .8],
                        'feather': .02, 'min_vertices': 2}], 'min_vertices': 4, 'max_vertices': 20}


class Group:
    def __init__(self, name, values=None):
        self.name, self.values = name, dict(values or {})

    def remove(self, ids):
        for index in ids:
            self.values.pop(index, None)

    def add(self, ids, weight, mode):
        assert mode == 'REPLACE'
        for index in ids:
            self.values[index] = weight


class Groups(list):
    def get(self, name):
        return next((group for group in self if group.name == name), None)

    def new(self, name):
        group = Group(name)
        self.append(group)
        return group


def fixture():
    points = np.array([[0, 0, 0], [.3, 0, 0], [0, .3, 1], [.3, .3, .7],
                       [.72, .5, .5], [.9, .4, .6], [.92, .6, .6], [.92, .6, .6]])
    polygons = [SimpleNamespace(vertices=(0, 1, 2), material_index=0),
                SimpleNamespace(vertices=(0, 2, 3), material_index=0),
                SimpleNamespace(vertices=(4, 5, 6), material_index=1),
                SimpleNamespace(vertices=(4, 5, 7), material_index=1)]
    obj = SimpleNamespace(type='MESH', data=SimpleNamespace(vertices=list(map(tuple, points)), polygons=polygons,
        materials=[SimpleNamespace(name='Body material'), SimpleNamespace(name='Part material')],
        uv_layers={'UV': [(0, 0)]*len(points)}), vertex_groups=Groups([Group('Body', {i: 1 for i in range(len(points))})]))
    rig = SimpleNamespace(type='ARMATURE', data=SimpleNamespace(bones={b['name']: deepcopy(b) for b in rig_config()['bones']}))
    return obj, rig, points


class PartRigTests(unittest.TestCase):
    def test_derived_subtree_preserves_global_coordinates_and_original_hierarchy(self):
        authored, part = rig_config(), part_config()
        saved = deepcopy(authored)
        derived = part_rig_config(part, authored)
        self.assertEqual([bone['name'] for bone in derived['bones']], ['PartRoot', 'Wing'])
        self.assertIsNone(derived['bones'][0]['parent'])
        self.assertEqual(derived['bones'][0]['head'], [.7, .5, .35])
        self.assertEqual(derived['base_bones'], ['PartRoot'])
        self.assertEqual(authored, saved)
        derived['bones'][0]['head'][0] = 0
        self.assertEqual(authored, saved)

    def test_selected_weights_use_full_bounds_and_identical_seam_positions(self):
        _, _, points = fixture()
        weights, report = part_skin_weights(points[4:], [0]*3, [1]*3, part_config(), rig_config())
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        np.testing.assert_array_equal(weights[2], weights[3])
        np.testing.assert_array_equal(weights[0], [1, 0])
        np.testing.assert_array_equal(weights[1:], [[0, 1]]*3)
        self.assertEqual(report['original_mesh_bounds'], {'min': [0., 0., 0.], 'max': [1., 1., 1.]})
        self.assertEqual(report['positive_vertices_per_bone'], {'PartRoot': 1, 'Wing': 3})
        self.assertLessEqual(report['max_influences'], 4)

    def test_native_assignment_changes_only_selected_weights(self):
        obj, rig, points = fixture()
        mesh_before, bones_before = deepcopy(obj.data), deepcopy(rig.data.bones)
        report = apply_part_rigs(obj, rig, points, [0]*3, [1]*3, [part_config()], rig_config())
        self.assertEqual(obj.data, mesh_before)
        self.assertEqual(rig.data.bones, bones_before)
        self.assertEqual(obj.vertex_groups.get('Body').values, {i: 1 for i in range(4)})
        self.assertEqual(obj.vertex_groups.get('PartRoot').values, {4: 1})
        self.assertEqual(obj.vertex_groups.get('Wing').values, {5: 1, 6: 1, 7: 1})
        self.assertEqual(report['parts'][0]['materials'], ['Part material'])
        self.assertEqual(report['parts'][0]['faces_by_material'], {'Part material': 2})
        self.assertTrue(report['outside_vertex_weights_unchanged'])

    def test_shared_boundary_missing_material_or_bone_rejects_before_mutation(self):
        for issue in ('boundary', 'material', 'unused-material', 'bone'):
            obj, rig, points = fixture()
            part = part_config()
            if issue == 'boundary':
                obj.data.polygons[0].vertices = (0, 1, 4)
            elif issue == 'material':
                part['materials'] = ['Missing']
            elif issue == 'unused-material':
                obj.data.materials.append(SimpleNamespace(name='Unused'))
                part['materials'].append('Unused')
            else:
                del rig.data.bones['Wing']
            before = deepcopy(obj.vertex_groups[0].values)
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                apply_part_rigs(obj, rig, points, [0]*3, [1]*3, [part], rig_config())
            self.assertEqual(obj.vertex_groups[0].values, before)
            self.assertEqual(len(obj.vertex_groups), 1)

    def test_different_parts_cannot_share_vertices(self):
        obj, rig, points = fixture()
        obj.data.materials.append(SimpleNamespace(name='Second part'))
        obj.data.polygons[3].material_index = 2
        first, second = part_config(), part_config()
        first['min_vertices'] = second['min_vertices'] = 1
        second.update(name='second', materials=['Second part'])
        with self.assertRaisesRegex(ValueError, 'boundary shares'):
            apply_part_rigs(obj, rig, points, [0]*3, [1]*3, [first, second], rig_config())
        self.assertEqual(len(obj.vertex_groups), 1)

    def test_missing_layer_coverage_and_later_overwrite_cannot_silently_pass(self):
        _, _, points = fixture()
        part = part_config()
        part['layers'][0].update(min=[0, 0, 0], max=[.1, .1, .1])
        with self.assertRaisesRegex(ValueError, 'missed expected geometry'):
            part_skin_weights(points[4:], [0]*3, [1]*3, part, rig_config())
        part = part_config()
        part['layers'].append({'bones': ['PartRoot'], 'min': [0, 0, 0], 'max': [1, 1, 1],
                               'feather': 0, 'min_vertices': 1})
        with self.assertRaisesRegex(ValueError, 'no final weight coverage'):
            part_skin_weights(points[4:], [0]*3, [1]*3, part, rig_config())

    def test_strict_validation_rejects_unknown_controls_counts_and_material_ownership(self):
        valid, authored = part_config(), rig_config()
        copied = validate_part_rigs([valid], authored)
        copied[0]['materials'][0] = 'Different'
        self.assertEqual(valid['materials'], ['Part material'])
        bad = [[], [valid]*5, [valid, valid], [{**valid, 'script': 'execute'}],
               [{**valid, 'min_vertices': 0}], [{**valid, 'max_vertices': 100001}],
               [{**valid, 'max_vertices': True}], [{**valid, 'root_bone': 'Missing'}],
               [{**valid, 'materials': ['Part material']*2}], [{**valid, 'materials': [str(i) for i in range(9)]}],
               [valid, {**valid, 'name': 'different'}],
               [{**valid, 'layers': [{**valid['layers'][0], 'bones': ['Body']}]}],
               [{**valid, 'layers': [{**valid['layers'][0], 'min_vertices': 0}]}]]
        for parts in bad:
            with self.subTest(parts=parts), self.assertRaises(ValueError):
                validate_part_rigs(parts, authored)

    def test_oversized_nonfinite_and_outside_bounds_points_are_rejected(self):
        _, _, points = fixture()
        for bad in (np.zeros((100001, 3)), points[4:]*np.nan, points[4:].astype(bool),
                    [[True, .5, .5]]*4, points[4:]+2):
            with self.subTest(shape=np.shape(bad)), self.assertRaises(ValueError):
                part_skin_weights(bad, [0]*3, [1]*3, part_config(), rig_config())


if __name__ == '__main__':
    unittest.main()
