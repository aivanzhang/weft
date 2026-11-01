"""Analytic color/region and isolated-material tests; not native rendering evidence."""

from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from avatar_assets.color_corrections import select_color_faces, srgb_to_linear, validate_corrections
from scripts.blender_color_corrections import apply_color_corrections
from tests.test_surface_profiles import Material, mesh


def region(center=(.2, .2, .2), radius=(.12, .12, .12)):
    return {'center': list(center), 'radius': list(radius)}


def correction(name='hidden-leak', **changes):
    return {'name': name, 'regions': [region()], 'dominant_channel': 1, 'dominant_ratio': 1.3,
            'color_srgb': [.8, .76, .7], 'roughness': .78, 'min_faces': 1, 'max_faces': 1, **changes}


def geometry():
    points = np.array([np.array(center)+offset
                       for center in ((.2, .2, .2), (.2, .2, .8), (.2, .2, .2), (.7, .2, .2))
                       for offset in ((-.02, -.02, 0), (.02, -.02, 0), (0, .04, 0))])
    colors = np.array([[.12, .64, .15]]*6+[[.5, .5, .5]]*3+[[.12, .64, .15]]*3)
    return points, colors, [(i, i+1, i+2) for i in range(0, 12, 3)]


class ColorCorrectionTests(unittest.TestCase):
    def select(self, corrections, points=None, colors=None, faces=None):
        defaults = geometry()
        return select_color_faces(defaults[2] if faces is None else faces, defaults[1] if colors is None else colors,
                                  defaults[0] if points is None else points, [0]*3, [1]*3, corrections)

    def test_validation_detaches_data_and_rejects_limits_nonfinite_and_extra_fields(self):
        original = [correction()]
        copied = validate_corrections(original)
        copied[0]['regions'][0]['center'][0] = .9
        copied[0]['color_srgb'][0] = .1
        self.assertEqual(original, [correction()])
        cases = [[], [correction()]*5, [correction(), correction()], [correction(name='../bad')],
                 [correction(script='run')], [correction(regions=[])], [correction(regions=[region()]*5)],
                 [correction(regions=[region(radius=(0, 1, 1))])], [correction(regions=[region(center=(-1.01, 0, 0))])],
                 [correction(regions=[region(radius=(3.01, 1, 1))])], [correction(regions=[region(center=(0, 0, float('nan')))])],
                 [correction(dominant_channel=True)], [correction(dominant_channel=3)],
                 [correction(dominant_ratio=1.049)], [correction(dominant_ratio=5.01)],
                 [correction(dominant_ratio=float('inf'))], [correction(color_srgb=[0, 0])],
                 [correction(color_srgb=[1.01, 0, 0])], [correction(color_srgb=[False, 0, 0])],
                 [correction(color_srgb=[float('nan'), 0, 0])], [correction(roughness=.039)],
                 [correction(roughness=1.01)], [correction(min_faces=0)], [correction(max_faces=100001)],
                 [correction(min_faces=2, max_faces=1)], [correction(max_faces=True)]]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_corrections(value)

    def test_spatial_and_color_selection_preserves_green_shoulder_and_neutral_faces(self):
        owners, reports = self.select([correction()])
        np.testing.assert_array_equal(owners, [0, -1, -1, -1])
        self.assertEqual(reports, [{'name': 'hidden-leak', 'matched_faces': 1, 'selected_faces': 1}])
        points, colors, faces = geometry()
        colors[:3] = [[.8, .9, .1], [0, .9, .1], [0, .9, .1]]
        owners, _ = self.select([correction(regions=[region(radius=(.001, .001, .001))])], points, colors, faces)
        self.assertEqual(owners[0], 0)  # Face centroid and mean RGB, not every source vertex.

    def test_region_union_last_match_and_final_min_max_ownership(self):
        broad = correction(regions=[region(), region((.7, .2, .2))], max_faces=1)
        later = correction('later', color_srgb=[.1, .1, .1])
        owners, reports = self.select([broad, later])
        np.testing.assert_array_equal(owners, [1, -1, -1, 0])
        self.assertEqual([r['matched_faces'] for r in reports], [2, 1])
        self.assertEqual([r['selected_faces'] for r in reports], [1, 1])
        with self.assertRaisesRegex(ValueError, 'selected 1 final faces'):
            self.select([{**broad, 'min_faces': 2, 'max_faces': 2}, later])
        with self.assertRaisesRegex(ValueError, 'selected 0 final faces'):
            self.select([correction(), later])
        with self.assertRaisesRegex(ValueError, 'selected 2 final faces'):
            self.select([broad])

    def test_dominance_is_strict_and_supports_each_channel(self):
        points, colors, faces = geometry()
        colors[:3] = [.2, .25, .1]
        config = correction(dominant_ratio=1.25, regions=[region(), region((.7, .2, .2))], max_faces=2)
        owners, _ = self.select([config], points, colors, faces)
        np.testing.assert_array_equal(owners, [-1, -1, -1, 0])
        colors[:3, 1] = .25001
        owners, _ = self.select([config], points, colors, faces)
        self.assertEqual(owners[0], 0)
        for channel in (0, 1, 2):
            colors[:3] = .1
            colors[:3, channel] = .9
            owners, _ = self.select([correction(dominant_channel=channel)], points, colors, faces)
            np.testing.assert_array_equal(owners, [0, -1, -1, -1])

    def test_geometry_color_corner_and_numeric_bounds(self):
        points, colors, faces = geometry()
        cases = [(faces, colors[:-1], points), (faces, colors, points*np.nan),
                 (faces, colors+1, points), (faces, colors-.8, points), (faces, colors*np.inf, points),
                 ([(0, 1, 99)], colors, points), ([(0., 1., 2.)], colors, points),
                 (np.array([[-1, 1, 2]]), colors, points), (np.array([[0., 1., 2.]]), colors, points),
                 (np.array(1), colors, points), (faces, colors, np.array(1)),
                 (np.zeros((100001, 3), dtype=int), colors, points),
                 (np.zeros((75001, 4), dtype=int), colors, points),
                 ([(0, 1, 2, 3)]*75001, colors, points),
                 (faces, np.zeros((100001, 3)), np.zeros((100001, 3))),
                 (faces, colors, np.full(points.shape, 1e308))]
        for bad_faces, bad_colors, bad_points in cases:
            with self.subTest(kind=type(bad_faces)), self.assertRaises(ValueError):
                select_color_faces(bad_faces, bad_colors, bad_points, [0]*3, [1]*3, [correction()])
        for low, span in (([0]*3, [0]*3), ([0]*3, [-1, 1, 1]), ([float('nan'), 0, 0], [1]*3),
                          ([0, 0], [1]*3), ([0]*3, [True, 1, 1])):
            with self.subTest(low=low, span=span), self.assertRaises(ValueError):
                select_color_faces(faces, colors, points, low, span, [correction()])

    def test_srgb_conversion_uses_piecewise_transfer_not_direct_assignment(self):
        result = srgb_to_linear([0, .04045, 1])
        self.assertEqual(result[0], 0)
        self.assertAlmostEqual(result[1], .0031308049535603713)
        self.assertEqual(result[2], 1)
        self.assertAlmostEqual(srgb_to_linear([.5]*3)[0], .21404114048223255)
        for color in ([1.1, 0, 0], [0, 0], [True, 0, 0], [float('nan'), 0, 0]):
            with self.subTest(color=color), self.assertRaises(ValueError):
                srgb_to_linear(color)

    def test_copies_disconnect_base_color_and_preserve_originals_geometry_uv_skin(self):
        points, colors, faces = geometry()
        original = Material()
        obj = mesh(points, faces, [original])
        before = deepcopy((obj.data.vertices, obj.data.uv_layers))
        report, affected = apply_color_corrections(obj, colors, points, [0]*3, [1]*3, [correction()])
        self.assertEqual(affected, [0, 1, 2])
        self.assertEqual([p.material_index for p in obj.data.polygons], [1, 0, 0, 0])
        self.assertEqual((report['new_material_count'], report['faces_reassigned']), (1, 1))
        self.assertTrue(report['original_texture_bytes_unchanged'])
        self.assertTrue(report['visible_selected_surface_color_authored'])
        self.assertTrue(report['fur_exclusion_required'])
        json.dumps(report, allow_nan=False)
        corrected = obj.data.materials[1]
        texture, uv_node, bsdf, normal = corrected.node_tree.nodes
        self.assertEqual(bsdf.inputs['Base Color'].default_value, (*srgb_to_linear(correction()['color_srgb']), 1.))
        self.assertEqual(bsdf.inputs['Metallic'].default_value, 0)
        self.assertEqual(bsdf.inputs['Roughness'].default_value, .78)
        self.assertEqual(normal.inputs['Strength'].default_value, .1)
        self.assertFalse(bsdf.inputs['Base Color'].links or bsdf.inputs['Metallic'].links
                         or bsdf.inputs['Roughness'].links or normal.inputs['Strength'].links)
        self.assertIs(texture.image, original.node_tree.nodes[0].image)
        self.assertEqual(texture.inputs['Vector'].links[0].from_node.name, 'OriginalUV')
        self.assertEqual(len(original.node_tree.links), 5)
        self.assertEqual(original.node_tree.nodes[2].inputs['Base Color'].default_value, (1, 1, 1, 1))
        self.assertEqual(len(original.node_tree.nodes[2].inputs['Base Color'].links), 1)
        self.assertEqual((obj.data.vertices, obj.data.uv_layers), before)
        np.testing.assert_array_equal(colors, geometry()[1])

    def test_multiple_source_slots_are_copied_separately_and_overlap_has_one_owner(self):
        points, colors, faces = geometry()
        originals = [Material('first'), Material('second')]
        obj = mesh(points, faces, originals[:], [0, 0, 0, 1])
        config = [correction(regions=[region(), region((.7, .2, .2))], max_faces=2)]
        report, affected = apply_color_corrections(obj, colors, points, [0]*3, [1]*3, config)
        self.assertEqual(report['new_material_count'], 2)
        self.assertEqual(affected, [0, 1, 2, 9, 10, 11])
        for original, copied in zip(originals, obj.data.materials[2:]):
            self.assertIs(original.node_tree.nodes[0].image, copied.node_tree.nodes[0].image)
        self.assertEqual([p.material_index for p in obj.data.polygons], [2, 0, 0, 3])
        obj = mesh(points, faces, [Material()])
        report, affected = apply_color_corrections(obj, colors, points, [0]*3, [1]*3,
              [config[0], correction('later', roughness=.5)])
        self.assertEqual([r['selected_faces'] for r in report['corrections']], [1, 1])
        self.assertEqual([p.material_index for p in obj.data.polygons], [2, 0, 0, 1])

    def test_selection_contract_and_copy_limits_fail_before_mesh_mutation(self):
        points, colors, faces = geometry()
        obj = mesh(points, faces, [Material()])
        before = Material.copies
        with self.assertRaisesRegex(ValueError, 'outside its bounds'):
            apply_color_corrections(obj, colors, points, [0]*3, [1]*3, [correction(min_faces=2, max_faces=2)])
        self.assertEqual(Material.copies, before)
        many = mesh(points, [faces[0]]*33, [Material(str(i)) for i in range(33)], list(range(33)))
        with self.assertRaisesRegex(ValueError, '32 new materials'):
            apply_color_corrections(many, colors, points, [0]*3, [1]*3, [correction(max_faces=33)])
        self.assertEqual(Material.copies, before)
        obj.data.materials[0].node_tree.nodes.extend([SimpleNamespace(type='UNSUPPORTED')]*253)
        with self.assertRaisesRegex(ValueError, 'bounded node graph'):
            apply_color_corrections(obj, colors, points, [0]*3, [1]*3, [correction()])
        self.assertEqual(Material.copies, before)
        self.assertEqual([p.material_index for p in obj.data.polygons], [0]*4)
        original = Material()
        original.copy = lambda: SimpleNamespace(name='shallow-copy', node_tree=SimpleNamespace(
            nodes=original.node_tree.nodes, links=original.node_tree.links))
        obj = mesh(points, faces, [original])
        with self.assertRaisesRegex(ValueError, 'isolate'):
            apply_color_corrections(obj, colors, points, [0]*3, [1]*3, [correction()])
        self.assertEqual(len(original.node_tree.links), 5)
        self.assertEqual(len(obj.data.materials), 1)


if __name__ == '__main__':
    unittest.main()
