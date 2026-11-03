"""Analytic material refinements, not Blender render or quality acceptance."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from avatar_assets.surface_profiles import validate_profiles
from scripts.blender_surface_profiles import apply_surface_profiles, select_surface_faces


def profile(name='matte', **changes):
    return {'name': name, 'regions': [], 'brightness': [0, 1.01], 'max_chroma': 1,
            'roughness': .72, 'metallic': 0, 'normal_strength': .35,
            'min_faces': 1, 'max_faces': 100000, **changes}


def geometry():
    points = np.array([[.05, .4, .5], [.45, .4, .5], [.25, .7, .5],
                       [.55, .4, .5], [.95, .4, .5], [.75, .7, .5]])
    colors = np.array([[.2]*3]*3 + [[.8]*3]*3)
    return points, colors, [(0, 1, 2), (3, 4, 5)]


class Links(list):
    def remove(self, link):
        super().remove(link)
        link.to_socket.links.remove(link)


def socket(name, value=0):
    return SimpleNamespace(name=name, default_value=value, links=[])


def connect(tree, source, output, target, name):
    link = SimpleNamespace(from_node=source, from_socket=socket(output),
                           to_node=target, to_socket=target.inputs[name])
    tree.links.append(link)
    target.inputs[name].links.append(link)


class Material:
    copies = 0

    def __init__(self, name='source'):
        self.name, self.use_nodes = name, True
        image = SimpleNamespace(name='unaltered-color', size=(4096, 4096))
        texture = SimpleNamespace(name='ColorTexture', type='TEX_IMAGE', image=image,
                                  inputs={'Vector': socket('Vector', (0, 0, 0))})
        uv = SimpleNamespace(name='OriginalUV', type='UVMAP', inputs={})
        bsdf = SimpleNamespace(name='Principled', type='BSDF_PRINCIPLED', inputs={
            'Base Color': socket('Base Color', (1, 1, 1, 1)),
            'Roughness': socket('Roughness', .9), 'Metallic': socket('Metallic', .2)})
        normal = SimpleNamespace(name='Normal', type='NORMAL_MAP', inputs={'Strength': socket('Strength', .8)})
        self.node_tree = SimpleNamespace(nodes=[texture, uv, bsdf, normal], links=Links())
        connect(self.node_tree, uv, 'UV', texture, 'Vector')
        connect(self.node_tree, texture, 'Color', bsdf, 'Base Color')
        connect(self.node_tree, texture, 'Color', bsdf, 'Roughness')
        connect(self.node_tree, texture, 'Color', bsdf, 'Metallic')
        connect(self.node_tree, texture, 'Color', normal, 'Strength')

    def copy(self):
        type(self).copies += 1
        images = {id(node.image): node.image for node in self.node_tree.nodes if node.type == 'TEX_IMAGE'}
        return deepcopy(self, images)


def mesh(points, faces, materials, slots=None):
    return SimpleNamespace(type='MESH', data=SimpleNamespace(
        vertices=[SimpleNamespace(co=tuple(point), groups={'root': 1}) for point in points],
        polygons=[SimpleNamespace(vertices=face, material_index=(slots[index] if slots else 0))
                  for index, face in enumerate(faces)],
        materials=materials, uv_layers={'original': [(.1, .2)]*len(points)}))


class SurfaceProfileTests(unittest.TestCase):
    def test_validation_is_detached_and_rejects_unbounded_fields(self):
        valid = [profile()]
        copied = validate_profiles(valid)
        copied[0]['brightness'][0] = .1
        self.assertEqual(valid[0]['brightness'][0], 0)
        bad = [[], [profile()]*9, [profile(), profile()], [profile(name='../bad')],
               [profile(roughness=.039)], [profile(metallic=True)], [profile(normal_strength=float('nan'))],
               [profile(brightness=[.5, .5])], [profile(max_chroma=1.1)], [profile(max_faces=100001)],
               [profile(min_faces=3, max_faces=2)], [profile(script='execute')],
               [profile(regions=[{'center': [0, 0, 0], 'radius': [0, 1, 1]}])],
               [profile(regions=[{'center': [0, 0, 0], 'radius': [1, 1, 1]}]*9)]]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_profiles(value)

    def test_centroid_rgb_selection_and_last_match_precedence(self):
        points, colors, faces = geometry()
        profiles = [profile(), profile('bright', regions=[{'center': [.75, .5, .5], 'radius': [.15]*3}],
                                       brightness=[.7, 1.01], max_chroma=.1, metallic=.9)]
        owners, reports = select_surface_faces(faces, colors, points, [0]*3, [1]*3, profiles)
        np.testing.assert_array_equal(owners, [0, 1])
        self.assertEqual([row['matched_faces'] for row in reports], [2, 1])
        self.assertEqual([row['selected_faces'] for row in reports], [1, 1])
        with self.assertRaisesRegex(ValueError, r'matte=0 allowed\[1,100000\]'):
            select_surface_faces(faces, colors, points, [0]*3, [1]*3, [profile(), profile('override')])

    def test_all_final_count_violations_are_reported_after_overlap_assignment(self):
        points, colors, faces = geometry()
        faces = np.array([faces[0]]*3+[faces[1]])
        profiles = [profile('base', max_faces=2),
                    profile('bright', brightness=[.7,1.01], min_faces=2, max_faces=3)]
        original = deepcopy(profiles)
        with self.assertRaises(ValueError) as caught:
            select_surface_faces(faces, colors, points, [0]*3, [1]*3, profiles)
        self.assertEqual(str(caught.exception),
            'Surface profile final face counts outside declared bounds: '
            'base=3 allowed[1,2]; bright=1 allowed[2,3]')
        self.assertEqual(profiles, original)
        obj = mesh(points, faces.tolist(), [Material()]); copies = Material.copies
        with self.assertRaises(ValueError) as applied:
            apply_surface_profiles(obj, colors, points, [0]*3, [1]*3, profiles)
        self.assertEqual(str(applied.exception), str(caught.exception))
        self.assertEqual(Material.copies, copies)
        self.assertEqual(len(obj.data.materials), 1)
        self.assertTrue(all(face.material_index == 0 for face in obj.data.polygons))

    def test_eight_maximum_length_profile_failures_fit_bounded_message(self):
        points, colors, faces = geometry()
        profiles = [profile(f'p{i}'+'x'*62, min_faces=3) for i in range(8)]
        with self.assertRaises(ValueError) as caught:
            select_surface_faces(faces, colors, points, [0]*3, [1]*3, profiles)
        message = str(caught.exception)
        self.assertLess(len(message.encode()), 1024)
        for index, row in enumerate(profiles):
            self.assertIn(f"{row['name']}={2 if index == 7 else 0} allowed[3,100000]",message)
        self.assertEqual(message.count('allowed['),8)

    def test_chroma_filter_and_invalid_geometry_fail_explicitly(self):
        points, colors, faces = geometry()
        colors[3:] = [1, 0, 0]
        owners, _ = select_surface_faces(faces, colors, points, [0]*3, [1]*3, [profile(max_chroma=.1)])
        np.testing.assert_array_equal(owners, [0, -1])
        cases = [(faces, colors[:-1], points), ([(0, 1, 99)], colors, points),
                 ([(0., 1., 2.)], colors, points), (faces, colors, points*np.nan),
                 (faces, colors, np.array(1)), (np.array(1), colors, points),
                 (np.zeros((100001, 3), dtype=int), colors, points),
                 (faces, np.zeros((100001, 3)), np.zeros((100001, 3)))]
        for bad_faces, bad_colors, bad_points in cases:
            with self.subTest(faces=str(type(bad_faces))), self.assertRaises(ValueError):
                select_surface_faces(bad_faces, bad_colors, bad_points, [0]*3, [1]*3, [profile()])

    def test_clones_change_only_response_and_keep_texture_uv_skin(self):
        points, colors, faces = geometry()
        original = Material()
        obj = mesh(points, faces, [original])
        coords = [vertex.co for vertex in obj.data.vertices]
        uv = deepcopy(obj.data.uv_layers)
        profiles = [profile(max_faces=2), profile('bright-metal', brightness=[.7, 1.01],
                                                  roughness=.25, metallic=.8, normal_strength=.1)]
        result = apply_surface_profiles(obj, colors, points, [0]*3, [1]*3, profiles)
        self.assertEqual((result['new_material_count'], result['faces_reassigned']), (2, 2))
        self.assertEqual([polygon.material_index for polygon in obj.data.polygons], [1, 2])
        for refined, settings in zip(obj.data.materials[1:], profiles):
            texture, uv_node, bsdf, normal = refined.node_tree.nodes
            self.assertIs(texture.image, original.node_tree.nodes[0].image)
            self.assertEqual(texture.inputs['Vector'].links[0].from_node.name, 'OriginalUV')
            self.assertEqual(bsdf.inputs['Base Color'].links[0].from_node.name, 'ColorTexture')
            self.assertEqual(bsdf.inputs['Roughness'].default_value, settings['roughness'])
            self.assertEqual(bsdf.inputs['Metallic'].default_value, settings['metallic'])
            self.assertEqual(normal.inputs['Strength'].default_value, settings['normal_strength'])
            self.assertFalse(bsdf.inputs['Roughness'].links or bsdf.inputs['Metallic'].links or normal.inputs['Strength'].links)
        self.assertEqual(len(original.node_tree.links), 5)
        self.assertEqual(original.node_tree.nodes[2].inputs['Roughness'].default_value, .9)
        self.assertEqual([vertex.co for vertex in obj.data.vertices], coords)
        self.assertEqual(obj.data.uv_layers, uv)
        self.assertTrue(all(vertex.groups == {'root': 1} for vertex in obj.data.vertices))
        self.assertIn('no repainting', result['texture_scope'])

    def test_multiple_source_materials_keep_their_own_images(self):
        points, colors, faces = geometry()
        originals = [Material('first'), Material('second')]
        obj = mesh(points, faces, originals[:], [0, 1])
        result = apply_surface_profiles(obj, colors, points, [0]*3, [1]*3, [profile()])
        self.assertEqual(result['new_material_count'], 2)
        self.assertEqual(len(result['profiles'][0]['material_copies']), 2)
        for source, refined in zip(originals, obj.data.materials[2:]):
            self.assertIs(source.node_tree.nodes[0].image, refined.node_tree.nodes[0].image)
        self.assertIsNot(obj.data.materials[2].node_tree.nodes[0].image, obj.data.materials[3].node_tree.nodes[0].image)

    def test_copy_and_selection_limits_fail_before_mutation(self):
        points, colors, faces = geometry()
        obj = mesh(points, faces, [Material()])
        before = Material.copies
        with self.assertRaisesRegex(ValueError, 'final face counts outside declared bounds'):
            apply_surface_profiles(obj, colors, points, [0]*3, [1]*3, [profile(min_faces=3)])
        self.assertEqual(Material.copies, before)
        many = mesh(points, [faces[0]]*33, [Material(str(i)) for i in range(33)], list(range(33)))
        with self.assertRaisesRegex(ValueError, '32 new materials'):
            apply_surface_profiles(many, colors, points, [0]*3, [1]*3, [profile()])
        self.assertEqual(Material.copies, before)
        self.assertEqual(len(many.data.materials), 33)


if __name__ == '__main__':
    unittest.main()
