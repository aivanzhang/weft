"""Analytic baked-frame preservation and hostile-storage tests; no visual quality claims."""

from copy import deepcopy
from io import BytesIO
import math
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from PIL import Image

from avatar_assets.generated_tangents import repair
from avatar_assets.three_d import _GLB
from tests.test_three_d import glb_bytes


def fixture():
    binary, views, accessors = bytearray(), [], []

    def add(rows, kind, component=5126):
        binary.extend(b'\0'*(-len(binary) % 4))
        offset = len(binary)
        for row in rows:
            binary.extend(struct.pack('<'+{5126: 'f', 5123: 'H'}[component]*len(row), *row))
        views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(binary)-offset})
        accessors.append({'bufferView': len(views)-1, 'componentType': component, 'count': len(rows), 'type': kind})
        return len(accessors)-1

    primitives, expected = [], []
    charts = [[(.1, .4), (.1, .1), (.4, .4)],
              [(.6, .1), (.6, .4), (.9, .1)],
              [(.1, .6), (.4, .6), (.1, .9)]]
    for uv1 in (True, False):
        points, normals, coordinates, targets = [], [], [], []
        for face in range(3 if uv1 else 1):
            points.extend([(face*2., 0, 0), (face*2.+1, 0, 0), (face*2., 1, 0)])
            coordinates.extend([(u, 1-v) for u, v in charts[face]])
            for angle in ((-.25, 0., .25) if face == 2 else (0., 0., 0.)):
                normal = (math.sin(angle), 0., math.cos(angle))
                normals.append(normal)
                if face == 0:
                    targets.append(((0., 1., 0.), (-1., 0., 0.), normal, 1.))
                elif face == 1:
                    targets.append(((0., 1., 0.), (1., 0., 0.), normal, -1.))
                else:
                    targets.append(((math.cos(angle), 0., -math.sin(angle)), (0., 1., 0.), normal, 1.))
        count = len(points)
        attrs = {'POSITION': add(points, 'VEC3'), 'NORMAL': add(normals, 'VEC3'),
                 'TEXCOORD_0': add([(.2, .3), (.4, .3), (.2, .5)]*(count//3), 'VEC2'),
                 'TANGENT': add([(1., 0., 0., 1.)]*count, 'VEC4'),
                 'JOINTS_0': add([(0, 0, 0, 0)]*count, 'VEC4', 5123),
                 'WEIGHTS_0': add([(1., 0., 0., 0.)]*count, 'VEC4')}
        if uv1:
            attrs['TEXCOORD_1'] = add(coordinates, 'VEC2')
            expected = targets
        primitives.append({'attributes': attrs, 'indices': add([(i,) for i in range(count)], 'SCALAR', 5123),
                           'material': 0 if uv1 else 1})
    identity = [1., 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    inverse = add([identity], 'MAT4')
    times = add([(0.,), (1.,)], 'SCALAR')
    rotations = add([(0., 0., 0., 1.)]*2, 'VEC4')
    png = BytesIO()
    Image.new('RGB', (1, 1), (166, 179, 238)).save(png, format='PNG')
    offset = len(binary)
    binary.extend(png.getvalue())
    views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(png.getvalue())})
    doc = {'asset': {'version': '2.0', 'generator': 'analytic-baked-normal-frame-test'},
           'buffers': [{'byteLength': len(binary)}], 'bufferViews': views, 'accessors': accessors,
           'meshes': [{'primitives': primitives}],
           'materials': [{'normalTexture': {'index': 0, 'texCoord': value},
                          'pbrMetallicRoughness': {'baseColorTexture': {'index': 0, 'texCoord': 0}}}
                         for value in (1, 0)],
           'textures': [{'source': 0}], 'images': [{'bufferView': len(views)-1, 'mimeType': 'image/png'}],
           'nodes': [{'mesh': 0, 'skin': 0}, {'name': 'analytic-joint'},
                     {'mesh': 0, 'skin': 0, 'translation': [2., 0., 0.], 'scale': [-1., 1., 1.]}],
           'skins': [{'joints': [1], 'inverseBindMatrices': inverse}],
           'animations': [{'samplers': [{'input': times, 'output': rotations}],
                           'channels': [{'sampler': 0, 'target': {'node': 1, 'path': 'rotation'}}]}]}
    return doc, binary, expected


class UV1TangentRepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name)/'native.glb'
        self.target = Path(self.temporary.name)/'corrected.glb'
        self.doc, self.binary, self.expected = fixture()

    def execute(self, **options):
        self.source.write_bytes(glb_bytes(self.doc, self.binary))
        return repair(self.source, self.target, **options)

    def accessor(self, name='TANGENT'):
        index = self.doc['meshes'][0]['primitives'][0]['attributes'][name]
        return index, self.doc['accessors'][index]

    def overwrite(self, name, values):
        _, spec = self.accessor(name)
        offset = self.doc['bufferViews'][spec['bufferView']]['byteOffset']
        struct.pack_into('<'+'f'*len(values), self.binary, offset, *values)

    def test_known_rotated_mirrored_and_smooth_frames_preserve_baked_world_normal(self):
        original = glb_bytes(self.doc, self.binary)
        report = self.execute(correct_normal_uv1=True)
        self.assertEqual(self.source.read_bytes(), original)
        result = _GLB(self.target)
        self.assertEqual(result.doc, self.doc)
        index, spec = self.accessor()
        actual = np.asarray(result.accessor(index, 'VEC4', (5126,)))
        normal = np.asarray(result.accessor(self.accessor('NORMAL')[0], 'VEC3', (5126,)))
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        # Independently chosen physical axes, not UV derivatives from the implementation.
        rgb = np.array([.65, .7, (1+math.sqrt(.75))*.5])
        color = rgb*2-1
        for i, (tangent, bitangent, n, sign) in enumerate(self.expected):
            np.testing.assert_allclose(actual[i, :3], tangent, atol=1e-6)
            self.assertEqual(actual[i, 3], sign)
            observed = actual[i, :3]*color[0]+np.cross(normal[i], actual[i, :3])*actual[i, 3]*color[1]+normal[i]*color[2]
            expected = np.asarray(tangent)*color[0]+np.asarray(bitangent)*color[1]+np.asarray(n)*color[2]
            np.testing.assert_allclose(observed, expected, atol=1e-6)
        view = self.doc['bufferViews'][spec['bufferView']]
        start, end = view['byteOffset'], view['byteOffset']+view['byteLength']
        self.assertEqual(result.binary[:start], self.binary[:start])
        self.assertEqual(result.binary[end:], self.binary[end:])
        correction = report['normal_uv1_correction']
        self.assertEqual((len(correction['primitives']), correction['vertices']), (1, 9))
        self.assertFalse(correction['quality_approved'])
        self.assertEqual(report['repaired_vertices'], [])

    def test_opt_in_is_required_and_boolean(self):
        # Keep original frames valid; the default must not silently reinterpret UV1.
        self.overwrite('NORMAL', (0., 0., 1.)*9)
        self.execute()
        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())
        for value in (1, 0, None, 'yes'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'explicitly boolean'):
                repair(self.source, self.target.with_name(str(value)+'.glb'), correct_normal_uv1=value)

    def test_healthy_uv0_only_keeps_legacy_behavior(self):
        self.doc['materials'][0]['normalTexture']['texCoord'] = 0
        self.overwrite('NORMAL', (0., 0., 1.)*9)
        report = self.execute(correct_normal_uv1=True)
        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())
        self.assertEqual(report['normal_uv1_correction']['primitives'], [])

    def test_degenerate_geometry_uv_or_normal_fails_without_writing(self):
        for name, values in (('POSITION', (0., 0., 0.)*3), ('TEXCOORD_1', (.4, .4)*3),
                             ('NORMAL', (0., 0., 0.)), ('NORMAL', (0., 0., 2.)),
                             ('NORMAL', (0., 1., 0.)),
                             ('TEXCOORD_1', (1.2, .3)), ('TEXCOORD_1', (float('nan'), .3))):
            with self.subTest(name=name, values=values):
                self.doc, self.binary, _ = fixture()
                self.overwrite(name, values)
                with self.assertRaises(ValueError):
                    self.execute(correct_normal_uv1=True)
                self.assertFalse(self.target.exists())

    def test_shared_corner_or_unused_vertices_are_not_averaged(self):
        primitive = self.doc['meshes'][0]['primitives'][0]
        spec = self.doc['accessors'][primitive['indices']]
        offset = self.doc['bufferViews'][spec['bufferView']]['byteOffset']
        struct.pack_into('<H', self.binary, offset+6, 0)
        with self.assertRaisesRegex(ValueError, 'every corner vertex exactly once'):
            self.execute(correct_normal_uv1=True)
        self.assertFalse(self.target.exists())
        struct.pack_into('<H', self.binary, offset+6, 3)
        spec['count'] -= 3
        with self.assertRaisesRegex(ValueError, 'every corner vertex exactly once'):
            self.execute(correct_normal_uv1=True)

    def test_opposed_corner_normals_cannot_create_mixed_triangle_handedness(self):
        self.overwrite('NORMAL', (0., 0., -1.))
        with self.assertRaisesRegex(ValueError, 'inconsistent tangent handedness'):
            self.execute(correct_normal_uv1=True)
        self.assertFalse(self.target.exists())

    def test_oversized_and_nontriangular_inputs_fail_closed(self):
        for change in ('vertices', 'faces', 'mode', 'indices', 'morph', 'selector'):
            with self.subTest(change=change):
                self.doc, self.binary, _ = fixture()
                primitive = self.doc['meshes'][0]['primitives'][0]
                if change == 'vertices': self.accessor('POSITION')[1]['count'] = 6145
                if change == 'faces': self.doc['accessors'][primitive['indices']]['count'] = 6147
                if change == 'mode': primitive['mode'] = 5
                if change == 'indices': primitive.pop('indices')
                if change == 'morph': primitive['targets'] = [{'POSITION': self.accessor('POSITION')[0]}]
                if change == 'selector': self.doc['materials'][0]['normalTexture']['texCoord'] = True
                with self.assertRaises(ValueError):
                    self.execute(correct_normal_uv1=True)
                self.assertFalse(self.target.exists())

    def test_tangent_storage_aliases_are_rejected(self):
        for alias in ('accessor', 'view', 'image', 'skin', 'animation', 'primitive', 'interleaved', 'sparse'):
            with self.subTest(alias=alias):
                self.doc, self.binary, _ = fixture()
                index, spec = self.accessor()
                view_index = spec['bufferView']
                if alias == 'accessor': self.doc['accessors'].append(deepcopy(spec))
                if alias == 'view': self.doc['bufferViews'].append(deepcopy(self.doc['bufferViews'][view_index]))
                if alias == 'image': self.doc['images'][0]['bufferView'] = view_index
                if alias == 'skin': self.doc['skins'][0]['inverseBindMatrices'] = index
                if alias == 'animation': self.doc['animations'][0]['samplers'][0]['output'] = index
                if alias == 'primitive':
                    self.doc['meshes'][0]['primitives'][1]['attributes']['TANGENT'] = index
                if alias == 'interleaved': self.doc['bufferViews'][view_index]['byteStride'] = 32
                if alias == 'sparse': self.doc['accessors'][-1]['sparse'] = {'count': 1}
                with self.assertRaises(ValueError):
                    self.execute(correct_normal_uv1=True)
                self.assertFalse(self.target.exists())

    def test_aggregate_primitive_limit_fails_without_partial_derivative(self):
        primitive = self.doc['meshes'][0]['primitives'][0]
        _, spec = self.accessor()
        view = self.doc['bufferViews'][spec['bufferView']]
        original = bytes(self.binary[view['byteOffset']:view['byteOffset']+view['byteLength']])
        for _ in range(4):
            self.binary.extend(b'\0'*(-len(self.binary) % 4))
            offset = len(self.binary)
            self.binary.extend(original)
            self.doc['bufferViews'].append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(original)})
            accessor = {**spec, 'bufferView': len(self.doc['bufferViews'])-1}
            self.doc['accessors'].append(accessor)
            copied = deepcopy(primitive)
            copied['attributes']['TANGENT'] = len(self.doc['accessors'])-1
            self.doc['meshes'][0]['primitives'].append(copied)
        self.doc['buffers'][0]['byteLength'] = len(self.binary)
        with self.assertRaisesRegex(ValueError, 'four primitives'):
            self.execute(correct_normal_uv1=True)
        self.assertFalse(self.target.exists())

    def test_existing_target_is_never_overwritten(self):
        self.target.write_bytes(b'preserved prior result')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.execute(correct_normal_uv1=True)
        self.assertEqual(self.target.read_bytes(), b'preserved prior result')


if __name__ == '__main__':
    unittest.main()
