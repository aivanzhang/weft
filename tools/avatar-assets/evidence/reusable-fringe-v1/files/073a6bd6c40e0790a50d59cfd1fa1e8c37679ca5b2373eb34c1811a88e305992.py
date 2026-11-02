"""Analytic unused-tangent omission; no native or generated-avatar quality claims."""

import base64
from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from avatar_assets.fringe_export import omit_unused_fringe_tangents
from avatar_assets.three_d import _GLB, validate_render_attributes
from tests.test_three_d import glb_bytes


def fixture(corners=960, zero_count=151):
    binary, views, accessors = bytearray(), [], []
    def add(rows, kind, component=5126):
        binary.extend(b'\0'*(-len(binary) % 4))
        offset = len(binary)
        array = np.asarray(rows, dtype={5126: '<f4', 5123: '<u2'}[component])
        binary.extend(array.tobytes())
        views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(binary)-offset})
        accessors.append({'bufferView': len(views)-1, 'componentType': component, 'count': len(array), 'type': kind})
        return len(accessors)-1
    primitives = []
    for selected, count in ((False, 3), (True, corners)):
        points = np.tile([[0., 0, 0], [.01, 0, 0], [0, .01, 0]], (count//3, 1))
        points[:, 0] += np.repeat(np.arange(count//3)*.02, 3)
        normal = [1., 0, 0] if selected else [0., 0, 1.]
        tangent = np.tile([0., 1, 0, 1.] if selected else [1., 0, 0, 1.], (count, 1))
        if selected:
            tangent[:zero_count, :3] = 0
        attrs = {'POSITION': add(points, 'VEC3'), 'NORMAL': add(np.tile(normal, (count, 1)), 'VEC3'),
                 'TEXCOORD_0': add(np.tile([[0., 0], [1., 0], [0, 1.]], (count//3, 1)), 'VEC2'),
                 'TANGENT': add(tangent, 'VEC4'), 'JOINTS_0': add(np.zeros((count, 4)), 'VEC4', 5123),
                 'WEIGHTS_0': add(np.tile([1., 0, 0, 0], (count, 1)), 'VEC4'),
                 'COLOR_0': add(np.tile([.5, .6, .7, 1.], (count, 1)), 'VEC4')}
        primitives.append({'attributes': attrs, 'indices': add(np.arange(count)[:, None], 'SCALAR', 5123),
                           'material': 1 if selected else 0, 'extras': {'analytic': True}})
    inverse = add([np.eye(4).ravel()], 'MAT4')
    times = add([[0.], [1.]], 'SCALAR')
    rotation = add([[0., 0, 0, 1.], [0, 0, .5, np.sqrt(.75)]], 'VEC4')
    png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==')
    offset = len(binary); binary.extend(png)
    views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(png)})
    doc = {'asset': {'version': '2.0', 'generator': 'analytic-unused-fringe-frame-fixture'},
           'buffers': [{'byteLength': len(binary)}], 'bufferViews': views, 'accessors': accessors,
           'meshes': [{'primitives': [primitives[0]]}, {'primitives': [primitives[1]]}],
           'nodes': [{'name': 'Body', 'mesh': 0, 'skin': 0}, {'name': 'Root'}, {'name': 'Fringe', 'mesh': 1, 'skin': 0}],
           'skins': [{'joints': [1], 'skeleton': 1, 'inverseBindMatrices': inverse}],
           'animations': [{'name': 'analytic-turn', 'samplers': [{'input': times, 'output': rotation}],
                           'channels': [{'sampler': 0, 'target': {'node': 1, 'path': 'rotation'}}]}],
           'scene': 0, 'scenes': [{'nodes': [0, 1, 2]}],
           'materials': [{'name': 'NormalMappedBody', 'normalTexture': {'index': 0},
                          'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'metallicFactor': 0, 'roughnessFactor': .92}},
                         {'name': 'SourceColoredFringe', 'alphaMode': 'MASK',
                          'alphaCutoff': float(np.float32(.18)),
                          'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'metallicFactor': 0,
                                                 'roughnessFactor': float(np.float32(.92))}}],
           'images': [{'bufferView': len(views)-1, 'mimeType': 'image/png'}], 'textures': [{'source': 0}],
           'extras': {'fixture_scope': '151 zero XYZ tangents among960 analytic corners, not an actual native export'}}
    return doc, binary


def padded_bin(raw):
    size = struct.unpack_from('<I', raw, 12)[0]
    return raw[28+size:]


class FringeExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source, self.target = self.root/'source.glb', self.root/'derivative.glb'
        self.doc, self.binary = fixture()

    def write(self):
        self.source.write_bytes(glb_bytes(self.doc, self.binary))

    def execute(self, names=None):
        self.write()
        return omit_unused_fringe_tangents(self.source, self.target, ['SourceColoredFringe'] if names is None else names)

    def test_exact_json_allowlist_and_padded_binary_preserve_all_other_data(self):
        self.write(); original = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, 'tangents are not unit'):
            validate_render_attributes(self.source)
        report = omit_unused_fringe_tangents(self.source, self.target, ['SourceColoredFringe'])
        expected = deepcopy(self.doc)
        removed = expected['meshes'][1]['primitives'][0]['attributes'].pop('TANGENT')
        actual = _GLB(self.target)
        self.assertEqual(actual.doc, expected)
        self.assertEqual(padded_bin(self.target.read_bytes()), padded_bin(original))
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(actual.doc['accessors'][removed], self.doc['accessors'][removed])
        self.assertEqual(actual.doc['asset']['generator'], self.doc['asset']['generator'])
        self.assertIn('TANGENT', actual.doc['meshes'][0]['primitives'][0]['attributes'])
        self.assertEqual(report['selected_primitives'][0]['zero_xyz_tangents'], 151)
        self.assertEqual(report['selected_primitives'][0]['vertices'], 960)
        self.assertEqual(report['json_changed_paths'], ['/meshes/1/primitives/0/attributes/TANGENT'])
        self.assertEqual(report['padded_binary_sha256'], report['result_padded_binary_sha256'])
        self.assertFalse(report['quality_approved'])
        self.assertEqual(validate_render_attributes(self.target)['status'], 'validated_render_attributes')

    def test_absent_binding_is_an_explicit_byte_exact_noop_to_a_new_path(self):
        self.doc['meshes'][1]['primitives'][0]['attributes'].pop('TANGENT')
        report = self.execute()
        self.assertEqual(report['removed_bindings'], 0)
        self.assertEqual(report['json_changed_paths'], [])
        self.assertEqual(self.source.read_bytes(), self.target.read_bytes())

    def test_other_invalid_tangents_are_not_hidden_or_repaired(self):
        index = self.doc['meshes'][0]['primitives'][0]['attributes']['TANGENT']
        view = self.doc['bufferViews'][self.doc['accessors'][index]['bufferView']]
        struct.pack_into('<fff', self.binary, view['byteOffset'], 0, 0, 0)
        self.execute()
        with self.assertRaisesRegex(ValueError, 'tangents are not unit'):
            validate_render_attributes(self.target)

    def test_missing_duplicate_ambiguous_or_unused_material_names_reject(self):
        self.write()
        for names in ([], ['Missing'], ['SourceColoredFringe']*2, [''] , [' '], [True], ['x'*129],
                      ['a', 'b', 'c', 'd', 'e'], 'SourceColoredFringe'):
            with self.subTest(names=names), self.assertRaises(ValueError):
                omit_unused_fringe_tangents(self.source, self.target, names)
            self.assertFalse(self.target.exists())
        self.doc['materials'].append(deepcopy(self.doc['materials'][1]))
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.execute()
        self.doc['materials'][-1]['name'] = 'UnusedFringe'
        with self.assertRaisesRegex(ValueError, 'unused'):
            self.execute(['UnusedFringe'])

    def test_normal_maps_extensions_other_shader_routes_and_wrong_policy_reject(self):
        original = deepcopy(self.doc)
        mutations = [lambda m: m.update(normalTexture={'index': 0}), lambda m: m.update(normalTexture=None),
            lambda m: m.update(extensions={}), lambda m: m.update(emissiveTexture={'index': 0}),
            lambda m: m.update(doubleSided=True), lambda m: m.update(doubleSided=0), lambda m: m.update(alphaMode='BLEND'),
            lambda m: m.update(alphaCutoff=.180001), lambda m: m['pbrMetallicRoughness'].update(roughnessFactor=.920001),
            lambda m: m['pbrMetallicRoughness'].update(metallicFactor=True),
            lambda m: m['pbrMetallicRoughness'].update(metallicRoughnessTexture={'index': 0}),
            lambda m: m['pbrMetallicRoughness']['baseColorTexture'].update(extensions={}),
            lambda m: m['pbrMetallicRoughness']['baseColorTexture'].update(texCoord=True),
            lambda m: m['pbrMetallicRoughness']['baseColorTexture'].update(texCoord=1)]
        for mutate in mutations:
            self.doc = deepcopy(original); mutate(self.doc['materials'][1])
            with self.subTest(material=self.doc['materials'][1]), self.assertRaises(ValueError):
                self.execute()
            self.assertFalse(self.target.exists())
        self.doc = original
        self.doc['materials'][1].update(doubleSided=False, alphaCutoff=.18)
        self.doc['materials'][1]['pbrMetallicRoughness']['roughnessFactor'] = .92
        self.execute()

    def test_missing_skin_attributes_morphs_and_unindexed_or_oversized_primitives_reject(self):
        original = deepcopy(self.doc)
        for mutate in (lambda p: p.pop('indices'), lambda p: p.update(mode=1),
                       lambda p: p.update(targets=[]), lambda p: p.update(extensions={}),
                       lambda p: p['attributes'].pop('JOINTS_0'), lambda p: p['attributes'].update(TANGENT=None)):
            self.doc = deepcopy(original); mutate(self.doc['meshes'][1]['primitives'][0])
            with self.assertRaises(ValueError): self.execute()
            self.assertFalse(self.target.exists())
        for mutate in (lambda d: d['nodes'][2].pop('skin'), lambda d: d['meshes'][1].update(weights=[1]),
                       lambda d: d['nodes'][2].update(weights=[1]),
                       lambda d: d['meshes'][1].update(primitives=d['meshes'][1]['primitives']*5),
                       lambda d: d['accessors'][d['meshes'][1]['primitives'][0]['attributes']['POSITION']].update(count=100001)):
            self.doc = deepcopy(original); mutate(self.doc)
            with self.assertRaises(ValueError): self.execute()
            self.assertFalse(self.target.exists())

    def test_nonfinite_data_duplicate_json_and_invalid_source_paths_reject(self):
        self.write()
        for source in (None, True, self.root, self.root/'missing.glb'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                omit_unused_fringe_tangents(source, self.target, ['SourceColoredFringe'])
        link = self.root/'link.glb'; link.symlink_to(self.source)
        with self.assertRaises(ValueError): omit_unused_fringe_tangents(link, self.target, ['SourceColoredFringe'])
        index = self.doc['meshes'][1]['primitives'][0]['attributes']['TANGENT']
        offset = self.doc['bufferViews'][self.doc['accessors'][index]['bufferView']]['byteOffset']
        struct.pack_into('<f', self.binary, offset, float('nan'))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'): self.execute()
        self.doc, self.binary = fixture()
        encoded = json.dumps(self.doc).replace('"asset":', '"duplicate":1,"duplicate":2,"asset":', 1).encode()
        encoded += b' '*(-len(encoded) % 4)
        binary = bytes(self.binary)+b'\0'*(-len(self.binary) % 4)
        self.source.write_bytes(struct.pack('<4sII', b'glTF', 2, 28+len(encoded)+len(binary))+
            struct.pack('<II',len(encoded),0x4E4F534A)+encoded+struct.pack('<II',len(binary),0x004E4942)+binary)
        with self.assertRaisesRegex(ValueError, 'Duplicate'): omit_unused_fringe_tangents(self.source,self.target,['SourceColoredFringe'])
        self.assertFalse(self.target.exists())

    def test_new_target_only_and_file_bound_fail_before_parsing(self):
        self.write()
        original = self.source.read_bytes()
        with self.assertRaises(ValueError): omit_unused_fringe_tangents(self.source,self.source,['SourceColoredFringe'])
        self.assertEqual(self.source.read_bytes(), original)
        self.target.write_bytes(b'Keep existing output')
        with self.assertRaises(ValueError): omit_unused_fringe_tangents(self.source,self.target,['SourceColoredFringe'])
        self.assertEqual(self.target.read_bytes(), b'Keep existing output')
        dangling = self.root/'dangling.glb'; dangling.symlink_to(self.root/'absent.glb')
        with self.assertRaises(ValueError): omit_unused_fringe_tangents(self.source,dangling,['SourceColoredFringe'])
        huge = self.root/'oversized.glb'
        with huge.open('wb') as handle: handle.truncate(64*1024**2+1)
        with patch('avatar_assets.fringe_export._GLB', side_effect=AssertionError('Parsed oversized source')):
            with self.assertRaises(ValueError): omit_unused_fringe_tangents(huge,self.root/'new.glb',['SourceColoredFringe'])
        self.assertFalse((self.root/'new.glb').exists())


if __name__ == '__main__':
    unittest.main()
