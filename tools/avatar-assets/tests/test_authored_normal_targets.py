"""Analytic UV1-source/UV2-atlas custody fixtures; no generated-avatar quality claims."""

from copy import deepcopy
import hashlib
from io import BytesIO
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from avatar_assets.generated_tangents import NORMAL_UV1_FRAME_POLICY, repair, validate_authored_normal_targets
from avatar_assets.three_d import _GLB
from tests.test_three_d import glb_bytes
from tests.test_uv1_tangent_repair import fixture


def append_accessor(doc, binary, rows, kind):
    binary.extend(b'\0'*(-len(binary) % 4))
    start = len(binary)
    for row in rows:
        binary.extend(struct.pack('<'+'f'*len(row), *row))
    doc['bufferViews'].append({'buffer': 0, 'byteOffset': start, 'byteLength': len(binary)-start})
    doc['accessors'].append({'bufferView': len(doc['bufferViews'])-1, 'componentType': 5126,
                             'count': len(rows), 'type': kind})
    doc['buffers'][0]['byteLength'] = len(binary)
    return len(doc['accessors'])-1


def overwrite(doc, binary, accessor, values):
    spec = doc['accessors'][accessor]
    offset = doc['bufferViews'][spec['bufferView']]['byteOffset']+spec.get('byteOffset', 0)
    struct.pack_into('<'+'f'*len(values), binary, offset, *values)


def authored_fixture():
    doc, binary, expected = fixture()
    source = doc['meshes'][0]['primitives'][0]
    known = [(*tangent, sign) for tangent, _, _, sign in expected]
    overwrite(doc, binary, source['attributes']['TANGENT'], [v for row in known for v in row])
    doc['materials'][0]['name'] = 'PreservedSourceUV1'
    doc['materials'][1]['name'] = 'PreservedSourceUV0'
    target = deepcopy(source)
    target['attributes']['TEXCOORD_2'] = source['attributes']['TEXCOORD_1']
    # UV1 deliberately has a different direction, so a hard-coded UV1 correction fails.
    target['attributes']['TEXCOORD_1'] = source['attributes']['TEXCOORD_0']
    target['attributes']['TANGENT'] = append_accessor(doc, binary, [(1.,0.,0.,1.)]*9, 'VEC4')
    target['attributes']['NORMAL'] = append_accessor(doc, binary, [row[2] for row in expected], 'VEC3')
    png = BytesIO()
    image = Image.new('RGB', (2, 2))
    image.putdata([(166,179,238), (128,160,247), (180,128,243), (140,145,252)])
    image.save(png, format='PNG')
    content = png.getvalue()
    binary.extend(b'\0'*(-len(binary) % 4))
    start = len(binary); binary.extend(content)
    doc['bufferViews'].append({'buffer': 0, 'byteOffset': start, 'byteLength': len(content)})
    doc['images'].append({'bufferView': len(doc['bufferViews'])-1, 'mimeType': 'image/png'})
    doc['textures'].append({'source': len(doc['images'])-1})
    doc['materials'].append({'name': 'AuthoredCoatUV2', 'normalTexture': {'index': len(doc['textures'])-1, 'texCoord': 2},
                             'pbrMetallicRoughness': deepcopy(doc['materials'][0]['pbrMetallicRoughness'])})
    target['material'] = len(doc['materials'])-1
    doc['meshes'][0]['primitives'].append(target)
    doc['buffers'][0]['byteLength'] = len(binary)
    targets = [{'material': 'AuthoredCoatUV2', 'normal_texcoord': 2, 'selected_faces': 3,
                'texture_sha256': hashlib.sha256(content).hexdigest(), 'frame_policy': NORMAL_UV1_FRAME_POLICY}]
    return doc, binary, expected, targets


class AuthoredNormalTargetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ordinal = 0
        self.reset()

    def reset(self):
        self.doc, self.binary, self.expected, self.targets = authored_fixture()

    def primitive(self, index=2):
        return self.doc['meshes'][0]['primitives'][index]

    def execute(self, targets=None, **kwargs):
        self.ordinal += 1
        self.source = self.root/f'source-{self.ordinal}.glb'
        self.target = self.root/f'result-{self.ordinal}.glb'
        self.original = glb_bytes(self.doc, self.binary)
        self.source.write_bytes(self.original)
        return repair(self.source, self.target, authored_normal_targets=self.targets if targets is None else targets, **kwargs)

    def rejected(self, **kwargs):
        with self.assertRaises(ValueError):
            self.execute(**kwargs)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.source.read_bytes(), self.original)

    def known_frames(self, result, primitive=2):
        attrs = self.primitive(primitive)['attributes']
        tangent = np.asarray(result.accessor(attrs['TANGENT'], 'VEC4', (5126,)))
        normals = np.asarray(result.accessor(attrs['NORMAL'], 'VEC3', (5126,)))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        # Actual fixture PNG vector projected onto independently known physical axes.
        rgb_vector = np.array([166., 179., 238.])/255*2-1
        for i, (t, b, n, sign) in enumerate(self.expected):
            np.testing.assert_allclose(tangent[i, :3], t, atol=1e-6)
            self.assertEqual(tangent[i, 3], sign)
            world = tangent[i, :3]*rgb_vector[0]+np.cross(normals[i], tangent[i, :3])*sign*rgb_vector[1]+normals[i]*rgb_vector[2]
            expected = np.asarray(t)*rgb_vector[0]+np.asarray(b)*rgb_vector[1]+np.asarray(n)*rgb_vector[2]
            np.testing.assert_allclose(world, expected, atol=1e-6)

    def test_selected_uv2_frame_changes_only_its_tangent_bytes_and_preserves_source_uv1(self):
        report = self.execute()
        result = _GLB(self.target)
        self.known_frames(result)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(result.doc, self.doc)
        accessor = self.doc['accessors'][self.primitive()['attributes']['TANGENT']]
        view = self.doc['bufferViews'][accessor['bufferView']]
        json_size = struct.unpack_from('<I', self.original, 12)[0]
        begin = 28+json_size+view['byteOffset']; end = begin+view['byteLength']
        changed = self.target.read_bytes()
        self.assertEqual(changed[:begin], self.original[:begin])
        self.assertEqual(changed[end:], self.original[end:])
        self.assertNotEqual(changed[begin:end], self.original[begin:end])
        proof = report['authored_normal_correction']
        self.assertEqual(proof['targets'], self.targets)
        self.assertEqual(proof['policy'], NORMAL_UV1_FRAME_POLICY)
        self.assertEqual(proof['vertices'], 9)
        self.assertEqual([(r['primitive'], r['material'], r['normal_texcoord']) for r in proof['primitives']],
                         [(2, 'AuthoredCoatUV2', 2)])
        self.assertEqual(report['repaired_vertices'], [])
        self.assertFalse(proof['quality_approved'])
        self.assertTrue(proof['authored_correction_changes_only_selected_tangent_float4_bytes'])

    def test_explicit_uv1_target_uses_the_same_bound_policy(self):
        attrs = self.primitive()['attributes']
        attrs['TEXCOORD_1'] = attrs['TEXCOORD_2']
        self.doc['materials'][2]['normalTexture']['texCoord'] = 1
        self.targets[0]['normal_texcoord'] = 1
        report = self.execute()
        self.known_frames(_GLB(self.target))
        self.assertEqual(report['authored_normal_correction']['primitives'][0]['normal_texcoord'], 1)

    def test_empty_targets_preserve_healthy_source_uv1_and_uv2_bytes(self):
        attrs = self.primitive()['attributes']
        values = [v for t, _, _, sign in self.expected for v in (*t, sign)]
        overwrite(self.doc, self.binary, attrs['TANGENT'], values)
        report = self.execute(targets=[])
        self.assertEqual(self.source.read_bytes(), self.target.read_bytes())
        self.assertEqual(report['authored_normal_correction']['primitives'], [])
        self.assertEqual(report['authored_normal_correction']['targets'], [])

    def test_schema_is_bounded_and_checked_before_any_model_read(self):
        bad = [True, {}, 'targets', (), [None], self.targets*2,
               [{**self.targets[0], 'extra': True}], [{k:v for k,v in self.targets[0].items() if k != 'material'}]]
        for key, values in [('material', ['', 'x'*257]), ('normal_texcoord', [True, 0, 3, '2']),
                            ('selected_faces', [True, 0, 2049, 3.0]),
                            ('texture_sha256', ['', 'A'*64, 'z'*64, None]),
                            ('frame_policy', [None, True, 'old-policy'])]:
            bad.extend([{**self.targets[0], key: value}] for value in values)
        bad.append([{**self.targets[0], 'material': 'Material'+str(i)} for i in range(5)])
        for targets in bad:
            with self.subTest(targets=targets), patch('avatar_assets.generated_tangents._GLB') as loader:
                with self.assertRaises(ValueError):
                    repair(self.root/'absent.glb', self.root/'not-written.glb', authored_normal_targets=targets)
                loader.assert_not_called()
                self.assertFalse((self.root/'not-written.glb').exists())
        copied = validate_authored_normal_targets(self.targets)
        copied[0]['material'] = 'Changed caller copy'
        self.assertEqual(self.targets[0]['material'], 'AuthoredCoatUV2')

    def test_legacy_flag_or_policy_cannot_be_combined_even_with_empty_targets(self):
        for targets in ([], self.targets):
            for args in ({'correct_normal_uv1': True, 'normal_uv1_frame_policy': NORMAL_UV1_FRAME_POLICY},
                         {'normal_uv1_frame_policy': NORMAL_UV1_FRAME_POLICY}):
                with self.subTest(targets=targets, args=args), patch('avatar_assets.generated_tangents._GLB') as loader:
                    with self.assertRaises(ValueError):
                        repair(self.root/'absent.glb', self.root/'not-written.glb', authored_normal_targets=targets, **args)
                    loader.assert_not_called()

    def test_absent_ambiguous_unbound_or_wrong_material_uv_count_and_png_are_rejected(self):
        for mode in ('absent', 'ambiguous', 'unused', 'uv', 'missing_uv', 'missing_normal',
                     'few_faces', 'many_faces', 'wrong_hash', 'wrong_texture', 'not_png'):
            with self.subTest(mode=mode):
                self.reset()
                if mode == 'absent': self.targets[0]['material'] = 'No such material'
                if mode == 'ambiguous': self.doc['materials'].append(deepcopy(self.doc['materials'][2]))
                if mode == 'unused': self.primitive()['material'] = 0
                if mode == 'uv': self.doc['materials'][2]['normalTexture']['texCoord'] = 1
                if mode == 'missing_uv': self.primitive()['attributes'].pop('TEXCOORD_2')
                if mode == 'missing_normal': self.doc['materials'][2].pop('normalTexture')
                if mode == 'few_faces': self.targets[0]['selected_faces'] = 2
                if mode == 'many_faces': self.targets[0]['selected_faces'] = 4
                if mode == 'wrong_hash': self.targets[0]['texture_sha256'] = '0'*64
                if mode == 'wrong_texture': self.doc['materials'][2]['normalTexture']['index'] = 0
                if mode == 'not_png': self.doc['images'][1]['mimeType'] = 'image/jpeg'
                self.rejected()

    def test_untargeted_uv1_and_uv2_invalid_frames_fail_instead_of_rare_fallback(self):
        for channel in (1, 2):
            for mode in ('zero_xyz', 'nonorthogonal', 'bad_w', 'nonunit_normal'):
                with self.subTest(channel=channel, mode=mode):
                    self.reset()
                    source = self.primitive(0)['attributes']
                    if channel == 2:
                        source['TEXCOORD_2'] = source['TEXCOORD_1']
                        self.doc['materials'][0]['normalTexture']['texCoord'] = 2
                    if mode == 'zero_xyz': overwrite(self.doc, self.binary, source['TANGENT'], [0,0,0,1])
                    if mode == 'nonorthogonal': overwrite(self.doc, self.binary, source['TANGENT'], [0,0,1,1])
                    if mode == 'bad_w': overwrite(self.doc, self.binary, source['TANGENT'], [0,1,0,0])
                    if mode == 'nonunit_normal': overwrite(self.doc, self.binary, source['NORMAL'], [0,0,2])
                    self.rejected()

    def test_target_storage_aliases_are_rejected_without_partial_output(self):
        for mode in ('accessor', 'view', 'image', 'skin', 'animation', 'primitive', 'interleaved', 'sparse'):
            with self.subTest(mode=mode):
                self.reset()
                index = self.primitive()['attributes']['TANGENT']
                spec = self.doc['accessors'][index]; vi = spec['bufferView']
                if mode == 'accessor': self.doc['accessors'].append(deepcopy(spec))
                if mode == 'view': self.doc['bufferViews'].append(deepcopy(self.doc['bufferViews'][vi]))
                if mode == 'image': self.doc['images'][0]['bufferView'] = vi
                if mode == 'skin': self.doc['skins'][0]['inverseBindMatrices'] = index
                if mode == 'animation': self.doc['animations'][0]['samplers'][0]['output'] = index
                if mode == 'primitive': self.primitive(0)['attributes']['TANGENT'] = index
                if mode == 'interleaved': self.doc['bufferViews'][vi]['byteStride'] = 32
                if mode == 'sparse': spec['sparse'] = {'count': 1}
                self.rejected()

    def test_extensions_and_remote_data_urls_never_become_authored_bakes(self):
        for mode in ('extension', 'remote_image', 'data_image', 'remote_buffer'):
            with self.subTest(mode=mode):
                self.reset()
                if mode == 'extension': self.doc['materials'][2]['normalTexture']['extensions'] = {'KHR_texture_transform': {'offset': [0,0]}}
                if mode == 'remote_image': self.doc['images'][1]['uri'] = 'https://invalid.example/normal.png'
                if mode == 'data_image': self.doc['images'][1]['uri'] = 'data:image/png;base64,AA=='
                if mode == 'remote_buffer': self.doc['buffers'][0]['uri'] = 'https://invalid.example/model.bin'
                self.rejected()

    def test_multiple_primitives_bind_aggregate_faces_and_keep_global_four_primitive_cap(self):
        for count in (2, 4, 5):
            with self.subTest(count=count):
                self.reset()
                for _ in range(count-1):
                    copied = deepcopy(self.primitive())
                    copied['attributes']['TANGENT'] = append_accessor(self.doc, self.binary, [(1.,0.,0.,1.)]*9, 'VEC4')
                    self.doc['meshes'][0]['primitives'].append(copied)
                self.targets[0]['selected_faces'] = count*3
                if count == 5:
                    self.rejected()
                else:
                    report = self.execute()
                    self.assertEqual(len(report['authored_normal_correction']['primitives']), count)
                    self.assertEqual(report['authored_normal_correction']['vertices'], count*9)

    def test_shared_target_corners_or_nontriangles_do_not_invent_averaged_frames(self):
        for mode in ('shared_corner', 'triangle_strip'):
            with self.subTest(mode=mode):
                self.reset()
                if mode == 'shared_corner':
                    spec = self.doc['accessors'][self.primitive()['indices']]
                    offset = self.doc['bufferViews'][spec['bufferView']]['byteOffset']
                    struct.pack_into('<H', self.binary, offset+6, 0)
                else:
                    self.primitive()['mode'] = 5
                self.rejected()


if __name__ == '__main__':
    unittest.main()
