"""Analytic multipart custody/transform fixtures; no generated-avatar quality claim."""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from avatar_assets.compose_sources import (MAX_CONFIG_BYTES, MAX_SOURCE_BYTES, _add_counts,
    _bounded_sha, _copy_bounded, _file_state, _image_inventory, _identity_instance,
    _metadata_inventory, compose, validate_config)
from avatar_assets.fringe_export import _array
from avatar_assets.three_d import _GLB


def glb_bytes(doc, binary):
    encoded = json.dumps(doc, separators=(',', ':')).encode(); encoded += b' '*(-len(encoded) % 4)
    binary = bytes(binary)+b'\0'*(-len(binary) % 4)
    return (struct.pack('<4sII', b'glTF', 2, 28+len(encoded)+len(binary))
            +struct.pack('<II', len(encoded), 0x4E4F534A)+encoded
            +struct.pack('<II', len(binary), 0x004E4942)+binary)


def fixture(*, constant=False, multiple=False, tangent=True):
    binary, views, accessors = bytearray(), [], []
    def add(value, kind, component=5126):
        binary.extend(b'\0'*(-len(binary) % 4)); offset = len(binary)
        data = np.asarray(value, dtype='<f4' if component == 5126 else '<u2')
        binary.extend(data.tobytes()); views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': data.nbytes})
        accessors.append({'bufferView': len(views)-1, 'componentType': component, 'count': len(data), 'type': kind})
        return len(accessors)-1
    primitives = []
    for index in range(2 if multiple else 1):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], dtype=float)+[index*2, 0, 0]
        attrs = {'POSITION': add(points, 'VEC3'), 'NORMAL': add([[0, 0, 1]]*3, 'VEC3')}
        if not constant:
            attrs['TEXCOORD_0'] = add([[-.1, .2], [1.1, .2], [.3, .95]], 'VEC2')
        if tangent:
            attrs['TANGENT'] = add([[1, 0, 0, -1]]*3, 'VEC4')
        primitives.append({'attributes': attrs, 'indices': add([[0], [1], [2]], 'SCALAR', 5123),
                           'material': index, 'extras': {'original_order': index}})
    doc = {'asset': {'version': '2.0', 'generator': 'analytic-compose-fixture'}, 'scene': 0,
           'scenes': [{'nodes': [0]}], 'nodes': [{'name': 'OriginalMesh', 'mesh': 0}],
           'meshes': [{'name': 'OriginalMesh', 'primitives': primitives}],
           'accessors': accessors, 'bufferViews': views,
           'materials': [{'name': 'Paint', 'pbrMetallicRoughness': {'baseColorFactor': [.2, .4, .6, 1],
                          'metallicFactor': .15, 'roughnessFactor': .7}}]}
    if multiple:
        doc['materials'].append({'pbrMetallicRoughness': {'baseColorFactor': [.9, .7, .3, 1]}, 'doubleSided': True})
    if not constant:
        image = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [200, 100, 50]]], dtype=np.uint8)
        stream = io.BytesIO(); Image.fromarray(image).save(stream, format='PNG'); packed = stream.getvalue()
        binary.extend(b'\0'*(-len(binary) % 4)); offset = len(binary); binary.extend(packed)
        views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(packed)})
        doc.update(images=[{'name': 'OriginalPixels', 'bufferView': len(views)-1, 'mimeType': 'image/png'}],
                   samplers=[{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}],
                   textures=[{'source': 0, 'sampler': 0}])
        doc['materials'][0]['pbrMetallicRoughness']['baseColorTexture'] = {'index': 0}
        doc['materials'][0]['normalTexture'] = {'index': 0, 'scale': .3}
    doc['buffers'] = [{'byteLength': len(binary)}]
    return doc, binary


class ComposeSourcesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.counter = 0
        self.a, self.ab = fixture(multiple=True)
        self.b, self.bb = fixture(tangent=False)
        self.config = {'schema_version': 1, 'id': 'analytic-pair', 'prompt': 'Two independently textured analytic pieces',
            'rights': 'Locally authored analytic triangles and2x2 PNG; no generated or external artwork',
            'space': 'gltf_y_up_right_handed', 'requested_rig_height_m': 1.2,
            'components': [
                {'id': 'main', 'source': 'a.glb', 'source_sha256': '0'*64, 'source_meters_per_unit': .5,
                 'placement': {'translation_m': [2, 0, 1], 'rotation_xyzw': [0, 0, 0, 1], 'uniform_scale': 2}},
                {'id': 'part', 'source': 'b.glb', 'source_sha256': '0'*64, 'source_meters_per_unit': .01,
                 'placement': {'translation_m': [-1, .5, 0], 'rotation_xyzw': [0, np.sqrt(.5), 0, np.sqrt(.5)], 'uniform_scale': 2}}]}

    def save(self):
        for name, doc, binary in (('a.glb', self.a, self.ab), ('b.glb', self.b, self.bb)):
            data = glb_bytes(doc, binary); (self.root/name).write_bytes(data)
            next(p for p in self.config['components'] if p['source'] == name)['source_sha256'] = hashlib.sha256(data).hexdigest()
        path = self.root/'compose.json'; path.write_text(json.dumps(self.config, indent=1)+'\n'); return path

    def execute(self, saved=False):
        path = self.root/'compose.json' if saved else self.save()
        self.counter += 1; out = self.root/('out-'+str(self.counter))
        return compose(path, out), out

    def test_real_bounded_worker_composes_rotation_scale_materials_and_unchanged_bytes(self):
        config = self.save(); originals = {p.name: p.read_bytes() for p in (config, self.root/'a.glb', self.root/'b.glb')}
        result, out = self.execute(saved=True)
        self.assertEqual(result['status'], 'unrigged_local_assembly', result)
        actual = _GLB(out/'model.glb'); before_a = _GLB(self.root/'a.glb'); before_b = _GLB(self.root/'b.glb')
        self.assertEqual(actual.doc['nodes'], [{'name': 'analytic-pair', 'mesh': 0}])
        self.assertNotIn('skins', actual.doc); self.assertNotIn('animations', actual.doc)
        self.assertEqual([m['name'] for m in actual.doc['materials']], ['main__m0', 'main__m1', 'part__m0'])
        for output_index, source, source_index in ((0, before_a, 0), (1, before_a, 1), (2, before_b, 0)):
            original = source.doc['meshes'][0]['primitives'][source_index]; primitive = actual.doc['meshes'][0]['primitives'][output_index]
            a, b = original['attributes'], primitive['attributes']
            points = _array(source, a['POSITION'], 'VEC3', {5126}).astype(float)
            expected = points+[2, 0, 1] if output_index < 2 else points[:, [2, 1, 0]]*[.02, .02, -.02]+[-1, .5, 0]
            np.testing.assert_allclose(_array(actual, b['POSITION'], 'VEC3', {5126}), expected, atol=1e-7)
            np.testing.assert_array_equal(_array(actual, b['TEXCOORD_0'], 'VEC2', {5126}), _array(source, a['TEXCOORD_0'], 'VEC2', {5126}))
            np.testing.assert_array_equal(_array(actual, primitive['indices'], 'SCALAR', {5123}), [[0], [1], [2]])
            expected_normal = [0, 0, 1] if output_index < 2 else [1, 0, 0]
            np.testing.assert_allclose(_array(actual, b['NORMAL'], 'VEC3', {5126}), [expected_normal]*3, atol=1e-7)
            if output_index < 2:
                np.testing.assert_array_equal(_array(actual, b['TANGENT'], 'VEC4', {5126}), [[1, 0, 0, -1]]*3)
            else:
                self.assertNotIn('TANGENT', b)
            self.assertEqual(primitive['extras'], original['extras'])
        for index, source in ((0, before_a), (1, before_b)):
            image = actual.doc['images'][index]; view = actual.doc['bufferViews'][image['bufferView']]
            source_image = source.doc['images'][0]; source_view = source.doc['bufferViews'][source_image['bufferView']]
            self.assertEqual(actual.binary[view['byteOffset']:view['byteOffset']+view['byteLength']],
                             source.binary[source_view['byteOffset']:source_view['byteOffset']+source_view['byteLength']])
        expected_materials = deepcopy(before_a.doc['materials']+before_b.doc['materials'])
        for index, name in enumerate(('main__m0', 'main__m1', 'part__m0')):
            expected_materials[index]['name'] = name
        expected_materials[2]['pbrMetallicRoughness']['baseColorTexture']['index'] = 1
        expected_materials[2]['normalTexture']['index'] = 1
        self.assertEqual(actual.doc['materials'], expected_materials)
        for name, data in originals.items(): self.assertEqual((self.root/name).read_bytes(), data)
        self.assertEqual((out/'config.original.json').read_bytes(), originals['compose.json'])
        receipt = json.loads((out/'worker-result.json').read_text())
        self.assertEqual(receipt['measured_totals']['vertices'], 9)
        self.assertEqual(receipt['measured_totals']['triangles'], 3)
        self.assertEqual(receipt['measured_totals']['image_pixels'], 8)
        canonical = receipt['requested_native_canonical_transform']
        self.assertFalse(canonical['applied']); self.assertAlmostEqual(canonical['uniform_scale'], .6)
        np.testing.assert_allclose(canonical['blender_offset_before_scale'], [2., -.49, 0.], atol=1e-7)
        manifest = json.loads((out/'manifest.json').read_text())
        self.assertEqual(manifest['kind'], 'unrigged_source_assembly')
        self.assertFalse(manifest['rig_created']); self.assertFalse(manifest['animation_verified'])
        self.assertNotIn('manifest.json', [row['path'] for row in manifest['artifacts']])
        for row in manifest['artifacts']:
            self.assertEqual(hashlib.sha256((out/row['path']).read_bytes()).hexdigest(), row['sha256'])
        job = json.loads((out/'job.json').read_text()); self.assertTrue(job['process_exit_verified'])
        self.assertEqual(job['limits']['rss_mb'], 512); self.assertEqual(job['limits']['wall_seconds'], 60)
        second, second_out = self.execute(saved=True)
        self.assertEqual(second['status'], 'unrigged_local_assembly')
        self.assertEqual((out/'model.glb').read_bytes(), (second_out/'model.glb').read_bytes())

    def test_constant_and_default_materials_need_no_images_or_uvs(self):
        self.a, self.ab = fixture(constant=True, multiple=True)
        self.b, self.bb = fixture(constant=True, tangent=False)
        self.b['materials'][0] = {}
        result, out = self.execute()
        self.assertEqual(result['status'], 'unrigged_local_assembly', result)
        doc = _GLB(out/'model.glb').doc
        self.assertNotIn('images', doc); self.assertNotIn('textures', doc); self.assertNotIn('samplers', doc)
        self.assertEqual(doc['materials'][2], {'name': 'part__m0'})
        self.assertTrue(all('TEXCOORD_0' not in p['attributes'] for p in doc['meshes'][0]['primitives']))

    def test_rotated_tangents_keep_exact_handedness(self):
        self.b, self.bb = fixture(tangent=True)
        result, out = self.execute(); self.assertEqual(result['status'], 'unrigged_local_assembly', result)
        glb = _GLB(out/'model.glb'); a = glb.doc['meshes'][0]['primitives'][2]['attributes']
        value = _array(glb, a['TANGENT'], 'VEC4', {5126})
        np.testing.assert_allclose(value[:, :3], [[0, 0, -1]]*3, atol=1e-7)
        np.testing.assert_array_equal(value[:, 3], [-1]*3)

    def test_wrong_hash_and_existing_output_preserve_originals_and_failures(self):
        config = self.save(); data = json.loads(config.read_text()); data['components'][1]['source_sha256'] = '0'*64
        config.write_text(json.dumps(data)); result, out = self.execute(saved=True)
        self.assertEqual(result['status'], 'unsupported'); self.assertIn('SHA256', result['error']['message'])
        self.assertEqual((out/'inputs/part.glb').read_bytes(), (self.root/'b.glb').read_bytes())
        self.assertFalse((out/'model.glb').exists()); self.assertFalse((out/'job.json').exists())
        manifest = (out/'manifest.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'exists'): compose(config, out)
        self.assertEqual((out/'manifest.json').read_bytes(), manifest)

    def test_invalid_and_nonfinite_configuration_rejects_before_worker(self):
        for mutate in (lambda c: c.update(schema_version=True), lambda c: c.update(prompt=''),
                lambda c: c.update(rights=''), lambda c: c.update(space='guess'),
                lambda c: c.update(requested_rig_height_m=float('nan')),
                lambda c: c['components'][0].update(source_meters_per_unit=True),
                lambda c: c['components'][0]['placement'].update(uniform_scale=-1),
                lambda c: c['components'][0]['placement'].update(uniform_scale=[1, 2, 1]),
                lambda c: c['components'][0]['placement'].update(rotation_xyzw=[0, 0, 0, 0]),
                lambda c: c['components'][0]['placement'].update(translation_m=[float('inf'), 0, 0]),
                lambda c: c['components'][1].update(id='main'), lambda c: c.update(components=c['components']*2)):
            data = deepcopy(self.config); mutate(data)
            with self.assertRaises(ValueError): validate_config(data)
        config = self.save(); config.write_bytes(b' '* (MAX_CONFIG_BYTES+1))
        result, out = self.execute(saved=True); self.assertEqual(result['status'], 'unsupported'); self.assertFalse((out/'job.json').exists())

    def test_bad_glb_features_are_retained_as_explicit_worker_rejections(self):
        initial = deepcopy(self.b)
        mutations = [lambda d: d.update(skins=[{'joints': [0]}]), lambda d: d.update(animations=[{}]),
            lambda d: d['meshes'][0]['primitives'][0]['attributes'].update(JOINTS_0=0),
            lambda d: d['meshes'][0]['primitives'][0].update(targets=[]),
            lambda d: d['meshes'][0]['primitives'][0].pop('indices'),
            lambda d: d['images'][0].update(uri='data:image/png;base64,AAAA'),
            lambda d: d['buffers'][0].update(uri='https://unrelated.invalid/model.bin'),
            lambda d: d.update(extensionsUsed=['KHR_mesh_quantization']),
            lambda d: d['nodes'][0].update(scale=[-1, 1, 1]),
            lambda d: d['nodes'].append({'mesh': 0}),
            lambda d: d.update(materials=[])]
        for mutate in mutations:
            self.b = deepcopy(initial); mutate(self.b)
            with self.subTest(source=self.b):
                result, out = self.execute()
                self.assertEqual(result['status'], 'unsupported')
                self.assertFalse((out/'model.glb').exists())
                self.assertEqual(json.loads((out/'worker-result.json').read_text())['status'], 'unsupported')
                self.assertTrue(json.loads((out/'job.json').read_text())['process_exit_verified'])

    def test_identity_defaults_and_inactive_or_nonidentity_frames(self):
        doc, _ = fixture()
        for node in ({'mesh': 0}, {'mesh': 0, 'translation': [0, 0, 0], 'rotation': [0, 0, 0, -1], 'scale': [1, 1, 1]},
                     {'mesh': 0, 'matrix': np.eye(4).ravel().tolist()}):
            doc['nodes'] = [node]; self.assertEqual(_identity_instance(doc), 0)
        for nodes, roots in (([{'mesh': 0, 'translation': [0, 0, .001]}], [0]),
                             ([{'mesh': 0, 'scale': [1, 2, 1]}], [0]),
                             ([{'mesh': 0, 'matrix': [1, .1, 0, 0]+np.eye(4).ravel().tolist()[4:]}], [0]),
                             ([{}, {'mesh': 0}], [0]), ([{'mesh': 0, 'children': [0]}], [0])):
            doc['nodes'] = nodes; doc['scenes'][0]['nodes'] = roots
            with self.assertRaises(ValueError): _identity_instance(doc)

    def test_exact_geometry_scalar_and_image_resource_boundaries(self):
        totals = {'vertices': 99999, 'triangles': 99999, 'decoded_scalars': 1999999}
        _add_counts(totals, vertices=1, triangles=1, scalars=1)
        self.assertEqual(totals, {'vertices': 100000, 'triangles': 100000, 'decoded_scalars': 2000000})
        for key in ('vertices', 'triangles', 'scalars'):
            with self.assertRaises(ValueError): _add_counts(dict(totals), **{key: 1})
        config = self.save(); huge = self.root/'a.glb'
        with huge.open('r+b') as stream: stream.truncate(MAX_SOURCE_BYTES+1)
        result, out = self.execute(saved=True); self.assertEqual(result['status'], 'unsupported')
        self.assertIn('64MiB', result['error']['message']); self.assertFalse((out/'inputs/main.glb').exists())
        self.b, self.bb = fixture(); path = self.root/'image.glb'; path.write_bytes(glb_bytes(self.b, self.bb)); glb = _GLB(path)
        with self.assertRaisesRegex(ValueError, '96Mi-pixels'):
            _image_inventory(glb, {'image_pixels': 96*1024**2-3})
        rows = _image_inventory(glb, {'image_pixels': 96*1024**2-4})
        self.assertEqual(rows[0]['width'], 2)

    def test_worker_env_is_explicit_and_unknown_resource_exit_is_not_success(self):
        self.save()
        captured = {}
        def fail(command, log, output, **limits):
            captured.update(command=command, limits=limits)
            return {'process_exit_verified': True, 'exit_code': -15, 'failure': 'resident-memory budget exceeded'}
        with patch.dict('os.environ', {'MESHY_API_KEY': 'not-a-real-key', 'SECRET_TEST': 'must-not-pass'}), \
                patch('avatar_assets.runner.run_bounded', side_effect=fail):
            result, out = self.execute(saved=True)
        self.assertEqual(result['status'], 'unsupported')
        command = captured['command']; self.assertEqual(command[:2], ['/usr/bin/env', '-i'])
        self.assertFalse(any('SECRET_TEST' in x or 'MESHY_API_KEY' in x or x.startswith(('HOME=', 'CODEX_HOME=')) for x in command))
        self.assertEqual(captured['limits'], {'timeout': 60, 'memory_mb': 512, 'output_mb': 128})
        self.assertFalse((out/'model.glb').exists()); self.assertTrue((out/'manifest.json').exists())

    def test_bounded_snapshot_rejects_growth_and_post_hash_identity_changes(self):
        source = self.root/'growing.bin'; source.write_bytes(b'abcdef')
        with self.assertRaisesRegex(ValueError, 'size'):
            _copy_bounded(source, self.root/'copy.bin', 4)
        self.assertFalse((self.root/'copy.bin').exists())
        state = _file_state(source.lstat()); source.rename(self.root/'old.bin'); source.write_bytes(b'abcdef')
        with self.assertRaisesRegex(ValueError, 'identity'):
            _bounded_sha(source, 6, state)
        link = self.root/'link.bin'; link.symlink_to(source)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            _bounded_sha(link, 6)
        _copy_bounded(source, self.root/'stable.bin', 6)
        self.assertEqual((self.root/'stable.bin').read_bytes(), b'abcdef')

    def test_aggregate_metadata_fails_before_vertex_decode_or_output_allocation(self):
        self.a['accessors'][0]['count'] = 100001
        self.save(); config = deepcopy(self.config)
        for part in config['components']: part['source'] = str(self.root/part['source'])
        with patch('avatar_assets.fringe_export._array', side_effect=AssertionError('Must not decode')):
            with self.assertRaisesRegex(ValueError, '100k vertices'):
                _metadata_inventory(config)
        self.assertFalse(list(self.root.glob('out-*')))

    def test_aggregate_json_limit_rejects_before_writing_a_model(self):
        self.a['meshes'][0]['primitives'][0]['extras'] = {'bounded_original_note': 'a'*(1100*1024)}
        self.b['meshes'][0]['primitives'][0]['extras'] = {'bounded_original_note': 'b'*(1100*1024)}
        result, out = self.execute()
        self.assertEqual(result['status'], 'unsupported'); self.assertIn('JSON exceeds2MiB', result['error']['message'])
        self.assertFalse((out/'model.glb').exists())

    def test_final_metadata_output_limit_cannot_report_supported(self):
        self.save()
        with patch('avatar_assets.compose_sources.MAX_OUTPUT_BYTES', 1024):
            result, out = self.execute(saved=True)
        self.assertEqual(result['status'], 'unsupported')
        self.assertIn('including metadata', result['error']['message'])
        self.assertTrue((out/'model.glb').exists())
        self.assertEqual(json.loads((out/'manifest.json').read_text())['status'], 'unsupported')

    def test_changed_prepared_config_or_code_cannot_be_accepted_after_worker(self):
        from avatar_assets.runner import run_bounded
        self.save()
        def mutate(command, log, output, **limits):
            job = run_bounded(command, log, output, **limits)
            (output/'config.worker.json').write_text('{}')
            return job
        with patch('avatar_assets.runner.run_bounded', side_effect=mutate):
            result, out = self.execute(saved=True)
        self.assertEqual(result['status'], 'unsupported')
        self.assertIn('size', result['error']['message'])
        self.assertTrue((out/'model.glb').exists())


if __name__ == '__main__':
    unittest.main()
