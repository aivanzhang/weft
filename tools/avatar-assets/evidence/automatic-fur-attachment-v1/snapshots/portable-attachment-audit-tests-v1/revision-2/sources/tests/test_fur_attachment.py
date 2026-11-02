"""Analytic exported skins exercise custody and attachment; these are not avatar assets."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from avatar_assets import fur_attachment as audit
from avatar_assets import fur_bindings
from avatar_assets.safety import sha256
from tests.test_three_d import glb_bytes


def fixture(root, *, mixed=False, seam=False, reordered=False, parts=1, six_bones=False,
            historical=False, no_fur=False, coat_palette_reversed=False):
    root.mkdir(exist_ok=True)
    (root/'sources').mkdir(exist_ok=True)
    binary, views, accessors = bytearray(), [], []

    def add(rows, kind, component=5126):
        binary.extend(b'\0'*(-len(binary) % 4))
        offset = len(binary)
        for row in rows:
            binary.extend(struct.pack('<'+{5126: 'f', 5123: 'H'}[component]*len(row), *row))
        views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(binary)-offset})
        accessors.append({'bufferView': len(views)-1, 'componentType': component,
                          'count': len(rows), 'type': kind})
        return len(accessors)-1

    source = np.array([[-.5, 0, 0], [.5, 0, 0], [0, 1, 0], [-.5, 1, 0]], dtype=float)
    source_ids = np.array([0, 1, 2, 3, 0] if seam else [0, 1, 2, 3])
    triangles = [[0, 1, 2], [4 if seam else 0, 2, 3]]
    if six_bones:
        joints_by_id = np.array([[0, 1, 0, 0], [2, 3, 0, 0], [4, 5, 0, 0], [0, 1, 0, 0]])
        weights_by_id = np.tile([.5, .5, 0, 0], (4, 1))
        bary = np.array([1/3, 1/3, 1/3])
    else:
        joints_by_id = np.tile([0, 1, 0, 0], (4, 1))
        weights_by_id = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [.5, .5, 0, 0], [1, 0, 0, 0]]) if mixed else np.tile([.4, .6, 0, 0], (4, 1))
        bary = np.array([.25, .25, .5])
    permutation = np.arange(len(source_ids))[::-1] if reordered else np.arange(len(source_ids))
    reverse = np.argsort(permutation)
    body_attrs = {'POSITION': add(source[source_ids][permutation], 'VEC3'),
                  'JOINTS_0': add(joints_by_id[source_ids][permutation], 'VEC4', 5123),
                  'WEIGHTS_0': add(weights_by_id[source_ids][permutation], 'VEC4'),
                  audit.ATTRIBUTE: add(source_ids[permutation, None], 'SCALAR')}
    meshes = [{'primitives': [{'attributes': body_attrs, 'indices': add(reverse[np.array(triangles)].reshape(-1, 1), 'SCALAR', 5123),
                              'material': 0}]}]
    nodes = [{'name': 'Felt form', 'mesh': 0, 'skin': 0}]
    part_reports, receipts = [], []
    for part in range(parts):
        name = 'PortableCoat_dusk'+('' if part == 0 else f'.{part:03}')
        inset = np.array([0., .0002, 0.])
        midpoint = bary @ source[:3]-inset
        xyz = np.array([midpoint+[-.02, 0, 0], midpoint+[.02, 0, 0],
                        midpoint+[-.02, .015, 0], midpoint+[.02, .015, 0],
                        midpoint+[-.01, .03, 0], midpoint+[.01, .03, 0]])
        coat_joints = np.array([0, 1, 2, 3] if six_bones else [0, 1, 0, 0])
        coat_weights = np.full(4, .25) if six_bones else bary @ weights_by_id[:3]
        if coat_palette_reversed:
            coat_joints = 1-coat_joints
        card_order = np.array([5, 2, 4, 1, 3, 0]) if reordered else np.arange(6)
        card_inverse = np.argsort(card_order)
        attrs = {'POSITION': add(xyz[card_order], 'VEC3'),
                 'JOINTS_0': add(np.tile(coat_joints, (6, 1))[card_order], 'VEC4', 5123),
                 'WEIGHTS_0': add(np.tile(coat_weights, (6, 1))[card_order], 'VEC4'),
                 audit.ATTRIBUTE: add(card_order[:, None], 'SCALAR')}
        indices = card_inverse[np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]])]
        meshes.append({'primitives': [{'attributes': attrs, 'indices': add(indices.reshape(-1, 1), 'SCALAR', 5123), 'material': 1}]})
        nodes.append({'name': name, 'mesh': part+1, 'skin': 1 if coat_palette_reversed else 0})
        bindings = {'triangle_index': np.array([0], dtype=np.int32),
                    'barycentric': bary[None, :], 'root_inset': np.array([[0, 0, .0002]])}
        correspondence = {'source_node': 'Felt form', 'coat_node': name,
                          'source_vertex_count': 4, 'source_triangle_count': 2,
                          'source_vertex_ids': np.array([[0, 1, 2]], dtype=np.int32)}
        receipt = fur_bindings.write_bindings(root, name, bindings, 1,
                                             correspondence=None if historical else correspondence)
        receipts.append(receipt)
        part_reports.append({'cards': 1, 'vertices': 6, 'geometry_profile': ['tuft-v1', 'combed-v1', 'legacy'][part % 3],
                             'root_summary': {'roots': 1, 'root_placement': 'skin-consistent-v1', 'binding_archive': receipt}})
    joint_count = 6 if six_bones else 2
    joint_start = len(nodes)
    nodes.extend({'name': f'hinge_{i}', **({'children': [joint_start+1]} if i == 0 else {})}
                 for i in range(joint_count))
    identity_matrix = np.eye(4).reshape(-1, order='F')
    inverse = add(np.tile(identity_matrix, (joint_count, 1)), 'MAT4')
    skins = [{'joints': list(range(joint_start, joint_start+joint_count)), 'inverseBindMatrices': inverse}]
    if coat_palette_reversed:
        skins.append({'joints': [joint_start+1, joint_start], 'inverseBindMatrices': inverse})
    q = np.array([0, 0, .5, np.sqrt(.75)]) if not six_bones else np.array([0, 0, 0, 1.])
    times = add([[0], [.5], [1]], 'SCALAR')
    rotations = add([[0, 0, 0, 1], q, [0, 0, 0, 1]], 'VEC4')
    long_times = add([[0], [2], [4]], 'SCALAR')
    scale = add([[1, 1, 1], [1, 1.2, 1], [1, 1, 1]], 'VEC3')
    clips = [{'name': 'sway', 'samplers': [{'input': times, 'output': rotations}],
              'channels': [{'sampler': 0, 'target': {'node': joint_start+1, 'path': 'rotation'}}]},
             {'name': 'reach', 'samplers': [{'input': long_times, 'output': scale}],
              'channels': [{'sampler': 0, 'target': {'node': joint_start, 'path': 'scale'}}]}]
    doc = {'asset': {'version': '2.0', 'generator': 'analytic-attachment-fixture'},
           'buffers': [{'byteLength': len(binary)}], 'bufferViews': views, 'accessors': accessors,
           'nodes': nodes, 'meshes': meshes, 'skins': skins, 'animations': clips,
           'materials': [{'name': 'ochre fabric'}, {'name': 'ivory strands'}],
           'scene': 0, 'scenes': [{'nodes': [*range(parts+1), joint_start,
                                           *range(joint_start+2, joint_start+joint_count)]}]}
    if no_fur:
        doc['meshes'] = doc['meshes'][:1]
        body_attrs.pop(audit.ATTRIBUTE)
        for node in doc['nodes'][1:parts+1]:
            node.pop('mesh'); node.pop('skin')
    state_ids = ['sway', 'reach']
    source_hash = hashlib.sha256(b'analytic source, not a generated avatar').hexdigest()
    design = {'schema_version': 1, 'id': 'analytic-attachment', 'source_sha256': source_hash,
              'states': [{'id': s} for s in state_ids],
              'portable_fur': None if no_fur else {'bands': [{'count': parts}]}}
    identity = {'schema_version': 1, 'policy': audit.IDENTITY_POLICY, 'attribute': audit.ATTRIBUTE,
                'space': 'applied_rest_Blender_XYZ_meters', 'source_node': 'Felt form',
                'source_vertex_count': 4, 'source_triangle_count': 2,
                'coats': [{'node': row['export_correspondence']['coat_node'], 'vertices': 6, 'roots': 1}
                          for row in receipts] if not historical else [],
                'recorded_roots': parts, 'export_attributes': True, 'native_attribute_type': 'INT',
                'native_attribute_domain': 'POINT', 'export_attribute_type': 'SCALAR', 'export_component_type': 5126,
                'identity_scope': 'IDs are local to each named mesh; bottom card endpoints are 6*r and 6*r+1',
                'attachment_verified': False, 'source_skin_restoration_performed': False}
    finish = {'schema_version': 1, 'source_sha256': source_hash, 'source_unchanged': True,
              'states': design['states'], 'portable_fur': None if no_fur else part_reports,
              'portable_fur_identity': None if historical or no_fur else identity}
    return {'root': root, 'doc': doc, 'binary': binary, 'design': design, 'finish': finish,
            'receipts': receipts, 'state_ids': state_ids}


def persist(f):
    root = f['root']
    (root/'model.glb').write_bytes(glb_bytes(f['doc'], f['binary']))
    for name, value in (('sources/finish-design.json', f['design']), ('finish-report.json', f['finish'])):
        (root/name).write_text(json.dumps(value, sort_keys=True))
    artifacts = [{'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
                 for path in sorted(root.rglob('*')) if path.is_file() and path.name != 'manifest.json']
    manifest = {'schema_version': 1, 'id': f['design']['id'], 'status': 'needs_visual_review',
                'source_sha256': f['design']['source_sha256'], 'states': f['design']['states'],
                'finish': f['finish'], 'artifacts': artifacts}
    (root/'manifest.json').write_text(json.dumps(manifest, sort_keys=True))
    return {'schema_version': 1, 'source_sha256': f['design']['source_sha256'],
            'design_sha256': sha256(root/'sources/finish-design.json'), 'model_sha256': sha256(root/'model.glb'),
            'state_ids': f['state_ids']}


def set_row(f, mesh, semantic, row, values):
    attrs = f['doc']['meshes'][mesh]['primitives'][0]['attributes']
    accessor = f['doc']['accessors'][attrs[semantic]]
    view = f['doc']['bufferViews'][accessor['bufferView']]
    fmt = {5126: 'f', 5123: 'H'}[accessor['componentType']]
    width = len(values)
    struct.pack_into('<'+fmt*width, f['binary'], view['byteOffset']+row*width*struct.calcsize(fmt), *values)


class FurAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()/'pack'

    def run_fixture(self, **options):
        f = fixture(self.root, **options)
        intent = persist(f)
        return f, intent, audit.audit_finish_pack(self.root, intent)

    def test_valid_animated_actual_skin_with_inset_and_reordered_palette(self):
        _, intent, result = self.run_fixture(coat_palette_reversed=True)
        self.assertTrue(result['passed'])
        self.assertEqual(result['status'], 'measured')
        self.assertEqual([len(s['times']) for s in result['states']], [25, 97])
        self.assertEqual(result['metrics']['root_pose_checks'], 122)
        self.assertLess(result['metrics']['max_attachment_error_m'], 1e-7)
        with patch.object(audit, '_measured', side_effect=AssertionError('numeric rerun')):
            with patch.object(audit.three_d, '_GLB', side_effect=AssertionError('model redecode')):
                self.assertEqual(audit.validate_report(self.root, intent, result), result)

    def test_correct_interpolated_weights_do_not_hide_midpoint_detachment(self):
        _, _, result = self.run_fixture(mixed=True)
        self.assertFalse(result['passed'])
        self.assertEqual(result['status'], 'measured')
        self.assertEqual(result['metrics']['named_weight_failures'], 0)
        self.assertEqual(result['metrics']['rest_midpoint_failures'], 0)
        self.assertGreater(result['metrics']['max_attachment_error_m'], .02)
        self.assertTrue(any(0 < w['second'] < 1 for w in result['witnesses'] if w['kind'] == 'animated_attachment'))

    def test_influence_mismatch_is_measured_not_malformed(self):
        f = fixture(self.root)
        for row in range(6):
            set_row(f, 1, 'WEIGHTS_0', row, [1, 0, 0, 0])
        result = audit.audit_finish_pack(self.root, persist(f))
        self.assertEqual(result['status'], 'measured')
        self.assertFalse(result['passed'])
        self.assertEqual(result['metrics']['named_weight_failures'], 1)

    def test_full_inherited_weights_are_not_truncated_to_producer_top_four(self):
        _, _, result = self.run_fixture(six_bones=True)
        self.assertFalse(result['passed'])
        self.assertGreater(result['metrics']['max_named_weight_error'], .16)
        self.assertEqual(result['metrics']['named_weight_failures'], 1)

    def test_detached_rest_midpoint_fails_fixed_gate(self):
        f = fixture(self.root)
        for row, x in ((0, -.02), (1, .02)):
            set_row(f, 1, 'POSITION', row, [x, .5018, 0])
        result = audit.audit_finish_pack(self.root, persist(f))
        self.assertFalse(result['passed'])
        self.assertEqual(result['metrics']['rest_midpoint_failures'], 1)
        self.assertGreater(result['metrics']['max_rest_midpoint_error_m'], .0019)

    def test_seams_reordered_rows_and_multiple_bands_use_ids(self):
        _, _, result = self.run_fixture(seam=True, reordered=True, parts=3)
        self.assertTrue(result['passed'])
        self.assertEqual(result['roots'], 3)
        self.assertEqual(result['metrics']['exported_vertices'], 23)
        self.assertEqual(result['metrics']['root_pose_checks'], 366)

    def test_conflicting_seam_positions_reject_before_measurement(self):
        f = fixture(self.root, seam=True)
        set_row(f, 0, 'POSITION', 4, [-.49, 0, 0])
        with self.assertRaisesRegex(ValueError, 'Ambiguous seam'):
            audit.audit_finish_pack(self.root, persist(f))

    def test_conflicting_seam_named_weights_reject(self):
        f = fixture(self.root, seam=True)
        set_row(f, 0, 'WEIGHTS_0', 4, [.5, .5, 0, 0])
        with self.assertRaisesRegex(ValueError, 'Ambiguous seam'):
            audit.audit_finish_pack(self.root, persist(f))

    def test_missing_export_identity_rejects_without_position_guessing(self):
        f = fixture(self.root)
        f['doc']['meshes'][1]['primitives'][0]['attributes'].pop(audit.ATTRIBUTE)
        with self.assertRaisesRegex(ValueError, 'Missing exported'):
            audit.audit_finish_pack(self.root, persist(f))

    def test_noninteger_or_out_of_range_export_ids_reject(self):
        f = fixture(self.root)
        set_row(f, 1, audit.ATTRIBUTE, 0, [.5])
        with self.assertRaisesRegex(ValueError, 'Invalid exported point IDs'):
            audit.audit_finish_pack(self.root, persist(f))

    def test_missing_source_id_rejects(self):
        f = fixture(self.root)
        set_row(f, 1, audit.ATTRIBUTE, 5, [4])
        with self.assertRaises(ValueError):
            audit.audit_finish_pack(self.root, persist(f))

    def test_wrong_source_triangle_winding_rejects(self):
        f = fixture(self.root)
        accessor = f['doc']['accessors'][f['doc']['meshes'][0]['primitives'][0]['indices']]
        offset = f['doc']['bufferViews'][accessor['bufferView']]['byteOffset']
        struct.pack_into('<3H', f['binary'], offset, 0, 2, 1)
        with self.assertRaisesRegex(ValueError, 'oriented source triangle'):
            audit.audit_finish_pack(self.root, persist(f))

    def test_unrecorded_mesh_owner_rejects(self):
        f = fixture(self.root)
        f['doc']['nodes'].append({'name': 'Unrecorded', 'mesh': 1, 'skin': 0})
        f['doc']['scenes'][0]['nodes'].append(len(f['doc']['nodes'])-1)
        with self.assertRaisesRegex(ValueError, 'Unrecorded'):
            audit.audit_finish_pack(self.root, persist(f))

    def test_nonidentity_mesh_is_explicitly_unsupported(self):
        f = fixture(self.root)
        f['doc']['nodes'][1]['translation'] = [1, 0, 0]
        result = audit.audit_finish_pack(self.root, persist(f))
        self.assertEqual(result['status'], 'unsupported')
        self.assertFalse(result['passed'])

    def test_historical_bindings_are_unsupported_not_reconstructed(self):
        _, _, result = self.run_fixture(historical=True)
        self.assertEqual(result['status'], 'unsupported')
        self.assertFalse(result['passed'])

    def test_missing_new_binding_cannot_masquerade_as_historical(self):
        f = fixture(self.root)
        f['finish']['portable_fur'][0]['root_summary'].pop('binding_archive')
        with self.assertRaisesRegex(ValueError, 'Advertised v2'):
            audit.audit_finish_pack(self.root, persist(f))

    def test_missing_requested_band_roots_cannot_pass_as_complete(self):
        f = fixture(self.root)
        f['design']['portable_fur']['bands'].append({'count': 1})
        with self.assertRaisesRegex(ValueError, 'exact requested band total'):
            audit.audit_finish_pack(self.root, persist(f))

    def test_cached_error_encoding_is_not_mutable(self):
        _, intent, result = self.run_fixture()
        result['error_encoding'] = 'different errors'
        with self.assertRaisesRegex(ValueError, 'scope'):
            audit.validate_report(self.root, intent, result)

    def test_no_requested_portable_fur_is_not_applicable(self):
        _, _, result = self.run_fixture(no_fur=True)
        self.assertEqual(result['status'], 'not_applicable')
        self.assertTrue(result['passed'])

    def test_missing_requested_parts_is_not_an_opt_out(self):
        f = fixture(self.root)
        f['finish']['portable_fur'] = []
        with self.assertRaisesRegex(ValueError, 'no recorded parts'):
            audit.audit_finish_pack(self.root, persist(f))

    def test_hash_changed_model_rejects(self):
        f = fixture(self.root)
        intent = persist(f)
        with (self.root/'model.glb').open('ab') as stream:
            stream.write(b'x')
        with self.assertRaisesRegex(ValueError, 'changed attachment input'):
            audit.audit_finish_pack(self.root, intent)

    def test_hash_changed_archive_rejects(self):
        f = fixture(self.root)
        intent = persist(f)
        path = self.root/f['receipts'][0]['path']
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'changed attachment input'):
            audit.audit_finish_pack(self.root, intent)

    def test_bound_but_corrupt_archive_rejects_before_numeric_audit(self):
        f = fixture(self.root)
        path = self.root/f['receipts'][0]['path']
        path.write_bytes(b'not a ZIP file')
        f['receipts'][0].update(bytes=path.stat().st_size, sha256=sha256(path))
        with self.assertRaises(ValueError):
            audit.audit_finish_pack(self.root, persist(f))

    def test_oversized_counts_reject_before_model_decode(self):
        f = fixture(self.root)
        f['finish']['portable_fur'][0]['cards'] = 5001
        intent = persist(f)
        with patch.object(audit.three_d, '_GLB', side_effect=AssertionError('must not decode')):
            with self.assertRaisesRegex(ValueError, 'root count'):
                audit.audit_finish_pack(self.root, intent)

    def test_oversized_json_rejects_before_loading(self):
        f = fixture(self.root)
        intent = persist(f)
        (self.root/'manifest.json').write_bytes(b' '*(audit.MAX_JSON+1))
        with self.assertRaisesRegex(ValueError, 'too large'):
            audit.audit_finish_pack(self.root, intent)

    def test_binding_path_symlink_is_rejected(self):
        f = fixture(self.root)
        intent = persist(f)
        path = self.root/f['receipts'][0]['path']
        other = self.root/'copy.npz'
        path.rename(other)
        path.symlink_to(other)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            audit.audit_finish_pack(self.root, intent)

    def test_cached_numeric_failure_cannot_be_relabelled_pass(self):
        _, intent, result = self.run_fixture(mixed=True)
        changed = deepcopy(result); changed['passed'] = True
        with self.assertRaisesRegex(ValueError, 'contradicts numeric'):
            audit.validate_report(self.root, intent, changed)
        changed = deepcopy(result); changed['metrics']['attachment_failures'] = 0
        with self.assertRaisesRegex(ValueError, 'count differs'):
            audit.validate_report(self.root, intent, changed)

    def test_cached_report_rebinds_archives_without_rerunning_poses(self):
        f, intent, result = self.run_fixture()
        path = self.root/f['receipts'][0]['path']
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'changed attachment input'):
            audit.validate_report(self.root, intent, result)

    def test_actual_clip_names_and_four_second_bound(self):
        f = fixture(self.root)
        f['doc']['animations'][0]['name'] = 'unrequested'
        with self.assertRaisesRegex(ValueError, 'Actual clips'):
            audit.audit_finish_pack(self.root, persist(f))

    def test_inverse_bind_difference_is_not_hidden_by_matching_bone_names(self):
        f = fixture(self.root, coat_palette_reversed=True)
        original = f['doc']['skins'][0]['inverseBindMatrices']
        old = f['doc']['accessors'][original]
        start = len(f['binary'])
        offset = f['doc']['bufferViews'][old['bufferView']]['byteOffset']
        f['binary'].extend(f['binary'][offset:offset+128])
        struct.pack_into('<f', f['binary'], start+12*4, .002)
        f['doc']['bufferViews'].append({'buffer': 0, 'byteOffset': start, 'byteLength': 128})
        f['doc']['accessors'].append({**old, 'bufferView': len(f['doc']['bufferViews'])-1})
        f['doc']['skins'][1]['inverseBindMatrices'] = len(f['doc']['accessors'])-1
        f['doc']['buffers'][0]['byteLength'] = len(f['binary'])
        result = audit.audit_finish_pack(self.root, persist(f))
        self.assertEqual(result['metrics']['named_weight_failures'], 0)
        self.assertGreater(result['metrics']['max_attachment_error_m'], .001)
        self.assertFalse(result['passed'])


if __name__ == '__main__':
    unittest.main()
