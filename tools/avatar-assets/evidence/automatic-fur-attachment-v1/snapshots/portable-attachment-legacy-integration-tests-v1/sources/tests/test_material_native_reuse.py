"""Analytic receipt custody only; fixture bytes are not native or media evidence."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from avatar_assets import finish_generated
from avatar_assets.fur_bindings import write_bindings
from avatar_assets.safety import artifacts, read_json, sha256, write_json


SCRIPT_SOURCES = (
    'blender_finish_generated.py', 'blender_native_groom.py', 'blender_comb_groom.py',
    'blender_verify.py', 'blender_fur_cards.py', 'blender_creature_rig.py', 'creature_rig_math.py',
    'motion_samples.py', 'native_render_tasks.py', 'deformation_diagnostics.py',
    'blender_surface_profiles.py', 'surface_smoothing.py', 'surface_integrity.py', 'groom_flow.py',
    'blender_color_corrections.py', 'blender_part_rigs.py', 'blender_normal_continuity.py',
    'normal_uv_export.py', 'blender_normal_uv_export.py', 'fur_root_sampling.py',
    'material_fur.py', 'blender_material_colors.py', 'blender_material_coat.py', 'blender_material_groom.py',
    'blender_fur_identity.py')
CONFIG_SOURCES = ('surface_profiles.py', 'surface_smoothing_config.py', 'color_corrections.py',
                  'part_rigs.py', 'normal_continuity.py', 'fur_bindings.py')
ANALYTIC = b'ANALYTIC COPY-CUSTODY BYTES; NOT A NATIVE MODEL OR MEDIA\n'


class MaterialNativeReuseTests(unittest.TestCase):
    def setUp(self, *, portable_bindings=True):
        self.temporary = tempfile.TemporaryDirectory(prefix='material-native-reuse-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root/'source'; self.source.mkdir()
        self.target = self.root/'target'; self.target.mkdir()
        (self.source/'sources').mkdir(); (self.source/'material-groom').mkdir()
        for directory, names in (('scripts', SCRIPT_SOURCES), ('avatar_assets', CONFIG_SOURCES)):
            for name in names:
                shutil.copyfile(finish_generated.ROOT/directory/name, self.source/'sources'/name)
        self.cfg = {'export_model': True, 'material_fur': {'version': 1, 'materials': ['coat', 'trim']},
                    'portable_fur': {'settings': {'profile': 'tuft-v1'}}}
        self.job = {'exit_code': 0, 'failure': None, 'process_exit_verified': True,
                    'scope': 'Analytic custody fixture; no native process was launched'}
        with (self.source/'material-groom/correspondence.npz').open('xb') as stream:
            np.savez(stream, source_vertex_ids=np.arange(6, dtype='<i4'),
                     part_00_face_ids=np.array([0], dtype='<i4'),
                     part_00_loop_ids=np.array([0, 1, 2], dtype='<i4'),
                     part_01_face_ids=np.array([1], dtype='<i4'),
                     part_01_loop_ids=np.array([3, 4, 5], dtype='<i4'))
        self.groom = {'schema_version': 1, 'kind': 'material_owned_native_grooms',
            'policy': 'material-owned-native-groom-v1', 'status': 'native_grooms_pending_render_review',
            'original_body_unchanged': True, 'native_only': True, 'quality_approved': False,
            'requested_parents': 32, 'actual_parents': 32, 'allocated_parents': [16, 16],
            'scope': 'Analytic custody data; counts are fixture values, not measured hair',
            'correspondence': {'path': 'material-groom/correspondence.npz',
                               'sha256': sha256(self.source/'material-groom/correspondence.npz')}}
        self.coat = {'schema_version': 1, 'policy': 'material-owned-tufts-v1',
                     'requested_cards': 100, 'created_cards': 100,
                     'body_materials_or_uvs_changed': False, 'scope': 'Analytic custody fixture'}
        self.normal_uv = {'schema_version': 1, 'policy': 'live-normal-uv-export-v1',
                          'status': 'supported', 'errors': [], 'scope': 'Analytic custody fixture'}
        self.finish_report = {'groom': deepcopy(self.groom), 'material_coat': deepcopy(self.coat),
                              'normal_uv_export': deepcopy(self.normal_uv)}
        if portable_bindings:
            self.cfg['portable_fur']['bands'] = [{'count': 100}]
            receipt = write_bindings(self.source, 'PortableCoat_fixture',
                {'triangle_index': [0]*100, 'barycentric': [[.2, .3, .5]]*100,
                 'root_inset': [[0., 0., .0005]]*100}, 100,
                correspondence={'source_node': 'analytic-body', 'coat_node': 'PortableCoat_fixture',
                                'source_vertex_count': 3, 'source_triangle_count': 1,
                                'source_vertex_ids': [[0, 1, 2]]*100})
            self.finish_report['portable_fur'] = [{'cards': 100, 'geometry_profile': 'tuft-v1',
                'root_summary': {'root_placement': 'area-stratified-v1', 'binding_archive': receipt}}]
        write_json(self.source/'finish-config.json', self.cfg)
        write_json(self.source/'finish-report.json', self.finish_report)
        write_json(self.source/'material-groom/report.json', self.groom)
        write_json(self.source/'material-coat.json', self.coat)
        write_json(self.source/'normal-uv-export.json', self.normal_uv)
        write_json(self.source/'job.json', self.job)
        for name in ('source.blend', 'portrait.png', 'finish.log', 'finish-monitor.json', 'finish-handle.json',
                     'model-native-export.glb', 'views/front.png', 'renders/idle/0000.png', 'textures/coat.png'):
            path = self.source/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(ANALYTIC+name.encode())
        self.seal()
        native = patch.object(finish_generated, 'run_bounded', side_effect=AssertionError('Reuse must not run a native job'))
        self.native = native.start(); self.addCleanup(native.stop)

    def seal(self):
        (self.source/'manifest.json').write_text(json.dumps({'schema_version': 1, 'status': 'analytic_fixture',
                                                          'artifacts': artifacts(self.source)}, sort_keys=True))

    def update_groom(self, value, *, embedded=True):
        (self.source/'material-groom/report.json').write_text(json.dumps(value))
        if embedded:
            self.finish_report['groom'] = deepcopy(value)
            (self.source/'finish-report.json').write_text(json.dumps(self.finish_report))
        self.seal()

    def assert_rejected(self, message=None):
        before, manifest = artifacts(self.source), (self.source/'manifest.json').read_bytes()
        context = self.assertRaisesRegex(ValueError, message) if message else self.assertRaises(ValueError)
        with context:
            finish_generated.reuse_native(self.source, self.target, self.cfg)
        self.assertEqual(artifacts(self.source), before)
        self.assertEqual((self.source/'manifest.json').read_bytes(), manifest)
        self.assertFalse((self.target/'native-reuse.json').exists())
        self.assertFalse((self.target/'model.glb').exists())
        self.assertFalse((self.target/'source.blend').exists())
        self.native.assert_not_called()

    def test_copy_preserves_exact_material_receipts_correspondence_and_related_assets(self):
        before, manifest = artifacts(self.source), (self.source/'manifest.json').read_bytes()
        actual_job = finish_generated.reuse_native(self.source, self.target, self.cfg)
        self.assertEqual(actual_job, self.job)
        for name in ('material-groom/report.json', 'material-groom/correspondence.npz', 'material-coat.json',
                     'normal-uv-export.json', 'source.blend', 'portrait.png', 'finish-report.json',
                     'job.json', 'views/front.png', 'renders/idle/0000.png', 'textures/coat.png'):
            with self.subTest(path=name):
                self.assertEqual((self.target/name).read_bytes(), (self.source/name).read_bytes())
        self.assertEqual((self.target/'model.glb').read_bytes(), (self.source/'model-native-export.glb').read_bytes())
        receipt = read_json(self.target/'native-reuse.json')
        self.assertFalse(receipt['native_execution_repeated'])
        self.assertEqual(receipt['source'], str(self.source))
        self.assertEqual(receipt['source_manifest_sha256'], sha256(self.source/'manifest.json'))
        self.assertEqual(receipt['source_export_sha256'], sha256(self.source/'model-native-export.glb'))
        self.assertEqual(read_json(self.target/'material-groom/report.json'), read_json(self.target/'finish-report.json')['groom'])
        self.assertEqual(read_json(self.target/'material-coat.json'), read_json(self.target/'finish-report.json')['material_coat'])
        self.assertEqual(artifacts(self.source), before)
        self.assertEqual((self.source/'manifest.json').read_bytes(), manifest)
        self.native.assert_not_called()

    def test_native_only_material_groom_does_not_require_or_invent_portable_receipt(self):
        self.cfg.pop('portable_fur'); self.finish_report.pop('material_coat')
        (self.source/'material-coat.json').unlink()
        (self.source/'finish-config.json').write_text(json.dumps(self.cfg))
        (self.source/'finish-report.json').write_text(json.dumps(self.finish_report)); self.seal()
        before = artifacts(self.source)
        finish_generated.reuse_native(self.source, self.target, self.cfg)
        self.assertTrue((self.target/'material-groom/report.json').is_file())
        self.assertFalse((self.target/'material-coat.json').exists())
        self.assertEqual(artifacts(self.source), before)
        self.native.assert_not_called()

    def test_unsealed_correspondence_change_fails_manifest_custody(self):
        with (self.source/'material-groom/correspondence.npz').open('ab') as stream:
            stream.write(b'changed analytic evidence')
        self.assert_rejected('changed since its manifest')

    def test_resealed_missing_groom_report_is_rejected(self):
        (self.source/'material-groom/report.json').unlink(); self.seal()
        self.assert_rejected('regular local file')

    def test_resealed_changed_groom_report_cannot_disagree_with_embedded_report(self):
        changed = deepcopy(self.groom); changed['allocated_parents'] = [15, 17]
        self.update_groom(changed, embedded=False)
        self.assert_rejected('native custody differs')

    def test_resealed_unsupported_groom_status_is_not_reusable(self):
        changed = deepcopy(self.groom); changed['status'] = 'unsupported'
        self.update_groom(changed)
        self.assert_rejected('native custody differs')

    def test_resealed_changed_body_claim_is_not_reusable(self):
        changed = deepcopy(self.groom); changed['original_body_unchanged'] = False
        self.update_groom(changed)
        self.assert_rejected('native custody differs')

    def test_resealed_missing_correspondence_is_rejected(self):
        (self.source/'material-groom/correspondence.npz').unlink(); self.seal()
        self.assert_rejected('regular local file')

    def test_resealed_changed_correspondence_cannot_keep_the_old_bound_hash(self):
        with (self.source/'material-groom/correspondence.npz').open('ab') as stream:
            stream.write(b'changed IDs')
        self.seal(); self.assert_rejected('correspondence is missing or changed')

    def test_resealed_alternate_correspondence_path_is_not_an_allowed_substitution(self):
        shutil.copyfile(self.source/'material-groom/correspondence.npz', self.source/'different.npz')
        changed = deepcopy(self.groom); changed['correspondence']['path'] = 'different.npz'
        self.update_groom(changed)
        self.assert_rejected('correspondence is missing or changed')

    def test_resealed_oversized_correspondence_is_rejected_even_with_matching_hash(self):
        path = self.source/'material-groom/correspondence.npz'
        with path.open('r+b') as stream:
            stream.truncate(8*1024**2+1)
        changed = deepcopy(self.groom); changed['correspondence']['sha256'] = sha256(path)
        self.update_groom(changed)
        self.assert_rejected('Input too large')

    def test_resealed_missing_portable_receipt_is_rejected(self):
        (self.source/'material-coat.json').unlink(); self.seal()
        self.assert_rejected('regular local file')

    def test_resealed_changed_portable_receipt_cannot_disagree_with_embedded_report(self):
        changed = deepcopy(self.coat); changed['created_cards'] = 99
        (self.source/'material-coat.json').write_text(json.dumps(changed)); self.seal()
        self.assert_rejected('coat receipt differs')

    def test_resealed_null_portable_receipt_cannot_match_a_missing_embedded_field(self):
        (self.source/'material-coat.json').write_text('null\n')
        self.finish_report.pop('material_coat')
        (self.source/'finish-report.json').write_text(json.dumps(self.finish_report)); self.seal()
        self.assert_rejected('coat receipt differs')

    def test_resealed_wrong_portable_policy_is_rejected_even_when_both_receipts_match(self):
        changed = deepcopy(self.coat); changed['policy'] = 'unrelated-analytic-policy'
        (self.source/'material-coat.json').write_text(json.dumps(changed))
        self.finish_report['material_coat'] = changed
        (self.source/'finish-report.json').write_text(json.dumps(self.finish_report)); self.seal()
        self.assert_rejected('coat receipt differs')

    def test_changed_requested_material_ownership_cannot_reuse_the_old_design(self):
        self.cfg['material_fur']['materials'] = ['coat']
        self.assert_rejected('config differs')

    def test_resealed_stale_material_backend_is_rejected(self):
        path = self.source/'sources/blender_material_groom.py'
        path.write_bytes(path.read_bytes()+b'\n# Analytic stale backend\n'); self.seal()
        self.assert_rejected('Material-owned fur backend changed')


if __name__ == '__main__':
    unittest.main()
