"""Analytic report fixtures; no generated avatar or native renderer is invoked."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from avatar_assets.deformation_gate import audit_finish_pack


SOURCE = 'a'*64


def outlier(rest=.004, excess=.4):
    return {'endpoint_indices': [0, 1], 'endpoint_influences': [{'Body': 1.0}, {'Hand': 1.0}],
            'rest_length': rest, 'posed_length': rest+excess, 'excess_length': excess,
            'stretch_ratio': (rest+excess)/rest, 'second': 2.,
            'rest_normalized_bounds': [[.19, .4, .42], [.191, .4, .42]]}


class DeformationGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='avatar-deformation-gate-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root/'sources').mkdir()
        self.design = {'schema_version': 1, 'id': 'analytic-gate', 'source_sha256': SOURCE,
                       'states': [{'id': 'wave'}],
                       'authored_rig': {'height_meters': 1.2,
                                        'bones': [{'name': 'Body'}, {'name': 'Hand'}]},
                       'authored_actions': [{'id': 'wave', 'keys': [
                           {'second': 0, 'rotations': {}},
                           {'second': 2, 'rotations': {'Hand': [0, 30, 0]}},
                           {'second': 4, 'rotations': {}}]}]}
        self.report = {'schema_version': 1, 'source_sha256': SOURCE, 'source_unchanged': True,
                       'authored_rig': {'design': deepcopy(self.design['authored_rig']),
                                        'height_meters': 1.2, 'authored_not_recovered': True},
                       'states': [{'id': 'wave', 'duration_seconds': 4,
                                   'frames': [0, 48, 96], 'seconds': [0, 2, 4],
                                   'sampling_method': 'authored_keyframes',
                                   'deformation_outliers': [outlier()]}]}
        self.rig_proof = None

    def native_measurement(self, *, source_skin=False, height=1.2):
        if source_skin:
            self.design.pop('authored_rig')
            self.report['authored_rig'] = None
        self.report['deformation_measurement'] = {
            'schema_version': 1, 'status': 'measured', 'rig_route': 'source_skin' if source_skin else 'authored',
            'coordinate_space': 'mesh_local_after_world_transform_applied', 'units': 'meters',
            'rest_sha256': 'b'*64, 'edges_sha256': 'c'*64,
            'rest_array_encoding': 'little-endian float64 XYZ', 'edge_array_encoding': 'little-endian int32 pairs',
            'vertices': 8, 'edges': 12, 'bone_names': ['Body', 'Hand'],
            'rest_bounds': {'min': [0, 0, 0], 'max': [.2, .3, height], 'height': height, 'height_basis': 'z_span'},
            'pose_samples_measured': 3, 'unchanged_topology_required': True,
        }
        self.report['states'][0].update(deformation_samples_measured=3, deformation_rest_sha256='b'*64)
        self.rig_proof = {'schema_version': 1, 'status': 'validated_local_structure',
                          'loading_verified': True, 'units': 'meters', 'bones': ['Body', 'Hand'], 'sha256': 'd'*64}

    def save(self):
        artifacts = []
        for name, data in (('finish-report.json', self.report), ('sources/finish-design.json', self.design)):
            path = self.root/name
            path.write_text(json.dumps(data))
            artifacts.append({'path': name, 'bytes': path.stat().st_size,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        self.design_hash = artifacts[1]['sha256']
        manifest = {'schema_version': 1, 'status': 'needs_visual_review', 'id': self.design['id'],
                    'source_sha256': SOURCE, 'states': self.design['states'],
                    'finish': self.report, 'artifacts': artifacts}
        if self.rig_proof is not None:
            path = self.root/'rig-verification.json'
            path.write_text(json.dumps(self.rig_proof))
            artifacts.extend([{'path': 'rig-verification.json', 'bytes': path.stat().st_size,
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
                              {'path': 'model.glb', 'bytes': 100, 'sha256': 'd'*64}])
            manifest['rig_verification'] = self.rig_proof
        (self.root/'manifest.json').write_text(json.dumps(manifest))

    def audit(self, **overrides):
        settings = {'expected_source_sha256': SOURCE, 'expected_design_sha256': self.design_hash,
                    'expected_state_ids': ['wave'], 'authored_height': 1.2}
        settings.update(overrides)
        return audit_finish_pack(self.root, **settings)

    def test_severe_spike_rejects_and_preserves_targeted_repair_evidence(self):
        self.save()
        result = self.audit()
        self.assertFalse(result['passed'])
        self.assertEqual(result['status'], 'severe_sampled_deformation')
        self.assertEqual(result['states'][0]['offending_outliers'], [outlier()])
        self.assertEqual(result['bindings']['design_sha256'], self.design_hash)
        self.assertFalse(result['visual_approval'])
        self.assertFalse(result['physics_verified'])

    def test_both_fixed_thresholds_are_required_and_strict(self):
        for row in (outlier(.01, .07), outlier(.001, .03*1.2), outlier(.1, .04)):
            with self.subTest(row=row):
                self.report['states'][0]['deformation_outliers'] = [row]
                self.save()
                self.assertTrue(self.audit()['passed'])
        self.report['states'][0]['deformation_outliers'] = [outlier(.001, .03601)]
        self.save()
        self.assertFalse(self.audit()['passed'])

    def test_empty_diagnostics_are_only_no_reported_spike_not_approval(self):
        self.report['states'][0]['deformation_outliers'] = []
        self.save()
        result = self.audit()
        self.assertEqual(result['status'], 'no_reported_severe_spike')
        self.assertIn('not exhaustive', result['scope'])
        self.assertFalse(result['visual_approval'])

    def test_source_design_state_and_height_bindings_reject(self):
        self.save()
        for override in ({'expected_source_sha256': 'b'*64}, {'expected_design_sha256': 'c'*64},
                         {'expected_state_ids': ['other']}, {'authored_height': 1.3},
                         {'authored_height': True}, {'expected_state_ids': ['wave', 'wave']},
                         {'expected_state_ids': ['s'+str(i) for i in range(9)]}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.audit(**override)

    def test_changed_report_bytes_and_duplicate_manifest_rows_reject(self):
        self.save()
        path = self.root/'finish-report.json'
        path.write_text(path.read_text()+' ')
        with self.assertRaises(ValueError):
            self.audit()
        self.save()
        path = self.root/'manifest.json'
        data = json.loads(path.read_text())
        data['artifacts'].append(data['artifacts'][0])
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            self.audit()

    def test_missing_or_oversized_outlier_evidence_rejects(self):
        for rows in (None, {}, [outlier()]*5):
            with self.subTest(rows=rows):
                self.report['states'][0]['deformation_outliers'] = rows
                self.save()
                with self.assertRaises(ValueError):
                    self.audit()

    def test_malformed_coordinates_weights_indices_and_lengths_reject(self):
        edits = [
            {'endpoint_indices': [False, 1]}, {'endpoint_indices': [0, 100000]},
            {'endpoint_indices': [1, 1]}, {'endpoint_influences': [{'Unknown': 1}, {'Hand': 1}]},
            {'endpoint_influences': [{'Body': .5}, {'Hand': 1}]},
            {'endpoint_influences': [{'Body': True}, {'Hand': 1}]},
            {'rest_normalized_bounds': [[0, 0, 2], [0, 0, 0]]},
            {'rest_normalized_bounds': [[0, 0, float('nan')], [0, 0, 0]]},
            {'second': 1}, {'stretch_ratio': float('inf')}, {'rest_length': 0},
            {'excess_length': .01}, {'stretch_ratio': 3}, {'unsafe_override': 0},
        ]
        for changes in edits:
            with self.subTest(changes=changes):
                row = outlier(); row.update(changes)
                self.report['states'][0]['deformation_outliers'] = [row]
                self.save()
                with self.assertRaises(ValueError):
                    self.audit()

    def test_sampler_binding_and_duplicate_edge_evidence_reject(self):
        original = deepcopy(self.report)
        for key, value in (('seconds', [0, 1, 4]), ('frames', [0, 48, True]),
                           ('sampling_method', 'legacy_uniform_five'),
                           ('deformation_outliers', [outlier(), outlier()])):
            self.report = deepcopy(original)
            self.report['states'][0][key] = value
            self.save()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit()

    def test_source_skin_requires_native_measurement(self):
        self.design.pop('authored_rig')
        self.report['authored_rig'] = None
        self.save()
        with self.assertRaisesRegex(ValueError, 'native deformation measurement'):
            self.audit()

    def test_source_skin_uses_measured_height_not_plan_height(self):
        self.native_measurement(source_skin=True, height=.2)
        self.report['states'][0]['deformation_outliers'] = [outlier(.001, .02)]
        self.save()
        result = self.audit(authored_height=1.2)
        self.assertFalse(result['passed'])
        self.assertEqual(result['thresholds']['excess_m_strictly_greater_than'], .006)
        self.assertEqual(result['thresholds']['height_source'], 'native_measured_rest')
        self.assertIsNone(result['bindings']['authored_height_m'])
        self.assertEqual(result['bindings']['requested_height_m'], 1.2)
        self.assertEqual(result['bindings']['exported_model_sha256'], 'd'*64)
        self.report['deformation_measurement']['rest_bounds'].update(max=[.2, .3, 5], height=5)
        self.save()
        result = self.audit()
        self.assertTrue(result['passed'])
        self.assertEqual(result['thresholds']['height_m'], 5)
        self.assertFalse(result['visual_approval'])

    def test_measured_empty_outliers_are_bound_to_actual_samples(self):
        self.native_measurement(source_skin=True)
        self.report['states'][0]['deformation_outliers'] = []
        self.save()
        result = self.audit()
        self.assertTrue(result['passed'])
        self.assertEqual(result['measurement_scope'], 'bound_native_rest_and_pose_samples')
        self.assertEqual(result['measurement']['pose_samples_measured'], 3)

    def test_flat_source_uses_maximum_span_only_when_z_is_degenerate(self):
        self.native_measurement(source_skin=True)
        bounds = self.report['deformation_measurement']['rest_bounds']
        bounds.update(max=[.2, .3, 0], height=.3, height_basis='maximum_span')
        self.save()
        self.assertEqual(self.audit()['thresholds']['height_m'], .3)
        bounds['height_basis'] = 'z_span'
        self.save()
        with self.assertRaisesRegex(ValueError, 'height disagrees'):
            self.audit()

    def test_actual_exported_bones_and_model_hash_must_match_measurement(self):
        self.native_measurement(source_skin=True)
        original = deepcopy(self.rig_proof)
        for field, value in (('bones', ['Body', 'Other']), ('sha256', 'e'*64), ('loading_verified', False)):
            self.rig_proof = deepcopy(original)
            self.rig_proof[field] = value
            self.save()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit()
        self.rig_proof = None
        self.save()
        with self.assertRaises(ValueError):
            self.audit()

    def test_invalid_native_marker_and_pose_bindings_fail_closed(self):
        self.native_measurement(source_skin=True)
        baseline = deepcopy(self.report)
        changes = [('schema_version', True), ('rig_route', 'authored'), ('units', 'centimeters'),
                   ('coordinate_space', 'unapplied_world'), ('vertices', True), ('vertices', 100001),
                   ('vertices', 1), ('edges', 300001), ('bone_names', ['Body', 'Body']),
                   ('bone_names', ['Body']*65), ('rest_sha256', 'wrong'),
                   ('pose_samples_measured', 2), ('unchanged_topology_required', False),
                   ('rest_array_encoding', 'float32'), ('extra', 'override')]
        for field, value in changes:
            self.report = deepcopy(baseline)
            self.report['deformation_measurement'][field] = value
            self.save()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit()
        for field, value in (('deformation_samples_measured', True), ('deformation_samples_measured', 2),
                             ('deformation_rest_sha256', 'e'*64)):
            self.report = deepcopy(baseline)
            self.report['states'][0][field] = value
            self.save()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit()

    def test_inconsistent_or_nonfinite_measured_bounds_reject(self):
        self.native_measurement(source_skin=True)
        baseline = deepcopy(self.report)
        for field, value in (('height', 3), ('height', float('nan')), ('min', [0, 0, 2]),
                             ('max', [0, 0, 0]), ('max', [0, True, 1.2]), ('height_basis', 'planned_height')):
            self.report = deepcopy(baseline)
            self.report['deformation_measurement']['rest_bounds'][field] = value
            self.save()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit()

    def test_authored_marker_cannot_change_height_or_downgrade_to_legacy(self):
        self.native_measurement()
        self.save()
        self.assertEqual(self.audit()['measurement_scope'], 'bound_native_rest_and_pose_samples')
        self.report['deformation_measurement']['rest_bounds'].update(max=[.2, .3, 2], height=2)
        self.save()
        with self.assertRaisesRegex(ValueError, 'authored bones or height'):
            self.audit()
        self.report.pop('deformation_measurement')
        self.save()
        with self.assertRaisesRegex(ValueError, 'marker is missing'):
            self.audit()

    def test_oversized_input_fails_before_parsing(self):
        self.save()
        (self.root/'finish-report.json').write_bytes(b' '*((1024*1024)+1))
        with self.assertRaisesRegex(ValueError, 'Input too large'):
            self.audit()


if __name__ == '__main__':
    unittest.main()
