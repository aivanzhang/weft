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

    def test_missing_authored_diagnostics_fail_explicitly(self):
        self.design.pop('authored_rig')
        self.report['authored_rig'] = None
        self.save()
        with self.assertRaisesRegex(ValueError, 'pre-rigged diagnostics are unsupported'):
            self.audit()

    def test_oversized_input_fails_before_parsing(self):
        self.save()
        (self.root/'finish-report.json').write_bytes(b' '*((1024*1024)+1))
        with self.assertRaisesRegex(ValueError, 'Input too large'):
            self.audit()


if __name__ == '__main__':
    unittest.main()
