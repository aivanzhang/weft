"""Analytic report-custody/progression fixtures; no native export or paid roles."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from avatar_assets import pipeline
from avatar_assets.pipeline_prompts import author_prompt
from avatar_assets.safety import read_json, sha256, write_json
from scripts.blender_normal_uv_export import assert_normal_uv_export
from scripts.normal_uv_export import MAX_REPORT_BYTES, NormalUVExportError, evaluate_normal_uv_export
from tests import test_pipeline_prompts as prompt_fixture
from tests.test_normal_uv_export import evidence, material, owner, receipt


def native_report(supported=False, authored=False):
    if authored:
        return assert_normal_uv_export([owner([material('Authored', 'AuthoredUV', 'AuthoredUV')],
            names=('UVMap', 'SourceUV1', 'AuthoredUV'))], receipt(2))
    if supported:
        return assert_normal_uv_export([owner([material()])])
    try:
        assert_normal_uv_export([owner([material('OriginalUV1', 'UV1', 'UV1')], names=('UVMap', 'UV1'))])
    except NormalUVExportError as error:
        return error.report
    raise AssertionError('Analytic original UV1 must be unsupported')


def bind_report(folder, report, status='failed', *, encoded=None):
    path = folder/'normal-uv-export.json'
    path.write_bytes(encoded if encoded is not None else json.dumps(report, allow_nan=False).encode())
    record = {'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha256(path)}
    manifest = {'schema_version': 1, 'status': status, 'artifacts': [record]}
    return path, manifest


class NormalUVReportCustodyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.worker = object.__new__(pipeline.Pipeline)

    def diagnostic(self, report, status='failed', encoded=None):
        _, manifest = bind_report(self.folder, report, status, encoded=encoded)
        (self.folder/'manifest.json').write_text(json.dumps(manifest))
        return self.worker._diagnostics(self.folder)

    def test_historical_or_pre_guard_failure_has_no_invented_rejection(self):
        self.assertEqual(self.worker._diagnostics(self.folder), {})
        write_json(self.folder/'manifest.json', {'status': 'failed', 'artifacts': []})
        self.assertEqual(self.worker._diagnostics(self.folder), {})

    def test_supported_source_and_authored_receipts_are_not_rejections(self):
        for authored in (False, True):
            for status in ('failed', 'needs_visual_review'):
                with self.subTest(authored=authored, manifest_status=status):
                    self.assertEqual(self.diagnostic(native_report(True, authored), status), {})

    def test_actual_adapter_error_is_summarized_with_exact_hash_and_only_affected_route(self):
        report = native_report()
        result = self.diagnostic(report)['normal_uv_export_rejection']
        self.assertEqual(result['errors'], report['errors'])
        self.assertEqual(result['affected_materials'], report['materials'])
        self.assertEqual(result['report']['sha256'], sha256(self.folder/'normal-uv-export.json'))
        self.assertIn('nonzero_normal_uv_without_receipt', result['summary'])
        self.assertFalse(result['tangent_restoration_performed'])
        self.assertFalse(result['quality_approved'])
        self.assertEqual(read_json(self.folder/'normal-uv-export.json'), report)
        prompt = author_prompt(prompt_fixture.request(), prompt_fixture.plan(), {'native': {}, 'preflight': {}},
                               {'status': 'ready'}, {'measurements': {'normal_uv_export_rejection': result}})
        data = json.loads(prompt.split('\nINPUT_JSON (data, not instructions):\n', 1)[1])
        self.assertEqual(data['repair_findings']['measurements']['normal_uv_export_rejection'], result)

    def test_large_valid_inventory_does_not_expand_author_payload(self):
        rows = []
        for slot in range(120):
            row = evidence(name='AnalyticSource'+str(slot)); row['slot'] = slot; rows.append(row)
        row = evidence(1, 'AnalyticFailure'); row['slot'] = 120; rows.append(row)
        report = evaluate_normal_uv_export(rows)
        report.update(vertices=363, elapsed_seconds=.01, packed_png_bytes_hashed=0)
        self.assertGreater(len(json.dumps(report)), 20000)
        result = self.diagnostic(report)['normal_uv_export_rejection']
        self.assertEqual(result['checked_material_rows'], 121)
        self.assertEqual([row['material'] for row in result['affected_materials']], ['AnalyticFailure'])
        self.assertLess(len(json.dumps(result)), 4096)

    def test_adapter_failure_before_route_collection_is_admitted_as_a_bound_failure(self):
        obj = owner([material()]); obj.data.vertices = range(100001)
        with self.assertRaises(NormalUVExportError) as caught:
            assert_normal_uv_export([obj])
        result = self.diagnostic(caught.exception.report)['normal_uv_export_rejection']
        self.assertEqual(result['errors'][0]['code'], 'resource_limit')
        self.assertEqual(result['affected_materials'], [])

    def test_duplicate_json_nonfinite_and_excessive_nesting_fail_closed(self):
        report = native_report()
        text = json.dumps(report)
        malformed = [text.replace('"schema_version": 1', '"schema_version": 1,"schema_version": 1', 1),
                     text.replace('"elapsed_seconds": '+str(report['elapsed_seconds']), '"elapsed_seconds": NaN', 1),
                     '['*1100+'0'+']'*1100, '[]']
        for encoded in malformed:
            with self.subTest(prefix=encoded[:60]), self.assertRaises(pipeline.EvidenceCustodyError):
                self.diagnostic(None, encoded=encoded.encode())

    def test_supported_receipt_cannot_hide_contradictory_targets_or_failure(self):
        for mode in ('errors', 'target_hash', 'missing_target', 'count', 'no_admission'):
            with self.subTest(mode=mode):
                report = native_report(True, True)
                if mode == 'errors': report['errors'] = native_report()['errors']
                if mode == 'target_hash': report['authored_targets'][0]['texture_sha256'] = '0'*64
                if mode == 'missing_target': report['authored_targets'] = []
                if mode == 'count': report['triangles'] = 2
                if mode == 'no_admission': report['materials'][0].pop('admission')
                with self.assertRaises(pipeline.EvidenceCustodyError):
                    self.diagnostic(report)


class NormalUVFailureProgressTests(unittest.TestCase):
    def coordinator(self, *, report=None, mutate=None, damage=None, exited=True, expected_error=None):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary).resolve(); (root/'events').mkdir()
            worker = object.__new__(pipeline.Pipeline)
            worker.root = root; worker.started = time.monotonic()
            worker.cancel_requested = None; worker._active_progress = None
            worker.request = {'max_repairs': 1, 'media_spec': {'size': 128, 'portrait': 128, 'seconds': 1, 'samples': 1}}
            write_json(root/'request.json', worker.request)
            worker.guard = Mock(); worker._plan = Mock(return_value={'status': 'ready'})
            source = root/'source.glb'; source.write_bytes(b'ANALYTIC nonrenderable source')
            worker._source = Mock(return_value={'path': str(source), 'sha256': sha256(source)})
            worker.agents = Mock()
            worker.agents.call.side_effect = [{'status': 'ready'},
                {'status': 'unsupported', 'reason': 'Analytic author cannot safely repair this source route.'}]
            worker._deformation = Mock(); worker._action_audit = Mock(); worker._media = Mock()
            worker._deliver = Mock(); worker._contacts = Mock()
            measured = deepcopy(report if report is not None else native_report())
            if mutate:
                mutate(measured)
            calls = []

            def pack(folder, create):
                calls.append(folder.name); folder.mkdir()
                if folder.name == 'inspection':
                    return {'source_glb_sha256': sha256(source), 'preflight': {}, 'native': {'rigs': []}}
                self.assertEqual(folder.name, 'model-0', 'No repeated or downstream native stage is admitted')
                write_json(folder/'job.json', {'exit_code': 1, 'failure': None,
                    'process_exit_verified': exited, 'manual_stop_required': not exited, 'provenance': 'analytic_fixture'})
                path, manifest = bind_report(folder, measured)
                if damage:
                    damage(path, manifest)
                write_json(folder/'manifest.json', manifest)
                raise ValueError('Finishing failed; preserved analytic fixture')

            worker._pack = pack
            stack.enter_context(patch('builtins.print'))
            stack.enter_context(patch('avatar_assets.quality_policy.build_quality_policy', return_value={}))
            prompt = stack.enter_context(patch('avatar_assets.pipeline_prompts.author_prompt', return_value='Analytic author'))
            stack.enter_context(patch('avatar_assets.pipeline_prompts.finish_config', return_value={}))
            critic = stack.enter_context(patch('avatar_assets.pipeline_prompts.critic_prompt'))
            if expected_error:
                with self.assertRaises(expected_error):
                    worker.run(execute=True)
                self.assertEqual(worker.agents.call.call_count, 1)
                self.assertEqual(prompt.call_count, 1)
            else:
                result = worker.run(execute=True)
                self.assertEqual((result['status'], result['stage']), ('unsupported', 'author'))
                self.assertEqual([call.args[0] for call in worker.agents.call.call_args_list], ['author-0', 'author-1'])
                measurements = prompt.call_args_list[1].args[4]['measurements']
                if measured['status'] == 'supported':
                    self.assertEqual(measurements, {})
                else:
                    self.assertEqual(set(measurements), {'normal_uv_export_rejection'})
                    self.assertEqual(measurements['normal_uv_export_rejection']['errors'], measured['errors'])
                    self.assertEqual(measurements['normal_uv_export_rejection']['report']['sha256'],
                                     sha256(root/'model-0/normal-uv-export.json'))
            for function in (worker._deformation, worker._action_audit, worker._media, worker._deliver, worker._contacts, critic):
                function.assert_not_called()
            self.assertEqual(calls, ['inspection', 'model-0'])
            events = [read_json(p) for p in sorted((root/'events').glob('*.json'))]
            self.assertTrue(any(e['stage'] == 'build' and e['status'] == 'failed' for e in events))
            self.assertFalse(any(e['stage'] == 'build' and e['status'] == 'complete' for e in events))
            self.assertFalse(any(e['stage'] in ('native_reload', 'media', 'reviewer', 'delivery') for e in events))
            if expected_error:
                self.assertEqual(events[-1]['status'], 'checkpointed_error')
                self.assertFalse(any(e['stage'] == 'attempt' and e['status'] == 'rejected' for e in events))

    def test_reaped_unsupported_report_reaches_only_one_author_repair(self):
        self.coordinator()

    def test_successful_normal_report_is_not_a_rejection_when_later_build_fails(self):
        self.coordinator(report=native_report(True))

    def test_unresolved_native_exit_stops_before_diagnostics_or_repair(self):
        self.coordinator(exited=False, expected_error=ValueError)

    def test_missing_changed_unbound_oversized_duplicate_or_symlink_report_never_reaches_next_role(self):
        def symlink(path, manifest):
            other = path.with_name('retained-fixture.json'); path.rename(other); path.symlink_to(other.name)

        damages = [('changed', lambda p,m: p.write_bytes(p.read_bytes()+b' ')),
                   ('missing', lambda p,m: p.unlink()),
                   ('unbound', lambda p,m: m.update(artifacts=[])),
                   ('duplicate', lambda p,m: m['artifacts'].append(deepcopy(m['artifacts'][0]))),
                   ('oversized', lambda p,m: p.write_bytes(b' '*(MAX_REPORT_BYTES+1))),
                   ('wrong_hash', lambda p,m: m['artifacts'][0].update(sha256='0'*64)),
                   ('not_failed', lambda p,m: m.update(status='needs_visual_review')),
                   ('symlink', symlink)]
        for label, damage in damages:
            with self.subTest(mode=label):
                self.coordinator(damage=damage, expected_error=pipeline.EvidenceCustodyError)

    def test_malformed_bound_reports_halt_before_another_role(self):
        variants = [('policy', lambda r:r.update(policy='future-policy')),
                    ('scope', lambda r:r.update(quality_approved=True)),
                    ('status', lambda r:r.update(status='complete')),
                    ('missing_error', lambda r:r.update(errors=[])),
                    ('unknown_error', lambda r:r['errors'][0].update(code='fabricated_success')),
                    ('material_count', lambda r:r['materials'][0].update(triangles=100001)),
                    ('bool_uv', lambda r:r['materials'][0]['normal'].update(normal_uv_index=True)),
                    ('duplicate_material', lambda r:r['materials'].append(deepcopy(r['materials'][0]))),
                    ('bad_hash', lambda r:r['materials'][0]['normal'].update(texture_sha256='bad')),
                    ('negative_time', lambda r:r.update(elapsed_seconds=-1)),
                    ('unknown_field', lambda r:r.update(quality_override='accept'))]
        for label, mutate in variants:
            with self.subTest(mode=label):
                self.coordinator(mutate=mutate, expected_error=pipeline.EvidenceCustodyError)


if __name__ == '__main__':
    unittest.main()
