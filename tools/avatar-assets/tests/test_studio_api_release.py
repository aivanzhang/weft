"""Settled API custody fixtures; transport, key reads and OS inventory are mocked."""

import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from avatar_assets import openai_runner as api
from avatar_assets import studio_inspection as inspection
from avatar_assets.studio_jobs import StudioJobs, _digest
from avatar_assets.studio_worker import MEDIA_SPECS
from tests.test_studio_jobs import FixtureExecutor, write


ABSENT = ([(90000, 1, 90000)], 'f' * 64)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StudioApiReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'project/tools/avatar-assets'
        self.root.mkdir(parents=True)
        self.source = self.root / 'outputs/saved-paid-source.glb'
        self.source.parent.mkdir()
        self.source.write_bytes(b'ANALYTIC NON-MODEL SOURCE; NOT A GENERATED AVATAR')
        self.reference = None
        if getattr(self,'fixture_reference',False):
            from PIL import Image
            self.reference=self.root/'outputs/reference.png'
            Image.new('RGB',(8,8),(200,100,20)).save(self.reference)
        self.executor = FixtureExecutor()
        self.jobs = StudioJobs(self.root, executor=self.executor)
        self.job = self.jobs.create('An original dancing character', reuse_model=self.source, reference=self.reference,
                                    custom_actions=['Dance with a visible rhythm'])['id']
        self.jobs.plan(self.job)
        self.claim()
        self.output = self.jobs.resolve_job(self.job)
        self.model = getattr(self,'fixture_model',api.MODEL)
        self.backend = {'provider': 'openai', 'model': self.model, 'max_calls': getattr(self,'fixture_calls',3),
                        'max_repairs': getattr(self,'fixture_repairs',0), 'timeout': 240, 'max_output_tokens': 16384}
        streaming=getattr(self,'fixture_stream',False)
        if streaming:self.backend['transport']='sse-v1'
        with self.jobs._db() as db:
            row = self.jobs._row(db, self.job)
            self.request = json.loads(row['request'])
        copied = self.output / 'inputs/source-model.glb'
        copied.parent.mkdir(parents=True)
        copied.write_bytes(self.source.read_bytes())
        prepared = {**self.request, 'schema_version': 1, 'id': self.output.name,
                    'states': [], 'max_credits': 0, 'max_repairs': self.backend['max_repairs'], 'max_agent_calls': self.backend['max_calls'],
                    'agent_timeout': 240, 'agent_model': self.model, 'agent_backend': self.backend,
                    'media_spec': MEDIA_SPECS['preview'], 'created': '2026-11-02T11:00:00-08:00',
                    'reuse_model': {'path': 'inputs/source-model.glb', 'original_path': str(self.source),
                                    'bytes': copied.stat().st_size, 'sha256': checksum(copied)}}
        if self.reference is not None:
            target=self.output/'inputs/reference.png';target.write_bytes(self.reference.read_bytes())
            prepared['reference']={'path':'inputs/reference.png','original_path':str(self.reference),
                                   'bytes':target.stat().st_size,'sha256':checksum(target)}
        write(self.output / 'request.json', prepared)
        request_hash = checksum(self.output / 'request.json')
        snapshot = self.output / 'sources/openai_runner.py'
        snapshot.parent.mkdir()
        snapshot.write_bytes(Path(api.__file__).read_bytes())
        snapshots=[{'path': 'openai_runner.py', 'bytes': snapshot.stat().st_size, 'sha256': checksum(snapshot)}]
        if self.model != api.MODEL:
            from avatar_assets import openai_models
            target=snapshot.with_name('openai_models.py');target.write_bytes(Path(openai_models.__file__).read_bytes())
            snapshots.append({'path':'openai_models.py','bytes':target.stat().st_size,'sha256':checksum(target)})
        if streaming:
            from avatar_assets import openai_stream
            target=snapshot.with_name('openai_stream.py');target.write_bytes(Path(openai_stream.__file__).read_bytes())
            snapshots.append({'path':'openai_stream.py','bytes':target.stat().st_size,'sha256':checksum(target)})
        write(self.output / 'input-receipt.json', {'request_sha256': request_hash, 'sources': snapshots})
        write(self.jobs.jobs_dir / self.job / 'pipeline-request.json', {'schema_version': 1,
              'studio_request_digest': row['request_digest'], 'pipeline_request_sha256': request_hash})
        write(self.output / 'estimate.json', {'provenance': 'analytic cost fixture'})
        write(self.output / 'events/0000.json', {'stage': 'planner', 'provenance': 'analytic fixture'})
        write(self.output / 'status.json', {'status': 'checkpointed_error', 'automatic_retry': False,
              'error': 'Analytic semantic validator rejected an oversized state ID'})
        self.prior = self.jobs.state_dir / self.data['invocation_dir']
        self.seal(self.prior, self.data, safe=True)
        self.prior_pid = self.data['owner_pid']
        self.jobs._worker_complete(self.job, self.token, 'awaiting_authorization', safe_to_release=True)
        ledger = api.OpenAILedger(self.root / '.local/openai')
        approval = {'decision': 'approved', 'user_reply': 'Analytic allowance only; no real spending'}
        ledger.configure(17000000, approval)
        ledger.authorize_job(self.output.name, request_hash, self.backend['max_calls'], 17000000, approval)
        self.answer = getattr(self,'fixture_answer',{'state_id': 'dance_' + 'a' * 65})
        schema = {'type': 'object', 'properties': {'state_id': {'type': 'string'}},
                  'required': ['state_id'], 'additionalProperties': False}
        schema = getattr(self,'fixture_schema',schema)
        response = {'id': 'resp_analytic_planner', 'model': self.model, 'service_tier': 'default',
                    'status': 'completed', 'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120},
                    'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                                'content': [{'type': 'output_text', 'text': json.dumps(self.answer)}]}]}
        def respond(*args,**kwargs):
            if streaming:
                write(kwargs['trace_path'],{'schema_version':1,'transport':'sse-v1','status':'completed',
                    'client_request_id':kwargs['client_request_id'],'response_id':response['id']})
            return json.dumps(response).encode()
        try:
            with patch.object(api, 'read_api_key', return_value='ANALYTIC NEVER REAL'), \
                    patch('avatar_assets.openai_stream.post_stream_response' if streaming else
                          'avatar_assets.openai_runner.post_response',side_effect=respond) as transport:
                runner = api.OpenAIRunner(self.output / 'agents', job_id=self.output.name,
                    job_digest=request_hash, ledger=ledger, max_calls=self.backend['max_calls'], model=self.model,
                    **({'transport':'sse-v1'} if streaming else {}))
                runner.call('planner', 'Plan this analytic dance; no tool execution', schema,
                            images=[self.output/'inputs/reference.png'] if self.reference is not None else [])
                self.assertEqual(transport.call_count, 1)
        finally:
            ledger.close()
        self.ledger = self.root / '.local/openai/ledger.sqlite3'
        self.jobs.plan(self.job)
        self.claim()
        self.invocation = self.jobs.state_dir / self.data['invocation_dir']
        self.seal(self.invocation, self.data, safe=False)
        self.jobs._worker_complete(self.job, self.token, 'needs_inspection', safe_to_release=False)
        self.owner_pid, self.identity = self.data['owner_pid'], self.data['owner_identity']
        for path in (self.jobs.state_dir / 'worker.lock', self.jobs.jobs_dir / self.job / 'worker.lock',
                     self.output / 'pipeline.lock'):
            path.touch()

    def claim(self):
        with self.jobs._db() as db:
            self.data = json.loads(self.jobs._row(db, self.job)['data'])
        self.token = self.data['token']
        self.jobs._worker_claim(self.job, self.token, self.data['owner_pid'], self.data['owner_identity'])

    @staticmethod
    def seal(folder, data, *, safe):
        child = data['owner_pid'] + 10000
        write(folder / 'owner.json', {'pid': data['owner_pid'], 'start_identity': data['owner_identity'],
                                     'token': data['token']})
        for name in ('worker-job.json', 'worker-monitor.json'):
            write(folder / name, {'pid': child, 'status': 'finished', 'exit_code': 0 if safe else 1,
                                 'process_exit_verified': True, 'manual_stop_required': False})
        write(folder / 'worker-handle.json', {'pid': child, 'monitor_checkpoint': 'worker-monitor.json'})
        for name in ('result.json', 'supervisor-result.json'):
            write(folder / name, {'status': 'awaiting_authorization' if safe else 'needs_inspection',
                                 'safe_to_release': safe})

    def release(self):
        return self.jobs.release_inspected(self.job, expected_pid=self.owner_pid,
            expected_identity=self.identity, reason='Preserve settled API response and semantic failure; no retry.')

    def retained(self):
        with self.jobs._db() as db:
            self.assertEqual(self.jobs._row(db, self.job)['status'], 'needs_inspection')
            self.assertEqual(tuple(db.execute('SELECT job,token FROM active').fetchone()), (self.job, self.token))
        self.assertFalse(list((self.jobs.jobs_dir / self.job).glob('inspection-release-*.json')))
        self.assertEqual(len(self.executor.calls), 2)

    def blocked(self, pattern):
        with patch.object(inspection, '_process_snapshot', return_value=ABSENT), \
                patch.object(api, 'read_api_key', side_effect=AssertionError('No key read')), \
                patch.object(api, 'post_response', side_effect=AssertionError('No API calls')), \
                self.assertRaisesRegex((ValueError, OSError), pattern):
            self.release()
        self.retained()

    def sql(self, statement, arguments=()):
        with sqlite3.connect(self.ledger) as db:
            db.execute(statement, arguments)

    def change(self, path, change):
        row = json.loads(path.read_text())
        change(row)
        write(path, row)

    def test_settled_semantically_invalid_answer_releases_only_as_failed_without_calls_or_reset(self):
        before = {str(p): checksum(p) for folder in (self.output, self.invocation, self.prior)
                  for p in folder.rglob('*') if p.is_file()}
        ledger_hash = checksum(self.ledger)
        with patch.object(inspection, '_process_snapshot', return_value=ABSENT), \
                patch.object(api, 'read_api_key', side_effect=AssertionError('No key read')), \
                patch.object(api, 'post_response', side_effect=AssertionError('No API calls')), \
                patch.object(self.jobs, '_dispatch', side_effect=AssertionError('No implicit dispatch')):
            result = self.release()
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['lease_released'] and result['no_dispatch'] and result['no_cost_reset'])
        self.assertEqual(checksum(self.ledger), ledger_hash)
        self.assertEqual(before, {str(p): checksum(p) for folder in (self.output, self.invocation, self.prior)
                                 for p in folder.rglob('*') if p.is_file()})
        receipt = self.jobs.state_dir / result['receipt']
        raw = receipt.read_text()
        self.assertNotIn(self.token, raw)
        self.assertNotIn('ANALYTIC NEVER REAL', raw)
        proof = json.loads(raw)['proof']
        self.assertEqual(proof['api_planning']['calls'][0]['actual_microdollars'], 550)
        self.assertEqual(proof['api_planning']['calls'][0]['reserved_microdollars'], 5618640)
        self.assertEqual(proof['api_planning']['call_ceiling_preserved'], 3)
        self.assertEqual(proof['previous_planning_invocations'], 1)
        self.assertIn(self.prior_pid, proof['known_pids'])
        self.assertEqual(len(proof['locks']), 4)
        self.assertGreater(len(json.loads((self.output / 'agents/planner/answer.json').read_text())['state_id']), 64)
        self.assertFalse((self.output / 'plan.json').exists())
        self.assertFalse(self.jobs.get(self.job)['capabilities']['plan'])
        with self.jobs._db() as db:
            self.assertFalse(db.execute('SELECT * FROM active').fetchall())

    def test_uncertain_or_absent_api_reservation_cannot_release(self):
        self.sql("UPDATE calls SET state='reserved_uncertain'")
        self.blocked('API calls are uncertain|price binding has no original call')
        self.sql('DELETE FROM calls')
        self.blocked('API calls are uncertain|price binding has no original call')

    def test_failed_receipt_with_settled_charge_remains_outside_scope(self):
        self.change(self.output / 'agents/planner/receipt.json', lambda row: row.update(status='failed_or_uncertain'))
        self.blocked('response receipt')

    def test_exact_allowance_request_and_private_binding_are_required(self):
        self.sql('UPDATE jobs SET digest=?', ('0' * 64,))
        self.blocked('allowance differs')
        self.change(self.jobs.jobs_dir / self.job / 'pipeline-request.json',
                    lambda row: row.update(pipeline_request_sha256='0' * 64))
        self.blocked('private Studio binding')

    def test_response_bytes_are_bound_and_no_missing_answer_is_synthesized(self):
        path = self.output / 'agents/planner/response.json'
        self.change(path, lambda row: row.update(id='resp_changed'))
        self.blocked('artifact hashes changed')
        (self.output / 'agents/planner/answer.json').unlink()
        self.blocked('files are missing')

    def test_usage_and_settled_amount_are_not_accepted_from_only_one_receipt(self):
        self.sql('UPDATE calls SET actual=1')
        self.blocked('settled token cost differs')
        self.sql('UPDATE calls SET actual=550,usage=?', (json.dumps({'input_tokens': 0}),))
        self.blocked('usage differs')

    def test_reservation_is_not_refunded_during_failed_release(self):
        self.sql('UPDATE calls SET reserved=550')
        self.blocked('settled cost or reservation differs|price profile differs from its reservation')

    def test_extra_roles_plans_native_or_generation_work_need_different_inspection(self):
        for name in ('plan.json', 'generation', 'inspection', 'model-0'):
            path = self.output / name
            with self.subTest(name=name):
                path.mkdir() if '.' not in name else write(path, {'status': 'ready'})
                self.blocked('plan, native output|Remote-generation artifacts')
                path.rmdir() if path.is_dir() else path.unlink()
        (self.output / 'agents/author-0').mkdir()
        self.blocked('inventory is unsafe or oversized|Only one API')

    def test_source_snapshot_and_reused_model_are_immutable(self):
        path = self.output / 'sources/openai_runner.py'
        original = path.read_bytes()
        path.write_bytes(original + b'changed')
        self.blocked('source snapshot changed')
        path.write_bytes(original)
        self.source.write_bytes(self.source.read_bytes() + b'changed')
        self.blocked('source bytes or identity changed')

    def test_backend_field_removal_cannot_hide_an_api_runner(self):
        self.change(self.output / 'request.json', lambda row: row.pop('agent_backend'))
        self.blocked('server AI configuration')

    def test_busy_runner_lock_and_live_prior_owner_fail_closed(self):
        with (self.output / 'agents/runner.lock').open('rb') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.blocked('still owns its lock')
        with patch.object(inspection, '_process_snapshot', return_value=([(self.prior_pid, 1, self.prior_pid)], 'e' * 64)), \
                self.assertRaisesRegex(ValueError, 'still present'):
            self.release()
        self.retained()

    def test_ledger_journal_or_hash_race_does_not_release(self):
        journal = Path(str(self.ledger) + '-wal')
        journal.touch()
        self.blocked('active journal')
        journal.unlink()
        real = inspection.collect_api_planning
        def changed(*args):
            proof = real(*args)
            self.sql('UPDATE calls SET actual=1')
            return proof
        with patch.object(inspection, 'collect_api_planning', side_effect=changed):
            self.blocked('evidence changed during inspection')

    def test_symlink_or_oversized_response_is_not_decoded(self):
        path = self.output / 'agents/planner/response.json'
        original = path.read_bytes()
        path.unlink()
        path.symlink_to(self.source)
        self.blocked('symlink|unsafe')
        path.unlink()
        path.write_bytes(b' ' * (4 * 1024**2 + 1))
        self.blocked('oversized')
        path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
