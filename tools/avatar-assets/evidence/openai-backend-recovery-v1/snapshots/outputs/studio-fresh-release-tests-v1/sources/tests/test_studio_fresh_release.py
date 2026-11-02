"""Analytic custody only: no real models, workers, process scans or provider calls."""

import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from avatar_assets import studio_inspection as inspection
from avatar_assets.studio_jobs import StudioJobs, _digest
from tests.test_studio_jobs import FixtureExecutor, write


ABSENT = ([(90000, 1, 90000)], 'f' * 64)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StudioFreshReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'project/tools/avatar-assets'
        self.root.mkdir(parents=True)
        self.executor = FixtureExecutor()
        self.jobs = StudioJobs(self.root, executor=self.executor)
        self.job = self.jobs.create('An original dancing character', style='Short tidy fur',
                                    custom_actions=['Dance visibly without props'])['id']
        self.jobs.plan(self.job)
        self.claim()
        self.output = self.jobs.resolve_job(self.job)
        with self.jobs._db() as db:
            self.request = json.loads(self.jobs._row(db, self.job)['request'])
        write(self.output / 'request.json', {**self.request, 'schema_version': 1,
              'id': self.output.name, 'max_credits': 35, 'max_agent_calls': 5, 'max_repairs': 1})
        write(self.output / 'plan.json', {'subject': self.request['prompt'], 'status': 'ready'})
        self.generation = self.output / 'generation'
        self.prepare_generation()
        self.jobs._worker_complete(self.job, self.token, 'awaiting_authorization', safe_to_release=True)
        self.jobs.start(self.job, self.jobs.get(self.job)['plan_id'])
        self.claim()
        self.jobs._worker_complete(self.job, self.token, 'needs_inspection', safe_to_release=False,
                                   error='Analytic author account-limit failure; no answer or accepted delivery.')
        self.invocation = self.jobs.state_dir / self.data['invocation_dir']
        self.owner_pid, self.identity = self.data['owner_pid'], self.data['owner_identity']
        write(self.invocation / 'owner.json', {'pid': self.owner_pid,
              'start_identity': self.identity, 'token': self.token})
        write(self.invocation / 'worker-job.json', self.exited(20001, exit_code=1))
        for name in ('result.json', 'supervisor-result.json'):
            write(self.invocation / name, {'status': 'needs_inspection', 'safe_to_release': False})
        write(self.output / 'inspection/job.json', self.exited(30001))
        write(self.output / 'agents/author-0/agent-process.json', self.exited(40001, exit_code=1))
        write(self.output / 'agents/usage.json', {'calls': 2, 'status': 'failed_or_uncertain'})
        self.locks = [self.jobs.state_dir / 'worker.lock', self.jobs.jobs_dir / self.job / 'worker.lock',
                      self.output / 'pipeline.lock', self.generation / 'job.lock']
        for path in self.locks:
            path.touch()

    def claim(self):
        with self.jobs._db() as db:
            self.data = json.loads(self.jobs._row(db, self.job)['data'])
        self.token = self.data['token']
        self.jobs._worker_claim(self.job, self.token, self.data['owner_pid'], self.data['owner_identity'])

    @staticmethod
    def exited(pid, **fields):
        return {'pid': pid, 'status': 'finished', 'exit_code': 0, 'process_exit_verified': True,
                'manual_stop_required': False, **fields}

    def prepare_generation(self):
        sources = []
        for name in ('meshy.py', 'meshy_access.py'):
            path = self.generation / 'sources' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('ANALYTIC SNAPSHOT; NEVER EXECUTED: ' + name).encode())
            sources.append({'path': name, 'bytes': path.stat().st_size, 'sha256': checksum(path)})
        self.source_hashes = {row['path']: row['sha256'] for row in sources}
        self.config = {'schema_version': 1, 'id': self.output.name + '-mesh', 'rig': 'none', 'states': [],
                       'prompt': 'Static neutral original character', 'style': self.request['style'],
                       'generation': {'mode': 'text', 'geometry_resolution': '4k', 'target_polycount': 35000,
                                      'pose_mode': '', 'texture_resolution': '4k'}}
        self.plan = {'schema_version': 1, 'id': self.config['id'], 'model': 'meshy-7.1', 'max_credits': 35,
                     'estimated_credits': 35, 'estimated_paid_calls': 2, 'automatic_paid_retries': 0,
                     'config': self.config, 'sources': sources}
        write(self.generation / 'plan.json', self.plan)
        write(self.generation / 'inputs/config.json', self.config)
        write(self.output / 'generation-config.json', self.config)
        write(self.output / 'generation-receipt.json', {'plan_sha256': checksum(self.generation / 'plan.json')})
        write(self.output / 'generation-cost.json', {'estimated_credits': 35, 'paid_submissions': 2,
              'provider_reported_credits': 35, 'reserved_credits': 35, 'usd_cost': None})
        self.ledger = self.root.parents[1] / '.local/meshy/ledger.sqlite3'
        self.ledger.parent.mkdir(parents=True)
        with sqlite3.connect(self.ledger) as db:
            db.executescript('''CREATE TABLE budget(id INTEGER,credit_limit INTEGER);
                CREATE TABLE jobs(id TEXT,output TEXT,plan_hash TEXT,credit_limit INTEGER);
                CREATE TABLE budget_grants(job TEXT,credits INTEGER,plan_hash TEXT,approval TEXT);
                CREATE TABLE calls(job TEXT,step TEXT,credits INTEGER,state TEXT,task_id TEXT,
                                   endpoint TEXT,request TEXT,response TEXT);''')
            db.execute('INSERT INTO budget VALUES(1,210)')
            db.execute('INSERT INTO jobs VALUES(?,?,?,?)',
                       (self.plan['id'], str(self.generation), _digest(self.plan), 35))
            approval = {'decision': 'approved', 'job_id': self.plan['id'], 'plan_digest': _digest(self.plan),
                        'additional_credits': 35, 'user_reply': 'ANALYTIC explicit allowance fixture'}
            db.execute('INSERT INTO budget_grants VALUES(?,?,?,?)',
                       (self.plan['id'], 35, _digest(self.plan), json.dumps(approval)))
            for step, credits in (('mesh', 25), ('texture', 10)):
                self.prepare_stage(db, step, credits)
        self.ledger.chmod(0o600)

    def prepare_stage(self, db, step, credits):
        task = step + '-analytic-task'
        folder = self.generation / 'stages' / step
        body = ({'ai_model': 'meshy-7.1', 'model_type': 'standard', 'geometry_resolution': '4k',
                 'should_remesh': True, 'target_polycount': 35000, 'topology': 'triangle', 'pose_mode': '',
                 'target_formats': ['glb'], 'mode': 'preview', 'prompt': self.config['prompt']}
                if step == 'mesh' else
                {'mode': 'refine', 'ai_model': 'meshy-7.1', 'preview_task_id': 'mesh-analytic-task',
                 'enable_pbr': True, 'texture_resolution': '4k', 'texture_prompt': self.config['style'],
                 'target_formats': ['glb']})
        reserved = {'endpoint': '/openapi/v2/text-to-3d', 'estimated_credits': credits,
                    'body': body, 'request_sha256': _digest(body)}
        response = {'id': task, 'status': 'SUCCEEDED', 'consumed_credits': credits,
                    'model_urls': {'glb': 'https://example.invalid/analytic.glb'},
                    'thumbnail_url': 'https://example.invalid/analytic.png'}
        db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?,?,?)',
                   (self.plan['id'], step, credits, 'SUCCEEDED', task, reserved['endpoint'],
                    json.dumps(reserved), json.dumps(response)))
        for name, value in [('request-000.json', reserved),
                            ('submission-000.json', {'task_id': task, 'estimated_credits': credits}),
                            ('status-000.json', {'task': response})]:
            write(folder / name, {**value, 'execution_sources': self.source_hashes})
        for name, digest in self.source_hashes.items():
            target = folder / 'tools' / (digest[:16] + '-' + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((self.generation / 'sources' / name).read_bytes())
        downloads = folder / 'downloads'
        downloads.mkdir()
        files, artifacts = {}, []
        for name in ('model.glb', 'thumbnail.png'):
            path = downloads / name
            path.write_bytes(('ANALYTIC NON-MEDIA ' + step + ':' + name).encode())
            files[name] = {'bytes': path.stat().st_size, 'sha256': checksum(path), 'elapsed_seconds': 0.01}
            artifacts.append({'path': name, 'bytes': path.stat().st_size, 'sha256': checksum(path)})
        write(folder / 'download-000.json', {'files': files, 'execution_sources': self.source_hashes})
        write(downloads / 'manifest.json', {'schema_version': 1, 'kind': 'meshy_original_downloads',
              'status': 'downloaded_not_visually_accepted', 'task_id': task, 'stage': step,
              'plan_sha256': checksum(self.generation / 'plan.json'), 'downloads': files, 'artifacts': artifacts})

    def release(self):
        return self.jobs.release_inspected(self.job, expected_pid=self.owner_pid,
                expected_identity=self.identity, reason='Preserve paid source and failed author; no replay or refund.')

    def retained(self):
        with self.jobs._db() as db:
            self.assertEqual(self.jobs._row(db, self.job)['status'], 'needs_inspection')
            self.assertEqual(tuple(db.execute('SELECT job,token FROM active').fetchone()), (self.job, self.token))
        self.assertFalse(list((self.jobs.jobs_dir / self.job).glob('inspection-release-*.json')))
        self.assertEqual(len(self.executor.calls), 2)

    def blocked(self, pattern):
        with patch.object(inspection, '_process_snapshot', return_value=ABSENT), \
                self.assertRaisesRegex((ValueError, OSError), pattern):
            self.release()
        self.retained()

    def change_json(self, path, change):
        value = json.loads(path.read_text())
        change(value)
        write(path, value)

    def sql(self, statement, arguments=()):
        with sqlite3.connect(self.ledger) as db:
            db.execute(statement, arguments)

    def test_completed_paid_sources_release_failed_without_mutating_outputs_ledger_or_calls(self):
        before = {str(p): checksum(p) for folder in (self.output, self.invocation)
                  for p in folder.rglob('*') if p.is_file()}
        ledger_hash = checksum(self.ledger)
        with patch.object(inspection, '_process_snapshot', return_value=ABSENT), \
                patch.object(self.jobs, '_dispatch', side_effect=AssertionError('No release dispatch')):
            result = self.release()
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['lease_released'] and result['no_dispatch'] and result['no_cost_reset'])
        self.assertEqual(checksum(self.ledger), ledger_hash)
        self.assertEqual(before, {str(p): checksum(p) for folder in (self.output, self.invocation)
                                 for p in folder.rglob('*') if p.is_file()})
        receipt_path = self.jobs.state_dir / result['receipt']
        self.assertEqual(checksum(receipt_path), result['sha256'])
        raw = receipt_path.read_text()
        self.assertNotIn(self.token, raw)
        self.assertNotIn('command', raw)
        proof = json.loads(raw)['proof']
        self.assertEqual(len(proof['locks']), 4)
        self.assertEqual(proof['generation']['policy'], 'completed-text-generation-custody-v1')
        self.assertEqual(proof['generation']['spent_credits_preserved'], 35)
        self.assertEqual([s['status'] for s in proof['generation']['stages']], ['SUCCEEDED', 'SUCCEEDED'])
        self.assertEqual(proof['generation']['new_provider_calls'], 0)
        self.assertEqual(len(self.executor.calls), 2)
        with self.jobs._db() as db:
            self.assertFalse(db.execute('SELECT * FROM active').fetchall())
        self.assertFalse(self.jobs.get(self.job)['capabilities']['plan'])
        with self.assertRaisesRegex(ValueError, 'Only a retained'):
            self.release()

    def test_uncertain_failed_or_running_remote_tasks_never_release(self):
        for state in ('submitting', 'uncertain', 'PENDING', 'IN_PROGRESS', 'FAILED', 'CANCELED'):
            with self.subTest(state=state):
                self.sql("UPDATE calls SET state=? WHERE step='texture'", (state,))
                self.blocked('not fully successful')

    def test_mismatched_task_or_final_status_is_not_a_success_receipt(self):
        path = self.generation / 'stages/texture/status-000.json'
        self.change_json(path, lambda v: v['task'].update(id='unrelated-task'))
        self.blocked('unrelated task')

    def test_extra_paid_attempt_and_terminal_history_are_rejected(self):
        folder = self.generation / 'stages/mesh'
        write(folder / 'request-001.json', {'not': 'authorized'})
        self.blocked('retries, failures')
        (folder / 'request-001.json').unlink()
        write(folder / 'status-001.json', json.loads((folder / 'status-000.json').read_text()))
        self.blocked('continued after terminal')

    def test_missing_download_and_changed_download_bytes_keep_lease(self):
        path = self.generation / 'stages/texture/downloads/model.glb'
        original = path.read_bytes()
        path.unlink()
        self.blocked('incomplete')
        path.write_bytes(original + b'changed')
        self.blocked('bytes/hash changed')

    def test_download_manifest_is_bound_to_task_plan_and_complete_inventory(self):
        path = self.generation / 'stages/mesh/downloads/manifest.json'
        original = path.read_bytes()
        for field, value in [('task_id', 'other'), ('plan_sha256', '0' * 64), ('stage', 'texture')]:
            with self.subTest(field=field):
                self.change_json(path, lambda row: row.update({field: value}))
                self.blocked('manifest differs')
                path.write_bytes(original)
        self.change_json(path, lambda row: row['artifacts'].pop())
        self.blocked('artifact inventory')

    def test_approved_canonical_plan_and_exact_full_grant_are_required(self):
        for statement, args in [
            ('UPDATE jobs SET plan_hash=?', ('0' * 64,)),
            ('UPDATE budget_grants SET credits=?', (34,)),
            ('DELETE FROM budget_grants', ()),
        ]:
            with self.subTest(statement=statement):
                original = self.ledger.read_bytes()
                self.sql(statement, args)
                self.blocked('plan or exact job grant')
                self.ledger.write_bytes(original)

    def test_canonical_approval_and_charge_are_not_inferred_from_downloads(self):
        original = self.ledger.read_bytes()
        self.sql('UPDATE budget_grants SET approval=?', (json.dumps({'decision': 'unapproved'}),))
        self.blocked('approval is missing')
        self.ledger.write_bytes(original)
        self.sql("UPDATE calls SET credits=9 WHERE step='texture'")
        self.blocked('exact charge')

    def test_submitted_body_must_match_the_approved_generation_plan(self):
        with sqlite3.connect(self.ledger) as db:
            reserved = json.loads(db.execute("SELECT request FROM calls WHERE step='mesh'").fetchone()[0])
        reserved['body']['prompt'] = 'Unrelated substituted subject'
        reserved['request_sha256'] = _digest(reserved['body'])
        self.sql("UPDATE calls SET request=? WHERE step='mesh'", (json.dumps(reserved),))
        path = self.generation / 'stages/mesh/request-000.json'
        write(path, {**reserved, 'execution_sources': self.source_hashes})
        self.blocked('Submitted remote body')

    def test_source_receipts_and_snapshot_bytes_are_bound(self):
        path = self.generation / 'stages/mesh/status-000.json'
        original = path.read_bytes()
        self.change_json(path, lambda row: row['execution_sources'].update({'meshy.py': '0' * 64}))
        self.blocked('implementation differs')
        path.write_bytes(original)
        source = self.generation / 'sources/meshy.py'
        source.write_bytes(source.read_bytes() + b'CHANGED')
        self.blocked('implementation snapshots')

    def test_symlink_extra_files_and_ledger_journal_fail_closed(self):
        extra = self.generation / 'stages/texture/downloads/extra.png'
        extra.symlink_to(self.output / 'request.json')
        self.blocked('symlink|unsafe')
        extra.unlink()
        journal = Path(str(self.ledger) + '-wal')
        journal.touch()
        self.blocked('active journal')
        journal.unlink()
        self.ledger.chmod(0o644)
        self.blocked('owner-only')

    def test_request_cost_or_completed_delivery_cannot_be_reinterpreted(self):
        path = self.output / 'request.json'
        original = path.read_bytes()
        self.change_json(path, lambda row: row.update(custom_actions=['Stand still']))
        self.blocked('pipeline request differs')
        path.write_bytes(original)
        cost = self.output / 'generation-cost.json'
        self.change_json(cost, lambda row: row.update(reserved_credits=0))
        self.blocked('generation cost receipt')
        write(self.output / 'delivery.json', {'status': 'complete'})
        self.blocked('completed delivery')

    def test_remote_evidence_race_blocks_commit_and_preserves_old_owner(self):
        real = inspection.collect_generation
        def changed(*args):
            proof = real(*args)
            path = self.generation / 'stages/texture/downloads/model.glb'
            path.write_bytes(path.read_bytes() + b'CHANGED AFTER INSPECTION')
            return proof
        with patch.object(inspection, 'collect_generation', side_effect=changed):
            self.blocked('changed during inspection')

    def test_live_local_owner_still_blocks_completed_remote_tasks(self):
        with patch.object(inspection, '_process_snapshot', return_value=([(self.owner_pid, 1, self.owner_pid)], 'e' * 64)), \
                self.assertRaisesRegex(ValueError, 'still present'):
            self.release()
        self.retained()

    def test_api_agent_custody_cannot_use_legacy_process_release(self):
        self.change_json(self.output / 'request.json', lambda row: row.update(agent_backend={'provider': 'openai'}))
        self.blocked('API-backed agent custody needs separate inspection')

    def test_new_server_gate_allows_only_explicit_plan_retry_before_any_plan(self):
        other = self.jobs.create('A separate server configuration gate fixture')['id']
        # The existing failed owner's lease prevents a launch until its separate release.
        with patch.object(inspection, '_process_snapshot', return_value=ABSENT):
            self.release()
        self.jobs.plan(other)
        with self.jobs._db() as db:
            data = json.loads(self.jobs._row(db, other)['data'])
        self.jobs._worker_claim(other, data['token'], data['owner_pid'], data['owner_identity'])
        self.jobs._worker_complete(other, data['token'], 'awaiting_authorization', safe_to_release=True,
                                   error='Analytic API configuration gate before planning')
        calls = len(self.executor.calls)
        gated = self.jobs.get(other)
        self.assertIsNone(gated['plan_id'])
        self.assertTrue(gated['capabilities']['plan'])
        self.assertFalse(gated['capabilities']['start'])
        self.assertEqual(len(self.executor.calls), calls)
        self.jobs.plan(other)
        self.assertEqual(len(self.executor.calls), calls + 1)
        self.jobs.plan(other)
        self.assertEqual(len(self.executor.calls), calls + 1)

    def test_new_invocations_pin_server_backend_sources_without_rewriting_prior_receipts(self):
        sources = json.loads((self.invocation / 'invocation.json').read_text())['sources']
        self.assertEqual([row['path'] for row in sources], ['sources/' + name for name in
            ('studio_jobs.py', 'studio_worker.py', 'agent_backend.py', 'openai_runner.py')])
        for row in sources:
            self.assertEqual(checksum(self.invocation / row['path']), row['sha256'])


if __name__ == '__main__':
    unittest.main()
