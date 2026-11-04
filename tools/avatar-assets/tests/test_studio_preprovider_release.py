"""A planning owner that died before provider contact releases on proven folder absence."""

import json
import tempfile
import unittest
from pathlib import Path

from avatar_assets.studio_jobs import StudioJobs
from tests.test_studio_jobs import FixtureExecutor, write


class PreProviderReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'project/tools/avatar-assets'
        self.root.mkdir(parents=True)
        self.executor = FixtureExecutor()
        self.jobs = StudioJobs(self.root, executor=self.executor)
        self.job = self.jobs.create('A fixture axolotl explorer')['id']
        self.jobs.plan(self.job)
        with self.jobs._db() as db:
            row = self.jobs._row(db, self.job)
            self.data = json.loads(row['data'])
        self.token = self.data['token']
        self.owner_pid = self.data['owner_pid']
        self.identity = self.data['owner_identity']
        self.jobs._worker_claim(self.job, self.token, self.owner_pid, self.identity)
        self.jobs._worker_complete(self.job, self.token, 'needs_inspection', safe_to_release=False,
                                   error='Pipeline backend changed; preserve this job and prepare a new one')
        self.output = self.jobs.resolve_job(self.job)
        # The real plan run prepares the output folder before its backend gate refuses.
        write(Path(self.output) / 'request.json', {'schema_version': 1, 'prompt': 'A fixture axolotl explorer'})
        self.invocation = self.jobs.state_dir / self.data['invocation_dir']
        child = {'pid': 90001, 'status': 'finished', 'exit_code': 1, 'process_exit_verified': True,
                 'manual_stop_required': False, 'monitor_checkpoint': 'worker-monitor.json'}
        write(self.invocation / 'owner.json',
              {'pid': self.owner_pid, 'start_identity': self.identity, 'token': self.token})
        write(self.invocation / 'worker-job.json', child)
        write(self.invocation / 'worker-monitor.json', child)
        write(self.invocation / 'worker-handle.json',
              {'pid': 90001, 'monitor_checkpoint': 'worker-monitor.json'})
        for name in ('result.json', 'supervisor-result.json'):
            write(self.invocation / name,
                  {'status': 'needs_inspection', 'safe_to_release': False, 'provenance': 'analytic_fixture'})
        # Dispatch machinery guarantees these two; the pipeline was never constructed, so no pipeline.lock.
        for lock in (self.jobs.state_dir / 'worker.lock', self.jobs.jobs_dir / self.job / 'worker.lock'):
            lock.touch()

    def release(self):
        return self.jobs.release_inspected(self.job, expected_pid=self.owner_pid,
                                           expected_identity=self.identity,
                                           reason='Backend changed during planning; reviewed custody.')

    def test_planning_owner_without_provider_folders_releases_exact_lease(self):
        output = self.root / 'outputs' / Path(self.output).name
        self.assertFalse((output / 'generation').exists())
        self.assertFalse((output / 'agents').exists())
        result = self.release()
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['lease_released'])
        with self.jobs._db() as db:
            self.assertIsNone(db.execute('SELECT job FROM active WHERE id=1').fetchone())
            row = self.jobs._row(db, self.job)
            self.assertEqual(row['status'], 'failed')
        receipt = json.loads((self.jobs.state_dir / result['receipt']).read_text())
        proof = receipt['proof']
        self.assertEqual(proof['no_provider_contact'],
                         {'generation_absent': True, 'agents_absent': True, 'operation': 'plan',
                          'scope': 'Planning exited before creating generation or agent folders; '
                                   'no remote or paid custody exists to settle.'})
        self.assertEqual(proof['previous_planning_invocations'], 0)
        self.assertNotIn('generation', proof)

    def test_paid_planning_without_generation_releases_with_ledger_scope(self):
        output = self.root / 'outputs' / Path(self.output).name
        write(output / 'agents/runner.json', {'execution_kind': 'openai_responses_api'})
        write(output / 'agents/planner/receipt.json', {'state': 'settled'})
        (output / 'agents/runner.lock').touch()
        result = self.release()
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['lease_released'])
        receipt = json.loads((self.jobs.state_dir / result['receipt']).read_text())
        proof = receipt['proof']
        self.assertNotIn('no_provider_contact', proof)
        self.assertNotIn('generation', proof)
        self.assertEqual(proof['planning_only']['generation_absent'], True)
        self.assertEqual(proof['planning_only']['agents_inventory'], ['planner', 'runner.json', 'runner.lock'])
        self.assertTrue(any(lock.endswith('agents/runner.lock') for lock in proof['locks']))

    def test_generation_folder_presence_still_requires_full_custody(self):
        output = self.root / 'outputs' / Path(self.output).name
        (output / 'generation').mkdir()
        with self.assertRaisesRegex(ValueError, 'custody lock is missing|generation'):
            self.release()

    def test_reuse_planning_owner_without_provider_folders_releases(self):
        # Free the setUp job's lease first so the reuse job can dispatch and claim.
        self.release()
        source = self.root / 'outputs/fixture.glb'
        source.write_bytes(b'ANALYTIC FIXTURE; NOT A GENERATED MODEL')
        job = self.jobs.create('A fixture reuse re-author', reuse_model=source)['id']
        self.jobs.plan(job)
        with self.jobs._db() as db:
            data = json.loads(self.jobs._row(db, job)['data'])
        self.jobs._worker_claim(job, data['token'], data['owner_pid'], data['owner_identity'])
        self.jobs._worker_complete(job, data['token'], 'needs_inspection', safe_to_release=False,
                                   error='Prepared pipeline request differs from the immutable Studio request')
        invocation = self.jobs.state_dir / data['invocation_dir']
        child = {'pid': 90002, 'status': 'finished', 'exit_code': 1, 'process_exit_verified': True,
                 'manual_stop_required': False, 'monitor_checkpoint': 'worker-monitor.json'}
        write(invocation / 'owner.json',
              {'pid': data['owner_pid'], 'start_identity': data['owner_identity'], 'token': data['token']})
        write(invocation / 'worker-job.json', child)
        write(invocation / 'worker-monitor.json', child)
        write(invocation / 'worker-handle.json', {'pid': 90002, 'monitor_checkpoint': 'worker-monitor.json'})
        for name in ('result.json', 'supervisor-result.json'):
            write(invocation / name,
                  {'status': 'needs_inspection', 'safe_to_release': False, 'provenance': 'analytic_fixture'})
        write(Path(self.jobs.resolve_job(job)) / 'request.json', {'schema_version': 1})
        (self.jobs.jobs_dir / job / 'worker.lock').touch()
        result = self.jobs.release_inspected(job, expected_pid=data['owner_pid'],
                                             expected_identity=data['owner_identity'],
                                             reason='Reuse planning custody; config drift reviewed.')
        self.assertEqual(result['status'], 'failed')
        receipt = json.loads((self.jobs.state_dir / result['receipt']).read_text())
        self.assertIn('no_provider_contact', receipt['proof'])

    def test_queued_work_still_blocks_release(self):
        second = self.jobs.create('Another fixture creature')['id']
        self.jobs.plan(second)
        with self.assertRaisesRegex(ValueError, 'Queued work must be resolved separately'):
            self.release()


if __name__ == '__main__':
    unittest.main()
