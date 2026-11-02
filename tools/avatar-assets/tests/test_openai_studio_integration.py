"""Analytic integration fixtures; no model, provider, network, browser or server calls."""

import hashlib
import json
from pathlib import Path
import sqlite3
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from avatar_assets import pipeline, studio
from avatar_assets.agent_backend import DEFAULT, pipeline_limits, readiness, server_settings, validate_settings
from avatar_assets.openai_runner import OpenAILedger, reserve_microdollars
from avatar_assets.studio_jobs import StudioJobs
from avatar_assets.studio_worker import run_child
from test_studio_jobs import FixtureExecutor


APPROVAL = {'decision': 'approved', 'user_reply': 'Analytic fixture approval only; no real spending.'}


class OpenAIStudioIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name).resolve()/'project'
        self.root = self.project/'tools/avatar-assets'
        self.root.mkdir(parents=True)
        for name, source in pipeline._pipeline_sources().items():
            target = self.root/('scripts' if name.startswith('scripts/') else 'avatar_assets')/Path(name).name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
        self.patch = patch.object(pipeline, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.settings = {**DEFAULT, 'max_calls': 3, 'max_repairs': 0}
        self.out = self.root/'outputs/api-fixture'
        self.no_network = patch('avatar_assets.openai_runner.post_response', side_effect=AssertionError('No real API calls'))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)
        self.no_cli = patch('avatar_assets.agent_runner.AgentRunner', side_effect=AssertionError('No CLI fallback'))
        self.no_cli.start()
        self.addCleanup(self.no_cli.stop)

    def config(self, value=None):
        path = self.root/'.local/openai-backend.json'
        path.parent.mkdir(mode=0o700, exist_ok=True)
        path.write_text(json.dumps(value or self.settings))
        path.chmod(0o600)
        return path

    def key(self):
        path = self.project/'.local/credentials/openai-api-key'
        path.parent.mkdir(mode=0o700, parents=True)
        path.write_text('analytic-fixture-not-an-api-key')
        path.chmod(0o600)
        return path

    def prepare(self):
        return pipeline.prepare('A copper telescope that scans and nods.', self.out,
                                agent_backend=self.settings, **pipeline_limits(self.settings))

    def grant(self, out=None):
        out = out or self.out
        ledger = OpenAILedger(self.root/'.local/openai')
        self.addCleanup(ledger.close)
        total = 3 * reserve_microdollars(self.settings['max_output_tokens'])
        ledger.configure(total, APPROVAL)
        ledger.authorize_job(out.name, hashlib.sha256((out/'request.json').read_bytes()).hexdigest(), 3, total, APPROVAL)
        return ledger

    def test_missing_configuration_is_readonly_and_saving_is_free(self):
        with patch('avatar_assets.openai_runner.read_api_key', side_effect=AssertionError('Do not read key')):
            self.assertEqual(server_settings(self.root), DEFAULT)
            ready = readiness(self.root)
            self.assertFalse(ready['can_run'])
            self.assertFalse(ready['configured'])
            self.assertFalse((self.root/'.local').exists())
            jobs = StudioJobs(self.root, executor=FixtureExecutor())
            saved = jobs.create('A soft panda', custom_actions=['Dance and bow'])
            self.assertEqual(saved['status'], 'created')
            self.assertFalse(jobs.resolve_job(saved['id']).exists())
            self.assertFalse((self.root/'.local/openai').exists())

    def test_owner_settings_forbid_client_secrets_urls_and_extra_repairs(self):
        for value in ({**DEFAULT, 'api_key': 'never'}, {**DEFAULT, 'base_url': 'https://example.com'},
                      {**DEFAULT, 'provider': 'legacy'}, {**DEFAULT, 'max_calls': True},
                      {**DEFAULT, 'max_calls': 3}, {**DEFAULT, 'timeout': 999},
                      {**DEFAULT, 'max_output_tokens': 10**9}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_settings(value)
        self.assertEqual(validate_settings(self.settings)['max_calls'], 3)

    def test_configuration_modes_and_symlinks_fail_closed(self):
        path = self.config()
        path.chmod(0o644)
        with self.assertRaises(ValueError):
            server_settings(self.root)
        self.assertFalse(readiness(self.root)['can_run'])
        path.chmod(0o600)
        other = path.with_suffix('.saved')
        path.rename(other)
        path.symlink_to(other)
        with self.assertRaises(ValueError):
            server_settings(self.root)

    def test_prepare_pins_api_settings_sources_and_separate_dollar_estimate(self):
        result = self.prepare()
        request = json.loads((self.out/'request.json').read_text())
        self.assertEqual(request['agent_backend'], self.settings)
        self.assertEqual(request['max_agent_calls'], 3)
        self.assertEqual(request['max_repairs'], 0)
        self.assertEqual(result['estimate']['new_meshy_credits'], 35)
        self.assertEqual(result['estimate']['maximum_reservation_microdollars'], 16855920)
        self.assertTrue((self.out/'sources/openai_runner.py').is_file())
        self.assertTrue((self.out/'sources/agent_backend.py').is_file())
        self.assertFalse((self.root/'.local/openai').exists())
        with self.assertRaises(ValueError):
            pipeline.Pipeline(self.out)

    def test_wrong_digest_and_key_never_allow_runner_or_cli(self):
        self.config()
        self.prepare()
        self.grant()
        digest = hashlib.sha256((self.out/'request.json').read_bytes()).hexdigest()
        self.assertFalse(readiness(self.root, job_id=self.out.name, job_digest=digest)['can_run'])
        self.key()
        self.assertFalse(readiness(self.root, job_id=self.out.name, job_digest='f'*64)['can_run'])
        with patch('avatar_assets.openai_runner.read_api_key', side_effect=AssertionError('Readiness must not read key')):
            self.assertTrue(readiness(self.root, job_id=self.out.name, job_digest=digest)['can_run'])
        self.config({**self.settings, 'timeout': 300})
        with self.assertRaises(ValueError):
            pipeline.Pipeline(self.out)

    def test_exact_grant_selects_openai_runner_with_no_cli_fallback(self):
        self.config()
        self.key()
        self.prepare()
        ledger = self.grant()
        with patch('avatar_assets.openai_runner.OpenAIRunner') as runner, \
             patch('avatar_assets.openai_runner.OpenAILedger', return_value=ledger):
            coordinator = pipeline.Pipeline(self.out)
        self.assertIs(coordinator.agents, runner.return_value)
        self.assertEqual(runner.call_args.kwargs['job_id'], self.out.name)
        self.assertEqual(runner.call_args.kwargs['job_digest'], hashlib.sha256((self.out/'request.json').read_bytes()).hexdigest())
        self.assertEqual(runner.call_args.kwargs['max_calls'], 3)
        self.assertEqual(runner.call_args.kwargs['max_output_tokens'], 16384)

    def test_worker_prepares_then_waits_for_explicit_request_grant(self):
        self.config()
        self.key()
        executor = FixtureExecutor()
        jobs = StudioJobs(self.root, executor=executor)
        job = jobs.create('A soft panda', custom_actions=['Dance and bow'])
        jobs.plan(job['id'])
        invocation = executor.calls[-1]
        with jobs._db() as db:
            row = jobs._row(db, job['id'])
            data = json.loads(row['data'])
        jobs._worker_claim(job['id'], invocation['token'], data['owner_pid'], data['owner_identity'])
        with patch.object(pipeline, 'Pipeline') as constructor:
            blocked = run_child(jobs, job['id'], invocation['token'])
        constructor.assert_not_called()
        self.assertEqual(blocked['status'], 'awaiting_authorization')
        out = jobs.resolve_job(job['id'])
        saved_bytes = (out/'request.json').read_bytes()
        self.assertEqual(blocked['pipeline_request_sha256'], hashlib.sha256(saved_bytes).hexdigest())
        self.assertTrue((jobs.jobs_dir/job['id']/'pipeline-request.json').is_file())
        self.grant(out)
        fake = SimpleNamespace(request=json.loads(saved_bytes), run=Mock(return_value={'status': 'planned'}))
        with patch.object(pipeline, 'Pipeline', return_value=fake):
            result = run_child(jobs, job['id'], invocation['token'])
        self.assertEqual(result['status'], 'planned')
        self.assertEqual((out/'request.json').read_bytes(), saved_bytes)
        fake.run.assert_called_once()
        self.assertTrue(fake.run.call_args.kwargs['plan_only'])

    def test_worker_rejects_unfinished_legacy_job_without_rewriting(self):
        self.config()
        executor = FixtureExecutor()
        jobs = StudioJobs(self.root, executor=executor)
        job = jobs.create('A soft panda')
        jobs.plan(job['id'])
        invocation = executor.calls[-1]
        with jobs._db() as db:
            data = json.loads(jobs._row(db, job['id'])['data'])
        jobs._worker_claim(job['id'], invocation['token'], data['owner_pid'], data['owner_identity'])
        out = jobs.resolve_job(job['id'])
        pipeline.prepare('A soft panda', out, max_agent_calls=5, max_credits=35)
        original = (out/'request.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'Studio contract'):
            run_child(jobs, job['id'], invocation['token'])
        self.assertEqual((out/'request.json').read_bytes(), original)

    def test_exhausted_or_too_small_grants_do_not_report_ready(self):
        self.config()
        self.key()
        self.prepare()
        ledger = self.grant()
        digest = hashlib.sha256((self.out/'request.json').read_bytes()).hexdigest()
        reservation = reserve_microdollars(self.settings['max_output_tokens'])
        for number in range(3):
            self.assertTrue(readiness(self.root, job_id=self.out.name, job_digest=digest)['can_run'])
            ledger.reserve(self.out.name, digest, 3, 'fixture-'+str(number), 'a'*64, reservation)
        ready = readiness(self.root, job_id=self.out.name, job_digest=digest)
        self.assertFalse(ready['can_run'])
        self.assertEqual(ready['remaining_calls'], 0)
        self.assertEqual(ready['job_remaining_microdollars'], 0)
        self.assertEqual(ready['project_remaining_microdollars'], 0)
        self.assertIn('cannot fund', ready['reason'])

    def test_dollar_headroom_blocks_even_with_available_call_count(self):
        self.config()
        self.key()
        self.prepare()
        ledger = OpenAILedger(self.root/'.local/openai')
        self.addCleanup(ledger.close)
        digest = hashlib.sha256((self.out/'request.json').read_bytes()).hexdigest()
        ledger.configure(1000, APPROVAL)
        ledger.authorize_job(self.out.name, digest, 3, 1000, APPROVAL)
        ready = readiness(self.root, job_id=self.out.name, job_digest=digest)
        self.assertFalse(ready['can_run'])
        self.assertEqual(ready['remaining_calls'], 3)
        self.assertIn('cannot fund', ready['reason'])

    def test_sealed_api_result_needs_no_live_key_grant_or_cli(self):
        self.prepare()
        packs = []
        for number in range(5):
            folder = self.out/('analytic-pack-'+str(number))
            folder.mkdir()
            (folder/'manifest.json').write_text('{"provenance":"analytic_fixture"}')
            packs.append({'path': folder.name, 'manifest_sha256': hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()})
        delivery = {'status': 'complete', 'packs': packs,
                    'request_sha256': hashlib.sha256((self.out/'request.json').read_bytes()).hexdigest()}
        (self.out/'delivery.json').write_text(json.dumps(delivery))
        (self.out/'delivery-receipt.json').write_text(json.dumps({'delivery.json': hashlib.sha256((self.out/'delivery.json').read_bytes()).hexdigest()}))
        with patch('avatar_assets.openai_runner.OpenAIRunner', side_effect=AssertionError('No live runner')), \
             patch('avatar_assets.cli.verify_pack') as verify:
            coordinator = pipeline.Pipeline(self.out)
            result = coordinator.run()
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['resumed_without_generation'])
        self.assertEqual(verify.call_count, 5)
        self.assertFalse((self.root/'.local').exists())

    def accounting_fixture(self, *, retirements=False):
        self.config()
        self.key()
        folder = self.root/'.local/openai'
        folder.mkdir(mode=0o700)
        path = folder/'ledger.sqlite3'
        db = sqlite3.connect(path)
        self.addCleanup(db.close)
        path.chmod(0o600)
        db.executescript("""
            CREATE TABLE budget(id INTEGER PRIMARY KEY, ceiling INTEGER, approval TEXT);
            CREATE TABLE jobs(id TEXT PRIMARY KEY, digest TEXT, calls INTEGER, ceiling INTEGER, approval TEXT);
            CREATE TABLE calls(job TEXT, role TEXT, request_digest TEXT, reserved INTEGER,
                state TEXT, response_id TEXT, actual INTEGER, usage TEXT, PRIMARY KEY(job,role));
        """)
        if retirements:
            db.execute('CREATE TABLE retirements(parent TEXT PRIMARY KEY,successor TEXT UNIQUE,actual INTEGER,approval TEXT)')
        db.execute('INSERT INTO budget VALUES(1,17000000,?)', ('analytic_fixture',))
        db.execute('INSERT INTO jobs VALUES(?,?,?,?,?)', (self.out.name, 'a'*64, 3, 16855920, 'analytic_fixture'))
        db.commit()
        return db

    def test_unmigrated_ledger_uses_validated_settled_cost_without_rewriting_reservations(self):
        db = self.accounting_fixture()
        reservation = reserve_microdollars()
        db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?,?,?)',
                   (self.out.name, 'planner', 'b'*64, reservation, 'settled', 'resp_fixture', 1000,
                    json.dumps({'input_tokens':400,'output_tokens':0,'total_tokens':400})))
        db.commit()
        with patch('avatar_assets.openai_runner.OpenAILedger', side_effect=AssertionError('No migration')):
            ready = readiness(self.root, job_id=self.out.name, job_digest='a'*64)
        self.assertTrue(ready['can_run'])
        self.assertFalse(ready['job_retired'])
        self.assertEqual(ready['project_remaining_microdollars'], 17000000-1000)
        self.assertEqual(ready['job_remaining_microdollars'], 16855920-1000)
        self.assertEqual(ready['reserved_microdollars'],reservation)
        self.assertEqual(ready['effective_exposure_microdollars'],1000)
        self.assertEqual(db.execute('SELECT reserved FROM calls').fetchone()[0],reservation)
        self.assertIsNone(db.execute("SELECT 1 FROM sqlite_master WHERE name='retirements'").fetchone())

    def test_retired_parent_cost_and_active_full_holds_match_global_remaining(self):
        db = self.accounting_fixture(retirements=True)
        reservation = reserve_microdollars()
        db.execute('INSERT INTO jobs VALUES(?,?,?,?,?)', ('retired-parent', 'b'*64, 3, 16855920, 'analytic_fixture'))
        for job, state, actual in [('retired-parent', 'settled', 1000), (self.out.name, 'reserved_uncertain', None)]:
            db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?,?,?)',
                       (job, 'planner', 'c'*64, reservation, state, 'resp_fixture' if actual else None, actual,
                        json.dumps({'input_tokens':400,'output_tokens':0,'total_tokens':400}) if actual else None))
        db.execute('INSERT INTO retirements VALUES(?,?,?,?)', ('retired-parent', self.out.name, 1000, 'analytic_fixture'))
        db.commit()
        ready = readiness(self.root, job_id=self.out.name, job_digest='a'*64)
        self.assertTrue(ready['can_run'])
        self.assertEqual(ready['project_remaining_microdollars'], 17000000-1000-reservation)
        self.assertEqual(ready['job_remaining_microdollars'], 16855920-reservation)
        parent = readiness(self.root, job_id='retired-parent', job_digest='b'*64)
        self.assertFalse(parent['can_run'])
        self.assertTrue(parent['job_retired'])
        self.assertIn('retired', parent['reason'])
        self.assertEqual(parent['remaining_calls'], 2)
        db.execute("UPDATE calls SET state='settled',actual=500,response_id='resp_fixture',usage=? WHERE job=?",
                   (json.dumps({'input_tokens':200,'output_tokens':0,'total_tokens':200}),self.out.name))
        db.commit()
        settled=readiness(self.root,job_id=self.out.name,job_digest='a'*64)
        self.assertEqual(settled['project_remaining_microdollars'],17000000-1000-500)
        self.assertEqual(settled['reserved_microdollars'],reservation)
        self.assertEqual(settled['retired_actual_microdollars'],1000)
        self.assertEqual(settled['effective_exposure_microdollars'],500)

    def test_malformed_settled_usage_fails_closed_in_readonly_readiness(self):
        db=self.accounting_fixture()
        db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?,?,?)',
                   (self.out.name,'planner','b'*64,reserve_microdollars(),'settled','resp_fixture',1,'{}'))
        db.commit()
        ready=readiness(self.root,job_id=self.out.name,job_digest='a'*64)
        self.assertFalse(ready['can_run'])
        self.assertIn('inspection',ready['reason'])

    def test_malformed_retirement_cost_fails_closed(self):
        db = self.accounting_fixture(retirements=True)
        db.execute('INSERT INTO retirements VALUES(?,?,?,?)', ('retired-parent', self.out.name, 'unknown', 'analytic_fixture'))
        db.commit()
        ready = readiness(self.root, job_id=self.out.name, job_digest='a'*64)
        self.assertFalse(ready['can_run'])
        self.assertIn('inspection', ready['reason'])

    def test_session_reports_safe_api_readiness_without_binding_a_socket(self):
        manager = SimpleNamespace(state_dir=self.root/'.local/studio')
        manager.state_dir.mkdir(parents=True)
        with patch.object(studio.HTTPServer, '__init__', return_value=None), \
             patch('avatar_assets.openai_runner.read_api_key', side_effect=AssertionError('No key read')):
            server = studio.StudioServer(self.root, manager, 0)
        session = server.session
        self.assertEqual(session['agent_backend']['provider'], 'openai')
        self.assertFalse(session['agent_backend']['can_run'])
        self.assertIn('separate', session['planning_cost_note'])
        self.assertNotIn('credentials/', json.dumps(session))
        self.assertNotIn(str(self.project), json.dumps(session))


if __name__ == '__main__':
    unittest.main()
