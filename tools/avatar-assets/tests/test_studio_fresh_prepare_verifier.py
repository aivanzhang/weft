"""Fresh zero-call prepared-request inspection; no grant, key read or dispatch."""
from contextlib import closing
import json
import shutil
import sqlite3
import unittest
from unittest.mock import patch
import test_studio_live_verifier as existing


class FreshPrepareVerifierTests(unittest.TestCase):
    def setUp(self):
        self.fixture=existing.StudioInspectionTests('runTest');self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture;f.api_fixture(reuse=False)
        (f.output/'plan.json').unlink();shutil.rmtree(f.output/'generation')
        for name in ('generation-config.json','generation-receipt.json'):(f.output/name).unlink()
        f.data['plan_id']=None;f.update_row(status='awaiting_authorization',data=json.dumps(f.data))
        self.ledger=f.root/'.local/openai/ledger.sqlite3';self.ledger.parent.mkdir()
        with closing(sqlite3.connect(self.ledger)) as db,db:db.execute('CREATE TABLE calls(job TEXT,role TEXT)')

    def test_fresh_zero_call_admission_returns_no_plan_or_grant(self):
        with self.assertRaises((ValueError,OSError)):existing.verifier.inspect(existing.JOB)
        with patch('avatar_assets.openai_runner.read_api_key',side_effect=AssertionError('No key read')):
            value=existing.verifier.inspect(existing.JOB,allow_unplanned=True)
        self.assertFalse(value['plan_available']);self.assertIsNone(value['plan_id'])
        self.assertEqual(value['generation_mode'],'new_meshy_generation')
        self.assertFalse(value['source_generation']['authorization_checked'])
        self.assertFalse(value['agent_backend']['plan_verified'])

    def test_reserved_call_or_generation_artifact_blocks_preparation_admission(self):
        with closing(sqlite3.connect(self.ledger)) as db,db:db.execute('INSERT INTO calls VALUES(?,?)',(self.fixture.output.name,'planner'))
        with self.assertRaisesRegex(ValueError,'zero API calls'):existing.verifier.inspect(existing.JOB,allow_unplanned=True)
        with closing(sqlite3.connect(self.ledger)) as db,db:db.execute('DELETE FROM calls')
        (self.fixture.output/'generation').mkdir()
        with self.assertRaisesRegex(ValueError,'execution artifact'):existing.verifier.inspect(existing.JOB,allow_unplanned=True)

    def test_wrong_credit_limit_or_changed_source_pin_rejected(self):
        f=self.fixture;file=f.output/'request.json';before=file.read_bytes()
        value=json.loads(before);value['max_credits']=70;existing.write(file,value)
        with self.assertRaisesRegex(ValueError,'35-credit'):existing.verifier.inspect(existing.JOB,allow_unplanned=True)
        file.write_bytes(before);(f.output/'sources/meshy.py').write_text('Tampered analytic source')
        with self.assertRaisesRegex(ValueError,'source pin changed'):existing.verifier.inspect(existing.JOB,allow_unplanned=True)
