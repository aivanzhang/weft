"""Harness-only admission: saved requests are not plans or permission to call providers."""
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3
import unittest
from unittest.mock import patch
import test_studio_live_verifier as existing


class UnplannedPreparationInspectionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=existing.StudioInspectionTests('runTest')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture;f.api_fixture(reuse=True)
        (f.output/'plan.json').unlink()
        shutil.rmtree(f.output/'generation')
        for name in ('generation-config.json','generation-receipt.json'):
            (f.output/name).unlink(missing_ok=True)
        f.data['plan_id']=None;f.update_row(status='awaiting_authorization',data=existing.json.dumps(f.data))
        self.ledger=f.root/'.local/openai/ledger.sqlite3';self.ledger.parent.mkdir()
        with closing(sqlite3.connect(self.ledger)) as db,db:
            db.execute('CREATE TABLE calls(job TEXT,role TEXT)')

    def inspect(self,**kwargs):return existing.verifier.inspect(existing.JOB,**kwargs)

    def test_default_remains_strict_and_opt_in_returns_no_plan_or_authority(self):
        with self.assertRaises((ValueError,OSError)):self.inspect()
        with patch('avatar_assets.openai_runner.read_api_key',side_effect=AssertionError('No key read')):
            result=self.inspect(allow_unplanned=True)
        self.assertFalse(result['plan_available']);self.assertIsNone(result['plan_id'])
        self.assertFalse(result['agent_backend']['plan_verified'])
        self.assertFalse(result['agent_backend']['authorization_checked'])
        self.assertFalse((self.fixture.output/'agents').exists())

    def test_invalid_status_active_lease_queue_or_plan_identifier_rejected(self):
        f=self.fixture
        for status in ('planned','failed','running','unsupported','created'):
            f.update_row(status=status)
            with self.subTest(status=status),self.assertRaises(ValueError):self.inspect(allow_unplanned=True)
        f.update_row(status='awaiting_authorization')
        f.data['plan_id']='f'*64;f.update_row(data=existing.json.dumps(f.data))
        with self.assertRaisesRegex(ValueError,'plan ID differs'):self.inspect(allow_unplanned=True)
        f.data['plan_id']=None;f.update_row(data=existing.json.dumps(f.data))
        with closing(sqlite3.connect(f.database)) as db,db:db.execute('INSERT INTO active VALUES(1,?)',(existing.JOB,))
        with self.assertRaisesRegex(ValueError,'released'):self.inspect(allow_unplanned=True)

    def test_any_api_reservation_or_agent_artifact_rejected(self):
        with closing(sqlite3.connect(self.ledger)) as db,db:db.execute('INSERT INTO calls VALUES(?,?)',(self.fixture.output.name,'planner'))
        with self.assertRaisesRegex(ValueError,'zero API calls'):self.inspect(allow_unplanned=True)
        with closing(sqlite3.connect(self.ledger)) as db,db:db.execute('DELETE FROM calls')
        (self.fixture.output/'agents').mkdir()
        with self.assertRaisesRegex(ValueError,'execution artifact'):self.inspect(allow_unplanned=True)

    def test_tampered_request_source_or_backend_pin_is_not_admitted(self):
        f=self.fixture
        target=f.output/'request.json';original=target.read_bytes()
        changed=dict(f.prepared);changed['prompt']='Changed prompt';existing.write(target,changed)
        with self.assertRaisesRegex(ValueError,'Studio input'):self.inspect(allow_unplanned=True)
        target.write_bytes(original)
        source=f.output/'inputs/source-model.glb';before=source.read_bytes();source.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.inspect(allow_unplanned=True)
        source.write_bytes(before)
        (f.output/'sources/meshy.py').write_text('Changed source pin')
        with self.assertRaisesRegex(ValueError,'source pin changed'):self.inspect(allow_unplanned=True)

    def test_present_invalid_plan_and_symlink_are_not_treated_as_unplanned(self):
        path=self.fixture.output/'plan.json';existing.write(path,{})
        with self.assertRaises(ValueError):self.inspect(allow_unplanned=True)
        path.unlink();path.symlink_to(self.fixture.output/'absent.json')
        with self.assertRaises(ValueError):self.inspect(allow_unplanned=True)


if __name__=='__main__':unittest.main()
