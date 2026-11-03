"""Bounded analytic receipts/ledger only; no keys, provider, native or browser calls."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from avatar_assets import studio_ai_status as status
from avatar_assets.openai_runner import digest,MODEL


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')


def pin(path):return {'path':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


class StudioAiStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.output=self.root/'outputs/studio-analytic'
        self.binding=self.root/'.local/studio/jobs/analytic/pipeline-request.json';self.request_digest='d'*64
        self.prepared={'agent_backend':{'provider':'openai','max_calls':7,'model':MODEL},'max_agent_calls':7}
        write(self.output/'request.json',self.prepared);checksum=pin(self.output/'request.json')['sha256']
        write(self.binding,{'schema_version':1,'studio_request_digest':self.request_digest,'pipeline_request_sha256':checksum})
        write(self.output/'input-receipt.json',{'request_sha256':checksum})
        self.settings={'job_id':self.output.name,'job_digest':checksum,'max_calls':7,'model':MODEL}
        write(self.output/'agents/runner.json',self.settings)
        self.folder=self.output/'agents/author-0'
        self.call_request={'role':'author-0','settings_sha256':digest(self.settings),'prompt_sha256':'b'*64,'schema_sha256':'c'*64,'images':[]}
        write(self.folder/'request.json',self.call_request)
        self.receipt={'status':'failed_or_uncertain','execution_kind':'openai_responses_api',
            'reserved_microdollars':5618640,'automatic_resubmission_allowed':False,'elapsed_seconds':240.009,
            'error':'RAW PROVIDER ERROR sk-secret must never appear','artifacts':[pin(self.folder/'request.json')]}
        write(self.folder/'receipt.json',self.receipt)
        self.ledger=self.root/'.local/openai/ledger.sqlite3';self.ledger.parent.mkdir(parents=True)
        with closing(sqlite3.connect(self.ledger)) as db,db:
            db.execute('CREATE TABLE jobs(id TEXT PRIMARY KEY,digest TEXT,calls INTEGER)')
            db.execute('CREATE TABLE calls(job TEXT,role TEXT,request_digest TEXT,reserved INTEGER,state TEXT,response_id TEXT,actual INTEGER,usage TEXT,PRIMARY KEY(job,role))')
            db.execute('INSERT INTO jobs VALUES(?,?,?)',(self.output.name,checksum,7))
            db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?,?,?)',(self.output.name,'author-0',digest(self.call_request),5618640,'reserved_uncertain',None,None,None))
        self.ledger.chmod(0o600)

    def snapshot(self,**options):
        return status.failure_snapshot(self.root,self.output,self.binding,self.request_digest,
            status=options.get('status','needs_inspection'),operation=options.get('operation','execute'))

    def sql(self,query,args=()):
        with closing(sqlite3.connect(self.ledger)) as db,db:db.execute(query,args)

    def settle(self,answer=None):
        usage={'input_tokens':100,'output_tokens':20,'total_tokens':120}
        self.sql("UPDATE calls SET state='settled',response_id='resp_analytic',actual=550,usage=?",(json.dumps(usage),))
        if answer is not None:
            write(self.folder/'answer.json',answer)
            self.receipt.update(status='succeeded',response_id='resp_analytic',actual_microdollars_upper_estimate=550,
                usage=usage,answer_sha256=pin(self.folder/'answer.json')['sha256'])
            self.receipt['artifacts'].append(pin(self.folder/'answer.json'));write(self.folder/'receipt.json',self.receipt)

    def test_timeout_is_specific_unknown_charge_not_raw_error_and_read_only(self):
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file()}
        result=self.snapshot()
        self.assertEqual(result['code'],'ai_charge_unknown');self.assertTrue(result['verified'])
        self.assertIn('240 seconds',result['message']);self.assertIn('$5.618640',result['message'])
        self.assertNotIn('sk-secret',str(result));self.assertFalse(result['automatic_retry'])
        self.assertIsNone(result['charge']['actual_microdollars'])
        self.assertEqual(before,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file()})

    def test_settled_invalid_response_is_not_unknown_transport(self):
        self.settle();result=self.snapshot()
        self.assertEqual(result['code'],'ai_response_unusable');self.assertEqual(result['charge']['status'],'settled')
        self.assertEqual(result['charge']['actual_microdollars'],550)

    def test_settled_refusal_is_distinct_and_does_not_echo_answer(self):
        self.settle({'status':'unsupported','reason':'RAW MODEL DETAIL never display directly','design':None})
        result=self.snapshot(status='unsupported')
        self.assertEqual(result['code'],'ai_declined');self.assertNotIn('RAW MODEL',str(result))

    def test_native_failure_after_ready_answer_is_not_mislabeled_ai_failure(self):
        self.settle({'status':'ready','design':{'fixture':True}})
        self.assertIsNone(self.snapshot())

    def test_request_and_receipt_tampering_are_unverified_not_billing_claims(self):
        self.call_request['role']='reviewer-0';write(self.folder/'request.json',self.call_request)
        self.assertEqual(self.snapshot()['code'],'ai_status_unverified')
        self.call_request['role']='author-0';write(self.folder/'request.json',self.call_request)
        self.receipt['reserved_microdollars']=1;write(self.folder/'receipt.json',self.receipt)
        self.assertEqual(self.snapshot()['charge']['status'],'unverified')

    def test_bound_settled_answer_tamper_is_rejected(self):
        self.settle({'status':'unsupported'});write(self.folder/'answer.json',{'status':'ready'})
        self.assertEqual(self.snapshot()['code'],'ai_status_unverified')

    def test_symlink_and_oversized_receipt_fail_closed(self):
        receipt=self.folder/'receipt.json';receipt.unlink();receipt.symlink_to(self.folder/'request.json')
        self.assertEqual(self.snapshot()['code'],'ai_status_unverified')
        receipt.unlink();receipt.write_text('x'*(64*1024+1))
        self.assertEqual(self.snapshot()['code'],'ai_status_unverified')

    def test_missing_or_nonprivate_ledger_is_not_fabricated_cost(self):
        self.ledger.chmod(0o644);self.assertEqual(self.snapshot()['charge']['status'],'unverified')
        self.ledger.unlink();self.assertEqual(self.snapshot()['charge']['status'],'unverified')

    def test_old_non_api_and_live_jobs_do_not_read_ledger(self):
        with patch.object(status.sqlite3,'connect',side_effect=AssertionError('No ledger read')):
            self.assertIsNone(self.snapshot(status='running'))
            write(self.output/'request.json',{'legacy':True});self.assertIsNone(self.snapshot())

    def test_oversized_ledger_usage_and_more_than_cap_are_rejected_before_payload_read(self):
        self.sql('UPDATE calls SET usage=?',('x'*16385,));self.assertEqual(self.snapshot()['code'],'ai_status_unverified')
        self.sql('UPDATE calls SET usage=NULL')
        for i in range(8):
            self.sql('INSERT INTO calls VALUES(?,?,?,?,?,?,?,?)',(self.output.name,'extra-'+str(i),'a'*64,1,'reserved_uncertain',None,None,None))
        self.assertEqual(self.snapshot()['code'],'ai_status_unverified')

    def test_api_snapshot_uses_verified_failure_without_changing_job_capabilities(self):
        from avatar_assets.studio_jobs import StudioJobs
        from tests.test_studio_jobs import FixtureExecutor
        for path in (self.root/'.local/studio',self.root/'.local/studio/jobs'):path.chmod(0o700)
        jobs=StudioJobs(self.root,executor=FixtureExecutor());job=jobs.create('Analytic object')['id']
        with jobs._db(True) as db:
            row=jobs._row(db,job);data=json.loads(row['data']);data['error']='Generic controller error'
            jobs._update(db,row,data,'needs_inspection')
        failure=self.snapshot()
        with patch('avatar_assets.studio_ai_status.failure_snapshot',return_value=failure):result=jobs.get(job)
        self.assertEqual(result['error']['message'],failure['message'])
        self.assertEqual(result['ai_failure'],failure)
        self.assertFalse(result['capabilities']['start']);self.assertFalse(result['capabilities']['retry'])
        with jobs._db() as db:self.assertEqual(jobs._row(db,job)['status'],'needs_inspection')

    def test_delivery_artifact_failure_has_priority_without_ai_status_lookup(self):
        from avatar_assets.studio_jobs import StudioJobs
        from tests.test_studio_jobs import FixtureExecutor
        for path in (self.root/'.local/studio',self.root/'.local/studio/jobs'):path.chmod(0o700)
        jobs=StudioJobs(self.root,executor=FixtureExecutor())
        for missing_model in (False,True):
            job=jobs.create('Analytic completed object')['id'];folder=jobs.resolve_job(job)
            if missing_model:write(folder/'delivery.json',{'status':'complete','model':'model-0/model.glb'})
            with jobs._db(True) as db:
                row=jobs._row(db,job);data=json.loads(row['data']);data['error']='Old AI error'
                jobs._update(db,row,data,'complete')
            with patch('avatar_assets.studio_ai_status.failure_snapshot',side_effect=AssertionError('Artifact error must win')):
                result=jobs.get(job)
            self.assertEqual(result['status'],'needs_inspection');self.assertIsNone(result['ai_failure'])
            self.assertIn('model is missing' if missing_model else 'missing its sealed delivery record',result['error']['message'])
            self.assertFalse(result['capabilities']['start'])


if __name__=='__main__':unittest.main()
