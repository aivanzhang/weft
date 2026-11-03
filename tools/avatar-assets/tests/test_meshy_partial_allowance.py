"""Temporary-ledger credit accounting only; no provider requests or real approvals."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from avatar_assets.meshy_access import Ledger,digest
from avatar_assets.studio_jobs import StudioJobs


class PartialMeshyAllowanceTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve();self.ledger=Ledger(self.root);self.addCleanup(self.ledger.close)
        self.ledger.configure(140)
        historical={'id':'historical','max_credits':140}
        self.ledger.bind_job(historical,self.root/'historical')
        self.ledger.reserve('historical','prior','/analytic',{'estimated_credits':132},132)
        self.ledger.update('historical','prior','SUCCEEDED','analytic-prior',{'consumed_credits':132})
        self.plan={'id':'target','max_credits':35,'estimated_credits':35}
        self.out=self.root/'target'

    def approval(self,plan=None,credits=27):
        plan=plan or self.plan
        return {'decision':'approved','job_id':plan['id'],'plan_digest':digest(plan),
                'additional_credits':credits,'user_reply':'ANALYTIC explicit partial allowance, not real spending.'}

    def rows(self,table):
        return [tuple(row) for row in self.ledger.db.execute('SELECT * FROM '+table)]

    def reserve(self,name,step,credits):
        self.ledger.reserve(name,step,'/analytic',{'estimated_credits':credits},credits)

    def test_twenty_seven_targeted_plus_eight_base_funds_exact_thirty_five(self):
        budget=self.rows('budget');old_calls=self.rows('calls');approval=self.approval()
        self.ledger.authorize_job(self.plan,self.out,approval)
        self.reserve('target','geometry',25);self.reserve('target','texture',10)
        self.assertEqual(self.rows('budget'),budget)
        self.assertEqual(self.rows('calls')[0],old_calls[0])
        self.assertEqual(self.ledger.summary()['reserved_credits'],167)
        self.assertEqual(self.ledger.summary()['project_credit_limit'],167)
        self.assertEqual(self.ledger.summary()['job_scoped_allowances'],{'target':27})
        self.assertEqual(json.loads(self.rows('budget_grants')[0][3]),approval)
        with self.assertRaisesRegex(ValueError,'allowance exhausted'):self.reserve('target','extra',1)
        self.assertIsNone(self.ledger.call('target','extra'))

    def test_unused_other_job_grant_cannot_replace_missing_base_or_expand_target(self):
        other={**self.plan,'id':'other'}
        self.ledger.authorize_job(other,self.root/'other',self.approval(other,35))
        self.ledger.authorize_job(self.plan,self.out,self.approval())
        ungranted={**self.plan,'id':'ungranted'};self.ledger.bind_job(ungranted,self.root/'ungranted')
        with self.assertRaisesRegex(ValueError,'allowance exhausted'):self.reserve('ungranted','too-much',9)
        self.reserve('ungranted','consume-base',8)
        self.reserve('target','own-grant-only',27)
        with self.assertRaisesRegex(ValueError,'allowance exhausted'):self.reserve('target','borrow-other',8)
        self.assertIsNone(self.ledger.call('other','anything'))
        self.reserve('other','own-grant',35)

    def test_partial_grant_idempotency_preserves_approval_and_rejects_top_up(self):
        approval=self.approval();self.ledger.authorize_job(self.plan,self.out,approval)
        before={table:self.rows(table) for table in ('budget','jobs','calls','budget_grants')}
        self.ledger.authorize_job(self.plan,self.out,approval)
        self.assertEqual(before,{table:self.rows(table) for table in before})
        for change in ({'additional_credits':28},{'user_reply':'Changed approval'}):
            with self.assertRaisesRegex(ValueError,'immutable'):
                self.ledger.authorize_job(self.plan,self.out,{**approval,**change})
        self.assertEqual(before,{table:self.rows(table) for table in before})

    def test_invalid_partial_or_nonfinite_ceiling_rejects_before_binding(self):
        for amount in (0,-1,36,True,27.0,500):
            with self.subTest(amount=amount):
                with self.assertRaisesRegex(ValueError,'finite job ceiling'):
                    self.ledger.authorize_job(self.plan,self.out,self.approval(credits=amount))
        for ceiling in (True,35.0,float('inf'),0,501):
            value={**self.plan,'max_credits':ceiling}
            with self.subTest(ceiling=ceiling):
                with self.assertRaises(ValueError):self.ledger.authorize_job(value,self.out,self.approval(value,1))
        self.assertEqual(len(self.rows('jobs')),1);self.assertEqual(self.rows('budget_grants'),[])

    def test_full_grants_still_work_and_partial_grants_cannot_be_retroactive(self):
        full={**self.plan,'id':'full'}
        self.ledger.authorize_job(full,self.root/'full',self.approval(full,35));self.reserve('full','all',35)
        self.ledger.bind_job(self.plan,self.out);self.reserve('target','before-approval',1)
        with self.assertRaisesRegex(ValueError,'before any paid'):
            self.ledger.authorize_job(self.plan,self.out,self.approval())
        self.assertEqual(self.ledger.summary()['job_scoped_allowances'],{'full':35})

    def test_unknown_reservation_keeps_its_base_and_partial_allowance(self):
        self.ledger.authorize_job(self.plan,self.out,self.approval())
        self.reserve('target','unknown',35)
        self.ledger.update('target','unknown','uncertain')
        other={**self.plan,'id':'unfunded'};self.ledger.bind_job(other,self.root/'unfunded')
        with self.assertRaisesRegex(ValueError,'allowance exhausted'):self.reserve('unfunded','new',1)
        with self.assertRaisesRegex(ValueError,'already reserved'):self.reserve('target','unknown',35)
        self.assertEqual(self.ledger.summary()['unknown_submissions'],1)
        self.assertEqual(self.ledger.call('target','unknown')['credits'],35)

    def test_concurrent_targets_cannot_both_consume_the_same_base_eight(self):
        for name in ('one','two'):
            plan={**self.plan,'id':name}
            self.ledger.authorize_job(plan,self.root/name,self.approval(plan,27));self.reserve(name,'geometry',25)
        barrier=threading.Barrier(2)
        def claim(name):
            ledger=Ledger(self.root)
            try:
                barrier.wait(timeout=5)
                try:ledger.reserve(name,'texture','/analytic',{'estimated_credits':10},10)
                except ValueError:return 'blocked'
                return 'reserved'
            finally:ledger.close()
        with ThreadPoolExecutor(max_workers=2) as workers:results=list(workers.map(claim,('one','two')))
        self.assertEqual(sorted(results),['blocked','reserved'])
        usage=dict(self.ledger.db.execute('SELECT job,SUM(credits) FROM calls GROUP BY job'))
        grants=dict(self.ledger.db.execute('SELECT job,credits FROM budget_grants'))
        self.assertEqual(sum(max(0,amount-grants.get(name,0)) for name,amount in usage.items()),140)

    def test_existing_studio_authorization_accepts_partial_grant_read_only(self):
        tool=self.root/'tools/avatar-assets';generation=tool/'outputs/studio-fixture/generation';generation.mkdir(parents=True)
        (generation/'plan.json').write_text(json.dumps(self.plan))
        self.ledger.authorize_job(self.plan,generation,self.approval())
        row={'request':json.dumps({'reuse_model':None}),'output':'studio-fixture'}
        holder=SimpleNamespace(root=tool,_plan_id=lambda row:'analytic-plan-id')
        path=self.root/'.local/meshy/ledger.sqlite3';before=hashlib.sha256(path.read_bytes()).hexdigest()
        result=StudioJobs._authorization(holder,row)
        self.assertTrue(result['can_start']);self.assertEqual(result['remaining_base_credits'],8)
        self.assertEqual(result['remaining_job_credits'],35)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)


if __name__=='__main__':unittest.main()
