"""Temporary-ledger authorization fixtures, never provider or native calls."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from avatar_assets import standing_authorization as standing
from avatar_assets.agent_backend import DEFAULT
from avatar_assets.openai_runner import OpenAILedger, accounting_snapshot, reserve_microdollars

APPROVAL = {'decision':'approved','user_reply':'Analytic standing approval only; no real spending.'}


class StandingAuthorizationTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()/'project';(self.root/'outputs').mkdir(parents=True)
        self.source=b'analytic source bytes, not a GLB fixture'
        self.source_hash=hashlib.sha256(self.source).hexdigest()
        self.settings={**DEFAULT,'max_calls':7,'max_repairs':2}
        self.parent=self.request('studio-root')
        self.ledger=OpenAILedger(self.root/'.local/openai');self.addCleanup(self.ledger.close)
        self.ledger.configure(17_000_000,APPROVAL)
        self.ledger.authorize_job(self.parent.name,self.hash(self.parent/'request.json'),7,17_000_000,APPROVAL)
        self.settle(self.parent)
        self.ledger.tighten_budget(15_000_000,{**APPROVAL,'operation':'tighten_project_ceiling','max_microdollars':15_000_000})
        receipt=self.root/'outputs/approval.json'
        self.write(receipt,{**APPROVAL,'standing_retries':True,'max_total_microdollars':15_000_000})
        self.policy={'schema_version':1,'status':'active','policy_id':'test-standing',
            'approval_receipt':{'path':'outputs/approval.json','sha256':self.hash(receipt)},
            'max_total_microdollars':15_000_000,'model':self.settings['model'],'max_calls':7,'max_repairs':2,
            'max_successors':2,'allowed_source_sha256':self.source_hash,
            'lineage_root_job_id':self.parent.name,'lineage_root_request_sha256':self.hash(self.parent/'request.json')}
        self.policy_path=self.root/standing.POLICY_FILE;self.write(self.policy_path,self.policy,private=True)
        blocker=patch('avatar_assets.openai_runner.post_response',side_effect=AssertionError('No provider calls'))
        blocker.start();self.addCleanup(blocker.stop)

    @staticmethod
    def hash(path):return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def write(path,value,private=False):
        path.write_text(json.dumps(value,sort_keys=True)+'\n')
        if private:path.chmod(0o600)

    def request(self,name):
        output=self.root/'outputs'/name;(output/'inputs').mkdir(parents=True)
        (output/'inputs/source-model.glb').write_bytes(self.source)
        value={'schema_version':1,'id':name,'created':'analytic','prompt':'A hinged red lamp dances',
            'style':'Painted metal','custom_actions':['Dance without replacing the articulated lamp'],
            'states':[],'preset':'preview','media_spec':{'size':720,'seconds':4},'max_credits':0,
            'reference':None,'reuse_model':{'path':'inputs/source-model.glb','original_path':'analytic source',
                'bytes':len(self.source),'sha256':self.source_hash},'agent_backend':self.settings,
            'max_agent_calls':7,'max_repairs':2,'agent_model':self.settings['model'],'agent_timeout':self.settings['timeout']}
        self.write(output/'request.json',value);return output

    def settle(self,output):
        self.ledger.reserve(output.name,self.hash(output/'request.json'),7,'planner','a'*64,reserve_microdollars())
        self.ledger.settle(output.name,'planner','resp_analytic',
                          {'input_tokens':40000,'output_tokens':0,'total_tokens':40000},100000)

    def authorize(self,child,parent=None):
        return standing.authorize_successor(self.root,parent or self.parent,child,
                                             expected_policy_sha256=self.hash(self.policy_path))

    def test_readonly_status_and_eligibility_do_not_change_ledger_or_create_grants(self):
        path=self.root/'.local/openai/ledger.sqlite3';before=path.read_bytes()
        self.assertTrue(standing.standing_policy_status(self.root)['enabled'])
        result=standing.retry_eligibility(self.root,self.parent)
        self.assertTrue(result['eligible']);self.assertEqual(result['grant_microdollars'],14_900_000)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],1)

    def test_exact_child_grant_is_idempotent_and_preserves_parent_history(self):
        child=self.request('studio-next')
        before={table:[tuple(r) for r in self.ledger.db.execute('SELECT * FROM '+table)] for table in ('budget','calls')}
        original=tuple(self.ledger.db.execute('SELECT * FROM jobs').fetchone())
        result=self.authorize(child)
        self.assertEqual(result['status'],'authorized');self.assertEqual(result['grant_microdollars'],14_900_000)
        path=self.root/'.local/openai/ledger.sqlite3';old=path.read_bytes()
        self.assertEqual(self.authorize(child)['status'],'already_authorized');self.assertEqual(path.read_bytes(),old)
        self.assertEqual(tuple(self.ledger.db.execute('SELECT * FROM jobs WHERE id=?',(self.parent.name,)).fetchone()),original)
        for table,rows in before.items():self.assertEqual([tuple(r) for r in self.ledger.db.execute('SELECT * FROM '+table)],rows)
        self.assertEqual(accounting_snapshot(self.ledger.db)['project_exposure_microdollars'],100000)
        with self.assertRaisesRegex(ValueError,'Retired'):
            self.ledger.reserve(self.parent.name,self.hash(self.parent/'request.json'),7,'author-0','b'*64,1)

    def test_policy_tamper_missing_revoked_permissions_and_receipt_changes_disable(self):
        self.policy_path.unlink();self.assertFalse(standing.retry_eligibility(self.root,self.parent)['eligible'])
        for value in ({**self.policy,'status':'revoked'}, {**self.policy,'max_calls':8},
                      {**self.policy,'max_successors':1000}, {**self.policy,'extra':True}):
            self.write(self.policy_path,value,private=True)
            self.assertFalse(standing.standing_policy_status(self.root)['enabled'])
        self.write(self.policy_path,self.policy,private=True);self.policy_path.chmod(0o644)
        self.assertFalse(standing.retry_eligibility(self.root,self.parent)['eligible'])
        self.policy_path.chmod(0o600)
        (self.root/'outputs/approval.json').write_text('{}')
        self.assertFalse(standing.standing_policy_status(self.root)['enabled'])

    def test_policy_hash_change_between_retry_and_prepare_cannot_grant(self):
        child=self.request('studio-next');old=self.hash(self.policy_path)
        self.write(self.policy_path,{**self.policy,'max_successors':1},private=True)
        with self.assertRaisesRegex(ValueError,'policy changed'):
            standing.authorize_successor(self.root,self.parent,child,expected_policy_sha256=old)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],1)

    def test_child_intent_or_source_change_rejects_without_retiring_parent(self):
        child=self.request('studio-next');request=json.loads((child/'request.json').read_text())
        changed={**request,'prompt':'Replace the lamp with a different subject'}
        self.write(child/'request.json',changed)
        with self.assertRaisesRegex(ValueError,'intent'):self.authorize(child)
        self.write(child/'request.json',request)
        (child/'inputs/source-model.glb').write_bytes(b'x'*len(self.source))
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.authorize(child)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM retirements').fetchone()[0],0)

    def test_unknown_parent_charge_is_not_released_or_granted(self):
        self.ledger.reserve(self.parent.name,self.hash(self.parent/'request.json'),7,'author-0','c'*64,reserve_microdollars())
        self.assertFalse(standing.retry_eligibility(self.root,self.parent)['eligible'])
        with self.assertRaisesRegex(ValueError,'unknown charges'):self.authorize(self.request('studio-next'))
        self.assertEqual(self.ledger.call(self.parent.name,'author-0')['state'],'reserved_uncertain')

    def test_current_project_cap_must_already_be_at_most_fifteen_dollars(self):
        self.ledger.db.execute('DELETE FROM budget_reductions');self.ledger.db.commit()
        self.assertFalse(standing.retry_eligibility(self.root,self.parent)['eligible'])
        with self.assertRaisesRegex(ValueError,'ceiling exceeds'):self.authorize(self.request('studio-next'))
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM budget_reductions').fetchone()[0],0)

    def test_finite_lineage_cannot_reset_count_or_fork_parent(self):
        first=self.request('studio-one');self.authorize(first);self.settle(first)
        self.assertFalse(standing.retry_eligibility(self.root,self.parent)['eligible'])
        with self.assertRaisesRegex(ValueError,'another successor'):self.authorize(self.request('studio-fork'))
        second=self.request('studio-two');self.authorize(second,parent=first);self.settle(second)
        self.assertFalse(standing.retry_eligibility(self.root,second)['eligible'])
        with self.assertRaisesRegex(ValueError,'successor limit'):
            self.authorize(self.request('studio-three'),parent=second)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM retirements').fetchone()[0],2)

    def test_unrelated_parent_and_root_request_tampering_fail_closed(self):
        foreign=self.request('studio-unrelated')
        self.assertFalse(standing.retry_eligibility(self.root,foreign)['eligible'])
        value=json.loads((self.parent/'request.json').read_text());value['created']='changed'
        self.write(self.parent/'request.json',value)
        self.assertFalse(standing.retry_eligibility(self.root,self.parent)['eligible'])

    def test_changed_existing_grant_is_not_treated_as_idempotent(self):
        child=self.request('studio-next');self.authorize(child)
        self.ledger.db.execute('UPDATE jobs SET digest=? WHERE id=?',('f'*64,child.name));self.ledger.db.commit()
        with self.assertRaisesRegex(ValueError,'pinned request'):self.authorize(child)


if __name__=='__main__':unittest.main()
