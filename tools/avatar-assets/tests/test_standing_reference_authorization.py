"""Reference-bound standing policies use temporary ledgers and analytic image bytes."""
import hashlib
import json
import unittest

from avatar_assets import standing_authorization as standing
from tests.test_standing_authorization import StandingAuthorizationTests


class StandingReferenceAuthorizationTests(StandingAuthorizationTests):
    def setUp(self):
        self.reference=b'ANALYTIC IMAGE HASH FIXTURE; NOT A GENERATED AVATAR'
        self.reference_hash=hashlib.sha256(self.reference).hexdigest()
        super().setUp()
        self.policy.update(schema_version=2,allowed_reference_sha256=self.reference_hash)
        self.write(self.policy_path,self.policy,private=True)

    def request(self,name,parent=None):
        output=super().request(name,parent)
        path=output/'inputs/reference.png';path.write_bytes(self.reference)
        value=json.loads((output/'request.json').read_text())
        value['reference']={'path':'inputs/reference.png','original_path':'analytic original reference',
                            'bytes':len(self.reference),'sha256':self.reference_hash}
        self.write(output/'request.json',value)
        return output

    def test_v2_reference_grant_and_idempotence_ignore_original_absolute_path(self):
        ledger=self.root/'.local/openai/ledger.sqlite3';before=ledger.read_bytes()
        status=standing.standing_policy_status(self.root)
        self.assertEqual(status['reference_sha256'],self.reference_hash)
        self.assertTrue(standing.retry_eligibility(self.root,self.parent)['eligible'])
        self.assertEqual(ledger.read_bytes(),before)
        child=self.request('studio-reference-child')
        value=json.loads((child/'request.json').read_text());value['reference']['original_path']='/another/original/location.png'
        self.write(child/'request.json',value)
        self.assertEqual(self.authorize(child)['status'],'authorized')
        before=ledger.read_bytes();self.assertEqual(self.authorize(child)['status'],'already_authorized')
        self.assertEqual(ledger.read_bytes(),before)

    def test_v1_rejects_reference_and_v2_requires_explicit_reference_field(self):
        legacy={k:v for k,v in self.policy.items() if k!='allowed_reference_sha256'};legacy['schema_version']=1
        self.write(self.policy_path,legacy,private=True)
        self.assertTrue(standing.standing_policy_status(self.root)['enabled'])
        self.assertFalse(standing.retry_eligibility(self.root,self.parent)['eligible'])
        with self.assertRaisesRegex(ValueError,'v1.*references'):self.authorize(self.request('studio-v1-reference'))
        self.write(self.policy_path,{**legacy,'schema_version':2},private=True)
        self.assertFalse(standing.standing_policy_status(self.root)['enabled'])
        for invalid in ('bad',True,42):
            self.write(self.policy_path,{**self.policy,'allowed_reference_sha256':invalid},private=True)
            self.assertFalse(standing.standing_policy_status(self.root)['enabled'])

    def test_different_missing_or_new_reference_rejected_without_retirement(self):
        child=self.request('studio-ref-change');value=json.loads((child/'request.json').read_text())
        changed={**value,'reference':{**value['reference'],'sha256':'f'*64}}
        self.write(child/'request.json',changed)
        with self.assertRaisesRegex(ValueError,'approved reference'):self.authorize(child)
        self.write(child/'request.json',{**value,'reference':None})
        with self.assertRaisesRegex(ValueError,'approved reference'):self.authorize(child)
        self.write(child/'request.json',value);self.policy['allowed_reference_sha256']=None
        self.write(self.policy_path,self.policy,private=True)
        with self.assertRaisesRegex(ValueError,'absence'):self.authorize(child)
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM retirements').fetchone()[0],0)

    def test_reference_changed_bytes_symlink_escape_and_size_rejected(self):
        child=self.request('studio-ref-tamper');path=child/'inputs/reference.png'
        value=json.loads((child/'request.json').read_text())
        path.write_bytes(b'x'*len(self.reference))
        with self.assertRaisesRegex(ValueError,'reference bytes changed'):self.authorize(child)
        path.unlink();path.symlink_to(self.parent/'inputs/reference.png')
        with self.assertRaisesRegex(ValueError,'symlink|escapes'):self.authorize(child)
        path.unlink();path.write_bytes(self.reference)
        for fields,pattern in (({'path':'../studio-root/inputs/reference.png'},'relative path'),
                               ({'path':str(path)},'relative path'),({'bytes':8*1024**2+1},'approved reference'),
                               ({'bytes':len(self.reference)+1},'reference size')):
            self.write(child/'request.json',{**value,'reference':{**value['reference'],**fields}})
            with self.assertRaisesRegex(ValueError,pattern):self.authorize(child)
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM retirements').fetchone()[0],0)

    def test_reference_policy_hash_change_cannot_authorize_saved_retry(self):
        child=self.request('studio-ref-policy');old=self.hash(self.policy_path)
        self.write(self.policy_path,{**self.policy,'allowed_reference_sha256':'f'*64},private=True)
        with self.assertRaisesRegex(ValueError,'policy changed'):
            standing.authorize_successor(self.root,self.parent,child,expected_policy_sha256=old)
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM retirements').fetchone()[0],0)


class StandingOptionalReferenceTests(unittest.TestCase):
    def test_v2_null_reference_preserves_existing_no_reference_workflow(self):
        fixture=StandingAuthorizationTests('runTest');fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.policy.update(schema_version=2,allowed_reference_sha256=None)
        fixture.write(fixture.policy_path,fixture.policy,private=True)
        self.assertTrue(standing.retry_eligibility(fixture.root,fixture.parent)['eligible'])
        self.assertEqual(fixture.authorize(fixture.request('studio-no-reference'))['status'],'authorized')


if __name__=='__main__':unittest.main()
