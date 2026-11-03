"""Explicit preference custody using analytic artifacts and temporary Studio ledgers."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from avatar_assets import appearance_policy as appearance, pipeline_continuation as cont
from avatar_assets.safety import read_json, sha256, write_json
from avatar_assets.studio_jobs import StudioJobs, _write_private
from test_studio_jobs import FixtureExecutor
import test_repair_composition_continuation as composed
import test_pipeline_continuation as continuation_fixtures
from test_pipeline_prompts import request as fixture_request
from test_author_contract import typed
import test_standing_authorization as standing_fixtures


class AppearancePolicyTests(unittest.TestCase):
    prepare=composed.ComposedContinuationTests.prepare
    make_terminal=composed.ComposedContinuationTests.make_terminal
    visual=composed.ComposedContinuationTests.visual
    seal=staticmethod(composed.ComposedContinuationTests.seal)

    def setUp(self):
        continuation_fixtures.ContinuationTests.setUp(self)
        design=json.loads(self.answer['design_json'])
        for action in design['authored_actions']:action['interpolation']='LINEAR'
        self.answer=typed(design)
        selected=self.root/'outputs/selected-source.glb'
        shutil.copyfile(self.original,selected); self.original=selected
        self.jobs=StudioJobs(self.root,executor=FixtureExecutor())
        request=fixture_request()
        original=self.jobs.create(request['prompt'],style=request['style'],reuse_model=self.original)
        self.parent_id=original['id']; self.parent=self.jobs.resolve_job(self.parent_id)
        self.prepare(self.parent); self.make_terminal(self.parent); self.visual()
        self.terminal(self.parent_id)
        self.network.reset_mock(); self.network.side_effect=AssertionError('No provider calls')
        self.addCleanup(patch.stopall)
        patch('avatar_assets.studio_inspection.collect',return_value={'fixture':'Analytic local custody'}).start()
        patch('avatar_assets.studio_inspection.recheck').start()
        patch('avatar_assets.standing_authorization.retry_eligibility',return_value={'eligible':True,'policy_sha256':'e'*64}).start()

    def terminal(self, identifier):
        folder=self.jobs.jobs_dir/identifier/'run-0001'; folder.mkdir(mode=0o700)
        with self.jobs._db(True) as db:
            row=self.jobs._row(db,identifier); data=json.loads(row['data'])
            data.update(operation='execute',token='a'*32,owner_pid=900001,owner_identity='analytic-dead',
                        invocation_dir=str(folder.relative_to(self.jobs.state_dir)),invocations=1)
            self.jobs._update(db,row,data,'unsupported')
            _write_private(folder/'invocation.json',{'job_id':identifier,'token':'a'*32,'request_digest':row['request_digest']})
        for path in self.jobs._retry_locks(identifier,self.jobs.resolve_job(identifier),global_lock=True):
            path.parent.mkdir(parents=True,exist_ok=True); path.touch(mode=0o600)

    def create_child(self):
        before=cont._inventory(self.parent)
        result=self.jobs.retry(self.parent_id,appearance_policy='preserve_current')
        self.assertEqual(before,cont._inventory(self.parent))
        child=self.jobs.resolve_job(result['id'])
        saved=read_json(self.jobs.jobs_dir/result['id']/'request.json')
        self.prepare(child,self.parent,appearance_policy=saved['appearance_policy'])
        request=read_json(child/'request.json')
        return result,child,request

    def test_explicit_retry_pins_latest_baseline_and_prepare_copies_exact_preference(self):
        result,child,request=self.create_child()
        policy=appearance.verify_policy(self.root,child,request)
        self.assertEqual(result['request']['appearance_policy'],'preserve_current')
        self.assertEqual(policy['baseline']['parent_output'],str(self.parent.relative_to(self.root)))
        self.assertEqual(policy['baseline']['model_manifest_sha256'],sha256(self.parent/'model-0/manifest.json'))
        original=self.root/policy['authorization']['path']
        self.assertEqual(original.read_bytes(),(child/appearance.SNAPSHOT).read_bytes())
        self.assertFalse(read_json(original)['human_asset_approval'])
        self.assertEqual(appearance.verify_origin(self.root,policy),read_json(original))
        self.assertFalse(self.jobs.executor.calls); self.network.assert_not_called()
        self.assertEqual(self.jobs.retry(self.parent_id,appearance_policy='preserve_current')['id'],result['id'])

    def test_ordinary_retry_inherits_policy_and_cannot_rebase_or_remove_it(self):
        result,child,request=self.create_child(); self.terminal(result['id'])
        next_result=self.jobs.retry(result['id'])
        saved=read_json(self.jobs.jobs_dir/next_result['id']/'request.json')
        self.assertEqual(saved['appearance_policy'],request['appearance_policy'])
        for changed in (None,{**request['appearance_policy'],'baseline':{**request['appearance_policy']['baseline'],'attempt':1}}):
            other={**request,'appearance_policy':changed}
            with self.assertRaisesRegex(ValueError,'remove or change'):cont._match(other,request)
        self.assertFalse(self.jobs.executor.calls)

    def test_existing_normal_retry_cannot_be_relabelled_by_later_preference(self):
        first=self.jobs.retry(self.parent_id)
        with self.assertRaisesRegex(ValueError,'immutable appearance preference'):
            self.jobs.retry(self.parent_id,appearance_policy='preserve_current')
        saved=read_json(self.jobs.jobs_dir/first['id']/'request.json')
        self.assertNotIn('appearance_policy',saved)

    def test_receipt_and_canonical_request_tampering_block_authority(self):
        result,child,request=self.create_child(); policy=request['appearance_policy']
        snapshot=child/appearance.SNAPSHOT; original=snapshot.read_bytes()
        changed=read_json(snapshot); changed['human_asset_approval']=True; write_json(snapshot,changed)
        with self.assertRaisesRegex(ValueError,'receipt changed'):appearance.verify_policy(self.root,child,request)
        snapshot.write_bytes(original)
        saved=self.jobs.jobs_dir/result['id']/'request.json'; value=read_json(saved)
        value['prompt']='Different intent'; write_json(saved,value); saved.chmod(0o600)
        with self.assertRaisesRegex(ValueError,'canonical explicit Retry'):appearance.verify_origin(self.root,policy)
        self.network.assert_not_called()

    def test_portable_custody_rejects_changed_baseline_and_wrong_source(self):
        _,child,request=self.create_child()
        wrong=deepcopy(request); wrong['reuse_model']['sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'request or source'):appearance.verify_policy(self.root,child,wrong)
        path=self.parent/'finish-0.json'; path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'changed artifact'):appearance.verify_policy(self.root,child,request)
        self.network.assert_not_called()

    def test_no_qualifying_review_is_not_a_preservation_baseline(self):
        captured=cont.collect(self.root,self.parent)
        for row in captured['context']['findings']['historical_visual_reviews']:row['passed_checks']=['identity','framing']
        with self.assertRaisesRegex(ValueError,'No eligible stable'):
            appearance.build_policy(self.root,self.parent,captured,parent_job_id=self.parent_id,child_job_id='b'*24,
                receipt_path='.local/studio/jobs/'+'b'*24+'/appearance-preference.json')

    def test_policy_schema_rejects_unknown_fields_paths_and_boolean_attempt(self):
        _,_,request=self.create_child(); policy=request['appearance_policy']
        bads=[]
        for mutation in ({'extra':True},{'mode':'improve_fur'},{'schema_version':True}):bads.append({**policy,**mutation})
        bads.append({**policy,'baseline':{**policy['baseline'],'attempt':True}})
        bads.append({**policy,'authorization':{**policy['authorization'],'path':'../preference.json'}})
        for bad in bads:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):appearance.validate_policy(bad)

    def test_portable_receipt_reads_without_private_registry(self):
        _,child,request=self.create_child()
        self.jobs.db_path.rename(self.jobs.db_path.with_suffix('.saved'))
        self.assertEqual(appearance.verify_policy(self.root,child,request),request['appearance_policy'])
        with self.assertRaises(ValueError):appearance.verify_origin(self.root,request['appearance_policy'])


class AppearanceStandingTests(unittest.TestCase):
    setUp=standing_fixtures.StandingAuthorizationTests.setUp
    request=standing_fixtures.StandingAuthorizationTests.request
    settle=standing_fixtures.StandingAuthorizationTests.settle
    authorize=standing_fixtures.StandingAuthorizationTests.authorize
    hash=staticmethod(standing_fixtures.StandingAuthorizationTests.hash)
    write=staticmethod(standing_fixtures.StandingAuthorizationTests.write)

    def policy_value(self):
        return {'schema_version':1,'mode':'preserve_current','baseline':{'parent_output':'outputs/'+self.parent.name,
            'attempt':0,**{key:'a'*64 for key in appearance.BASE_FIELDS-{'parent_output','attempt','source_sha256'}},
            'source_sha256':self.source_hash},'authorization':{'path':'.local/studio/jobs/'+'b'*24+'/appearance-preference.json','sha256':'c'*64}}

    def test_validated_preference_does_not_change_cap_or_reset_calls_and_cannot_change_intent(self):
        child=self.request('studio-next'); value=read_json(child/'request.json'); value['appearance_policy']=self.policy_value()
        self.write(child/'request.json',value)
        before=[tuple(row) for row in self.ledger.db.execute('SELECT * FROM calls')]
        with patch.object(appearance,'verify_origin'),patch.object(appearance,'verify_policy'):
            result=self.authorize(child)
            self.assertEqual(result['grant_microdollars'],14_900_000)
            self.assertEqual(self.authorize(child)['status'],'already_authorized')
        self.assertEqual(before,[tuple(row) for row in self.ledger.db.execute('SELECT * FROM calls')])
        self.assertEqual(self.ledger.db.execute('SELECT MAX(ceiling) FROM budget_reductions').fetchone()[0],15_000_000)

    def test_unverified_preference_blocks_grant_before_retirement(self):
        child=self.request('studio-next'); value=read_json(child/'request.json'); value['appearance_policy']=self.policy_value()
        self.write(child/'request.json',value)
        with self.assertRaises((ValueError,OSError)):self.authorize(child)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM retirements').fetchone()[0],0)


if __name__=='__main__':unittest.main()
