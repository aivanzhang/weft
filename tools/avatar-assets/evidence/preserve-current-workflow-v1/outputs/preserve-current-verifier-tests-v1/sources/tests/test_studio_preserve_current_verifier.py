"""Read-only canonical inspection fixtures; no server, browser or provider calls."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch
import test_studio_live_verifier as fixtures
from avatar_assets.studio_jobs import _digest


class PreserveCurrentInspectionTests(unittest.TestCase):
    setUp = fixtures.StudioInspectionTests.setUp
    pins = fixtures.StudioInspectionTests.pins
    make_fixture = fixtures.StudioInspectionTests.make_fixture
    save_generation = fixtures.StudioInspectionTests.save_generation
    update_row = fixtures.StudioInspectionTests.update_row
    api_fixture = fixtures.StudioInspectionTests.api_fixture
    continuation_fixture = fixtures.StudioInspectionTests.continuation_fixture

    def policy_fixture(self):
        self.continuation_fixture()
        self.policy = {'schema_version':1, 'mode':'preserve_current',
            'baseline':{'parent_output':'outputs/studio-'+'b'*24, 'attempt':2,
                'source_sha256':self.prepared['reuse_model']['sha256'],
                **{name:'e'*64 for name in ['model_manifest_sha256','design_sha256','review_sha256','descriptor_sha256']}},
            'authorization':{'path':'.local/studio/jobs/'+fixtures.JOB+'/appearance-preference.json','sha256':'f'*64}}
        self.request['appearance_policy']=deepcopy(self.policy)
        self.prepared['appearance_policy']=deepcopy(self.policy)
        self.reseal()

    def reseal(self):
        fixtures.write(self.job/'request.json',self.request)
        fixtures.write(self.output/'request.json',self.prepared)
        fixtures.write(self.output/'input-receipt.json',{'request_sha256':fixtures.digest(self.output/'request.json'),
            'sources':self.pins(self.output/'sources')})
        fixtures.write(self.job/'pipeline-request.json',{'schema_version':1,'studio_request_digest':_digest(self.request),
            'pipeline_request_sha256':fixtures.digest(self.output/'request.json')})
        self.data['plan_id']=_digest({'request':_digest(self.request),'plan':self.plan,'generation':None})
        self.update_row(request=json.dumps(self.request),request_digest=_digest(self.request),data=json.dumps(self.data))

    def test_exact_policy_delegates_canonical_origin_and_prepared_source_checks(self):
        self.policy_fixture()
        with patch('avatar_assets.appearance_policy.verify_origin',return_value={}) as origin, \
             patch('avatar_assets.appearance_policy.verify_policy',return_value=self.policy) as source, \
             patch('avatar_assets.pipeline_continuation.verify',return_value={}):
            result=fixtures.verifier.inspect(fixtures.JOB)
            origin.assert_called_once_with(self.root,self.policy)
            source.assert_called_once_with(self.root,self.output,self.prepared)
        self.assertEqual(result['status'],'planned')
        self.assertEqual(result['plan_id'],self.data['plan_id'])

    def test_origin_or_source_failure_is_not_bypassed(self):
        self.policy_fixture()
        with patch('avatar_assets.appearance_policy.verify_origin',side_effect=ValueError('Origin receipt changed')):
            with self.assertRaisesRegex(ValueError,'Origin receipt changed'):fixtures.verifier.inspect(fixtures.JOB)
        with patch('avatar_assets.appearance_policy.verify_origin',return_value={}), \
             patch('avatar_assets.appearance_policy.verify_policy',side_effect=ValueError('Baseline source changed')), \
             patch('avatar_assets.pipeline_continuation.verify',return_value={}):
            with self.assertRaisesRegex(ValueError,'Baseline source changed'):fixtures.verifier.inspect(fixtures.JOB)

    def test_invalid_extra_or_changed_policy_still_rejects(self):
        self.policy_fixture()
        with patch('avatar_assets.appearance_policy.verify_origin',return_value={}), \
             patch('avatar_assets.pipeline_continuation.verify',return_value={}):
            self.prepared['appearance_policy']['baseline']['source_sha256']='0'*64;self.reseal()
            with self.assertRaisesRegex(ValueError,'appearance_policy'):fixtures.verifier.inspect(fixtures.JOB)
            self.prepared['appearance_policy']=deepcopy(self.policy)
            self.request['appearance_policy']['authorization']['path']='../outside.json';self.reseal()
            with self.assertRaisesRegex(ValueError,'receipt path'):fixtures.verifier.inspect(fixtures.JOB)
            self.request['appearance_policy']=deepcopy(self.policy);self.request['budget']=99;self.reseal()
            with self.assertRaisesRegex(ValueError,'canonical Studio request'):fixtures.verifier.inspect(fixtures.JOB)
