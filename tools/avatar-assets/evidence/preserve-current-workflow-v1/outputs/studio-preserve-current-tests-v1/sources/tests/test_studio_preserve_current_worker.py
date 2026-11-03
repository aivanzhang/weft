"""Worker request binding fixtures; no native, model, provider, ledger or real job."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch

from avatar_assets.studio_worker import _validate_prepared_request, PIPELINE_LIMITS, MEDIA_SPECS, run_child
import test_studio_jobs as fixtures


class PreserveCurrentWorkerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve(); self.output = self.root/'outputs/studio-fixture'
        self.policy = {'schema_version':1, 'mode':'preserve_current',
                       'baseline':{'descriptor_sha256':'a'*64},
                       'authorization':{'path':'.local/studio/jobs/fixture/appearance-preference.json','sha256':'b'*64}}
        self.request = {'prompt':'Copper telescope', 'style':'Brushed copper',
                        'custom_actions':['Scan'], 'preset':'preview', 'reference':None, 'reuse_model':None,
                        'continuation_from':'c'*24, 'appearance_policy':self.policy}
        self.prepared = {'schema_version':1, 'id':self.output.name, 'prompt':self.request['prompt'],
                         'style':self.request['style'], 'custom_actions':['Scan'], 'states':[],
                         'preset':'preview', 'media_spec':MEDIA_SPECS['preview'], 'max_credits':35,
                         **PIPELINE_LIMITS, 'reference':None, 'reuse_model':None, 'created':'analytic',
                         'continuation':{'parent':'outputs/studio-'+'c'*24}, 'appearance_policy':deepcopy(self.policy)}
        # Isolate worker wiring from the separately tested authorization implementation.
        helper = ModuleType('avatar_assets.appearance_policy'); helper.verify_policy = Mock(return_value={})
        self.verify = helper.verify_policy
        self.module = patch.dict('sys.modules', {'avatar_assets.appearance_policy':helper})
        self.module.start(); self.addCleanup(self.module.stop)
        continuation = patch('avatar_assets.pipeline_continuation.verify', return_value={})
        continuation.start(); self.addCleanup(continuation.stop)

    def test_exact_policy_is_verified_not_just_copied(self):
        _validate_prepared_request(self.prepared, self.request, self.output)
        self.verify.assert_called_once_with(self.root, self.output, self.prepared)
        self.verify.side_effect = ValueError('Preference receipt changed')
        with self.assertRaisesRegex(ValueError, 'Preference receipt changed'):
            _validate_prepared_request(self.prepared, self.request, self.output)

    def test_removed_injected_or_changed_policy_fails_before_authorization(self):
        for mutate in [lambda p:p.pop('appearance_policy'),
                       lambda p:p['appearance_policy']['authorization'].update(sha256='f'*64),
                       lambda p:p.update(appearance_policy='preserve_current')]:
            candidate = deepcopy(self.prepared); mutate(candidate)
            with self.assertRaises(ValueError): _validate_prepared_request(candidate, self.request, self.output)
        ordinary = deepcopy(self.request); ordinary.pop('appearance_policy')
        with self.assertRaisesRegex(ValueError, 'fields differ'):
            _validate_prepared_request(self.prepared, ordinary, self.output)
        self.verify.assert_not_called()

    def test_worker_forwards_exact_binding_and_keeps_old_requests_unmodified(self):
        for include_policy in [True, False]:
            request = deepcopy(self.request)
            if not include_policy: request.pop('appearance_policy')
            output = self.root/'outputs'/('studio-preserve' if include_policy else 'studio-ordinary')
            state = self.root/'analytic-state'; job_id = output.name
            (state/'jobs'/job_id).mkdir(parents=True)
            jobs = SimpleNamespace(root=self.root, state_dir=state, jobs_dir=state/'jobs',
                resolve_job=lambda identifier: output if identifier==job_id else self.root/'outputs'/('studio-'+identifier),
                _plan_id=lambda row:'analytic-plan')
            fixture = fixtures.FixturePipeline(); observed = []
            def prepare(prompt, target, **kwargs):
                observed.append(deepcopy(kwargs))
                parent = kwargs.pop('continuation_from')
                fixture.prepare(prompt, target, **kwargs)
                saved = json.loads((target/'request.json').read_text())
                saved['continuation']={'parent':'outputs/'+parent.name}
                (target/'request.json').write_text(json.dumps(saved))
            module = SimpleNamespace(prepare=prepare, Pipeline=lambda folder:SimpleNamespace(
                request=json.loads((folder/'request.json').read_text()),run=lambda **kwargs:{'status':'planned'}))
            with patch('avatar_assets.studio_worker._current', return_value=(
                {'request_digest':'analytic'},request,{'invocation_dir':'invocation','operation':'plan'})):
                result = run_child(jobs,job_id,'analytic-token',pipeline_module=module)
            self.assertEqual(result['status'],'planned')
            if include_policy:self.assertEqual(observed[0]['appearance_policy'],self.policy)
            else:self.assertNotIn('appearance_policy',observed[0])
