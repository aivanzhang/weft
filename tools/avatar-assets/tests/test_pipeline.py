"""Coordinator tests use explicit analytic agents and never invoke paid providers."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
from PIL import Image

from avatar_assets import pipeline
from avatar_assets.safety import artifacts, read_json, write_json
from test_pipeline_prompts import plan


class AnalyticAgent:
    def __init__(self, result=None):
        self.result = result or plan()
        self.roles = []

    def call(self, role, prompt, schema, images=()):
        self.roles.append(role)
        return deepcopy(self.result)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root/'avatar_assets').mkdir()
        for name in pipeline.PIPELINE_SOURCES:
            shutil.copyfile(pipeline.ROOT/'avatar_assets'/name, self.root/'avatar_assets'/name)
        self.patch = patch.object(pipeline, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.out = self.root/'outputs/test-pipeline'

    def prepare(self, **kwargs):
        return pipeline.prepare('A copper telescope that scans and nods.', self.out, **kwargs)

    def test_prepare_is_free_bounded_and_preserves_request(self):
        with patch('avatar_assets.meshy.run') as paid:
            result = self.prepare()
        self.assertEqual(result['estimate']['new_meshy_credits'], 35)
        self.assertEqual(read_json(self.out/'request.json')['prompt'], 'A copper telescope that scans and nods.')
        paid.assert_not_called()
        self.assertFalse((self.out/'agents').exists())
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.prepare()

    def test_custom_actions_are_exact_and_bounded_before_creation(self):
        actions = ['Dance with alternating steps and arm swings.', 'Fly up, circle, then land.']
        self.prepare(custom_actions=actions)
        self.assertEqual(read_json(self.out/'request.json')['custom_actions'], actions)
        self.assertEqual(read_json(self.out/'estimate.json')['maximum_agent_calls'], 10)

    def test_plan_only_reuse_does_not_start_native_or_paid_work(self):
        source = self.root/'analytic.glb'
        source.write_bytes(b'Explicit non-renderable pipeline test fixture')
        with patch('avatar_assets.provider_model.preflight'):
            self.prepare(reuse_model=source)
        with patch('avatar_assets.provider_model.preview') as native, patch('avatar_assets.meshy.run') as paid:
            result = pipeline.run(self.out, execute=True, plan_only=True, agents=AnalyticAgent())
        self.assertEqual(result['status'], 'planned')
        native.assert_not_called()
        paid.assert_not_called()
        self.assertFalse((self.out/'inspection').exists())

    def test_cancellation_preserves_plan_without_starting_generation(self):
        self.prepare(max_credits=35)
        observations = iter((False, False, True))
        with patch('avatar_assets.meshy.run') as paid:
            result = pipeline.run(self.out, execute=True, agents=AnalyticAgent(),
                                  cancel_requested=lambda: next(observations))
        self.assertEqual(result['status'], 'cancelled')
        self.assertTrue((self.out/'plan.json').exists())
        self.assertFalse((self.out/'generation').exists())
        paid.assert_not_called()

    def test_prepare_only_resume_cannot_make_calls(self):
        with patch.object(pipeline, 'run') as run, self.assertRaisesRegex(ValueError, 'new job'):
            pipeline.main(SimpleNamespace(resume='outputs/existing', prepare_only=True))
        run.assert_not_called()

    def test_invalid_requests_fail_before_creating_output(self):
        for values in ({'max_repairs': 3}, {'max_agent_calls': 100}, {'max_credits': -1},
                       {'states': ['idle', 'idle']}, {'preset': 'unbounded'}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.prepare(**values)
            self.assertFalse(self.out.exists())
        with self.assertRaises(ValueError):
            pipeline.prepare('x'*4001, self.out)
        with self.assertRaisesRegex(ValueError, 'under this tool'):
            pipeline.prepare('Telescope', self.root/'escape')

    def test_request_and_backend_tampering_block_resume(self):
        self.prepare()
        original = (self.out/'request.json').read_bytes()
        (self.out/'request.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'request changed'):
            pipeline.Pipeline(self.out, agents=AnalyticAgent())
        (self.out/'request.json').write_bytes(original)
        (self.root/'avatar_assets/pipeline.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'backend changed'):
            pipeline.Pipeline(self.out, agents=AnalyticAgent())

    def test_no_execution_or_insufficient_cap_never_submits(self):
        self.prepare(max_credits=0)
        agent = AnalyticAgent()
        with patch('avatar_assets.meshy.run') as paid:
            result = pipeline.run(self.out, execute=True, agents=agent)
        self.assertEqual(result['status'], 'awaiting_generation_authorization')
        self.assertEqual(agent.roles, ['planner'])
        paid.assert_not_called()
        self.assertTrue((self.out/'generation/plan.json').is_file())

    def test_changed_generation_plan_cannot_expand_spend(self):
        self.prepare(max_credits=35)
        worker = pipeline.Pipeline(self.out, agents=AnalyticAgent())
        worker._source(plan(), False)
        path = self.out/'generation/plan.json'
        changed = read_json(path)
        changed['max_credits'] = 500
        path.write_text(json.dumps(changed))
        with patch('avatar_assets.meshy.run') as paid, self.assertRaisesRegex(ValueError, 'Generation plan changed'):
            worker._source(plan(), True)
        paid.assert_not_called()

    def test_unsupported_plan_does_not_claim_artifact_or_call_provider(self):
        self.prepare()
        answer = plan()
        answer.update(status='unsupported', reason='Physical flight requires an unsupported simulation.')
        agent = AnalyticAgent(answer)
        with patch('avatar_assets.meshy.run') as paid:
            result = pipeline.run(self.out, execute=True, agents=agent)
            resumed = pipeline.run(self.out, execute=True, agents=agent)
        self.assertEqual(result['status'], 'unsupported')
        self.assertEqual(resumed, result)
        self.assertEqual(agent.roles, ['planner'])
        self.assertNotIn('model', result)
        paid.assert_not_called()

    def test_failed_native_pack_is_not_automatically_replayed(self):
        self.prepare()
        worker = pipeline.Pipeline(self.out, agents=AnalyticAgent())
        failed = self.out/'failed'
        failed.mkdir()
        (failed/'failure.txt').write_text('Analytic native failure fixture')
        write_json(failed/'manifest.json', {'schema_version': 1, 'status': 'failed', 'artifacts': artifacts(failed)})
        create = Mock()
        with self.assertRaisesRegex(ValueError, 'Retained failed'):
            worker._pack(failed, create)
        create.assert_not_called()

    def test_nonowner_cannot_write_lifecycle_events(self):
        self.prepare()
        with pipeline._lock(self.out), self.assertRaisesRegex(ValueError, 'already running'):
            pipeline.run(self.out, agents=AnalyticAgent())
        self.assertEqual(list((self.out/'events').iterdir()), [])
        self.assertFalse((self.out/'status.json').exists())

    def test_interrupted_old_status_temp_does_not_break_checkpoint(self):
        self.prepare()
        (self.out/'status.tmp').write_text('old interrupted lifecycle fixture')
        pipeline.Pipeline(self.out, agents=AnalyticAgent()).event('fixture', 'checked')
        self.assertEqual(read_json(self.out/'status.json')['status'], 'checked')
        self.assertTrue((self.out/'status.tmp').exists())

    def test_montage_does_not_collide_with_requested_state_names(self):
        self.prepare()
        worker = pipeline.Pipeline(self.out, agents=AnalyticAgent())
        pack, preview = self.out/'analytic-pack', self.out/'analytic-preview'
        pack.mkdir(); preview.mkdir()
        names = ['all-states', 'motion-montage', 'front', 'portrait', 'scan', 'nod']
        write_json(pack/'manifest.json', {'states': [{'id': n} for n in names]})
        write_json(pack/'finish-report.json', {'states': [{'id': n, 'frames': [0,96]} for n in names]})
        for n in names:
            folder = pack/'renders'/n
            folder.mkdir(parents=True)
            for i in range(2):
                Image.new('RGB', (8,8), (i*30,20,10)).save(folder/f'{i:04d}.png')
        for p in [preview/'front.png',preview/'three-quarter.png',pack/'portrait.png']:
            Image.new('RGB',(8,8),'white').save(p)
        evidence = worker._evidence(0,pack,preview)
        self.assertEqual(len(evidence),4)
        self.assertEqual(evidence[0]['state_ids'],names)
        self.assertEqual(Path(evidence[0]['path']).name,'motion-montage.png')
        self.assertEqual(len({e['id'] for e in evidence}),4)

    def test_repair_keeps_failed_state_measurements_and_key_pose_images(self):
        self.prepare()
        worker = pipeline.Pipeline(self.out, agents=AnalyticAgent())
        pack = self.out/'rejected-pack'
        folder = pack/'renders/wave'; folder.mkdir(parents=True)
        seconds = [0, 1, 1.5, 2, 2.5, 3, 4]
        write_json(pack/'finish-report.json', {'states': [{'id':'wave', 'frames':[s*24 for s in seconds],
            'seconds':seconds}], 'authored_rig': {'positive_vertices_per_bone': {'Hand': 13}, 'layers': []}})
        write_json(pack/'rig-verification.json', {'motion': {'animation_verified':False,
            'animations':[{'name':'wave','motion_observed':False}]}})
        for i in range(7):
            Image.new('RGB',(384,384),(i*20,20,10)).save(folder/f'{i:04d}.png')
        facts = worker._diagnostics(pack)
        self.assertFalse(facts['skin_motion']['animation_verified'])
        self.assertEqual(facts['weight_coverage']['positive_vertices_per_bone']['Hand'],13)
        _, paths = worker._contacts(0,pack)
        with Image.open(paths[0]) as image:
            self.assertEqual(image.size,(1280,720))
        self.assertEqual(worker._contacts(0,pack)[1],paths)

    def test_repair_receives_surface_failure_before_rendering(self):
        self.prepare()
        worker = pipeline.Pipeline(self.out, agents=AnalyticAgent())
        pack = self.out/'rejected-surface'; pack.mkdir()
        report = {'area_ratio_below_0_01_count': 252,
                  'regression_gate': {'status': 'failed', 'quality_proof': False}}
        write_json(pack/'surface-integrity.json', report)
        self.assertEqual(worker._diagnostics(pack)['surface_integrity'], report)
        self.assertNotIn('sampled_states', worker._diagnostics(pack))
        write_json(pack/'finish-report.json', {'authored_rig': None, 'states': []})
        self.assertEqual(worker._diagnostics(pack)['weight_coverage'], {})

    def test_resumed_failed_stage_preserves_exact_repair_reason(self):
        self.prepare()
        worker = pipeline.Pipeline(self.out, agents=AnalyticAgent())
        pack = self.out/'rejected-native'
        def create():
            pack.mkdir()
            (pack/'report.txt').write_text('Analytic missing wing coverage')
            write_json(pack/'manifest.json', {'status':'failed',
                                              'artifacts':artifacts(pack)})
            raise ValueError('No measured wing motion')
        with self.assertRaises(ValueError) as first:
            worker._pack(pack, create)
        replay = Mock()
        with self.assertRaises(ValueError) as resumed:
            worker._pack(pack, replay)
        self.assertEqual(str(first.exception),str(resumed.exception))
        replay.assert_not_called()
        (pack/'report.txt').write_text('changed')
        with self.assertRaisesRegex(ValueError,'artifacts changed'):
            worker._pack(pack,replay)

    def test_unsealed_stage_is_interrupted_not_a_repair_candidate(self):
        self.prepare()
        worker = pipeline.Pipeline(self.out, agents=AnalyticAgent())
        pack = self.out/'unsealed'; pack.mkdir()
        replay = Mock()
        with self.assertRaises(pipeline.InterruptedStage):
            worker._pack(pack,replay)
        replay.assert_not_called()

    def test_action_review_cannot_borrow_a_different_states_motion(self):
        evidence = [{'id': 'dance-sheet', 'state_ids': ['dance']}, {'id': 'fly-sheet', 'state_ids': ['fly']}]
        review = {'verdict': 'accept', 'state_reviews': [{'id': 'dance', 'evidence_ids': ['dance-sheet'],
                  'requirement_reviews': [{'id': 'rhythm', 'evidence_ids': ['fly-sheet']}]}]}
        with self.assertRaisesRegex(ValueError, 'own state'):
            pipeline.Pipeline._bind_review(review, evidence)
        review['state_reviews'][0]['requirement_reviews'][0]['evidence_ids'] = ['dance-sheet']
        pipeline.Pipeline._bind_review(review, evidence)

    def test_sealed_delivery_is_read_only_after_backend_upgrade(self):
        self.prepare()
        delivery = {'status': 'complete', 'request_sha256': pipeline.sha256(self.out/'request.json'), 'packs': []}
        for i in range(5):
            folder = self.out/f'pack-{i}'
            folder.mkdir()
            (folder/'fixture.txt').write_text('Analytic completed-pack fixture, not a model')
            write_json(folder/'manifest.json', {'artifacts': artifacts(folder)})
            delivery['packs'].append({'path':folder.name,'manifest_sha256':pipeline.sha256(folder/'manifest.json')})
        write_json(self.out/'delivery.json', delivery)
        write_json(self.out/'delivery-receipt.json', {'delivery.json':pipeline.sha256(self.out/'delivery.json')})
        (self.root/'avatar_assets/pipeline.py').write_text('Changed live backend cannot run this old job')
        agents = AnalyticAgent()
        result = pipeline.run(self.out, execute=True, agents=agents)
        self.assertTrue(result['resumed_without_generation'])
        self.assertEqual(agents.roles, [])
        (self.out/'pack-0/fixture.txt').write_text('Tampered')
        with self.assertRaisesRegex(ValueError, 'Artifact changed'):
            pipeline.run(self.out, execute=True, agents=agents)


if __name__ == '__main__':
    unittest.main()
