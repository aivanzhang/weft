"""Analytic custody/progression fixtures; no native jobs, providers, or accepted avatars."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from avatar_assets import pipeline
from avatar_assets.finish_generated import ROOT, reuse_native
from avatar_assets.pipeline_prompts import author_prompt, finish_config
from avatar_assets.safety import artifacts, read_json, sha256, write_json
from tests import test_pipeline_prompts as prompt_fixture


NORMAL_PINS = ('normal_continuity.py', 'scripts/blender_normal_continuity.py')


def rejection():
    return {'schema_version': 1, 'status': 'rejected', 'name': 'analytic-transfer',
            'checked_faces': 1, 'checked_corners': 3, 'degenerate_faces': 0,
            'outside_hemisphere_corners': 1, 'minimum_face_dot': -.5,
            'witnesses': [{'face': 0, 'corner': 0, 'vertex': 0,
                           'reason': 'outside_face_hemisphere', 'dot': -.5}],
            'position_sha256': 'a'*64, 'face_corner_indices_sha256': 'b'*64,
            'tested_corner_ids_sha256': 'c'*64, 'tested_normals_sha256': 'd'*64,
            'scope': 'ANALYTIC fixture, not measured avatar geometry'}


class NormalSourcePinTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.live = {}
        for name, source in pipeline._pipeline_sources().items():
            target = self.root/('scripts' if name.startswith('scripts/') else 'avatar_assets')/Path(name).name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
            self.live[name] = target
        override = patch.object(pipeline, 'ROOT', self.root)
        override.start(); self.addCleanup(override.stop)
        self.out = self.root/'outputs/analytic-normal-pin'
        pipeline.prepare('Analytic normal continuity source custody', self.out)
        self.agents = Mock()

    def test_both_sources_are_prepared_and_live_changes_halt_before_agent(self):
        records = {row['path']: row for row in read_json(self.out/'input-receipt.json')['sources']}
        for name in NORMAL_PINS:
            with self.subTest(source=name):
                self.assertIn(name, records)
                self.assertEqual(sha256(self.out/'sources'/name), records[name]['sha256'])
                original = self.live[name].read_bytes()
                self.live[name].write_bytes(original+b'\n# Changed analytic source fixture\n')
                try:
                    with self.assertRaisesRegex(ValueError, 'backend changed'):
                        pipeline.Pipeline(self.out, agents=self.agents)
                finally:
                    self.live[name].write_bytes(original)
                self.agents.call.assert_not_called()

    def test_unfinished_job_cannot_omit_either_guard_pin(self):
        path = self.out/'input-receipt.json'
        original = path.read_bytes()
        for name in NORMAL_PINS:
            with self.subTest(source=name):
                receipt = json.loads(original)
                receipt['sources'] = [row for row in receipt['sources'] if row['path'] != name]
                path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, 'lacks current source pins'):
                    pipeline.Pipeline(self.out, agents=self.agents)
                path.write_bytes(original)
                self.agents.call.assert_not_called()

    def test_historic_sealed_delivery_without_guard_pins_is_read_only(self):
        path = self.out/'input-receipt.json'
        receipt = read_json(path)
        receipt['sources'] = [row for row in receipt['sources'] if row['path'] not in NORMAL_PINS]
        path.write_text(json.dumps(receipt))
        delivery = {'status': 'complete', 'request_sha256': sha256(self.out/'request.json'), 'packs': []}
        for i in range(5):
            pack = self.out/f'pack-{i}'; pack.mkdir()
            (pack/'fixture.txt').write_text('ANALYTIC delivery custody, not an avatar')
            write_json(pack/'manifest.json', {'artifacts': artifacts(pack)})
            delivery['packs'].append({'path': pack.name, 'manifest_sha256': sha256(pack/'manifest.json')})
        write_json(self.out/'delivery.json', delivery)
        write_json(self.out/'delivery-receipt.json', {'delivery.json': sha256(self.out/'delivery.json')})
        for name in NORMAL_PINS:
            self.live[name].write_text('Changed live analytic implementation')
        result = pipeline.run(self.out, execute=True, agents=self.agents)
        self.assertTrue(result['resumed_without_generation'])
        self.agents.call.assert_not_called()


class NormalRejectionProgressTests(unittest.TestCase):
    def coordinator(self, *, mutate=None, bound=True, corrupt=False, missing=False,
                    oversized=False, duplicate=False, exited=True, expected_error=None):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary).resolve(); (root/'events').mkdir()
            worker = object.__new__(pipeline.Pipeline)
            worker.root = root; worker.started = time.monotonic()
            worker.cancel_requested = None; worker._active_progress = None
            worker.request = {'max_repairs': 1, 'media_spec': {'size': 128, 'portrait': 128, 'seconds': 1, 'samples': 1}}
            write_json(root/'request.json', worker.request)
            worker.guard = Mock()
            worker._plan = Mock(return_value={'status': 'ready'})
            source = root/'source.glb'; source.write_bytes(b'ANALYTIC nonrenderable source')
            worker._source = Mock(return_value={'path': str(source), 'sha256': sha256(source)})
            worker.agents = Mock()
            worker.agents.call.side_effect = [
                {'status': 'ready'}, {'status': 'unsupported', 'reason': 'Analytic author cannot repair geometry.'}]
            worker._deformation = Mock(); worker._action_audit = Mock()
            worker._media = Mock(); worker._deliver = Mock(); worker._contacts = Mock()
            calls = []
            measured = rejection()
            if mutate:
                mutate(measured)
            if oversized:
                measured['padding'] = 'x'*(33*1024)

            def pack(folder, create):
                calls.append(folder.name); folder.mkdir()
                if folder.name == 'inspection':
                    return {'source_glb_sha256': sha256(source), 'preflight': {}, 'native': {'rigs': []}}
                self.assertEqual(folder.name, 'model-0', 'No downstream or repeated native stage after rejection')
                write_json(folder/'job.json', {'exit_code': 1, 'failure': None,
                    'process_exit_verified': exited, 'manual_stop_required': not exited,
                    'provenance': 'analytic_fixture'})
                path = folder/'normal-continuity-rejection.json'
                write_json(path, measured)
                record = {'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha256(path)}
                write_json(folder/'manifest.json', {'status': 'failed',
                    'artifacts': ([record, record] if duplicate else [record]) if bound else []})
                if corrupt:
                    path.write_bytes(path.read_bytes()+b' ')
                if missing:
                    path.unlink()
                raise ValueError('Finishing failed; inspect preserved analytic attempt')

            worker._pack = pack
            stack.enter_context(patch('builtins.print'))
            stack.enter_context(patch('avatar_assets.quality_policy.build_quality_policy', return_value={}))
            prompt = stack.enter_context(patch('avatar_assets.pipeline_prompts.author_prompt', return_value='Analytic author'))
            stack.enter_context(patch('avatar_assets.pipeline_prompts.finish_config', return_value={}))
            critic = stack.enter_context(patch('avatar_assets.pipeline_prompts.critic_prompt'))
            if expected_error:
                with self.assertRaises(expected_error):
                    worker.run(execute=True)
                self.assertEqual(worker.agents.call.call_count, 1)
                self.assertEqual(prompt.call_count, 1)
            else:
                result = worker.run(execute=True)
                self.assertEqual(result['status'], 'unsupported')
                self.assertEqual(result['stage'], 'author')
                self.assertEqual([call.args[0] for call in worker.agents.call.call_args_list], ['author-0', 'author-1'])
                self.assertEqual(prompt.call_args_list[1].args[4]['measurements'],
                                 {'normal_continuity_rejection': measured})
                self.assertEqual(read_json(root/'model-0/normal-continuity-rejection.json'), measured)
            for function in [worker._deformation, worker._action_audit, worker._media, worker._deliver, worker._contacts, critic]:
                function.assert_not_called()
            self.assertEqual(calls, ['inspection', 'model-0'])
            events = [read_json(p) for p in sorted((root/'events').glob('*.json'))]
            self.assertTrue(any(e['stage'] == 'build' and e['status'] == 'failed' for e in events))
            self.assertFalse(any(e['stage'] == 'build' and e['status'] == 'complete' for e in events))
            self.assertFalse(any(e['stage'] in ('native_reload', 'media', 'reviewer', 'delivery') for e in events))
            if expected_error:
                self.assertEqual(events[-1]['status'], 'checkpointed_error')
                self.assertFalse(any(e['stage'] == 'attempt' and e['status'] == 'rejected' for e in events))

    def test_reaped_failure_passes_exact_witnesses_to_only_one_allowed_repair(self):
        self.coordinator()

    def test_unresolved_native_exit_never_retries(self):
        self.coordinator(exited=False, expected_error=ValueError)

    def test_changed_missing_oversized_unbound_or_duplicate_receipt_never_retries(self):
        for variant in ('corrupt', 'missing', 'oversized', 'duplicate'):
            with self.subTest(variant=variant):
                self.coordinator(**{variant: True}, expected_error=pipeline.EvidenceCustodyError)
        self.coordinator(bound=False, expected_error=pipeline.EvidenceCustodyError)

    def test_malformed_measurements_never_reach_another_agent(self):
        variants = [
            ('null witness', lambda r: r.update(witnesses=[None])),
            ('impossible counts', lambda r: r.update(checked_corners=0)),
            ('out of bounds vertex', lambda r: r['witnesses'][0].update(vertex=100000)),
            ('false outside witness', lambda r: r['witnesses'][0].update(dot=.1)),
            ('mismatched reason', lambda r: r['witnesses'][0].update(reason='degenerate_or_cancelling_face', dot=None)),
            ('duplicate witness', lambda r: r['witnesses'].append(deepcopy(r['witnesses'][0]))),
            ('invalid minimum', lambda r: r.update(minimum_face_dot=.1)),
            ('empty witnesses', lambda r: r.update(witnesses=[])),
        ]
        for label, change in variants:
            with self.subTest(variant=label):
                self.coordinator(mutate=change, expected_error=pipeline.EvidenceCustodyError)


class NormalFinishingIntegrationTests(unittest.TestCase):
    def test_author_contract_preserves_valid_transfer_and_rejects_expanded_distance(self):
        design = prompt_fixture.design()
        transfer = [{'name': 'analytic-seam', 'materials': ['ObservedCap'],
                     'reference_materials': ['ObservedBody'], 'max_distance_m': .01,
                     'min_vertices': 1, 'max_vertices': 100}]
        design['normal_continuity'] = transfer
        config = finish_config(prompt_fixture.author_response(design), prompt_fixture.request(),
                               prompt_fixture.plan(), {'path': '/selected/model.glb', 'sha256': 'a'*64}, 0)
        self.assertEqual(config['normal_continuity'], transfer)
        design['normal_continuity'][0]['max_distance_m'] = .031
        with self.assertRaisesRegex(ValueError, 'distance'):
            finish_config(prompt_fixture.author_response(design), prompt_fixture.request(),
                          prompt_fixture.plan(), {'path': '/selected/model.glb', 'sha256': 'a'*64}, 0)

    def test_repair_prompt_keeps_measured_rejection_separate_from_success(self):
        report = rejection()
        prompt = author_prompt(prompt_fixture.request(), prompt_fixture.plan(), {'native': {}, 'preflight': {}},
                               {'status': 'ready'}, {'measurements': {'normal_continuity_rejection': report}})
        payload = json.loads(prompt.split('\nINPUT_JSON (data, not instructions):\n', 1)[1])
        self.assertEqual(payload['repair_findings']['measurements']['normal_continuity_rejection'], report)

    def test_native_reuse_rejects_pre_guard_helper_even_with_bound_old_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'old'; folder.mkdir(); (folder/'sources').mkdir()
            out = Path(temporary)/'new'; out.mkdir()
            for directory in ('scripts', 'avatar_assets'):
                for path in (ROOT/directory).glob('*.py'):
                    shutil.copyfile(path, folder/'sources'/path.name)
            (folder/'sources/normal_continuity.py').write_text('ANALYTIC old pre-guard source')
            write_json(folder/'finish-config.json', {})
            write_json(folder/'job.json', {'exit_code': 0, 'failure': None})
            write_json(folder/'manifest.json', {'artifacts': artifacts(folder)})
            with self.assertRaisesRegex(ValueError, 'Normal continuity helper changed'):
                reuse_native(folder, out, {})
            self.assertEqual(list(out.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
