"""Analytic executable fixtures; these tests make no model/provider request."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from avatar_assets.agent_runner import AgentRunner, DISABLED_FEATURES, _events, _execute, _validate


SCHEMA = {'type': 'object', 'properties': {'value': {'type': 'integer', 'minimum': 0, 'maximum': 5}},
          'required': ['value'], 'additionalProperties': False}


class AgentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mode = self.root/'mode.txt'
        self.mode.write_text('success')
        self.counter = self.root/'calls.txt'
        self.seen = self.root/'seen.json'
        self.executable = self.root/'analytic-codex-fixture'
        code = '''import json, os, pathlib, sys, time
if '--version' in sys.argv:
    if pathlib.Path(MODE).read_text() == 'hang_version':
        time.sleep(10)
    print('analytic-agent-fixture 1')
    raise SystemExit(0)
counter = pathlib.Path(COUNTER)
counter.write_text(counter.read_text()+'x' if counter.exists() else 'x')
stdin = sys.stdin.read()
pathlib.Path(SEEN).write_text(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'prompt': stdin,
    'secret_passed': 'OPENAI_API_KEY' in os.environ or 'MESHY_API_KEY' in os.environ or 'FAKE_SECRET' in os.environ}))
mode = pathlib.Path(MODE).read_text()
answer = pathlib.Path(sys.argv[sys.argv.index('--output-last-message')+1])
if mode == 'hang':
    time.sleep(10)
if mode == 'flood':
    print('x'*10000, flush=True)
    time.sleep(10)
if mode == 'exit':
    print('simulated failure', file=sys.stderr)
    raise SystemExit(7)
if mode == 'large_answer':
    answer.write_text(' '*1100000)
    raise SystemExit(0)
answer.write_text('{bad' if mode == 'bad_json' else json.dumps({'value': 100 if mode == 'bad_schema' else 3}))
print(json.dumps({'type':'thread.started','thread_id':'analytic-thread'}))
if mode == 'tool':
    print(json.dumps({'type':'item.completed','item':{'type':'command_execution','command':'fixture-only'}}))
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':answer.read_text()}}))
if mode != 'incomplete':
    print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':5}}))
'''
        code = code.replace('COUNTER', repr(str(self.counter))).replace('SEEN', repr(str(self.seen))).replace('MODE', repr(str(self.mode)))
        self.executable.write_text('#!'+sys.executable+'\n'+code)
        self.executable.chmod(0o700)

    def runner(self, **kwargs):
        return AgentRunner(self.root/'agents', executable=self.executable, fixture=True, **kwargs)

    def test_success_records_usage_and_resumes_without_request(self):
        runner = self.runner()
        self.assertEqual(runner.call('planner', 'Make a cube', SCHEMA), {'value': 3})
        self.assertEqual(self.counter.read_text(), 'x')
        again = self.runner()
        self.assertEqual(again.call('planner', 'Make a cube', SCHEMA), {'value': 3})
        self.assertEqual(self.counter.read_text(), 'x')
        folder = self.root/'agents/planner'
        receipt = json.loads((folder/'receipt.json').read_text())
        self.assertEqual(receipt['execution_kind'], 'analytic_fixture')
        self.assertEqual(receipt['usage']['input_tokens'], 10)
        self.assertEqual(receipt['thread_id'], 'analytic-thread')
        self.assertFalse(receipt['tool_activity'])
        self.assertEqual((folder/'prompt.txt').read_text(), 'Make a cube')
        self.assertTrue(receipt['artifacts'])

    def test_tool_disabled_command_isolated_cwd_and_filtered_environment(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'do-not-pass', 'MESHY_API_KEY': 'do-not-pass', 'FAKE_SECRET': 'do-not-pass'}):
            self.runner().call('planner', 'Describe a robot', SCHEMA)
        seen = json.loads(self.seen.read_text())
        self.assertFalse(seen['secret_passed'])
        self.assertFalse(Path(seen['cwd']).exists())
        self.assertNotEqual(seen['cwd'], str(self.root))
        args = seen['argv']
        self.assertEqual(args[args.index('--sandbox')+1], 'read-only')
        for option in ('--ignore-user-config', '--ephemeral', '--skip-git-repo-check', '--output-schema', '--json'):
            self.assertIn(option, args)
        disabled = {args[i+1] for i, value in enumerate(args) if value == '--disable'}
        self.assertEqual(disabled, set(DISABLED_FEATURES))
        self.assertIn('web_search="disabled"', args)
        self.assertIn('mcp_servers={}', args)
        self.assertIn('project_doc_max_bytes=0', args)
        self.assertFalse(any('dangerously' in value for value in args))

    def test_changed_or_tampered_success_is_not_reused(self):
        runner = self.runner()
        runner.call('planner', 'Make a cube', SCHEMA)
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            runner.call('planner', 'Make a sphere', SCHEMA)
        (self.root/'agents/planner/answer.json').write_text('{"value":4}')
        with self.assertRaisesRegex(ValueError, 'artifacts changed'):
            runner.call('planner', 'Make a cube', SCHEMA)
        self.assertEqual(self.counter.read_text(), 'x')

    def test_persistent_call_ceiling_cannot_be_reset(self):
        runner = self.runner(max_calls=1)
        runner.call('planner', 'Make a cube', SCHEMA)
        with self.assertRaisesRegex(ValueError, 'allowance exhausted'):
            self.runner(max_calls=1).call('critic', 'Review cube', SCHEMA)
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            self.runner(max_calls=2)
        self.assertEqual(self.counter.read_text(), 'x')

    def test_uncertain_reservation_is_preserved_and_consumes_allowance(self):
        runner = self.runner(max_calls=1)
        (self.root/'agents/planner').mkdir()
        with self.assertRaisesRegex(ValueError, 'reservation is incomplete'):
            runner.call('planner', 'Make cube', SCHEMA)
        with self.assertRaisesRegex(ValueError, 'allowance exhausted'):
            runner.call('author', 'Make cube', SCHEMA)
        self.assertFalse(self.counter.exists())

    def test_failed_response_never_auto_retries(self):
        for mode in ('exit', 'bad_json', 'bad_schema', 'tool', 'incomplete'):
            with self.subTest(mode=mode):
                self.mode.write_text(mode)
                runner = AgentRunner(self.root/('agents-'+mode), executable=self.executable, fixture=True)
                with self.assertRaises(ValueError):
                    runner.call('planner', 'Make cube', SCHEMA)
                count = self.counter.read_text()
                self.mode.write_text('success')
                with self.assertRaisesRegex(ValueError, 'no automatic resubmission'):
                    runner.call('planner', 'Make cube', SCHEMA)
                self.assertEqual(self.counter.read_text(), count)
                receipt = json.loads((runner.root/'planner/receipt.json').read_text())
                self.assertEqual(receipt['status'], 'failed_or_uncertain')

    def test_timeout_stops_owned_process_and_leaves_receipt(self):
        self.mode.write_text('hang')
        runner = self.runner(timeout=.2)

        def execute_stage(command, cwd, out, timeout, stdin=None, prefix='agent'):
            # Isolate the intentional agent hang from cold version-probe startup.
            return _execute(command, cwd, out, 2 if prefix == 'version' else timeout, stdin, prefix)

        with patch('avatar_assets.agent_runner._execute', side_effect=execute_stage), \
                self.assertRaisesRegex(ValueError, 'wall-time'):
            runner.call('planner', 'Make cube', SCHEMA)
        version = json.loads((runner.root/'planner/version-process.json').read_text())
        self.assertEqual(version['exit_code'], 0)
        self.assertIsNone(version['failure'])
        report = json.loads((runner.root/'planner/agent-process.json').read_text())
        self.assertEqual(report['limits']['wall_seconds'], .2)
        self.assertIsNotNone(report['exit_code'])
        self.assertFalse(report['manual_stop_required'])
        self.assertTrue(report['cleanup'])
        self.assertLess(report['elapsed_seconds'], 5)

    def test_version_timeout_stops_owned_process_and_leaves_receipt(self):
        self.mode.write_text('hang_version')
        runner = self.runner(timeout=.2)
        with self.assertRaisesRegex(ValueError, 'wall-time'):
            runner.call('planner', 'Make cube', SCHEMA)
        folder = runner.root/'planner'
        report = json.loads((folder/'version-process.json').read_text())
        self.assertEqual(report['limits']['wall_seconds'], .2)
        self.assertIsNotNone(report['exit_code'])
        self.assertFalse(report['manual_stop_required'])
        self.assertTrue(report['cleanup'])
        self.assertLess(report['elapsed_seconds'], 5)
        self.assertFalse((folder/'agent-process.json').exists())
        self.assertFalse(self.counter.exists())
        receipt = json.loads((folder/'receipt.json').read_text())
        self.assertEqual(receipt['status'], 'failed_or_uncertain')
        self.assertFalse(receipt['automatic_resubmission_allowed'])
        self.assertIn('version-process.json', {row['path'] for row in receipt['artifacts']})
        with self.assertRaisesRegex(ValueError, 'no automatic resubmission'):
            runner.call('planner', 'Make cube', SCHEMA)
        self.assertFalse(self.counter.exists())

    def test_log_budget_stops_flood_and_caps_stored_log(self):
        self.mode.write_text('flood')
        with patch('avatar_assets.agent_runner.MAX_STREAM_BYTES', 1024):
            runner = self.runner()
            with self.assertRaisesRegex(ValueError, 'log byte budget'):
                runner.call('planner', 'Make cube', SCHEMA)
        self.assertLessEqual((runner.root/'planner/agent.jsonl').stat().st_size, 1024)
        report = json.loads((runner.root/'planner/agent-process.json').read_text())
        self.assertIsNotNone(report['exit_code'])

    def test_answer_and_memory_limits_fail_closed(self):
        self.mode.write_text('large_answer')
        runner = self.runner()
        with self.assertRaisesRegex(ValueError, 'answer byte budget'):
            runner.call('planner', 'Make cube', SCHEMA)
        self.mode.write_text('hang')
        with patch('avatar_assets.agent_runner.resident_bytes', return_value=2*1024**3):
            with self.assertRaisesRegex(ValueError, 'resident-memory'):
                runner.call('critic', 'Review cube', SCHEMA)
        process_report = runner.root/'critic/agent-process.json'
        if not process_report.exists():
            process_report = runner.root/'critic/version-process.json'
        report = json.loads(process_report.read_text())
        self.assertIsNotNone(report['exit_code'])
        self.assertFalse(report['manual_stop_required'])

    def test_selected_images_copied_and_hash_bound(self):
        image = self.root/'reference.png'
        Image.new('RGB', (4, 4), '#24a243').save(image)
        runner = self.runner()
        runner.call('planner', 'Look at this', SCHEMA, [image])
        self.assertEqual(image.read_bytes(), (runner.root/'planner/image-00.png').read_bytes())
        Image.new('RGB', (4, 4), '#aa0000').save(image)
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            runner.call('planner', 'Look at this', SCHEMA, [image])
        self.assertEqual(self.counter.read_text(), 'x')

    def test_invalid_and_oversized_inputs_fail_before_call(self):
        runner = self.runner()
        invalid = [('bad/role', 'x', SCHEMA, []), ('planner', '', SCHEMA, []),
                   ('planner', 'x'*131073, SCHEMA, []), ('planner', 'x', {'type':'object','not':{}}, []),
                   ('planner', 'x', SCHEMA, ['none']*9)]
        for args in invalid:
            with self.subTest(args=str(args)[:80]), self.assertRaises(ValueError):
                runner.call(*args)
        oversized = self.root/'wide.png'
        Image.new('RGB', (9000, 1)).save(oversized)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            runner.call('planner', 'x', SCHEMA, [oversized])
        self.assertFalse(self.counter.exists())
        self.assertEqual([p for p in runner.root.iterdir() if p.is_dir()], [])

    def test_ownership_lock_prevents_duplicate_call(self):
        runner = self.runner()
        with runner._lock(), self.assertRaisesRegex(ValueError, 'Another process owns'):
            runner.call('planner', 'Make cube', SCHEMA)
        self.assertFalse(self.counter.exists())

    def test_schema_nested_types_and_local_definitions(self):
        schema = {'type':'object', 'properties':{'items':{'type':'array','maxItems':2,'items':{'$ref':'#/$defs/item'}}},
                  'required':['items'], 'additionalProperties':False,
                  '$defs':{'item':{'anyOf':[{'type':'string','minLength':1},{'type':'null'}]}}}
        _validate({'items':['a',None]},schema)
        for value in ({'items':[False]}, {'items':['']}, {'items':['a','b','c']}, {'items':[], 'extra':1}):
            with self.assertRaises(ValueError):
                _validate(value,schema)

    def test_observed_cli_bootstrap_warnings_are_preserved(self):
        messages = [
            'Under-development features enabled: omit_skill_descriptions, skip_host_skill_discovery. '
            'Under-development features are incomplete and may behave unpredictably. '
            'To suppress this warning, set `suppress_unstable_features_warning = true` in /tmp/fake-home/.codex/config.toml.',
            'Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; '
            'enable `features.code_mode_host` and install `codex-code-mode-host`.',
        ]
        events = [{'type':'thread.started','thread_id':'fixture-thread'}]
        events += [{'type':'item.completed','item':{'type':'error','message':text}} for text in messages]
        events += [{'type':'turn.started'}, {'type':'turn.completed','usage':{'input_tokens':15,'output_tokens':3}}]
        path = self.root/'events.jsonl'
        path.write_text(''.join(json.dumps(event)+'\n' for event in events))
        report = _events(path)
        self.assertEqual(report['bootstrap_warnings'],messages)
        self.assertEqual(report['usage']['output_tokens'],3)
        self.assertFalse(report['tool_activity'])

    def test_unknown_errors_and_tool_events_remain_rejected(self):
        code_mode = ('Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; '
                     'enable `features.code_mode_host` and install `codex-code-mode-host`.')
        cases = [
            [{'type':'item.completed','item':{'type':'error','message':'Provider authentication failed'}}],
            [{'type':'item.completed','item':{'type':'error','message':code_mode+' Extra failure'}}],
            [{'type':'turn.started'},{'type':'item.completed','item':{'type':'error','message':code_mode}}],
            [{'type':'error','message':code_mode}],
            [{'type':'turn.failed','error':{'message':code_mode}}],
            [{'type':'item.completed','item':{'type':'command_execution','command':'fixture-only'}}],
        ]
        path = self.root/'events.jsonl'
        for events in cases:
            events = events+[{'type':'turn.completed','usage':{}}]
            path.write_text(''.join(json.dumps(event)+'\n' for event in events))
            with self.subTest(events=events), self.assertRaises(ValueError):
                _events(path)


if __name__ == '__main__':
    unittest.main()
