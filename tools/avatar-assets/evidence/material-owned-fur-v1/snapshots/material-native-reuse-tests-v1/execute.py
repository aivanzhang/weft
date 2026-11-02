import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, sha256, write_json

test = ROOT/'tests/test_material_native_reuse.py'
constants = {}
for node in ast.parse(test.read_text()).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id in ('SCRIPT_SOURCES', 'CONFIG_SOURCES'):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
paths = ['tests/test_material_native_reuse.py', 'avatar_assets/finish_generated.py',
         'avatar_assets/assemble_generated.py', 'avatar_assets/provider_model.py',
         'avatar_assets/three_d.py', 'avatar_assets/runner.py', 'avatar_assets/safety.py',
         'avatar_assets/__init__.py', 'tests/__init__.py']
paths += ['scripts/'+name for name in constants['SCRIPT_SOURCES']]
paths += ['avatar_assets/'+name for name in constants['CONFIG_SOURCES']]
actual = 'outputs/fox-material-fur-finish-v1/'
paths += [actual+name for name in ('material-groom/report.json', 'material-groom/correspondence.npz',
                                 'material-coat.json', 'finish-report.json', 'manifest.json')]
paths = sorted(set(paths))
assert sha256(ROOT/'avatar_assets/finish_generated.py') == '1a2e03a2a6708e7d36e6be39f19e044f6521cbf6b5734b04a50bd461e9d644f1', 'Production source changed before freeze'
inputs = []
for name in paths:
    source = ROOT/name
    assert source.is_file() and not source.is_symlink() and source.stat().st_size <= 8*1024**2
    target = OUT/'sources'/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    inputs.append({'path': name, 'sha256': sha256(source), 'bytes': source.stat().st_size,
                   'snapshot': str(target.relative_to(OUT))})
command = ['/usr/bin/env', 'PYTHONDONTWRITEBYTECODE=1', 'OPENBLAS_NUM_THREADS=1',
           'OMP_NUM_THREADS=1', 'TMPDIR='+str(OUT/'tmp'), sys.executable, str(OUT/'run_tests.py')]
write_json(OUT/'prepared.json', {
    'schema_version': 1, 'at': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
    'command': command, 'equivalent_test_command': '.venv/bin/python -m unittest -v tests.test_material_native_reuse',
    'unique_test_methods': 17, 'inputs': inputs,
    'runner_sources': [{'path': name, 'sha256': sha256(OUT/name)} for name in ('run_tests.py', 'execute.py')],
    'limits': {'attempts': 1, 'seconds': 30, 'memory_mb': 512, 'output_mb': 16,
               'native_jobs': 0, 'provider_calls': 0, 'ai_role_calls': 0},
    'static_review_finding': {
        'finding': 'Previously a JSON-null material-coat receipt compared equal to a missing embedded field.',
        'verification_history': 'Found by source inspection; no old-source test execution or failing test result is claimed.',
        'root_fix': 'Require dictionary, material-owned-tufts-v1 policy and exact embedded equality.',
        'prior_source_hash': None, 'reason': 'The pre-fix source hash was not captured before root patched it.'},
    'scope': 'Analytic temporary-file copy/custody tests plus one actual read-only fox receipt comparison; no media validity or quality claim.'})
job = run_bounded(command, OUT/'cpu.log', OUT, timeout=30, memory_mb=512, output_mb=16)
write_json(OUT/'cpu-job.json', job)
unchanged = [{'path': row['path'], 'unchanged': (ROOT/row['path']).stat().st_size == row['bytes']
              and sha256(ROOT/row['path']) == row['sha256']} for row in inputs]
result = json.loads((OUT/'test-result.json').read_text()) if (OUT/'test-result.json').is_file() else {}
passed = (job.get('process_exit_verified') and job.get('exit_code') == 0 and not job.get('failure')
          and result.get('passed') is True and all(row['unchanged'] for row in unchanged))
write_json(OUT/'receipt.json', {'schema_version': 1, 'status': 'passed' if passed else 'failed',
    'test_result': result, 'bounded_job': job, 'source_unchanged': unchanged,
    'native_jobs': 0, 'provider_calls': 0, 'ai_role_calls': 0,
    'limitations': ['Analytic placeholder model/image bytes are never parsed or rendered.',
                    'Actual fox comparison checks receipt consistency only, not animation or visual quality.',
                    'Resealed negatives exercise dedicated custody guards independently of generic manifest tamper detection.']})
write_json(OUT/'manifest.json', {'schema_version': 1, 'kind': 'material_native_reuse_test_evidence',
    'status': 'passed' if passed else 'failed', 'artifacts': artifacts(OUT)})
print(json.dumps({'status': 'passed' if passed else 'failed', 'tests_run': result.get('tests_run'),
    'elapsed_seconds': job.get('elapsed_seconds'), 'sampled_peak_rss_bytes': job.get('sampled_peak_rss_bytes'),
    'manifest_sha256': sha256(OUT/'manifest.json')}, sort_keys=True))
raise SystemExit(0 if passed else 1)
